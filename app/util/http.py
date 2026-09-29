from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from app.config import AppConfig, get_config
from app.util.credential_hygiene import (
    sanitize_json_value,
    sanitize_url_credentials,
    validate_sec_user_agent,
)
from app.util.hashing import sha256_text


_GLOBAL_LIMITERS: dict[float, "RateLimiter"] = {}
_GLOBAL_LIMITERS_LOCK = threading.Lock()
_GLOBAL_METRICS = {"request_count": 0, "throttled_count": 0}
_GLOBAL_METRICS_LOCK = threading.Lock()
_GLOBAL_DOMAIN_COUNTS: dict[str, int] = {}
_GLOBAL_BUDGET_OVERRIDE_LOCK = threading.Lock()


class AllowlistError(ValueError):
    pass


class DomainBudgetExceeded(RuntimeError):
    pass


class ResponseTooLarge(requests.RequestException):
    """A response body larger than the caller's byte cap; nothing is cached."""


class TooManyRedirects(requests.RequestException):
    """More redirect hops than ``MAX_REDIRECTS``."""


MAX_REDIRECTS = 5
DEFAULT_MAX_RESPONSE_BYTES = 250 * 1024 * 1024
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class NetworkDisabledError(requests.ConnectionError):
    """A request that would leave the machine while ``VOE_NET_PROVIDER=disabled``.

    Subclasses ``requests.ConnectionError`` so callers that already degrade
    gracefully on an unreachable host treat "offline" the same way.
    """


def network_disabled(cfg: AppConfig | None = None) -> bool:
    """True when ``VOE_NET_PROVIDER`` (the single network switch) is ``disabled``."""

    cfg = cfg or get_config()
    return str(getattr(cfg, "net_provider", "enabled")).strip().lower() == "disabled"


def require_network(url: str, cfg: AppConfig | None = None) -> None:
    """Raise :class:`NetworkDisabledError` if the network switch is off."""

    if network_disabled(cfg):
        raise NetworkDisabledError(
            f"offline: VOE_NET_PROVIDER=disabled and {sanitize_url_credentials(url)} "
            "is not in the local cache"
        )


class RateLimiter:
    def __init__(self, rate_per_sec: float) -> None:
        self.rate_per_sec = max(rate_per_sec, 0.1)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + (1.0 / self.rate_per_sec)


class HttpClient:
    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or get_config()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.cfg.sec_user_agent})
        self.rate_limiter = self._global_rate_limiter(self.cfg.sec_rate_limit_per_sec)

    @staticmethod
    def _global_rate_limiter(rate_per_sec: float) -> RateLimiter:
        with _GLOBAL_LIMITERS_LOCK:
            if rate_per_sec not in _GLOBAL_LIMITERS:
                _GLOBAL_LIMITERS[rate_per_sec] = RateLimiter(rate_per_sec)
            return _GLOBAL_LIMITERS[rate_per_sec]

    def _allowed_hosts(self) -> set[str]:
        hosts = {"sec.gov", "www.sec.gov", "data.sec.gov"}
        if self.cfg.allow_fred:
            hosts.update({"fred.stlouisfed.org", "api.stlouisfed.org"})
        if not self.cfg.safe_mode and self.cfg.price_provider.lower() == "stooq":
            hosts.update({"stooq.com", "www.stooq.com"})
        if not self.cfg.safe_mode and self.cfg.eodhd_apikey:
            hosts.update({"eodhd.com"})
        if (
            not self.cfg.safe_mode
            and self.cfg.research_enable_transcripts
            and self.cfg.research_transcript_provider.lower() == "alpha_vantage"
            and self.cfg.research_alpha_vantage_api_key
        ):
            hosts.update({"alphavantage.co", "www.alphavantage.co"})
        if (
            not self.cfg.safe_mode
            and self.cfg.research_external_news_enabled
            and self.cfg.research_external_news_provider.lower() == "alpha_vantage"
            and self.cfg.research_alpha_vantage_api_key
        ):
            hosts.update({"alphavantage.co", "www.alphavantage.co"})
        if not self.cfg.safe_mode and self.cfg.research_enable_wikipedia:
            hosts.update({"wikipedia.org", "en.wikipedia.org"})
        if (
            not self.cfg.safe_mode
            and self.cfg.events_dockets_enabled
            and self.cfg.courtlistener_api_token
        ):
            hosts.update({"courtlistener.com", "www.courtlistener.com"})
        if (
            not self.cfg.safe_mode
            and self.cfg.events_adverse_news_enabled
            and self.cfg.research_alpha_vantage_api_key
        ):
            hosts.update({"alphavantage.co", "www.alphavantage.co"})
        if not self.cfg.safe_mode and (self.cfg.research_ir_press_enabled or self.cfg.research_company_news_enabled):
            hosts.update({h.lower() for h in self.cfg.research_allowlist_domains})
        return hosts

    @staticmethod
    def _host_match(host: str, candidate: str) -> bool:
        candidate = candidate.strip().lower()
        if candidate.startswith("*."):
            suffix = candidate[2:]
            if not suffix:
                return False
            return host == suffix or host.endswith(f".{suffix}")
        return host == candidate or host.endswith(f".{candidate}")

    def _budget_for_host(self, host: str) -> int | None:
        host = host.lower()
        if host == "data.sec.gov":
            return self.cfg.max_requests_data_sec_domain
        if host == "www.sec.gov":
            return self.cfg.max_requests_www_sec_domain
        if host == "sec.gov" or host.endswith(".sec.gov"):
            return self.cfg.max_requests_sec_domain
        if host == "stooq.com" or host.endswith(".stooq.com"):
            return self.cfg.max_requests_stooq_domain
        if host == "eodhd.com" or host.endswith(".eodhd.com"):
            return self.cfg.max_requests_eodhd_domain
        if host == "alphavantage.co" or host.endswith(".alphavantage.co"):
            return self.cfg.max_requests_alpha_vantage_domain
        if host == "wikipedia.org" or host.endswith(".wikipedia.org"):
            return self.cfg.max_requests_wikipedia_domain
        if host == "courtlistener.com" or host.endswith(".courtlistener.com"):
            return self.cfg.max_requests_courtlistener_domain
        if not self.cfg.safe_mode and (self.cfg.research_ir_press_enabled or self.cfg.research_company_news_enabled):
            for allowed in self.cfg.research_allowlist_domains:
                if self._host_match(host, allowed.lower()):
                    return self.cfg.max_requests_press_domain
        return None

    def _consume_domain_budget(self, url: str) -> None:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            return
        budget = self._budget_for_host(host)
        if budget is None:
            return
        with _GLOBAL_METRICS_LOCK:
            current = _GLOBAL_DOMAIN_COUNTS.get(host, 0)
            if current >= budget:
                raise DomainBudgetExceeded(f"Domain budget exceeded for {host} ({budget} requests)")
            _GLOBAL_DOMAIN_COUNTS[host] = current + 1

    def _check_allowlist(self, url: str) -> None:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host:
            raise AllowlistError(f"URL has no hostname: {url}")
        if host in self._allowed_hosts():
            return
        if host.endswith(".sec.gov"):
            return
        if self.cfg.allow_fred and (host.endswith(".stlouisfed.org") or host.endswith(".fred.stlouisfed.org")):
            return
        if (
            not self.cfg.safe_mode
            and self.cfg.research_enable_transcripts
            and self.cfg.research_transcript_provider.lower() == "alpha_vantage"
            and self.cfg.research_alpha_vantage_api_key
            and (host == "alphavantage.co" or host.endswith(".alphavantage.co"))
        ):
            return
        if (
            not self.cfg.safe_mode
            and self.cfg.research_external_news_enabled
            and self.cfg.research_external_news_provider.lower() == "alpha_vantage"
            and self.cfg.research_alpha_vantage_api_key
            and (host == "alphavantage.co" or host.endswith(".alphavantage.co"))
        ):
            return
        if not self.cfg.safe_mode and self.cfg.research_enable_wikipedia and (
            host == "wikipedia.org" or host.endswith(".wikipedia.org")
        ):
            return
        if not self.cfg.safe_mode and (self.cfg.research_ir_press_enabled or self.cfg.research_company_news_enabled):
            for allowed in self.cfg.research_allowlist_domains:
                if self._host_match(host, allowed.lower()):
                    return
        raise AllowlistError(f"Host not allowlisted in v0: {host}")

    def _require_valid_sec_user_agent(self, url: str) -> None:
        host = (urlparse(url).hostname or "").lower()
        if host == "sec.gov" or host.endswith(".sec.gov"):
            validate_sec_user_agent(self.session.headers.get("User-Agent"))

    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path:
        key = json.dumps(
            {
                "u": sanitize_url_credentials(url),
                "p": sanitize_json_value(params or {}),
            },
            sort_keys=True,
        )
        digest = sha256_text(key)
        folder = self.cfg.cache_dir / "http"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{digest}.cache"

    def cache_path(self, url: str, params: dict[str, Any] | None = None) -> Path:
        return self._cache_path(url, params)

    def read_cached_json(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        cache_path = self.cache_path(url, params)
        if not cache_path.exists():
            return {}
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def get_bytes(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
        cache_ttl_seconds: int | None = 24 * 3600,
        timeout: float | None = None,
        max_bytes: int | None = None,
    ) -> bytes:
        """GET ``url``; answer from the disk cache when fresh.

        Redirects are followed by hand (at most ``MAX_REDIRECTS``), each hop
        re-checked against the allowlist, so the SEC identity never follows a
        redirect to an unlisted host. A response that ended on a different
        host is returned but never cached under the original URL's key. A body
        over ``max_bytes`` raises :class:`ResponseTooLarge`.
        """
        self._check_allowlist(url)
        cache_path = self._cache_path(url, params)
        if use_cache and cache_path.exists():
            if cache_ttl_seconds is None:
                return cache_path.read_bytes()
            age_seconds = time.time() - cache_path.stat().st_mtime
            if age_seconds <= cache_ttl_seconds:
                return cache_path.read_bytes()

        require_network(url, self.cfg)
        self._require_valid_sec_user_agent(url)
        timeout = timeout or self.cfg.http_timeout_seconds
        backoff = self.cfg.sec_backoff_seconds
        last_exc: Exception | None = None
        for attempt in range(self.cfg.sec_max_retries + 1):
            self._consume_domain_budget(url)
            self.rate_limiter.acquire()
            try:
                response, final_host = self._get_following_redirects(url, params, timeout)
                with _GLOBAL_METRICS_LOCK:
                    _GLOBAL_METRICS["request_count"] += 1
                if response.status_code == 429:
                    with _GLOBAL_METRICS_LOCK:
                        _GLOBAL_METRICS["throttled_count"] += 1
                    sleep_s = backoff * (2**attempt)
                    time.sleep(sleep_s)
                    continue
                if response.status_code >= 500:
                    sleep_s = backoff * (2**attempt)
                    time.sleep(sleep_s)
                    continue
                response.raise_for_status()
                payload = _read_capped(response, max_bytes or DEFAULT_MAX_RESPONSE_BYTES, url)
                same_host = final_host == (urlparse(url).hostname or "").lower()
                if use_cache and same_host:
                    cache_path.write_bytes(payload)
                return payload
            except (ResponseTooLarge, TooManyRedirects):
                raise
            except requests.HTTPError as exc:
                # A client error (404 unknown CIK, 403 refused identity) is an
                # answer, not a transient fault: retrying it only stalls the
                # caller for the whole backoff ladder before failing anyway.
                # 429 never reaches here; it is handled above.
                status = exc.response.status_code if exc.response is not None else None
                if status is not None and 400 <= status < 500:
                    raise
                last_exc = exc
                sleep_s = backoff * (2**attempt)
                time.sleep(sleep_s)
            except requests.RequestException as exc:
                last_exc = exc
                sleep_s = backoff * (2**attempt)
                time.sleep(sleep_s)

        if last_exc:
            raise last_exc
        raise RuntimeError(f"GET failed after retries: {url}")

    def _get_following_redirects(
        self, url: str, params: dict[str, Any] | None, timeout: float
    ) -> tuple[requests.Response, str]:
        current_url, current_params = url, params
        for hop in range(MAX_REDIRECTS + 1):
            if hop:
                self._consume_domain_budget(current_url)
            response = self.session.get(
                current_url,
                params=current_params,
                timeout=timeout,
                allow_redirects=False,
                stream=True,
            )
            location = (getattr(response, "headers", None) or {}).get("Location")
            if response.status_code not in _REDIRECT_STATUSES or not location:
                return response, (urlparse(current_url).hostname or "").lower()
            target = urljoin(getattr(response, "url", None) or current_url, location)
            _close_quietly(response)
            self._check_allowlist(target)
            self._require_valid_sec_user_agent(target)
            current_url, current_params = target, None
        raise TooManyRedirects(
            f"more than {MAX_REDIRECTS} redirects from {sanitize_url_credentials(url)}"
        )

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
        cache_ttl_seconds: int | None = 24 * 3600,
    ) -> dict[str, Any]:
        raw = self.get_bytes(
            url,
            params=params,
            use_cache=use_cache,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        return json.loads(raw.decode("utf-8"))

    def metrics(self) -> dict[str, int]:
        with _GLOBAL_METRICS_LOCK:
            payload: dict[str, int] = dict(_GLOBAL_METRICS)
            for host, count in _GLOBAL_DOMAIN_COUNTS.items():
                payload[f"domain_count:{host}"] = count
            return payload


def _close_quietly(response: requests.Response) -> None:
    try:
        response.close()
    except Exception:  # noqa: BLE001 - a fake or already-closed response
        pass


def _read_capped(response: requests.Response, max_bytes: int, url: str) -> bytes:
    def _too_large() -> ResponseTooLarge:
        _close_quietly(response)
        return ResponseTooLarge(
            f"response from {sanitize_url_credentials(url)} is larger than "
            f"{max_bytes // (1024 * 1024)} MB; refusing to download it"
        )

    try:
        declared = int((getattr(response, "headers", None) or {}).get("Content-Length") or 0)
    except (TypeError, ValueError):
        declared = 0
    if declared > max_bytes:
        raise _too_large()
    if not isinstance(response, requests.Response) or response._content is not False:
        body = response.content  # already in memory (a stand-in or non-streamed response)
        if len(body) > max_bytes:
            raise _too_large()
        return body
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=1 << 16):
        total += len(chunk)
        if total > max_bytes:
            raise _too_large()
        chunks.append(chunk)
    return b"".join(chunks)


def reset_domain_request_counts() -> None:
    """Zero the per-host request counters that the domain budgets are checked against.

    The budgets are sized for one batch run. A long-lived process (the MCP
    server) calls this at the start of each unit of work, so a budget caps a
    single request rather than the lifetime of the process.
    """
    with _GLOBAL_METRICS_LOCK:
        _GLOBAL_DOMAIN_COUNTS.clear()


def effective_sec_domain_budgets() -> dict[str, int]:
    cfg = get_config()
    return {
        "sec.gov": int(cfg.max_requests_sec_domain),
        "data.sec.gov": int(cfg.max_requests_data_sec_domain),
        "www.sec.gov": int(cfg.max_requests_www_sec_domain),
    }


@contextmanager
def temporary_sec_domain_budget(*, sec_budget: int | None = None):
    if sec_budget is None:
        yield effective_sec_domain_budgets()
        return

    cfg = get_config()
    target = max(1, int(sec_budget))
    with _GLOBAL_BUDGET_OVERRIDE_LOCK:
        prior = (
            int(cfg.max_requests_sec_domain),
            int(cfg.max_requests_data_sec_domain),
            int(cfg.max_requests_www_sec_domain),
        )
        cfg.max_requests_sec_domain = max(prior[0], target)
        cfg.max_requests_data_sec_domain = max(prior[1], target)
        cfg.max_requests_www_sec_domain = max(prior[2], target)
        try:
            yield effective_sec_domain_budgets()
        finally:
            cfg.max_requests_sec_domain = prior[0]
            cfg.max_requests_data_sec_domain = prior[1]
            cfg.max_requests_www_sec_domain = prior[2]
