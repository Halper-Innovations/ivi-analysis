from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.util.credential_hygiene import InvalidSecUserAgentError
from app.util.http import DomainBudgetExceeded, HttpClient


FACTS_REASON_CODES = {
    "CACHE_HIT",
    "FETCH_OK",
    "FETCH_4XX",
    "FETCH_5XX",
    "BUDGET_EXHAUSTED",
    "SEC_USER_AGENT_INVALID",
    "OFFLINE_NO_CACHE",
    "EXCEPTION",
}


# A cached payload younger than this is served without a request. The facts
# writer and the HTTP cache use the same 24h horizon; SEC fair access asks
# clients not to re-download unchanged multi-megabyte payloads.
COMPANYFACTS_CACHE_TTL_SECONDS = 24 * 3600


def normalize_cik(cik: str | int | None) -> str:
    token = str(cik or "").strip()
    digits = "".join(ch for ch in token if ch.isdigit())
    return digits.zfill(10) if digits else ""


def companyfacts_cache_path(
    cik: str | int,
    cfg: AppConfig | None = None,
    *,
    create_parent: bool = True,
) -> Path:
    cfg = cfg or get_config()
    cik_norm = normalize_cik(cik)
    base = cfg.cache_dir / "companyfacts"
    if create_parent:
        base.mkdir(parents=True, exist_ok=True)
    return base / f"{cik_norm}.json"


def _copy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _network_disabled(cfg: AppConfig) -> bool:
    # VOE_NET_PROVIDER is the only network switch. The LLM provider setting
    # says nothing about network access: SEC companyfacts are free and the
    # deterministic valuation needs them with no LLM configured.
    net_provider = str(os.getenv("VOE_NET_PROVIDER", cfg.net_provider)).strip().lower()
    return net_provider == "disabled"


def _cache_is_fresh(meta: dict[str, Any]) -> bool:
    retrieved = str(meta.get("retrieved_at") or "")
    try:
        retrieved_at = datetime.fromisoformat(retrieved.replace("Z", "+00:00"))
    except ValueError:
        return False
    if retrieved_at.tzinfo is None:
        return False
    age = (datetime.now(timezone.utc) - retrieved_at).total_seconds()
    return 0 <= age <= COMPANYFACTS_CACHE_TTL_SECONDS


def _read_cached_companyfacts(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not path.exists():
        return None, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, {}
    if not isinstance(payload, dict):
        return None, {}

    # Support both wrapped and raw SEC payload cache formats.
    if isinstance(payload.get("companyfacts"), dict):
        companyfacts = payload.get("companyfacts")
        metadata = {
            "retrieved_at": payload.get("retrieved_at"),
            "source_url": payload.get("source_url"),
            "http_status": payload.get("http_status"),
            "size_bytes": payload.get("size_bytes"),
        }
        return _copy_payload(companyfacts), metadata

    if isinstance(payload.get("facts"), dict):
        return _copy_payload(payload), {}
    return None, {}


def _write_cached_companyfacts(
    *,
    path: Path,
    cik: str,
    source_url: str,
    http_status: int,
    companyfacts: dict[str, Any],
) -> dict[str, Any]:
    raw = json.dumps(companyfacts, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload = {
        "cik": cik,
        "retrieved_at": utc_now_iso(),
        "source_url": source_url,
        "http_status": int(http_status),
        "size_bytes": len(raw),
        "companyfacts": companyfacts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def fetch_company_facts(
    cik: str | int,
    *,
    user_agent: str | None = None,
    sec_budget: int | None = None,
    cfg: AppConfig | None = None,
    cache_only: bool = False,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    cik_norm = normalize_cik(cik)
    cache_path = companyfacts_cache_path(
        cik_norm,
        cfg=cfg,
        create_parent=not cache_only,
    )
    source_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_norm}.json"
    now = utc_now_iso()

    base = {
        "cik": cik_norm,
        "status": "MISSING",
        "reason_code": "EXCEPTION",
        "reason_detail": "",
        "source_resolution": "unknown",
        "cache_path": str(cache_path),
        "source_url": source_url,
        "http_status": None,
        "size_bytes": None,
        "retrieved_at": None,
        "companyfacts": None,
        "network_attempted": False,
        "attempts_made": 0,
        "retries_configured": 0,
        "backoff_seconds": None,
        "timeout_seconds": None,
        "last_exception_type": None,
        "last_exception_message": None,
        "derived_from": [str(cache_path)],
        "generated_at": now,
    }

    if not cik_norm:
        base["reason_code"] = "EXCEPTION"
        base["reason_detail"] = "CIK is missing or invalid."
        return base

    cached_payload, cached_meta = _read_cached_companyfacts(cache_path)
    if cache_only or _network_disabled(cfg):
        if isinstance(cached_payload, dict):
            base.update(
                {
                    "status": "OK",
                    "reason_code": "CACHE_HIT",
                    "reason_detail": (
                        "Resolved from companyfacts cache in explicit cache-only mode."
                        if cache_only
                        else (
                            "Resolved from companyfacts cache while network provider is disabled."
                        )
                    ),
                    "source_resolution": "companyfacts_cache",
                    "http_status": cached_meta.get("http_status"),
                    "size_bytes": cached_meta.get("size_bytes"),
                    "retrieved_at": cached_meta.get("retrieved_at"),
                    "companyfacts": cached_payload,
                }
            )
            return base
        base.update(
            {
                "reason_code": "OFFLINE_NO_CACHE",
                "reason_detail": (
                    "Explicit cache-only mode found no cached companyfacts payload."
                    if cache_only
                    else (
                        "Network provider disabled and no cached companyfacts payload is available."
                    )
                ),
            }
        )
        return base

    if isinstance(cached_payload, dict) and _cache_is_fresh(cached_meta):
        base.update(
            {
                "status": "OK",
                "reason_code": "CACHE_HIT",
                "reason_detail": "Resolved from a companyfacts cache fetched within the last 24h.",
                "source_resolution": "companyfacts_cache",
                "http_status": cached_meta.get("http_status"),
                "size_bytes": cached_meta.get("size_bytes"),
                "retrieved_at": cached_meta.get("retrieved_at"),
                "companyfacts": cached_payload,
            }
        )
        return base

    if sec_budget is not None and int(sec_budget) <= 0:
        base.update(
            {
                "reason_code": "BUDGET_EXHAUSTED",
                "reason_detail": "SEC request budget exhausted before companyfacts fetch.",
            }
        )
        return base

    http = HttpClient(cfg)
    if user_agent:
        http.session.headers.update({"User-Agent": str(user_agent)})
    elif cfg.sec_user_agent:
        http.session.headers.update({"User-Agent": str(cfg.sec_user_agent)})

    try:
        http._require_valid_sec_user_agent(source_url)  # noqa: SLF001
    except InvalidSecUserAgentError as exc:
        base.update(
            {
                "reason_code": "SEC_USER_AGENT_INVALID",
                "reason_detail": str(exc),
                "last_exception_type": type(exc).__name__,
                "last_exception_message": str(exc),
            }
        )
        return base

    retries = max(0, int(cfg.sec_max_retries))
    backoff = max(0.1, float(cfg.sec_backoff_seconds))
    timeout = max(5.0, float(cfg.http_timeout_seconds))
    base["retries_configured"] = int(retries)
    base["backoff_seconds"] = float(backoff)
    base["timeout_seconds"] = float(timeout)
    last_exc: Exception | None = None
    last_status: int | None = None

    for attempt in range(retries + 1):
        base["network_attempted"] = True
        base["attempts_made"] = int(attempt + 1)
        try:
            http._check_allowlist(source_url)  # noqa: SLF001
            http._consume_domain_budget(source_url)  # noqa: SLF001
            http.rate_limiter.acquire()
            response = http.session.get(source_url, timeout=timeout, allow_redirects=False)
            last_status = int(response.status_code)
        except DomainBudgetExceeded as exc:
            base.update(
                {
                    "reason_code": "BUDGET_EXHAUSTED",
                    "reason_detail": str(exc),
                }
            )
            if isinstance(cached_payload, dict):
                base.update(
                    {
                        "status": "OK",
                        "reason_code": "CACHE_HIT",
                        "reason_detail": "Domain budget exhausted; used cached companyfacts payload.",
                        "source_resolution": "companyfacts_cache",
                        "http_status": cached_meta.get("http_status"),
                        "size_bytes": cached_meta.get("size_bytes"),
                        "retrieved_at": cached_meta.get("retrieved_at"),
                        "companyfacts": cached_payload,
                    }
                )
            return base
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            base["last_exception_type"] = type(exc).__name__
            base["last_exception_message"] = str(exc)
            if isinstance(cached_payload, dict):
                base.update(
                    {
                        "status": "OK",
                        "reason_code": "CACHE_HIT",
                        "reason_detail": "SEC fetch failed; used cached companyfacts payload.",
                        "source_resolution": "companyfacts_cache",
                        "http_status": cached_meta.get("http_status"),
                        "size_bytes": cached_meta.get("size_bytes"),
                        "retrieved_at": cached_meta.get("retrieved_at"),
                        "companyfacts": cached_payload,
                    }
                )
                return base
            if attempt < retries:
                time.sleep(backoff * (2**attempt))
                continue
            break

        if last_status is None:
            continue
        if 200 <= last_status < 300:
            try:
                companyfacts = response.json()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                break
            if not isinstance(companyfacts, dict):
                last_exc = RuntimeError("Companyfacts payload is not a JSON object.")
                break
            cache_meta = _write_cached_companyfacts(
                path=cache_path,
                cik=cik_norm,
                source_url=source_url,
                http_status=last_status,
                companyfacts=companyfacts,
            )
            base.update(
                {
                    "status": "OK",
                    "reason_code": "FETCH_OK",
                    "reason_detail": "Fetched companyfacts from SEC and updated local cache.",
                    "source_resolution": "companyfacts_fetch",
                    "http_status": int(last_status),
                    "size_bytes": cache_meta.get("size_bytes"),
                    "retrieved_at": cache_meta.get("retrieved_at"),
                    "companyfacts": companyfacts,
                }
            )
            return base

        if 500 <= last_status < 600:
            if attempt < retries:
                time.sleep(backoff * (2**attempt))
                continue
            base.update(
                {
                    "reason_code": "FETCH_5XX",
                    "reason_detail": f"SEC companyfacts returned HTTP {last_status}.",
                    "http_status": int(last_status),
                }
            )
            break

        if 400 <= last_status < 500:
            base.update(
                {
                    "reason_code": "FETCH_4XX",
                    "reason_detail": f"SEC companyfacts returned HTTP {last_status}.",
                    "http_status": int(last_status),
                }
            )
            break

    if isinstance(cached_payload, dict):
        base.update(
            {
                "status": "OK",
                "reason_code": "CACHE_HIT",
                "reason_detail": "SEC fetch failed; used cached companyfacts payload.",
                "source_resolution": "companyfacts_cache",
                "http_status": cached_meta.get("http_status"),
                "size_bytes": cached_meta.get("size_bytes"),
                "retrieved_at": cached_meta.get("retrieved_at"),
                "companyfacts": cached_payload,
            }
        )
        return base

    if base["reason_code"] not in {"FETCH_4XX", "FETCH_5XX"}:
        base.update(
            {
                "reason_code": "EXCEPTION",
                "reason_detail": str(last_exc) if last_exc else "Unknown companyfacts fetch error.",
                "http_status": int(last_status) if isinstance(last_status, int) else None,
            }
        )
    if last_exc is not None:
        base["last_exception_type"] = type(last_exc).__name__
        base["last_exception_message"] = str(last_exc)
    return base
