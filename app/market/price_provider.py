from __future__ import annotations

import csv
import io
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlencode

import requests

from app.config import AppConfig, get_config
from app.db import get_db, utc_now_iso
from app.util.credential_hygiene import (
    redact_credential_text,
    sanitize_json_value,
    sanitize_url_credentials,
)
from app.util.http import DomainBudgetExceeded, HttpClient


PriceConfidence = Literal["HIGH", "MEDIUM", "LOW"]
ReasonCode = Literal[
    "CACHE_HIT",
    "CACHE_MISS",
    "PROVIDER_OK",
    "DNS_FAILURE",
    "TLS_FAILURE",
    "TIMEOUT",
    "HTTP_4XX",
    "HTTP_5XX",
    "RATE_LIMIT",
    "PARSE_ERROR",
    "SYMBOL_NOT_FOUND",
    "PROVIDER_NO_DATA",
    "NON_TRADING_DAY_NO_FALLBACK",
    "OFFLINE_NO_CACHE",
    "BUDGET_EXHAUSTED",
]


@dataclass(eq=True, frozen=True)
class PriceSnapshot:
    ticker: str
    as_of_date: str
    price: float
    currency: str = "USD"
    source: str = "stooq"
    retrieved_at: str = ""
    url: str | None = None
    confidence: PriceConfidence = "MEDIUM"
    # Unadjusted close at as_of_date when the provider exposes one (EODHD).
    # `price` stays the adjusted close: return legs need one consistent
    # adjusted series, while deploy classification against as-of per-share
    # anchors needs the raw quote (split-basis re-basing decision).
    raw_price: float | None = None
    # Share volume for as_of_date when the provider history carries it
    # (Stooq/EODHD daily rows do) — the input to the dollar-ADV liquidity
    # layer. None when the source has no volume (quotes, caches, fallbacks).
    volume: float | None = None

    def __post_init__(self) -> None:
        sanitized_url = sanitize_url_credentials(self.url)
        if sanitized_url != self.url:
            object.__setattr__(self, "url", sanitized_url)


@dataclass(eq=True, frozen=True)
class SymbolOverride:
    ticker: str
    symbol: str
    valid_from: date | None = None
    valid_to: date | None = None


class PriceProvider(Protocol):
    provider_name: str

    def get_price_asof(self, ticker: str, as_of_date: str) -> PriceSnapshot | None:
        ...

    def get_last_diagnostic(self, ticker: str, as_of_date: str) -> dict[str, Any] | None:
        ...


def _parse_date(value: str) -> date | None:
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except Exception:
        return None


def _parse_iso_datetime(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone()


def _cache_ttl_seconds(cfg: AppConfig | None) -> int:
    effective = int(getattr(cfg, "quote_ttl_seconds", 86400) or 86400)
    return max(1, effective)


def _snapshot_is_fresh(snapshot: PriceSnapshot | None, *, ttl_seconds: int) -> bool:
    if snapshot is None:
        return False
    retrieved = _parse_iso_datetime(snapshot.retrieved_at)
    if retrieved is None:
        return False
    now = datetime.now(retrieved.tzinfo) if retrieved.tzinfo is not None else datetime.now()
    return (now - retrieved).total_seconds() <= max(1, int(ttl_seconds))


def _normalize_reason(code: str | None) -> ReasonCode:
    value = str(code or "").strip().upper()
    aliases = {
        "PROVIDER_HTTP_ERROR": "HTTP_4XX",
        "PROVIDER_PARSE_ERROR": "PARSE_ERROR",
        "SYMBOL_UNMAPPED": "SYMBOL_NOT_FOUND",
        "UNEXPECTED_EXCEPTION": "PROVIDER_NO_DATA",
    }
    value = aliases.get(value, value)
    valid = {
        "CACHE_HIT",
        "CACHE_MISS",
        "PROVIDER_OK",
        "DNS_FAILURE",
        "TLS_FAILURE",
        "TIMEOUT",
        "HTTP_4XX",
        "HTTP_5XX",
        "RATE_LIMIT",
        "PARSE_ERROR",
        "SYMBOL_NOT_FOUND",
        "PROVIDER_NO_DATA",
        "NON_TRADING_DAY_NO_FALLBACK",
        "OFFLINE_NO_CACHE",
        "BUDGET_EXHAUSTED",
    }
    if value in valid:
        return value  # type: ignore[return-value]
    return "PROVIDER_NO_DATA"


def _as_snapshot(payload: dict[str, object] | None) -> PriceSnapshot | None:
    if not isinstance(payload, dict):
        return None
    ticker = str(payload.get("ticker") or "").upper()
    as_of_date = str(payload.get("as_of_date") or "")
    price = payload.get("price")
    if not ticker or not as_of_date or not isinstance(price, (int, float)):
        return None
    confidence = str(payload.get("confidence") or "LOW").upper()
    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        confidence = "LOW"
    raw_price = payload.get("raw_price")
    volume = payload.get("volume")
    return PriceSnapshot(
        ticker=ticker,
        as_of_date=as_of_date,
        price=float(price),
        currency=str(payload.get("currency") or "USD"),
        source=str(payload.get("source") or "UNKNOWN"),
        retrieved_at=str(payload.get("retrieved_at") or ""),
        url=str(payload.get("url")) if payload.get("url") else None,
        confidence=confidence,  # type: ignore[arg-type]
        raw_price=float(raw_price) if isinstance(raw_price, (int, float)) else None,
        volume=float(volume) if isinstance(volume, (int, float)) else None,
    )


_ERROR_DETAIL_MAX_CHARS = 240
_RETRYABLE_FAILURE_CODES = {"DNS_FAILURE", "TLS_FAILURE", "TIMEOUT", "HTTP_5XX", "RATE_LIMIT"}
_PRIMARY_RETRY_BACKOFF_SECONDS = (0.2, 0.6)
_OFFLINE_TERMINAL_SUGGESTION = "Run companyfacts-fetch and price-fetch in an online environment once to seed caches, then rerun depth."


def _truncate_error_detail(value: str | None, *, max_chars: int = _ERROR_DETAIL_MAX_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def price_reason_suggestion(reason_code: str | None) -> str:
    code = _normalize_reason(reason_code)
    mapping = {
        "DNS_FAILURE": "Retry later, check connectivity, or increase fallback-days.",
        "TLS_FAILURE": "Retry later, check connectivity, or increase fallback-days.",
        "TIMEOUT": "Retry later, check connectivity, or increase fallback-days.",
        "RATE_LIMIT": "Reduce workers or increase budget.",
        "SYMBOL_NOT_FOUND": "Add symbol override entry to config/price_symbol_overrides.csv.",
        "OFFLINE_NO_CACHE": _OFFLINE_TERMINAL_SUGGESTION,
        "NON_TRADING_DAY_NO_FALLBACK": "Increase fallback-days to walk back to a prior trading day.",
        "BUDGET_EXHAUSTED": "Increase request budget or retry after budget reset.",
        "HTTP_4XX": "Inspect provider request parameters and symbol mapping.",
        "HTTP_5XX": "Retry later or switch provider endpoint.",
        "PARSE_ERROR": "Inspect provider response parsing assumptions.",
        "PROVIDER_NO_DATA": "Retry with symbol override and fallback-days.",
        "CACHE_MISS": "Run price-fetch to prewarm cache.",
        "CACHE_HIT": "",
        "PROVIDER_OK": "",
    }
    return mapping.get(code, "Inspect provider_attempts and cache metadata for this ticker.")


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _parse_candidate_asof(value: Any) -> str | None:
    token = str(value or "").strip()
    if not token:
        return None
    if _parse_date(token) is None:
        return None
    return token


def _snapshot_from_historical_valuation_payload(payload: dict[str, Any], *, ticker: str) -> PriceSnapshot | None:
    input_snapshot = payload.get("input_snapshot") if isinstance(payload.get("input_snapshot"), dict) else {}
    price_value = _to_float((input_snapshot or {}).get("current_price"))
    if price_value is None or price_value <= 0:
        return None
    used_asof = (
        _parse_candidate_asof((input_snapshot or {}).get("price_asof_used"))
        or _parse_candidate_asof((input_snapshot or {}).get("price_as_of_date"))
        or _parse_candidate_asof(payload.get("as_of_date"))
    )
    if used_asof is None:
        return None
    retrieved_at = str(payload.get("generated_at") or payload.get("created_at") or f"{used_asof}T00:00:00+00:00")
    return PriceSnapshot(
        ticker=ticker.upper(),
        as_of_date=used_asof,
        price=price_value,
        currency=str((input_snapshot or {}).get("currency") or "USD"),
        source="historical_run_artifacts",
        retrieved_at=retrieved_at,
        url=None,
        confidence="MEDIUM",
    )


def _snapshot_from_historical_valuation_coverage_entry(entry: dict[str, Any], *, ticker: str) -> PriceSnapshot | None:
    if str(entry.get("price_status") or "").upper() != "OK":
        return None
    price_value = (
        _to_float(entry.get("price_value"))
        or _to_float(entry.get("current_price"))
        or _to_float(entry.get("price"))
    )
    if price_value is None or price_value <= 0:
        return None
    used_asof = _parse_candidate_asof(entry.get("price_asof_used"))
    if used_asof is None:
        return None
    return PriceSnapshot(
        ticker=ticker.upper(),
        as_of_date=used_asof,
        price=price_value,
        currency=str(entry.get("currency") or "USD"),
        source="historical_run_artifacts",
        retrieved_at=f"{used_asof}T00:00:00+00:00",
        url=None,
        confidence="MEDIUM",
    )


def _snapshot_from_historical_price_coverage_entry(entry: dict[str, Any], *, ticker: str) -> PriceSnapshot | None:
    result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
    if str((result or {}).get("status") or "").upper() != "OK":
        return None
    output_fields = entry.get("output_fields") if isinstance(entry.get("output_fields"), dict) else {}
    price_value = _to_float((output_fields or {}).get("current_price"))
    if price_value is None or price_value <= 0:
        return None
    used_asof = (
        _parse_candidate_asof((output_fields or {}).get("price_asof_used"))
        or _parse_candidate_asof(entry.get("asof_final_used"))
    )
    if used_asof is None:
        return None
    source = str((output_fields or {}).get("price_source") or "historical_run_artifacts")
    confidence = str((output_fields or {}).get("confidence") or "MEDIUM").upper()
    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        confidence = "MEDIUM"
    return PriceSnapshot(
        ticker=ticker.upper(),
        as_of_date=used_asof,
        price=price_value,
        currency="USD",
        source=source,
        retrieved_at=f"{used_asof}T00:00:00+00:00",
        url=None,
        confidence=confidence,  # type: ignore[arg-type]
    )


# Staleness ceilings, measured against the REQUESTED as-of date (never
# wall-clock), so offline/backtest replay stays deterministic. A cached quote
# older than the ceiling is NO_PRICE, not a price.
DB_QUOTE_MAX_AGE_DAYS = 7
HISTORICAL_RUN_MAX_AGE_DAYS = 30


def resolve_price_from_historical_runs(
    ticker: str,
    as_of_date: str,
    *,
    sectors_dir: Path | None = None,
    max_runs: int = 50,
    max_age_days: int = HISTORICAL_RUN_MAX_AGE_DAYS,
) -> PriceSnapshot | None:
    """
    Resolve price deterministically from prior sector run artifacts (no network).

    Run directory ordering is deterministic: descending lexicographic directory name,
    then source precedence inside each run:
    valuation_<T>.json > valuation_coverage.json > price_coverage.json.

    Candidates more than ``max_age_days`` before the requested as-of are
    rejected — an any-age artifact price backing a trigger is a phantom.
    """
    target = _parse_date(as_of_date)
    ticker_norm = str(ticker or "").strip().upper()
    if target is None or not ticker_norm:
        return None
    min_used = target - timedelta(days=max(0, int(max_age_days)))
    root = sectors_dir or get_config().sectors_dir
    if not root.exists():
        return None

    run_dirs = sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name, reverse=True)
    source_priority = {
        "valuation_file": 3,
        "valuation_coverage": 2,
        "price_coverage": 1,
    }
    candidates: list[tuple[date, int, str, PriceSnapshot]] = []
    for run_dir in run_dirs[: max(1, int(max_runs))]:
        run_id = run_dir.name
        valuation_path = run_dir / f"valuation_{ticker_norm}.json"
        if valuation_path.exists():
            try:
                valuation_payload = json.loads(valuation_path.read_text(encoding="utf-8"))
            except Exception:
                valuation_payload = {}
            if isinstance(valuation_payload, dict):
                snapshot = _snapshot_from_historical_valuation_payload(valuation_payload, ticker=ticker_norm)
                if snapshot is not None:
                    used = _parse_date(snapshot.as_of_date)
                    if used is not None and min_used <= used <= target:
                        candidates.append((used, source_priority["valuation_file"], run_id, snapshot))

        valuation_coverage_path = run_dir / "valuation_coverage.json"
        if valuation_coverage_path.exists():
            try:
                coverage_payload = json.loads(valuation_coverage_path.read_text(encoding="utf-8"))
            except Exception:
                coverage_payload = {}
            entries = coverage_payload.get("entries") if isinstance(coverage_payload, dict) else None
            if isinstance(entries, list):
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    if str(entry.get("ticker") or "").strip().upper() != ticker_norm:
                        continue
                    snapshot = _snapshot_from_historical_valuation_coverage_entry(entry, ticker=ticker_norm)
                    if snapshot is None:
                        continue
                    used = _parse_date(snapshot.as_of_date)
                    if used is not None and min_used <= used <= target:
                        candidates.append((used, source_priority["valuation_coverage"], run_id, snapshot))
                    break

        price_coverage_path = run_dir / "price_coverage.json"
        if price_coverage_path.exists():
            try:
                coverage_payload = json.loads(price_coverage_path.read_text(encoding="utf-8"))
            except Exception:
                coverage_payload = {}
            entries = coverage_payload.get("entries") if isinstance(coverage_payload, dict) else None
            if isinstance(entries, list):
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    if str(entry.get("ticker") or "").strip().upper() != ticker_norm:
                        continue
                    snapshot = _snapshot_from_historical_price_coverage_entry(entry, ticker=ticker_norm)
                    if snapshot is None:
                        continue
                    used = _parse_date(snapshot.as_of_date)
                    if used is not None and min_used <= used <= target:
                        candidates.append((used, source_priority["price_coverage"], run_id, snapshot))
                    break

    if not candidates:
        return None
    candidates.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    best = candidates[0][3]
    confidence: PriceConfidence = "HIGH" if best.as_of_date == str(as_of_date) else "MEDIUM"
    return PriceSnapshot(
        ticker=best.ticker,
        as_of_date=best.as_of_date,
        price=float(best.price),
        currency=best.currency,
        source="historical_run_artifacts",
        retrieved_at=best.retrieved_at or f"{best.as_of_date}T00:00:00+00:00",
        url=best.url,
        confidence=confidence,
    )


def classify_price_error(
    exc: Exception | str,
    *,
    additional_sensitive_names: tuple[str, ...] = (),
) -> tuple[ReasonCode, bool, str, str]:
    error_detail = _truncate_error_detail(
        redact_credential_text(
            str(exc),
            additional_sensitive_names=additional_sensitive_names,
        )
    )
    text = error_detail.lower()
    reason: ReasonCode = "PROVIDER_NO_DATA"
    retryable = False

    if isinstance(exc, DomainBudgetExceeded) or "budget" in text:
        reason = "BUDGET_EXHAUSTED"
    elif isinstance(exc, requests.Timeout):
        reason = "TIMEOUT"
        retryable = True
    elif isinstance(exc, requests.exceptions.SSLError):
        reason = "TLS_FAILURE"
        retryable = True
    elif isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        status = int(getattr(response, "status_code", 0) or 0)
        if status == 429:
            reason = "RATE_LIMIT"
            retryable = True
        elif 500 <= status:
            reason = "HTTP_5XX"
            retryable = True
        elif 400 <= status:
            reason = "HTTP_4XX"
    elif isinstance(exc, requests.ConnectionError):
        if any(marker in text for marker in ("ssl", "tls", "certificate")):
            reason = "TLS_FAILURE"
        else:
            reason = "DNS_FAILURE"
        retryable = True
    elif isinstance(exc, requests.RequestException):
        if any(marker in text for marker in ("timeout", "timed out")):
            reason = "TIMEOUT"
        elif any(marker in text for marker in ("ssl", "tls", "certificate")):
            reason = "TLS_FAILURE"
        elif any(marker in text for marker in ("name resolution", "nodename nor servname", "temporary failure in name resolution")):
            reason = "DNS_FAILURE"
        else:
            reason = "PROVIDER_NO_DATA"
        retryable = reason in {"DNS_FAILURE", "TLS_FAILURE", "TIMEOUT"}
    elif any(marker in text for marker in ("timeout", "timed out")):
        reason = "TIMEOUT"
        retryable = True
    elif any(marker in text for marker in ("name resolution", "nodename nor servname", "temporary failure in name resolution")):
        reason = "DNS_FAILURE"
        retryable = True
    elif any(marker in text for marker in ("ssl", "tls", "certificate")):
        reason = "TLS_FAILURE"
        retryable = True
    elif "429" in text or "rate limit" in text or "too many requests" in text:
        reason = "RATE_LIMIT"
        retryable = True
    elif "5xx" in text or "503" in text or "502" in text or "500" in text:
        reason = "HTTP_5XX"
        retryable = True
    elif "4xx" in text or "404" in text or "400" in text:
        reason = "HTTP_4XX"
    elif "parse" in text or "csv" in text or "json" in text:
        reason = "PARSE_ERROR"
    elif "symbol" in text and ("not found" in text or "unknown" in text):
        reason = "SYMBOL_NOT_FOUND"
    elif "no data" in text:
        reason = "PROVIDER_NO_DATA"

    suggestion = price_reason_suggestion(reason)
    return reason, retryable, suggestion, error_detail


def _snapshot_payload(snapshot: PriceSnapshot) -> dict[str, Any]:
    return {
        "ticker": snapshot.ticker,
        "as_of_date": snapshot.as_of_date,
        "price": snapshot.price,
        "currency": snapshot.currency,
        "source": snapshot.source,
        "retrieved_at": snapshot.retrieved_at,
        "url": sanitize_url_credentials(snapshot.url),
        "confidence": snapshot.confidence,
        "raw_price": snapshot.raw_price,
        "volume": snapshot.volume,
    }


def _load_run_scoped_output(path: Path, *, requested_as_of_date: str) -> tuple[PriceSnapshot | None, dict[str, Any] | None]:
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    if not isinstance(payload, dict):
        return None, None
    if str(payload.get("status") or "").upper() != "OK":
        return None, payload.get("diagnostic") if isinstance(payload.get("diagnostic"), dict) else None
    requested = str(payload.get("requested_as_of_date") or "")
    # Exact date match required: an undated artifact is not evidence for a dated request.
    if not requested.strip() or requested.strip() != str(requested_as_of_date).strip():
        return None, payload.get("diagnostic") if isinstance(payload.get("diagnostic"), dict) else None
    raw_snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else None
    raw_price = (raw_snapshot or {}).get("price")
    if not (
        isinstance(raw_price, (int, float))
        and not isinstance(raw_price, bool)
        and math.isfinite(float(raw_price))
        and float(raw_price) > 0
    ):
        return None, payload.get("diagnostic") if isinstance(payload.get("diagnostic"), dict) else None
    snapshot = _as_snapshot(raw_snapshot)
    diag = payload.get("diagnostic") if isinstance(payload.get("diagnostic"), dict) else None
    if isinstance(diag, dict):
        diag = sanitize_json_value(diag)
    return snapshot, diag


def _load_disk_cache_snapshot(path: Path, *, requested_as_of_date: str, ttl_seconds: int = 86400) -> PriceSnapshot | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return None
    candidates: list[tuple[str, PriceSnapshot]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("requested_as_of_date") or "") != str(requested_as_of_date):
            continue
        snapshot = _as_snapshot(entry.get("snapshot") if isinstance(entry.get("snapshot"), dict) else None)
        if snapshot is None:
            continue
        if not _snapshot_is_fresh(snapshot, ttl_seconds=ttl_seconds):
            continue
        candidates.append((str(entry.get("source") or "").lower(), snapshot))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].as_of_date, item[1].source))
    return candidates[0][1]


def _load_db_quote_snapshot(
    *, ticker: str, as_of_date: str, max_age_days: int = DB_QUOTE_MAX_AGE_DAYS
) -> PriceSnapshot | None:
    try:
        with get_db() as conn:
            row = conn.execute(
                """
                SELECT provider, price, status, source_url, as_of_date, fetched_at, currency
                FROM price_quotes
                WHERE ticker = ?
                  AND as_of_date <= ?
                  AND status = 'OK'
                  AND price IS NOT NULL
                ORDER BY as_of_date DESC, fetched_at DESC
                LIMIT 1
                """,
                (ticker.upper(), as_of_date),
            ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    price = row["price"]
    if not isinstance(price, (int, float)):
        return None
    used_asof = str(row["as_of_date"] or "").strip()
    if not used_asof:
        return None
    # An any-age cached quote must not back an as-of read — over-ceiling
    # rows are NO_PRICE, not a price.
    used_date = _parse_date(used_asof)
    target_date = _parse_date(str(as_of_date))
    if used_date is None or target_date is None:
        return None
    if (target_date - used_date).days > max(0, int(max_age_days)):
        return None
    confidence: PriceConfidence = "HIGH" if used_asof == str(as_of_date) else "MEDIUM"
    return PriceSnapshot(
        ticker=ticker.upper(),
        as_of_date=used_asof,
        price=float(price),
        currency=str(row["currency"] or "USD"),
        source=str(row["provider"] or "db_quote_cache"),
        retrieved_at=str(row["fetched_at"] or utc_now_iso()),
        url=str(row["source_url"]) if row["source_url"] else None,
        confidence=confidence,
    )


def _cached_snapshot_diagnostic(
    *,
    ticker: str,
    requested_as_of: str,
    snapshot: PriceSnapshot,
    cache_path: Path,
    cache_label: str,
    price_source: str,
    reason_detail: str,
) -> dict[str, Any]:
    diag = _empty_diagnostic(
        ticker=ticker,
        requested_as_of=requested_as_of,
        cache_path=cache_path,
        provider=cache_label,
    )
    diag["resolved_symbol"] = f"{ticker.lower()}.us"
    diag["attempted_symbols"] = [f"{ticker.lower()}.us"]
    diag["cache"] = {
        "hit": True,
        "path": str(cache_path),
        "snapshot_found": True,
        "cached_as_of_used": snapshot.as_of_date,
    }
    target = _parse_date(requested_as_of)
    used = _parse_date(snapshot.as_of_date)
    fallback_days_checked = 1
    if target is not None and used is not None and used <= target:
        fallback_days_checked = int((target - used).days) + 1
    diag["market_day"] = {
        "requested_day_type": "TRADING" if snapshot.as_of_date == requested_as_of else "NON_TRADING",
        "fallback_days_checked": max(1, fallback_days_checked),
        "asof_used": snapshot.as_of_date,
    }
    diag["asof_final_used"] = snapshot.as_of_date
    diag["provider_attempts"] = [
        {
            "provider": cache_label,
            "status": "CACHE_HIT",
            "url": snapshot.url,
            "took_ms": 0,
        }
    ]
    diag["result"] = {
        "status": "OK",
        "reason_code": "CACHE_HIT",
        "reason_detail": reason_detail,
        "retryable": False,
        "error_detail": "",
        "suggestion": "",
    }
    diag["output_fields"] = {
        "current_price": float(snapshot.price),
        "price_asof_used": snapshot.as_of_date,
        "asof_final_used": snapshot.as_of_date,
        "price_source": price_source,
        "confidence": snapshot.confidence,
    }
    diag["local_fallbacks"] = {
        "run_scoped_output_checked": cache_label == "run_scoped_output",
        "run_scoped_output_hit": cache_label == "run_scoped_output",
        "disk_cache_checked": cache_label in {"run_scoped_output", "disk_cache"},
        "disk_cache_hit": cache_label == "disk_cache",
        "db_quote_cache_checked": cache_label in {"run_scoped_output", "disk_cache", "db_quote_cache"},
        "db_quote_cache_hit": cache_label == "db_quote_cache",
        "historical_run_artifacts_checked": False,
        "historical_run_artifacts_hit": False,
        "any_hit": True,
    }
    return diag


def _empty_diagnostic(
    *,
    ticker: str,
    requested_as_of: str,
    cache_path: Path | None = None,
    provider: str = "unknown",
) -> dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "requested_as_of": str(requested_as_of),
        "resolved_symbol": None,
        "attempted_symbols": [],
        "asof_final_used": None,
        "provider_attempts": [],
        "retry_count": 0,
        "total_attempts": 0,
        "cache": {
            "hit": False,
            "path": str(cache_path) if cache_path else "",
            "snapshot_found": False,
            "cached_as_of_used": None,
        },
        "market_day": {
            "requested_day_type": "UNKNOWN",
            "fallback_days_checked": 0,
            "asof_used": None,
        },
        "result": {
            "status": "UNKNOWN",
            "reason_code": "CACHE_MISS",
            "reason_detail": "No cached snapshot found.",
            "retryable": False,
            "error_detail": "",
            "suggestion": price_reason_suggestion("CACHE_MISS"),
        },
        "output_fields": {
            "current_price": "UNKNOWN",
            "price_asof_used": None,
            "asof_final_used": None,
            "price_source": provider if provider else None,
            "confidence": None,
        },
        "local_fallbacks": {
            "run_scoped_output_checked": False,
            "run_scoped_output_hit": False,
            "disk_cache_checked": False,
            "disk_cache_hit": False,
            "db_quote_cache_checked": False,
            "db_quote_cache_hit": False,
            "historical_run_artifacts_checked": False,
            "historical_run_artifacts_hit": False,
            "any_hit": False,
        },
        "suggestions": [],
    }


def _reason_detail(code: ReasonCode, default_message: str | None = None) -> str:
    if default_message:
        return str(default_message)
    messages = {
        "CACHE_HIT": "Price resolved from disk cache.",
        "CACHE_MISS": "No cached snapshot found.",
        "PROVIDER_OK": "Provider returned a valid closing price.",
        "DNS_FAILURE": "Provider request failed due to DNS resolution.",
        "TLS_FAILURE": "Provider request failed due to TLS/SSL negotiation.",
        "TIMEOUT": "Provider request timed out.",
        "HTTP_4XX": "Provider returned an HTTP 4xx error.",
        "HTTP_5XX": "Provider returned an HTTP 5xx error.",
        "RATE_LIMIT": "Provider rate limit reached.",
        "PARSE_ERROR": "Provider response could not be parsed.",
        "SYMBOL_NOT_FOUND": "Provider did not recognize the symbol mapping.",
        "PROVIDER_NO_DATA": "Provider returned no usable close data.",
        "NON_TRADING_DAY_NO_FALLBACK": "No fallback close found inside fallback window.",
        "OFFLINE_NO_CACHE": "Network provider is disabled and no cached snapshot is available.",
        "BUDGET_EXHAUSTED": "Provider request budget was exhausted.",
    }
    return messages[code]


def _reason_priority(code: str | None) -> int:
    ranking = {
        "CACHE_HIT": 100,
        "PROVIDER_OK": 95,
        "SYMBOL_NOT_FOUND": 82,
        "NON_TRADING_DAY_NO_FALLBACK": 78,
        "OFFLINE_NO_CACHE": 76,
        "BUDGET_EXHAUSTED": 74,
        "RATE_LIMIT": 73,
        "HTTP_5XX": 72,
        "HTTP_4XX": 71,
        "PARSE_ERROR": 70,
        "DNS_FAILURE": 68,
        "TLS_FAILURE": 67,
        "TIMEOUT": 66,
        "PROVIDER_NO_DATA": 40,
        "CACHE_MISS": 20,
    }
    return int(ranking.get(str(code or "").upper().strip(), 10))


def _network_disabled(cfg: AppConfig) -> bool:
    # VOE_NET_PROVIDER is the only network switch. A disabled LLM provider
    # does not imply offline: free quotes must work with no LLM configured.
    net_provider = str(os.getenv("VOE_NET_PROVIDER", cfg.net_provider)).strip().lower()
    return net_provider == "disabled"


class PriceCache:
    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or get_config()
        self.base_dir = self.cfg.cache_dir / "prices"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for_ticker(self, ticker: str) -> Path:
        return self.base_dir / f"{ticker.upper()}.json"

    def load(self, ticker: str, *, requested_as_of_date: str, source: str) -> PriceSnapshot | None:
        path = self.path_for_ticker(ticker)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("requested_as_of_date") or "") != str(requested_as_of_date):
                continue
            if str(entry.get("source") or "").lower() != str(source).lower():
                continue
            snapshot = _as_snapshot(entry.get("snapshot") if isinstance(entry.get("snapshot"), dict) else None)
            if snapshot and _snapshot_is_fresh(snapshot, ttl_seconds=_cache_ttl_seconds(self.cfg)):
                return snapshot
        return None

    def store(
        self,
        ticker: str,
        *,
        requested_as_of_date: str,
        source: str,
        snapshot: PriceSnapshot,
    ) -> PriceSnapshot:
        path = self.path_for_ticker(ticker)
        entries: list[dict[str, object]] = []
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                existing = payload.get("entries") if isinstance(payload, dict) else []
                if isinstance(existing, list):
                    entries = [row for row in existing if isinstance(row, dict)]
            except Exception:
                entries = []

        for entry in entries:
            if str(entry.get("requested_as_of_date") or "") != str(requested_as_of_date):
                continue
            if str(entry.get("source") or "").lower() != str(source).lower():
                continue
            cached = _as_snapshot(entry.get("snapshot") if isinstance(entry.get("snapshot"), dict) else None)
            if cached and _snapshot_is_fresh(cached, ttl_seconds=_cache_ttl_seconds(self.cfg)):
                return cached

        entries = [
            entry
            for entry in entries
            if not (
                str(entry.get("requested_as_of_date") or "") == str(requested_as_of_date)
                and str(entry.get("source") or "").lower() == str(source).lower()
            )
        ]

        entries.append(
            {
                "requested_as_of_date": str(requested_as_of_date),
                "source": str(source).lower(),
                "snapshot": {
                    "ticker": snapshot.ticker,
                    "as_of_date": snapshot.as_of_date,
                    "price": snapshot.price,
                    "currency": snapshot.currency or "USD",
                    "source": snapshot.source,
                    "retrieved_at": snapshot.retrieved_at,
                    "url": snapshot.url,
                    "confidence": snapshot.confidence,
                    "raw_price": snapshot.raw_price,
                    "volume": snapshot.volume,
                },
            }
        )
        entries.sort(
            key=lambda row: (
                str(row.get("requested_as_of_date") or ""),
                str(row.get("source") or ""),
            )
        )
        path.write_text(
            json.dumps({"ticker": ticker.upper(), "entries": entries}, indent=2),
            encoding="utf-8",
        )
        return snapshot


def _normalize_symbol_value(value: str) -> str:
    symbol = str(value or "").strip().lower()
    if not symbol:
        return ""
    if "." not in symbol:
        symbol = f"{symbol}.us"
    return symbol


def _load_symbol_overrides(path: Path) -> dict[str, list[SymbolOverride]]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    try:
        reader = csv.DictReader(io.StringIO(text))
    except Exception:
        return {}
    overrides: dict[str, list[SymbolOverride]] = {}
    for row in reader:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        symbol = _normalize_symbol_value(str(row.get("stooq_symbol") or row.get("symbol") or ""))
        if not ticker or not symbol:
            continue
        valid_from = _parse_date(str(row.get("valid_from") or "").strip())
        valid_to = _parse_date(str(row.get("valid_to") or "").strip())
        overrides.setdefault(ticker, []).append(
            SymbolOverride(
                ticker=ticker,
                symbol=symbol,
                valid_from=valid_from,
                valid_to=valid_to,
            )
        )
    for ticker in list(overrides.keys()):
        overrides[ticker] = sorted(
            overrides[ticker],
            key=lambda row: (
                row.valid_from or date.min,
                row.valid_to or date.max,
                row.symbol,
            ),
            reverse=True,
        )
    return overrides


class StooqProvider:
    provider_name = "stooq"
    history_url = "https://stooq.com/q/d/l/"

    def __init__(
        self,
        cfg: AppConfig | None = None,
        *,
        cache: PriceCache | None = None,
        fallback_days: int | None = None,
        symbol_overrides_path: Path | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.cfg = cfg or get_config()
        # Honor an explicit VOE_STOOQ_HISTORY_URL override (applies to all Stooq
        # providers); otherwise keep the per-class default class attr so subclasses
        # like StooqSecondaryProvider (www.stooq.com) are not shadowed.
        if self.cfg.stooq_history_url and self.cfg.stooq_history_url != StooqProvider.history_url:
            self.history_url = self.cfg.stooq_history_url
        self.cache = cache or PriceCache(self.cfg)
        self.timeout_seconds = max(5.0, float(self.cfg.http_timeout_seconds))
        self.fallback_days = max(0, int(fallback_days if fallback_days is not None else self.cfg.price_fallback_days))
        self.symbol_overrides_path = symbol_overrides_path or self.cfg.price_symbol_overrides_path
        self.symbol_overrides = _load_symbol_overrides(self.symbol_overrides_path)
        if max_retries is None:
            self.retry_backoff_seconds = _PRIMARY_RETRY_BACKOFF_SECONDS
        else:
            retry_limit = max(0, min(len(_PRIMARY_RETRY_BACKOFF_SECONDS), int(max_retries)))
            self.retry_backoff_seconds = _PRIMARY_RETRY_BACKOFF_SECONDS[:retry_limit]
        self._diag: dict[tuple[str, str], dict[str, Any]] = {}
        # Single-slot full-history download memo (symbol, download_result),
        # OFF by default — normal flows keep pure TTL-cache freshness. The
        # backfill enables it because the disk cache keys by exact
        # requested_as_of_date, so a 63-anchor ADV backfill re-downloads the
        # SAME full history 63 times per name without it (observed 2026-07-16:
        # ~4h for 1,131 names, EODHD rate-limited out after ~31). Backfills
        # iterate ticker-major, so one slot gives a full hit rate and each
        # symbol's anchors resolve seconds after its download; failed/empty
        # downloads memoize too, so a dead symbol costs one request, not 63.
        self._history_memo: tuple[str, Any] | None = None
        self._history_memo_enabled = False

    def enable_history_memo(self) -> None:
        """Coalesce full-history downloads per symbol for this instance's
        lifetime (backfill flows only — see __init__ note)."""
        self._history_memo_enabled = True

    def _download_history_memoized(self, symbol: str) -> Any:
        if not self._history_memo_enabled:
            return self._download_daily_history(symbol)
        memo = self._history_memo
        if memo is not None and memo[0] == symbol:
            if isinstance(memo[1], Exception):
                raise memo[1]
            return memo[1]
        # Hard failures memoize too (negative memo): a rate-limited or dead
        # provider fails ONCE per symbol instead of once per anchor (with
        # retry backoff, per-anchor re-probes added minutes per name).
        # Transient network classes stay un-memoized so the caller's in-call
        # retry loop still reaches the network.
        try:
            download_result = self._download_daily_history(symbol)
        except Exception as exc:
            code, _retryable, _suggestion, _detail = classify_price_error(
                exc,
                additional_sensitive_names=(self.cfg.stooq_apikey_param,),
            )
            if code not in {"DNS_FAILURE", "TLS_FAILURE", "TIMEOUT"}:
                self._history_memo = (symbol, exc)
            raise
        self._history_memo = (symbol, download_result)
        return download_result

    def _set_diag(self, ticker: str, as_of_date: str, payload: dict[str, Any]) -> None:
        self._diag[(ticker.upper(), str(as_of_date))] = payload

    def get_last_diagnostic(self, ticker: str, as_of_date: str) -> dict[str, Any] | None:
        payload = self._diag.get((ticker.upper(), str(as_of_date)))
        return sanitize_json_value(json.loads(json.dumps(payload))) if isinstance(payload, dict) else None

    def _default_symbol(self, ticker: str) -> str:
        return f"{ticker.lower()}.us"

    def _resolve_symbol(self, ticker: str, as_of_date: str) -> tuple[str, bool, list[str]]:
        default_symbol = self._default_symbol(ticker)
        target = _parse_date(as_of_date)
        candidates = self.symbol_overrides.get(ticker.upper()) or []
        if target is not None and candidates:
            eligible = [
                row
                for row in candidates
                if (row.valid_from is None or row.valid_from <= target)
                and (row.valid_to is None or row.valid_to >= target)
            ]
            if eligible:
                chosen = eligible[0].symbol
                attempted = [chosen]
                if chosen != default_symbol:
                    attempted.append(default_symbol)
                return chosen, True, attempted
        return default_symbol, False, [default_symbol]

    def _download_daily_history(self, symbol: str) -> tuple[str, str, int]:
        params = {"s": symbol, "i": "d"}
        if self.cfg.stooq_apikey:
            params[self.cfg.stooq_apikey_param] = self.cfg.stooq_apikey
        url = self.history_url
        request_url = sanitize_url_credentials(
            f"{url}?{urlencode(params)}",
            additional_sensitive_names=(self.cfg.stooq_apikey_param,),
        ) or url
        http = HttpClient(self.cfg)
        http._consume_domain_budget(url)  # noqa: SLF001
        response = requests.get(
            url,
            params=params,
            timeout=self.timeout_seconds,
            headers={"User-Agent": self.cfg.sec_user_agent},
        )
        status = int(response.status_code)
        response.raise_for_status()
        return response.text, request_url, status

    def _parse_history(self, csv_text: str) -> tuple[dict[date, float], int]:
        try:
            reader = csv.DictReader(io.StringIO(csv_text))
        except Exception as exc:
            raise ValueError(f"csv reader init failed: {exc}") from exc
        rows: dict[date, float] = {}
        raw_rows = 0
        for row in reader:
            raw_rows += 1
            row_date = _parse_date(str(row.get("Date") or row.get("date") or ""))
            close_value = row.get("Close") or row.get("close")
            if row_date is None or close_value in {None, "", "N/D"}:
                continue
            try:
                close = float(str(close_value).strip())
            except Exception:
                continue
            if close > 0:
                rows[row_date] = close
                volume_value = row.get("Volume") or row.get("volume")
                try:
                    volume = float(str(volume_value).strip())
                except (TypeError, ValueError):
                    volume = None
                if volume is not None and volume >= 0:
                    self._volumes_by_day[row_date] = volume
        return rows, raw_rows

    def get_price_asof(self, ticker: str, as_of_date: str) -> PriceSnapshot | None:
        ticker_norm = str(ticker).upper().strip()
        as_of_norm = str(as_of_date).strip()
        cache_path = self.cache.path_for_ticker(ticker_norm)
        diag = _empty_diagnostic(
            ticker=ticker_norm,
            requested_as_of=as_of_norm,
            cache_path=cache_path,
            provider=self.provider_name,
        )
        self._set_diag(ticker_norm, as_of_norm, diag)
        target = _parse_date(as_of_norm)
        if not ticker_norm or target is None:
            diag["result"] = {
                "status": "UNKNOWN",
                "reason_code": "PARSE_ERROR",
                "reason_detail": "Invalid ticker or as_of_date format.",
                "retryable": False,
                "error_detail": "Invalid ticker or as_of_date format.",
                "suggestion": price_reason_suggestion("PARSE_ERROR"),
            }
            self._set_diag(ticker_norm, as_of_norm, diag)
            return None

        resolved_symbol, mapped, attempted_symbols = self._resolve_symbol(ticker_norm, as_of_norm)
        diag["resolved_symbol"] = resolved_symbol
        diag["attempted_symbols"] = attempted_symbols

        cached = self.cache.load(ticker_norm, requested_as_of_date=as_of_norm, source=self.provider_name)
        if cached is not None:
            diag["cache"] = {
                "hit": True,
                "path": str(cache_path),
                "snapshot_found": True,
                "cached_as_of_used": cached.as_of_date,
            }
            diag["market_day"]["asof_used"] = cached.as_of_date
            diag["asof_final_used"] = cached.as_of_date
            diag["provider_attempts"].append(
                {
                    "provider": self.provider_name,
                    "status": "CACHE_HIT",
                    "url": cached.url,
                    "took_ms": 0,
                }
            )
            diag["result"] = {
                "status": "OK",
                "reason_code": "CACHE_HIT",
                "reason_detail": _reason_detail("CACHE_HIT"),
                "retryable": False,
                "error_detail": "",
                "suggestion": "",
            }
            diag["output_fields"] = {
                "current_price": float(cached.price),
                "price_asof_used": cached.as_of_date,
                "asof_final_used": cached.as_of_date,
                "price_source": cached.source,
                "confidence": cached.confidence,
            }
            self._set_diag(ticker_norm, as_of_norm, diag)
            return cached

        diag["cache"]["hit"] = False
        diag["cache"]["snapshot_found"] = False
        diag["result"] = {
            "status": "UNKNOWN",
            "reason_code": "CACHE_MISS",
            "reason_detail": _reason_detail("CACHE_MISS"),
            "retryable": False,
            "error_detail": "",
            "suggestion": price_reason_suggestion("CACHE_MISS"),
        }
        if _network_disabled(self.cfg):
            suggestion = price_reason_suggestion("OFFLINE_NO_CACHE")
            diag["provider_attempts"].append(
                {
                    "provider": self.provider_name,
                    "status": "OFFLINE_NO_CACHE",
                    "error_code": "OFFLINE_NO_CACHE",
                    "error_summary": "Network provider disabled and no cache hit.",
                    "error_detail": "Network provider disabled and no cache hit.",
                    "retryable": False,
                    "suggestion": suggestion,
                    "url": self.history_url,
                    "took_ms": 0,
                }
            )
            diag["total_attempts"] = 1
            diag["result"] = {
                "status": "UNKNOWN",
                "reason_code": "OFFLINE_NO_CACHE",
                "reason_detail": _reason_detail("OFFLINE_NO_CACHE"),
                "retryable": False,
                "error_detail": "Network provider disabled and no cache hit.",
                "suggestion": suggestion,
            }
            self._set_diag(ticker_norm, as_of_norm, diag)
            return None

        request_url = self.history_url
        http_status: int | None = None
        csv_text = ""
        total_attempts = 0
        retry_count = 0
        final_failure: tuple[ReasonCode, bool, str, str] | None = None

        for attempt_idx in range(len(self.retry_backoff_seconds) + 1):
            total_attempts += 1
            start = time.perf_counter()
            try:
                download_result = self._download_history_memoized(resolved_symbol)
                if isinstance(download_result, tuple) and len(download_result) == 3:
                    csv_text, request_url, http_status = download_result
                elif isinstance(download_result, tuple) and len(download_result) == 2:
                    csv_text, request_url = download_result
                    http_status = 200
                else:
                    raise ValueError("Unexpected provider download result format.")
                final_failure = None
                break
            except Exception as exc:  # noqa: BLE001
                code, retryable, suggestion, error_detail = classify_price_error(
                    exc,
                    additional_sensitive_names=(self.cfg.stooq_apikey_param,),
                )
                if code == "BUDGET_EXHAUSTED":
                    retryable = False
                diag["provider_attempts"].append(
                    {
                        "provider": self.provider_name,
                        "status": code,
                        "http_status": int(getattr(getattr(exc, "response", None), "status_code", 0) or 0) or None,
                        "error_code": code,
                        "error_summary": _reason_detail(code),
                        "error_detail": error_detail,
                        "retryable": bool(retryable),
                        "suggestion": suggestion,
                        "attempt": int(attempt_idx + 1),
                        "url": self.history_url,
                        "took_ms": int((time.perf_counter() - start) * 1000.0),
                    }
                )
                final_failure = (code, retryable, suggestion, error_detail)
                should_retry = bool(code in {"DNS_FAILURE", "TLS_FAILURE", "TIMEOUT"} and attempt_idx < len(self.retry_backoff_seconds))
                if should_retry:
                    retry_count += 1
                    time.sleep(self.retry_backoff_seconds[attempt_idx])
                    continue
                break

        diag["retry_count"] = int(retry_count)
        diag["total_attempts"] = int(total_attempts)
        if final_failure is not None:
            code, retryable, suggestion, error_detail = final_failure
            diag["result"] = {
                "status": "UNKNOWN",
                "reason_code": code,
                "reason_detail": _reason_detail(code),
                "retryable": bool(retryable),
                "error_detail": error_detail,
                "suggestion": suggestion,
            }
            self._set_diag(ticker_norm, as_of_norm, diag)
            return None

        self._raw_closes_by_day: dict[date, float] = {}
        self._volumes_by_day: dict[date, float] = {}
        try:
            closes_by_day, raw_rows = self._parse_history(csv_text)
        except Exception as exc:  # noqa: BLE001
            reason, retryable, suggestion, error_detail = classify_price_error(exc)
            reason = "PARSE_ERROR"
            diag["provider_attempts"].append(
                {
                    "provider": self.provider_name,
                    "status": reason,
                    "http_status": http_status,
                    "error_code": reason,
                    "error_summary": _reason_detail(reason),
                    "error_detail": error_detail,
                    "retryable": bool(retryable),
                    "suggestion": suggestion,
                    "attempt": int(total_attempts + 1),
                    "url": request_url,
                    "took_ms": 0,
                }
            )
            diag["total_attempts"] = int(total_attempts + 1)
            diag["result"] = {
                "status": "UNKNOWN",
                "reason_code": reason,
                "reason_detail": _reason_detail(reason),
                "retryable": bool(retryable),
                "error_detail": error_detail,
                "suggestion": suggestion,
            }
            self._set_diag(ticker_norm, as_of_norm, diag)
            return None

        if not closes_by_day:
            code: ReasonCode = "SYMBOL_NOT_FOUND" if (raw_rows == 0 or (not mapped)) else "PROVIDER_NO_DATA"
            suggestion = price_reason_suggestion(code)
            diag["provider_attempts"].append(
                {
                    "provider": self.provider_name,
                    "status": code,
                    "http_status": http_status,
                    "error_code": code,
                    "error_summary": "No rows in provider response.",
                    "error_detail": "No rows in provider response.",
                    "retryable": False,
                    "suggestion": suggestion,
                    "url": request_url,
                    "took_ms": 0,
                }
            )
            diag["total_attempts"] = int(diag.get("total_attempts") or 0) + 1
            diag["result"] = {
                "status": "UNKNOWN",
                "reason_code": code,
                "reason_detail": _reason_detail(code),
                "retryable": False,
                "error_detail": "No rows in provider response.",
                "suggestion": suggestion,
            }
            self._set_diag(ticker_norm, as_of_norm, diag)
            return None

        requested_day_type = "TRADING" if target in closes_by_day else "NON_TRADING"
        diag["market_day"]["requested_day_type"] = requested_day_type

        selected_day: date | None = None
        for offset in range(0, self.fallback_days + 1):
            candidate = target - timedelta(days=offset)
            diag["market_day"]["fallback_days_checked"] = int(offset) + 1
            if candidate in closes_by_day:
                selected_day = candidate
                break

        if selected_day is None:
            reason: ReasonCode = "NON_TRADING_DAY_NO_FALLBACK" if requested_day_type == "NON_TRADING" else "PROVIDER_NO_DATA"
            suggestion = price_reason_suggestion(reason)
            diag["provider_attempts"].append(
                {
                    "provider": self.provider_name,
                    "status": reason,
                    "http_status": http_status,
                    "error_code": reason,
                    "error_summary": f"No close found within fallback window={self.fallback_days} days.",
                    "error_detail": f"No close found within fallback window={self.fallback_days} days.",
                    "retryable": False,
                    "suggestion": suggestion,
                    "url": request_url,
                    "took_ms": 0,
                }
            )
            diag["total_attempts"] = int(diag.get("total_attempts") or 0) + 1
            diag["result"] = {
                "status": "UNKNOWN",
                "reason_code": reason,
                "reason_detail": _reason_detail(reason),
                "retryable": False,
                "error_detail": f"No close found within fallback window={self.fallback_days} days.",
                "suggestion": suggestion,
            }
            self._set_diag(ticker_norm, as_of_norm, diag)
            return None

        used_asof = selected_day.isoformat()
        price = float(closes_by_day[selected_day])
        confidence: PriceConfidence = "HIGH" if used_asof == as_of_norm else "MEDIUM"
        raw_close = (getattr(self, "_raw_closes_by_day", None) or {}).get(selected_day)
        snapshot = PriceSnapshot(
            ticker=ticker_norm,
            as_of_date=used_asof,
            price=price,
            currency="USD",
            source=self.provider_name,
            retrieved_at=utc_now_iso(),
            url=request_url,
            confidence=confidence,
            raw_price=float(raw_close) if isinstance(raw_close, (int, float)) and raw_close > 0 else None,
            volume=(getattr(self, "_volumes_by_day", None) or {}).get(selected_day),
        )
        snapshot = self.cache.store(
            ticker_norm,
            requested_as_of_date=as_of_norm,
            source=self.provider_name,
            snapshot=snapshot,
        )
        diag["provider_attempts"].append(
            {
                "provider": self.provider_name,
                "status": "PROVIDER_OK",
                "http_status": http_status,
                "url": request_url,
                "took_ms": int((time.perf_counter() - start) * 1000.0),
            }
        )
        diag["market_day"]["asof_used"] = used_asof
        diag["asof_final_used"] = used_asof
        diag["result"] = {
            "status": "OK",
            "reason_code": "PROVIDER_OK",
            "reason_detail": _reason_detail("PROVIDER_OK"),
            "retryable": False,
            "error_detail": "",
            "suggestion": "",
        }
        diag["output_fields"] = {
            "current_price": float(snapshot.price),
            "price_asof_used": snapshot.as_of_date,
            "asof_final_used": used_asof,
            "price_source": snapshot.source,
            "confidence": snapshot.confidence,
        }
        self._set_diag(ticker_norm, as_of_norm, diag)
        return snapshot


class StooqSecondaryProvider(StooqProvider):
    provider_name = "stooq_secondary"
    history_url = "https://www.stooq.com/q/d/l/"


class EODHDProvider(StooqProvider):
    """
    Licensed historical-EOD provider (https://eodhd.com), delisting-capable.

    Reuses StooqProvider's cache/retry/diagnostic/at-or-before machinery; only
    the transport (windowed JSON request) and symbol mapping (TICKER.EXCHANGE)
    differ. Chain assembly only constructs this provider when cfg.eodhd_apikey
    is set, so it is inert without a key.
    """

    provider_name = "eodhd"
    history_url = "https://eodhd.com/api/eod"

    def __init__(
        self,
        cfg: AppConfig | None = None,
        *,
        cache: PriceCache | None = None,
        fallback_days: int | None = None,
        symbol_overrides_path: Path | None = None,
        max_retries: int | None = None,
    ) -> None:
        super().__init__(
            cfg,
            cache=cache,
            fallback_days=fallback_days,
            symbol_overrides_path=symbol_overrides_path,
            max_retries=max_retries,
        )
        # Parent __init__ may have applied the Stooq history-URL override;
        # always rebuild from the EODHD base URL.
        base = str(self.cfg.eodhd_base_url or "https://eodhd.com").rstrip("/")
        self.history_url = f"{base}/api/eod"
        self._window_target: date | None = None

    def _default_symbol(self, ticker: str) -> str:
        exchange = str(self.cfg.eodhd_exchange or "US").strip().upper() or "US"
        return f"{ticker.upper()}.{exchange}"

    def _resolve_symbol(self, ticker: str, as_of_date: str) -> tuple[str, bool, list[str]]:
        # The overrides CSV is Stooq-format; EODHD mapping is the canonical
        # TICKER.EXCHANGE. Report mapped=True so empty-vs-unusable payloads
        # classify like Stooq's mapped path (zero rows -> SYMBOL_NOT_FOUND,
        # unusable rows -> PROVIDER_NO_DATA).
        default_symbol = self._default_symbol(ticker)
        return default_symbol, True, [default_symbol]

    def get_price_asof(self, ticker: str, as_of_date: str) -> PriceSnapshot | None:
        # Stash the target so the windowed from/to request ends at the
        # requested date; the parent drives caching, retries, diagnostics,
        # and last-close-at-or-before-date selection.
        self._window_target = _parse_date(str(as_of_date).strip())
        try:
            return super().get_price_asof(ticker, as_of_date)
        finally:
            self._window_target = None

    def _download_daily_history(self, symbol: str) -> tuple[str, str, int]:
        target = self._window_target or date.today()
        window_start = target - timedelta(days=max(7, self.fallback_days))
        params = {
            "api_token": self.cfg.eodhd_apikey or "",
            "from": window_start.isoformat(),
            "to": target.isoformat(),
            "fmt": "json",
        }
        url = f"{self.history_url}/{symbol}"
        request_url = sanitize_url_credentials(f"{url}?{urlencode(params)}") or url
        http = HttpClient(self.cfg)
        http._consume_domain_budget(url)  # noqa: SLF001
        response = requests.get(
            url,
            params=params,
            timeout=self.timeout_seconds,
            headers={"User-Agent": self.cfg.sec_user_agent},
        )
        status = int(response.status_code)
        response.raise_for_status()
        return response.text, request_url, status

    def _parse_history(self, json_text: str) -> tuple[dict[date, float], int]:
        try:
            payload = json.loads(json_text)
        except Exception as exc:
            raise ValueError(f"json parse failed: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError("json payload is not a list of EOD rows")
        rows: dict[date, float] = {}
        raw_closes: dict[date, float] = {}
        raw_rows = 0
        for row in payload:
            if not isinstance(row, dict):
                continue
            raw_rows += 1
            row_date = _parse_date(str(row.get("date") or ""))
            close_value = row.get("adjusted_close")
            if close_value in (None, ""):
                close_value = row.get("close")
            if row_date is None or close_value in (None, ""):
                continue
            try:
                close = float(str(close_value).strip())
            except Exception:
                continue
            if close > 0:
                rows[row_date] = close
            # Keep the unadjusted close beside the adjusted one (split-basis
            # re-basing: deploy classification needs the quote live saw at T).
            raw_value = row.get("close")
            try:
                raw_close = float(str(raw_value).strip()) if raw_value not in (None, "") else None
            except Exception:
                raw_close = None
            if raw_close is not None and raw_close > 0 and row_date is not None:
                raw_closes[row_date] = raw_close
            volume_value = row.get("volume")
            try:
                volume = float(str(volume_value).strip()) if volume_value not in (None, "") else None
            except Exception:
                volume = None
            if volume is not None and volume >= 0 and row_date is not None:
                self._volumes_by_day[row_date] = volume
        self._raw_closes_by_day = raw_closes
        return rows, raw_rows


class FallbackProvider:
    provider_name = "fallback"

    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or get_config()
        self._diag: dict[tuple[str, str], dict[str, Any]] = {}

    def get_price_asof(self, ticker: str, as_of_date: str) -> PriceSnapshot | None:
        ticker_norm = str(ticker).upper().strip()
        as_of_norm = str(as_of_date).strip()
        diag = _empty_diagnostic(
            ticker=ticker_norm,
            requested_as_of=as_of_norm,
            cache_path=None,
            provider=self.provider_name,
        )
        diag["provider_attempts"].append(
            {
                "provider": self.provider_name,
                "status": "PROVIDER_NO_DATA",
                "error_code": "PROVIDER_NO_DATA",
                "error_summary": "Fallback provider intentionally returns no data.",
                "error_detail": "Fallback provider intentionally returns no data.",
                "retryable": False,
                "suggestion": price_reason_suggestion("PROVIDER_NO_DATA"),
                "took_ms": 0,
            }
        )
        diag["total_attempts"] = 1
        diag["result"] = {
            "status": "UNKNOWN",
            "reason_code": "PROVIDER_NO_DATA",
            "reason_detail": _reason_detail("PROVIDER_NO_DATA"),
            "retryable": False,
            "error_detail": "Fallback provider intentionally returns no data.",
            "suggestion": price_reason_suggestion("PROVIDER_NO_DATA"),
        }
        self._diag[(ticker_norm, as_of_norm)] = diag
        return None

    def get_last_diagnostic(self, ticker: str, as_of_date: str) -> dict[str, Any] | None:
        payload = self._diag.get((ticker.upper(), str(as_of_date)))
        return sanitize_json_value(json.loads(json.dumps(payload))) if isinstance(payload, dict) else None


class ChainedPriceProvider:
    provider_name = "stooq+fallback"

    def __init__(self, providers: list[PriceProvider]) -> None:
        self.providers = providers
        self._diag: dict[tuple[str, str], dict[str, Any]] = {}

    def enable_history_memo(self) -> None:
        for provider in self.providers:
            enable = getattr(provider, "enable_history_memo", None)
            if callable(enable):
                enable()

    def get_price_asof(self, ticker: str, as_of_date: str) -> PriceSnapshot | None:
        ticker_norm = str(ticker).upper().strip()
        as_of_norm = str(as_of_date).strip()
        combined = _empty_diagnostic(
            ticker=ticker_norm,
            requested_as_of=as_of_norm,
            cache_path=None,
            provider=self.provider_name,
        )
        chosen_snapshot: PriceSnapshot | None = None
        latest_result: dict[str, Any] | None = None
        total_attempts = 0
        total_retries = 0
        # The chain runs EVERY configured provider until one succeeds.
        # The old min(len, 2) cap made the third provider dead code whenever
        # EODHD led the chain, and breaking on a non-retryable first-provider
        # failure ignored that providers have different symbology — a
        # SYMBOL_NOT_FOUND at one is not evidence about the next.
        for provider in self.providers:
            snapshot = provider.get_price_asof(ticker_norm, as_of_norm)
            provider_diag = provider.get_last_diagnostic(ticker_norm, as_of_norm) if hasattr(provider, "get_last_diagnostic") else None
            provider_result: dict[str, Any] | None = None
            if isinstance(provider_diag, dict):
                if provider_diag.get("resolved_symbol"):
                    combined["resolved_symbol"] = provider_diag.get("resolved_symbol")
                attempted_symbols = provider_diag.get("attempted_symbols")
                if isinstance(attempted_symbols, list):
                    merged_attempts = [
                        str(symbol)
                        for symbol in attempted_symbols
                        if str(symbol).strip()
                    ]
                    combined["attempted_symbols"] = merged_attempts
                if provider_diag.get("asof_final_used"):
                    combined["asof_final_used"] = provider_diag.get("asof_final_used")
                if isinstance(provider_diag.get("cache"), dict):
                    combined["cache"] = provider_diag["cache"]
                if isinstance(provider_diag.get("market_day"), dict):
                    combined["market_day"] = provider_diag["market_day"]
                attempts = provider_diag.get("provider_attempts")
                if isinstance(attempts, list):
                    combined["provider_attempts"].extend(attempts)
                total_attempts += int(provider_diag.get("total_attempts") or len(attempts or []))
                total_retries += int(provider_diag.get("retry_count") or 0)
                result = provider_diag.get("result")
                if isinstance(result, dict):
                    provider_result = json.loads(json.dumps(result))
                    latest_result = provider_result
                outputs = provider_diag.get("output_fields")
                if isinstance(outputs, dict):
                    combined["output_fields"] = outputs
                suggestion = str((provider_result or {}).get("suggestion") or "")
                if suggestion:
                    combined["suggestions"] = [suggestion]
            if snapshot is not None:
                chosen_snapshot = snapshot
                break
        if latest_result:
            combined["result"] = latest_result
        combined["total_attempts"] = int(total_attempts)
        combined["retry_count"] = int(total_retries)
        self._diag[(ticker_norm, as_of_norm)] = combined
        return chosen_snapshot

    def get_last_diagnostic(self, ticker: str, as_of_date: str) -> dict[str, Any] | None:
        payload = self._diag.get((ticker.upper(), str(as_of_date)))
        return sanitize_json_value(json.loads(json.dumps(payload))) if isinstance(payload, dict) else None


class YahooFinanceProvider:
    """Price provider using yfinance. Free, no API key, good US stock coverage."""
    provider_name = "yahoo"

    def __init__(self, cfg: AppConfig | None = None, fallback_days: int | None = None, **kwargs: Any) -> None:
        self.cfg = cfg or get_config()
        self.fallback_days = fallback_days or 5
        self.timeout_seconds = max(1.0, float(kwargs.get("timeout_seconds") or self.cfg.http_timeout_seconds))
        # Existing trigger/research consumers retain adjusted-close behavior.
        # The factual quote writer opts out so it can persist a truthful
        # UNADJUSTED price-basis contract without changing trigger semantics.
        self.auto_adjust = bool(kwargs.get("auto_adjust", True))
        self._diag: dict[tuple[str, str], dict[str, Any]] = {}

    def get_price_asof(self, ticker: str, as_of_date: str) -> PriceSnapshot | None:
        ticker_norm = ticker.upper().strip()
        as_of_norm = as_of_date.strip()
        if _network_disabled(self.cfg):
            # Yahoo has no local cache: with the network switch off it has
            # nothing to serve, and it says so rather than calling out.
            self._diag[(ticker_norm, as_of_norm)] = {
                "status": "OFFLINE_NO_CACHE",
                "provider": "yahoo",
                "ticker": ticker_norm,
            }
            return None
        try:
            import yfinance as yf
            tk = yf.Ticker(ticker_norm)
            target = _parse_date(as_of_norm)
            if target is None:
                target = date.today()
            # Fetch a window around the target date to handle weekends/holidays
            start = target - timedelta(days=self.fallback_days + 1)
            end = target + timedelta(days=1)
            history_kwargs = {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "auto_adjust": self.auto_adjust,
            }
            try:
                hist = tk.history(**history_kwargs, timeout=self.timeout_seconds)
            except TypeError:
                hist = tk.history(**history_kwargs)
            if hist.empty:
                self._diag[(ticker_norm, as_of_norm)] = {
                    "status": "PROVIDER_NO_DATA",
                    "provider": "yahoo",
                    "ticker": ticker_norm,
                }
                return None
            # Find the closest date on or before target with a valid (non-NaN)
            # close. rows AFTER the target are never used — substituting a
            # future close for an as-of read is look-ahead, not a fallback.
            import math
            valid = hist[hist.index.date <= target]
            if valid.empty:
                self._diag[(ticker_norm, as_of_norm)] = {
                    "status": "PROVIDER_NO_DATA_ON_OR_BEFORE_ASOF",
                    "provider": "yahoo",
                    "ticker": ticker_norm,
                }
                return None
            # Drop rows where Close is NaN (market holidays, incomplete data)
            valid = valid[valid["Close"].apply(lambda x: not math.isnan(x))]
            if valid.empty:
                self._diag[(ticker_norm, as_of_norm)] = {
                    "status": "PROVIDER_NAN_PRICE",
                    "provider": "yahoo",
                    "ticker": ticker_norm,
                }
                return None
            row = valid.iloc[-1]
            price_date = valid.index[-1]
            price = float(row["Close"])
            price_date_str = price_date.strftime("%Y-%m-%d")
            # Yfinance history carries Volume — attach it (this provider
            # wrote 66k volume-less rows on the 2026-07-16 ADV backfill).
            volume: float | None = None
            try:
                raw_volume = float(row["Volume"])
                if not math.isnan(raw_volume) and raw_volume >= 0:
                    volume = raw_volume
            except (KeyError, TypeError, ValueError):
                volume = None
            self._diag[(ticker_norm, as_of_norm)] = {
                "status": "OK",
                "provider": "yahoo",
                "ticker": ticker_norm,
                "price_date": price_date_str,
            }
            # HIGH confidence is reserved for an exact as-of match; a
            # weekend/holiday backfill within the window is MEDIUM.
            return PriceSnapshot(
                ticker=ticker_norm,
                as_of_date=price_date_str,
                price=price,
                currency="USD",
                source="yahoo",
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                confidence="HIGH" if price_date_str == target.isoformat() else "MEDIUM",
                volume=volume,
            )
        except Exception as exc:
            self._diag[(ticker_norm, as_of_norm)] = {
                "status": "ERROR",
                "provider": "yahoo",
                "ticker": ticker_norm,
                "error": redact_credential_text(str(exc)),
            }
            return None

    def get_last_diagnostic(self, ticker: str, as_of_date: str) -> dict[str, Any] | None:
        payload = self._diag.get((ticker.upper().strip(), as_of_date.strip()))
        return sanitize_json_value(json.loads(json.dumps(payload))) if isinstance(payload, dict) else None


def _eodhd_chain_head(
    cfg: AppConfig,
    *,
    fallback_days: int | None = None,
    symbol_overrides_path: Path | None = None,
    max_retries: int | None = None,
) -> list[PriceProvider]:
    # EODHD leads the chain only when an API key is configured; without a key
    # the provider is never constructed.
    if not cfg.eodhd_apikey:
        return []
    return [
        EODHDProvider(
            cfg,
            fallback_days=fallback_days,
            symbol_overrides_path=symbol_overrides_path,
            max_retries=max_retries,
        )
    ]


def get_default_provider(
    cfg: AppConfig | None = None,
    *,
    fallback_days: int | None = None,
    symbol_overrides_path: Path | None = None,
    max_retries: int | None = None,
) -> PriceProvider:
    cfg = cfg or get_config()
    provider_name = str(cfg.price_provider or "").strip().lower()
    if provider_name in {"disabled", "off", "none"}:
        return FallbackProvider(cfg)
    eodhd_head = _eodhd_chain_head(
        cfg,
        fallback_days=fallback_days,
        symbol_overrides_path=symbol_overrides_path,
        max_retries=max_retries,
    )
    if provider_name == "yahoo":
        yahoo = YahooFinanceProvider(cfg, fallback_days=fallback_days)
        if eodhd_head:
            return ChainedPriceProvider([*eodhd_head, yahoo])
        return yahoo
    if provider_name == "stooq":
        return ChainedPriceProvider(
            [
                *eodhd_head,
                StooqProvider(
                    cfg,
                    fallback_days=fallback_days,
                    symbol_overrides_path=symbol_overrides_path,
                    max_retries=max_retries,
                ),
                StooqSecondaryProvider(
                    cfg,
                    fallback_days=fallback_days,
                    symbol_overrides_path=symbol_overrides_path,
                    max_retries=max_retries,
                ),
            ]
        )
    # "auto" or unrecognized: EODHD when keyed, then Yahoo (free, keyless),
    # then Stooq only when a Stooq key is configured (keyless Stooq requests
    # are refused, so chaining one would only add a guaranteed failure).
    stooq_tail: list[PriceProvider] = []
    if cfg.stooq_apikey:
        stooq_tail.append(
            StooqProvider(
                cfg,
                fallback_days=fallback_days,
                symbol_overrides_path=symbol_overrides_path,
                max_retries=max_retries,
            )
        )
    return ChainedPriceProvider(
        [
            *eodhd_head,
            YahooFinanceProvider(cfg, fallback_days=fallback_days),
            *stooq_tail,
        ]
    )


def build_price_provider(
    *,
    cfg: AppConfig | None = None,
    with_prices: bool = True,
    fallback_days: int | None = None,
    symbol_overrides_path: Path | None = None,
    max_retries: int | None = None,
) -> PriceProvider:
    cfg = cfg or get_config()
    if not with_prices:
        return FallbackProvider(cfg)
    provider = get_default_provider(
        cfg,
        fallback_days=fallback_days,
        symbol_overrides_path=symbol_overrides_path,
        max_retries=max_retries,
    )
    if isinstance(provider, FallbackProvider):
        return ChainedPriceProvider(
            [
                *_eodhd_chain_head(
                    cfg,
                    fallback_days=fallback_days,
                    symbol_overrides_path=symbol_overrides_path,
                    max_retries=max_retries,
                ),
                StooqProvider(
                    cfg,
                    fallback_days=fallback_days,
                    symbol_overrides_path=symbol_overrides_path,
                    max_retries=max_retries,
                ),
                StooqSecondaryProvider(
                    cfg,
                    fallback_days=fallback_days,
                    symbol_overrides_path=symbol_overrides_path,
                    max_retries=max_retries,
                ),
            ]
        )
    return provider


def write_prices_for_run(
    *,
    tickers: list[str],
    as_of_date: str,
    run_id: str,
    fallback_days: int | None = None,
    local_only: bool = False,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    out_dir = cfg.outputs_dir / "prices" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = build_price_provider(
        cfg=cfg,
        with_prices=True,
        fallback_days=fallback_days if fallback_days is not None else cfg.price_fallback_days,
    )
    cache = PriceCache(cfg)

    rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    ok_count = 0
    for ticker in sorted({str(symbol).strip().upper() for symbol in tickers if str(symbol).strip()}):
        ticker_path = out_dir / f"{ticker}.json"
        snapshot, diagnostic = _load_run_scoped_output(ticker_path, requested_as_of_date=as_of_date)

        if snapshot is not None:
            if not isinstance(diagnostic, dict):
                diagnostic = _cached_snapshot_diagnostic(
                    ticker=ticker,
                    requested_as_of=as_of_date,
                    snapshot=snapshot,
                    cache_path=ticker_path,
                    cache_label="run_scoped_output",
                    price_source="run_scoped_output",
                    reason_detail="Price resolved from outputs/prices/<run_id> cache.",
                )
        else:
            disk_path = cache.path_for_ticker(ticker)
            snapshot = _load_disk_cache_snapshot(
                disk_path,
                requested_as_of_date=as_of_date,
                ttl_seconds=_cache_ttl_seconds(cfg),
            )
            if snapshot is not None:
                diagnostic = _cached_snapshot_diagnostic(
                    ticker=ticker,
                    requested_as_of=as_of_date,
                    snapshot=snapshot,
                    cache_path=disk_path,
                    cache_label="disk_cache",
                    price_source="disk_cache",
                    reason_detail="Price resolved from data/cache/prices cache.",
                )
            else:
                snapshot = _load_db_quote_snapshot(ticker=ticker, as_of_date=as_of_date)
                if snapshot is not None:
                    diagnostic = _cached_snapshot_diagnostic(
                        ticker=ticker,
                        requested_as_of=as_of_date,
                        snapshot=snapshot,
                        cache_path=cfg.db_path,
                        cache_label="db_quote_cache",
                        price_source="db_quote_cache",
                        reason_detail="Price resolved from price_quotes cache.",
                    )
                else:
                    historical = resolve_price_from_historical_runs(
                        ticker=ticker,
                        as_of_date=as_of_date,
                        sectors_dir=cfg.sectors_dir,
                    )
                    if historical is not None:
                        diagnostic = _cached_snapshot_diagnostic(
                            ticker=ticker,
                            requested_as_of=as_of_date,
                            snapshot=historical,
                            cache_path=cfg.sectors_dir,
                            cache_label="historical_run_artifacts",
                            price_source="historical_run_artifacts",
                            reason_detail="Price resolved from prior sector run artifacts.",
                        )
                        local_flags = diagnostic.get("local_fallbacks")
                        if isinstance(local_flags, dict):
                            local_flags["historical_run_artifacts_checked"] = True
                            local_flags["historical_run_artifacts_hit"] = True
                            local_flags["db_quote_cache_checked"] = True
                            local_flags["disk_cache_checked"] = True
                            local_flags["run_scoped_output_checked"] = True
                            local_flags["any_hit"] = True
                        snapshot = PriceSnapshot(
                            ticker=ticker,
                            as_of_date=historical.as_of_date,
                            price=float(historical.price),
                            currency=historical.currency,
                            source="historical_run_artifacts",
                            retrieved_at=historical.retrieved_at,
                            url=historical.url,
                            confidence=historical.confidence,
                        )
                    elif local_only:
                        snapshot = None
                        diagnostic = _empty_diagnostic(
                            ticker=ticker,
                            requested_as_of=as_of_date,
                            cache_path=cache.path_for_ticker(ticker),
                            provider="local_only",
                        )
                        diagnostic["local_fallbacks"] = {
                            "run_scoped_output_checked": True,
                            "run_scoped_output_hit": False,
                            "disk_cache_checked": True,
                            "disk_cache_hit": False,
                            "db_quote_cache_checked": True,
                            "db_quote_cache_hit": False,
                            "historical_run_artifacts_checked": True,
                            "historical_run_artifacts_hit": False,
                            "any_hit": False,
                        }
                        diagnostic["result"] = {
                            "status": "UNKNOWN",
                            "reason_code": "OFFLINE_NO_CACHE",
                            "reason_detail": _reason_detail("OFFLINE_NO_CACHE"),
                            "retryable": False,
                            "error_detail": "Local-only seed found no run-scoped, disk, DB, or historical artifact price.",
                            "terminal": True,
                            "suggestion": _OFFLINE_TERMINAL_SUGGESTION,
                        }
                        diagnostic["suggestions"] = [_OFFLINE_TERMINAL_SUGGESTION]
                    else:
                        snapshot = provider.get_price_asof(ticker, as_of_date)
                        diagnostic = provider.get_last_diagnostic(ticker, as_of_date) if hasattr(provider, "get_last_diagnostic") else None

        if isinstance(diagnostic, dict):
            local_flags = diagnostic.get("local_fallbacks")
            if not isinstance(local_flags, dict):
                local_flags = {
                    "run_scoped_output_checked": True,
                    "run_scoped_output_hit": False,
                    "disk_cache_checked": True,
                    "disk_cache_hit": False,
                    "db_quote_cache_checked": True,
                    "db_quote_cache_hit": False,
                    "historical_run_artifacts_checked": True,
                    "historical_run_artifacts_hit": False,
                    "any_hit": False,
                }
                diagnostic["local_fallbacks"] = local_flags
            local_flags["any_hit"] = bool(
                local_flags.get("run_scoped_output_hit")
                or local_flags.get("disk_cache_hit")
                or local_flags.get("db_quote_cache_hit")
                or local_flags.get("historical_run_artifacts_hit")
            )
            result_bucket = diagnostic.get("result") if isinstance(diagnostic.get("result"), dict) else {}
            reason_value = _normalize_reason((result_bucket or {}).get("reason_code"))
            if snapshot is None and reason_value == "OFFLINE_NO_CACHE" and not bool(local_flags.get("any_hit")):
                result_bucket = dict(result_bucket or {})
                result_bucket["terminal"] = True
                result_bucket["suggestion"] = _OFFLINE_TERMINAL_SUGGESTION
                if not str(result_bucket.get("error_detail") or "").strip():
                    result_bucket["error_detail"] = "No local offline price sources were available."
                diagnostic["result"] = result_bucket
                diagnostic["suggestions"] = [_OFFLINE_TERMINAL_SUGGESTION]

            diagnostic = sanitize_json_value(diagnostic)

        status = "OK" if snapshot is not None else "MISSING"
        if status == "OK":
            ok_count += 1
        result_bucket = diagnostic.get("result") if isinstance(diagnostic, dict) else {}
        reason_code = str((result_bucket or {}).get("reason_code") or ("PROVIDER_OK" if snapshot is not None else "PROVIDER_NO_DATA"))
        reason_counts[reason_code] = reason_counts.get(reason_code, 0) + 1

        ticker_payload = {
            "ticker": ticker,
            "requested_as_of_date": as_of_date,
            "status": status,
            "reason_code": reason_code,
            "diagnostic": diagnostic,
            "snapshot": _snapshot_payload(snapshot) if snapshot is not None else None,
        }
        ticker_path.write_text(json.dumps(ticker_payload, indent=2), encoding="utf-8")
        rows.append(
            {
                "ticker": ticker,
                "status": status,
                "as_of_used": snapshot.as_of_date if snapshot is not None else None,
                "price": snapshot.price if snapshot is not None else None,
                "source": snapshot.source if snapshot is not None else None,
                "reason_code": reason_code,
                "path": str(ticker_path),
            }
        )

    summary = {
        "run_id": run_id,
        "requested_as_of_date": as_of_date,
        "output_dir": str(out_dir),
        "provider_effective": str(getattr(provider, "provider_name", "unknown")),
        "fallback_days": int(fallback_days if fallback_days is not None else cfg.price_fallback_days),
        "local_only": bool(local_only),
        "ticker_count": len(rows),
        "ok_count": int(ok_count),
        "missing_count": int(len(rows) - ok_count),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: kv[0])),
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    summary_path = out_dir / "prices_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
