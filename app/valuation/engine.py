from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

from app.config import get_config
from app.db import get_db, utc_now_iso
from app.market.price_provider import (
    PriceSnapshot,
    build_price_provider,
    price_reason_suggestion,
    resolve_price_from_historical_runs,
)
from app.market.shares_guard import GUARD_REFUSED
from app.market.shares_provider import SharesSnapshot, build_shares_provider
from app.util.hashing import sha256_text
from app.valuation.fcf import resolve_fcf_asof
from app.valuation.facts import write_facts_coverage_for_run
from app.valuation.fundamentals import UNKNOWN, FUNDAMENTALS_VERSION, build_fundamentals_frame
from app.valuation.guards import validate_denominators
from app.valuation.net_debt import resolve_net_debt_proxy
from app.valuation.reverse_dcf import implied_growth_from_price
from app.valuation.shares import resolve_shares_asof


VALUATION_VERSION = "v1.1"
ALLOWED_VALUATION_REASON_CODES = {
    "PRICE_UNKNOWN",
    "MISSING_FCF",
    "MISSING_SHARES",
    "MISSING_NET_DEBT",
    "INVALID_DENOMINATOR",
    "MODEL_PRECONDITION_FAILED",
    "ENGINE_EXCEPTION",
    "VALUATION_ANOMALY",
}
_VALUATION_ANOMALY_ABS_MAX = 100.0


def _is_num(value: Any) -> bool:
    """A finite real number: bool, NaN and infinity are not values (a True
    share count read as 1.0, and NaN passed every ``<= 0`` guard)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def _is_ticker_symbol(value: str) -> bool:
    token = str(value or "").strip().upper()
    if token in {"", "SUMMARY"}:
        return False
    return bool(_TICKER_RE.match(token))


def _signal_status(derived_signals: dict[str, Any], signal: str) -> str:
    node = derived_signals.get(signal) if isinstance(derived_signals, dict) else None
    value = node.get("value") if isinstance(node, dict) else UNKNOWN
    return "OK" if _is_num(value) else "UNKNOWN"


def _snapshot_from_payload(payload: dict[str, Any] | None) -> PriceSnapshot | None:
    if not isinstance(payload, dict):
        return None
    ticker = str(payload.get("ticker") or "").upper()
    as_of_date = str(payload.get("as_of_date") or "")
    price = payload.get("price")
    if not ticker or not as_of_date or not _is_num(price):
        return None
    confidence = str(payload.get("confidence") or "LOW").upper()
    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        confidence = "LOW"
    return PriceSnapshot(
        ticker=ticker,
        as_of_date=as_of_date,
        price=float(price),
        currency=str(payload.get("currency") or "USD"),
        source=str(payload.get("source") or "UNKNOWN"),
        retrieved_at=str(payload.get("retrieved_at") or utc_now_iso()),
        url=str(payload.get("url")) if payload.get("url") else None,
        confidence=confidence,  # type: ignore[arg-type]
    )


def _positive_finite_price(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _load_run_scoped_price_snapshot(run_id: str, ticker: str, as_of_date: str) -> tuple[PriceSnapshot | None, Path]:
    cfg = get_config()
    path = cfg.outputs_dir / "prices" / run_id / f"{ticker.upper()}.json"
    if not path.exists():
        return None, path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, path
    if not isinstance(payload, dict):
        return None, path
    status = str(payload.get("status") or "").upper()
    requested = str(payload.get("requested_as_of_date") or "")
    if status != "OK":
        return None, path
    # Exact date match required: an undated artifact is not evidence for a dated request.
    if not requested.strip() or requested.strip() != str(as_of_date).strip():
        return None, path
    raw_snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else None
    if not _positive_finite_price((raw_snapshot or {}).get("price")):
        return None, path
    snapshot = _snapshot_from_payload(raw_snapshot)
    if snapshot is None:
        return None, path
    return snapshot, path


def _load_disk_cache_snapshot(ticker: str, as_of_date: str) -> tuple[PriceSnapshot | None, Path]:
    cfg = get_config()
    path = cfg.cache_dir / "prices" / f"{ticker.upper()}.json"
    if not path.exists():
        return None, path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, path
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return None, path
    candidates: list[tuple[str, PriceSnapshot]] = []
    for row in entries:
        if not isinstance(row, dict):
            continue
        if str(row.get("requested_as_of_date") or "") != str(as_of_date):
            continue
        source = str(row.get("source") or "")
        snapshot = _snapshot_from_payload(row.get("snapshot") if isinstance(row.get("snapshot"), dict) else None)
        if snapshot is None:
            continue
        candidates.append((source, snapshot))
    if not candidates:
        return None, path
    candidates.sort(key=lambda item: (item[0], item[1].as_of_date, item[1].source))
    return candidates[0][1], path


def _price_source_resolution(coverage: dict[str, Any]) -> str:
    output_fields = coverage.get("output_fields") if isinstance(coverage, dict) else None
    source = (output_fields.get("price_source") if isinstance(output_fields, dict) else None) or ""
    source_norm = str(source).strip().lower()
    if source_norm in {"run_scoped_output", "run_scoped_price_fetch"}:
        return "run_scoped_output"
    if source_norm in {"disk_cache"}:
        return "disk_cache"
    if source_norm in {"db_quote_cache"}:
        return "db_quote_cache"
    if source_norm in {"historical_run_artifacts"}:
        return "historical_run_artifacts"
    if source_norm in {"stooq", "stooq_secondary", "fallback", "stooq+fallback"}:
        return "provider_live_fetch"
    if source_norm in {"disabled", "unknown", ""}:
        return "provider_live_fetch"
    return "provider_live_fetch"


def get_default_provider(cfg=None, *, with_prices: bool = True, fallback_days: int | None = None):
    return build_price_provider(cfg=cfg, with_prices=with_prices, fallback_days=fallback_days)


def get_default_shares_provider(cfg=None, *, cache_only: bool = False):
    return build_shares_provider(cfg=cfg, cache_only=cache_only)


def _parse_iso(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return datetime.now(timezone.utc)


def _latest_price_quote(ticker: str, as_of_date: str) -> PriceSnapshot | None:
    try:
        with get_db() as conn:
            row = conn.execute(
                """
                SELECT provider, price, status, source_url, as_of_date, fetched_at
                FROM price_quotes
                WHERE ticker = ? AND as_of_date <= ?
                  AND status = 'OK'
                  AND price IS NOT NULL
                ORDER BY as_of_date DESC, fetched_at DESC
                LIMIT 1
                """,
                (ticker, as_of_date),
            ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    if str(row["status"] or "").upper() != "OK":
        return None
    if not _is_num(row["price"]):
        return None
    used_asof = str(row["as_of_date"] or as_of_date)
    return PriceSnapshot(
        ticker=ticker.upper(),
        as_of_date=used_asof,
        price=float(row["price"]),
        currency="USD",
        source=str(row["provider"] or "db_cache"),
        retrieved_at=str(row["fetched_at"] or utc_now_iso()),
        url=str(row["source_url"]) if row["source_url"] else None,
        confidence="HIGH" if used_asof == str(as_of_date) else "MEDIUM",
    )


def _price_evidence_id(ticker: str, snapshot: PriceSnapshot) -> str:
    raw = f"{ticker.upper()}|{snapshot.source}|{snapshot.as_of_date}|{snapshot.price}"
    return f"price_{sha256_text(raw)[:16]}"


def _upsert_price_quote(snapshot: PriceSnapshot) -> None:
    payload = {
        "ticker": snapshot.ticker,
        "as_of_date": snapshot.as_of_date,
        "price": snapshot.price,
        "currency": snapshot.currency or "USD",
        "source": snapshot.source,
        "url": snapshot.url,
        "confidence": snapshot.confidence,
        "retrieved_at": snapshot.retrieved_at,
    }
    fetched_at = snapshot.retrieved_at or utc_now_iso()
    expires_at = (_parse_iso(fetched_at) + timedelta(days=7)).isoformat()
    quote_hash = sha256_text(json.dumps(payload, sort_keys=True))
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO price_quotes(
                ticker, provider, as_of_date, price, currency, source_url,
                status, fetched_at, expires_at, raw_json, quote_hash
            ) VALUES(?, ?, ?, ?, ?, ?, 'OK', ?, ?, ?, ?)
            ON CONFLICT(ticker, provider, as_of_date) DO UPDATE SET
                price=excluded.price,
                currency=excluded.currency,
                source_url=excluded.source_url,
                status=excluded.status,
                fetched_at=excluded.fetched_at,
                expires_at=excluded.expires_at,
                raw_json=excluded.raw_json,
                quote_hash=excluded.quote_hash
            """,
            (
                snapshot.ticker,
                snapshot.source,
                snapshot.as_of_date,
                float(snapshot.price),
                snapshot.currency or "USD",
                snapshot.url or "",
                fetched_at,
                expires_at,
                json.dumps(payload, sort_keys=True),
                quote_hash,
            ),
        )


def _persist_price_evidence_item(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
    snapshot: PriceSnapshot,
) -> str | None:
    evidence_id = _price_evidence_id(ticker, snapshot)
    excerpt = (
        f"{ticker.upper()} close price {float(snapshot.price):.4f} {snapshot.currency or 'USD'} "
        f"as of {snapshot.as_of_date} (source={snapshot.source})."
    )
    source_url = snapshot.url or f"source:{snapshot.source}"
    item_hash = sha256_text(
        json.dumps(
            {
                "ticker": ticker.upper(),
                "as_of_date": as_of_date,
                "source": snapshot.source,
                "price_asof": snapshot.as_of_date,
                "price": float(snapshot.price),
            },
            sort_keys=True,
        )
    )
    now = utc_now_iso()
    citations = [{"source_url": source_url, "snippet": excerpt[:180], "section_label": "price_snapshot"}]
    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO evidence_items(
                    evidence_id, ticker, as_of_date, run_id, adapter_run_id, source_type, source_url, source_title,
                    source_published_at, retrieved_at, excerpt_text, excerpt_hash, content_hash, dedupe_key,
                    citations_json, derived_from_json, item_hash, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evidence_id) DO UPDATE SET
                    run_id=excluded.run_id,
                    adapter_run_id=excluded.adapter_run_id,
                    source_type=excluded.source_type,
                    source_url=excluded.source_url,
                    source_title=excluded.source_title,
                    source_published_at=excluded.source_published_at,
                    retrieved_at=excluded.retrieved_at,
                    excerpt_text=excluded.excerpt_text,
                    excerpt_hash=excluded.excerpt_hash,
                    content_hash=excluded.content_hash,
                    dedupe_key=excluded.dedupe_key,
                    citations_json=excluded.citations_json,
                    derived_from_json=excluded.derived_from_json,
                    item_hash=excluded.item_hash
                """,
                (
                    evidence_id,
                    ticker.upper(),
                    as_of_date,
                    run_id,
                    run_id,
                    "price",
                    source_url,
                    f"{ticker.upper()} price snapshot",
                    snapshot.as_of_date,
                    snapshot.retrieved_at or now,
                    excerpt,
                    sha256_text(excerpt),
                    sha256_text(excerpt),
                    item_hash,
                    json.dumps(citations),
                    json.dumps(["valuation.input_snapshot.current_price", f"source:{snapshot.source}"]),
                    item_hash,
                    now,
                ),
            )
    except Exception:
        return None
    return evidence_id


def _shares_evidence_id(ticker: str, snapshot: SharesSnapshot) -> str:
    raw = (
        f"{ticker.upper()}|{snapshot.source}|{snapshot.as_of_date}|{float(snapshot.shares_outstanding)}|"
        f"{snapshot.filing_accession or ''}|{snapshot.filing_date or ''}"
    )
    return f"shares_{sha256_text(raw)[:16]}"


def _persist_shares_evidence_item(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
    snapshot: SharesSnapshot,
) -> str | None:
    evidence_id = snapshot.evidence_id or _shares_evidence_id(ticker, snapshot)
    excerpt = (
        f"{ticker.upper()} shares outstanding {float(snapshot.shares_outstanding):.0f} "
        f"(resolved_via={snapshot.resolved_via}, filing_date={snapshot.filing_date or snapshot.as_of_date})."
    )
    source_url = snapshot.url or f"source:{snapshot.source}"
    item_hash = sha256_text(
        json.dumps(
            {
                "ticker": ticker.upper(),
                "as_of_date": as_of_date,
                "shares_asof": snapshot.as_of_date,
                "shares_outstanding": float(snapshot.shares_outstanding),
                "source": snapshot.source,
                "resolved_via": snapshot.resolved_via,
                "filing_accession": snapshot.filing_accession,
            },
            sort_keys=True,
        )
    )
    now = utc_now_iso()
    section_label = "cover_page" if str(snapshot.resolved_via).upper() == "COVER_PAGE" else "xbrl_concept"
    citations = [{"source_url": source_url, "snippet": excerpt[:180], "section_label": section_label}]
    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO evidence_items(
                    evidence_id, ticker, as_of_date, run_id, adapter_run_id, source_type, source_url, source_title,
                    source_published_at, retrieved_at, excerpt_text, excerpt_hash, content_hash, dedupe_key,
                    citations_json, derived_from_json, item_hash, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evidence_id) DO UPDATE SET
                    run_id=excluded.run_id,
                    adapter_run_id=excluded.adapter_run_id,
                    source_type=excluded.source_type,
                    source_url=excluded.source_url,
                    source_title=excluded.source_title,
                    source_published_at=excluded.source_published_at,
                    retrieved_at=excluded.retrieved_at,
                    excerpt_text=excluded.excerpt_text,
                    excerpt_hash=excluded.excerpt_hash,
                    content_hash=excluded.content_hash,
                    dedupe_key=excluded.dedupe_key,
                    citations_json=excluded.citations_json,
                    derived_from_json=excluded.derived_from_json,
                    item_hash=excluded.item_hash
                """,
                (
                    evidence_id,
                    ticker.upper(),
                    as_of_date,
                    run_id,
                    run_id,
                    "shares",
                    source_url,
                    f"{ticker.upper()} shares snapshot",
                    snapshot.filing_date or snapshot.as_of_date,
                    snapshot.retrieved_at or now,
                    excerpt,
                    sha256_text(excerpt),
                    sha256_text(excerpt),
                    item_hash,
                    json.dumps(citations),
                    json.dumps(["valuation.input_snapshot.shares_outstanding", f"source:{snapshot.source}"]),
                    item_hash,
                    now,
                ),
            )
    except Exception:
        return None
    return evidence_id


def _snapshot_from_legacy_quote_obj(ticker: str, requested_as_of: str, quote: Any) -> PriceSnapshot | None:
    price = getattr(quote, "price", None)
    if not _is_num(price):
        return None
    status = str(getattr(quote, "status", "OK") or "OK").upper()
    if status not in {"OK", "SUCCESS", ""}:
        return None
    used_asof = str(getattr(quote, "as_of_date", requested_as_of) or requested_as_of)
    source = str(getattr(quote, "provider", "") or getattr(quote, "source", "") or "legacy")
    source_url = getattr(quote, "source_url", None)
    confidence = "HIGH" if used_asof == requested_as_of else "MEDIUM"
    return PriceSnapshot(
        ticker=ticker.upper(),
        as_of_date=used_asof,
        price=float(price),
        currency=str(getattr(quote, "currency", "USD") or "USD"),
        source=source,
        retrieved_at=str(getattr(quote, "fetched_at", "") or utc_now_iso()),
        url=str(source_url) if source_url else None,
        confidence=confidence,  # type: ignore[arg-type]
    )


def _fetch_provider_snapshot(
    ticker: str,
    as_of_date: str,
    *,
    with_prices: bool,
    fallback_days: int | None = None,
) -> tuple[PriceSnapshot | None, dict[str, Any] | None]:
    provider = get_default_provider(get_config(), with_prices=with_prices, fallback_days=fallback_days)
    diag: dict[str, Any] | None = None
    if hasattr(provider, "get_price_asof"):
        snapshot = provider.get_price_asof(ticker=ticker, as_of_date=as_of_date)
        if hasattr(provider, "get_last_diagnostic"):
            try:
                diag = provider.get_last_diagnostic(ticker, as_of_date)
            except Exception:
                diag = None
        if snapshot and _is_num(snapshot.price):
            return snapshot, diag
        return None, diag
    if hasattr(provider, "get_quote"):
        quote = provider.get_quote(ticker=ticker, as_of_date=as_of_date)
        return _snapshot_from_legacy_quote_obj(ticker, as_of_date, quote), diag
    return None, diag


def _default_price_coverage_entry(
    *,
    ticker: str,
    requested_as_of: str,
    with_prices: bool,
) -> dict[str, Any]:
    cache_path = get_config().cache_dir / "prices" / f"{ticker.upper()}.json"
    if with_prices:
        reason_code = "CACHE_MISS"
        reason_detail = "No cached snapshot and no provider result."
    else:
        reason_code = "PROVIDER_NO_DATA"
        reason_detail = "Price lookup disabled via --no-with-prices."
    return {
        "ticker": ticker.upper(),
        "requested_as_of": requested_as_of,
        "resolved_symbol": None,
        "attempted_symbols": [],
        "asof_final_used": None,
        "source_resolution": "unknown",
        "provider_attempts": [],
        "cache": {
            "hit": False,
            "path": str(cache_path),
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
            "reason_code": reason_code,
            "reason_detail": reason_detail,
        },
        "output_fields": {
            "current_price": UNKNOWN,
            "price_asof_used": None,
            "asof_final_used": None,
            "price_source": None,
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


def _price_coverage_suggestions(entry: dict[str, Any]) -> list[str]:
    result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
    reason_code = str((result or {}).get("reason_code") or "UNKNOWN").upper()
    suggestions: list[str] = []
    if reason_code in {"SYMBOL_UNMAPPED", "SYMBOL_NOT_FOUND"}:
        suggestions.append("Add symbol override entry to config/price_symbol_overrides.csv.")
    if reason_code == "NON_TRADING_DAY_NO_FALLBACK":
        suggestions.append("Increase fallback-days for price fetch/prewarm to walk back to a prior trading day.")
    if reason_code == "OFFLINE_NO_CACHE":
        suggestions.append(price_reason_suggestion("OFFLINE_NO_CACHE"))
    if reason_code in {"DNS_FAILURE", "TLS_FAILURE", "TIMEOUT"}:
        suggestions.append("Retry later, check connectivity, or increase fallback-days.")
    if reason_code == "RATE_LIMIT":
        suggestions.append("Reduce workers or increase budget.")
    if reason_code in {"PROVIDER_NO_DATA", "HTTP_4XX", "HTTP_5XX", "PARSE_ERROR"}:
        suggestions.append("Inspect provider_attempts and retry with explicit ticker prewarm.")
    if reason_code == "BUDGET_EXHAUSTED":
        suggestions.append("Increase domain request budget or retry after budget reset.")
    if not suggestions:
        fallback = price_reason_suggestion(reason_code)
        suggestions.append(fallback or "Inspect provider_attempts and cache metadata for this ticker.")
    return suggestions


def _select_price_snapshot(
    ticker: str,
    as_of_date: str,
    *,
    run_id: str | None = None,
    with_prices: bool,
    fallback_days: int | None = None,
) -> tuple[PriceSnapshot | None, str, dict[str, Any]]:
    coverage = _default_price_coverage_entry(
        ticker=ticker,
        requested_as_of=as_of_date,
        with_prices=with_prices,
    )
    local_flags = coverage.get("local_fallbacks") if isinstance(coverage.get("local_fallbacks"), dict) else {}
    if not isinstance(local_flags, dict):
        local_flags = {}
        coverage["local_fallbacks"] = local_flags
    if not with_prices:
        coverage["source_resolution"] = "prices_disabled"
        return None, "prices_disabled", coverage
    local_flags["run_scoped_output_checked"] = True
    if run_id:
        run_snapshot, run_path = _load_run_scoped_price_snapshot(run_id, ticker, as_of_date)
    else:
        run_snapshot, run_path = None, get_config().outputs_dir / "prices" / "unknown" / f"{ticker.upper()}.json"
    if run_snapshot is not None:
        local_flags["run_scoped_output_hit"] = True
        local_flags["any_hit"] = True
        coverage["resolved_symbol"] = str(run_snapshot.ticker).lower() if run_snapshot.ticker else None
        coverage["attempted_symbols"] = [f"{str(run_snapshot.ticker).lower()}.us"] if run_snapshot.ticker else []
        coverage["asof_final_used"] = run_snapshot.as_of_date
        coverage["cache"] = {
            "hit": True,
            "path": str(run_path),
            "snapshot_found": True,
            "cached_as_of_used": run_snapshot.as_of_date,
        }
        coverage["market_day"]["asof_used"] = run_snapshot.as_of_date
        coverage["provider_attempts"] = [
            {
                "provider": "run_scoped_price_fetch",
                "status": "CACHE_HIT",
                "url": run_snapshot.url,
                "took_ms": 0,
            }
        ]
        coverage["result"] = {
            "status": "OK",
            "reason_code": "CACHE_HIT",
            "reason_detail": "Resolved from outputs/prices/<run_id> cache.",
        }
        coverage["source_resolution"] = "run_scoped_output"
        coverage["output_fields"] = {
            "current_price": float(run_snapshot.price),
            "price_asof_used": run_snapshot.as_of_date,
            "asof_final_used": run_snapshot.as_of_date,
            "price_source": "run_scoped_output",
            "confidence": run_snapshot.confidence,
        }
        coverage["suggestions"] = []
        return run_snapshot, "run_scoped_output", coverage

    local_flags["disk_cache_checked"] = True
    disk_snapshot, disk_path = _load_disk_cache_snapshot(ticker, as_of_date)
    if disk_snapshot is not None:
        local_flags["disk_cache_hit"] = True
        local_flags["any_hit"] = True
        coverage["resolved_symbol"] = str(disk_snapshot.ticker).lower() if disk_snapshot.ticker else None
        coverage["attempted_symbols"] = [f"{str(disk_snapshot.ticker).lower()}.us"] if disk_snapshot.ticker else []
        coverage["asof_final_used"] = disk_snapshot.as_of_date
        coverage["cache"] = {
            "hit": True,
            "path": str(disk_path),
            "snapshot_found": True,
            "cached_as_of_used": disk_snapshot.as_of_date,
        }
        coverage["market_day"]["asof_used"] = disk_snapshot.as_of_date
        coverage["provider_attempts"] = [
            {
                "provider": str(disk_snapshot.source or "disk_cache"),
                "status": "CACHE_HIT",
                "url": disk_snapshot.url,
                "took_ms": 0,
            }
        ]
        coverage["result"] = {
            "status": "OK",
            "reason_code": "CACHE_HIT",
            "reason_detail": "Resolved from data/cache/prices cache.",
        }
        coverage["source_resolution"] = "disk_cache"
        coverage["output_fields"] = {
            "current_price": float(disk_snapshot.price),
            "price_asof_used": disk_snapshot.as_of_date,
            "asof_final_used": disk_snapshot.as_of_date,
            "price_source": "disk_cache",
            "confidence": disk_snapshot.confidence,
        }
        coverage["suggestions"] = []
        return disk_snapshot, "disk_cache", coverage

    local_flags["db_quote_cache_checked"] = True
    db_snapshot = _latest_price_quote(ticker, as_of_date)
    if db_snapshot is not None:
        local_flags["db_quote_cache_hit"] = True
        local_flags["any_hit"] = True
        coverage["resolved_symbol"] = str(db_snapshot.ticker).lower() if db_snapshot.ticker else None
        coverage["attempted_symbols"] = [f"{str(db_snapshot.ticker).lower()}.us"] if db_snapshot.ticker else []
        coverage["asof_final_used"] = db_snapshot.as_of_date
        requested_day_type = "TRADING" if db_snapshot.as_of_date == str(as_of_date) else "NON_TRADING"
        fallback_days_checked = 1
        requested_dt = None
        used_dt = None
        try:
            requested_dt = datetime.strptime(str(as_of_date), "%Y-%m-%d").date()
            used_dt = datetime.strptime(str(db_snapshot.as_of_date), "%Y-%m-%d").date()
        except Exception:
            requested_dt = None
            used_dt = None
        if requested_dt is not None and used_dt is not None and used_dt <= requested_dt:
            fallback_days_checked = int((requested_dt - used_dt).days) + 1
        coverage["cache"] = {
            "hit": True,
            "path": str(get_config().db_path),
            "snapshot_found": True,
            "cached_as_of_used": db_snapshot.as_of_date,
        }
        coverage["market_day"] = {
            "requested_day_type": requested_day_type,
            "fallback_days_checked": max(1, fallback_days_checked),
            "asof_used": db_snapshot.as_of_date,
        }
        coverage["provider_attempts"] = [
            {
                "provider": "db_quote_cache",
                "status": "CACHE_HIT",
                "url": db_snapshot.url,
                "took_ms": 0,
            }
        ]
        coverage["result"] = {
            "status": "OK",
            "reason_code": "CACHE_HIT",
            "reason_detail": "Resolved from price_quotes cache.",
        }
        coverage["source_resolution"] = "db_quote_cache"
        coverage["output_fields"] = {
            "current_price": float(db_snapshot.price),
            "price_asof_used": db_snapshot.as_of_date,
            "asof_final_used": db_snapshot.as_of_date,
            "price_source": "db_quote_cache",
            "confidence": db_snapshot.confidence,
        }
        coverage["suggestions"] = []
        return db_snapshot, "db_quote_cache", coverage

    local_flags["historical_run_artifacts_checked"] = True
    historical = resolve_price_from_historical_runs(
        ticker=ticker,
        as_of_date=as_of_date,
        sectors_dir=get_config().sectors_dir,
    )
    if historical is not None:
        local_flags["historical_run_artifacts_hit"] = True
        local_flags["any_hit"] = True
        coverage["resolved_symbol"] = str(historical.ticker).lower() if historical.ticker else None
        coverage["attempted_symbols"] = [f"{str(historical.ticker).lower()}.us"] if historical.ticker else []
        coverage["asof_final_used"] = historical.as_of_date
        coverage["cache"] = {
            "hit": True,
            "path": str(get_config().sectors_dir),
            "snapshot_found": True,
            "cached_as_of_used": historical.as_of_date,
        }
        coverage["market_day"] = {
            "requested_day_type": "TRADING" if historical.as_of_date == str(as_of_date) else "NON_TRADING",
            "fallback_days_checked": 1,
            "asof_used": historical.as_of_date,
        }
        coverage["provider_attempts"] = [
            {
                "provider": "historical_run_artifacts",
                "status": "CACHE_HIT",
                "url": None,
                "took_ms": 0,
            }
        ]
        coverage["result"] = {
            "status": "OK",
            "reason_code": "CACHE_HIT",
            "reason_detail": "Resolved from prior sector run artifacts.",
            "retryable": False,
            "error_detail": "",
            "suggestion": "",
        }
        coverage["source_resolution"] = "historical_run_artifacts"
        coverage["output_fields"] = {
            "current_price": float(historical.price),
            "price_asof_used": historical.as_of_date,
            "asof_final_used": historical.as_of_date,
            "price_source": "historical_run_artifacts",
            "confidence": historical.confidence,
        }
        coverage["suggestions"] = []
        return historical, "historical_run_artifacts", coverage

    fetched, provider_diag = _fetch_provider_snapshot(
        ticker,
        as_of_date,
        with_prices=with_prices,
        fallback_days=fallback_days,
    )
    if isinstance(provider_diag, dict):
        coverage.update(provider_diag)
        coverage["local_fallbacks"] = local_flags
    coverage["source_resolution"] = "provider_live_fetch"
    if fetched is None:
        result = coverage.get("result") if isinstance(coverage.get("result"), dict) else {}
        reason_code = str((result or {}).get("reason_code") or "").upper()
        force_terminal_offline = False
        if reason_code == "OFFLINE_NO_CACHE" and not bool(local_flags.get("any_hit")):
            result = dict(result or {})
            result["terminal"] = True
            result["suggestion"] = price_reason_suggestion("OFFLINE_NO_CACHE")
            if not str(result.get("error_detail") or "").strip():
                result["error_detail"] = "No local offline price sources were available."
            coverage["result"] = result
            force_terminal_offline = True
        coverage["suggestions"] = _price_coverage_suggestions(coverage)
        if force_terminal_offline:
            coverage["suggestions"] = [price_reason_suggestion("OFFLINE_NO_CACHE")]
        return None, "provider_missing", coverage
    _upsert_price_quote(fetched)
    output_fields = coverage.get("output_fields")
    if not isinstance(output_fields, dict):
        output_fields = {}
    output_fields.update(
        {
            "current_price": float(fetched.price),
            "price_asof_used": fetched.as_of_date,
            "asof_final_used": fetched.as_of_date,
            "price_source": fetched.source,
            "confidence": fetched.confidence,
        }
    )
    coverage["output_fields"] = output_fields
    coverage["source_resolution"] = "provider_live_fetch"
    coverage["suggestions"] = []
    return fetched, "provider_live", coverage


def _default_shares_coverage_entry(
    *,
    ticker: str,
    requested_as_of: str,
) -> dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "requested_as_of": requested_as_of,
        "shares_status": "UNKNOWN",
        "shares_reason_code": "NO_CURRENT_RUN_SHARES",
        "shares_reason_detail": "No shares snapshot was resolved.",
        "shares_value": UNKNOWN,
        "shares_asof_used": None,
        "shares_source": None,
        "shares_source_resolution": "unknown",
        "confidence": None,
        "resolved_via": "UNKNOWN",
        "cache": {"hit": False, "path": None, "snapshot_found": False, "cached_as_of_used": None},
        "provider_attempts": [],
        "derived_from": [f"sector.shares_resolver[{ticker.upper()}]"],
        "shares_evidence_ref": None,
    }


def _shares_source_resolution(coverage: dict[str, Any]) -> str:
    source = str(coverage.get("shares_source_resolution") or "").strip().lower().replace("-", "_")
    if source in {
        "current_run_fundamentals",
        "current_run_dossier",
        "historical_dossier",
        "companyfacts_cache",
        "companyfacts_fetch",
        "derived_market_cap_price",
        "unknown",
    }:
        return source
    shares_source = str(coverage.get("shares_source") or "").strip().lower()
    if shares_source.startswith("historical_dossier"):
        return "historical_dossier"
    if shares_source in {"current_run_fundamentals", "current_run_dossier"}:
        return shares_source
    if shares_source in {"companyfacts_cache", "companyfacts_fetch"}:
        return shares_source
    if shares_source == "derived_market_cap_price":
        return "derived_market_cap_price"
    return "unknown"


def _default_fcf_coverage_entry(
    *,
    ticker: str,
    requested_as_of: str,
) -> dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "requested_as_of": requested_as_of,
        "fcf_status": "UNKNOWN",
        "fcf_reason_code": "DOSSIER_NO_FCF",
        "fcf_reason_detail": "No FCF snapshot was resolved.",
        "fcf_value": UNKNOWN,
        "fcf_asof_used": None,
        "fcf_source": None,
        "fcf_source_resolution": "unknown",
        "cfo_status": "UNKNOWN",
        "cfo_reason_code": "MISSING_CFO",
        "cfo_value": UNKNOWN,
        "cfo_asof_used": None,
        "capex_status": "UNKNOWN",
        "capex_reason_code": "MISSING_CAPEX",
        "capex_value": UNKNOWN,
        "capex_asof_used": None,
        "derived_from": [f"sector.fcf_resolver[{ticker.upper()}]"],
        "fcf_evidence_ref": None,
    }


def _default_net_debt_coverage_entry(
    *,
    ticker: str,
    requested_as_of: str,
) -> dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "requested_as_of": requested_as_of,
        "net_debt_status": "UNKNOWN",
        "net_debt_reason_code": "NO_NET_DEBT_PROXY",
        "net_debt_reason_detail": "No net debt snapshot was resolved.",
        "net_debt_value": UNKNOWN,
        "net_debt_asof_used": None,
        "net_debt_source": None,
        "net_debt_source_resolution": "unknown",
        "derived_from": [f"sector.net_debt_coverage.entries[{ticker.upper()}]"],
        "net_debt_evidence_ref": None,
        "total_debt_value": UNKNOWN,
        "cash_equivalents_value": UNKNOWN,
        "tags_used": {},
        "fact_dates_used": {},
    }


def _net_debt_source_resolution(coverage: dict[str, Any]) -> str:
    source = str(coverage.get("net_debt_source_resolution") or "").strip().lower().replace("-", "_")
    if source in {"current_run_fundamentals", "companyfacts_cache", "companyfacts_fetch"}:
        return source
    source_value = str(coverage.get("net_debt_source") or "").strip().lower()
    if source_value in {"current_run_fundamentals", "companyfacts_cache", "companyfacts_fetch"}:
        return source_value
    refs = [str(ref) for ref in (coverage.get("derived_from") or []) if str(ref).strip()]
    if any(ref.startswith("facts_coverage.rows[") or ref.startswith("companyfacts.") for ref in refs):
        return "companyfacts_cache"
    return "unknown"


def _fcf_source_resolution(coverage: dict[str, Any]) -> str:
    source = str(coverage.get("fcf_source_resolution") or "").strip().lower().replace("-", "_")
    if source in {
        "current_run_dossier",
        "historical_dossier",
        "companyfacts_cache",
        "companyfacts_fetch",
        "derived_cfo_minus_capex",
        "unknown",
    }:
        return source
    fcf_source = str(coverage.get("fcf_source") or "").strip().lower()
    if fcf_source.startswith("historical_dossier"):
        return "historical_dossier"
    if fcf_source == "current_run_dossier":
        return "current_run_dossier"
    if fcf_source in {"companyfacts_cache", "companyfacts_fetch"}:
        return fcf_source
    refs = [str(ref) for ref in (coverage.get("derived_from") or []) if str(ref).strip()]
    if any("derived:fcf=cfo-capex" in ref for ref in refs):
        return "derived_cfo_minus_capex"
    return "unknown"


def _select_shares_snapshot(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str | None,
) -> tuple[SharesSnapshot | None, dict[str, Any]]:
    coverage = _default_shares_coverage_entry(ticker=ticker, requested_as_of=as_of_date)
    provider = get_default_shares_provider(get_config(), cache_only=False)
    try:
        snapshot = provider.get_shares_asof(ticker=ticker, as_of_date=as_of_date, run_id=run_id)
    except TypeError:
        snapshot = provider.get_shares_asof(ticker=ticker, as_of_date=as_of_date)  # type: ignore[call-arg]
    diag = None
    if hasattr(provider, "get_last_diagnostic"):
        try:
            diag = provider.get_last_diagnostic(ticker=ticker, as_of_date=as_of_date, run_id=run_id)  # type: ignore[misc]
        except TypeError:
            try:
                diag = provider.get_last_diagnostic(ticker=ticker, as_of_date=as_of_date)  # type: ignore[misc]
            except Exception:
                diag = None
        except Exception:
            diag = None
    if isinstance(diag, dict):
        result = diag.get("result") if isinstance(diag.get("result"), dict) else {}
        outputs = diag.get("output_fields") if isinstance(diag.get("output_fields"), dict) else {}
        cache = diag.get("cache") if isinstance(diag.get("cache"), dict) else {}
        coverage["provider_attempts"] = [row for row in (diag.get("provider_attempts") or []) if isinstance(row, dict)]
        coverage["cache"] = {
            "hit": bool(cache.get("hit")),
            "path": str(cache.get("path") or coverage["cache"]["path"]),
            "snapshot_found": bool(cache.get("snapshot_found")),
            "cached_as_of_used": cache.get("cached_as_of_used"),
        }
        coverage["shares_status"] = str(result.get("status") or "UNKNOWN").upper()
        if coverage["shares_status"] not in {"OK", "UNKNOWN"}:
            coverage["shares_status"] = "UNKNOWN"
        coverage["shares_reason_code"] = str(result.get("reason_code") or "NO_FILINGS")
        coverage["shares_reason_detail"] = str(result.get("reason_detail") or "")
        coverage["shares_value"] = (
            float(outputs.get("shares_outstanding"))
            if _is_num(outputs.get("shares_outstanding"))
            else UNKNOWN
        )
        coverage["shares_asof_used"] = outputs.get("shares_asof_used")
        coverage["shares_source"] = outputs.get("shares_source")
        coverage["confidence"] = outputs.get("confidence")
        coverage["resolved_via"] = outputs.get("resolved_via") or "UNKNOWN"
        coverage["shares_source_resolution"] = (
            "run_scoped_output"
            if str(outputs.get("shares_source") or "").strip().lower() == "run_scoped_output"
            else (
                "disk_cache"
                if str(outputs.get("shares_source") or "").strip().lower() == "disk_cache"
                else "provider_lookup"
            )
        )
    if snapshot is not None and _is_num(snapshot.shares_outstanding):
        coverage["shares_status"] = "OK"
        coverage["shares_reason_code"] = str(coverage.get("shares_reason_code") or "PROVIDER_OK")
        coverage["shares_reason_detail"] = str(coverage.get("shares_reason_detail") or "Shares snapshot resolved.")
        coverage["shares_value"] = float(snapshot.shares_outstanding)
        coverage["shares_asof_used"] = snapshot.as_of_date
        if not coverage.get("shares_source"):
            coverage["shares_source"] = snapshot.source
        if not coverage.get("confidence"):
            coverage["confidence"] = snapshot.confidence
        coverage["resolved_via"] = snapshot.resolved_via
        if str(coverage.get("shares_source") or "").strip().lower() == "run_scoped_output":
            coverage["shares_source_resolution"] = "run_scoped_output"
        elif str(coverage.get("shares_source") or "").strip().lower() == "disk_cache":
            coverage["shares_source_resolution"] = "disk_cache"
        else:
            coverage["shares_source_resolution"] = "provider_lookup"
        return snapshot, coverage
    return None, coverage


def _select_net_debt_snapshot(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str | None,
    fundamentals_payload: dict[str, Any],
    latest_row: dict[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    coverage = _default_net_debt_coverage_entry(ticker=ticker, requested_as_of=as_of_date)
    latest_value = latest_row.get("net_debt", UNKNOWN)
    row_traces = fundamentals_payload.get("row_traces") if isinstance(fundamentals_payload.get("row_traces"), dict) else {}
    if _is_num(latest_value):
        year = int(latest_row.get("year", 0))
        trace_bucket = (
            (row_traces.get(str(year)) or {}).get("net_debt")
            if isinstance(row_traces.get(str(year)), dict)
            else None
        )
        refs = [str(x) for x in ((trace_bucket or {}).get("derived_from") or []) if str(x).strip()]
        if not refs:
            refs = [f"fundamentals.rows[{year}].net_debt"]
        coverage.update(
            {
                "net_debt_status": "OK",
                "net_debt_reason_code": "OK",
                "net_debt_reason_detail": "Resolved from in-memory fundamentals payload.",
                "net_debt_value": float(latest_value),
                "net_debt_asof_used": as_of_date or fundamentals_payload.get("as_of_date"),
                "net_debt_source": "current_run_fundamentals",
                "net_debt_source_resolution": "current_run_fundamentals",
                "derived_from": refs,
            }
        )
        return float(latest_value), coverage

    resolved = resolve_net_debt_proxy(
        ticker=ticker,
        as_of_date=as_of_date,
        run_id=run_id,
    )
    if isinstance(resolved, dict):
        coverage.update(
            {
                "net_debt_status": str(resolved.get("status") or "UNKNOWN").upper(),
                "net_debt_reason_code": str(resolved.get("reason_code") or "UNKNOWN").upper(),
                "net_debt_reason_detail": str(resolved.get("reason_code") or "UNKNOWN"),
                "net_debt_value": resolved.get("net_debt_proxy", UNKNOWN),
                "net_debt_asof_used": as_of_date,
                "net_debt_source": "companyfacts_cache",
                "total_debt_value": (
                    resolved.get("total_debt", {}).get("value", UNKNOWN)
                    if isinstance(resolved.get("total_debt"), dict)
                    else UNKNOWN
                ),
                "cash_equivalents_value": (
                    resolved.get("cash_equivalents", {}).get("value", UNKNOWN)
                    if isinstance(resolved.get("cash_equivalents"), dict)
                    else UNKNOWN
                ),
                "tags_used": {
                    "debt_tag": resolved.get("total_debt", {}).get("tag") if isinstance(resolved.get("total_debt"), dict) else None,
                    "cash_tag": resolved.get("cash_equivalents", {}).get("tag") if isinstance(resolved.get("cash_equivalents"), dict) else None,
                },
                "fact_dates_used": {
                    "debt_date": resolved.get("total_debt", {}).get("date") if isinstance(resolved.get("total_debt"), dict) else None,
                    "cash_date": resolved.get("cash_equivalents", {}).get("date") if isinstance(resolved.get("cash_equivalents"), dict) else None,
                },
                "derived_from": [str(ref) for ref in (resolved.get("derived_from") or []) if str(ref).strip()],
            }
        )
        coverage["net_debt_source_resolution"] = _net_debt_source_resolution(coverage)
        # The returned value feeds FCF-multiple equity bridges (post-lease
        # flows under ASC 842), so prefer the lease-EXCLUSIVE variant — the
        # lease-inclusive proxy double-counts the lease (audit:
        # lease-double-count-vs-postlease-fcf, review issue 1).
        if _is_num(resolved.get("net_debt_proxy_lease_exclusive")):
            coverage["net_debt_reason_detail"] = "Resolved from companyfacts-backed net debt proxy (lease-exclusive)."
            # Coverage must report the value the bridge actually uses — the
            # lease-INCLUSIVE proxy beside a '(lease-exclusive)' reason was
            # self-contradicting metadata (review EVB-5).
            coverage["net_debt_value"] = float(resolved["net_debt_proxy_lease_exclusive"])
            return float(resolved["net_debt_proxy_lease_exclusive"]), coverage
        if _is_num(resolved.get("net_debt_proxy")):
            coverage["net_debt_reason_detail"] = "Resolved from companyfacts-backed net debt proxy."
            return float(resolved["net_debt_proxy"]), coverage
    return None, coverage


def _price_source_label(snapshot: PriceSnapshot | None, origin: str) -> str:
    if snapshot is None:
        return "UNKNOWN"
    if origin in {"run_scoped_output", "disk_cache"}:
        return origin
    source = str(snapshot.source or "").strip()
    return source if source else "UNKNOWN"


def _quality_bucket(latest_row: dict[str, Any], derived_signals: dict[str, Any]) -> tuple[str, list[str]]:
    op_margin = latest_row.get("op_margin", UNKNOWN)
    fcf_margin = latest_row.get("fcf_margin", UNKNOWN)
    gm_margin = latest_row.get("gross_margin", UNKNOWN)
    om_slope = (derived_signals.get("operating_margin_trend_slope") or {}).get("value", UNKNOWN)
    refs = [
        "fundamentals.rows[-1].op_margin",
        "fundamentals.rows[-1].fcf_margin",
        "fundamentals.rows[-1].gross_margin",
        "fundamentals.derived_signals.operating_margin_trend_slope",
    ]
    score = 0
    if _is_num(op_margin) and float(op_margin) > 0.15:
        score += 1
    if _is_num(fcf_margin) and float(fcf_margin) > 0.10:
        score += 1
    if _is_num(gm_margin) and float(gm_margin) > 0.45:
        score += 1
    if _is_num(om_slope) and float(om_slope) > 0:
        score += 1
    if score >= 3:
        return "HIGH", refs
    if score >= 2:
        return "MED", refs
    return "LOW", refs


def _multiple_band(quality_bucket: str) -> tuple[float, float]:
    key = str(quality_bucket).upper()
    if key == "HIGH":
        return 20.0, 14.0
    if key == "MED":
        return 15.0, 11.0
    return 10.0, 8.0


def _owner_earnings_intrinsic(
    *,
    fcf_values: list[float],
    shares: float | str,
    net_debt: float | str,
    base_multiple: float,
    conservative_multiple: float,
) -> tuple[float | str, float | str, list[str]]:
    refs = [
        "fundamentals.rows[*].fcf",
        "fundamentals.rows[-1].shares_outstanding",
        "fundamentals.rows[-1].net_debt",
    ]
    if not _is_num(shares) or float(shares) <= 0:
        return UNKNOWN, UNKNOWN, refs
    positive_fcf = [float(value) for value in fcf_values if _is_num(value) and float(value) > 0]
    if not positive_fcf:
        return UNKNOWN, UNKNOWN, refs
    normalized_fcf = float(median(positive_fcf))
    # An unknown net debt is not zero: the caller's preflight already refuses on
    # MISSING_NET_DEBT, and this helper refuses too rather than value a company
    # whose balance sheet it cannot read.
    if not _is_num(net_debt):
        return UNKNOWN, UNKNOWN, refs
    # The stream is LEVERED: cash from operations is after the interest already
    # paid (app/valuation/fcf.py), so capitalising it yields equity value
    # directly. Subtracting net debt on top charged the debt twice. Net debt
    # stays an input for lineage and the refusal above, not for the arithmetic.
    equity_value_base = normalized_fcf * float(base_multiple)
    equity_value_cons = normalized_fcf * float(conservative_multiple)
    intrinsic_base = equity_value_base / float(shares)
    intrinsic_cons = equity_value_cons / float(shares)
    return intrinsic_base, intrinsic_cons, refs


def _valuation_reason_detail(reason_code: str | None) -> str | None:
    mapping = {
        "PRICE_UNKNOWN": "Current price is unavailable; implied return cannot be computed.",
        "MISSING_FCF": "No positive FCF observations are available for owner-earnings normalization.",
        "MISSING_SHARES": "Latest shares_outstanding is missing.",
        "MISSING_NET_DEBT": "Latest net_debt is missing.",
        "INVALID_DENOMINATOR": "A required denominator is zero or negative.",
        "MODEL_PRECONDITION_FAILED": "A valuation model precondition failed.",
        "ENGINE_EXCEPTION": "Unexpected valuation engine exception.",
        "VALUATION_ANOMALY": "Implied return exceeded the anomaly threshold and was excluded from ranking.",
    }
    key = str(reason_code or "").strip().upper()
    return mapping.get(key) if key else None


def _latest_numeric_from_rows(rows: list[dict[str, Any]], field: str) -> float | str:
    for row in reversed(rows):
        value = row.get(field, UNKNOWN)
        if _is_num(value):
            return float(value)
    return UNKNOWN


def _valuation_preflight(
    *,
    latest_row: dict[str, Any],
    derived_signals: dict[str, Any],
    current_price: float | str,
    shares_latest: float | str,
    shares_effective: float | str,
    shares_status: str,
    shares_reason_code: str,
    fcf_status: str,
    fcf_reason_code: str,
    net_debt_latest: float | str,
    net_debt_effective: float | str,
    revenue: float | str,
    fcf_values: list[Any],
) -> dict[str, Any]:
    positive_fcf = [float(value) for value in fcf_values if _is_num(value) and float(value) > 0]
    fcf_latest = latest_row.get("fcf", UNKNOWN)
    valuation_inputs = {
        "price_current": "OK" if _is_num(current_price) and float(current_price) > 0 else "UNKNOWN",
        "fcf_latest": "OK" if _is_num(fcf_latest) else "UNKNOWN",
        "shares_latest": "OK" if _is_num(shares_latest) else "UNKNOWN",
        "shares_effective": "OK" if _is_num(shares_effective) else "UNKNOWN",
        "shares_status": shares_status if shares_status in {"OK", "UNKNOWN"} else "UNKNOWN",
        "shares_reason_code": str(shares_reason_code or "UNKNOWN"),
        "fcf_status": fcf_status if fcf_status in {"OK", "UNKNOWN"} else "UNKNOWN",
        "fcf_reason_code": str(fcf_reason_code or "UNKNOWN"),
        "net_debt_latest": "OK" if _is_num(net_debt_latest) else "UNKNOWN",
        "net_debt_effective": "OK" if _is_num(net_debt_effective) else "UNKNOWN",
        "revenue_latest": "OK" if _is_num(revenue) else "UNKNOWN",
        "revenue_cagr_5y": _signal_status(derived_signals, "revenue_cagr_5y"),
        "revenue_cagr_10y": _signal_status(derived_signals, "revenue_cagr_10y"),
        "fcf_margin_trend": _signal_status(derived_signals, "fcf_margin_trend_slope"),
        "operating_margin_trend": _signal_status(derived_signals, "operating_margin_trend_slope"),
        "positive_fcf_observations": int(len(positive_fcf)),
    }

    reason_code: str | None = None
    if not (_is_num(current_price) and float(current_price) > 0):
        reason_code = "PRICE_UNKNOWN"
    elif str(shares_status).upper() != "OK":
        reason_code = "MISSING_SHARES"
    elif not _is_num(net_debt_effective):
        reason_code = "MISSING_NET_DEBT"
    elif str(fcf_status).upper() != "OK" or len(positive_fcf) == 0:
        reason_code = "MISSING_FCF"

    denominator_ok, denominator_reason, denominator_details = validate_denominators(
        {
            "shares_outstanding": shares_effective,
            "current_price": current_price,
        }
    )
    valuation_inputs["denominator_check"] = "OK" if denominator_ok else "UNKNOWN"
    if denominator_reason:
        valuation_inputs["denominator_reason_code"] = denominator_reason
    if denominator_details:
        valuation_inputs["denominator_details"] = denominator_details
    if reason_code is None and not denominator_ok:
        reason_code = "INVALID_DENOMINATOR"

    reason_detail = _valuation_reason_detail(reason_code)
    if reason_code == "MISSING_SHARES":
        code = str(shares_reason_code or "UNKNOWN")
        reason_detail = f"{_valuation_reason_detail(reason_code)} (shares_reason_code={code})"
    if reason_code == "MISSING_FCF":
        code = str(fcf_reason_code or "UNKNOWN")
        reason_detail = f"{_valuation_reason_detail(reason_code)} (fcf_reason_code={code})"
    if reason_code == "INVALID_DENOMINATOR" and denominator_reason:
        reason_detail = f"{_valuation_reason_detail(reason_code)} (subreason={denominator_reason})"

    return {
        "can_compute": reason_code is None,
        "reason_code": reason_code,
        "reason_detail": reason_detail,
        "valuation_inputs": valuation_inputs,
        "positive_fcf_values": positive_fcf,
        "denominator_reason_code": denominator_reason,
    }


def _load_fundamentals_payload(run_id: str, ticker: str, *, output_dir: Path | None = None) -> dict[str, Any] | None:
    cfg = get_config()
    out_dir = output_dir or (cfg.sectors_dir / run_id)
    path = out_dir / f"fundamentals_{ticker}.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None
    dossier_path = cfg.dossiers_dir / run_id / ticker / "dossier.json"
    if not dossier_path.exists():
        return None
    try:
        dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(dossier, dict):
        return None
    return build_fundamentals_frame(dossier)


def build_ticker_valuation(
    fundamentals_payload: dict[str, Any],
    *,
    run_id: str | None = None,
    as_of_date: str | None = None,
    with_prices: bool = True,
    fallback_days: int | None = None,
) -> dict[str, Any]:
    ticker = str(fundamentals_payload.get("ticker") or "").upper()
    rows = [row for row in (fundamentals_payload.get("rows") or []) if isinstance(row, dict)]
    rows = sorted(rows, key=lambda row: int(row.get("year", 0)))
    derived_signals = fundamentals_payload.get("derived_signals") or {}
    as_of = str(as_of_date or fundamentals_payload.get("as_of_date") or "")

    gaps: list[str] = []
    claims: dict[str, dict[str, Any]] = {}

    if not rows:
        gaps.append("NO_FUNDAMENTALS_ROWS")
        ticker_cov = _default_price_coverage_entry(
            ticker=ticker,
            requested_as_of=as_of,
            with_prices=with_prices,
        )
        ticker_shares_cov = _default_shares_coverage_entry(
            ticker=ticker,
            requested_as_of=as_of,
        )
        ticker_fcf_cov = _default_fcf_coverage_entry(
            ticker=ticker,
            requested_as_of=as_of,
        )
        return {
            "valuation_version": VALUATION_VERSION,
            "fundamentals_version": fundamentals_payload.get("fundamentals_version", FUNDAMENTALS_VERSION),
            "ticker": ticker,
            "as_of_date": as_of,
            "valuation_status": "UNKNOWN",
            "valuation_reason_code": "MODEL_PRECONDITION_FAILED",
            "valuation_reason_detail": "No fundamentals rows are available.",
            "valuation_inputs": {},
            "intrinsic_per_share_base": UNKNOWN,
            "intrinsic_per_share_conservative": UNKNOWN,
            "valuation_gap": UNKNOWN,
            "implied_return_base": UNKNOWN,
            "implied_return_conservative": UNKNOWN,
            "implied_fcf_growth": UNKNOWN,
            "assumption_card": {},
            "input_snapshot": {},
            "price_gap": {
                "reason_code": ticker_cov.get("result", {}).get("reason_code"),
                "reason_detail": ticker_cov.get("result", {}).get("reason_detail"),
                "provider": None,
                "as_of_used": None,
                "cache_hit": False,
                "derived_from": [f"sector.price_coverage.entries[{ticker}]"],
            },
            "price_coverage_entry": ticker_cov,
            "shares_coverage_entry": ticker_shares_cov,
            "fcf_coverage_entry": ticker_fcf_cov,
            "price_evidence": None,
            "shares_evidence": None,
            "fcf_evidence": None,
            "gaps": gaps,
            "claims": claims,
            "generated_at": utc_now_iso(),
        }

    latest = rows[-1]
    shares_latest = latest.get("shares_outstanding", UNKNOWN)
    net_debt_latest = latest.get("net_debt", UNKNOWN)
    revenue = latest.get("revenue", UNKNOWN)
    fcf_values: list[Any] = []
    fcf_margin = latest.get("fcf_margin", UNKNOWN)
    op_margin = latest.get("op_margin", UNKNOWN)
    margin_for_reverse = fcf_margin if _is_num(fcf_margin) else op_margin
    run_scope_id = str(run_id or fundamentals_payload.get("run_id") or "").strip() or None

    shares_coverage_entry: dict[str, Any]
    if as_of:
        shares_value, shares_coverage_entry = resolve_shares_asof(
            ticker=ticker,
            as_of_date=as_of,
            run_id=run_scope_id,
        )
    else:
        shares_value, shares_coverage_entry = (None, _default_shares_coverage_entry(ticker=ticker, requested_as_of=as_of))
    # A count the share-count guard refused stays refused: the in-memory rows below come
    # from the same filings, so falling back to them would put the slip straight back.
    shares_guard = shares_coverage_entry.get("shares_guard")
    guard_refused = isinstance(shares_guard, dict) and shares_guard.get("outcome") == GUARD_REFUSED
    if not _is_num(shares_value) and not guard_refused:
        row_traces = fundamentals_payload.get("row_traces") if isinstance(fundamentals_payload.get("row_traces"), dict) else {}
        for row in reversed(rows):
            value = row.get("shares_outstanding", UNKNOWN)
            if not (_is_num(value) and float(value) > 0):
                continue
            year = int(row.get("year", 0))
            trace_bucket = (
                (row_traces.get(str(year)) or {}).get("shares_outstanding")
                if isinstance(row_traces.get(str(year)), dict)
                else None
            )
            refs = [str(x) for x in ((trace_bucket or {}).get("derived_from") or []) if str(x).strip()]
            if not refs:
                refs = [f"fundamentals.rows[{year}].shares_outstanding"]
            shares_value = float(value)
            shares_coverage_entry.update(
                {
                    "shares_status": "OK",
                    "shares_reason_code": "OK",
                    "shares_reason_detail": "Resolved from in-memory fundamentals payload.",
                    "shares_value": float(value),
                    "shares_asof_used": as_of or fundamentals_payload.get("as_of_date"),
                    "shares_source": "current_run_fundamentals",
                    "derived_from": refs,
                }
            )
            break
    shares = float(shares_value) if _is_num(shares_value) else UNKNOWN
    if _is_num(shares):
        shares_coverage_entry["shares_status"] = "OK"
        shares_coverage_entry["shares_value"] = float(shares)
    shares_coverage_entry["shares_source_resolution"] = _shares_source_resolution(shares_coverage_entry)

    net_debt_value, net_debt_coverage_entry = _select_net_debt_snapshot(
        ticker=ticker,
        as_of_date=as_of,
        run_id=run_scope_id,
        fundamentals_payload=fundamentals_payload,
        latest_row=latest,
    )
    net_debt = float(net_debt_value) if _is_num(net_debt_value) else _latest_numeric_from_rows(rows, "net_debt")
    net_debt_status = str(net_debt_coverage_entry.get("net_debt_status") or "UNKNOWN").upper()
    if net_debt_status not in {"OK", "UNKNOWN"}:
        net_debt_status = "UNKNOWN"
    if net_debt_status != "OK" and _is_num(net_debt):
        net_debt_status = "OK"
        net_debt_coverage_entry["net_debt_status"] = "OK"
        net_debt_coverage_entry["net_debt_value"] = float(net_debt)
    net_debt_reason_code = str(net_debt_coverage_entry.get("net_debt_reason_code") or "NO_NET_DEBT_PROXY").upper()
    net_debt_source = net_debt_coverage_entry.get("net_debt_source")
    net_debt_asof_used = net_debt_coverage_entry.get("net_debt_asof_used")
    net_debt_source_resolution = _net_debt_source_resolution(net_debt_coverage_entry)
    net_debt_coverage_entry["net_debt_source_resolution"] = net_debt_source_resolution
    net_debt_refs = [str(ref) for ref in (net_debt_coverage_entry.get("derived_from") or []) if str(ref).strip()]
    net_debt_evidence_ref = net_debt_refs[0] if net_debt_refs else f"sector.net_debt_coverage.entries[{ticker}]"
    net_debt_coverage_entry["net_debt_evidence_ref"] = net_debt_evidence_ref
    net_debt_evidence = (
        {
            "id": f"netdebt_{sha256_text(f'{ticker}|{net_debt_asof_used}|{net_debt_source}|{net_debt}')[:16]}",
            "type": "net_debt",
            "ticker": ticker,
            "requested_as_of_date": as_of,
            "as_of_date": net_debt_asof_used or as_of,
            "net_debt": float(net_debt),
            "source": net_debt_source,
            "source_resolution": net_debt_source_resolution,
            "derived_from": net_debt_refs or [f"sector.net_debt_coverage.entries[{ticker}]"],
        }
        if net_debt_status == "OK" and _is_num(net_debt)
        else None
    )

    fcf_coverage_entry: dict[str, Any]
    if as_of:
        fcf_value, fcf_coverage_entry = resolve_fcf_asof(
            ticker=ticker,
            as_of_date=as_of,
            run_id=run_scope_id,
        )
    else:
        fcf_value, fcf_coverage_entry = (None, _default_fcf_coverage_entry(ticker=ticker, requested_as_of=as_of))
    if not _is_num(fcf_value):
        row_traces = fundamentals_payload.get("row_traces") if isinstance(fundamentals_payload.get("row_traces"), dict) else {}
        for row in reversed(rows):
            year = int(row.get("year", 0))
            fcf_row = row.get("fcf", UNKNOWN)
            if _is_num(fcf_row):
                trace_bucket = (
                    (row_traces.get(str(year)) or {}).get("fcf")
                    if isinstance(row_traces.get(str(year)), dict)
                    else None
                )
                refs = [str(x) for x in ((trace_bucket or {}).get("derived_from") or []) if str(x).strip()]
                if not refs:
                    refs = [f"fundamentals.rows[{year}].fcf"]
                fcf_value = float(fcf_row)
                fcf_coverage_entry.update(
                    {
                        "fcf_status": "OK",
                        "fcf_reason_code": "OK",
                        "fcf_reason_detail": "Resolved from in-memory fundamentals payload.",
                        "fcf_value": float(fcf_row),
                        "fcf_asof_used": as_of or fundamentals_payload.get("as_of_date"),
                        "fcf_source": "current_run_dossier",
                        "cfo_status": "OK" if _is_num(row.get("cfo")) else "UNKNOWN",
                        "cfo_reason_code": "OK" if _is_num(row.get("cfo")) else "MISSING_CFO",
                        "cfo_value": float(row.get("cfo")) if _is_num(row.get("cfo")) else UNKNOWN,
                        "cfo_asof_used": as_of or fundamentals_payload.get("as_of_date"),
                        "capex_status": "OK" if _is_num(row.get("capex")) else "UNKNOWN",
                        "capex_reason_code": "OK" if _is_num(row.get("capex")) else "MISSING_CAPEX",
                        "capex_value": float(row.get("capex")) if _is_num(row.get("capex")) else UNKNOWN,
                        "capex_asof_used": as_of or fundamentals_payload.get("as_of_date"),
                        "derived_from": refs,
                    }
                )
                break
            cfo_row = row.get("cfo", UNKNOWN)
            capex_row = row.get("capex", UNKNOWN)
            if _is_num(cfo_row) and _is_num(capex_row):
                cfo_bucket = (
                    (row_traces.get(str(year)) or {}).get("cfo")
                    if isinstance(row_traces.get(str(year)), dict)
                    else None
                )
                capex_bucket = (
                    (row_traces.get(str(year)) or {}).get("capex")
                    if isinstance(row_traces.get(str(year)), dict)
                    else None
                )
                refs = [str(x) for x in ((cfo_bucket or {}).get("derived_from") or []) if str(x).strip()]
                refs += [str(x) for x in ((capex_bucket or {}).get("derived_from") or []) if str(x).strip()]
                refs.append("derived:fcf=cfo-capex")
                # Capex is a spend: subtract its magnitude whatever sign the
                # filer tagged it with (a negated capex turned FCF into CFO + capex).
                fcf_value = float(cfo_row) - abs(float(capex_row))
                fcf_coverage_entry.update(
                    {
                        "fcf_status": "OK",
                        "fcf_reason_code": "OK",
                        "fcf_reason_detail": "Derived from in-memory fundamentals payload (cfo - capex).",
                        "fcf_value": float(fcf_value),
                        "fcf_asof_used": as_of or fundamentals_payload.get("as_of_date"),
                        "fcf_source": "current_run_dossier",
                        "cfo_status": "OK",
                        "cfo_reason_code": "OK",
                        "cfo_value": float(cfo_row),
                        "cfo_asof_used": as_of or fundamentals_payload.get("as_of_date"),
                        "capex_status": "OK",
                        "capex_reason_code": "OK",
                        "capex_value": float(capex_row),
                        "capex_asof_used": as_of or fundamentals_payload.get("as_of_date"),
                        "derived_from": refs,
                    }
                )
                break
    if _is_num(fcf_value):
        fcf_values = [float(fcf_value)]
        fcf_coverage_entry["fcf_status"] = "OK"
        fcf_coverage_entry["fcf_value"] = float(fcf_value)
    fcf_coverage_entry["fcf_source_resolution"] = _fcf_source_resolution(fcf_coverage_entry)

    snapshot: PriceSnapshot | None
    price_origin: str
    price_coverage_entry: dict[str, Any]
    if as_of:
        snapshot, price_origin, price_coverage_entry = _select_price_snapshot(
            ticker,
            as_of,
            run_id=run_scope_id,
            with_prices=with_prices,
            fallback_days=fallback_days,
        )
    else:
        snapshot, price_origin = None, "missing_as_of"
        price_coverage_entry = _default_price_coverage_entry(
            ticker=ticker,
            requested_as_of=as_of,
            with_prices=with_prices,
        )

    current_price = float(snapshot.price) if snapshot is not None else UNKNOWN
    current_price_source = _price_source_label(snapshot, price_origin)
    price_source = snapshot.source if snapshot is not None else UNKNOWN
    price_source_url = snapshot.url if snapshot is not None else None
    price_asof_used = snapshot.as_of_date if snapshot is not None else UNKNOWN
    price_confidence = snapshot.confidence if snapshot is not None else "LOW"
    run_id_for_evidence = str(
        fundamentals_payload.get("run_id")
        or fundamentals_payload.get("dossier_run_id")
        or f"valuation_{as_of or 'unknown'}"
    )
    price_evidence_id = (
        _persist_price_evidence_item(
            ticker=ticker,
            as_of_date=as_of or "UNKNOWN",
            run_id=run_id_for_evidence,
            snapshot=snapshot,
        )
        if snapshot is not None and as_of
        else None
    )
    price_evidence_ref = (
        f"valuation.price_evidence.{price_evidence_id}"
        if price_evidence_id
        else (
            f"price:{snapshot.source}:{snapshot.as_of_date}"
            if snapshot is not None
            else None
        )
    )
    price_evidence = (
        {
            "id": price_evidence_id or _price_evidence_id(ticker, snapshot),
            "type": "price",
            "ticker": snapshot.ticker,
            "requested_as_of_date": as_of,
            "as_of_date": snapshot.as_of_date,
            "price": float(snapshot.price),
            "currency": snapshot.currency or "USD",
            "source": snapshot.source,
            "retrieved_at": snapshot.retrieved_at,
            "url": snapshot.url,
            "confidence": snapshot.confidence,
        }
        if snapshot is not None
        else None
    )
    shares_status = str(shares_coverage_entry.get("shares_status") or "UNKNOWN").upper()
    if shares_status not in {"OK", "UNKNOWN"}:
        shares_status = "UNKNOWN"
    if shares_status != "OK" and _is_num(shares) and float(shares) > 0:
        shares_status = "OK"
    shares_reason_code = str(shares_coverage_entry.get("shares_reason_code") or "NO_CURRENT_RUN_SHARES").upper()
    shares_source = shares_coverage_entry.get("shares_source")
    shares_asof_used = shares_coverage_entry.get("shares_asof_used")
    shares_confidence = shares_coverage_entry.get("confidence") or ("HIGH" if shares_status == "OK" else None)
    shares_source_resolution = _shares_source_resolution(shares_coverage_entry)
    shares_coverage_entry["shares_source_resolution"] = shares_source_resolution

    shares_refs = [str(ref) for ref in (shares_coverage_entry.get("derived_from") or []) if str(ref).strip()]
    shares_evidence_ref = shares_refs[0] if shares_refs else f"sector.shares_coverage.entries[{ticker}]"
    shares_coverage_entry["shares_evidence_ref"] = shares_evidence_ref
    shares_evidence = (
        {
            "id": f"shares_{sha256_text(f'{ticker}|{shares_asof_used}|{shares_source}|{shares}')[:16]}",
            "type": "shares",
            "ticker": ticker,
            "requested_as_of_date": as_of,
            "as_of_date": shares_asof_used or as_of,
            "shares_outstanding": float(shares),
            # resolve_shares_asof and the fundamentals rows both carry MILLIONS
            # (shares.py stamps shares_unit "shares_millions"); the old label
            # "shares" misstated the value a million-fold.
            "unit": "shares_millions",
            "source": shares_source,
            "confidence": shares_confidence,
            "source_resolution": shares_source_resolution,
            "derived_from": shares_refs or [f"sector.shares_coverage.entries[{ticker}]"],
        }
        if shares_status == "OK" and _is_num(shares)
        else None
    )

    fcf_status = str(fcf_coverage_entry.get("fcf_status") or "UNKNOWN").upper()
    if fcf_status not in {"OK", "UNKNOWN"}:
        fcf_status = "UNKNOWN"
    if fcf_status != "OK" and _is_num(fcf_value):
        fcf_status = "OK"
    fcf_reason_code = str(fcf_coverage_entry.get("fcf_reason_code") or "DOSSIER_NO_FCF").upper()
    fcf_source = fcf_coverage_entry.get("fcf_source")
    fcf_asof_used = fcf_coverage_entry.get("fcf_asof_used")
    fcf_source_resolution = _fcf_source_resolution(fcf_coverage_entry)
    fcf_coverage_entry["fcf_source_resolution"] = fcf_source_resolution
    fcf_refs = [str(ref) for ref in (fcf_coverage_entry.get("derived_from") or []) if str(ref).strip()]
    fcf_evidence_ref = fcf_refs[0] if fcf_refs else f"sector.fcf_coverage.entries[{ticker}]"
    fcf_coverage_entry["fcf_evidence_ref"] = fcf_evidence_ref
    fcf_evidence = (
        {
            "id": f"fcf_{sha256_text(f'{ticker}|{fcf_asof_used}|{fcf_source}|{fcf_value}')[:16]}",
            "type": "fcf",
            "ticker": ticker,
            "requested_as_of_date": as_of,
            "as_of_date": fcf_asof_used or as_of,
            "fcf": float(fcf_value),
            "source": fcf_source,
            "source_resolution": fcf_source_resolution,
            "derived_from": fcf_refs or [f"sector.fcf_coverage.entries[{ticker}]"],
        }
        if fcf_status == "OK" and _is_num(fcf_value)
        else None
    )

    price_gap = None
    if not _is_num(current_price):
        gaps.append("MISSING_PRICE")
        result_bucket = price_coverage_entry.get("result") if isinstance(price_coverage_entry, dict) else {}
        output_bucket = price_coverage_entry.get("output_fields") if isinstance(price_coverage_entry, dict) else {}
        cache_bucket = price_coverage_entry.get("cache") if isinstance(price_coverage_entry, dict) else {}
        price_gap = {
            "reason_code": (
                result_bucket.get("reason_code")
                if isinstance(result_bucket, dict)
                else "PROVIDER_NO_DATA"
            ),
            "reason_detail": (
                result_bucket.get("reason_detail")
                if isinstance(result_bucket, dict)
                else "No price snapshot found."
            ),
            "provider": (
                output_bucket.get("price_source")
                if isinstance(output_bucket, dict)
                else None
            ) or (price_source if price_source != UNKNOWN else None),
            "as_of_used": (
                output_bucket.get("price_asof_used")
                if isinstance(output_bucket, dict)
                else None
            ),
            "cache_hit": bool(cache_bucket.get("hit")) if isinstance(cache_bucket, dict) else False,
            "derived_from": [f"sector.price_coverage.entries[{ticker}]"],
        }
    shares_gap = None
    if shares_status != "OK":
        shares_gap = {
            "reason_code": shares_reason_code,
            "reason_detail": str(shares_coverage_entry.get("shares_reason_detail") or ""),
            "source": shares_source,
            "as_of_used": shares_asof_used,
            "cache_hit": bool((shares_coverage_entry.get("cache") or {}).get("hit")),
            "derived_from": [f"sector.shares_coverage.entries[{ticker}]"],
        }
    fcf_gap = None
    if fcf_status != "OK":
        fcf_gap = {
            "reason_code": fcf_reason_code,
            "reason_detail": str(fcf_coverage_entry.get("fcf_reason_detail") or ""),
            "source": fcf_source,
            "as_of_used": fcf_asof_used,
            "derived_from": [f"sector.fcf_coverage.entries[{ticker}]"],
        }
    net_debt_gap = None
    if net_debt_status != "OK":
        net_debt_gap = {
            "reason_code": net_debt_reason_code,
            "reason_detail": str(net_debt_coverage_entry.get("net_debt_reason_detail") or ""),
            "source": net_debt_source,
            "as_of_used": net_debt_asof_used,
            "derived_from": [f"sector.net_debt_coverage.entries[{ticker}]"],
        }

    preflight = _valuation_preflight(
        latest_row=latest,
        derived_signals=derived_signals,
        current_price=current_price,
        shares_latest=shares_latest,
        shares_effective=shares,
        shares_status=shares_status,
        shares_reason_code=shares_reason_code,
        fcf_status=fcf_status,
        fcf_reason_code=fcf_reason_code,
        net_debt_latest=net_debt_latest,
        net_debt_effective=net_debt,
        revenue=revenue,
        fcf_values=fcf_values,
    )
    valuation_inputs = preflight.get("valuation_inputs") if isinstance(preflight, dict) else {}
    if not isinstance(valuation_inputs, dict):
        valuation_inputs = {}
    valuation_reason_code = str(preflight.get("reason_code") or "").strip().upper() or None
    valuation_reason_detail = str(preflight.get("reason_detail") or "").strip() or None

    if valuation_reason_code == "MISSING_SHARES":
        gaps.append("MISSING_SHARES")
    if valuation_reason_code == "MISSING_NET_DEBT":
        gaps.append("MISSING_NET_DEBT")
    if valuation_reason_code == "MISSING_FCF":
        gaps.append("MISSING_OWNER_EARNINGS_INPUTS")
    if valuation_reason_code == "INVALID_DENOMINATOR":
        gaps.append("INVALID_DENOMINATOR")

    quality_bucket, quality_refs = _quality_bucket(latest, derived_signals)
    base_multiple, conservative_multiple = _multiple_band(quality_bucket)

    intrinsic_base: float | str = UNKNOWN
    intrinsic_cons: float | str = UNKNOWN
    shares_trace_refs = [shares_evidence_ref] if shares_evidence_ref else [f"sector.shares_coverage.entries[{ticker}]"]
    fcf_trace_refs = [fcf_evidence_ref] if fcf_evidence_ref else [f"sector.fcf_coverage.entries[{ticker}]"]
    intrinsic_refs = [
        "fundamentals.rows[*].fcf",
        "fundamentals.rows[-1].shares_outstanding",
        "fundamentals.rows[-1].net_debt",
    ] + shares_trace_refs + fcf_trace_refs + ([net_debt_evidence_ref] if net_debt_evidence_ref else [])
    implied_return_base: float | str = UNKNOWN
    implied_return_cons: float | str = UNKNOWN
    valuation_gap: float | str = UNKNOWN
    valuation_anomaly_detail: dict[str, Any] | None = None

    try:
        if bool(preflight.get("can_compute")):
            intrinsic_base, intrinsic_cons, intrinsic_refs = _owner_earnings_intrinsic(
                fcf_values=list(preflight.get("positive_fcf_values") or []),
                shares=shares,
                net_debt=net_debt,
                base_multiple=base_multiple,
                conservative_multiple=conservative_multiple,
            )
            intrinsic_refs = list(
                dict.fromkeys(
                    [
                        str(ref)
                        for ref in (
                            intrinsic_refs
                            + shares_trace_refs
                            + fcf_trace_refs
                            + ([net_debt_evidence_ref] if net_debt_evidence_ref else [])
                        )
                        if str(ref).strip()
                    ]
                )
            )
            if _is_num(intrinsic_base) and _is_num(current_price) and float(current_price) > 0:
                implied_return_base = (float(intrinsic_base) / float(current_price)) - 1.0
            else:
                implied_return_base = UNKNOWN
            if _is_num(intrinsic_cons) and _is_num(current_price) and float(current_price) > 0:
                implied_return_cons = (float(intrinsic_cons) / float(current_price)) - 1.0
            else:
                implied_return_cons = UNKNOWN
            valuation_gap = implied_return_base if _is_num(implied_return_base) else UNKNOWN
            if not _is_num(intrinsic_base) and not valuation_reason_code:
                valuation_reason_code = "MODEL_PRECONDITION_FAILED"
                valuation_reason_detail = _valuation_reason_detail(valuation_reason_code)
                gaps.append("MISSING_OWNER_EARNINGS_INPUTS")
            if _is_num(intrinsic_base) and not _is_num(implied_return_base) and not valuation_reason_code:
                valuation_reason_code = "INVALID_DENOMINATOR"
                valuation_reason_detail = _valuation_reason_detail(valuation_reason_code)
                gaps.append("INVALID_DENOMINATOR")
    except Exception as exc:  # noqa: BLE001
        valuation_reason_code = "ENGINE_EXCEPTION"
        valuation_reason_detail = str(exc)
        intrinsic_base = UNKNOWN
        intrinsic_cons = UNKNOWN
        implied_return_base = UNKNOWN
        implied_return_cons = UNKNOWN
        valuation_gap = UNKNOWN
        gaps.append("ENGINE_EXCEPTION")

    if _is_num(implied_return_base) and abs(float(implied_return_base)) > _VALUATION_ANOMALY_ABS_MAX:
        valuation_anomaly_detail = {
            "reason_code": "VALUATION_ANOMALY",
            "raw_implied_return_base": float(implied_return_base),
            "intrinsic_per_share_base": float(intrinsic_base) if _is_num(intrinsic_base) else UNKNOWN,
            "current_price": float(current_price) if _is_num(current_price) else UNKNOWN,
            "shares_outstanding": float(shares) if _is_num(shares) else UNKNOWN,
            "fcf_latest": float(fcf_value) if _is_num(fcf_value) else UNKNOWN,
            "net_debt": float(net_debt) if _is_num(net_debt) else UNKNOWN,
        }
        valuation_reason_code = "VALUATION_ANOMALY"
        valuation_reason_detail = _valuation_reason_detail(valuation_reason_code)
        implied_return_base = UNKNOWN
        implied_return_cons = UNKNOWN
        valuation_gap = UNKNOWN
        gaps.append("VALUATION_ANOMALY")

    reverse_dcf_out, reverse_warnings = implied_growth_from_price(
        market_price=float(current_price) if _is_num(current_price) else None,
        shares_outstanding=float(shares) if _is_num(shares) else None,
        net_debt=float(net_debt) if _is_num(net_debt) else None,
        base_revenue=float(revenue) if _is_num(revenue) else None,
        margin=float(margin_for_reverse) if _is_num(margin_for_reverse) else None,
        discount_rate=0.10,
        terminal_growth=0.02,
    )
    implied_growth = reverse_dcf_out.get("implied_growth", UNKNOWN)
    if reverse_dcf_out.get("implied_growth_saturated"):
        # A clipped bound is not a solve: flag-ignoring consumers (the
        # value-first rubric scores valuation['implied_fcf_growth']; claims
        # persist it to the peer scoreboard) must see UNKNOWN, mirroring
        # evidence/enrichment (review RDCF-1).
        implied_growth = UNKNOWN
        gaps.append("REVERSE_DCF_SATURATED")
    elif not _is_num(implied_growth):
        gaps.append("MISSING_REVERSE_DCF_INPUTS")

    if (not _is_num(implied_return_base)) and (not valuation_reason_code):
        valuation_reason_code = "MODEL_PRECONDITION_FAILED"
        valuation_reason_detail = _valuation_reason_detail(valuation_reason_code)

    claims["intrinsic_per_share_base"] = {
        "value": intrinsic_base,
        "derived_from": intrinsic_refs + ["valuation.assumption_card.base_multiple"],
    }
    claims["intrinsic_per_share_conservative"] = {
        "value": intrinsic_cons,
        "derived_from": intrinsic_refs + ["valuation.assumption_card.conservative_multiple"],
    }
    claims["implied_return_base"] = {
        "value": implied_return_base,
        "derived_from": [
            "valuation.claims.intrinsic_per_share_base.value",
            "valuation.input_snapshot.current_price",
        ]
        + ([price_evidence_ref] if price_evidence_ref else [])
        + ([shares_evidence_ref] if shares_evidence_ref else [])
        + ([fcf_evidence_ref] if fcf_evidence_ref else []),
    }
    claims["implied_return_conservative"] = {
        "value": implied_return_cons,
        "derived_from": [
            "valuation.claims.intrinsic_per_share_base.value",
            "valuation.claims.intrinsic_per_share_conservative.value",
            "valuation.input_snapshot.current_price",
        ]
        + ([price_evidence_ref] if price_evidence_ref else [])
        + ([shares_evidence_ref] if shares_evidence_ref else [])
        + ([fcf_evidence_ref] if fcf_evidence_ref else []),
    }
    claims["valuation_gap"] = {
        "value": valuation_gap,
        "derived_from": [
            "valuation.claims.intrinsic_per_share_base.value",
            "valuation.input_snapshot.current_price",
        ]
        + ([price_evidence_ref] if price_evidence_ref else [])
        + ([shares_evidence_ref] if shares_evidence_ref else [])
        + ([fcf_evidence_ref] if fcf_evidence_ref else [])
        + ([net_debt_evidence_ref] if net_debt_evidence_ref else []),
    }
    claims["implied_fcf_growth"] = {
        "value": implied_growth,
        "derived_from": [
            "valuation.input_snapshot.current_price",
            "valuation.input_snapshot.shares_outstanding",
            "valuation.input_snapshot.net_debt",
            "valuation.input_snapshot.revenue_latest",
            "valuation.input_snapshot.margin_for_reverse_dcf",
        ],
    }

    status = "OK" if (_is_num(intrinsic_base) and _is_num(implied_return_base)) else "UNKNOWN"
    if status == "OK":
        valuation_reason_code = None
        valuation_reason_detail = None

    return {
        "valuation_version": VALUATION_VERSION,
        "fundamentals_version": fundamentals_payload.get("fundamentals_version", FUNDAMENTALS_VERSION),
        "ticker": ticker,
        "as_of_date": as_of,
        "valuation_status": status,
        "valuation_reason_code": valuation_reason_code if valuation_reason_code in ALLOWED_VALUATION_REASON_CODES else (
            "MODEL_PRECONDITION_FAILED" if status != "OK" else None
        ),
        "valuation_reason_detail": valuation_reason_detail,
        "valuation_inputs": valuation_inputs,
        "intrinsic_per_share_base": intrinsic_base if _is_num(intrinsic_base) else UNKNOWN,
        "intrinsic_per_share_conservative": intrinsic_cons if _is_num(intrinsic_cons) else UNKNOWN,
        "valuation_gap": valuation_gap if _is_num(valuation_gap) else UNKNOWN,
        "implied_return_base": implied_return_base if _is_num(implied_return_base) else UNKNOWN,
        "implied_return_conservative": implied_return_cons if _is_num(implied_return_cons) else UNKNOWN,
        "implied_fcf_growth": float(implied_growth) if _is_num(implied_growth) else UNKNOWN,
        "assumption_card": {
            "model": "owner_earnings_plus_reverse_dcf",
            "horizon_years": 10,
            "discount_rate": 0.10,
            "terminal_growth": 0.02,
            "base_multiple": float(base_multiple),
            "conservative_multiple": float(conservative_multiple),
            "quality_bucket": quality_bucket,
            "quality_bucket_derived_from": quality_refs,
            # The count of free-cash-flow observations the median actually used:
            # the engine capitalises the latest year only (fcf_values above), so
            # the label must not claim the row count.
            "normalization_window_years": int(len(preflight.get("positive_fcf_values") or [])),
        },
        "input_snapshot": {
            "current_price": current_price if _is_num(current_price) else UNKNOWN,
            "price_source": price_source,
            "price_asof_used": price_asof_used,
            "price_provider": price_source,
            "price_source_url": price_source_url,
            "price_as_of_date": price_asof_used,
            "current_price_source": current_price_source,
            "price_source_resolution": price_origin,
            "price_confidence": price_confidence,
            "shares_outstanding": shares if _is_num(shares) else UNKNOWN,
            "shares_status": shares_status,
            "shares_reason_code": shares_reason_code,
            "shares_asof_used": shares_asof_used,
            "shares_source": shares_source,
            "shares_source_resolution": shares_source_resolution,
            "shares_confidence": shares_confidence,
            "fcf_latest": float(fcf_value) if _is_num(fcf_value) else UNKNOWN,
            "fcf_status": fcf_status,
            "fcf_reason_code": fcf_reason_code,
            "fcf_asof_used": fcf_asof_used,
            "fcf_source": fcf_source,
            "fcf_source_resolution": fcf_source_resolution,
            "cfo_status": fcf_coverage_entry.get("cfo_status"),
            "cfo_reason_code": fcf_coverage_entry.get("cfo_reason_code"),
            "cfo_value": fcf_coverage_entry.get("cfo_value"),
            "cfo_asof_used": fcf_coverage_entry.get("cfo_asof_used"),
            "capex_status": fcf_coverage_entry.get("capex_status"),
            "capex_reason_code": fcf_coverage_entry.get("capex_reason_code"),
            "capex_value": fcf_coverage_entry.get("capex_value"),
            "capex_asof_used": fcf_coverage_entry.get("capex_asof_used"),
            "net_debt": net_debt if _is_num(net_debt) else UNKNOWN,
            "net_debt_status": net_debt_status,
            "net_debt_reason_code": net_debt_reason_code,
            "net_debt_asof_used": net_debt_asof_used,
            "net_debt_source": net_debt_source,
            "net_debt_source_resolution": net_debt_source_resolution,
            "revenue_latest": revenue if _is_num(revenue) else UNKNOWN,
            "margin_for_reverse_dcf": margin_for_reverse if _is_num(margin_for_reverse) else UNKNOWN,
            "normalized_fcf_median": float(median([float(v) for v in fcf_values if _is_num(v) and float(v) > 0]))
            if [v for v in fcf_values if _is_num(v) and float(v) > 0]
            else UNKNOWN,
        },
        "reverse_dcf_outputs": reverse_dcf_out,
        "warnings": [str(w) for w in reverse_warnings],
        "valuation_anomaly_detail": valuation_anomaly_detail,
        "price_gap": price_gap,
        "shares_gap": shares_gap,
        "fcf_gap": fcf_gap,
        "net_debt_gap": net_debt_gap,
        "price_coverage_entry": price_coverage_entry,
        "shares_coverage_entry": shares_coverage_entry,
        "fcf_coverage_entry": fcf_coverage_entry,
        "net_debt_coverage_entry": net_debt_coverage_entry,
        "price_evidence": price_evidence,
        "shares_evidence": shares_evidence,
        "fcf_evidence": fcf_evidence,
        "net_debt_evidence": net_debt_evidence,
        "gaps": sorted(set(gaps)),
        "claims": claims,
        "generated_at": utc_now_iso(),
    }


def write_valuations_for_run(
    *,
    run_id: str,
    tickers: list[str] | None = None,
    as_of_date: str | None = None,
    output_dir: Path | None = None,
    with_prices: bool = True,
) -> dict[str, Any]:
    cfg = get_config()
    out_dir = output_dir or (cfg.sectors_dir / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        candidates = sorted(
            [
                symbol
                for symbol in (p.stem.replace("fundamentals_", "").upper() for p in out_dir.glob("fundamentals_*.json"))
                if _is_ticker_symbol(symbol)
            ]
        )
        if not candidates:
            dossier_dir = cfg.dossiers_dir / run_id
            candidates = (
                sorted([p.name for p in dossier_dir.iterdir() if p.is_dir() and _is_ticker_symbol(p.name)])
                if dossier_dir.exists()
                else []
            )
    else:
        candidates = sorted({str(t).strip().upper() for t in tickers if _is_ticker_symbol(str(t))})

    rows: list[dict[str, Any]] = []
    coverage_entries: list[dict[str, Any]] = []
    shares_coverage_entries: list[dict[str, Any]] = []
    fcf_coverage_entries: list[dict[str, Any]] = []
    net_debt_coverage_entries: list[dict[str, Any]] = []
    prices_ok = 0
    prices_unknown = 0
    shares_ok = 0
    shares_unknown = 0
    fcf_ok = 0
    fcf_unknown = 0
    net_debt_ok = 0
    net_debt_unknown = 0
    for ticker in candidates:
        fundamentals = _load_fundamentals_payload(run_id, ticker, output_dir=out_dir)
        if not fundamentals:
            coverage_entries.append(
                _default_price_coverage_entry(
                    ticker=ticker,
                    requested_as_of=str(as_of_date or ""),
                    with_prices=with_prices,
                )
            )
            shares_coverage_entries.append(
                _default_shares_coverage_entry(
                    ticker=ticker,
                    requested_as_of=str(as_of_date or ""),
                )
            )
            fcf_coverage_entries.append(
                _default_fcf_coverage_entry(
                    ticker=ticker,
                    requested_as_of=str(as_of_date or ""),
                )
            )
            net_debt_coverage_entries.append(
                _default_net_debt_coverage_entry(
                    ticker=ticker,
                    requested_as_of=str(as_of_date or ""),
                )
            )
            prices_unknown += 1
            shares_unknown += 1
            fcf_unknown += 1
            net_debt_unknown += 1
            rows.append({"ticker": ticker, "status": "MISSING_FUNDAMENTALS", "path": None})
            continue
        valuation = build_ticker_valuation(
            fundamentals,
            run_id=run_id,
            as_of_date=as_of_date,
            with_prices=with_prices,
            fallback_days=cfg.price_fallback_days,
        )
        out_path = out_dir / f"valuation_{ticker}.json"
        out_path.write_text(json.dumps(valuation, indent=2), encoding="utf-8")
        coverage_entry = valuation.get("price_coverage_entry")
        if isinstance(coverage_entry, dict):
            coverage_entries.append(coverage_entry)
        shares_coverage_entry = valuation.get("shares_coverage_entry")
        if isinstance(shares_coverage_entry, dict):
            shares_coverage_entries.append(shares_coverage_entry)
        fcf_coverage_entry = valuation.get("fcf_coverage_entry")
        if isinstance(fcf_coverage_entry, dict):
            fcf_coverage_entries.append(fcf_coverage_entry)
        net_debt_coverage_entry = valuation.get("net_debt_coverage_entry")
        if isinstance(net_debt_coverage_entry, dict):
            net_debt_coverage_entries.append(net_debt_coverage_entry)
        rows.append(
            {
                "ticker": ticker,
                "status": str(valuation.get("valuation_status") or "UNKNOWN"),
                "valuation_reason_code": valuation.get("valuation_reason_code"),
                "path": str(out_path),
                "gaps": valuation.get("gaps") or [],
                "current_price_source": ((valuation.get("input_snapshot") or {}).get("current_price_source") or "UNKNOWN"),
                "shares_source_resolution": ((valuation.get("input_snapshot") or {}).get("shares_source_resolution") or "UNKNOWN"),
                "fcf_source_resolution": ((valuation.get("input_snapshot") or {}).get("fcf_source_resolution") or "UNKNOWN"),
                "net_debt_source_resolution": ((valuation.get("input_snapshot") or {}).get("net_debt_source_resolution") or "UNKNOWN"),
            }
        )
        price_value = (valuation.get("input_snapshot") or {}).get("current_price", UNKNOWN)
        if _is_num(price_value):
            prices_ok += 1
        else:
            prices_unknown += 1
        shares_value = (valuation.get("input_snapshot") or {}).get("shares_outstanding", UNKNOWN)
        if _is_num(shares_value) and float(shares_value) > 0:
            shares_ok += 1
        else:
            shares_unknown += 1
        fcf_value = (valuation.get("input_snapshot") or {}).get("fcf_latest", UNKNOWN)
        if _is_num(fcf_value):
            fcf_ok += 1
        else:
            fcf_unknown += 1
        net_debt_value = (valuation.get("input_snapshot") or {}).get("net_debt", UNKNOWN)
        if _is_num(net_debt_value):
            net_debt_ok += 1
        else:
            net_debt_unknown += 1

    rows_sorted = sorted(rows, key=lambda row: str(row.get("ticker") or ""))
    coverage_entries = sorted(
        [row for row in coverage_entries if isinstance(row, dict)],
        key=lambda row: str(row.get("ticker") or ""),
    )
    normalized_coverage_entries: list[dict[str, Any]] = []
    for row in coverage_entries:
        entry = dict(row)
        result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
        status = str((result or {}).get("status") or "UNKNOWN").upper()
        if "source_resolution" not in entry:
            entry["source_resolution"] = _price_source_resolution(entry)
        entry["suggestions"] = _price_coverage_suggestions(entry) if status != "OK" else []
        normalized_coverage_entries.append(entry)
    coverage_entries = normalized_coverage_entries
    shares_coverage_entries = sorted(
        [row for row in shares_coverage_entries if isinstance(row, dict)],
        key=lambda row: str(row.get("ticker") or ""),
    )
    fcf_coverage_entries = sorted(
        [row for row in fcf_coverage_entries if isinstance(row, dict)],
        key=lambda row: str(row.get("ticker") or ""),
    )
    net_debt_coverage_entries = sorted(
        [row for row in net_debt_coverage_entries if isinstance(row, dict)],
        key=lambda row: str(row.get("ticker") or ""),
    )
    reason_counts: dict[str, int] = {}
    for row in coverage_entries:
        result = row.get("result") if isinstance(row, dict) else None
        code = str((result or {}).get("reason_code") or "UNKNOWN")
        reason_counts[code] = reason_counts.get(code, 0) + 1
    shares_reason_counts: dict[str, int] = {}
    for row in shares_coverage_entries:
        code = str(row.get("shares_reason_code") or "UNKNOWN")
        shares_reason_counts[code] = shares_reason_counts.get(code, 0) + 1
    fcf_reason_counts: dict[str, int] = {}
    for row in fcf_coverage_entries:
        code = str(row.get("fcf_reason_code") or "UNKNOWN")
        fcf_reason_counts[code] = fcf_reason_counts.get(code, 0) + 1
    net_debt_reason_counts: dict[str, int] = {}
    for row in net_debt_coverage_entries:
        code = str(row.get("net_debt_reason_code") or "UNKNOWN")
        net_debt_reason_counts[code] = net_debt_reason_counts.get(code, 0) + 1

    provider = get_default_provider(cfg, with_prices=with_prices)
    summary = {
        "run_id": run_id,
        "valuation_version": VALUATION_VERSION,
        "output_dir": str(out_dir),
        "with_prices": bool(with_prices),
        "price_provider_config": str(cfg.price_provider),
        "price_provider_effective": str(getattr(provider, "provider_name", "unknown")),
        "ticker_count": len(rows_sorted),
        "ok_count": len([row for row in rows_sorted if row.get("status") == "OK"]),
        "unknown_count": len([row for row in rows_sorted if row.get("status") == "UNKNOWN"]),
        "prices_ok": int(prices_ok),
        "prices_unknown": int(prices_unknown),
        "shares_ok": int(shares_ok),
        "shares_unknown": int(shares_unknown),
        "fcf_ok": int(fcf_ok),
        "fcf_unknown": int(fcf_unknown),
        "net_debt_ok": int(net_debt_ok),
        "net_debt_unknown": int(net_debt_unknown),
        "rows": rows_sorted,
        "generated_at": utc_now_iso(),
    }
    summary_path = out_dir / "valuation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    coverage_path = out_dir / "price_coverage.json"
    coverage_payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "with_prices": bool(with_prices),
        "provider": str(getattr(provider, "provider_name", "unknown")),
        "ticker_count": len(coverage_entries),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: kv[0])),
        "entries": coverage_entries,
        "generated_at": utc_now_iso(),
    }
    coverage_path.write_text(json.dumps(coverage_payload, indent=2), encoding="utf-8")
    shares_coverage_path = out_dir / "shares_coverage.json"
    shares_coverage_payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(shares_coverage_entries),
        "reason_counts": dict(sorted(shares_reason_counts.items(), key=lambda kv: kv[0])),
        "entries": shares_coverage_entries,
        "generated_at": utc_now_iso(),
    }
    shares_coverage_path.write_text(json.dumps(shares_coverage_payload, indent=2), encoding="utf-8")
    fcf_coverage_path = out_dir / "fcf_coverage.json"
    fcf_coverage_payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(fcf_coverage_entries),
        "reason_counts": dict(sorted(fcf_reason_counts.items(), key=lambda kv: kv[0])),
        "entries": fcf_coverage_entries,
        "generated_at": utc_now_iso(),
    }
    fcf_coverage_path.write_text(json.dumps(fcf_coverage_payload, indent=2), encoding="utf-8")
    net_debt_coverage_path = out_dir / "net_debt_coverage.json"
    net_debt_coverage_payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(net_debt_coverage_entries),
        "reason_counts": dict(sorted(net_debt_reason_counts.items(), key=lambda kv: kv[0])),
        "entries": net_debt_coverage_entries,
        "generated_at": utc_now_iso(),
    }
    net_debt_coverage_path.write_text(json.dumps(net_debt_coverage_payload, indent=2), encoding="utf-8")
    facts_coverage_payload = write_facts_coverage_for_run(
        run_id=run_id,
        as_of_date=str(as_of_date or ""),
        tickers=candidates,
        output_dir=out_dir,
        cfg=cfg,
        shares_entries=shares_coverage_entries,
        fcf_entries=fcf_coverage_entries,
    )
    facts_coverage_path = Path(str(facts_coverage_payload.get("facts_coverage_path") or out_dir / "facts_coverage.json"))
    summary["facts_ok"] = int((facts_coverage_payload.get("status_counts") or {}).get("OK", 0))
    summary["facts_partial"] = int((facts_coverage_payload.get("status_counts") or {}).get("PARTIAL", 0))
    summary["facts_unknown"] = int((facts_coverage_payload.get("status_counts") or {}).get("UNKNOWN", 0))
    summary["summary_path"] = str(summary_path)
    summary["price_coverage_path"] = str(coverage_path)
    summary["shares_coverage_path"] = str(shares_coverage_path)
    summary["fcf_coverage_path"] = str(fcf_coverage_path)
    summary["net_debt_coverage_path"] = str(net_debt_coverage_path)
    summary["facts_coverage_path"] = str(facts_coverage_path)
    return summary
