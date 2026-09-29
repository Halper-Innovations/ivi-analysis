"""Materialized EODHD quote/split lineage for offline market-cap arithmetic.

The producer is intentionally parent-side and EODHD-only.  It persists an
immutable provider response plus the canonical proof envelopes required by the
existing financial-integrity validator, then stores the proof-carrying quote in
``price_quotes``.  Offline children can read that dedicated row without
enabling any provider.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from app.autonomous.financial_integrity import (
    PRICE_BASIS_SPLIT_ADJUSTED,
    PRICE_BASIS_UNADJUSTED,
    SPLIT_PROOF_KIND_NO_INTERVENING_SPLIT,
    SPLIT_PROOF_KIND_SPLIT_EVENT,
    authoritative_split_proof_reference,
    canonical_split_proof_cache_path,
    canonical_split_proof_cache_root,
    canonical_split_proof_raw_cache_path,
    split_proof_materialization_envelope,
    split_proof_raw_materialization_envelope,
    stable_quote_hash,
)
from app.config import AppConfig, get_config
from app.db import connect
from app.util.http import HttpClient


SPLIT_LINEAGE_QUOTE_PROVIDER = "eodhd_split_lineage"
SPLIT_LINEAGE_QUOTE_SCHEMA_VERSION = "ivi.eodhd_split_lineage_quote.v1"
SPLIT_LINEAGE_QUOTE_RECORD_TYPE = "proof_carrying_price_quote"

_REQUIRED_COMPANY_COLUMNS = frozenset({"ticker", "cik"})
_REQUIRED_REGISTRANT_COLUMNS = frozenset({"cik", "primary_ticker", "all_tickers"})
_REQUIRED_FACT_COLUMNS = frozenset(
    {
        "ticker",
        "period_end",
        "line_item",
        "value",
        "units",
        "source_url",
        "filed_date",
    }
)
_SEC_COMPANYFACTS_PATH_RE = re.compile(
    r"^/api/xbrl/companyfacts/CIK(?P<cik>\d{1,10})\.json$",
    re.IGNORECASE,
)
_SEC_COMPANYFACTS_HOSTS = frozenset({"data.sec.gov", "www.sec.gov", "sec.gov"})
_REQUIRED_QUOTE_COLUMNS = frozenset(
    {
        "ticker",
        "provider",
        "as_of_date",
        "price",
        "currency",
        "price_basis",
        "split_adjustment_factor",
        "split_effective_date",
        "source_url",
        "status",
        "fetched_at",
        "expires_at",
        "raw_json",
        "quote_hash",
    }
)

ByteFetcher = Callable[[str, dict[str, Any]], bytes]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Inspect a table before every query against an existing schema."""

    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _normalized_cik(value: Any) -> str | None:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    return digits.zfill(10) if digits else None


def _iso_date(value: Any) -> str | None:
    token = str(value or "").strip()[:10]
    try:
        return date.fromisoformat(token).isoformat()
    except ValueError:
        return None


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _safe_failure_reason(exc: Exception) -> str:
    detail = str(exc).strip()
    return detail if re.fullmatch(r"[A-Z0-9_]+", detail) else type(exc).__name__


def _split_factor(value: Any) -> float | None:
    direct = _positive_number(value)
    if direct is not None:
        return direct
    token = str(value or "").strip()
    if "/" not in token:
        return None
    numerator, denominator = token.split("/", 1)
    top = _positive_number(numerator)
    bottom = _positive_number(denominator)
    return top / bottom if top is not None and bottom is not None else None


def _eodhd_url(cfg: AppConfig, path: str) -> str:
    base = str(cfg.eodhd_base_url or "https://eodhd.com").rstrip("/")
    parsed = urlparse(base)
    if parsed.scheme.lower() != "https" or (
        (parsed.hostname or "").lower() != "eodhd.com"
        and not (parsed.hostname or "").lower().endswith(".eodhd.com")
    ):
        raise ValueError("EODHD_BASE_URL_NOT_AUTHORITATIVE")
    return f"{base}{path}"


def _default_fetcher(cfg: AppConfig) -> ByteFetcher:
    # Relight children stay in safe mode.  Only this parent-owned, EODHD-only
    # client receives a non-mutating config copy that authorizes the subscribed
    # endpoint.
    parent_cfg = cfg if not cfg.safe_mode else cfg.model_copy(update={"safe_mode": False})
    client = HttpClient(parent_cfg)

    def fetch(url: str, params: dict[str, Any]) -> bytes:
        return client.get_bytes(
            url,
            params=params,
            use_cache=False,
            cache_ttl_seconds=None,
        )

    return fetch


def _companyfacts_source_cik(value: Any) -> str | None:
    parsed = urlparse(str(value or "").strip())
    host = (parsed.hostname or "").lower()
    match = _SEC_COMPANYFACTS_PATH_RE.fullmatch(parsed.path)
    if parsed.scheme.lower() != "https" or host not in _SEC_COMPANYFACTS_HOSTS or match is None:
        return None
    return _normalized_cik(match.group("cik"))


def _exact_ticker_aliases(value: Any) -> set[str]:
    try:
        decoded = json.loads(str(value or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()
    if not isinstance(decoded, list):
        return set()
    return {str(item).strip().upper() for item in decoded if str(item).strip()}


def _fallback_issuer_cik(
    conn: sqlite3.Connection,
    *,
    ticker: str,
) -> tuple[str, str] | None:
    company_columns = _table_columns(conn, "companies")
    if _REQUIRED_COMPANY_COLUMNS.issubset(company_columns):
        row = conn.execute(
            "SELECT cik FROM companies WHERE ticker = ? LIMIT 1",
            (ticker,),
        ).fetchone()
        candidate = _normalized_cik(row["cik"] if row is not None else None)
        if candidate is not None:
            return candidate, "companies"

    registrant_columns = _table_columns(conn, "sec_registrants")
    if not _REQUIRED_REGISTRANT_COLUMNS.issubset(registrant_columns):
        return None
    active_clause = " AND removed_at IS NULL" if "removed_at" in registrant_columns else ""
    rows = conn.execute(
        f"""
        SELECT cik
        FROM sec_registrants
        WHERE UPPER(primary_ticker) = ?
        {active_clause}
        """,
        (ticker,),
    ).fetchall()
    primary_candidates = {
        candidate for row in rows if (candidate := _normalized_cik(row["cik"])) is not None
    }
    if len(primary_candidates) == 1:
        return primary_candidates.pop(), "sec_registrants_primary_ticker"
    if len(primary_candidates) > 1:
        return None

    rows = conn.execute(
        f"""
        SELECT cik, all_tickers
        FROM sec_registrants
        WHERE 1 = 1
        {active_clause}
        """
    ).fetchall()
    alias_candidates = {
        candidate
        for row in rows
        if ticker in _exact_ticker_aliases(row["all_tickers"])
        if (candidate := _normalized_cik(row["cik"])) is not None
    }
    if len(alias_candidates) == 1:
        return alias_candidates.pop(), "sec_registrants_all_tickers_exact"
    return None


def _load_issuer_and_shares(
    *,
    ticker: str,
    as_of_date: str,
    db_path: Path,
    cfg: AppConfig,
) -> dict[str, Any] | None:
    if not db_path.is_file():
        return None
    try:
        conn = connect(db_path, cfg=cfg)
        try:
            if not _REQUIRED_FACT_COLUMNS.issubset(_table_columns(conn, "companyfacts_facts")):
                return None
            rows = conn.execute(
                """
                SELECT period_end, filed_date, value, units, source_url
                FROM companyfacts_facts
                WHERE ticker = ?
                  AND line_item = 'shares_outstanding'
                  AND value IS NOT NULL
                  AND value > 0
                  AND period_end <= ?
                  AND filed_date IS NOT NULL
                  AND filed_date <> ''
                  AND filed_date <= ?
                ORDER BY period_end DESC, filed_date DESC
                """,
                (ticker, as_of_date, as_of_date),
            ).fetchall()
            fallback_issuer = _fallback_issuer_cik(conn, ticker=ticker)
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    for row in rows:
        value = _positive_number(row["value"])
        unit = str(row["units"] or "").strip()
        period_end = _iso_date(row["period_end"])
        filed_date = _iso_date(row["filed_date"])
        if value is None or period_end is None or filed_date is None:
            continue
        if unit == "shares":
            shares_mm = value / 1_000_000.0
        elif unit == "shares_millions":
            shares_mm = value
        else:
            continue
        issuer_cik = _companyfacts_source_cik(row["source_url"])
        issuer_cik_source = "companyfacts_source_url"
        if issuer_cik is None:
            if fallback_issuer is None:
                continue
            issuer_cik, issuer_cik_source = fallback_issuer
        return {
            "issuer_cik": issuer_cik,
            "issuer_cik_source": issuer_cik_source,
            "shares_period_end": period_end,
            "shares_filed_date": filed_date,
            "shares_mm": shares_mm,
            "shares_source_url": str(row["source_url"] or "").strip() or None,
        }
    return None


def _parse_quote_payload(
    raw_bytes: bytes,
    *,
    requested_as_of: str,
    fallback_days: int,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    requested = date.fromisoformat(requested_as_of)
    earliest = requested - timedelta(days=max(0, int(fallback_days)))
    candidates: list[tuple[str, float, float | None]] = []
    for row in payload:
        if not isinstance(row, Mapping):
            continue
        row_date = _iso_date(row.get("date"))
        close = _positive_number(row.get("close"))
        adjusted_close = _positive_number(row.get("adjusted_close"))
        if row_date is None or close is None:
            continue
        parsed = date.fromisoformat(row_date)
        if earliest <= parsed <= requested:
            candidates.append((row_date, close, adjusted_close))
    if not candidates:
        return None
    quote_date, close, adjusted_close = max(candidates, key=lambda item: item[0])
    return {
        "as_of_date": quote_date,
        "close": close,
        "adjusted_close": adjusted_close,
    }


def _parse_split_payload(raw_bytes: bytes) -> tuple[list[dict[str, Any]], Any] | None:
    try:
        payload = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            return None
        effective_date = _iso_date(item.get("date") or item.get("effective_date"))
        factor = _split_factor(item.get("split") or item.get("factor"))
        if effective_date is None or factor is None:
            return None
        rows.append(
            {
                "effective_date": effective_date,
                "factor": factor,
                "filed_date": _iso_date(item.get("filed_date")),
            }
        )
    return rows, payload


def _materialize_exact_provider_response(
    *,
    raw_bytes: bytes,
    ticker: str,
    issuer_cik: str,
) -> tuple[str, str]:
    digest = hashlib.sha256(raw_bytes).hexdigest()
    root = canonical_split_proof_cache_root()
    path = (
        root / "provider_responses" / "eodhd" / issuer_cik / ticker / f"{digest}.json"
    ).resolve()
    _atomic_write_bytes(path, raw_bytes)
    return path.relative_to(root).as_posix(), digest


def _materialize_proof(
    *,
    record: dict[str, Any],
    raw_payload: Any,
) -> dict[str, Any]:
    raw_envelope = split_proof_raw_materialization_envelope(record, raw_payload)
    raw_bytes = _canonical_json_bytes(raw_envelope)
    raw_path = canonical_split_proof_raw_cache_path(record, raw_payload)
    _atomic_write_bytes(raw_path, raw_bytes)
    bound_record = {
        **record,
        "raw_relative_path": raw_path.relative_to(canonical_split_proof_cache_root()).as_posix(),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_payload_sha256": raw_envelope["payload_sha256"],
    }
    proof_envelope = split_proof_materialization_envelope(bound_record)
    proof_bytes = _canonical_json_bytes(proof_envelope)
    proof_path = canonical_split_proof_cache_path(bound_record)
    _atomic_write_bytes(proof_path, proof_bytes)
    return {
        **bound_record,
        "materialized_path": str(proof_path),
        "materialized_sha256": hashlib.sha256(proof_bytes).hexdigest(),
    }


def _proof_and_quote_basis(
    *,
    ticker: str,
    issuer_cik: str,
    shares_period_end: str,
    quote: dict[str, Any],
    split_rows: list[dict[str, Any]],
    split_payload: Any,
    split_source_url: str,
    retrieved_on: str,
    exact_response_relative_path: str,
    exact_response_sha256: str,
) -> dict[str, Any]:
    quote_date = str(quote["as_of_date"])
    window_events = [
        row for row in split_rows if shares_period_end <= str(row["effective_date"]) <= quote_date
    ]
    common = {
        "ticker": ticker,
        "issuer_cik": issuer_cik,
        "source": "eodhd_splits",
        "source_reference": split_source_url,
        "source_provider": "eodhd",
        "provider_symbol": ticker,
        "retrieved_at": retrieved_on,
        "provider_response_relative_path": exact_response_relative_path,
        "provider_response_sha256": exact_response_sha256,
    }
    if not window_events:
        proof = _materialize_proof(
            record={
                **common,
                "proof_kind": SPLIT_PROOF_KIND_NO_INTERVENING_SPLIT,
                "status": "PASS",
                "period_start": shares_period_end,
                "period_end": quote_date,
                "verified_as_of": retrieved_on,
            },
            raw_payload=split_payload,
        )
        if not authoritative_split_proof_reference(
            proof,
            expected_ticker=ticker,
            expected_issuer_cik=issuer_cik,
            expected_as_of_date=retrieved_on,
        ):
            raise ValueError("MATERIALIZED_SPLIT_PROOF_INVALID")
        return {
            "price": float(quote["close"]),
            "raw_price": float(quote["close"]),
            "price_basis": PRICE_BASIS_UNADJUSTED,
            "split_adjustment_factor": 1.0,
            "split_effective_date": None,
            "split_event": None,
            "no_intervening_split_proof": proof,
        }
    if len(window_events) != 1:
        raise ValueError("MULTIPLE_INTERVENING_SPLITS_UNSUPPORTED")
    event = window_events[0]
    filed_date = event.get("filed_date")
    adjusted_close = _positive_number(quote.get("adjusted_close"))
    close = _positive_number(quote.get("close"))
    factor = _positive_number(event.get("factor"))
    if (
        filed_date is None
        or adjusted_close is None
        or close is None
        or factor is None
        or not math.isclose(close / adjusted_close, factor, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise ValueError("INTERVENING_SPLIT_PROOF_INCOMPLETE")
    proof = _materialize_proof(
        record={
            **common,
            "proof_kind": SPLIT_PROOF_KIND_SPLIT_EVENT,
            "factor": factor,
            "effective_date": event["effective_date"],
            "filed_date": filed_date,
        },
        raw_payload=split_payload,
    )
    if not authoritative_split_proof_reference(
        proof,
        expected_ticker=ticker,
        expected_issuer_cik=issuer_cik,
        expected_as_of_date=quote_date,
    ):
        raise ValueError("MATERIALIZED_SPLIT_PROOF_INVALID")
    return {
        "price": adjusted_close,
        "raw_price": close,
        "price_basis": PRICE_BASIS_SPLIT_ADJUSTED,
        "split_adjustment_factor": factor,
        "split_effective_date": event["effective_date"],
        "split_event": proof,
        "no_intervening_split_proof": None,
    }


def _persist_quote(
    *,
    ticker: str,
    issuer_cik: str,
    issuer_cik_source: str,
    shares_period_end: str,
    quote_date: str,
    quote_source_url: str,
    fetched_at: str,
    quote_basis: dict[str, Any],
    db_path: Path,
    cfg: AppConfig,
) -> str:
    quote_payload = {
        "schema_version": SPLIT_LINEAGE_QUOTE_SCHEMA_VERSION,
        "record_type": SPLIT_LINEAGE_QUOTE_RECORD_TYPE,
        "ticker": ticker,
        "issuer_cik": issuer_cik,
        "issuer_cik_source": issuer_cik_source,
        "shares_period_end": shares_period_end,
        "as_of_date": quote_date,
        "price": quote_basis["price"],
        "raw_price": quote_basis["raw_price"],
        "currency": "USD",
        "source": SPLIT_LINEAGE_QUOTE_PROVIDER,
        "source_url": quote_source_url,
        "confidence": "HIGH",
        "price_basis": quote_basis["price_basis"],
        "split_adjustment_factor": quote_basis["split_adjustment_factor"],
        "split_effective_date": quote_basis["split_effective_date"],
        "split_event": quote_basis["split_event"],
        "no_intervening_split_proof": quote_basis["no_intervening_split_proof"],
    }
    quote_hash = stable_quote_hash(quote_payload)
    expires_at = (
        datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
        + timedelta(seconds=max(1, int(cfg.quote_ttl_seconds)))
    ).isoformat()
    conn = connect(db_path, cfg=cfg)
    try:
        if not _REQUIRED_QUOTE_COLUMNS.issubset(_table_columns(conn, "price_quotes")):
            raise ValueError("PRICE_QUOTES_SCHEMA_UNSUPPORTED")
        conn.execute(
            """
            INSERT INTO price_quotes(
                ticker, provider, as_of_date, price, currency, price_basis,
                split_adjustment_factor, split_effective_date, source_url,
                status, fetched_at, expires_at, raw_json, quote_hash
            )
            VALUES(?, ?, ?, ?, 'USD', ?, ?, ?, ?, 'OK', ?, ?, ?, ?)
            ON CONFLICT(ticker, provider, as_of_date) DO UPDATE SET
                price = excluded.price,
                currency = excluded.currency,
                price_basis = excluded.price_basis,
                split_adjustment_factor = excluded.split_adjustment_factor,
                split_effective_date = excluded.split_effective_date,
                source_url = excluded.source_url,
                status = excluded.status,
                fetched_at = excluded.fetched_at,
                expires_at = excluded.expires_at,
                raw_json = excluded.raw_json,
                quote_hash = excluded.quote_hash
            """,
            (
                ticker,
                SPLIT_LINEAGE_QUOTE_PROVIDER,
                quote_date,
                quote_basis["price"],
                quote_basis["price_basis"],
                quote_basis["split_adjustment_factor"],
                quote_basis["split_effective_date"],
                quote_source_url,
                fetched_at,
                expires_at,
                _canonical_json_bytes(quote_payload).decode("utf-8"),
                quote_hash,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return quote_hash


def _ready_result_from_persisted_quote(payload: Mapping[str, Any]) -> dict[str, Any]:
    proof_key = (
        "split_event"
        if payload.get("price_basis") == PRICE_BASIS_SPLIT_ADJUSTED
        else "no_intervening_split_proof"
    )
    proof = payload[proof_key]
    shares_period_end = payload.get("shares_period_end")
    if not shares_period_end and proof_key == "no_intervening_split_proof":
        shares_period_end = proof.get("period_start")
    return {
        "ticker": str(payload["ticker"]).strip().upper(),
        "status": "READY",
        "issuer_cik": str(payload["issuer_cik"]),
        "issuer_cik_source": payload.get("issuer_cik_source"),
        "shares_period_end": shares_period_end,
        "quote_as_of_date": str(payload["as_of_date"]),
        "price_basis": str(payload["price_basis"]),
        "quote_hash": stable_quote_hash(payload),
        "proof_kind": str(proof["proof_kind"]),
        "materialized_path": str(proof["materialized_path"]),
    }


def produce_split_lineage_evidence(
    *,
    ticker: str,
    as_of_date: str,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    fetch_bytes: ByteFetcher | None = None,
    retrieved_on: str | None = None,
) -> dict[str, Any]:
    """Reuse a valid persisted quote or fetch one with authoritative split lineage."""

    resolved_cfg = cfg or get_config()
    path = Path(db_path) if db_path is not None else Path(resolved_cfg.db_path)
    ticker_norm = str(ticker or "").strip().upper()
    as_of_norm = _iso_date(as_of_date)
    retrieved_norm = _iso_date(retrieved_on or datetime.now(timezone.utc).date().isoformat())
    if not ticker_norm or as_of_norm is None or retrieved_norm is None:
        return {"ticker": ticker_norm, "status": "UNKNOWN", "reason": "INVALID_REQUEST"}
    persisted = load_persisted_split_lineage_quote(
        ticker_norm,
        as_of_norm,
        db_path=path,
        cfg=resolved_cfg,
    )
    if persisted is not None:
        return _ready_result_from_persisted_quote(persisted)
    if retrieved_norm > as_of_norm:
        return {
            "ticker": ticker_norm,
            "status": "UNKNOWN",
            "reason": "PROOF_RETRIEVED_AFTER_RUN_AS_OF",
        }
    metadata = _load_issuer_and_shares(
        ticker=ticker_norm,
        as_of_date=as_of_norm,
        db_path=path,
        cfg=resolved_cfg,
    )
    if metadata is None:
        return {
            "ticker": ticker_norm,
            "status": "UNKNOWN",
            "reason": "ISSUER_OR_SHARES_MISSING",
        }
    if fetch_bytes is None and not str(resolved_cfg.eodhd_apikey or "").strip():
        return {"ticker": ticker_norm, "status": "UNKNOWN", "reason": "EODHD_APIKEY_MISSING"}
    fetch = fetch_bytes or _default_fetcher(resolved_cfg)
    token = str(resolved_cfg.eodhd_apikey or "")
    exchange = str(resolved_cfg.eodhd_exchange or "US").strip().upper() or "US"
    quote_source_url = _eodhd_url(resolved_cfg, f"/api/eod/{ticker_norm}.{exchange}")
    split_source_url = _eodhd_url(resolved_cfg, f"/api/splits/{ticker_norm}")
    try:
        requested = date.fromisoformat(as_of_norm)
        quote_bytes = fetch(
            quote_source_url,
            {
                "api_token": token,
                "from": (
                    requested - timedelta(days=max(7, int(resolved_cfg.price_fallback_days)))
                ).isoformat(),
                "to": as_of_norm,
                "fmt": "json",
            },
        )
        quote = _parse_quote_payload(
            quote_bytes,
            requested_as_of=as_of_norm,
            fallback_days=max(7, int(resolved_cfg.price_fallback_days)),
        )
        if quote is None:
            raise ValueError("QUOTE_RESPONSE_UNUSABLE")
        split_bytes = fetch(
            split_source_url,
            {
                "api_token": token,
                "from": metadata["shares_period_end"],
                "to": quote["as_of_date"],
                "fmt": "json",
            },
        )
        parsed_splits = _parse_split_payload(split_bytes)
        if parsed_splits is None:
            raise ValueError("SPLIT_RESPONSE_UNUSABLE")
        split_rows, split_payload = parsed_splits
        response_relative_path, response_sha256 = _materialize_exact_provider_response(
            raw_bytes=split_bytes,
            ticker=ticker_norm,
            issuer_cik=metadata["issuer_cik"],
        )
        quote_basis = _proof_and_quote_basis(
            ticker=ticker_norm,
            issuer_cik=metadata["issuer_cik"],
            shares_period_end=metadata["shares_period_end"],
            quote=quote,
            split_rows=split_rows,
            split_payload=split_payload,
            split_source_url=split_source_url,
            retrieved_on=retrieved_norm,
            exact_response_relative_path=response_relative_path,
            exact_response_sha256=response_sha256,
        )
        fetched_at = datetime.now(timezone.utc).isoformat()
        quote_hash = _persist_quote(
            ticker=ticker_norm,
            issuer_cik=metadata["issuer_cik"],
            issuer_cik_source=metadata["issuer_cik_source"],
            shares_period_end=metadata["shares_period_end"],
            quote_date=quote["as_of_date"],
            quote_source_url=quote_source_url,
            fetched_at=fetched_at,
            quote_basis=quote_basis,
            db_path=path,
            cfg=resolved_cfg,
        )
    except Exception as exc:  # noqa: BLE001 - one name must not abort the relight
        return {
            "ticker": ticker_norm,
            "status": "UNKNOWN",
            "reason": _safe_failure_reason(exc),
        }
    proof = quote_basis["split_event"] or quote_basis["no_intervening_split_proof"]
    return {
        "ticker": ticker_norm,
        "status": "READY",
        "issuer_cik": metadata["issuer_cik"],
        "issuer_cik_source": metadata["issuer_cik_source"],
        "shares_period_end": metadata["shares_period_end"],
        "quote_as_of_date": quote["as_of_date"],
        "price_basis": quote_basis["price_basis"],
        "quote_hash": quote_hash,
        "proof_kind": proof["proof_kind"],
        "materialized_path": proof["materialized_path"],
    }


def prepare_relight_split_lineage_evidence(
    *,
    tickers: list[str],
    as_of_date: str,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    fetch_bytes: ByteFetcher | None = None,
    retrieved_on: str | None = None,
) -> dict[str, Any]:
    """Prepare each relight name independently; failures remain fail-closed."""

    normalized = list(
        dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip())
    )
    results = [
        produce_split_lineage_evidence(
            ticker=ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            cfg=cfg,
            fetch_bytes=fetch_bytes,
            retrieved_on=retrieved_on,
        )
        for ticker in normalized
    ]
    ready = sum(result["status"] == "READY" for result in results)
    return {
        "as_of_date": str(as_of_date)[:10],
        "requested": len(normalized),
        "ready": ready,
        "unknown": len(results) - ready,
        "results": results,
    }


def _exact_provider_response_is_valid(proof: Mapping[str, Any]) -> bool:
    relative_token = str(proof.get("provider_response_relative_path") or "").strip()
    expected_sha256 = str(proof.get("provider_response_sha256") or "").strip().lower()
    relative = Path(relative_token)
    if (
        not relative_token
        or relative.is_absolute()
        or ".." in relative.parts
        or len(expected_sha256) != 64
    ):
        return False
    root = canonical_split_proof_cache_root()
    lexical = root / relative
    try:
        resolved = lexical.resolve(strict=True)
        payload = resolved.read_bytes()
    except OSError:
        return False
    return bool(
        str(lexical) == str(resolved) and hashlib.sha256(payload).hexdigest() == expected_sha256
    )


def load_persisted_split_lineage_quote(
    ticker: str,
    as_of_date: str,
    *,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any] | None:
    """Read one dedicated quote row and revalidate all materialized evidence."""

    resolved_cfg = cfg or get_config()
    path = Path(db_path) if db_path is not None else Path(resolved_cfg.db_path)
    ticker_norm = str(ticker or "").strip().upper()
    as_of_norm = _iso_date(as_of_date)
    if not ticker_norm or as_of_norm is None or not path.is_file():
        return None
    try:
        conn = connect(path, cfg=resolved_cfg)
        try:
            if not _REQUIRED_QUOTE_COLUMNS.issubset(_table_columns(conn, "price_quotes")):
                return None
            row = conn.execute(
                """
                SELECT ticker, provider, as_of_date, price, currency, price_basis,
                       split_adjustment_factor, split_effective_date, source_url,
                       status, fetched_at, raw_json, quote_hash
                FROM price_quotes
                WHERE ticker = ?
                  AND provider = ?
                  AND as_of_date <= ?
                  AND status = 'OK'
                  AND price IS NOT NULL
                ORDER BY as_of_date DESC, fetched_at DESC
                LIMIT 1
                """,
                (ticker_norm, SPLIT_LINEAGE_QUOTE_PROVIDER, as_of_norm),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    quote_date = _iso_date(row["as_of_date"])
    if quote_date is None:
        return None
    if (date.fromisoformat(as_of_norm) - date.fromisoformat(quote_date)).days > max(
        7, int(resolved_cfg.price_fallback_days)
    ):
        return None
    try:
        payload = json.loads(str(row["raw_json"] or ""))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("schema_version") != SPLIT_LINEAGE_QUOTE_SCHEMA_VERSION
        or payload.get("record_type") != SPLIT_LINEAGE_QUOTE_RECORD_TYPE
        or str(payload.get("ticker") or "").strip().upper() != ticker_norm
        or payload.get("as_of_date") != quote_date
        or _positive_number(payload.get("price")) != _positive_number(row["price"])
        or str(payload.get("currency") or "").upper() != str(row["currency"] or "").upper()
        or str(payload.get("price_basis") or "") != str(row["price_basis"] or "")
        or stable_quote_hash(payload) != str(row["quote_hash"] or "")
    ):
        return None
    proof_key = (
        "split_event"
        if payload.get("price_basis") == PRICE_BASIS_SPLIT_ADJUSTED
        else "no_intervening_split_proof"
    )
    proof = payload.get(proof_key)
    if not isinstance(proof, Mapping):
        return None
    issuer_cik = _normalized_cik(payload.get("issuer_cik"))
    if (
        issuer_cik is None
        or not _exact_provider_response_is_valid(proof)
        or not authoritative_split_proof_reference(
            proof,
            expected_ticker=ticker_norm,
            expected_issuer_cik=issuer_cik,
            expected_as_of_date=as_of_norm,
        )
    ):
        return None
    return dict(payload)


__all__ = [
    "SPLIT_LINEAGE_QUOTE_PROVIDER",
    "load_persisted_split_lineage_quote",
    "prepare_relight_split_lineage_evidence",
    "produce_split_lineage_evidence",
]
