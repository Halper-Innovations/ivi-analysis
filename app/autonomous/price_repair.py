"""Point-in-time price source exhaustion for autonomous-sector v2.

This module is intentionally a one-ticker resolver.  It keeps every source
attempt visible, validates every candidate against the requested date, and
never performs FX conversion.  Callers may persist the winning immutable
snapshot into their injected database, but persistence failure does not erase
otherwise valid evidence.

The precedence contract is:

``cap packet -> injected price_quotes -> current-run output -> disk cache ->
prior sector artifacts -> date-aware provider``.

V1 does not import or call this module.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Literal

from app.config import AppConfig, get_config
from app.market.price_provider import (
    PriceSnapshot,
    build_price_provider,
    classify_price_error,
    resolve_price_from_historical_runs,
)
from app.util.credential_hygiene import sanitize_json_value, sanitize_url_credentials

PriceRepairStatus = Literal["RESOLVED", "NEEDS_DATA", "INCOMPLETE"]

_RETRYABLE_REASON_CODES = frozenset(
    {
        "BUDGET_EXHAUSTED",
        "DNS_FAILURE",
        "HTTP_5XX",
        "RATE_LIMIT",
        "TIMEOUT",
        "TLS_FAILURE",
    }
)
_TERMINAL_MISSING_REASON_CODES = frozenset(
    {
        "HTTP_4XX",
        "NON_TRADING_DAY_NO_FALLBACK",
        "OFFLINE_NO_CACHE",
        "PARSE_ERROR",
        "PROVIDER_NO_DATA",
        "SYMBOL_NOT_FOUND",
    }
)


@dataclass(frozen=True)
class PriceSourceAttempt:
    """One transparent source decision in the ordered repair chain."""

    source: str
    status: Literal["HIT", "MISS", "REJECTED", "ERROR", "SKIPPED"]
    reason_code: str
    retryable: bool = False
    terminal_for_attempt: bool = False
    detail: str = ""
    observed_price: float | None = None
    observed_as_of_date: str | None = None
    observed_currency: str | None = None
    source_url: str | None = None
    provider_diagnostic: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_url", sanitize_url_credentials(self.source_url))
        if self.provider_diagnostic is not None:
            object.__setattr__(
                self,
                "provider_diagnostic",
                sanitize_json_value(self.provider_diagnostic),
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V2PriceResolution:
    """Immutable result suitable for a v2 repair checkpoint or packet."""

    ticker: str
    requested_as_of_date: str
    status: PriceRepairStatus
    reason_code: str
    retryable: bool
    terminal_for_attempt: bool
    source_resolution: str | None
    snapshot: PriceSnapshot | None
    attempts: tuple[PriceSourceAttempt, ...]
    persisted: bool = False
    persistence_error: str | None = None

    @property
    def resolved(self) -> bool:
        return self.status == "RESOLVED" and self.snapshot is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "requested_as_of_date": self.requested_as_of_date,
            "status": self.status,
            "reason_code": self.reason_code,
            "retryable": self.retryable,
            "terminal_for_attempt": self.terminal_for_attempt,
            "source_resolution": self.source_resolution,
            "snapshot": asdict(self.snapshot) if self.snapshot is not None else None,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "persisted": self.persisted,
            "persistence_error": self.persistence_error,
        }


@dataclass(frozen=True)
class _CandidateDecision:
    status: Literal["VALID", "NON_USD", "INVALID"]
    reason_code: str
    detail: str
    snapshot: PriceSnapshot | None


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value or "").strip()[:10])
    except ValueError:
        return None


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _object_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, PriceSnapshot):
        return asdict(value)
    keys = (
        "ticker",
        "price",
        "price_used",
        "current_price",
        "as_of_date",
        "price_as_of_date",
        "price_asof_used",
        "currency",
        "price_currency",
        "source",
        "provider",
        "price_source",
        "url",
        "source_url",
        "price_source_url",
        "retrieved_at",
        "fetched_at",
        "confidence",
        "price_confidence",
        "raw_price",
        "volume",
    )
    return {key: getattr(value, key) for key in keys if hasattr(value, key)}


def _coerce_snapshot(
    value: Any,
    *,
    ticker: str,
    default_source: str,
) -> PriceSnapshot | None:
    payload = _object_payload(value)
    raw_price = _first_present(payload, "price_used", "price", "current_price")
    if isinstance(raw_price, bool) or not isinstance(raw_price, (int, float)):
        return None
    observed_as_of = _first_present(
        payload,
        "price_as_of_date",
        "price_asof_used",
        "as_of_date",
    )
    if observed_as_of is None:
        return None
    confidence = str(_first_present(payload, "price_confidence", "confidence") or "MEDIUM").upper()
    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        confidence = "LOW"
    source = str(_first_present(payload, "price_source", "source", "provider") or default_source)
    source_url = _first_present(
        payload,
        "price_source_url",
        "url",
        "source_url",
    )
    retrieved_at = str(
        _first_present(payload, "retrieved_at", "fetched_at")
        or f"{str(observed_as_of)[:10]}T00:00:00+00:00"
    )
    raw_unadjusted = payload.get("raw_price")
    volume = payload.get("volume")
    return PriceSnapshot(
        ticker=str(payload.get("ticker") or ticker).strip().upper(),
        as_of_date=str(observed_as_of).strip()[:10],
        price=float(raw_price),
        currency=str(_first_present(payload, "price_currency", "currency") or "UNKNOWN")
        .strip()
        .upper(),
        source=source,
        retrieved_at=retrieved_at,
        url=str(source_url).strip() if source_url else None,
        confidence=confidence,  # type: ignore[arg-type]
        raw_price=(
            float(raw_unadjusted)
            if isinstance(raw_unadjusted, (int, float)) and not isinstance(raw_unadjusted, bool)
            else None
        ),
        volume=(
            float(volume)
            if isinstance(volume, (int, float)) and not isinstance(volume, bool)
            else None
        ),
    )


def _validate_snapshot(
    snapshot: PriceSnapshot | None,
    *,
    ticker: str,
    requested_as_of: date,
    max_age_days: int,
) -> _CandidateDecision:
    if snapshot is None:
        return _CandidateDecision(
            status="INVALID",
            reason_code="INVALID_PRICE_EVIDENCE",
            detail="Source did not expose a numeric price and dated provenance.",
            snapshot=None,
        )
    if snapshot.ticker.strip().upper() != ticker:
        return _CandidateDecision(
            status="INVALID",
            reason_code="PRICE_TICKER_MISMATCH",
            detail=f"Evidence ticker {snapshot.ticker!r} does not match {ticker!r}.",
            snapshot=snapshot,
        )
    if not math.isfinite(float(snapshot.price)) or float(snapshot.price) <= 0:
        return _CandidateDecision(
            status="INVALID",
            reason_code="NON_POSITIVE_PRICE",
            detail="Price must be finite and greater than zero.",
            snapshot=snapshot,
        )
    observed_as_of = _parse_date(snapshot.as_of_date)
    if observed_as_of is None:
        return _CandidateDecision(
            status="INVALID",
            reason_code="INVALID_PRICE_ASOF",
            detail="Price evidence has no valid ISO as-of date.",
            snapshot=snapshot,
        )
    if observed_as_of > requested_as_of:
        return _CandidateDecision(
            status="INVALID",
            reason_code="FUTURE_PRICE",
            detail=(
                f"Price date {observed_as_of.isoformat()} is after requested "
                f"date {requested_as_of.isoformat()}."
            ),
            snapshot=snapshot,
        )
    age_days = (requested_as_of - observed_as_of).days
    if age_days > max_age_days:
        return _CandidateDecision(
            status="INVALID",
            reason_code="STALE_PRICE",
            detail=f"Price is {age_days} days old; maximum is {max_age_days}.",
            snapshot=snapshot,
        )
    currency = str(snapshot.currency or "").strip().upper()
    if currency in {"", "UNKNOWN", "N/A", "NONE"}:
        return _CandidateDecision(
            status="INVALID",
            reason_code="PRICE_CURRENCY_UNRESOLVED",
            detail="Price evidence does not identify its currency; USD cannot be assumed.",
            snapshot=snapshot,
        )
    if currency != "USD":
        return _CandidateDecision(
            status="NON_USD",
            reason_code="NON_USD_PRICE_UNSUPPORTED",
            detail=(
                f"Price currency {snapshot.currency!r} requires FX normalization, "
                "which autonomous-sector v2 does not perform."
            ),
            snapshot=snapshot,
        )
    return _CandidateDecision(
        status="VALID",
        reason_code="PRICE_RESOLVED",
        detail="Positive USD price is on or before the requested date and within age bounds.",
        snapshot=snapshot,
    )


def _attempt_from_decision(
    source: str,
    decision: _CandidateDecision,
    *,
    provider_diagnostic: dict[str, Any] | None = None,
) -> PriceSourceAttempt:
    snapshot = decision.snapshot
    if decision.status == "VALID":
        status: Literal["HIT", "MISS", "REJECTED", "ERROR", "SKIPPED"] = "HIT"
    else:
        status = "REJECTED"
    return PriceSourceAttempt(
        source=source,
        status=status,
        reason_code=decision.reason_code,
        retryable=False,
        terminal_for_attempt=False,
        detail=decision.detail,
        observed_price=float(snapshot.price) if snapshot is not None else None,
        observed_as_of_date=snapshot.as_of_date if snapshot is not None else None,
        observed_currency=snapshot.currency if snapshot is not None else None,
        source_url=snapshot.url if snapshot is not None else None,
        provider_diagnostic=provider_diagnostic,
    )


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _db_candidates(db_path: Path, ticker: str) -> tuple[list[dict[str, Any]], str | None]:
    if not db_path.exists():
        return [], "Injected database does not exist."
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT ticker, provider, as_of_date, price, currency,
                       source_url, fetched_at
                FROM price_quotes
                WHERE ticker = ?
                  AND UPPER(status) = 'OK'
                  AND price IS NOT NULL
                ORDER BY as_of_date DESC, fetched_at DESC
                """,
                (ticker,),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return [], f"Injected price_quotes lookup failed: {exc}"
    return [dict(row) for row in rows], None


def _run_candidates(
    *,
    cfg: AppConfig,
    run_id: str | None,
    ticker: str,
    requested_as_of_date: str,
) -> tuple[list[dict[str, Any]], str | None]:
    if not run_id:
        return [], "No run_id was supplied."
    path = cfg.outputs_dir / "prices" / run_id / f"{ticker}.json"
    payload = _safe_json(path)
    if payload is None:
        return [], f"No readable current-run price artifact at {path}."
    artifact_requested = str(payload.get("requested_as_of_date") or "").strip()
    if artifact_requested != str(requested_as_of_date).strip():
        return [], (
            f"Current-run artifact requested {artifact_requested or 'no date'}, "
            f"not {requested_as_of_date}."
        )
    if str(payload.get("status") or "").upper() != "OK":
        diagnostic = payload.get("diagnostic")
        result = diagnostic.get("result") if isinstance(diagnostic, dict) else None
        reason = result.get("reason_code") if isinstance(result, dict) else None
        return [], f"Current-run artifact is not OK ({reason or 'UNKNOWN'})."
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, dict):
        return [], "Current-run artifact has no snapshot payload."
    price = snapshot.get("price")
    if not (
        isinstance(price, (int, float))
        and not isinstance(price, bool)
        and math.isfinite(float(price))
        and float(price) > 0
    ):
        return [], "Current-run artifact snapshot price is not a finite positive number."
    return [snapshot], None


def _disk_candidates(
    *,
    cfg: AppConfig,
    ticker: str,
    requested_as_of_date: str,
) -> tuple[list[dict[str, Any]], str | None]:
    path = cfg.cache_dir / "prices" / f"{ticker}.json"
    payload = _safe_json(path)
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return [], f"No readable disk-cache entries at {path}."
    candidates: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("requested_as_of_date") or "") != requested_as_of_date:
            continue
        snapshot = entry.get("snapshot")
        if isinstance(snapshot, dict):
            candidates.append(snapshot)
    candidates.sort(
        key=lambda row: (
            str(row.get("as_of_date") or ""),
            str(row.get("retrieved_at") or ""),
            str(row.get("source") or ""),
        ),
        reverse=True,
    )
    if not candidates:
        return [], "Disk cache has no entry for the requested as-of date."
    return candidates, None


def _evaluate_source_candidates(
    *,
    source: str,
    candidates: Iterable[Any],
    ticker: str,
    requested_as_of: date,
    max_age_days: int,
) -> tuple[_CandidateDecision | None, PriceSourceAttempt]:
    decisions: list[_CandidateDecision] = []
    for candidate in candidates:
        snapshot = _coerce_snapshot(candidate, ticker=ticker, default_source=source)
        decision = _validate_snapshot(
            snapshot,
            ticker=ticker,
            requested_as_of=requested_as_of,
            max_age_days=max_age_days,
        )
        decisions.append(decision)
        if decision.status == "VALID":
            return decision, _attempt_from_decision(source, decision)
    if not decisions:
        return None, PriceSourceAttempt(
            source=source,
            status="MISS",
            reason_code="PRICE_NOT_FOUND",
            detail="Source exposed no price candidate.",
        )
    priority = {
        "PRICE_CURRENCY_UNRESOLVED": 7,
        "NON_USD_PRICE_UNSUPPORTED": 6,
        "FUTURE_PRICE": 5,
        "STALE_PRICE": 4,
        "NON_POSITIVE_PRICE": 3,
        "PRICE_TICKER_MISMATCH": 2,
        "INVALID_PRICE_ASOF": 1,
        "INVALID_PRICE_EVIDENCE": 0,
    }
    representative = max(
        decisions,
        key=lambda item: priority.get(item.reason_code, -1),
    )
    return None, _attempt_from_decision(source, representative)


def _terminal_evidence_reason(
    attempts: Iterable[PriceSourceAttempt],
    *,
    fallback: str,
) -> str:
    reasons = {attempt.reason_code for attempt in attempts}
    if "PRICE_CURRENCY_UNRESOLVED" in reasons:
        return "PRICE_CURRENCY_UNRESOLVED"
    if "NON_USD_PRICE_UNSUPPORTED" in reasons:
        return "NON_USD_PRICE_UNSUPPORTED"
    return fallback


def _persistence_timestamp(value: str) -> str:
    text = str(value or "").strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _persist_snapshot(
    *,
    db_path: Path,
    snapshot: PriceSnapshot,
    cfg: AppConfig,
    source_resolution: str,
) -> tuple[bool, str | None]:
    if not db_path.exists():
        return False, "Injected database does not exist; immutable snapshot returned only."
    fetched_at = _persistence_timestamp(snapshot.retrieved_at)
    expires_at = (
        datetime.fromisoformat(fetched_at) + timedelta(seconds=max(1, int(cfg.quote_ttl_seconds)))
    ).isoformat()
    provenance = {
        "mode": "autonomous_sector_v2_price_repair",
        "source_resolution": source_resolution,
        "ticker": snapshot.ticker,
        "as_of_date": snapshot.as_of_date,
        "price": snapshot.price,
        "currency": snapshot.currency,
        "source": snapshot.source,
        "source_url": snapshot.url,
        "confidence": snapshot.confidence,
    }
    raw_json = json.dumps(provenance, sort_keys=True)
    quote_hash = sha256(raw_json.encode("utf-8")).hexdigest()
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(price_quotes)").fetchall()
            }
            if not columns:
                return False, "Injected database has no price_quotes table."
            if "volume" in columns:
                conn.execute(
                    """
                    INSERT INTO price_quotes(
                        ticker, provider, as_of_date, price, currency, source_url,
                        status, fetched_at, expires_at, raw_json, quote_hash, volume
                    ) VALUES(?, ?, ?, ?, ?, ?, 'OK', ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, provider, as_of_date) DO UPDATE SET
                        price=excluded.price,
                        currency=excluded.currency,
                        source_url=excluded.source_url,
                        status=excluded.status,
                        fetched_at=excluded.fetched_at,
                        expires_at=excluded.expires_at,
                        raw_json=excluded.raw_json,
                        quote_hash=excluded.quote_hash,
                        volume=COALESCE(excluded.volume, price_quotes.volume)
                    """,
                    (
                        snapshot.ticker,
                        snapshot.source or source_resolution,
                        snapshot.as_of_date,
                        float(snapshot.price),
                        snapshot.currency,
                        snapshot.url,
                        fetched_at,
                        expires_at,
                        raw_json,
                        quote_hash,
                        snapshot.volume,
                    ),
                )
            else:
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
                        snapshot.source or source_resolution,
                        snapshot.as_of_date,
                        float(snapshot.price),
                        snapshot.currency,
                        snapshot.url,
                        fetched_at,
                        expires_at,
                        raw_json,
                        quote_hash,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return False, f"Injected price_quotes persistence failed: {exc}"
    return True, None


def _finish_with_decision(
    *,
    ticker: str,
    requested_as_of_date: str,
    source: str,
    decision: _CandidateDecision,
    attempts: list[PriceSourceAttempt],
    db_path: Path,
    cfg: AppConfig,
    persist: bool,
) -> V2PriceResolution:
    if decision.status == "NON_USD":
        return V2PriceResolution(
            ticker=ticker,
            requested_as_of_date=requested_as_of_date,
            status="NEEDS_DATA",
            reason_code=decision.reason_code,
            retryable=False,
            terminal_for_attempt=True,
            source_resolution=source,
            snapshot=decision.snapshot,
            attempts=tuple(attempts),
        )
    assert decision.snapshot is not None
    persisted = False
    persistence_error: str | None = None
    if persist:
        persisted, persistence_error = _persist_snapshot(
            db_path=db_path,
            snapshot=decision.snapshot,
            cfg=cfg,
            source_resolution=source,
        )
    return V2PriceResolution(
        ticker=ticker,
        requested_as_of_date=requested_as_of_date,
        status="RESOLVED",
        reason_code="PRICE_RESOLVED",
        retryable=False,
        terminal_for_attempt=True,
        source_resolution=source,
        snapshot=decision.snapshot,
        attempts=tuple(attempts),
        persisted=persisted,
        persistence_error=persistence_error,
    )


def _provider_reason(
    diagnostic: dict[str, Any] | None,
) -> tuple[str, bool, str]:
    payload = diagnostic or {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    reason = (
        str(
            (result or {}).get("reason_code")
            or (result or {}).get("status")
            or payload.get("status")
            or "PROVIDER_NO_DATA"
        )
        .strip()
        .upper()
    )
    detail = str(
        (result or {}).get("error_detail")
        or (result or {}).get("reason_detail")
        or payload.get("error")
        or "Provider returned no usable price."
    )
    child_attempts = payload.get("provider_attempts")
    child_reasons: list[str] = []
    if isinstance(child_attempts, list):
        for child in child_attempts:
            if not isinstance(child, dict):
                continue
            token = (
                str(
                    child.get("error_code") or child.get("reason_code") or child.get("status") or ""
                )
                .strip()
                .upper()
            )
            if token:
                child_reasons.append(token)
    retryable_candidates = [
        token for token in [reason, *child_reasons] if token in _RETRYABLE_REASON_CODES
    ]
    if retryable_candidates:
        return retryable_candidates[0], True, detail
    terminal_candidates = [
        token for token in [reason, *child_reasons] if token in _TERMINAL_MISSING_REASON_CODES
    ]
    if reason == "PROVIDER_NO_DATA" and terminal_candidates:
        reason = terminal_candidates[0]
    if reason not in _TERMINAL_MISSING_REASON_CODES:
        reason = "PROVIDER_NO_DATA"
    return reason, False, detail


def resolve_v2_price(
    ticker: str,
    *,
    as_of_date: str,
    db_path: str | Path,
    cap_stage_price: Any = None,
    run_id: str | None = None,
    cfg: AppConfig | None = None,
    provider: Any | None = None,
    allow_provider: bool = True,
    persist: bool = True,
    max_age_days: int | None = None,
) -> V2PriceResolution:
    """Exhaust v2 price sources for one ticker with point-in-time guards.

    ``db_path`` is mandatory: neither lookup nor persistence may silently use
    the process-global database.  Supplying ``provider`` is the hermetic test
    and embedding hook; otherwise the configured date-aware chain is built
    only after every local source misses.
    """

    ticker_norm = str(ticker or "").strip().upper()
    requested_text = str(as_of_date or "").strip()[:10]
    requested = _parse_date(requested_text)
    resolved_cfg = cfg or get_config()
    injected_db = Path(db_path)
    fallback_age = max(
        0,
        int(
            max_age_days
            if max_age_days is not None
            else getattr(resolved_cfg, "price_fallback_days", 7)
        ),
    )
    attempts: list[PriceSourceAttempt] = []
    if not ticker_norm:
        return V2PriceResolution(
            ticker=ticker_norm,
            requested_as_of_date=requested_text,
            status="NEEDS_DATA",
            reason_code="INVALID_TICKER",
            retryable=False,
            terminal_for_attempt=True,
            source_resolution=None,
            snapshot=None,
            attempts=(),
        )
    if requested is None:
        return V2PriceResolution(
            ticker=ticker_norm,
            requested_as_of_date=requested_text,
            status="NEEDS_DATA",
            reason_code="INVALID_REQUEST_ASOF",
            retryable=False,
            terminal_for_attempt=True,
            source_resolution=None,
            snapshot=None,
            attempts=(),
        )

    if cap_stage_price is None:
        attempts.append(
            PriceSourceAttempt(
                source="cap_stage",
                status="MISS",
                reason_code="CAP_STAGE_PRICE_MISSING",
                detail="Immutable cap-stage packet carried no price evidence.",
            )
        )
    else:
        decision, attempt = _evaluate_source_candidates(
            source="cap_stage",
            candidates=[cap_stage_price],
            ticker=ticker_norm,
            requested_as_of=requested,
            max_age_days=fallback_age,
        )
        attempts.append(attempt)
        if decision is not None:
            return _finish_with_decision(
                ticker=ticker_norm,
                requested_as_of_date=requested_text,
                source="cap_stage",
                decision=decision,
                attempts=attempts,
                db_path=injected_db,
                cfg=resolved_cfg,
                persist=persist,
            )

    db_candidates, db_error = _db_candidates(injected_db, ticker_norm)
    decision, attempt = _evaluate_source_candidates(
        source="price_quotes",
        candidates=db_candidates,
        ticker=ticker_norm,
        requested_as_of=requested,
        max_age_days=fallback_age,
    )
    if db_error and not db_candidates:
        attempt = PriceSourceAttempt(
            source="price_quotes",
            status="MISS",
            reason_code="PRICE_QUOTES_UNAVAILABLE",
            detail=db_error,
        )
    attempts.append(attempt)
    if decision is not None:
        # A price_quotes hit is already persisted in the injected database.
        result = _finish_with_decision(
            ticker=ticker_norm,
            requested_as_of_date=requested_text,
            source="price_quotes",
            decision=decision,
            attempts=attempts,
            db_path=injected_db,
            cfg=resolved_cfg,
            persist=False,
        )
        if result.resolved:
            return replace(result, persisted=True)
        return result

    run_candidates, run_error = _run_candidates(
        cfg=resolved_cfg,
        run_id=run_id,
        ticker=ticker_norm,
        requested_as_of_date=requested_text,
    )
    decision, attempt = _evaluate_source_candidates(
        source="run_scoped_output",
        candidates=run_candidates,
        ticker=ticker_norm,
        requested_as_of=requested,
        max_age_days=fallback_age,
    )
    if run_error and not run_candidates:
        attempt = PriceSourceAttempt(
            source="run_scoped_output",
            status="SKIPPED" if not run_id else "MISS",
            reason_code="RUN_PRICE_NOT_FOUND",
            detail=run_error,
        )
    attempts.append(attempt)
    if decision is not None:
        return _finish_with_decision(
            ticker=ticker_norm,
            requested_as_of_date=requested_text,
            source="run_scoped_output",
            decision=decision,
            attempts=attempts,
            db_path=injected_db,
            cfg=resolved_cfg,
            persist=persist,
        )

    disk_candidates, disk_error = _disk_candidates(
        cfg=resolved_cfg,
        ticker=ticker_norm,
        requested_as_of_date=requested_text,
    )
    decision, attempt = _evaluate_source_candidates(
        source="disk_cache",
        candidates=disk_candidates,
        ticker=ticker_norm,
        requested_as_of=requested,
        max_age_days=fallback_age,
    )
    if disk_error and not disk_candidates:
        attempt = PriceSourceAttempt(
            source="disk_cache",
            status="MISS",
            reason_code="DISK_PRICE_NOT_FOUND",
            detail=disk_error,
        )
    attempts.append(attempt)
    if decision is not None:
        return _finish_with_decision(
            ticker=ticker_norm,
            requested_as_of_date=requested_text,
            source="disk_cache",
            decision=decision,
            attempts=attempts,
            db_path=injected_db,
            cfg=resolved_cfg,
            persist=persist,
        )

    historical = resolve_price_from_historical_runs(
        ticker=ticker_norm,
        as_of_date=requested_text,
        sectors_dir=resolved_cfg.sectors_dir,
        max_age_days=fallback_age,
    )
    decision, attempt = _evaluate_source_candidates(
        source="prior_sector_artifacts",
        candidates=[historical] if historical is not None else [],
        ticker=ticker_norm,
        requested_as_of=requested,
        max_age_days=fallback_age,
    )
    attempts.append(attempt)
    if decision is not None:
        return _finish_with_decision(
            ticker=ticker_norm,
            requested_as_of_date=requested_text,
            source="prior_sector_artifacts",
            decision=decision,
            attempts=attempts,
            db_path=injected_db,
            cfg=resolved_cfg,
            persist=persist,
        )

    if not allow_provider:
        attempts.append(
            PriceSourceAttempt(
                source="provider",
                status="SKIPPED",
                reason_code="OFFLINE_NO_CACHE",
                retryable=False,
                terminal_for_attempt=True,
                detail="Provider access is disabled and every local source missed.",
            )
        )
        terminal_reason = _terminal_evidence_reason(
            attempts,
            fallback="OFFLINE_NO_CACHE",
        )
        return V2PriceResolution(
            ticker=ticker_norm,
            requested_as_of_date=requested_text,
            status="NEEDS_DATA",
            reason_code=terminal_reason,
            retryable=False,
            terminal_for_attempt=True,
            source_resolution=None,
            snapshot=None,
            attempts=tuple(attempts),
        )

    try:
        effective_provider = provider or build_price_provider(
            cfg=resolved_cfg,
            with_prices=True,
            fallback_days=fallback_age,
        )
        provider_snapshot = effective_provider.get_price_asof(
            ticker_norm,
            requested_text,
        )
        diagnostic = None
        get_diagnostic = getattr(effective_provider, "get_last_diagnostic", None)
        if callable(get_diagnostic):
            diagnostic = get_diagnostic(ticker_norm, requested_text)
            if not isinstance(diagnostic, dict):
                diagnostic = None
    except Exception as exc:  # noqa: BLE001 - classified into truthful result
        reason, classified_retryable, _suggestion, detail = classify_price_error(exc)
        retryable = bool(classified_retryable or reason in _RETRYABLE_REASON_CODES)
        attempts.append(
            PriceSourceAttempt(
                source="provider",
                status="ERROR",
                reason_code=str(reason),
                retryable=retryable,
                terminal_for_attempt=not retryable,
                detail=detail,
            )
        )
        return V2PriceResolution(
            ticker=ticker_norm,
            requested_as_of_date=requested_text,
            status="INCOMPLETE" if retryable else "NEEDS_DATA",
            reason_code=str(reason),
            retryable=retryable,
            terminal_for_attempt=not retryable,
            source_resolution=None,
            snapshot=None,
            attempts=tuple(attempts),
        )

    if provider_snapshot is not None:
        decision = _validate_snapshot(
            _coerce_snapshot(
                provider_snapshot,
                ticker=ticker_norm,
                default_source="provider",
            ),
            ticker=ticker_norm,
            requested_as_of=requested,
            max_age_days=fallback_age,
        )
        attempts.append(
            _attempt_from_decision(
                "provider",
                decision,
                provider_diagnostic=diagnostic,
            )
        )
        if decision.status in {"VALID", "NON_USD"}:
            return _finish_with_decision(
                ticker=ticker_norm,
                requested_as_of_date=requested_text,
                source="provider",
                decision=decision,
                attempts=attempts,
                db_path=injected_db,
                cfg=resolved_cfg,
                persist=persist,
            )
        return V2PriceResolution(
            ticker=ticker_norm,
            requested_as_of_date=requested_text,
            status="NEEDS_DATA",
            reason_code=decision.reason_code,
            retryable=False,
            terminal_for_attempt=True,
            source_resolution="provider",
            snapshot=decision.snapshot,
            attempts=tuple(attempts),
        )

    reason, retryable, detail = _provider_reason(diagnostic)
    attempts.append(
        PriceSourceAttempt(
            source="provider",
            status="ERROR" if retryable else "MISS",
            reason_code=reason,
            retryable=retryable,
            terminal_for_attempt=not retryable,
            detail=detail,
            provider_diagnostic=diagnostic,
        )
    )
    terminal_reason = (
        reason
        if retryable
        else _terminal_evidence_reason(attempts, fallback=reason)
    )
    return V2PriceResolution(
        ticker=ticker_norm,
        requested_as_of_date=requested_text,
        status="INCOMPLETE" if retryable else "NEEDS_DATA",
        reason_code=terminal_reason,
        retryable=retryable,
        terminal_for_attempt=not retryable,
        source_resolution=None,
        snapshot=None,
        attempts=tuple(attempts),
    )


__all__ = [
    "PriceRepairStatus",
    "PriceSourceAttempt",
    "V2PriceResolution",
    "resolve_v2_price",
]
