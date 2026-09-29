"""Immutable model recommendation staking, reads, and historical seed import."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from app.calibration.decision_ledger import select_benchmark_symbol
from app.calibration.recommendation_schema import ensure_recommendation_ledger_schema
from app.config import get_config
from app.db import connect, utc_now_iso


RECOMMENDATION_TYPES = frozenset({"BUY_AT_LIMIT", "WATCH", "AVOID"})
RECORD_VINTAGES = frozenset({"LIVE", "SEED"})
BENCHMARK_SYMBOLS = frozenset({"IWM", "SPY"})
SEED_MODEL_ID = "UNATTRIBUTED_HISTORICAL_MODEL"
SEED_MODEL_VINTAGE = "SEED_2026_07_28"
SEED_POLICY_ID = "recovery-plan-2026-07-28-phase-2-seed-v1"
SEED_POLICY_HASH = hashlib.sha256(SEED_POLICY_ID.encode("utf-8")).hexdigest()
SEED_HORIZONS = (90, 365)
MISSING_RISK_FLAG = "HISTORICAL_RISK_FLAGS_NOT_RECORDED"
MISSING_PRE_MORTEM = "Historical pre-mortem and falsifiers were not recorded."

_LEDGER_COLUMNS = (
    "recommendation_id",
    "ticker",
    "recommendation_type",
    "record_vintage",
    "model_id",
    "model_vintage",
    "thesis_reference",
    "thesis_summary",
    "trigger_price",
    "trigger_price_source",
    "trigger_price_as_of",
    "trigger_price_age_seconds",
    "target_price",
    "target_price_source",
    "conviction_grade",
    "capacity_class",
    "adv_dollar_20d",
    "adv_as_of",
    "pre_mortem",
    "risk_flags_json",
    "policy_hash",
    "source_run_id",
    "horizons_json",
    "benchmark_symbol",
    "staked_at",
    "recorded_at",
    "source_disposition_id",
    "corrects_recommendation_id",
    "correction_reason",
)

_SEED_REQUIRED_COLUMNS = {
    "dispositions": {
        "id",
        "ticker",
        "watchlist_id",
        "kind",
        "status",
        "opened_at",
        "trigger_snapshot_json",
        "pre_mortem",
    },
    "watchlist": {
        "id",
        "ticker",
        "status",
        "conviction_grade",
        "valuation_anchor_method",
        "buy_price_target",
        "thesis_text",
        "key_risks_json",
        "falsifiers_json",
        "source_run_id",
        "cap_band",
        "adv_dollar_20d",
        "adv_asof",
        "capacity_class",
    },
    "watchlist_price_snapshots": {
        "id",
        "watchlist_id",
        "price",
        "checked_at",
        "source",
    },
}


@dataclass(frozen=True, slots=True)
class RecommendationDraft:
    """Complete input required to stake one live model recommendation."""

    recommendation_id: str
    ticker: str
    recommendation_type: str
    model_id: str
    model_vintage: str
    thesis_reference: str
    thesis_summary: str
    trigger_price: float
    trigger_price_source: str
    trigger_price_as_of: str
    trigger_price_age_seconds: int
    target_price: float
    target_price_source: str
    conviction_grade: str
    capacity_class: str
    adv_dollar_20d: float | None
    adv_as_of: str | None
    pre_mortem: str
    risk_flags: tuple[str, ...]
    policy_hash: str
    source_run_id: str
    horizons: tuple[int, ...]
    benchmark_symbol: str
    staked_at: str
    corrects_recommendation_id: str | None = None
    correction_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _RecommendationRecord:
    recommendation_id: str
    ticker: str
    recommendation_type: str
    record_vintage: str
    model_id: str
    model_vintage: str
    thesis_reference: str
    thesis_summary: str
    trigger_price: float
    trigger_price_source: str
    trigger_price_as_of: str
    trigger_price_age_seconds: int
    target_price: float
    target_price_source: str
    conviction_grade: str
    capacity_class: str
    adv_dollar_20d: float | None
    adv_as_of: str | None
    pre_mortem: str
    risk_flags: tuple[str, ...]
    policy_hash: str
    source_run_id: str
    horizons: tuple[int, ...]
    benchmark_symbol: str
    staked_at: str
    recorded_at: str
    source_disposition_id: int | None
    corrects_recommendation_id: str | None
    correction_reason: str | None


def _required_text(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} is required")
    return normalized


def _timestamp(value: Any, field: str) -> tuple[str, datetime]:
    normalized = _required_text(value, field)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return normalized, parsed


def _positive_price(value: Any, field: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive number") from exc
    if normalized <= 0:
        raise ValueError(f"{field} must be a positive number")
    return normalized


def _positive_adv(value: Any) -> float | None:
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("adv_dollar_20d must be positive when supplied") from exc
    if normalized <= 0:
        raise ValueError("adv_dollar_20d must be positive when supplied")
    return normalized


def _string_tuple(values: Iterable[Any], field: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values if str(value).strip())
    if not normalized:
        raise ValueError(f"{field} must contain at least one explicit value")
    return normalized


def _horizons(values: Iterable[Any]) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("horizons must contain positive day counts")
    if len(set(normalized)) != len(normalized):
        raise ValueError("horizons must not contain duplicates")
    return normalized


def _policy_hash(value: Any) -> str:
    normalized = _required_text(value, "policy_hash").lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("policy_hash must be a 64-character SHA-256 hex digest")
    return normalized


def _normalize_record(record: _RecommendationRecord) -> _RecommendationRecord:
    recommendation_type = _required_text(
        record.recommendation_type, "recommendation_type"
    ).upper()
    if recommendation_type not in RECOMMENDATION_TYPES:
        raise ValueError("recommendation_type must be BUY_AT_LIMIT, WATCH, or AVOID")
    record_vintage = _required_text(record.record_vintage, "record_vintage").upper()
    if record_vintage not in RECORD_VINTAGES:
        raise ValueError("record_vintage must be LIVE or SEED")
    if record_vintage == "LIVE" and record.source_disposition_id is not None:
        raise ValueError("live recommendations cannot carry a seed disposition id")
    if record_vintage == "SEED" and record.source_disposition_id is None:
        raise ValueError("seed recommendations require a source disposition id")

    correction_id = (
        _required_text(record.corrects_recommendation_id, "corrects_recommendation_id")
        if record.corrects_recommendation_id is not None
        else None
    )
    correction_reason = (
        _required_text(record.correction_reason, "correction_reason")
        if record.correction_reason is not None
        else None
    )
    if (correction_id is None) != (correction_reason is None):
        raise ValueError("corrections require both predecessor id and reason")

    staked_at, staked_dt = _timestamp(record.staked_at, "staked_at")
    trigger_as_of, trigger_dt = _timestamp(record.trigger_price_as_of, "trigger_price_as_of")
    age_seconds = int(record.trigger_price_age_seconds)
    actual_age_seconds = int((staked_dt - trigger_dt).total_seconds())
    if age_seconds < 0 or actual_age_seconds != age_seconds:
        raise ValueError("trigger_price_age_seconds must match staked_at minus trigger_price_as_of")
    recorded_at, _ = _timestamp(record.recorded_at, "recorded_at")

    benchmark_symbol = _required_text(record.benchmark_symbol, "benchmark_symbol").upper()
    if benchmark_symbol not in BENCHMARK_SYMBOLS:
        raise ValueError("benchmark_symbol must be IWM or SPY")
    adv_as_of = (
        _required_text(record.adv_as_of, "adv_as_of")
        if record.adv_as_of is not None
        else None
    )

    return _RecommendationRecord(
        recommendation_id=_required_text(record.recommendation_id, "recommendation_id"),
        ticker=_required_text(record.ticker, "ticker").upper(),
        recommendation_type=recommendation_type,
        record_vintage=record_vintage,
        model_id=_required_text(record.model_id, "model_id"),
        model_vintage=_required_text(record.model_vintage, "model_vintage"),
        thesis_reference=_required_text(record.thesis_reference, "thesis_reference"),
        thesis_summary=_required_text(record.thesis_summary, "thesis_summary"),
        trigger_price=_positive_price(record.trigger_price, "trigger_price"),
        trigger_price_source=_required_text(
            record.trigger_price_source, "trigger_price_source"
        ),
        trigger_price_as_of=trigger_as_of,
        trigger_price_age_seconds=age_seconds,
        target_price=_positive_price(record.target_price, "target_price"),
        target_price_source=_required_text(record.target_price_source, "target_price_source"),
        conviction_grade=_required_text(record.conviction_grade, "conviction_grade").upper(),
        capacity_class=_required_text(record.capacity_class, "capacity_class").upper(),
        adv_dollar_20d=_positive_adv(record.adv_dollar_20d),
        adv_as_of=adv_as_of,
        pre_mortem=_required_text(record.pre_mortem, "pre_mortem"),
        risk_flags=_string_tuple(record.risk_flags, "risk_flags"),
        policy_hash=_policy_hash(record.policy_hash),
        source_run_id=_required_text(record.source_run_id, "source_run_id"),
        horizons=_horizons(record.horizons),
        benchmark_symbol=benchmark_symbol,
        staked_at=staked_at,
        recorded_at=recorded_at,
        source_disposition_id=record.source_disposition_id,
        corrects_recommendation_id=correction_id,
        correction_reason=correction_reason,
    )


def _record_values(record: _RecommendationRecord) -> tuple[Any, ...]:
    payload = {
        field: getattr(record, field)
        for field in _LEDGER_COLUMNS
        if field not in {"risk_flags_json", "horizons_json"}
    }
    payload["risk_flags_json"] = json.dumps(list(record.risk_flags))
    payload["horizons_json"] = json.dumps(list(record.horizons))
    return tuple(payload[field] for field in _LEDGER_COLUMNS)


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    payload = {key: row[key] for key in row.keys()}
    payload["risk_flags"] = json.loads(payload["risk_flags_json"])
    payload["horizons"] = json.loads(payload["horizons_json"])
    return payload


def _get_from_connection(
    conn: sqlite3.Connection, recommendation_id: str
) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM recommendation_ledger WHERE recommendation_id = ?",
        (recommendation_id,),
    ).fetchone()
    return _row_dict(row) if row is not None else None


def _insert_record(conn: sqlite3.Connection, record: _RecommendationRecord) -> dict[str, Any]:
    normalized = _normalize_record(record)
    if normalized.corrects_recommendation_id is not None:
        predecessor = _get_from_connection(conn, normalized.corrects_recommendation_id)
        if predecessor is None:
            raise ValueError("corrected recommendation does not exist")
        if predecessor["ticker"] != normalized.ticker:
            raise ValueError("a correction must keep the predecessor ticker")

    placeholders = ", ".join("?" for _ in _LEDGER_COLUMNS)
    conn.execute(
        f"INSERT INTO recommendation_ledger({', '.join(_LEDGER_COLUMNS)}) "
        f"VALUES ({placeholders})",
        _record_values(normalized),
    )
    stored = _get_from_connection(conn, normalized.recommendation_id)
    if stored is None:
        raise RuntimeError("recommendation insert did not persist")
    return stored


def stake_recommendation(
    draft: RecommendationDraft,
    *,
    db_path: str | Path | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Stake one immutable LIVE recommendation.

    Corrections use the same API with ``corrects_recommendation_id`` and
    ``correction_reason`` populated; no update path exists.
    """

    owns_connection = conn is None
    if conn is None:
        conn = connect(db_path)
        ensure_recommendation_ledger_schema(conn)
    try:
        record = _RecommendationRecord(
            **{
                field: getattr(draft, field)
                for field in RecommendationDraft.__dataclass_fields__
            },
            record_vintage="LIVE",
            recorded_at=utc_now_iso(),
            source_disposition_id=None,
        )
        stored = _insert_record(conn, record)
        if owns_connection:
            conn.commit()
        return stored
    finally:
        if owns_connection:
            conn.close()


def _readonly_connection(db_path: str | Path | None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else Path(get_config().db_path)
    uri = f"{path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def get_recommendation(
    recommendation_id: str,
    *,
    db_path: str | Path | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    """Read one recommendation without creating or mutating schema."""

    owns_connection = conn is None
    if conn is None:
        conn = _readonly_connection(db_path)
    try:
        return _get_from_connection(conn, recommendation_id)
    finally:
        if owns_connection:
            conn.close()


def list_recommendations(
    *,
    ticker: str | None = None,
    recommendation_type: str | None = None,
    record_vintage: str | None = None,
    limit: int = 200,
    db_path: str | Path | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """List recommendations from the append-only ledger without writes."""

    if limit < 1 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")
    clauses: list[str] = []
    params: list[Any] = []
    if ticker:
        clauses.append("ticker = ?")
        params.append(str(ticker).strip().upper())
    if recommendation_type:
        normalized_type = str(recommendation_type).strip().upper()
        if normalized_type not in RECOMMENDATION_TYPES:
            raise ValueError("recommendation_type must be BUY_AT_LIMIT, WATCH, or AVOID")
        clauses.append("recommendation_type = ?")
        params.append(normalized_type)
    if record_vintage:
        normalized_vintage = str(record_vintage).strip().upper()
        if normalized_vintage not in RECORD_VINTAGES:
            raise ValueError("record_vintage must be LIVE or SEED")
        clauses.append("record_vintage = ?")
        params.append(normalized_vintage)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(int(limit))

    owns_connection = conn is None
    if conn is None:
        conn = _readonly_connection(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT *
            FROM recommendation_ledger
            {where}
            ORDER BY staked_at DESC, id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [_row_dict(row) for row in rows]
    finally:
        if owns_connection:
            conn.close()


def _require_seed_source_columns(conn: sqlite3.Connection) -> None:
    """PRAGMA every existing source table before the seed SELECT."""

    for table, required in _SEED_REQUIRED_COLUMNS.items():
        columns = {
            str(row["name"] if isinstance(row, sqlite3.Row) else row[1])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        missing = sorted(required - columns)
        if missing:
            raise RuntimeError(f"{table} is missing required seed columns: {', '.join(missing)}")


def _json_string_list(raw: Any) -> tuple[str, ...]:
    try:
        payload = json.loads(str(raw or "[]"))
    except (TypeError, ValueError) as exc:
        raise ValueError("historical seed JSON must be a list of strings") from exc
    if not isinstance(payload, list):
        raise ValueError("historical seed JSON must be a list of strings")
    return tuple(str(value).strip() for value in payload if str(value).strip())


def _seed_source_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    _require_seed_source_columns(conn)
    return conn.execute(
        """
        SELECT
            d.id AS disposition_id,
            d.ticker,
            d.opened_at,
            d.trigger_snapshot_json,
            d.pre_mortem,
            w.id AS watchlist_id,
            w.status AS watchlist_status,
            w.conviction_grade,
            w.valuation_anchor_method,
            w.buy_price_target,
            w.thesis_text,
            w.key_risks_json,
            w.falsifiers_json,
            w.source_run_id,
            w.cap_band,
            w.adv_dollar_20d,
            w.adv_asof,
            w.capacity_class,
            p.price AS trigger_price,
            p.checked_at AS trigger_price_as_of,
            p.source AS trigger_price_source
        FROM dispositions d
        JOIN watchlist w ON w.id = d.watchlist_id
        JOIN watchlist_price_snapshots p ON p.id = (
            SELECT p2.id
            FROM watchlist_price_snapshots p2
            WHERE p2.watchlist_id = w.id
              AND p2.checked_at <= d.opened_at
            ORDER BY p2.checked_at DESC, p2.id DESC
            LIMIT 1
        )
        WHERE d.kind = 'AT_TARGET'
          AND d.status = 'OPEN'
        ORDER BY d.id
        """
    ).fetchall()


def _recommendation_type_for_grade(grade: str) -> str:
    """Reuse the existing decision-ledger grade semantics in the new vocabulary."""

    if grade == "ACTIONABLE":
        return "BUY_AT_LIMIT"
    if grade == "AVOID":
        return "AVOID"
    return "WATCH"


def _seed_record(row: sqlite3.Row, *, recorded_at: str) -> _RecommendationRecord:
    try:
        trigger_snapshot = json.loads(str(row["trigger_snapshot_json"] or "{}"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"disposition {row['disposition_id']} has invalid trigger snapshot") from exc
    if not isinstance(trigger_snapshot, dict):
        raise ValueError(f"disposition {row['disposition_id']} has invalid trigger snapshot")

    grade = str(
        trigger_snapshot.get("conviction_grade") or row["conviction_grade"] or ""
    ).strip().upper()
    target_price = trigger_snapshot.get("buy_price_target", row["buy_price_target"])
    staked_at, staked_dt = _timestamp(row["opened_at"], "opened_at")
    trigger_as_of, trigger_dt = _timestamp(row["trigger_price_as_of"], "trigger_price_as_of")
    age_seconds = int((staked_dt - trigger_dt).total_seconds())
    risks = _json_string_list(row["key_risks_json"]) or (MISSING_RISK_FLAG,)
    falsifiers = _json_string_list(row["falsifiers_json"])
    pre_mortem = str(row["pre_mortem"] or "").strip()
    if not pre_mortem and falsifiers:
        pre_mortem = "Historical falsifiers: " + "; ".join(falsifiers)
    if not pre_mortem:
        pre_mortem = MISSING_PRE_MORTEM
    source_run_id = _required_text(row["source_run_id"], "source_run_id")
    watchlist_id = int(row["watchlist_id"])
    anchor_method = str(row["valuation_anchor_method"] or "UNKNOWN").strip()
    disposition_id = int(row["disposition_id"])

    return _RecommendationRecord(
        recommendation_id=f"seed-at-target-disposition-{disposition_id}",
        ticker=str(row["ticker"]),
        recommendation_type=_recommendation_type_for_grade(grade),
        record_vintage="SEED",
        model_id=SEED_MODEL_ID,
        model_vintage=SEED_MODEL_VINTAGE,
        thesis_reference=f"watchlist:{watchlist_id};run:{source_run_id}",
        thesis_summary=str(row["thesis_text"] or ""),
        trigger_price=float(row["trigger_price"]),
        trigger_price_source=f"watchlist_price_snapshots:{row['trigger_price_source']}",
        trigger_price_as_of=trigger_as_of,
        trigger_price_age_seconds=age_seconds,
        target_price=float(target_price),
        target_price_source=f"watchlist.buy_price_target:{anchor_method}",
        conviction_grade=grade,
        capacity_class=str(row["capacity_class"] or "ADV_UNKNOWN"),
        adv_dollar_20d=row["adv_dollar_20d"],
        adv_as_of=row["adv_asof"],
        pre_mortem=pre_mortem,
        risk_flags=risks,
        policy_hash=SEED_POLICY_HASH,
        source_run_id=source_run_id,
        horizons=SEED_HORIZONS,
        benchmark_symbol=select_benchmark_symbol(market_cap_category=row["cap_band"]),
        staked_at=staked_at,
        recorded_at=recorded_at,
        source_disposition_id=disposition_id,
        corrects_recommendation_id=None,
        correction_reason=None,
    )


def preview_open_at_target_seed(
    *,
    db_path: str | Path | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Derive seed records without writing the ledger or source tables."""

    owns_connection = conn is None
    if conn is None:
        conn = _readonly_connection(db_path)
    try:
        recorded_at = utc_now_iso()
        records = [
            _normalize_record(_seed_record(row, recorded_at=recorded_at))
            for row in _seed_source_rows(conn)
        ]
        return [
            {
                field: (
                    list(value)
                    if field in {"risk_flags", "horizons"}
                    else value
                )
                for field in _RecommendationRecord.__dataclass_fields__
                if (value := getattr(record, field)) is not None
            }
            for record in records
        ]
    finally:
        if owns_connection:
            conn.close()


def seed_open_at_target_dispositions(
    *,
    db_path: str | Path | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, int]:
    """Backfill open AT_TARGET history as SEED rows, never LIVE rows."""

    owns_connection = conn is None
    if conn is None:
        conn = connect(db_path)
        ensure_recommendation_ledger_schema(conn)
    try:
        records = [
            _normalize_record(_seed_record(row, recorded_at=utc_now_iso()))
            for row in _seed_source_rows(conn)
        ]
        inserted = 0
        existing = 0
        for record in records:
            row = conn.execute(
                """
                SELECT recommendation_id
                FROM recommendation_ledger
                WHERE record_vintage = 'SEED'
                  AND source_disposition_id = ?
                """,
                (record.source_disposition_id,),
            ).fetchone()
            if row is not None:
                existing += 1
                continue
            _insert_record(conn, record)
            inserted += 1
        if owns_connection:
            conn.commit()
        return {"source_rows": len(records), "inserted": inserted, "existing": existing}
    finally:
        if owns_connection:
            conn.close()
