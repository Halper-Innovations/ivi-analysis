from __future__ import annotations

import json
import logging
import math
import sqlite3
from hashlib import sha256
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_config
from app.autonomous.artifact_financial_audit import (
    PASS as FINANCIAL_INTEGRITY_PASS,
    artifact_decision_eligibility,
    financial_integrity_manifest_is_usable,
)
from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    SectorCompanyFinancialPacket,
)
from app.watchlist.contract import (
    AUTONOMOUS_SECTOR_PIPELINE_V2,
    WATCHLIST_CANDIDATE_DISPOSITIONS,
    WATCHLIST_CONFIDENCE_LABELS,
    WATCHLIST_CONVICTION_GRADES,
    WATCHLIST_CONVICTION_SOURCES,
    WATCHLIST_DECISION_BASES,
    WATCHLIST_SCAN_FAMILIES,
    WATCHLIST_STATUSES,
    WatchlistEntry,
    is_price_trigger_eligible,
)
from app.watchlist.lineage import watchlist_row_is_decision_eligible
from app.watchlist import anchor_sanity
from app.watchlist.margin_of_safety import compute_buy_target
from app.watchlist.volatility import realized_volatility
from app.watchlist.schema import ensure_watchlist_schema, resolve_db_path
from app.calibration.decision_ledger import snapshot_decision


# The flat BUY_PRICE_DISCOUNT_TO_ANCHOR=0.25 haircut is gone; the operative
# haircut is the per-name bounded discount in margin_of_safety.compute_buy_target.
# The single neutral-fallback magnitude (0.25 for a range-less WATCHLIST_ONLY
# name, so the ~137 legacy rows do not shift) lives in margin_of_safety.DISCOUNT_CONFIG.
WATCHLIST_POPULATION_VERDICTS = {"ACTIONABLE", "WATCHLIST_ONLY", "DATA_INCOMPLETE"}

logger = logging.getLogger(__name__)

_V2_PROTECTIVE_OR_TERMINAL_STATUSES = {
    "UNCERTAIN",
    "PRICE_DATA_SUSPECT",
    "QUARANTINE",
    "CONTRADICTED",
    "RESOLVED",
    "REMOVED",
}


@dataclass(frozen=True)
class WatchlistPopulationResult:
    added_or_updated: int
    skipped: int
    entry_ids: list[int] = field(default_factory=list)
    skipped_reasons: dict[str, str] = field(default_factory=dict)
    adverse_check: dict | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    from app.db import connect

    ensure_watchlist_schema(db_path)
    return connect(resolve_db_path(db_path))


def _json_list(items: list[str]) -> str:
    return json.dumps([str(item) for item in items], sort_keys=True)


def _read_json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _read_json_object(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid current-event watermark JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Current-event watermark must be a JSON object")
    return dict(parsed)


def _row_to_entry(row: sqlite3.Row) -> WatchlistEntry:
    return WatchlistEntry(
        id=int(row["id"]),
        ticker=str(row["ticker"]),
        status=str(row["status"]),
        conviction_grade=row["conviction_grade"],
        confidence=row["confidence"],
        conviction_source=row["conviction_source"],
        scan_family=row["scan_family"],
        valuation_anchor_method=row["valuation_anchor_method"],
        valuation_anchor_value=row["valuation_anchor_value"],
        buy_price_target=row["buy_price_target"],
        current_price_at_addition=row["current_price_at_addition"],
        thesis_text=row["thesis_text"],
        key_risks=_read_json_list(row["key_risks_json"]),
        falsifiers=_read_json_list(row["falsifiers_json"]),
        open_questions=_read_json_list(row["open_questions_json"]),
        source_run_id=str(row["source_run_id"]),
        source_sector=row["source_sector"],
        added_at=str(row["added_at"]),
        last_evaluated_at=row["last_evaluated_at"],
        current_event_watermark=_read_json_object(row["current_event_watermark_json"]),
        status_reason=row["status_reason"],
        market_cap_mm=row["market_cap_mm"],
        cap_source=row["cap_source"],
        cap_band=row["cap_band"],
        cap_asof=row["cap_asof"],
        pipeline_version=row["pipeline_version"],
        candidate_disposition=row["candidate_disposition"],
        decision_basis=row["decision_basis"],
        selection_validation_status=row["selection_validation_status"],
        event_pending=row["event_pending"],
    )


def watchlist_entry_revision_fingerprint(entry: WatchlistEntry) -> str:
    """Hash the complete normalized current-row state for transactional CAS."""

    payload = json.dumps(
        entry.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _raise_reevaluation_state_conflict(
    *,
    ticker: str,
    expected_revision: str,
    observed_revision: str | None,
) -> None:
    from app.autonomous.financial_integrity import (
        INVALID_FINANCIAL_INPUT,
        FinancialIntegrityGateResult,
        FinancialIntegrityViolation,
        InvalidFinancialInputError,
    )

    violation = FinancialIntegrityViolation(
        code="WATCHLIST_REEVALUATION_STATE_MUTATED",
        ticker=ticker.upper(),
        field="watchlist.current_entry_revision",
        source_values={
            "expected_revision": expected_revision,
            "observed_revision": observed_revision,
        },
        expected_relationship=(
            "the current watchlist row remains byte-semantically identical "
            "between paid-response authorization and publication"
        ),
        observed_relationship=f"{observed_revision or 'MISSING'} != {expected_revision}",
        reason=(
            "The watchlist row changed while reevaluation was in flight; "
            "the stale result cannot update state, history, or artifacts."
        ),
        terminal_status=INVALID_FINANCIAL_INPUT,
    )
    raise InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context=f"watchlist_reevaluation_cas:{ticker.upper()}",
            run_as_of_date=datetime.now(timezone.utc).date().isoformat(),
            status=INVALID_FINANCIAL_INPUT,
            violations=(violation,),
            scope_fingerprint=observed_revision or "",
        )
    )


def _raise_reevaluation_evidence_conflict(
    *,
    ticker: str,
    expected_fingerprint: str,
    observed_fingerprint: str,
) -> None:
    from app.autonomous.financial_integrity import (
        INVALID_FINANCIAL_INPUT,
        FinancialIntegrityGateResult,
        FinancialIntegrityViolation,
        InvalidFinancialInputError,
    )

    violation = FinancialIntegrityViolation(
        code="WATCHLIST_REEVALUATION_EVIDENCE_MUTATED",
        ticker=ticker.upper(),
        field="watchlist.reevaluation_evidence_fingerprint",
        source_values={
            "expected_fingerprint": expected_fingerprint,
            "observed_fingerprint": observed_fingerprint,
        },
        expected_relationship=(
            "the database evidence remains unchanged before the transactional "
            "write and the adapter-backed event snapshot is bound exactly"
        ),
        observed_relationship=f"{observed_fingerprint} != {expected_fingerprint}",
        reason=(
            "Filing or event evidence changed before publication authorization; "
            "the stale result cannot advance the evidence watermark. SQLite "
            "locking does not freeze adapter or filesystem sources."
        ),
        terminal_status=INVALID_FINANCIAL_INPUT,
    )
    raise InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context=f"watchlist_reevaluation_evidence_cas:{ticker.upper()}",
            run_as_of_date=datetime.now(timezone.utc).date().isoformat(),
            status=INVALID_FINANCIAL_INPUT,
            violations=(violation,),
            scope_fingerprint=observed_fingerprint,
        )
    )


def _watchlist_state_value(value: Any, field_name: str) -> Any:
    if isinstance(value, dict):
        return value.get(field_name)
    try:
        return value[field_name]
    except (KeyError, TypeError, IndexError):
        return getattr(value, field_name, None)


def _validate_v2_watchlist_state(value: Any) -> None:
    """Fail closed on every persisted or mutated v2 watchlist state."""

    pipeline = str(_watchlist_state_value(value, "pipeline_version") or "").strip().lower()
    if pipeline != AUTONOMOUS_SECTOR_PIPELINE_V2:
        return

    disposition = str(_watchlist_state_value(value, "candidate_disposition") or "").strip().upper()
    basis = str(_watchlist_state_value(value, "decision_basis") or "").strip().upper()
    source = str(_watchlist_state_value(value, "conviction_source") or "").strip().lower()
    grade = str(_watchlist_state_value(value, "conviction_grade") or "").strip().upper()
    status = str(_watchlist_state_value(value, "status") or "").strip().upper()
    validation = (
        str(_watchlist_state_value(value, "selection_validation_status") or "").strip().upper()
    )

    if not disposition or not basis:
        raise ValueError("v2 watchlist rows require disposition and decision basis")
    if disposition in {"OUT_OF_SCOPE", "SCREENED_OUT"}:
        raise ValueError(f"v2 {disposition} candidates cannot enter the watchlist")

    eligibility_state = {
        "pipeline_version": pipeline,
        "candidate_disposition": disposition,
        "decision_basis": basis,
        "conviction_grade": grade,
        "selection_validation_status": validation,
        "status": status,
    }
    trigger_eligible = is_price_trigger_eligible(eligibility_state)
    if grade == "ACTIONABLE" and not trigger_eligible:
        raise ValueError("v2 ACTIONABLE requires validated underwriting")
    if status in {"DEPLOY_READY", "BUY_CONFIRMED"} and not trigger_eligible:
        raise ValueError("v2 at-target status requires validated underwriting")

    if disposition == "READY_FOR_UNDERWRITING":
        valid = (
            basis == "SCREEN"
            and source == "sector_screen"
            and grade == "WATCHLIST_ONLY"
            and status in ({"ACTIVE"} | _V2_PROTECTIVE_OR_TERMINAL_STATUSES)
        )
        if not valid:
            raise ValueError("v2 READY_FOR_UNDERWRITING provenance is inconsistent")
        return

    if disposition == "NEEDS_DATA":
        valid = (
            basis in {"SCREEN", "UNDERWRITING"}
            and source == ("sector_screen" if basis == "SCREEN" else "company_autonomy")
            and grade == "DATA_INCOMPLETE"
            and status in _V2_PROTECTIVE_OR_TERMINAL_STATUSES
        )
        if not valid:
            raise ValueError("v2 NEEDS_DATA provenance is inconsistent")
        return

    if disposition == "UNDERWRITTEN":
        validated = (
            basis == "VALIDATED_UNDERWRITING"
            and source == "sector_final_decision"
            and grade == "ACTIONABLE"
            and validation == "VALIDATED"
            and status
            in ({"ACTIVE", "DEPLOY_READY", "BUY_CONFIRMED"} | _V2_PROTECTIVE_OR_TERMINAL_STATUSES)
        )
        watchlist_only = (
            basis == "UNDERWRITING"
            and source == "company_autonomy"
            and grade == "WATCHLIST_ONLY"
            and status in ({"ACTIVE"} | _V2_PROTECTIVE_OR_TERMINAL_STATUSES)
        )
        if not (validated or watchlist_only):
            raise ValueError("v2 UNDERWRITTEN provenance is inconsistent")
        return

    raise ValueError(f"Unsupported candidate disposition: {disposition}")


def _validate_v2_watchlist_mutation(
    row: sqlite3.Row,
    *,
    status: str | None = None,
    conviction_grade: str | None = None,
) -> None:
    state = dict(row)
    if status is not None:
        state["status"] = status
    if conviction_grade is not None:
        state["conviction_grade"] = conviction_grade
    _validate_v2_watchlist_state(state)


def _entry_values(entry: WatchlistEntry) -> dict[str, Any]:
    normalized = entry.normalized()
    if normalized.status not in WATCHLIST_STATUSES:
        raise ValueError(f"Unsupported watchlist status: {normalized.status}")
    if (
        normalized.conviction_grade
        and normalized.conviction_grade not in WATCHLIST_CONVICTION_GRADES
    ):
        raise ValueError(f"Unsupported watchlist conviction grade: {normalized.conviction_grade}")
    if normalized.confidence and normalized.confidence not in WATCHLIST_CONFIDENCE_LABELS:
        raise ValueError(f"Unsupported watchlist confidence: {normalized.confidence}")
    if (
        normalized.conviction_source
        and normalized.conviction_source not in WATCHLIST_CONVICTION_SOURCES
    ):
        raise ValueError(f"Unsupported watchlist conviction source: {normalized.conviction_source}")
    if normalized.scan_family not in WATCHLIST_SCAN_FAMILIES:
        raise ValueError(f"Unsupported watchlist scan family: {normalized.scan_family}")
    if (
        normalized.candidate_disposition
        and normalized.candidate_disposition not in WATCHLIST_CANDIDATE_DISPOSITIONS
    ):
        raise ValueError(f"Unsupported candidate disposition: {normalized.candidate_disposition}")
    if normalized.decision_basis and normalized.decision_basis not in WATCHLIST_DECISION_BASES:
        raise ValueError(f"Unsupported decision basis: {normalized.decision_basis}")
    _validate_v2_watchlist_state(normalized)
    return {
        "ticker": normalized.ticker,
        "status": normalized.status,
        "conviction_grade": normalized.conviction_grade,
        "confidence": normalized.confidence,
        "conviction_source": normalized.conviction_source,
        "scan_family": normalized.scan_family,
        "valuation_anchor_method": normalized.valuation_anchor_method,
        "valuation_anchor_value": normalized.valuation_anchor_value,
        "buy_price_target": normalized.buy_price_target,
        "current_price_at_addition": normalized.current_price_at_addition,
        "thesis_text": normalized.thesis_text,
        "key_risks_json": _json_list(normalized.key_risks),
        "falsifiers_json": _json_list(normalized.falsifiers),
        "open_questions_json": _json_list(normalized.open_questions),
        "source_run_id": normalized.source_run_id,
        "source_sector": normalized.source_sector,
        "added_at": normalized.added_at,
        "last_evaluated_at": normalized.last_evaluated_at,
        "current_event_watermark_json": (
            json.dumps(
                normalized.current_event_watermark,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            if normalized.current_event_watermark is not None
            else None
        ),
        "status_reason": normalized.status_reason,
        "market_cap_mm": normalized.market_cap_mm,
        "cap_source": normalized.cap_source,
        "cap_band": normalized.cap_band,
        "cap_asof": normalized.cap_asof,
        "pipeline_version": normalized.pipeline_version,
        "candidate_disposition": normalized.candidate_disposition,
        "decision_basis": normalized.decision_basis,
        "selection_validation_status": normalized.selection_validation_status,
    }


def _history_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _insert_history(
    conn: sqlite3.Connection,
    *,
    watchlist_id: int,
    field_name: str,
    old_value: Any,
    new_value: Any,
    source: str,
    source_run_id: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO watchlist_history (
            watchlist_id, changed_at, field_name, old_value, new_value, source, source_run_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            watchlist_id,
            _utc_now_iso(),
            field_name,
            _history_value(old_value),
            _history_value(new_value),
            source,
            source_run_id,
        ),
    )


def add_or_update(
    entry: WatchlistEntry,
    db_path: str | Path | None = None,
    *,
    history_source: str = "sector_run",
) -> int:
    values = _entry_values(entry)
    conn = _connect(db_path)
    try:
        existing = conn.execute(
            "SELECT * FROM watchlist WHERE ticker = ? AND source_run_id = ?",
            (values["ticker"], values["source_run_id"]),
        ).fetchone()
        if existing is None:
            columns = ", ".join(values)
            placeholders = ", ".join("?" for _field in values)
            cursor = conn.execute(
                f"INSERT INTO watchlist ({columns}) VALUES ({placeholders})",
                tuple(values.values()),
            )
            row_id = int(cursor.lastrowid)
            _insert_history(
                conn,
                watchlist_id=row_id,
                field_name="created",
                old_value=None,
                new_value=values["status"],
                source=history_source,
                source_run_id=values["source_run_id"],
            )
            conn.commit()
            return row_id

        row_id = int(existing["id"])
        # status and status_reason are owned by the trigger / reevaluation path.
        # Re-populating an existing (ticker, source_run_id) row must NOT overwrite
        # them, otherwise a trigger-set DEPLOY_READY or PRICE_DATA_SUSPECT can be
        # reverted to ACTIVE by a recompute against the stale current_price_at_addition.
        _population_protected_fields = {
            "ticker",
            "source_run_id",
            "added_at",
            "status",
            "status_reason",
            "current_event_watermark_json",
        }
        changed_fields = [
            field_name
            for field_name, new_value in values.items()
            if field_name not in _population_protected_fields and existing[field_name] != new_value
        ]
        if not changed_fields:
            return row_id

        assignments = ", ".join(f"{field_name} = ?" for field_name in changed_fields)
        conn.execute(
            f"UPDATE watchlist SET {assignments} WHERE id = ?",
            tuple(values[field_name] for field_name in changed_fields) + (row_id,),
        )
        for field_name in changed_fields:
            _insert_history(
                conn,
                watchlist_id=row_id,
                field_name=field_name.replace("_json", ""),
                old_value=existing[field_name],
                new_value=values[field_name],
                source=history_source,
                source_run_id=values["source_run_id"],
            )
        conn.commit()
        return row_id
    finally:
        conn.close()


# Current-row selection policy (2026-07-21).
#
# A ticker can accumulate multiple watchlist rows across re-analysis runs
# (UNIQUE is per (ticker, source_run_id), not per ticker). Every "current
# view" (show/list/queue/digest/stats) must collapse those rows to ONE
# current row per ticker. After the existing scan-family scope and sector
# pipeline-version boundary are applied, MAX(id) is authoritative. A newer
# weaker, ACTIVE, AVOID, or REMOVED assessment therefore supersedes older
# optimism. Historical rows stay append-only and queryable. Status/trigger
# mutations target this same authoritative row.


def current_watchlist_cte(latest_where: str = "") -> str:
    """Body of the ``latest_watchlist(ticker, latest_id)`` CTE.

    ``latest_where`` is an optional ``WHERE ...`` clause (e.g. scan-family or
    ticker scoping) applied to the candidate rows before selection; its bind
    parameters come first in the enclosing query's parameter list.
    """
    return f"""
        SELECT ticker, MAX(id) AS latest_id
        FROM (
            SELECT
                versioned.*,
                MAX(
                    CASE WHEN id = latest_sector_id
                         THEN LOWER(COALESCE(pipeline_version, 'v1')) END
                ) OVER (PARTITION BY ticker) AS latest_sector_pipeline
            FROM (
                SELECT
                    scoped.*,
                    MAX(
                        CASE WHEN source_run_id LIKE 'autonomous_sector_%'
                             THEN id END
                    ) OVER (PARTITION BY ticker) AS latest_sector_id
                FROM watchlist AS scoped
                {latest_where}
            ) AS versioned
        ) AS watchlist
        WHERE NOT (
            source_run_id LIKE 'autonomous_sector_%'
            AND LOWER(COALESCE(pipeline_version, 'v1'))
                != latest_sector_pipeline
        )
        GROUP BY ticker
    """


def _current_row_for_ticker(conn: sqlite3.Connection, ticker: str) -> sqlite3.Row | None:
    row = conn.execute(
        f"""
        WITH latest_watchlist AS ({current_watchlist_cte("WHERE ticker = ?")})
        SELECT w.*
        FROM watchlist w
        JOIN latest_watchlist ON latest_watchlist.latest_id = w.id
        """,
        (ticker.upper(),),
    ).fetchone()
    if row is None:
        return None
    # Apply invalidation only after authoritative MAX(id) selection.  Doing it
    # inside the CTE would silently resurrect an older optimistic assessment.
    return row if watchlist_row_is_decision_eligible(row) else None


def list_active(
    *,
    status: str | None = None,
    sector: str | None = None,
    scan_family: str | None = None,
    db_path: str | Path | None = None,
) -> list[WatchlistEntry]:
    normalized_scan_family = scan_family.lower() if scan_family else None
    if normalized_scan_family and normalized_scan_family not in WATCHLIST_SCAN_FAMILIES:
        raise ValueError(f"Unsupported watchlist scan family: {scan_family}")
    if not financial_integrity_manifest_is_usable():
        return []
    conn = _connect(db_path)
    try:
        normalized_status = status.upper() if status else None
        conditions = ["w.status = ?"] if normalized_status else ["w.status != 'REMOVED'"]
        params: list[Any] = [normalized_status] if normalized_status else []
        if sector:
            conditions.append("w.source_sector = ?")
            params.append(sector)
        where_clause = " AND ".join(conditions)
        latest_conditions: list[str] = []
        latest_params: list[Any] = []
        if normalized_scan_family:
            latest_conditions.append("scan_family = ?")
            latest_params.append(normalized_scan_family)
        latest_where = f"WHERE {' AND '.join(latest_conditions)}" if latest_conditions else ""
        rows = conn.execute(
            f"""
            WITH latest_watchlist AS ({current_watchlist_cte(latest_where)})
            SELECT w.*
            FROM watchlist w
            JOIN latest_watchlist ON latest_watchlist.latest_id = w.id
            WHERE {where_clause}
            ORDER BY w.status, w.ticker
            """,
            latest_params + params,
        ).fetchall()
        rows = [row for row in rows if watchlist_row_is_decision_eligible(row)]
        return [_row_to_entry(row) for row in rows]
    finally:
        conn.close()


def list_current_for_market_data(
    *,
    db_path: str | Path | None = None,
) -> list[WatchlistEntry]:
    """Return one authoritative non-removed row per ticker without decision gates.

    This selector exists only for factual market-data maintenance. Decision and
    presentation consumers must continue to use ``list_active`` or another
    authorization-aware read model.
    """

    conn = _connect(db_path)
    try:
        rows = conn.execute(
            f"""
            WITH latest_watchlist AS ({current_watchlist_cte()})
            SELECT w.*
            FROM watchlist w
            JOIN latest_watchlist ON latest_watchlist.latest_id = w.id
            WHERE w.status != 'REMOVED'
            ORDER BY w.ticker
            """
        ).fetchall()
        return [_row_to_entry(row) for row in rows]
    finally:
        conn.close()


def get_latest(ticker: str, db_path: str | Path | None = None) -> WatchlistEntry | None:
    """Return the ticker's current row per the selection policy above."""
    if not financial_integrity_manifest_is_usable():
        return None
    conn = _connect(db_path)
    try:
        row = _current_row_for_ticker(conn, ticker)
        return _row_to_entry(row) if row else None
    finally:
        conn.close()


def get_history(ticker: str, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT h.*
            FROM watchlist_history h
            JOIN watchlist w ON w.id = h.watchlist_id
            WHERE w.ticker = ?
            ORDER BY h.id
            """,
            (ticker.upper(),),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def add_price_snapshot(
    watchlist_id: int,
    *,
    price: float,
    checked_at: str | None = None,
    source: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO watchlist_price_snapshots (watchlist_id, price, checked_at, source)
            VALUES (?, ?, ?, ?)
            """,
            (int(watchlist_id), float(price), checked_at or _utc_now_iso(), source),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def get_latest_price(watchlist_id: int, db_path: str | Path | None = None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT id, watchlist_id, price, checked_at, source
            FROM watchlist_price_snapshots
            WHERE watchlist_id = ?
            ORDER BY checked_at DESC, id DESC
            LIMIT 1
            """,
            (int(watchlist_id),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def record_trigger_status_change(
    ticker: str,
    *,
    status: str,
    reason: str,
    db_path: str | Path | None = None,
) -> None:
    normalized_status = status.upper()
    if normalized_status not in WATCHLIST_STATUSES:
        raise ValueError(f"Unsupported watchlist status: {status}")
    conn = _connect(db_path)
    try:
        row = _current_row_for_ticker(conn, ticker)
        if row is None:
            raise ValueError(f"No watchlist entry found for {ticker.upper()}")
        row_id = int(row["id"])
        _validate_v2_watchlist_mutation(row, status=normalized_status)
        conn.execute(
            """
            UPDATE watchlist
            SET status = ?, status_reason = ?
            WHERE id = ?
            """,
            (normalized_status, reason, row_id),
        )
        if row["status"] != normalized_status:
            _insert_history(
                conn,
                watchlist_id=row_id,
                field_name="status",
                old_value=row["status"],
                new_value=normalized_status,
                source="trigger",
                source_run_id=None,
            )
        if row["status_reason"] != reason:
            _insert_history(
                conn,
                watchlist_id=row_id,
                field_name="status_reason",
                old_value=row["status_reason"],
                new_value=reason,
                source="trigger",
                source_run_id=None,
            )
        conn.commit()
    finally:
        conn.close()


def mark_status(
    ticker: str,
    status: str,
    reason: str,
    source: str,
    source_run_id: str | None = None,
    db_path: str | Path | None = None,
) -> None:
    normalized_status = status.upper()
    if normalized_status not in WATCHLIST_STATUSES:
        raise ValueError(f"Unsupported watchlist status: {status}")
    conn = _connect(db_path)
    try:
        row = _current_row_for_ticker(conn, ticker)
        if row is None:
            raise ValueError(f"No watchlist entry found for {ticker.upper()}")
        row_id = int(row["id"])
        _validate_v2_watchlist_mutation(row, status=normalized_status)
        changed_at = _utc_now_iso()
        conn.execute(
            """
            UPDATE watchlist
            SET status = ?, status_reason = ?, last_evaluated_at = ?
            WHERE id = ?
            """,
            (normalized_status, reason, changed_at, row_id),
        )
        if row["status"] != normalized_status:
            _insert_history(
                conn,
                watchlist_id=row_id,
                field_name="status",
                old_value=row["status"],
                new_value=normalized_status,
                source=source,
                source_run_id=source_run_id,
            )
        if row["status_reason"] != reason:
            _insert_history(
                conn,
                watchlist_id=row_id,
                field_name="status_reason",
                old_value=row["status_reason"],
                new_value=reason,
                source=source,
                source_run_id=source_run_id,
            )
        conn.commit()
    finally:
        conn.close()


def record_reevaluation_result(
    ticker: str,
    *,
    status: str,
    reason: str,
    evaluation: str,
    source_run_id: str | None,
    conviction_grade: str | None = None,
    expected_entry_revision: str | None = None,
    expected_evidence_fingerprint: str | None = None,
    evidence_since: str | None = None,
    evidence_as_of_date: str | None = None,
    include_current_events: bool = True,
    use_current_event_watermark: bool = True,
    publication_artifact: dict[str, Any] | None = None,
    db_path: str | Path | None = None,
) -> None:
    normalized_status = status.upper()
    if normalized_status not in WATCHLIST_STATUSES:
        raise ValueError(f"Unsupported watchlist status: {status}")
    normalized_grade = conviction_grade.upper() if conviction_grade else None
    if normalized_grade and normalized_grade not in WATCHLIST_CONVICTION_GRADES:
        raise ValueError(f"Unsupported watchlist conviction grade: {conviction_grade}")
    if expected_evidence_fingerprint is not None and (
        evidence_since is None or evidence_as_of_date is None
    ):
        raise ValueError("Evidence CAS requires since and as-of boundaries")
    if publication_artifact is not None and (
        not source_run_id
        or expected_entry_revision is None
        or expected_evidence_fingerprint is None
    ):
        raise ValueError(
            "Reevaluation publication requires run id, entry revision, and evidence fingerprint"
        )
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = _current_row_for_ticker(conn, ticker)
        if row is None:
            if expected_entry_revision is not None:
                _raise_reevaluation_state_conflict(
                    ticker=ticker,
                    expected_revision=expected_entry_revision,
                    observed_revision=None,
                )
            raise ValueError(f"No watchlist entry found for {ticker.upper()}")
        if expected_entry_revision is not None:
            row_entry = _row_to_entry(row)
            observed_revision = watchlist_entry_revision_fingerprint(row_entry)
            if observed_revision != expected_entry_revision:
                _raise_reevaluation_state_conflict(
                    ticker=ticker,
                    expected_revision=expected_entry_revision,
                    observed_revision=observed_revision,
                )
        else:
            row_entry = _row_to_entry(row)
        publication_current_event_watermark = row_entry.current_event_watermark
        if expected_evidence_fingerprint is not None:
            from app.watchlist.reevaluation import (
                detect_new_evidence_snapshot,
                reevaluation_evidence_fingerprint,
            )

            evidence_watermark = (
                row_entry.current_event_watermark if use_current_event_watermark else None
            )
            current_snapshot = detect_new_evidence_snapshot(
                ticker,
                since=str(evidence_since),
                as_of_date=str(evidence_as_of_date),
                db_path=db_path,
                include_current_events=include_current_events,
                current_event_watermark=evidence_watermark,
                _conn=conn,
            )
            current_evidence = current_snapshot.evidence
            observed_evidence_fingerprint = reevaluation_evidence_fingerprint(
                current_evidence,
                ticker=ticker,
                since=str(evidence_since),
                as_of_date=str(evidence_as_of_date),
                include_current_events=include_current_events,
                current_event_watermark=evidence_watermark,
                proposed_current_event_watermark=current_snapshot.current_event_watermark,
            )
            if observed_evidence_fingerprint != expected_evidence_fingerprint:
                _raise_reevaluation_evidence_conflict(
                    ticker=ticker,
                    expected_fingerprint=expected_evidence_fingerprint,
                    observed_fingerprint=observed_evidence_fingerprint,
                )
            if include_current_events:
                publication_current_event_watermark = current_snapshot.current_event_watermark
        row_id = int(row["id"])
        _validate_v2_watchlist_mutation(
            row,
            status=normalized_status,
            conviction_grade=normalized_grade or row["conviction_grade"],
        )
        changed_at = _utc_now_iso()
        conn.execute(
            """
            UPDATE watchlist
            SET status = ?, status_reason = ?, last_evaluated_at = ?,
                current_event_watermark_json = ?
            WHERE id = ?
            """,
            (
                normalized_status,
                reason,
                changed_at,
                (
                    json.dumps(
                        publication_current_event_watermark,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    if publication_current_event_watermark is not None
                    else None
                ),
                row_id,
            ),
        )
        # The conviction GRADE is orthogonal to the price STATUS. A re-evaluation
        # may promote a resolved DATA_INCOMPLETE row's grade (e.g. to
        # WATCHLIST_ONLY) without disturbing the status ladder.
        if normalized_grade and row["conviction_grade"] != normalized_grade:
            conn.execute(
                "UPDATE watchlist SET conviction_grade = ? WHERE id = ?",
                (normalized_grade, row_id),
            )
            _insert_history(
                conn,
                watchlist_id=row_id,
                field_name="conviction_grade",
                old_value=row["conviction_grade"],
                new_value=normalized_grade,
                source="reevaluation",
                source_run_id=source_run_id,
            )
        if row["status"] != normalized_status:
            _insert_history(
                conn,
                watchlist_id=row_id,
                field_name="status",
                old_value=row["status"],
                new_value=normalized_status,
                source="reevaluation",
                source_run_id=source_run_id,
            )
        if row["status_reason"] != reason:
            _insert_history(
                conn,
                watchlist_id=row_id,
                field_name="status_reason",
                old_value=row["status_reason"],
                new_value=reason,
                source="reevaluation",
                source_run_id=source_run_id,
            )
        _insert_history(
            conn,
            watchlist_id=row_id,
            field_name="reevaluation",
            old_value=None,
            new_value=evaluation,
            source="reevaluation",
            source_run_id=source_run_id,
        )
        if publication_artifact is not None:
            _insert_reevaluation_publication(
                conn,
                run_id=str(source_run_id),
                watchlist_id=row_id,
                ticker=str(ticker).strip().upper(),
                evaluation=evaluation,
                state_applied=True,
                entry_revision=str(expected_entry_revision),
                evidence_fingerprint=str(expected_evidence_fingerprint),
                artifact=publication_artifact,
            )
        conn.commit()
    finally:
        conn.close()


def _insert_reevaluation_publication(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    watchlist_id: int,
    ticker: str,
    evaluation: str,
    state_applied: bool,
    entry_revision: str,
    evidence_fingerprint: str,
    artifact: dict[str, Any],
) -> None:
    artifact_json = json.dumps(
        artifact,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    artifact_sha256 = sha256(artifact_json.encode("utf-8")).hexdigest()
    conn.execute(
        """
        INSERT INTO watchlist_reevaluation_publications (
            run_id, watchlist_id, ticker, evaluation, state_applied,
            entry_revision, evidence_fingerprint, artifact_json,
            artifact_sha256, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            int(watchlist_id),
            ticker,
            evaluation,
            1 if state_applied else 0,
            entry_revision,
            evidence_fingerprint,
            artifact_json,
            artifact_sha256,
            _utc_now_iso(),
        ),
    )


def record_reevaluation_attempt_artifact(
    ticker: str,
    *,
    source_run_id: str,
    evaluation: str,
    expected_entry_revision: str,
    evidence_fingerprint: str,
    publication_artifact: dict[str, Any],
    db_path: str | Path | None = None,
) -> None:
    """Durably record a non-state-applying reevaluation attempt.

    Failed and budget-blocked substantive work remains auditable without
    consuming the entry's evidence/event watermark or changing its status.
    """

    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = _current_row_for_ticker(conn, ticker)
        if row is None:
            _raise_reevaluation_state_conflict(
                ticker=ticker,
                expected_revision=expected_entry_revision,
                observed_revision=None,
            )
        row_entry = _row_to_entry(row)
        observed_revision = watchlist_entry_revision_fingerprint(row_entry)
        if observed_revision != expected_entry_revision:
            _raise_reevaluation_state_conflict(
                ticker=ticker,
                expected_revision=expected_entry_revision,
                observed_revision=observed_revision,
            )
        _insert_reevaluation_publication(
            conn,
            run_id=source_run_id,
            watchlist_id=int(row["id"]),
            ticker=str(ticker).strip().upper(),
            evaluation=evaluation,
            state_applied=False,
            entry_revision=expected_entry_revision,
            evidence_fingerprint=evidence_fingerprint,
            artifact=publication_artifact,
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def remove(ticker: str, reason: str, db_path: str | Path | None = None) -> None:
    mark_status(ticker, "REMOVED", reason, "manual", None, db_path)


def _latest_watchlist_rows(
    conn: sqlite3.Connection,
    *,
    include_removed: bool = False,
    sector: str | None = None,
    scan_family: str | None = None,
) -> list[sqlite3.Row]:
    normalized_scan_family = scan_family.lower() if scan_family else None
    if normalized_scan_family and normalized_scan_family not in WATCHLIST_SCAN_FAMILIES:
        raise ValueError(f"Unsupported watchlist scan family: {scan_family}")
    if not financial_integrity_manifest_is_usable():
        return []
    conditions: list[str] = []
    params: list[Any] = []
    if not include_removed:
        conditions.append("w.status != 'REMOVED'")
    if sector:
        conditions.append("w.source_sector = ?")
        params.append(sector)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    latest_conditions: list[str] = []
    latest_params: list[Any] = []
    if normalized_scan_family:
        latest_conditions.append("scan_family = ?")
        latest_params.append(normalized_scan_family)
    latest_where = f"WHERE {' AND '.join(latest_conditions)}" if latest_conditions else ""
    rows = conn.execute(
        f"""
        WITH latest_watchlist AS ({current_watchlist_cte(latest_where)})
        SELECT w.*
        FROM watchlist w
        JOIN latest_watchlist ON latest_watchlist.latest_id = w.id
        {where_clause}
        ORDER BY w.ticker
        """,
        latest_params + params,
    ).fetchall()
    return [row for row in rows if watchlist_row_is_decision_eligible(row)]


def stats(db_path: str | Path | None = None, *, all_rows: bool = False) -> dict[str, Any]:
    if not all_rows and not financial_integrity_manifest_is_usable():
        return {
            "mode": "current_unique",
            "total_entries": 0,
            "all_rows_total": 0,
            "unique_tickers": 0,
            "duplicate_rows": 0,
            "status_counts": {},
            "sector_counts": {},
            "oldest_entry": None,
            "newest_entry": None,
        }
    conn = _connect(db_path)
    try:
        duplicate_row = conn.execute(
            """
            SELECT COUNT(*) AS all_nonremoved_rows, COUNT(DISTINCT ticker) AS unique_tickers
            FROM watchlist
            WHERE status != 'REMOVED'
            """
        ).fetchone()
        all_nonremoved_rows = int(duplicate_row["all_nonremoved_rows"] or 0)
        unique_tickers = int(duplicate_row["unique_tickers"] or 0)
        duplicate_rows = max(0, all_nonremoved_rows - unique_tickers)

        if not all_rows:
            rows = _latest_watchlist_rows(conn, include_removed=False)
            status_counts: dict[str, int] = {}
            sector_counts: dict[str, int] = {}
            for row in rows:
                status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
                sector = str(row["source_sector"] or "UNKNOWN")
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            oldest = min((row["added_at"] for row in rows if row["added_at"]), default=None)
            newest = max((row["added_at"] for row in rows if row["added_at"]), default=None)
            return {
                "mode": "current_unique",
                "total_entries": len(rows),
                "all_rows_total": all_nonremoved_rows,
                "unique_tickers": len(rows),
                "duplicate_rows": duplicate_rows,
                "status_counts": dict(sorted(status_counts.items())),
                "sector_counts": dict(sorted(sector_counts.items())),
                "oldest_entry": oldest,
                "newest_entry": newest,
            }

        status_counts = {
            str(row["status"]): int(row["count"])
            for row in conn.execute(
                "SELECT status, COUNT(*) AS count FROM watchlist GROUP BY status ORDER BY status"
            ).fetchall()
        }
        sector_counts = {
            str(row["source_sector"] or "UNKNOWN"): int(row["count"])
            for row in conn.execute(
                """
                SELECT source_sector, COUNT(*) AS count
                FROM watchlist
                GROUP BY source_sector
                ORDER BY source_sector
                """
            ).fetchall()
        }
        row = conn.execute(
            "SELECT COUNT(*) AS total, MIN(added_at) AS oldest, MAX(added_at) AS newest FROM watchlist"
        ).fetchone()
        return {
            "mode": "all_rows",
            "total_entries": int(row["total"] or 0),
            "all_rows_total": all_nonremoved_rows,
            "unique_tickers": unique_tickers,
            "duplicate_rows": duplicate_rows,
            "status_counts": status_counts,
            "sector_counts": sector_counts,
            "oldest_entry": row["oldest"],
            "newest_entry": row["newest"],
        }
    finally:
        conn.close()


PRICE_BASIS_SNAPSHOT = "SNAPSHOT"
PRICE_BASIS_ADDITION = "PRICE_AT_ADDITION"


def _display_price_for_queue(row: sqlite3.Row) -> tuple[float | None, str | None]:
    """The price a queue row shows, and which kind of price it is.

    A price snapshot is an observation with a timestamp. The price recorded when the name
    was added is the price on THAT day, possibly months old; before it was labelled it
    stood in as the "latest" price with nothing to tell the two apart, so a stale figure
    fed the distance-from-buy sort and the today-page waterline as if it were live.
    The fallback is kept (investor buy-now renders it with an explicit no-snapshot
    caveat) but every row now says which one it is: ``latest_price_basis``.
    """
    latest_price = row["latest_price"]
    if latest_price is not None:
        return float(latest_price), PRICE_BASIS_SNAPSHOT
    addition_price = row["current_price_at_addition"]
    if addition_price is not None:
        return float(addition_price), PRICE_BASIS_ADDITION
    return None, None


def watchlist_queue(
    *,
    limit: int = 25,
    sector: str | None = None,
    include_price_suspect: bool = False,
    band: str | None = None,
    scan_family: str | None = None,
    db_path: str | Path | None = None,
    conn: sqlite3.Connection | None = None,
    manifest_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Ranked review queue with presented statuses.

    When ``conn`` is provided the caller owns the connection (it is neither
    schema-ensured nor closed here) — this is how the read-only web layer
    shares the exact presentation derivation without a write path.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    normalized_scan_family = scan_family.lower() if scan_family else None
    if normalized_scan_family and normalized_scan_family not in WATCHLIST_SCAN_FAMILIES:
        raise ValueError(f"Unsupported watchlist scan family: {scan_family}")

    band_bounds: tuple[float | None, float | None] | None = None
    if band:
        from app.autonomous.sector_candidates import MARKET_CAP_FOCUS_TIERS

        band_key = str(band).strip().lower()
        if band_key not in MARKET_CAP_FOCUS_TIERS:
            raise ValueError(f"Unknown cap band: {band}")
        band_bounds = MARKET_CAP_FOCUS_TIERS[band_key]

    if not financial_integrity_manifest_is_usable(manifest_path):
        return []

    owns_conn = conn is None
    if conn is None:
        conn = _connect(db_path)
    try:
        conditions = ["w.status != 'REMOVED'"]
        params: list[Any] = []
        if not include_price_suspect:
            conditions.append("w.status != 'PRICE_DATA_SUSPECT'")
        if sector:
            conditions.append("w.source_sector = ?")
            params.append(sector)
        where_clause = " AND ".join(conditions)
        latest_conditions: list[str] = []
        latest_params: list[Any] = []
        if normalized_scan_family:
            latest_conditions.append("scan_family = ?")
            latest_params.append(normalized_scan_family)
        latest_where = f"WHERE {' AND '.join(latest_conditions)}" if latest_conditions else ""
        rows = conn.execute(
            f"""
            WITH latest_watchlist AS ({current_watchlist_cte(latest_where)}),
            latest_snapshot AS (
                SELECT id, watchlist_id, price, checked_at, source
                FROM (
                    SELECT
                        s.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY s.watchlist_id
                            ORDER BY s.checked_at DESC, s.id DESC
                        ) AS snapshot_rank
                    FROM watchlist_price_snapshots s
                )
                WHERE snapshot_rank = 1
            )
            SELECT
                w.*,
                latest_snapshot.price AS latest_price,
                latest_snapshot.checked_at AS latest_price_checked_at,
                latest_snapshot.source AS latest_price_source
            FROM watchlist w
            JOIN latest_watchlist ON latest_watchlist.latest_id = w.id
            LEFT JOIN latest_snapshot ON latest_snapshot.watchlist_id = w.id
            WHERE {where_clause}
            """,
            latest_params + params,
        ).fetchall()
        rows = [row for row in rows if watchlist_row_is_decision_eligible(row, manifest_path)]

        from app.events.cheapness import cheapness_headline, latest_cheapness_by_ticker

        cheapness_reports = latest_cheapness_by_ticker(conn)

        queue_rows: list[dict[str, Any]] = []
        for row in rows:
            market_cap_mm = row["market_cap_mm"]
            if band_bounds is not None:
                # Band integrity: a row may only appear in a band-scoped queue
                # when its cap is computable AND inside the band. UNKNOWN_CAP
                # rows are never presentable as in-band output.
                cap_min, cap_max = band_bounds
                if market_cap_mm is None:
                    continue
                if cap_min is not None and float(market_cap_mm) < float(cap_min):
                    continue
                if cap_max is not None and float(market_cap_mm) >= float(cap_max):
                    continue
            price, price_basis = _display_price_for_queue(row)
            buy_target = (
                float(row["buy_price_target"]) if row["buy_price_target"] is not None else None
            )
            distance = None
            if price is not None and buy_target and buy_target > 0:
                distance = ((price - buy_target) / buy_target) * 100.0
            event_pending = row["event_pending"]
            # An open EVENT_PENDING flag blocks DEPLOY_READY presentation:
            # the row renders as EVENT_PENDING until the analyst disposes the
            # event (ivi events dispose). The stored status is untouched.
            presented_status = row["status"]
            trigger_eligible = is_price_trigger_eligible(row)
            if not trigger_eligible and str(row["status"]) in {"DEPLOY_READY", "BUY_CONFIRMED"}:
                presented_status = (
                    "UNCERTAIN"
                    if str(row["conviction_grade"] or "").upper() == "DATA_INCOMPLETE"
                    else "ACTIVE"
                )
            if (
                trigger_eligible
                and event_pending
                and str(row["status"]) in {"DEPLOY_READY", "BUY_CONFIRMED"}
            ):
                presented_status = "EVENT_PENDING"
            queue_rows.append(
                {
                    "id": int(row["id"]),
                    "ticker": str(row["ticker"]),
                    "status": row["status"],
                    "presented_status": presented_status,
                    "event_pending": event_pending,
                    "conviction_grade": row["conviction_grade"],
                    "confidence": row["confidence"],
                    "conviction_source": row["conviction_source"],
                    "pipeline_version": row["pipeline_version"],
                    "candidate_disposition": row["candidate_disposition"],
                    "decision_basis": row["decision_basis"],
                    "selection_validation_status": row["selection_validation_status"],
                    "price_trigger_eligible": trigger_eligible,
                    "scan_family": row["scan_family"],
                    "latest_price": price,
                    "latest_price_basis": price_basis,
                    "price_at_addition": (
                        float(row["current_price_at_addition"])
                        if row["current_price_at_addition"] is not None
                        else None
                    ),
                    "latest_price_source": row["latest_price_source"],
                    "latest_price_checked_at": row["latest_price_checked_at"],
                    "buy_price_target": buy_target,
                    "distance_from_buy_pct": distance,
                    "valuation_anchor_method": row["valuation_anchor_method"],
                    "valuation_anchor_value": (
                        float(row["valuation_anchor_value"])
                        if row["valuation_anchor_value"] is not None
                        else None
                    ),
                    "source_sector": row["source_sector"],
                    "status_reason": row["status_reason"],
                    "added_at": row["added_at"],
                    "last_evaluated_at": row["last_evaluated_at"],
                    "falsifiers": _read_json_list(row["falsifiers_json"]),
                    "market_cap_mm": float(market_cap_mm) if market_cap_mm is not None else None,
                    "cap_source": row["cap_source"],
                    "cap_band": row["cap_band"],
                    "cap_band_label": row["cap_band"] or "UNKNOWN_CAP",
                    "adv_dollar_20d": row["adv_dollar_20d"],
                    "adv_dollar_60d": row["adv_dollar_60d"],
                    "adv_asof": row["adv_asof"],
                    "capacity_class": row["capacity_class"] or "ADV_UNKNOWN",
                    "cheapness": cheapness_headline(
                        cheapness_reports.get(str(row["ticker"]).upper())
                    ),
                }
            )

        conviction_priority = {
            "ACTIONABLE": 0,
            "WATCHLIST_ONLY": 1,
            "DATA_INCOMPLETE": 2,
            "AVOID": 3,
        }
        basis_priority = {
            "VALIDATED_UNDERWRITING": 0,
            "UNDERWRITING": 1,
            "SCREEN": 2,
        }
        # EVENT_PENDING ranks with ACTIVE: a flagged name needs an analyst
        # pass, never the at-target slot.
        status_priority = {
            "BUY_CONFIRMED": 0,
            "DEPLOY_READY": 0,
            "EVENT_PENDING": 1,
            "ACTIVE": 1,
            "UNCERTAIN": 2,
            "PRICE_DATA_SUSPECT": 3,
        }
        confidence_priority = {"HIGH": 0, "MODERATE": 1, "LOW": 2}

        def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
            distance = item["distance_from_buy_pct"]
            return (
                conviction_priority.get(str(item.get("conviction_grade") or ""), 9),
                basis_priority.get(str(item.get("decision_basis") or ""), 0),
                status_priority.get(str(item.get("presented_status") or ""), 9),
                confidence_priority.get(str(item.get("confidence") or ""), 9),
                distance if distance is not None else float("inf"),
                str(item.get("ticker") or ""),
            )

        return sorted(queue_rows, key=sort_key)[:limit]
    finally:
        if owns_conn:
            conn.close()


def _valuation_anchor(packet: SectorCompanyFinancialPacket) -> float | None:
    value = packet.valuation.get("valuation_anchor") if isinstance(packet.valuation, dict) else None
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def _anchor_method(packet: SectorCompanyFinancialPacket) -> str | None:
    value = packet.valuation.get("anchor_method") if isinstance(packet.valuation, dict) else None
    return str(value) if value else None


def _intrinsic_bound(packet: SectorCompanyFinancialPacket, key: str) -> float | None:
    valuation = packet.valuation if isinstance(packet.valuation, dict) else {}
    value = valuation.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _explicit_buy_price_target(packet: SectorCompanyFinancialPacket) -> float | None:
    """The buy target the model wrote by hand, if it wrote one.

    Separated out because it is the one number in the row that no check had
    ever seen: the add-time sanity band is applied to the valuation anchor, and
    this field bypassed it entirely on its way to becoming the price the
    DEPLOY_READY trigger fires on.
    """
    valuation = packet.valuation if isinstance(packet.valuation, dict) else {}
    for key in ("buy_below_price", "buy_price_target", "buy_below"):
        value = valuation.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        ):
            return float(value)
    return None


def _buy_price_target(
    packet: SectorCompanyFinancialPacket,
    *,
    conviction_grade: str | None = None,
    confidence: str | None = None,
    db_path: str | Path | None = None,
) -> float | None:
    # An explicit LLM-supplied target is used only when it is at least as strict
    # as the computed margin-of-safety target: a buy target above that is a "buy"
    # with less discount to intrinsic value than the name's grade requires (and
    # above the anchor, a "buy now" on a name trading over its own valuation).
    # The magnitude check on the explicit number lives at the call site in
    # _entry_from_candidate, where the trusted price reference is.
    explicit = _explicit_buy_price_target(packet)
    anchor = _valuation_anchor(packet)
    if anchor is None:
        return explicit
    # Per-name margin of safety scaled by conviction grade, intrinsic-method
    # dispersion, and (gated on the price-history backfill) realized volatility. realized_volatility returns
    # None in production today, so the volatility term is inert.
    computed = compute_buy_target(
        anchor,
        conviction_grade=conviction_grade,
        confidence=confidence,
        intrinsic_low=_intrinsic_bound(packet, "intrinsic_range_low"),
        intrinsic_high=_intrinsic_bound(packet, "intrinsic_range_high"),
        realized_volatility=realized_volatility(packet.ticker, db_path=db_path),
    )
    if explicit is not None:
        # An explicit target may be stricter than the name's own margin of safety,
        # never looser: capping it only at the anchor let a target 1% under
        # intrinsic value deploy with no margin of safety at all.
        return min(explicit, computed) if computed is not None else min(explicit, float(anchor))
    return computed


def _artifact_pipeline_version(artifact: AutonomousSectorFinancialRunArtifact) -> str:
    explicit = str(getattr(artifact, "pipeline_version", "") or "").strip().lower()
    if explicit:
        return explicit
    # Defensive compatibility while v2 artifacts are being introduced: a
    # versioned contract string is enough to identify the new semantics, but
    # historical v1 artifacts remain legacy.
    contract_version = str(getattr(artifact, "contract_version", "") or "").strip().lower()
    return "v2" if contract_version.endswith("_v2") else "v1"


def _object_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _candidate_disposition_map(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> dict[str, Any]:
    rows = getattr(artifact, "candidate_dispositions", []) or []
    dispositions: dict[str, Any] = {}
    for row in rows:
        ticker = str(_object_value(row, "ticker", "") or "").strip().upper()
        if ticker:
            dispositions[ticker] = row
    return dispositions


@dataclass(frozen=True)
class _V2CandidateSemantics:
    terminal_state: str
    grade: str | None
    status: str | None
    conviction_source: str | None
    decision_basis: str | None
    selection_validation_status: str | None
    confidence: str | None
    watchlist_eligible: bool
    outcome_eligible: bool


def _v2_candidate_semantics(
    artifact: AutonomousSectorFinancialRunArtifact,
    ticker: str,
    disposition: Any | None,
) -> _V2CandidateSemantics | None:
    if disposition is None:
        return None
    terminal_state = str(_object_value(disposition, "terminal_state", "") or "").strip().upper()
    if terminal_state not in WATCHLIST_CANDIDATE_DISPOSITIONS:
        return None
    raw_verdict = str(_object_value(disposition, "underwriting_verdict", "") or "").strip().upper()
    confidence = _normalize_confidence(_object_value(disposition, "underwriting_confidence"))
    watchlist_eligible = bool(_object_value(disposition, "watchlist_eligible", False))

    if terminal_state == "OUT_OF_SCOPE":
        return _V2CandidateSemantics(
            terminal_state, None, None, None, None, None, None, False, False
        )
    if terminal_state == "SCREENED_OUT":
        return _V2CandidateSemantics(
            terminal_state,
            "AVOID",
            None,
            "sector_screen",
            "SCREEN",
            None,
            None,
            False,
            True,
        )
    if terminal_state == "NEEDS_DATA":
        review_status = str(_object_value(disposition, "review_status", "") or "").strip().upper()
        if review_status == "COMPLETED":
            return _V2CandidateSemantics(
                terminal_state,
                "DATA_INCOMPLETE",
                "UNCERTAIN",
                "company_autonomy",
                "UNDERWRITING",
                None,
                confidence,
                watchlist_eligible,
                True,
            )
        return _V2CandidateSemantics(
            terminal_state,
            "DATA_INCOMPLETE",
            "UNCERTAIN",
            "sector_screen",
            "SCREEN",
            None,
            None,
            watchlist_eligible,
            True,
        )
    if terminal_state == "READY_FOR_UNDERWRITING":
        return _V2CandidateSemantics(
            terminal_state,
            "WATCHLIST_ONLY",
            "ACTIVE",
            "sector_screen",
            "SCREEN",
            None,
            None,
            watchlist_eligible,
            True,
        )

    # UNDERWRITTEN. A child may recommend ACTIONABLE, but the watchlist grade
    # stays capped until the final selection has a matching validation record.
    if raw_verdict == "AVOID":
        return _V2CandidateSemantics(
            terminal_state,
            "AVOID",
            None,
            "company_autonomy",
            "UNDERWRITING",
            None,
            confidence,
            False,
            True,
        )
    if raw_verdict == "NO_WINNER" or not raw_verdict:
        return _V2CandidateSemantics(
            terminal_state,
            None,
            None,
            "company_autonomy",
            "UNDERWRITING",
            None,
            confidence,
            False,
            False,
        )
    if raw_verdict == "DATA_INCOMPLETE":
        return _V2CandidateSemantics(
            terminal_state,
            "DATA_INCOMPLETE",
            "UNCERTAIN",
            "company_autonomy",
            "UNDERWRITING",
            None,
            confidence,
            watchlist_eligible,
            True,
        )
    if raw_verdict in {"WATCHLIST", "WATCHLIST_ONLY"}:
        return _V2CandidateSemantics(
            terminal_state,
            "WATCHLIST_ONLY",
            "ACTIVE",
            "company_autonomy",
            "UNDERWRITING",
            None,
            confidence,
            watchlist_eligible,
            True,
        )

    selected_ticker = str(getattr(artifact, "selected_ticker", "") or "").upper()
    final_verdict = str(getattr(artifact, "final_verdict", "") or "").upper()
    decision_status = str(getattr(artifact, "decision_status", "") or "").upper()
    validation = getattr(artifact, "selection_validation", None)
    validation_status = str(_object_value(validation, "status", "") or "").strip().upper()
    validation_ticker = str(_object_value(validation, "selected_ticker", "") or "").strip().upper()
    matching_validation_status = validation_status if validation_ticker == ticker.upper() else None
    is_validated_selection = (
        raw_verdict == "ACTIONABLE"
        and ticker.upper() == selected_ticker
        and final_verdict == "SELECTED"
        and decision_status == "COMPLETE"
        and matching_validation_status == "VALIDATED"
    )
    if is_validated_selection:
        return _V2CandidateSemantics(
            terminal_state,
            "ACTIONABLE",
            None,
            "sector_final_decision",
            "VALIDATED_UNDERWRITING",
            matching_validation_status,
            confidence,
            True,
            True,
        )
    if raw_verdict != "ACTIONABLE":
        return _V2CandidateSemantics(
            terminal_state,
            None,
            None,
            "company_autonomy",
            "UNDERWRITING",
            matching_validation_status,
            confidence,
            False,
            False,
        )
    return _V2CandidateSemantics(
        terminal_state,
        "WATCHLIST_ONLY",
        "ACTIVE",
        "company_autonomy",
        "UNDERWRITING",
        matching_validation_status,
        confidence,
        watchlist_eligible,
        True,
    )


def _candidate_verdict(
    artifact: AutonomousSectorFinancialRunArtifact,
    ticker: str,
    ranking: dict[str, Any] | None,
) -> str | None:
    if _artifact_pipeline_version(artifact) == AUTONOMOUS_SECTOR_PIPELINE_V2:
        semantics = _v2_candidate_semantics(
            artifact, ticker, _candidate_disposition_map(artifact).get(ticker.upper())
        )
        return semantics.grade if semantics else None
    if artifact.selected_ticker and artifact.selected_ticker.upper() == ticker.upper():
        if artifact.final_verdict == "SELECTED":
            return "ACTIONABLE"
        if artifact.final_verdict == "WATCHLIST":
            return "WATCHLIST_ONLY"
        if artifact.final_verdict == "DATA_INCOMPLETE":
            return "DATA_INCOMPLETE"
    if ranking:
        for key in ("company_autonomy_verdict", "final_verdict", "verdict"):
            value = ranking.get(key)
            if value:
                return str(value).upper()
    return None


def _candidate_conviction_source(
    artifact: AutonomousSectorFinancialRunArtifact,
    ticker: str,
    ranking: dict[str, Any] | None,
) -> str | None:
    if _artifact_pipeline_version(artifact) == AUTONOMOUS_SECTOR_PIPELINE_V2:
        semantics = _v2_candidate_semantics(
            artifact, ticker, _candidate_disposition_map(artifact).get(ticker.upper())
        )
        return semantics.conviction_source if semantics else None
    if artifact.selected_ticker and artifact.selected_ticker.upper() == ticker.upper():
        if artifact.final_verdict in {"SELECTED", "WATCHLIST"}:
            return "sector_final_decision"
    if ranking:
        if ranking.get("company_autonomy_verdict"):
            return "company_autonomy"
        if ranking.get("final_verdict") or ranking.get("verdict"):
            return "relative_ranking"
    return None


def _normalize_confidence(value: Any) -> str | None:
    normalized = str(value or "").strip().upper()
    return normalized if normalized in WATCHLIST_CONFIDENCE_LABELS else None


def _confidence_from_business_caps(confidence_caps: list[Any]) -> str | None:
    """Confidence inferred from a ranking row's caps, when it carries no explicit label.

    Caps can only LOWER confidence: two distinct business-quality caps (or the suspicious
    valuation-anchor cap) give LOW, one gives MODERATE. No business-quality cap is an
    absence of evidence, not evidence of strength, so it yields None (unknown) rather than
    HIGH — a HIGH here would also pick the thinnest margin-of-safety base for an ACTIONABLE
    name (0.10 instead of the neutral 0.20). HIGH has to come from an explicit label.
    """
    try:
        from app.autonomous.sector_runtime import (
            SUSPICIOUS_VALUATION_ANCHOR_CAP,
            classify_audit_signal,
        )
    except Exception:
        business_caps = [str(item).upper() for item in confidence_caps if str(item).strip()]
        suspicious_cap = "SUSPICIOUS_MAGNITUDE_DCF"
    else:
        business_caps = [
            str(item).upper()
            for item in confidence_caps
            if str(item).strip() and classify_audit_signal(str(item)) == "BUSINESS_QUALITY"
        ]
        suspicious_cap = SUSPICIOUS_VALUATION_ANCHOR_CAP
    if not business_caps:
        return None
    if len(set(business_caps)) >= 2 or suspicious_cap in set(business_caps):
        return "LOW"
    return "MODERATE"


def _candidate_confidence(
    artifact: AutonomousSectorFinancialRunArtifact,
    ticker: str,
    ranking: dict[str, Any] | None,
) -> str | None:
    if _artifact_pipeline_version(artifact) == AUTONOMOUS_SECTOR_PIPELINE_V2:
        semantics = _v2_candidate_semantics(
            artifact, ticker, _candidate_disposition_map(artifact).get(ticker.upper())
        )
        return semantics.confidence if semantics else None
    if artifact.selected_ticker and artifact.selected_ticker.upper() == ticker.upper():
        if artifact.final_decision:
            confidence = _normalize_confidence(artifact.final_decision.confidence)
            if confidence:
                return confidence
        confidence = _normalize_confidence(artifact.confidence)
        if confidence:
            return confidence
    if ranking:
        for key in ("company_autonomy_confidence", "confidence", "confidence_ceiling"):
            confidence = _normalize_confidence(ranking.get(key))
            if confidence:
                return confidence
        verdict = _candidate_verdict(artifact, ticker, ranking)
        if verdict in WATCHLIST_POPULATION_VERDICTS and "confidence_caps" in ranking:
            return _confidence_from_business_caps(list(ranking.get("confidence_caps") or []))
    return None


def _cap_fields_for_candidate(
    packet: SectorCompanyFinancialPacket,
    *,
    as_of_date: str | None,
    db_path: str | Path | None = None,
) -> tuple[float | None, str | None, str | None, str | None]:
    """(market_cap_mm, cap_source, cap_band, cap_asof) for one intake row.

    Prefers the sweep's load-time chain result carried on the packet; falls
    back to a light offline chain pass (strict tier skipped, stale-shares
    tier priced from the packet) for artifacts that predate cap provenance.
    """
    asof = str(as_of_date or "").strip() or None
    packet_source = getattr(packet, "market_cap_source", None)
    if packet_source:
        packet_mm = getattr(packet, "market_cap_mm", None)
        band = getattr(packet, "market_cap_category", None)
        return (
            float(packet_mm) if isinstance(packet_mm, (int, float)) else None,
            str(packet_source),
            str(band) if band else None,
            asof,
        )
    from app.autonomous.cap_resolver import classify_market_cap_for_band_filter

    current_price = packet.current_price if isinstance(packet.current_price, (int, float)) else None
    classification = classify_market_cap_for_band_filter(
        packet.ticker,
        as_of_date=asof or _utc_now_iso()[:10],
        asof_price=None,
        current_price=current_price,
        db_path=db_path,
        price_lookup=lambda _ticker, _asof: None,
    )
    return (
        classification.market_cap_mm,
        classification.cap_source,
        classification.cap_band,
        classification.as_of_date,
    )


def _memo_candidate_payload(
    artifact: AutonomousSectorFinancialRunArtifact, ticker: str
) -> dict[str, Any]:
    memo_body = artifact.memo_body if isinstance(artifact.memo_body, dict) else {}
    candidates = (
        memo_body.get("candidates") if isinstance(memo_body.get("candidates"), dict) else {}
    )
    payload = candidates.get(ticker.upper()) or candidates.get(ticker)
    return dict(payload) if isinstance(payload, dict) else {}


def _strings_from_payload(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    return [str(item) for item in value] if isinstance(value, list) else []


def _entry_from_candidate(
    artifact: AutonomousSectorFinancialRunArtifact,
    *,
    ticker: str,
    packet: SectorCompanyFinancialPacket,
    ranking: dict[str, Any] | None,
    added_at: str,
    db_path: str | Path | None = None,
) -> WatchlistEntry | None:
    pipeline_version = _artifact_pipeline_version(artifact)
    v2_semantics = None
    v2_disposition = None
    if pipeline_version == AUTONOMOUS_SECTOR_PIPELINE_V2:
        v2_disposition = _candidate_disposition_map(artifact).get(ticker.upper())
        v2_semantics = _v2_candidate_semantics(artifact, ticker, v2_disposition)
    verdict = v2_semantics.grade if v2_semantics else _candidate_verdict(artifact, ticker, ranking)
    # Build an entry for ANY emitted verdict (incl AVOID) so the caller can
    # snapshot it into the decision ledger. The caller still gates the watchlist
    # ROW write to WATCHLIST_POPULATION_VERDICTS. A None verdict (no emitted
    # decision at all) is not recordable.
    if verdict is None:
        return None
    confidence = _candidate_confidence(artifact, ticker, ranking)
    # Thread the conviction grade + confidence into the per-name discount.
    buy_price_target = _buy_price_target(
        packet,
        conviction_grade=verdict,
        confidence=confidence,
        db_path=db_path,
    )
    if buy_price_target is None and pipeline_version != AUTONOMOUS_SECTOR_PIPELINE_V2:
        return None
    current_price = (
        float(packet.current_price) if isinstance(packet.current_price, (int, float)) else None
    )
    if v2_semantics is not None and v2_semantics.grade != "ACTIONABLE":
        status = v2_semantics.status or "ACTIVE"
    else:
        status = (
            "DEPLOY_READY"
            if current_price is not None
            and buy_price_target is not None
            and current_price <= buy_price_target
            else "ACTIVE"
        )
    status_reason: str | None = None
    # Add-time DEPLOY_READY refuses stale prices. The packet's price is
    # dated by the run's as-of date; when an old artifact is (re)populated
    # long after that date, the crossing lands as ACTIVE with an explicit
    # reason instead of presenting a months-old print as a live trigger.
    if status == "DEPLOY_READY":
        artifact_asof = str(getattr(artifact, "as_of_date", "") or "").strip()
        try:
            asof_age_days = (
                datetime.fromisoformat(added_at.replace("Z", "+00:00")).date()
                - datetime.fromisoformat(artifact_asof).date()
            ).days
        except ValueError:
            asof_age_days = None
        from app.watchlist.triggers import _price_trigger_max_age_days

        if asof_age_days is not None and asof_age_days > _price_trigger_max_age_days():
            status = "ACTIVE"
            status_reason = (
                f"ADD_TIME_PRICE_STALE:artifact_asof={artifact_asof}:"
                f"age_days={asof_age_days}:ceiling={_price_trigger_max_age_days()}"
            )
    # Quarantine per-share anchors that fall outside the [0.2x, 5.0x]
    # band of the TRUSTED trailing fiscal-year median price so a magnitude/units
    # artifact is never shipped as a buy target. Prefer the trailing median; fall
    # back to the addition price only when no trailing reference is available.
    from app.watchlist import triggers

    anchor = _valuation_anchor(packet)
    trusted_ref = (
        triggers._historical_fiscal_year_median_price(ticker, db_path=db_path) or current_price
    )
    anchor_result = anchor_sanity.evaluate_anchor(anchor, trusted_ref)
    # The same band, applied to a model-written buy target. That field is the
    # number the DEPLOY_READY trigger fires on and nothing had ever checked it:
    # a target of 7,000 against a reference of 100 shipped as "buy now" with no
    # reason string at all, while the identical slip in the anchor was
    # quarantined here. A COMPUTED target is a discount to an anchor this band
    # has already accepted, so it is not re-checked.
    explicit_target = _explicit_buy_price_target(packet)
    explicit_result = (
        anchor_sanity.evaluate_anchor(explicit_target, trusted_ref)
        if explicit_target is not None
        else None
    )
    if (
        anchor_result.verdict.startswith("QUARANTINE")
        and anchor is not None
        and trusted_ref is not None
    ):
        # The anchor is the root cause when both are out of band — a computed
        # target inherits the anchor's units slip — so it keeps its own reason.
        status = "QUARANTINE"
        status_reason = f"{anchor_result.verdict}:anchor={anchor:.2f}:ref={trusted_ref:.2f}:ratio={anchor_result.ratio}"
    elif (
        explicit_result is not None
        and explicit_result.verdict.startswith("QUARANTINE")
        and trusted_ref is not None
    ):
        status = "QUARANTINE"
        status_reason = (
            f"QUARANTINE_BUY_TARGET_OUT_OF_BAND:target={explicit_target:.2f}:"
            f"ref={trusted_ref:.2f}:ratio={explicit_result.ratio}"
        )
    elif anchor is not None and trusted_ref is None:
        # Anchor present but NO trusted reference (FY-median AND current price
        # both missing) is UNASSESSABLE. Surface it in status_reason rather than
        # silently admitting the name with an unverified anchor (plan line 242).
        # A None current_price can never reach DEPLOY_READY here, so the status is
        # left as ACTIVE; the marker just makes the unassessable anchor auditable.
        status_reason = f"UNASSESSABLE_NO_REFERENCE:anchor={anchor:.2f}"
    memo_payload = _memo_candidate_payload(artifact, ticker)
    thesis_text = str(memo_payload.get("thesis") or "").strip()
    if not thesis_text and ranking:
        thesis_text = str(ranking.get("positioning_summary") or "").strip()
    if not thesis_text and artifact.final_decision and artifact.selected_ticker == ticker:
        thesis_text = artifact.final_decision.thesis
    open_questions = _strings_from_payload(memo_payload, "open_questions")
    # Surface the fetchable data-resolution codes (e.g. NO_FILING) in
    # open_questions so the operator sees exactly which fetch unblocks a
    # DATA_INCOMPLETE candidate. Conviction grade falls out of the verdict.
    if verdict == "DATA_INCOMPLETE" and artifact.final_decision is not None:
        for code in artifact.final_decision.data_resolution_needed:
            code_str = str(code).strip()
            if code_str and code_str not in open_questions:
                open_questions.append(code_str)
    if v2_disposition is not None:
        for code in list(_object_value(v2_disposition, "reason_codes", []) or []):
            code_str = str(code).strip()
            if code_str and code_str not in open_questions:
                open_questions.append(code_str)
    market_cap_mm, cap_source, cap_band, cap_asof = _cap_fields_for_candidate(
        packet,
        as_of_date=getattr(artifact, "as_of_date", None),
        db_path=db_path,
    )
    # Structural exclusion gate at intake: surfaced, never silent. A candidate
    # that reached this point with a verdict still lands as a row, but a
    # structural trigger forces QUARANTINE with the exact literal reason
    # string(s); an anchor-sanity quarantine reason is appended, not lost.
    # The schema is ensured first so a fresh DB reads as "minimal", not
    # "missing", to the gate's fail-closed check.
    from app.autonomous.structural_gate import evaluate_structural_gate

    ensure_watchlist_schema(db_path)
    gate_result = evaluate_structural_gate(
        ticker,
        as_of_date=str(getattr(artifact, "as_of_date", "") or added_at[:10]),
        price=current_price,
        market_cap_mm=market_cap_mm,
        db_path=db_path,
    )
    if gate_result.quarantined or gate_result.excluded_error:
        # Fail-closed: an EXCLUDED_ERROR (engine DB unreadable mid-sweep)
        # quarantines just like a structural trigger — a going-concern name
        # must not land presentable because the DB was briefly unreadable.
        # When both fire, both reason strings are kept.
        status = "QUARANTINE"
        gate_reasons = ";".join(
            part for part in (gate_result.reason_string, gate_result.degraded_string) if part
        )
        status_reason = gate_reasons if status_reason is None else f"{gate_reasons}|{status_reason}"
    # Grade-vs-narrative consistency: the grade is deterministic and the thesis
    # is LLM text, so they can diverge (the known memo/eval-gate mismatch
    # failure mode). A divergence is flagged in status_reason — never silently
    # persisted — but the deterministic grade is kept as-is.
    from app.autonomous.semantic_consistency import (
        grade_narrative_mismatch,
        ungrounded_missing_data_claims,
    )

    consistency_flags: list[str] = []
    mismatch = grade_narrative_mismatch(thesis_text, verdict)
    if mismatch:
        consistency_flags.append(f"GRADE_NARRATIVE_MISMATCH:{mismatch}")
    if ungrounded_missing_data_claims(thesis_text, packet):
        consistency_flags.append(
            "UNGROUNDED_MISSING_DATA_CLAIM:thesis claims missing data with no "
            "matching gap recorded in the packet"
        )
    if consistency_flags:
        flag_string = "|".join(consistency_flags)
        logger.warning(
            "watchlist consistency flag for %s (%s): %s",
            ticker,
            artifact.run_id,
            flag_string,
        )
        status_reason = flag_string if status_reason is None else f"{status_reason}|{flag_string}"
    return WatchlistEntry(
        ticker=ticker,
        status=status,
        conviction_grade=verdict,
        confidence=confidence,
        conviction_source=(
            v2_semantics.conviction_source
            if v2_semantics is not None
            else _candidate_conviction_source(artifact, ticker, ranking)
        ),
        valuation_anchor_method=_anchor_method(packet),
        valuation_anchor_value=_valuation_anchor(packet),
        buy_price_target=buy_price_target,
        current_price_at_addition=current_price,
        thesis_text=thesis_text or None,
        key_risks=_strings_from_payload(memo_payload, "key_risks"),
        falsifiers=_strings_from_payload(memo_payload, "falsifiers"),
        open_questions=open_questions,
        source_run_id=artifact.run_id,
        source_sector=artifact.sector,
        added_at=added_at,
        status_reason=status_reason,
        market_cap_mm=market_cap_mm,
        cap_source=cap_source,
        cap_band=cap_band,
        cap_asof=cap_asof,
        pipeline_version=(
            AUTONOMOUS_SECTOR_PIPELINE_V2
            if pipeline_version == AUTONOMOUS_SECTOR_PIPELINE_V2
            else None
        ),
        candidate_disposition=(v2_semantics.terminal_state if v2_semantics is not None else None),
        decision_basis=(v2_semantics.decision_basis if v2_semantics is not None else None),
        selection_validation_status=(
            v2_semantics.selection_validation_status if v2_semantics is not None else None
        ),
    )


def populate_from_sector_artifact(
    artifact: AutonomousSectorFinancialRunArtifact,
    db_path: str | Path | None = None,
) -> WatchlistPopulationResult:
    pipeline_version = _artifact_pipeline_version(artifact)
    if pipeline_version == AUTONOMOUS_SECTOR_PIPELINE_V2:
        artifact._validate_v2()
    execution_status = str(getattr(artifact, "execution_status", None) or artifact.status).upper()
    if execution_status != "COMPLETED":
        return WatchlistPopulationResult(added_or_updated=0, skipped=0)

    integrity_status = artifact_decision_eligibility(artifact.to_dict())
    if integrity_status != FINANCIAL_INTEGRITY_PASS:
        affected_tickers = sorted(
            {
                str(packet.ticker).strip().upper()
                for packet in artifact.company_packets
                if str(packet.ticker).strip()
            }
        )
        reason = f"INVALID_FINANCIAL_INPUT:{integrity_status}"
        return WatchlistPopulationResult(
            added_or_updated=0,
            skipped=len(affected_tickers),
            skipped_reasons={ticker: reason for ticker in affected_tickers},
        )

    packets = {packet.ticker.upper(): packet for packet in artifact.company_packets}
    rankings = {
        str(row.get("ticker") or "").upper(): row
        for row in artifact.relative_ranking
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }
    disposition_map = _candidate_disposition_map(artifact)
    candidate_tickers: list[str] = []
    if pipeline_version == AUTONOMOUS_SECTOR_PIPELINE_V2:
        candidate_tickers.extend(disposition_map)
    else:
        for row in artifact.relative_ranking:
            ticker = str(row.get("ticker") or "").upper() if isinstance(row, dict) else ""
            if ticker and ticker not in candidate_tickers:
                candidate_tickers.append(ticker)
        selected_tickers = (
            artifact.candidate_selection.get("selected_tickers", [])
            if isinstance(artifact.candidate_selection, dict)
            else []
        )
        for ticker in selected_tickers:
            normalized = str(ticker).upper()
            if normalized and normalized not in candidate_tickers:
                candidate_tickers.append(normalized)

    added_at = _utc_now_iso()
    entry_ids: list[int] = []
    skipped_reasons: dict[str, str] = {}
    actionable_tickers: list[str] = []
    added_tickers: list[str] = []
    for ticker in candidate_tickers:
        packet = packets.get(ticker)
        semantics = (
            _v2_candidate_semantics(artifact, ticker, disposition_map.get(ticker))
            if pipeline_version == AUTONOMOUS_SECTOR_PIPELINE_V2
            else None
        )
        if semantics is not None and semantics.terminal_state == "OUT_OF_SCOPE":
            skipped_reasons[ticker] = "DISPOSITION_OUT_OF_SCOPE"
            continue
        if packet is None:
            # A v2 disposition is the source of truth for research-queue
            # visibility.  An absent packet is itself incomplete data, not a
            # reason to drop an otherwise eligible NEEDS_DATA/READY row (or a
            # screen outcome) from the audit trail.
            if semantics is None or not (
                semantics.watchlist_eligible or semantics.outcome_eligible
            ):
                skipped_reasons[ticker] = "PACKET_MISSING"
                continue
            reason_codes = list(
                _object_value(disposition_map.get(ticker), "reason_codes", []) or []
            )
            packet = SectorCompanyFinancialPacket(
                ticker=ticker,
                financial_status="INCOMPLETE",
                model_fit_status="INCOMPLETE",
                data_quality_status="NEEDS_DATA",
                blockers=[str(code) for code in reason_codes],
            )
        ranking = rankings.get(ticker)
        verdict = semantics.grade if semantics else _candidate_verdict(artifact, ticker, ranking)
        entry = _entry_from_candidate(
            artifact,
            ticker=ticker,
            packet=packet,
            ranking=ranking,
            added_at=added_at,
            db_path=db_path,
        )
        if entry is None:
            # A None verdict has no emitted decision to record; otherwise the
            # anchor produced no computable buy-price target.
            skipped_reasons[ticker] = (
                "DISPOSITION_OR_VERDICT_MISSING"
                if pipeline_version == AUTONOMOUS_SECTOR_PIPELINE_V2
                else ("VERDICT_MISSING" if verdict is None else "BUY_PRICE_TARGET_MISSING")
            )
            continue
        # Snapshot EVERY emitted verdict (incl AVOID) into the
        # ticker_outcomes decision ledger so the calibration loop can resolve
        # realized + excess returns and measure grade predictiveness — including
        # the AVOID sign-inverted segment — per the "record ALL
        # grades" goal. NON-FATAL: a ledger failure must never abort population.
        try:
            # Select the cap-appropriate benchmark
            # (SPY large-cap, IWM small/mid) from the per-name cap category when
            # present, else the sweep-level market_cap_focus. The chosen symbol
            # is stored per-outcome so the resolver/report stay benchmark-agnostic.
            snapshot_decision(
                entry,
                run_id=artifact.run_id,
                as_of_date=artifact.as_of_date,
                grade=entry.conviction_grade,
                status=entry.status,
                horizon_days=365,
                market_cap_focus=getattr(artifact, "market_cap_focus", None),
                market_cap_category=getattr(packet, "market_cap_category", None),
            )
        except Exception:  # pragma: no cover - defensive, exercised by test
            logger.warning(
                "snapshot_decision failed for %s (run_id=%s); watchlist population continues",
                ticker,
                artifact.run_id,
                exc_info=True,
            )
        # The watchlist ROW write stays gated to actionable / watchlist /
        # data-incomplete verdicts; AVOID (and any other non-population verdict)
        # is recorded in the ledger above but never added as a watchlist row.
        if pipeline_version == AUTONOMOUS_SECTOR_PIPELINE_V2 and (
            semantics is None or not semantics.watchlist_eligible
        ):
            disposition = semantics.terminal_state if semantics is not None else "MISSING"
            skipped_reasons[ticker] = f"DISPOSITION_{disposition}"
            continue
        if verdict not in WATCHLIST_POPULATION_VERDICTS:
            skipped_reasons[ticker] = f"VERDICT_{verdict}"
            continue
        entry_ids.append(add_or_update(entry, db_path=db_path))
        added_tickers.append(ticker)
        if entry.conviction_grade == "ACTIONABLE":
            actionable_tickers.append(ticker)

    # Grade-time adverse-events check: a name graded ACTIONABLE this run gets
    # docket/news-checked NOW, not at the next daily heartbeat, so a
    # lawsuit-cheap name is flagged EVENT_PENDING on the first surface it
    # appears on. Best-effort — never aborts population.
    adverse_check: dict | None = None
    if actionable_tickers:
        try:
            from app.db import get_db
            from app.events.grade_time import grade_time_adverse_check

            with get_db() as conn:
                adverse_check = grade_time_adverse_check(conn, actionable_tickers)
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "grade-time adverse check failed for %s; watchlist population continues",
                actionable_tickers,
                exc_info=True,
            )

    # Retroactive queue protection — every name added this run inherits
    # EVENT_PENDING protection for M&A/delisting/dilution filings ALREADY on
    # file (cache-only, no network). Best-effort — never aborts population.
    if added_tickers:
        try:
            from app.db import connect as _db_connect
            from app.events.grade_time import retroactive_queue_protection

            conn = _db_connect(resolve_db_path(db_path))
            try:
                retroactive_queue_protection(conn, added_tickers)
            finally:
                conn.close()
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "retroactive queue-protection scan failed for %s; watchlist population continues",
                added_tickers,
                exc_info=True,
            )

    return WatchlistPopulationResult(
        added_or_updated=len(entry_ids),
        skipped=len(skipped_reasons),
        entry_ids=entry_ids,
        skipped_reasons=skipped_reasons,
        adverse_check=adverse_check,
    )


def _sector_artifact_path_for_run_id(
    source_run_id: str, *, runs_dir: str | Path | None = None
) -> Path | None:
    base = Path(runs_dir) if runs_dir is not None else Path(get_config().runs_dir)
    candidates = [
        base / "autonomous_sector" / source_run_id / "autonomous_sector_run.json",
        base / "_archive" / "autonomous_sector" / source_run_id / "autonomous_sector_run.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_sector_artifact_for_run_id(
    source_run_id: str,
    *,
    runs_dir: str | Path | None = None,
) -> AutonomousSectorFinancialRunArtifact | None:
    path = _sector_artifact_path_for_run_id(source_run_id, runs_dir=runs_dir)
    if path is None:
        return None
    try:
        return AutonomousSectorFinancialRunArtifact.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def backfill_decision_clarity_from_artifacts(
    db_path: str | Path | None = None,
    *,
    runs_dir: str | Path | None = None,
    history_source: str = "decision_clarity_backfill",
) -> dict[str, Any]:
    conn = _connect(db_path)
    updated: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM watchlist
            WHERE status != 'REMOVED'
              AND conviction_grade IN ('ACTIONABLE', 'SELECTED', 'WATCHLIST_ONLY')
              AND (
                confidence IS NULL OR confidence = ''
                OR conviction_source IS NULL OR conviction_source = ''
              )
            ORDER BY id
            """
        ).fetchall()
        for row in rows:
            ticker = str(row["ticker"]).upper()
            artifact = _load_sector_artifact_for_run_id(
                str(row["source_run_id"]), runs_dir=runs_dir
            )
            if artifact is None:
                skipped[ticker] = "ARTIFACT_MISSING"
                continue
            rankings = {
                str(item.get("ticker") or "").upper(): item
                for item in artifact.relative_ranking
                if isinstance(item, dict) and str(item.get("ticker") or "").strip()
            }
            ranking = rankings.get(ticker)
            confidence = _candidate_confidence(artifact, ticker, ranking)
            conviction_source = _candidate_conviction_source(artifact, ticker, ranking)
            updates: dict[str, Any] = {}
            if (row["confidence"] is None or row["confidence"] == "") and confidence is not None:
                updates["confidence"] = confidence
            if (
                row["conviction_source"] is None or row["conviction_source"] == ""
            ) and conviction_source is not None:
                updates["conviction_source"] = conviction_source
            if not updates:
                missing_fields = []
                if row["confidence"] is None or row["confidence"] == "":
                    missing_fields.append("CONFIDENCE")
                if row["conviction_source"] is None or row["conviction_source"] == "":
                    missing_fields.append("CONVICTION_SOURCE")
                skipped[ticker] = (
                    f"{'_AND_'.join(missing_fields)}_MISSING" if missing_fields else "NO_UPDATE"
                )
                continue
            assignments = ", ".join(f"{field_name} = ?" for field_name in updates)
            conn.execute(
                f"UPDATE watchlist SET {assignments} WHERE id = ?",
                tuple(updates.values()) + (int(row["id"]),),
            )
            for field_name, new_value in updates.items():
                _insert_history(
                    conn,
                    watchlist_id=int(row["id"]),
                    field_name=field_name,
                    old_value=row[field_name],
                    new_value=new_value,
                    source=history_source,
                    source_run_id=str(row["source_run_id"]),
                )
            updated.append(
                {
                    "id": int(row["id"]),
                    "ticker": ticker,
                    "confidence": updates.get("confidence", row["confidence"]),
                    "conviction_source": updates.get("conviction_source", row["conviction_source"]),
                    "updated_fields": sorted(updates),
                }
            )
        conn.commit()
        return {"updated": updated, "skipped": skipped, "updated_count": len(updated)}
    finally:
        conn.close()


def backfill_confidence_from_artifacts(
    db_path: str | Path | None = None,
    *,
    runs_dir: str | Path | None = None,
) -> dict[str, Any]:
    return backfill_decision_clarity_from_artifacts(db_path, runs_dir=runs_dir)
