from __future__ import annotations

import json
from typing import Any

from app.db import get_db, utc_now_iso
from app.outcomes.lineage import (
    bind_outcome_row,
    clear_outcome_lineage,
    outcome_row_is_decision_eligible,
    refresh_outcome_integrity_fingerprint,
)


ALLOWED_DECISIONS = {"BUY", "WATCH", "SHORT", "PASS"}
ALLOWED_STATUS = {"OPEN", "CLOSED"}


def _normalize_optional_token(value: str | None, *, lower: bool = False) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    return normalized.lower() if lower else normalized.upper()


def _validate_v2_outcome_provenance(
    *,
    decision: str,
    pipeline_version: str | None,
    candidate_disposition: str | None,
    decision_basis: str | None,
    selection_validation_status: str | None,
    grade: str | None,
    status: str | None,
    source_sector: str | None,
) -> tuple[str | None, str | None, str | None, str | None, str | None, str | None]:
    """Fail closed on direct v2 outcome writes while leaving history untouched."""

    pipeline = _normalize_optional_token(pipeline_version, lower=True)
    disposition = _normalize_optional_token(candidate_disposition)
    basis = _normalize_optional_token(decision_basis)
    validation = _normalize_optional_token(selection_validation_status)
    grade_norm = _normalize_optional_token(grade)
    status_norm = _normalize_optional_token(status)
    if pipeline != "v2":
        return (
            pipeline_version,
            candidate_disposition,
            decision_basis,
            selection_validation_status,
            grade,
            status,
        )
    if not disposition or not basis or not str(source_sector or "").strip():
        raise ValueError(
            "v2 outcomes require candidate disposition, decision basis, and source sector"
        )
    if disposition == "OUT_OF_SCOPE":
        raise ValueError("v2 OUT_OF_SCOPE candidates cannot create outcome rows")
    protective_statuses = {"QUARANTINE", "PRICE_DATA_SUSPECT", "CONTRADICTED"}
    if disposition == "SCREENED_OUT":
        valid = (
            basis == "SCREEN"
            and grade_norm == "AVOID"
            and validation != "VALIDATED"
            and decision == "PASS"
            and status_norm in protective_statuses | {"ACTIVE", None}
        )
    elif disposition == "READY_FOR_UNDERWRITING":
        valid = (
            basis == "SCREEN"
            and grade_norm == "WATCHLIST_ONLY"
            and validation != "VALIDATED"
            and decision == "WATCH"
            and status_norm in protective_statuses | {"ACTIVE"}
        )
    elif disposition == "NEEDS_DATA":
        valid = (
            basis in {"SCREEN", "UNDERWRITING"}
            and grade_norm == "DATA_INCOMPLETE"
            and validation != "VALIDATED"
            and decision == "WATCH"
            and status_norm in protective_statuses | {"UNCERTAIN"}
        )
    elif disposition == "UNDERWRITTEN":
        valid = (
            basis == "UNDERWRITING"
            and grade_norm in {"WATCHLIST_ONLY", "AVOID"}
            and validation != "VALIDATED"
            and decision == ("PASS" if grade_norm == "AVOID" else "WATCH")
            and (
                status_norm in protective_statuses | {"ACTIVE", None}
                if grade_norm == "AVOID"
                else status_norm in protective_statuses | {"ACTIVE"}
            )
        ) or (
            basis == "VALIDATED_UNDERWRITING"
            and grade_norm == "ACTIONABLE"
            and validation == "VALIDATED"
            and (
                (decision == "BUY" and status_norm in {"ACTIVE", "DEPLOY_READY", "BUY_CONFIRMED"})
                or (decision == "WATCH" and status_norm in protective_statuses)
            )
        )
    else:
        valid = False
    if not valid:
        raise ValueError("v2 outcome provenance is inconsistent")
    if grade_norm == "ACTIONABLE" and not (
        disposition == "UNDERWRITTEN"
        and basis == "VALIDATED_UNDERWRITING"
        and validation == "VALIDATED"
    ):
        raise ValueError("v2 ACTIONABLE outcomes require validated underwriting")
    return pipeline, disposition, basis, validation, grade_norm, status_norm


def archive_outcome_row(conn: Any, outcome_id: int) -> None:
    """Copy a ticker_outcomes row into its history table before mutation.

    Full-row JSON so schema drift never loses fields; best-effort — archival
    must never block the decision-record write itself.
    """
    try:
        row = conn.execute(
            "SELECT * FROM ticker_outcomes WHERE id = ?", (int(outcome_id),)
        ).fetchone()
        if row is None:
            return
        payload = {key: row[key] for key in row.keys()}
        conn.execute(
            "INSERT INTO ticker_outcomes_history("
            "source_id, ticker, as_of_date, run_id, row_json, archived_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                int(outcome_id),
                payload.get("ticker"),
                payload.get("as_of_date"),
                payload.get("run_id"),
                json.dumps(payload, default=str),
                utc_now_iso(),
            ),
        )
    except Exception:  # noqa: BLE001
        pass


def add_outcome(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
    decision: str,
    conviction: int,
    horizon_days: int,
    notes: str = "",
    thesis_tags: list[str] | None = None,
    discovery_run_id: str | None = None,
    deep_run_id: str | None = None,
    entry_price: float | None = None,
    entry_price_source: str | None = None,
    entry_date: str | None = None,
    grade: str | None = None,
    status: str | None = None,
    benchmark_symbol: str | None = None,
    buy_price_target: float | None = None,
    cap_category: str | None = None,
    pipeline_version: str | None = None,
    candidate_disposition: str | None = None,
    decision_basis: str | None = None,
    selection_validation_status: str | None = None,
    source_sector: str | None = None,
) -> dict[str, Any]:
    decision_norm = decision.strip().upper()
    if decision_norm not in ALLOWED_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(ALLOWED_DECISIONS)}")
    if conviction < 1 or conviction > 5:
        raise ValueError("conviction must be in range 1-5")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be > 0")
    ticker_norm = ticker.strip().upper()
    tags = [tag.strip() for tag in (thesis_tags or []) if tag.strip()]
    now = utc_now_iso()
    with get_db() as conn:
        existing = conn.execute(
            "SELECT * FROM ticker_outcomes WHERE ticker = ? AND as_of_date = ? AND run_id = ?",
            (ticker_norm, as_of_date, run_id),
        ).fetchone()
        if existing is not None and str(existing["outcome_status"] or "").upper() != "OPEN":
            raise ValueError(
                "refusing to overwrite a finalized outcome row; "
                "CLOSED/UNRESOLVED measurements are immutable"
            )
        # Validate the row that the UPSERT will actually create. Provenance is
        # COALESCE-preserved on conflict, while decision/grade/status overwrite;
        # checking only the incoming payload would let a legacy-looking update
        # corrupt an existing v2 cohort row.
        effective_pipeline_version = pipeline_version or (
            existing["pipeline_version"] if existing is not None else None
        )
        effective_candidate_disposition = candidate_disposition or (
            existing["candidate_disposition"] if existing is not None else None
        )
        effective_decision_basis = decision_basis or (
            existing["decision_basis"] if existing is not None else None
        )
        effective_validation_status = selection_validation_status or (
            existing["selection_validation_status"] if existing is not None else None
        )
        effective_source_sector = source_sector or (
            existing["source_sector"] if existing is not None else None
        )
        (
            pipeline_version,
            candidate_disposition,
            decision_basis,
            selection_validation_status,
            grade,
            status,
        ) = _validate_v2_outcome_provenance(
            decision=decision_norm,
            pipeline_version=effective_pipeline_version,
            candidate_disposition=effective_candidate_disposition,
            decision_basis=effective_decision_basis,
            selection_validation_status=effective_validation_status,
            grade=grade,
            status=status,
            source_sector=effective_source_sector,
        )
        source_sector = effective_source_sector
        if existing is not None:
            archive_outcome_row(conn, int(existing["id"]))
        conn.execute(
            """
            INSERT INTO ticker_outcomes(
                ticker, as_of_date, run_id, discovery_run_id, deep_run_id, decision,
                conviction, horizon_days, thesis_tags_json, notes, outcome_status,
                close_date, realized_return_pct, max_drawdown_pct,
                entry_price, entry_price_source, entry_date, grade, status, benchmark_symbol,
                cap_category, pipeline_version, candidate_disposition, decision_basis,
                selection_validation_status, source_sector,
                buy_price_target, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', NULL, NULL, NULL,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, as_of_date, run_id) DO UPDATE SET
                discovery_run_id=COALESCE(excluded.discovery_run_id, ticker_outcomes.discovery_run_id),
                deep_run_id=COALESCE(excluded.deep_run_id, ticker_outcomes.deep_run_id),
                decision=excluded.decision,
                conviction=excluded.conviction,
                horizon_days=excluded.horizon_days,
                thesis_tags_json=excluded.thesis_tags_json,
                notes=excluded.notes,
                entry_price=COALESCE(excluded.entry_price, ticker_outcomes.entry_price),
                entry_price_source=COALESCE(excluded.entry_price_source, ticker_outcomes.entry_price_source),
                entry_date=COALESCE(excluded.entry_date, ticker_outcomes.entry_date),
                grade=excluded.grade,
                status=excluded.status,
                benchmark_symbol=COALESCE(excluded.benchmark_symbol, ticker_outcomes.benchmark_symbol),
                cap_category=COALESCE(excluded.cap_category, ticker_outcomes.cap_category),
                pipeline_version=COALESCE(excluded.pipeline_version, ticker_outcomes.pipeline_version),
                candidate_disposition=COALESCE(
                    excluded.candidate_disposition, ticker_outcomes.candidate_disposition
                ),
                decision_basis=COALESCE(excluded.decision_basis, ticker_outcomes.decision_basis),
                selection_validation_status=COALESCE(
                    excluded.selection_validation_status,
                    ticker_outcomes.selection_validation_status
                ),
                source_sector=COALESCE(excluded.source_sector, ticker_outcomes.source_sector),
                buy_price_target=COALESCE(excluded.buy_price_target, ticker_outcomes.buy_price_target),
                updated_at=excluded.updated_at
            """,
            (
                ticker_norm,
                as_of_date,
                run_id,
                discovery_run_id,
                deep_run_id,
                decision_norm,
                int(conviction),
                int(horizon_days),
                json.dumps(tags),
                notes.strip(),
                None if entry_price is None else float(entry_price),
                entry_price_source,
                entry_date,
                grade,
                status,
                benchmark_symbol,
                cap_category,
                pipeline_version,
                candidate_disposition,
                decision_basis,
                selection_validation_status,
                source_sector,
                None if buy_price_target is None else float(buy_price_target),
                now,
                now,
            ),
        )
        row = conn.execute(
            """
            SELECT *
            FROM ticker_outcomes
            WHERE ticker = ? AND as_of_date = ? AND run_id = ?
            LIMIT 1
            """,
            (ticker_norm, as_of_date, run_id),
        ).fetchone()
        if row is not None:
            # An UPSERT can change any decision-bearing field.  Clear the old
            # authorization first, then bind the exact newly materialized row
            # only when this run actually emitted this ticker decision.
            clear_outcome_lineage(conn, int(row["id"]))
            bind_outcome_row(conn, int(row["id"]))
            row = conn.execute(
                "SELECT * FROM ticker_outcomes WHERE id = ?",
                (int(row["id"]),),
            ).fetchone()
    if not row:
        raise RuntimeError("failed to upsert outcome")
    return _row_to_dict(row)


def close_outcome(
    *,
    ticker: str,
    run_id: str,
    realized_return_pct: float,
    close_date: str,
    max_drawdown_pct: float | None = None,
) -> dict[str, Any]:
    ticker_norm = ticker.strip().upper()
    now = utc_now_iso()
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM ticker_outcomes
            WHERE ticker = ? AND run_id = ?
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (ticker_norm, run_id),
        ).fetchone()
        if not row:
            raise ValueError(f"No outcome found for ticker={ticker_norm} run_id={run_id}")
        if str(row["outcome_status"] or "").upper() != "OPEN":
            raise ValueError(
                "refusing to overwrite a finalized outcome row; "
                "CLOSED/UNRESOLVED measurements are immutable"
            )
        if not outcome_row_is_decision_eligible(row):
            raise ValueError(
                "refusing to close outcome without exact authorized source and row lineage"
            )
        archive_outcome_row(conn, int(row["id"]))
        conn.execute(
            """
            UPDATE ticker_outcomes
            SET outcome_status = 'CLOSED',
                close_date = ?,
                realized_return_pct = ?,
                max_drawdown_pct = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (close_date, float(realized_return_pct), max_drawdown_pct, now, int(row["id"])),
        )
        if not refresh_outcome_integrity_fingerprint(conn, int(row["id"])):
            raise RuntimeError("outcome source authorization changed while closing row")
        updated = conn.execute(
            "SELECT * FROM ticker_outcomes WHERE id = ?", (int(row["id"]),)
        ).fetchone()
    if not updated:
        raise RuntimeError("failed to update outcome")
    return _row_to_dict(updated)


def list_outcomes(
    *,
    open_only: bool = False,
    run_id: str | None = None,
    ticker: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if open_only:
        clauses.append("outcome_status = 'OPEN'")
    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.strip().upper())
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(int(limit))
    with get_db() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM ticker_outcomes
            {where_sql}
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def _row_to_dict(row) -> dict[str, Any]:
    payload = dict(row)
    payload["thesis_tags"] = json.loads(payload.get("thesis_tags_json") or "[]")
    payload.pop("thesis_tags_json", None)
    return payload
