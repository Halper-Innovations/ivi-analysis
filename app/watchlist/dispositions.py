"""Disposition ledger: the signal-to-execution bridge.

The audit chain used to end at "Review at target" — nothing linked a
DEPLOY_READY surfacing to an executed trade, a documented pass, or a
deferral. This module makes a typed, append-only disposition record the
mandatory terminus of every at-target surfacing:

- The daily pass opens one OPEN row per at-target presentation
  (``sync_at_target_dispositions``); the digest and ``ivi investor buy-now``
  render open rows until the operator closes them.
- ``close_disposition`` records ACTED / PASSED / DEFERRED with operator,
  typed reason code, structured rationale, intended size, sizing rationale,
  and pre-mortem — and writes the ticker_outcomes journal row
  (run_id='journal_live') so the behavioral kill-criterion tally counts
  automatically. Never auto-emitted: the CLI requires explicit confirmation
  (the strategic review's named failure mode is process theater).
- Event disposals (``ivi events dispose``) land in the same ledger as
  EVENT_DISPOSAL rows — one typed trail for every owner decision.

Rows are never deleted and a decided row is never re-decided.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.watchlist.lineage import watchlist_row_is_decision_eligible
from app.watchlist.schema import ensure_watchlist_schema, resolve_db_path

JOURNAL_RUN_ID = "journal_live"

KIND_AT_TARGET = "AT_TARGET"
KIND_EVENT_DISPOSAL = "EVENT_DISPOSAL"
KIND_MANUAL = "MANUAL"

STATUS_OPEN = "OPEN"
CLOSE_STATUSES = ("ACTED", "PASSED", "DEFERRED")

# Typed close reasons — free text goes in rationale, never in the code.
REASON_CODES = (
    "ENTERED_POSITION",
    "PASSED_VALUATION",
    "PASSED_LIQUIDITY",
    "PASSED_THESIS_CHANGED",
    "PASSED_EVENT_RISK",
    "PASSED_PORTFOLIO_FIT",
    "DEFERRED_PENDING_EVENT",
    "DEFERRED_PENDING_REVIEW",
    "EVENT_REVIEWED",
    "OTHER",
)

# Deterministic mapping into the journal ledger's decision vocabulary.
_STATUS_TO_DECISION = {"ACTED": "BUY", "PASSED": "PASS", "DEFERRED": "WATCH"}
_DEFAULT_JOURNAL_HORIZON_DAYS = 365


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    from app.db import connect, init_db

    ensure_watchlist_schema(db_path)
    conn = connect(resolve_db_path(db_path))
    try:
        conn.execute("SELECT 1 FROM dispositions LIMIT 1")
    except sqlite3.Error:
        init_db(conn=conn)
    return conn


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    payload = {key: row[key] for key in row.keys()}
    try:
        payload["trigger_snapshot"] = json.loads(payload.get("trigger_snapshot_json") or "{}")
    except (TypeError, ValueError):
        payload["trigger_snapshot"] = {}
    return payload


def _decorate_open_disposition(
    row: sqlite3.Row,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    from app.watchlist.contract import is_price_trigger_eligible

    payload = _row_dict(row)
    if str(payload.get("kind")) != KIND_AT_TARGET:
        payload["source_state"] = "NOT_APPLICABLE"
        payload["is_current_actionable"] = False
        return payload

    source_id = payload.get("watchlist_id")
    current_id = payload.get("current_watchlist_id")
    authoritative = source_id is not None and source_id == current_id
    source_run_eligible = watchlist_row_is_decision_eligible(
        payload,
        manifest_path,
    )
    if authoritative:
        payload["source_state"] = (
            "CURRENT" if source_run_eligible else "CURRENT_FINANCIAL_INTEGRITY_BLOCKED"
        )
    else:
        payload["source_state"] = "SUPERSEDED_STALE_SOURCE"
    payload["financial_integrity_eligible"] = source_run_eligible
    payload["is_current_actionable"] = bool(
        authoritative
        and source_run_eligible
        and str(payload.get("current_watchlist_status") or "").upper()
        in {"DEPLOY_READY", "BUY_CONFIRMED"}
        and not payload.get("current_event_pending")
        and str(payload.get("current_conviction_grade") or "").upper() == "ACTIONABLE"
        and is_price_trigger_eligible(
            {
                "status": payload.get("current_watchlist_status"),
                "conviction_grade": payload.get("current_conviction_grade"),
                "pipeline_version": payload.get("current_pipeline_version"),
                "candidate_disposition": payload.get("current_candidate_disposition"),
                "decision_basis": payload.get("current_decision_basis"),
                "selection_validation_status": payload.get("current_selection_validation_status"),
            }
        )
    )
    if not authoritative:
        payload["resolution_required"] = (
            "Explicit PASSED or DEFERRED closure required; the source assessment "
            "is no longer authoritative."
        )
    return payload


def open_dispositions(
    db_path: str | Path | None = None,
    *,
    kind: str | None = None,
    conn: sqlite3.Connection | None = None,
    manifest_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Open rows with the platform's own currency judgment attached.

    When ``conn`` is provided the caller owns the connection (neither
    schema-ensured nor closed here) — how the read-only web layer shares
    this derivation without a write path.
    """
    from app.watchlist.store import current_watchlist_cte

    owns_conn = conn is None
    if conn is None:
        conn = _connect(db_path)
    try:
        kind_clause = "AND d.kind = ?" if kind else ""
        params: tuple[Any, ...] = (STATUS_OPEN, kind) if kind else (STATUS_OPEN,)
        rows = conn.execute(
            f"""
            WITH latest_watchlist AS ({current_watchlist_cte()})
            SELECT d.*,
                   w.id AS current_watchlist_id,
                   w.source_run_id AS current_source_run_id,
                   w.status AS current_watchlist_status,
                   w.conviction_grade AS current_conviction_grade,
                   w.event_pending AS current_event_pending,
                   w.pipeline_version AS current_pipeline_version,
                   w.candidate_disposition AS current_candidate_disposition,
                   w.decision_basis AS current_decision_basis,
                   w.selection_validation_status AS current_selection_validation_status,
                   w.*
            FROM dispositions d
            LEFT JOIN latest_watchlist l ON l.ticker = d.ticker
            LEFT JOIN watchlist w ON w.id = l.latest_id
            WHERE d.status = ? {kind_clause}
            ORDER BY d.opened_at, d.id
            """,
            params,
        ).fetchall()
        return [_decorate_open_disposition(row, manifest_path=manifest_path) for row in rows]
    finally:
        if owns_conn:
            conn.close()


def sync_at_target_dispositions(
    db_path: str | Path | None = None,
    *,
    opened_by: str = "system:daily",
) -> dict[str, int]:
    """Open one OPEN AT_TARGET disposition per presentable at-target name.

    Idempotent per authoritative watchlist row. A stale OPEN disposition does
    not suppress a new disposition for a newer authoritative assessment.
    Event-blocked rows (EVENT_PENDING) are not presentable and therefore do
    not open dispositions — they carry their own queue.
    """
    from app.watchlist.contract import is_price_trigger_eligible
    from app.watchlist.store import current_watchlist_cte

    conn = _connect(db_path)
    try:
        at_target = conn.execute(
            f"""
            WITH latest_watchlist AS ({current_watchlist_cte()})
            SELECT w.*
            FROM watchlist w
            JOIN latest_watchlist ON latest_watchlist.latest_id = w.id
            WHERE w.status IN ('DEPLOY_READY', 'BUY_CONFIRMED')
              AND (w.event_pending IS NULL OR w.event_pending = '')
              AND UPPER(COALESCE(w.conviction_grade, '')) != 'AVOID'
            """
        ).fetchall()
        existing_open_ids = {
            int(row["watchlist_id"])
            for row in conn.execute(
                "SELECT DISTINCT watchlist_id FROM dispositions "
                "WHERE status = ? AND kind = ? AND watchlist_id IS NOT NULL",
                (STATUS_OPEN, KIND_AT_TARGET),
            ).fetchall()
        }
        opened = 0
        now = _utc_now_iso()
        seen: set[str] = set()
        for row in at_target:
            if not watchlist_row_is_decision_eligible(row):
                continue
            if not is_price_trigger_eligible(row):
                continue
            ticker = str(row["ticker"]).upper()
            if int(row["id"]) in existing_open_ids or ticker in seen:
                continue
            seen.add(ticker)
            snapshot = {
                "watchlist_status": row["status"],
                "conviction_grade": row["conviction_grade"],
                "buy_price_target": row["buy_price_target"],
            }
            conn.execute(
                """
                INSERT INTO dispositions(
                    ticker, watchlist_id, kind, status, opened_at, opened_by,
                    trigger_snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    int(row["id"]),
                    KIND_AT_TARGET,
                    STATUS_OPEN,
                    now,
                    opened_by,
                    json.dumps(snapshot, default=str),
                ),
            )
            opened += 1
        conn.commit()
        return {"opened": opened, "already_open": len(existing_open_ids)}
    finally:
        conn.close()


def close_disposition(
    *,
    ticker: str,
    status: str,
    operator: str,
    reason_code: str,
    rationale: str,
    intended_size: str | None = None,
    sizing_rationale: str | None = None,
    pre_mortem: str | None = None,
    disposition_id: int | None = None,
    db_path: str | Path | None = None,
    write_journal: bool = True,
) -> dict[str, Any]:
    """Close the ticker's OPEN disposition (opening a MANUAL one if absent).

    ACTED closes are refused while the row carries an open EVENT_PENDING
    flag — the event queue must be disposed first; that is the whole point
    of queue protection. Closed rows write the journal ledger
    (ticker_outcomes, run_id='journal_live') unless write_journal=False.
    """
    status_norm = str(status).strip().upper()
    if status_norm not in CLOSE_STATUSES:
        raise ValueError(f"status must be one of {CLOSE_STATUSES}")
    reason_norm = str(reason_code).strip().upper()
    if reason_norm not in REASON_CODES:
        raise ValueError(f"reason_code must be one of {REASON_CODES}")
    if not str(operator).strip():
        raise ValueError("operator is required — the ledger attributes every decision")
    if not str(rationale).strip():
        raise ValueError("rationale is required")
    ticker_norm = str(ticker).strip().upper()

    conn = _connect(db_path)
    try:
        # EVENT_PENDING enforcement at close time.
        if status_norm == "ACTED":
            from app.watchlist.store import current_watchlist_cte

            row = conn.execute(
                f"""
                WITH latest_watchlist AS ({current_watchlist_cte("WHERE ticker = ?")})
                SELECT w.*
                FROM watchlist w
                JOIN latest_watchlist l ON l.latest_id = w.id
                """,
                (ticker_norm,),
            ).fetchone()
            if row is not None and row["event_pending"]:
                raise ValueError(
                    f"refusing ACTED close: {ticker_norm} carries open "
                    f"{row['event_pending']} — dispose the event first "
                    "(`ivi events list --open`)"
                )
            if row is None or not watchlist_row_is_decision_eligible(row, ticker=ticker_norm):
                raise ValueError(
                    "refusing ACTED close: the current watchlist source run is "
                    "not financial-integrity decision-eligible"
                )

        if disposition_id is not None:
            open_row = conn.execute(
                "SELECT * FROM dispositions WHERE id = ?", (int(disposition_id),)
            ).fetchone()
            if open_row is None:
                raise ValueError(f"no disposition with id {disposition_id}")
            if str(open_row["ticker"]).upper() != ticker_norm:
                raise ValueError(
                    f"disposition {disposition_id} belongs to "
                    f"{str(open_row['ticker']).upper()}, not {ticker_norm}"
                )
            if str(open_row["status"]) != STATUS_OPEN:
                raise ValueError(
                    f"disposition {disposition_id} already decided "
                    f"({open_row['status']}) — the ledger is append-only"
                )
        else:
            from app.watchlist.store import current_watchlist_cte

            open_row = conn.execute(
                f"""
                WITH latest_watchlist AS ({current_watchlist_cte("WHERE ticker = ?")})
                SELECT d.*
                FROM dispositions d
                LEFT JOIN latest_watchlist l ON l.ticker = d.ticker
                WHERE d.ticker = ? AND d.status = ?
                ORDER BY CASE WHEN d.watchlist_id = l.latest_id THEN 0 ELSE 1 END,
                         d.opened_at, d.id
                LIMIT 1
                """,
                (ticker_norm, ticker_norm, STATUS_OPEN),
            ).fetchone()

        if (
            open_row is not None
            and status_norm == "ACTED"
            and str(open_row["kind"]) == KIND_AT_TARGET
        ):
            from app.watchlist.store import current_watchlist_cte

            current = conn.execute(
                f"""
                WITH latest_watchlist AS ({current_watchlist_cte("WHERE ticker = ?")})
                SELECT latest_id FROM latest_watchlist
                """,
                (ticker_norm,),
            ).fetchone()
            current_id = int(current["latest_id"]) if current is not None else None
            if open_row["watchlist_id"] != current_id:
                raise ValueError(
                    "refusing ACTED close: superseded/stale-source AT_TARGET "
                    "dispositions require an explicit PASSED or DEFERRED closure"
                )

        now = _utc_now_iso()
        if open_row is None:
            conn.execute(
                """
                INSERT INTO dispositions(
                    ticker, kind, status, opened_at, opened_by, trigger_snapshot_json
                ) VALUES (?, ?, ?, ?, ?, '{}')
                """,
                (ticker_norm, KIND_MANUAL, STATUS_OPEN, now, operator),
            )
            open_row = conn.execute(
                "SELECT * FROM dispositions WHERE ticker = ? AND status = ? "
                "ORDER BY id DESC LIMIT 1",
                (ticker_norm, STATUS_OPEN),
            ).fetchone()

        outcome_run_id = JOURNAL_RUN_ID if write_journal else None
        conn.execute(
            """
            UPDATE dispositions
            SET status = ?, decided_at = ?, operator = ?, reason_code = ?,
                rationale = ?, intended_size = ?, sizing_rationale = ?,
                pre_mortem = ?, outcome_run_id = ?
            WHERE id = ?
            """,
            (
                status_norm,
                now,
                operator,
                reason_norm,
                rationale,
                intended_size,
                sizing_rationale,
                pre_mortem,
                outcome_run_id,
                int(open_row["id"]),
            ),
        )
        conn.commit()
        closed = conn.execute(
            "SELECT * FROM dispositions WHERE id = ?", (int(open_row["id"]),)
        ).fetchone()
    finally:
        conn.close()

    journal_row: dict[str, Any] | None = None
    if write_journal:
        journal_row = _write_journal_entry(
            ticker=ticker_norm,
            status=status_norm,
            rationale=rationale,
            sizing_rationale=sizing_rationale,
            pre_mortem=pre_mortem,
            watchlist_id=(
                int(open_row["watchlist_id"]) if open_row["watchlist_id"] is not None else None
            ),
            db_path=db_path,
        )
    result = _row_dict(closed)
    result["journal"] = journal_row
    return result


def _write_journal_entry(
    *,
    ticker: str,
    status: str,
    rationale: str,
    sizing_rationale: str | None,
    pre_mortem: str | None,
    watchlist_id: int | None,
    db_path: str | Path | None,
) -> dict[str, Any]:
    """ticker_outcomes journal row via the existing add_outcome machinery.

    Conviction maps from the row's grade (ACTIONABLE -> 4, else 3) —
    deterministic and documented, never fabricated per-close.
    """
    from datetime import date as _date

    from app.outcomes.store import add_outcome

    conn = _connect(db_path)
    try:
        if watchlist_id is not None:
            row = conn.execute(
                "SELECT conviction_grade, buy_price_target FROM watchlist WHERE id = ?",
                (watchlist_id,),
            ).fetchone()
        else:
            from app.watchlist.store import current_watchlist_cte

            row = conn.execute(
                f"""
                WITH latest_watchlist AS ({current_watchlist_cte("WHERE ticker = ?")})
                SELECT w.conviction_grade, w.buy_price_target
                FROM watchlist w
                JOIN latest_watchlist l ON l.latest_id = w.id
                """,
                (ticker,),
            ).fetchone()
    finally:
        conn.close()
    grade = str(row["conviction_grade"] or "") if row is not None else ""
    conviction = 4 if grade.upper() == "ACTIONABLE" else 3
    notes_parts = [rationale.strip()]
    if sizing_rationale:
        notes_parts.append(f"sizing: {sizing_rationale.strip()}")
    if pre_mortem:
        notes_parts.append(f"pre-mortem: {pre_mortem.strip()}")
    return add_outcome(
        ticker=ticker,
        as_of_date=_date.today().isoformat(),
        run_id=JOURNAL_RUN_ID,
        decision=_STATUS_TO_DECISION[status],
        conviction=conviction,
        horizon_days=_DEFAULT_JOURNAL_HORIZON_DAYS,
        notes=" | ".join(notes_parts),
        thesis_tags=["source:disposition"],
        buy_price_target=(
            float(row["buy_price_target"])
            if row is not None and row["buy_price_target"] is not None
            else None
        ),
    )


def record_event_disposal(
    *,
    ticker: str | None,
    event_id: int,
    operator: str,
    reason_code: str,
    note: str,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """A typed EVENT_DISPOSAL ledger row as a byproduct of `ivi events dispose`.

    Opened and closed in one step (the disposal IS the decision); never
    writes the journal — event review is queue work, not a trade decision.
    """
    reason_norm = str(reason_code).strip().upper()
    if reason_norm not in REASON_CODES:
        raise ValueError(f"reason_code must be one of {REASON_CODES}")
    if not str(operator).strip():
        raise ValueError("operator is required")
    now = _utc_now_iso()
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO dispositions(
                ticker, kind, status, opened_at, opened_by,
                trigger_snapshot_json, decided_at, operator, reason_code,
                rationale, event_id
            ) VALUES (?, ?, 'PASSED', ?, ?, '{}', ?, ?, ?, ?, ?)
            """,
            (
                str(ticker or "UNKNOWN").upper(),
                KIND_EVENT_DISPOSAL,
                now,
                operator,
                now,
                operator,
                reason_norm,
                note,
                int(event_id),
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM dispositions ORDER BY id DESC LIMIT 1").fetchone()
        return _row_dict(row)
    finally:
        conn.close()


def journal_entry_count(db_path: str | Path | None = None) -> int:
    """Kill-criterion tally: journaled live decisions (run_id='journal_live')."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM ticker_outcomes WHERE run_id = ?",
            (JOURNAL_RUN_ID,),
        ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


__all__ = [
    "CLOSE_STATUSES",
    "JOURNAL_RUN_ID",
    "KIND_AT_TARGET",
    "KIND_EVENT_DISPOSAL",
    "KIND_MANUAL",
    "REASON_CODES",
    "close_disposition",
    "journal_entry_count",
    "open_dispositions",
    "record_event_disposal",
    "sync_at_target_dispositions",
]
