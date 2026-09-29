"""Idempotent corporate-event store: upserts, forward-only lifecycle, skips, scans.

All functions take an open sqlite3 connection; callers own the get_db()
context. Lifecycle rank DETECTED(0) < QUALIFIED(1) < SURFACED(2) < DECIDED(3);
EXPIRED is terminal from any rank < 3. Backward transitions are refused
(no-op returning False) so re-running a backfill over already-surfaced events
cannot regress them.

Two event families share the table and lifecycle:
- opportunity events (spec 2026-06-09): spinoff / ch11_emergence / busted_ipo
- queue-protection events: filings that mean a watchlist name is carrying a
  known event (merger paper, activist stakes, dilution shelves, 8-K items).
  These go DETECTED -> DECIDED via an explicit analyst disposal
  (mark_decided); while open they render as EVENT_PENDING:<TYPE> flags and
  block DEPLOY_READY presentation.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

OPPORTUNITY_EVENT_TYPES = ("spinoff", "ch11_emergence", "busted_ipo")
QUEUE_PROTECTION_EVENT_TYPES = (
    "merger",
    "material_agreement",
    "bankruptcy",
    "obligation_acceleration",
    "restructuring",
    "delisting_notice",
    "activist",
    "dilution",
    "restatement",
    "unreviewed_8k",
    "litigation_docket",
    "adverse_news",
)
EVENT_TYPES = OPPORTUNITY_EVENT_TYPES + QUEUE_PROTECTION_EVENT_TYPES

STATUS_RANK = {"DETECTED": 0, "QUALIFIED": 1, "SURFACED": 2, "DECIDED": 3}

SKIP_REASONS = {
    "PRIOR_REPORTING_S1",
    "FOLLOWON_424B4",
    "NO_CH11_LINK_8A",
    "AMBIGUOUS_NAME_MATCH",
    "SUBMISSIONS_FETCH_FAILED",
    "INDEX_ROW_MALFORMED",
    "QUALIFIER_NOT_IMPLEMENTED",
    # Busted-IPO price qualifier (transient — event stays DETECTED, re-polled):
    "TICKER_UNRESOLVED",
    "NOT_SEASONED",
    "NO_PRICE_HISTORY",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def event_pending_flag(event_type: str) -> str:
    return f"EVENT_PENDING:{event_type.upper()}"


def find_active_event(conn: sqlite3.Connection, *, cik: str, event_type: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM corporate_events
        WHERE cik = ? AND event_type = ? AND status NOT IN ('DECIDED', 'EXPIRED')
        ORDER BY id DESC LIMIT 1
        """,
        (cik, event_type),
    ).fetchone()


def upsert_event(
    conn: sqlite3.Connection,
    *,
    cik: str,
    event_type: str,
    anchor_accession: str,
    company_name: str,
    detection_date: str,
    source_mode: str,
    detail: dict | None = None,
) -> int:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event_type: {event_type}")
    now = _now()
    conn.execute(
        """
        INSERT INTO corporate_events(
            cik, event_type, anchor_accession, company_name, status,
            detection_date, detail_json, source_mode, detected_at, updated_at)
        VALUES(?, ?, ?, ?, 'DETECTED', ?, ?, ?, ?, ?)
        ON CONFLICT(cik, event_type, anchor_accession) DO NOTHING
        """,
        (
            cik, event_type, anchor_accession, company_name,
            detection_date, json.dumps(detail or {}), source_mode, now, now,
        ),
    )
    row = conn.execute(
        "SELECT id FROM corporate_events WHERE cik=? AND event_type=? AND anchor_accession=?",
        (cik, event_type, anchor_accession),
    ).fetchone()
    return int(row["id"])


def attach_filing(
    conn: sqlite3.Connection,
    *,
    event_id: int,
    cik: str,
    accession: str,
    form_type: str,
    filing_date: str,
    role: str,
    detail: dict | None = None,
) -> bool:
    cursor = conn.execute(
        """
        INSERT INTO corporate_event_filings(
            event_id, cik, accession, form_type, filing_date, role, detail_json, created_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id, accession) DO NOTHING
        """,
        (event_id, cik, accession, form_type, filing_date, role,
         json.dumps(detail or {}), _now()),
    )
    return cursor.rowcount > 0


def _transition(
    conn: sqlite3.Connection,
    *,
    event_id: int,
    new_status: str,
    stamp_column: str,
    extra_sets: dict[str, str] | None = None,
) -> bool:
    row = conn.execute(
        "SELECT status FROM corporate_events WHERE id=?", (event_id,)
    ).fetchone()
    if row is None:
        return False
    current = row["status"]
    if current == "EXPIRED" or current not in STATUS_RANK:
        return False
    if STATUS_RANK[current] >= STATUS_RANK[new_status]:
        return False
    now = _now()
    sets = {stamp_column: now, "status": new_status, "updated_at": now}
    if extra_sets:
        sets.update(extra_sets)
    assignments = ", ".join(f"{col} = ?" for col in sets)
    conn.execute(
        f"UPDATE corporate_events SET {assignments} WHERE id = ?",
        (*sets.values(), event_id),
    )
    return True


def mark_qualified(conn: sqlite3.Connection, *, event_id: int, qualification_date: str) -> bool:
    return _transition(
        conn, event_id=event_id, new_status="QUALIFIED", stamp_column="qualified_at",
        extra_sets={"qualification_date": qualification_date},
    )


def mark_surfaced(conn: sqlite3.Connection, *, event_id: int) -> bool:
    return _transition(conn, event_id=event_id, new_status="SURFACED", stamp_column="surfaced_at")


def mark_decided(conn: sqlite3.Connection, *, event_id: int, note: str | None = None) -> bool:
    """Analyst disposal: closes an open event (queue-protection or opportunity)."""
    moved = _transition(conn, event_id=event_id, new_status="DECIDED", stamp_column="decided_at")
    if moved and note:
        merge_event_detail(conn, event_id=event_id, detail={"decision_note": note})
    return moved


def mark_expired(conn: sqlite3.Connection, *, event_id: int, expiry_reason: str) -> bool:
    row = conn.execute(
        "SELECT status FROM corporate_events WHERE id=?", (event_id,)
    ).fetchone()
    if row is None:
        return False
    current = row["status"]
    if current == "EXPIRED" or STATUS_RANK.get(current, 3) >= STATUS_RANK["DECIDED"]:
        return False
    now = _now()
    conn.execute(
        """
        UPDATE corporate_events
        SET status='EXPIRED', expiry_reason=?, expired_at=?, updated_at=?
        WHERE id=?
        """,
        (expiry_reason, now, now, event_id),
    )
    return True


def record_skip(
    conn: sqlite3.Connection,
    *,
    scan_date: str,
    cik: str,
    accession: str,
    form_type: str,
    detector: str,
    reason_code: str,
    detail: dict | None = None,
) -> bool:
    if reason_code not in SKIP_REASONS:
        raise ValueError(f"Unknown skip reason_code: {reason_code}")
    cursor = conn.execute(
        """
        INSERT INTO corporate_event_skips(
            scan_date, cik, accession, form_type, detector, reason_code, detail_json, created_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cik, accession, detector) DO NOTHING
        """,
        (scan_date, cik, accession, form_type, detector, reason_code,
         json.dumps(detail or {}), _now()),
    )
    return cursor.rowcount > 0


def record_scan(
    conn: sqlite3.Connection,
    *,
    scan_date: str,
    mode: str,
    status: str,
    counters: dict | None = None,
) -> None:
    counters = dict(counters or {})
    now = _now()
    core = {
        key: counters.pop(key, None)
        for key in ("index_rows", "candidate_rows", "events_created", "events_updated", "skips_recorded")
    }
    # Archive the prior scan record before any overwrite — the scan
    # ledger is part of the decision record and must stay reconstructible.
    try:
        prior = conn.execute(
            "SELECT * FROM corporate_event_scans WHERE scan_date = ?", (scan_date,)
        ).fetchone()
        if prior is not None:
            payload = {key: prior[key] for key in prior.keys()}
            conn.execute(
                "INSERT INTO corporate_event_scans_history("
                "source_id, scan_date, row_json, archived_at) VALUES (?, ?, ?, ?)",
                (
                    payload.get("id"),
                    scan_date,
                    json.dumps(payload, default=str),
                    now,
                ),
            )
    except Exception:  # noqa: BLE001 - archival never blocks the scan record
        pass
    conn.execute(
        """
        INSERT INTO corporate_event_scans(
            scan_date, mode, status, index_rows, candidate_rows,
            events_created, events_updated, skips_recorded, detail_json,
            created_at, updated_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(scan_date) DO UPDATE SET
            mode=excluded.mode,
            status=excluded.status,
            index_rows=excluded.index_rows,
            candidate_rows=excluded.candidate_rows,
            events_created=excluded.events_created,
            events_updated=excluded.events_updated,
            skips_recorded=excluded.skips_recorded,
            detail_json=excluded.detail_json,
            updated_at=excluded.updated_at
        WHERE excluded.status = 'OK' OR corporate_event_scans.status != 'OK'
        """,
        (
            scan_date, mode, status,
            core["index_rows"], core["candidate_rows"], core["events_created"],
            core["events_updated"], core["skips_recorded"],
            json.dumps(counters), now, now,
        ),
    )


def set_ticker(conn: sqlite3.Connection, *, event_id: int, ticker: str) -> None:
    conn.execute(
        "UPDATE corporate_events SET ticker=?, ticker_state='RESOLVED', updated_at=? WHERE id=?",
        (ticker, _now(), event_id),
    )


def reset_ticker(conn: sqlite3.Connection, *, event_id: int) -> None:
    conn.execute(
        "UPDATE corporate_events SET ticker=NULL, ticker_state='UNKNOWN_TICKER', updated_at=? WHERE id=?",
        (_now(), event_id),
    )


def merge_event_detail(conn: sqlite3.Connection, *, event_id: int, detail: dict) -> None:
    row = conn.execute(
        "SELECT detail_json FROM corporate_events WHERE id=?", (event_id,)
    ).fetchone()
    if row is None:
        return
    merged = json.loads(row["detail_json"] or "{}")
    merged.update(detail)
    conn.execute(
        "UPDATE corporate_events SET detail_json=?, updated_at=? WHERE id=?",
        (json.dumps(merged), _now(), event_id),
    )


def open_queue_protection_events(
    conn: sqlite3.Connection, *, ciks: list[str] | None = None
) -> dict[str, list[sqlite3.Row]]:
    """Open (non-terminal) queue-protection events grouped by CIK.

    These are the events that render as EVENT_PENDING flags and block
    DEPLOY_READY presentation until an analyst disposal (mark_decided).
    """
    placeholders = ",".join("?" for _ in QUEUE_PROTECTION_EVENT_TYPES)
    sql = (
        "SELECT * FROM corporate_events "
        f"WHERE event_type IN ({placeholders}) "
        "AND status NOT IN ('DECIDED', 'EXPIRED')"
    )
    params: list[str] = list(QUEUE_PROTECTION_EVENT_TYPES)
    if ciks is not None:
        cik_marks = ",".join("?" for _ in ciks)
        sql += f" AND cik IN ({cik_marks})"
        params.extend(ciks)
    sql += " ORDER BY cik, detection_date, id"
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in conn.execute(sql, params).fetchall():
        grouped.setdefault(row["cik"], []).append(row)
    return grouped
