"""Events read model: the triage deck over ``corporate_events``.

Two lanes share the table and lifecycle (see :mod:`app.events.store`); the
lane split imports the production tuples so the deck can never drift from
the detectors. Cards group into lifecycle columns DETECTED → QUALIFIED →
SURFACED → DECIDED, with EXPIRED terminal; each open card carries the exact
``ivi events dispose`` command — the CLI stays the sole write path.

The scan-integrity strip walks business days (Mon–Fri, no holiday calendar —
the same convention as the dead-man's previous-business-day check) up to the
previous business day: a business day with no OK scan row is a visible gap,
because an unscanned day is a hole in the record.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.events.store import (
    OPPORTUNITY_EVENT_TYPES,
    QUEUE_PROTECTION_EVENT_TYPES,
    STATUS_RANK,
)

LANES = {
    "opportunity": OPPORTUNITY_EVENT_TYPES,
    "queue_protection": QUEUE_PROTECTION_EVENT_TYPES,
}

# Column order on the deck: the lifecycle left to right, then the terminal
# shelf. STATUS_RANK is the store's own ordering; EXPIRED sits outside it.
COLUMN_STATUSES = (*sorted(STATUS_RANK, key=STATUS_RANK.get), "EXPIRED")

_OPEN_STATUSES = ("DETECTED", "QUALIFIED", "SURFACED")


def lane_for(event_type: str) -> str:
    return "opportunity" if event_type in OPPORTUNITY_EVENT_TYPES else "queue_protection"


def dispose_command(event_id: int) -> str:
    return (
        f'ivi events dispose {event_id} --note "<why>" '
        "--reason-code EVENT_REVIEWED|PASSED_EVENT_RISK|OTHER"
    )


def _parse_detail(raw: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _age_days(detection_date: str | None, *, today: date) -> int | None:
    if not detection_date:
        return None
    try:
        detected = date.fromisoformat(str(detection_date))
    except ValueError:
        return None
    return (today - detected).days


def _filings_for(
    conn: sqlite3.Connection, event_ids: list[int]
) -> dict[int, list[dict[str, Any]]]:
    if not event_ids:
        return {}
    placeholders = ",".join("?" for _ in event_ids)
    rows = conn.execute(
        f"""
        SELECT event_id, accession, form_type, filing_date, role
        FROM corporate_event_filings
        WHERE event_id IN ({placeholders})
        ORDER BY filing_date ASC, id ASC
        """,
        event_ids,
    ).fetchall()
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["event_id"]), []).append(
            {
                "accession": row["accession"],
                "form_type": row["form_type"],
                "filing_date": row["filing_date"],
                "role": row["role"],
            }
        )
    return grouped


def events_deck(
    conn: sqlite3.Connection,
    *,
    lane: str | None = None,
    event_type: str | None = None,
    q: str | None = None,
    per_column: int = 40,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The triage deck: lanes → lifecycle columns → cards, plus totals.

    ``per_column`` caps each column's card list (the DETECTED shelf holds
    hundreds of busted-IPO candidates); column totals always count the full
    filtered population so nothing truncates silently.
    """
    if lane is not None and lane not in LANES:
        raise ValueError(f"Unknown lane: {lane}")
    if per_column <= 0:
        raise ValueError("per_column must be positive")

    conditions: list[str] = []
    params: list[Any] = []
    if lane is not None:
        marks = ",".join("?" for _ in LANES[lane])
        conditions.append(f"event_type IN ({marks})")
        params.extend(LANES[lane])
    if event_type is not None:
        conditions.append("event_type = ?")
        params.append(event_type)
    if q:
        needle = f"%{str(q).strip().upper()}%"
        conditions.append("(UPPER(ticker) LIKE ? OR UPPER(company_name) LIKE ?)")
        params.extend([needle, needle])
    where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    rows = conn.execute(
        f"""
        SELECT id, cik, event_type, company_name, ticker, ticker_state, status,
               detection_date, qualification_date, expiry_reason, detail_json
        FROM corporate_events
        {where_sql}
        ORDER BY detection_date DESC, id DESC
        """,
        params,
    ).fetchall()

    today = (now or datetime.now(timezone.utc)).date()
    lanes: dict[str, dict[str, Any]] = {
        name: {
            "lane": name,
            "total": 0,
            "open_total": 0,
            "type_counts": {},
            "columns": {status: {"status": status, "total": 0, "cards": []} for status in COLUMN_STATUSES},
        }
        for name in LANES
    }
    unknown_ticker_total = 0

    for row in rows:
        row_lane = lane_for(row["event_type"])
        bucket = lanes[row_lane]
        bucket["total"] += 1
        bucket["type_counts"][row["event_type"]] = (
            bucket["type_counts"].get(row["event_type"], 0) + 1
        )
        status = str(row["status"])
        if status in _OPEN_STATUSES:
            bucket["open_total"] += 1
        if str(row["ticker_state"] or "") == "UNKNOWN_TICKER":
            unknown_ticker_total += 1
        column = bucket["columns"].get(status)
        if column is None:
            # Never drop an unexpected lifecycle value silently.
            column = {"status": status, "total": 0, "cards": []}
            bucket["columns"][status] = column
        column["total"] += 1
        if len(column["cards"]) < per_column:
            column["cards"].append(
                {
                    "id": int(row["id"]),
                    "lane": row_lane,
                    "event_type": row["event_type"],
                    "status": status,
                    "company_name": row["company_name"],
                    "ticker": row["ticker"],
                    "ticker_state": row["ticker_state"],
                    "cik": row["cik"],
                    "detection_date": row["detection_date"],
                    "qualification_date": row["qualification_date"],
                    "age_days": _age_days(row["detection_date"], today=today),
                    "expiry_reason": row["expiry_reason"],
                    "detail": _parse_detail(row["detail_json"]),
                    "filings": [],
                    "dispose_command": (
                        dispose_command(int(row["id"])) if status in _OPEN_STATUSES else None
                    ),
                }
            )

    card_ids = [
        card["id"]
        for bucket in lanes.values()
        for column in bucket["columns"].values()
        for card in column["cards"]
    ]
    filings = _filings_for(conn, card_ids)
    for bucket in lanes.values():
        for column in bucket["columns"].values():
            for card in column["cards"]:
                card["filings"] = filings.get(card["id"], [])
        # Deck order is the store's lifecycle order, unknown statuses last.
        bucket["columns"] = [
            bucket["columns"][status]
            for status in (*COLUMN_STATUSES, *sorted(set(bucket["columns"]) - set(COLUMN_STATUSES)))
        ]

    return {
        "lanes": [lanes["opportunity"], lanes["queue_protection"]],
        "unknown_ticker_total": unknown_ticker_total,
    }


def scan_strip(
    conn: sqlite3.Connection,
    *,
    business_days: int = 30,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The last N business days of the scan ledger, newest first.

    Ends at the previous business day (today's daily scan may simply not
    have run yet — the dead-man check uses the same anchor). A business day
    whose row is missing or non-OK counts as a gap.
    """
    if business_days <= 0:
        raise ValueError("business_days must be positive")
    today = (now or datetime.now(timezone.utc)).date()

    days: list[date] = []
    cursor = today - timedelta(days=1)
    while len(days) < business_days:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)

    placeholders = ",".join("?" for _ in days)
    rows = conn.execute(
        f"""
        SELECT scan_date, mode, status, index_rows, candidate_rows,
               events_created, events_updated
        FROM corporate_event_scans
        WHERE scan_date IN ({placeholders})
        """,
        [day.isoformat() for day in days],
    ).fetchall()
    by_date = {str(row["scan_date"]): row for row in rows}

    strip: list[dict[str, Any]] = []
    gap_count = 0
    for day in days:
        key = day.isoformat()
        row = by_date.get(key)
        status = str(row["status"]) if row is not None else "MISSING"
        if status != "OK":
            gap_count += 1
        strip.append(
            {
                "date": key,
                "status": status,
                "mode": row["mode"] if row is not None else None,
                "index_rows": row["index_rows"] if row is not None else None,
                "candidate_rows": row["candidate_rows"] if row is not None else None,
                "events_created": row["events_created"] if row is not None else None,
                "events_updated": row["events_updated"] if row is not None else None,
            }
        )
    return {
        "days": strip,
        "gap_count": gap_count,
        "window_business_days": business_days,
    }
