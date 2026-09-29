"""EVENT_PENDING flag sync between corporate events and the watchlist.

A watchlist name with an open (non-terminal) queue-protection event carries
EVENT_PENDING:<TYPE> flags on every surface, and DEPLOY_READY presentation is
blocked until an analyst pass disposes the event (`ivi events dispose`).
The flags are persisted on watchlist.event_pending (comma-joined, NULL when
clear) by sync_event_pending_flags, which runs at the end of every poll and
after every disposal; each change writes a watchlist_history row so relabels
are auditable.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from app.events import store

EVENT_PENDING_HISTORY_SOURCE = "events_feed"


def open_flags_by_ticker(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """ticker -> sorted EVENT_PENDING flags from open queue-protection events."""
    placeholders = ",".join("?" for _ in store.QUEUE_PROTECTION_EVENT_TYPES)
    rows = conn.execute(
        "SELECT DISTINCT ticker, event_type FROM corporate_events "
        f"WHERE event_type IN ({placeholders}) "
        "AND status NOT IN ('DECIDED', 'EXPIRED') AND ticker IS NOT NULL",
        list(store.QUEUE_PROTECTION_EVENT_TYPES),
    ).fetchall()
    flags: dict[str, set[str]] = {}
    for row in rows:
        flags.setdefault(str(row["ticker"]).upper(), set()).add(
            store.event_pending_flag(row["event_type"])
        )
    return {ticker: sorted(values) for ticker, values in flags.items()}


def open_flags_by_cik(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """cik -> sorted EVENT_PENDING flags from open queue-protection events.

    Events are keyed by CIK at detection; ticker resolution can lag for
    weeks (UNKNOWN_TICKER). Matching protection by CIK means an unresolved
    ticker can never orphan the flag on a watchlist row.
    """
    placeholders = ",".join("?" for _ in store.QUEUE_PROTECTION_EVENT_TYPES)
    rows = conn.execute(
        "SELECT DISTINCT cik, event_type FROM corporate_events "
        f"WHERE event_type IN ({placeholders}) "
        "AND status NOT IN ('DECIDED', 'EXPIRED')",
        list(store.QUEUE_PROTECTION_EVENT_TYPES),
    ).fetchall()
    flags: dict[str, set[str]] = {}
    for row in rows:
        cik = str(row["cik"] or "").strip().lstrip("0")
        if not cik:
            continue
        flags.setdefault(cik, set()).add(store.event_pending_flag(row["event_type"]))
    return {cik: sorted(values) for cik, values in flags.items()}


def _watchlist_ticker_ciks(tickers: list[str]) -> dict[str, str]:
    """ticker -> normalized CIK via the cached SEC registry map (no network)."""
    try:
        from app.universe.ticker_cik_map import load_ticker_cik_map

        mapping = load_ticker_cik_map(refresh_if_missing=False) or {}
    except Exception:  # noqa: BLE001 - missing cache means ticker-only matching
        return {}
    out: dict[str, str] = {}
    for ticker in tickers:
        cik = mapping.get(str(ticker).upper())
        if cik:
            out[str(ticker).upper()] = str(cik).strip().lstrip("0")
    return out


def sync_event_pending_flags(conn: sqlite3.Connection) -> dict[str, int]:
    """Reconcile watchlist.event_pending with open queue-protection events.

    Idempotent; tolerates a DB without the watchlist tables (tmp DBs in
    detector tests) by reporting zeros.
    """
    try:
        watch_rows = conn.execute(
            "SELECT id, ticker, event_pending FROM watchlist WHERE status != 'REMOVED'"
        ).fetchall()
    except sqlite3.OperationalError:
        return {"flagged": 0, "cleared": 0, "unchanged": 0}
    by_ticker = open_flags_by_ticker(conn)
    by_cik = open_flags_by_cik(conn)
    ticker_ciks = _watchlist_ticker_ciks(
        [str(row["ticker"]) for row in watch_rows]
    )
    now = datetime.now(timezone.utc).isoformat()
    flagged = cleared = unchanged = 0
    for row in watch_rows:
        ticker = str(row["ticker"]).upper()
        # CIK match is primary (survives ticker-resolution lag); the
        # ticker match remains as fallback for names outside the registry map.
        merged = set(by_ticker.get(ticker, []))
        row_cik = ticker_ciks.get(ticker)
        if row_cik:
            merged.update(by_cik.get(row_cik, []))
        desired = ",".join(sorted(merged)) or None
        current = row["event_pending"]
        if desired == current:
            unchanged += 1
            continue
        conn.execute(
            "UPDATE watchlist SET event_pending = ? WHERE id = ?",
            (desired, row["id"]),
        )
        conn.execute(
            """
            INSERT INTO watchlist_history(
                watchlist_id, changed_at, field_name, old_value, new_value, source)
            VALUES(?, ?, 'event_pending', ?, ?, ?)
            """,
            (row["id"], now, current, desired, EVENT_PENDING_HISTORY_SOURCE),
        )
        if desired is None:
            cleared += 1
        else:
            flagged += 1
    return {"flagged": flagged, "cleared": cleared, "unchanged": unchanged}
