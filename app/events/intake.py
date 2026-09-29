"""Route QUALIFIED opportunity events into watchlist intake.

Only opportunity events (spinoff / ch11_emergence / busted_ipo) surface —
queue-protection events are flags on names already on the watchlist, never
new rows. QUALIFIED events with an unresolved ticker are counted and retried
on the next poll; they are never silently dropped.
"""

from __future__ import annotations

from app.db import get_db
from app.events import store
from app.watchlist.contract import WatchlistEntry
from app.watchlist.store import add_or_update


def surface_qualified_events(*, as_of: str) -> dict[str, int]:
    placeholders = ",".join("?" for _ in store.OPPORTUNITY_EVENT_TYPES)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM corporate_events "
            f"WHERE status = 'QUALIFIED' AND event_type IN ({placeholders})",
            list(store.OPPORTUNITY_EVENT_TYPES),
        ).fetchall()
        events = [dict(row) for row in rows]

    surfaced = 0
    awaiting_ticker = 0
    for event in events:
        if event["ticker_state"] != "RESOLVED" or not event["ticker"]:
            awaiting_ticker += 1
            continue
        entry = WatchlistEntry(
            ticker=event["ticker"],
            status="ACTIVE",
            conviction_grade="DATA_INCOMPLETE",
            conviction_source="corporate_event",
            source_run_id=(
                f"events_{event['event_type']}_{event['cik']}_{event['qualification_date']}"
            ),
            added_at=as_of,
            status_reason=f"{event['event_type']} qualified {event['qualification_date']}",
            source_sector=None,
        )
        add_or_update(entry, history_source="events_feed")
        with get_db() as conn:
            store.mark_surfaced(conn, event_id=int(event["id"]))
        surfaced += 1
    return {"surfaced": surfaced, "awaiting_ticker": awaiting_ticker}
