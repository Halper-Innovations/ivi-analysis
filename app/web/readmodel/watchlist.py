"""Watchlist read model: the ranked review queue over a read-only connection.

The presentation derivation (presented status, EVENT_PENDING blocking,
trigger eligibility, ranking) lives in :func:`app.watchlist.store.watchlist_queue`
and is shared — this module only supplies the read-only connection, so the
web layer can never diverge from the CLI's queue nor touch a write path
(``_connect`` runs schema-ensure, which writes).
"""

from __future__ import annotations

import sqlite3
from typing import Any

from app.watchlist.store import watchlist_queue
from app.web.readmodel.db import table_exists


def queue_rows(
    conn: sqlite3.Connection,
    *,
    limit: int = 500,
    sector: str | None = None,
    include_price_suspect: bool = True,
    band: str | None = None,
    scan_family: str | None = None,
) -> list[dict[str, Any]]:
    """The review queue as the CLI ranks it, over a caller-owned ro connection.

    Defaults differ from the CLI deliberately: the web UI shows the whole
    floor (``limit=500``) including PRICE_DATA_SUSPECT rows — staleness is
    rendered as frost, not hidden.

    A database that has no watchlist tables yet has an empty queue.
    """

    if not (table_exists(conn, "watchlist") and table_exists(conn, "watchlist_price_snapshots")):
        return []
    return watchlist_queue(
        limit=limit,
        sector=sector,
        include_price_suspect=include_price_suspect,
        band=band,
        scan_family=scan_family,
        conn=conn,
    )
