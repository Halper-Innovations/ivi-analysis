"""Company search read model: ticker + name lookup for the ⌘K palette.

Searches the SEC registrant census (ticker and company name) unioned with
watchlist tickers the census misses, over the caller's read-only
connection. Ranking is deterministic and computed in SQL: exact ticker,
ticker prefix, name prefix, then substring; covered names (on the
watchlist, valued, or with cached facts) sort ahead within a tier, so the
platform's own coverage surfaces before census-only rows.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from app.web.readmodel.company import has_authorized_valuation
from app.web.readmodel.db import table_exists

_SEARCH_SQL_TEMPLATE = """
WITH matches AS (
    SELECT
        r.primary_ticker AS ticker,
        r.name AS name,
        COALESCE(r.sector, r.sic_description) AS sector
    FROM sec_registrants r
    WHERE r.primary_ticker IS NOT NULL
      AND (
        UPPER(r.primary_ticker) LIKE :like ESCAPE '\\'
        OR UPPER(r.name) LIKE :like ESCAPE '\\'
      )
    {watchlist_union}
)
SELECT
    m.ticker,
    m.name,
    m.sector,
    EXISTS (
        SELECT 1 FROM companyfacts_facts f WHERE f.ticker = m.ticker
    ) AS has_facts,
    EXISTS (
        SELECT 1 FROM valuations v WHERE v.ticker = m.ticker
    ) AS has_raw_valuation,
    CASE
        WHEN UPPER(m.ticker) = :exact THEN 0
        WHEN UPPER(m.ticker) LIKE :prefix ESCAPE '\\' THEN 1
        WHEN m.name IS NOT NULL AND UPPER(m.name) LIKE :prefix ESCAPE '\\' THEN 2
        ELSE 3
    END AS tier
FROM matches m
ORDER BY tier, m.ticker
"""

# Watchlist tickers the census misses. Left out when the database has no
# watchlist table yet (not populated), so search still answers from the census.
_WATCHLIST_UNION = """UNION ALL
    SELECT w.ticker, NULL, w.source_sector
    FROM watchlist w
    WHERE UPPER(w.ticker) LIKE :like ESCAPE '\\'
      AND NOT EXISTS (
        SELECT 1 FROM sec_registrants r WHERE r.primary_ticker = w.ticker
      )
    GROUP BY w.ticker"""


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_companies(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 20,
    eligible_watchlist_tickers: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Ranked ticker/name matches for ``query``; empty query returns []."""

    q = query.strip().upper()
    if not q:
        return []
    escaped = _escape_like(q)
    rows = conn.execute(
        _SEARCH_SQL_TEMPLATE.format(
            watchlist_union=_WATCHLIST_UNION if table_exists(conn, "watchlist") else ""
        ),
        {
            "exact": q,
            "prefix": f"{escaped}%",
            "like": f"%{escaped}%",
        },
    ).fetchall()
    eligible_watchlist = {str(ticker).upper() for ticker in (eligible_watchlist_tickers or set())}
    results = [
        {
            "ticker": row["ticker"],
            "name": row["name"],
            "sector": row["sector"],
            "covered": bool(row["has_facts"])
            or str(row["ticker"]).upper() in eligible_watchlist
            or (
                bool(row["has_raw_valuation"])
                and has_authorized_valuation(conn, str(row["ticker"]).upper())
            ),
            "_tier": int(row["tier"]),
        }
        for row in rows
    ]
    results.sort(key=lambda row: (row["_tier"], not row["covered"], row["ticker"]))
    return [{key: value for key, value in row.items() if key != "_tier"} for row in results[:limit]]
