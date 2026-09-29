"""CIK -> ticker resolution pass for corporate events.

Brand-new SpinCo / post-reorg CIKs are not in company_tickers.json until the
SEC refreshes it (7-day-TTL cache), so unresolved events keep
ticker_state='UNKNOWN_TICKER' and are retried on every poll — never dropped.
Resolution prefers detail_json["relisting_cik"] (cross-CIK ch11 relink) over
the debtor CIK so intake routes the new post-reorg equity, never the
cancelled pre-petition ticker.
"""

from __future__ import annotations

import json
import sqlite3

from app.events import store


def invert_ticker_map(mapping: dict[str, str]) -> dict[str, str]:
    """ticker->cik (unpadded int-string) -> cik10->primary ticker.

    CIK->tickers is one-to-many; the primary ticker is chosen
    deterministically: shortest, then alphabetical.
    """
    inverted: dict[str, str] = {}
    for ticker, cik in mapping.items():
        cik10 = str(cik).zfill(10)
        current = inverted.get(cik10)
        if current is None or (len(ticker), ticker) < (len(current), current):
            inverted[cik10] = ticker
    return inverted


def resolve_unknown_tickers(
    conn: sqlite3.Connection, *, mapping: dict[str, str] | None = None
) -> dict[str, int]:
    if mapping is None:
        from app.universe.ticker_cik_map import load_ticker_cik_map

        mapping = load_ticker_cik_map()
    by_cik = invert_ticker_map(mapping)
    rows = conn.execute(
        """
        SELECT id, cik, detail_json FROM corporate_events
        WHERE ticker_state = 'UNKNOWN_TICKER'
          AND status NOT IN ('DECIDED', 'EXPIRED')
        """
    ).fetchall()
    resolved = 0
    still_unknown = 0
    for row in rows:
        detail = json.loads(row["detail_json"] or "{}")
        resolution_cik = str(detail.get("relisting_cik") or row["cik"]).zfill(10)
        ticker = by_cik.get(resolution_cik)
        if ticker is None:
            still_unknown += 1
            continue
        store.set_ticker(conn, event_id=int(row["id"]), ticker=ticker)
        resolved += 1
    return {"resolved": resolved, "still_unknown": still_unknown}
