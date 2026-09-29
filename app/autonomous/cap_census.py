"""Unknown-cap census resolution through the band-filter chain.

A June 2026 universe census
counted 1,130 classified tickers with no computable strict as-of market cap.
This module re-runs the census methodology (latest sector_inference
membership, strict cap from the latest scorecard price) and then pushes every
strict-unknown name through the stale-shares fallback tier to answer the
sizing question: how many resolve from data we already have, and how many
genuinely need a paid fundamentals feed (the inert EODHD tier-2 slot).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from app.config import get_config
from app.autonomous.cap_resolver import (
    CAP_SOURCE_STALE_SHARES,
    CAP_SOURCE_UNKNOWN,
    classify_market_cap_for_band_filter,
    default_price_lookup,
)
from app.valuation.lineage import latest_decision_eligible_valuation_row

logger = logging.getLogger(__name__)


def _scorecard_price_from_json(outputs_json: str | None) -> float | None:
    if not outputs_json:
        return None
    try:
        scorecard = json.loads(outputs_json)
    except json.JSONDecodeError:
        return None
    pzd = scorecard.get("pricing_zone_detail") or {}
    price = pzd.get("current_price")
    if isinstance(price, (int, float)) and price > 0:
        return float(price)
    return None


def census_cap_resolution(
    *,
    as_of_date: str | None = None,
    db_path: str | Path | None = None,
    price_lookup=None,
    max_tickers: int | None = None,
) -> dict[str, Any]:
    """Classify every census ticker through the chain and tally resolution.

    Census membership: every ticker with a latest sector_inference row.
    Its newest scorecard is selected and then authorized; an invalid newest
    row is treated as missing rather than falling back to an older row.
    """
    asof = str(as_of_date or "").strip() or date.today().isoformat()
    path = Path(db_path) if db_path is not None else Path(get_config().db_path)
    lookup = price_lookup or default_price_lookup()

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        sector_rows = conn.execute(
            """
            SELECT si.ticker
            FROM sector_inference si
            WHERE si.as_of_date = (
                SELECT MAX(si2.as_of_date)
                FROM sector_inference si2
                WHERE si2.ticker = si.ticker
            )
            ORDER BY si.ticker
            """
        ).fetchall()
        if max_tickers is not None:
            sector_rows = sector_rows[: int(max_tickers)]
        rows: list[dict[str, Any]] = []
        for sector_row in sector_rows:
            ticker = str(sector_row["ticker"]).upper()
            scorecard = latest_decision_eligible_valuation_row(
                conn,
                ticker=ticker,
                method="scorecard",
            )
            rows.append(
                {
                    "ticker": ticker,
                    "scorecard_asof": (scorecard["as_of_date"] if scorecard is not None else None),
                    "outputs_json": (scorecard["outputs_json"] if scorecard is not None else None),
                }
            )
    finally:
        conn.close()

    counts: dict[str, Any] = {
        "census_tickers": len(rows),
        "strict_resolved": 0,
        "stale_shares_resolved": 0,
        "still_unknown": 0,
        "stale_shares_by_band": {},
        "unknown_reasons": {},
    }
    stale_resolved: list[dict[str, Any]] = []
    still_unknown: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        ticker = str(row["ticker"]).upper()
        classification = classify_market_cap_for_band_filter(
            ticker,
            as_of_date=str(row["scorecard_asof"] or asof),
            asof_price=_scorecard_price_from_json(row["outputs_json"]),
            db_path=path,
            price_lookup=lookup,
        )
        if classification.cap_source == CAP_SOURCE_STALE_SHARES:
            counts["stale_shares_resolved"] += 1
            band = classification.band_label
            counts["stale_shares_by_band"][band] = counts["stale_shares_by_band"].get(band, 0) + 1
            stale_resolved.append(
                {
                    "ticker": ticker,
                    "market_cap_mm": classification.market_cap_mm,
                    "cap_band": classification.cap_band,
                    "detail": classification.detail,
                }
            )
        elif classification.cap_source == CAP_SOURCE_UNKNOWN:
            counts["still_unknown"] += 1
            reason = classification.detail or "unknown"
            counts["unknown_reasons"][reason] = counts["unknown_reasons"].get(reason, 0) + 1
            still_unknown.append({"ticker": ticker, "reason": reason})
        else:
            counts["strict_resolved"] += 1
        if (idx + 1) % 500 == 0:
            logger.info(
                "census cap resolution: %d/%d examined (%d stale-resolved, %d unknown)",
                idx + 1,
                len(rows),
                counts["stale_shares_resolved"],
                counts["still_unknown"],
            )

    return {
        "as_of_date": asof,
        "counts": counts,
        "stale_shares_resolved": stale_resolved,
        "still_unknown": still_unknown,
    }


__all__ = ["census_cap_resolution"]
