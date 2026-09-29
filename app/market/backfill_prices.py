"""Backfill missing current_price in cached scorecards.

Fetches prices via the configured price provider (Yahoo → Stooq chain)
and patches the scorecard's pricing_zone_detail.current_price in the
valuations table. Only touches scorecards where current_price is NULL
or NaN.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BackfillSummary:
    total_scorecards: int
    missing_price: int
    fetched: int
    updated: int
    failed: int
    skipped_existing: int
    failed_tickers: list[str]


def _needs_price(scorecard: dict[str, Any]) -> bool:
    """Return True if the scorecard's current_price is missing or NaN."""
    pzd = scorecard.get("pricing_zone_detail") or {}
    price = pzd.get("current_price")
    if price is None:
        return True
    if isinstance(price, float) and (math.isnan(price) or price <= 0):
        return True
    return False


def backfill_prices(
    *,
    db_path: str | Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    as_of_date: str | None = None,
) -> BackfillSummary:
    """Fetch and patch missing prices in cached scorecards.

    Parameters
    ----------
    db_path : path to engine.db
    dry_run : if True, fetch prices but don't write to DB
    force : if True, re-fetch prices even for scorecards that have one
    as_of_date : price as-of date (default: today)
    """
    from app.config import get_config
    from app.market.price_provider import get_default_provider

    cfg = get_config()
    if db_path is None:
        db_path = cfg.db_path

    effective_date = as_of_date or date.today().isoformat()
    provider = get_default_provider(cfg)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Load all scorecards
        rows = conn.execute(
            """
            SELECT id, ticker, as_of_date, outputs_json
            FROM valuations
            WHERE method = 'scorecard'
              AND as_of_date = (
                  SELECT MAX(v2.as_of_date)
                  FROM valuations v2
                  WHERE v2.ticker = valuations.ticker AND v2.method = 'scorecard'
              )
            ORDER BY ticker
            """
        ).fetchall()

        total = len(rows)
        missing = 0
        fetched = 0
        updated = 0
        failed = 0
        skipped = 0
        failed_tickers: list[str] = []

        for row in rows:
            ticker = row["ticker"]
            sc = json.loads(row["outputs_json"] or "{}")

            if not force and not _needs_price(sc):
                skipped += 1
                continue

            missing += 1

            # Fetch price
            try:
                snapshot = provider.get_price_asof(ticker, effective_date)
                price = snapshot.price if snapshot and isinstance(snapshot.price, (int, float)) else None
            except Exception as exc:
                logger.warning("backfill %s: price fetch failed: %s", ticker, exc)
                failed += 1
                failed_tickers.append(ticker)
                continue

            if price is None or (isinstance(price, float) and math.isnan(price)):
                logger.debug("backfill %s: no price available", ticker)
                failed += 1
                failed_tickers.append(ticker)
                continue

            fetched += 1

            if dry_run:
                logger.info("backfill %s: would set price to %.4f (dry run)", ticker, price)
                continue

            # Patch the scorecard
            pzd = sc.get("pricing_zone_detail") or {}
            pzd["current_price"] = price
            sc["pricing_zone_detail"] = pzd

            conn.execute(
                "UPDATE valuations SET outputs_json = ? WHERE id = ?",
                (json.dumps(sc), row["id"]),
            )
            updated += 1

            if updated % 50 == 0:
                conn.commit()
                logger.info("backfill: %d/%d updated so far", updated, missing)

        if not dry_run:
            conn.commit()

        logger.info(
            "backfill complete: %d total, %d missing, %d fetched, %d updated, %d failed",
            total, missing, fetched, updated, failed,
        )

        return BackfillSummary(
            total_scorecards=total,
            missing_price=missing,
            fetched=fetched,
            updated=updated,
            failed=failed,
            skipped_existing=skipped,
            failed_tickers=sorted(failed_tickers),
        )
    finally:
        conn.close()
