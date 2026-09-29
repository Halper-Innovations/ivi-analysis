"""Realized-volatility input layer for the per-name margin of safety.

This is the explicit hook for a future price-history backfill. Today ``price_quotes`` rows are written with
``status='UNKNOWN'`` and only span ~5 weeks for a handful of tickers, so
``realized_volatility`` returns ``None`` for virtually every production name and
the discount function (``app/watchlist/margin_of_safety.py``) falls back to its
neutral volatility term (``w_vol`` contributes 0 until a backfill provides an ``'OK'``
price history). Once daily quotes land (both ``'OK'`` and
``'parsed'`` are valid cached states), this accessor begins returning a real
annualized volatility with no other code change.

Pure read layer: no LLM, no network, no writes.

Volatility convention
----------------------
Annualized volatility = stdev(daily log returns) * sqrt(252), using the
POPULATION stdev (``statistics.pstdev`` / ddof=0). This matches the established
codebase convention (``app/alpha/cross_sectional_ranker.py`` z-scores use
``statistics.pstdev``) and reproduces the plan's pinned literal
(``0.01 * sqrt(252) == 0.158745`` for a balanced alternating +/-0.01 series).
"""

from __future__ import annotations

import math
import sqlite3
import statistics
from pathlib import Path

from app.db import connect as db_connect
from app.watchlist.schema import resolve_db_path

# Cached price-quote statuses that represent a real resolved price.
# Both 'OK' and 'parsed' are valid cached states.
_USABLE_STATUSES: tuple[str, ...] = ("OK", "parsed")

# Trading days per year for annualization.
_TRADING_DAYS_PER_YEAR: int = 252


def realized_volatility(
    ticker: str,
    *,
    db_path: str | Path | None = None,
    min_observations: int = 10,
    lookback_days: int = 180,
) -> float | None:
    """Return the annualized realized volatility for ``ticker``, or ``None``.

    Reads ``price_quotes`` for the ticker, restricted to usable statuses
    (``'OK'``/``'parsed'``) within a trailing ``lookback_days`` window, collapses
    to one price per distinct ``as_of_date`` (keeping the latest ``fetched_at``),
    and computes the annualized standard deviation of daily log returns.

    Returns ``None`` when fewer than ``min_observations`` distinct-day quotes are
    available, so the caller falls back to a neutral volatility term. A
    zero-variance series (all prices equal) returns ``0.0``.
    """
    path = resolve_db_path(db_path)
    prices = _load_daily_prices(
        ticker=ticker,
        path=path,
        lookback_days=lookback_days,
    )
    if len(prices) < min_observations:
        return None

    returns: list[float] = []
    for prev, curr in zip(prices, prices[1:]):
        if prev <= 0.0 or curr <= 0.0:
            continue
        returns.append(math.log(curr / prev))

    if len(returns) < 1:
        return None

    daily_stdev = statistics.pstdev(returns)
    return daily_stdev * math.sqrt(_TRADING_DAYS_PER_YEAR)


def _load_daily_prices(
    *,
    ticker: str,
    path: Path,
    lookback_days: int,
) -> list[float]:
    """Return prices ordered by trade date, one per distinct ``as_of_date``.

    When multiple rows share an ``as_of_date`` (e.g. different providers), the
    row with the latest ``fetched_at`` wins.
    """
    if not path.exists():
        return []

    # Trailing window cutoff against the most recent usable trade date.
    cutoff_sql = (
        "date((SELECT MAX(as_of_date) FROM price_quotes "
        "WHERE ticker = ? AND status IN (%s)), ?)"
        % ",".join("?" for _ in _USABLE_STATUSES)
    )

    by_date: dict[str, tuple[str, float]] = {}
    try:
        with db_connect(path) as conn:
            rows = conn.execute(
                """
                SELECT as_of_date, price, fetched_at
                FROM price_quotes
                WHERE ticker = ?
                  AND status IN (%s)
                  AND price IS NOT NULL
                  AND as_of_date >= %s
                ORDER BY as_of_date ASC
                """
                % (",".join("?" for _ in _USABLE_STATUSES), cutoff_sql),
                (
                    ticker,
                    *_USABLE_STATUSES,
                    ticker,
                    *_USABLE_STATUSES,
                    f"-{int(lookback_days)} days",
                ),
            ).fetchall()
    except sqlite3.Error:
        return []

    for as_of_date, price, fetched_at in rows:
        if price is None:
            continue
        key = str(as_of_date)
        existing = by_date.get(key)
        fetched = str(fetched_at or "")
        if existing is None or fetched >= existing[0]:
            by_date[key] = (fetched, float(price))

    return [by_date[d][1] for d in sorted(by_date)]
