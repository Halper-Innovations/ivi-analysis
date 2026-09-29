"""Split-basis reconciliation between backtest prices and as-of share counts.

Audit coverage gap 3: backtest prices are adjusted_close — split-adjusted to
the DOWNLOAD date — while ensure_valuation divides intrinsic value by as-of
share counts. Any split/reverse split between T and now puts price and
per-share anchor on different share bases (MoS off by the split factor), and
reverse splits concentrate in the micro band.

This module quantifies the affected names so the backtest diagnostic can report
them (and sensitivity-exclude them) without re-running the backtest. It is a
measurement instrument, not a signal change: detection compares the
as-of-visible share count against the latest known count — a large ratio is a
capital event (split, reverse split, major issuance/buyback) after T, which
also proxies adjusted-price-basis divergence.
"""
from __future__ import annotations

from typing import Any

from app.db import get_db

# Outside this band the as-of vs latest share-count ratio is implausible as
# organic drift and indicates a capital event between T and now.
CAPITAL_EVENT_RATIO_HIGH = 1.8
CAPITAL_EVENT_RATIO_LOW = 1.0 / 1.8


def share_count_basis_profile(ticker: str, as_of_cutoff: str) -> dict[str, Any]:
    """Profile the as-of vs latest share-count basis for one ticker.

    Returns dict with shares_asof, shares_latest, basis_ratio
    (latest / as-of) and classification:
      - NO_SHARES_DATA: nothing to compare
      - NO_ASOF_SHARES: ticker has counts but none visible at the cutoff
      - STABLE_BASIS: ratio within the organic band
      - CAPITAL_EVENT_SUSPECT: ratio outside the band — adjusted prices and
        as-of per-share anchors are on different bases for dates before the
        event; the diagnostic must quantify/exclude these rows.
    """
    upper = str(ticker or "").upper()
    with get_db() as conn:
        latest_row = conn.execute(
            "SELECT value FROM companyfacts_facts "
            "WHERE ticker = ? AND line_item = 'shares_outstanding' "
            "AND period_type = 'FY' AND value IS NOT NULL "
            "ORDER BY period_end DESC LIMIT 1",
            (upper,),
        ).fetchone()
        asof_row = conn.execute(
            "SELECT value FROM companyfacts_facts "
            "WHERE ticker = ? AND line_item = 'shares_outstanding' "
            "AND period_type = 'FY' AND value IS NOT NULL AND period_end <= ? "
            "ORDER BY period_end DESC LIMIT 1",
            (upper, str(as_of_cutoff)),
        ).fetchone()

    shares_latest = float(latest_row["value"]) if latest_row else None
    shares_asof = float(asof_row["value"]) if asof_row else None

    if shares_latest is None:
        classification = "NO_SHARES_DATA"
        ratio = None
    elif shares_asof is None or shares_asof <= 0:
        classification = "NO_ASOF_SHARES"
        ratio = None
    else:
        ratio = shares_latest / shares_asof
        classification = (
            "CAPITAL_EVENT_SUSPECT"
            if ratio >= CAPITAL_EVENT_RATIO_HIGH or ratio <= CAPITAL_EVENT_RATIO_LOW
            else "STABLE_BASIS"
        )

    return {
        "ticker": upper,
        "as_of_cutoff": str(as_of_cutoff),
        "shares_asof": shares_asof,
        "shares_latest": shares_latest,
        "basis_ratio": ratio,
        "classification": classification,
    }


def split_affected_tickers(tickers: list[str], as_of_cutoff: str) -> list[dict[str, Any]]:
    """Profile a universe and return only the CAPITAL_EVENT_SUSPECT rows —
    the quantification input for the backtest diagnostic's split-basis section."""
    out: list[dict[str, Any]] = []
    for ticker in tickers:
        profile = share_count_basis_profile(ticker, as_of_cutoff)
        if profile["classification"] == "CAPITAL_EVENT_SUSPECT":
            out.append(profile)
    return out
