"""SBC dilution trajectory detection.

Tracks SBC/revenue ratio trends and share count changes to detect
accelerating stock-based compensation and net shareholder dilution.
"""

from __future__ import annotations

from typing import Any

from app.valuation.share_splits import (
    REASON_SHARE_COUNT_BREAK_UNCORROBORATED,
    corroborating_split_ratio,
    is_share_count_break,
)

# The trend is the fitted change in the SBC/revenue ratio across the most recent
# fiscal years (a least-squares line over at least three ratio years inside this
# window), not latest minus the oldest ratio ever filed: one old data point must not
# turn flat recent years into an acceleration.
TREND_WINDOW_YEARS = 5
TREND_MIN_POINTS = 3
TREND_THRESHOLD = 0.01  # one percentage point of revenue across the window


def _fitted_change(points: list[tuple[int, float]]) -> float:
    """Least-squares slope times the year span: the change the line fits across the window."""
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = sxy / sxx
    return slope * (points[-1][0] - points[0][0])


def compute_sbc_trajectory(
    facts: dict[str, list[tuple[int, float]]],
    *,
    split_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Detect SBC trajectory and dilution signals.

    Returns dict with sbc_revenue_ratios, sbc_revenue_trend, shares_yoy_changes,
    sbc_flags, latest_sbc_revenue_ratio, shares_count_breaks.

    A share-count year whose move is a break (the shared rule in
    app.valuation.share_splits) is a split only when a filed split ratio in
    ``split_rows`` corroborates it; its change is then measured on the
    split-adjusted count, so a 2-for-1 year is not +100% dilution. A break
    nothing corroborates is neither a split nor ordinary dilution: the year is
    left out of the year-over-year changes (so it cannot start or extend a
    dilution streak), listed in ``shares_count_breaks`` and flagged
    SHARE_COUNT_BREAK_UNCORROBORATED, so the gap is visible rather than guessed.
    """
    sbc_flags: list[str] = []
    sbc_revenue_ratios: list[dict[str, Any]] = []
    shares_yoy_changes: list[dict[str, Any]] = []
    shares_count_breaks: list[int] = []

    def _by_year(key: str) -> dict[int, float]:
        return {int(yr): float(v) for yr, v in (facts.get(key) or [])}

    sbc_map = _by_year("sbc")
    rev_map = _by_year("revenue")
    shares_map = _by_year("shares_outstanding")
    repurchases_map = _by_year("share_repurchases_amount")

    # ── SBC/revenue ratios per year ────────────────────────────────────────
    common_years = sorted(set(sbc_map.keys()) & set(rev_map.keys()))
    for year in common_years:
        rev = rev_map[year]
        if rev > 0:
            ratio = sbc_map[year] / rev
            sbc_revenue_ratios.append({
                "year": year,
                "sbc": sbc_map[year],
                "revenue": rev,
                "ratio": round(ratio, 4),
            })

    # ── SBC/revenue trend ──────────────────────────────────────────────────
    sbc_revenue_trend = "UNKNOWN"
    latest_sbc_revenue_ratio: float | None = None

    sbc_revenue_trend_change: float | None = None
    trend_window: list[int] = []

    if sbc_revenue_ratios:
        latest_ratio_year = sbc_revenue_ratios[-1]["year"]
        window_points = [
            (int(r["year"]), float(r["ratio"]))
            for r in sbc_revenue_ratios
            if r["year"] > latest_ratio_year - TREND_WINDOW_YEARS
        ]
        latest_sbc_revenue_ratio = sbc_revenue_ratios[-1]["ratio"]
        if len(window_points) >= TREND_MIN_POINTS:
            trend_window = [window_points[0][0], window_points[-1][0]]
            change = _fitted_change(window_points)
            sbc_revenue_trend_change = round(change, 4)

            if change > TREND_THRESHOLD:  # increased > 1 pct point
                sbc_revenue_trend = "INCREASING"
                sbc_flags.append("SBC_ACCELERATING")
            elif change < -TREND_THRESHOLD:  # decreased > 1 pct point
                sbc_revenue_trend = "DECREASING"
                sbc_flags.append("SBC_IMPROVING")
            else:
                sbc_revenue_trend = "STABLE"

    # A current burden is a level measurement, independent of the three-point
    # history needed to describe its trend.
    if latest_sbc_revenue_ratio is not None and latest_sbc_revenue_ratio > 0.08:
        sbc_flags.append("SBC_BURDEN_EXTREME")

    # ── Shares YoY changes ─────────────────────────────────────────────────
    shares_years = sorted(shares_map.keys())
    for i in range(1, len(shares_years)):
        prev_yr, curr_yr = shares_years[i - 1], shares_years[i]
        if curr_yr != prev_yr + 1:
            continue
        prev_shares = shares_map[prev_yr]
        curr_shares = shares_map[curr_yr]
        if prev_shares > 0 and curr_shares > 0:
            entry: dict[str, Any] = {"year": curr_yr, "shares": curr_shares}
            if is_share_count_break(prev_shares, curr_shares):
                factor = corroborating_split_ratio(prev_shares, curr_shares, curr_yr, split_rows)
                if factor is None:
                    shares_count_breaks.append(curr_yr)
                    continue
                prev_shares = prev_shares * factor
                entry["split_factor"] = factor
            change_pct = (curr_shares - prev_shares) / prev_shares
            entry["change_pct"] = round(change_pct, 4)
            shares_yoy_changes.append(entry)
    if shares_count_breaks:
        sbc_flags.append(REASON_SHARE_COUNT_BREAK_UNCORROBORATED)

    # ── Net dilution despite buybacks ──────────────────────────────────────
    if len(shares_yoy_changes) >= 2:
        # Group the rising years into runs of adjacent fiscal years. Only a run of
        # 2+ years is a dilution streak, and only a buyback inside such a run is a
        # buyback "despite" which the count kept rising: a repurchase in an isolated
        # rising year elsewhere in the history does not credit a later streak.
        streaks: list[list[int]] = []
        current: list[int] = []
        previous_year: int | None = None
        for entry in shares_yoy_changes:
            if previous_year is not None and entry["year"] != previous_year + 1:
                current = []
            previous_year = entry["year"]
            if entry["change_pct"] > 0:
                if not current:
                    streaks.append(current)
                current.append(entry["year"])
            else:
                current = []

        if any(
            len(streak) >= 2 and any(repurchases_map.get(yr, 0) > 0 for yr in streak)
            for streak in streaks
        ):
            sbc_flags.append("NET_DILUTION_DESPITE_BUYBACKS")

    return {
        "sbc_revenue_ratios": sbc_revenue_ratios,
        "sbc_revenue_trend": sbc_revenue_trend,
        "shares_yoy_changes": shares_yoy_changes,
        "shares_count_breaks": shares_count_breaks,
        "sbc_flags": sbc_flags,
        "latest_sbc_revenue_ratio": latest_sbc_revenue_ratio,
        "sbc_revenue_trend_change": sbc_revenue_trend_change,
        "sbc_revenue_trend_window": trend_window,
    }
