"""Stable share-count selection for per-share valuation.

Pure, dependency-free helper that selects a STABLE share count from the full
FY shares_outstanding series. Detects when the latest FY value deviates >50%
from the trailing-3 median and falls back to that median, returning
(stable_shares, flag).

Used by valuation_writer DCF/EPV share selection so the magnitude/units logic
lives in one testable place.
"""

from __future__ import annotations

from statistics import median

# Latest-FY share count must be within this fractional band of the trailing
# median, else it is treated as an outlier and the median is substituted.
SHARES_OUTLIER_DEVIATION_THRESHOLD = 0.5

# An independent share count within this fractional band of the latest FY
# count corroborates a genuine capital event (split / offering / buyback) —
# the latest count is kept instead of the stale median.
CAPITAL_EVENT_AGREEMENT_TOLERANCE = 0.2

# Two share counts this many times apart (either way) are a scale slip in one of
# them -- a count filed in thousands, or a thousandfold off -- not a capital event:
# the share-count guard's own contradiction multiple (app/market/shares_guard.py
# SHARES_MAX_MOVE; pinned equal by a test rather than imported, so this helper
# stays dependency-free). A move that large is never "corroborated" into the
# divisor, and a corroborating count that far from a stable series is the slip,
# not the series.
SCALE_SLIP_MULTIPLE = 100.0


def _scale_slip(a: float, b: float) -> bool:
    ratio = a / b
    return ratio >= SCALE_SLIP_MULTIPLE or ratio <= 1.0 / SCALE_SLIP_MULTIPLE


def select_stable_shares(
    series: list[tuple[int, float]],
    *,
    corroborating_count: float | None = None,
) -> tuple[float, str | None]:
    """Select a stable share count from a (fiscal_year, shares) series.

    ``series`` is a list of (fiscal_year, shares_millions) tuples in DESCENDING
    fiscal-year order (latest first).

    ``corroborating_count`` is an INDEPENDENT recent share count (e.g. the
    latest as-of-visible quarterly cover-page dei count, in millions). A >50%
    deviation cannot by itself distinguish a corrupt latest-FY fact from a
    genuine capitalization event (reverse split, ATM/secondary, large buyback)
    — audit finding stable-shares-stale-on-capital-events. When the
    corroborating count agrees with the latest FY value (within 20%), the
    event is genuine and the latest count is kept.

    Returns ``(stable_shares, flag)`` where ``flag`` is one of:
      - ``'SHARES_MISSING'`` when the series is empty (shares 0.0)
      - ``'SHARES_SINGLE_YEAR_NO_STABILITY_CHECK'`` when only one year exists and no
        independent count contradicts it
      - ``'SHARES_SINGLE_YEAR_CONTRADICTED'`` when only one year exists and the
        independent count disagrees with it by more than 20% (shares 0.0: refused)
      - ``'SHARES_CAPITAL_EVENT'`` when the latest FY deviates >50% from the
        trailing-3 median but an independent count corroborates it (latest kept)
      - ``'SHARES_LATEST_FY_OUTLIER'`` when the latest FY deviates >50% and no
        corroboration agrees (the median is returned instead) -- including a move of
        SCALE_SLIP_MULTIPLE or more, which is a scale slip even when corroborated
      - ``'SHARES_SERIES_CONTRADICTED'`` when the fiscal-year series is stable but the
        independent count disagrees with the latest FY by more than 20% (a split or a
        large issuance since the last fiscal-year row) -- shares 0.0: refused, like the
        single-year case, rather than dividing by a count the newer one contradicts
      - ``'SHARES_CORROBORATING_COUNT_SCALE_SLIP'`` when the series is stable and the
        independent count is SCALE_SLIP_MULTIPLE or more away from it: the independent
        count is the slip; the latest FY is returned and the slip is named
      - ``None`` when the latest FY value is stable and nothing contradicts it
    """
    if not series:
        return 0.0, "SHARES_MISSING"
    if len(series) == 1:
        latest = series[0][1]
        # One filed year cannot be checked against a trailing median, but it CAN be
        # checked against the independent count the caller passed for exactly that
        # purpose. When the two disagree by more than the capital-event tolerance the
        # lone year is unproven (a seven-year-old count on the point-in-time path was
        # accepted this way, 5x too small, against a fresh cover-page count of 1,024M
        # for a 47M "series"), and the honest answer is no
        # count rather than a stale one.
        if (
            corroborating_count is not None
            and corroborating_count > 0
            and latest > 0
            and abs(corroborating_count - latest) / latest > CAPITAL_EVENT_AGREEMENT_TOLERANCE
        ):
            return 0.0, "SHARES_SINGLE_YEAR_CONTRADICTED"
        return latest, "SHARES_SINGLE_YEAR_NO_STABILITY_CHECK"

    latest = series[0][1]
    # "trailing-3 median": median over the three most-recent fiscal years
    # (including the latest). This median is 16.384636
    # for the BTM fixture and 32.815201 for the BKNG fixture, which only holds
    # when the latest year is included in the 3-year window.
    trailing = [v for _, v in series[:3]]
    med = median(trailing)
    corro = float(corroborating_count) if corroborating_count is not None else 0.0
    independent = corro > 0 and latest > 0
    if med > 0 and abs(latest - med) / med > SHARES_OUTLIER_DEVIATION_THRESHOLD:
        if (
            independent
            and abs(corro - latest) / latest <= CAPITAL_EVENT_AGREEMENT_TOLERANCE
            # A thousandfold move "corroborated" by a count slipped the same way is a
            # scale slip carried into both, not a capital event.
            and not _scale_slip(latest, med)
        ):
            return latest, "SHARES_CAPITAL_EVENT"
        return med, "SHARES_LATEST_FY_OUTLIER"
    if independent:
        # A stable series was never checked against the newer count, so a
        # pre-split count divided post-split quotes without a flag.
        if _scale_slip(corro, latest):
            return latest, "SHARES_CORROBORATING_COUNT_SCALE_SLIP"
        if abs(corro - latest) / latest > CAPITAL_EVENT_AGREEMENT_TOLERANCE:
            return 0.0, "SHARES_SERIES_CONTRADICTED"
    return latest, None
