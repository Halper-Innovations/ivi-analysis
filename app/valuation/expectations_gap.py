"""Canonical expectations-gap signal.

Single authority for turning a reverse-DCF implied growth and a supportable
growth estimate into the "expectations gap" used by both the conviction verdict
path (sector_runtime) and the lead memo section (report_renderer).

Pure module: no DB, no LLM, no network imports.

Frame (Mauboussin/Rappaport "expectations investing"):
    gap = implied_growth - supportable_growth
    negative gap = the market pays for LESS growth than the business can
                   support  -> CHEAP_VS_EXPECTATIONS (bullish)
    positive gap = the market pays for MORE growth than is supportable
                   -> EXPENSIVE_VS_EXPECTATIONS (bearish)
"""

from __future__ import annotations

from app.fundamentals.normalize import UNKNOWN

# Bucket thresholds (5pp symmetric; a product-defining default).
CHEAP_GAP_THRESHOLD = -0.05
EXPENSIVE_GAP_THRESHOLD = 0.05

# Supportable-growth cap mirrors the high-scenario clamp in valuation_writer
# (the 0.15 ceiling on the trailing-CAGR high scenario). The floor mirrors
# the reverse-DCF solver's lower bound.
SUPPORTABLE_GROWTH_CAP = 0.15
SUPPORTABLE_GROWTH_FLOOR = -0.25

# Multiplier applied to supportable growth when a structural quality headwind
# is present (50% haircut).
QUALITY_HAIRCUT = 0.50

# Bucket labels.
BUCKET_CHEAP = "CHEAP_VS_EXPECTATIONS"
BUCKET_FAIRLY_PRICED = "FAIRLY_PRICED_EXPECTATIONS"
BUCKET_EXPENSIVE = "EXPENSIVE_VS_EXPECTATIONS"
BUCKET_UNRELIABLE = "EXPECTATIONS_GAP_UNRELIABLE"

# Structural quality headwinds that trigger the supportable-growth haircut.
# Subset of FALLBACK_STRUCTURAL_VALUATION_HEADWINDS (sector_runtime.py).
_QUALITY_HAIRCUT_FLAGS = {
    "EARNINGS_QUALITY_HEADWIND",
    "ACCOUNTING_QUALITY_HEADWIND",
    "LOW_ACCOUNTING_QUALITY_HEADWIND",
}


def estimate_supportable_growth(
    revenue_cagr_5y: float | None,
    owner_earnings_cagr_5y: float | None,
    quality_flags: list[str] | None,
) -> tuple[float | None, str]:
    """Estimate the growth the business can support, with a basis label.

    Prefers the trailing revenue CAGR, falling back to owner-earnings CAGR.
    Caps at SUPPORTABLE_GROWTH_CAP and floors at SUPPORTABLE_GROWTH_FLOOR
    (the solver's lower bound, NOT 0.0 — flooring negative history at zero
    flipped decliners priced for milder decline into CHEAP_VS_EXPECTATIONS;
    audit: supportable-growth-clamp-sign-bias). The quality haircut applies
    only to POSITIVE supportable growth (halving a negative would RAISE it).

    Returns (supportable_growth, basis_label). When neither input is available
    returns (None, 'SUPPORTABLE_GROWTH_UNKNOWN').
    """
    flags = quality_flags or []

    base = revenue_cagr_5y if revenue_cagr_5y is not None else owner_earnings_cagr_5y
    if base is None:
        return (None, "SUPPORTABLE_GROWTH_UNKNOWN")
    basis_prefix = "REVENUE_CAGR_5Y" if revenue_cagr_5y is not None else "OWNER_EARNINGS_CAGR_5Y"

    capped = min(max(base, SUPPORTABLE_GROWTH_FLOOR), SUPPORTABLE_GROWTH_CAP)
    was_capped = capped != base

    if capped > 0 and any(flag in _QUALITY_HAIRCUT_FLAGS for flag in flags):
        return (round(capped * QUALITY_HAIRCUT, 6), "CAGR_QUALITY_HAIRCUT")

    if was_capped:
        return (capped, basis_prefix + "_CAPPED")
    return (capped, basis_prefix)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compute_expectations_gap(
    implied_growth,
    supportable_growth: float | None,
    implied_growth_saturated: bool,
    *,
    saturated_bound: str | None = None,
    margin_sign: str | None = None,
) -> dict:
    """Bucket the gap between price-implied growth and supportable growth.

    Returns a dict with keys: gap, bucket, supportable_growth,
    implied_growth_saturated, line (+ gap_is_upper_bound for saturated-LOW).

    The signal is silent (EXPECTATIONS_GAP_UNRELIABLE, gap=None) whenever the
    implied leg is UNKNOWN/non-numeric, the supportable leg is missing, or the
    solver saturated at the HIGH bound / on a non-positive margin. A LOW-bound
    saturation with POSITIVE margin and known supportable growth is the
    opposite of unreliable — the price sits below even the worst-case-growth
    value — and is bucketed CHEAP with the gap reported as an UPPER BOUND (gap <= value)
    (audit: sat-conflates-deep-cheap-with-unsolvable).
    """
    if (
        implied_growth_saturated
        and saturated_bound == "LOW"
        and margin_sign == "POSITIVE"
        and supportable_growth is not None
        and _is_number(implied_growth)
        # The bound only establishes gap <= (bound - supportable): CHEAP is
        # claimable only when that ceiling itself clears the CHEAP threshold
        # (review issue 3 — supportable near the -0.25 floor gave gap <= 0
        # labeled CHEAP).
        and round(float(implied_growth) - float(supportable_growth), 4) <= CHEAP_GAP_THRESHOLD
    ):
        gap_upper_bound = round(float(implied_growth) - float(supportable_growth), 4)
        line = (
            f"Market implies <= {implied_growth:.0%} growth (saturated low); "
            f"supportable {supportable_growth:.0%}; gap <= {gap_upper_bound:.0%} "
            f"({BUCKET_CHEAP})"
        )
        return {
            "gap": gap_upper_bound,
            # The true implied growth is at or BELOW the bound, so the true gap is
            # at or below this number: an upper bound, never a floor.
            "gap_is_upper_bound": True,
            "bucket": BUCKET_CHEAP,
            "supportable_growth": supportable_growth,
            "implied_growth_saturated": True,
            "line": line,
        }

    if (
        implied_growth is UNKNOWN
        or not _is_number(implied_growth)
        or supportable_growth is None
        or implied_growth_saturated
    ):
        return {
            "gap": None,
            "bucket": BUCKET_UNRELIABLE,
            "supportable_growth": supportable_growth,
            "implied_growth_saturated": implied_growth_saturated,
            "line": None,
        }

    gap = round(implied_growth - supportable_growth, 4)
    if gap <= CHEAP_GAP_THRESHOLD:
        bucket = BUCKET_CHEAP
    elif gap >= EXPENSIVE_GAP_THRESHOLD:
        bucket = BUCKET_EXPENSIVE
    else:
        bucket = BUCKET_FAIRLY_PRICED

    # Canonical one-liner — the SINGLE authority consumed verbatim by both the
    # verdict path and the lead memo section (report_renderer._render_expectations_gap).
    line = (
        f"Market implies {implied_growth:.0%} growth; "
        f"supportable {supportable_growth:.0%}; gap {gap:.0%} ({bucket})"
    )

    return {
        "gap": gap,
        "bucket": bucket,
        "supportable_growth": supportable_growth,
        "implied_growth_saturated": implied_growth_saturated,
        "line": line,
    }
