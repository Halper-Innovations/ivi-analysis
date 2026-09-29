"""Detect and classify disagreements between valuation methods.

When DCF and EPV disagree, the gap is information: DCF includes growth
assumptions, EPV values current earnings at perpetuity. The difference
is the market's growth premium. If EPV/DCF < 0.5, more than half the
DCF value is growth expectation — a fragile assumption.

Tension types:
  NONE — methods agree (all overvalued or all undervalued)
  GROWTH_VS_EARNINGS_POWER — DCF > EPV significantly (growth dependency)
  ASSET_VS_EARNINGS — NCAV disagrees with earnings methods
  INSUFFICIENT_METHODS — fewer than 2 methods produced values
"""
from __future__ import annotations

from typing import Any


def _compute_assumption_sensitivity(
    methods: dict[str, float],
    current_price: float | None,
    wacc: float,
    terminal_growth: float,
) -> dict[str, str]:
    """Quantify how sensitive each method is to key assumption changes."""
    result: dict[str, str] = {}
    price = current_price or 1.0

    # DCF sensitivity to growth: d(DCF)/d(g) ≈ DCF / (wacc - g) per 1pp
    dcf = methods.get("dcf")
    spread = wacc - terminal_growth
    if dcf is not None and spread > 0.005:
        delta_per_pp = dcf * 0.01 / spread
        pct_of_price = abs(delta_per_pp) / price if price > 0 else 0
        severity = "HIGH" if pct_of_price > 0.02 else "MODERATE" if pct_of_price > 0.005 else "LOW"
        result["dcf_to_growth"] = f"{severity} — 1pp growth change ≈ ${abs(delta_per_pp):.2f}/share DCF impact"

    # EPV sensitivity to margin: d(EPV)/d(margin) ≈ EPV * 0.01 / wacc
    epv = methods.get("epv")
    if epv is not None and wacc > 0:
        delta_per_pp = epv * 0.01 / wacc
        pct_of_price = abs(delta_per_pp) / price if price > 0 else 0
        severity = "HIGH" if pct_of_price > 0.02 else "MODERATE" if pct_of_price > 0.005 else "LOW"
        result["epv_to_margin"] = f"{severity} — 1pp margin change ≈ ${abs(delta_per_pp):.2f}/share EPV impact"

    return result


def _build_adjustment_reasoning(
    tension_type: str,
    methods: dict[str, float],
    intrinsic_range: dict[str, float | None],
    growth_dependency_ratio: float | None,
    method_count: int,
) -> str:
    """Generate plain-English reasoning for the intrinsic range."""
    low = intrinsic_range.get("low")
    high = intrinsic_range.get("high")
    rng = f"${low:.0f}–${high:.0f}" if low is not None and high is not None else "unknown"

    if tension_type == "INSUFFICIENT_METHODS":
        return f"Only {method_count} method(s) produced values. Cannot assess cross-method tension."

    if tension_type == "GROWTH_VS_EARNINGS_POWER":
        dcf = methods.get("dcf", 0)
        epv = methods.get("epv", 0)
        gap = dcf - epv
        gdr = growth_dependency_ratio or 0
        return (
            f"The intrinsic range of {rng} reflects the gap between EPV "
            f"(${epv:.0f}, current earnings at perpetuity) and DCF (${dcf:.0f}, "
            f"assumes growth). The ${gap:.0f}/share difference ({gdr:.0%}) is growth "
            f"premium — if growth stalls, intrinsic value drops to ${epv:.0f}."
        )

    if tension_type == "ASSET_VS_EARNINGS":
        ncav = methods.get("ncav", 0)
        earnings_methods = {k: v for k, v in methods.items() if k != "ncav"}
        avg_earnings = sum(earnings_methods.values()) / max(1, len(earnings_methods))
        return (
            f"The intrinsic range of {rng} spans asset value (NCAV ${ncav:.0f}) "
            f"and earnings-derived estimates (avg ${avg_earnings:.0f}). The gap suggests "
            f"the company may be worth more in liquidation than as a going concern, "
            f"or that current earnings are temporarily depressed."
        )

    # NONE — methods agree
    return (
        f"Methods agree within a narrow range of {rng}, suggesting limited "
        f"assumption sensitivity. Valuation is relatively robust across approaches."
    )


def analyze_method_tensions(
    *,
    dcf_value: float | None,
    epv_value: float | None,
    graham_value: float | None,
    ncav_value: float | None,
    current_price: float | None,
    revenue_cagr_5y: float | None,
    wacc: float,
    terminal_growth: float,
) -> dict[str, Any]:
    """Analyze tensions between valuation method outputs.

    All values are per-share. Returns a dict with:
      - methods_agree: bool
      - tension_type: str
      - tension_description: str
      - consensus_direction: UNDERVALUED / OVERVALUED / MIXED / UNKNOWN
      - consensus_strength: int (how many methods agree on direction)
      - sensitivity: dict with growth_dependency_ratio etc.
      - intrinsic_range: {low, mid, high}
    """
    # Collect available methods
    methods: dict[str, float] = {}
    if dcf_value is not None:
        methods["dcf"] = dcf_value
    if epv_value is not None:
        methods["epv"] = epv_value
    if graham_value is not None:
        methods["graham"] = graham_value
    if ncav_value is not None:
        methods["ncav"] = ncav_value

    method_count = len(methods)

    if method_count < 2:
        return {
            "methods_agree": method_count == 1,
            "tension_type": "INSUFFICIENT_METHODS",
            "tension_description": f"Only {method_count} method(s) produced values.",
            "method_count": method_count,
            "consensus_direction": "UNKNOWN",
            "consensus_strength": method_count,
            "sensitivity": {},
            "intrinsic_range": _intrinsic_range(methods),
            "overvalued_count": 0,
            "undervalued_count": 0,
            "assumption_sensitivity": {},
            "adjustment_reasoning": f"Only {method_count} method(s) produced values. Cannot assess cross-method tension.",
        }

    # Compute discounts (positive = undervalued)
    discounts: dict[str, float | None] = {}
    if current_price and current_price > 0:
        for name, value in methods.items():
            if value > 0:
                discounts[name] = (value - current_price) / value
            else:
                discounts[name] = -1.0
    else:
        for name in methods:
            discounts[name] = None

    # Count directions
    undervalued = [n for n, d in discounts.items() if d is not None and d > 0.10]
    overvalued = [n for n, d in discounts.items() if d is not None and d < -0.10]
    fair = [n for n in discounts if n not in undervalued and n not in overvalued and discounts[n] is not None]

    # Consensus. Direction is measured against the price; with no price there
    # is no direction to agree on, so agreement is UNKNOWN, not FAIR — reading
    # the missing measurement as agreement handed conviction 25 of 25 points
    # for agreement nobody measured.
    if all(d is None for d in discounts.values()):
        consensus = "UNKNOWN"
        strength = 0
        agree = False
    elif len(undervalued) >= 2 and not overvalued:
        consensus = "UNDERVALUED"
        strength = len(undervalued)
        agree = True
    elif len(overvalued) >= 2 and not undervalued:
        consensus = "OVERVALUED"
        strength = len(overvalued)
        agree = True
    elif not undervalued and not overvalued:
        consensus = "FAIR"
        strength = len(fair)
        agree = True
    else:
        consensus = "MIXED"
        strength = max(len(undervalued), len(overvalued))
        agree = False

    # Growth dependency: EPV/DCF ratio
    growth_dependency_ratio = None
    if dcf_value and epv_value and dcf_value > 0:
        growth_dependency_ratio = 1.0 - (epv_value / dcf_value)
        # If ratio > 0.5, more than half the DCF is growth premium

    sensitivity = {}
    if growth_dependency_ratio is not None:
        sensitivity["growth_dependency_ratio"] = round(growth_dependency_ratio, 3)
        if dcf_value and epv_value:
            sensitivity["growth_value_per_share"] = round(dcf_value - epv_value, 2)
            sensitivity["earnings_power_per_share"] = round(epv_value, 2)

    # Classify tension
    tension_type = "NONE"
    tension_description = "Methods are broadly consistent."

    # GROWTH_VS_EARNINGS_POWER: DCF materially above EPV
    if dcf_value and epv_value and growth_dependency_ratio is not None and growth_dependency_ratio > 0.40:
        # Check if price exploits this tension
        dcf_disc = discounts.get("dcf")
        epv_disc = discounts.get("epv")
        if dcf_disc is not None and epv_disc is not None and dcf_disc > 0 and epv_disc < 0:
            tension_type = "GROWTH_VS_EARNINGS_POWER"
            growth_val = dcf_value - epv_value
            tension_description = (
                f"DCF (${dcf_value:.0f}) assumes growth, but EPV (${epv_value:.0f}) values "
                f"current earnings only. ${growth_val:.0f}/share ({growth_dependency_ratio:.0%}) "
                f"of DCF value is growth premium. If growth stalls, intrinsic value "
                f"drops to ${epv_value:.0f}."
            )
        elif growth_dependency_ratio > 0.60:
            tension_type = "GROWTH_VS_EARNINGS_POWER"
            growth_val = dcf_value - epv_value
            tension_description = (
                f"DCF (${dcf_value:.0f}) is {growth_dependency_ratio:.0%} growth value vs "
                f"EPV (${epv_value:.0f}). Growth dependency is high — "
                f"${growth_val:.0f}/share depends on sustained {(revenue_cagr_5y or 0):.1%} growth."
            )

    # If tension detected, methods do not agree
    if tension_type != "NONE":
        agree = False

    # ASSET_VS_EARNINGS: NCAV disagrees with earnings methods
    if tension_type == "NONE" and ncav_value is not None and ncav_value > 0:
        earnings_avg = sum(v for k, v in methods.items() if k != "ncav") / max(1, method_count - 1)
        if ncav_value > earnings_avg * 1.5 or (current_price and ncav_value > current_price and earnings_avg < current_price * 0.5):
            tension_type = "ASSET_VS_EARNINGS"
            tension_description = (
                f"NCAV (${ncav_value:.0f}) shows asset value above earnings-derived "
                f"estimates (avg ${earnings_avg:.0f}). Company may be worth more dead "
                f"than alive, or earnings are temporarily depressed."
            )

    intrinsic_range = _intrinsic_range(methods)

    return {
        "methods_agree": agree,
        "tension_type": tension_type,
        "tension_description": tension_description,
        "method_count": method_count,
        "consensus_direction": consensus,
        "consensus_strength": strength,
        "overvalued_count": len(overvalued),
        "undervalued_count": len(undervalued),
        "growth_value_pct": growth_dependency_ratio if growth_dependency_ratio and growth_dependency_ratio > 0 else 0,
        "sensitivity": sensitivity,
        "intrinsic_range": intrinsic_range,
        "method_values": {k: round(v, 2) for k, v in methods.items()},
        "discounts": {k: round(v, 3) for k, v in discounts.items() if v is not None},
        "assumption_sensitivity": _compute_assumption_sensitivity(methods, current_price, wacc, terminal_growth),
        "adjustment_reasoning": _build_adjustment_reasoning(
            tension_type, methods, intrinsic_range, growth_dependency_ratio, method_count,
        ),
    }


def _intrinsic_range(methods: dict[str, float]) -> dict[str, float | None]:
    """Compute low/mid/high from available method values."""
    values = [v for v in methods.values() if v is not None]
    if not values:
        return {"low": None, "mid": None, "high": None}
    return {
        "low": min(values),
        "mid": sum(values) / len(values),
        "high": max(values),
    }
