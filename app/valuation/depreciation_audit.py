"""Depreciation policy audit.

Detects declining depreciation rates (possible earnings inflation via
useful life extensions) and capex coverage gaps (underinvestment risk).
"""

from __future__ import annotations

from typing import Any

from app.valuation.adjacent_years import trailing_adjacent_run


def compute_depreciation_audit(
    facts: dict[str, list[tuple[int, float]]],
) -> dict[str, Any]:
    """Audit depreciation policy and capex coverage.

    Returns dict with depreciation_rates, capex_coverage_ratios,
    depreciation_flags, latest_depreciation_rate, latest_capex_coverage.
    """
    depreciation_rates: list[dict[str, Any]] = []
    capex_coverage_ratios: list[dict[str, Any]] = []
    depreciation_flags: list[str] = []
    audit_reason_codes: list[str] = []

    def _by_year(key: str) -> dict[int, float]:
        return {int(yr): float(v) for yr, v in (facts.get(key) or [])}

    ppe_map = _by_year("gross_ppe")
    da_map = _by_year("depreciation_amortization")
    capex_map = _by_year("capex")

    # -- Depreciation rates per year ----------------------------------------
    common_rate_years = sorted(set(ppe_map.keys()) & set(da_map.keys()))
    for year in common_rate_years:
        ppe = ppe_map[year]
        da = da_map[year]
        if ppe > 0:
            rate = da / ppe
            depreciation_rates.append({
                "year": year,
                "gross_ppe": ppe,
                "da": da,
                "rate": round(rate, 4),
            })

    # -- Capex coverage ratios per year -------------------------------------
    common_capex_years = sorted(set(capex_map.keys()) & set(da_map.keys()))
    for year in common_capex_years:
        da = da_map[year]
        capex = capex_map[year]
        if da > 0:
            ratio = capex / da
            capex_coverage_ratios.append({
                "year": year,
                "capex": capex,
                "da": da,
                "ratio": round(ratio, 2),
            })

    # -- Depreciation rate trend (recent 3 years only) -----------------------
    # Both trends need three CALENDAR-ADJACENT years ending at the latest year;
    # a gap leaves the trend unassessed (no flag) instead of comparing across it.
    adjacent_rates = trailing_adjacent_run(depreciation_rates, lambda e: e["year"])
    adjacent_coverage = trailing_adjacent_run(capex_coverage_ratios, lambda e: e["year"])
    if (len(depreciation_rates) >= 3 and len(adjacent_rates) < 3) or (
        len(capex_coverage_ratios) >= 3 and len(adjacent_coverage) < 3
    ):
        audit_reason_codes.append("NON_ADJACENT_YEARS")
    if len(adjacent_rates) >= 3:
        recent_rates = adjacent_rates[-3:]
        three_yr_ago_rate = recent_rates[0]["rate"]
        latest_rate = recent_rates[-1]["rate"]
        # Declined > 1.0 pct points over 3 years (e.g., 10% -> 8.5%)
        if three_yr_ago_rate - latest_rate > 0.01:
            depreciation_flags.append("DEPRECIATION_RATE_DECLINING")

    # -- Capex coverage flags -----------------------------------------------
    if len(adjacent_coverage) >= 3:
        recent_3 = adjacent_coverage[-3:]

        # All below 0.8 for 3 consecutive years
        if all(entry["ratio"] < 0.8 for entry in recent_3):
            depreciation_flags.append("CAPEX_BELOW_DEPRECIATION")

        # All above 2.0 for 3 consecutive years
        if all(entry["ratio"] > 2.0 for entry in recent_3):
            depreciation_flags.append("HEAVY_INVESTMENT_CYCLE")

    # -- Latest values ------------------------------------------------------
    latest_depreciation_rate = depreciation_rates[-1]["rate"] if depreciation_rates else None
    latest_capex_coverage = capex_coverage_ratios[-1]["ratio"] if capex_coverage_ratios else None

    return {
        "depreciation_rates": depreciation_rates,
        "capex_coverage_ratios": capex_coverage_ratios,
        "depreciation_flags": depreciation_flags,
        "reason_codes": audit_reason_codes,
        "latest_depreciation_rate": latest_depreciation_rate,
        "latest_capex_coverage": latest_capex_coverage,
    }
