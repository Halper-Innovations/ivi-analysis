"""Nonrecurring item detection.

Detects restructuring charges, operating income spikes, and goodwill
impairment from companyfacts data. Produces flags and adjusted earnings
for downstream consumers. Does NOT automatically override valuation inputs.

Two rules:

* Only the most recent ``NONRECURRING_WINDOW_YEARS`` fiscal years can raise a flag or
  appear in ``nonrecurring_years``; a charge from a decade ago is history, not a
  headwind on today's valuation. Adjusted earnings are still built for every year on
  file (an old year's add-back belongs to that old year).
* A restructuring charge is a pre-tax figure. It is added back to operating income in
  full but to net income only net of tax, at the issuer's effective rate for the same
  fiscal year when it is usable, else at ``STATUTORY_TAX_RATE_FALLBACK``. The output
  says which one was used, year by year.
"""

from __future__ import annotations

from typing import Any

from app.valuation.dcf_lite import DEFAULT_TAX_RATE, TAX_RATE_CEILING, TAX_RATE_FLOOR

# Three fiscal years, ending with the latest fiscal year on file.
NONRECURRING_WINDOW_YEARS = 3
# Same blended (federal plus state) statutory rate the DCF uses when no issuer rate exists.
STATUTORY_TAX_RATE_FALLBACK = DEFAULT_TAX_RATE
TAX_RATE_SOURCE_EFFECTIVE = "EFFECTIVE_SAME_PERIOD"
TAX_RATE_SOURCE_STATUTORY = "STATUTORY_FALLBACK"


def _restructuring_tax_rate(
    year: int, tax_map: dict[int, float], pretax_map: dict[int, float]
) -> tuple[float, str]:
    """Same-year effective rate when it is a plausible rate, else the statutory fallback."""
    tax = tax_map.get(year)
    pretax = pretax_map.get(year)
    if tax is not None and pretax is not None and pretax > 0:
        rate = tax / pretax
        if TAX_RATE_FLOOR <= rate <= TAX_RATE_CEILING:
            return rate, TAX_RATE_SOURCE_EFFECTIVE
    return STATUTORY_TAX_RATE_FALLBACK, TAX_RATE_SOURCE_STATUTORY


def detect_nonrecurring_items(
    facts: dict[str, list[tuple[int, float]]],
) -> dict[str, Any]:
    """Detect nonrecurring items and compute adjusted earnings."""
    nonrecurring_years: set[int] = set()
    nonrecurring_flags: list[str] = []
    restructuring_by_year: dict[int, float] = {}
    goodwill_impairment_years: list[int] = []
    adjusted_oi: dict[int, float] = {}
    adjusted_ni: dict[int, float] = {}

    def _by_year(key: str) -> dict[int, float]:
        return {int(yr): float(v) for yr, v in (facts.get(key) or [])}

    oi_map = _by_year("operating_income")
    ni_map = _by_year("net_income")
    rev_map = _by_year("revenue")
    restructuring_map = _by_year("restructuring_charges")
    goodwill_map = _by_year("goodwill")
    tax_map = _by_year("income_tax_expense")
    pretax_map = _by_year("pretax_income")

    # The window ends at the latest fiscal year on file across every series read here.
    all_years = [
        y for m in (oi_map, ni_map, rev_map, restructuring_map, goodwill_map) for y in m
    ]
    latest_year = max(all_years) if all_years else None
    window_first_year = (
        latest_year - NONRECURRING_WINDOW_YEARS + 1 if latest_year is not None else None
    )

    def _in_window(year: int) -> bool:
        return window_first_year is not None and year >= window_first_year

    # Heuristic 1: Explicit restructuring charges (all years kept for the add-back;
    # only in-window years flag)
    for year, charge in restructuring_map.items():
        if charge > 0:
            restructuring_by_year[year] = charge
            if _in_window(year):
                nonrecurring_years.add(year)
                if "RESTRUCTURING_CHARGE_DETECTED" not in nonrecurring_flags:
                    nonrecurring_flags.append("RESTRUCTURING_CHARGE_DETECTED")

    # Heuristic 2: OI spike without revenue change
    oi_years = sorted(oi_map.keys())
    for i in range(1, len(oi_years)):
        prev_yr, curr_yr = oi_years[i - 1], oi_years[i]
        prev_oi, curr_oi = oi_map[prev_yr], oi_map[curr_yr]
        if prev_oi == 0:
            continue
        oi_change = abs(curr_oi - prev_oi) / abs(prev_oi)
        prev_rev = rev_map.get(prev_yr)
        curr_rev = rev_map.get(curr_yr)
        if prev_rev and curr_rev and prev_rev > 0:
            rev_change = abs(curr_rev - prev_rev) / abs(prev_rev)
        else:
            rev_change = None
        if oi_change > 0.30 and rev_change is not None and rev_change < 0.10 and _in_window(curr_yr):
            nonrecurring_years.add(curr_yr)
            if "POSSIBLE_NONRECURRING_OI_SPIKE" not in nonrecurring_flags:
                nonrecurring_flags.append("POSSIBLE_NONRECURRING_OI_SPIKE")

    # Heuristic 3: Goodwill impairment. Only a fall between ADJACENT fiscal years
    # can be dated to one year; a fall across missing years is unattributable.
    # A fall is still only "likely" an impairment (a disposal or currency move also
    # lowers the balance), which is why this flag never produces an add-back.
    gw_years = sorted(goodwill_map.keys())
    for i in range(1, len(gw_years)):
        prev_yr, curr_yr = gw_years[i - 1], gw_years[i]
        if curr_yr != prev_yr + 1:
            continue
        prev_gw, curr_gw = goodwill_map[prev_yr], goodwill_map[curr_yr]
        if prev_gw > 0 and (prev_gw - curr_gw) / prev_gw > 0.10 and _in_window(curr_yr):
            nonrecurring_years.add(curr_yr)
            goodwill_impairment_years.append(curr_yr)
    if goodwill_impairment_years:
        nonrecurring_flags.append("GOODWILL_IMPAIRMENT_LIKELY")

    # Adjusted earnings: add back restructuring charges. The charge is pre-tax, so it
    # goes into operating income in full and into net income net of tax.
    tax_rate_by_year: dict[int, float] = {}
    tax_rate_source_by_year: dict[int, str] = {}
    for year in sorted(set(oi_map.keys()) | set(ni_map.keys())):
        charge = restructuring_by_year.get(year, 0.0)
        if year in oi_map:
            adjusted_oi[year] = oi_map[year] + charge
        if year in ni_map:
            if charge:
                rate, source = _restructuring_tax_rate(year, tax_map, pretax_map)
                tax_rate_by_year[year] = rate
                tax_rate_source_by_year[year] = source
                adjusted_ni[year] = ni_map[year] + charge * (1.0 - rate)
            else:
                adjusted_ni[year] = ni_map[year]

    # Restructuring magnitude
    total_restructuring = sum(restructuring_by_year.values())
    total_oi = sum(abs(v) for v in oi_map.values()) if oi_map else 0.0
    restructuring_magnitude_pct = (
        total_restructuring / total_oi if total_oi > 0 else None
    )

    return {
        "has_nonrecurring": bool(nonrecurring_flags),
        "nonrecurring_years": sorted(nonrecurring_years),
        "nonrecurring_flags": nonrecurring_flags,
        "restructuring_by_year": restructuring_by_year,
        "goodwill_impairment_years": sorted(goodwill_impairment_years),
        "adjusted_operating_income": adjusted_oi,
        "adjusted_net_income": adjusted_ni,
        "restructuring_magnitude_pct": restructuring_magnitude_pct,
        "restructuring_tax_rate_by_year": tax_rate_by_year,
        "restructuring_tax_rate_source_by_year": tax_rate_source_by_year,
        "statutory_tax_rate_fallback": STATUTORY_TAX_RATE_FALLBACK,
        "nonrecurring_window_years": NONRECURRING_WINDOW_YEARS,
        "nonrecurring_window_first_year": window_first_year,
        "nonrecurring_window_last_year": latest_year,
    }
