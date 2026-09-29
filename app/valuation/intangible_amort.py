"""Intangible amortization add-back for EPV / normalized earnings.

Problem
-------
For serial acquirers (COLL, JAZZ, ANIP, and essentially every pharma / industrial
roll-up that's done M&A in the last 10 years), GAAP operating income is crushed
by purchase-price intangible amortization — a non-cash charge allocating past
acquisition costs over a 5-15 year useful life. When EPV = NOPAT / WACC is
computed from GAAP operating income, it treats this non-cash charge as if it
were a real economic expense, producing systematically suppressed or even
negative EPVs that don't reflect the cash earning power of the business.

Symptom in the wild (healthcare_pharma_2026-04-14 scan): COLL ($222M/yr
intangible amort), ANIP ($91M D&A mostly intangibles), and JAZZ (heavy GW
Pharma amort) all showed deeply suppressed or negative EPVs that the AI had
to explicitly correct during Stage 4 analysis.

Fix
---
Estimate intangible amortization from the gap between D&A and capex, bounded
by balance-sheet intangibles so we don't over-credit companies with modest
intangible bases. Add back this number to operating income before computing
NOPAT / WACC.

Heuristic (conservative):
    intangible_amort_addback = min(
        max(0, D&A − capex),               # excess D&A is the proxy
        0.10 × intangible_assets,           # capped at ~10-year useful life
    )

The module exposes pure functions so callers can:
- Compute the add-back per year from XBRL facts
- Produce an "intangible amort adjusted" operating income series for EPV
- Read the company-level metadata (e.g. "significant amortization distortion")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Conservative useful-life assumption for intangible asset base.
# 10 years is the SEC-typical range for customer relationships, developed
# technology, and trademarks in purchase-price allocation. Pharma patents can
# be shorter (5-7 years), software longer (3-5 years). 10% is a safe midpoint
# that won't materially over-credit any single company.
_INTANGIBLE_USEFUL_LIFE_RATIO = 0.10


@dataclass
class IntangibleAmortAddBack:
    """Result of the intangible amort computation for a single fiscal year."""

    fiscal_year: int
    d_and_a: float | None
    capex: float | None
    intangible_assets: float | None
    excess_da_over_capex: float  # max(0, D&A - capex)
    intangible_cap: float  # 10% of intangibles
    addback: float  # min(excess, cap), or 0 if insufficient data
    # Explains why the addback is what it is — useful for auditing the scorecard
    reason: str


@dataclass
class IntangibleAmortSeries:
    """Multi-year result + distortion assessment for a ticker."""

    ticker: str
    years: list[IntangibleAmortAddBack] = field(default_factory=list)
    # Is the company materially affected by intangible amort?
    # True when multi-year average addback exceeds 5% of revenue (rough rule of
    # thumb: meaningful distortion to operating margins).
    is_materially_distorted: bool = False
    average_addback: float = 0.0
    average_addback_pct_of_revenue: float | None = None


def compute_single_year_addback(
    *,
    fiscal_year: int,
    d_and_a: float | None,
    capex: float | None,
    intangible_assets: float | None,
    intangible_amortization: float | None = None,
) -> IntangibleAmortAddBack:
    """Compute intangible amortization add-back for one fiscal year.

    If `intangible_amortization` is provided (i.e., the company reports
    `AmortizationOfIntangibleAssets` directly via XBRL), use that value
    verbatim — it's the ground-truth answer. Otherwise fall back to the
    (D&A − capex) heuristic capped at 10% of intangibles.

    Returns a zero add-back with an explanatory reason when inputs are missing
    or the heuristic doesn't fire.
    """
    # Direct path: company reports intangible amort separately. This is the
    # most accurate answer — no estimation needed.
    if isinstance(intangible_amortization, (int, float)) and intangible_amortization >= 0:
        return IntangibleAmortAddBack(
            fiscal_year=fiscal_year,
            d_and_a=d_and_a,
            capex=capex,
            intangible_assets=intangible_assets,
            excess_da_over_capex=0.0,
            intangible_cap=0.0,
            addback=float(intangible_amortization),
            reason="DIRECT_FROM_AMORTIZATION_OF_INTANGIBLE_ASSETS",
        )

    if d_and_a is None or capex is None:
        return IntangibleAmortAddBack(
            fiscal_year=fiscal_year,
            d_and_a=d_and_a,
            capex=capex,
            intangible_assets=intangible_assets,
            excess_da_over_capex=0.0,
            intangible_cap=0.0,
            addback=0.0,
            reason="INSUFFICIENT_DATA_DA_OR_CAPEX",
        )

    capex_abs = abs(capex)  # capex sometimes comes through as negative
    excess = max(0.0, d_and_a - capex_abs)

    if excess <= 0:
        return IntangibleAmortAddBack(
            fiscal_year=fiscal_year,
            d_and_a=d_and_a,
            capex=capex,
            intangible_assets=intangible_assets,
            excess_da_over_capex=0.0,
            intangible_cap=0.0,
            addback=0.0,
            reason="NO_EXCESS_DA_OVER_CAPEX",
        )

    if intangible_assets is None or intangible_assets <= 0:
        # Excess D&A exists but no intangible assets on balance sheet — the
        # excess is probably from software/accelerated depreciation, not
        # purchase-price amortization. Don't add back.
        return IntangibleAmortAddBack(
            fiscal_year=fiscal_year,
            d_and_a=d_and_a,
            capex=capex,
            intangible_assets=intangible_assets,
            excess_da_over_capex=excess,
            intangible_cap=0.0,
            addback=0.0,
            reason="NO_MATERIAL_INTANGIBLES",
        )

    cap = _INTANGIBLE_USEFUL_LIFE_RATIO * intangible_assets
    addback = min(excess, cap)
    reason = (
        "CAPPED_BY_INTANGIBLE_USEFUL_LIFE" if excess > cap
        else "EXCESS_DA_BELOW_INTANGIBLE_CAP"
    )
    return IntangibleAmortAddBack(
        fiscal_year=fiscal_year,
        d_and_a=d_and_a,
        capex=capex,
        intangible_assets=intangible_assets,
        excess_da_over_capex=excess,
        intangible_cap=cap,
        addback=addback,
        reason=reason,
    )


def compute_series(
    *,
    ticker: str,
    years_data: list[dict[str, Any]],
    revenue_by_year: dict[int, float] | None = None,
) -> IntangibleAmortSeries:
    """Compute the multi-year intangible amort add-back series.

    Parameters
    ----------
    ticker : str
    years_data : list of dicts per fiscal year, each with:
        - fiscal_year (int)
        - d_and_a (float | None) — depreciation_amortization
        - capex (float | None)
        - intangible_assets (float | None) — balance sheet intangibles (net)
    revenue_by_year : optional dict {fiscal_year: revenue} used to flag whether
        the distortion is material (>5% of revenue).

    The caller is responsible for pulling these fields from companyfacts_facts.
    """
    years: list[IntangibleAmortAddBack] = []
    for row in years_data:
        years.append(
            compute_single_year_addback(
                fiscal_year=int(row["fiscal_year"]),
                d_and_a=row.get("d_and_a"),
                capex=row.get("capex"),
                intangible_assets=row.get("intangible_assets"),
                intangible_amortization=row.get("intangible_amortization"),
            )
        )

    # A missing basis is not a measured zero. Include actual zero-charge years
    # while keeping unsupported estimates out of either aggregation operand.
    known_years = [
        y for y in years
        if y.reason in {
            "DIRECT_FROM_AMORTIZATION_OF_INTANGIBLE_ASSETS",
            "NO_EXCESS_DA_OVER_CAPEX",
            "CAPPED_BY_INTANGIBLE_USEFUL_LIFE",
            "EXCESS_DA_BELOW_INTANGIBLE_CAP",
        }
        or (y.reason == "NO_MATERIAL_INTANGIBLES" and y.intangible_assets is not None)
    ]
    avg = sum(y.addback for y in known_years) / len(known_years) if known_years else 0.0

    avg_pct_rev: float | None = None
    if revenue_by_year:
        paired = [
            (y.addback, revenue_by_year[y.fiscal_year])
            for y in known_years
            if y.fiscal_year in revenue_by_year
            and revenue_by_year[y.fiscal_year]
            and revenue_by_year[y.fiscal_year] > 0
        ]
        if paired:
            # Match years before forming either operand, including known zero
            # charges; charge-only average divided by all-year revenue overstates it.
            avg_pct_rev = sum(addback for addback, _ in paired) / sum(
                revenue for _, revenue in paired
            )

    is_material = bool(avg_pct_rev is not None and avg_pct_rev >= 0.05)

    return IntangibleAmortSeries(
        ticker=ticker,
        years=years,
        is_materially_distorted=is_material,
        average_addback=avg,
        average_addback_pct_of_revenue=avg_pct_rev,
    )


def adjust_operating_income_series(
    oi_series: list[tuple[int, float]],
    addback_series: IntangibleAmortSeries,
) -> list[tuple[int, float]]:
    """Apply the add-back to an operating income series.

    Returns a new list with (fiscal_year, oi + addback) per year. Years with
    no matching add-back row pass through unchanged.
    """
    addback_by_year = {y.fiscal_year: y.addback for y in addback_series.years}
    return [
        (fy, oi + addback_by_year.get(fy, 0.0))
        for (fy, oi) in oi_series
    ]
