"""app/valuation/reinvestment_efficiency.py labels whole-history statistics as 5y and 3y.

The output keys are
``revenue_cagr_5y_proxy`` … ``shares_cagr_5y_proxy`` and
``capex_burden_vs_cfo_median_3y`` (reinvestment_efficiency.py:497-501), but
``_series_cagr`` compounds from the first numeric row to the last with no window
(159-179) and ``_capex_burden`` takes the median over every row (182-200). Observed on
ten rows whose revenue is flat for five years and then grows 10% a year for five:
the "5y" figure is 0.0544 (a nine-interval rate); the five-year rate is 0.10. Observed
on ten rows whose capex burden is 0.5 for seven years and 0.2 for the last three: the
"median_3y" is 0.5. The window the label promises is the window the number must use.
"""

from __future__ import annotations

import math

from app.valuation.reinvestment_efficiency import compute_reinvestment_efficiency


def _rows() -> list[dict]:
    rows = []
    revenue = 100.0
    for year in range(2016, 2026):
        if year > 2020:
            revenue *= 1.10
        cfo = 10.0
        capex = 2.0 if year >= 2023 else 5.0
        rows.append(
            {
                "year": year,
                "revenue": round(revenue, 6),
                "cfo": cfo,
                "fcf": cfo - capex,
                "capex": capex,
                "shares_outstanding": 10.0,
            }
        )
    return rows


def _compute() -> dict:
    return compute_reinvestment_efficiency(
        "TST",
        "2026-01-01",
        fundamentals={"rows": _rows()},
        owner_quality_payload={},
        intangible_payload={},
        maintenance_capex_payload={},
        capital_allocation_discipline_payload={},
        evidence_sufficiency_payload={},
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
        row_derived_from=[],
        cfg=None,
    )


def test_the_five_year_growth_rate_is_measured_over_five_years():
    out = _compute()
    assert math.isclose(out["revenue_cagr_5y_proxy"], 0.10, rel_tol=1e-6)


def test_the_three_year_capex_burden_is_the_median_of_the_last_three_years():
    out = _compute()
    assert math.isclose(out["capex_burden_vs_cfo_median_3y"], 0.20, rel_tol=1e-9)
