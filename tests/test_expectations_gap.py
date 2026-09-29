"""Unit tests for app/valuation/expectations_gap.py.

Exact literal assertions only; no values recomputed from the logic under test.
"""

from app.fundamentals.normalize import UNKNOWN
from app.valuation.expectations_gap import (
    compute_expectations_gap,
    estimate_supportable_growth,
)


def test_estimate_supportable_growth_revenue_cagr_uncapped():
    assert estimate_supportable_growth(
        revenue_cagr_5y=0.06,
        owner_earnings_cagr_5y=None,
        quality_flags=[],
    ) == (0.06, "REVENUE_CAGR_5Y")


def test_estimate_supportable_growth_revenue_cagr_capped():
    assert estimate_supportable_growth(
        revenue_cagr_5y=0.30,
        owner_earnings_cagr_5y=None,
        quality_flags=[],
    ) == (0.15, "REVENUE_CAGR_5Y_CAPPED")


def test_estimate_supportable_growth_quality_haircut():
    assert estimate_supportable_growth(
        revenue_cagr_5y=0.10,
        owner_earnings_cagr_5y=0.10,
        quality_flags=["EARNINGS_QUALITY_HEADWIND"],
    ) == (0.05, "CAGR_QUALITY_HAIRCUT")


def test_estimate_supportable_growth_unknown():
    assert estimate_supportable_growth(
        revenue_cagr_5y=None,
        owner_earnings_cagr_5y=None,
        quality_flags=[],
    ) == (None, "SUPPORTABLE_GROWTH_UNKNOWN")


def test_compute_expectations_gap_expensive():
    result = compute_expectations_gap(
        implied_growth=0.12,
        supportable_growth=0.06,
        implied_growth_saturated=False,
    )
    assert result["gap"] == 0.06
    assert result["bucket"] == "EXPENSIVE_VS_EXPECTATIONS"


def test_compute_expectations_gap_cheap():
    result = compute_expectations_gap(
        implied_growth=0.04,
        supportable_growth=0.10,
        implied_growth_saturated=False,
    )
    assert result["gap"] == -0.06
    assert result["bucket"] == "CHEAP_VS_EXPECTATIONS"


def test_compute_expectations_gap_fairly_priced():
    result = compute_expectations_gap(
        implied_growth=0.08,
        supportable_growth=0.06,
        implied_growth_saturated=False,
    )
    assert result["gap"] == 0.02
    assert result["bucket"] == "FAIRLY_PRICED_EXPECTATIONS"


def test_compute_expectations_gap_saturated_unreliable():
    result = compute_expectations_gap(
        implied_growth=0.60,
        supportable_growth=0.06,
        implied_growth_saturated=True,
    )
    assert result["bucket"] == "EXPECTATIONS_GAP_UNRELIABLE"


def test_compute_expectations_gap_unknown_implied_unreliable():
    result = compute_expectations_gap(
        implied_growth=UNKNOWN,
        supportable_growth=0.06,
        implied_growth_saturated=False,
    )
    assert result["bucket"] == "EXPECTATIONS_GAP_UNRELIABLE"


def test_compute_expectations_gap_missing_supportable_unreliable():
    result = compute_expectations_gap(
        implied_growth=0.10,
        supportable_growth=None,
        implied_growth_saturated=False,
    )
    assert result["bucket"] == "EXPECTATIONS_GAP_UNRELIABLE"
