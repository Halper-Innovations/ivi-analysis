"""Independent literal checks for cash-flow magnitudes reaching the published lens."""

import pytest

from app.valuation.lenses import fcf_yield_anchor


@pytest.mark.parametrize("reported_capex", [-20.0, 20.0])
def test_fcf_yield_deducts_capex_magnitude(reported_capex):
    """CFO 100 less spending 20 = 80; equity 80/.08 = 1000; /10 shares = 100."""
    facts = {
        "cfo": [(2025, 100.0), (2024, 100.0), (2023, 100.0)],
        "capex": [(2025, reported_capex), (2024, reported_capex), (2023, reported_capex)],
    }
    result = fcf_yield_anchor(facts, shares=10.0, category="TRADITIONAL_OPERATING")
    assert result["basis"]["fcf_normalized"] == 80.0
    assert result["value_per_share"] == 100.0
    assert result["status"] == "OK"


def test_fcf_yield_refuses_spending_above_cash_flow_even_when_negated():
    """CFO 100 less spending 120 = -20: no positive capitalizable cash flow."""
    facts = {
        "cfo": [(2025, 100.0), (2024, 100.0), (2023, 100.0)],
        "capex": [(2025, -120.0), (2024, -120.0), (2023, -120.0)],
    }
    result = fcf_yield_anchor(facts, shares=10.0, category="TRADITIONAL_OPERATING")
    assert result["basis"]["fcf_normalized"] == -20.0
    assert result["status"] == "FCF_YIELD_NOT_MEANINGFUL"
    assert result["value_per_share"] is None
