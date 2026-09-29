"""Tests for app.valuation.method_tension."""
from __future__ import annotations

import pytest


def test_all_methods_agree_overvalued():
    """When all methods say overvalued, no tension."""
    from app.valuation.method_tension import analyze_method_tensions
    result = analyze_method_tensions(
        dcf_value=67.0, epv_value=56.0, graham_value=27.0, ncav_value=-5.0,
        current_price=253.0, revenue_cagr_5y=0.087, wacc=0.10, terminal_growth=0.015,
    )
    assert result["methods_agree"] is True
    assert result["tension_type"] == "NONE"
    assert result["overvalued_count"] == 4


def test_growth_vs_earnings_power_tension():
    """DCF says undervalued but EPV says overvalued → growth dependency."""
    from app.valuation.method_tension import analyze_method_tensions
    result = analyze_method_tensions(
        dcf_value=210.0, epv_value=39.0, graham_value=50.0, ncav_value=None,
        current_price=190.0, revenue_cagr_5y=0.12, wacc=0.10, terminal_growth=0.015,
    )
    assert result["methods_agree"] is False
    assert result["tension_type"] == "GROWTH_VS_EARNINGS_POWER"
    assert "growth" in result["tension_description"].lower()
    assert result["growth_value_pct"] > 0.5  # >50% of DCF is growth value


def test_asset_vs_earnings_tension():
    """NCAV shows net-net value but earnings methods say overvalued."""
    from app.valuation.method_tension import analyze_method_tensions
    result = analyze_method_tensions(
        dcf_value=5.0, epv_value=3.0, graham_value=None, ncav_value=12.0,
        current_price=8.0, revenue_cagr_5y=-0.05, wacc=0.10, terminal_growth=0.015,
    )
    assert result["tension_type"] == "ASSET_VS_EARNINGS"


def test_all_undervalued_consensus():
    """When multiple methods agree on undervaluation, high consensus."""
    from app.valuation.method_tension import analyze_method_tensions
    result = analyze_method_tensions(
        dcf_value=150.0, epv_value=120.0, graham_value=100.0, ncav_value=None,
        current_price=80.0, revenue_cagr_5y=0.05, wacc=0.10, terminal_growth=0.015,
    )
    assert result["methods_agree"] is True
    assert result["consensus_direction"] == "UNDERVALUED"
    assert result["consensus_strength"] >= 3  # 3 methods agree


def test_sensitivity_computation():
    """Should compute how sensitive DCF is to growth assumption changes."""
    from app.valuation.method_tension import analyze_method_tensions
    result = analyze_method_tensions(
        dcf_value=210.0, epv_value=39.0, graham_value=50.0, ncav_value=None,
        current_price=190.0, revenue_cagr_5y=0.12, wacc=0.10, terminal_growth=0.015,
    )
    sens = result.get("sensitivity", {})
    # Growth sensitivity should be computed
    assert "growth_dependency_ratio" in sens
    # EPV/DCF ratio shows how much is growth vs current earnings
    assert sens["growth_dependency_ratio"] > 0


def test_insufficient_methods():
    """With only 1 method, tension analysis should report insufficient."""
    from app.valuation.method_tension import analyze_method_tensions
    result = analyze_method_tensions(
        dcf_value=None, epv_value=None, graham_value=50.0, ncav_value=None,
        current_price=40.0, revenue_cagr_5y=0.05, wacc=0.10, terminal_growth=0.015,
    )
    assert result["tension_type"] == "INSUFFICIENT_METHODS"
    assert result["method_count"] == 1


def test_no_price():
    """Without a price, can still analyze method agreement."""
    from app.valuation.method_tension import analyze_method_tensions
    result = analyze_method_tensions(
        dcf_value=100.0, epv_value=40.0, graham_value=60.0, ncav_value=None,
        current_price=None, revenue_cagr_5y=0.08, wacc=0.10, terminal_growth=0.015,
    )
    # Can still detect growth vs earnings tension
    assert result["tension_type"] in ("GROWTH_VS_EARNINGS_POWER", "NONE")
    # Growth dependency from EPV/DCF ratio
    assert "growth_dependency_ratio" in result.get("sensitivity", {})
    # ...but direction agreement is unmeasured without a price.
    assert result["consensus_direction"] == "UNKNOWN"
    assert result["consensus_strength"] == 0
    assert result["methods_agree"] is False


def test_intrinsic_range():
    """Should produce a range from min to max of all methods."""
    from app.valuation.method_tension import analyze_method_tensions
    result = analyze_method_tensions(
        dcf_value=150.0, epv_value=80.0, graham_value=100.0, ncav_value=20.0,
        current_price=90.0, revenue_cagr_5y=0.06, wacc=0.10, terminal_growth=0.015,
    )
    r = result["intrinsic_range"]
    assert r["low"] == 20.0  # NCAV
    assert r["high"] == 150.0  # DCF
    assert r["mid"] == pytest.approx((20 + 80 + 100 + 150) / 4, rel=0.01)


def test_assumption_sensitivity_growth():
    """assumption_sensitivity should quantify DCF sensitivity to growth changes."""
    from app.valuation.method_tension import analyze_method_tensions
    result = analyze_method_tensions(
        dcf_value=210.0, epv_value=39.0, graham_value=50.0, ncav_value=None,
        current_price=190.0, revenue_cagr_5y=0.12, wacc=0.10, terminal_growth=0.015,
    )
    sens = result.get("assumption_sensitivity", {})
    assert sens["dcf_to_growth"] == "HIGH — 1pp growth change ≈ $24.71/share DCF impact"


def test_assumption_sensitivity_epv_margin():
    """assumption_sensitivity should include EPV margin sensitivity when EPV is available."""
    from app.valuation.method_tension import analyze_method_tensions
    result = analyze_method_tensions(
        dcf_value=150.0, epv_value=120.0, graham_value=100.0, ncav_value=None,
        current_price=80.0, revenue_cagr_5y=0.05, wacc=0.10, terminal_growth=0.015,
    )
    sens = result.get("assumption_sensitivity", {})
    assert sens["epv_to_margin"] == "HIGH — 1pp margin change ≈ $12.00/share EPV impact"


def test_adjustment_reasoning_growth_tension():
    """adjustment_reasoning should explain the growth vs earnings power gap."""
    from app.valuation.method_tension import analyze_method_tensions
    result = analyze_method_tensions(
        dcf_value=210.0, epv_value=39.0, graham_value=50.0, ncav_value=None,
        current_price=190.0, revenue_cagr_5y=0.12, wacc=0.10, terminal_growth=0.015,
    )
    reasoning = result.get("adjustment_reasoning", "")
    assert "$39" in reasoning
    assert "$210" in reasoning
    assert "$171" in reasoning
    assert "81%" in reasoning


def test_adjustment_reasoning_no_tension():
    """adjustment_reasoning should describe agreement when methods agree."""
    from app.valuation.method_tension import analyze_method_tensions
    result = analyze_method_tensions(
        dcf_value=150.0, epv_value=120.0, graham_value=100.0, ncav_value=None,
        current_price=80.0, revenue_cagr_5y=0.05, wacc=0.10, terminal_growth=0.015,
    )
    reasoning = result.get("adjustment_reasoning", "")
    assert len(reasoning) > 0


def test_adjustment_reasoning_insufficient():
    """adjustment_reasoning should note insufficient methods."""
    from app.valuation.method_tension import analyze_method_tensions
    result = analyze_method_tensions(
        dcf_value=None, epv_value=None, graham_value=50.0, ncav_value=None,
        current_price=40.0, revenue_cagr_5y=0.05, wacc=0.10, terminal_growth=0.015,
    )
    reasoning = result.get("adjustment_reasoning", "")
    assert "1" in reasoning or "insufficient" in reasoning.lower()
