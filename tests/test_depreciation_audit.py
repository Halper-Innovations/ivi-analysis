"""Tests for app.valuation.depreciation_audit."""

from __future__ import annotations


def test_depreciation_rate_declining():
    """Rate drops >0.5 pct points over 3yr -> DEPRECIATION_RATE_DECLINING."""
    from app.valuation.depreciation_audit import compute_depreciation_audit
    facts = {
        "gross_ppe": [(2023, 1000.0), (2024, 1050.0), (2025, 1100.0)],
        "depreciation_amortization": [(2023, 100.0), (2024, 97.0), (2025, 93.0)],
        "capex": [(2023, 110.0), (2024, 105.0), (2025, 100.0)],
    }
    result = compute_depreciation_audit(facts)
    # Rate: 10.0% -> 9.24% -> 8.45% -- declined 1.55 pct points
    assert "DEPRECIATION_RATE_DECLINING" in result["depreciation_flags"]
    assert result["latest_depreciation_rate"] is not None
    assert result["latest_depreciation_rate"] < 0.09


def test_capex_below_depreciation():
    """capex/DA < 0.8 for 3+ consecutive years -> CAPEX_BELOW_DEPRECIATION."""
    from app.valuation.depreciation_audit import compute_depreciation_audit
    facts = {
        "gross_ppe": [(2023, 1000.0), (2024, 1000.0), (2025, 1000.0)],
        "depreciation_amortization": [(2023, 100.0), (2024, 100.0), (2025, 100.0)],
        "capex": [(2023, 70.0), (2024, 65.0), (2025, 60.0)],
    }
    result = compute_depreciation_audit(facts)
    # Coverage: 0.70, 0.65, 0.60 -- all < 0.8
    assert "CAPEX_BELOW_DEPRECIATION" in result["depreciation_flags"]


def test_heavy_investment_cycle():
    """capex/DA > 2.0 for 3+ consecutive years -> HEAVY_INVESTMENT_CYCLE."""
    from app.valuation.depreciation_audit import compute_depreciation_audit
    facts = {
        "gross_ppe": [(2023, 1000.0), (2024, 1100.0), (2025, 1200.0)],
        "depreciation_amortization": [(2023, 100.0), (2024, 110.0), (2025, 120.0)],
        "capex": [(2023, 250.0), (2024, 280.0), (2025, 300.0)],
    }
    result = compute_depreciation_audit(facts)
    # Coverage: 2.50, 2.55, 2.50 -- all > 2.0; rate stable at 10%
    assert "HEAVY_INVESTMENT_CYCLE" in result["depreciation_flags"]
    # Not a headwind -- informational only
    assert "DEPRECIATION_RATE_DECLINING" not in result["depreciation_flags"]
    assert "CAPEX_BELOW_DEPRECIATION" not in result["depreciation_flags"]


def test_healthy_company_no_flags():
    """Stable rate + capex ~ depreciation -> no headwind flags."""
    from app.valuation.depreciation_audit import compute_depreciation_audit
    facts = {
        "gross_ppe": [(2023, 1000.0), (2024, 1020.0), (2025, 1040.0)],
        "depreciation_amortization": [(2023, 100.0), (2024, 102.0), (2025, 104.0)],
        "capex": [(2023, 105.0), (2024, 108.0), (2025, 110.0)],
    }
    result = compute_depreciation_audit(facts)
    # Rate: ~10.0% stable; coverage: ~1.03-1.06
    assert result["depreciation_flags"] == []


def test_missing_data_graceful():
    """No gross_ppe -> graceful empty result."""
    from app.valuation.depreciation_audit import compute_depreciation_audit
    result = compute_depreciation_audit({})
    assert result["depreciation_flags"] == []
    assert result["depreciation_rates"] == []
    assert result["capex_coverage_ratios"] == []
    assert result["latest_depreciation_rate"] is None
    assert result["latest_capex_coverage"] is None
