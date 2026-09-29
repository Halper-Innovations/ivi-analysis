"""Exact-literal regressions for known valuation edge cases."""

from app.valuation import returns_persistence

def test_returns_latest_proxy_does_not_fabricate_third_year():
    """Two known fiscal returns remain two observations when latest is supplied as a proxy."""
    result = returns_persistence.compute_returns_persistence(
        "FIXTURE", "2025-03-01",
        fundamentals={"rows": [{"year": 2023, "roic": 0.2}, {"year": 2024, "roic": 0.2}]},
        roic_proxy=0.2, facts_status="OK", price_status="OK", shares_status="OK",
    )
    assert result["returns_support_signals"] == ["HIGH_RETURN_ON_CAPITAL_PRESENT"]
    assert result["returns_persistence_class"] == "MODERATE_RETURNS_PERSISTENCE"


def test_contradictory_undated_proxy_keeps_level_without_inventing_annual_history():
    result = returns_persistence.compute_returns_persistence(
        "FIXTURE", "2025-03-01",
        fundamentals={"rows": [{"year": year, "roic": 0.2} for year in [2021, 2022, 2023]]},
        roic_proxy=-0.5, facts_status="OK", price_status="OK", shares_status="OK",
        row_derived_from=["current-proxy"],
    )
    assert result["latest_roic_proxy"] == -0.5
    assert "current-proxy" in result["derived_from"]
    assert "current-proxy" in result["claims"]["returns_persistence_class"]["derived_from"]
    assert result["returns_support_signals"] == []
    assert result["returns_headwind_signals"] == ["LOW_RETURN_ON_CAPITAL"]
    assert "RETURNS_SERIES_ALIGNMENT_UNKNOWN" in result["returns_persistence_reason_codes"]
    assert result["returns_persistence_class"] == "MODERATE_RETURNS_PERSISTENCE"


def test_matching_proxy_preserves_three_actual_annual_observations():
    result = returns_persistence.compute_returns_persistence(
        "FIXTURE", "2025-03-01",
        fundamentals={"rows": [{"year": year, "roic": 0.2} for year in [2021, 2022, 2023]]},
        roic_proxy=0.2, facts_status="OK", price_status="OK", shares_status="OK",
    )
    assert result["returns_support_signals"] == [
        "HIGH_RETURN_ON_CAPITAL_PRESENT", "RETURNS_STABILITY_PRESENT"
    ]
    assert "RETURNS_SERIES_ALIGNMENT_UNKNOWN" not in result["returns_persistence_reason_codes"]
    assert result["returns_persistence_class"] == "HIGH_RETURNS_PERSISTENCE"


def test_contradictory_proxy_does_not_erase_observed_historical_volatility():
    result = returns_persistence.compute_returns_persistence(
        "FIXTURE", "2025-03-01",
        fundamentals={"rows": [{"year": year, "roic": value}
                               for year, value in [(2021, 0.2), (2022, -0.2), (2023, 0.2)]]},
        roic_proxy=0.5, facts_status="OK", price_status="OK", shares_status="OK",
    )
    assert result["latest_roic_proxy"] == 0.5
    assert result["returns_support_signals"] == ["HIGH_RETURN_ON_CAPITAL_PRESENT"]
    assert result["returns_headwind_signals"] == ["RETURNS_VOLATILITY_HEADWIND"]
    assert "RETURNS_SERIES_ALIGNMENT_UNKNOWN" in result["returns_persistence_reason_codes"]
