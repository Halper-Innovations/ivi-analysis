"""Exact-literal regressions for the returns CAGR calculations."""

from app.valuation import returns_persistence

def test_returns_cagr_does_not_discard_latest_negative_endpoint():
    """CAGR from 100 to -1 is undefined; +21% describes only the obsolete 2022->2023 pair."""
    value, _ = returns_persistence._series_cagr([
        {"year": 2022, "invested_capital": 100.0},
        {"year": 2023, "invested_capital": 121.0},
        {"year": 2024, "invested_capital": -1.0},
    ], "invested_capital")
    assert value == "UNKNOWN"

