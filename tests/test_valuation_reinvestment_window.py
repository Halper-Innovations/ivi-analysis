"""Exact-literal regressions for the valuation math."""

from app.valuation import reinvestment_efficiency

def test_reinvestment_capex_burden_is_recent_three_year_median():
    """Latest three ratios = [.1, .9, .9], whose median is .9; five-year median is .1."""
    result = reinvestment_efficiency.compute_reinvestment_efficiency(
        "FIXTURE", "2025-03-01",
        fundamentals={"rows": [
            {"year": year, "cfo": 100.0, "capex": capex}
            for year, capex in zip(range(2020, 2025), [10.0, 10.0, 10.0, 90.0, 90.0], strict=True)
        ]},
        facts_status="OK", price_status="OK", shares_status="OK",
    )
    assert result["capex_burden_vs_cfo_median_3y"] == 0.9


def test_reinvestment_missing_recent_cashflow_does_not_pull_in_old_low_capex():
    result = reinvestment_efficiency.compute_reinvestment_efficiency(
        "FIXTURE", "2025-03-01", fundamentals={"rows": [
            {"year": 2021, "cfo": 100.0, "capex": 1.0},
            {"year": 2022, "cfo": 0.0, "capex": 20.0},
            {"year": 2023, "cfo": 0.0, "capex": 20.0},
            {"year": 2024, "cfo": 0.0, "capex": 20.0},
        ]}, facts_status="OK", price_status="OK", shares_status="OK",
    )
    assert result["capex_burden_vs_cfo_median_3y"] == "UNKNOWN"
