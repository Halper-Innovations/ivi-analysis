"""Hand-derived, scratch-only regressions for one formula defect."""

from app.valuation import intangible_economics


def test_intangible_fundamentals_preserves_cash_for_liquidity_ratio():
    """Cash/revenue = 30/100 = .30; the row's cash is a known dollar numerator."""
    result = intangible_economics.compute_intangible_economics(
        "FIXTURE",
        "2025-03-01",
        owner_quality_payload={"oe_quality_total": 0.0},
        fundamentals={"rows": [{"year": 2024, "cash": 30.0, "revenue": 100.0}]},
    )
    assert result["cash_pct_revenue"] == 0.3


def test_known_zero_cash_is_retained_but_missing_cash_stays_unknown():
    def compute(row):
        return intangible_economics.compute_intangible_economics(
            "FIXTURE",
            "2025-03-01",
            owner_quality_payload={"oe_quality_total": 0.0},
            fundamentals={"rows": [row]},
        )

    assert compute({"year": 2024, "cash": 0.0, "revenue": 100.0})["cash_pct_revenue"] == 0.0
    assert compute({"year": 2024, "revenue": 100.0})["cash_pct_revenue"] == "UNKNOWN"
