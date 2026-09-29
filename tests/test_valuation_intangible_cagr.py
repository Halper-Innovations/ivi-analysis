"""Hand-derived, scratch-only regressions for one formula defect."""

from app.valuation import intangible_economics


def test_intangible_fcf_cagr_keeps_latest_loss_as_endpoint():
    """Per-share FCF = 1, 1.21, -.5; latest CAGR is undefined, not stale +21%."""
    result = intangible_economics.compute_intangible_economics(
        "FIXTURE",
        "2025-03-01",
        owner_quality_payload={"oe_quality_total": 0.0},
        fundamentals={
            "rows": [
                {
                    "year": year,
                    "revenue": 100.0,
                    "shares_outstanding": 10.0,
                    "fcf": value,
                    "cfo": value,
                    "capex": 0.0,
                }
                for year, value in [(2022, 10.0), (2023, 12.1), (2024, -5.0)]
            ]
        },
    )
    assert result["fcf_per_share_cagr_proxy"] == "UNKNOWN"
    assert result["owner_earnings_per_share_cagr_proxy"] == "UNKNOWN"


def test_positive_growth_and_zero_endpoint_keep_correct_meaning():
    assert (
        round(
            intangible_economics._series_cagr(
                [{"year": 2022, "value": 100.0}, {"year": 2024, "value": 121.0}]
            )[0],
            8,
        )
        == 0.1
    )
    assert (
        intangible_economics._series_cagr(
            [
                {"year": 2022, "value": 100.0},
                {"year": 2023, "value": 121.0},
                {"year": 2024, "value": 0.0},
            ]
        )[0]
        == "UNKNOWN"
    )
