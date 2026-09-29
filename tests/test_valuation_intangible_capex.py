"""Hand-derived, scratch-only regressions for one formula defect."""

from app.valuation import intangible_economics
import pytest


def _companyfacts(tags):
    return {
        "facts": {
            "us-gaap": {
                tag: {
                    "units": {
                        "USD": [
                            {
                                "start": f"{year}-01-01",
                                "end": f"{year}-12-31",
                                "filed": f"{year + 1}-02-01",
                                "fp": "FY",
                                "form": "10-K",
                                "val": value,
                            }
                            for year, value in rows
                        ]
                    }
                }
                for tag, rows in tags.items()
            }
        }
    }


def test_intangible_fundamentals_owner_earnings_uses_capex_magnitude():
    """OE = CFO - .60*abs(capex) = 100 - .60*20 = 88, regardless of filed sign."""
    series = intangible_economics._build_series_from_fundamentals(
        {"rows": [{"year": 2024, "cfo": 100.0, "capex": -20.0}]}
    )
    assert series["owner_earnings"][0]["value"] == 88.0


@pytest.mark.parametrize(
    "metric, expected", [("fcf", 80_000_000.0), ("owner_earnings", 88_000_000.0)]
)
def test_intangible_raw_cashflows_use_capex_magnitude(metric, expected):
    """USD FCF = 100m-20m=80m; proxy OE = 100m-.60*20m=88m."""
    series = intangible_economics._build_series_from_companyfacts(
        companyfacts=_companyfacts(
            {
                "NetCashProvidedByUsedInOperatingActivities": [(2024, 100_000_000.0)],
                "PaymentsToAcquirePropertyPlantAndEquipment": [(2024, -20_000_000.0)],
            }
        ),
        as_of_date="2025-03-01",
    )
    assert series[metric][0]["value"] == expected


def test_positive_capex_preserves_cashflow_amounts():
    normalized = intangible_economics._build_series_from_fundamentals(
        {"rows": [{"year": 2024, "cfo": 100.0, "capex": 20.0}]}
    )
    raw = intangible_economics._build_series_from_companyfacts(
        companyfacts=_companyfacts(
            {
                "NetCashProvidedByUsedInOperatingActivities": [(2024, 100_000_000.0)],
                "PaymentsToAcquirePropertyPlantAndEquipment": [(2024, 20_000_000.0)],
            }
        ),
        as_of_date="2025-03-01",
    )
    assert normalized["owner_earnings"][0]["value"] == 88.0
    assert raw["fcf"][0]["value"] == 80_000_000.0
    assert raw["owner_earnings"][0]["value"] == 88_000_000.0
