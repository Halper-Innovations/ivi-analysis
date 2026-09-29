"""A missing latest R&D expense cannot turn a historical delta into a current one."""

from app.valuation.rnd_capitalization import compute_rnd_adjusted_earnings


def _financials(*, current_rnd=None):
    annual_years = range(2020, 2025)
    rnd = [(2020, 30_000_000.0), (2021, 30_000_000.0), (2022, 30_000_000.0), (2023, 60_000_000.0)]
    if current_rnd is not None:
        rnd.append((2024, current_rnd))
    tags = {
        "ResearchAndDevelopmentExpense": rnd,
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            (year, 100_000_000.0) for year in annual_years
        ],
        "GrossProfit": [(year, 60_000_000.0) for year in annual_years],
        "OperatingIncomeLoss": [(year, 100_000_000.0) for year in annual_years],
        "NetCashProvidedByUsedInOperatingActivities": [
            (year, 100_000_000.0) for year in annual_years
        ],
        "PaymentsToAcquirePropertyPlantAndEquipment": [
            (year, 20_000_000.0) for year in annual_years
        ],
    }
    return {
        "facts": {
            "us-gaap": {
                tag: {
                    "units": {
                        "USD": [
                            {
                                "val": value,
                                "start": f"{year}-01-01",
                                "end": f"{year}-12-31",
                                "filed": f"{year + 1}-02-01",
                                "form": "10-K",
                                "fp": "FY",
                            }
                            for year, value in pairs
                        ]
                    }
                }
                for tag, pairs in tags.items()
            }
        }
    }


def _compute(as_of_date="2025-03-01", *, current_rnd=None):
    return compute_rnd_adjusted_earnings(
        "FIXTURE",
        as_of_date,
        category="ENTERPRISE_SOFTWARE",
        companyfacts=_financials(current_rnd=current_rnd),
    )


def test_absent_latest_rnd_refuses_current_adjustment_and_preserves_history():
    """FY2024 financials are known, but its absent R&D cannot inherit FY2023's delta30."""
    result = _compute()
    # Historical 2023: 60 - (30+30+30)/3 = 30, with unamortized asset90.
    assert [row["year"] for row in result["time_series"]] == [2020, 2021, 2022, 2023]
    assert result["time_series"][-1]["rnd_adjustment"] == 30.0
    assert result["time_series"][-1]["rnd_capitalized_asset"] == 90.0
    assert result["status"] == "INSUFFICIENT_RND_HISTORY"
    assert result["latest_year"] == 2024
    assert result["gaap_operating_income"] == 100.0
    assert result["gaap_owner_earnings"] == 80.0
    for key in (
        "rnd_adjustment",
        "rnd_amortization_current",
        "rnd_capitalized_asset",
        "adjusted_operating_income",
        "adjusted_owner_earnings",
    ):
        assert result[key] == "UNKNOWN"


def test_unfiled_later_financial_year_does_not_invalidate_known_current_rnd():
    """As of March2024 only FY2023 is available; its complete delta remains exactly30."""
    result = _compute("2024-03-01")
    assert result["status"] == "OK"
    assert result["latest_year"] == 2023
    assert result["rnd_adjustment"] == 30.0
    assert result["adjusted_operating_income"] == 130.0


def test_explicit_zero_latest_rnd_is_known_and_amortizes_earlier_spend():
    """2024 expense0 - (2021:30+2022:30+2023:60)/3 = -40, not missing data."""
    result = _compute(current_rnd=0.0)
    assert result["status"] == "OK"
    assert result["latest_year"] == 2024
    assert result["rnd_amortization_current"] == 40.0
    assert result["rnd_adjustment"] == -40.0
    assert result["rnd_capitalized_asset"] == 50.0
    assert result["adjusted_operating_income"] == 60.0
    assert result["adjusted_owner_earnings"] == 40.0
