"""Exact-literal regressions for R&D capitalization vintages."""

from app.valuation.rnd_capitalization import compute_rnd_adjusted_earnings

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


def test_rnd_missing_internal_vintage_is_not_complete():
    """2024's 3-year amortization needs 2021/2022/2023; 2022 is missing, not zero."""
    years = [2019, 2021, 2023, 2024]
    result = compute_rnd_adjusted_earnings(
        "FIXTURE", "2025-03-01", category="ENTERPRISE_SOFTWARE",
        companyfacts=_companyfacts({
            "ResearchAndDevelopmentExpense": [(year, 30_000_000.0) for year in years],
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                (year, 100_000_000.0) for year in years
            ],
            "GrossProfit": [(year, 50_000_000.0) for year in years],
            "OperatingIncomeLoss": [(2024, 100_000_000.0)],
            "NetCashProvidedByUsedInOperatingActivities": [(2024, 100_000_000.0)],
            "PaymentsToAcquirePropertyPlantAndEquipment": [(2024, 20_000_000.0)],
        }),
    )
    assert result["time_series"][-1]["vintage_complete"] is False



def _rnd_stack_result(years_and_values):
    years = [year for year, _ in years_and_values]
    return compute_rnd_adjusted_earnings(
        "FIXTURE", "2025-03-01", category="ENTERPRISE_SOFTWARE",
        companyfacts=_companyfacts({
            "ResearchAndDevelopmentExpense": years_and_values,
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                (year, 100_000_000.0) for year in years
            ],
            "GrossProfit": [(year, 50_000_000.0) for year in years],
            "OperatingIncomeLoss": [(year, 100_000_000.0) for year in years],
            "NetCashProvidedByUsedInOperatingActivities": [
                (year, 100_000_000.0) for year in years
            ],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                (year, 20_000_000.0) for year in years
            ],
        }),
    )


def test_incomplete_latest_rnd_stack_cannot_reuse_an_older_complete_adjustment():
    # 2021 has a complete 2018-2021 stack; 2023 lacks the 2022 expense.
    result = _rnd_stack_result([(year, 30_000_000.0) for year in [2018, 2019, 2020, 2021, 2023]])
    by_year = {row["year"]: row for row in result["time_series"]}
    assert by_year[2021]["vintage_complete"] is True
    assert result["status"] == "INSUFFICIENT_RND_HISTORY"
    # The writer checks this status before selecting adjustments or using the
    # top-level adjusted-owner-earnings fallback. Unsupported amounts stay unknown.
    for key in ["rnd_adjustment", "rnd_amortization_current", "rnd_capitalized_asset",
                "adjusted_operating_income", "adjusted_owner_earnings"]:
        assert result[key] == "UNKNOWN"
        assert by_year[2023][key] == "UNKNOWN"


def test_explicit_zero_rnd_vintage_is_known_and_keeps_the_exact_amortization():
    result = _rnd_stack_result([(2021, 30_000_000.0), (2022, 0.0),
                                (2023, 30_000_000.0), (2024, 30_000_000.0)])
    assert result["status"] == "OK"
    assert result["time_series"][-1]["vintage_complete"] is True
    # (30 + 0 + 30) / 3 = 20; current expense30 - amortization20 = adjustment10.
    assert result["rnd_amortization_current"] == 20.0
    assert result["rnd_adjustment"] == 10.0
    assert result["rnd_capitalized_asset"] == 50.0
