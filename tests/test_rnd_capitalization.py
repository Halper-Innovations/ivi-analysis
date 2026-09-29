from __future__ import annotations

from app.valuation.rnd_capitalization import (
    FLAG_INPUTS_SCALED_TO_MILLIONS,
    FLAG_RND_ADJUSTMENT_CAPPED,
    STATUS_INSUFFICIENT_RND_HISTORY,
    STATUS_OK,
    STATUS_SKIPPED_LOW_GROSS_MARGIN,
    compute_rnd_adjusted_earnings,
)
from app.valuation.tech_category import ENTERPRISE_SOFTWARE


def _companyfacts(*, tags: dict[str, list[tuple[int, float]]]) -> dict:
    return {
        "entityName": "Example Tech Co.",
        "facts": {
            "us-gaap": {
                tag: {
                    "units": {
                        "USD": [
                            {
                                "end": f"{year}-12-31",
                                "filed": f"{year + 1}-02-01",
                                "val": value,
                            }
                            for year, value in rows
                        ]
                    }
                }
                for tag, rows in tags.items()
            }
        },
    }


def test_rnd_capitalization_computes_asset_amortization_and_adjusted_earnings():
    payload = _companyfacts(
        tags={
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                (2021, 900.0),
                (2022, 1000.0),
                (2023, 1100.0),
                (2024, 1200.0),
            ],
            "GrossProfit": [(2021, 630.0), (2022, 700.0), (2023, 770.0), (2024, 840.0)],
            "ResearchAndDevelopmentExpense": [
                (2021, 100.0),
                (2022, 120.0),
                (2023, 150.0),
                (2024, 180.0),
            ],
            "OperatingIncomeLoss": [(2021, 180.0), (2022, 190.0), (2023, 195.0), (2024, 200.0)],
            "NetCashProvidedByUsedInOperatingActivities": [
                (2021, 210.0),
                (2022, 220.0),
                (2023, 230.0),
                (2024, 240.0),
            ],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                (2021, 30.0),
                (2022, 32.0),
                (2023, 34.0),
                (2024, 36.0),
            ],
            "ShareBasedCompensation": [(2021, 8.0), (2022, 8.0), (2023, 9.0), (2024, 10.0)],
        },
    )

    result = compute_rnd_adjusted_earnings(
        "TECH",
        "2025-03-01",
        category=ENTERPRISE_SOFTWARE,
        companyfacts=payload,
    )

    assert result is not None
    assert result["status"] == STATUS_OK
    assert result["amortization_life"] == 3
    assert result["latest_year"] == 2024
    assert result["rnd_amortization_current"] == 0.000123
    assert result["rnd_adjustment"] == 0.000057
    assert result["rnd_capitalized_asset"] == 0.00032
    assert result["adjusted_operating_income"] == 0.000257
    assert result["guardrails"]["input_unit"] == "USD"
    assert result["guardrails"]["output_unit"] == "USD_millions"
    assert result["guardrails"]["input_unit_scale"] == 1_000_000.0


def test_rnd_capitalization_caps_large_adjustment_against_operating_income():
    payload = _companyfacts(
        tags={
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                (2021, 900.0),
                (2022, 1000.0),
                (2023, 1100.0),
                (2024, 1200.0),
            ],
            "GrossProfit": [(2021, 630.0), (2022, 700.0), (2023, 770.0), (2024, 840.0)],
            "ResearchAndDevelopmentExpense": [
                (2021, 100.0),
                (2022, 120.0),
                (2023, 150.0),
                (2024, 180.0),
            ],
            "OperatingIncomeLoss": [(2021, 180.0), (2022, 190.0), (2023, 195.0), (2024, 50.0)],
            "NetCashProvidedByUsedInOperatingActivities": [
                (2021, 210.0),
                (2022, 220.0),
                (2023, 230.0),
                (2024, 240.0),
            ],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                (2021, 30.0),
                (2022, 32.0),
                (2023, 34.0),
                (2024, 36.0),
            ],
        },
    )

    result = compute_rnd_adjusted_earnings(
        "TECH",
        "2025-03-01",
        category=ENTERPRISE_SOFTWARE,
        companyfacts=payload,
    )

    assert result is not None
    assert result["status"] == STATUS_OK
    assert FLAG_RND_ADJUSTMENT_CAPPED in result["flags"]
    assert result["rnd_adjustment"] == 0.00002
    assert result["adjusted_operating_income"] == 0.00007


def test_rnd_capitalization_requires_life_plus_one_years_of_history():
    payload = _companyfacts(
        tags={
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                (2022, 1000.0),
                (2023, 1100.0),
                (2024, 1200.0),
            ],
            "GrossProfit": [(2022, 700.0), (2023, 770.0), (2024, 840.0)],
            "ResearchAndDevelopmentExpense": [(2022, 120.0), (2023, 150.0), (2024, 180.0)],
            "OperatingIncomeLoss": [(2022, 190.0), (2023, 195.0), (2024, 200.0)],
        },
    )

    result = compute_rnd_adjusted_earnings(
        "TECH",
        "2025-03-01",
        category=ENTERPRISE_SOFTWARE,
        companyfacts=payload,
    )

    assert result is not None
    assert result["status"] == STATUS_INSUFFICIENT_RND_HISTORY


def test_rnd_capitalization_skips_when_gross_margin_is_too_low():
    payload = _companyfacts(
        tags={
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                (2021, 900.0),
                (2022, 1000.0),
                (2023, 1100.0),
                (2024, 1200.0),
            ],
            "GrossProfit": [(2021, 200.0), (2022, 220.0), (2023, 240.0), (2024, 250.0)],
            "ResearchAndDevelopmentExpense": [
                (2021, 100.0),
                (2022, 120.0),
                (2023, 150.0),
                (2024, 180.0),
            ],
            "OperatingIncomeLoss": [(2021, 50.0), (2022, 55.0), (2023, 60.0), (2024, 65.0)],
        },
    )

    result = compute_rnd_adjusted_earnings(
        "TECH",
        "2025-03-01",
        category=ENTERPRISE_SOFTWARE,
        companyfacts=payload,
    )

    assert result is not None
    assert result["status"] == STATUS_SKIPPED_LOW_GROSS_MARGIN


def test_rnd_capitalization_scales_large_companyfacts_to_millions():
    payload = _companyfacts(
        tags={
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                (2021, 900_000_000.0),
                (2022, 1_000_000_000.0),
                (2023, 1_100_000_000.0),
                (2024, 1_200_000_000.0),
            ],
            "GrossProfit": [
                (2021, 630_000_000.0),
                (2022, 700_000_000.0),
                (2023, 770_000_000.0),
                (2024, 840_000_000.0),
            ],
            "ResearchAndDevelopmentExpense": [
                (2021, 100_000_000.0),
                (2022, 120_000_000.0),
                (2023, 150_000_000.0),
                (2024, 180_000_000.0),
            ],
            "OperatingIncomeLoss": [
                (2021, 180_000_000.0),
                (2022, 190_000_000.0),
                (2023, 195_000_000.0),
                (2024, 200_000_000.0),
            ],
            "NetCashProvidedByUsedInOperatingActivities": [
                (2021, 210_000_000.0),
                (2022, 220_000_000.0),
                (2023, 230_000_000.0),
                (2024, 240_000_000.0),
            ],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                (2021, 30_000_000.0),
                (2022, 32_000_000.0),
                (2023, 34_000_000.0),
                (2024, 36_000_000.0),
            ],
            "ShareBasedCompensation": [
                (2021, 8_000_000.0),
                (2022, 8_000_000.0),
                (2023, 9_000_000.0),
                (2024, 10_000_000.0),
            ],
        },
    )

    result = compute_rnd_adjusted_earnings(
        "TECH",
        "2025-03-01",
        category=ENTERPRISE_SOFTWARE,
        companyfacts=payload,
    )

    assert result is not None
    assert result["status"] == STATUS_OK
    assert FLAG_INPUTS_SCALED_TO_MILLIONS in result["flags"]
    assert abs(result["rnd_adjustment"] - 56.666667) < 0.01


def test_rnd_capitalization_scales_sub_million_raw_usd_without_magnitude_guess():
    payload = _companyfacts(
        tags={
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                (2021, 900_000.0),
                (2022, 1_000_000.0),
                (2023, 1_100_000.0),
                (2024, 1_200_000.0),
            ],
            "GrossProfit": [
                (2021, 630_000.0),
                (2022, 700_000.0),
                (2023, 770_000.0),
                (2024, 840_000.0),
            ],
            "ResearchAndDevelopmentExpense": [
                (2021, 100_000.0),
                (2022, 120_000.0),
                (2023, 150_000.0),
                (2024, 180_000.0),
            ],
            "OperatingIncomeLoss": [
                (2021, 180_000.0),
                (2022, 190_000.0),
                (2023, 195_000.0),
                (2024, 200_000.0),
            ],
            "NetCashProvidedByUsedInOperatingActivities": [
                (2021, 210_000.0),
                (2022, 220_000.0),
                (2023, 230_000.0),
                (2024, 240_000.0),
            ],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                (2021, 30_000.0),
                (2022, 32_000.0),
                (2023, 34_000.0),
                (2024, 36_000.0),
            ],
        },
    )

    result = compute_rnd_adjusted_earnings(
        "MICRO",
        "2025-03-01",
        category=ENTERPRISE_SOFTWARE,
        companyfacts=payload,
    )

    assert result["status"] == STATUS_OK
    assert result["rnd_adjustment"] == 0.056667
    assert result["adjusted_operating_income"] == 0.256667
    assert result["guardrails"]["avg_rnd_to_revenue_3y"] == 0.135455
    assert result["guardrails"]["avg_gross_margin_3y"] == 0.7
    assert result["guardrails"]["input_unit"] == "USD"
    assert result["guardrails"]["output_unit"] == "USD_millions"
    assert result["guardrails"]["input_unit_scale"] == 1_000_000.0
