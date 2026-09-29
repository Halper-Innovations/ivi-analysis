from __future__ import annotations

from app.config import AppConfig
from app.valuation.tech_category import (
    ENTERPRISE_SOFTWARE,
    TRADITIONAL_OPERATING,
    classify_company_category,
)


def _companyfacts(*, entity_name: str, tags: dict[str, list[tuple[int, float]]]) -> dict:
    return {
        "entityName": entity_name,
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


def test_classifies_enterprise_software_from_rnd_margin_and_capex_profile():
    payload = _companyfacts(
        entity_name="Example Software Inc.",
        tags={
            "RevenueFromContractWithCustomerExcludingAssessedTax": [(2022, 1000.0), (2023, 1100.0), (2024, 1200.0)],
            "GrossProfit": [(2022, 700.0), (2023, 781.0), (2024, 864.0)],
            "ResearchAndDevelopmentExpense": [(2022, 180.0), (2023, 198.0), (2024, 216.0)],
            "PaymentsToAcquirePropertyPlantAndEquipment": [(2022, 35.0), (2023, 40.0), (2024, 45.0)],
        },
    )

    result = classify_company_category("SWCO", "2025-03-01", companyfacts=payload)

    assert result["category"] == ENTERPRISE_SOFTWARE
    assert result["confidence"] == "HIGH"
    assert result["metrics_used"]["avg_rnd_to_revenue_3y"] > 0.15
    assert result["metrics_used"]["avg_gross_margin_3y"] > 0.65
    assert result["metrics_used"]["avg_capex_to_revenue_3y"] < 0.08


def test_banks_always_route_to_traditional_operating():
    payload = _companyfacts(
        entity_name="Example Regional Bank",
        tags={
            "Deposits": [(2023, 2500.0), (2024, 2700.0)],
            "Loans": [(2023, 1800.0), (2024, 1900.0)],
            "RevenueFromContractWithCustomerExcludingAssessedTax": [(2023, 300.0), (2024, 320.0)],
            "ResearchAndDevelopmentExpense": [(2023, 80.0), (2024, 90.0)],
            "GrossProfit": [(2023, 220.0), (2024, 235.0)],
        },
    )

    result = classify_company_category("BANK", "2025-03-01", companyfacts=payload)

    assert result["category"] == TRADITIONAL_OPERATING
    assert result["confidence"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["metrics_used"]["issuer_classification"] == "financial"


def test_low_rnd_industrial_defaults_to_traditional_operating():
    payload = _companyfacts(
        entity_name="Example Industrial Systems",
        tags={
            "RevenueFromContractWithCustomerExcludingAssessedTax": [(2022, 1000.0), (2023, 1050.0), (2024, 1100.0)],
            "GrossProfit": [(2022, 420.0), (2023, 441.0), (2024, 462.0)],
            "ResearchAndDevelopmentExpense": [(2022, 12.0), (2023, 13.0), (2024, 15.0)],
            "PaymentsToAcquirePropertyPlantAndEquipment": [(2022, 90.0), (2023, 95.0), (2024, 100.0)],
        },
    )

    result = classify_company_category("INDU", "2025-03-01", companyfacts=payload)

    assert result["category"] == TRADITIONAL_OPERATING
    assert result["metrics_used"]["avg_rnd_to_revenue_3y"] < 0.03


def test_cik_override_forces_enterprise_software_on_known_edge_case():
    payload = _companyfacts(
        entity_name="Oracle Corporation",
        tags={
            "RevenueFromContractWithCustomerExcludingAssessedTax": [(2022, 42000.0), (2023, 45000.0), (2024, 50000.0)],
            "GrossProfit": [(2022, 24000.0), (2023, 27000.0), (2024, 30000.0)],
            "ResearchAndDevelopmentExpense": [(2022, 7000.0), (2023, 8000.0), (2024, 9000.0)],
            "PaymentsToAcquirePropertyPlantAndEquipment": [(2022, 4000.0), (2023, 4500.0), (2024, 5000.0)],
        },
    )
    cfg = AppConfig(tech_category_cik_overrides={"0001341439": ENTERPRISE_SOFTWARE})

    result = classify_company_category(
        "ORCL",
        "2025-03-01",
        companyfacts=payload,
        facts_row={"cik": "0001341439"},
        cfg=cfg,
    )

    assert result["category"] == ENTERPRISE_SOFTWARE
    assert result["metrics_used"]["cik_override_applied"] is True
    assert result["metrics_used"]["decision_path"] == "cik_override:0001341439"
