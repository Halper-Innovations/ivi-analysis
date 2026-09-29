"""Hand-derived, scratch-only regressions for one formula defect."""

from app.valuation import intangible_economics


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


def test_intangible_companyfacts_debt_and_cfo_use_same_currency_scale(monkeypatch):
    """$2bn net debt / $500m CFO = 4.0, independent of the two producer units."""
    companyfacts = _companyfacts(
        {
            "NetCashProvidedByUsedInOperatingActivities": [(2024, 500_000_000.0)],
            "PaymentsToAcquirePropertyPlantAndEquipment": [(2024, 100_000_000.0)],
        }
    )
    monkeypatch.setattr(intangible_economics, "_load_companyfacts_payload", lambda _: companyfacts)
    result = intangible_economics.compute_intangible_economics(
        "FIXTURE",
        "2025-03-01",
        facts_row={},
        owner_payload={"series": []},
        owner_quality_payload={"oe_quality_total": 0.0},
        net_debt_resolved={"net_debt_proxy": 2_000.0},
    )
    assert result["net_debt_to_cfo_proxy"] == 4.0


def test_normalized_and_raw_derived_debt_ratios_are_not_rescaled():
    normalized = intangible_economics.compute_intangible_economics(
        "FIXTURE",
        "2025-03-01",
        owner_quality_payload={"oe_quality_total": 0.0},
        fundamentals={"rows": [{"year": 2024, "cfo": 500.0}]},
        net_debt_resolved={"net_debt_proxy": 2000.0},
    )
    assert normalized["net_debt_to_cfo_proxy"] == 4.0
    raw = intangible_economics._balance_sheet_optionality(
        {
            "net_debt": [{"year": 2024, "value": 2_000_000_000.0}],
            "cfo": [{"year": 2024, "value": 500_000_000.0}],
        }
    )
    assert raw["net_debt_to_cfo_proxy"]["value"] == 4.0
