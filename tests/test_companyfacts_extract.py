from __future__ import annotations

import json
from pathlib import Path

from app.market.company_facts_extract import (
    extract_cash_equivalents_asof,
    extract_company_facts_asof,
    extract_total_debt_asof,
)


def _fixture(name: str) -> dict:
    path = Path(__file__).parent / "fixtures" / "companyfacts" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_extract_companyfacts_prefers_priority_tag_and_latest_end_date():
    payload = _fixture("AAA")
    extracted = extract_company_facts_asof(payload, "2026-02-14")
    shares = extracted["shares_outstanding_asof"]
    assert isinstance(shares, dict)
    assert shares["value"] == 1000.0
    assert shares["taxonomy"] == "dei"
    assert shares["tag"] == "EntityCommonStockSharesOutstanding"
    assert shares["fact_end_date"] == "2025-12-31"


def test_extract_companyfacts_derives_fcf_from_cfo_minus_capex():
    payload = _fixture("BBB")
    extracted = extract_company_facts_asof(payload, "2026-02-14")
    cfo = extracted["cfo_asof"]
    capex = extracted["capex_asof"]
    fcf = extracted["fcf_asof"]
    assert isinstance(cfo, dict)
    assert isinstance(capex, dict)
    assert isinstance(fcf, dict)
    assert cfo["value"] == 410.0
    assert capex["value"] == 90.0
    assert fcf["value"] == 320.0
    assert fcf["computation"] == "CFO_MINUS_CAPEX"
    assert len(fcf["derived_from"]) >= 2


def test_extract_companyfacts_respects_asof_window():
    payload = _fixture("CCC")
    extracted = extract_company_facts_asof(payload, "2024-02-14")
    shares = extracted["shares_outstanding_asof"]
    assert isinstance(shares, dict)
    assert shares["fact_end_date"] == "2023-12-31"

    none_window = extract_company_facts_asof(payload, "2023-01-01")
    assert none_window["shares_outstanding_asof"] is None
    assert none_window["fcf_asof"] is None


def test_extract_companyfacts_requires_exact_units_and_filed_asof_visibility():
    payload = {
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {
                        "USD": [
                            {
                                "end": "2023-12-31",
                                "filed": "2024-02-20",
                                "val": 900_000.0,
                            },
                            {
                                "end": "2024-12-31",
                                "filed": "2025-02-20",
                                "val": 1_200_000.0,
                            },
                            {
                                "end": "2025-12-31",
                                "val": 1_500_000.0,
                            },
                        ],
                        "USD_millions": [
                            {
                                "end": "2024-09-30",
                                "filed": "2024-11-01",
                                "val": 999.0,
                            }
                        ],
                    }
                }
            }
        }
    }

    extracted = extract_company_facts_asof(payload, "2025-01-15")

    assert extracted["cfo_asof"]["value"] == 900_000.0
    assert extracted["cfo_asof"]["unit"] == "USD"
    assert extracted["cfo_asof"]["filed_date"] == "2024-02-20"


def test_extract_total_debt_prefers_current_plus_noncurrent_sum():
    payload = _fixture("NET_DEBT_OK")
    debt = extract_total_debt_asof(payload, "2026-02-14")
    assert isinstance(debt, dict)
    # Raw XBRL USD (whole dollars): 80M + 220M = 300M.
    assert debt["value"] == 300000000.0
    assert debt["tag"] == "DebtCurrent_plus_LongTermDebtNoncurrent"
    assert debt["computation"] == "SUM_COMPONENTS"
    assert debt["fact_end_date"] == "2025-12-31"
    assert len(debt["derived_from"]) >= 2


def test_extract_total_debt_fallback_and_cash_equivalents():
    debt_only = _fixture("NET_DEBT_DEBT_ONLY")
    cash_only = _fixture("NET_DEBT_CASH_ONLY")
    debt = extract_total_debt_asof(debt_only, "2026-02-14")
    cash = extract_cash_equivalents_asof(cash_only, "2026-02-14")
    assert isinstance(debt, dict)
    assert debt["value"] == 150000000.0
    assert debt["tag"] == "LongTermDebt"
    assert isinstance(cash, dict)
    assert cash["value"] == 95000000.0
    assert cash["tag"] == "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"


def test_extract_companyfacts_uses_expanded_capex_alias_and_bridge_context():
    payload = {
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "val": 500.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
                "PaymentsToAcquirePremisesAndEquipment": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "val": 120.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
            }
        }
    }
    extracted = extract_company_facts_asof(payload, "2026-03-19")
    capex = extracted["capex_asof"]
    fcf = extracted["fcf_asof"]
    assert isinstance(capex, dict)
    assert capex["tag"] == "PaymentsToAcquirePremisesAndEquipment"
    assert isinstance(fcf, dict)
    assert fcf["value"] == 380.0
    assert fcf["resolution"] == "DERIVED"
    assert fcf["reason_code"] == "COMPANYFACTS_CFO_CAPEX_HIT"
    assert fcf["bridge_context"]["formula"] == "FCF = CFO - abs(CapEx)"


def test_extract_total_debt_supports_combined_alias_fallback():
    payload = {
        "facts": {
            "us-gaap": {
                "DebtLongtermAndShorttermCombinedAmount": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "val": 275.0,
                                "form": "10-K",
                            }
                        ]
                    }
                }
            }
        }
    }
    debt = extract_total_debt_asof(payload, "2026-03-19")
    assert isinstance(debt, dict)
    assert debt["tag"] == "DebtLongtermAndShorttermCombinedAmount"
    assert debt["value"] == 275.0
    assert debt["resolution"] == "RESOLVED"


def test_extract_total_debt_prefers_requested_fallback_order():
    payload = {
        "facts": {
            "us-gaap": {
                "LongTermDebt": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "val": 150.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
                "NotesPayable": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "val": 90.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
            }
        }
    }
    debt = extract_total_debt_asof(payload, "2026-03-19")
    assert isinstance(debt, dict)
    assert debt["tag"] == "LongTermDebt"
    assert debt["value"] == 150.0


def test_extract_cash_equivalents_prefers_cash_before_restricted_cash_aliases():
    payload = {
        "facts": {
            "us-gaap": {
                "Cash": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "val": 80.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "val": 120.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
            }
        }
    }
    cash = extract_cash_equivalents_asof(payload, "2026-03-19")
    assert isinstance(cash, dict)
    assert cash["tag"] == "Cash"
    assert cash["value"] == 80.0


def test_extract_cash_equivalents_prefers_fresher_alias_over_priority_order():
    payload = {
        "facts": {
            "us-gaap": {
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-12-31",
                                "filed": "2025-02-01",
                                "val": 70.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
                "Cash": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-09-30",
                                "filed": "2025-11-01",
                                "val": 95.0,
                                "form": "10-Q",
                            }
                        ]
                    }
                },
            }
        }
    }
    cash = extract_cash_equivalents_asof(payload, "2025-12-15")
    assert isinstance(cash, dict)
    assert cash["tag"] == "Cash"
    assert cash["value"] == 95.0
    assert cash["fact_end_date"] == "2025-09-30"


def test_extract_companyfacts_does_not_derive_fcf_from_mismatched_periods():
    payload = {
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-09-30",
                                "filed": "2025-11-01",
                                "val": 410.0,
                                "form": "10-Q",
                            }
                        ]
                    }
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-12-31",
                                "filed": "2025-02-01",
                                "val": 90.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
            }
        }
    }
    extracted = extract_company_facts_asof(payload, "2025-12-15")
    assert extracted["fcf_asof"] is None


def test_extract_companyfacts_prefers_direct_fcf_when_dates_tie():
    payload = {
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-05",
                                "val": 500.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-05",
                                "val": 120.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
                "FreeCashFlow": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-05",
                                "val": 370.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
            }
        }
    }
    extracted = extract_company_facts_asof(payload, "2026-03-19")
    fcf = extracted["fcf_asof"]
    assert isinstance(fcf, dict)
    assert fcf["tag"] == "FreeCashFlow"
    assert fcf["value"] == 370.0
    assert fcf["computation"] == "DIRECT"


def test_extract_total_debt_does_not_sum_mismatched_component_periods():
    payload = {
        "facts": {
            "us-gaap": {
                "DebtCurrent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-09-30",
                                "filed": "2025-11-01",
                                "val": 120.0,
                                "form": "10-Q",
                            }
                        ]
                    }
                },
                "LongTermDebtNoncurrent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-12-31",
                                "filed": "2025-02-01",
                                "val": 180.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
            }
        }
    }
    debt = extract_total_debt_asof(payload, "2025-12-15")
    assert isinstance(debt, dict)
    assert debt["tag"] == "LongTermDebtNoncurrent"
    assert debt["value"] == 180.0
    assert debt["resolution"] == "RESOLVED"


def test_extract_total_debt_prefers_direct_fact_when_dates_tie():
    payload = {
        "facts": {
            "us-gaap": {
                "DebtCurrent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-05",
                                "val": 100.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
                "LongTermDebtNoncurrent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-05",
                                "val": 200.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
                "DebtLongtermAndShorttermCombinedAmount": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-05",
                                "val": 295.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
            }
        }
    }
    debt = extract_total_debt_asof(payload, "2026-03-19")
    assert isinstance(debt, dict)
    # Migrated 2026-09-29: DebtCurrent now counts toward the completeness floor, and the
    # old 290 sat 3.4% under noncurrent 200 + current 100; 295 is inside the 2% allowance.
    assert debt["tag"] == "DebtLongtermAndShorttermCombinedAmount"
    assert debt["value"] == 295.0
    assert debt["resolution"] == "RESOLVED"


def test_extract_operating_lease_liability_sums_current_and_noncurrent():
    """FIX 1: extract operating lease liability (current + noncurrent) for net-debt parity."""
    from app.market.company_facts_extract import extract_operating_lease_liability_asof

    payload = {
        "facts": {
            "us-gaap": {
                "OperatingLeaseLiabilityCurrent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "val": 40.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
                "OperatingLeaseLiabilityNoncurrent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "val": 160.0,
                                "form": "10-K",
                            }
                        ]
                    }
                },
            }
        }
    }
    lease = extract_operating_lease_liability_asof(payload, "2026-03-19")
    assert isinstance(lease, dict)
    assert lease["value"] == 200.0
    assert lease["tag"] == "OperatingLeaseLiabilityCurrent_plus_Noncurrent"
    assert lease["computation"] == "SUM_COMPONENTS"


def test_extract_operating_lease_liability_prefers_direct_total():
    from app.market.company_facts_extract import extract_operating_lease_liability_asof

    payload = {
        "facts": {
            "us-gaap": {
                "OperatingLeaseLiability": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "val": 250.0,
                                "form": "10-K",
                            }
                        ]
                    }
                }
            }
        }
    }
    lease = extract_operating_lease_liability_asof(payload, "2026-03-19")
    assert isinstance(lease, dict)
    assert lease["value"] == 250.0
    assert lease["tag"] == "OperatingLeaseLiability"


def test_extract_operating_lease_liability_returns_none_when_absent():
    from app.market.company_facts_extract import extract_operating_lease_liability_asof

    payload = {"facts": {"us-gaap": {}}}
    assert extract_operating_lease_liability_asof(payload, "2026-03-19") is None
