# tests/test_companyfacts.py
from __future__ import annotations
import pytest
from unittest.mock import patch


def _fake_payload(tag: str, value: float, end: str, form: str = "10-K",
                  start: str | None = None, unit: str = "USD") -> dict:
    accn = "0000320193-24-000001"
    fact = {"accn": accn, "end": end, "val": value, "form": form}
    if start:
        fact["start"] = start
    return {
        "facts": {
            "us-gaap": {
                tag: {
                    "units": {
                        unit: [fact]
                    }
                }
            }
        }
    }


def test_tag_alias_primary_preferred():
    """First matching tag wins."""
    payload = _fake_payload(
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        391035000000.0,
        "2024-09-28",
        start="2023-09-30",
    )
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts
        facts = fetch_annual_facts("0000320193", years_back=2)
    rev_facts = [f for f in facts if f["line_item"] == "revenue"]
    assert len(rev_facts) >= 1
    assert rev_facts[0]["value"] == pytest.approx(391035.0, rel=0.01)


def test_tag_alias_fallback():
    """Falls back to secondary tag when primary absent."""
    payload = _fake_payload(
        "Revenues",
        391035000000.0,
        "2024-09-28",
        start="2023-09-30",
    )
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts
        facts = fetch_annual_facts("0000320193", years_back=2)
    rev_facts = [f for f in facts if f["line_item"] == "revenue"]
    assert len(rev_facts) >= 1


def test_quarterly_filing_excluded():
    """10-Q forms must not appear in annual facts."""
    payload = _fake_payload(
        "Revenues", 100000000000.0, "2024-03-31", form="10-Q", start="2024-01-01"
    )
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts
        facts = fetch_annual_facts("0000320193", years_back=2)
    assert facts == []


def test_unit_normalization_base_usd():
    """Values in base USD are scaled to millions."""
    payload = _fake_payload(
        "Revenues", 391_035_000_000.0, "2024-09-28", start="2023-09-30"
    )
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts
        facts = fetch_annual_facts("0000320193", years_back=2)
    rev = next((f for f in facts if f["line_item"] == "revenue"), None)
    assert rev is not None
    assert rev["value"] == pytest.approx(391035.0, rel=0.01)


def test_implausible_value_excluded():
    """Values outside plausibility bounds are excluded."""
    payload = _fake_payload("Revenues", 0.001, "2024-09-28", start="2023-09-30")
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts
        facts = fetch_annual_facts("0000320193", years_back=2)
    assert facts == []


def test_fiscal_year_assigned():
    """fiscal_year is derived from the end date."""
    payload = _fake_payload(
        "Revenues", 391_035_000_000.0, "2024-09-28", start="2023-09-30"
    )
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts
        facts = fetch_annual_facts("0000320193", years_back=2)
    rev = next((f for f in facts if f["line_item"] == "revenue"), None)
    assert rev["fiscal_year"] == 2024


def test_bank_like_amount_tags_are_normalized():
    payload = {
        "facts": {
            "us-gaap": {
                "Deposits": {
                    "units": {
                        "USD": [
                            {"accn": "0000019617-24-000001", "end": "2024-12-31", "start": "2024-01-01", "val": 2500000000000.0, "form": "10-K"}
                        ]
                    }
                },
                "LoansAndLeasesReceivableNetReportedAmount": {
                    "units": {
                        "USD": [
                            {"accn": "0000019617-24-000001", "end": "2024-12-31", "start": "2024-01-01", "val": 1400000000000.0, "form": "10-K"}
                        ]
                    }
                },
            }
        }
    }
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts

        facts = fetch_annual_facts("0000019617", years_back=2)

    by_item = {fact["line_item"]: fact["value"] for fact in facts}
    assert by_item["deposits"] == pytest.approx(2_500_000.0, rel=0.01)
    assert by_item["loans"] == pytest.approx(1_400_000.0, rel=0.01)


def test_foreign_annual_forms_are_included():
    payload = _fake_payload(
        "Revenues",
        125_000_000_000.0,
        "2024-12-31",
        form="20-F",
        start="2024-01-01",
    )
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts

        facts = fetch_annual_facts("0001234567", years_back=2)

    rev = next((f for f in facts if f["line_item"] == "revenue"), None)
    assert rev is not None
    assert rev["value"] == pytest.approx(125_000.0, rel=0.01)


def test_additional_bank_aliases_are_normalized():
    payload = {
        "facts": {
            "us-gaap": {
                "LoansReceivableNetReportedAmount": {
                    "units": {
                        "USD": [
                            {"accn": "0000019617-24-000001", "end": "2024-12-31", "start": "2024-01-01", "val": 1300000000000.0, "form": "10-K"}
                        ]
                    }
                },
                "AllowanceForCreditLossesOnFinancingReceivables": {
                    "units": {
                        "USD": [
                            {"accn": "0000019617-24-000001", "end": "2024-12-31", "start": "2024-01-01", "val": 42000000000.0, "form": "10-K"}
                        ]
                    }
                },
                "PaymentsToAcquirePremisesAndEquipment": {
                    "units": {
                        "USD": [
                            {"accn": "0000019617-24-000001", "end": "2024-12-31", "start": "2024-01-01", "val": 9000000000.0, "form": "10-K"}
                        ]
                    }
                },
            }
        }
    }
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts

        facts = fetch_annual_facts("0000019617", years_back=2)

    by_item = {fact["line_item"]: fact["value"] for fact in facts}
    assert by_item["loans"] == pytest.approx(1_300_000.0, rel=0.01)
    assert by_item["allowance_for_credit_losses"] == pytest.approx(42_000.0, rel=0.01)
    assert by_item["capex"] == pytest.approx(9_000.0, rel=0.01)


def test_jpm_style_financing_receivable_tags_are_normalized():
    payload = {
        "facts": {
            "us-gaap": {
                "FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss": {
                    "units": {
                        "USD": [
                            {"accn": "0000019617-26-000001", "end": "2025-12-31", "val": 1408905000000.0, "form": "10-K"}
                        ]
                    }
                },
                "FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest": {
                    "units": {
                        "USD": [
                            {"accn": "0000019617-26-000001", "end": "2025-12-31", "val": 25765000000.0, "form": "10-K"}
                        ]
                    }
                },
            }
        }
    }
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts

        facts = fetch_annual_facts("0000019617", years_back=2)

    by_item = {fact["line_item"]: fact["value"] for fact in facts}
    assert by_item["loans"] == pytest.approx(1_408_905.0, rel=0.01)
    assert by_item["allowance_for_credit_losses"] == pytest.approx(25_765.0, rel=0.01)


def test_bank_credit_series_aliases_are_normalized():
    payload = {
        "facts": {
            "us-gaap": {
                "ProvisionForLoanLeaseAndOtherLosses": {
                    "units": {
                        "USD": [
                            {"accn": "0000019617-26-000001", "end": "2025-12-31", "val": 10462000000.0, "form": "10-K"}
                        ]
                    }
                },
                "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoffAfterRecovery": {
                    "units": {
                        "USD": [
                            {"accn": "0000019617-26-000001", "end": "2025-12-31", "val": 3142000000.0, "form": "10-K"}
                        ]
                    }
                },
                "FinancingReceivableRecordedInvestmentNonaccrualStatus": {
                    "units": {
                        "USD": [
                            {"accn": "0000019617-26-000001", "end": "2025-12-31", "val": 8650000000.0, "form": "10-K"}
                        ]
                    }
                },
            }
        }
    }
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts

        facts = fetch_annual_facts("0000019617", years_back=2)

    by_item = {fact["line_item"]: fact["value"] for fact in facts}
    assert by_item["provision_for_credit_losses"] == pytest.approx(10_462.0, rel=0.01)
    assert by_item["net_charge_offs"] == pytest.approx(3_142.0, rel=0.01)
    assert by_item["nonaccrual_loans"] == pytest.approx(8_650.0, rel=0.01)


def test_annual_restatement_prefers_latest_filed_value():
    """A later-filed 10-K/A restatement must override the original 10-K value."""
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            # Original 10-K filed 2024-02-15 (SEC ascending order = first).
                            {"accn": "0000000001-24-000001", "end": "2023-12-31",
                             "start": "2023-01-01", "val": 100_000_000_000.0,
                             "form": "10-K", "filed": "2024-02-15"},
                            # Restated 10-K/A filed later — the correct value.
                            {"accn": "0000000001-24-000009", "end": "2023-12-31",
                             "start": "2023-01-01", "val": 120_000_000_000.0,
                             "form": "10-K/A", "filed": "2024-08-20"},
                        ]
                    }
                }
            }
        }
    }
    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    facts = normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
    rev = [f for f in facts if f["line_item"] == "revenue"]
    assert len(rev) == 1
    assert rev[0]["value"] == pytest.approx(120_000.0, rel=0.001)


def test_quarterly_restatement_prefers_latest_filed_value():
    """A later-filed 10-Q/A restatement must override the original 10-Q value."""
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"accn": "0000000001-24-000002", "end": "2024-03-31",
                             "start": "2024-01-01", "val": 25_000_000_000.0,
                             "form": "10-Q", "fp": "Q1", "fy": 2024, "filed": "2024-05-01"},
                            {"accn": "0000000001-24-000010", "end": "2024-03-31",
                             "start": "2024-01-01", "val": 27_000_000_000.0,
                             "form": "10-Q/A", "fp": "Q1", "fy": 2024, "filed": "2024-11-01"},
                        ]
                    }
                }
            }
        }
    }
    from app.ingest.companyfacts import normalize_quarterly_facts_from_raw

    facts = normalize_quarterly_facts_from_raw(payload, cik="0000000001", years_back=10)
    rev = [f for f in facts if f["line_item"] == "revenue"]
    assert len(rev) == 1
    assert rev[0]["value"] == pytest.approx(27_000.0, rel=0.001)


# ── Quarterly normalization tests ──────────────────────────────────────────────

def _fake_quarterly_payload(tag: str, value: float, end: str, fp: str,
                            form: str = "10-Q", start: str | None = None,
                            fy: int | None = None, unit: str = "USD") -> dict:
    """Build a minimal SEC CompanyFacts payload with quarterly metadata."""
    fact = {
        "accn": "0000320193-24-000001",
        "end": end,
        "val": value,
        "form": form,
        "fp": fp,
    }
    if fy is not None:
        fact["fy"] = fy
    if start:
        fact["start"] = start
    return {
        "facts": {
            "us-gaap": {
                tag: {
                    "units": {
                        unit: [fact]
                    }
                }
            }
        }
    }


def test_quarterly_normalization_basic():
    """Quarterly revenue from 10-Q with fp=Q1 should be extracted."""
    payload = _fake_quarterly_payload(
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        94930000000.0,
        "2024-12-28",
        fp="Q1",
        start="2024-09-29",
        fy=2025,
    )
    from app.ingest.companyfacts import normalize_quarterly_facts_from_raw
    facts = normalize_quarterly_facts_from_raw(payload, cik="0000320193", years_back=2)
    rev = [f for f in facts if f["line_item"] == "revenue"]
    assert len(rev) == 1
    assert rev[0]["period_type"] == "Q1"
    assert rev[0]["fiscal_year"] == 2025
    assert rev[0]["value"] == pytest.approx(94930.0, rel=0.01)


def test_quarterly_normalization_excludes_annual():
    """Annual 10-K filings should not appear in quarterly output."""
    payload = _fake_quarterly_payload(
        "Revenues", 391035000000.0, "2024-09-28",
        fp="FY", form="10-K", start="2023-09-30",
    )
    from app.ingest.companyfacts import normalize_quarterly_facts_from_raw
    facts = normalize_quarterly_facts_from_raw(payload, cik="0000320193", years_back=2)
    assert len(facts) == 0


def test_quarterly_normalization_balance_sheet_instant():
    """Balance sheet items (no start date) from 10-Q should be extracted."""
    payload = _fake_quarterly_payload(
        "CashAndCashEquivalentsAtCarryingValue",
        30299000000.0,
        "2024-12-28",
        fp="Q1",
        form="10-Q",
        fy=2025,
    )
    from app.ingest.companyfacts import normalize_quarterly_facts_from_raw
    facts = normalize_quarterly_facts_from_raw(payload, cik="0000320193", years_back=2)
    cash = [f for f in facts if f["line_item"] == "cash"]
    assert len(cash) == 1
    assert cash[0]["period_type"] == "Q1"


def test_quarterly_deduplicates_same_period():
    """Only one value per (line_item, fiscal_year, period_type) — first tag wins."""
    payload = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [
                        {"accn": "A", "end": "2024-12-28", "val": 94930000000.0,
                         "form": "10-Q", "fp": "Q1", "fy": 2025, "start": "2024-09-29"},
                    ]}
                },
                "Revenues": {
                    "units": {"USD": [
                        {"accn": "B", "end": "2024-12-28", "val": 94930000000.0,
                         "form": "10-Q", "fp": "Q1", "fy": 2025, "start": "2024-09-29"},
                    ]}
                },
            }
        }
    }
    from app.ingest.companyfacts import normalize_quarterly_facts_from_raw
    facts = normalize_quarterly_facts_from_raw(payload, cik="0000320193", years_back=2)
    rev = [f for f in facts if f["line_item"] == "revenue"]
    assert len(rev) == 1


def test_annual_normalization_includes_period_type():
    """Annual normalization must now include period_type='FY' in output."""
    payload = _fake_payload(
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        391035000000.0,
        "2024-09-28",
        start="2023-09-30",
    )
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts
        facts = fetch_annual_facts("0000320193", years_back=2)
    assert all(f.get("period_type") == "FY" for f in facts)


def test_annual_total_debt_derives_from_aligned_components():
    payload = {
        "facts": {
            "us-gaap": {
                "DebtCurrent": {
                    "units": {
                        "USD": [
                            {"accn": "0000000001-25-000001", "end": "2024-12-31", "val": 100_000_000.0, "form": "10-K"}
                        ]
                    }
                },
                "LongTermDebtNoncurrent": {
                    "units": {
                        "USD": [
                            {"accn": "0000000001-25-000001", "end": "2024-12-31", "val": 500_000_000.0, "form": "10-K"}
                        ]
                    }
                },
            }
        }
    }

    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    facts = normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
    total_debt = next((fact for fact in facts if fact["line_item"] == "total_debt"), None)
    assert total_debt is not None
    assert total_debt["value"] == 600.0
    assert total_debt["period_type"] == "FY"


def test_annual_total_debt_direct_tag_beats_component_sum_on_same_period():
    payload = {
        "facts": {
            "us-gaap": {
                "DebtLongtermAndShorttermCombinedAmount": {
                    "units": {
                        "USD": [
                            {"accn": "0000000001-25-000001", "end": "2024-12-31", "val": 595_000_000.0, "form": "10-K"}
                        ]
                    }
                },
                "DebtCurrent": {
                    "units": {
                        "USD": [
                            {"accn": "0000000001-25-000001", "end": "2024-12-31", "val": 100_000_000.0, "form": "10-K"}
                        ]
                    }
                },
                "LongTermDebtNoncurrent": {
                    "units": {
                        "USD": [
                            {"accn": "0000000001-25-000001", "end": "2024-12-31", "val": 500_000_000.0, "form": "10-K"}
                        ]
                    }
                },
            }
        }
    }

    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    facts = normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
    total_debt = next((fact for fact in facts if fact["line_item"] == "total_debt"), None)
    assert total_debt is not None
    # Migrated 2026-09-29: DebtCurrent now counts toward the completeness floor, and the
    # old 550 sat 8% under noncurrent 500 + current 100; 595 is inside the 2% allowance.
    assert total_debt["value"] == 595.0


def test_fixed_asof_annual_normalization_excludes_later_restatement():
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "accn": "original",
                                "end": "2024-12-31",
                                "start": "2024-01-01",
                                "val": 100_000_000.0,
                                "form": "10-K",
                                "filed": "2025-02-15",
                            },
                            {
                                "accn": "amended",
                                "end": "2024-12-31",
                                "start": "2024-01-01",
                                "val": 125_000_000.0,
                                "form": "10-K/A",
                                "filed": "2025-07-01",
                            },
                        ]
                    }
                }
            }
        }
    }

    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    facts = normalize_annual_facts_from_raw(
        payload,
        cik="0000000001",
        years_back=10,
        filed_as_of="2025-06-01",
    )
    revenue = next(fact for fact in facts if fact["line_item"] == "revenue")
    assert revenue["value"] == 100.0
    assert revenue["filed_date"] == "2025-02-15"
    assert revenue["accession"] == "original"


def test_fixed_asof_component_sum_keeps_latest_component_filing_provenance():
    payload = {
        "facts": {
            "us-gaap": {
                "DebtCurrent": {
                    "units": {
                        "USD": [
                            {
                                "accn": "current",
                                "end": "2024-12-31",
                                "val": 100_000_000.0,
                                "form": "10-K",
                                "filed": "2025-02-15",
                            }
                        ]
                    }
                },
                "LongTermDebtNoncurrent": {
                    "units": {
                        "USD": [
                            {
                                "accn": "noncurrent",
                                "end": "2024-12-31",
                                "val": 500_000_000.0,
                                "form": "10-K/A",
                                "filed": "2025-03-01",
                            }
                        ]
                    }
                },
            }
        }
    }

    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    facts = normalize_annual_facts_from_raw(
        payload,
        cik="0000000001",
        years_back=10,
        filed_as_of="2025-06-01",
    )
    total_debt = next(fact for fact in facts if fact["line_item"] == "total_debt")
    assert total_debt["value"] == 600.0
    assert total_debt["filed_date"] == "2025-03-01"
    assert total_debt["form"] == "10-K/A"
    assert total_debt["accession"] == "noncurrent"


def test_quarterly_total_debt_derives_from_aligned_components():
    payload = {
        "facts": {
            "us-gaap": {
                "DebtCurrent": {
                    "units": {
                        "USD": [
                            {
                                "accn": "0000000001-25-000002",
                                "end": "2025-03-31",
                                "val": 120_000_000.0,
                                "form": "10-Q",
                                "fp": "Q2",
                                "fy": 2025,
                            }
                        ]
                    }
                },
                "LongTermDebtNoncurrent": {
                    "units": {
                        "USD": [
                            {
                                "accn": "0000000001-25-000002",
                                "end": "2025-03-31",
                                "val": 480_000_000.0,
                                "form": "10-Q",
                                "fp": "Q2",
                                "fy": 2025,
                            }
                        ]
                    }
                },
            }
        }
    }

    from app.ingest.companyfacts import normalize_quarterly_facts_from_raw

    facts = normalize_quarterly_facts_from_raw(payload, cik="0000000001", years_back=10)
    total_debt = next((fact for fact in facts if fact["line_item"] == "total_debt"), None)
    assert total_debt is not None
    assert total_debt["value"] == 600.0
    assert total_debt["period_type"] == "Q2"
    assert total_debt["fiscal_year"] == 2025


# ---------------------------------------------------------------------------
# Shares unit mismatch detection
# ---------------------------------------------------------------------------

class TestSharesUnitMismatch:
    """SEC XBRL sometimes reports shares in thousands instead of raw counts.
    The normalizer must detect this and produce correct millions values."""

    def test_raw_shares_normalized_to_millions(self):
        """Standard case: 267M shares reported as 267,479,000 raw."""
        payload = _fake_payload(
            "CommonStockSharesOutstanding",
            267_479_000,
            "2025-07-31",
            unit="shares",
        )
        with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
            from app.ingest.companyfacts import fetch_annual_facts
            facts = fetch_annual_facts("0001618732", years_back=2)
        shares = next((f for f in facts if f["line_item"] == "shares_outstanding"), None)
        assert shares is not None
        assert shares["value"] == pytest.approx(267.479, rel=0.01)

    def test_dei_shares_outstanding_preferred_over_us_gaap(self):
        """Cover-page DEI shares should win over secondary us-gaap share tags."""
        payload = {
            "facts": {
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {
                            "shares": [
                                {"accn": "0001618732-25-000001", "end": "2025-07-31", "val": 1_200_000_000, "form": "10-K"}
                            ]
                        }
                    }
                },
                "us-gaap": {
                    "CommonStockSharesOutstanding": {
                        "units": {
                            "shares": [
                                {"accn": "0001618732-25-000001", "end": "2025-07-31", "val": 1_000_000_000, "form": "10-K"}
                            ]
                        }
                    }
                },
            }
        }
        with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
            from app.ingest.companyfacts import fetch_annual_facts
            facts = fetch_annual_facts("0001618732", years_back=2)
        shares = next((f for f in facts if f["line_item"] == "shares_outstanding"), None)
        assert shares is not None
        assert shares["value"] == pytest.approx(1200.0, rel=0.01)

    def test_weighted_average_shares_do_not_normalize_as_shares_outstanding(self):
        payload = _fake_payload(
            "WeightedAverageNumberOfSharesOutstandingBasic",
            267_479_000,
            "2025-07-31",
            unit="shares",
        )
        with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
            from app.ingest.companyfacts import fetch_annual_facts
            facts = fetch_annual_facts("0001618732", years_back=2)
        shares = [f for f in facts if f["line_item"] == "shares_outstanding"]
        assert shares == []

    def test_tiny_shares_rejected(self):
        """A value that's implausible even after thousands correction is rejected."""
        payload = _fake_payload(
            "CommonStockSharesOutstanding",
            500,  # 500 raw shares -> 0.0005M or 0.5M (thousands). 0.5M is below floor of 1.0M.
            "2025-07-31",
            unit="shares",
        )
        with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
            from app.ingest.companyfacts import fetch_annual_facts
            facts = fetch_annual_facts("0001618732", years_back=2)
        shares = [f for f in facts if f["line_item"] == "shares_outstanding"]
        assert shares == []


def test_quarterly_dei_shares_outstanding_are_normalized():
    payload = {
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {
                                "accn": "0000320193-25-000001",
                                "end": "2024-12-28",
                                "val": 15_000_000_000,
                                "form": "10-Q",
                                "fp": "Q1",
                                "fy": 2025,
                            }
                        ]
                    }
                }
            }
        }
    }
    from app.ingest.companyfacts import normalize_quarterly_facts_from_raw

    facts = normalize_quarterly_facts_from_raw(payload, cik="0000320193", years_back=2)
    shares = next((f for f in facts if f["line_item"] == "shares_outstanding"), None)
    assert shares is not None
    assert shares["period_type"] == "Q1"
    assert shares["fiscal_year"] == 2025
    assert shares["value"] == pytest.approx(15000.0, rel=0.01)


# ── EVB-3: senior-claims component summation (review finding) ────────────────

def _senior_claims_payload(tag_values: dict[str, float], end: str = "2024-12-31") -> dict:
    accn = "0000320193-24-000001"
    return {
        "facts": {
            "us-gaap": {
                tag: {"units": {"USD": [{"accn": accn, "end": end, "val": val, "form": "10-K"}]}}
                for tag, val in tag_values.items()
            }
        }
    }


def test_preferred_zero_psv_does_not_mask_temporary_equity():
    """Review EVB-3: redeemable-preferred filers routinely tag
    PreferredStockValue = 0 (no permanent class) while the real preferred
    sits in temporary equity — first-tag-wins kept the 0 and dropped
    billions of senior claims (false-cheap direction on structured-capital
    names). A zero-valued higher-priority tag must not mask a populated
    lower-priority component."""
    payload = _senior_claims_payload({
        "PreferredStockValue": 0.0,
        "TemporaryEquityCarryingAmountAttributableToParent": 2_283_500_000.0,
    })
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts
        facts = fetch_annual_facts("0000320193", years_back=2)
    pref = [f for f in facts if f["line_item"] == "preferred_equity"]
    assert len(pref) == 1
    assert pref[0]["value"] == pytest.approx(2283.5)


def test_preferred_permanent_and_temporary_are_summed():
    """Permanent preferred and temporary-equity redeemable preferred are
    disjoint senior claims — SUMMED, not first-tag-wins."""
    payload = _senior_claims_payload({
        "PreferredStockValue": 100_000_000.0,
        "TemporaryEquityCarryingAmountAttributableToParent": 200_000_000.0,
    })
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts
        facts = fetch_annual_facts("0000320193", years_back=2)
    pref = [f for f in facts if f["line_item"] == "preferred_equity"]
    assert len(pref) == 1
    assert pref[0]["value"] == pytest.approx(300.0)


def test_nci_minority_and_redeemable_are_summed():
    """MinorityInterest (permanent) and RedeemableNCI (temporary) are
    disjoint claims that must be summed (review EVB-3: both present for
    4.8% of filers; first-tag-wins dropped the redeemable leg)."""
    payload = _senior_claims_payload({
        "MinorityInterest": 50_000_000.0,
        "RedeemableNoncontrollingInterestEquityCarryingAmount": 30_000_000.0,
    })
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts
        facts = fetch_annual_facts("0000320193", years_back=2)
    nci = [f for f in facts if f["line_item"] == "noncontrolling_interest"]
    assert len(nci) == 1
    assert nci[0]["value"] == pytest.approx(80.0)


def test_nci_single_tag_value_unchanged():
    payload = _senior_claims_payload({"MinorityInterest": 50_000_000.0})
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts
        facts = fetch_annual_facts("0000320193", years_back=2)
    nci = [f for f in facts if f["line_item"] == "noncontrolling_interest"]
    assert len(nci) == 1
    assert nci[0]["value"] == pytest.approx(50.0)


def test_preferred_psv_only_zero_still_emits_zero_row():
    """A filer with only PreferredStockValue=0 keeps an explicit 0 row (no
    senior claims, affirmatively known)."""
    payload = _senior_claims_payload({"PreferredStockValue": 0.0})
    with patch("app.ingest.companyfacts._fetch_raw", return_value=payload):
        from app.ingest.companyfacts import fetch_annual_facts
        facts = fetch_annual_facts("0000320193", years_back=2)
    pref = [f for f in facts if f["line_item"] == "preferred_equity"]
    assert len(pref) == 1
    assert pref[0]["value"] == 0.0
