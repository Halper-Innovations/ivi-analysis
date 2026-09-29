"""Filed-input regressions: pure functions, no database, cache, or provider access.

USD facts mirror the stated accessions; normalized writer inputs are USD millions.
Unknown debt assertions mean completeness cannot be established from raw tags alone.
They do not propose a max(), blind sum, or issuer-specific production override.
"""

import pytest

from app.ingest.companyfacts import normalize_annual_facts_from_raw
from app.market.company_facts_extract import extract_total_debt_asof
from app.valuation.valuation_writer import _compute_owner_earnings, _revenue_cagr


def _raw(tags, *, end="2025-12-31", filed="2026-03-31", duration=False):
    """One annual USD fact per tag, with a fixed publication cutoff."""
    return {
        "facts": {
            "us-gaap": {
                tag: {
                    "units": {
                        "USD": [
                            {
                                "val": value,
                                "end": end,
                                "filed": filed,
                                "fy": 2025,
                                "fp": "FY",
                                "form": "20-F" if filed == "2026-03-31" else "10-K",
                                **({"start": "2025-01-01"} if duration else {}),
                            }
                        ]
                    }
                }
                for tag, value in tags.items()
            }
        }
    }


def _annual(raw, line_item):
    return [
        row["value"]
        for row in normalize_annual_facts_from_raw(raw, cik="0000000001", filed_as_of="2026-09-09")
        if row["line_item"] == line_item
    ]


def _payroll_adjusted_cashflow_facts():
    # Three complete years avoid unrelated capex/CFO history confidence failures.
    return {
        "cfo": [(2025, 6_416.0), (2024, 7_450.0), (2023, 4_843.0)],
        "capex": [(2025, 852.0), (2024, 683.0), (2023, 623.0)],
        "sbc": [(2025, 1_002.0), (2024, 1_230.0), (2023, 1_475.0)],
        "revenue": [(2025, 33_172.0), (2024, 31_797.0), (2023, 29_771.0)],
        "total_debt": [(2025, 11_583.0)],
        # Actual normalized omission: current annual interest is absent.
        "interest_expense": [(2023, 347.0)],
    }


def test_missing_current_interest_does_not_claim_normal_fcff_confidence():
    """A known borrower with only 2023 interest has unknown 2025 addback, not known zero."""
    result = _compute_owner_earnings(_payroll_adjusted_cashflow_facts())
    assert result["confidence"] == "LOWER"


def test_known_current_interest_uses_current_year_after_tax_amount():
    """441 * (1 - .21) = 348.39; stale 347 must never substitute for 2025."""
    facts = _payroll_adjusted_cashflow_facts()
    facts["interest_expense"] = [(2025, 441.0), (2023, 347.0)]
    result = _compute_owner_earnings(facts)
    assert result["interest_addback"] == pytest.approx(348.39, abs=1e-12)
    assert result["confidence"] == "NORMAL"


def _bosc_aggregate_conflict():
    # Actual 0001213900-26-037333, all instants 2025-12-31. The balance sheet's
    # broad current line is 775k; scheduled long-term principal due is 148k.
    # ShortTermBorrowings covers only 286k, so raw tags do not reconcile all debt.
    return _raw(
        {
            "LongTermDebt": 1_120_000,
            "LongTermDebtCurrent": 775_000,
            "LongTermDebtNoncurrent": 972_000,
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths": 148_000,
            "ShortTermBorrowings": 286_000,
        }
    )


def test_bosc_aggregate_conflict_requires_unknown_total_on_asof_path():
    """Raw-only complete total is UNKNOWN; filed statement separately establishes $1.747m."""
    assert extract_total_debt_asof(_bosc_aggregate_conflict(), "2026-09-09") is None


def test_bosc_aggregate_conflict_requires_no_normalized_complete_total():
    """Missing reconciliation must omit total_debt; 1.12 is only narrow long-term debt."""
    assert _annual(_bosc_aggregate_conflict(), "total_debt") == []


def _paypal_partial_instrument():
    # Actual undimensioned CompanyFacts, 0001633917-26-000024. The filing's
    # dimensioned 200m commercial paper, 1396m current term, and 76m discounts
    # are absent from this payload. Never invent those amounts from raw tags.
    return _raw(
        {
            "DebtInstrumentCarryingAmount": 11_459_000_000,
            "LongTermDebtNoncurrent": 9_987_000_000,
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths": 1_397_000_000,
        },
        filed="2026-02-03",
    )


def test_paypal_partial_instrument_does_not_establish_total_on_asof_path():
    """Raw-only total is UNKNOWN; statement debt=9987+1396+200=11583m, principal=11659m."""
    assert extract_total_debt_asof(_paypal_partial_instrument(), "2026-09-09") is None


def test_paypal_partial_instrument_does_not_normalize_to_complete_debt():
    """The 11459m candidate excludes 200m CP and retains 76m issuance costs versus book."""
    assert _annual(_paypal_partial_instrument(), "total_debt") == []


_BOSC_REVENUE = [
    (2025, 50.569),
    (2024, 39.949),
    (2023, 44.179),
    (2022, 41.511),
    (2021, 33.634),
    (2020, 33.551),
]


def test_bosc_displayed_five_year_cagr_has_five_actual_intervals():
    """2020->2025: (50.569/33.551)^(1/5)-1 = 8.551487566640548%; stored value is correct."""
    assert _revenue_cagr(_BOSC_REVENUE, n=6) == pytest.approx(0.08551487566640548, abs=1e-14)


def test_bosc_five_observation_dcf_growth_uses_four_intervals():
    """2021->2025: (50.569/33.634)^(1/4)-1 = 10.732845834614269%, before the DCF cap."""
    assert _revenue_cagr(_BOSC_REVENUE, n=5) == pytest.approx(0.10732845834614269, abs=1e-14)
