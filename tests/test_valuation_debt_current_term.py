"""Hand-derived, scratch-only regressions for one formula defect."""

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


def test_total_debt_includes_disclosed_current_and_noncurrent_components():
    """Same-date current debt775000 + noncurrent972000 = 1747000 USD, not972000."""
    from app.market.company_facts_extract import extract_total_debt_asof

    result = extract_total_debt_asof(
        _companyfacts(
            {
                "LongTermDebtCurrent": [(2024, 775_000.0)],
                "LongTermDebtNoncurrent": [(2024, 972_000.0)],
            }
        ),
        "2025-03-01",
    )
    assert result["value"] == 1_747_000.0


def test_broad_current_debt_already_includes_current_long_term_component():
    from app.market.company_facts_extract import extract_total_debt_asof

    result = extract_total_debt_asof(
        _companyfacts(
            {
                "DebtCurrent": [(2024, 1_000_000.0)],
                "LongTermDebtCurrent": [(2024, 775_000.0)],
                "LongTermDebtNoncurrent": [(2024, 972_000.0)],
            }
        ),
        "2025-03-01",
    )
    assert result["value"] == 1_972_000.0
    assert result["tag"] == "DebtCurrent_plus_LongTermDebtNoncurrent"


def test_direct_total_precedence_and_period_alignment_are_preserved():
    from app.market.company_facts_extract import extract_total_debt_asof

    direct = extract_total_debt_asof(
        _companyfacts(
            {
                "Debt": [(2024, 2_000_000.0)],
                "LongTermDebtCurrent": [(2024, 775_000.0)],
                "LongTermDebtNoncurrent": [(2024, 972_000.0)],
            }
        ),
        "2025-03-01",
    )
    assert direct["value"] == 2_000_000.0
    assert direct["tag"] == "Debt"
    mismatch = extract_total_debt_asof(
        _companyfacts(
            {
                "LongTermDebtCurrent": [(2023, 775_000.0)],
                "LongTermDebtNoncurrent": [(2024, 972_000.0)],
            }
        ),
        "2025-03-01",
    )
    assert mismatch["value"] == 972_000.0  # Existing incomplete fallback, never a cross-date sum.


# Each tag is already treated by the extractor as a direct aggregate candidate.
# Later filing of a narrower current LT component must not displace that
# same-period complete debt evidence merely because the component was filed later.
@pytest.mark.parametrize(
    "aggregate_tag",
    [
        "DebtAndCapitalLeaseObligations",
        "DebtLongtermAndShorttermCombinedAmount",
        "Debt",
    ],
)
def test_later_filed_current_term_does_not_replace_a_complete_same_period_aggregate(aggregate_tag):
    from app.market.company_facts_extract import extract_total_debt_asof

    result = extract_total_debt_asof(_later_component_payload(aggregate_tag), "2025-06-01")
    # Full debt = noncurrent100 + current LT10 + short-term borrowings20 = 130m.
    assert result["value"] == 130_000_000.0
    assert result["tag"] == aggregate_tag
    assert result["filed_date"] == "2025-02-01"


@pytest.mark.parametrize(
    "aggregate_tag", ["LongTermDebtAndCapitalLeaseObligations", "DebtInstrumentCarryingAmount"]
)
def test_a_narrow_aggregate_is_completed_from_the_same_balance_sheet(aggregate_tag):
    """Migrated: both tags used to sit in the parametrization above as complete
    aggregates. LongTermDebtAndCapitalLeaseObligations is the NONCURRENT long-term debt and
    lease line, and DebtInstrumentCarryingAmount an instrument-level concept (no longer a
    total-debt candidate at all). The complete 130m is now assembled from the balance
    sheet's own lines: noncurrent 100 + current portion 10 + short-term borrowings 20."""
    from app.market.company_facts_extract import extract_total_debt_asof

    result = extract_total_debt_asof(_later_component_payload(aggregate_tag), "2025-06-01")
    assert result["value"] == 130_000_000.0
    assert result["tag"] == (
        "LongTermDebtNoncurrent_plus_LongTermDebtCurrent_plus_ShortTermBorrowings"
    )
    assert result["filed_date"] == "2025-05-01"


def _later_component_payload(aggregate_tag):
    tags = [
        (aggregate_tag, 130_000_000.0, "2025-02-01"),
        ("LongTermDebtNoncurrent", 100_000_000.0, "2025-02-01"),
        # Current portion 10 below short-term borrowings 20, so the one cannot hold the
        # other (2026-09-29: a current portion at or above the short-term borrowings with
        # nothing to reconcile them is UNKNOWN, pinned in tests/test_debt_completeness.py).
        ("LongTermDebtCurrent", 10_000_000.0, "2025-05-01"),
        ("ShortTermBorrowings", 20_000_000.0, "2025-05-01"),
        ("CashAndCashEquivalentsAtCarryingValue", 5_000_000.0, "2025-05-01"),
    ]
    return {
        "facts": {
            "us-gaap": {
                tag: {
                    "units": {
                        "USD": [
                            {
                                "val": value,
                                "end": "2024-12-31",
                                "filed": filed,
                                "form": "10-K",
                                "fy": 2024,
                                "accn": filed,
                            }
                        ]
                    }
                }
                for tag, value, filed in tags
            }
        }
    }


def test_net_debt_consumer_keeps_the_complete_aggregate(monkeypatch):
    from types import SimpleNamespace
    from app.valuation import net_debt

    payload = _later_component_payload("DebtLongtermAndShorttermCombinedAmount")
    monkeypatch.setattr(net_debt, "_load_companyfacts_payload", lambda _: payload)
    result = net_debt.resolve_net_debt_proxy(
        "FIXTURE", "2025-06-01", facts_row={}, cfg=SimpleNamespace()
    )
    assert result["total_debt"]["value"] == 130.0
    assert result["net_debt_proxy"] == 125.0  # Complete debt130 - cash5.
