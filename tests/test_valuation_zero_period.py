"""Hand-derived, scratch-only regressions for one formula defect."""

from types import SimpleNamespace
from app.valuation import evidenced_zero


def test_evidenced_zero_requires_raw_basis_on_normalized_balance_sheet_date(monkeypatch):
    """A 2023 balance sheet repeated in FY2024 filing is not raw evidence for 2024's zero."""
    companyfacts = {
        "facts": {
            "us-gaap": {
                "Liabilities": {
                    "units": {
                        "USD": [
                            {
                                "fy": 2024,
                                "end": "2023-12-31",
                                "filed": "2025-02-01",
                                "val": 90_000_000,
                                "accn": "prior-comparative",
                                "form": "10-K",
                            }
                        ]
                    }
                }
            }
        }
    }
    monkeypatch.setattr(
        evidenced_zero, "_materialized_companyfacts", lambda **_: (companyfacts, {})
    )

    def basis(*args, **kwargs):
        if kwargs["line_item"] != "total_liabilities":
            return None
        return {
            "line_item": "total_liabilities",
            "fiscal_year": 2024,
            "period_end": "2024-12-31",
            "accession": "current-annual",
            "value": 100.0,
        }

    monkeypatch.setattr(evidenced_zero, "_normalized_basis_row", basis)
    result, proofs = evidenced_zero.resolve_evidenced_zero_facts(
        {"cash": [(2024, 20.0)], "total_liabilities": [(2024, 100.0)]},
        ticker="FIXTURE",
        conn=object(),
        as_of_date="2025-03-01",
        issuer_cik="42",
        cfg=SimpleNamespace(),
    )
    assert result.get("total_debt", []) == []
    assert proofs == []


def test_raw_basis_same_period_can_use_another_eligible_accession():
    records = [{"period_end": "2024-12-31", "accession": "restatement", "filed_date": "2025-02-02"}]
    assert (
        evidenced_zero._basis_raw_evidence(
            records, basis_row={"period_end": "2024-12-31", "accession": "initial"}
        )
        == records
    )


def test_raw_basis_needs_a_known_matching_period():
    records = [{"period_end": "2023-12-31", "accession": "same", "filed_date": "2025-02-02"}]
    assert (
        evidenced_zero._basis_raw_evidence(
            records, basis_row={"period_end": "2024-12-31", "accession": "same"}
        )
        == []
    )
