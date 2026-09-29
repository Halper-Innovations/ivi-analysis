"""Annual interest mapping regressions using filed USD facts and pure normalization."""

import pytest

from app.ingest.companyfacts import normalize_annual_facts_from_raw


@pytest.mark.parametrize(
    ("amount", "start", "end", "filed", "accession", "expected"),
    [
        (
            441_000_000,
            "2025-01-01",
            "2025-12-31",
            "2026-02-03",
            "0001633917-26-000024",
            441.0,
        ),
        (
            589_000_000,
            "2024-10-01",
            "2025-09-30",
            "2025-11-06",
            "0001403161-25-000089",
            589.0,
        ),
    ],
)
def test_nonoperating_interest_expense_is_available_for_fcff(
    amount, start, end, filed, accession, expected
):
    """PayPal441m and Visa589m are ordinary interest expense, USD scaled to millions."""
    raw = {
        "facts": {
            "us-gaap": {
                "InterestExpenseNonoperating": {
                    "units": {
                        "USD": [
                            {
                                "val": amount,
                                "start": start,
                                "end": end,
                                "filed": filed,
                                "accn": accession,
                                "fy": 2025,
                                "fp": "FY",
                                "form": "10-K",
                            }
                        ]
                    }
                }
            }
        }
    }
    rows = normalize_annual_facts_from_raw(raw, cik="0000000001", filed_as_of="2026-09-09")
    assert [row["value"] for row in rows if row["line_item"] == "interest_expense"] == [expected]
