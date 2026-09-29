"""The tangible-book floor deducts only what the equity figure contains.

No other test covered this case. Each docstring derives the correct answer by hand; none is copied from
implementation output.
"""

from __future__ import annotations

from app.valuation.lenses import tangible_floor


# ── Tangible floor: goodwill filed in other years is not zero in this one ────────


def test_tangible_floor_refuses_when_goodwill_is_filed_but_not_for_the_equity_year():
    """Equity 500 at FY2024; goodwill 300 filed at FY2023 but no FY2024 goodwill row.

    The company carries goodwill; assuming zero for FY2024 would put 0.65 x 500 / 10
    = 32.5 a share on the floor where 0.65 x (500 - 300) / 10 = 13.0 is the most the
    FY2023 goodwill level would allow. The goodwill for the equity year is unknown,
    so the tangible-book floor cannot be formed.
    """
    result = tangible_floor(
        {"equity": [(2024, 500.0)], "goodwill": [(2023, 300.0)]},
        shares=10.0,
    )
    assert result["status"] == "METHOD_INSUFFICIENT_DATA"
    assert result["value_per_share"] is None
    assert "GOODWILL_YEAR_MISSING" in result["flags"]


def test_tangible_floor_with_no_goodwill_ever_filed_keeps_the_zero_assumption():
    """Control: a company that never filed goodwill: 0.65 x 500 / 10 = 32.5, flagged."""
    result = tangible_floor({"equity": [(2024, 500.0)]}, shares=10.0)
    assert result["status"] == "OK"
    assert result["value_per_share"] == 32.5
    assert "GOODWILL_MISSING_ZERO_ASSUMED" in result["flags"]


def test_tangible_floor_does_not_deduct_nci_from_parent_only_equity():
    """Parent equity 600, consolidated equity 700 (so NCI 100 is outside the 600).

    Goodwill 100, intangibles 50, NCI 100, 100 shares. The parent's tangible book
    is 600 - 100 - 50 = 450 (4.50 a share); deducting NCI again gives 350, which
    charges the minority holders' stake to the parent twice. Floor 0.65 x 4.50.
    """
    result = tangible_floor(
        {
            "equity": [(2023, 600.0)],
            "equity_including_nci": [(2023, 700.0)],
            "goodwill": [(2023, 100.0)],
            "intangible_assets": [(2023, 50.0)],
            "noncontrolling_interest": [(2023, 100.0)],
        },
        shares=100.0,
    )
    assert result["basis"]["tangible_book_per_share"] == 4.5
    assert result["value_per_share"] == 0.65 * 4.5
    assert "EQUITY_PARENT_ONLY_NCI_NOT_DEDUCTED" in result["flags"]


def test_tangible_floor_deducts_nci_from_consolidated_equity():
    """Control: equity equals the consolidated 700, so NCI 100 is inside it.

    700 - 100 - 50 - 100 = 450 -> 4.50 a share.
    """
    result = tangible_floor(
        {
            "equity": [(2023, 700.0)],
            "equity_including_nci": [(2023, 700.0)],
            "goodwill": [(2023, 100.0)],
            "intangible_assets": [(2023, 50.0)],
            "noncontrolling_interest": [(2023, 100.0)],
        },
        shares=100.0,
    )
    assert result["basis"]["tangible_book_per_share"] == 4.5


def test_annual_ingest_keeps_the_consolidated_equity_beside_the_parent_equity():
    """Parent 600m and consolidated 700m filed for FY2024 both reach the store (in millions)."""
    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    def _usd(value):
        return {
            "units": {
                "USD": [
                    {
                        "start": "2024-01-01",
                        "end": "2024-12-31",
                        "filed": "2025-02-01",
                        "fy": 2024,
                        "fp": "FY",
                        "form": "10-K",
                        "val": value,
                        "accn": "0000000000-25-000001",
                    }
                ]
            }
        }

    raw = {
        "facts": {
            "us-gaap": {
                "StockholdersEquity": _usd(600e6),
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": _usd(
                    700e6
                ),
            }
        }
    }
    rows = normalize_annual_facts_from_raw(raw, cik="1", filed_as_of="2025-06-01")
    values = {row["line_item"]: row["value"] for row in rows}
    assert values["equity"] == 600.0
    assert values["equity_including_nci"] == 700.0
