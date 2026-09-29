"""The share-count guard on the ingest normalizer's per-period share pick.

The live buy targets divide by the fiscal-year ``shares_outstanding`` rows the normalizer
writes, and its pick -- the first tag in _SHARES_TAG_PRIORITY, the cover-page count first --
was never checked: ResMed's FY2021 row was 0.145681 million shares (the cover's 145,681),
Packaging Corp's FY2025 row 89,213 million, and seven of Chesapeake Utilities' 10-Q covers
reached the quarterly rows a thousandfold too large. Each pick is now judged by
app.market.shares_guard.check_share_count as of its own filing; a refused pick falls to the
next share tag for the same period, else the period has no share row. Trimmed real payloads;
values in the normalizer's millions of shares.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.ingest.companyfacts as companyfacts

FIXTURES = Path(__file__).parent / "fixtures" / "companyfacts_shares_guard"


def _real(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _shares(payload: dict, *, annual: bool = True, **kwargs) -> dict:
    normalize = (
        companyfacts.normalize_annual_facts_from_raw
        if annual
        else companyfacts.normalize_quarterly_facts_from_raw
    )
    rows = normalize(payload, cik="0000000001", years_back=30, filed_as_of="2026-09-28", **kwargs)
    return {
        (r["fiscal_year"], r["period_type"]): r for r in rows if r["line_item"] == "shares_outstanding"
    }


@pytest.fixture
def unguarded(monkeypatch):
    """The normalizer as it was: every pick accepted."""

    def run(payload: dict, *, annual: bool = True) -> dict:
        with monkeypatch.context() as patch:
            patch.setattr(
                companyfacts,
                "check_share_count",
                lambda *_a, **_k: {"decision": "accept", "by": "off", "references": []},
            )
            return _shares(payload, annual=annual)

    return run


def test_resmed_fy2021_row_is_the_balance_sheet_count_not_the_slipped_cover(unguarded):
    payload = _real("RMD_0000943819.json")
    before = unguarded(payload)[(2021, "FY")]
    assert (before["value"], before["tag"]) == (0.145681, "EntityCommonStockSharesOutstanding")
    after = _shares(payload)[(2021, "FY")]
    assert (after["value"], after["tag"]) == (145.648358, "CommonStockSharesOutstanding")
    assert after["period_end"] == "2021-06-30"


def test_packaging_corp_fy2025_slipped_cover_leaves_no_row(unguarded):
    payload = _real("PKG_0000075677.json")
    assert unguarded(payload)[(2025, "FY")]["value"] == 89_213.394
    after = _shares(payload)
    assert (2025, "FY") not in after, "no other share tag for that year: UNKNOWN, not the slip"
    assert after[(2024, "FY")]["value"] == unguarded(payload)[(2024, "FY")]["value"]


def test_chesapeake_utilities_slipped_10q_covers_leave_no_quarterly_rows(unguarded):
    payload = _real("CPK_0000019745.json")
    before = unguarded(payload, annual=False)
    after = _shares(payload, annual=False)
    dropped = sorted(set(before) - set(after))
    assert dropped == [
        (2023, "Q3"),
        (2024, "Q1"),
        (2024, "Q2"),
        (2024, "Q3"),
        (2025, "Q1"),
        (2025, "Q2"),
        (2025, "Q3"),
    ]
    assert all(before[key]["value"] > 17_000 for key in dropped), "each a thousandfold slip"
    assert {k: before[k] for k in after} == after
    assert _shares(payload) == unguarded(payload), "its 10-K covers were right"


@pytest.mark.parametrize("name", ["MCD_0000063908.json", "LARK_0001141688.json"])
@pytest.mark.parametrize("annual", [True, False])
def test_the_guard_is_invisible_where_it_accepts(unguarded, name, annual):
    """Migrated 2026-09-29: the weighted-average fallback is now decided per fiscal year, so
    the unguarded normalizer also reaches years no outstanding count reported; those rows
    are the guard's to judge (next test). Every outstanding-count pick is unchanged."""
    payload = _real(name)
    before = unguarded(payload, annual=annual)
    after = _shares(payload, annual=annual)
    fallback = "WeightedAverageNumberOfDilutedSharesOutstanding"
    assert {k: r for k, r in after.items() if r["tag"] != fallback} == {
        k: r for k, r in before.items() if r["tag"] != fallback
    }
    assert all(before[k] == r for k, r in after.items() if r["tag"] == fallback)


def test_a_per_year_weighted_fallback_still_goes_through_the_guard(unguarded):
    """LARK reports no outstanding count for FY2020; its diluted weighted average there is
    5,241 million, a thousandfold slip against about 5.2 million shares. The per-year
    fallback reaches the year and the guard refuses it: no row (UNKNOWN)."""
    payload = _real("LARK_0001141688.json")
    row = unguarded(payload)[(2020, "FY")]
    assert (row["value"], row["tag"]) == (5241.496, "WeightedAverageNumberOfDilutedSharesOutstanding")
    assert (2020, "FY") not in _shares(payload)


def test_a_refused_cover_falls_to_the_same_years_balance_sheet_count():
    accn = "0000000001-26-000010"
    payload = {
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"end": "2026-02-10", "val": 50_000, "accn": accn, "filed": "2026-02-20",
                             "fy": 2025, "fp": "FY", "form": "10-K"}
                        ]
                    }
                }
            },
            "us-gaap": {
                "CommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"end": "2025-12-31", "val": 50_000_000, "accn": accn,
                             "filed": "2026-02-20", "fy": 2025, "fp": "FY", "form": "10-K"}
                        ]
                    }
                },
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"start": "2025-01-01", "end": "2025-12-31", "val": 49_800_000,
                             "accn": accn, "filed": "2026-02-20", "fy": 2025, "fp": "FY",
                             "form": "10-K"}
                        ]
                    }
                },
            },
        }
    }
    row = _shares(payload)[(2025, "FY")]
    assert (row["value"], row["tag"]) == (50.0, "CommonStockSharesOutstanding")


def test_aemetis_cover_counts_survive_statements_filed_in_thousands(unguarded):
    """Aemetis states its balance-sheet and diluted counts in thousands (and once, 2,088
    for 20,088) beside correct cover pages. Counted as two votes, those two share tags
    outvoted the income reference and the cover fell through to 0.020088 million shares;
    as one class of evidence they tie with it and the cover's own history keeps it. The
    one pre-guard row that was a thousands-scale balance-sheet count (FY2014) is dropped."""
    payload = _real("AMTX_0000738214.json")
    before = unguarded(payload)
    after = _shares(payload)
    assert before[(2014, "FY")]["value"] == 0.02065
    assert (2014, "FY") not in after
    assert {k: v for k, v in before.items() if k != (2014, "FY")} == after
    assert after[(2017, "FY")]["value"] == 20.22289
