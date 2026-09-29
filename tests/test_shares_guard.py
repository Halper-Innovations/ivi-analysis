"""Both answers for the share-count guard (app/market/shares_guard.py).

"Before" is the unguarded chooser's answer -- the same `_extract_from_priority` over
SHARES_TAG_PRIORITY the spine has always used, its sort left alone; "after" is the guarded
pick every consumer now gets from `extract_shares_outstanding_asof`. Synthetic payloads pin
the guard's shape (ported from census-tested cases in a sibling tool): the type exclusion,
the same-filing references, the continuity check, the fall-through, the honest UNKNOWN, the
split that must pass, the run of consecutive slips that the single-prior form lets through.
Trimmed excerpts of three real SEC payloads pin the cases the guard exists for: ResMed's
FY2021 cover (145,681 for 145.6 million, on its 10-K and its 10-K/A), Chesapeake Utilities'
10-Q covers (x1,000, every quarter from late 2023 to late 2025, with no balance-sheet tag),
and Packaging Corp's 2026 10-K cover (89.2 billion for 89.2 million).

Values in these tests are raw share counts unless a name says millions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app.autonomous.cap_resolver import STALE_SHARES_MAX_AGE_DAYS
from app.market import shares_guard as guard_mod
from app.market.company_facts_extract import (
    SHARES_TAG_PRIORITY,
    _extract_from_priority,
    extract_company_facts_asof,
    extract_shares_outstanding_asof,
)

FIXTURES = Path(__file__).parent / "fixtures" / "companyfacts_shares_guard"
COVER = ("dei", "EntityCommonStockSharesOutstanding")
BALANCE = ("us-gaap", "CommonStockSharesOutstanding")
WAD = ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")


# --- helpers -----------------------------------------------------------------


def payload(rows_by_tag: dict[tuple[str, str], list[dict]]) -> dict:
    facts: dict = {}
    for (tax, tag), rows in rows_by_tag.items():
        facts.setdefault(tax, {})[tag] = {"units": {"shares": rows}}
    return {"cik": "0", "entityName": "Synthetic", "facts": facts}


def row(end: str, val, filed: str, form: str = "10-Q", accn: str | None = None) -> dict:
    return {"end": end, "val": val, "filed": filed, "form": form, "fp": "Q1", "accn": accn or f"a-{filed}"}


def asof_of(p: dict, k: int) -> str:
    """The k-th (0-based) distinct filing date in a synthetic payload: an as-of on the
    filing day sees that filing and everything before it."""
    filed = set()
    for tags in p["facts"].values():
        for node in tags.values():
            for r in node["units"]["shares"]:
                filed.add(r["filed"])
    return sorted(filed)[k]


def quarterly(tag_key, values, start_year=2022):
    """Four filings a year: cover counts a few weeks after quarter end, balance-sheet
    counts at quarter end, both filed the same day under the same accession number."""
    out = []
    ends = ["03-31", "06-30", "09-30", "12-31"]
    filings = ["05-05", "08-05", "11-05"]
    cover_ends = ["05-01", "08-01", "11-01", "02-20"]
    i = 0
    for year in range(start_year, start_year + 3):
        for q in range(4):
            if i >= len(values):
                break
            next_year = year + (1 if q == 3 else 0)
            filed = f"{next_year}-{filings[q] if q < 3 else '02-25'}"
            form = "10-K" if q == 3 else "10-Q"
            if tag_key == COVER:
                out.append(row(f"{next_year}-{cover_ends[q]}", values[i], filed, form))
            else:
                out.append(row(f"{year}-{ends[q]}", values[i], filed, form))
            i += 1
    return out


def with_wad(p: dict, values, start_year=2022) -> dict:
    """Add the diluted weighted-average count to the same filings (a duration fact)."""
    p["facts"].setdefault("us-gaap", {})[WAD[1]] = {
        "units": {"shares": [{**r, "start": "2020-01-01"} for r in quarterly(BALANCE, values, start_year)]}
    }
    return p


def unguarded(p: dict, asof: str) -> dict | None:
    return _extract_from_priority(
        companyfacts=p,
        as_of_date=asof,
        priority=SHARES_TAG_PRIORITY,
        expected_unit_exact=("shares",),
        expected_unit_prefixes=("shares",),
    )


def before(p: dict, asof: str):
    fact = unguarded(p, asof)
    return None if fact is None else fact["value"]


def after(p: dict, asof: str, **knobs):
    """The guarded pick: (value or None, fact, guard record). With no knobs this is the
    production entry point itself."""
    if knobs:
        fact, guard = guard_mod.guard_shares_fact(
            p, asof, unguarded=unguarded(p, asof), priority=SHARES_TAG_PRIORITY, **knobs
        )
    else:
        fact, guard = extract_shares_outstanding_asof(p, asof)
    return (None if fact is None else fact["value"]), fact, guard


def real(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# --- constants and the bound this repository already uses ---------------------


def test_the_constants_are_the_ported_ones_and_the_stale_bound_is_the_repos_own():
    assert guard_mod.SHARES_MAX_MOVE == 100.0
    assert guard_mod.SHARES_CORROBORATION_X == 2.0
    assert guard_mod.SHARES_PRIOR_WINDOW == 4
    assert guard_mod.FALLTHROUGH_MAX_AGE_DAYS == 366
    assert guard_mod.REFERENCES_DECIDE == "majority"
    # One bound, one number: the cap chain's stale-shares window.
    assert guard_mod.SHARES_STALE_DAYS == STALE_SHARES_MAX_AGE_DAYS == 400


@pytest.mark.parametrize(
    ("candidate", "reference", "verdict"),
    [
        (100.0, 100.0, "corroborates"),
        (200.0, 100.0, "corroborates"),  # exactly 2x still corroborates
        (50.0, 100.0, "corroborates"),
        (201.0, 100.0, "disagrees"),
        (3_300.0, 100.0, "disagrees"),  # between the bands: history decides
        (9_999.0, 100.0, "disagrees"),
        (10_000.0, 100.0, "contradicts"),  # exactly 100x contradicts
        (1.0, 100.0, "contradicts"),
        (0.145681, 145.648358, "contradicts"),  # ResMed's slip
        (89_213.394, 89.6, "contradicts"),  # Packaging Corp's slip
    ],
)
def test_reference_bands(candidate, reference, verdict):
    assert guard_mod._verdict(candidate, reference, 100.0, 2.0)[0] == verdict


# --- the type exclusion --------------------------------------------------------


def test_nan_infinity_and_booleans_are_not_counts():
    p = payload(
        {
            COVER: [
                row("2025-05-01", 100e6, "2025-05-05"),
                row("2025-08-01", float("nan"), "2025-08-05"),
                row("2025-11-01", True, "2025-11-05"),
                row("2026-02-20", float("inf"), "2026-02-25", "10-K"),
            ],
            BALANCE: [
                row("2025-03-31", 100e6, "2025-05-05"),
                row("2025-06-30", 101e6, "2025-08-05"),
                row("2025-09-30", 102e6, "2025-11-05"),
                row("2025-12-31", 103e6, "2026-02-25", "10-K"),
            ],
        }
    )
    assert math.isinf(before(p, "2026-03-01")), "the unguarded chooser takes the infinity row"
    value, fact, guard = after(p, "2026-03-01")
    assert value == 103e6, "the balance-sheet count, the freshest real number"
    assert fact["tag"] == "CommonStockSharesOutstanding"
    assert guard["outcome"] == "TYPE_EXCLUDED"
    assert guard["reason_code"] == "COMPANYFACTS_HIT_GUARDED"
    assert guard["rejected"] == []


def test_only_non_counts_is_an_explicit_refusal_not_a_miss():
    p = payload({COVER: [row("2025-05-01", 0, "2025-05-05")]})
    assert before(p, "2025-06-01") == 0.0
    value, fact, guard = after(p, "2025-06-01")
    assert value is None and fact is None
    assert guard["outcome"] == "REFUSED"
    assert guard["reason_code"] == "SHARES_NOT_A_COUNT"


def test_no_share_tag_at_all_is_no_candidate():
    value, fact, guard = after(payload({}), "2025-06-01")
    assert value is None and fact is None
    assert guard["outcome"] == "NO_CANDIDATE"
    assert guard["reason_code"] is None


# --- continuity ------------------------------------------------------------------


class TestContinuity:
    def setup_method(self):
        covers = [100e6, 101e6, 102e6, 103e6, 104e6, 105e6, 106e6, 107_000, 108e6]  # 8th: a 1,000x slip
        balances = [100e6, 101e6, 102e6, 103e6, 104e6, 105e6, 106e6, 107e6, 108e6]
        self.p = payload({COVER: quarterly(COVER, covers), BALANCE: quarterly(BALANCE, balances)})
        self.slip_asof = asof_of(self.p, 7)
        self.after_asof = asof_of(self.p, 8)

    def test_before_the_unguarded_chooser_takes_the_slipped_cover(self):
        assert before(self.p, self.slip_asof) == 107_000

    def test_after_the_guard_falls_through_to_the_same_filings_balance_sheet(self):
        value, fact, guard = after(self.p, self.slip_asof)
        assert value == 107e6
        assert guard["outcome"] == "FELL_THROUGH"
        assert guard["reason_code"] == "COMPANYFACTS_HIT_GUARDED"
        assert len(guard["rejected"]) == 1
        rejected = guard["rejected"][0]
        assert "EntityCommonStockSharesOutstanding" in rejected["ref"]
        assert rejected["reason"] == "SHARES_CONTRADICTED"
        assert rejected["references"][0]["basis"] == "balance_sheet"
        assert rejected["references"][0]["ratio"] < 1.0 / guard_mod.SHARES_MAX_MOVE
        assert fact["filed_date"] == self.slip_asof, "the balance-sheet count from the SAME filing"
        assert fact["derived_from"] == [
            f"companyfacts.us-gaap.CommonStockSharesOutstanding[end_date=2023-12-31,"
            f"unit=shares,filed={self.slip_asof},accn=a-{self.slip_asof}]"
        ]
        assert guard["accepted"]["by"] == "history"

    def test_healed_filing_is_the_unguarded_fact_verbatim(self):
        fact_before = unguarded(self.p, self.after_asof)
        value, fact, guard = after(self.p, self.after_asof)
        assert value == 108e6
        assert fact == fact_before
        assert guard["outcome"] == "PASS"
        assert guard["rejected"] == []

    def test_two_bases_slipped_alike_pass_unless_a_third_reference_contradicts(self):
        # The same filing carries cover AND balance sheet both at 107,000: they corroborate
        # each other and, with nothing else to consult, pass -- the residual the ruling
        # accepted rather than guess. Here the third reference is the diluted
        # weighted-average count in that filing (the sibling tool used its release's
        # market share count, which this repository has no equivalent of at the chooser).
        covers = [100e6, 101e6, 102e6, 103e6, 104e6, 105e6, 106e6, 107_000]
        balances = [100e6, 101e6, 102e6, 103e6, 104e6, 105e6, 106e6, 107_000]
        p = payload({COVER: quarterly(COVER, covers), BALANCE: quarterly(BALANCE, balances)})
        asof = asof_of(p, 7)
        assert before(p, asof) == 107_000
        value, _fact, guard = after(p, asof)
        assert value == 107_000, "two bases agreeing: accepted regardless of history"
        assert guard["accepted"]["by"] == "reference"

        p = with_wad(p, [100e6, 101e6, 102e6, 103e6, 104e6, 105e6, 106e6, 107e6])
        value, _fact, guard = after(p, asof)
        assert value is None, (
            "the diluted count against both: a tie on the cover goes to history, which "
            "rejects it; the balance sheet is then contradicted outright"
        )
        assert guard["outcome"] == "REFUSED"
        assert {r["reason"] for r in guard["rejected"]} == {
            "SHARES_DISCONTINUITY",
            "SHARES_CONTRADICTED",
        }
        assert guard["reason_code"] == "SHARES_DISCONTINUITY"
        assert guard["references_conflicted"] is True

    def test_cover_only_history_every_candidate_slipped_is_unknown(self):
        covers = [100e6, 101e6, 102e6, 103e6, 104e6, 105e6, 106e6, 107_000]
        p = payload({COVER: quarterly(COVER, covers)})
        value, _fact, guard = after(p, asof_of(p, 7))
        assert value is None
        assert guard["reason_code"] == "SHARES_DISCONTINUITY"

    def test_a_fifty_for_one_split_passes(self):
        covers = [10e6, 10e6, 10e6, 10e6, 500e6, 500e6]  # a 50-for-1 forward split
        balances = [10e6, 10e6, 10e6, 10e6, 500e6, 500e6]
        p = payload({COVER: quarterly(COVER, covers), BALANCE: quarterly(BALANCE, balances)})
        asof = asof_of(p, 4)  # the 5th filing, the first post-split one, is the latest
        value, fact, guard = after(p, asof)
        assert value == 500e6
        assert fact == unguarded(p, asof)
        assert guard["rejected"] == []
        assert guard["accepted"]["by"] == "reference"
        # With no second basis in the filing, history alone must still let it through.
        p = payload({COVER: quarterly(COVER, covers)})
        value, _fact, guard = after(p, asof)
        assert value == 500e6
        assert guard["accepted"]["by"] == "history"
        assert guard["accepted"]["ratio"] == 50.0

    def test_first_filing_has_no_prior_and_is_accepted_unchecked(self):
        p = payload(
            {COVER: [row("2025-05-01", 100e6, "2025-05-05")], BALANCE: [row("2025-03-31", 100e6, "2025-05-05")]}
        )
        value, _fact, guard = after(p, "2025-06-01")
        assert value == 100e6
        assert guard["accepted"]["by"] == "reference", "the same filing's balance sheet agrees"
        p = payload({COVER: [row("2025-05-01", 100e6, "2025-05-05")]})
        value, _fact, guard = after(p, "2025-06-01")
        assert value == 100e6
        assert guard["unchecked"] is True
        assert guard["accepted"]["by"] == "unchecked"
        assert guard["outcome"] == "PASS"


class TestConsecutiveSlips:
    """Two filings in a row carry the same thousandfold slip. Against the single prior
    filing the second slip passes at a ratio of 1.0; against the median of a window it
    does not. Both answers, on record."""

    def setup_method(self):
        covers = [100e6, 101e6, 102e6, 103e6, 104e6, 105e6, 106_000, 107_000]
        self.p = payload({COVER: quarterly(COVER, covers)})  # no balance sheet: history only
        self.asof = asof_of(self.p, 7)

    def test_single_prior_form_lets_the_second_slip_through(self):
        value, _fact, guard = after(self.p, self.asof, prior_window=1)
        assert value == 107_000, "ratio 1.0 against the previous, equally slipped, filing"
        assert guard["rejected"] == []

    def test_windowed_form_catches_it(self):
        value, _fact, guard = after(self.p, self.asof)
        assert value is None, "no other basis to fall to: an honest UNKNOWN"
        assert guard["reason_code"] == "SHARES_DISCONTINUITY"


# --- the completion's refinements, each on the shape that forced it -------------


def test_a_basis_wrong_more_often_than_right_is_settled_by_the_diluted_count():
    # Every 10-Q cover x1,000, every 10-K right, no balance-sheet tag.
    covers, wads = [], []
    for i in range(8):
        is_k = i % 4 == 3
        covers.append(23e6 + i * 1e5 if is_k else (23e6 + i * 1e5) * 1000)
        wads.append(23e6 + i * 1e5)
    p = with_wad(payload({COVER: quarterly(COVER, covers)}), wads)
    for k in range(4, 8):
        asof = asof_of(p, k)
        value, _fact, guard = after(p, asof)
        if k % 4 == 3:  # the 10-K: right, corroborated regardless of a mostly-wrong history
            assert value == covers[k]
            assert guard["accepted"]["by"] == "reference"
        else:  # the 10-Q: contradicted by its own diluted count, and no other basis
            assert before(p, asof) == covers[k], "the unguarded chooser believes the slip"
            assert value is None
            assert guard["reason_code"] == "SHARES_CONTRADICTED"


def test_a_transformation_corroborated_in_the_same_filing_beats_history():
    covers = [1_000, 1_000, 1_071_666_977]  # a shell's 1,000 shares, then the merged company
    balances = [1_000, 1_000, 1_071_666_977]
    p = payload({COVER: quarterly(COVER, covers), BALANCE: quarterly(BALANCE, balances)})
    asof = asof_of(p, 2)
    value, _fact, guard = after(p, asof)
    assert value == 1_071_666_977
    assert guard["accepted"]["by"] == "reference"
    # ...and with the diluted count as the only reference, the same.
    p = with_wad(payload({COVER: quarterly(COVER, covers)}), [1_000, 1_000, 1_098_000_000])
    value, _fact, guard = after(p, asof)
    assert value == 1_071_666_977
    assert guard["accepted"]["by"] == "reference"


def test_disagreement_band_falls_to_history_with_a_note():
    # Cover 100M, same-filing balance sheet 30M (3.3x: between the bands) -> judged by
    # history like a single-basis filing; the note is set.
    covers = [100e6, 100e6, 100e6, 100e6, 100e6]
    balances = [100e6, 100e6, 100e6, 100e6, 30e6]
    p = payload({COVER: quarterly(COVER, covers), BALANCE: quarterly(BALANCE, balances)})
    value, _fact, guard = after(p, asof_of(p, 4))
    assert value == 100e6
    assert guard["accepted"]["by"] == "history"
    assert guard["bases_disagreed"] is True


def test_a_stale_cover_is_not_the_current_count():
    # A company whose only non-dimensional cover count is from 2010.
    p = payload(
        {COVER: [row("2009-11-13", 470_210_301, "2009-11-20"), row("2010-01-27", 469_280_842, "2010-02-03", "10-K")]}
    )
    assert before(p, "2026-08-01") == 469_280_842, "the unguarded chooser returns 2010's count"
    value, _fact, guard = after(p, "2026-08-01")
    assert value is None
    assert guard["reason_code"] == "SHARES_STALE"
    assert guard["rejected"][0]["age_days"] > guard_mod.SHARES_STALE_DAYS
    value, _fact, _guard = after(p, "2010-06-01")
    assert value == 469_280_842, "fresh at the time"


def test_the_fall_through_never_reaches_an_older_fiscal_year():
    # Staleness stops a years-old fall-through first ...
    p = payload(
        {
            COVER: [row("2024-08-19", 13_326_944, "2024-08-21"), row("2025-04-28", 30_959, "2025-04-30", "10-K/A")],
            BALANCE: [row("2021-10-29", 3_162_500, "2021-11-15")],
        }
    )
    value, _fact, guard = after(p, "2025-05-15")
    assert value is None
    assert {r["reason"] for r in guard["rejected"]} == {"SHARES_DISCONTINUITY", "SHARES_STALE"}
    # ... and the fiscal-year fence stops one that is not yet stale but is from an earlier
    # fiscal year than the refused count.
    p = payload(
        {
            COVER: [row("2024-08-19", 13_326_944, "2024-08-21"), row("2025-04-28", 30_959, "2025-04-30", "10-K/A")],
            BALANCE: [row("2024-04-20", 3_162_500, "2024-05-10")],  # 376 days before the as-of
        }
    )
    value, _fact, guard = after(p, "2025-05-01")
    assert value is None
    assert guard["stopped"]["reason"] == "SHARES_FALLTHROUGH_TOO_OLD"
    assert guard["reason_code"] == "SHARES_DISCONTINUITY", "the cover was rejected by history first"


class TestConflictingReferences:
    """The same filing's diluted weighted-average count filed in thousands contradicts a
    count the balance sheet corroborates. Under "contradiction wins" any contradiction
    rejects; under the majority rule the two references tie and history decides."""

    def setup_method(self):
        covers = [8.5e6, 8.6e6, 8.64e6]
        balances = [8.5e6, 8.6e6, 8.568e6]
        self.p = with_wad(
            payload({COVER: quarterly(COVER, covers), BALANCE: quarterly(BALANCE, balances)}),
            [8_500, 8_600, 8_470],  # in thousands
        )
        self.asof = asof_of(self.p, 2)

    def test_contradiction_wins_rejects_on_the_single_contradiction(self):
        value, _fact, guard = after(self.p, self.asof, references_decide="contradiction_wins")
        assert value is None
        assert guard["reason_code"] == "SHARES_CONTRADICTED"

    def test_majority_tie_falls_to_history_with_the_note(self):
        value, _fact, guard = after(self.p, self.asof)
        assert value == 8.64e6
        assert guard["accepted"]["by"] == "history"
        assert guard["references_conflicted"] is True
        assert guard["bases_disagreed"] is True

    def test_majority_still_rejects_a_real_slip(self):
        # A 10-Q cover x1,000 and the same filing's diluted count against it.
        p = payload({COVER: [row("2025-08-04", 23_544_479_000, "2025-08-07")]})
        p["facts"]["us-gaap"] = {
            WAD[1]: {"units": {"shares": [row("2025-06-30", 23_402_000, "2025-08-07") | {"start": "2025-04-01"}]}}
        }
        value, _fact, guard = after(p, "2025-08-10")
        assert value is None
        assert guard["reason_code"] == "SHARES_CONTRADICTED"


# --- the real payloads ------------------------------------------------------------


class TestResMed:
    """ResMed FY2021: the cover page of the 10-K (2021-08-17) and of the 10-K/A
    (2021-08-19) both read 145,681 shares; the 10-K's balance sheet reads 145,648,358."""

    def setup_method(self):
        self.p = real("RMD_0000943819.json")

    def test_before_the_cover_page_wins_a_thousandfold_small(self):
        fact = unguarded(self.p, "2021-09-15")
        assert fact["value"] == 145_681.0
        assert "accn=0000943819-21-000020" in fact["derived_from"][0], "the 10-K/A's cover page"

    def test_after_the_guard_takes_the_balance_sheet_count(self):
        value, fact, guard = after(self.p, "2021-09-15")
        assert value == 145_648_358.0
        assert fact["fact_end_date"] == "2021-06-30"
        assert fact["filed_date"] == "2021-08-17"
        assert fact["derived_from"] == [
            "companyfacts.us-gaap.CommonStockSharesOutstanding[end_date=2021-06-30,"
            "unit=shares,filed=2021-08-17,accn=0000943819-21-000017]"
        ]
        assert guard["outcome"] == "FELL_THROUGH"
        assert guard["reason_code"] == "COMPANYFACTS_HIT_GUARDED"
        assert len(guard["rejected"]) == 1
        rejected = guard["rejected"][0]
        assert "accn=0000943819-21-000020" in rejected["ref"]
        # The 10-K/A re-filed only the cover, so no same-filing reference exists and
        # history decides: 145,681 against a median of ~145.5 million.
        assert rejected["reason"] == "SHARES_DISCONTINUITY"
        assert rejected["references"] == []
        assert rejected["ratio"] == pytest.approx(0.001, abs=1e-4)
        assert guard["accepted"]["by"] == "reference", "the 10-K's own diluted count agrees"

    def test_between_the_10k_and_the_10ka_the_same_filing_contradicts_the_cover(self):
        value, _fact, guard = after(self.p, "2021-08-18")
        assert value == 145_648_358.0
        rejected = guard["rejected"][0]
        assert rejected["reason"] == "SHARES_CONTRADICTED"
        assert [(r["basis"], r["verdict"]) for r in rejected["references"]] == [
            ("balance_sheet", "contradicts"),
            ("diluted_weighted_average", "contradicts"),
            ("net_income_per_diluted_eps", "contradicts"),
        ]

    def test_single_prior_form_would_not_have_caught_resmed(self):
        # The 10-K/A re-filed the 10-K's 145,681: against the single prior filing the
        # ratio is exactly 1.0.
        value, _fact, _guard = after(self.p, "2021-09-15", prior_window=1)
        assert value == 145_681.0

    def test_extract_company_facts_asof_carries_the_guarded_count_and_the_record(self):
        extracted = extract_company_facts_asof(self.p, "2021-09-15")
        assert extracted["shares_outstanding_asof"]["value"] == 145_648_358.0
        assert extracted["shares_outstanding_asof"]["resolution"] == "RESOLVED"
        assert extracted["shares_outstanding_guard"]["outcome"] == "FELL_THROUGH"


def test_chesapeake_utilities_10q_cover_slip_is_refused_and_its_10k_passes():
    p = real("CPK_0000019745.json")
    # The 10-Q of 2025-08-07: cover 23,544,479,000; the same filing's diluted count 23.4M.
    assert before(p, "2025-08-10") == 23_544_479_000.0
    value, fact, guard = after(p, "2025-08-10")
    assert value is None and fact is None
    assert guard["reason_code"] == "SHARES_CONTRADICTED"
    assert [(r["basis"], r["verdict"]) for r in guard["rejected"][0]["references"]] == [
        ("diluted_weighted_average", "contradicts"),
        ("net_income_per_diluted_eps", "contradicts"),
    ]
    # The 10-K of 2025-02-26 is right, and its own diluted count says so.
    value, fact, guard = after(p, "2025-03-01")
    assert value == 22_982_417.0
    assert fact == unguarded(p, "2025-03-01")
    assert guard["outcome"] == "PASS"
    assert guard["accepted"]["by"] == "reference"


def test_packaging_corp_2026_10k_cover_slip_is_refused():
    p = real("PKG_0000075677.json")
    assert before(p, "2026-03-15") == 89_213_394_000.0
    value, _fact, guard = after(p, "2026-03-15")
    assert value is None
    assert guard["reason_code"] == "SHARES_CONTRADICTED"
    assert guard["rejected"][0]["references"][0]["ratio"] == pytest.approx(995.685, abs=1e-3)
    assert guard["rejected"][0]["references"][1]["basis"] == "net_income_per_diluted_eps"
    assert guard["rejected"][0]["references"][1]["ratio"] == pytest.approx(988.827, abs=1e-3)
    # The next 10-Q is right again.
    value, _fact, guard = after(p, "2026-05-10")
    assert value == 89_098_647.0
    assert guard["outcome"] == "PASS"


@pytest.mark.parametrize("name", ["RMD_0000943819.json", "CPK_0000019745.json", "PKG_0000075677.json"])
def test_wherever_nothing_is_refused_the_unguarded_fact_is_returned_verbatim(name):
    p = real(name)
    filed = sorted(
        {
            r["filed"]
            for tags in p["facts"].values()
            for node in tags.values()
            for rows in node["units"].values()
            for r in rows
        }
    )
    touched = []
    for asof in filed:
        fact_before = unguarded(p, asof)
        _value, fact, guard = after(p, asof)
        if guard["rejected"] or guard["outcome"] != "PASS":
            touched.append(asof)
            continue
        assert fact == fact_before, asof  # the chooser's own fact, every key
    assert len(touched) < len(filed) / 2, "the guard touches the slips, not the company"
