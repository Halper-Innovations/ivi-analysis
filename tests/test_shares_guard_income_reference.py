"""The share-count guard's third reference -- net income / EPS in the same filing -- and its
pre-conversion rule for a first filing after an IPO (app/market/shares_guard.py).

The defect: a filing's diluted weighted-average count is itself sometimes filed on
another scale, and alone it contradicted, and so refused, a correct cover-page count. About
ten correct counts were refused this way; McDonald's (diluted count filed as 713.5 "shares")
and Landmark Bancorp (filed a thousandfold too large) are pinned from trimmed real payloads.
The income reference ties that vote and history decides, while the slips the guard exists
for -- ResMed FY2021, Chesapeake Utilities' 10-Qs, Packaging Corp's 2026 10-K -- stay
refused (tests/test_shares_guard.py, whose fixtures now carry the same filings' income and
EPS rows). Values are raw share counts unless a name says millions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.market import shares_guard as guard_mod
from app.market.company_facts_extract import (
    SHARES_TAG_PRIORITY,
    _extract_from_priority,
    extract_shares_outstanding_asof,
)

FIXTURES = Path(__file__).parent / "fixtures" / "companyfacts_shares_guard"
ACCN = "a-2025-08-05"


def _real(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _unguarded(p: dict, asof: str):
    fact = _extract_from_priority(
        companyfacts=p,
        as_of_date=asof,
        priority=SHARES_TAG_PRIORITY,
        expected_unit_exact=("shares",),
        expected_unit_prefixes=("shares",),
    )
    return None if fact is None else fact["value"]


def _duration(val, *, start="2025-04-01", end="2025-06-30", accn=ACCN, filed="2025-08-05"):
    return {"start": start, "end": end, "val": val, "accn": accn, "filed": filed, "form": "10-Q"}


def _payload(*, cover, wad=None, income=None, eps=None, income_tag="NetIncomeLoss",
             eps_tag="EarningsPerShareDiluted", prior_covers=(100e6, 101e6, 102e6, 103e6)):
    """Four prior quarterly cover counts, then the filing under test (filed 2025-08-05)."""
    covers = [
        {"end": f"2024-{m:02d}-01", "val": v, "accn": f"p-{m}", "filed": f"2024-{m:02d}-05",
         "form": "10-Q"}
        for m, v in zip((2, 5, 8, 11), prior_covers, strict=True)
    ]
    covers.append({"end": "2025-08-01", "val": cover, "accn": ACCN, "filed": "2025-08-05",
                   "form": "10-Q"})
    us_gaap: dict = {}
    if wad is not None:
        us_gaap["WeightedAverageNumberOfDilutedSharesOutstanding"] = {
            "units": {"shares": [_duration(wad)]}
        }
    if income is not None:
        us_gaap[income_tag] = {"units": {"USD": [_duration(income)]}}
    if eps is not None:
        us_gaap[eps_tag] = {"units": {"USD/shares": [_duration(eps)]}}
    return {
        "facts": {
            "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": covers}}},
            "us-gaap": us_gaap,
        }
    }


def _after(p: dict, asof: str = "2025-08-10"):
    fact, guard = extract_shares_outstanding_asof(p, asof)
    return (None if fact is None else fact["value"]), guard


# --- the income reference itself ---------------------------------------------------


def test_a_diluted_count_filed_in_millions_no_longer_refuses_a_right_cover_alone():
    # Diluted count filed as 104 "shares" (millions): alone it contradicts the cover.
    p = _payload(cover=104e6, wad=104.0)
    value, guard = _after(p)
    assert value is None
    assert guard["reason_code"] == "SHARES_CONTRADICTED"
    # Net income 208m / diluted EPS 2.00 = 104m: the income reference corroborates, and
    # being computed from two USD figures it settles the scale -- the cover is accepted.
    p = _payload(cover=104e6, wad=104.0, income=208e6, eps=2.00)
    value, guard = _after(p)
    assert value == 104e6
    assert guard["outcome"] == "PASS"
    assert guard["references_conflicted"] is True
    assert guard["accepted"]["by"] == "reference"
    reference = guard["accepted"]["references"][1]
    assert reference["basis"] == "net_income_per_diluted_eps"
    assert reference["value"] == 104e6
    assert reference["verdict"] == "corroborates"
    assert (reference["net_income"], reference["eps"]) == (208e6, 2.00)


def test_the_income_reference_joins_the_diluted_count_against_a_slipped_cover():
    p = _payload(cover=104e9, wad=104e6, income=208e6, eps=2.00)
    value, guard = _after(p)
    assert value is None
    assert [(r["basis"], r["verdict"]) for r in guard["rejected"][0]["references"]] == [
        ("diluted_weighted_average", "contradicts"),
        ("net_income_per_diluted_eps", "contradicts"),
    ]


@pytest.mark.parametrize(
    ("income", "eps"),
    [
        (2e6, 0.04),  # |EPS| below 0.05: rounding to the cent would dominate
        (-208e6, 2.00),  # a loss with a positive EPS: signs disagree
        (0.0, 0.10),  # no income
    ],
)
def test_an_unusable_income_pair_is_no_reference(income, eps):
    p = _payload(cover=104e6, wad=104.0, income=income, eps=eps)
    value, guard = _after(p)
    assert value is None, "the diluted count alone still decides"
    assert [r["basis"] for r in guard["rejected"][0]["references"]] == ["diluted_weighted_average"]


def test_a_loss_divided_by_a_loss_per_share_is_a_count():
    p = _payload(cover=104e6, wad=104.0, income=-156e6, eps=-1.50)
    value, guard = _after(p)
    assert value == 104e6
    assert guard["accepted"]["references"][1]["value"] == 104e6


def test_basic_eps_is_used_only_when_no_diluted_pair_exists():
    p = _payload(cover=104e6, wad=104.0, income=208e6, eps=2.00, eps_tag="EarningsPerShareBasic")
    value, guard = _after(p)
    assert value == 104e6
    assert guard["accepted"]["references"][1]["basis"] == "net_income_per_basic_eps"


def test_profit_loss_stands_in_when_net_income_loss_is_not_tagged():
    # Ashland tags only ProfitLoss.
    p = _payload(cover=104e6, wad=104.0, income=208e6, eps=2.00, income_tag="ProfitLoss")
    value, guard = _after(p)
    assert value == 104e6
    assert "ProfitLoss" in guard["accepted"]["references"][1]["ref"]


def test_the_freshest_period_then_the_longest_duration_is_used():
    p = _payload(cover=104e6, wad=104e6)
    six_months = {"start": "2025-01-01", "end": "2025-06-30"}
    last_year = {"start": "2024-04-01", "end": "2024-06-30"}
    p["facts"]["us-gaap"]["NetIncomeLoss"] = {
        "units": {"USD": [_duration(100e6), _duration(300e6, **six_months), _duration(90e6, **last_year)]}
    }
    p["facts"]["us-gaap"]["EarningsPerShareDiluted"] = {
        "units": {
            "USD/shares": [_duration(1.0), _duration(3.0, **six_months), _duration(0.1, **last_year)]
        }
    }
    reference = guard_mod._income_reference(p, ACCN, guard_mod._parse("2025-08-10"))
    assert reference["value"] == 100e6  # 300m / 3.00 over the six months to June
    assert "start=2025-01-01" in reference["ref"]


def test_another_filings_income_is_never_a_reference():
    p = _payload(cover=104e6, wad=104.0, income=208e6, eps=2.00)
    for tag in ("NetIncomeLoss", "EarningsPerShareDiluted"):
        for rows in p["facts"]["us-gaap"][tag]["units"].values():
            rows[0]["accn"] = "another-filing"
    value, _guard = _after(p)
    assert value is None


# --- the real payloads ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("asof", "expected", "wad_ratio"),
    [
        ("2025-08-10", 713_604_434.0, 994_432.043),
        ("2026-05-10", 710_505_859.0, 995_803.587),
    ],
)
def test_mcdonalds_cover_count_is_accepted_despite_its_diluted_count_in_millions(
    asof, expected, wad_ratio
):
    p = _real("MCD_0000063908.json")
    value, guard = _after(p, asof)
    assert value == expected == _unguarded(p, asof)
    assert guard["outcome"] == "PASS"
    assert guard["accepted"]["by"] == "reference"
    wad, income = guard["accepted"]["references"]
    assert (wad["basis"], wad["verdict"]) == ("diluted_weighted_average", "contradicts")
    assert wad["ratio"] == pytest.approx(wad_ratio, abs=1e-3)
    assert (income["basis"], income["verdict"]) == ("net_income_per_diluted_eps", "corroborates")


def test_landmark_bancorp_cover_count_is_accepted_despite_a_thousandfold_diluted_count():
    p = _real("LARK_0001141688.json")
    value, guard = _after(p, "2026-05-20")
    assert value == 6_097_552.0
    assert guard["outcome"] == "PASS"
    wad, income = guard["accepted"]["references"]
    assert wad["ratio"] == pytest.approx(0.001, abs=1e-5)
    assert income["value"] == pytest.approx(5_066_000 / 0.83)
    assert income["verdict"] == "corroborates"


# --- pre-conversion references (a first filing after an IPO) ----------------------------


def test_kailera_first_filing_cover_is_not_refused_by_its_pre_ipo_balance_sheet():
    """Kailera's first 10-Q: 29,953 common shares and 78.8 million convertible preferred at
    2026-03-31; 129.6 million common on the cover dated after the IPO. Before this rule the
    cover was refused and the guard fell through to the 29,953 pre-IPO common count."""
    p = _real("KLRA_0002096997.json")
    value, guard = _after(p, "2026-09-28")
    assert value == 129_565_608.0
    assert guard["outcome"] == "PASS"
    assert guard["unchecked"] is True
    assert guard["references_pre_conversion"] is True
    assert {r["verdict"] for r in guard["accepted"]["references"]} == {"pre_conversion"}


def test_without_convertible_preferred_the_same_shape_is_still_a_contradiction():
    p = _real("KLRA_0002096997.json")
    del p["facts"]["us-gaap"]["TemporaryEquitySharesOutstanding"]
    value, guard = _after(p, "2026-09-28")
    assert value == 29_953.0, "the pre-rule answer: the cover refused, the balance sheet taken"
    assert guard["outcome"] == "FELL_THROUGH"


def test_pre_conversion_references_do_not_unlock_a_slip_that_history_catches():
    p = _payload(cover=104e9, wad=104e6, income=208e6, eps=2.00)
    p["facts"]["us-gaap"]["CommonStockSharesOutstanding"] = {
        "units": {"shares": [{"end": "2025-06-30", "val": 1e6, "accn": ACCN, "filed": "2025-08-05", "form": "10-Q"}]}
    }
    p["facts"]["us-gaap"]["TemporaryEquitySharesOutstanding"] = {
        "units": {"shares": [{"end": "2025-06-30", "val": 5e6, "accn": ACCN, "filed": "2025-08-05", "form": "10-Q"}]}
    }
    value, guard = _after(p)
    assert guard["references_pre_conversion"] is True
    assert guard["rejected"][0]["reason"] == "SHARES_DISCONTINUITY"
    assert value is None, "and the pre-offering balance sheet is contradicted in its own period"
    assert guard["rejected"][1]["reason"] == "SHARES_CONTRADICTED"


# --- the per-count entry point the ingest normalizer uses ------------------------------


def test_check_share_count_judges_one_given_count_as_of_its_own_filing():
    p = _payload(cover=104e9, wad=104e6, income=208e6, eps=2.00)
    judged = guard_mod.check_share_count(
        p, taxonomy="dei", tag="EntityCommonStockSharesOutstanding", value=104e9,
        end="2025-08-01", filed="2025-08-05", accn=ACCN,
    )
    assert judged["decision"] == "reject"
    assert judged["reason"] == "SHARES_CONTRADICTED"
    judged = guard_mod.check_share_count(
        p, taxonomy="dei", tag="EntityCommonStockSharesOutstanding", value=103e6,
        end="2024-11-01", filed="2024-11-05", accn="p-11",
    )
    assert (judged["decision"], judged["by"]) == ("accept", "history")
    assert guard_mod.check_share_count(
        p, taxonomy="dei", tag="EntityCommonStockSharesOutstanding", value=0,
        end="2025-08-01", filed="2025-08-05", accn=ACCN,
    )["reason"] == "SHARES_NOT_A_COUNT"


# --- the statements are one class of evidence on scale ---------------------------------


def _with_statements(p: dict, *, balance, wad, balance_end="2025-06-30"):
    p["facts"]["us-gaap"]["CommonStockSharesOutstanding"] = {
        "units": {"shares": [{"end": balance_end, "val": balance, "accn": ACCN,
                              "filed": "2025-08-05", "form": "10-Q"}]}
    }
    p["facts"]["us-gaap"]["WeightedAverageNumberOfDilutedSharesOutstanding"] = {
        "units": {"shares": [_duration(wad)]}
    }
    return p


def test_two_statement_counts_in_thousands_cast_one_vote_against_a_right_cover():
    # Balance sheet and diluted count both in thousands: one class, one vote, against the
    # income reference's agreement -- the cover stands. Two votes would have refused it and
    # fallen through to the 104,000 balance-sheet count.
    p = _with_statements(_payload(cover=104e6, income=208e6, eps=2.00), balance=104_000, wad=103_500)
    value, guard = _after(p)
    assert value == 104e6
    assert guard["accepted"]["by"] == "reference"
    # Two statement counts that do not even agree with each other still cast one vote.
    p = _with_statements(_payload(cover=104e6, income=208e6, eps=2.00), balance=10_400, wad=103_500)
    value, _guard = _after(p)
    assert value == 104e6


def test_a_statement_count_is_not_vouched_for_by_its_own_statements():
    # The cover is gone (no cover in this filing); the balance-sheet count in thousands is
    # corroborated only by the diluted count from the same statements, and the income
    # reference contradicts it: refused, not accepted on history.
    p = _with_statements(_payload(cover=104e6, income=208e6, eps=2.00), balance=104_000, wad=103_500)
    del p["facts"]["dei"]
    value, guard = _after(p)
    assert value is None
    assert guard["rejected"][0]["reason"] == "SHARES_CONTRADICTED"


def test_the_income_reference_alone_never_refuses_a_cover_after_a_large_issuance():
    """Modelled on Empery Digital's 10-Q of 2025-08-12: 533,008 shares at 30 June, a large
    placement in July, 47,444,907 on the cover dated 8 August. The balance-sheet and
    diluted counts sit in the 2x-100x band (a real move, not a slip); the income
    reference, a quarter's average, lands at 111x. Alone it would refuse the cover and
    fall through to the stale 30 June count; it only confirms or ties the statements."""
    p = _with_statements(
        _payload(
            cover=47_444_907,
            income=-2_300_000,
            eps=-5.38,  # 427,509 shares on average over the quarter
            prior_covers=(520_000, 525_000, 530_000, 532_000),
        ),
        balance=533_008,
        wad=515_500,
    )
    value, guard = _after(p)
    assert value == 47_444_907
    assert guard["accepted"]["by"] == "history"
    income = guard["accepted"]["references"][2]
    assert (income["basis"], income["verdict"]) == ("net_income_per_diluted_eps", "contradicts")


def test_a_right_cover_after_a_run_of_slipped_covers_is_accepted_on_the_income_reference():
    # Garmin: its 2016 10-Q covers read 208 and 198 billion; its 10-K cover (198 million)
    # is right, but the median of its last four covers is a slip, so history refuses it.
    # The income reference agrees with the cover and settles the scale.
    p = _with_statements(
        _payload(
            cover=198_077_418,
            income=510_800_000,
            eps=2.70,  # 189.2 million shares on average
            prior_covers=(208_077_418, 208_077_418_000, 208_077_418_000, 198_077_418_000),
        ),
        balance=188_565,  # the statements in thousands
        wad=189_343,
    )
    value, guard = _after(p)
    assert value == 198_077_418
    assert guard["accepted"]["by"] == "reference"
    # Without the income reference the statements in thousands refuse the cover outright.
    del p["facts"]["us-gaap"]["NetIncomeLoss"]
    _value, guard = _after(p)
    assert (guard["rejected"][0]["value"], guard["rejected"][0]["reason"]) == (
        198_077_418,
        "SHARES_CONTRADICTED",
    )
