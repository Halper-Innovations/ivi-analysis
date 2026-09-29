"""Restatement selection in the component-summed and grouped-sum resolvers must
not depend on SEC array order.

Both resolvers used to keep the LAST array element per (period, tag), so a
restated comparative could lose to the stale original it restates purely
because of where it sat in the array. The rule now matches the single-tag
path: same tag, later ``filed`` wins; on the same filed date an amended form
wins; point-in-time visibility (filed <= as_of) still applies first.

The live excerpt is public SEC data (CIK 0000040545, one tag, all twelve
facts, in the order the API served them).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from app.ingest.companyfacts import (
    normalize_annual_facts_from_raw,
    normalize_quarterly_facts_from_raw,
)

_EXCERPT = (
    Path(__file__).parent
    / "fixtures"
    / "companyfacts"
    / "DEBT_RESTATED_COMPARATIVE_0000040545.json"
)


def _excerpt() -> dict[str, Any]:
    return json.loads(_EXCERPT.read_text(encoding="utf-8"))


def _reversed_arrays(raw: dict[str, Any]) -> dict[str, Any]:
    flipped = copy.deepcopy(raw)
    for tags in flipped["facts"].values():
        for node in tags.values():
            for unit, facts in node["units"].items():
                node["units"][unit] = list(reversed(facts))
    return flipped


def _series(
    raw: dict[str, Any], line_item: str, *, as_of: str | None
) -> dict[int, tuple[float, str, str, str]]:
    rows = normalize_annual_facts_from_raw(raw, cik="0000000001", years_back=10, filed_as_of=as_of)
    return {
        int(r["fiscal_year"]): (r["value"], r["filed_date"], r["form"], r["accession"])
        for r in rows
        if r["line_item"] == line_item
    }


def _fact(end: str, val: float, form: str, filed: str, accn: str, **extra: Any) -> dict[str, Any]:
    return {"end": end, "val": val, "form": form, "filed": filed, "accn": accn, **extra}


def _payload(tags: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {"facts": {"us-gaap": {tag: {"units": {"USD": facts}} for tag, facts in tags.items()}}}


# ── live excerpt: direct total tag (component-summed resolver) ────────────────


def test_live_excerpt_restated_comparative_wins_in_both_array_orders():
    raw = _excerpt()
    served = _series(raw, "total_debt", as_of="2017-06-01")
    flipped = _series(_reversed_arrays(raw), "total_debt", as_of="2017-06-01")
    assert served == flipped
    # The FY2016 10-K restated the 2015-12-31 balance to 197,602; the stale
    # FY2015 original (198,276) sat LAST in the served array and used to win.
    assert served == {
        2014: (261424.0, "2016-02-26", "10-K", "0000040545-16-000145"),
        2015: (197602.0, "2017-02-24", "10-K", "0000040545-17-000010"),
        2016: (136210.0, "2017-02-24", "10-K", "0000040545-17-000010"),
    }


def test_live_excerpt_restatement_filed_after_as_of_is_invisible():
    raw = _excerpt()
    for payload in (raw, _reversed_arrays(raw)):
        assert _series(payload, "total_debt", as_of="2017-02-23") == {
            2014: (261424.0, "2016-02-26", "10-K", "0000040545-16-000145"),
            2015: (198276.0, "2016-02-26", "10-K", "0000040545-16-000145"),
        }


def test_live_excerpt_restatement_filed_on_as_of_is_visible():
    raw = _excerpt()
    for payload in (raw, _reversed_arrays(raw)):
        assert _series(payload, "total_debt", as_of="2017-02-24")[2015] == (
            197602.0,
            "2017-02-24",
            "10-K",
            "0000040545-17-000010",
        )


# ── synthetic: current + noncurrent component sum ─────────────────────────────


def _component_debt_payload() -> dict[str, Any]:
    return _payload(
        {
            "DebtCurrent": [
                _fact("2014-12-31", 100e6, "10-K", "2015-02-01", "orig"),
                _fact("2014-12-31", 200e6, "10-K", "2016-02-01", "restated"),
            ],
            "LongTermDebtNoncurrent": [
                _fact("2014-12-31", 50e6, "10-K", "2015-02-01", "orig"),
            ],
        }
    )


def test_component_sum_uses_later_filed_component_in_both_orders():
    raw = _component_debt_payload()
    asc = _series(raw, "total_debt", as_of="2016-06-01")
    desc = _series(_reversed_arrays(raw), "total_debt", as_of="2016-06-01")
    assert asc == desc == {2014: (250.0, "2016-02-01", "10-K", "restated")}


def test_component_sum_as_of_boundary():
    raw = _component_debt_payload()
    for payload in (raw, _reversed_arrays(raw)):
        assert _series(payload, "total_debt", as_of="2016-01-31") == {
            2014: (150.0, "2015-02-01", "10-K", "orig")
        }
        assert _series(payload, "total_debt", as_of="2016-02-01") == {
            2014: (250.0, "2016-02-01", "10-K", "restated")
        }


def test_operating_lease_direct_total_uses_later_filed_value_in_both_orders():
    raw = _payload(
        {
            "OperatingLeaseLiability": [
                _fact("2020-12-31", 80e6, "10-K", "2021-12-01", "restated"),
                _fact("2020-12-31", 40e6, "10-K", "2021-02-01", "orig"),
            ]
        }
    )
    for payload in (raw, _reversed_arrays(raw)):
        assert _series(payload, "operating_lease_liability", as_of="2022-01-01") == {
            2020: (80.0, "2021-12-01", "10-K", "restated")
        }
        assert _series(payload, "operating_lease_liability", as_of="2021-11-30") == {
            2020: (40.0, "2021-02-01", "10-K", "orig")
        }


def test_same_filed_date_amendment_beats_original_in_both_orders():
    raw = _payload(
        {
            "DebtAndCapitalLeaseObligations": [
                _fact("2014-12-31", 300e6, "10-K", "2015-03-01", "orig"),
                _fact("2014-12-31", 310e6, "10-K/A", "2015-03-01", "amend"),
            ]
        }
    )
    for payload in (raw, _reversed_arrays(raw)):
        assert _series(payload, "total_debt", as_of="2020-01-01") == {
            2014: (310.0, "2015-03-01", "10-K/A", "amend")
        }


def test_exact_tie_is_resolved_the_same_way_in_both_orders():
    """Same filed date, same form, two accessions: no economic winner, but the
    pick must not depend on array order."""
    raw = _payload(
        {
            "DebtAndCapitalLeaseObligations": [
                _fact("2014-12-31", 300e6, "10-K", "2015-03-01", "0000000001-15-000001"),
                _fact("2014-12-31", 320e6, "10-K", "2015-03-01", "0000000001-15-000002"),
            ]
        }
    )
    for payload in (raw, _reversed_arrays(raw)):
        assert _series(payload, "total_debt", as_of="2020-01-01") == {
            2014: (320.0, "2015-03-01", "10-K", "0000000001-15-000002")
        }


def test_quarterly_component_resolver_uses_later_filed_value_in_both_orders():
    raw = _payload(
        {
            "DebtAndCapitalLeaseObligations": [
                _fact("2016-06-30", 900e6, "10-Q/A", "2016-11-09", "amend", fy=2016, fp="Q2"),
                _fact("2016-06-30", 800e6, "10-Q", "2016-08-01", "orig", fy=2016, fp="Q2"),
            ]
        }
    )
    results = []
    for payload in (raw, _reversed_arrays(raw)):
        rows = normalize_quarterly_facts_from_raw(payload, cik="0000000001", years_back=20)
        results.append(
            [
                (r["fiscal_year"], r["period_type"], r["value"], r["accession"])
                for r in rows
                if r["line_item"] == "total_debt"
            ]
        )
    assert results[0] == results[1] == [(2016, "Q2", 900.0, "amend")]


# ── synthetic: grouped-sum resolver (preferred equity, NCI) ───────────────────


def test_preferred_equity_uses_later_filed_value_in_both_orders():
    raw = _payload(
        {
            "PreferredStockValue": [
                _fact("2014-12-31", 300e6, "10-K", "2015-02-01", "a1"),
                _fact("2014-12-31", 700e6, "10-K/A", "2016-02-01", "a2"),
            ]
        }
    )
    asc = _series(raw, "preferred_equity", as_of="2016-06-01")
    desc = _series(_reversed_arrays(raw), "preferred_equity", as_of="2016-06-01")
    assert asc == desc == {2014: (700.0, "2016-02-01", "10-K/A", "a2")}


def test_preferred_equity_restatement_after_as_of_is_invisible_and_boundary_included():
    raw = _payload(
        {
            "PreferredStockValue": [
                _fact("2014-12-31", 700e6, "10-K/A", "2016-02-01", "a2"),
                _fact("2014-12-31", 300e6, "10-K", "2015-02-01", "a1"),
            ]
        }
    )
    for payload in (raw, _reversed_arrays(raw)):
        assert _series(payload, "preferred_equity", as_of="2016-01-31") == {
            2014: (300.0, "2015-02-01", "10-K", "a1")
        }
        assert _series(payload, "preferred_equity", as_of="2016-02-01") == {
            2014: (700.0, "2016-02-01", "10-K/A", "a2")
        }


def test_noncontrolling_interest_sum_uses_later_filed_component_in_both_orders():
    raw = _payload(
        {
            "MinorityInterest": [
                _fact("2014-12-31", 50e6, "10-K", "2015-02-01", "orig"),
                _fact("2014-12-31", 65e6, "10-K", "2016-02-01", "restated"),
            ],
            "RedeemableNoncontrollingInterestEquityCarryingAmount": [
                _fact("2014-12-31", 30e6, "10-K", "2015-02-01", "orig"),
            ],
        }
    )
    asc = _series(raw, "noncontrolling_interest", as_of="2016-06-01")
    desc = _series(_reversed_arrays(raw), "noncontrolling_interest", as_of="2016-06-01")
    assert asc == desc == {2014: (95.0, "2016-02-01", "10-K", "restated")}
