"""Three owner's-delegate decisions from the 2026-09-29 basket review.

1. Short-term investments count as cash in net debt. Only cash and equivalents
   used to offset debt, so Microsoft's 55.9 billion of short-term investments
   (June 2026) left its net debt at +19.4 billion when its balance sheet shows
   net cash of 36.5 billion. Only CURRENT short-term investments from the cash's
   own balance-sheet date count, never long-term investments, and a combined
   cash-and-short-term-investments line is never counted twice.
2. Real-estate investment trusts (SEC SIC 6798): earnings power and EV/EBIT are
   NOT_APPLICABLE (REIT_DEPRECIATION_DISTORTS_EARNINGS). Other methods unchanged.
3. The capital-structure ratios and the discount rate's leverage rule refuse
   debt from a year more than one fiscal year behind the latest balance sheet,
   the rule net debt already used (Ford).

The MSFT and KO fixtures are the issuers' real SEC companyfacts, trimmed to the
concepts and 10-K balance-sheet dates these tests read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingest.companyfacts import normalize_annual_facts_from_raw
from app.market.company_facts_extract import short_term_investments_addition
from app.valuation.net_debt import resolve_net_debt_proxy
from tests.test_basket_net_debt import _init_cfg, _scorecard, _seed
from tests.test_valuation_writer import _make_conn

_FIXTURES = Path(__file__).parent / "fixtures" / "companyfacts"
_AS_OF_DATE = "2026-09-29"


def _raw(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _sti_rows(raw: dict, cik: str) -> list[tuple[int, str, float, str]]:
    rows = normalize_annual_facts_from_raw(raw, cik=cik, years_back=3, filed_as_of=_AS_OF_DATE)
    return sorted(
        (row["fiscal_year"], row["period_end"], row["value"], row["source_tags"])
        for row in rows
        if row["line_item"] == "short_term_investments"
    )


# ── 1. short-term investments ────────────────────────────────────────────────


def test_microsoft_short_term_investments_are_normalized_beside_its_cash():
    """Microsoft tags cash (20,935 at 2026-06-30), ShortTermInvestments (55,908)
    and the combined line (76,843 = the two). The part is read, once."""
    rows = _sti_rows(_raw("MSFT_0000789019_cashlike.json"), "0000789019")
    assert rows == [
        (2025, "2025-06-30", 64323.0, "ShortTermInvestments"),
        (2026, "2026-06-30", 55908.0, "ShortTermInvestments"),
    ]


def test_combined_line_less_cash_when_no_part_is_tagged():
    """Coca-Cola tags only cash (10,270 at 2025-12-31) and the combined line
    (13,872); the difference, 3,602, is its short-term investments."""
    rows = _sti_rows(_raw("KO_0000021344_cashlike.json"), "0000021344")
    derivation = "CashCashEquivalentsAndShortTermInvestments - CashAndCashEquivalentsAtCarryingValue"
    assert rows == [
        (2024, "2024-12-31", 2020.0, derivation),
        (2025, "2025-12-31", 3602.0, derivation),
    ]


def test_combined_line_as_the_cash_figure_is_not_counted_twice():
    """Coca-Cola's payload with the cash-only tag removed: the combined line
    becomes the cash figure, so nothing is added to it, not even a separately
    tagged ShortTermInvestments at the same date."""
    raw = _raw("KO_0000021344_cashlike.json")
    gaap = raw["facts"]["us-gaap"]
    del gaap["CashAndCashEquivalentsAtCarryingValue"]
    gaap["ShortTermInvestments"] = {
        "units": {
            "USD": [
                {"end": "2025-12-31", "val": 3602000000, "accn": "x", "fy": 2025,
                 "fp": "FY", "form": "10-K", "filed": "2026-02-20"}
            ]
        }
    }
    assert _sti_rows(raw, "0000021344") == []


def test_short_term_investments_from_another_date_are_not_added():
    """A short-term investment reported only for a different balance-sheet date
    than the cash is not this balance sheet's: nothing is added."""
    raw = _raw("MSFT_0000789019_cashlike.json")
    gaap = raw["facts"]["us-gaap"]
    for tag in ("ShortTermInvestments", "CashCashEquivalentsAndShortTermInvestments"):
        gaap[tag]["units"]["USD"] = [
            row for row in gaap[tag]["units"]["USD"] if row["end"] == "2025-06-30"
        ]
    assert _sti_rows(raw, "0000789019") == [
        (2025, "2025-06-30", 64323.0, "ShortTermInvestments"),
    ]


@pytest.mark.parametrize(
    ("lines", "cash_tag", "expected"),
    [
        # Apple, FY2025 (2025-09-27): current marketable securities 18,763.
        ({"MarketableSecuritiesCurrent": 18763.0}, "CashAndCashEquivalentsAtCarryingValue",
         (18763.0, ["MarketableSecuritiesCurrent"])),
        # Available-for-sale and held-to-maturity current lines are disjoint: summed.
        ({"AvailableForSaleSecuritiesDebtSecuritiesCurrent": 300.0,
          "HeldToMaturitySecuritiesCurrent": 30.0}, "CashAndCashEquivalentsAtCarryingValue",
         (330.0, ["AvailableForSaleSecuritiesDebtSecuritiesCurrent",
                  "HeldToMaturitySecuritiesCurrent"])),
        # A total tag wins over the lines it may contain; never both.
        ({"ShortTermInvestments": 100.0, "MarketableSecuritiesCurrent": 60.0,
          "AvailableForSaleSecuritiesDebtSecuritiesCurrent": 60.0},
         "CashAndCashEquivalentsAtCarryingValue", (100.0, ["ShortTermInvestments"])),
        # The cash figure already is the combined line: nothing to add.
        ({"ShortTermInvestments": 100.0}, "CashCashEquivalentsAndShortTermInvestments", None),
        # A combined line no larger than cash adds nothing.
        ({"CashCashEquivalentsAndShortTermInvestments": 90.0},
         "CashAndCashEquivalentsAtCarryingValue", None),
        # A reported zero adds nothing.
        ({"ShortTermInvestments": 0.0}, "CashAndCashEquivalentsAtCarryingValue", None),
    ],
)
def test_short_term_investments_addition_rule(lines, cash_tag, expected):
    assert short_term_investments_addition(lines, cash_tag=cash_tag, cash_value=100.0) == expected


def test_asof_net_debt_counts_microsofts_short_term_investments(monkeypatch, tmp_path):
    """As-of path: debt 40,294 - cash 20,935 - short-term investments 55,908 =
    -36,549 (net cash), with the component and its tag in the provenance."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    path = tmp_path / "msft.json"
    path.write_text(json.dumps(_raw("MSFT_0000789019_cashlike.json")), encoding="utf-8")

    result = resolve_net_debt_proxy(
        "MSFT", _AS_OF_DATE, facts_row={"cache_path": str(path)}, cfg=cfg
    )

    assert result["status"] == "OK"
    assert result["cash_equivalents"]["value"] == 20935.0
    assert result["net_debt_proxy_lease_exclusive"] == -36549.0
    assert result["net_debt_proxy"] == -36549.0 + 21925.0
    assert result["net_debt_flags"] == ["SHORT_TERM_INVESTMENTS_INCLUDED", "LEASE_ADJUSTED"]
    sti = result["short_term_investments"]
    assert (sti["value"], sti["period_end"], sti["tags"], sti["derivation"]) == (
        55908.0,
        "2026-06-30",
        ["ShortTermInvestments"],
        "ShortTermInvestments",
    )
    assert sti["derived_from"] == [
        "companyfacts.us-gaap.ShortTermInvestments[end_date=2026-06-30,unit=USD,"
        "filed=2026-07-29,accn=0001193125-26-323660]"
    ]


def test_asof_short_term_investments_respect_the_filing_cutoff(monkeypatch, tmp_path):
    """Before the FY2026 10-K was filed (2026-07-29) the June 2025 balance sheet
    is the latest public one: 43,151 - 30,242 - 64,323 = -51,414."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    path = tmp_path / "msft.json"
    path.write_text(json.dumps(_raw("MSFT_0000789019_cashlike.json")), encoding="utf-8")

    result = resolve_net_debt_proxy("MSFT", "2026-07-28", facts_row={"cache_path": str(path)}, cfg=cfg)

    assert result["net_debt_proxy_lease_exclusive"] == -51414.0
    assert result["short_term_investments"]["period_end"] == "2025-06-30"


def test_valuation_net_debt_counts_short_term_investments(monkeypatch, tmp_path):
    """The valuation bridge: debt 50 - (cash 30 + short-term investments 25) = -5,
    flagged, with the amount and its tag recorded beside the flags."""
    from tests.test_basket_ford import _v1_net_debt
    from tests.test_valuation_writer import _seed_companyfacts

    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed_companyfacts(conn, ticker="CASHLIKE")
    conn.execute(
        "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_end, line_item, value, "
        "units, source_url, fetched_at, filed_date, accession, source_tags) VALUES('CASHLIKE', "
        "2024, '2024-12-31', 'short_term_investments', 25.0, 'USD_millions', "
        "'https://example.test', '2026-01-01T00:00:00+00:00', '2025-02-15', 'x', "
        "'ShortTermInvestments')"
    )
    conn.commit()

    value, flags = _v1_net_debt(conn, cfg, "CASHLIKE")
    assert value == -5.0
    assert flags == ["SHORT_TERM_INVESTMENTS_INCLUDED"]
    outputs = json.loads(
        conn.execute(
            "SELECT outputs_json FROM valuations WHERE ticker = 'CASHLIKE' "
            "AND method = 'scorecard' ORDER BY id DESC LIMIT 1"
        ).fetchone()["outputs_json"]
    )
    assert outputs["quality_context"]["net_debt_short_term_investments"] == {
        "value": 25.0,
        "fiscal_year": 2024,
        "line_item": "short_term_investments",
        "source_tags": "ShortTermInvestments",
    }


# ── 2. REITs ──────────────────────────────────────────────────────────────────


def _method_outputs(conn, ticker: str, method: str) -> dict:
    row = conn.execute(
        "SELECT outputs_json FROM valuations WHERE ticker = ? AND method = ? "
        "ORDER BY id DESC LIMIT 1",
        (ticker, method),
    ).fetchone()
    return json.loads(row["outputs_json"])


@pytest.mark.parametrize(("sic", "not_applicable"), [(6798, True), (6500, False)])
def test_realty_income_is_a_reit_so_epv_and_ev_ebit_are_not_applicable(
    monkeypatch, tmp_path, sic, not_applicable
):
    """Realty Income (SIC 6798) on its trimmed real balance sheet: earnings power
    and EV/EBIT are NOT_APPLICABLE; the DCF and Graham rows are written as
    before. The same issuer filed under 6500 (real estate, not a REIT) keeps
    both methods."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed(conn, cfg, ticker="O", cik="0000726728")
    conn.execute(
        "INSERT INTO sec_registrants(cik, primary_ticker, all_tickers, exchange_scope, sic, "
        "operating_status, first_seen_at, last_seen_at) VALUES('0000726728', 'O', '[\"O\"]', "
        "'US', ?, 'ACTIVE', '2026-09-29', '2026-09-29')",
        (sic,),
    )
    conn.commit()

    _scorecard(conn, cfg, ticker="O", cik="0000726728")

    epv, ev_ebit = _method_outputs(conn, "O", "epv"), _method_outputs(conn, "O", "ev_ebit")
    expected = {
        "status": "NOT_APPLICABLE",
        "reason_code": "REIT_DEPRECIATION_DISTORTS_EARNINGS",
        "value_per_share": None,
        "flags": ["REIT_DEPRECIATION_DISTORTS_EARNINGS"],
    }
    if not_applicable:
        assert {k: epv[k] for k in expected} == expected
        assert {k: ev_ebit[k] for k in expected} == expected
    else:
        assert epv["status"] != "NOT_APPLICABLE"
        assert ev_ebit["status"] != "NOT_APPLICABLE"
    assert _method_outputs(conn, "O", "dcf")["status"] == "OK"
    assert _method_outputs(conn, "O", "graham")["status"] == "OK"


# ── 3. stale debt in the ratios and the discount rate ────────────────────────


def _ford_facts(debt_year: int, debt: float) -> dict[str, list[tuple[int, float]]]:
    """Ford's real annual figures ($M): its last total debt in companyfacts is
    2020 (471, a fragment) while its balance sheet runs to 2025."""
    return {
        "total_debt": [(debt_year, debt)],
        "cash": [(2025, 23356.0), (2024, 22935.0), (2020, 25243.0)],
        "equity": [(2025, 35952.0), (2024, 44835.0), (2020, 30690.0)],
        "operating_income": [(2025, -9169.0), (2024, 5219.0), (2020, -4408.0)],
        "depreciation_amortization": [(2025, 15974.0), (2024, 7567.0), (2020, 8774.0)],
        "total_assets": [(2025, 289160.0), (2024, 285196.0), (2020, 267261.0)],
    }


def test_capital_structure_refuses_fords_2020_debt():
    """Before: debt/equity 471 / 30,690 = 0.015 and cash covering debt 54 times,
    from 2020, beside a 2025 balance sheet."""
    from app.valuation.valuation_writer import _capital_structure_health

    result = _capital_structure_health(_ford_facts(2020, 471.0))
    assert (
        result["de_ratio"],
        result["cash_coverage"],
        result["net_debt_to_ebitda"],
        result["flags"],
    ) == (None, None, None, ["DEBT_STALE_YEAR"])


def test_capital_structure_accepts_debt_one_year_behind():
    from app.valuation.valuation_writer import _capital_structure_health

    result = _capital_structure_health(_ford_facts(2024, 8967.0))
    assert result["de_ratio"] == 8967.0 / 44835.0
    assert result["cash_coverage"] == 22935.0 / 8967.0
    assert result["net_debt_to_ebitda"] == (8967.0 - 22935.0) / (5219.0 + 7567.0)
    assert result["flags"] == []


def test_discount_rate_leverage_rule_refuses_fords_2020_debt():
    """The leverage rule read 2020's -5.67x net debt/EBITDA; now it has no ratio."""
    from app.valuation.valuation_writer import (
        _compute_quality_wacc,
        _latest_net_debt_and_ebitda,
    )

    assert _latest_net_debt_and_ebitda(_ford_facts(2020, 471.0)) is None
    assert _latest_net_debt_and_ebitda(_ford_facts(2024, 8967.0)) == (
        8967.0 - 22935.0,
        5219.0 + 7567.0,
    )
    wacc = _compute_quality_wacc(_ford_facts(2020, 471.0))
    leverage = next(r for r in wacc["rule_evaluations"] if r["code"] == "LEVERAGE_RISK")
    assert (leverage["metric_value"], leverage["fired"]) == (None, False)
