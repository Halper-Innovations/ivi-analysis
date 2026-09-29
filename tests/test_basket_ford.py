"""Ford in the 2026-09-29 basket: a share count and a stale net debt.

Ford (CIK 0000037996) reported SHARES_MISSING and EQUITY_MISSING though its
companyfacts payload is 3.7 MB. Its outstanding shares are tagged only per class,
and its balance-sheet debt only per segment, and companyfacts keeps neither.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.ingest.companyfacts import normalize_annual_facts_from_raw
from app.valuation.valuation_writer import ensure_valuation
from tests.test_basket_net_debt import _AS_OF_DATE, _FIXTURES, _init_cfg
from tests.test_valuation_writer import _make_conn


def _v1_net_debt(conn, cfg, ticker: str) -> tuple[object, list[str]]:
    with patch("app.valuation.valuation_writer.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda _self: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        ensure_valuation(
            ticker,
            "2026-03-19",
            provider=None,
            price_override=50.0,
            force_refresh=True,
            cfg=cfg,
            raise_on_error=True,
        )
    row = conn.execute(
        "SELECT inputs_json, outputs_json FROM valuations "
        "WHERE ticker = ? AND method = 'scorecard' ORDER BY id DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    inputs, outputs = json.loads(row["inputs_json"]), json.loads(row["outputs_json"])
    return inputs["net_debt"], outputs["quality_context"]["net_debt_flags"]


@pytest.mark.parametrize(
    ("debt_year", "net_debt", "flags"),
    [
        # Ford's shape: the last total debt in companyfacts is 2020 (471, a
        # fragment) while the balance sheet runs to 2025. Netting the 2020 pair
        # gave Ford 24.7 billion of net CASH and a 56-dollar DCF.
        (2021, "UNKNOWN", ["NET_DEBT_STALE_YEAR", "NET_DEBT_UNKNOWN"]),
        # One fiscal year behind is still accepted, as before.
        (2023, 20.0, []),
    ],
)
def test_net_debt_from_a_stale_year_is_refused(monkeypatch, tmp_path, debt_year, net_debt, flags):
    from tests.test_valuation_writer import _seed_companyfacts

    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed_companyfacts(conn, ticker="STALEDEBT")
    conn.execute(
        "UPDATE companyfacts_facts SET fiscal_year = ?, period_end = ? "
        "WHERE ticker = 'STALEDEBT' AND line_item IN "
        "('total_debt', 'preferred_equity', 'noncontrolling_interest')",
        (debt_year, f"{debt_year}-12-31"),
    )
    conn.execute(
        "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_end, line_item, value, "
        "units, source_url, fetched_at, filed_date, accession) VALUES('STALEDEBT', ?, ?, "
        "'cash', 30.0, 'USD_millions', 'https://example.test', '2026-01-01T00:00:00+00:00', "
        "?, 'x')",
        (debt_year, f"{debt_year}-12-31", f"{debt_year + 1}-02-15"),
    )
    conn.commit()

    value, net_debt_flags = _v1_net_debt(conn, cfg, "STALEDEBT")
    assert value == net_debt
    assert net_debt_flags == flags


def test_ford_share_count_comes_from_the_diluted_weighted_average():
    """Ford reports its outstanding shares only per class (Common and Class B
    cover-page counts), and companyfacts drops dimensioned facts: its last
    undimensioned cover count is from 2011. With no outstanding-count fact in
    the window the fiscal-year count falls back to the year's diluted weighted
    average (every class), judged by the share guard — 3,979 million for 2025
    against about 3.98 billion outstanding. Before, the year had no share row
    and every per-share method refused (SHARES_MISSING, and Graham reported
    EQUITY_MISSING)."""
    raw = json.loads((_FIXTURES / "F_0000037996_shares.json").read_text(encoding="utf-8"))
    rows = normalize_annual_facts_from_raw(
        raw, cik="0000037996", years_back=10, filed_as_of=_AS_OF_DATE
    )
    shares = {
        row["fiscal_year"]: (row["value"], row["tag"])
        for row in rows
        if row["line_item"] == "shares_outstanding"
    }
    tag = "WeightedAverageNumberOfDilutedSharesOutstanding"
    assert shares[2025] == (3979.0, tag)
    assert shares[2024] == (4021.0, tag)
    assert shares[2023] == (4041.0, tag)
    assert min(shares) == 2016


def test_an_outstanding_count_in_one_year_leaves_the_fallback_on_for_the_others():
    """Migrated 2026-09-29. The fallback used to be all-or-nothing for the window: one
    cover count anywhere left every other year without a row (Ford FY2011-2019 had no share
    count because of one old cover fact). It is now decided per fiscal year: the year with
    an outstanding count keeps it, and only years no outstanding-count tag reported fall
    back to the guarded diluted weighted average."""
    raw = json.loads((_FIXTURES / "F_0000037996_shares.json").read_text(encoding="utf-8"))
    raw["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"].append(
        {
            "end": "2025-01-31",
            "val": 3_970_000_000,
            "accn": "0000037996-25-000009",
            "fy": 2024,
            "fp": "FY",
            "form": "10-K",
            "filed": "2025-02-06",
        }
    )
    rows = normalize_annual_facts_from_raw(
        raw, cik="0000037996", years_back=10, filed_as_of=_AS_OF_DATE
    )
    shares = {
        row["fiscal_year"]: (row["value"], row["tag"])
        for row in rows
        if row["line_item"] == "shares_outstanding"
    }
    tag = "WeightedAverageNumberOfDilutedSharesOutstanding"
    assert shares[2024] == (3970.0, "EntityCommonStockSharesOutstanding")
    assert shares[2025] == (3979.0, tag)
    assert shares[2023] == (4041.0, tag)
    assert min(shares) == 2016
