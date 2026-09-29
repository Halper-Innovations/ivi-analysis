"""Net debt for five large issuers that report debt and cash plainly.

A fresh-install basket run (2026-09-29) left DCF and earnings power blank with
NET_DEBT_UNKNOWN for JNJ, PG, COST, Realty Income (O) and LeMaitre (LMAT). The
debt and cash were never the problem: each time a senior claim (preferred stock
or a minority interest) was not in the normalized annual facts for the
balance-sheet year, and the evidenced-zero rule refused to call it zero.

Each fixture is the issuer's real SEC companyfacts payload, trimmed to the
balance-sheet concepts (annual reports, plus quarterly reports for the claim
concepts) from 2021 on. The expected net debt is debt - cash + preferred +
minority interest off the latest annual balance sheet, checked against the
10-K's own balance sheet.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.config import get_config
from app.db import init_db
from app.ingest.companyfacts import normalize_annual_facts_from_raw
from app.valuation.valuation_writer import _load_valuation_facts, ensure_valuation
from tests.test_valuation_writer import _make_conn

_AS_OF_DATE = "2026-09-29"
_FIXTURES = Path(__file__).parent / "fixtures" / "companyfacts"


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _seed(conn, cfg, *, ticker: str, cik: str) -> None:
    """Normalize the trimmed payload into the table and materialize it in the cache."""
    raw = json.loads((_FIXTURES / f"{ticker}_{cik}_net_debt.json").read_text(encoding="utf-8"))
    rows = normalize_annual_facts_from_raw(raw, cik=cik, years_back=6, filed_as_of=_AS_OF_DATE)
    for row in rows:
        conn.execute(
            "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, period_end, "
            "line_item, value, units, source_url, fetched_at, filed_date, form, accession) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, '2026-09-29T00:00:00+00:00', ?, ?, ?)",
            (
                ticker,
                row["fiscal_year"],
                row["period_type"],
                row["period_end"],
                row["line_item"],
                row["value"],
                row["units"],
                row["source_url"],
                row["filed_date"],
                row["form"],
                row["accession"],
            ),
        )
    # Flow rows the writer needs to run at all; they do not touch net debt.
    latest = max(int(row["fiscal_year"]) for row in rows if row["line_item"] == "cash")
    period_end = next(
        row["period_end"]
        for row in rows
        if row["line_item"] == "cash" and int(row["fiscal_year"]) == latest
    )
    source_url = rows[0]["source_url"]
    for offset in range(5):
        year = latest - offset
        for line_item, value in (
            ("cfo", 1200.0),
            ("capex", 150.0),
            ("operating_income", 1000.0),
            ("net_income", 700.0),
            ("revenue", 5000.0 - 100.0 * offset),
            ("shares_outstanding", 100.0),
        ):
            conn.execute(
                "INSERT OR IGNORE INTO companyfacts_facts(ticker, fiscal_year, period_type, "
                "period_end, line_item, value, units, source_url, fetched_at, filed_date, "
                "form, accession) VALUES(?, ?, 'FY', ?, ?, ?, 'USD_millions', ?, "
                "'2026-09-29T00:00:00+00:00', ?, '10-K', ?)",
                (
                    ticker,
                    year,
                    f"{year}{period_end[4:]}",
                    line_item,
                    value,
                    source_url,
                    f"{year + 1}-02-01" if period_end[5:7] == "12" else f"{year}-{period_end[5:7]}-28",
                    f"{ticker}-{year}",
                ),
            )
    conn.commit()
    cache_path = cfg.cache_dir / "companyfacts" / f"{cik}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "cik": cik,
                "retrieved_at": "2026-09-29T12:00:00+00:00",
                "source_url": source_url,
                "http_status": 200,
                "companyfacts": raw,
            }
        ),
        encoding="utf-8",
    )


def _scorecard(conn, cfg, *, ticker: str, cik: str) -> tuple[dict, dict]:
    with patch("app.valuation.valuation_writer.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda _self: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        ensure_valuation(
            ticker,
            _AS_OF_DATE,
            provider=None,
            price_override=50.0,
            force_refresh=True,
            cfg=cfg,
            issuer_cik=cik,
            issuer_aliases=(ticker,),
            require_filed_asof=True,
            raise_on_error=True,
        )
    row = conn.execute(
        "SELECT inputs_json, outputs_json FROM valuations "
        "WHERE ticker = ? AND method = 'scorecard' ORDER BY id DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    return json.loads(row["inputs_json"]), json.loads(row["outputs_json"])


@pytest.mark.parametrize(
    ("ticker", "cik", "net_debt", "derivations"),
    [
        # Johnson & Johnson, FY2025 (balance sheet 2025-12-28): debt 47,938,
        # cash 19,709, no preferred, no minority interest. The only minority
        # interest in the window is Kenvue's 1,260 at 2023-07-02, a 10-Q taken
        # before the August 2023 exchange offer; the 2023 and 2024 annual
        # balance sheets present equity with none. It no longer blocks.
        (
            "JNJ",
            "0000200406",
            28229.0,
            {"noncontrolling_interest": "EVIDENCED_ZERO_SENIOR_CLAIM_ABSENCE"},
        ),
        # Costco, FY2025 (2025-08-31): debt 5,788, cash 14,161 -> net cash.
        # The Taiwan minority interest (5 at 2023-05-07) was bought out; the
        # FY2023 10-K reports it as zero at 2023-09-03 and it is not tagged
        # since.
        (
            "COST",
            "0000909832",
            -8373.0,
            {"noncontrolling_interest": "EVIDENCED_ZERO_SENIOR_CLAIM_ABSENCE"},
        ),
        # Realty Income, FY2025 (2025-12-31): debt 25,031.947, cash 434.842,
        # minority interest 685.273. The 167.394 of temporary-equity preferred
        # it carried at 2024-03-31 and 2024-06-30 was gone by 2024-09-30 and
        # is reported as zero in the 2024 10-K; the 2025 10-K's zero
        # comparatives for earlier dates no longer block the zero either.
        (
            "O",
            "0000726728",
            25282.378,
            {"preferred_equity": "EVIDENCED_ZERO_SENIOR_CLAIM_ABSENCE"},
        ),
        # LeMaitre, FY2025 (2025-12-31): convertible notes 168.645, cash
        # 28.244. The 2025 10-K does not tag preferred stock; the 2026 10-Qs
        # report it as zero AT 2025-12-31, which is direct evidence. (This
        # fixture is trimmed of LeMaitre's 330.9 of current marketable
        # securities; with them, which count as cash since 2026-09-29, it is
        # net cash of 190.5 — see tests/test_basket_cashlike.py.)
        (
            "LMAT",
            "0001158895",
            140.401,
            {
                "preferred_equity": "EVIDENCED_ZERO_SENIOR_CLAIM_REPORTED",
                "noncontrolling_interest": "EVIDENCED_ZERO_SENIOR_CLAIM_ABSENCE",
            },
        ),
    ],
)
def test_basket_net_debt_resolves_to_the_balance_sheet_figure(
    monkeypatch, tmp_path, ticker, cik, net_debt, derivations
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed(conn, cfg, ticker=ticker, cik=cik)

    _facts, proofs = _load_valuation_facts(
        ticker, conn, as_of_date=_AS_OF_DATE, issuer_cik=cik, issuer_aliases=(ticker,), cfg=cfg
    )
    assert {proof["line_item"]: proof["derivation"] for proof in proofs} == derivations

    inputs, outputs = _scorecard(conn, cfg, ticker=ticker, cik=cik)
    assert inputs["net_debt"] == pytest.approx(net_debt, abs=1e-6)
    assert "NET_DEBT_UNKNOWN" not in outputs["quality_context"]["net_debt_flags"]


def test_pg_preferred_missing_at_year_end_still_refuses(monkeypatch, tmp_path):
    """Procter & Gamble, FY2026 (2026-06-30): debt 34,138, cash 8,700, minority
    interest 230 — and ESOP convertible preferred stock the 10-K tags only
    under PG's own extension concept, which companyfacts omits. The latest
    standard-tagged value is 759 at 2026-03-31 (a 10-Q), and no annual balance
    sheet after it shows the claim gone, so the year-end figure is not in the
    data. Refusing is right: a quarter-old figure is not the balance-sheet
    value. It resolves once a later filing reports the claim at 2026-06-30 and
    it is normalized."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed(conn, cfg, ticker="PG", cik="0000080424")

    facts, proofs = _load_valuation_facts(
        "PG", conn, as_of_date=_AS_OF_DATE, issuer_cik="0000080424", issuer_aliases=("PG",), cfg=cfg
    )
    assert proofs == []
    assert "preferred_equity" not in facts
    inputs, outputs = _scorecard(conn, cfg, ticker="PG", cik="0000080424")
    assert inputs["net_debt"] == "UNKNOWN"
    assert outputs["quality_context"]["net_debt_flags"] == [
        "SENIOR_CLAIMS_UNKNOWN",
        "NET_DEBT_UNKNOWN",
        "LEASE_EXCLUDED_POSTLEASE_FLOWS",
    ]
