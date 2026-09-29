"""Every "is this a financial issuer?" call site uses the SIC code first.

The substring rule reads XBRL tag NAMES and calls NVIDIA, Tesla and Costco financial
(71 of the top 200 liquid US names). Only the net-debt bridge used the SEC SIC code
(tests/test_net_debt_resolver.py); fundamentals, discovery, the sector packets, the
research plan, the gaps report, the synthesis agent and the tech-category classifier
kept the substring rule. They all resolve through
``app.util.issuer_classification.resolve_issuer_classification`` now: SIC when one is on
file, the substring rule only when none is (or ``VOE_ISSUER_CLASSIFICATION_BY_SIC=false``),
and the returned source string says which rule answered.
"""

from __future__ import annotations

import json

import pytest

import app.autonomous.sector_financial_packets as sfp
from app.db import get_db, init_db, utc_now_iso
from app.util.issuer_classification import (
    lookup_registrant_sic,
    resolve_issuer_classification,
)

# Tag / line-item names that trip the substring rule on an operating company.
BANK_SHAPED = {"deposits": 5.0, "loans": 7.0}
BANK_SHAPED_ITEMS = ["deposits", "loans"]


def _init_cfg(monkeypatch, tmp_path, *, by_sic: str | None = None):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    if by_sic is None:
        monkeypatch.delenv("VOE_ISSUER_CLASSIFICATION_BY_SIC", raising=False)
    else:
        monkeypatch.setenv("VOE_ISSUER_CLASSIFICATION_BY_SIC", by_sic)
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _seed_registrant(cfg, *, cik: str, sic: int | None, ticker: str, aliases=()) -> None:
    with get_db(cfg) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sec_registrants(cik, primary_ticker, all_tickers, name, sic,"
            " exchange, exchange_scope, operating_status, in_scope, first_seen_at, last_seen_at)"
            " VALUES(?, ?, ?, ?, ?, 'NASDAQ', 'IN_SCOPE', 'OPERATING', 1, ?, ?)",
            (cik, ticker, json.dumps([ticker, *aliases]), ticker, sic, utc_now_iso(), utc_now_iso()),
        )


@pytest.fixture(autouse=True)
def _clear_config_cache():
    from app.config import get_config

    yield
    get_config.cache_clear()


# ── the shared resolver ──────────────────────────────────────────────────────


def test_sic_decides_when_on_file_even_against_bank_shaped_line_items(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _seed_registrant(cfg, cik="0001045810", sic=3674, ticker="NVDA")
    assert resolve_issuer_classification(
        ticker="NVDA", line_items=BANK_SHAPED_ITEMS, cfg=cfg
    ) == ("operating", "sic")
    # By CIK too (unpadded and padded forms both find the row).
    assert resolve_issuer_classification(cik=1045810, line_items=BANK_SHAPED_ITEMS, cfg=cfg) == (
        "operating",
        "sic",
    )
    assert resolve_issuer_classification(
        cik="0001045810", line_items=BANK_SHAPED_ITEMS, cfg=cfg
    ) == ("operating", "sic")


def test_a_bank_by_sic_is_financial_without_any_bank_line_items(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _seed_registrant(cfg, cik="0000019617", sic=6021, ticker="JPM")
    assert resolve_issuer_classification(ticker="JPM", line_items=["revenue"], cfg=cfg) == (
        "financial",
        "sic",
    )


def test_ticker_lookup_finds_an_alias_ticker(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _seed_registrant(cfg, cik="0001652044", sic=7370, ticker="GOOGL", aliases=["GOOG"])
    assert lookup_registrant_sic(ticker="goog", cfg=cfg) == ("7370", "OK")


def test_opting_out_restores_the_substring_rule(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path, by_sic="false")
    _seed_registrant(cfg, cik="0001045810", sic=3674, ticker="NVDA")
    assert resolve_issuer_classification(
        ticker="NVDA", line_items=BANK_SHAPED_ITEMS, cfg=cfg
    ) == ("financial", "substring")


@pytest.mark.parametrize(
    ("registrant", "kwargs", "reason"),
    [
        (None, {"ticker": "NVDA"}, "NO_REGISTRANT_ROW"),
        (("0001045810", None, "NVDA"), {"ticker": "NVDA"}, "NO_SIC_ON_FILE"),
        (None, {}, "NO_IDENTIFIER"),
    ],
)
def test_no_usable_sic_falls_back_to_the_substring_rule_and_says_why(
    monkeypatch, tmp_path, registrant, kwargs, reason
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    if registrant:
        _seed_registrant(cfg, cik=registrant[0], sic=registrant[1], ticker=registrant[2])
    assert resolve_issuer_classification(line_items=BANK_SHAPED_ITEMS, cfg=cfg, **kwargs) == (
        "financial",
        f"substring:{reason}",
    )


def test_a_missing_registrants_table_is_a_fallback_not_a_crash(monkeypatch, tmp_path):
    import sqlite3

    cfg = _init_cfg(monkeypatch, tmp_path)
    bare = sqlite3.connect(":memory:")
    assert resolve_issuer_classification(
        ticker="NVDA", line_items=BANK_SHAPED_ITEMS, conn=bare, cfg=cfg
    ) == ("financial", "substring:LOOKUP_FAILED")


# ── the call sites ───────────────────────────────────────────────────────────


def test_fundamentals_pipeline_stores_the_sic_class_and_its_source(monkeypatch, tmp_path):
    from app.fundamentals.metrics import compute_fundamentals_for_ticker

    cfg = _init_cfg(monkeypatch, tmp_path)
    _seed_registrant(cfg, cik="0001045810", sic=3674, ticker="NVDA")
    rows = {
        "revenue": 100.0,
        "cfo": 25.0,
        "capex": 5.0,
        "cash": 10.0,
        "total_debt": 30.0,
        **BANK_SHAPED,
    }
    with get_db(cfg) as conn:
        conn.executemany(
            "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, period_end,"
            " filed_date, line_item, value, units, source_url, fetched_at, form, accession)"
            " VALUES('NVDA', 2025, 'FY', '2025-12-31', '2026-02-15', ?, ?, 'USD_millions',"
            " 'https://data.sec.gov', '2026-03-20T00:00:00+00:00', '10-K', '0000000001-26-000001')",
            list(rows.items()),
        )
    assert compute_fundamentals_for_ticker("NVDA", as_of_date="2026-03-19") is True
    with get_db(cfg) as conn:
        row = conn.execute(
            "SELECT metrics_json, quality_flags_json FROM fundamentals WHERE ticker='NVDA'"
        ).fetchone()
    metrics = json.loads(row["metrics_json"])
    flags = json.loads(row["quality_flags_json"])
    assert metrics["issuer_classification"] == "operating"
    assert metrics["fcf"] == 20.0
    assert metrics["fcf_applicability"] == "standard"
    assert flags["issuer_classification_source"] == "sic"


def test_fundamentals_without_a_sic_keep_the_substring_answer_and_flag_it(monkeypatch, tmp_path):
    from app.fundamentals.metrics import compute_fundamentals_for_ticker

    cfg = _init_cfg(monkeypatch, tmp_path)
    with get_db(cfg) as conn:
        conn.executemany(
            "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, period_end,"
            " filed_date, line_item, value, units, source_url, fetched_at, form, accession)"
            " VALUES('ZZZ', 2025, 'FY', '2025-12-31', '2026-02-15', ?, ?, 'USD_millions',"
            " 'https://data.sec.gov', '2026-03-20T00:00:00+00:00', '10-K', '0000000002-26-000001')",
            [("revenue", 100.0), ("cfo", 25.0), ("capex", 5.0), *BANK_SHAPED.items()],
        )
    assert compute_fundamentals_for_ticker("ZZZ", as_of_date="2026-03-19") is True
    with get_db(cfg) as conn:
        row = conn.execute(
            "SELECT metrics_json, quality_flags_json FROM fundamentals WHERE ticker='ZZZ'"
        ).fetchone()
    assert json.loads(row["metrics_json"])["issuer_classification"] == "financial"
    assert (
        json.loads(row["quality_flags_json"])["issuer_classification_source"]
        == "substring:NO_REGISTRANT_ROW"
    )


def _packet(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "financials": [{"line_item": item, "citation": {"snippet": ""}} for item in BANK_SHAPED_ITEMS],
    }


@pytest.mark.parametrize(
    "getter",
    [
        pytest.param(
            lambda p: __import__("app.research.engine", fromlist=["x"])._issuer_classification_from_packet(p),
            id="research_engine",
        ),
        pytest.param(
            lambda p: __import__("app.report.gaps", fromlist=["x"])._issuer_classification_from_packet(p),
            id="report_gaps",
        ),
        pytest.param(
            lambda p: __import__("app.llm.synthesis_agent", fromlist=["x"])._issuer_classification_from_packet(p),
            id="synthesis_agent",
        ),
    ],
)
def test_packet_fallbacks_use_the_sic_code(monkeypatch, tmp_path, getter):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _seed_registrant(cfg, cik="0001045810", sic=3674, ticker="NVDA")
    _seed_registrant(cfg, cik="0000019617", sic=6021, ticker="JPM")
    assert getter(_packet("NVDA")) == "operating"
    # Same line items, no registrant row: the substring rule still answers.
    assert getter(_packet("NOROW")) == "financial"
    # A packet that already carries a classification is left alone.
    assert getter({"ticker": "NVDA", "fundamentals": {"issuer_classification": "financial"}}) == "financial"


def test_discovery_metrics_use_the_sic_code(monkeypatch, tmp_path):
    from app.discovery.metrics import compute_discovery_metrics

    cfg = _init_cfg(monkeypatch, tmp_path)
    _seed_registrant(cfg, cik="0001045810", sic=3674, ticker="NVDA")
    with get_db(cfg) as conn:
        now = utc_now_iso()
        conn.execute(
            "INSERT INTO filings(cik, ticker, accession, form_type, filing_date, period_end,"
            " primary_doc_url, local_path, hash, ingested_as_of, status, created_at, updated_at)"
            " VALUES('0001045810', 'NVDA', '0001045810-26-000001', '10-K', '2026-02-13',"
            " '2025-12-31', 'https://www.sec.gov/example', NULL, NULL, '2026-02-13', 'parsed', ?, ?)",
            (now, now),
        )
        conn.executemany(
            "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_end, line_item, value, units,"
            " source_url, fetched_at) VALUES('NVDA', 2025, '2025-12-31', ?, ?, 'USD_millions',"
            " 'https://www.sec.gov/companyfacts', ?)",
            [
                (item, value, now)
                for item, value in [
                    ("revenue", 100.0),
                    ("gross_profit", 60.0),
                    ("operating_income", 20.0),
                    ("cfo", 40.0),
                    ("capex", 10.0),
                    ("cash", 200.0),
                    ("total_debt", 150.0),
                    ("shares_outstanding", 50.0),
                    *BANK_SHAPED.items(),
                ]
            ],
        )
        conn.execute(
            "UPDATE companyfacts_facts SET filed_date='2026-02-13', form='10-K',"
            " accession='0001045810-26-000001' WHERE ticker='NVDA'"
        )
        result = compute_discovery_metrics(
            conn, ticker="NVDA", selected_accessions=["0001045810-26-000001"]
        )
    assert result is not None
    assert result.metrics["issuer_classification"] == "operating"
    assert result.metrics["fcf_applicability"] == "standard"
    assert result.metrics["fcf"] == 30.0


def test_sector_packet_rows_use_the_sic_code(monkeypatch, tmp_path):
    """Gross margin and ROIC were suppressed / put on the financial equity basis for any
    operating company whose facts carried bank-shaped line items."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    _seed_registrant(cfg, cik="0001045810", sic=3674, ticker="NVDA")
    by_year = {
        2024: {
            "revenue": 1000.0,
            "gross_profit": 600.0,
            "operating_income": 100.0,
            "income_tax_expense": 20.0,
            "pretax_income": 100.0,
            "total_debt": 300.0,
            "equity": 200.0,
            "cash": 100.0,
            **BANK_SHAPED,
        }
    }
    monkeypatch.setattr(sfp, "_annual_fact_rows", lambda ticker, **_kwargs: by_year)

    gross = sfp._gross_margin_metrics("NVDA", as_of_date=None)
    assert gross["gross_margin"] == pytest.approx(0.6)
    roic = sfp._returns_on_capital_metrics("NVDA", as_of_date=None)
    assert roic["roic"] == pytest.approx(80.0 / 400.0)
    assert roic["invested_capital_basis"] != "equity_only_financial"

    # No registrant row: the per-year substring rule answers as before.
    assert sfp._gross_margin_metrics("NOROW", as_of_date=None)["gross_margin"] is None


def test_tech_category_classifier_uses_the_sic_code(monkeypatch, tmp_path):
    from app.valuation.tech_category import _infer_companyfacts_issuer_classification

    cfg = _init_cfg(monkeypatch, tmp_path)
    _seed_registrant(cfg, cik="0001045810", sic=3674, ticker="NVDA")
    bank_shaped_tags = {
        "cik": 1045810,
        "entityName": "NVIDIA CORP",
        "facts": {"us-gaap": {"Deposits": {}, "Loans": {}, "InvestmentSecurities": {}}},
    }
    assert (
        _infer_companyfacts_issuer_classification(bank_shaped_tags, ticker="NVDA", cfg=cfg)
        == "operating"
    )
    # Unknown registrant: the tag-name rule answers, unchanged.
    assert (
        _infer_companyfacts_issuer_classification(
            {**bank_shaped_tags, "cik": 999}, ticker="NOROW", cfg=cfg
        )
        == "financial"
    )
