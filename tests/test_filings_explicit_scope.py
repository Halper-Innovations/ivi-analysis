"""Explicit-ticker filing ingest must bypass the active-universe snapshot.

Regression for the silent no-op that held census-discovered names: with an
active_universe state set, `ingest_with_policy(tickers=[T])` selected
companies through the universe snapshot and dropped any name outside it —
the filing fetch "ran" and did nothing.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from app.ingest.filings import _companies_for_active_universe, _upsert_filing
from app.ingest.sec_client import FilingStub


@pytest.fixture()
def filings_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.config import get_config

    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    cfg = get_config()
    from app.db import init_db

    init_db(cfg)
    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO companies(ticker, cik, name, universe_id, created_at) "
        "VALUES ('INUV', '0000000001', 'In Universe Co', 'u_test', 'x')"
    )
    conn.execute(
        "INSERT INTO companies(ticker, cik, name, universe_id, created_at) "
        "VALUES ('OUTU', '0000000002', 'Outside Universe Co', NULL, 'x')"
    )
    conn.execute(
        "INSERT INTO state(key, value_json, updated_at) "
        "VALUES ('active_universe', '{\"universe_id\": \"u_test\"}', 'x')"
    )
    conn.commit()
    yield conn, cfg
    conn.close()
    get_config.cache_clear()


def test_explicit_tickers_bypass_active_universe(filings_env) -> None:
    conn, _ = filings_env
    rows = _companies_for_active_universe(conn, tickers=["OUTU"])
    assert [(r["ticker"], r["cik"]) for r in rows] == [("OUTU", "0000000002")]


def test_universe_scope_still_applies_without_explicit_tickers(filings_env) -> None:
    conn, _ = filings_env
    rows = _companies_for_active_universe(conn, tickers=None)
    assert [r["ticker"] for r in rows] == ["INUV"]


def test_alias_ingest_does_not_relabel_existing_canonical_filing(filings_env) -> None:
    conn, _ = filings_env
    filing = FilingStub(
        cik="1",
        accession="0000000001-25-000001",
        accession_nodash="000000000125000001",
        form_type="10-K",
        filing_date=date(2025, 2, 15),
        period_end="2024-12-31",
        primary_document="annual.htm",
        primary_doc_url="https://www.sec.gov/Archives/annual.htm",
        filing_index_url="https://www.sec.gov/Archives/index.json",
    )

    filing_id = _upsert_filing(conn, "INUV", filing, "2025-03-01")
    same_id = _upsert_filing(conn, "ALIAS", filing, "2025-03-02")
    row = conn.execute(
        "SELECT ticker, ingested_as_of FROM filings WHERE id = ?", (filing_id,)
    ).fetchone()

    assert same_id == filing_id
    assert (row["ticker"], row["ingested_as_of"]) == ("INUV", "2025-03-02")
