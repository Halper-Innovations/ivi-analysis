"""Verify that valuation and dossier queries exclude quarterly data by default."""
from __future__ import annotations

from app.db import get_db, init_db, utc_now_iso


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config as _gc
    _gc.cache_clear()
    cfg = _gc()
    init_db(cfg)
    return cfg


def _seed_mixed_data(ticker: str = "AAPL"):
    """Insert annual + quarterly rows for the same ticker."""
    now = utc_now_iso()
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    rows = [
        # Annual: full-year revenue $400B
        (ticker, 2024, "FY", "2024-09-28", "revenue", 400000.0, "USD_millions", source_url, now, "2024-11-01", "10-K", "0000320193-24-000123"),
        (ticker, 2024, "FY", "2024-09-28", "operating_income", 120000.0, "USD_millions", source_url, now, "2024-11-01", "10-K", "0000320193-24-000123"),
        (ticker, 2024, "FY", "2024-09-28", "total_assets", 350000.0, "USD_millions", source_url, now, "2024-11-01", "10-K", "0000320193-24-000123"),
        (ticker, 2024, "FY", "2024-09-28", "net_income", 95000.0, "USD_millions", source_url, now, "2024-11-01", "10-K", "0000320193-24-000123"),
        (ticker, 2024, "FY", "2024-09-28", "cfo", 110000.0, "USD_millions", source_url, now, "2024-11-01", "10-K", "0000320193-24-000123"),
        (ticker, 2024, "FY", "2024-09-28", "capex", 10000.0, "USD_millions", source_url, now, "2024-11-01", "10-K", "0000320193-24-000123"),
        (ticker, 2024, "FY", "2024-09-28", "equity", 60000.0, "USD_millions", source_url, now, "2024-11-01", "10-K", "0000320193-24-000123"),
        (ticker, 2024, "FY", "2024-09-28", "shares_outstanding", 15000.0, "shares_millions", source_url, now, "2024-11-01", "10-K", "0000320193-24-000123"),
        # Quarterly: Q1 revenue $95B — must NOT contaminate annual calculations
        (ticker, 2025, "Q1", "2024-12-28", "revenue", 95000.0, "USD_millions", source_url, now, "2025-01-31", "10-Q", "0000320193-25-000010"),
        (ticker, 2025, "Q1", "2024-12-28", "operating_income", 30000.0, "USD_millions", source_url, now, "2025-01-31", "10-Q", "0000320193-25-000010"),
        (ticker, 2025, "Q1", "2024-12-28", "total_assets", 360000.0, "USD_millions", source_url, now, "2025-01-31", "10-Q", "0000320193-25-000010"),
    ]
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES(?, '0000320193', 'Quarterly Isolation Test Co', ?)
            """,
            (ticker, now),
        )
        for r in rows:
            conn.execute(
                """INSERT INTO companyfacts_facts
                   (ticker, fiscal_year, period_type, period_end, line_item, value, units,
                    source_url, fetched_at, filed_date, form, accession)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                r,
            )


def test_valuation_writer_load_facts_excludes_quarterly(monkeypatch, tmp_path):
    """_load_facts should only return FY rows."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_mixed_data()
    from app.valuation.valuation_writer import _load_facts
    with get_db() as conn:
        facts = _load_facts("AAPL", conn)
    # Revenue should have exactly 1 entry (FY 2024), not 2
    rev = facts.get("revenue", [])
    assert len(rev) == 1
    assert rev[0][0] == 2024  # fiscal_year
    assert rev[0][1] == 400000.0  # full-year value


def test_annual_extractors_excludes_quarterly(monkeypatch, tmp_path):
    """_load_companyfacts_rows should only return FY rows."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_mixed_data()
    from app.dossier.collector import DossierFiling
    from app.dossier.extractors.annual_extractors import _load_companyfacts_rows
    rows = _load_companyfacts_rows(
        DossierFiling(
            ticker="AAPL",
            cik="0000320193",
            accession="0000320193-24-000123",
            form_type="10-K",
            filing_date="2024-11-01",
            period_end="2024-09-28",
            primary_doc_url="https://www.sec.gov/Archives/edgar/data/320193/annual.htm",
            local_path=None,
            filing_id=1,
        )
    )
    assert "revenue" in rows
    assert rows["revenue"]["value"] == 400000.0
    # Should NOT find Q1 2025 data when asking for FY data
    q1_rows = _load_companyfacts_rows(
        DossierFiling(
            ticker="AAPL",
            cik="0000320193",
            accession="0000320193-25-000999",
            form_type="10-K",
            filing_date="2025-11-01",
            period_end="2025-09-27",
            primary_doc_url="https://www.sec.gov/Archives/edgar/data/320193/future-annual.htm",
            local_path=None,
            filing_id=2,
        )
    )
    assert "revenue" not in q1_rows  # No FY row for 2025


def test_load_quarterly_facts(monkeypatch, tmp_path):
    """_load_quarterly_facts returns only quarterly rows, sorted by (fiscal_year, period_type) descending."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_mixed_data()
    from app.valuation.valuation_writer import _load_quarterly_facts
    with get_db() as conn:
        qfacts = _load_quarterly_facts("AAPL", conn)
    rev = qfacts.get("revenue", [])
    assert len(rev) == 1
    assert rev[0] == (2025, "Q1", 95000.0)


def test_latest_quarterly_value(monkeypatch, tmp_path):
    """_latest_quarterly returns the most recent quarterly value for a line item."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_mixed_data()
    # Add a Q2 row
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO companyfacts_facts
               (ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("AAPL", 2025, "Q2", "2025-03-29", "revenue", 97000.0, "USD_millions", "", now),
        )
    from app.valuation.valuation_writer import _load_quarterly_facts, _latest_quarterly
    with get_db() as conn:
        qfacts = _load_quarterly_facts("AAPL", conn)
    result = _latest_quarterly(qfacts, "revenue")
    assert result is not None
    fy, qtr, val = result
    assert fy == 2025
    assert qtr == "Q2"
    assert val == 97000.0
