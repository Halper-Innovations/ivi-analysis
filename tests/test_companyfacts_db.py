# tests/test_companyfacts_db.py
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


def test_companyfacts_facts_table_exists(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO companyfacts_facts
               (ticker, fiscal_year, period_end, line_item, value, units, source_url, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("AAPL", 2024, "2024-09-28", "revenue", 391035.0, "USD_millions",
             "https://data.sec.gov/api/xbrl/companyfacts/0000320193.json", now),
        )
        row = conn.execute(
            "SELECT value FROM companyfacts_facts WHERE ticker=? AND fiscal_year=? AND line_item=?",
            ("AAPL", 2024, "revenue"),
        ).fetchone()
    assert row["value"] == 391035.0


def test_companyfacts_facts_unique_constraint(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO companyfacts_facts
               (ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at)
               VALUES ('AAPL', 2024, 'FY', '2024-09-28', 'revenue', 391035.0, 'USD_millions',
                       'https://data.sec.gov', ?)""", (now,),
        )
        conn.execute(
            """INSERT INTO companyfacts_facts
               (ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at)
               VALUES ('AAPL', 2024, 'FY', '2024-09-28', 'revenue', 999.0, 'USD_millions',
                       'https://data.sec.gov', ?)
               ON CONFLICT(ticker, fiscal_year, period_type, line_item) DO UPDATE SET value=excluded.value""",
            (now,),
        )
        row = conn.execute(
            "SELECT value FROM companyfacts_facts WHERE ticker='AAPL' AND fiscal_year=2024 AND line_item='revenue'"
        ).fetchone()
    assert row["value"] == 999.0


def test_companyfacts_facts_period_type_column_exists(monkeypatch, tmp_path):
    """period_type column must exist with default 'FY' for new and migrated databases."""
    _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        # Insert without specifying period_type — should default to 'FY'
        conn.execute(
            """INSERT INTO companyfacts_facts
               (ticker, fiscal_year, period_end, line_item, value, units, source_url, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("MSFT", 2024, "2024-06-30", "revenue", 245122.0, "USD_millions",
             "https://data.sec.gov", now),
        )
        row = conn.execute(
            "SELECT period_type FROM companyfacts_facts WHERE ticker='MSFT' AND fiscal_year=2024"
        ).fetchone()
    assert row["period_type"] == "FY"


def test_companyfacts_facts_quarterly_rows_distinct_from_annual(monkeypatch, tmp_path):
    """Quarterly (Q1) and annual (FY) rows for same ticker/year/line_item coexist."""
    _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO companyfacts_facts
               (ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at)
               VALUES ('AAPL', 2024, 'FY', '2024-09-28', 'revenue', 391035.0, 'USD_millions',
                       'https://data.sec.gov', ?)""", (now,),
        )
        conn.execute(
            """INSERT INTO companyfacts_facts
               (ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at)
               VALUES ('AAPL', 2024, 'Q1', '2023-12-30', 'revenue', 119575.0, 'USD_millions',
                       'https://data.sec.gov', ?)""", (now,),
        )
        rows = conn.execute(
            "SELECT period_type, value FROM companyfacts_facts WHERE ticker='AAPL' AND fiscal_year=2024 AND line_item='revenue' ORDER BY period_type"
        ).fetchall()
    assert len(rows) == 2
    by_period = {r["period_type"]: r["value"] for r in rows}
    assert by_period["FY"] == 391035.0
    assert by_period["Q1"] == 119575.0
