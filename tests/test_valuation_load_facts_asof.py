from __future__ import annotations

from app.config import get_config
from app.db import get_db, init_db


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def test_load_facts_excludes_future_fiscal_years(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('XCO', '0000000789', 'XCO Test Co', '2022-01-01T00:00:00Z')
            """
        )
        for fy, pend, val in [(2022, "2022-12-31", 50.0), (2024, "2024-12-31", 999.0)]:
            conn.execute(
                "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, period_end, "
                "line_item, value, units, source_url, fetched_at, filed_date, form, accession) "
                "VALUES('XCO', ?, 'FY', ?, 'revenue', ?, 'USD_millions', "
                "'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000789.json', "
                "'2026-01-01T00:00:00Z', ?, '10-K', ?)",
                (fy, pend, val, f"{fy + 1}-02-15", f"0000000789-{str(fy + 1)[-2:]}-000001"),
            )
        conn.commit()
        from app.valuation.valuation_writer import _load_facts
        facts = _load_facts("XCO", conn, as_of_date="2023-06-30")
    revenue_years = [fy for fy, _ in facts.get("revenue", [])]
    assert revenue_years == [2022]
    assert facts["revenue"][0] == (2022, 50.0)


def test_load_facts_without_as_of_returns_all(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('XCO', '0000000789', 'XCO Test Co', '2022-01-01T00:00:00Z')
            """
        )
        for fy, pend in [(2022, "2022-12-31"), (2024, "2024-12-31")]:
            conn.execute(
                "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, period_end, "
                "line_item, value, units, source_url, fetched_at, filed_date, form, accession) "
                "VALUES('XCO', ?, 'FY', ?, 'revenue', 1.0, 'USD_millions', "
                "'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000789.json', "
                "'2026-01-01T00:00:00Z', ?, '10-K', ?)",
                (fy, pend, f"{fy + 1}-02-15", f"0000000789-{str(fy + 1)[-2:]}-000001"),
            )
        conn.commit()
        from app.valuation.valuation_writer import _load_facts
        facts = _load_facts("XCO", conn)
    assert sorted(fy for fy, _ in facts["revenue"]) == [2022, 2024]
