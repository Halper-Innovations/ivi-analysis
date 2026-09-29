"""Regression coverage for the companyfacts constraint rebuild."""

from __future__ import annotations

import sqlite3

from app.db import init_db


LEGACY_COMPANYFACTS_DDL = """
CREATE TABLE companyfacts_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    period_type TEXT NOT NULL DEFAULT 'FY',
    period_end TEXT NOT NULL,
    line_item TEXT NOT NULL,
    value REAL,
    units TEXT,
    source_url TEXT,
    fetched_at TEXT NOT NULL,
    filed_date TEXT,
    form TEXT,
    accession TEXT,
    UNIQUE(ticker, fiscal_year, line_item)
)
"""


def _open(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _unique_column_sets(conn: sqlite3.Connection) -> set[tuple[str, ...]]:
    unique_columns: set[tuple[str, ...]] = set()
    for index in conn.execute("PRAGMA index_list(companyfacts_facts)"):
        if not index["unique"]:
            continue
        columns = tuple(row["name"] for row in conn.execute(f"PRAGMA index_info({index['name']})"))
        unique_columns.add(columns)
    return unique_columns


def test_constraint_migration_preserves_filed_asof_provenance(tmp_path):
    conn = _open(tmp_path / "engine.db")
    conn.executescript(LEGACY_COMPANYFACTS_DDL)
    conn.execute(
        """
        INSERT INTO companyfacts_facts(
            id, ticker, fiscal_year, period_type, period_end, line_item,
            value, units, source_url, fetched_at, filed_date, form, accession
        ) VALUES(
            41, 'PITX', 2025, 'FY', '2025-12-31', 'revenue',
            125.0, 'USD_millions',
            'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000041.json',
            '2026-02-15T00:00:00Z', '2026-02-14', '10-K',
            '0000000041-26-000001'
        )
        """
    )
    conn.commit()

    init_db(conn=conn)
    init_db(conn=conn)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(companyfacts_facts)")}
    row = conn.execute(
        """
        SELECT id, period_type, filed_date, form, accession
        FROM companyfacts_facts
        WHERE ticker = 'PITX' AND line_item = 'revenue'
        """
    ).fetchone()
    unique_column_sets = _unique_column_sets(conn)
    conn.close()

    assert {"filed_date", "form", "accession"}.issubset(columns)
    assert (
        "ticker",
        "fiscal_year",
        "period_type",
        "line_item",
    ) in unique_column_sets
    assert dict(row) == {
        "id": 41,
        "period_type": "FY",
        "filed_date": "2026-02-14",
        "form": "10-K",
        "accession": "0000000041-26-000001",
    }
