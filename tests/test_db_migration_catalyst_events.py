# tests/test_db_migration_catalyst_events.py
"""catalyst_events cache table migration.

Asserts the catalyst_events table is created by the idempotent,
PRAGMA-table_info-guarded migration (mirroring the _migrate_ticker_outcomes
pattern). A pre-existing engine.db without the table gains it in place with no
data loss to other tables, and repeated init_db calls do not duplicate columns
or rows.
"""
from __future__ import annotations

import sqlite3

from app.db import (
    get_catalyst_event,
    init_db,
    upsert_catalyst_event,
)

EXPECTED_COLUMNS = {
    "id",
    "ticker",
    "as_of_date",
    "catalyst_type",
    "signal_label",
    "score",
    "detail_json",
    "source_url",
    "created_at",
}


def _open(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _column_names(conn) -> set[str]:
    return {r["name"] for r in conn.execute("PRAGMA table_info(catalyst_events)")}


def _table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def test_fresh_init_db_creates_catalyst_events(tmp_path):
    db_path = tmp_path / "engine.db"
    conn = _open(db_path)
    init_db(conn=conn)
    exists = _table_exists(conn, "catalyst_events")
    cols = _column_names(conn)
    conn.close()
    assert exists
    assert cols == EXPECTED_COLUMNS


def test_migration_adds_table_to_existing_db_without_data_loss(tmp_path):
    db_path = tmp_path / "engine.db"
    conn = _open(db_path)
    # An OLD engine.db with no catalyst_events table but a seeded companies row.
    conn.executescript(
        """
        CREATE TABLE companies (
            ticker TEXT PRIMARY KEY,
            name TEXT
        );
        """
    )
    conn.execute("INSERT INTO companies(ticker, name) VALUES('AAPL', 'Apple')")
    conn.commit()
    assert not _table_exists(conn, "catalyst_events")

    # Schema evolution must add the missing table in place.
    init_db(conn=conn)

    name = conn.execute("SELECT name FROM companies WHERE ticker='AAPL'").fetchone()["name"]
    exists = _table_exists(conn, "catalyst_events")
    cols = _column_names(conn)
    conn.close()

    assert name == "Apple"
    assert exists
    assert cols == EXPECTED_COLUMNS


def test_init_db_twice_is_idempotent_no_duplicate_columns(tmp_path):
    db_path = tmp_path / "engine.db"
    conn = _open(db_path)
    init_db(conn=conn)
    init_db(conn=conn)  # second call must not raise nor duplicate the table
    score_col_count = sum(
        1 for r in conn.execute("PRAGMA table_info(catalyst_events)") if r["name"] == "score"
    )
    conn.close()
    assert score_col_count == 1


def test_upsert_catalyst_event_is_idempotent_on_unique_key(tmp_path):
    db_path = tmp_path / "engine.db"
    conn = _open(db_path)
    init_db(conn=conn)

    upsert_catalyst_event(
        conn,
        ticker="AAA",
        as_of_date="2024-12-01",
        catalyst_type="INSIDER_BUY_CLUSTER",
        signal_label="WEAK",
        score=2.0,
        detail={"insider_distinct_buyers": 1},
        source_url="https://example.test/form4.xml",
    )
    # Re-run for the same (ticker, as_of_date, catalyst_type) overwrites in place.
    upsert_catalyst_event(
        conn,
        ticker="AAA",
        as_of_date="2024-12-01",
        catalyst_type="INSIDER_BUY_CLUSTER",
        signal_label="CONFIRMED",
        score=4.0,
        detail={"insider_distinct_buyers": 2},
        source_url="https://example.test/form4b.xml",
    )
    conn.commit()

    row_count = conn.execute(
        "SELECT COUNT(*) AS c FROM catalyst_events "
        "WHERE ticker='AAA' AND as_of_date='2024-12-01' "
        "AND catalyst_type='INSIDER_BUY_CLUSTER'"
    ).fetchone()["c"]
    cached = get_catalyst_event(
        conn,
        ticker="AAA",
        as_of_date="2024-12-01",
        catalyst_type="INSIDER_BUY_CLUSTER",
    )
    conn.close()

    assert row_count == 1
    assert cached is not None
    assert cached["signal_label"] == "CONFIRMED"
    assert cached["score"] == 4.0
    assert cached["detail"] == {"insider_distinct_buyers": 2}
    assert cached["source_url"] == "https://example.test/form4b.xml"
