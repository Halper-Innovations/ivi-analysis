# tests/test_db_migration_ticker_outcomes.py
"""ticker_outcomes schema migration (additive entry/grade/benchmark columns).

Asserts the additive migration is idempotent and upgrades an OLD table in place
without data loss. New columns must be nullable and read NULL for pre-existing rows.
"""

from __future__ import annotations

import sqlite3

from app.db import init_db, utc_now_iso

NEW_COLUMNS = {
    "entry_price",
    "entry_price_source",
    "entry_date",
    "grade",
    "status",
    "benchmark_symbol",
    "benchmark_return_pct",
    "excess_return_pct",
    "buy_price_target",
    "reached_buy_target",
    "pipeline_version",
    "candidate_disposition",
    "decision_basis",
    "selection_validation_status",
    "source_sector",
    "source_artifact_path",
    "source_artifact_sha256",
    "source_decision_fingerprint",
    "financial_integrity_fingerprint",
}

# The pre-migration ticker_outcomes schema (no entry/grade/benchmark columns).
OLD_TICKER_OUTCOMES_DDL = """
CREATE TABLE ticker_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    run_id TEXT NOT NULL,
    discovery_run_id TEXT,
    deep_run_id TEXT,
    decision TEXT NOT NULL,
    conviction INTEGER NOT NULL,
    horizon_days INTEGER NOT NULL,
    thesis_tags_json TEXT NOT NULL DEFAULT '[]',
    notes TEXT,
    outcome_status TEXT NOT NULL DEFAULT 'OPEN',
    close_date TEXT,
    realized_return_pct REAL,
    max_drawdown_pct REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(ticker, as_of_date, run_id)
)
"""


def _open(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _column_names(conn) -> set[str]:
    return {r["name"] for r in conn.execute("PRAGMA table_info(ticker_outcomes)")}


def test_fresh_init_db_has_new_columns(tmp_path):
    db_path = tmp_path / "engine.db"
    conn = _open(db_path)
    init_db(conn=conn)
    cols = _column_names(conn)
    conn.close()
    assert cols.issuperset(NEW_COLUMNS)


def test_migration_preserves_old_row_and_adds_null_entry_price(tmp_path):
    db_path = tmp_path / "engine.db"
    conn = _open(db_path)
    # Build the OLD ticker_outcomes table and seed one row.
    conn.executescript(OLD_TICKER_OUTCOMES_DDL)
    now = utc_now_iso()
    conn.execute(
        """INSERT INTO ticker_outcomes
           (ticker, as_of_date, run_id, decision, conviction, horizon_days, created_at, updated_at)
           VALUES ('AAPL', '2026-05-17', 'r1', 'WATCH', 3, 365, ?, ?)""",
        (now, now),
    )
    conn.commit()

    # init_db (CREATE TABLE IF NOT EXISTS is a no-op) runs the additive migration.
    init_db(conn=conn)

    count = conn.execute("SELECT COUNT(*) AS c FROM ticker_outcomes").fetchone()["c"]
    row = conn.execute(
        "SELECT decision, entry_price, pipeline_version, candidate_disposition, "
        "decision_basis, selection_validation_status, source_sector, "
        "source_artifact_path, source_artifact_sha256, "
        "source_decision_fingerprint, financial_integrity_fingerprint "
        "FROM ticker_outcomes WHERE ticker='AAPL'"
    ).fetchone()
    cols = _column_names(conn)
    conn.close()

    assert count == 1
    assert row["decision"] == "WATCH"
    assert row["entry_price"] is None
    assert row["pipeline_version"] is None
    assert row["candidate_disposition"] is None
    assert row["decision_basis"] is None
    assert row["selection_validation_status"] is None
    assert row["source_sector"] is None
    assert row["source_artifact_path"] is None
    assert row["source_artifact_sha256"] is None
    assert row["source_decision_fingerprint"] is None
    assert row["financial_integrity_fingerprint"] is None
    assert cols.issuperset(NEW_COLUMNS)


def test_init_db_twice_is_idempotent_no_duplicate_columns(tmp_path):
    db_path = tmp_path / "engine.db"
    conn = _open(db_path)
    init_db(conn=conn)
    init_db(conn=conn)  # second call must not raise nor duplicate columns
    entry_price_count = sum(
        1 for r in conn.execute("PRAGMA table_info(ticker_outcomes)") if r["name"] == "entry_price"
    )
    conn.close()
    assert entry_price_count == 1
