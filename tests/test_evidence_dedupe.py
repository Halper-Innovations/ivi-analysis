from __future__ import annotations

import sqlite3

from app.db import dedupe_evidence_items, init_db


def _legacy_evidence_items_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE evidence_items (
            evidence_id TEXT PRIMARY KEY,
            ticker TEXT,
            as_of_date TEXT,
            run_id TEXT,
            dedupe_key TEXT,
            item_hash TEXT,
            excerpt_hash TEXT,
            created_at TEXT
        )
        """
    )


def test_init_db_dedupes_evidence_items_before_unique_index_creation():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _legacy_evidence_items_table(conn)
    conn.execute(
        """
        INSERT INTO evidence_items(evidence_id, ticker, dedupe_key, item_hash, excerpt_hash, created_at)
        VALUES
          ('ev_old', 'AAPL', 'dup_1', 'h_old', 'h_old', '2026-02-10T00:00:00+00:00'),
          ('ev_new', 'AAPL', 'dup_1', 'h_new', 'h_new', '2026-02-11T00:00:00+00:00')
        """
    )

    init_db(conn=conn)

    count_row = conn.execute(
        "SELECT COUNT(*) AS n FROM evidence_items WHERE ticker='AAPL' AND dedupe_key='dup_1'"
    ).fetchone()
    assert int(count_row["n"] or 0) == 1
    keeper = conn.execute(
        "SELECT evidence_id FROM evidence_items WHERE ticker='AAPL' AND dedupe_key='dup_1'"
    ).fetchone()
    assert keeper["evidence_id"] == "ev_new"

    index_rows = conn.execute("PRAGMA index_list('evidence_items')").fetchall()
    dedupe_idx = [row for row in index_rows if row["name"] == "idx_evidence_items_dedupe_key"]
    assert dedupe_idx
    assert int(dedupe_idx[0]["unique"]) == 1


def test_dedupe_evidence_items_uses_rowid_when_created_at_missing():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _legacy_evidence_items_table(conn)
    conn.execute(
        """
        INSERT INTO evidence_items(evidence_id, ticker, dedupe_key, item_hash, excerpt_hash, created_at)
        VALUES
          ('ev_1', 'MSFT', 'dup_2', 'h1', 'h1', NULL),
          ('ev_2', 'MSFT', 'dup_2', 'h2', 'h2', NULL)
        """
    )

    summary = dedupe_evidence_items(conn)
    assert summary["duplicate_groups_before"] == 1
    assert summary["rows_deleted"] == 1
    assert summary["duplicate_groups_after"] == 0

    keeper = conn.execute(
        "SELECT evidence_id FROM evidence_items WHERE ticker='MSFT' AND dedupe_key='dup_2'"
    ).fetchone()
    assert keeper["evidence_id"] == "ev_2"
