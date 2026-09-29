from __future__ import annotations

import sqlite3

import pytest

from app.db import init_db


def test_synthesis_paid_attempt_ledger_migrates_existing_packet_table() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE synthesis_packets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            run_id TEXT NOT NULL,
            packet_path TEXT NOT NULL,
            packet_hash TEXT NOT NULL,
            packet_json TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            usage_json TEXT NOT NULL DEFAULT '{}',
            cost_estimate_usd REAL NOT NULL DEFAULT 0,
            from_cache INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            UNIQUE(ticker, as_of_date, run_id)
        )
        """
    )

    init_db(conn=conn)

    packet_columns = {row["name"] for row in conn.execute("PRAGMA table_info(synthesis_packets)")}
    attempt_columns = [
        row["name"] for row in conn.execute("PRAGMA table_info(synthesis_paid_attempts)")
    ]
    assert "paid_invocation_id" in packet_columns
    assert attempt_columns == [
        "attempt_id",
        "invocation_id",
        "physical_sequence",
        "ticker",
        "as_of_date",
        "run_id",
        "prompt_hash",
        "input_hash",
        "provider",
        "model",
        "schema_name",
        "status",
        "usage_json",
        "cost_estimate_usd",
        "created_at",
    ]

    conn.execute(
        """
        INSERT INTO synthesis_paid_attempts(
            attempt_id, invocation_id, physical_sequence, ticker, as_of_date,
            run_id, prompt_hash, input_hash, provider, model, schema_name,
            status, usage_json, cost_estimate_usd, created_at
        )
        VALUES(
            'invocation:1', 'invocation', 1, 'TEST', '2026-03-25',
            'run_1', 'prompt_hash', 'input_hash', 'openai', 'gpt-5-mini',
            'synthesis_packet_v1', 'OK', '{}', 0.0019,
            '2026-03-25T00:00:00+00:00'
        )
        """
    )
    with pytest.raises(
        sqlite3.IntegrityError,
        match="synthesis paid attempts are append-only",
    ):
        conn.execute(
            """
            UPDATE synthesis_paid_attempts
            SET cost_estimate_usd = 0.0
            WHERE attempt_id = 'invocation:1'
            """
        )
    with pytest.raises(
        sqlite3.IntegrityError,
        match="synthesis paid attempts are append-only",
    ):
        conn.execute("DELETE FROM synthesis_paid_attempts WHERE attempt_id = 'invocation:1'")
    conn.close()
