"""DDL for the immutable recommendation ledger."""

from __future__ import annotations

import sqlite3


RECOMMENDATION_LEDGER_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS recommendation_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id TEXT NOT NULL UNIQUE,
    ticker TEXT NOT NULL,
    recommendation_type TEXT NOT NULL
        CHECK (recommendation_type IN ('BUY_AT_LIMIT', 'WATCH', 'AVOID')),
    record_vintage TEXT NOT NULL
        CHECK (record_vintage IN ('LIVE', 'SEED')),
    model_id TEXT NOT NULL,
    model_vintage TEXT NOT NULL,
    thesis_reference TEXT NOT NULL,
    thesis_summary TEXT NOT NULL,
    trigger_price REAL NOT NULL CHECK (trigger_price > 0),
    trigger_price_source TEXT NOT NULL,
    trigger_price_as_of TEXT NOT NULL,
    trigger_price_age_seconds INTEGER NOT NULL
        CHECK (trigger_price_age_seconds >= 0),
    target_price REAL NOT NULL CHECK (target_price > 0),
    target_price_source TEXT NOT NULL,
    conviction_grade TEXT NOT NULL,
    capacity_class TEXT NOT NULL,
    adv_dollar_20d REAL CHECK (adv_dollar_20d IS NULL OR adv_dollar_20d > 0),
    adv_as_of TEXT,
    pre_mortem TEXT NOT NULL,
    risk_flags_json TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    source_run_id TEXT NOT NULL,
    horizons_json TEXT NOT NULL,
    benchmark_symbol TEXT NOT NULL
        CHECK (benchmark_symbol IN ('IWM', 'SPY')),
    staked_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    source_disposition_id INTEGER,
    corrects_recommendation_id TEXT
        REFERENCES recommendation_ledger(recommendation_id),
    correction_reason TEXT,
    CHECK (
        (record_vintage = 'SEED' AND source_disposition_id IS NOT NULL)
        OR
        (record_vintage = 'LIVE' AND source_disposition_id IS NULL)
    ),
    CHECK (
        (corrects_recommendation_id IS NULL AND correction_reason IS NULL)
        OR
        (corrects_recommendation_id IS NOT NULL AND correction_reason IS NOT NULL)
    ),
    CHECK (
        corrects_recommendation_id IS NULL
        OR corrects_recommendation_id != recommendation_id
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_recommendation_ledger_seed_source
    ON recommendation_ledger(source_disposition_id)
    WHERE record_vintage = 'SEED';

CREATE INDEX IF NOT EXISTS idx_recommendation_ledger_ticker_staked
    ON recommendation_ledger(ticker, staked_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_recommendation_ledger_vintage_type
    ON recommendation_ledger(record_vintage, recommendation_type, staked_at);

CREATE TRIGGER IF NOT EXISTS recommendation_ledger_no_update
BEFORE UPDATE ON recommendation_ledger
BEGIN
    SELECT RAISE(ABORT, 'recommendation ledger rows are immutable');
END;

CREATE TRIGGER IF NOT EXISTS recommendation_ledger_no_delete
BEFORE DELETE ON recommendation_ledger
BEGIN
    SELECT RAISE(ABORT, 'recommendation ledger rows are immutable');
END;
"""


def ensure_recommendation_ledger_schema(conn: sqlite3.Connection) -> None:
    """Create only the new recommendation ledger objects."""

    conn.executescript(RECOMMENDATION_LEDGER_SCHEMA_SQL)
