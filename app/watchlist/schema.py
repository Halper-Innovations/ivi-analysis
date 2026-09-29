from __future__ import annotations

import sqlite3
from pathlib import Path

from app.config import get_config


WATCHLIST_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    status TEXT NOT NULL,
    conviction_grade TEXT,
    confidence TEXT,
    conviction_source TEXT,
    scan_family TEXT NOT NULL DEFAULT 'normal',
    valuation_anchor_method TEXT,
    valuation_anchor_value REAL,
    buy_price_target REAL,
    current_price_at_addition REAL,
    thesis_text TEXT,
    key_risks_json TEXT,
    falsifiers_json TEXT,
    open_questions_json TEXT,
    source_run_id TEXT NOT NULL,
    source_sector TEXT,
    added_at TEXT NOT NULL,
    last_evaluated_at TEXT,
    current_event_watermark_json TEXT,
    status_reason TEXT,
    UNIQUE(ticker, source_run_id)
);

CREATE TABLE IF NOT EXISTS watchlist_history (
    id INTEGER PRIMARY KEY,
    watchlist_id INTEGER NOT NULL REFERENCES watchlist(id),
    changed_at TEXT NOT NULL,
    field_name TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    source TEXT NOT NULL,
    source_run_id TEXT
);

CREATE TABLE IF NOT EXISTS watchlist_price_snapshots (
    id INTEGER PRIMARY KEY,
    watchlist_id INTEGER NOT NULL REFERENCES watchlist(id),
    price REAL NOT NULL,
    checked_at TEXT NOT NULL,
    source TEXT,
    quote_as_of_date TEXT,
    currency TEXT,
    price_basis TEXT,
    quote_snapshot_id TEXT,
    price_quote_id INTEGER
);

CREATE TABLE IF NOT EXISTS watchlist_reevaluation_publications (
    run_id TEXT PRIMARY KEY,
    watchlist_id INTEGER NOT NULL REFERENCES watchlist(id),
    ticker TEXT NOT NULL,
    evaluation TEXT NOT NULL,
    state_applied INTEGER NOT NULL CHECK(state_applied IN (0, 1)),
    entry_revision TEXT NOT NULL,
    evidence_fingerprint TEXT NOT NULL,
    artifact_json TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS watchlist_reevaluation_publications_no_update
BEFORE UPDATE ON watchlist_reevaluation_publications
BEGIN
    SELECT RAISE(ABORT, 'watchlist reevaluation publications are append-only');
END;

CREATE TRIGGER IF NOT EXISTS watchlist_reevaluation_publications_no_delete
BEFORE DELETE ON watchlist_reevaluation_publications
BEGIN
    SELECT RAISE(ABORT, 'watchlist reevaluation publications are append-only');
END;

CREATE INDEX IF NOT EXISTS idx_watchlist_ticker ON watchlist(ticker);
CREATE INDEX IF NOT EXISTS idx_watchlist_status ON watchlist(status);
CREATE INDEX IF NOT EXISTS idx_watchlist_price_snapshots_latest
    ON watchlist_price_snapshots(watchlist_id, checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_watchlist_reevaluation_publications_ticker
    ON watchlist_reevaluation_publications(ticker, created_at DESC);
"""


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    return Path(db_path) if db_path is not None else Path(get_config().db_path)


def apply_watchlist_schema(conn: sqlite3.Connection) -> None:
    """Create the watchlist tables and add any missing columns on ``conn``.

    ``init_db`` calls this so a freshly initialized database already has every
    table the web read model and the read-only CLI paths query. The caller
    owns the transaction (commit).
    """

    conn.executescript(WATCHLIST_SCHEMA_SQL)
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(watchlist)").fetchall()}
    if "confidence" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN confidence TEXT")
    if "conviction_source" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN conviction_source TEXT")
    if "scan_family" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN scan_family TEXT NOT NULL DEFAULT 'normal'")
    # Band-filter cap chain provenance (app/autonomous/cap_resolver.py):
    # cap in millions, which chain tier resolved it, canonical band token
    # (NULL renders as UNKNOWN_CAP), and the classification as-of date.
    if "market_cap_mm" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN market_cap_mm REAL")
    if "cap_source" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN cap_source TEXT")
    if "cap_band" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN cap_band TEXT")
    if "cap_asof" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN cap_asof TEXT")
    # Queue-protection flags (app/events/flags.py): comma-joined
    # EVENT_PENDING:<TYPE> tokens while the ticker has open corporate
    # events; NULL when clear. Blocks DEPLOY_READY presentation.
    if "event_pending" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN event_pending TEXT")
    # Liquidity layer (app/market/adv.py): 20d/60d dollar-ADV, the
    # as-of they were computed, and the capacity band. NULL ADV renders
    # as ADV_UNKNOWN — never fabricated.
    if "adv_dollar_20d" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN adv_dollar_20d REAL")
    if "adv_dollar_60d" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN adv_dollar_60d REAL")
    if "adv_asof" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN adv_asof TEXT")
    if "capacity_class" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN capacity_class TEXT")
    # Autonomous-sector v2 provenance. Nullable by design: historical,
    # filing-watch, manual, and corporate-event rows retain their prior semantics.
    if "pipeline_version" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN pipeline_version TEXT")
    if "candidate_disposition" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN candidate_disposition TEXT")
    if "decision_basis" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN decision_basis TEXT")
    if "selection_validation_status" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN selection_validation_status TEXT")
    # Exact adapter-backed current-event identity/timestamp snapshot from
    # the last reevaluation publication. NULL means legacy/uninitialized;
    # an initialized snapshot with zero events is stored as a JSON object.
    if "current_event_watermark_json" not in columns:
        conn.execute("ALTER TABLE watchlist ADD COLUMN current_event_watermark_json TEXT")


def ensure_watchlist_schema(db_path: str | Path | None = None) -> None:
    from app.db import connect

    resolved = resolve_db_path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(resolved)
    try:
        apply_watchlist_schema(conn)
        conn.commit()
    finally:
        conn.close()
