from __future__ import annotations

import sqlite3

import pytest

from app.config import get_config
from app.db import get_db, init_db


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def test_corporate_events_tables_exist_with_expected_columns(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(corporate_events)")}
        assert cols == {
            "id", "cik", "event_type", "anchor_accession", "company_name",
            "ticker", "ticker_state", "status", "detection_date",
            "qualification_date", "expiry_reason", "detail_json", "source_mode",
            "detected_at", "qualified_at", "surfaced_at", "decided_at",
            "expired_at", "updated_at",
        }
        for table in ("corporate_event_filings", "corporate_event_skips", "corporate_event_scans"):
            n = conn.execute(
                "SELECT COUNT(*) c FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()["c"]
            assert n == 1


def test_corporate_events_unique_key(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    row = ("0000123456", "spinoff", "0001234567-26-000001", "SPINCO CORP",
           "DETECTED", "2026-01-05", "daily", "2026-01-06T00:00:00Z", "2026-01-06T00:00:00Z")
    sql = ("INSERT INTO corporate_events(cik, event_type, anchor_accession, company_name, "
           "status, detection_date, source_mode, detected_at, updated_at) "
           "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)")
    with get_db() as conn:
        conn.execute(sql, row)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql, row)


def test_corporate_event_scans_unique_on_scan_date(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    sql = ("INSERT INTO corporate_event_scans(scan_date, mode, status, created_at, updated_at) "
           "VALUES(?, ?, ?, ?, ?)")
    with get_db() as conn:
        conn.execute(sql, ("2026-06-01", "daily", "OK", "x", "x"))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql, ("2026-06-01", "daily", "OK", "x", "x"))


def test_migration_is_idempotent(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    # Running init_db twice must be harmless.
    init_db()
    with get_db() as conn:
        n = conn.execute(
            "SELECT COUNT(*) c FROM sqlite_master WHERE type='table' AND name='corporate_events'"
        ).fetchone()["c"]
        assert n == 1
