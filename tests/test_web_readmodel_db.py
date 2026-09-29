from __future__ import annotations

import sqlite3

import pytest

from app.web.readmodel.db import OfflineError, open_readonly, readonly_db


def _make_engine_db(tmp_path):
    db_path = tmp_path / "engine.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE watchlist (id INTEGER PRIMARY KEY, ticker TEXT)")
    conn.execute("INSERT INTO watchlist (ticker) VALUES ('AAA')")
    conn.commit()
    conn.close()
    return db_path


def test_open_readonly_reads_rows(tmp_path):
    db_path = _make_engine_db(tmp_path)
    conn = open_readonly(db_path)
    try:
        row = conn.execute("SELECT ticker FROM watchlist WHERE id = 1").fetchone()
        assert row["ticker"] == "AAA"
    finally:
        conn.close()


def test_open_readonly_rejects_writes(tmp_path):
    db_path = _make_engine_db(tmp_path)
    conn = open_readonly(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO watchlist (ticker) VALUES ('BBB')")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM watchlist")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE scribble (id INTEGER)")
    finally:
        conn.close()


def test_open_readonly_missing_db_raises_named_precondition(tmp_path):
    missing = tmp_path / "nope" / "engine.db"
    with pytest.raises(OfflineError) as excinfo:
        open_readonly(missing)
    assert excinfo.value.precondition == "engine_db_missing"
    assert str(missing) in excinfo.value.detail
    # A failed open must not create the file (plain sqlite3.connect would).
    assert not missing.exists()


def test_readonly_db_context_manager_closes(tmp_path):
    db_path = _make_engine_db(tmp_path)
    with readonly_db(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM watchlist").fetchone()["n"] == 1
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
