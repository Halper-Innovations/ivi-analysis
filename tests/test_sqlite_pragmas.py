from __future__ import annotations

from app.db import get_db, init_db


def _init_temp_db(monkeypatch, tmp_path, busy_timeout_ms: int = 6500):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SQLITE_BUSY_TIMEOUT_MS", str(busy_timeout_ms))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_sqlite_pragmas_are_set(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path, busy_timeout_ms=7777)

    with get_db() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = int(conn.execute("PRAGMA synchronous").fetchone()[0])
        foreign_keys = int(conn.execute("PRAGMA foreign_keys").fetchone()[0])
        busy_timeout = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])

    assert str(journal_mode).lower() == "wal"
    # SQLite encodes NORMAL as 1.
    assert synchronous == 1
    assert foreign_keys == 1
    assert busy_timeout == 7777
