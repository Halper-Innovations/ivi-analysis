from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from app.agent.queue import (
    cancel_job,
    claim_next_job,
    dead_letter_count,
    enqueue_job,
    list_dead_letters,
    mark_job_failure,
    retry_dead_letter,
    requeue_stale_running_jobs,
)
from app.agent.workers import JOB_HANDLERS, process_next_job
from app.db import get_db, init_db


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_requeue_stale_running_jobs_after_restart(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    job_id = enqueue_job("compute_fundamentals", {"ticker": "AAPL"})
    claimed = claim_next_job()
    assert claimed is not None and claimed["id"] == job_id

    stale_started = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    with get_db() as conn:
        conn.execute("UPDATE jobs SET started_at=?, status='running' WHERE id=?", (stale_started, job_id))

    recovered = requeue_stale_running_jobs(timeout_seconds=60)
    assert recovered == 1
    with get_db() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
    assert row["status"] == "pending"


def test_dead_letter_on_permanent_failure(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    job_id = enqueue_job("parse_filing", {"filing_id": 123}, max_attempts=1)
    claimed = claim_next_job()
    assert claimed is not None

    mark_job_failure(job_id, "permanent failure", attempts=1, transient=False, error_type="permanent")

    with get_db() as conn:
        row = conn.execute("SELECT status, error_type FROM jobs WHERE id=?", (job_id,)).fetchone()
    assert row["status"] == "dead_letter"
    assert row["error_type"] == "permanent"
    assert dead_letter_count() == 1


def test_cancelled_jobs_are_not_claimed(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    job_id = enqueue_job("run_valuation", {"ticker": "MSFT"})
    assert cancel_job(job_id, reason="test cancellation") is True
    assert claim_next_job() is None


def test_sqlite_lock_is_transient_retry_not_dead_letter(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    attempts = {"n": 0}

    def flaky_handler(payload):
        _ = payload
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setitem(JOB_HANDLERS, "test_sqlite_lock", flaky_handler)
    job_id = enqueue_job("test_sqlite_lock", {}, max_attempts=3)

    assert process_next_job() is True
    with get_db() as conn:
        row = conn.execute("SELECT status, error_type FROM jobs WHERE id=?", (job_id,)).fetchone()
    assert row["status"] == "pending"
    assert row["error_type"] == "sqlite_locked"
    assert dead_letter_count() == 0

    with get_db() as conn:
        conn.execute("UPDATE jobs SET scheduled_at=? WHERE id=?", (datetime.now(timezone.utc).isoformat(), job_id))

    assert process_next_job() is True
    with get_db() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
    assert row["status"] == "done"


def test_deadletter_retry_requeues_job(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    job_id = enqueue_job("parse_filing", {"filing_id": 999}, max_attempts=1)
    claimed = claim_next_job()
    assert claimed and claimed["id"] == job_id

    mark_job_failure(job_id, "hard fail", attempts=1, transient=False, error_type="permanent")
    rows = list_dead_letters(limit=5)
    assert rows

    new_job_id = retry_dead_letter(rows[0]["id"])
    assert isinstance(new_job_id, int)
    assert new_job_id != job_id
