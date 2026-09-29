from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from app.db import get_db, utc_now_iso


class TransientJobError(RuntimeError):
    pass


class PermanentJobError(RuntimeError):
    pass


class CancelledJobError(RuntimeError):
    pass


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def enqueue_job(
    job_type: str,
    payload: dict[str, Any] | None = None,
    scheduled_at: datetime | None = None,
    max_attempts: int = 5,
) -> int:
    payload = payload or {}
    scheduled_at = scheduled_at or now_utc()
    payload_json = json.dumps(payload, sort_keys=True)
    with get_db() as conn:
        existing = conn.execute(
            """
            SELECT id
            FROM jobs
            WHERE job_type = ? AND payload_json = ? AND status IN ('pending', 'running')
            LIMIT 1
            """,
            (job_type, payload_json),
        ).fetchone()
        if existing:
            return int(existing["id"])

        conn.execute(
            """
            INSERT INTO jobs(
                job_type, payload_json, scheduled_at, attempts, max_attempts,
                status, created_at, updated_at
            ) VALUES(?, ?, ?, 0, ?, 'pending', ?, ?)
            """,
            (
                job_type,
                payload_json,
                scheduled_at.isoformat(),
                max(1, max_attempts),
                utc_now_iso(),
                utc_now_iso(),
            ),
        )
        row = conn.execute("SELECT last_insert_rowid() AS id").fetchone()
        return int(row["id"])


def cancel_job(job_id: int, reason: str = "cancelled by operator") -> bool:
    with get_db() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return False
        if row["status"] in {"done", "failed", "dead_letter", "cancelled"}:
            return False
        conn.execute(
            """
            UPDATE jobs
            SET status='cancelled', cancelled_at=?, cancel_reason=?, updated_at=?
            WHERE id=?
            """,
            (utc_now_iso(), reason[:500], utc_now_iso(), job_id),
        )
        return True


def cancel_jobs(job_type: str | None = None, reason: str = "cancelled by operator") -> int:
    with get_db() as conn:
        if job_type:
            rows = conn.execute(
                "SELECT id FROM jobs WHERE job_type = ? AND status IN ('pending', 'running')",
                (job_type,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT id FROM jobs WHERE status IN ('pending', 'running')").fetchall()
        ids = [int(row["id"]) for row in rows]
    count = 0
    for job_id in ids:
        if cancel_job(job_id, reason=reason):
            count += 1
    return count


def requeue_stale_running_jobs(timeout_seconds: int) -> int:
    if timeout_seconds <= 0:
        return 0
    cutoff = now_utc() - timedelta(seconds=timeout_seconds)
    recovered = 0
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id
            FROM jobs
            WHERE status='running' AND started_at IS NOT NULL AND started_at < ?
            """,
            (cutoff.isoformat(),),
        ).fetchall()
        for row in rows:
            conn.execute(
                """
                UPDATE jobs
                SET status='pending',
                    scheduled_at=?,
                    last_error=?,
                    updated_at=?,
                    started_at=NULL,
                    error_type='restart_requeue'
                WHERE id=?
                """,
                (utc_now_iso(), "Recovered stale running job after restart", utc_now_iso(), int(row["id"])),
            )
            recovered += 1
    return recovered


def claim_next_job() -> dict[str, Any] | None:
    try:
        with get_db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT *
                FROM jobs
                WHERE status = 'pending'
                  AND scheduled_at <= ?
                  AND cancelled_at IS NULL
                ORDER BY scheduled_at ASC, id ASC
                LIMIT 1
                """,
                (utc_now_iso(),),
            ).fetchone()
            if not row:
                return None
            conn.execute(
                """
                UPDATE jobs
                SET status='running', attempts=attempts+1, started_at=?, updated_at=?
                WHERE id=?
                """,
                (utc_now_iso(), utc_now_iso(), row["id"]),
            )
            return {
                "id": int(row["id"]),
                "job_type": row["job_type"],
                "payload": json.loads(row["payload_json"]),
                "attempts": int(row["attempts"]) + 1,
                "max_attempts": int(row["max_attempts"] or 5),
            }
    except sqlite3.OperationalError as exc:
        if "database is locked" in str(exc).lower():
            return None
        raise


def mark_job_success(job_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status='done', last_error=NULL, error_type=NULL, updated_at=?, started_at=NULL
            WHERE id=?
            """,
            (utc_now_iso(), job_id),
        )


def _to_dead_letter(job_id: int, error_type: str, error: str) -> None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT job_type, payload_json, attempts FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return
        conn.execute(
            """
            INSERT INTO dead_letter_jobs(job_id, job_type, payload_json, attempts, error_type, error_message, moved_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, row["job_type"], row["payload_json"], int(row["attempts"]), error_type, error[:1000], utc_now_iso()),
        )
        conn.execute(
            "UPDATE jobs SET status='dead_letter', last_error=?, error_type=?, updated_at=?, started_at=NULL WHERE id=?",
            (error[:1000], error_type, utc_now_iso(), job_id),
        )


def mark_job_failure(
    job_id: int,
    error: str,
    attempts: int,
    *,
    transient: bool = True,
    max_attempts: int = 5,
    error_type: str = "unknown",
) -> None:
    with get_db() as conn:
        row = conn.execute("SELECT cancelled_at FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row and row["cancelled_at"] is not None:
            conn.execute(
                "UPDATE jobs SET status='cancelled', last_error=?, error_type=?, updated_at=?, started_at=NULL WHERE id=?",
                ("Cancelled before completion", "cancelled", utc_now_iso(), job_id),
            )
            return

    if transient and attempts < max_attempts:
        delay = 2 ** min(6, attempts)
        next_time = now_utc() + timedelta(seconds=delay)
        with get_db() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status='pending',
                    scheduled_at=?,
                    last_error=?,
                    error_type=?,
                    updated_at=?,
                    started_at=NULL
                WHERE id=?
                """,
                (next_time.isoformat(), error[:1000], error_type, utc_now_iso(), job_id),
            )
        return

    _to_dead_letter(job_id, error_type=error_type, error=error)


def backlog_size() -> int:
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE status='pending'").fetchone()
        return int(row["n"])


def recent_errors(limit: int = 50) -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, job_type, attempts, last_error, error_type, status, updated_at
            FROM jobs
            WHERE status IN ('failed', 'dead_letter') OR (last_error IS NOT NULL AND last_error != '')
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "job_type": row["job_type"],
            "attempts": int(row["attempts"]),
            "last_error": row["last_error"],
            "error_type": row["error_type"],
            "status": row["status"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


def dead_letter_count() -> int:
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM dead_letter_jobs").fetchone()
        return int(row["n"])


def list_dead_letters(limit: int = 50, job_type: str | None = None) -> list[dict[str, Any]]:
    with get_db() as conn:
        if job_type:
            rows = conn.execute(
                """
                SELECT id, job_id, job_type, payload_json, attempts, error_type, error_message, moved_at
                FROM dead_letter_jobs
                WHERE job_type = ?
                ORDER BY moved_at DESC
                LIMIT ?
                """,
                (job_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, job_id, job_type, payload_json, attempts, error_type, error_message, moved_at
                FROM dead_letter_jobs
                ORDER BY moved_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "job_id": row["job_id"],
            "job_type": row["job_type"],
            "payload": json.loads(row["payload_json"]),
            "attempts": int(row["attempts"]),
            "error_type": row["error_type"],
            "error_message": row["error_message"],
            "moved_at": row["moved_at"],
        }
        for row in rows
    ]


def retry_dead_letter(dead_letter_id: int) -> int | None:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT job_type, payload_json
            FROM dead_letter_jobs
            WHERE id = ?
            """,
            (dead_letter_id,),
        ).fetchone()
    if not row:
        return None
    payload = json.loads(row["payload_json"])
    return enqueue_job(row["job_type"], payload)


def retry_dead_letter_all(job_type: str | None = None, limit: int | None = None) -> int:
    with get_db() as conn:
        if job_type:
            rows = conn.execute(
                """
                SELECT id
                FROM dead_letter_jobs
                WHERE job_type = ?
                ORDER BY moved_at DESC
                """,
                (job_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id
                FROM dead_letter_jobs
                ORDER BY moved_at DESC
                """
            ).fetchall()
    ids = [int(row["id"]) for row in rows]
    if limit is not None and limit > 0:
        ids = ids[:limit]
    retried = 0
    for dead_letter_id in ids:
        if retry_dead_letter(dead_letter_id) is not None:
            retried += 1
    return retried


def purge_dead_letters(older_than_days: int) -> int:
    cutoff = now_utc() - timedelta(days=max(0, older_than_days))
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM dead_letter_jobs WHERE moved_at < ?",
            (cutoff.isoformat(),),
        ).fetchone()
        count = int(row["n"])
        conn.execute("DELETE FROM dead_letter_jobs WHERE moved_at < ?", (cutoff.isoformat(),))
    return count
