"""Heartbeat run ledger.

Shell heartbeats record each step's exit code here via
``ivi ops heartbeat-record``; the 16:30 catch-up checker and the dead-man
check read completion from this table instead of grepping log files for
done markers. A heartbeat is COMPLETE for a date when its ``_complete``
step was recorded with exit code 0.

Writes are best-effort by contract: the shell helpers append ``|| true``
so a broken engine.db (the exact failure the ledger exists to expose)
cannot also break the heartbeat that would report it — the alert path is
independent of this table.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


COMPLETE_STEP = "_complete"

STATUS_OK = "OK"
STATUS_FAILED = "FAILED"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    from app.db import connect, init_db

    conn = connect(db_path)
    try:
        conn.execute("SELECT 1 FROM heartbeat_runs LIMIT 1")
    except sqlite3.Error:
        init_db(conn=conn)
    return conn


def record_step(
    *,
    heartbeat: str,
    step: str,
    exit_code: int,
    run_date: str | None = None,
    detail: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Append one step outcome; returns the recorded row as a dict."""
    row = {
        "heartbeat": str(heartbeat).strip(),
        "run_date": str(run_date or date.today().isoformat()),
        "step": str(step).strip(),
        "status": STATUS_OK if int(exit_code) == 0 else STATUS_FAILED,
        "exit_code": int(exit_code),
        "detail": detail,
        "recorded_at": _utc_now_iso(),
    }
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO heartbeat_runs(
                heartbeat, run_date, step, status, exit_code, detail, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["heartbeat"],
                row["run_date"],
                row["step"],
                row["status"],
                row["exit_code"],
                row["detail"],
                row["recorded_at"],
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return row


def derive_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Completion state over one heartbeat-date's ordered step rows (pure).

    status: COMPLETE (a _complete row with exit 0), FAILED (latest _complete
    nonzero, or any failed step and no clean completion afterward), STARTED
    (steps recorded, no completion), MISSING (no rows). The dead-man check
    and the web UI ops deck both read completion through this one
    derivation.
    """
    completes = [r for r in rows if r["step"] == COMPLETE_STEP]
    failed_steps = [
        r for r in rows if r["step"] != COMPLETE_STEP and r["status"] == STATUS_FAILED
    ]
    if completes:
        last = completes[-1]
        status = "COMPLETE" if int(last["exit_code"] or 0) == 0 else "FAILED"
    elif failed_steps:
        status = "FAILED"
    elif rows:
        status = "STARTED"
    else:
        status = "MISSING"
    return {"status": status, "failed_steps": [r["step"] for r in failed_steps]}


def heartbeat_status(
    *,
    heartbeat: str,
    run_date: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Completion state for one heartbeat on one date (see derive_status)."""
    target_date = str(run_date or date.today().isoformat())
    conn = _connect(db_path)
    try:
        rows = [
            dict(r)
            for r in conn.execute(
                """
                SELECT step, status, exit_code, detail, recorded_at
                FROM heartbeat_runs
                WHERE heartbeat = ? AND run_date = ?
                ORDER BY recorded_at ASC, id ASC
                """,
                (str(heartbeat).strip(), target_date),
            ).fetchall()
        ]
    finally:
        conn.close()

    derived = derive_status(rows)
    return {
        "heartbeat": str(heartbeat).strip(),
        "run_date": target_date,
        "status": derived["status"],
        "steps": rows,
        "failed_steps": derived["failed_steps"],
    }


__all__ = [
    "COMPLETE_STEP",
    "STATUS_FAILED",
    "STATUS_OK",
    "derive_status",
    "heartbeat_status",
    "record_step",
]
