from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from app.agent.queue import (
    backlog_size,
    cancel_jobs,
    dead_letter_count,
    enqueue_job,
    recent_errors,
    requeue_stale_running_jobs,
)
from app.agent.workers import process_next_job, worker_loop
from app.config import get_config
from app.db import get_db, get_state, upsert_state, utc_now_iso
from app.logging import get_logger
from app.report.run_manifest import write_run_manifest
from app.util.http import HttpClient


logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _schedule_periodic_jobs(
    force_full_run: bool = False,
    as_of_date: str | None = None,
    include_research: bool = False,
    include_synthesis: bool = False,
) -> None:
    now = _utc_now()
    cfg = get_config()
    jobs_to_enqueue: list[tuple[str, dict[str, Any]]] = []
    with get_db() as conn:
        hourly = get_state(conn, "scheduler_last_hourly") or {}
        nightly = get_state(conn, "scheduler_last_nightly") or {}
        weekly = get_state(conn, "scheduler_last_weekly") or {}

        hourly_due = True
        if hourly.get("ts"):
            hourly_ts = datetime.fromisoformat(hourly["ts"])
            hourly_due = now - hourly_ts >= timedelta(minutes=60)

        nightly_due = force_full_run
        if not force_full_run:
            if nightly.get("date") != now.date().isoformat() and now.hour >= 2:
                nightly_due = True

        weekly_due = False
        if weekly.get("iso_week") != f"{now.isocalendar().year}-{now.isocalendar().week}" and now.weekday() == 0:
            weekly_due = True

        if hourly_due:
            jobs_to_enqueue.append(("check_edgar_updates", {"as_of_date": as_of_date}))
            upsert_state(conn, "scheduler_last_hourly", {"ts": utc_now_iso()})

        if nightly_due:
            nightly_run_id = f"scheduler_{now.strftime('%Y%m%dT%H%M%SZ')}"
            jobs_to_enqueue.append(("check_edgar_updates", {"as_of_date": as_of_date}))
            jobs_to_enqueue.append(
                (
                    "compute_fundamentals",
                    {
                        "as_of_date": as_of_date,
                        "include_research": include_research,
                        "include_synthesis": include_synthesis,
                        "run_id": nightly_run_id,
                    },
                )
            )
            jobs_to_enqueue.append(
                (
                    "run_valuation",
                    {
                        "as_of_date": as_of_date,
                        "include_research": include_research,
                        "include_synthesis": include_synthesis,
                        "run_id": nightly_run_id,
                    },
                )
            )
            jobs_to_enqueue.append(("build_rankings", {}))
            if cfg.nightly_research_cycle_enabled:
                jobs_to_enqueue.append(
                    (
                        "run_nightly_research_cycle",
                        {
                            "as_of_date": as_of_date,
                            "top_k": cfg.nightly_research_cycle_top_k,
                            "max_iterations": 1,
                        },
                    )
                )
            upsert_state(conn, "scheduler_last_nightly", {"date": now.date().isoformat(), "ts": utc_now_iso()})

        if weekly_due:
            jobs_to_enqueue.append(("build_weekly_digest", {}))
            upsert_state(
                conn,
                "scheduler_last_weekly",
                {"iso_week": f"{now.isocalendar().year}-{now.isocalendar().week}", "ts": utc_now_iso()},
            )

    for job_type, payload in jobs_to_enqueue:
        enqueue_job(job_type, payload)


def _stage_last_success(conn) -> dict[str, str]:
    rows = conn.execute("SELECT key, value_json FROM state WHERE key LIKE 'stage_last_success:%'").fetchall()
    out: dict[str, str] = {}
    for row in rows:
        stage = row["key"].split(":", 1)[1]
        payload = json.loads(row["value_json"])
        out[stage] = payload.get("ts")
    return out


def write_agent_status() -> None:
    cfg = get_config()
    cfg.status_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        stage_times = _stage_last_success(conn)

    status = {
        "generated_at": utc_now_iso(),
        "last_successful_run_times": stage_times,
        "job_backlog_size": backlog_size(),
        "dead_letter_count": dead_letter_count(),
        "last_50_errors": recent_errors(limit=50),
        "sec_request_metrics": HttpClient(get_config()).metrics(),
    }
    cfg.status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def run_scheduler_once(
    as_of_date: str | None = None,
    *,
    with_research: bool = False,
    with_synthesis: bool = False,
) -> None:
    cfg = get_config()
    as_of_date = as_of_date or _utc_now().date().isoformat()
    requeue_stale_running_jobs(cfg.running_job_timeout_seconds)
    include_research = with_research or cfg.research_enable_in_run_all
    _schedule_periodic_jobs(
        force_full_run=True,
        as_of_date=as_of_date,
        include_research=include_research,
        include_synthesis=with_synthesis,
    )

    idle_rounds = 0
    while True:
        progressed = process_next_job()
        if progressed:
            idle_rounds = 0
            continue
        idle_rounds += 1
        if idle_rounds >= 2:
            break
        time.sleep(0.5)

    write_agent_status()
    write_run_manifest(as_of_date=as_of_date)
    logger.info("scheduler_once_complete", extra={"stage_name": "scheduler"})


def run_scheduler_forever() -> None:
    cfg = get_config()
    stop_event = threading.Event()
    requeue_stale_running_jobs(cfg.running_job_timeout_seconds)

    workers = [
        threading.Thread(target=worker_loop, args=(stop_event,), name=f"worker-{idx+1}", daemon=True)
        for idx in range(max(1, cfg.scheduler_worker_count))
    ]
    for t in workers:
        t.start()

    try:
        while True:
            _schedule_periodic_jobs(
                force_full_run=False,
                as_of_date=_utc_now().date().isoformat(),
                include_research=cfg.research_enable_in_run_all,
                include_synthesis=False,
            )
            write_agent_status()
            write_run_manifest(as_of_date=_utc_now().date().isoformat())
            time.sleep(max(1, cfg.scheduler_poll_seconds))
    except KeyboardInterrupt:
        logger.info("scheduler_stopping", extra={"stage_name": "scheduler"})
    finally:
        stop_event.set()
        for t in workers:
            t.join(timeout=3)
        write_agent_status()
        write_run_manifest(as_of_date=_utc_now().date().isoformat())


def cancel_scheduled_jobs(job_type: str | None = None, reason: str = "cancelled by operator") -> int:
    return cancel_jobs(job_type=job_type, reason=reason)
