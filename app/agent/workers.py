from __future__ import annotations

import sqlite3
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from app.agent.analyst_agent import run_analyst_agent_for_ticker, run_red_team_for_ticker
from app.agent.queue import (
    CancelledJobError,
    PermanentJobError,
    TransientJobError,
    claim_next_job,
    enqueue_job,
    mark_job_failure,
    mark_job_success,
)
from app.config import get_config
from app.db import get_db, get_state, upsert_state, utc_now_iso
from app.evidence.packet_builder import build_packet_for_ticker
from app.fundamentals.metrics import compute_all_fundamentals, compute_fundamentals_for_ticker
from app.ingest.filings import download_filing_by_id, ingest_since
from app.logging import get_logger
from app.llm.synthesis_agent import run_synthesis_for_ticker
from app.parse.filing_parser import parse_filing_by_id
from app.research import run_research_agent_for_ticker
from app.report.memo_builder import build_memo_for_ticker, build_weekly_digest
from app.score.ranker import score_and_rank, score_ticker
from app.valuation.sanity_checks import run_all_valuations, run_valuation_for_ticker
from app.util.http import DomainBudgetExceeded


logger = get_logger(__name__)
_DB_WRITE_SEMAPHORE = threading.BoundedSemaphore(max(1, int(get_config().max_db_write_concurrency)))


def _update_stage_success(stage: str) -> None:
    with get_db() as conn:
        upsert_state(conn, f"stage_last_success:{stage}", {"ts": utc_now_iso()})


def _ticker_for_filing(filing_id: int) -> str | None:
    with get_db() as conn:
        row = conn.execute("SELECT ticker FROM filings WHERE id = ?", (filing_id,)).fetchone()
        if not row:
            return None
        return row["ticker"]


def _handle_check_edgar_updates(payload: dict[str, Any]) -> None:
    forms = payload.get("forms", ["10-K", "10-Q", "8-K"])
    as_of_date = payload.get("as_of_date")
    with get_db() as conn:
        state = get_state(conn, "last_edgar_check") or {}
        last_date = state.get("date")
    if last_date:
        since = date.fromisoformat(last_date)
    else:
        since = date.today() - timedelta(days=2)

    ingest_since(since, forms, as_of_date=as_of_date)

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, ticker
            FROM filings
            WHERE status IN ('downloaded', 'new')
            ORDER BY COALESCE(filing_date, '1900-01-01') DESC
            LIMIT 500
            """
        ).fetchall()
        upsert_state(conn, "last_edgar_check", {"date": as_of_date or date.today().isoformat(), "ts": utc_now_iso()})

    for row in rows:
        enqueue_job("parse_filing", {"filing_id": int(row["id"]), "ticker": row["ticker"]})

    _update_stage_success("check_edgar_updates")


def _handle_ingest_filing(payload: dict[str, Any]) -> None:
    filing_id = payload.get("filing_id")
    if filing_id is None:
        raise ValueError("ingest_filing requires filing_id")
    ok = download_filing_by_id(int(filing_id))
    if not ok:
        raise TransientJobError(f"Failed to download filing_id={filing_id}")
    ticker = _ticker_for_filing(int(filing_id))
    enqueue_job("parse_filing", {"filing_id": int(filing_id), "ticker": ticker})
    _update_stage_success("ingest_filing")


def _handle_parse_filing(payload: dict[str, Any]) -> None:
    filing_id = payload.get("filing_id")
    if filing_id is None:
        raise ValueError("parse_filing requires filing_id")
    ok = parse_filing_by_id(int(filing_id))
    if not ok:
        raise TransientJobError(f"Failed to parse filing_id={filing_id}")
    ticker = payload.get("ticker") or _ticker_for_filing(int(filing_id))
    if ticker:
        enqueue_job("compute_fundamentals", {"ticker": ticker})
    _update_stage_success("parse_filing")


def _handle_compute_fundamentals(payload: dict[str, Any]) -> None:
    ticker = payload.get("ticker")
    as_of_date = payload.get("as_of_date")
    include_research = bool(payload.get("include_research"))
    include_synthesis = bool(payload.get("include_synthesis"))
    run_id = payload.get("run_id")
    if ticker:
        ok = compute_fundamentals_for_ticker(ticker, as_of_date=as_of_date)
        if ok:
            enqueue_job(
                "run_valuation",
                {
                    "ticker": ticker,
                    "as_of_date": as_of_date,
                    "include_research": include_research,
                    "include_synthesis": include_synthesis,
                    "run_id": run_id,
                },
            )
    else:
        compute_all_fundamentals(as_of_date=as_of_date)
        with get_db() as conn:
            rows = conn.execute("SELECT DISTINCT ticker FROM fundamentals").fetchall()
        for row in rows:
            enqueue_job(
                "run_valuation",
                {
                    "ticker": row["ticker"],
                    "as_of_date": as_of_date,
                    "include_research": include_research,
                    "include_synthesis": include_synthesis,
                    "run_id": run_id,
                },
            )
    _update_stage_success("compute_fundamentals")


def _handle_run_valuation(payload: dict[str, Any]) -> None:
    ticker = payload.get("ticker")
    as_of_date = payload.get("as_of_date")
    include_research = bool(payload.get("include_research"))
    include_synthesis = bool(payload.get("include_synthesis"))
    run_id = payload.get("run_id")
    if ticker:
        ok = run_valuation_for_ticker(ticker, run_as_of_date=as_of_date)
        if ok:
            enqueue_job(
                "build_evidence_packet",
                {
                    "ticker": ticker,
                    "as_of_date": as_of_date,
                    "include_research": include_research,
                    "include_synthesis": include_synthesis,
                    "run_id": run_id,
                },
            )
    else:
        run_all_valuations(as_of_date=as_of_date)
        with get_db() as conn:
            rows = conn.execute("SELECT DISTINCT ticker FROM fundamentals").fetchall()
        for row in rows:
            enqueue_job(
                "build_evidence_packet",
                {
                    "ticker": row["ticker"],
                    "as_of_date": as_of_date,
                    "include_research": include_research,
                    "include_synthesis": include_synthesis,
                    "run_id": run_id,
                },
            )
    _update_stage_success("run_valuation")


def _handle_build_evidence_packet(payload: dict[str, Any]) -> None:
    ticker = payload.get("ticker")
    as_of_date = payload.get("as_of_date")
    include_research = bool(payload.get("include_research"))
    include_synthesis = bool(payload.get("include_synthesis"))
    run_id = payload.get("run_id")
    if not ticker:
        raise ValueError("build_evidence_packet requires ticker")
    packet = build_packet_for_ticker(ticker, as_of_date=as_of_date, dossier_run_id=run_id)
    if packet:
        if include_research:
            run_id = run_id or f"rq_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            enqueue_job(
                "run_research_agent",
                {
                    "ticker": ticker,
                    "as_of_date": as_of_date,
                    "run_id": run_id,
                    "include_synthesis": include_synthesis,
                },
            )
        else:
            enqueue_job(
                "run_analyst_agent",
                {
                    "ticker": ticker,
                    "as_of_date": as_of_date,
                    "run_id": run_id,
                    "include_synthesis": include_synthesis,
                },
            )
    _update_stage_success("build_evidence_packet")


def _handle_run_research_agent(payload: dict[str, Any]) -> None:
    ticker = payload.get("ticker")
    as_of_date = payload.get("as_of_date")
    run_id = payload.get("run_id")
    include_synthesis = bool(payload.get("include_synthesis"))
    if not ticker:
        raise ValueError("run_research_agent requires ticker")
    path = run_research_agent_for_ticker(ticker=ticker, as_of_date=as_of_date, run_id=run_id)
    if path is None:
        raise TransientJobError(f"Research packet not produced for {ticker}")
    enqueue_job(
        "run_analyst_agent",
        {
            "ticker": ticker,
            "as_of_date": as_of_date,
            "run_id": run_id,
            "include_synthesis": include_synthesis,
        },
    )
    _update_stage_success("run_research_agent")


def _handle_run_analyst_agent(payload: dict[str, Any]) -> None:
    ticker = payload.get("ticker")
    as_of_date = payload.get("as_of_date")
    run_id = payload.get("run_id")
    include_synthesis = bool(payload.get("include_synthesis"))
    if not ticker:
        raise ValueError("run_analyst_agent requires ticker")
    bundle = run_analyst_agent_for_ticker(ticker)
    if bundle is None:
        raise TransientJobError(f"Analyst bundle not produced for {ticker}")
    enqueue_job("run_red_team", {"ticker": ticker})
    enqueue_job(
        "score_ticker",
        {"ticker": ticker, "as_of_date": as_of_date, "run_id": run_id, "include_synthesis": include_synthesis},
    )
    _update_stage_success("run_analyst_agent")


def _handle_run_red_team(payload: dict[str, Any]) -> None:
    ticker = payload.get("ticker")
    if not ticker:
        raise ValueError("run_red_team requires ticker")
    red = run_red_team_for_ticker(ticker)
    if red is None:
        raise TransientJobError(f"Red team output missing for {ticker}")
    _update_stage_success("run_red_team")


def _handle_score_ticker(payload: dict[str, Any]) -> None:
    ticker = payload.get("ticker")
    run_id = payload.get("run_id")
    as_of_date = payload.get("as_of_date")
    include_synthesis = bool(payload.get("include_synthesis"))
    if ticker:
        ok = score_ticker(ticker, run_id=run_id)
        if ok:
            if include_synthesis and run_id and as_of_date:
                enqueue_job(
                    "run_synthesis_agent",
                    {"ticker": ticker, "run_id": run_id, "as_of_date": as_of_date},
                )
            enqueue_job("build_memo", {"ticker": ticker})
    else:
        score_and_rank()
    _update_stage_success("score_ticker")


def _handle_run_synthesis_agent(payload: dict[str, Any]) -> None:
    ticker = payload.get("ticker")
    run_id = payload.get("run_id")
    as_of_date = payload.get("as_of_date")
    if not ticker or not run_id or not as_of_date:
        raise ValueError("run_synthesis_agent requires ticker, run_id, and as_of_date")
    run_synthesis_for_ticker(ticker=ticker, as_of_date=as_of_date, run_id=run_id)
    _update_stage_success("run_synthesis_agent")


def _handle_build_memo(payload: dict[str, Any]) -> None:
    ticker = payload.get("ticker")
    if not ticker:
        raise ValueError("build_memo requires ticker")
    build_memo_for_ticker(ticker)
    enqueue_job("build_rankings", {})
    _update_stage_success("build_memo")


def _handle_build_rankings(payload: dict[str, Any]) -> None:
    _ = payload
    score_and_rank()
    _update_stage_success("build_rankings")


def _handle_build_weekly_digest(payload: dict[str, Any]) -> None:
    _ = payload
    build_weekly_digest()
    _update_stage_success("build_weekly_digest")


def _handle_run_nightly_research_cycle(payload: dict[str, Any]) -> None:
    from app.discovery.runner import run_discovery
    from app.research.cycle import run_research_cycle

    as_of_date = payload.get("as_of_date") or date.today().isoformat()
    top_k = int(payload.get("top_k") or get_config().nightly_research_cycle_top_k or 10)
    max_iterations = int(payload.get("max_iterations") or 1)
    discovery_summary = run_discovery(
        as_of_date=as_of_date,
        top_k=top_k,
        advance_top=top_k,
    )
    discovery_run_id = str(discovery_summary.get("run_id") or "").strip()
    if not discovery_run_id:
        raise TransientJobError("nightly discovery did not produce a run_id")

    run_research_cycle(
        discovery_run_id=discovery_run_id,
        as_of_date=as_of_date,
        max_iterations=max_iterations,
        top_k=top_k,
    )
    _update_stage_success("run_nightly_research_cycle")


JOB_HANDLERS: dict[str, Callable[[dict[str, Any]], None]] = {
    "check_edgar_updates": _handle_check_edgar_updates,
    "ingest_filing": _handle_ingest_filing,
    "parse_filing": _handle_parse_filing,
    "compute_fundamentals": _handle_compute_fundamentals,
    "run_valuation": _handle_run_valuation,
    "build_evidence_packet": _handle_build_evidence_packet,
    "run_research_agent": _handle_run_research_agent,
    "run_analyst_agent": _handle_run_analyst_agent,
    "run_red_team": _handle_run_red_team,
    "score_ticker": _handle_score_ticker,
    "run_synthesis_agent": _handle_run_synthesis_agent,
    "build_memo": _handle_build_memo,
    "build_rankings": _handle_build_rankings,
    "build_weekly_digest": _handle_build_weekly_digest,
    "run_nightly_research_cycle": _handle_run_nightly_research_cycle,
}


def process_next_job() -> bool:
    try:
        job = claim_next_job()
    except sqlite3.OperationalError as exc:
        if "database is locked" in str(exc).lower():
            logger.warning(
                "job_claim_locked",
                extra={"stage_name": "job", "stage_error": str(exc)},
            )
            return False
        raise
    if not job:
        return False

    job_id = job["id"]
    attempts = job["attempts"]
    max_attempts = job.get("max_attempts", 5)
    job_type = job["job_type"]
    payload = job["payload"]

    handler = JOB_HANDLERS.get(job_type)
    if handler is None:
        mark_job_failure(job_id, f"Unknown job type: {job_type}", attempts, transient=False, max_attempts=max_attempts)
        return True

    with _DB_WRITE_SEMAPHORE:
        try:
            with get_db() as conn:
                cancelled = conn.execute("SELECT cancelled_at FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if cancelled and cancelled["cancelled_at"] is not None:
                    raise CancelledJobError(f"Job {job_id} cancelled before execution")
            handler(payload)
            mark_job_success(job_id)
            logger.info("job_done", extra={"stage_name": "job", "stage_job_type": job_type, "stage_job_id": job_id})
        except CancelledJobError as exc:
            mark_job_failure(job_id, str(exc), attempts, transient=False, max_attempts=max_attempts, error_type="cancelled")
            logger.info(
                "job_cancelled",
                extra={"stage_name": "job", "stage_job_type": job_type, "stage_job_id": job_id, "stage_error": str(exc)},
            )
        except DomainBudgetExceeded as exc:
            mark_job_failure(
                job_id,
                str(exc),
                attempts,
                transient=False,
                max_attempts=max_attempts,
                error_type="budget_exceeded",
            )
            logger.error(
                "job_budget_exceeded",
                extra={"stage_name": "job", "stage_job_type": job_type, "stage_job_id": job_id, "stage_error": str(exc)},
            )
        except PermanentJobError as exc:
            mark_job_failure(job_id, str(exc), attempts, transient=False, max_attempts=max_attempts, error_type="permanent")
            logger.error(
                "job_permanent_failure",
                extra={"stage_name": "job", "stage_job_type": job_type, "stage_job_id": job_id, "stage_error": str(exc)},
            )
        except TransientJobError as exc:
            mark_job_failure(job_id, str(exc), attempts, transient=True, max_attempts=max_attempts, error_type="transient")
            logger.warning(
                "job_retry",
                extra={"stage_name": "job", "stage_job_type": job_type, "stage_job_id": job_id, "stage_error": str(exc)},
            )
        except sqlite3.OperationalError as exc:
            message = str(exc)
            if "database is locked" in message.lower():
                mark_job_failure(
                    job_id,
                    message,
                    attempts,
                    transient=True,
                    max_attempts=max_attempts,
                    error_type="sqlite_locked",
                )
                logger.warning(
                    "job_retry_sqlite_lock",
                    extra={"stage_name": "job", "stage_job_type": job_type, "stage_job_id": job_id, "stage_error": message},
                )
            else:
                mark_job_failure(
                    job_id,
                    message,
                    attempts,
                    transient=False,
                    max_attempts=max_attempts,
                    error_type="sqlite_operational",
                )
                logger.error(
                    "job_sqlite_operational_failure",
                    extra={"stage_name": "job", "stage_job_type": job_type, "stage_job_id": job_id, "stage_error": message},
                )
        except Exception as exc:  # noqa: BLE001
            mark_job_failure(job_id, str(exc), attempts, transient=False, max_attempts=max_attempts, error_type="unexpected")
            logger.error(
                "job_failed",
                extra={"stage_name": "job", "stage_job_type": job_type, "stage_job_id": job_id, "stage_error": str(exc)},
            )
    return True


def worker_loop(stop_event: threading.Event, idle_sleep_seconds: float = 1.0) -> None:
    while not stop_event.is_set():
        try:
            progressed = process_next_job()
            if not progressed:
                time.sleep(idle_sleep_seconds)
        except Exception as exc:  # noqa: BLE001
            logger.error("worker_loop_error", extra={"stage_name": "job", "stage_error": str(exc)})
            time.sleep(idle_sleep_seconds)
