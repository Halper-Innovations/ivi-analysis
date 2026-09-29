from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import get_db, utc_now_iso
from app.discovery.runner import load_discovery_candidates
from app.evidence.packet_builder import build_packet_for_ticker
from app.fundamentals.metrics import compute_fundamentals_for_ticker
from app.ingest.filings import ingest_with_policy
from app.llm.synthesis_agent import run_synthesis_for_ticker
from app.parse.filing_parser import parse_pending_filings
from app.report.memo_builder import build_memo_for_ticker
from app.research.engine import run_research_gap_closer
from app.score.ranker import score_ticker
from app.valuation.sanity_checks import run_valuation_for_ticker


def _default_cycle_run_id(discovery_run_id: str, as_of_date: str) -> str:
    clean = as_of_date.replace("-", "")
    return f"cycle_{discovery_run_id}_{clean}"


def _load_advance_tickers(discovery_run_id: str, top_k: int) -> list[dict[str, Any]]:
    candidates = load_discovery_candidates(discovery_run_id)
    advances = [c for c in candidates if str(c.get("stage") or "") == "ADVANCE_TO_DEEP"]
    advances = sorted(
        advances,
        key=lambda c: (
            -float(c.get("discovery_score") or 0.0),
            str(c.get("ticker") or ""),
        ),
    )
    return advances[: max(1, int(top_k))]


def _cycle_row(conn, *, ticker: str, run_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT summary_json
        FROM research_cycles
        WHERE ticker = ? AND run_id = ?
        LIMIT 1
        """,
        (ticker, run_id),
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["summary_json"] or "{}")
    except Exception:
        return {}


def _persist_cycle_summary(
    *,
    ticker: str,
    run_id: str,
    discovery_run_id: str,
    as_of_date: str,
    payload: dict[str, Any],
) -> None:
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO research_cycles(
                ticker, run_id, discovery_run_id, as_of_date, summary_json, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, run_id) DO UPDATE SET
                discovery_run_id=excluded.discovery_run_id,
                as_of_date=excluded.as_of_date,
                summary_json=excluded.summary_json,
                updated_at=excluded.updated_at
            """,
            (ticker, run_id, discovery_run_id, as_of_date, json.dumps(payload), now, now),
        )


def _evidence_count(ticker: str, run_id: str) -> int:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM evidence_items
            WHERE ticker = ? AND run_id = ?
            """,
            (ticker, run_id),
        ).fetchone()
    return int(row["n"] or 0)


def _latest_score(ticker: str, run_id: str) -> float | None:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT total_score
            FROM scores
            WHERE ticker = ? AND run_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (ticker, run_id),
        ).fetchone()
    if not row:
        return None
    value = row["total_score"]
    return float(value) if isinstance(value, (int, float)) else None


def _latest_research_quality(ticker: str, run_id: str) -> dict[str, Any]:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT packet_path
            FROM research_packets
            WHERE ticker = ? AND run_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (ticker, run_id),
        ).fetchone()
    if not row:
        return {}
    path = Path(row["packet_path"])
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    quality = payload.get("quality")
    return quality if isinstance(quality, dict) else {}


def _latest_gaps_payload(ticker: str, run_id: str) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.gaps_dir / f"{ticker}_{run_id}.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _choose_gap_actions(
    *,
    ticker: str,
    run_id: str,
    as_of_date: str,
    sources: set[str] | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    gaps_payload = _latest_gaps_payload(ticker, run_id)
    missing_metrics = [str(x) for x in (gaps_payload.get("missing_key_metrics") or [])]
    parse_warnings = [str(x) for x in (gaps_payload.get("parse_warnings") or [])]
    research_warnings = [str(x) for x in (gaps_payload.get("research_warnings") or [])]
    missing_sources = [str(x) for x in (gaps_payload.get("missing_research_sources") or [])]
    missing_price = bool(gaps_payload.get("missing_price"))

    if any(m.upper() == "SHARES_OUTSTANDING" for m in missing_metrics):
        actions.append(
            {
                "action_code": "A_SHARES_OUTSTANDING_COVER_PAGES",
                "action": "Pull prior 10-Q/10-K cover pages and extract shares outstanding trend.",
                "source_filters": ["edgar"],
            }
        )
    if any(m.upper() in {"FCF", "CFO"} for m in missing_metrics) or any("cash flow" in w.lower() for w in parse_warnings):
        actions.append(
            {
                "action_code": "A_CASH_FLOW_TABLE_REPARSE",
                "action": "Re-parse cash flow table and bridge CFO/capex extraction.",
                "source_filters": ["edgar"],
            }
        )
    if missing_price:
        actions.append(
            {
                "action_code": "A_MARKET_CAP_RECOVERY",
                "action": "Retry market cap inputs via shares extraction and price quote refresh.",
                "source_filters": ["edgar"],
            }
        )
    if "sec_exhibits" in [s.lower() for s in missing_sources] or any("earnings" in w.lower() for w in research_warnings):
        actions.append(
            {
                "action_code": "A_SEC_EXHIBITS_REFRESH",
                "action": "Pull latest 8-K exhibits for earnings release / presentation signals.",
                "source_filters": ["exhibits"],
            }
        )
    if any(s.lower() in {"company_news", "ir_press", "homepage"} for s in missing_sources):
        actions.append(
            {
                "action_code": "A_ALLOWLISTED_NEWS_REFRESH",
                "action": "Refresh allowlisted IR/news/homepage evidence adapters.",
                "source_filters": ["news", "homepage"],
            }
        )
    if not actions:
        actions.append(
            {
                "action_code": "A_GENERAL_RESEARCH_REFRESH",
                "action": "Run deterministic research refresh on current evidence scope.",
                "source_filters": ["news", "exhibits", "homepage"],
            }
        )

    allowed = {token.strip().lower() for token in (sources or set()) if token.strip()}
    if allowed:
        for action in actions:
            next_filters = set(action["source_filters"]).intersection(allowed) or set(allowed)
            action["source_filters"] = sorted(next_filters)

    unique: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    for action in actions:
        code = str(action["action_code"])
        if code in seen_codes:
            continue
        seen_codes.add(code)
        unique.append(action)
    return unique[:3]


def _synthesis_spend(run_id: str) -> float:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_estimate_usd), 0) AS spent FROM synthesis_packets WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return float(row["spent"] or 0.0)


def _baseline_deep_stage(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
) -> dict[str, Any]:
    ingest_with_policy(
        as_of_date=as_of_date,
        run_id=run_id,
        tickers=[ticker],
        limit=1,
    )
    parse_pending_filings(limit=25, tickers=[ticker])
    compute_fundamentals_for_ticker(ticker, as_of_date=as_of_date)
    run_valuation_for_ticker(ticker, run_as_of_date=as_of_date)
    build_packet_for_ticker(ticker, as_of_date=as_of_date, dossier_run_id=run_id)
    build_memo_for_ticker(ticker, memo_mode="triage", run_id=run_id, as_of_date=as_of_date)
    return {"baseline_completed": True}


def run_research_cycle(
    *,
    discovery_run_id: str,
    as_of_date: str,
    max_iterations: int = 2,
    top_k: int = 10,
    budget_usd: float | None = None,
    sources: set[str] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    deep_run_id = run_id or _default_cycle_run_id(discovery_run_id, as_of_date)
    target_budget = float(budget_usd) if budget_usd is not None else float(cfg.openai_budget_usd_per_run)
    target_budget = max(0.0, target_budget)
    candidates = _load_advance_tickers(discovery_run_id, top_k)
    tickers = [str(row.get("ticker") or "").upper() for row in candidates if str(row.get("ticker") or "").strip()]

    ticker_summaries: list[dict[str, Any]] = []
    completed = 0
    for ticker in tickers:
        with get_db() as conn:
            existing = _cycle_row(conn, ticker=ticker, run_id=deep_run_id)
        if existing and bool(existing.get("completed")):
            ticker_summaries.append(existing)
            completed += 1
            continue

        summary = existing or {
            "ticker": ticker,
            "run_id": deep_run_id,
            "discovery_run_id": discovery_run_id,
            "as_of_date": as_of_date,
            "iterations_run": 0,
            "gaps_before": None,
            "gaps_after": None,
            "evidence_items_added_count": 0,
            "score_delta": 0.0,
            "synthesis_updated": False,
            "actions_taken": [],
            "completed": False,
            "stop_reason": "",
            "updated_at": utc_now_iso(),
        }

        start_iter = int(summary.get("iterations_run") or 0)
        baseline_done = bool(summary.get("baseline_done"))
        if not baseline_done:
            _baseline_deep_stage(ticker=ticker, as_of_date=as_of_date, run_id=deep_run_id)
            baseline_done = True
            summary["baseline_done"] = True
            _persist_cycle_summary(
                ticker=ticker,
                run_id=deep_run_id,
                discovery_run_id=discovery_run_id,
                as_of_date=as_of_date,
                payload=summary,
            )

        # Ensure there is a baseline research packet/score for this run.
        run_research_gap_closer(
            as_of_date=as_of_date,
            run_id=deep_run_id,
            limit=1,
            source_filters=sources,
            tickers=[ticker],
        )
        score_ticker(ticker, run_id=deep_run_id)
        build_memo_for_ticker(ticker, memo_mode="triage", run_id=deep_run_id, as_of_date=as_of_date)

        base_score = _latest_score(ticker, deep_run_id)
        gap_before = _latest_research_quality(ticker, deep_run_id).get("gap_score")
        summary["gaps_before"] = gap_before

        stop_reason = ""
        synthesis_updated = bool(summary.get("synthesis_updated"))
        total_added = int(summary.get("evidence_items_added_count") or 0)
        for iteration in range(start_iter, max(1, int(max_iterations))):
            actions = _choose_gap_actions(ticker=ticker, run_id=deep_run_id, as_of_date=as_of_date, sources=sources)
            source_filters: set[str] = set()
            for action in actions:
                source_filters.update({str(x).lower() for x in action.get("source_filters", set())})

            before_count = _evidence_count(ticker, deep_run_id)
            before_gap = _latest_research_quality(ticker, deep_run_id).get("gap_score")

            run_research_gap_closer(
                as_of_date=as_of_date,
                run_id=deep_run_id,
                limit=1,
                source_filters=source_filters or sources,
                tickers=[ticker],
            )
            score_ticker(ticker, run_id=deep_run_id)
            build_memo_for_ticker(ticker, memo_mode="triage", run_id=deep_run_id, as_of_date=as_of_date)

            after_count = _evidence_count(ticker, deep_run_id)
            added = max(0, after_count - before_count)
            total_added += added
            after_gap = _latest_research_quality(ticker, deep_run_id).get("gap_score")

            summary["actions_taken"].append(
                {
                    "iteration": iteration + 1,
                    "actions": actions,
                    "source_filters": sorted(source_filters),
                    "evidence_items_added": added,
                    "gap_before": before_gap,
                    "gap_after": after_gap,
                }
            )

            # Re-run synthesis only if budget remains and provider is enabled.
            if target_budget > 0 and _synthesis_spend(deep_run_id) < target_budget:
                synth = run_synthesis_for_ticker(ticker=ticker, as_of_date=as_of_date, run_id=deep_run_id)
                if synth is not None:
                    synthesis_updated = True

            summary["iterations_run"] = iteration + 1
            summary["gaps_after"] = after_gap
            summary["evidence_items_added_count"] = total_added
            summary["synthesis_updated"] = synthesis_updated
            summary["completed"] = False
            summary["updated_at"] = utc_now_iso()
            _persist_cycle_summary(
                ticker=ticker,
                run_id=deep_run_id,
                discovery_run_id=discovery_run_id,
                as_of_date=as_of_date,
                payload=summary,
            )

            improvement = None
            if isinstance(before_gap, (int, float)) and isinstance(after_gap, (int, float)):
                improvement = float(before_gap) - float(after_gap)

            if added <= 0:
                stop_reason = "NO_NEW_EVIDENCE"
                break
            if improvement is not None and improvement < float(cfg.research_cycle_gap_improvement_threshold):
                stop_reason = "DIMINISHING_RETURNS"
                break
            if target_budget > 0 and _synthesis_spend(deep_run_id) >= target_budget:
                stop_reason = "SYNTHESIS_BUDGET_REACHED"
                break

        latest_score = _latest_score(ticker, deep_run_id)
        score_delta = 0.0
        if isinstance(base_score, (int, float)) and isinstance(latest_score, (int, float)):
            score_delta = round(float(latest_score) - float(base_score), 4)
        summary["score_delta"] = score_delta
        summary["stop_reason"] = stop_reason or "MAX_ITERATIONS_REACHED"
        summary["completed"] = True
        summary["updated_at"] = utc_now_iso()

        _persist_cycle_summary(
            ticker=ticker,
            run_id=deep_run_id,
            discovery_run_id=discovery_run_id,
            as_of_date=as_of_date,
            payload=summary,
        )

        ticker_summaries.append(summary)
        completed += 1

    return {
        "run_id": deep_run_id,
        "discovery_run_id": discovery_run_id,
        "as_of_date": as_of_date,
        "target_tickers": tickers,
        "processed_count": completed,
        "max_iterations": int(max_iterations),
        "budget_usd": target_budget,
        "summaries": ticker_summaries,
    }
