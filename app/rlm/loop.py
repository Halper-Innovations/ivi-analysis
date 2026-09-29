from __future__ import annotations

import json
import math
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.config import get_config
from app.db import get_db, utc_now_iso
from app.rlm.critic import (
    NO_PROGRESS_GAP_REDUCTION_THRESHOLD,
    apply_stop_rules,
    evaluate_progress,
)
from app.rlm.executor import execute_actions
from app.rlm.planner import generate_plan
from app.llm.usage_capture import (
    attach_provider_usage_to_exception,
    attached_provider_usage_records,
    provider_usage_budget,
    provider_usage_capture,
)
from app.rlm.state import (
    LoopState,
    get_max_iteration_for_run,
    get_rlm_run_row,
    init_loop_state,
    insert_rlm_iteration_row,
    load_loop_state,
    persist_iteration_artifacts,
    rlm_run_dir,
    save_loop_state,
    upsert_rlm_run_row,
)
from app.sector.cycle import run_sector_cycle
from app.sector.decision_pack import build_sector_decision_pack
from app.util.http import HttpClient, effective_sec_domain_budgets
from app.valuation.value_gates import normalize_gate_thresholds, write_value_gates_for_run


HEARTBEAT_INTERVAL_SECONDS = 5
STALE_HEARTBEAT_SECONDS = 10 * 60
TERMINAL_STATUSES = {"DONE", "STOPPED", "NEEDS_HUMAN", "CANCELLED"}
RLM_LOG_FILE = "rlm.log"
RLM_DELTA_METRICS = [
    "whale_signature_score",
    "revenue_cagr_5y",
    "revenue_cagr_10y",
    "dilution_rate_shares_cagr",
    "net_debt_latest",
    "risk_factor_keyword_delta",
    "valuation_gap",
    "intrinsic_per_share_base",
    "implied_return_base",
    "implied_fcf_growth",
    "quality_score",
    "growth_score",
    "capital_discipline_score",
    "valuation_score",
    "risk_penalty",
    "score_total",
]


class _RunRefusedError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_error(exc: Exception | str) -> str:
    return str(exc).replace("\n", " ").strip()[:400]


def _provider_usage_cost(records: list[dict[str, Any]]) -> float:
    total = 0.0
    for record in records:
        value = record.get("cost_estimate_usd")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("provider usage record has non-numeric cost")
        cost = float(value)
        if not math.isfinite(cost) or cost < 0.0:
            raise RuntimeError("provider usage record has invalid cost")
        total += cost
    return round(total, 6)


def _debit_llm_cost(*, state: LoopState, cost_usd: float) -> None:
    cost = float(cost_usd)
    if not math.isfinite(cost) or cost < 0.0:
        raise RuntimeError("RLM LLM cost debit must be finite and non-negative")
    state.llm_cost_used = round(float(state.llm_cost_used) + cost, 6)
    state.budgets_remaining.llm_budget_remaining = max(
        0.0,
        round(float(state.budgets_remaining.llm_budget_remaining) - cost, 6),
    )


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_gate_override_inputs(
    *,
    mos_min: float | None = None,
    valuation_gap_min: float | None = None,
    net_debt_to_cfo_max: float | None = None,
    dilution_max: float | None = None,
) -> dict[str, float]:
    raw = {
        "mos_min": mos_min,
        "valuation_gap_min": valuation_gap_min,
        "net_debt_to_cfo_max": net_debt_to_cfo_max,
        "dilution_max": dilution_max,
    }
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _normalize_seed_tickers(seed_tickers: list[str] | None) -> list[str]:
    if not seed_tickers:
        return []
    return sorted({str(ticker).upper().strip() for ticker in seed_tickers if str(ticker).strip()})


def _run_log_path(run_id: str) -> Path:
    return rlm_run_dir(run_id) / RLM_LOG_FILE


def _append_run_log(run_id: str, *, level: str, event: str, fields: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "ts": _utc_now(),
        "level": str(level).upper(),
        "event": str(event),
    }
    for key, value in (fields or {}).items():
        if key.lower().endswith("key") or key.lower().endswith("token"):
            payload[key] = "***REDACTED***"
        else:
            payload[key] = value
    path = _run_log_path(run_id)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _sec_domain_counts(metrics: dict[str, Any]) -> dict[str, int]:
    return {
        host: int(metrics.get(f"domain_count:{host}", 0))
        for host in ("data.sec.gov", "www.sec.gov", "sec.gov")
    }


def _sec_domain_usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        host: max(0, int(after.get(host, 0)) - int(before.get(host, 0)))
        for host in ("data.sec.gov", "www.sec.gov", "sec.gov")
    }


def _artifact_paths(run_id: str) -> dict[str, str | None]:
    cfg = get_config()
    sector_dir = cfg.sectors_dir / run_id
    dossier_dir = cfg.dossiers_dir / run_id

    def _pick(primary: Path, fallback: Path | None = None) -> str | None:
        if primary.exists():
            return str(primary)
        if fallback and fallback.exists():
            return str(fallback)
        return None

    return {
        "sector_summary_path": _pick(sector_dir / "sector_summary.json"),
        "peer_rankings_path": _pick(sector_dir / "peer_rankings.json", dossier_dir / "peer_rankings.json"),
        "peer_report_path": _pick(sector_dir / "peer_report.md", dossier_dir / "peer_report.md"),
        "peer_scoreboard_path": _pick(sector_dir / "peer_scoreboard.json", dossier_dir / "peer_scoreboard.json"),
        "fundamentals_summary_path": _pick(sector_dir / "fundamentals_summary.json"),
        "valuation_summary_path": _pick(sector_dir / "valuation_summary.json"),
        "price_coverage_path": _pick(sector_dir / "price_coverage.json"),
        "shares_coverage_path": _pick(sector_dir / "shares_coverage.json"),
        "fcf_coverage_path": _pick(sector_dir / "fcf_coverage.json"),
        "facts_coverage_path": _pick(sector_dir / "facts_coverage.json"),
        "valuation_coverage_path": _pick(sector_dir / "valuation_coverage.json"),
        "value_gates_path": _pick(sector_dir / "value_gates.json"),
        "value_gates_calibration_path": _pick(sector_dir / "value_gates_calibration.json"),
        "universe_scout_linkage_path": _pick(sector_dir / "universe_scout_linkage.json"),
        "whale_signals_summary_path": _pick(
            sector_dir / "whale_signals_summary.json",
            dossier_dir / "whale_signals_summary.json",
        ),
        "sector_synthesis_path": _pick(sector_dir / "sector_synthesis.json"),
        "decision_pack_path": _pick(sector_dir / "decision_pack.json"),
        "decision_pack_md_path": _pick(sector_dir / "decision_pack.md"),
        "dossier_summary_path": _pick(dossier_dir / "dossier_summary.json"),
        "run_log_path": _pick(sector_dir / RLM_LOG_FILE),
    }


def _top_k_from_artifacts(artifacts: dict[str, str | None], top_k: int) -> list[str]:
    rankings_path = artifacts.get("peer_rankings_path")
    if not rankings_path:
        return []
    payload = _safe_json(Path(rankings_path))
    value_rank = [str(t).upper() for t in (payload.get("value_first_rank") or []) if str(t).strip()]
    if value_rank:
        return value_rank[: max(1, int(top_k))]
    ranked = [str(t).upper() for t in (payload.get("future_whale_rank") or []) if str(t).strip()]
    return ranked[: max(1, int(top_k))]


def _gap_summary_from_scoreboard(path_value: str | None) -> dict[str, dict[str, Any]]:
    if not path_value:
        return {}
    payload = _safe_json(Path(path_value))
    rows = [row for row in (payload.get("rows") or []) if isinstance(row, dict)]
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            continue
        gaps = row.get("gaps") or (row.get("whale_summary") or {}).get("gaps") or []
        gap_entries = [gap for gap in gaps if isinstance(gap, dict)]
        missing_metrics: list[str] = []
        for gap in gap_entries:
            for metric in gap.get("missing_metrics") or []:
                name = str(metric).strip()
                if name:
                    missing_metrics.append(name)
        summary[ticker] = {
            "gap_count": len(gap_entries),
            "missing_metrics": sorted(set(missing_metrics)),
            "derived_from": [f"peer_scoreboard.rows[{ticker}].gaps"],
        }
    return summary


def _peer_set_from_summary(run_id: str) -> list[str]:
    cfg = get_config()
    summary_path = cfg.sectors_dir / run_id / "sector_summary.json"
    payload = _safe_json(summary_path)
    peers = [str(t).upper() for t in (payload.get("peer_tickers") or []) if str(t).strip()]
    if peers:
        return peers
    peer_payload = _safe_json(cfg.sectors_dir / run_id / "sector_peers.json")
    return [str(t).upper() for t in (peer_payload.get("selected_tickers") or []) if str(t).strip()]


def _sec_budget_remaining_estimate() -> int:
    budgets = effective_sec_domain_budgets()
    metrics = HttpClient().metrics()
    remaining = 0
    for host in ("data.sec.gov", "www.sec.gov"):
        budget = int(budgets.get(host, 0))
        used = int(metrics.get(f"domain_count:{host}", 0))
        remaining += max(0, budget - used)
    return max(0, int(remaining))


def _dossier_progress_counts(run_id: str) -> dict[str, int]:
    payload = _safe_json(get_config().dossiers_dir / run_id / "dossier_summary.json")
    tickers_requested = [str(t).upper() for t in (payload.get("tickers_requested") or []) if str(t).strip()]
    built = [str(t).upper() for t in (payload.get("tickers_built") or []) if str(t).strip()]
    failed = [str(t).upper() for t in (payload.get("tickers_failed") or []) if str(t).strip()]
    skipped = [str(t).upper() for t in (payload.get("tickers_skipped") or []) if str(t).strip()]
    skipped_budget = [str(t).upper() for t in (payload.get("tickers_skipped_budget") or []) if str(t).strip()]
    ticker_results = payload.get("ticker_results") if isinstance(payload.get("ticker_results"), dict) else {}
    for ticker, row in ticker_results.items():
        ticker_token = str(ticker or "").upper().strip()
        if not ticker_token:
            continue
        status = str((row or {}).get("status") or "").upper()
        dossier_json_path = str((row or {}).get("dossier_json_path") or "").strip()
        if status == "OK" or dossier_json_path:
            built.append(ticker_token)
    dossier_root = get_config().dossiers_dir / run_id
    if dossier_root.exists():
        for path in dossier_root.glob("*/dossier.json"):
            ticker_token = path.parent.name.upper().strip()
            if ticker_token:
                built.append(ticker_token)
    return {
        "dossier_requested": len(sorted(set(tickers_requested))),
        "dossier_ok": len(sorted(set(built))),
        "dossier_skipped_preflight": len(sorted(set(skipped))),
        "dossier_skipped_budget": len(sorted(set(skipped_budget))),
        "dossier_failed": len(sorted(set(failed))),
    }


def _research_synthesis_counts(run_id: str) -> dict[str, int]:
    try:
        with get_db() as conn:
            research_row = conn.execute(
                "SELECT COUNT(*) AS n FROM research_packets WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            synthesis_row = conn.execute(
                "SELECT COUNT(*) AS n FROM synthesis_packets WHERE run_id = ?",
                (run_id,),
            ).fetchone()
    except Exception:
        return {"research_packets": 0, "synthesis_packets": 0}
    research_n = int(research_row["n"]) if research_row and research_row["n"] is not None else 0
    synthesis_n = int(synthesis_row["n"]) if synthesis_row and synthesis_row["n"] is not None else 0
    return {
        "research_packets": research_n,
        "synthesis_packets": synthesis_n,
    }


def _progress_counts(*, run_id: str, state: LoopState | None = None) -> dict[str, int]:
    dossier_counts = _dossier_progress_counts(run_id)
    research_counts = _research_synthesis_counts(run_id)
    return {
        "peers_selected": len(state.peer_set) if state is not None else len(_peer_set_from_summary(run_id)),
        "dossier_requested": dossier_counts["dossier_requested"],
        "dossier_ok": dossier_counts["dossier_ok"],
        "dossier_skipped_preflight": dossier_counts["dossier_skipped_preflight"],
        "dossier_skipped_budget": dossier_counts["dossier_skipped_budget"],
        "dossier_failed": dossier_counts["dossier_failed"],
        "research_packets": research_counts["research_packets"],
        "synthesis_packets": research_counts["synthesis_packets"],
    }


def _scoreboard_rows_by_ticker(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in (payload.get("rows") or []):
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper()
        if ticker:
            out[ticker] = row
    return out


def _valuation_coverage_summary(payload: dict[str, Any]) -> tuple[int, dict[str, int]]:
    entries = [row for row in (payload.get("entries") or []) if isinstance(row, dict)]
    known = 0
    unknown_reason_counts: dict[str, int] = {}
    for row in entries:
        implied = row.get("implied_return_base", "UNKNOWN")
        if isinstance(implied, (int, float)):
            known += 1
            continue
        code = str(row.get("valuation_reason_code") or "MODEL_PRECONDITION_FAILED")
        unknown_reason_counts[code] = unknown_reason_counts.get(code, 0) + 1
    return int(known), dict(sorted(unknown_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _scoreboard_delta_payload(
    *,
    run_id: str,
    iteration: int,
    before_scoreboard: dict[str, Any],
    after_scoreboard: dict[str, Any],
    before_valuation_coverage: dict[str, Any],
    after_valuation_coverage: dict[str, Any],
    calibration_required_note: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before_rows = _scoreboard_rows_by_ticker(before_scoreboard)
    after_rows = _scoreboard_rows_by_ticker(after_scoreboard)
    all_tickers = sorted(set(before_rows.keys()).union(after_rows.keys()))
    rows: list[dict[str, Any]] = []
    metrics_changed = 0
    for ticker in all_tickers:
        before_metrics = (before_rows.get(ticker) or {}).get("metric_values") or {}
        after_metrics = (after_rows.get(ticker) or {}).get("metric_values") or {}
        metric_deltas: dict[str, dict[str, Any]] = {}
        ticker_changed = False
        for metric in RLM_DELTA_METRICS:
            prev = before_metrics.get(metric, "UNKNOWN")
            curr = after_metrics.get(metric, "UNKNOWN")
            delta: float | str = "UNKNOWN"
            if isinstance(prev, (int, float)) and isinstance(curr, (int, float)):
                delta = round(float(curr) - float(prev), 6)
            metric_deltas[metric] = {
                "previous": prev,
                "current": curr,
                "delta": delta,
                "derived_from": [
                    f"scoreboard_before.rows[{ticker}].metric_values.{metric}",
                    f"scoreboard_after.rows[{ticker}].metric_values.{metric}",
                ],
            }
            if isinstance(delta, (int, float)) and abs(float(delta)) > 0:
                ticker_changed = True
                metrics_changed += 1
        rows.append(
            {
                "ticker": ticker,
                "metric_deltas": metric_deltas,
                "changed": ticker_changed,
                "derived_from": [
                    f"scoreboard_before.rows[{ticker}]",
                    f"scoreboard_after.rows[{ticker}]",
                ],
            }
        )
    tickers_changed = len([row for row in rows if bool(row.get("changed"))])
    known_before, unknown_reasons_before = _valuation_coverage_summary(before_valuation_coverage)
    known_after, unknown_reasons_after = _valuation_coverage_summary(after_valuation_coverage)
    payload = {
        "run_id": run_id,
        "iteration": int(iteration),
        "metrics": list(RLM_DELTA_METRICS),
        "rows": rows,
        "metrics_changed": int(metrics_changed),
        "tickers_changed": int(tickers_changed),
        "coverage_delta": {
            "known_implied_return_count_before": int(known_before),
            "known_implied_return_count_after": int(known_after),
            "known_implied_return_count_delta": int(known_after - known_before),
            "unknown_reasons_before": unknown_reasons_before,
            "unknown_reasons_after": unknown_reasons_after,
        },
        "derived_from": [
            "sector.peer_scoreboard.before",
            "sector.peer_scoreboard.after",
            "sector.valuation_coverage.before",
            "sector.valuation_coverage.after",
        ],
    }
    if isinstance(calibration_required_note, dict):
        payload["calibration_required"] = calibration_required_note
    return payload


def _value_gates_rows_by_ticker(value_gates_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in (value_gates_payload.get("entries") or []):
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper().strip()
        if ticker:
            out[ticker] = row
    return out


def _calibration_required_note(*, state: LoopState, top_k: int) -> dict[str, Any] | None:
    if str(state.mode).lower() != "depth":
        return None
    value_gates_payload = _safe_json(Path(str(state.artifacts.get("value_gates_path") or "")))
    rows_by_ticker = _value_gates_rows_by_ticker(value_gates_payload)
    if not rows_by_ticker:
        return None
    top_scope = _top_k_from_artifacts(state.artifacts, top_k=top_k) or list(state.top_k_current)
    top_scope = [str(t).upper().strip() for t in top_scope if str(t).strip()][: max(1, int(top_k))]
    if not top_scope:
        return None

    pass_count = 0
    watch_count = 0
    fail_count = 0
    blocker_counts: dict[str, int] = {}
    for ticker in top_scope:
        row = rows_by_ticker.get(ticker)
        if not isinstance(row, dict):
            watch_count += 1
            continue
        status = str(row.get("gate_status") or "WATCH").upper()
        if status == "PASS":
            pass_count += 1
        elif status == "FAIL":
            fail_count += 1
            blocker = str(row.get("primary_blocker") or "UNKNOWN")
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        else:
            watch_count += 1

    if pass_count > 0 or watch_count > 0:
        return None

    calibration_payload = _safe_json(Path(str(state.artifacts.get("value_gates_calibration_path") or "")))
    missing_input_breakdown = (
        calibration_payload.get("missing_input_breakdown")
        if isinstance(calibration_payload.get("missing_input_breakdown"), dict)
        else {}
    )
    blockers = [
        {"primary_blocker": str(name), "count": int(count)}
        for name, count in sorted(blocker_counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:3]
    ]
    if not blockers:
        blocker_hist = (
            calibration_payload.get("blocker_histogram")
            if isinstance(calibration_payload.get("blocker_histogram"), dict)
            else {}
        )
        blockers = [
            {"primary_blocker": str(name), "count": int(count)}
            for name, count in sorted(blocker_hist.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:3]
        ]

    return {
        "note_code": "CALIBRATION_REQUIRED",
        "run_id": state.run_id,
        "top_k_scope": top_scope,
        "counts_top_k": {
            "PASS": int(pass_count),
            "WATCH": int(watch_count),
            "FAIL": int(fail_count),
        },
        "top_blockers": blockers,
        "missing_input_breakdown": missing_input_breakdown,
        "suggestion": f"Run `python -m app.cli value-gates-calibration-open --run-id {state.run_id}`.",
        "derived_from": [
            str(state.artifacts.get("value_gates_path") or ""),
            str(state.artifacts.get("value_gates_calibration_path") or ""),
        ],
    }


def _initialize_state(
    *,
    run_id: str,
    sector: str,
    as_of_date: str,
    iterations: int,
    budget_usd: float,
    top_k: int,
    mode: str,
    with_prices: bool,
    gate_threshold_overrides: dict[str, float] | None = None,
    gate_thresholds_effective: dict[str, float] | None = None,
) -> LoopState:
    sec_remaining = _sec_budget_remaining_estimate()
    state = init_loop_state(
        run_id=run_id,
        sector=sector,
        as_of_date=as_of_date,
        max_iterations=max(1, int(iterations)),
        llm_budget_usd=max(0.0, float(budget_usd)),
        sec_budget_count=max(0, int(sec_remaining)),
        gate_threshold_overrides=gate_threshold_overrides,
        gate_thresholds_effective=gate_thresholds_effective,
    )
    state.mode = str(mode).lower()
    state.rlm_version = "v1.1" if str(mode).lower() == "depth" else "v0"
    state.with_prices = bool(with_prices)
    state.artifacts = _artifact_paths(run_id)
    state.peer_set = sorted(set(_peer_set_from_summary(run_id)))
    state.top_k_current = _top_k_from_artifacts(state.artifacts, top_k=top_k)
    state.gap_summary = _gap_summary_from_scoreboard(state.artifacts.get("peer_scoreboard_path"))
    state.evidence_delta_counters = {
        "evidence_count_topk": 0,
        "gap_count_topk": 0,
    }
    return state


def _reconstruct_state_from_db(
    *,
    run_id: str,
    sector: str,
    as_of_date: str,
    iterations: int,
    budget_usd: float,
    top_k: int,
    mode: str,
    with_prices: bool,
    gate_threshold_overrides: dict[str, float] | None = None,
    gate_thresholds_effective: dict[str, float] | None = None,
) -> LoopState:
    row = get_rlm_run_row(run_id)
    if not row:
        return _initialize_state(
            run_id=run_id,
            sector=sector,
            as_of_date=as_of_date,
            iterations=iterations,
            budget_usd=budget_usd,
            top_k=top_k,
            mode=mode,
            with_prices=with_prices,
        )

    state = init_loop_state(
        run_id=run_id,
        sector=sector,
        as_of_date=as_of_date,
        max_iterations=max(1, int(iterations)),
        llm_budget_usd=max(0.0, float(budget_usd)),
        sec_budget_count=max(0, int(_sec_budget_remaining_estimate())),
        gate_threshold_overrides=gate_threshold_overrides,
        gate_thresholds_effective=gate_thresholds_effective,
    )
    state.mode = str(mode).lower()
    state.rlm_version = "v1.1" if str(mode).lower() == "depth" else "v0"
    state.with_prices = bool(with_prices)
    state.iteration = max(0, int(row.get("iterations") or 0))
    state.status = str(row.get("status") or "RUNNING").upper()
    state.created_at = str(row.get("created_at") or state.created_at)
    state.updated_at = str(row.get("updated_at") or state.updated_at)
    stop_reason = str(row.get("stop_reason") or "").strip()
    if stop_reason:
        state.stop_reason_code = stop_reason.split(":", 1)[0].strip().upper() or None
        state.stop_summary = stop_reason

    state.artifacts = _artifact_paths(run_id)
    state.peer_set = sorted(set(_peer_set_from_summary(run_id)))
    state.top_k_current = _top_k_from_artifacts(state.artifacts, top_k=top_k)
    state.gap_summary = _gap_summary_from_scoreboard(state.artifacts.get("peer_scoreboard_path"))
    state.evidence_delta_counters = {
        "evidence_count_topk": 0,
        "gap_count_topk": 0,
    }
    return state


def _ensure_sector_baseline(
    *,
    run_id: str,
    sector: str,
    as_of_date: str,
    peer_limit: int,
    min_peers: int,
    limit_dossiers: int,
    years_back: int,
    min_annual_filings: int,
    workers: int,
    with_research: bool,
    with_synthesis: bool,
) -> dict[str, Any]:
    cfg = get_config()
    run_dir = cfg.sectors_dir / run_id
    summary_path = cfg.sectors_dir / run_id / "sector_summary.json"
    if summary_path.exists():
        summary = _safe_json(summary_path)
        peer_rankings = run_dir / "peer_rankings.json"
        peer_scoreboard = run_dir / "peer_scoreboard.json"
        if peer_rankings.exists() and peer_scoreboard.exists():
            return summary

    return run_sector_cycle(
        sector=sector,
        as_of_date=as_of_date,
        peer_limit=max(1, int(peer_limit)),
        min_peers_dossierable=max(1, int(min_peers)),
        limit_dossiers=max(1, int(limit_dossiers)),
        years_back=max(1, int(years_back)),
        min_annual_filings=max(1, int(min_annual_filings)),
        workers=max(1, int(workers)),
        with_research=bool(with_research),
        with_synthesis=bool(with_synthesis),
        run_id=run_id,
        peer_mode="hybrid",
    )


def _planner_stop_reason(planner_json: dict[str, Any]) -> tuple[str, str] | None:
    actions = planner_json.get("actions") or []
    for action in actions:
        if not isinstance(action, dict):
            continue
        if str(action.get("action_type") or "").upper() not in {"STOP", "ACTION_STOP"}:
            continue
        reason = str(action.get("reason_code") or "PLANNER_STOP")
        summary = str(action.get("summary") or "Planner requested stop.")
        return reason, summary
    return None


def _heartbeat_path(run_id: str) -> Path:
    return rlm_run_dir(run_id) / "rlm_heartbeat.json"


def _read_heartbeat(run_id: str) -> dict[str, Any]:
    path = _heartbeat_path(run_id)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_pid_active(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _heartbeat_age_seconds(payload: dict[str, Any]) -> float | None:
    updated_at = payload.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at.strip():
        return None
    try:
        ts = datetime.fromisoformat(updated_at)
    except Exception:
        return None
    return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())


def _run_appears_active(run_id: str) -> bool:
    heartbeat = _read_heartbeat(run_id)
    pid = heartbeat.get("pid")
    return _is_pid_active(int(pid)) if isinstance(pid, int) else False


def _cancel_requested(run_id: str) -> tuple[bool, str | None]:
    row = get_rlm_run_row(run_id)
    if not row:
        return False, None
    status = str(row.get("status") or "").upper()
    if status != "CANCELLED":
        return False, None
    reason = str(row.get("stop_reason") or "").strip() or "Cancelled by operator"
    return True, reason


def _write_heartbeat(runtime: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        payload = dict(runtime)
    payload["updated_at"] = _utc_now()
    path = _heartbeat_path(str(payload.get("run_id") or ""))
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _heartbeat_pump(runtime: dict[str, Any], lock: threading.Lock, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        _write_heartbeat(runtime, lock)
        stop_event.wait(HEARTBEAT_INTERVAL_SECONDS)


def _write_final_decision_pack(*, state: LoopState, top_k: int) -> dict[str, str] | None:
    try:
        decision = build_sector_decision_pack(
            sector=state.sector,
            as_of_date=state.as_of_date,
            run_id=state.run_id,
            top_n=max(1, int(top_k)),
        )
    except Exception:
        return None

    base_json_path = Path(str(decision.get("decision_pack_path") or ""))
    base_md_path = Path(str(decision.get("decision_pack_md_path") or ""))
    if not base_json_path.exists() or not base_md_path.exists():
        return None

    base_payload = _safe_json(base_json_path)
    trace_summary = {
        "run_id": state.run_id,
        "sector": state.sector,
        "as_of_date": state.as_of_date,
        "status": state.status,
        "iterations_completed": int(state.iteration),
        "stop_reason_code": state.stop_reason_code,
        "stop_summary": state.stop_summary,
        "no_progress_streak": int(state.no_progress_streak),
        "ranking_stable_streak": int(state.ranking_stable_streak),
        "llm_cost_used": round(float(state.llm_cost_used), 6),
        "llm_budget_remaining": round(float(state.budgets_remaining.llm_budget_remaining), 6),
        "sec_budget_remaining": int(state.budgets_remaining.sec_budget_remaining),
        "actions_executed": len(state.action_history),
        "sec_requests_used_total": int(state.sec_requests_used),
        "derived_from": [
            "rlm_state.action_history",
            "rlm_state.progress_history",
            "rlm_state.budgets_remaining",
            "sector.decision_pack",
        ],
    }
    final_payload = {
        "run_id": state.run_id,
        "sector": state.sector,
        "as_of_date": state.as_of_date,
        "created_at": _utc_now(),
        "decision_pack": base_payload,
        "rlm_trace_summary": trace_summary,
    }

    out_dir = rlm_run_dir(state.run_id)
    final_json = out_dir / "final_decision_pack.json"
    final_md = out_dir / "final_decision_pack.md"
    final_json.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")

    md_text = base_md_path.read_text(encoding="utf-8")
    extra_lines = [
        "",
        "## RLM Trace Summary",
        f"- Status: `{state.status}`",
        f"- Iterations completed: `{int(state.iteration)}`",
        f"- Stop reason: `{state.stop_reason_code}`",
        f"- Stop summary: {state.stop_summary or 'N/A'}",
        f"- LLM budget remaining: `{float(state.budgets_remaining.llm_budget_remaining):.6f}`",
        f"- SEC budget remaining (estimate): `{int(state.budgets_remaining.sec_budget_remaining)}`",
    ]
    final_md.write_text(md_text.rstrip() + "\n" + "\n".join(extra_lines) + "\n", encoding="utf-8")

    return {
        "final_decision_pack_path": str(final_json),
        "final_decision_pack_md_path": str(final_md),
        "base_decision_pack_path": str(base_json_path),
        "base_decision_pack_md_path": str(base_md_path),
    }


def force_restart_sector_rlm_run(*, run_id: str) -> dict[str, Any]:
    cfg = get_config()
    run_dir = cfg.sectors_dir / run_id
    removed_run_dir = False
    if run_dir.exists():
        shutil.rmtree(run_dir)
        removed_run_dir = True

    with get_db() as conn:
        deleted_iterations = conn.execute("DELETE FROM rlm_iterations WHERE run_id = ?", (run_id,)).rowcount
        deleted_runs = conn.execute("DELETE FROM rlm_runs WHERE run_id = ?", (run_id,)).rowcount

    return {
        "run_id": run_id,
        "deleted_rlm_iterations": int(deleted_iterations or 0),
        "deleted_rlm_runs": int(deleted_runs or 0),
        "removed_run_dir": bool(removed_run_dir),
        "run_dir": str(run_dir),
    }


def cancel_sector_rlm_run(*, run_id: str, reason: str) -> dict[str, Any]:
    reason_clean = reason.strip() or "Cancelled by operator"
    row = get_rlm_run_row(run_id)
    state = load_loop_state(run_id)

    if row is None and state is None:
        return {
            "run_id": run_id,
            "status": "MISSING",
            "message": "RLM run not found",
        }

    stop_reason = f"CANCELLED: {reason_clean}"
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            UPDATE rlm_runs
            SET status = 'CANCELLED', stop_reason = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (stop_reason, now, run_id),
        )

    if state is not None:
        state.status = "CANCELLED"
        state.stop_reason_code = "CANCELLED"
        state.stop_summary = reason_clean
        save_loop_state(state)
        upsert_rlm_run_row(state)

    heartbeat = _read_heartbeat(run_id)
    heartbeat.update(
        {
            "run_id": run_id,
            "phase": "finalize",
            "status": "CANCELLED",
            "last_action": "CANCEL",
            "last_progress_summary": reason_clean,
            "updated_at": _utc_now(),
        }
    )
    _heartbeat_path(run_id).write_text(json.dumps(heartbeat, indent=2), encoding="utf-8")
    _append_run_log(run_id, level="WARN", event="cancel_requested", fields={"reason": reason_clean})

    return {
        "run_id": run_id,
        "status": "CANCELLED",
        "reason": reason_clean,
        "heartbeat_path": str(_heartbeat_path(run_id)),
    }


def run_sector_rlm_loop(
    *,
    sector: str,
    as_of_date: str,
    run_id: str,
    peer_limit: int,
    min_peers: int,
    limit_dossiers: int,
    years_back: int,
    workers: int,
    iterations: int,
    top_k: int,
    with_research: bool,
    with_synthesis: bool,
    with_prices: bool = True,
    budget_usd: float | None = None,
    resume: bool = False,
    mode: str = "auto",
    force_restart: bool = False,
    seed_tickers: list[str] | None = None,
    parent_run_id: str | None = None,
    shortlist_source: str | None = None,
    gate_mos_min: float | None = None,
    gate_valuation_gap_min: float | None = None,
    gate_net_debt_to_cfo_max: float | None = None,
    gate_dilution_max: float | None = None,
    timeout_per_stage: float = 30.0,
) -> dict[str, Any]:
    cfg = get_config()
    llm_budget = float(budget_usd if budget_usd is not None else cfg.openai_budget_usd_per_run)
    mode_norm = str(mode).lower()
    seed_tickers_norm = _normalize_seed_tickers(seed_tickers)
    gate_threshold_overrides = _normalize_gate_override_inputs(
        mos_min=gate_mos_min,
        valuation_gap_min=gate_valuation_gap_min,
        net_debt_to_cfo_max=gate_net_debt_to_cfo_max,
        dilution_max=gate_dilution_max,
    )
    gate_thresholds_effective = normalize_gate_thresholds(gate_threshold_overrides)

    state: LoopState | None = None
    result: dict[str, Any] | None = None
    heartbeat_thread: threading.Thread | None = None
    heartbeat_stop_event = threading.Event()
    heartbeat_lock = threading.Lock()
    heartbeat_runtime: dict[str, Any] = {}

    def _update_heartbeat(
        *,
        phase: str,
        iteration_value: int | None = None,
        current_ticker: str | None = None,
        last_action: str | None = None,
        last_progress_summary: str | None = None,
        force_write: bool = False,
    ) -> None:
        nonlocal heartbeat_runtime
        if state is None:
            return
        progress = _progress_counts(run_id=state.run_id, state=state)
        with heartbeat_lock:
            heartbeat_runtime.update(
                {
                    "run_id": state.run_id,
                    "pid": os.getpid(),
                    "status": state.status,
                    "phase": phase,
                    "iteration": int(iteration_value if iteration_value is not None else state.iteration),
                    "current_ticker": current_ticker,
                    "last_action": last_action,
                    "last_progress_summary": last_progress_summary,
                    "top_k_current": list(state.top_k_current),
                    "peer_count": len(state.peer_set),
                    "dossier_ok_count": int(progress["dossier_ok"]),
                    "dossier_skipped_count": int(progress["dossier_skipped_preflight"]) + int(progress["dossier_skipped_budget"]),
                    "dossier_failed_count": int(progress["dossier_failed"]),
                    "sec_budget_remaining": int(state.budgets_remaining.sec_budget_remaining),
                    "llm_budget_remaining": float(state.budgets_remaining.llm_budget_remaining),
                    "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
                }
            )
        if force_write:
            _write_heartbeat(heartbeat_runtime, heartbeat_lock)

    def _persist_state() -> None:
        if state is None:
            return
        save_loop_state(state)
        upsert_rlm_run_row(state)

    def _refresh_value_gates(*, phase: str, iteration_value: int) -> dict[str, Any]:
        if state is None or str(state.mode).lower() != "depth":
            return {}
        run_dir = cfg.sectors_dir / state.run_id
        try:
            summary = write_value_gates_for_run(
                run_id=state.run_id,
                output_dir=run_dir,
                tickers=_top_k_from_artifacts(state.artifacts, top_k=top_k) or state.top_k_current or state.peer_set,
                threshold_overrides=state.gate_thresholds_effective,
            )
        except Exception as exc:  # noqa: BLE001
            _append_run_log(
                run_id,
                level="WARN",
                event="value_gates_refresh_failed",
                fields={"iteration": int(iteration_value), "phase": phase, "error": _sanitize_error(exc)},
            )
            return {}
        path_value = summary.get("value_gates_path")
        if isinstance(path_value, str) and path_value:
            state.artifacts["value_gates_path"] = path_value
        calibration_path = summary.get("value_gates_calibration_path")
        if isinstance(calibration_path, str) and calibration_path:
            state.artifacts["value_gates_calibration_path"] = calibration_path
        counts = (((summary.get("summary") or {}).get("counts")) if isinstance(summary.get("summary"), dict) else {})
        _append_run_log(
            run_id,
            level="INFO",
            event="value_gates_refreshed",
            fields={
                "iteration": int(iteration_value),
                "phase": phase,
                "pass_count": int((counts or {}).get("PASS", 0)) if isinstance(counts, dict) else 0,
                "watch_count": int((counts or {}).get("WATCH", 0)) if isinstance(counts, dict) else 0,
                "fail_count": int((counts or {}).get("FAIL", 0)) if isinstance(counts, dict) else 0,
            },
        )
        return summary if isinstance(summary, dict) else {}

    try:
        if force_restart:
            force_restart_sector_rlm_run(run_id=run_id)
            _append_run_log(run_id, level="INFO", event="force_restart", fields={"run_id": run_id})

        existing_state = load_loop_state(run_id)
        run_row = get_rlm_run_row(run_id)
        existing_status = (
            str(existing_state.status).upper()
            if existing_state is not None
            else str((run_row or {}).get("status") or "").upper()
        )

        if run_row is not None and existing_status in TERMINAL_STATUSES and not force_restart:
            raise _RunRefusedError(
                f"Run {run_id} is already {existing_status}. Use `sector-rlm-resume --force-restart` to restart."
            )
        if run_row is not None and existing_status == "RUNNING" and _run_appears_active(run_id) and not resume:
            raise _RunRefusedError(
                f"Run {run_id} is RUNNING with an active PID. Use `sector-rlm-status --run-id {run_id}` first."
            )

        if existing_state is not None:
            state = existing_state
            state.status = "RUNNING"
            state.artifacts.update(_artifact_paths(run_id))
            _append_run_log(run_id, level="INFO", event="resume_from_state", fields={"iteration": int(state.iteration)})
        elif run_row is not None:
            state = _reconstruct_state_from_db(
                run_id=run_id,
                sector=str(run_row.get("sector") or sector),
                as_of_date=str(run_row.get("as_of_date") or as_of_date),
                iterations=max(1, int(iterations)),
                budget_usd=llm_budget,
                top_k=top_k,
                mode=mode_norm,
                with_prices=with_prices,
                gate_threshold_overrides=gate_threshold_overrides,
                gate_thresholds_effective=gate_thresholds_effective,
            )
            state.status = "RUNNING"
            _append_run_log(run_id, level="INFO", event="resume_from_db", fields={"iteration": int(state.iteration)})
        else:
            if mode_norm != "depth":
                _ensure_sector_baseline(
                    run_id=run_id,
                    sector=sector,
                    as_of_date=as_of_date,
                    peer_limit=peer_limit,
                    min_peers=min_peers,
                    limit_dossiers=limit_dossiers,
                    years_back=years_back,
                    min_annual_filings=2,
                    workers=workers,
                    with_research=with_research,
                    with_synthesis=with_synthesis,
                )
            state = _initialize_state(
                run_id=run_id,
                sector=sector,
                as_of_date=as_of_date,
                iterations=iterations,
                budget_usd=llm_budget,
                top_k=top_k,
                mode=mode_norm,
                with_prices=with_prices,
                gate_threshold_overrides=gate_threshold_overrides,
                gate_thresholds_effective=gate_thresholds_effective,
            )
            _append_run_log(run_id, level="INFO", event="run_initialized", fields={"iteration": int(state.iteration)})

        state.mode = mode_norm
        state.rlm_version = "v1.1" if mode_norm == "depth" else "v0"
        state.with_prices = bool(with_prices)
        state.budgets_remaining.max_iterations = max(1, int(iterations))
        if budget_usd is not None:
            state.budgets_remaining.llm_budget_remaining = max(0.0, float(budget_usd) - float(state.llm_cost_used))
        if gate_threshold_overrides:
            state.gate_threshold_overrides = dict(gate_threshold_overrides)
            state.gate_thresholds_effective = dict(gate_thresholds_effective)
        elif state.gate_thresholds_effective:
            state.gate_thresholds_effective = normalize_gate_thresholds(state.gate_thresholds_effective)
        else:
            state.gate_thresholds_effective = normalize_gate_thresholds(state.gate_threshold_overrides)

        state.artifacts.update(_artifact_paths(run_id))
        if seed_tickers_norm:
            state.peer_set = list(seed_tickers_norm)
            state.top_k_current = list(seed_tickers_norm[: max(1, int(top_k))])
        if not state.peer_set:
            state.peer_set = sorted(set(_peer_set_from_summary(run_id)))
        if not state.top_k_current:
            state.top_k_current = _top_k_from_artifacts(state.artifacts, top_k=top_k)
        state.gap_summary = _gap_summary_from_scoreboard(state.artifacts.get("peer_scoreboard_path"))
        state.artifacts["run_log_path"] = str(_run_log_path(run_id))
        if parent_run_id or shortlist_source or seed_tickers_norm:
            linkage_path = rlm_run_dir(run_id) / "universe_scout_linkage.json"
            linkage_payload = {
                "run_id": run_id,
                "parent_run_id": str(parent_run_id or ""),
                "shortlist_source": str(shortlist_source or ""),
                "seed_tickers": list(seed_tickers_norm),
                "generated_at": _utc_now(),
                "derived_from": [str(shortlist_source or "")] if str(shortlist_source or "").strip() else [],
            }
            linkage_path.write_text(json.dumps(linkage_payload, indent=2), encoding="utf-8")
            state.artifacts["universe_scout_linkage_path"] = str(linkage_path)

        persisted_max = get_max_iteration_for_run(run_id)
        if persisted_max >= int(state.iteration):
            state.iteration = int(persisted_max + 1)

        _persist_state()

        heartbeat_runtime = {
            "run_id": state.run_id,
            "pid": os.getpid(),
            "status": state.status,
            "phase": "planner",
            "iteration": int(state.iteration),
            "current_ticker": None,
            "last_action": "START",
            "last_progress_summary": "Loop initialized",
            "top_k_current": list(state.top_k_current),
            "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
            "run_log_path": str(_run_log_path(state.run_id)),
        }
        _write_heartbeat(heartbeat_runtime, heartbeat_lock)
        heartbeat_thread = threading.Thread(
            target=_heartbeat_pump,
            args=(heartbeat_runtime, heartbeat_lock, heartbeat_stop_event),
            daemon=True,
            name=f"rlm-heartbeat-{run_id}",
        )
        heartbeat_thread.start()

        _append_run_log(
            run_id,
            level="INFO",
            event="loop_start",
            fields={
                "sector": state.sector,
                "as_of_date": state.as_of_date,
                "iterations": int(state.budgets_remaining.max_iterations),
                "top_k": int(top_k),
                "with_research": bool(with_research),
                "with_synthesis": bool(with_synthesis),
                "with_prices": bool(with_prices),
                "mode": state.mode,
                "seed_ticker_count": len(seed_tickers_norm),
                "parent_run_id": str(parent_run_id or ""),
                "shortlist_source": str(shortlist_source or ""),
                "gate_threshold_overrides": state.gate_threshold_overrides,
                "gate_thresholds_effective": state.gate_thresholds_effective,
            },
        )

        max_iterations = int(state.budgets_remaining.max_iterations)
        while int(state.iteration) < max_iterations:
            cancel_now, cancel_reason = _cancel_requested(run_id)
            if cancel_now:
                state.status = "CANCELLED"
                state.stop_reason_code = "CANCELLED"
                state.stop_summary = cancel_reason or "Cancelled by operator"
                _append_run_log(run_id, level="WARN", event="cancelled", fields={"reason": state.stop_summary})
                _update_heartbeat(
                    phase="finalize",
                    iteration_value=int(state.iteration),
                    last_action="CANCEL",
                    last_progress_summary=state.stop_summary,
                    force_write=True,
                )
                break

            if float(state.budgets_remaining.llm_budget_remaining) <= 0.0:
                state.status = "STOPPED"
                state.stop_reason_code = "LLM_BUDGET_EXHAUSTED"
                state.stop_summary = "LLM budget reached zero before planner execution."
                _append_run_log(run_id, level="WARN", event="stop_budget", fields={"type": "llm"})
                break
            if int(state.budgets_remaining.sec_budget_remaining) <= 0:
                state.status = "STOPPED"
                state.stop_reason_code = "SEC_BUDGET_EXHAUSTED"
                state.stop_summary = "SEC budget reached zero before planner execution."
                _append_run_log(run_id, level="WARN", event="stop_budget", fields={"type": "sec"})
                break

            iter_idx = int(state.iteration)
            _append_run_log(
                run_id,
                level="INFO",
                event="iteration_start",
                fields={
                    "iteration": iter_idx,
                    "peer_count": len(state.peer_set),
                    "sec_budget_remaining": int(state.budgets_remaining.sec_budget_remaining),
                    "llm_budget_remaining": float(state.budgets_remaining.llm_budget_remaining),
                },
            )
            _refresh_value_gates(phase="pre_planner", iteration_value=iter_idx)
            metrics_before = _sec_domain_counts(HttpClient().metrics())
            scoreboard_before_path = state.artifacts.get("peer_scoreboard_path")
            scoreboard_before = _safe_json(Path(str(scoreboard_before_path))) if scoreboard_before_path else {}
            valuation_coverage_before_path = state.artifacts.get("valuation_coverage_path")
            valuation_coverage_before = (
                _safe_json(Path(str(valuation_coverage_before_path)))
                if valuation_coverage_before_path
                else {}
            )

            _update_heartbeat(
                phase="planner",
                iteration_value=iter_idx,
                last_action="PLANNER",
                last_progress_summary="Generating plan",
            )
            # Provider calls already carry bounded physical/overall timeouts.
            # A shorter daemon-thread wrapper cannot cancel them; it would
            # orphan paid work, lose later integrity exceptions/cost, and let
            # a zero-cost fallback publish while the provider is still live.
            with (
                provider_usage_capture("rlm_planner") as planner_usage,
                provider_usage_budget(float(state.budgets_remaining.llm_budget_remaining)),
            ):
                try:
                    planner_output, planner_meta = generate_plan(
                        state=state,
                        top_k=top_k,
                        mode=state.mode,
                    )
                except BaseException as exc:
                    attach_provider_usage_to_exception(exc, planner_usage)
                    failed_usage = attached_provider_usage_records(exc)
                    _debit_llm_cost(
                        state=state,
                        cost_usd=_provider_usage_cost(failed_usage),
                    )
                    raise
            if planner_usage:
                llm_cost = _provider_usage_cost(planner_usage)
            else:
                # Preserve compatibility for deterministic planner doubles and
                # disabled-provider metadata that do not emit physical usage.
                llm_cost = float(planner_meta.get("cost_estimate_usd") or 0.0)
            _debit_llm_cost(state=state, cost_usd=llm_cost)
            planner_meta["provider_usage"] = list(planner_usage)
            planner_meta["cost_estimate_usd"] = llm_cost
            _append_run_log(
                run_id,
                level="INFO",
                event="planner_complete",
                fields={
                    "iteration": iter_idx,
                    "provider": planner_meta.get("provider"),
                    "model": planner_meta.get("model"),
                    "cost_estimate_usd": llm_cost,
                    "actions_count": len(planner_output.actions),
                },
            )

            def _executor_progress(payload: dict[str, Any]) -> None:
                ticker = str(payload.get("current_ticker") or "").upper() or None
                action_type = str(payload.get("action_type") or "EXECUTOR")
                _update_heartbeat(
                    phase="executor",
                    iteration_value=iter_idx,
                    current_ticker=ticker,
                    last_action=action_type,
                    last_progress_summary=f"Executing {action_type}",
                )

            _update_heartbeat(
                phase="executor",
                iteration_value=iter_idx,
                last_action="EXECUTOR",
                last_progress_summary="Executing plan actions",
            )
            with (
                provider_usage_capture("rlm_executor") as executor_usage,
                provider_usage_budget(float(state.budgets_remaining.llm_budget_remaining)),
            ):
                try:
                    actions_result = execute_actions(
                        state=state,
                        planner_output=planner_output,
                        top_k=top_k,
                        years_back_default=years_back,
                        workers=workers,
                        with_research=with_research,
                        with_synthesis=with_synthesis,
                        with_prices=with_prices,
                        mode=state.mode,
                        timeout_per_stage=timeout_per_stage,
                        progress_hook=_executor_progress,
                    )
                except BaseException as exc:
                    attach_provider_usage_to_exception(exc, executor_usage)
                    failed_usage = attached_provider_usage_records(exc)
                    _debit_llm_cost(
                        state=state,
                        cost_usd=_provider_usage_cost(failed_usage),
                    )
                    raise
            executor_cost = _provider_usage_cost(executor_usage)
            _debit_llm_cost(state=state, cost_usd=executor_cost)
            actions_result["provider_usage"] = list(executor_usage)
            actions_result["cost_estimate_usd"] = executor_cost
            for action_result in actions_result.get("results") or []:
                if str(action_result.get("status") or "").upper() != "STAGE_TIMEOUT":
                    continue
                _append_run_log(
                    run_id,
                    level="WARN",
                    event="STAGE_TIMEOUT",
                    fields={
                        "iteration": iter_idx,
                        "stage": str(action_result.get("effective_action_type") or action_result.get("action_type") or "UNKNOWN"),
                        "action_index": action_result.get("index"),
                        "details": action_result.get("details"),
                    },
                )
            _append_run_log(
                run_id,
                level="INFO",
                event="executor_complete",
                fields={
                    "iteration": iter_idx,
                    "executed_count": actions_result.get("executed_count"),
                    "planner_requested_stop": bool(actions_result.get("planner_requested_stop")),
                    "cost_estimate_usd": executor_cost,
                },
            )

            cancel_now, cancel_reason = _cancel_requested(run_id)
            if cancel_now:
                state.status = "CANCELLED"
                state.stop_reason_code = "CANCELLED"
                state.stop_summary = cancel_reason or "Cancelled by operator"
                _append_run_log(run_id, level="WARN", event="cancelled", fields={"reason": state.stop_summary})
                _update_heartbeat(
                    phase="finalize",
                    iteration_value=iter_idx,
                    last_action="CANCEL",
                    last_progress_summary=state.stop_summary,
                    force_write=True,
                )
                break

            state.artifacts.update(_artifact_paths(run_id))
            _refresh_value_gates(phase="post_executor", iteration_value=iter_idx)
            state.budgets_remaining.sec_budget_remaining = _sec_budget_remaining_estimate()
            state.gap_summary = _gap_summary_from_scoreboard(state.artifacts.get("peer_scoreboard_path"))
            calibration_note = _calibration_required_note(state=state, top_k=top_k)
            if isinstance(calibration_note, dict):
                _append_run_log(
                    run_id,
                    level="WARN",
                    event="calibration_required",
                    fields={
                        "iteration": iter_idx,
                        "top_blockers": calibration_note.get("top_blockers"),
                    },
                )

            metrics_after = _sec_domain_counts(HttpClient().metrics())
            sec_delta = _sec_domain_usage_delta(metrics_before, metrics_after)
            state.sec_requests_used = int(state.sec_requests_used) + int(sum(sec_delta.values()))

            scoreboard_after_path = state.artifacts.get("peer_scoreboard_path")
            scoreboard_after = _safe_json(Path(str(scoreboard_after_path))) if scoreboard_after_path else {}
            valuation_coverage_after_path = state.artifacts.get("valuation_coverage_path")
            valuation_coverage_after = (
                _safe_json(Path(str(valuation_coverage_after_path)))
                if valuation_coverage_after_path
                else {}
            )
            delta_payload = _scoreboard_delta_payload(
                run_id=run_id,
                iteration=iter_idx,
                before_scoreboard=scoreboard_before,
                after_scoreboard=scoreboard_after,
                before_valuation_coverage=valuation_coverage_before,
                after_valuation_coverage=valuation_coverage_after,
                calibration_required_note=calibration_note,
            )
            delta_path = str(rlm_run_dir(run_id) / "rlm_iterations" / f"iter_{int(iter_idx):03d}_scoreboard_delta.json")
            delta_obj = Path(delta_path)
            delta_obj.parent.mkdir(parents=True, exist_ok=True)
            delta_obj.write_text(json.dumps(delta_payload, indent=2), encoding="utf-8")

            _update_heartbeat(
                phase="critic",
                iteration_value=iter_idx,
                last_action="CRITIC",
                last_progress_summary="Evaluating iteration progress",
            )
            critic_report = evaluate_progress(
                state=state,
                top_k=top_k,
                scoreboard_delta_path=delta_path,
                calibration_required_note=calibration_note,
            )

            if bool(critic_report.no_new_evidence_topk) and float(critic_report.gap_reduction_topk) < float(
                NO_PROGRESS_GAP_REDUCTION_THRESHOLD
            ):
                state.no_progress_streak += 1
            else:
                state.no_progress_streak = 0

            if bool(critic_report.ranking_stable) and str(critic_report.confidence).upper() == "HIGH":
                state.ranking_stable_streak += 1
            else:
                state.ranking_stable_streak = 0

            state.evidence_delta_counters = {
                "evidence_count_topk": int(critic_report.evidence_count_topk),
                "gap_count_topk": int(critic_report.gap_count_topk),
            }
            state.top_k_current = list(critic_report.current_top_k)
            progress_payload = critic_report.model_dump(mode="json")
            terminal_offline_tickers = [
                str(t).upper().strip()
                for t in (planner_meta.get("price_terminal_offline_tickers") or [])
                if str(t).strip()
            ]
            if terminal_offline_tickers:
                progress_payload["planner_note"] = "price_terminal_offline"
                progress_payload["planner_note_tickers"] = sorted(set(terminal_offline_tickers))
            state.progress_history.append(progress_payload)

            planner_dump = planner_output.model_dump(mode="json")
            action_entry = {
                "iteration": iter_idx,
                "created_at": _utc_now(),
                "planner": planner_dump,
                "planner_meta": planner_meta,
                "execution": actions_result,
                "sec_domain_requests": sec_delta,
                "sec_requests_iteration_total": int(sum(sec_delta.values())),
                "scoreboard_delta_path": delta_path,
            }
            state.action_history.append(action_entry)

            iteration_paths = persist_iteration_artifacts(
                run_id=run_id,
                iteration=iter_idx,
                planner_payload=planner_dump,
                actions_payload=actions_result,
                critic_payload=critic_report.model_dump(mode="json"),
                delta_payload=delta_payload,
            )
            state.artifacts["latest_scoreboard_delta_path"] = iteration_paths.get("scoreboard_delta_path")

            insert_rlm_iteration_row(
                run_id=run_id,
                iteration=iter_idx,
                planner_json=planner_dump,
                critic_json=critic_report.model_dump(mode="json"),
                progress_json={
                    "no_progress_streak": int(state.no_progress_streak),
                    "ranking_stable_streak": int(state.ranking_stable_streak),
                    "iteration_artifacts": iteration_paths,
                    "sec_domain_requests": sec_delta,
                    "llm_cost_used_total": float(state.llm_cost_used),
                    "planner_note": progress_payload.get("planner_note"),
                    "planner_note_tickers": progress_payload.get("planner_note_tickers"),
                    "calibration_required": calibration_note,
                },
            )
            _append_run_log(
                run_id,
                level="INFO",
                event="critic_complete",
                fields={
                    "iteration": iter_idx,
                    "confidence": critic_report.confidence,
                    "no_new_evidence_topk": bool(critic_report.no_new_evidence_topk),
                    "gap_reduction_topk": float(critic_report.gap_reduction_topk),
                    "ranking_stable": bool(critic_report.ranking_stable),
                    "improvement_score": float(critic_report.improvement_score),
                    "delta_path": iteration_paths.get("scoreboard_delta_path"),
                },
            )

            state.iteration = iter_idx + 1

            planner_stop = _planner_stop_reason(planner_dump)
            if planner_stop is not None or bool(actions_result.get("planner_requested_stop")):
                reason_code, summary = planner_stop or ("PLANNER_STOP", "Planner requested stop.")
                if str(reason_code).upper() == "DISABLED_PROVIDER_COMPLETED_BASELINE":
                    state.status = "DONE"
                else:
                    state.status = "STOPPED"
                state.stop_reason_code = reason_code
                state.stop_summary = summary
                _append_run_log(
                    run_id,
                    level="INFO",
                    event="planner_stop",
                    fields={"reason_code": reason_code, "summary": summary},
                )
                _persist_state()
                break

            stop_decision = apply_stop_rules(
                state=state,
                critic_report=critic_report,
                max_iterations=max_iterations,
            )
            if stop_decision.should_stop:
                state.status = stop_decision.status
                state.stop_reason_code = stop_decision.reason_code
                state.stop_summary = stop_decision.summary
                _append_run_log(
                    run_id,
                    level="INFO",
                    event="stop_rule_triggered",
                    fields={"reason_code": stop_decision.reason_code, "status": stop_decision.status},
                )
                _persist_state()
                break

            state.status = "RUNNING"
            _persist_state()

        if state.status == "RUNNING":
            if int(state.iteration) >= int(max_iterations):
                state.status = "DONE"
                state.stop_reason_code = "MAX_ITERATIONS_REACHED"
                state.stop_summary = f"Reached iteration cap ({int(max_iterations)})."
            else:
                state.status = "STOPPED"
                state.stop_reason_code = state.stop_reason_code or "STOPPED"
                state.stop_summary = state.stop_summary or "Loop stopped."

    except _RunRefusedError:
        raise
    except InvalidFinancialInputError as exc:
        if state is not None:
            state.status = "FAILED"
            state.stop_reason_code = str(exc.status)
            state.stop_summary = _sanitize_error(exc)
            _append_run_log(
                run_id,
                level="ERROR",
                event="financial_integrity_failure",
                fields={
                    "status": str(exc.status),
                    "error": state.stop_summary,
                },
            )
        raise
    except KeyboardInterrupt:
        if state is None:
            state = init_loop_state(
                run_id=run_id,
                sector=sector,
                as_of_date=as_of_date,
                max_iterations=max(1, int(iterations)),
                llm_budget_usd=max(0.0, float(llm_budget)),
                sec_budget_count=max(0, int(_sec_budget_remaining_estimate())),
            )
            state.mode = mode_norm
            state.rlm_version = "v1.1" if mode_norm == "depth" else "v0"
        state.status = "CANCELLED"
        state.stop_reason_code = "INTERRUPTED"
        state.stop_summary = "Interrupted by user"
        _append_run_log(run_id, level="WARN", event="interrupted", fields={"reason": state.stop_summary})
    except Exception as exc:  # noqa: BLE001
        if state is None:
            state = init_loop_state(
                run_id=run_id,
                sector=sector,
                as_of_date=as_of_date,
                max_iterations=max(1, int(iterations)),
                llm_budget_usd=max(0.0, float(llm_budget)),
                sec_budget_count=max(0, int(_sec_budget_remaining_estimate())),
            )
            state.mode = mode_norm
            state.rlm_version = "v1.1" if mode_norm == "depth" else "v0"
        state.status = "FAILED"
        state.stop_reason_code = "EXCEPTION"
        state.stop_summary = _sanitize_error(exc)
        _append_run_log(run_id, level="ERROR", event="exception", fields={"error": state.stop_summary})
    finally:
        if state is not None:
            _update_heartbeat(
                phase="finalize",
                iteration_value=int(state.iteration),
                last_action="FINALIZE",
                last_progress_summary=state.stop_summary,
                force_write=True,
            )
            if state.status in {"DONE", "STOPPED", "NEEDS_HUMAN", "CANCELLED"}:
                final_paths = _write_final_decision_pack(state=state, top_k=top_k) or {}
                if final_paths:
                    state.artifacts.update(final_paths)
            _persist_state()

            if heartbeat_thread is not None:
                heartbeat_stop_event.set()
                heartbeat_thread.join(timeout=HEARTBEAT_INTERVAL_SECONDS + 1)
            _write_heartbeat(
                heartbeat_runtime
                or {
                    "run_id": state.run_id,
                    "pid": os.getpid(),
                    "status": state.status,
                    "phase": "finalize",
                    "iteration": int(state.iteration),
                    "current_ticker": None,
                    "last_action": "FINALIZE",
                    "last_progress_summary": state.stop_summary,
                    "top_k_current": list(state.top_k_current),
                    "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
                },
                heartbeat_lock,
            )
            _append_run_log(
                run_id,
                level="INFO",
                event="loop_finalized",
                fields={
                    "status": state.status,
                    "stop_reason_code": state.stop_reason_code,
                    "iteration": int(state.iteration),
                },
            )
            result = {
                "run_id": state.run_id,
                "status": state.status,
                "mode": state.mode,
                "iteration": int(state.iteration),
                "stop_reason_code": state.stop_reason_code,
                "stop_summary": state.stop_summary,
                "state_path": str(rlm_run_dir(state.run_id) / "rlm_state.json"),
                "heartbeat_path": str(_heartbeat_path(state.run_id)),
                "run_log_path": str(_run_log_path(state.run_id)),
                "progress_counts": _progress_counts(run_id=state.run_id, state=state),
                "rubric_weights": state.rubric_weights,
                "gate_threshold_overrides": state.gate_threshold_overrides,
                "gate_thresholds_effective": state.gate_thresholds_effective,
                "artifacts": state.artifacts,
            }

    if result is None:
        raise RuntimeError("RLM loop did not produce a result")
    return result


def resume_sector_rlm_loop(
    *,
    run_id: str,
    iterations: int | None = None,
    top_k: int | None = None,
    workers: int | None = None,
    mode: str | None = None,
    force_restart: bool = False,
    timeout_per_stage: float = 30.0,
) -> dict[str, Any]:
    state = load_loop_state(run_id)
    row = get_rlm_run_row(run_id)
    if state is None and row is None:
        raise ValueError(f"RLM run not found: {run_id}")

    sector = state.sector if state is not None else str(row.get("sector") or "")
    as_of_date = state.as_of_date if state is not None else str(row.get("as_of_date") or "")
    if not sector or not as_of_date:
        raise ValueError(f"RLM run metadata missing for run_id={run_id}")

    max_iterations = (
        int(iterations)
        if iterations is not None
        else int(state.budgets_remaining.max_iterations if state is not None else max(2, int(row.get("iterations") or 0) + 1))
    )
    top_k_value = int(top_k) if top_k is not None else int(len(state.top_k_current) if state and state.top_k_current else 5)
    worker_count = int(workers) if workers is not None else 4
    run_mode = str(mode or (state.mode if state is not None else "auto")).lower()
    with_prices_value = bool(state.with_prices) if state is not None else True

    return run_sector_rlm_loop(
        sector=sector,
        as_of_date=as_of_date,
        run_id=run_id,
        peer_limit=25,
        min_peers=15,
        limit_dossiers=25,
        years_back=10,
        workers=max(1, worker_count),
        iterations=max(1, max_iterations),
        top_k=max(1, top_k_value),
        with_research=True,
        with_synthesis=True,
        with_prices=with_prices_value,
        budget_usd=None,
        resume=True,
        mode=run_mode,
        force_restart=force_restart,
        timeout_per_stage=timeout_per_stage,
    )


def sector_rlm_status(*, run_id: str) -> dict[str, Any]:
    state = load_loop_state(run_id)
    row = get_rlm_run_row(run_id)
    heartbeat = _read_heartbeat(run_id)
    heartbeat_age = _heartbeat_age_seconds(heartbeat)
    heartbeat_pid = heartbeat.get("pid") if isinstance(heartbeat.get("pid"), int) else None
    pid_active = _is_pid_active(heartbeat_pid)

    if state is None and row is None:
        return {
            "run_id": run_id,
            "status": "MISSING",
            "message": "rlm_state.json and rlm_runs row not found",
        }

    if state is None and row is not None:
        state = _reconstruct_state_from_db(
            run_id=run_id,
            sector=str(row.get("sector") or "UNKNOWN"),
            as_of_date=str(row.get("as_of_date") or "UNKNOWN"),
            iterations=max(1, int(row.get("iterations") or 1)),
            budget_usd=float(get_config().openai_budget_usd_per_run),
            top_k=5,
            mode="auto",
            with_prices=True,
        )

    latest_progress = state.progress_history[-1] if state and state.progress_history else {}
    db_status = str((row or {}).get("status") or (state.status if state else "UNKNOWN")).upper()
    stale_running = bool(
        db_status == "RUNNING"
        and (heartbeat_age is None or heartbeat_age > float(STALE_HEARTBEAT_SECONDS))
        and not pid_active
    )
    progress_counts = _progress_counts(run_id=run_id, state=state)
    if stale_running:
        next_operator_action = "sector-rlm-resume --run-id ... or sector-rlm-cancel --run-id ... --reason ..."
    elif db_status == "RUNNING":
        next_operator_action = "wait"
    elif db_status == "FAILED":
        next_operator_action = f"python -m app.cli sector-rlm-resume --run-id {run_id} --force-restart"
    else:
        next_operator_action = f"python -m app.cli sector-rlm-open --run-id {run_id}"

    status_payload = {
        "run_id": state.run_id,
        "status": db_status,
        "state_status": state.status,
        "mode": state.mode,
        "sector": state.sector,
        "as_of_date": state.as_of_date,
        "iteration": int(state.iteration),
        "max_iterations": int(state.budgets_remaining.max_iterations),
        "stop_reason_code": state.stop_reason_code,
        "stop_summary": state.stop_summary,
        "top_k_current": state.top_k_current,
        "peer_count": len(state.peer_set),
        "action_history_count": len(state.action_history),
        "progress_history_count": len(state.progress_history),
        "budgets_remaining": state.budgets_remaining.model_dump(mode="json"),
        "rubric_weights": state.rubric_weights,
        "gate_threshold_overrides": state.gate_threshold_overrides,
        "gate_thresholds_effective": state.gate_thresholds_effective,
        "latest_progress": latest_progress if isinstance(latest_progress, dict) else {},
        "artifacts": state.artifacts,
        "state_path": str(rlm_run_dir(run_id) / "rlm_state.json"),
        "heartbeat_path": str(_heartbeat_path(run_id)),
        "heartbeat": heartbeat,
        "heartbeat_age_seconds": heartbeat_age,
        "heartbeat_stale": bool(heartbeat_age is not None and heartbeat_age > float(STALE_HEARTBEAT_SECONDS)),
        "heartbeat_pid_active": bool(pid_active),
        "health": "STALE_RUN_DETECTED" if stale_running else "OK",
        "run_log_path": str(_run_log_path(run_id)),
        "progress_counts": progress_counts,
        "next_operator_action": next_operator_action,
    }
    if stale_running:
        status_payload["suggested_actions"] = [
            f"python -m app.cli sector-rlm-resume --run-id {run_id}",
            f"python -m app.cli sector-rlm-cancel --run-id {run_id} --reason 'stale run cleanup'",
        ]
    return status_payload


def sector_rlm_open(*, run_id: str) -> dict[str, Any]:
    payload = sector_rlm_status(run_id=run_id)
    if payload.get("status") == "MISSING":
        return payload
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    return {
        "run_id": run_id,
        "status": payload.get("status"),
        "mode": payload.get("mode"),
        "health": payload.get("health"),
        "run_dir": str(rlm_run_dir(run_id)),
        "state_path": payload.get("state_path"),
        "heartbeat_path": payload.get("heartbeat_path"),
        "run_log_path": payload.get("run_log_path"),
        "decision_pack_md": artifacts.get("final_decision_pack_md_path")
        or artifacts.get("decision_pack_md_path"),
        "decision_pack_json": artifacts.get("final_decision_pack_path")
        or artifacts.get("decision_pack_path"),
        "peer_report_path": artifacts.get("peer_report_path"),
        "peer_rankings_path": artifacts.get("peer_rankings_path"),
        "peer_scoreboard_path": artifacts.get("peer_scoreboard_path"),
        "shares_summary_path": artifacts.get("shares_summary_path"),
        "fundamentals_summary_path": artifacts.get("fundamentals_summary_path"),
        "valuation_summary_path": artifacts.get("valuation_summary_path"),
        "price_coverage_path": artifacts.get("price_coverage_path"),
        "shares_coverage_path": artifacts.get("shares_coverage_path"),
        "fcf_coverage_path": artifacts.get("fcf_coverage_path"),
        "facts_coverage_path": artifacts.get("facts_coverage_path"),
        "valuation_coverage_path": artifacts.get("valuation_coverage_path"),
        "value_gates_path": artifacts.get("value_gates_path"),
        "value_gates_calibration_path": artifacts.get("value_gates_calibration_path"),
        "universe_scout_linkage_path": artifacts.get("universe_scout_linkage_path"),
        "sector_synthesis_path": artifacts.get("sector_synthesis_path"),
    }


def _tail_last_lines(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if lines <= 0:
        return []
    return content[-int(lines) :]


def sector_rlm_tail(*, run_id: str, interval: float = 2.0, lines: int = 40, once: bool = False) -> dict[str, Any]:
    loops = 0
    last_status: dict[str, Any] = {}
    try:
        while True:
            status = sector_rlm_status(run_id=run_id)
            last_status = status
            heartbeat = status.get("heartbeat") if isinstance(status.get("heartbeat"), dict) else {}
            progress = status.get("progress_counts") if isinstance(status.get("progress_counts"), dict) else {}
            budgets = status.get("budgets_remaining") if isinstance(status.get("budgets_remaining"), dict) else {}
            skipped_total = int(progress.get("dossier_skipped_preflight", 0)) + int(
                progress.get("dossier_skipped_budget", 0)
            )
            line = (
                f"status={status.get('status')} iteration={status.get('iteration')} phase={heartbeat.get('phase')} "
                f"last_action={heartbeat.get('last_action')} current_ticker={heartbeat.get('current_ticker')} "
                f"heartbeat_age_seconds={status.get('heartbeat_age_seconds')} "
                f"sec_budget_remaining={budgets.get('sec_budget_remaining')} "
                f"llm_budget_remaining={budgets.get('llm_budget_remaining')} "
                f"peer_count={status.get('peer_count')} dossier_ok_count={progress.get('dossier_ok', 0)} "
                f"dossier_skipped_count={skipped_total} dossier_failed_count={progress.get('dossier_failed', 0)}"
            )
            print(line, flush=True)
            log_path = Path(str(status.get("run_log_path") or ""))
            tail_lines = _tail_last_lines(log_path, lines=int(lines))
            if tail_lines:
                print(f"--- {log_path} (last {int(lines)} lines) ---", flush=True)
                for row in tail_lines:
                    print(row, flush=True)
            loops += 1
            if once:
                break
            time.sleep(max(0.1, float(interval)))
    except KeyboardInterrupt:
        pass
    return {
        "run_id": run_id,
        "loops": loops,
        "last_status": last_status.get("status"),
        "run_log_path": str(_run_log_path(run_id)),
    }
