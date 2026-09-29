from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, TypedDict, cast

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.config import get_config
from app.db import utc_now_iso
from app.patterns.scanner import count_patterns_with_hits
from app.universe.autopilot import run_universe_autopilot
from app.universe.escalation import (
    ACTION_ADD_TO_ACTIVE_WATCHLIST,
    ACTION_BUILD_REFRESHED_MEMO,
    ACTION_CLEAR_BLOCKERS,
    ACTION_DEEPEN_FILING_DIFF,
    ACTION_NO_ACTION,
    ACTION_REBUILD_DEPTH_RUN,
    ACTION_RECHECK_PROMOTION,
    ACTION_RESOLVE_SYNTHESIS_GAP,
    ACTION_SCHEDULE_LIGHT_REFRESH,
    ACTION_TRACK_VARIANT_PERCEPTION,
    ACTION_RUN_TARGETED_PATTERN_SCAN,
)
from app.universe.memo_pack import write_investment_memo_pack


RUNNER_STATUS_RUNNING = "RUNNING"
RUNNER_STATUS_PARTIAL = "PARTIAL"
RUNNER_STATUS_DONE = "DONE"
RUNNER_STATUS_CANCELLED = "CANCELLED"
RUNNER_STATUS_FAILED = "FAILED"

RESULT_DONE = "DONE"
RESULT_SKIPPED = "SKIPPED"
RESULT_FAILED = "FAILED"
RESULT_PLANNED_ONLY = "PLANNED_ONLY"

STOP_COMPLETED = "COMPLETED"
STOP_MAX_ITEMS_REACHED = "MAX_ITEMS_REACHED"
STOP_CANCEL_REQUESTED = "CANCEL_REQUESTED"
STOP_EXCEPTION = "EXCEPTION"
STOP_DRY_RUN = "DRY_RUN"

UNKNOWN = "UNKNOWN"

_TERMINAL_RESULTS = {RESULT_DONE, RESULT_SKIPPED, RESULT_FAILED, RESULT_PLANNED_ONLY}


class EscalationItem(TypedDict, total=False):
    queue_rank: int
    ticker: str
    priority_lane: str
    action_type: str
    action_reason: str
    action_metadata: dict[str, Any]
    blocking_reason_code: str
    recommended_command: str
    source_campaign_run_id: str
    source_universe_run_ids: list[str]
    latest_metrics: dict[str, Any]
    artifacts_to_read: dict[str, Any]
    l4_signal_summary: dict[str, Any]
    l4_priority_boost: int
    status: str
    appearances_count: int


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.write("\n")


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _runner_paths(campaign_run_id: str) -> dict[str, Path]:
    root = get_config().campaigns_dir / campaign_run_id
    return {
        "root": root,
        "queue_path": root / "escalation_queue.json",
        "plan_path": root / "escalation_plan.json",
        "summary_path": root / "escalation_summary.json",
        "state_path": root / "escalation_state.json",
        "log_path": root / "escalation_log.jsonl",
        "results_path": root / "escalation_results.json",
        "active_watchlist_path": root / "active_watchlist.json",
        "campaign_state_path": root / "campaign_state.json",
        "promotion_state_path": root / "promotion_state.json",
        "priority_lanes_path": root / "priority_lanes.json",
        "master_shortlist_path": root / "master_shortlist.json",
        "master_shortlist_md_path": root / "master_shortlist.md",
        "master_watchlist_path": root / "master_watchlist_state.json",
    }


def _first_source_run_id(queue_item: dict[str, Any]) -> str:
    for run_id in queue_item.get("source_universe_run_ids") or []:
        token = str(run_id or "").strip()
        if token:
            return token
    return ""


def _source_run_root(run_id: str) -> Path:
    return get_config().outputs_dir / "universe" / str(run_id).strip()


def _source_run_as_of_date(run_id: str, fallback: str = "") -> str:
    autopilot_state = _safe_json(_source_run_root(run_id) / "autopilot" / "autopilot_state.json")
    return str(autopilot_state.get("as_of_date") or fallback or "").strip()


def _source_batch_state(run_id: str) -> dict[str, Any]:
    batch_run_id = f"{str(run_id).strip()}_depth_batch"
    return _safe_json(
        _source_run_root(run_id) / "depth_batches" / batch_run_id / "batch_state.json"
    )


def _completed_depth_tickers(batch_state: dict[str, Any]) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()
    for row in [value for value in (batch_state.get("completed_runs") or []) if isinstance(value, dict)]:
        ticker = str(row.get("ticker") or row.get("input_ticker") or "").strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


def _persist_filing_diff_report(*, ticker: str, run_id: str, report_payload: dict[str, Any]) -> list[str]:
    cfg = get_config()
    local_path = cfg.outputs_dir / "universe" / run_id / "filing_diffs" / f"{ticker}.json"
    global_path = cfg.outputs_dir / "diffs" / f"{ticker}_{run_id}_diff.json"
    _json_write(local_path, report_payload)
    _json_write(global_path, report_payload)
    return [str(local_path), str(global_path)]


def _persist_pattern_scan_report(*, run_id: str, report_payload: dict[str, Any]) -> list[str]:
    cfg = get_config()
    local_root = cfg.outputs_dir / "universe" / run_id / "pattern_scan"
    report_path = local_root / "pattern_scan_report.json"
    summary_path = local_root / "pattern_scan_summary.json"
    _json_write(report_path, report_payload)
    _json_write(
        summary_path,
        {
            "run_id": run_id,
            "peer_set_size": int(report_payload.get("peer_set_size") or 0),
            "patterns_tested": len(
                [value for value in (report_payload.get("pattern_results") or []) if isinstance(value, dict)]
            ),
            "patterns_with_signal": count_patterns_with_hits(report_payload),
            "total_hits": sum(
                int(result.get("hit_count") or 0)
                for result in (report_payload.get("pattern_results") or [])
                if isinstance(result, dict)
            ),
        },
    )
    return [str(report_path), str(summary_path)]


def _materialize_variant_report_for_run(*, ticker: str, run_id: str, as_of_date: str) -> str:
    cfg = get_config()
    global_path = cfg.outputs_dir / "variant_perceptions" / f"{ticker}_{as_of_date}.json"
    if not global_path.exists():
        return ""
    payload = _safe_json(global_path)
    if not payload:
        return ""
    local_path = cfg.outputs_dir / "universe" / run_id / "variant_perceptions" / global_path.name
    _json_write(local_path, payload)
    return str(local_path)


def _load_variant_report_for_queue_item(queue_item: dict[str, Any], fallback_as_of: str) -> tuple[dict[str, Any], str]:
    cfg = get_config()
    ticker = str(queue_item.get("ticker") or "").strip().upper()
    for run_id in queue_item.get("source_universe_run_ids") or []:
        run_id_norm = str(run_id or "").strip()
        if not run_id_norm:
            continue
        as_of_date = _source_run_as_of_date(run_id_norm, fallback_as_of)
        candidates: list[Path] = []
        if as_of_date:
            candidates.append(
                cfg.outputs_dir / "universe" / run_id_norm / "variant_perceptions" / f"{ticker}_{as_of_date}.json"
            )
            candidates.append(cfg.outputs_dir / "variant_perceptions" / f"{ticker}_{as_of_date}.json")
        candidates.extend(
            sorted(
                (cfg.outputs_dir / "universe" / run_id_norm / "variant_perceptions").glob(f"{ticker}_*.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if (cfg.outputs_dir / "universe" / run_id_norm / "variant_perceptions").exists()
            else []
        )
        for path in candidates:
            payload = _safe_json(path)
            if payload:
                return payload, str(path)
    return {}, ""


def _expected_resolution_date(time_horizon: str, *, base_date: date | None = None) -> str:
    base = base_date or date.today()
    token = str(time_horizon or "").upper()
    if token == "SHORT":
        return (base + timedelta(days=365)).isoformat()
    if token == "LONG":
        return (base + timedelta(days=365 * 5)).isoformat()
    return (base + timedelta(days=365 * 3)).isoformat()


def load_escalation_queue(campaign_run_id: str) -> list[EscalationItem]:
    payload = _safe_json(_runner_paths(campaign_run_id)["queue_path"])
    rows = [row for row in (payload.get("rows") or []) if isinstance(row, dict)]
    rows.sort(key=lambda row: (int(row.get("queue_rank") or 10**9), str(row.get("ticker") or "")))
    if not rows:
        raise ValueError(f"Missing or invalid escalation_queue.json for campaign_run_id={campaign_run_id}")
    return [cast(EscalationItem, row) for row in rows]


def _default_state(campaign_run_id: str, queue_items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "campaign_run_id": campaign_run_id,
        "queue_count": len(queue_items),
        "completed_count": 0,
        "cursor_next_idx": 0,
        "status": RUNNER_STATUS_RUNNING,
        "stop_reason_code": "",
        "stop_summary": "",
        "current_item": {},
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "results_summary": {
            "by_action_type": {},
            "by_outcome": {},
        },
        "queue_items": queue_items,
    }


def _write_empty_active_watchlist(paths: dict[str, Path], campaign_run_id: str) -> None:
    _json_write(
        paths["active_watchlist_path"],
        {
            "campaign_run_id": campaign_run_id,
            "generated_at": utc_now_iso(),
            "entry_count": 0,
            "entries": [],
        },
    )


def _reset_runtime_artifacts(paths: dict[str, Path], campaign_run_id: str) -> None:
    if paths["log_path"].exists():
        paths["log_path"].write_text("", encoding="utf-8")
    _write_empty_active_watchlist(paths, campaign_run_id)


def _result_map(results_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in [value for value in (results_payload.get("rows") or []) if isinstance(value, dict)]:
        rank = int(row.get("queue_rank") or -1)
        if rank >= 0:
            out[rank] = row
    return out


def _build_result_summary(results_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_action_type: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    completed_action_counts: dict[str, int] = {}
    lane_change_counts: dict[str, int] = {}
    for row in results_rows:
        action = str(row.get("action_type") or UNKNOWN)
        outcome = str(row.get("status") or UNKNOWN)
        by_action_type[action] = by_action_type.get(action, 0) + 1
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
        if outcome == RESULT_DONE:
            completed_action_counts[action] = completed_action_counts.get(action, 0) + 1
        lane_before = str(row.get("lane_before") or "").strip()
        lane_after = str(row.get("lane_after") or "").strip()
        if lane_before and lane_after and lane_before != lane_after:
            token = f"{lane_before}->{lane_after}"
            lane_change_counts[token] = lane_change_counts.get(token, 0) + 1
    return {
        "by_action_type": dict(sorted(by_action_type.items(), key=lambda item: item[0])),
        "by_outcome": dict(sorted(by_outcome.items(), key=lambda item: item[0])),
        "completed_action_counts": dict(sorted(completed_action_counts.items(), key=lambda item: item[0])),
        "lane_change_counts": dict(sorted(lane_change_counts.items(), key=lambda item: item[0])),
    }


def _active_watchlist_count(path: Path) -> int:
    payload = _safe_json(path)
    entries = [row for row in (payload.get("entries") or []) if isinstance(row, dict)]
    return len(entries)


def _write_state_and_results(paths: dict[str, Path], state: dict[str, Any], results_rows: list[dict[str, Any]]) -> None:
    summary = _build_result_summary(results_rows)
    state["updated_at"] = utc_now_iso()
    state["completed_count"] = len(results_rows)
    state["results_summary"] = {
        "by_action_type": summary["by_action_type"],
        "by_outcome": summary["by_outcome"],
    }
    _json_write(paths["state_path"], state)
    _json_write(
        paths["results_path"],
        {
            "campaign_run_id": str(state.get("campaign_run_id") or ""),
            "generated_at": utc_now_iso(),
            "result_count": len(results_rows),
            "completed_action_counts": summary["completed_action_counts"],
            "lane_change_counts": summary["lane_change_counts"],
            "rows": results_rows,
        },
    )
    planner_summary = _safe_json(paths["summary_path"])
    planner_summary["executed_queue_count"] = len(results_rows)
    planner_summary["completed_action_counts"] = summary["completed_action_counts"]
    planner_summary["lane_change_counts"] = summary["lane_change_counts"]
    planner_summary["active_watchlist_count"] = _active_watchlist_count(paths["active_watchlist_path"])
    _json_write(paths["summary_path"], planner_summary)
    _refresh_campaign_summary_only(str(state.get("campaign_run_id") or ""))


def _refresh_campaign_summary_only(campaign_run_id: str) -> None:
    from app.universe.campaign import _build_summary, _campaign_paths, _safe_json as _campaign_safe_json
    from app.universe.research_memory import update_research_memory_from_campaign

    paths = _campaign_paths(campaign_run_id)
    state = _campaign_safe_json(paths["state_path"])
    if not state:
        return
    campaign_summary = _build_summary(paths, state)
    campaign_summary["campaign_summary_path"] = str(paths["summary_path"])
    update_research_memory_from_campaign(
        campaign_run_id,
        campaign_summary=campaign_summary,
        master_shortlist=_campaign_safe_json(paths["master_shortlist_json_path"]),
        master_watchlist_state=_campaign_safe_json(paths["master_watchlist_state_path"]),
        promotion_state=_campaign_safe_json(paths["promotion_state_path"]),
        escalation_results={
            **_safe_json(_runner_paths(campaign_run_id)["results_path"]),
            "escalation_results_path": str(_runner_paths(campaign_run_id)["results_path"]),
        },
    )
    _json_write(paths["summary_path"], _build_summary(paths, state))


def _action_run_id(campaign_run_id: str, ticker: str, queue_rank: int, action_type: str) -> str:
    token = str(ticker or "").strip().lower()
    rank = f"{int(queue_rank):03d}"
    if action_type == ACTION_REBUILD_DEPTH_RUN:
        return f"{campaign_run_id}__rebuild__{token}__{rank}"
    if action_type == ACTION_CLEAR_BLOCKERS:
        return f"{campaign_run_id}__escalate__{token}__{rank}"
    if action_type == ACTION_SCHEDULE_LIGHT_REFRESH:
        return f"{campaign_run_id}__refresh__{token}__{rank}"
    return ""


def _targeted_autopilot_params(*, ticker: str, light: bool) -> dict[str, Any]:
    scout_params = {
        "tickers": [str(ticker).strip().upper()],
        "top_n": 1,
        "batch_size": 1,
        "max_batches": 1,
        "with_prices": True,
    }
    depth_batch_params = {
        "max_runs": 1,
        "mode": "depth",
        "iterations": 1 if light else 2,
        "top_k": 1 if light else 3,
        "with_prices": True,
    }
    common = {
        "rollup_params": {"top_n": 1, "policy": "value_first"},
        "dossier_pack_params": {"top_n": 1, "policy": "value_first"},
        "memo_pack_params": {"top_n": 1, "policy": "value_first"},
    }
    return {
        "scout_params": scout_params,
        "depth_batch_params": depth_batch_params,
        **common,
    }


def _run_action_autopilot(
    *,
    campaign_run_id: str,
    queue_item: dict[str, Any],
    execution_run_id: str,
    action_type: str,
    as_of_date: str,
) -> dict[str, Any]:
    ticker = str(queue_item.get("ticker") or "").strip().upper()
    light = action_type == ACTION_SCHEDULE_LIGHT_REFRESH
    params = _targeted_autopilot_params(ticker=ticker, light=light)
    result = run_universe_autopilot(
        universe_run_id=execution_run_id,
        as_of_date=as_of_date,
        scout_params=params["scout_params"],
        depth_batch_params=params["depth_batch_params"],
        rollup_params=params["rollup_params"],
        dossier_pack_params=params["dossier_pack_params"],
        memo_pack_params=params["memo_pack_params"],
        resume=True,
        force=False,
    )
    return result if isinstance(result, dict) else {}


def _targeted_paths(universe_run_id: str) -> dict[str, Path]:
    cfg = get_config()
    batch_run_id = f"{universe_run_id}_depth_batch"
    root = cfg.outputs_dir / "universe" / universe_run_id
    return {
        "autopilot_watchlist": root / "autopilot" / "watchlist_state.json",
        "memo_manifest": root / "depth_batches" / batch_run_id / "memo_pack" / "memo_pack_manifest.json",
        "global_shortlist": root / "depth_batches" / batch_run_id / "global_shortlist.json",
        "memo_pack_dir": root / "depth_batches" / batch_run_id / "memo_pack",
        "batch_run_id": Path(batch_run_id),
    }


def _find_targeted_candidate(universe_run_id: str, ticker: str, action_type: str) -> dict[str, Any]:
    ticker_norm = str(ticker or "").strip().upper()
    paths = _targeted_paths(universe_run_id)
    shortlist_payload = _safe_json(paths["global_shortlist"])
    watchlist_payload = _safe_json(paths["autopilot_watchlist"])
    memo_manifest = _safe_json(paths["memo_manifest"])
    shortlist_rows = [row for row in (shortlist_payload.get("rows") or []) if isinstance(row, dict)]
    row = next((value for value in shortlist_rows if str(value.get("ticker") or "").upper() == ticker_norm), {})
    memos = [value for value in (memo_manifest.get("memos") or []) if isinstance(value, dict)]
    memo_row = next((value for value in memos if str(value.get("ticker") or "").upper() == ticker_norm), {})
    watch_row = {}
    if isinstance(watchlist_payload.get("tickers"), dict):
        watch_row = watchlist_payload["tickers"].get(ticker_norm) if isinstance(watchlist_payload["tickers"].get(ticker_norm), dict) else {}
    source_runs = [
        {
            "campaign_item": f"ESCALATION_{action_type}",
            "universe_run_id": universe_run_id,
            "batch_run_id": f"{universe_run_id}_depth_batch",
        }
    ]
    return {
        "ticker": ticker_norm,
        "source_runs": source_runs,
        "best_rank_seen": int(memo_row.get("rank_global") or row.get("rank_global") or 1),
        "value_gate_status": str(
            row.get("value_gate_status") or watch_row.get("last_value_gate_status") or UNKNOWN
        ).upper(),
        "latest_value_gate_status": str(
            watch_row.get("last_value_gate_status") or row.get("value_gate_status") or UNKNOWN
        ).upper(),
        "implied_return_base": row.get("implied_return_base", watch_row.get("last_implied_return_base", UNKNOWN)),
        "latest_implied_return_base": watch_row.get("last_implied_return_base", row.get("implied_return_base", UNKNOWN)),
        "mos_epv": row.get("mos_epv", UNKNOWN),
        "mos_netnet": row.get("mos_netnet", UNKNOWN),
        "owner_earnings_yield_ev_3y": row.get("owner_earnings_yield_ev_3y", UNKNOWN),
        "yield_metric_used": str(row.get("yield_metric_used") or UNKNOWN),
        "primary_blocker": str(
            row.get("primary_blocker") or watch_row.get("last_primary_blocker_code") or UNKNOWN
        ).upper(),
        "latest_primary_blocker": str(
            watch_row.get("last_primary_blocker_code") or row.get("primary_blocker") or UNKNOWN
        ).upper(),
        "memo_path": str(memo_row.get("memo_md_path") or memo_row.get("memo_json_path") or ""),
        "composite_score_total": row.get("composite_score_total", UNKNOWN),
        "derived_from": [str(ref) for ref in (row.get("derived_from") or []) if str(ref).strip()],
        "watch_row": watch_row,
    }


def _merge_targeted_result_into_campaign(
    campaign_run_id: str,
    *,
    execution_run_id: str,
    ticker: str,
    action_type: str,
) -> dict[str, Any]:
    from app.universe.campaign import _campaign_markdown, _campaign_paths, _master_sort_key

    paths = _runner_paths(campaign_run_id)
    candidate = _find_targeted_candidate(execution_run_id, ticker, action_type)
    if not candidate.get("ticker"):
        return {}

    master_shortlist = _safe_json(paths["master_shortlist_path"])
    rows = [row for row in (master_shortlist.get("rows") or []) if isinstance(row, dict)]
    existing = next((row for row in rows if str(row.get("ticker") or "").upper() == str(candidate["ticker"])), None)
    if existing is None:
        merged = dict(candidate)
        rows.append(merged)
    else:
        all_sources = [value for value in (existing.get("source_runs") or []) if isinstance(value, dict)] + [
            value for value in (candidate.get("source_runs") or []) if isinstance(value, dict)
        ]
        deduped_sources: list[dict[str, Any]] = []
        seen_sources: set[tuple[str, str, str]] = set()
        for source in all_sources:
            key = (
                str(source.get("campaign_item") or ""),
                str(source.get("universe_run_id") or ""),
                str(source.get("batch_run_id") or ""),
            )
            if key in seen_sources:
                continue
            seen_sources.add(key)
            deduped_sources.append(
                {
                    "campaign_item": key[0],
                    "universe_run_id": key[1],
                    "batch_run_id": key[2],
                }
            )
        representative = sorted([existing, candidate], key=_master_sort_key)[0]
        merged = dict(representative)
        merged["ticker"] = str(candidate["ticker"])
        merged["source_runs"] = deduped_sources
        merged["best_rank_seen"] = min(
            int(existing.get("best_rank_seen") or 10**9),
            int(candidate.get("best_rank_seen") or 10**9),
        )
        merged["latest_value_gate_status"] = str(candidate.get("latest_value_gate_status") or UNKNOWN)
        merged["latest_primary_blocker"] = str(candidate.get("latest_primary_blocker") or UNKNOWN)
        merged["derived_from"] = sorted(
            {
                str(ref)
                for ref in list(existing.get("derived_from") or []) + list(candidate.get("derived_from") or [])
                if str(ref).strip()
            }
        )
        rows = [row for row in rows if str(row.get("ticker") or "").upper() != str(candidate["ticker"])]
        rows.append(merged)

    ranked = sorted(rows, key=_master_sort_key)
    for idx, row in enumerate(ranked, start=1):
        row["rank_master"] = int(idx)
    _json_write(
        paths["master_shortlist_path"],
        {
            "campaign_run_id": campaign_run_id,
            "generated_at": utc_now_iso(),
            "candidate_count_total": max(len(ranked), int(master_shortlist.get("candidate_count_total") or 0)),
            "candidate_count_ranked": len(ranked),
            "rows": ranked,
        },
    )
    _campaign_paths(campaign_run_id)["master_shortlist_md_path"].write_text(
        _campaign_markdown(campaign_run_id=campaign_run_id, rows=ranked[:50]),
        encoding="utf-8",
    )

    master_watch = _safe_json(paths["master_watchlist_path"])
    tickers = master_watch.get("tickers") if isinstance(master_watch.get("tickers"), dict) else {}
    watch_row = candidate.get("watch_row") if isinstance(candidate.get("watch_row"), dict) else {}
    ticker_norm = str(candidate.get("ticker") or "").upper()
    existing_watch = tickers.get(ticker_norm) if isinstance(tickers.get(ticker_norm), dict) else {}
    history = [row for row in (existing_watch.get("history") or []) if isinstance(row, dict)]
    history.append(
        {
            "campaign_item": f"ESCALATION_{action_type}",
            "universe_run_id": execution_run_id,
            "value_gate_status": str(candidate.get("latest_value_gate_status") or UNKNOWN),
            "implied_return_base": candidate.get("latest_implied_return_base", UNKNOWN),
            "primary_blocker": str(candidate.get("latest_primary_blocker") or UNKNOWN),
            "last_rank": int(candidate.get("best_rank_seen") or 0),
        }
    )
    tickers[ticker_norm] = {
        "first_seen_campaign_run_id": str(existing_watch.get("first_seen_campaign_run_id") or campaign_run_id),
        "last_seen_campaign_run_id": campaign_run_id,
        "first_seen_universe_run_id": str(existing_watch.get("first_seen_universe_run_id") or execution_run_id),
        "last_seen_universe_run_id": execution_run_id,
        "latest_value_gate_status": str(candidate.get("latest_value_gate_status") or UNKNOWN),
        "latest_implied_return_base": candidate.get("latest_implied_return_base", UNKNOWN),
        "latest_primary_blocker": str(candidate.get("latest_primary_blocker") or UNKNOWN),
        "appearances_count": int(existing_watch.get("appearances_count") or 0) + 1,
        "history": history[-10:],
    }
    gate_counts: dict[str, int] = {}
    for row in tickers.values():
        gate = str(row.get("latest_value_gate_status") or UNKNOWN)
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
    _json_write(
        paths["master_watchlist_path"],
        {
            "campaign_run_id": campaign_run_id,
            "generated_at": utc_now_iso(),
            "ticker_count": len(tickers),
            "gate_counts": dict(sorted(gate_counts.items(), key=lambda item: item[0])),
            "tickers": dict(sorted(tickers.items(), key=lambda item: item[0])),
        },
    )
    return {
        "ticker": ticker_norm,
        "latest_value_gate_status": str(candidate.get("latest_value_gate_status") or UNKNOWN),
        "latest_primary_blocker": str(candidate.get("latest_primary_blocker") or UNKNOWN),
    }


def _refresh_promotion_and_escalation(campaign_run_id: str) -> dict[str, Any]:
    from app.universe.escalation import write_escalation_artifacts
    from app.universe.promotion import write_promotion_artifacts

    promotion_paths = write_promotion_artifacts(campaign_run_id)
    escalation_paths = write_escalation_artifacts(campaign_run_id)
    _refresh_campaign_summary_only(campaign_run_id)
    return {
        **promotion_paths,
        **escalation_paths,
    }


def _result_for_recheck(campaign_run_id: str, ticker: str, lane_before: str) -> dict[str, Any]:
    _refresh_promotion_and_escalation(campaign_run_id)
    promotion_state = _safe_json(_runner_paths(campaign_run_id)["promotion_state_path"])
    row = next(
        (
            value
            for value in (promotion_state.get("rows") or [])
            if isinstance(value, dict) and str(value.get("ticker") or "").upper() == str(ticker).upper()
        ),
        {},
    )
    return {
        "lane_after": str(row.get("priority_lane") or lane_before or UNKNOWN),
        "blocker_after": str(
            row.get("latest_primary_blocker") or row.get("primary_blocker") or UNKNOWN
        ).upper(),
        "artifacts_written": [
            str(_runner_paths(campaign_run_id)["promotion_state_path"]),
            str(_runner_paths(campaign_run_id)["plan_path"]),
            str(_runner_paths(campaign_run_id)["summary_path"]),
        ],
    }


def _update_active_watchlist(paths: dict[str, Path], queue_item: dict[str, Any]) -> dict[str, Any]:
    payload = _safe_json(paths["active_watchlist_path"])
    entries = [row for row in (payload.get("entries") or []) if isinstance(row, dict)]
    ticker = str(queue_item.get("ticker") or "").upper()
    timestamp = utc_now_iso()
    lane = str(queue_item.get("priority_lane") or UNKNOWN)
    kept = [row for row in entries if str(row.get("ticker") or "").upper() != ticker]
    kept.append(
        {
            "ticker": ticker,
            "lane": lane,
            "priority_lane": lane,
            "reason": str(queue_item.get("action_reason") or UNKNOWN),
            "source_run_ids": [str(value) for value in (queue_item.get("source_universe_run_ids") or []) if str(value).strip()],
            "timestamp": timestamp,
            "updated_at": timestamp,
        }
    )
    kept.sort(key=lambda row: str(row.get("ticker") or ""))
    _json_write(
        paths["active_watchlist_path"],
        {
            "campaign_run_id": str(queue_item.get("source_campaign_run_id") or ""),
            "generated_at": utc_now_iso(),
            "entry_count": len(kept),
            "entries": kept,
        },
    )
    return {
        "active_watchlist_path": str(paths["active_watchlist_path"]),
        "active_watchlist_count": len(kept),
    }


def _latest_execution_run_id(results_rows: list[dict[str, Any]], ticker: str) -> str:
    for row in reversed(results_rows):
        if str(row.get("ticker") or "").upper() != str(ticker).upper():
            continue
        token = str(row.get("execution_run_id") or "").strip()
        if token:
            return token
    return ""


def _result_skipped_missing_artifacts(*, lane_before: str, blocker_before: str) -> dict[str, Any]:
    return {
        "status": RESULT_SKIPPED,
        "execution_run_id": "",
        "outcome_summary": {"reason": "SKIPPED_MISSING_ARTIFACTS"},
        "lane_before": lane_before,
        "lane_after": lane_before,
        "blocker_before": blocker_before,
        "blocker_after": blocker_before,
        "artifacts_written": [],
    }


def _result_failed(
    *,
    lane_before: str,
    blocker_before: str,
    reason: str,
    artifacts_written: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outcome = {"reason": reason}
    if isinstance(extra, dict):
        outcome.update(extra)
    return {
        "status": RESULT_FAILED,
        "execution_run_id": "",
        "outcome_summary": outcome,
        "lane_before": lane_before,
        "lane_after": lane_before,
        "blocker_before": blocker_before,
        "blocker_after": blocker_before,
        "artifacts_written": [str(path) for path in (artifacts_written or []) if str(path).strip()],
    }


def _execute_queue_item(
    *,
    campaign_run_id: str,
    queue_item: dict[str, Any],
    as_of_date: str,
    results_rows: list[dict[str, Any]],
    paths: dict[str, Path],
) -> dict[str, Any]:
    action_type = str(queue_item.get("action_type") or "")
    ticker = str(queue_item.get("ticker") or "").upper()
    queue_rank = int(queue_item.get("queue_rank") or 0)
    lane_before = str(queue_item.get("priority_lane") or UNKNOWN)
    blocker_before = str(queue_item.get("blocking_reason_code") or UNKNOWN).upper()

    if action_type in {ACTION_SCHEDULE_LIGHT_REFRESH, ACTION_CLEAR_BLOCKERS, ACTION_REBUILD_DEPTH_RUN}:
        execution_run_id = _action_run_id(campaign_run_id, ticker, queue_rank, action_type)
        payload = _run_action_autopilot(
            campaign_run_id=campaign_run_id,
            queue_item=queue_item,
            execution_run_id=execution_run_id,
            action_type=action_type,
            as_of_date=as_of_date,
        )
        merge_payload = _merge_targeted_result_into_campaign(
            campaign_run_id,
            execution_run_id=execution_run_id,
            ticker=ticker,
            action_type=action_type,
        )
        _refresh_campaign_summary_only(campaign_run_id)
        return {
            "status": RESULT_DONE if payload else RESULT_PLANNED_ONLY,
            "execution_run_id": execution_run_id,
            "outcome_summary": {
                "run_status": str(payload.get("status") or UNKNOWN),
                "merged_into_campaign": bool(merge_payload),
            },
            "lane_before": lane_before,
            "lane_after": "",
            "blocker_before": blocker_before,
            "blocker_after": str(merge_payload.get("latest_primary_blocker") or blocker_before),
            "artifacts_written": [
                str(paths["master_shortlist_path"]),
                str(paths["master_watchlist_path"]),
            ],
        }

    if action_type == ACTION_DEEPEN_FILING_DIFF:
        from app.diff.engine import build_filing_diff_report
        from app.synthesis.variant_builder import build_variant_perceptions

        source_run_id = _first_source_run_id(queue_item)
        if not source_run_id:
            return _result_failed(
                lane_before=lane_before,
                blocker_before=blocker_before,
                reason="MISSING_SOURCE_RUN_ID",
            )
        try:
            report = build_filing_diff_report(ticker=ticker, run_id=source_run_id, years_back=5)
        except InvalidFinancialInputError:
            raise
        except Exception as exc:  # noqa: BLE001
            return _result_failed(
                lane_before=lane_before,
                blocker_before=blocker_before,
                reason="FILING_DIFF_FAILED",
                extra={"error": str(exc)},
            )
        report_payload = report.model_dump(mode="json")
        artifacts_written = _persist_filing_diff_report(
            ticker=ticker,
            run_id=source_run_id,
            report_payload=report_payload,
        )
        changes = [value for value in (report_payload.get("changes") or []) if isinstance(value, dict)]
        high_materiality_count = len(
            [value for value in changes if str(value.get("materiality") or "").upper() == "HIGH"]
        )
        variant_report_path = ""
        variant_refresh_status = "SKIPPED"
        as_of_for_variant = _source_run_as_of_date(source_run_id, as_of_date)
        if as_of_for_variant:
            try:
                build_variant_perceptions(
                    ticker=ticker,
                    as_of_date=as_of_for_variant,
                    run_id=source_run_id,
                    cfg=get_config(),
                    diff_report=report_payload,
                    persist=True,
                )
                variant_report_path = _materialize_variant_report_for_run(
                    ticker=ticker,
                    run_id=source_run_id,
                    as_of_date=as_of_for_variant,
                )
                if variant_report_path:
                    artifacts_written.append(variant_report_path)
                variant_refresh_status = "DONE"
            except Exception as exc:  # noqa: BLE001
                variant_refresh_status = f"FAILED:{exc}"
        return {
            "status": RESULT_DONE,
            "execution_run_id": "",
            "outcome_summary": {
                "diff_report_path": artifacts_written[0] if artifacts_written else "",
                "changes_found": len(changes),
                "high_materiality_count": high_materiality_count,
                "variant_refresh_status": variant_refresh_status,
            },
            "lane_before": lane_before,
            "lane_after": lane_before,
            "blocker_before": blocker_before,
            "blocker_after": blocker_before,
            "artifacts_written": artifacts_written,
        }

    if action_type == ACTION_RUN_TARGETED_PATTERN_SCAN:
        from app.patterns.scanner import scan_peer_set, summarize_pattern_scan_for_ticker

        source_run_id = _first_source_run_id(queue_item)
        if not source_run_id:
            return _result_failed(
                lane_before=lane_before,
                blocker_before=blocker_before,
                reason="MISSING_SOURCE_RUN_ID",
            )
        batch_state = _source_batch_state(source_run_id)
        peer_tickers = _completed_depth_tickers(batch_state)
        if not peer_tickers:
            return _result_failed(
                lane_before=lane_before,
                blocker_before=blocker_before,
                reason="PATTERN_SCAN_PEER_SET_UNAVAILABLE",
            )
        try:
            report = scan_peer_set(run_id=source_run_id, tickers=peer_tickers, cfg=get_config())
        except Exception as exc:  # noqa: BLE001
            return _result_failed(
                lane_before=lane_before,
                blocker_before=blocker_before,
                reason="PATTERN_SCAN_FAILED",
                extra={"error": str(exc)},
            )
        report_payload = report.model_dump(mode="json")
        artifacts_written = _persist_pattern_scan_report(run_id=source_run_id, report_payload=report_payload)
        target_summary = summarize_pattern_scan_for_ticker(report, ticker)
        return {
            "status": RESULT_DONE,
            "execution_run_id": "",
            "outcome_summary": {
                "scan_report_path": artifacts_written[0] if artifacts_written else "",
                "hits_for_target_ticker": int(target_summary.get("pattern_hit_count") or 0),
            },
            "lane_before": lane_before,
            "lane_after": lane_before,
            "blocker_before": blocker_before,
            "blocker_after": blocker_before,
            "artifacts_written": artifacts_written,
        }

    if action_type == ACTION_TRACK_VARIANT_PERCEPTION:
        from app.calibration.perception_tracker import register_perception
        from app.synthesis.schemas import VariantPerception

        payload, _source_path = _load_variant_report_for_queue_item(queue_item, as_of_date)
        perceptions = [
            value
            for value in (payload.get("perceptions") or [])
            if isinstance(value, dict)
            and str(value.get("testable_prediction") or "").strip()
            and str(value.get("time_horizon") or "").strip()
        ]
        if not perceptions:
            return _result_failed(
                lane_before=lane_before,
                blocker_before=blocker_before,
                reason="VARIANT_REPORT_MISSING_OR_UNTRACKABLE",
            )
        tracking_dir = get_config().outputs_dir / "perception_tracking"
        artifacts_written: list[str] = []
        perception_ids: list[str] = []
        for perception_payload in perceptions:
            try:
                perception = VariantPerception.model_validate(perception_payload)
            except Exception:
                continue
            record = register_perception(perception)
            tracking_path = tracking_dir / f"{record.ticker}_{record.perception_id}.json"
            artifacts_written.append(str(tracking_path))
            perception_ids.append(record.perception_id)
        return {
            "status": RESULT_DONE,
            "execution_run_id": "",
            "outcome_summary": {
                "perceptions_registered": len(perception_ids),
                "perception_ids": perception_ids,
            },
            "lane_before": lane_before,
            "lane_after": lane_before,
            "blocker_before": blocker_before,
            "blocker_after": blocker_before,
            "artifacts_written": artifacts_written,
        }

    if action_type == ACTION_RESOLVE_SYNTHESIS_GAP:
        metadata = queue_item.get("action_metadata") if isinstance(queue_item.get("action_metadata"), dict) else {}
        gap_description = str(metadata.get("gap_description") or "").strip()
        recommended_action = str(metadata.get("recommended_action") or "").strip()
        if not gap_description:
            return _result_failed(
                lane_before=lane_before,
                blocker_before=blocker_before,
                reason="SYNTHESIS_GAP_METADATA_MISSING",
            )
        return {
            "status": RESULT_PLANNED_ONLY,
            "execution_run_id": "",
            "outcome_summary": {
                "status": RESULT_PLANNED_ONLY,
                "gap_description": gap_description,
                "recommended_action": recommended_action,
                "requires_human_review": True,
            },
            "lane_before": lane_before,
            "lane_after": lane_before,
            "blocker_before": blocker_before,
            "blocker_after": blocker_before,
            "artifacts_written": [],
        }

    if action_type == ACTION_RECHECK_PROMOTION:
        recheck = _result_for_recheck(campaign_run_id, ticker, lane_before)
        return {
            "status": RESULT_DONE,
            "execution_run_id": "",
            "outcome_summary": {"promotion_rechecked": True},
            "lane_before": lane_before,
            "lane_after": recheck["lane_after"],
            "blocker_before": blocker_before,
            "blocker_after": recheck["blocker_after"],
            "artifacts_written": recheck["artifacts_written"],
        }

    if action_type == ACTION_ADD_TO_ACTIVE_WATCHLIST:
        watch = _update_active_watchlist(paths, queue_item)
        return {
            "status": RESULT_DONE,
            "execution_run_id": "",
            "outcome_summary": {"active_watchlist_count": watch["active_watchlist_count"]},
            "lane_before": lane_before,
            "lane_after": lane_before,
            "blocker_before": blocker_before,
            "blocker_after": blocker_before,
            "artifacts_written": [watch["active_watchlist_path"]],
        }

    if action_type == ACTION_BUILD_REFRESHED_MEMO:
        target_run_id = _latest_execution_run_id(results_rows, ticker) or (
            (queue_item.get("source_universe_run_ids") or [None])[0] or ""
        )
        if not str(target_run_id).strip():
            return _result_skipped_missing_artifacts(lane_before=lane_before, blocker_before=blocker_before)
        batch_run_id = f"{target_run_id}_depth_batch"
        try:
            pack = write_investment_memo_pack(
                universe_run_id=str(target_run_id),
                batch_run_id=batch_run_id,
                top_n=1,
                policy="value_first",
            )
        except (ValueError, FileNotFoundError):
            return _result_skipped_missing_artifacts(lane_before=lane_before, blocker_before=blocker_before)
        return {
            "status": RESULT_DONE,
            "execution_run_id": "",
            "outcome_summary": {"memo_refresh": "DONE"},
            "lane_before": lane_before,
            "lane_after": lane_before,
            "blocker_before": blocker_before,
            "blocker_after": blocker_before,
            "artifacts_written": [
                str(pack.get("manifest_path") or ""),
                str(pack.get("watchlist_state_path") or ""),
            ],
        }

    if action_type == ACTION_NO_ACTION:
        return {
            "status": RESULT_PLANNED_ONLY,
            "execution_run_id": "",
            "outcome_summary": {"reason": "NO_ACTION"},
            "lane_before": lane_before,
            "lane_after": lane_before,
            "blocker_before": blocker_before,
            "blocker_after": blocker_before,
            "artifacts_written": [],
        }

    return {
        "status": RESULT_SKIPPED,
        "execution_run_id": "",
        "outcome_summary": {"reason": "UNSUPPORTED_ACTION"},
        "lane_before": lane_before,
        "lane_after": lane_before,
        "blocker_before": blocker_before,
        "blocker_after": blocker_before,
        "artifacts_written": [],
    }


def run_escalation_queue(
    campaign_run_id: str,
    *,
    max_items: int | None = None,
    dry_run: bool = False,
    resume: bool = True,
    force_restart: bool = False,
) -> dict[str, Any]:
    paths = _runner_paths(campaign_run_id)
    if force_restart:
        queue_items = load_escalation_queue(campaign_run_id)
        state = _default_state(campaign_run_id, queue_items)
        results_rows: list[dict[str, Any]] = []
        _reset_runtime_artifacts(paths, campaign_run_id)
    else:
        state = _safe_json(paths["state_path"]) if resume else {}
        results_payload = _safe_json(paths["results_path"])
        results_rows = [row for row in (results_payload.get("rows") or []) if isinstance(row, dict)]
        if state:
            queue_items = [row for row in (state.get("queue_items") or []) if isinstance(row, dict)]
            if not queue_items:
                queue_items = load_escalation_queue(campaign_run_id)
                state["queue_items"] = queue_items
        else:
            queue_items = load_escalation_queue(campaign_run_id)
            state = _default_state(campaign_run_id, queue_items)

    if str(state.get("status") or "").upper() == RUNNER_STATUS_CANCELLED and not force_restart:
        _write_state_and_results(paths, state, results_rows)
        return {
            "status": RUNNER_STATUS_CANCELLED,
            "campaign_run_id": campaign_run_id,
            "escalation_state_path": str(paths["state_path"]),
            "escalation_results_path": str(paths["results_path"]),
        }

    if bool(dry_run):
        state["status"] = RUNNER_STATUS_PARTIAL
        state["stop_reason_code"] = STOP_DRY_RUN
        state["stop_summary"] = "Dry-run only."
        _write_state_and_results(paths, state, results_rows)
        return {
            "status": RUNNER_STATUS_PARTIAL,
            "campaign_run_id": campaign_run_id,
            "stop_reason_code": STOP_DRY_RUN,
            "escalation_state_path": str(paths["state_path"]),
            "escalation_results_path": str(paths["results_path"]),
        }

    queue_items = [row for row in (state.get("queue_items") or []) if isinstance(row, dict)]
    state["status"] = RUNNER_STATUS_RUNNING
    state["stop_reason_code"] = ""
    state["stop_summary"] = ""
    cursor = max(0, int(state.get("cursor_next_idx") or 0))
    cap_remaining = int(max_items) if _is_num(max_items) and int(max_items) > 0 else None
    _write_state_and_results(paths, state, results_rows)
    _append_jsonl(
        paths["log_path"],
        {
            "ts": utc_now_iso(),
            "event": "escalation_run_started",
            "campaign_run_id": campaign_run_id,
            "cursor_next_idx": cursor,
            "max_items": cap_remaining,
        },
    )

    as_of_date = str(_safe_json(paths["campaign_state_path"]).get("as_of_date") or "")
    result_lookup = {int(row.get("queue_rank") or -1): row for row in results_rows}

    for idx in range(cursor, len(queue_items)):
        if cap_remaining is not None and cap_remaining <= 0:
            state["status"] = RUNNER_STATUS_PARTIAL
            state["stop_reason_code"] = STOP_MAX_ITEMS_REACHED
            state["stop_summary"] = "Invocation item cap reached."
            break

        latest_state = _safe_json(paths["state_path"])
        if str(latest_state.get("status") or "").upper() == RUNNER_STATUS_CANCELLED:
            state = latest_state
            break

        queue_item = queue_items[idx]
        queue_rank = int(queue_item.get("queue_rank") or -1)
        if queue_rank in result_lookup and str(result_lookup[queue_rank].get("status") or "") in _TERMINAL_RESULTS:
            state["cursor_next_idx"] = int(idx + 1)
            continue

        state["current_item"] = {
            "queue_rank": queue_rank,
            "ticker": str(queue_item.get("ticker") or ""),
            "action_type": str(queue_item.get("action_type") or ""),
            "execution_run_id": _action_run_id(
                campaign_run_id,
                str(queue_item.get("ticker") or ""),
                queue_rank,
                str(queue_item.get("action_type") or ""),
            ),
        }
        _write_state_and_results(paths, state, results_rows)
        _append_jsonl(
            paths["log_path"],
            {
                "ts": utc_now_iso(),
                "event": "escalation_item_started",
                "queue_rank": queue_rank,
                "ticker": queue_item.get("ticker"),
                "action_type": queue_item.get("action_type"),
            },
        )

        started_at = utc_now_iso()
        try:
            outcome = _execute_queue_item(
                campaign_run_id=campaign_run_id,
                queue_item=queue_item,
                as_of_date=as_of_date,
                results_rows=results_rows,
                paths=paths,
            )
            result_row = {
                "queue_rank": queue_rank,
                "ticker": str(queue_item.get("ticker") or ""),
                "action_type": str(queue_item.get("action_type") or ""),
                "status": str(outcome.get("status") or RESULT_SKIPPED),
                "execution_run_id": str(outcome.get("execution_run_id") or ""),
                "started_at": started_at,
                "finished_at": utc_now_iso(),
                "outcome_summary": outcome.get("outcome_summary") if isinstance(outcome.get("outcome_summary"), dict) else {},
                "lane_before": str(outcome.get("lane_before") or ""),
                "lane_after": str(outcome.get("lane_after") or ""),
                "blocker_before": str(outcome.get("blocker_before") or ""),
                "blocker_after": str(outcome.get("blocker_after") or ""),
                "artifacts_written": [str(path) for path in (outcome.get("artifacts_written") or []) if str(path).strip()],
                "recommended_command": str(queue_item.get("recommended_command") or ""),
                "l4_signal_summary": queue_item.get("l4_signal_summary") if isinstance(queue_item.get("l4_signal_summary"), dict) else {},
                "l4_priority_boost": int(queue_item.get("l4_priority_boost") or 0),
                "error_summary": "",
            }
        except InvalidFinancialInputError as exc:
            result_row = {
                "queue_rank": queue_rank,
                "ticker": str(queue_item.get("ticker") or ""),
                "action_type": str(queue_item.get("action_type") or ""),
                "status": RESULT_FAILED,
                "execution_run_id": _action_run_id(
                    campaign_run_id,
                    str(queue_item.get("ticker") or ""),
                    queue_rank,
                    str(queue_item.get("action_type") or ""),
                ),
                "started_at": started_at,
                "finished_at": utc_now_iso(),
                "outcome_summary": {"financial_integrity": exc.result.to_dict()},
                "lane_before": str(queue_item.get("priority_lane") or ""),
                "lane_after": "",
                "blocker_before": str(queue_item.get("blocking_reason_code") or ""),
                "blocker_after": "",
                "artifacts_written": [],
                "recommended_command": str(queue_item.get("recommended_command") or ""),
                "l4_signal_summary": (
                    queue_item.get("l4_signal_summary")
                    if isinstance(queue_item.get("l4_signal_summary"), dict)
                    else {}
                ),
                "l4_priority_boost": int(queue_item.get("l4_priority_boost") or 0),
                "error_summary": str(exc),
            }
            result_lookup[queue_rank] = result_row
            results_rows = [
                result_lookup[key] for key in sorted(result_lookup.keys()) if key >= 0
            ]
            state["status"] = RUNNER_STATUS_FAILED
            state["stop_reason_code"] = exc.status
            state["stop_summary"] = str(exc)
            state["financial_integrity"] = exc.result.to_dict()
            state["cursor_next_idx"] = int(idx + 1)
            state["current_item"] = {}
            _write_state_and_results(paths, state, results_rows)
            _append_jsonl(
                paths["log_path"],
                {
                    "ts": utc_now_iso(),
                    "event": "escalation_item_finished",
                    "queue_rank": queue_rank,
                    "ticker": queue_item.get("ticker"),
                    "action_type": queue_item.get("action_type"),
                    "status": RESULT_FAILED,
                    "stop_reason_code": exc.status,
                },
            )
            raise
        except Exception as exc:  # noqa: BLE001
            result_row = {
                "queue_rank": queue_rank,
                "ticker": str(queue_item.get("ticker") or ""),
                "action_type": str(queue_item.get("action_type") or ""),
                "status": RESULT_FAILED,
                "execution_run_id": _action_run_id(
                    campaign_run_id,
                    str(queue_item.get("ticker") or ""),
                    queue_rank,
                    str(queue_item.get("action_type") or ""),
                ),
                "started_at": started_at,
                "finished_at": utc_now_iso(),
                "outcome_summary": {},
                "lane_before": str(queue_item.get("priority_lane") or ""),
                "lane_after": "",
                "blocker_before": str(queue_item.get("blocking_reason_code") or ""),
                "blocker_after": "",
                "artifacts_written": [],
                "recommended_command": str(queue_item.get("recommended_command") or ""),
                "l4_signal_summary": queue_item.get("l4_signal_summary") if isinstance(queue_item.get("l4_signal_summary"), dict) else {},
                "l4_priority_boost": int(queue_item.get("l4_priority_boost") or 0),
                "error_summary": str(exc),
            }
            state["status"] = RUNNER_STATUS_FAILED
            state["stop_reason_code"] = STOP_EXCEPTION
            state["stop_summary"] = str(exc)

        result_lookup[queue_rank] = result_row
        results_rows = [result_lookup[key] for key in sorted(result_lookup.keys()) if key >= 0]
        state["cursor_next_idx"] = int(idx + 1)
        state["current_item"] = {}
        _write_state_and_results(paths, state, results_rows)
        _append_jsonl(
            paths["log_path"],
            {
                "ts": utc_now_iso(),
                "event": "escalation_item_finished",
                "queue_rank": queue_rank,
                "ticker": queue_item.get("ticker"),
                "action_type": queue_item.get("action_type"),
                "status": result_row["status"],
            },
        )

        if result_row["status"] == RESULT_FAILED and str(result_row.get("error_summary") or "").strip():
            break
        if cap_remaining is not None:
            cap_remaining -= 1

    if str(state.get("status") or "").upper() == RUNNER_STATUS_RUNNING:
        if int(state.get("cursor_next_idx") or 0) >= len(queue_items):
            state["status"] = RUNNER_STATUS_DONE
            state["stop_reason_code"] = STOP_COMPLETED
            state["stop_summary"] = "Escalation queue completed."
        else:
            state["status"] = RUNNER_STATUS_PARTIAL
            if not str(state.get("stop_reason_code") or "").strip():
                state["stop_reason_code"] = STOP_MAX_ITEMS_REACHED
                state["stop_summary"] = "Escalation queue paused before completion."

    _write_state_and_results(paths, state, results_rows)
    _append_jsonl(
        paths["log_path"],
        {
            "ts": utc_now_iso(),
            "event": "escalation_run_finished",
            "campaign_run_id": campaign_run_id,
            "status": state.get("status"),
            "stop_reason_code": state.get("stop_reason_code"),
        },
    )
    return {
        "status": str(state.get("status") or RUNNER_STATUS_PARTIAL),
        "campaign_run_id": campaign_run_id,
        "stop_reason_code": str(state.get("stop_reason_code") or ""),
        "stop_summary": str(state.get("stop_summary") or ""),
        "queue_count": len(queue_items),
        "completed_count": len(results_rows),
        "cursor_next_idx": int(state.get("cursor_next_idx") or 0),
        "escalation_state_path": str(paths["state_path"]),
        "escalation_results_path": str(paths["results_path"]),
        "active_watchlist_path": str(paths["active_watchlist_path"]),
    }


def resume_escalation_queue(campaign_run_id: str, *, max_items: int | None = None) -> dict[str, Any]:
    return run_escalation_queue(campaign_run_id, max_items=max_items, dry_run=False, resume=True, force_restart=False)


def cancel_escalation_queue(campaign_run_id: str, reason: str) -> dict[str, Any]:
    paths = _runner_paths(campaign_run_id)
    state = _safe_json(paths["state_path"])
    if not state:
        try:
            queue_items = load_escalation_queue(campaign_run_id)
        except ValueError:
            return {
                "status": "MISSING",
                "campaign_run_id": campaign_run_id,
                "escalation_state_path": str(paths["state_path"]),
            }
        state = _default_state(campaign_run_id, queue_items)
        _write_empty_active_watchlist(paths, campaign_run_id)
    state["status"] = RUNNER_STATUS_CANCELLED
    state["stop_reason_code"] = STOP_CANCEL_REQUESTED
    state["stop_summary"] = str(reason or "Cancelled by operator.")
    current = state.get("current_item") if isinstance(state.get("current_item"), dict) else {}
    execution_run_id = str(current.get("execution_run_id") or "")
    if execution_run_id:
        from app.universe.autopilot import cancel_universe_autopilot

        try:
            cancel_universe_autopilot(execution_run_id, str(reason or "Cancelled by operator."))
        except Exception:
            pass
    results_payload = _safe_json(paths["results_path"])
    results_rows = [row for row in (results_payload.get("rows") or []) if isinstance(row, dict)]
    _write_state_and_results(paths, state, results_rows)
    return {
        "status": "OK",
        "campaign_run_id": campaign_run_id,
        "run_status": RUNNER_STATUS_CANCELLED,
        "stop_reason_code": STOP_CANCEL_REQUESTED,
        "stop_summary": str(state.get("stop_summary") or ""),
        "escalation_state_path": str(paths["state_path"]),
        "escalation_results_path": str(paths["results_path"]),
    }


def open_escalation_status(campaign_run_id: str) -> dict[str, Any]:
    paths = _runner_paths(campaign_run_id)
    state = _safe_json(paths["state_path"])
    if not state:
        return {
            "status": "MISSING",
            "campaign_run_id": campaign_run_id,
            "escalation_state_path": str(paths["state_path"]),
            "escalation_results_path": str(paths["results_path"]),
        }
    results_payload = _safe_json(paths["results_path"])
    results_rows = [row for row in (results_payload.get("rows") or []) if isinstance(row, dict)]
    summary = _build_result_summary(results_rows)
    lane_changes = [
        {
            "ticker": str(row.get("ticker") or ""),
            "action_type": str(row.get("action_type") or ""),
            "lane_before": str(row.get("lane_before") or ""),
            "lane_after": str(row.get("lane_after") or ""),
        }
        for row in results_rows
        if str(row.get("lane_before") or "").strip() and str(row.get("lane_after") or "").strip() and row.get("lane_before") != row.get("lane_after")
    ]
    return {
        "status": "OK",
        "campaign_run_id": campaign_run_id,
        "run_status": str(state.get("status") or RUNNER_STATUS_RUNNING),
        "stop_reason_code": str(state.get("stop_reason_code") or ""),
        "stop_summary": str(state.get("stop_summary") or ""),
        "queue_count": int(state.get("queue_count") or 0),
        "completed_count": len(results_rows),
        "cursor_next_idx": int(state.get("cursor_next_idx") or 0),
        "current_item": state.get("current_item") if isinstance(state.get("current_item"), dict) else {},
        "counts_by_status": summary["by_outcome"],
        "completed_action_counts": summary["completed_action_counts"],
        "lane_change_counts": summary["lane_change_counts"],
        "top_lane_changes": lane_changes[:10],
        "active_watchlist_count": _active_watchlist_count(paths["active_watchlist_path"]),
        "escalation_state_path": str(paths["state_path"]),
        "escalation_results_path": str(paths["results_path"]),
        "active_watchlist_path": str(paths["active_watchlist_path"]),
    }
