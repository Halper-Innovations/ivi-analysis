from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import utc_now_iso


BATCH_STATUS_RUNNING = "RUNNING"
BATCH_STATUS_PARTIAL = "PARTIAL"
BATCH_STATUS_DONE = "DONE"
BATCH_STATUS_CANCELLED = "CANCELLED"
BATCH_STATUS_FAILED = "FAILED"

STOP_MAX_RUNS_REACHED = "MAX_RUNS_REACHED"
STOP_CANCEL_REQUESTED = "CANCEL_REQUESTED"
STOP_EXCEPTION = "EXCEPTION"
STOP_COMPLETED = "COMPLETED"

PLAN_STATUS_PENDING = "PENDING"
PLAN_STATUS_RUNNING = "RUNNING"
PLAN_STATUS_DONE = "DONE"
PLAN_STATUS_FAILED = "FAILED"
PLAN_STATUS_CANCELLED = "CANCELLED"
PLAN_STATUS_SKIPPED_EXISTING = "SKIPPED_EXISTING"

_RUN_TERMINAL_STATUSES = {"DONE", "COMPLETED", "FAILED", "CANCELLED"}
_QUEUE_SELECTION_POLICIES = {"QUEUE_ORDER"}


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


def _slug_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip())
    token = token.strip("_")
    return token or "UNKNOWN_SECTOR"


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _dedupe_tickers_keep_order(items: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        token = str(item or "").strip().upper()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _queue_sort_key(entry: dict[str, Any]) -> tuple[int, str, str, str]:
    rank = entry.get("rank")
    rank_value = int(rank) if _is_num(rank) else 10**9
    run_id_suggested = str(entry.get("run_id_suggested") or "")
    sector = str(entry.get("sector_suggested") or "UNKNOWN_SECTOR")
    tickers = [str(ticker) for ticker in (entry.get("tickers") or []) if str(ticker).strip()]
    first_ticker = tickers[0] if tickers else ""
    return (rank_value, run_id_suggested, sector, first_ticker)


def _default_batch_run_id() -> str:
    ts = utc_now_iso().replace(":", "").replace("-", "").replace(".", "").replace("+", "_")
    ts = re.sub(r"[^0-9T_Z]", "", ts)[:24]
    return f"depth_batch_{ts or 'run'}"


def _batch_dir(*, universe_run_id: str, batch_run_id: str) -> Path:
    cfg = get_config()
    return cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id


def _batch_paths(*, universe_run_id: str, batch_run_id: str) -> dict[str, Path]:
    base_dir = _batch_dir(universe_run_id=universe_run_id, batch_run_id=batch_run_id)
    return {
        "batch_dir": base_dir,
        "state_path": base_dir / "batch_state.json",
        "log_path": base_dir / "batch_log.jsonl",
        "summary_path": base_dir / "batch_summary.json",
    }


def _find_batch_paths_by_id(*, batch_run_id: str) -> dict[str, Any]:
    cfg = get_config()
    root = cfg.outputs_dir / "universe"
    matches = sorted(root.glob(f"*/depth_batches/{batch_run_id}/batch_state.json"))
    if not matches:
        return {}
    if len(matches) > 1:
        raise ValueError(f"Ambiguous batch_run_id={batch_run_id}; found multiple universe parents.")
    state_path = matches[0]
    batch_dir = state_path.parent
    universe_run_id = state_path.parents[2].name
    return {
        "universe_run_id": universe_run_id,
        "batch_dir": batch_dir,
        "state_path": state_path,
        "log_path": batch_dir / "batch_log.jsonl",
        "summary_path": batch_dir / "batch_summary.json",
    }


def _build_sector_rlm_command(
    *,
    run_id: str,
    sector: str,
    as_of_date: str,
    tickers: list[str],
    mode: str,
    iterations: int,
    top_k: int,
    peer_limit: int,
    min_peers_dossierable: int,
    limit_dossiers: int,
    workers: int,
    with_prices: bool,
    llm_budget: float | None,
) -> str:
    argv = [
        ".venv/bin/python",
        "-m",
        "app.cli",
        "sector-rlm",
        "--mode",
        str(mode),
        "--sector",
        str(sector),
        "--as-of",
        str(as_of_date),
        "--run-id",
        str(run_id),
        "--iterations",
        str(int(iterations)),
        "--top-k",
        str(int(top_k)),
        "--peer-limit",
        str(int(peer_limit)),
        "--min-peers-dossierable",
        str(int(min_peers_dossierable)),
        "--limit-dossiers",
        str(int(limit_dossiers)),
        "--workers",
        str(int(workers)),
        "--tickers",
        ",".join(_dedupe_tickers_keep_order(tickers)),
    ]
    if not with_prices:
        argv.append("--no-with-prices")
    if _is_num(llm_budget):
        argv.extend(["--budget-usd", str(float(llm_budget))])
    return " ".join(shlex.quote(str(token)) for token in argv)


def _top_counts(counts: dict[str, Any], *, top_n: int = 3) -> list[dict[str, Any]]:
    rows = [
        {"reason_code": str(key), "count": int(value)}
        for key, value in counts.items()
        if _is_num(value)
    ]
    rows.sort(key=lambda row: (-int(row["count"]), str(row["reason_code"])))
    return rows[: max(1, int(top_n))]


def load_depth_queue(run_id: str) -> list[dict[str, Any]]:
    cfg = get_config()
    path = cfg.outputs_dir / "universe" / run_id / "depth_queue.json"
    payload = _safe_json(path)
    entries = [row for row in (payload.get("entries") or []) if isinstance(row, dict)]
    if not entries:
        raise ValueError(f"Missing or invalid depth_queue.json for run_id={run_id}")
    entries = sorted(entries, key=_queue_sort_key)
    as_of_date = str(payload.get("as_of_date") or "").strip()
    out: list[dict[str, Any]] = []
    for row in entries:
        out.append(
            {
                "universe_run_id": run_id,
                "as_of_date": as_of_date,
                "source_depth_queue_path": str(path),
                "queue_rank": int(row.get("rank")) if _is_num(row.get("rank")) else None,
                "sector": str(row.get("sector_suggested") or "UNKNOWN_SECTOR"),
                "tickers": _dedupe_tickers_keep_order(list(row.get("tickers") or [])),
                "recommended_flags": row.get("recommended_flags") if isinstance(row.get("recommended_flags"), dict) else {},
                "source_command": str(row.get("command") or ""),
                "source_rationale": row.get("rationale") if isinstance(row.get("rationale"), dict) else {},
            }
        )
    return out


def create_batch_plan(
    queue: list[dict[str, Any]],
    max_runs: int | None,
    selection_policy: str = "QUEUE_ORDER",
    *,
    batch_run_id: str | None = None,
) -> list[dict[str, Any]]:
    policy = str(selection_policy or "QUEUE_ORDER").upper()
    if policy not in _QUEUE_SELECTION_POLICIES:
        raise ValueError(f"Unsupported selection_policy={selection_policy}")
    normalized = [row for row in queue if isinstance(row, dict)]
    normalized = sorted(
        normalized,
        key=lambda row: (
            int(row.get("queue_rank")) if _is_num(row.get("queue_rank")) else 10**9,
            str(row.get("sector") or ""),
            str(((row.get("tickers") or [None])[0]) or ""),
        ),
    )
    if _is_num(max_runs) and int(max_runs) > 0:
        normalized = normalized[: int(max_runs)]
    batch_id = str(batch_run_id or _default_batch_run_id())
    plan: list[dict[str, Any]] = []
    for idx, row in enumerate(normalized, start=1):
        tickers = _dedupe_tickers_keep_order(list(row.get("tickers") or []))
        recommended = row.get("recommended_flags") if isinstance(row.get("recommended_flags"), dict) else {}
        sector = str(row.get("sector") or "UNKNOWN_SECTOR")
        run_id = f"{batch_id}__{_slug_token(sector)}__{idx:03d}"
        plan.append(
            {
                "idx": int(idx - 1),
                "queue_rank": row.get("queue_rank"),
                "run_id": run_id,
                "sector": sector,
                "as_of_date": str(row.get("as_of_date") or ""),
                "tickers": tickers,
                "recommended_flags": recommended,
                "source_depth_queue_path": str(row.get("source_depth_queue_path") or ""),
                "source_command": str(row.get("source_command") or ""),
                "source_rationale": row.get("source_rationale") if isinstance(row.get("source_rationale"), dict) else {},
                "status": PLAN_STATUS_PENDING,
                "universe_run_id": str(row.get("universe_run_id") or ""),
            }
        )
    return plan


def _collect_run_metrics(*, run_id: str) -> dict[str, Any]:
    from app.rlm.loop import sector_rlm_open, sector_rlm_status
    from app.sector.cycle import sector_scoreboard_open
    from app.valuation.value_gates import open_value_gates_for_run

    status_payload = sector_rlm_status(run_id=run_id)
    open_payload = sector_rlm_open(run_id=run_id)
    scoreboard_payload = sector_scoreboard_open(run_id=run_id)
    gates_payload = open_value_gates_for_run(run_id=run_id, top_n=5)

    run_status = str(status_payload.get("status") or "UNKNOWN").upper()
    stop_reason_code = str(status_payload.get("stop_reason_code") or "UNKNOWN")
    valuation_reason_counts = (
        scoreboard_payload.get("valuation_unknown_reason_counts")
        if isinstance(scoreboard_payload.get("valuation_unknown_reason_counts"), dict)
        else (
            scoreboard_payload.get("valuation_coverage_reason_counts")
            if isinstance(scoreboard_payload.get("valuation_coverage_reason_counts"), dict)
            else {}
        )
    )
    price_unknown_reason_counts = (
        scoreboard_payload.get("price_unknown_reason_counts")
        if isinstance(scoreboard_payload.get("price_unknown_reason_counts"), dict)
        else {}
    )
    gates_counts = gates_payload.get("counts") if isinstance(gates_payload.get("counts"), dict) else {}
    artifacts = {
        "run_dir": str(open_payload.get("run_dir") or ""),
        "peer_scoreboard_path": str(scoreboard_payload.get("peer_scoreboard_path") or ""),
        "valuation_coverage_path": str(scoreboard_payload.get("valuation_coverage_path") or ""),
        "value_gates_path": str(gates_payload.get("value_gates_path") or ""),
    }
    return {
        "run_id": run_id,
        "status": run_status,
        "stop_reason_code": stop_reason_code,
        "known_implied_return_count": int(scoreboard_payload.get("implied_return_known_count") or 0),
        "unknown_implied_return_count": int(scoreboard_payload.get("implied_return_unknown_count") or 0),
        "pass_watch_fail_counts": {
            "PASS": int(gates_counts.get("PASS", 0)),
            "WATCH": int(gates_counts.get("WATCH", 0)),
            "FAIL": int(gates_counts.get("FAIL", 0)),
        },
        "dominant_blockers": {
            "valuation_reason_counts_top": _top_counts(valuation_reason_counts, top_n=3),
            "price_unknown_reason_counts_top": _top_counts(price_unknown_reason_counts, top_n=3),
        },
        "artifacts_paths": artifacts,
        "scoreboard_status": str(scoreboard_payload.get("status") or "UNKNOWN"),
        "value_gates_status": str(gates_payload.get("status") or "UNKNOWN"),
    }


def _execute_plan_item(
    *,
    plan_item: dict[str, Any],
    mode: str,
    iterations: int,
    top_k: int,
    workers: int,
    with_prices: bool,
    llm_budget: float | None,
) -> dict[str, Any]:
    from app.rlm.loop import run_sector_rlm_loop

    tickers = _dedupe_tickers_keep_order(list(plan_item.get("tickers") or []))
    peer_limit = max(1, len(tickers))
    top_k_eff = max(1, int(top_k))
    min_peers = max(1, min(peer_limit, top_k_eff))
    limit_dossiers = max(1, peer_limit)
    return run_sector_rlm_loop(
        sector=str(plan_item.get("sector") or "UNKNOWN_SECTOR"),
        as_of_date=str(plan_item.get("as_of_date") or ""),
        run_id=str(plan_item.get("run_id") or ""),
        peer_limit=peer_limit,
        min_peers=min_peers,
        limit_dossiers=limit_dossiers,
        years_back=10,
        workers=max(1, int(workers)),
        iterations=max(1, int(iterations)),
        top_k=top_k_eff,
        with_research=True,
        with_synthesis=True,
        with_prices=bool(with_prices),
        budget_usd=float(llm_budget) if _is_num(llm_budget) else None,
        resume=True,
        mode=str(mode or "depth").lower(),
        force_restart=False,
        seed_tickers=tickers,
        parent_run_id=str(plan_item.get("universe_run_id") or ""),
        shortlist_source=str(plan_item.get("source_depth_queue_path") or ""),
    )


def _summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    planned_runs = [row for row in (state.get("planned_runs") or []) if isinstance(row, dict)]
    completed_runs = [row for row in (state.get("completed_runs") or []) if isinstance(row, dict)]
    completed_count = len(completed_runs)
    done_count = len(
        [
            row
            for row in completed_runs
            if str(row.get("status") or "").upper() in {"DONE", "COMPLETED"}
        ]
    )
    failed_count = len([row for row in completed_runs if str(row.get("status") or "").upper() == "FAILED"])
    cancelled_count = len([row for row in completed_runs if str(row.get("status") or "").upper() == "CANCELLED"])
    implied_known_total = sum(int(row.get("known_implied_return_count") or 0) for row in completed_runs)
    implied_unknown_total = sum(int(row.get("unknown_implied_return_count") or 0) for row in completed_runs)

    valuation_reasons: dict[str, int] = {}
    price_reasons: dict[str, int] = {}
    for row in completed_runs:
        blockers = row.get("dominant_blockers") if isinstance(row.get("dominant_blockers"), dict) else {}
        for item in blockers.get("valuation_reason_counts_top") or []:
            if not isinstance(item, dict):
                continue
            code = str(item.get("reason_code") or "UNKNOWN")
            count = int(item.get("count") or 0)
            valuation_reasons[code] = valuation_reasons.get(code, 0) + count
        for item in blockers.get("price_unknown_reason_counts_top") or []:
            if not isinstance(item, dict):
                continue
            code = str(item.get("reason_code") or "UNKNOWN")
            count = int(item.get("count") or 0)
            price_reasons[code] = price_reasons.get(code, 0) + count

    return {
        "batch_run_id": str(state.get("batch_run_id") or ""),
        "universe_run_id": str(state.get("universe_run_id") or ""),
        "status": str(state.get("status") or BATCH_STATUS_RUNNING),
        "stop_reason_code": str(state.get("stop_reason_code") or STOP_COMPLETED),
        "stop_summary": str(state.get("stop_summary") or ""),
        "selection_policy": str(state.get("selection_policy") or "QUEUE_ORDER"),
        "max_runs": int(state.get("max_runs") or len(planned_runs)),
        "cursor_next_idx": int(state.get("cursor_next_idx") or 0),
        "total_planned": len(planned_runs),
        "completed_count": completed_count,
        "done_count": done_count,
        "failed_count": failed_count,
        "cancelled_count": cancelled_count,
        "known_implied_return_total": implied_known_total,
        "unknown_implied_return_total": implied_unknown_total,
        "dominant_blockers": {
            "valuation_reason_counts_top": _top_counts(valuation_reasons, top_n=5),
            "price_unknown_reason_counts_top": _top_counts(price_reasons, top_n=5),
        },
        "completed_runs": completed_runs,
        "updated_at": str(state.get("updated_at") or utc_now_iso()),
    }


def _write_state_and_summary(paths: dict[str, Path], state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now_iso()
    _json_write(paths["state_path"], state)
    _json_write(paths["summary_path"], _summarize_state(state))


def _effective_option(
    *,
    override: Any,
    plan_item: dict[str, Any],
    key: str,
    fallback: int,
) -> int:
    if _is_num(override):
        return max(1, int(override))
    recommended = plan_item.get("recommended_flags") if isinstance(plan_item.get("recommended_flags"), dict) else {}
    if _is_num(recommended.get(key)):
        return max(1, int(recommended.get(key)))
    return max(1, int(fallback))


def _effective_run_context(
    *,
    plan_item: dict[str, Any],
    mode: str,
    iterations: int | None,
    top_k: int | None,
    workers: int | None,
    with_prices: bool,
    llm_budget: float | None,
) -> dict[str, Any]:
    tickers = _dedupe_tickers_keep_order(list(plan_item.get("tickers") or []))
    peer_limit = max(1, len(tickers))
    iterations_eff = _effective_option(override=iterations, plan_item=plan_item, key="iterations", fallback=2)
    top_k_eff = _effective_option(override=top_k, plan_item=plan_item, key="top_k", fallback=min(10, peer_limit))
    workers_eff = max(1, int(workers)) if _is_num(workers) else 4
    min_peers_eff = _effective_option(
        override=None,
        plan_item=plan_item,
        key="min_peers_dossierable",
        fallback=min(peer_limit, top_k_eff),
    )
    limit_dossiers_eff = _effective_option(
        override=None,
        plan_item=plan_item,
        key="limit_dossiers",
        fallback=peer_limit,
    )
    min_peers_eff = max(1, min(min_peers_eff, peer_limit))
    limit_dossiers_eff = max(1, min(limit_dossiers_eff, peer_limit))
    return {
        "mode": str(mode or "depth").lower(),
        "iterations": iterations_eff,
        "top_k": top_k_eff,
        "workers": workers_eff,
        "peer_limit": peer_limit,
        "min_peers_dossierable": min_peers_eff,
        "limit_dossiers": limit_dossiers_eff,
        "with_prices": bool(with_prices),
        "llm_budget": float(llm_budget) if _is_num(llm_budget) else None,
    }


def _run_state_loop(
    *,
    state: dict[str, Any],
    paths: dict[str, Path],
    mode: str,
    iterations: int | None,
    top_k: int | None,
    workers: int | None,
    with_prices: bool,
    llm_budget: float | None,
    sec_budget: int | None,
    max_runs_this_invocation: int | None,
) -> dict[str, Any]:
    planned_runs = [row for row in (state.get("planned_runs") or []) if isinstance(row, dict)]
    completed_runs = [row for row in (state.get("completed_runs") or []) if isinstance(row, dict)]
    cursor = max(0, int(state.get("cursor_next_idx") or 0))
    cap_remaining = int(max_runs_this_invocation) if _is_num(max_runs_this_invocation) and int(max_runs_this_invocation) > 0 else None

    state["status"] = BATCH_STATUS_RUNNING
    state["stop_reason_code"] = ""
    state["stop_summary"] = ""
    _write_state_and_summary(paths, state)
    _append_jsonl(
        paths["log_path"],
        {
            "ts": utc_now_iso(),
            "event": "batch_run_started",
            "batch_run_id": state.get("batch_run_id"),
            "cursor_next_idx": cursor,
            "max_runs_this_invocation": cap_remaining,
        },
    )

    for idx in range(cursor, len(planned_runs)):
        if cap_remaining is not None and cap_remaining <= 0:
            state["status"] = BATCH_STATUS_PARTIAL
            state["stop_reason_code"] = STOP_MAX_RUNS_REACHED
            state["stop_summary"] = "Invocation run cap reached."
            break

        latest_state = _safe_json(paths["state_path"])
        if str(latest_state.get("status") or "").upper() == BATCH_STATUS_CANCELLED:
            state = latest_state
            state["stop_reason_code"] = STOP_CANCEL_REQUESTED
            if not str(state.get("stop_summary") or "").strip():
                state["stop_summary"] = "Batch cancelled by operator."
            break

        plan_item = planned_runs[idx]
        plan_item["status"] = PLAN_STATUS_RUNNING
        plan_item["started_at"] = utc_now_iso()

        context = _effective_run_context(
            plan_item=plan_item,
            mode=mode,
            iterations=iterations,
            top_k=top_k,
            workers=workers,
            with_prices=with_prices,
            llm_budget=llm_budget,
        )
        plan_item["cmd"] = _build_sector_rlm_command(
            run_id=str(plan_item.get("run_id") or ""),
            sector=str(plan_item.get("sector") or "UNKNOWN_SECTOR"),
            as_of_date=str(plan_item.get("as_of_date") or ""),
            tickers=_dedupe_tickers_keep_order(list(plan_item.get("tickers") or [])),
            mode=str(context["mode"]),
            iterations=int(context["iterations"]),
            top_k=int(context["top_k"]),
            peer_limit=int(context["peer_limit"]),
            min_peers_dossierable=int(context["min_peers_dossierable"]),
            limit_dossiers=int(context["limit_dossiers"]),
            workers=int(context["workers"]),
            with_prices=bool(context["with_prices"]),
            llm_budget=context["llm_budget"],
        )

        _write_state_and_summary(paths, state)
        _append_jsonl(
            paths["log_path"],
            {
                "ts": utc_now_iso(),
                "event": "depth_run_started",
                "idx": int(idx),
                "run_id": plan_item.get("run_id"),
                "sector": plan_item.get("sector"),
                "tickers": plan_item.get("tickers"),
            },
        )

        execution_error: str | None = None
        execution_payload: dict[str, Any] = {}
        try:
            from app.rlm.loop import sector_rlm_status

            existing_status_payload = sector_rlm_status(run_id=str(plan_item.get("run_id") or ""))
            existing_status = str(existing_status_payload.get("status") or "").upper()
            if existing_status in _RUN_TERMINAL_STATUSES:
                plan_item["status"] = PLAN_STATUS_SKIPPED_EXISTING
            else:
                execution_payload = _execute_plan_item(
                    plan_item=plan_item,
                    mode=str(context["mode"]),
                    iterations=int(context["iterations"]),
                    top_k=int(context["top_k"]),
                    workers=int(context["workers"]),
                    with_prices=bool(context["with_prices"]),
                    llm_budget=context["llm_budget"],
                )
            metrics = _collect_run_metrics(run_id=str(plan_item.get("run_id") or ""))
            run_status = str(metrics.get("status") or "UNKNOWN").upper()
            if plan_item["status"] != PLAN_STATUS_SKIPPED_EXISTING:
                if run_status in {"DONE", "COMPLETED"}:
                    plan_item["status"] = PLAN_STATUS_DONE
                elif run_status == "CANCELLED":
                    plan_item["status"] = PLAN_STATUS_CANCELLED
                elif run_status == "FAILED":
                    plan_item["status"] = PLAN_STATUS_FAILED
                else:
                    plan_item["status"] = PLAN_STATUS_DONE
            completed_runs.append(
                {
                    **metrics,
                    "idx": int(idx),
                    "sector": str(plan_item.get("sector") or ""),
                    "tickers": _dedupe_tickers_keep_order(list(plan_item.get("tickers") or [])),
                    "started_at": str(plan_item.get("started_at") or ""),
                    "finished_at": utc_now_iso(),
                    "execution_status": str(execution_payload.get("status") or ""),
                    "execution_stop_reason_code": str(execution_payload.get("stop_reason_code") or ""),
                }
            )
        except Exception as exc:  # noqa: BLE001
            execution_error = str(exc)
            plan_item["status"] = PLAN_STATUS_FAILED
            completed_runs.append(
                {
                    "idx": int(idx),
                    "run_id": str(plan_item.get("run_id") or ""),
                    "status": "FAILED",
                    "stop_reason_code": STOP_EXCEPTION,
                    "known_implied_return_count": 0,
                    "unknown_implied_return_count": 0,
                    "pass_watch_fail_counts": {"PASS": 0, "WATCH": 0, "FAIL": 0},
                    "dominant_blockers": {
                        "valuation_reason_counts_top": [],
                        "price_unknown_reason_counts_top": [],
                    },
                    "artifacts_paths": {},
                    "started_at": str(plan_item.get("started_at") or ""),
                    "finished_at": utc_now_iso(),
                    "error": execution_error,
                }
            )
            state["status"] = BATCH_STATUS_PARTIAL
            state["stop_reason_code"] = STOP_EXCEPTION
            state["stop_summary"] = execution_error

        plan_item["finished_at"] = utc_now_iso()
        state["cursor_next_idx"] = int(idx + 1)
        state["planned_runs"] = planned_runs
        state["completed_runs"] = completed_runs
        latest_after_item = _safe_json(paths["state_path"])
        if str(latest_after_item.get("status") or "").upper() == BATCH_STATUS_CANCELLED:
            state["status"] = BATCH_STATUS_CANCELLED
            state["stop_reason_code"] = str(latest_after_item.get("stop_reason_code") or STOP_CANCEL_REQUESTED)
            state["stop_summary"] = str(latest_after_item.get("stop_summary") or "Batch cancelled by operator.")
        _write_state_and_summary(paths, state)
        _append_jsonl(
            paths["log_path"],
            {
                "ts": utc_now_iso(),
                "event": "depth_run_finished",
                "idx": int(idx),
                "run_id": plan_item.get("run_id"),
                "status": plan_item.get("status"),
                "error": execution_error,
            },
        )

        if execution_error is not None:
            break

        if cap_remaining is not None:
            cap_remaining -= 1

        if plan_item["status"] == PLAN_STATUS_CANCELLED:
            state["status"] = BATCH_STATUS_CANCELLED
            state["stop_reason_code"] = STOP_CANCEL_REQUESTED
            if not str(state.get("stop_summary") or "").strip():
                state["stop_summary"] = "Depth run returned CANCELLED."
            break

    if str(state.get("status") or "").upper() == BATCH_STATUS_RUNNING:
        if int(state.get("cursor_next_idx") or 0) >= len(planned_runs):
            state["status"] = BATCH_STATUS_DONE
            state["stop_reason_code"] = STOP_COMPLETED
            state["stop_summary"] = "Batch completed."
        else:
            state["status"] = BATCH_STATUS_PARTIAL
            if not str(state.get("stop_reason_code") or "").strip():
                state["stop_reason_code"] = STOP_MAX_RUNS_REACHED
                state["stop_summary"] = "Batch paused before completion."

    if str(state.get("status") or "").upper() == BATCH_STATUS_CANCELLED:
        if not str(state.get("stop_reason_code") or "").strip():
            state["stop_reason_code"] = STOP_CANCEL_REQUESTED
        if not str(state.get("stop_summary") or "").strip():
            state["stop_summary"] = "Batch cancelled by operator."

    state["planned_runs"] = planned_runs
    state["completed_runs"] = completed_runs
    state["budgets_used"] = {
        "sec_budget": int(sec_budget) if _is_num(sec_budget) else None,
        "llm_budget": float(llm_budget) if _is_num(llm_budget) else None,
    }
    _write_state_and_summary(paths, state)
    _append_jsonl(
        paths["log_path"],
        {
            "ts": utc_now_iso(),
            "event": "batch_run_finished",
            "batch_run_id": state.get("batch_run_id"),
            "status": state.get("status"),
            "stop_reason_code": state.get("stop_reason_code"),
        },
    )
    summary = _safe_json(paths["summary_path"])
    return {
        "status": "OK",
        "batch_run_id": str(state.get("batch_run_id") or ""),
        "universe_run_id": str(state.get("universe_run_id") or ""),
        "run_status": str(state.get("status") or BATCH_STATUS_PARTIAL),
        "stop_reason_code": str(state.get("stop_reason_code") or STOP_COMPLETED),
        "cursor_next_idx": int(state.get("cursor_next_idx") or 0),
        "completed_count": len(completed_runs),
        "planned_count": len(planned_runs),
        "batch_state_path": str(paths["state_path"]),
        "batch_summary_path": str(paths["summary_path"]),
        "batch_log_path": str(paths["log_path"]),
        "summary": summary,
    }


def run_depth_batch(
    plan: list[dict[str, Any]],
    *,
    batch_run_id: str,
    dry_run: bool,
    max_runs: int | None,
    mode: str = "depth",
    iterations: int | None = None,
    top_k: int | None = None,
    workers: int | None = None,
    budgets: dict[str, Any] | None = None,
    provider_flags: dict[str, Any] | None = None,
    with_prices: bool = True,
    selection_policy: str = "QUEUE_ORDER",
) -> dict[str, Any]:
    if not plan:
        raise ValueError("Batch plan is empty.")
    universe_run_id = str((plan[0] or {}).get("universe_run_id") or "").strip()
    if not universe_run_id:
        raise ValueError("Batch plan missing universe_run_id.")
    paths = _batch_paths(universe_run_id=universe_run_id, batch_run_id=batch_run_id)
    paths["batch_dir"].mkdir(parents=True, exist_ok=True)
    now = utc_now_iso()

    if _is_num(max_runs) and int(max_runs) > 0:
        plan = plan[: int(max_runs)]

    budgets_eff = budgets if isinstance(budgets, dict) else {}
    llm_budget_eff = float(budgets_eff.get("llm_budget")) if _is_num(budgets_eff.get("llm_budget")) else None

    planned_runs: list[dict[str, Any]] = []
    for idx, item in enumerate(plan):
        row = dict(item)
        row["idx"] = int(idx)
        row["status"] = PLAN_STATUS_PENDING
        row["run_id"] = str(row.get("run_id") or f"{batch_run_id}__UNKNOWN_SECTOR__{idx + 1:03d}")
        context = _effective_run_context(
            plan_item=row,
            mode=str(mode or "depth"),
            iterations=int(iterations) if _is_num(iterations) else None,
            top_k=int(top_k) if _is_num(top_k) else None,
            workers=int(workers) if _is_num(workers) else None,
            with_prices=bool(with_prices),
            llm_budget=llm_budget_eff,
        )
        row["cmd"] = _build_sector_rlm_command(
            run_id=str(row.get("run_id") or ""),
            sector=str(row.get("sector") or "UNKNOWN_SECTOR"),
            as_of_date=str(row.get("as_of_date") or ""),
            tickers=_dedupe_tickers_keep_order(list(row.get("tickers") or [])),
            mode=str(context["mode"]),
            iterations=int(context["iterations"]),
            top_k=int(context["top_k"]),
            peer_limit=int(context["peer_limit"]),
            min_peers_dossierable=int(context["min_peers_dossierable"]),
            limit_dossiers=int(context["limit_dossiers"]),
            workers=int(context["workers"]),
            with_prices=bool(context["with_prices"]),
            llm_budget=context["llm_budget"],
        )
        planned_runs.append(row)

    state = {
        "batch_run_id": str(batch_run_id),
        "universe_run_id": universe_run_id,
        "created_at": now,
        "updated_at": now,
        "selection_policy": str(selection_policy or "QUEUE_ORDER").upper(),
        "max_runs": int(max_runs) if _is_num(max_runs) and int(max_runs) > 0 else len(planned_runs),
        "dry_run": bool(dry_run),
        "planned_runs": planned_runs,
        "completed_runs": [],
        "cursor_next_idx": 0,
        "status": BATCH_STATUS_RUNNING,
        "stop_reason_code": "",
        "stop_summary": "",
        "execution_defaults": {
            "mode": str(mode or "depth"),
            "iterations": int(iterations) if _is_num(iterations) else None,
            "top_k": int(top_k) if _is_num(top_k) else None,
            "workers": int(workers) if _is_num(workers) else None,
            "with_prices": bool(with_prices),
            "budgets": budgets if isinstance(budgets, dict) else {},
            "provider_flags": provider_flags if isinstance(provider_flags, dict) else {},
        },
    }

    _write_state_and_summary(paths, state)
    _append_jsonl(
        paths["log_path"],
        {
            "ts": utc_now_iso(),
            "event": "batch_initialized",
            "batch_run_id": batch_run_id,
            "universe_run_id": universe_run_id,
            "planned_count": len(planned_runs),
            "dry_run": bool(dry_run),
        },
    )

    if dry_run:
        state["status"] = BATCH_STATUS_DONE
        state["stop_reason_code"] = STOP_COMPLETED
        state["stop_summary"] = "Dry-run complete; no depth runs executed."
        _write_state_and_summary(paths, state)
        summary = _safe_json(paths["summary_path"])
        return {
            "status": "OK",
            "batch_run_id": batch_run_id,
            "universe_run_id": universe_run_id,
            "run_status": BATCH_STATUS_DONE,
            "stop_reason_code": STOP_COMPLETED,
            "planned_count": len(planned_runs),
            "completed_count": 0,
            "commands": [str(row.get("cmd") or "") for row in planned_runs if str(row.get("cmd") or "").strip()],
            "batch_state_path": str(paths["state_path"]),
            "batch_summary_path": str(paths["summary_path"]),
            "batch_log_path": str(paths["log_path"]),
            "summary": summary,
        }

    budgets_eff = budgets if isinstance(budgets, dict) else {}
    provider_flags_eff = provider_flags if isinstance(provider_flags, dict) else {}
    llm_provider = str(provider_flags_eff.get("llm_provider") or "").strip()
    if llm_provider:
        import os

        os.environ["VOE_LLM_PROVIDER"] = llm_provider
        get_config.cache_clear()

    return _run_state_loop(
        state=state,
        paths=paths,
        mode=str(mode or "depth"),
        iterations=int(iterations) if _is_num(iterations) else None,
        top_k=int(top_k) if _is_num(top_k) else None,
        workers=int(workers) if _is_num(workers) else None,
        with_prices=bool(with_prices),
        llm_budget=float(budgets_eff.get("llm_budget")) if _is_num(budgets_eff.get("llm_budget")) else None,
        sec_budget=int(budgets_eff.get("sec_budget")) if _is_num(budgets_eff.get("sec_budget")) else None,
        max_runs_this_invocation=None,
    )


def run_universe_depth_batch(
    *,
    universe_run_id: str,
    batch_run_id: str | None = None,
    max_runs: int | None = None,
    selection_policy: str = "QUEUE_ORDER",
    dry_run: bool = False,
    mode: str = "depth",
    iterations: int | None = None,
    top_k: int | None = None,
    workers: int | None = None,
    with_prices: bool = True,
    sec_budget: int | None = None,
    llm_budget: float | None = None,
    llm_provider: str | None = None,
) -> dict[str, Any]:
    queue = load_depth_queue(universe_run_id)
    batch_id = str(batch_run_id or _default_batch_run_id())
    plan = create_batch_plan(queue, max_runs=max_runs, selection_policy=selection_policy, batch_run_id=batch_id)
    return run_depth_batch(
        plan,
        batch_run_id=batch_id,
        dry_run=bool(dry_run),
        max_runs=max_runs,
        mode=mode,
        iterations=iterations,
        top_k=top_k,
        workers=workers,
        with_prices=with_prices,
        budgets={
            "sec_budget": int(sec_budget) if _is_num(sec_budget) else None,
            "llm_budget": float(llm_budget) if _is_num(llm_budget) else None,
        },
        provider_flags={"llm_provider": llm_provider} if llm_provider else {},
        selection_policy=selection_policy,
    )


def open_depth_batch_status(*, batch_run_id: str) -> dict[str, Any]:
    paths = _find_batch_paths_by_id(batch_run_id=batch_run_id)
    if not paths:
        return {
            "batch_run_id": batch_run_id,
            "status": "MISSING",
            "message": "batch_state.json not found",
        }
    state = _safe_json(paths["state_path"])
    summary = _safe_json(paths["summary_path"])
    planned_runs = [row for row in (state.get("planned_runs") or []) if isinstance(row, dict)]
    completed_runs = [row for row in (state.get("completed_runs") or []) if isinstance(row, dict)]
    run_status = str(state.get("status") or BATCH_STATUS_RUNNING)
    failures = len([row for row in completed_runs if str(row.get("status") or "").upper() == "FAILED"])
    last_run = completed_runs[-1] if completed_runs else {}
    if run_status in {BATCH_STATUS_RUNNING, BATCH_STATUS_PARTIAL}:
        suggestions = [
            f"python -m app.cli universe-depth-batch-resume --batch-run-id {batch_run_id}",
            f"python -m app.cli universe-depth-batch-cancel --batch-run-id {batch_run_id} --reason 'operator cancel'",
        ]
    elif run_status == BATCH_STATUS_DONE:
        suggestions = []
    else:
        suggestions = [f"python -m app.cli universe-depth-batch --universe-run-id {state.get('universe_run_id')} --batch-run-id {batch_run_id}"]
    return {
        "batch_run_id": batch_run_id,
        "status": "OK",
        "run_status": run_status,
        "stop_reason_code": str(state.get("stop_reason_code") or ""),
        "cursor_next_idx": int(state.get("cursor_next_idx") or 0),
        "completed_count": len(completed_runs),
        "planned_count": len(planned_runs),
        "last_run": {
            "run_id": str(last_run.get("run_id") or ""),
            "status": str(last_run.get("status") or ""),
            "stop_reason_code": str(last_run.get("stop_reason_code") or ""),
        },
        "failures": int(failures),
        "suggested_actions": suggestions,
        "batch_state_path": str(paths["state_path"]),
        "batch_summary_path": str(paths["summary_path"]),
        "batch_log_path": str(paths["log_path"]),
        "summary": summary,
    }


def resume_depth_batch(
    batch_run_id: str,
    *,
    max_runs: int | None = None,
    mode: str | None = None,
    iterations: int | None = None,
    top_k: int | None = None,
    workers: int | None = None,
    with_prices: bool | None = None,
    sec_budget: int | None = None,
    llm_budget: float | None = None,
    llm_provider: str | None = None,
) -> dict[str, Any]:
    paths = _find_batch_paths_by_id(batch_run_id=batch_run_id)
    if not paths:
        raise ValueError(f"Missing batch_state.json for batch_run_id={batch_run_id}")
    state = _safe_json(paths["state_path"])
    if not state:
        raise ValueError(f"Invalid batch state for batch_run_id={batch_run_id}")
    run_status = str(state.get("status") or "").upper()
    if run_status == BATCH_STATUS_DONE:
        summary = _safe_json(paths["summary_path"])
        return {
            "status": "OK",
            "batch_run_id": batch_run_id,
            "run_status": BATCH_STATUS_DONE,
            "summary": "Batch already completed.",
            "batch_state_path": str(paths["state_path"]),
            "batch_summary_path": str(paths["summary_path"]),
            "batch_log_path": str(paths["log_path"]),
            "summary_payload": summary,
        }
    if run_status == BATCH_STATUS_CANCELLED:
        raise ValueError("Batch is CANCELLED. Start a new batch_run_id to execute again.")

    defaults = state.get("execution_defaults") if isinstance(state.get("execution_defaults"), dict) else {}
    budgets = defaults.get("budgets") if isinstance(defaults.get("budgets"), dict) else {}
    provider_flags = defaults.get("provider_flags") if isinstance(defaults.get("provider_flags"), dict) else {}
    mode_eff = str(mode or defaults.get("mode") or "depth")
    with_prices_eff = bool(with_prices) if isinstance(with_prices, bool) else bool(defaults.get("with_prices", True))
    llm_budget_eff = float(llm_budget) if _is_num(llm_budget) else (
        float(budgets.get("llm_budget")) if _is_num(budgets.get("llm_budget")) else None
    )
    sec_budget_eff = int(sec_budget) if _is_num(sec_budget) else (
        int(budgets.get("sec_budget")) if _is_num(budgets.get("sec_budget")) else None
    )
    llm_provider_eff = str(llm_provider or provider_flags.get("llm_provider") or "").strip()
    if llm_provider_eff:
        import os

        os.environ["VOE_LLM_PROVIDER"] = llm_provider_eff
        get_config.cache_clear()

    return _run_state_loop(
        state=state,
        paths=paths,
        mode=mode_eff,
        iterations=int(iterations) if _is_num(iterations) else (
            int(defaults.get("iterations")) if _is_num(defaults.get("iterations")) else None
        ),
        top_k=int(top_k) if _is_num(top_k) else (
            int(defaults.get("top_k")) if _is_num(defaults.get("top_k")) else None
        ),
        workers=int(workers) if _is_num(workers) else (
            int(defaults.get("workers")) if _is_num(defaults.get("workers")) else None
        ),
        with_prices=with_prices_eff,
        llm_budget=llm_budget_eff,
        sec_budget=sec_budget_eff,
        max_runs_this_invocation=int(max_runs) if _is_num(max_runs) and int(max_runs) > 0 else None,
    )


def cancel_depth_batch(batch_run_id: str, reason: str) -> dict[str, Any]:
    paths = _find_batch_paths_by_id(batch_run_id=batch_run_id)
    if not paths:
        return {
            "batch_run_id": batch_run_id,
            "status": "MISSING",
            "message": "batch_state.json not found",
        }
    state = _safe_json(paths["state_path"])
    if not state:
        return {
            "batch_run_id": batch_run_id,
            "status": "MISSING",
            "message": "invalid batch_state.json payload",
        }
    state["status"] = BATCH_STATUS_CANCELLED
    state["stop_reason_code"] = STOP_CANCEL_REQUESTED
    state["stop_summary"] = str(reason or "Cancelled by operator.")
    planned_runs = [row for row in (state.get("planned_runs") or []) if isinstance(row, dict)]
    for row in planned_runs:
        if str(row.get("status") or "").upper() == PLAN_STATUS_RUNNING:
            row["status"] = PLAN_STATUS_CANCELLED
            row["finished_at"] = utc_now_iso()
    state["planned_runs"] = planned_runs
    _write_state_and_summary(paths, state)
    _append_jsonl(
        paths["log_path"],
        {
            "ts": utc_now_iso(),
            "event": "batch_cancelled",
            "batch_run_id": batch_run_id,
            "reason": state["stop_summary"],
        },
    )
    return {
        "batch_run_id": batch_run_id,
        "status": "OK",
        "run_status": BATCH_STATUS_CANCELLED,
        "stop_reason_code": STOP_CANCEL_REQUESTED,
        "stop_summary": state["stop_summary"],
        "batch_state_path": str(paths["state_path"]),
        "batch_summary_path": str(paths["summary_path"]),
        "batch_log_path": str(paths["log_path"]),
    }
