from __future__ import annotations

import json
import os
from typing import Any, Callable

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.autonomous.v1_financial_context import (
    BoundV1FinancialScope,
    bind_v1_financial_scope,
    build_canonical_v1_financial_context,
    financial_input_scenario,
)
from app.llm.providers.retry_guard import llm_physical_attempt_guard
from app.config import get_config
from app.db import utc_now_iso
from app.llm.providers import get_llm_provider
from app.llm.usage_capture import (
    attach_provider_usage_to_exception,
    provider_failed_attempt_capture,
    provider_usage_records,
    provider_usage_records_from_exception,
    provider_usage_request,
    record_provider_usage,
)
from app.rlm.schemas import Action, PlannerOutput, planner_schema_for_prompt
from app.rlm.state import LoopState
from app.util.hashing import sha256_text


_GATE_PASS = "PASS"
_GATE_WATCH = "WATCH"
_GATE_FAIL = "FAIL"
_REASON_PRICE_UNKNOWN = "PRICE_UNKNOWN"
_REASON_MISSING_INPUT_SHARES = "MISSING_INPUT_SHARES"
_REASON_MISSING_INPUT_FCF = "MISSING_INPUT_FCF"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _load_json(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {}
    try:
        from pathlib import Path

        path = Path(path_value)
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _top_scope_tickers(state: LoopState, *, top_k: int) -> list[str]:
    ordered = [str(t).upper() for t in (state.top_k_current or []) if str(t).strip()]
    if not ordered:
        ordered = [str(t).upper() for t in (state.peer_set or []) if str(t).strip()]
    return ordered[: max(1, int(top_k))]


def _value_gates_entries(state: LoopState) -> list[dict[str, Any]]:
    payload = _load_json(state.artifacts.get("value_gates_path"))
    return [row for row in (payload.get("entries") or []) if isinstance(row, dict)]


def _value_gate_map(state: LoopState) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("ticker") or "").upper(): row
        for row in _value_gates_entries(state)
        if str(row.get("ticker") or "").strip()
    }


def _value_gate_status(row: dict[str, Any] | None) -> str:
    status = str((row or {}).get("gate_status") or _GATE_WATCH).upper().strip()
    if status not in {_GATE_PASS, _GATE_WATCH, _GATE_FAIL}:
        return _GATE_WATCH
    return status


def _value_gate_summary_for_top_k(state: LoopState, *, top_k: int) -> dict[str, Any]:
    gate_by_ticker = _value_gate_map(state)
    top_tickers = _top_scope_tickers(state, top_k=top_k)
    counts = {_GATE_PASS: 0, _GATE_WATCH: 0, _GATE_FAIL: 0}
    watch_price: list[str] = []
    watch_facts: list[str] = []
    top_pass: list[dict[str, Any]] = []
    for ticker in top_tickers:
        row = gate_by_ticker.get(ticker) or {}
        status = str(row.get("gate_status") or _GATE_WATCH).upper()
        if status not in counts:
            status = _GATE_WATCH
        counts[status] += 1
        reasons = {str(item) for item in (row.get("gate_reasons") or [])}
        if status == _GATE_WATCH:
            if _REASON_PRICE_UNKNOWN in reasons:
                watch_price.append(ticker)
            if _REASON_MISSING_INPUT_SHARES in reasons or _REASON_MISSING_INPUT_FCF in reasons:
                watch_facts.append(ticker)
        if status == _GATE_PASS:
            top_pass.append(
                {
                    "ticker": ticker,
                    "valuation_gap": row.get("valuation_gap", "UNKNOWN"),
                }
            )
    top_pass = sorted(
        top_pass,
        key=lambda row: (
            (
                -float(row["valuation_gap"]),
                str(row["ticker"]),
            )
            if isinstance(row.get("valuation_gap"), (int, float))
            else (1e9, str(row["ticker"]))
        ),
    )
    return {
        "top_k_tickers": top_tickers,
        "counts_top_k": counts,
        "watch_price_targets": sorted(set(watch_price)),
        "watch_facts_targets": sorted(set(watch_facts)),
        "top_pass_tickers": top_pass[:10],
    }


def _artifact_summary(state: LoopState) -> dict[str, Any]:
    peer_rankings = _load_json(state.artifacts.get("peer_rankings_path"))
    scoreboard = _load_json(state.artifacts.get("peer_scoreboard_path"))
    whale_summary = _load_json(state.artifacts.get("whale_signals_summary_path"))
    value_gates = _load_json(state.artifacts.get("value_gates_path"))
    value_rank = [
        str(t).upper() for t in (peer_rankings.get("value_first_rank") or []) if str(t).strip()
    ]
    gate_summary = (
        value_gates.get("summary") if isinstance(value_gates.get("summary"), dict) else {}
    )
    return {
        "peer_count": len(state.peer_set),
        "top_k_current": list(state.top_k_current),
        "ranking_mode": peer_rankings.get("ranking_mode"),
        "value_first_rank_top10": value_rank[:10],
        "future_whale_rank_top10": (peer_rankings.get("future_whale_rank") or [])[:10],
        "whale_signature_rank_top10": (peer_rankings.get("whale_signature_rank") or [])[:10],
        "scoreboard_tickers": [
            str(row.get("ticker"))
            for row in (scoreboard.get("rows") or [])[:15]
            if isinstance(row, dict)
        ],
        "whale_summary_tickers": [
            str(row.get("ticker"))
            for row in (whale_summary.get("rows") or [])[:15]
            if isinstance(row, dict)
        ],
        "value_gates_counts": gate_summary.get("counts")
        if isinstance(gate_summary.get("counts"), dict)
        else {},
        "value_gates_top_pass": [
            {
                "ticker": str(row.get("ticker") or ""),
                "valuation_gap": row.get("valuation_gap", "UNKNOWN"),
            }
            for row in (gate_summary.get("top_pass_tickers") or [])[:10]
            if isinstance(row, dict)
        ],
    }


def _planner_input_payload(state: LoopState, *, top_k: int, mode: str) -> dict[str, Any]:
    latest_progress = state.progress_history[-1] if state.progress_history else {}
    value_gate_summary = _value_gate_summary_for_top_k(state, top_k=top_k)
    return {
        "rlm_version": "v1.1" if str(mode).lower() == "depth" else "v0",
        "mode": str(mode).lower(),
        "run_id": state.run_id,
        "sector": state.sector,
        "as_of_date": state.as_of_date,
        "iteration": int(state.iteration),
        "top_k": int(top_k),
        "budgets_remaining": state.budgets_remaining.model_dump(mode="json"),
        "no_progress_streak": int(state.no_progress_streak),
        "ranking_stable_streak": int(state.ranking_stable_streak),
        "gap_summary": state.gap_summary,
        "evidence_delta_counters": state.evidence_delta_counters,
        "rubric_weights": state.rubric_weights,
        "value_gate_summary_top_k": value_gate_summary,
        "artifact_summary": _artifact_summary(state),
        "latest_progress": latest_progress if isinstance(latest_progress, dict) else {},
    }


def _build_prompt(inputs: dict[str, Any], schema: dict[str, Any], *, mode: str) -> str:
    del schema
    mode_norm = str(mode).lower()
    if mode_norm == "depth":
        action_list = (
            "REFINE_PEER_SET, BUILD_DOSSIERS, BUILD_FUNDAMENTALS, VALUE_TICKER, "
            "PREWARM_PRICES, HYDRATE_PRICE_SNAPSHOT, HYDRATE_SHARES, HYDRATE_FCF, HYDRATE_FINANCIAL_FACTS, RECOMPUTE_VALUATION, REWEIGHT_RUBRIC, NARROW_PEER_SET, "
            "BUILD_SCOREBOARD, UPDATE_SCOREBOARD, RUN_RESEARCH_GAP_CLOSER, RUN_SYNTHESIS, "
            "UPDATE_DECISION_PACK, NO_OP, STOP."
        )
    else:
        action_list = (
            "REFINE_PEER_SET, BUILD_DOSSIERS, RUN_WHALE_SIGNALS, "
            "BUILD_SCOREBOARD, RUN_RESEARCH_GAP_CLOSER, RUN_SYNTHESIS, "
            "UPDATE_DECISION_PACK, NO_OP, STOP."
        )
    return (
        "You are Sector RLM Planner.\\n"
        "Return ONLY JSON that matches the PlannerOutput schema.\\n"
        "No prose, markdown, or extra keys.\\n"
        f"Finite action set only: {action_list}\\n"
        "Never invent data sources. Use only currently available run-scoped artifacts and budgets.\\n"
        "At most 6 actions. If no useful action remains under budgets, use NO_OP and/or STOP with reason_code.\\n\\n"
        f"INPUT_PAYLOAD:\\n{json.dumps(inputs, sort_keys=True)}"
    )


def _bind_planner_financial_scope(
    *,
    state: LoopState,
    planner_inputs: dict[str, Any],
    top_k: int,
    provider_request: dict[str, Any],
) -> tuple[BoundV1FinancialScope, Callable[[], tuple[Any, ...]]]:
    gate_map = _value_gate_map(state)
    tickers = _top_scope_tickers(state, top_k=top_k)
    if not tickers:
        # An empty initial iteration has no canonical finance-bearing scope.
        # Bind the literal empty scope so the deterministic gate emits
        # NEEDS_DATA; ``None`` is never authorization for planning or fallback.
        bound_scope = bind_v1_financial_scope(
            context=f"rlm_planner:{state.run_id}:{state.iteration}",
            run_as_of_date=state.as_of_date,
            packets=(),
            scenarios=(),
        )
        return bound_scope, lambda: ()
    context = build_canonical_v1_financial_context(
        tickers=tickers,
        as_of_date=state.as_of_date,
        db_path=get_config().db_path,
    )

    def current_scenarios() -> tuple[Any, ...]:
        return tuple(
            financial_input_scenario(
                context.packets.get(ticker)
                or {
                    "ticker": ticker,
                    "quote_snapshot_id": None,
                    "current_price": None,
                    "current_price_unit": None,
                    "price_basis": None,
                },
                financial_inputs={
                    "planner_inputs": planner_inputs,
                    "value_gate_entry": gate_map.get(ticker),
                    "provider_request": provider_request,
                },
            )
            for ticker in tickers
        )

    bound_scope = bind_v1_financial_scope(
        context=f"rlm_planner:{state.run_id}:{state.iteration}",
        run_as_of_date=state.as_of_date,
        packets=tuple(context.packets[ticker] for ticker in tickers if ticker in context.packets),
        scenarios=current_scenarios(),
    )
    return bound_scope, current_scenarios


def _shares_resolution_targets(state: LoopState, *, top_k: int) -> list[str]:
    valuation_cov = _load_json(state.artifacts.get("valuation_coverage_path"))
    entries = [row for row in (valuation_cov.get("entries") or []) if isinstance(row, dict)]
    if not entries:
        return []
    order = [str(t).upper() for t in (state.top_k_current or []) if str(t).strip()]
    if not order:
        order = [str(t).upper() for t in (state.peer_set or []) if str(t).strip()]
    order_rank = {ticker: idx for idx, ticker in enumerate(order)}
    candidates: list[str] = []
    for row in entries:
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        valuation_reason = str(row.get("valuation_reason_code") or "").upper().strip()
        shares_status = str(row.get("shares_status") or "").upper().strip()
        if valuation_reason == "MISSING_SHARES" or shares_status == "UNKNOWN":
            candidates.append(ticker)
    candidates = sorted(
        set(candidates), key=lambda ticker: (order_rank.get(ticker, 10_000), ticker)
    )
    return candidates[: max(1, int(top_k))]


def _fcf_resolution_targets(state: LoopState, *, top_k: int) -> list[str]:
    valuation_cov = _load_json(state.artifacts.get("valuation_coverage_path"))
    entries = [row for row in (valuation_cov.get("entries") or []) if isinstance(row, dict)]
    if not entries:
        return []
    order = [str(t).upper() for t in (state.top_k_current or []) if str(t).strip()]
    if not order:
        order = [str(t).upper() for t in (state.peer_set or []) if str(t).strip()]
    order_rank = {ticker: idx for idx, ticker in enumerate(order)}
    candidates: list[str] = []
    for row in entries:
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        valuation_reason = str(row.get("valuation_reason_code") or "").upper().strip()
        fcf_status = str(row.get("fcf_status") or "").upper().strip()
        if valuation_reason == "MISSING_FCF" or fcf_status == "UNKNOWN":
            candidates.append(ticker)
    candidates = sorted(
        set(candidates), key=lambda ticker: (order_rank.get(ticker, 10_000), ticker)
    )
    return candidates[: max(1, int(top_k))]


def _facts_hydration_targets(state: LoopState, *, top_k: int) -> list[str]:
    valuation_cov = _load_json(state.artifacts.get("valuation_coverage_path"))
    entries = [row for row in (valuation_cov.get("entries") or []) if isinstance(row, dict)]
    if not entries:
        return []
    order = [str(t).upper() for t in (state.top_k_current or []) if str(t).strip()]
    if not order:
        order = [str(t).upper() for t in (state.peer_set or []) if str(t).strip()]
    order_rank = {ticker: idx for idx, ticker in enumerate(order)}
    candidates: list[str] = []
    for row in entries:
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        price_status = str(row.get("price_status") or "UNKNOWN").upper().strip()
        valuation_reason = str(row.get("valuation_reason_code") or "").upper().strip()
        shares_missing = (
            valuation_reason == "MISSING_SHARES"
            or str(row.get("shares_status") or "").upper().strip() == "UNKNOWN"
        )
        fcf_missing = (
            valuation_reason == "MISSING_FCF"
            or str(row.get("fcf_status") or "").upper().strip() == "UNKNOWN"
        )
        if price_status == "OK" and (shares_missing or fcf_missing):
            candidates.append(ticker)
    candidates = sorted(
        set(candidates), key=lambda ticker: (order_rank.get(ticker, 10_000), ticker)
    )
    return candidates[: max(1, int(top_k))]


def _price_prewarm_targets(state: LoopState, *, top_k: int) -> list[str]:
    top_ranked = [str(t).upper() for t in (state.top_k_current or []) if str(t).strip()][
        : max(1, int(top_k))
    ]
    if not top_ranked:
        top_ranked = [str(t).upper() for t in (state.peer_set or []) if str(t).strip()][
            : max(1, int(top_k))
        ]
    scoreboard = _load_json(state.artifacts.get("peer_scoreboard_path"))
    scoreboard_rows = [row for row in (scoreboard.get("rows") or []) if isinstance(row, dict)]
    scoreboard_tickers = [
        str(row.get("ticker") or "").upper()
        for row in scoreboard_rows
        if str(row.get("ticker") or "").strip()
    ]
    candidates: list[str] = []
    for ticker in (
        top_ranked
        + scoreboard_tickers
        + [str(t).upper() for t in (state.peer_set or []) if str(t).strip()]
    ):
        if ticker and ticker not in candidates:
            candidates.append(ticker)
    return candidates


def _price_hydration_targets(state: LoopState, *, top_k: int) -> list[str]:
    valuation_cov = _load_json(state.artifacts.get("valuation_coverage_path"))
    entries = [row for row in (valuation_cov.get("entries") or []) if isinstance(row, dict)]
    order = [str(t).upper() for t in (state.top_k_current or []) if str(t).strip()]
    if not order:
        order = [str(t).upper() for t in (state.peer_set or []) if str(t).strip()]
    top_ranked = order[: max(1, int(top_k))]
    top_ranked_set = set(top_ranked)
    order_rank = {ticker: idx for idx, ticker in enumerate(top_ranked)}
    if not entries:
        fallback = _price_prewarm_targets(state, top_k=top_k)
        return fallback
    candidates: list[str] = []
    for row in entries:
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker or (top_ranked_set and ticker not in top_ranked_set):
            continue
        price_status = str(row.get("price_status") or "UNKNOWN").upper().strip()
        valuation_reason = str(row.get("valuation_reason_code") or "").upper().strip()
        if price_status != "OK" or valuation_reason == "PRICE_UNKNOWN":
            candidates.append(ticker)
    candidates = sorted(
        set(candidates), key=lambda ticker: (order_rank.get(ticker, 10_000), ticker)
    )
    return candidates


def _should_prewarm_prices(state: LoopState, *, iteration: int, top_k: int) -> bool:
    price_cov = _load_json(state.artifacts.get("price_coverage_path"))
    entries = [row for row in (price_cov.get("entries") or []) if isinstance(row, dict)]
    if not entries:
        return int(iteration) == 0
    valuation_cov = _load_json(state.artifacts.get("valuation_coverage_path"))
    valuation_entries = [
        row for row in (valuation_cov.get("entries") or []) if isinstance(row, dict)
    ]
    if valuation_entries:
        ranked = [str(t).upper() for t in (state.top_k_current or []) if str(t).strip()]
        if not ranked:
            ranked = [str(t).upper() for t in (state.peer_set or []) if str(t).strip()]
        ranked = ranked[: max(1, int(top_k))]
        ranked_set = set(ranked)
        for row in valuation_entries:
            ticker = str(row.get("ticker") or "").upper().strip()
            if ranked_set and ticker not in ranked_set:
                continue
            price_status = str(row.get("price_status") or "UNKNOWN").upper().strip()
            valuation_reason = str(row.get("valuation_reason_code") or "").upper().strip()
            if price_status != "OK" or valuation_reason == "PRICE_UNKNOWN":
                return True
    unknown_count = 0
    for row in entries:
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        status = str((result or {}).get("status") or "UNKNOWN").upper()
        if status != "OK":
            unknown_count += 1
    return unknown_count >= max(1, int(len(entries) * 0.5))


def _last_iteration_attempted_tickers(state: LoopState, *, action_types: set[str]) -> set[str]:
    if not state.action_history:
        return set()
    action_types_norm = {str(item).upper().strip() for item in action_types if str(item).strip()}
    if not action_types_norm:
        return set()
    last = state.action_history[-1] if state.action_history else {}
    attempted: set[str] = set()

    execution = last.get("execution") if isinstance(last, dict) else {}
    results = execution.get("results") if isinstance(execution, dict) else []
    for result in results if isinstance(results, list) else []:
        if not isinstance(result, dict):
            continue
        effective = str(result.get("effective_action_type") or "").upper().strip()
        original = str(result.get("action_type") or "").upper().strip()
        if effective not in action_types_norm and original not in action_types_norm:
            continue
        details = result.get("details") if isinstance(result.get("details"), dict) else {}
        for key in ("target_tickers", "tickers_requested", "tickers"):
            values = details.get(key)
            if isinstance(values, list):
                attempted.update(
                    str(symbol).upper().strip() for symbol in values if str(symbol).strip()
                )
        nested_summary = details.get("summary") if isinstance(details.get("summary"), dict) else {}
        nested_requested = nested_summary.get("tickers_requested")
        if isinstance(nested_requested, list):
            attempted.update(
                str(symbol).upper().strip() for symbol in nested_requested if str(symbol).strip()
            )

    planner = last.get("planner") if isinstance(last, dict) else {}
    planner_actions = planner.get("actions") if isinstance(planner, dict) else []
    for action in planner_actions if isinstance(planner_actions, list) else []:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("action_type") or "").upper().strip()
        if action_type not in action_types_norm:
            continue
        values = action.get("tickers")
        if isinstance(values, list):
            attempted.update(
                str(symbol).upper().strip() for symbol in values if str(symbol).strip()
            )
    return attempted


def _attempted_tickers_in_history(state: LoopState, *, action_types: set[str]) -> set[str]:
    action_types_norm = {str(item).upper().strip() for item in action_types if str(item).strip()}
    if not action_types_norm:
        return set()
    attempted: set[str] = set()
    for row in state.action_history:
        if not isinstance(row, dict):
            continue
        execution = row.get("execution") if isinstance(row.get("execution"), dict) else {}
        results = execution.get("results") if isinstance(execution.get("results"), list) else []
        for result in results:
            if not isinstance(result, dict):
                continue
            effective = str(result.get("effective_action_type") or "").upper().strip()
            original = str(result.get("action_type") or "").upper().strip()
            if effective not in action_types_norm and original not in action_types_norm:
                continue
            details = result.get("details") if isinstance(result.get("details"), dict) else {}
            for key in ("target_tickers", "tickers_requested", "tickers"):
                values = details.get(key)
                if isinstance(values, list):
                    attempted.update(
                        str(symbol).upper().strip() for symbol in values if str(symbol).strip()
                    )
            summary = details.get("summary") if isinstance(details.get("summary"), dict) else {}
            nested_requested = summary.get("tickers_requested")
            if isinstance(nested_requested, list):
                attempted.update(
                    str(symbol).upper().strip()
                    for symbol in nested_requested
                    if str(symbol).strip()
                )
    return attempted


def _planner_network_disabled() -> bool:
    # VOE_NET_PROVIDER is the only network switch; a disabled LLM provider
    # does not make market-data or SEC fetches offline.
    cfg = get_config()
    net_provider = str(os.getenv("VOE_NET_PROVIDER", cfg.net_provider)).strip().lower()
    return net_provider == "disabled"


def _price_terminal_offline_tickers(state: LoopState, *, top_k: int) -> list[str]:
    valuation_cov = _load_json(state.artifacts.get("valuation_coverage_path"))
    valuation_entries = [
        row for row in (valuation_cov.get("entries") or []) if isinstance(row, dict)
    ]
    if not valuation_entries:
        return []
    price_cov = _load_json(state.artifacts.get("price_coverage_path"))
    price_entries = [row for row in (price_cov.get("entries") or []) if isinstance(row, dict)]
    price_by_ticker = {
        str(row.get("ticker") or "").upper().strip(): row
        for row in price_entries
        if str(row.get("ticker") or "").strip()
    }
    ranked = [str(t).upper() for t in (state.top_k_current or []) if str(t).strip()]
    if not ranked:
        ranked = [str(t).upper() for t in (state.peer_set or []) if str(t).strip()]
    ranked = ranked[: max(1, int(top_k))]
    ranked_set = set(ranked)
    ranked_idx = {ticker: idx for idx, ticker in enumerate(ranked)}

    terminal: list[str] = []
    for row in valuation_entries:
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        if ranked_set and ticker not in ranked_set:
            continue
        price_status = str(row.get("price_status") or "UNKNOWN").upper().strip()
        price_reason = str(row.get("price_reason_code") or "").upper().strip()
        if price_status == "OK" or price_reason != "OFFLINE_NO_CACHE":
            continue
        price_row = price_by_ticker.get(ticker, {})
        result = price_row.get("result") if isinstance(price_row.get("result"), dict) else {}
        local = (
            price_row.get("local_fallbacks")
            if isinstance(price_row.get("local_fallbacks"), dict)
            else {}
        )
        terminal_flag = bool((result or {}).get("terminal"))
        local_checked = bool(
            local.get("run_scoped_output_checked")
            and local.get("disk_cache_checked")
            and local.get("db_quote_cache_checked")
            and local.get("historical_run_artifacts_checked")
        )
        local_miss = not bool(local.get("any_hit"))
        if terminal_flag or (local_checked and local_miss):
            terminal.append(ticker)

    return sorted(set(terminal), key=lambda ticker: (ranked_idx.get(ticker, 10_000), ticker))


def _offline_no_cache_tickers(state: LoopState, *, top_k: int) -> list[str]:
    valuation_cov = _load_json(state.artifacts.get("valuation_coverage_path"))
    entries = [row for row in (valuation_cov.get("entries") or []) if isinstance(row, dict)]
    if not entries:
        return []
    ranked = [str(t).upper() for t in (state.top_k_current or []) if str(t).strip()]
    if not ranked:
        ranked = [str(t).upper() for t in (state.peer_set or []) if str(t).strip()]
    ranked = ranked[: max(1, int(top_k))]
    ranked_set = set(ranked)
    ranked_idx = {ticker: idx for idx, ticker in enumerate(ranked)}
    out: list[str] = []
    for row in entries:
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        if ranked_set and ticker not in ranked_set:
            continue
        price_status = str(row.get("price_status") or "UNKNOWN").upper().strip()
        price_reason = str(row.get("price_reason_code") or "").upper().strip()
        if price_status != "OK" and price_reason == "OFFLINE_NO_CACHE":
            out.append(ticker)
    return sorted(set(out), key=lambda ticker: (ranked_idx.get(ticker, 10_000), ticker))


def _value_gate_buckets(state: LoopState, *, top_k: int) -> tuple[list[str], list[str], list[str]]:
    gate_by_ticker = _value_gate_map(state)
    ordered = _top_scope_tickers(state, top_k=top_k)
    pass_tickers: list[str] = []
    watch_tickers: list[str] = []
    fail_tickers: list[str] = []
    for ticker in ordered:
        row = gate_by_ticker.get(ticker) or {}
        status = str(row.get("gate_status") or _GATE_WATCH).upper()
        if status == _GATE_PASS:
            pass_tickers.append(ticker)
        elif status == _GATE_FAIL:
            fail_tickers.append(ticker)
        else:
            watch_tickers.append(ticker)
    return pass_tickers, watch_tickers, fail_tickers


def _calibration_required_for_top_k(
    state: LoopState,
    *,
    top_k: int,
) -> tuple[bool, list[str]]:
    gate_map = _value_gate_map(state)
    if not gate_map:
        return False, []
    ordered = _top_scope_tickers(state, top_k=top_k)
    if not ordered:
        return False, []
    pass_count = 0
    watch_count = 0
    fail_count = 0
    blocker_counts: dict[str, int] = {}
    for ticker in ordered:
        row = gate_map.get(ticker) or {}
        status = _value_gate_status(row)
        if status == _GATE_PASS:
            pass_count += 1
        elif status == _GATE_FAIL:
            fail_count += 1
            blocker = str(row.get("primary_blocker") or "UNKNOWN")
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        else:
            watch_count += 1
    calibration_required = pass_count == 0 and watch_count == 0 and fail_count > 0
    top_blockers = [
        str(name)
        for name, _count in sorted(blocker_counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:3]
    ]
    return calibration_required, top_blockers


def _value_gate_hydration_targets(state: LoopState, *, top_k: int) -> tuple[list[str], list[str]]:
    gate_by_ticker = _value_gate_map(state)
    ordered = _top_scope_tickers(state, top_k=top_k)
    price_targets: list[str] = []
    facts_targets: list[str] = []
    for ticker in ordered:
        row = gate_by_ticker.get(ticker) or {}
        status = str(row.get("gate_status") or _GATE_WATCH).upper()
        reasons = {str(item) for item in (row.get("gate_reasons") or [])}
        needs_price = _REASON_PRICE_UNKNOWN in reasons
        needs_facts = (
            _REASON_MISSING_INPUT_SHARES in reasons or _REASON_MISSING_INPUT_FCF in reasons
        )
        if status == _GATE_FAIL:
            if needs_price:
                price_targets.append(ticker)
            if needs_facts:
                facts_targets.append(ticker)
            continue
        if status == _GATE_WATCH:
            if needs_price:
                price_targets.append(ticker)
            if needs_facts:
                facts_targets.append(ticker)
    return price_targets, facts_targets


def _fallback_planner_output(state: LoopState, *, top_k: int, mode: str) -> PlannerOutput:
    mode_norm = str(mode).lower()
    planner_notes: list[str] = ["deterministic_disabled_provider_fallback"]
    if (
        mode_norm != "depth"
        and not state.peer_set
        and not state.artifacts.get("peer_rankings_path")
    ):
        actions = [
            Action(
                action_type="STOP",
                reason_code="NO_BASELINE_ARTIFACTS",
                summary="No baseline peer artifacts are available for recursive planning.",
            )
        ]
    elif mode_norm == "depth" and int(state.iteration) == 0:
        target_limit = max(1, int(top_k))
        actions: list[Action] = []
        if not state.peer_set:
            actions.append(
                Action(
                    action_type="REFINE_PEER_SET",
                    peer_mode="hybrid",
                    min_peers_dossierable=max(1, int(target_limit)),
                    min_annual_filings=2,
                )
            )
        prewarm_targets = _price_prewarm_targets(state, top_k=top_k)
        if _should_prewarm_prices(state, iteration=int(state.iteration), top_k=top_k):
            actions.append(
                Action(
                    action_type="HYDRATE_PRICE_SNAPSHOT",
                    tickers=prewarm_targets,
                    limit=max(1, len(prewarm_targets) or target_limit),
                    fallback_days=max(0, int(get_config().price_fallback_days)),
                )
            )
        actions.extend(
            [
                Action(
                    action_type="BUILD_DOSSIERS",
                    years_back=10,
                    limit=target_limit,
                ),
                Action(action_type="BUILD_FUNDAMENTALS", limit=target_limit),
                Action(action_type="VALUE_TICKER", limit=target_limit),
                Action(action_type="UPDATE_SCOREBOARD", metrics=[]),
            ]
        )
    elif mode_norm == "depth":
        gate_map = _value_gate_map(state)
        gate_available = bool(gate_map)
        pass_gate_targets, watch_gate_targets, fail_gate_targets = _value_gate_buckets(
            state, top_k=top_k
        )
        calibration_required, calibration_blockers = _calibration_required_for_top_k(
            state, top_k=top_k
        )
        if gate_available:
            planner_notes.append(
                f"value_gates_topk:pass={len(pass_gate_targets)},watch={len(watch_gate_targets)},fail={len(fail_gate_targets)}"
            )
        if calibration_required:
            planner_notes.append(
                "calibration_required:top_blockers="
                + ",".join(calibration_blockers if calibration_blockers else ["UNKNOWN"])
            )

        facts_targets = _facts_hydration_targets(state, top_k=top_k)
        shares_targets = _shares_resolution_targets(state, top_k=top_k)
        fcf_targets = _fcf_resolution_targets(state, top_k=top_k)
        price_targets = _price_hydration_targets(state, top_k=top_k)
        if gate_available:
            gate_price_targets, gate_facts_targets = _value_gate_hydration_targets(
                state, top_k=top_k
            )
            if gate_price_targets:
                price_targets = gate_price_targets
            if gate_facts_targets:
                facts_targets = gate_facts_targets
            watch_set = set(watch_gate_targets)
            shares_targets = [ticker for ticker in shares_targets if ticker in watch_set]
            fcf_targets = [ticker for ticker in fcf_targets if ticker in watch_set]

        attempted_price_tickers = _last_iteration_attempted_tickers(
            state,
            action_types={"HYDRATE_PRICE_SNAPSHOT", "PREWARM_PRICES"},
        )
        attempted_price_tickers_all = _attempted_tickers_in_history(
            state,
            action_types={"HYDRATE_PRICE_SNAPSHOT", "PREWARM_PRICES"},
        )
        network_disabled = _planner_network_disabled()
        offline_reason_tickers = (
            set(_offline_no_cache_tickers(state, top_k=top_k)) if network_disabled else set()
        )
        terminal_offline_tickers = (
            set(_price_terminal_offline_tickers(state, top_k=top_k)) if network_disabled else set()
        )
        offline_blocked_tickers = set(terminal_offline_tickers)
        offline_blocked_tickers.update(
            ticker for ticker in offline_reason_tickers if ticker in attempted_price_tickers_all
        )
        if offline_blocked_tickers:
            planner_notes.append(
                f"price_terminal_offline:{','.join(sorted(offline_blocked_tickers))}"
            )
        pending_price_targets = [
            ticker
            for ticker in price_targets
            if ticker not in attempted_price_tickers and ticker not in offline_blocked_tickers
        ]
        actions = []
        recompute_targets: list[str] = []
        if pending_price_targets and _should_prewarm_prices(
            state, iteration=int(state.iteration), top_k=top_k
        ):
            actions.append(
                Action(
                    action_type="HYDRATE_PRICE_SNAPSHOT",
                    tickers=pending_price_targets,
                    limit=max(1, len(pending_price_targets) or max(1, int(top_k))),
                    fallback_days=max(0, int(get_config().price_fallback_days)),
                )
            )
            recompute_targets.extend(pending_price_targets)
        if facts_targets:
            actions.append(
                Action(
                    action_type="HYDRATE_FINANCIAL_FACTS",
                    tickers=facts_targets,
                    limit=len(facts_targets),
                )
            )
            recompute_targets.extend(facts_targets)
        elif shares_targets:
            actions.append(
                Action(
                    action_type="HYDRATE_SHARES", tickers=shares_targets, limit=len(shares_targets)
                )
            )
            recompute_targets.extend(shares_targets)
        if (not facts_targets) and fcf_targets:
            actions.append(
                Action(action_type="HYDRATE_FCF", tickers=fcf_targets, limit=len(fcf_targets))
            )
            recompute_targets.extend(fcf_targets)
        recompute_targets = sorted(set(recompute_targets))
        if recompute_targets:
            actions.append(
                Action(
                    action_type="RECOMPUTE_VALUATION",
                    tickers=recompute_targets,
                    limit=len(recompute_targets),
                )
            )
        if actions:
            actions.extend(
                [
                    Action(action_type="UPDATE_SCOREBOARD", metrics=[]),
                    Action(
                        action_type="STOP",
                        reason_code="DISABLED_PROVIDER_COMPLETED_BASELINE",
                        summary="Disabled provider completed value-gate hydration follow-up pass.",
                    ),
                ]
            )
        elif gate_available and pass_gate_targets:
            pass_targets = pass_gate_targets[: max(1, int(top_k))]
            actions = [
                Action(
                    action_type="BUILD_DOSSIERS",
                    tickers=pass_targets,
                    years_back=10,
                    limit=len(pass_targets),
                ),
                Action(
                    action_type="RUN_RESEARCH_GAP_CLOSER",
                    tickers=pass_targets,
                    limit=len(pass_targets),
                ),
                Action(
                    action_type="RUN_SYNTHESIS",
                    target={"scope": "ticker", "value": pass_targets[0]},
                ),
                Action(action_type="UPDATE_SCOREBOARD", metrics=[]),
                Action(
                    action_type="STOP",
                    reason_code="DISABLED_PROVIDER_COMPLETED_BASELINE",
                    summary="Disabled provider completed PASS-ticker deepening pass.",
                ),
            ]
        elif calibration_required:
            actions = [
                Action(
                    action_type="STOP",
                    reason_code="CALIBRATION_REQUIRED",
                    summary=(
                        "No PASS/WATCH tickers in top-k; inspect value-gates calibration "
                        "artifact before additional depth actions."
                    ),
                )
            ]
        else:
            actions = [
                Action(
                    action_type="STOP",
                    reason_code="DISABLED_PROVIDER_COMPLETED_BASELINE",
                    summary="Disabled provider completed baseline and no value-gate upgrades remain.",
                )
            ]
    elif int(state.iteration) == 0:
        actions = [
            Action(
                action_type="REFINE_PEER_SET",
                peer_mode="hybrid",
                min_peers_dossierable=max(1, int(len(state.peer_set) or top_k)),
                min_annual_filings=2,
            ),
            Action(
                action_type="BUILD_DOSSIERS",
                tickers=list(state.top_k_current)
                if state.top_k_current
                else list(state.peer_set[: max(1, int(top_k))]),
                years_back=10,
                limit=max(1, int(top_k)),
            ),
            Action(action_type="RUN_WHALE_SIGNALS"),
            Action(action_type="BUILD_SCOREBOARD", metrics=[]),
            Action(action_type="UPDATE_DECISION_PACK", top_n=max(1, int(top_k))),
            Action(
                action_type="STOP",
                reason_code="DISABLED_PROVIDER_COMPLETED_BASELINE",
                summary="Disabled provider completed deterministic baseline depth loop iteration.",
            ),
        ]
    else:
        actions = [
            Action(
                action_type="STOP",
                reason_code="DISABLED_PROVIDER_COMPLETED_BASELINE",
                summary="Disabled provider baseline already completed.",
            )
        ]
    return PlannerOutput(
        rlm_version="v1.1" if mode_norm == "depth" else "v0",
        iteration=int(state.iteration),
        objective=f"Improve sector ranking quality for top {int(top_k)} candidates.",
        actions=actions,
        notes=";".join(planner_notes),
    )


def _apply_value_gate_policy_to_plan(
    *,
    state: LoopState,
    planner_output: PlannerOutput,
    top_k: int,
    mode: str,
) -> PlannerOutput:
    if str(mode).lower() != "depth":
        return planner_output
    gate_map = _value_gate_map(state)
    if not gate_map:
        return planner_output

    top_scope = _top_scope_tickers(state, top_k=top_k)
    pass_set = {
        ticker
        for ticker, row in gate_map.items()
        if str((row or {}).get("gate_status") or _GATE_WATCH).upper() == _GATE_PASS
    }
    calibration_required, calibration_blockers = _calibration_required_for_top_k(state, top_k=top_k)
    expensive_action_types = {
        "BUILD_DOSSIERS",
        "DEEPEN_DOSSIER",
        "RUN_RESEARCH_GAP_CLOSER",
        "RUN_SYNTHESIS",
    }
    bootstrap_dossiers_allowed = (
        int(state.iteration) == 0
        and not str(state.artifacts.get("dossier_summary_path") or "").strip()
    )

    filtered_actions: list[Action] = []
    stop_action: Action | None = None
    for action in planner_output.actions:
        action_type = str(action.action_type).upper()
        if action_type in {"STOP", "ACTION_STOP"}:
            stop_action = action
            continue
        if action_type not in expensive_action_types:
            filtered_actions.append(action)
            continue
        if bootstrap_dossiers_allowed and action_type == "BUILD_DOSSIERS":
            filtered_actions.append(action)
            continue

        if action_type == "RUN_SYNTHESIS":
            target = action.target or {}
            scope = str(target.get("scope") or "").lower()
            if scope == "ticker":
                ticker = str(target.get("value") or "").upper().strip()
                if ticker not in pass_set:
                    continue
                filtered_actions.append(action)
                continue
            # Sector synthesis is only allowed when at least one top-k PASS ticker exists.
            if not any(ticker in pass_set for ticker in top_scope):
                continue
            filtered_actions.append(action)
            continue

        requested = [
            str(t).upper().strip() for t in (action.tickers or top_scope) if str(t).strip()
        ]
        narrowed = [ticker for ticker in requested if ticker in pass_set]
        if not narrowed:
            continue
        filtered_actions.append(
            action.model_copy(
                update={
                    "tickers": narrowed,
                    "limit": len(narrowed),
                }
            )
        )

    # Keep NO_OP valid if present with other actions.
    if len(filtered_actions) > 1:
        filtered_actions = [action for action in filtered_actions if action.action_type != "NO_OP"]

    if stop_action is not None:
        if calibration_required and str(stop_action.reason_code or "").upper() not in {
            "CALIBRATION_REQUIRED"
        }:
            stop_action = Action(
                action_type="STOP",
                reason_code="CALIBRATION_REQUIRED",
                summary=(
                    "No PASS/WATCH tickers in top-k; inspect value-gates calibration artifact "
                    "before additional depth actions."
                ),
            )
        filtered_actions.append(stop_action)
    if not filtered_actions:
        if calibration_required:
            filtered_actions = [
                Action(
                    action_type="STOP",
                    reason_code="CALIBRATION_REQUIRED",
                    summary=(
                        "No PASS/WATCH tickers in top-k; inspect value-gates calibration artifact "
                        "before additional depth actions."
                    ),
                )
            ]
        else:
            filtered_actions = [
                Action(
                    action_type="STOP",
                    reason_code="VALUE_GATE_POLICY_NO_PASS",
                    summary="No PASS tickers available for expensive depth actions.",
                )
            ]
    if len(filtered_actions) > 6:
        capped = filtered_actions[:6]
        stop_candidates = [
            action for action in filtered_actions if action.action_type in {"STOP", "ACTION_STOP"}
        ]
        if stop_candidates:
            capped = [
                action for action in capped if action.action_type not in {"STOP", "ACTION_STOP"}
            ][:5]
            capped.append(stop_candidates[0])
        filtered_actions = capped

    notes = str(planner_output.notes or "").strip()
    policy_note = "value_gate_policy_enforced"
    if calibration_required:
        blocker_note = ",".join(calibration_blockers if calibration_blockers else ["UNKNOWN"])
        policy_note = f"{policy_note};CALIBRATION_REQUIRED:{blocker_note}"
    updated_notes = f"{notes};{policy_note}" if notes else policy_note
    return PlannerOutput.model_validate(
        {
            "rlm_version": planner_output.rlm_version,
            "iteration": int(planner_output.iteration),
            "objective": planner_output.objective,
            "actions": [action.model_dump(mode="json") for action in filtered_actions],
            "notes": updated_notes,
        }
    )


def generate_plan(
    *, state: LoopState, top_k: int, mode: str = "auto"
) -> tuple[PlannerOutput, dict[str, Any]]:
    schema = planner_schema_for_prompt()
    planner_inputs = _planner_input_payload(state, top_k=top_k, mode=mode)
    prompt = _build_prompt(planner_inputs, schema, mode=mode)
    prompt_hash = sha256_text(prompt)
    input_hash = sha256_text(json.dumps(planner_inputs, sort_keys=True))
    provider_request = {
        "prompt": prompt,
        "schema": schema,
        "schema_name": "sector_rlm_planner_v0",
    }
    financial_scope, current_financial_scenarios = _bind_planner_financial_scope(
        state=state,
        planner_inputs=planner_inputs,
        top_k=top_k,
        provider_request=provider_request,
    )
    provider = get_llm_provider()
    provider_name = getattr(provider, "provider_name", "disabled")
    cfg = get_config()
    model = (
        cfg.openai_model
        if provider_name == "openai"
        else str(getattr(provider, "model", "") or "disabled")
    )

    if not provider.enabled():
        financial_scope.require(scenarios=current_financial_scenarios())
        output = _fallback_planner_output(state, top_k=top_k, mode=mode)
        output = _apply_value_gate_policy_to_plan(
            state=state, planner_output=output, top_k=top_k, mode=mode
        )
        note_tickers: list[str] = []
        notes_text = str(output.notes or "")
        for note in [segment.strip() for segment in notes_text.split(";") if segment.strip()]:
            if not note.startswith("price_terminal_offline:"):
                continue
            _, raw = note.split(":", 1)
            note_tickers.extend([str(t).upper().strip() for t in raw.split(",") if str(t).strip()])
        note_tickers = sorted(set(note_tickers))
        meta = {
            "provider": provider_name,
            "model": model,
            "usage_input_tokens": 0,
            "usage_output_tokens": 0,
            "cost_estimate_usd": 0.0,
            "prompt_hash": prompt_hash,
            "input_hash": input_hash,
            "created_at": utc_now_iso(),
            "fallback": True,
            "price_terminal_offline_tickers": note_tickers,
        }
        return output, meta

    def require_exact_scope(_attempt=None):
        financial_scope.require(scenarios=current_financial_scenarios())

    schema_name = "sector_rlm_planner_v0"
    max_output_tokens = (
        int(cfg.openai_max_output_tokens)
        if provider_name == "openai"
        else int(cfg.anthropic_max_output_tokens)
    )
    provider_request.update(
        {
            "provider": provider_name,
            "model": model,
            "max_output_tokens": max_output_tokens,
        }
    )
    # Provider identity and request limits are part of the exact paid-input
    # authorization, so bind them before the first physical attempt.
    financial_scope, current_financial_scenarios = _bind_planner_financial_scope(
        state=state,
        planner_inputs=planner_inputs,
        top_k=top_k,
        provider_request=provider_request,
    )
    failed_attempts: list[dict[str, Any]] = []
    successful_attempts: list[dict[str, Any]] = []
    try:
        with provider_usage_request(
            provider=provider,
            prompt=prompt,
            schema=schema,
            schema_name=schema_name,
            max_output_tokens=max_output_tokens,
        ) as request_kwargs:
            try:
                with (
                    provider_failed_attempt_capture(
                        provider=provider,
                        prompt=prompt,
                        schema_name=schema_name,
                        estimated_output_tokens=max_output_tokens,
                    ) as failed_attempts,
                    llm_physical_attempt_guard(require_exact_scope),
                ):
                    require_exact_scope()
                    result = provider.synthesize_json(
                        prompt=prompt,
                        schema=schema,
                        schema_name=schema_name,
                        **request_kwargs,
                    )
            except BaseException as exc:
                successful_attempts = provider_usage_records_from_exception(
                    provider=provider,
                    error=exc,
                    prompt=prompt,
                    schema_name=schema_name,
                )
                for usage_record in successful_attempts:
                    record_provider_usage(usage_record)
                attach_provider_usage_to_exception(exc, [*failed_attempts, *successful_attempts])
                try:
                    require_exact_scope()
                except InvalidFinancialInputError as integrity_exc:
                    attach_provider_usage_to_exception(
                        integrity_exc,
                        [*failed_attempts, *successful_attempts],
                    )
                    raise integrity_exc from exc
                raise
            successful_attempts = provider_usage_records(
                provider=provider,
                result=result,
                prompt=prompt,
                schema_name=schema_name,
            )
            for usage_record in successful_attempts:
                record_provider_usage(usage_record)

        payload = json.loads(result.json_text)
        financial_scope.require(scenarios=current_financial_scenarios())
        if not isinstance(payload, dict):
            raise RuntimeError("planner output was not a JSON object")
        payload["rlm_version"] = "v1.1" if str(mode).lower() == "depth" else "v0"
        payload["iteration"] = int(state.iteration)
        try:
            output = PlannerOutput.model_validate(payload)
        except Exception:
            output = PlannerOutput(
                rlm_version="v1.1" if str(mode).lower() == "depth" else "v0",
                iteration=int(state.iteration),
                objective="Planner output failed validation; stopping safely.",
                actions=[
                    Action(
                        action_type="STOP",
                        reason_code="PLANNER_SCHEMA_VALIDATION_FAILED",
                        summary="Planner output failed schema validation; stopped for deterministic safety.",
                    )
                ],
                notes="planner_validation_fallback",
            )
        output = _apply_value_gate_policy_to_plan(
            state=state, planner_output=output, top_k=top_k, mode=mode
        )
    except BaseException as exc:
        attach_provider_usage_to_exception(exc, [*failed_attempts, *successful_attempts])
        raise

    input_tokens = result.usage_input_tokens or _estimate_tokens(prompt)
    output_tokens = result.usage_output_tokens or _estimate_tokens(result.json_text)
    physical_cost = round(
        sum(
            float(record.get("cost_estimate_usd") or 0.0)
            for record in [*failed_attempts, *successful_attempts]
        ),
        6,
    )
    meta = {
        "provider": provider_name,
        "model": result.model,
        "usage_input_tokens": int(input_tokens),
        "usage_output_tokens": int(output_tokens),
        "cost_estimate_usd": physical_cost,
        "provider_usage": [*failed_attempts, *successful_attempts],
        "prompt_hash": prompt_hash,
        "input_hash": input_hash,
        "created_at": utc_now_iso(),
        "fallback": False,
    }
    return output, meta
