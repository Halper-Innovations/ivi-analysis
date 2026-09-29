from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.config import get_config
from app.dossier.peer_report import build_peer_report_from_run
from app.dossier.runner import run_dossier_for_peer_set
from app.dossier.sections import SECTION_PATTERNS
from app.dossier.whale_signals import run_whale_signals_for_run
from app.logging import get_logger
from app.llm.synthesis_agent import run_synthesis_for_ticker
from app.market.price_prewarm import write_prices_prewarm_for_run
from app.research.engine import run_research_gap_closer
from app.rlm.schemas import Action, PlannerOutput
from app.rlm.state import LoopState
from app.sector.decision_pack import build_sector_decision_pack
from app.sector.peer_set import select_sector_peers
from app.sector.taxonomy import load_sector_taxonomy
from app.sector.synthesis import run_sector_synthesis
from app.valuation.engine import write_valuations_for_run
from app.valuation.fcf import write_fcf_coverage_for_run
from app.valuation.facts import write_facts_coverage_for_run
from app.valuation.fundamentals import write_fundamentals_for_run
from app.valuation.rubric import apply_value_first_overlay, sanitize_rubric_weights
from app.valuation.shares import write_shares_coverage_for_run


logger = get_logger(__name__)


class StageTimeoutError(TimeoutError):
    """Raised when an RLM planner/executor stage exceeds its time budget."""


def _run_with_timeout(*, fn: Callable[[], Any], timeout_seconds: float | None) -> Any:
    """Run to a definitive result; underlying I/O owns cancellable timeouts.

    Python cannot safely cancel a worker thread. Abandoning one here allowed
    provider calls and state mutations to continue after the loop had recorded
    a fallback or failure. ``timeout_seconds`` remains an API compatibility
    input, but no longer creates non-cancellable detached work.
    """

    _ = timeout_seconds
    return fn()


LEGACY_ACTION_MAP = {
    "CLOSE_GAPS": "RUN_RESEARCH_GAP_CLOSER",
    "DEEPEN_DOSSIER": "BUILD_DOSSIERS",
    "REBUILD_SCOREBOARD": "BUILD_SCOREBOARD",
    "UPDATE_SCOREBOARD": "BUILD_SCOREBOARD",
    "NARROW_PEER_SET": "REFINE_PEER_SET",
    "ACTION_BUILD_FUNDAMENTALS": "BUILD_FUNDAMENTALS",
    "ACTION_VALUE_TICKER": "VALUE_TICKER",
    "ACTION_REWEIGHT_RUBRIC": "REWEIGHT_RUBRIC",
    "ACTION_NARROW_PEER_SET": "REFINE_PEER_SET",
    "ACTION_STOP": "STOP",
    "RESOLVE_SHARES": "HYDRATE_SHARES",
}
_VALUE_GATE_PASS = "PASS"
_VALUE_GATE_WATCH = "WATCH"
_VALUE_GATE_FAIL = "FAIL"
_VALUE_GATE_REASON_PRICE_UNKNOWN = "PRICE_UNKNOWN"
_VALUE_GATE_REASON_MISSING_INPUT_SHARES = "MISSING_INPUT_SHARES"
_VALUE_GATE_REASON_MISSING_INPUT_FCF = "MISSING_INPUT_FCF"


def _load_peer_rankings(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _top_ranked_tickers(*, state: LoopState, top_k: int) -> list[str]:
    rankings = _load_peer_rankings(state.artifacts.get("peer_rankings_path"))
    ranked = [str(t).upper() for t in (rankings.get("future_whale_rank") or []) if str(t).strip()]
    if ranked:
        return ranked[: max(1, int(top_k))]
    if state.top_k_current:
        return [str(t).upper() for t in state.top_k_current[: max(1, int(top_k))]]
    return [str(t).upper() for t in state.peer_set[: max(1, int(top_k))]]


def _value_gate_map(state: LoopState) -> dict[str, dict[str, Any]]:
    payload = _load_peer_rankings(state.artifacts.get("value_gates_path"))
    out: dict[str, dict[str, Any]] = {}
    for row in (payload.get("entries") or []):
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper().strip()
        if ticker:
            out[ticker] = row
    return out


def _value_gate_status(row: dict[str, Any] | None) -> str:
    status = str((row or {}).get("gate_status") or _VALUE_GATE_WATCH).upper().strip()
    if status not in {_VALUE_GATE_PASS, _VALUE_GATE_WATCH, _VALUE_GATE_FAIL}:
        return _VALUE_GATE_WATCH
    return status


def _value_gate_reasons(row: dict[str, Any] | None) -> set[str]:
    return {str(item) for item in ((row or {}).get("gate_reasons") or [])}


def _apply_depth_value_gate_policy_to_action(
    *,
    state: LoopState,
    action: Action,
    effective_action: str,
    top_k: int,
) -> tuple[Action | None, str | None]:
    gate_map = _value_gate_map(state)
    if not gate_map:
        return action, None
    requested = [str(t).upper().strip() for t in (action.tickers or _top_ranked_tickers(state=state, top_k=top_k)) if str(t).strip()]
    if not requested and effective_action not in {"RUN_SYNTHESIS"}:
        return action, None

    if effective_action == "BUILD_DOSSIERS" and int(state.iteration) == 0 and not str(state.artifacts.get("dossier_summary_path") or "").strip():
        return action, "bootstrap_dossier_build"

    if effective_action in {"BUILD_DOSSIERS", "RUN_RESEARCH_GAP_CLOSER"}:
        narrowed = [ticker for ticker in requested if _value_gate_status(gate_map.get(ticker)) == _VALUE_GATE_PASS]
        if not narrowed:
            return None, "value_gates_no_pass_targets"
        return action.model_copy(update={"tickers": narrowed, "limit": len(narrowed)}), "value_gates_pass_only"

    if effective_action == "RUN_SYNTHESIS":
        target = action.target or {}
        scope = str(target.get("scope") or "").lower()
        if scope == "ticker":
            ticker = str(target.get("value") or "").upper().strip()
            if ticker and _value_gate_status(gate_map.get(ticker)) != _VALUE_GATE_PASS:
                return None, "value_gates_skip_non_pass_synthesis"
        elif scope == "sector":
            top_scope = _top_ranked_tickers(state=state, top_k=top_k)
            if not any(_value_gate_status(gate_map.get(ticker)) == _VALUE_GATE_PASS for ticker in top_scope):
                return None, "value_gates_skip_sector_synthesis_no_pass"
        return action, "value_gates_synthesis_checked"

    if effective_action == "HYDRATE_PRICE_SNAPSHOT":
        narrowed = []
        for ticker in requested:
            row = gate_map.get(ticker)
            status = _value_gate_status(row)
            reasons = _value_gate_reasons(row)
            if status == _VALUE_GATE_WATCH:
                narrowed.append(ticker)
                continue
            if status == _VALUE_GATE_FAIL and _VALUE_GATE_REASON_PRICE_UNKNOWN in reasons:
                narrowed.append(ticker)
        if not narrowed:
            return None, "value_gates_no_price_hydration_targets"
        return action.model_copy(update={"tickers": narrowed, "limit": len(narrowed)}), "value_gates_watch_or_fail_price"

    if effective_action == "HYDRATE_FINANCIAL_FACTS":
        narrowed = []
        for ticker in requested:
            row = gate_map.get(ticker)
            status = _value_gate_status(row)
            reasons = _value_gate_reasons(row)
            if status == _VALUE_GATE_WATCH:
                narrowed.append(ticker)
                continue
            if status == _VALUE_GATE_FAIL and (
                _VALUE_GATE_REASON_MISSING_INPUT_SHARES in reasons or _VALUE_GATE_REASON_MISSING_INPUT_FCF in reasons
            ):
                narrowed.append(ticker)
        if not narrowed:
            return None, "value_gates_no_facts_hydration_targets"
        return action.model_copy(update={"tickers": narrowed, "limit": len(narrowed)}), "value_gates_watch_or_fail_facts"

    if effective_action in {"HYDRATE_SHARES", "HYDRATE_FCF"}:
        narrowed = [ticker for ticker in requested if _value_gate_status(gate_map.get(ticker)) == _VALUE_GATE_WATCH]
        if not narrowed:
            return None, "value_gates_skip_fail_specific_hydration"
        return action.model_copy(update={"tickers": narrowed, "limit": len(narrowed)}), "value_gates_watch_only"

    if effective_action == "RECOMPUTE_VALUATION":
        narrowed = []
        for ticker in requested:
            row = gate_map.get(ticker)
            status = _value_gate_status(row)
            reasons = _value_gate_reasons(row)
            if status == _VALUE_GATE_WATCH:
                narrowed.append(ticker)
                continue
            if status == _VALUE_GATE_FAIL and (
                _VALUE_GATE_REASON_PRICE_UNKNOWN in reasons
                or _VALUE_GATE_REASON_MISSING_INPUT_SHARES in reasons
                or _VALUE_GATE_REASON_MISSING_INPUT_FCF in reasons
            ):
                narrowed.append(ticker)
        if not narrowed:
            return None, "value_gates_no_recompute_targets"
        return action.model_copy(update={"tickers": narrowed, "limit": len(narrowed)}), "value_gates_recompute_watch_or_missing_fail"

    return action, None


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _upsert_sector_summary_artifacts(*, run_dir: Path, updated_artifacts: dict[str, str]) -> None:
    if not updated_artifacts:
        return
    summary_path = run_dir / "sector_summary.json"
    if not summary_path.exists():
        return
    payload = _safe_json(summary_path)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    for key, value in updated_artifacts.items():
        if value:
            artifacts[key] = value
    payload["artifacts"] = artifacts
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_peer_artifacts_to_sector_dir(*, run_dir: Path, peer_report: dict[str, Any]) -> dict[str, str]:
    rankings_path = Path(peer_report["peer_rankings_path"])
    scoreboard_path = Path(peer_report["peer_scoreboard_path"])
    report_path = Path(peer_report["peer_report_path"])

    target_rankings = run_dir / "peer_rankings.json"
    target_scoreboard = run_dir / "peer_scoreboard.json"
    target_report = run_dir / "peer_report.md"

    target_rankings.write_text(rankings_path.read_text(encoding="utf-8"), encoding="utf-8")
    target_scoreboard.write_text(scoreboard_path.read_text(encoding="utf-8"), encoding="utf-8")
    target_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "peer_rankings_path": str(target_rankings),
        "peer_scoreboard_path": str(target_scoreboard),
        "peer_report_path": str(target_report),
    }


def _narrow_peer_set(*, state: LoopState, action: Action, top_k: int) -> dict[str, Any]:
    rules = action.filter_rules or {}
    ranked_payload = _load_peer_rankings(state.artifacts.get("peer_rankings_path"))
    ranking_rows = [row for row in (ranked_payload.get("rankings") or []) if isinstance(row, dict)]
    rank_order = [str(t).upper() for t in (ranked_payload.get("future_whale_rank") or []) if str(t).strip()]
    current = [str(t).upper() for t in state.peer_set if str(t).strip()]
    if not current:
        return {"status": "NOOP", "message": "peer_set empty"}

    ordered = [ticker for ticker in rank_order if ticker in current]
    ordered += [ticker for ticker in sorted(current) if ticker not in ordered]
    filtered = ordered[:]
    reasons: list[str] = []

    keep_top_n = rules.get("keep_top_n")
    if isinstance(keep_top_n, int) and keep_top_n > 0:
        filtered = filtered[: int(keep_top_n)]
        reasons.append(f"keep_top_n={int(keep_top_n)}")

    min_whale_score = rules.get("min_whale_signature_score")
    if isinstance(min_whale_score, (int, float)):
        whale_by_ticker = {
            str(row.get("ticker") or "").upper(): (row.get("whale_signature") or {}).get("score")
            for row in ranking_rows
        }
        filtered = [
            ticker
            for ticker in filtered
            if isinstance(whale_by_ticker.get(ticker), (int, float))
            and float(whale_by_ticker[ticker]) >= float(min_whale_score)
        ]
        reasons.append(f"min_whale_signature_score={float(min_whale_score)}")

    filtered = sorted(
        set(filtered),
        key=lambda ticker: ordered.index(ticker) if ticker in ordered else 10_000,
    )
    state.peer_set = filtered
    state.top_k_current = filtered[: max(1, int(top_k))]
    return {
        "status": "OK",
        "message": "peer set narrowed deterministically",
        "count_before": len(current),
        "count_after": len(filtered),
        "rules_applied": reasons,
    }


def _historical_sector_peer_seed(
    *,
    state: LoopState,
    min_target: int,
    max_peers: int,
) -> tuple[list[str], str | None, str | None]:
    cfg = get_config()
    candidates: list[tuple[str, str, list[str], str]] = []
    for summary_path in sorted(cfg.sectors_dir.glob("*/sector_summary.json")):
        run_dir = summary_path.parent
        seed_run_id = run_dir.name
        if seed_run_id == state.run_id:
            continue
        summary = _safe_json(summary_path)
        if str(summary.get("sector") or "").strip().lower() != str(state.sector).strip().lower():
            continue
        if str(summary.get("as_of_date") or "").strip() != str(state.as_of_date):
            continue
        peer_payload = _safe_json(run_dir / "sector_peers.json")
        selected = [str(t).upper() for t in (peer_payload.get("selected_tickers") or []) if str(t).strip()]
        if len(selected) < max(1, int(min_target)):
            continue
        seed_dossier_dir = cfg.dossiers_dir / seed_run_id
        with_dossier = [ticker for ticker in selected if (seed_dossier_dir / ticker / "dossier.json").exists()]
        without_dossier = [ticker for ticker in selected if ticker not in set(with_dossier)]
        selected = with_dossier + without_dossier
        updated_at = str(summary.get("updated_at") or "")
        candidates.append(
            (
                updated_at,
                seed_run_id,
                selected[: max(1, int(max_peers))],
                str(run_dir / "sector_peers.json"),
            )
        )
    if not candidates:
        return [], None, None
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    _, seed_run_id, selected, source_path = candidates[0]
    return selected, seed_run_id, source_path


def _exact_taxonomy_sector_seed(
    *,
    state: LoopState,
    min_target: int,
    max_peers: int,
) -> list[str]:
    cfg = get_config()
    try:
        taxonomy = load_sector_taxonomy(
            taxonomy_path=cfg.sector_taxonomy_path,
            overrides_path=cfg.sector_overrides_path,
        )
    except Exception:
        return []
    sector_norm = str(state.sector or "").strip().lower()
    selected = sorted(
        {
            str(ticker).upper().strip()
            for ticker, raw_sector in taxonomy.items()
            if str(raw_sector or "").strip().lower() == sector_norm and str(ticker).strip()
        }
    )
    if len(selected) < max(1, int(min_target)):
        return []
    return selected[: max(1, int(max_peers))]


def _historical_sector_run_ids(*, state: LoopState) -> list[str]:
    cfg = get_config()
    rows: list[tuple[str, str]] = []
    for summary_path in sorted(cfg.sectors_dir.glob("*/sector_summary.json")):
        run_id = summary_path.parent.name
        if run_id == state.run_id:
            continue
        summary = _safe_json(summary_path)
        if str(summary.get("sector") or "").strip().lower() != str(state.sector).strip().lower():
            continue
        if str(summary.get("as_of_date") or "").strip() != str(state.as_of_date):
            continue
        rows.append((str(summary.get("updated_at") or ""), run_id))
    rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [run_id for _, run_id in rows]


def _copy_missing_dossiers_from_history(
    *,
    state: LoopState,
    target_tickers: list[str],
    run_dir: Path,
) -> dict[str, str]:
    cfg = get_config()
    seed_summary = _safe_json(run_dir / "peer_selection_summary.json")
    candidate_runs: list[str] = []
    seed_run_id = str(seed_summary.get("seed_run_id") or "").strip()
    if seed_run_id:
        candidate_runs.append(seed_run_id)
    for run_id in _historical_sector_run_ids(state=state):
        if run_id not in candidate_runs:
            candidate_runs.append(run_id)

    copied: dict[str, str] = {}
    if not candidate_runs:
        return copied

    dest_root = cfg.dossiers_dir / state.run_id
    for ticker in [str(symbol).upper() for symbol in target_tickers if str(symbol).strip()]:
        dest_dir = dest_root / ticker
        dest_json = dest_dir / "dossier.json"
        if dest_json.exists():
            continue
        for source_run_id in candidate_runs:
            source_dir = cfg.dossiers_dir / source_run_id / ticker
            source_json = source_dir / "dossier.json"
            if not source_json.exists():
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_json, dest_json)
            source_md = source_dir / "dossier.md"
            if source_md.exists():
                shutil.copy2(source_md, dest_dir / "dossier.md")
            copied[ticker] = source_run_id
            break
    return copied


def _recompute_dossier_summary_lists(summary: dict[str, Any]) -> None:
    results = summary.get("ticker_results") if isinstance(summary.get("ticker_results"), dict) else {}
    built: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    skipped_budget: list[str] = []
    pending: list[str] = []
    for ticker, payload in sorted(results.items()):
        status = str((payload or {}).get("status") or "PENDING").upper()
        if status == "OK":
            built.append(str(ticker).upper())
        elif status == "FAILED":
            failed.append(str(ticker).upper())
        elif status == "SKIPPED":
            skipped.append(str(ticker).upper())
        elif status == "SKIPPED_BUDGET":
            skipped_budget.append(str(ticker).upper())
        else:
            pending.append(str(ticker).upper())
    summary["tickers_built"] = built
    summary["tickers_failed"] = failed
    summary["tickers_skipped"] = skipped
    summary["tickers_skipped_budget"] = skipped_budget
    summary["tickers_pending"] = pending
    summary["status"] = "DONE" if (not failed and not pending and not skipped_budget) else "PARTIAL"


def _apply_historical_dossier_copies(
    *,
    summary: dict[str, Any],
    summary_path: Path,
    state: LoopState,
    target_tickers: list[str],
    copied_sources: dict[str, str],
) -> dict[str, Any]:
    if not copied_sources:
        return summary
    results = summary.get("ticker_results")
    if not isinstance(results, dict):
        results = {}
        summary["ticker_results"] = results

    dossier_root = get_config().dossiers_dir / state.run_id
    for ticker in [str(symbol).upper() for symbol in target_tickers if str(symbol).strip()]:
        entry = results.get(ticker)
        if not isinstance(entry, dict):
            entry = {
                "status": "PENDING",
                "error": None,
                "dossier_json_path": None,
                "dossier_md_path": None,
            }
            results[ticker] = entry
        if ticker in copied_sources:
            ticker_dir = dossier_root / ticker
            entry["status"] = "OK"
            entry["error"] = None
            entry["dossier_json_path"] = str(ticker_dir / "dossier.json")
            md_path = ticker_dir / "dossier.md"
            entry["dossier_md_path"] = str(md_path) if md_path.exists() else None

    copied_rows = summary.get("copied_from_history")
    if not isinstance(copied_rows, list):
        copied_rows = []
    copied_rows.extend(
        [
            {"ticker": ticker, "source_run_id": source_run}
            for ticker, source_run in sorted(copied_sources.items(), key=lambda row: row[0])
        ]
    )
    summary["copied_from_history"] = copied_rows
    _recompute_dossier_summary_lists(summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _refine_peer_set(*, state: LoopState, action: Action, top_k: int) -> dict[str, Any]:
    cfg = get_config()
    run_dir = cfg.sectors_dir / state.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if action.filter_rules:
        return _narrow_peer_set(state=state, action=action, top_k=top_k)

    min_target = int(action.min_peers_dossierable or max(1, len(state.peer_set), int(top_k)))
    max_peers = max(min_target, len(state.peer_set), int(top_k), 25)
    peer_mode = action.peer_mode or "hybrid"
    include_foreign = bool(action.include_foreign) if action.include_foreign is not None else True
    include_otc = bool(action.include_otc) if action.include_otc is not None else False
    min_annual = int(action.min_annual_filings or 2)
    max_peer_scan = int(action.max_peer_scan or max(400, max_peers * 6))

    exact_taxonomy_seed = _exact_taxonomy_sector_seed(
        state=state,
        min_target=min_target,
        max_peers=max_peers,
    )
    if exact_taxonomy_seed:
        state.peer_set = exact_taxonomy_seed
        state.top_k_current = exact_taxonomy_seed[: max(1, int(top_k))]
        peers_path = run_dir / "sector_peers.json"
        summary_path = run_dir / "peer_selection_summary.json"
        seed_payload = {
            "run_id": state.run_id,
            "sector": state.sector,
            "as_of_date": state.as_of_date,
            "selected_tickers": exact_taxonomy_seed,
            "peer_selection_summary": {
                "mode": "exact_taxonomy_seed",
                "selected_count": len(exact_taxonomy_seed),
                "requested_min_peers": int(min_target),
                "max_peers": int(max_peers),
            },
        }
        peers_path.write_text(json.dumps(seed_payload, indent=2), encoding="utf-8")
        summary_path.write_text(
            json.dumps(seed_payload["peer_selection_summary"], indent=2),
            encoding="utf-8",
        )
        return {
            "status": "OK",
            "peer_mode": "exact_taxonomy_seed",
            "selected_count": len(exact_taxonomy_seed),
            "selected_tickers": exact_taxonomy_seed,
            "min_peers_dossierable": min_target,
            "min_annual_filings": min_annual,
            "max_peer_scan": max_peer_scan,
            "peer_selection_summary": seed_payload["peer_selection_summary"],
            "artifacts_written": [str(peers_path), str(summary_path)],
        }

    seeded, seed_run_id, seed_source_path = _historical_sector_peer_seed(
        state=state,
        min_target=min_target,
        max_peers=max_peers,
    )
    if seeded:
        state.peer_set = seeded
        state.top_k_current = seeded[: max(1, int(top_k))]
        peers_path = run_dir / "sector_peers.json"
        summary_path = run_dir / "peer_selection_summary.json"
        seed_payload = {
            "run_id": state.run_id,
            "sector": state.sector,
            "as_of_date": state.as_of_date,
            "selected_tickers": seeded,
            "peer_selection_summary": {
                "mode": "historical_seed",
                "seed_run_id": seed_run_id,
                "seed_source_path": seed_source_path,
                "selected_count": len(seeded),
                "requested_min_peers": int(min_target),
                "max_peers": int(max_peers),
            },
        }
        peers_path.write_text(json.dumps(seed_payload, indent=2), encoding="utf-8")
        summary_path.write_text(
            json.dumps(seed_payload["peer_selection_summary"], indent=2),
            encoding="utf-8",
        )
        return {
            "status": "OK",
            "peer_mode": "historical_seed",
            "selected_count": len(seeded),
            "selected_tickers": seeded,
            "min_peers_dossierable": min_target,
            "min_annual_filings": min_annual,
            "max_peer_scan": max_peer_scan,
            "peer_selection_summary": seed_payload["peer_selection_summary"],
            "artifacts_written": [str(peers_path), str(summary_path)],
        }

    payload = select_sector_peers(
        sector=state.sector,
        as_of_date=state.as_of_date,
        years_back=int(action.years_back or 10),
        limit=max_peers,
        min_peers=min_target,
        max_peers=max_peers,
        mode=peer_mode,
        min_annual_filings=min_annual,
        max_peer_scan=max_peer_scan,
        stop_when_min_reached=True,
        sic_expand=True,
        sic_family=True,
        include_foreign=include_foreign,
        include_otc=include_otc,
    )
    selected = [str(t).upper() for t in (payload.get("selected_tickers") or []) if str(t).strip()]
    state.peer_set = selected
    state.top_k_current = selected[: max(1, int(top_k))]

    peers_path = run_dir / "sector_peers.json"
    summary_path = run_dir / "peer_selection_summary.json"
    peers_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(payload.get("peer_selection_summary") or {}, indent=2),
        encoding="utf-8",
    )

    return {
        "status": "OK",
        "peer_mode": peer_mode,
        "selected_count": len(selected),
        "selected_tickers": selected,
        "min_peers_dossierable": min_target,
        "min_annual_filings": min_annual,
        "max_peer_scan": max_peer_scan,
        "peer_selection_summary": payload.get("peer_selection_summary") or {},
        "artifacts_written": [str(peers_path), str(summary_path)],
    }


def _build_dossiers(*, state: LoopState, action: Action, top_k: int, workers: int, years_back_default: int) -> dict[str, Any]:
    requested = action.tickers or _top_ranked_tickers(state=state, top_k=top_k)
    limit = int(action.limit or len(requested) or max(1, int(top_k)))
    target_tickers = requested[: max(1, limit)]
    cfg = get_config()
    run_dir = cfg.sectors_dir / state.run_id
    dossier_summary = run_dossier_for_peer_set(
        tickers=target_tickers,
        as_of_date=state.as_of_date,
        years_back=int(action.years_back or years_back_default),
        run_id=state.run_id,
        workers=max(1, int(workers)),
        min_annual_filings=int(action.min_annual_filings or 2),
    )
    summary_path = Path(
        str(
            dossier_summary.get("summary_path")
            or (cfg.dossiers_dir / state.run_id / "dossier_summary.json")
        )
    )
    copied_sources = _copy_missing_dossiers_from_history(
        state=state,
        target_tickers=target_tickers,
        run_dir=run_dir,
    )
    if copied_sources:
        dossier_summary = _apply_historical_dossier_copies(
            summary=dossier_summary,
            summary_path=summary_path,
            state=state,
            target_tickers=target_tickers,
            copied_sources=copied_sources,
        )
    _annotate_dossiers_with_price_hydration(state=state, target_tickers=target_tickers)
    return {
        "target_tickers": target_tickers,
        "dossier_summary": dossier_summary,
        "dossier_summary_path": dossier_summary.get("summary_path"),
        "copied_from_history": [
            {"ticker": ticker, "source_run_id": source_run}
            for ticker, source_run in sorted(copied_sources.items(), key=lambda row: row[0])
        ],
    }


def _annotate_dossiers_with_price_hydration(*, state: LoopState, target_tickers: list[str]) -> None:
    cfg = get_config()
    price_root = cfg.outputs_dir / "prices" / state.run_id
    dossier_root = cfg.dossiers_dir / state.run_id
    for ticker in sorted({str(symbol).upper().strip() for symbol in target_tickers if str(symbol).strip()}):
        dossier_path = dossier_root / ticker / "dossier.json"
        if not dossier_path.exists():
            continue
        try:
            dossier_payload = json.loads(dossier_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("failed to read dossier for price hydration annotation: %s", dossier_path)
            continue
        if not isinstance(dossier_payload, dict):
            continue
        price_hydrated = False
        price_snapshot_path = price_root / f"{ticker}.json"
        if price_snapshot_path.exists():
            try:
                price_payload = json.loads(price_snapshot_path.read_text(encoding="utf-8"))
            except Exception:
                price_payload = {}
            snapshot = price_payload.get("snapshot") if isinstance(price_payload, dict) else {}
            price_value = snapshot.get("price") if isinstance(snapshot, dict) else None
            price_hydrated = (
                str((price_payload or {}).get("status") or "").upper() == "OK"
                and isinstance(price_value, (int, float))
                and float(price_value) > 0
            )
        dossier_payload["price_hydrated"] = bool(price_hydrated)
        dossier_path.write_text(json.dumps(dossier_payload, indent=2), encoding="utf-8")


def _build_fundamentals(
    *,
    state: LoopState,
    action: Action,
    top_k: int,
    run_dir: Path,
    years_back_default: int,
) -> dict[str, Any]:
    requested = action.tickers or _top_ranked_tickers(state=state, top_k=top_k)
    limit = int(action.limit or len(requested) or max(1, int(top_k)))
    target_tickers = requested[: max(1, limit)]
    summary = write_fundamentals_for_run(
        run_id=state.run_id,
        tickers=target_tickers,
        years_back=int(action.years_back or years_back_default),
        output_dir=run_dir,
    )
    return {
        "target_tickers": target_tickers,
        "summary": summary,
        "summary_path": summary.get("summary_path"),
    }


def _build_valuations(
    *,
    state: LoopState,
    action: Action,
    top_k: int,
    run_dir: Path,
    with_prices: bool,
) -> dict[str, Any]:
    requested = action.tickers or _top_ranked_tickers(state=state, top_k=top_k)
    limit = int(action.limit or len(requested) or max(1, int(top_k)))
    target_tickers = requested[: max(1, limit)]
    summary = write_valuations_for_run(
        run_id=state.run_id,
        tickers=target_tickers,
        as_of_date=state.as_of_date,
        output_dir=run_dir,
        with_prices=with_prices,
    )
    return {
        "target_tickers": target_tickers,
        "summary": summary,
        "summary_path": summary.get("summary_path"),
        "price_coverage_path": summary.get("price_coverage_path"),
        "shares_coverage_path": summary.get("shares_coverage_path"),
        "fcf_coverage_path": summary.get("fcf_coverage_path"),
        "facts_coverage_path": summary.get("facts_coverage_path"),
    }


def _prewarm_prices(
    *,
    state: LoopState,
    action: Action,
    top_k: int,
) -> dict[str, Any]:
    requested = action.tickers or _top_ranked_tickers(state=state, top_k=top_k)
    limit = int(action.limit or len(requested) or max(1, int(top_k)))
    target_tickers = requested[: max(1, limit)]
    fallback_days = int(action.fallback_days if action.fallback_days is not None else get_config().price_fallback_days)
    summary = write_prices_prewarm_for_run(
        run_id=state.run_id,
        as_of_date=state.as_of_date,
        tickers=target_tickers,
        fallback_days=max(0, fallback_days),
    )
    return {
        "target_tickers": target_tickers,
        "fallback_days": max(0, fallback_days),
        "summary": summary,
        "prices_prewarm_path": summary.get("prices_prewarm_path"),
        "prices_summary_path": summary.get("prices_summary_path"),
    }


def _hydrate_price_snapshot(
    *,
    state: LoopState,
    action: Action,
    top_k: int,
) -> dict[str, Any]:
    details = _prewarm_prices(
        state=state,
        action=action,
        top_k=top_k,
    )
    details["hydration_action"] = "HYDRATE_PRICE_SNAPSHOT"
    return details


def _resolve_shares(
    *,
    state: LoopState,
    action: Action,
    top_k: int,
) -> dict[str, Any]:
    requested = action.tickers or _top_ranked_tickers(state=state, top_k=top_k)
    limit = int(action.limit or len(requested) or max(1, int(top_k)))
    target_tickers = requested[: max(1, limit)]
    summary = write_shares_coverage_for_run(
        as_of_date=state.as_of_date,
        tickers=target_tickers,
        run_id=state.run_id,
    )
    coverage_path = summary.get("shares_coverage_path") or str(get_config().sectors_dir / state.run_id / "shares_coverage.json")
    return {
        "target_tickers": target_tickers,
        "summary": summary,
        "shares_coverage_path": coverage_path,
    }


def _hydrate_fcf(
    *,
    state: LoopState,
    action: Action,
    top_k: int,
) -> dict[str, Any]:
    requested = action.tickers or _top_ranked_tickers(state=state, top_k=top_k)
    limit = int(action.limit or len(requested) or max(1, int(top_k)))
    target_tickers = requested[: max(1, limit)]
    summary = write_fcf_coverage_for_run(
        as_of_date=state.as_of_date,
        tickers=target_tickers,
        run_id=state.run_id,
    )
    coverage_path = summary.get("fcf_coverage_path") or str(get_config().sectors_dir / state.run_id / "fcf_coverage.json")
    return {
        "target_tickers": target_tickers,
        "summary": summary,
        "fcf_coverage_path": coverage_path,
    }


def _hydrate_financial_facts(
    *,
    state: LoopState,
    action: Action,
    top_k: int,
) -> dict[str, Any]:
    requested = action.tickers or _top_ranked_tickers(state=state, top_k=top_k)
    limit = int(action.limit or len(requested) or max(1, int(top_k)))
    target_tickers = requested[: max(1, limit)]
    summary = write_facts_coverage_for_run(
        run_id=state.run_id,
        as_of_date=state.as_of_date,
        tickers=target_tickers,
        output_dir=get_config().sectors_dir / state.run_id,
        cfg=get_config(),
        refresh=True,
    )
    coverage_path = summary.get("facts_coverage_path") or str(get_config().sectors_dir / state.run_id / "facts_coverage.json")
    return {
        "target_tickers": target_tickers,
        "summary": summary,
        "facts_coverage_path": coverage_path,
    }


def _reweight_rubric(*, state: LoopState, action: Action) -> dict[str, Any]:
    requested = action.weights or (
        action.filter_rules.get("weights") if isinstance(action.filter_rules, dict) else {}
    )
    normalized = sanitize_rubric_weights(requested if isinstance(requested, dict) else {})
    state.rubric_weights = normalized
    return {
        "status": "OK",
        "rubric_weights": normalized,
        "derived_from": ["planner.actions[*].weights"],
    }


def _run_whale_signals(*, state: LoopState, run_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    whale_summary = run_whale_signals_for_run(run_id=state.run_id)
    updated: dict[str, str] = {}
    whale_summary_path = Path(str(whale_summary.get("summary_path") or ""))
    if whale_summary_path.exists():
        target = run_dir / "whale_signals_summary.json"
        target.write_text(whale_summary_path.read_text(encoding="utf-8"), encoding="utf-8")
        updated["whale_signals_summary_path"] = str(target)
    return whale_summary, updated


def _build_scoreboard(
    *,
    state: LoopState,
    run_dir: Path,
    mode: str,
    with_prices: bool,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any] | None]:
    peer_report = build_peer_report_from_run(run_id=state.run_id, as_of_date=state.as_of_date)
    updated = _copy_peer_artifacts_to_sector_dir(run_dir=run_dir, peer_report=peer_report)
    overlay_summary: dict[str, Any] | None = None
    if str(mode).lower() == "depth":
        cfg = get_config()
        overlay_summary = apply_value_first_overlay(
            run_id=state.run_id,
            as_of_date=state.as_of_date,
            sector_run_dir=run_dir,
            dossier_run_dir=cfg.dossiers_dir / state.run_id,
            rubric_weights=state.rubric_weights,
            with_prices=with_prices,
        )
    return peer_report, updated, overlay_summary


def _update_decision_pack(
    *,
    state: LoopState,
    run_dir: Path,
    top_n: int,
    delta_path: str | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    decision = build_sector_decision_pack(
        sector=state.sector,
        as_of_date=state.as_of_date,
        run_id=state.run_id,
        top_n=max(1, int(top_n)),
    )
    json_path = Path(str(decision.get("decision_pack_path") or ""))
    md_path = Path(str(decision.get("decision_pack_md_path") or ""))

    delta_payload = _safe_json(Path(delta_path)) if delta_path else {}
    delta_summary = {
        "iteration": int(state.iteration),
        "delta_artifact_path": delta_path,
        "metrics_changed": int(delta_payload.get("metrics_changed", 0)),
        "tickers_changed": int(delta_payload.get("tickers_changed", 0)),
        "derived_from": [delta_path] if delta_path else [],
    }

    if json_path.exists():
        payload = _safe_json(json_path)
        payload["rlm_iteration"] = int(state.iteration)
        payload["rlm_delta_since_last_iter"] = delta_summary
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if md_path.exists():
        lines = [
            "",
            "## Delta Since Last Iter",
            f"- Iteration: `{int(state.iteration)}`",
            f"- Delta artifact: `{delta_path or 'N/A'}`",
            f"- Metrics changed: `{int(delta_summary['metrics_changed'])}`",
            f"- Tickers changed: `{int(delta_summary['tickers_changed'])}`",
        ]
        with md_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    updated = {
        "decision_pack_path": str(json_path) if json_path.exists() else str(run_dir / "decision_pack.json"),
        "decision_pack_md_path": str(md_path) if md_path.exists() else str(run_dir / "decision_pack.md"),
    }
    return {
        "decision": decision,
        "delta_summary": delta_summary,
    }, updated


def execute_actions(
    *,
    state: LoopState,
    planner_output: PlannerOutput,
    top_k: int,
    years_back_default: int,
    workers: int,
    with_research: bool,
    with_synthesis: bool,
    with_prices: bool = True,
    mode: str = "auto",
    timeout_per_stage: float = 30.0,
    progress_hook: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    run_dir = cfg.sectors_dir / state.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    updated_artifacts: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    planner_requested_stop = False
    known_sections = set(SECTION_PATTERNS.keys()) | {"full_document"}

    for idx, action in enumerate(planner_output.actions):
        effective_action = LEGACY_ACTION_MAP.get(action.action_type, action.action_type)
        policy_note: str | None = None
        if str(mode).lower() == "depth":
            gated_action, policy_note = _apply_depth_value_gate_policy_to_action(
                state=state,
                action=action,
                effective_action=effective_action,
                top_k=top_k,
            )
            if gated_action is None:
                result = {
                    "index": idx,
                    "action_type": action.action_type,
                    "effective_action_type": effective_action,
                    "status": "SKIPPED",
                    "details": {
                        "reason": policy_note or "value_gate_policy_skip",
                        "value_gate_policy": policy_note or "value_gate_policy_skip",
                    },
                    "artifacts_written": [],
                }
                results.append(result)
                continue
            action = gated_action
        if progress_hook is not None:
            current_ticker = action.tickers[0] if action.tickers else None
            progress_hook(
                {
                    "index": idx,
                    "action_type": effective_action,
                    "current_ticker": current_ticker,
                }
            )
        result: dict[str, Any] = {
            "index": idx,
            "action_type": action.action_type,
            "effective_action_type": effective_action,
            "status": "OK",
            "details": {},
            "artifacts_written": [],
        }
        if policy_note:
            result["details"]["value_gate_policy"] = policy_note

        def _execute_current_action() -> None:
            nonlocal planner_requested_stop
            if effective_action == "STOP":
                planner_requested_stop = True
                result["details"] = {
                    "reason_code": action.reason_code,
                    "summary": action.summary,
                }
            elif effective_action == "NO_OP":
                result["details"] = {
                    "summary": action.summary,
                    "reason_code": action.reason_code,
                }
            elif effective_action == "REFINE_PEER_SET":
                refined = _refine_peer_set(state=state, action=action, top_k=top_k)
                result["details"] = refined
                result["artifacts_written"] = list(refined.get("artifacts_written") or [])
            elif effective_action == "RUN_RESEARCH_GAP_CLOSER":
                if not with_research:
                    result["status"] = "SKIPPED"
                    result["details"] = {"reason": "with_research disabled"}
                else:
                    target_tickers = action.tickers or _top_ranked_tickers(state=state, top_k=top_k)
                    summary = run_research_gap_closer(
                        as_of_date=state.as_of_date,
                        run_id=state.run_id,
                        limit=int(action.limit or max(1, len(target_tickers))),
                        source_filters=set(action.sources) if action.sources else None,
                        tickers=target_tickers,
                    )
                    result["details"] = {
                        "target_tickers": target_tickers,
                        "summary": summary,
                        "gap_types": action.gap_types,
                    }
            elif effective_action == "BUILD_DOSSIERS":
                recognized = [section for section in action.sections if section in known_sections]
                ignored = [section for section in action.sections if section not in known_sections]
                details = _build_dossiers(
                    state=state,
                    action=action,
                    top_k=top_k,
                    workers=workers,
                    years_back_default=years_back_default,
                )
                details["recognized_sections"] = recognized
                details["ignored_sections"] = ignored
                result["details"] = details
                if details.get("dossier_summary_path"):
                    updated_artifacts["dossier_summary_path"] = str(details["dossier_summary_path"])
                    result["artifacts_written"].append(str(details["dossier_summary_path"]))
            elif effective_action == "BUILD_FUNDAMENTALS":
                details = _build_fundamentals(
                    state=state,
                    action=action,
                    top_k=top_k,
                    run_dir=run_dir,
                    years_back_default=years_back_default,
                )
                result["details"] = details
                summary_path = details.get("summary_path")
                if summary_path:
                    updated_artifacts["fundamentals_summary_path"] = str(summary_path)
                    result["artifacts_written"].append(str(summary_path))
            elif effective_action == "VALUE_TICKER":
                details = _build_valuations(
                    state=state,
                    action=action,
                    top_k=top_k,
                    run_dir=run_dir,
                    with_prices=with_prices,
                )
                result["details"] = details
                summary_payload = details.get("summary") or {}
                if int(summary_payload.get("prices_ok", 0)) == 0 and int(summary_payload.get("prices_unknown", 0)) > 0:
                    logger.warning(
                        "price_missing: implied_return metrics will be UNKNOWN; run with --with-prices (default) and ensure price snapshots are available",
                        extra={
                            "stage_name": "valuation",
                            "run_id": state.run_id,
                            "provider_config": summary_payload.get("price_provider_config"),
                            "provider_effective": summary_payload.get("price_provider_effective"),
                        },
                    )
                summary_path = details.get("summary_path")
                if summary_path:
                    updated_artifacts["valuation_summary_path"] = str(summary_path)
                    result["artifacts_written"].append(str(summary_path))
                coverage_path = details.get("price_coverage_path")
                if coverage_path:
                    updated_artifacts["price_coverage_path"] = str(coverage_path)
                    updated_artifacts["price_coverage"] = str(coverage_path)
                    result["artifacts_written"].append(str(coverage_path))
                shares_coverage_path = details.get("shares_coverage_path")
                if shares_coverage_path:
                    updated_artifacts["shares_coverage_path"] = str(shares_coverage_path)
                    updated_artifacts["shares_coverage"] = str(shares_coverage_path)
                    result["artifacts_written"].append(str(shares_coverage_path))
                fcf_coverage_path = details.get("fcf_coverage_path")
                if fcf_coverage_path:
                    updated_artifacts["fcf_coverage_path"] = str(fcf_coverage_path)
                    updated_artifacts["fcf_coverage"] = str(fcf_coverage_path)
                    result["artifacts_written"].append(str(fcf_coverage_path))
                facts_coverage_path = details.get("facts_coverage_path")
                if facts_coverage_path:
                    updated_artifacts["facts_coverage_path"] = str(facts_coverage_path)
                    updated_artifacts["facts_coverage"] = str(facts_coverage_path)
                    result["artifacts_written"].append(str(facts_coverage_path))
            elif effective_action == "PREWARM_PRICES":
                details = _prewarm_prices(
                    state=state,
                    action=action,
                    top_k=top_k,
                )
                result["details"] = details
                prewarm_path = details.get("prices_prewarm_path")
                if prewarm_path:
                    updated_artifacts["prices_prewarm_path"] = str(prewarm_path)
                    result["artifacts_written"].append(str(prewarm_path))
                prices_summary_path = details.get("prices_summary_path")
                if prices_summary_path:
                    updated_artifacts["prices_summary_path"] = str(prices_summary_path)
                    result["artifacts_written"].append(str(prices_summary_path))
            elif effective_action == "HYDRATE_PRICE_SNAPSHOT":
                details = _hydrate_price_snapshot(
                    state=state,
                    action=action,
                    top_k=top_k,
                )
                result["details"] = details
                prewarm_path = details.get("prices_prewarm_path")
                if prewarm_path:
                    updated_artifacts["prices_prewarm_path"] = str(prewarm_path)
                    result["artifacts_written"].append(str(prewarm_path))
                prices_summary_path = details.get("prices_summary_path")
                if prices_summary_path:
                    updated_artifacts["prices_summary_path"] = str(prices_summary_path)
                    result["artifacts_written"].append(str(prices_summary_path))
            elif effective_action == "HYDRATE_SHARES":
                details = _resolve_shares(
                    state=state,
                    action=action,
                    top_k=top_k,
                )
                result["details"] = details
                shares_coverage_path = details.get("shares_coverage_path")
                if shares_coverage_path:
                    updated_artifacts["shares_coverage_path"] = str(shares_coverage_path)
                    updated_artifacts["shares_coverage"] = str(shares_coverage_path)
                    result["artifacts_written"].append(str(shares_coverage_path))
            elif effective_action == "HYDRATE_FCF":
                details = _hydrate_fcf(
                    state=state,
                    action=action,
                    top_k=top_k,
                )
                result["details"] = details
                fcf_coverage_path = details.get("fcf_coverage_path")
                if fcf_coverage_path:
                    updated_artifacts["fcf_coverage_path"] = str(fcf_coverage_path)
                    updated_artifacts["fcf_coverage"] = str(fcf_coverage_path)
                    result["artifacts_written"].append(str(fcf_coverage_path))
            elif effective_action == "HYDRATE_FINANCIAL_FACTS":
                details = _hydrate_financial_facts(
                    state=state,
                    action=action,
                    top_k=top_k,
                )
                result["details"] = details
                facts_coverage_path = details.get("facts_coverage_path")
                if facts_coverage_path:
                    updated_artifacts["facts_coverage_path"] = str(facts_coverage_path)
                    updated_artifacts["facts_coverage"] = str(facts_coverage_path)
                    result["artifacts_written"].append(str(facts_coverage_path))
            elif effective_action == "RECOMPUTE_VALUATION":
                details = _build_valuations(
                    state=state,
                    action=action,
                    top_k=top_k,
                    run_dir=run_dir,
                    with_prices=with_prices,
                )
                result["details"] = details
                summary_path = details.get("summary_path")
                if summary_path:
                    updated_artifacts["valuation_summary_path"] = str(summary_path)
                    result["artifacts_written"].append(str(summary_path))
                coverage_path = details.get("price_coverage_path")
                if coverage_path:
                    updated_artifacts["price_coverage_path"] = str(coverage_path)
                    updated_artifacts["price_coverage"] = str(coverage_path)
                    result["artifacts_written"].append(str(coverage_path))
                shares_coverage_path = details.get("shares_coverage_path")
                if shares_coverage_path:
                    updated_artifacts["shares_coverage_path"] = str(shares_coverage_path)
                    updated_artifacts["shares_coverage"] = str(shares_coverage_path)
                    result["artifacts_written"].append(str(shares_coverage_path))
                fcf_coverage_path = details.get("fcf_coverage_path")
                if fcf_coverage_path:
                    updated_artifacts["fcf_coverage_path"] = str(fcf_coverage_path)
                    updated_artifacts["fcf_coverage"] = str(fcf_coverage_path)
                    result["artifacts_written"].append(str(fcf_coverage_path))
                facts_coverage_path = details.get("facts_coverage_path")
                if facts_coverage_path:
                    updated_artifacts["facts_coverage_path"] = str(facts_coverage_path)
                    updated_artifacts["facts_coverage"] = str(facts_coverage_path)
                    result["artifacts_written"].append(str(facts_coverage_path))
            elif effective_action == "REWEIGHT_RUBRIC":
                details = _reweight_rubric(state=state, action=action)
                result["details"] = details
            elif effective_action == "RUN_WHALE_SIGNALS":
                whale_summary, whale_artifacts = _run_whale_signals(state=state, run_dir=run_dir)
                updated_artifacts.update(whale_artifacts)
                result["details"] = {
                    "summary_path": whale_summary.get("summary_path"),
                    "ticker_count": len(whale_summary.get("rows") or []),
                }
                result["artifacts_written"] = list(whale_artifacts.values())
            elif effective_action == "BUILD_SCOREBOARD":
                peer_report, report_artifacts, overlay_summary = _build_scoreboard(
                    state=state,
                    run_dir=run_dir,
                    mode=mode,
                    with_prices=with_prices,
                )
                updated_artifacts.update(report_artifacts)
                result["details"] = {
                    "metrics_requested": action.metrics,
                    "peer_report": {
                        "peer_rankings_path": peer_report.get("peer_rankings_path"),
                        "peer_scoreboard_path": peer_report.get("peer_scoreboard_path"),
                        "peer_report_path": peer_report.get("peer_report_path"),
                        "ranked_count": len(peer_report.get("rankings") or []),
                    },
                }
                result["artifacts_written"] = list(report_artifacts.values())
                if overlay_summary is not None:
                    result["details"]["value_first_overlay"] = overlay_summary
                    shares_coverage_path = overlay_summary.get("shares_coverage_path")
                    if shares_coverage_path:
                        updated_artifacts["shares_coverage_path"] = str(shares_coverage_path)
                        updated_artifacts["shares_coverage"] = str(shares_coverage_path)
                        result["artifacts_written"].append(str(shares_coverage_path))
                    fcf_coverage_path = overlay_summary.get("fcf_coverage_path")
                    if fcf_coverage_path:
                        updated_artifacts["fcf_coverage_path"] = str(fcf_coverage_path)
                        updated_artifacts["fcf_coverage"] = str(fcf_coverage_path)
                        result["artifacts_written"].append(str(fcf_coverage_path))
                    facts_coverage_path = overlay_summary.get("facts_coverage_path")
                    if facts_coverage_path:
                        updated_artifacts["facts_coverage_path"] = str(facts_coverage_path)
                        updated_artifacts["facts_coverage"] = str(facts_coverage_path)
                        result["artifacts_written"].append(str(facts_coverage_path))
                    coverage_path = overlay_summary.get("valuation_coverage_path")
                    if coverage_path:
                        updated_artifacts["valuation_coverage_path"] = str(coverage_path)
                        updated_artifacts["valuation_coverage"] = str(coverage_path)
                        result["artifacts_written"].append(str(coverage_path))
                if str(mode).lower() == "depth":
                    decision, decision_artifacts = _update_decision_pack(
                        state=state,
                        run_dir=run_dir,
                        top_n=max(1, int(top_k)),
                        delta_path=state.artifacts.get("latest_scoreboard_delta_path"),
                    )
                    updated_artifacts.update(decision_artifacts)
                    result["details"]["decision_pack"] = decision
                    result["artifacts_written"] += list(decision_artifacts.values())
            elif effective_action == "RUN_SYNTHESIS":
                if not with_synthesis:
                    result["status"] = "SKIPPED"
                    result["details"] = {"reason": "with_synthesis disabled"}
                else:
                    target = action.target or {"scope": "sector", "value": state.sector}
                    scope = str(target.get("scope") or "").lower()
                    if scope == "ticker":
                        ticker = str(target.get("value") or "").upper()
                        path = run_synthesis_for_ticker(ticker=ticker, as_of_date=state.as_of_date, run_id=state.run_id)
                        result["details"] = {
                            "scope": "ticker",
                            "ticker": ticker,
                            "path": str(path) if path else None,
                        }
                    else:
                        summary = run_sector_synthesis(
                            sector=state.sector,
                            as_of_date=state.as_of_date,
                            run_id=state.run_id,
                        )
                        if summary.get("path"):
                            updated_artifacts["sector_synthesis_path"] = str(summary.get("path"))
                            result["artifacts_written"].append(str(summary.get("path")))
                        result["details"] = {"scope": "sector", "summary": summary}
            elif effective_action == "UPDATE_DECISION_PACK":
                top_n = int(action.top_n or top_k)
                delta_path = state.artifacts.get("latest_scoreboard_delta_path")
                decision, decision_artifacts = _update_decision_pack(
                    state=state,
                    run_dir=run_dir,
                    top_n=top_n,
                    delta_path=delta_path,
                )
                updated_artifacts.update(decision_artifacts)
                result["details"] = decision
                result["artifacts_written"] = list(decision_artifacts.values())
            else:
                result["status"] = "SKIPPED"
                result["details"] = {"reason": f"unsupported action_type={action.action_type}"}
        try:
            _run_with_timeout(
                fn=_execute_current_action,
                timeout_seconds=timeout_per_stage,
            )
        except InvalidFinancialInputError:
            raise
        except StageTimeoutError as exc:
            result["status"] = "STAGE_TIMEOUT"
            result["details"] = {
                "reason": "STAGE_TIMEOUT",
                "timeout_seconds": float(timeout_per_stage),
                "error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001
            result["status"] = "FAILED"
            result["details"] = {"error": str(exc)}
        results.append(result)

    state.artifacts.update(updated_artifacts)
    _upsert_sector_summary_artifacts(run_dir=run_dir, updated_artifacts=updated_artifacts)
    return {
        "planner_requested_stop": bool(planner_requested_stop),
        "executed_count": len(results),
        "results": results,
        "updated_artifacts": updated_artifacts,
        "peer_set_after": list(state.peer_set),
    }
