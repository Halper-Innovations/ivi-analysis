from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.rlm.schemas import CriticReport, StopDecision
from app.rlm.state import LoopState


NO_PROGRESS_GAP_REDUCTION_THRESHOLD = 0.05
TRACE_COMPLIANCE_MIN = 0.90
HIGH_CONFIDENCE_TRACE_MIN = 0.98
HIGH_CONFIDENCE_MAX_GAPS = 1

LOWER_BETTER_DELTA_METRICS = {
    "dilution_rate_shares_cagr",
    "net_debt_latest",
    "risk_factor_keyword_delta",
    "risk_penalty",
}
_VALUE_GATE_REASON_PRICE_UNKNOWN = "PRICE_UNKNOWN"
_VALUE_GATE_REASON_MISSING_INPUT_PREFIX = "MISSING_INPUT_"


def _read_json(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any, fallback: int = 0) -> int:
    return int(value) if isinstance(value, int) else fallback


def _top_k_from_rankings(rankings_payload: dict[str, Any], top_k: int) -> list[str]:
    from_value = [str(t).upper() for t in (rankings_payload.get("value_first_rank") or []) if str(t).strip()]
    if from_value:
        return from_value[: max(1, int(top_k))]
    from_future = [str(t).upper() for t in (rankings_payload.get("future_whale_rank") or []) if str(t).strip()]
    if from_future:
        return from_future[: max(1, int(top_k))]
    rows = [row for row in (rankings_payload.get("rankings") or []) if isinstance(row, dict)]
    ordered = [str(row.get("ticker") or "").upper() for row in rows if str(row.get("ticker") or "").strip()]
    return ordered[: max(1, int(top_k))]


def _scoreboard_rows_by_ticker(scoreboard_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [row for row in (scoreboard_payload.get("rows") or []) if isinstance(row, dict)]
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if ticker:
            out[ticker] = row
    return out


def _value_gate_rows_by_ticker(value_gates_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in (value_gates_payload.get("entries") or []):
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper()
        if ticker:
            out[ticker] = row
    return out


def _value_gate_progress(
    *,
    state: LoopState,
    top_k_tickers: list[str],
    value_gates_payload: dict[str, Any],
) -> tuple[dict[str, str], int, int, list[str]]:
    rows_by_ticker = _value_gate_rows_by_ticker(value_gates_payload)
    current_status: dict[str, str] = {}
    contradictions: list[str] = []
    for ticker in top_k_tickers:
        row = rows_by_ticker.get(ticker)
        if not isinstance(row, dict):
            contradictions.append(f"Missing value-gate row for top-k ticker {ticker}")
            current_status[ticker] = "WATCH"
            continue
        status = str(row.get("gate_status") or "WATCH").upper()
        if status not in {"PASS", "WATCH", "FAIL"}:
            status = "WATCH"
        current_status[ticker] = status

    previous_status: dict[str, str] = {}
    if state.progress_history:
        last = state.progress_history[-1]
        if isinstance(last, dict):
            previous_status = {
                str(ticker).upper(): str(status).upper()
                for ticker, status in (last.get("value_gate_status_top_k") or {}).items()
                if str(ticker).strip()
            }

    watch_to_pass = 0
    fail_confirmed = 0
    for ticker in top_k_tickers:
        curr = current_status.get(ticker, "WATCH")
        prev = previous_status.get(ticker)
        if prev == "WATCH" and curr == "PASS":
            watch_to_pass += 1
        if curr != "FAIL":
            continue
        row = rows_by_ticker.get(ticker) or {}
        reasons = {str(item) for item in (row.get("gate_reasons") or [])}
        missing_input = any(reason.startswith(_VALUE_GATE_REASON_MISSING_INPUT_PREFIX) for reason in reasons)
        if _VALUE_GATE_REASON_PRICE_UNKNOWN in reasons or missing_input:
            continue
        fail_confirmed += 1

    return current_status, int(watch_to_pass), int(fail_confirmed), contradictions


def _topk_evidence_and_gaps(*, top_k_tickers: list[str], scoreboard_payload: dict[str, Any]) -> tuple[int, int, list[str], float]:
    by_ticker = _scoreboard_rows_by_ticker(scoreboard_payload)
    evidence_count = 0
    gap_count = 0
    trace_checks = 0
    trace_pass = 0
    contradictions: list[str] = []

    for ticker in top_k_tickers:
        row = by_ticker.get(ticker)
        if not row:
            contradictions.append(f"Missing scoreboard row for top-k ticker {ticker}")
            continue

        metric_values = row.get("metric_values") or {}
        for value in metric_values.values():
            if isinstance(value, (int, float)):
                evidence_count += 1

        whale_summary = row.get("whale_summary") or {}
        gaps = whale_summary.get("gaps") or row.get("gaps") or []
        gap_count += len([gap for gap in gaps if isinstance(gap, dict)])

        metric_traces = row.get("metric_traces") or {}
        for trace_row in metric_traces.values():
            if not isinstance(trace_row, dict):
                continue
            trace_checks += 1
            refs = [str(ref) for ref in (trace_row.get("derived_from") or []) if str(ref).strip()]
            if refs:
                trace_pass += 1

        for signal_row in (whale_summary.get("top_signals") or []):
            if not isinstance(signal_row, dict):
                continue
            trace_checks += 1
            refs = [str(ref) for ref in (signal_row.get("derived_from") or []) if str(ref).strip()]
            if refs:
                trace_pass += 1

    trace_rate = float(trace_pass) / float(trace_checks) if trace_checks > 0 else 1.0
    return evidence_count, gap_count, contradictions, trace_rate


def _ranking_overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    left = [str(t).upper() for t in a if str(t).strip()]
    right = [str(t).upper() for t in b if str(t).strip()]
    if not left or not right:
        return 0.0
    denom = max(len(left), len(right), 1)
    overlap = len(set(left).intersection(set(right)))
    return float(overlap) / float(denom)


def _confidence(*, trace_rate: float, gap_count: int, evidence_delta: int, improvement_score: float) -> str:
    if (
        trace_rate >= HIGH_CONFIDENCE_TRACE_MIN
        and gap_count <= HIGH_CONFIDENCE_MAX_GAPS
        and evidence_delta >= 0
        and improvement_score >= 0
    ):
        return "HIGH"
    if trace_rate >= TRACE_COMPLIANCE_MIN and gap_count <= 5:
        return "MED"
    return "LOW"


def _scoreboard_delta_summary(
    *,
    delta_payload: dict[str, Any],
    top_k_tickers: list[str],
) -> tuple[int, int, int, float]:
    by_ticker = {
        str(row.get("ticker") or "").upper(): row
        for row in (delta_payload.get("rows") or [])
        if isinstance(row, dict)
    }
    improved = 0
    worsened = 0
    unknown = 0
    for ticker in top_k_tickers:
        row = by_ticker.get(ticker)
        if not isinstance(row, dict):
            continue
        metric_deltas = row.get("metric_deltas") or {}
        for metric, entry in metric_deltas.items():
            if not isinstance(entry, dict):
                continue
            delta = entry.get("delta")
            if not isinstance(delta, (int, float)):
                unknown += 1
                continue
            if metric in LOWER_BETTER_DELTA_METRICS:
                if float(delta) < 0:
                    improved += 1
                elif float(delta) > 0:
                    worsened += 1
            else:
                if float(delta) > 0:
                    improved += 1
                elif float(delta) < 0:
                    worsened += 1
    denom = max(1, improved + worsened)
    score = round((float(improved) - float(worsened)) / float(denom), 6)
    return improved, worsened, unknown, score


def evaluate_progress(
    *,
    state: LoopState,
    top_k: int,
    scoreboard_delta_path: str | None = None,
    calibration_required_note: dict[str, Any] | None = None,
) -> CriticReport:
    rankings_payload = _read_json(state.artifacts.get("peer_rankings_path"))
    scoreboard_payload = _read_json(state.artifacts.get("peer_scoreboard_path"))
    value_gates_payload = _read_json(state.artifacts.get("value_gates_path"))

    current_top_k = _top_k_from_rankings(rankings_payload, top_k=top_k)
    if not current_top_k:
        current_top_k = [str(t).upper() for t in state.top_k_current[: max(1, int(top_k))] if str(t).strip()]

    previous_top_k: list[str] = []
    if state.progress_history:
        last_progress = state.progress_history[-1]
        if isinstance(last_progress, dict):
            previous_top_k = [str(t).upper() for t in (last_progress.get("current_top_k") or []) if str(t).strip()]

    evidence_count, gap_count, contradictions, trace_rate = _topk_evidence_and_gaps(
        top_k_tickers=current_top_k,
        scoreboard_payload=scoreboard_payload,
    )
    gate_status_top_k, watch_to_pass_count, fail_confirmed_count, gate_contradictions = _value_gate_progress(
        state=state,
        top_k_tickers=current_top_k,
        value_gates_payload=value_gates_payload,
    )
    contradictions.extend(gate_contradictions)

    delta_payload = _read_json(scoreboard_delta_path)
    delta_improved, delta_worsened, delta_unknown, improvement_score = _scoreboard_delta_summary(
        delta_payload=delta_payload,
        top_k_tickers=current_top_k,
    )
    coverage_delta = delta_payload.get("coverage_delta") if isinstance(delta_payload.get("coverage_delta"), dict) else {}
    known_implied_delta = int(coverage_delta.get("known_implied_return_count_delta", 0)) if coverage_delta else 0
    if known_implied_delta > 0:
        delta_improved += int(known_implied_delta)
        denom = max(1, int(delta_improved + delta_worsened))
        improvement_score = round((float(delta_improved) - float(delta_worsened)) / float(denom), 6)

    if watch_to_pass_count > 0:
        delta_improved += int(watch_to_pass_count)
    if fail_confirmed_count > 0:
        delta_improved += int(fail_confirmed_count)
    if watch_to_pass_count > 0 or fail_confirmed_count > 0:
        denom = max(1, int(delta_improved + delta_worsened))
        improvement_score = round((float(delta_improved) - float(delta_worsened)) / float(denom), 6)

    prev_evidence = _safe_int(state.evidence_delta_counters.get("evidence_count_topk"), fallback=0)
    prev_gap_count = _safe_int(state.evidence_delta_counters.get("gap_count_topk"), fallback=0)

    evidence_delta = int(evidence_count - prev_evidence)
    if prev_gap_count > 0:
        gap_reduction = max(0.0, float(prev_gap_count - gap_count) / float(prev_gap_count))
    elif prev_gap_count == 0 and gap_count == 0:
        gap_reduction = 0.0
    else:
        gap_reduction = 0.0

    overlap = _ranking_overlap(current_top_k, previous_top_k)
    ranking_stable = bool(previous_top_k) and overlap >= 1.0
    no_new_evidence = evidence_delta <= 0 and delta_improved <= 0

    confidence = _confidence(
        trace_rate=trace_rate,
        gap_count=gap_count,
        evidence_delta=evidence_delta,
        improvement_score=improvement_score,
    )

    if not rankings_payload:
        contradictions.append("Missing peer_rankings artifact during critic evaluation")
    if not scoreboard_payload:
        contradictions.append("Missing peer_scoreboard artifact during critic evaluation")
    if state.artifacts.get("value_gates_path") and not value_gates_payload:
        contradictions.append("Missing value_gates artifact during critic evaluation")

    derived_from = [
        "sector.peer_rankings.future_whale_rank",
        "sector.peer_scoreboard.rows[*].metric_values",
        "sector.peer_scoreboard.rows[*].metric_traces",
        "sector.peer_scoreboard.rows[*].whale_summary.gaps",
    ]
    if state.artifacts.get("value_gates_path"):
        derived_from.append(str(state.artifacts.get("value_gates_path")))
    if scoreboard_delta_path:
        derived_from.append(scoreboard_delta_path)

    report = CriticReport(
        iteration=int(state.iteration),
        current_top_k=current_top_k,
        previous_top_k=previous_top_k,
        evidence_count_topk=evidence_count,
        evidence_delta_topk=evidence_delta,
        gap_count_topk=gap_count,
        gap_reduction_topk=round(gap_reduction, 6),
        ranking_overlap_topk=round(overlap, 6),
        ranking_stable=ranking_stable,
        no_new_evidence_topk=no_new_evidence,
        trace_compliance_rate=round(trace_rate, 6),
        confidence=confidence,
        contradictions=contradictions,
        delta_artifact_path=scoreboard_delta_path,
        delta_metrics_improved=int(delta_improved),
        delta_metrics_worsened=int(delta_worsened),
        delta_metrics_unknown=int(delta_unknown),
        improvement_score=float(improvement_score),
        watch_to_pass_count=int(watch_to_pass_count),
        fail_confirmed_count=int(fail_confirmed_count),
        value_gate_status_top_k=gate_status_top_k,
        calibration_required=calibration_required_note if isinstance(calibration_required_note, dict) else None,
        derived_from=derived_from,
    )
    return report


def apply_stop_rules(
    *,
    state: LoopState,
    critic_report: CriticReport,
    max_iterations: int,
    gap_reduction_threshold: float = NO_PROGRESS_GAP_REDUCTION_THRESHOLD,
) -> StopDecision:
    if int(state.iteration) >= int(max_iterations):
        return StopDecision(
            should_stop=True,
            status="DONE",
            reason_code="MAX_ITERATIONS_REACHED",
            summary=f"Reached iteration cap ({int(max_iterations)}).",
            derived_from=["state.iteration", "state.budgets_remaining.max_iterations"],
        )

    if float(state.budgets_remaining.llm_budget_remaining) <= 0.0:
        return StopDecision(
            should_stop=True,
            status="STOPPED",
            reason_code="LLM_BUDGET_EXHAUSTED",
            summary="LLM budget reached zero; stopping loop.",
            derived_from=["state.budgets_remaining.llm_budget_remaining"],
        )

    if int(state.budgets_remaining.sec_budget_remaining) <= 0:
        return StopDecision(
            should_stop=True,
            status="STOPPED",
            reason_code="SEC_BUDGET_EXHAUSTED",
            summary="SEC budget reached zero; stopping loop.",
            derived_from=["state.budgets_remaining.sec_budget_remaining"],
        )

    if float(critic_report.trace_compliance_rate) < TRACE_COMPLIANCE_MIN:
        return StopDecision(
            should_stop=True,
            status="NEEDS_HUMAN",
            reason_code="TRACE_COMPLIANCE_DROP",
            summary=(
                "Trace compliance dropped below threshold "
                f"({critic_report.trace_compliance_rate:.3f} < {TRACE_COMPLIANCE_MIN:.2f})."
            ),
            derived_from=["critic.trace_compliance_rate", "rlm.thresholds.trace_compliance_min"],
        )

    if state.no_progress_streak >= 2 and bool(critic_report.no_new_evidence_topk) and float(
        critic_report.gap_reduction_topk
    ) < float(gap_reduction_threshold):
        return StopDecision(
            should_stop=True,
            status="STOPPED",
            reason_code="NO_PROGRESS_STREAK",
            summary=(
                "No new top-k evidence and insufficient gap reduction for two consecutive iterations."
            ),
            derived_from=[
                "state.no_progress_streak",
                "critic.no_new_evidence_topk",
                "critic.gap_reduction_topk",
            ],
        )

    if (
        state.ranking_stable_streak >= 2
        and bool(critic_report.ranking_stable)
        and str(critic_report.confidence).upper() == "HIGH"
    ):
        return StopDecision(
            should_stop=True,
            status="DONE",
            reason_code="RANKING_STABLE_HIGH_CONFIDENCE",
            summary="Top-k ranking stable for two iterations with high confidence.",
            derived_from=[
                "state.ranking_stable_streak",
                "critic.ranking_stable",
                "critic.confidence",
            ],
        )

    return StopDecision(
        should_stop=False,
        status="RUNNING",
        reason_code="CONTINUE",
        summary="Continue recursive loop.",
        derived_from=["critic", "state.budgets_remaining"],
    )
