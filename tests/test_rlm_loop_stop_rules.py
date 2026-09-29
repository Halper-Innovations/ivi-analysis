from __future__ import annotations

from app.rlm.critic import apply_stop_rules
from app.rlm.schemas import CriticReport
from app.rlm.state import LoopBudgets, LoopState


def _state() -> LoopState:
    return LoopState(
        run_id="rlm_test",
        sector="Software",
        as_of_date="2026-02-13",
        iteration=0,
        budgets_remaining=LoopBudgets(
            sec_budget_remaining=100,
            llm_budget_remaining=5.0,
            max_iterations=3,
        ),
    )


def _critic(**overrides):
    payload = {
        "iteration": 0,
        "current_top_k": ["AAPL", "MSFT"],
        "previous_top_k": ["AAPL", "MSFT"],
        "evidence_count_topk": 10,
        "evidence_delta_topk": 0,
        "gap_count_topk": 2,
        "gap_reduction_topk": 0.0,
        "ranking_overlap_topk": 1.0,
        "ranking_stable": True,
        "no_new_evidence_topk": True,
        "trace_compliance_rate": 0.99,
        "confidence": "HIGH",
        "contradictions": [],
        "derived_from": ["x"],
    }
    payload.update(overrides)
    return CriticReport.model_validate(payload)


def test_stop_rule_max_iterations():
    state = _state()
    state.iteration = 3
    decision = apply_stop_rules(state=state, critic_report=_critic(), max_iterations=3)
    assert decision.should_stop is True
    assert decision.reason_code == "MAX_ITERATIONS_REACHED"


def test_stop_rule_budget_exhaustion():
    state = _state()
    state.budgets_remaining.llm_budget_remaining = 0.0
    decision = apply_stop_rules(state=state, critic_report=_critic(), max_iterations=3)
    assert decision.should_stop is True
    assert decision.reason_code == "LLM_BUDGET_EXHAUSTED"

    state = _state()
    state.budgets_remaining.sec_budget_remaining = 0
    decision = apply_stop_rules(state=state, critic_report=_critic(), max_iterations=3)
    assert decision.should_stop is True
    assert decision.reason_code == "SEC_BUDGET_EXHAUSTED"


def test_stop_rule_no_progress_streak():
    state = _state()
    state.no_progress_streak = 2
    decision = apply_stop_rules(state=state, critic_report=_critic(), max_iterations=3)
    assert decision.should_stop is True
    assert decision.reason_code == "NO_PROGRESS_STREAK"


def test_stop_rule_ranking_stable_high_confidence():
    state = _state()
    state.ranking_stable_streak = 2
    decision = apply_stop_rules(state=state, critic_report=_critic(), max_iterations=3)
    assert decision.should_stop is True
    assert decision.reason_code == "RANKING_STABLE_HIGH_CONFIDENCE"


def test_stop_rule_trace_drop_needs_human():
    state = _state()
    decision = apply_stop_rules(
        state=state,
        critic_report=_critic(trace_compliance_rate=0.5, ranking_stable=False, confidence="LOW"),
        max_iterations=3,
    )
    assert decision.should_stop is True
    assert decision.status == "NEEDS_HUMAN"
    assert decision.reason_code == "TRACE_COMPLIANCE_DROP"
