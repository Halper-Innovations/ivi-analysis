from __future__ import annotations

from app.decision.decision_block import (
    ACTION_REVIEW_AT_TARGET,
    DecisionBlock,
    action_from_status_and_grade,
    buy_now_imperative_line,
    pct_to_target,
    render_decision_block_markdown,
)


def test_action_review_at_target_literal():
    # Template-pass rule (2026-06-11): the deploy flag routes attention, it is
    # not a buy signal — the action verb must not read as a buy directive.
    assert ACTION_REVIEW_AT_TARGET == "Review at target"


def test_action_deploy_ready_actionable_is_review_at_target():
    assert (
        action_from_status_and_grade("DEPLOY_READY", "ACTIONABLE")
        == ACTION_REVIEW_AT_TARGET
    )


def test_action_deploy_ready_watchlist_only_is_review_at_target():
    assert (
        action_from_status_and_grade("DEPLOY_READY", "WATCHLIST_ONLY")
        == ACTION_REVIEW_AT_TARGET
    )


def test_action_deploy_ready_avoid_is_pass():
    assert action_from_status_and_grade("DEPLOY_READY", "AVOID") == "Pass"


def test_action_buy_confirmed_actionable_is_review_at_target():
    # BUY_CONFIRMED (catalyst-confirmed, price <= target) is the most actionable
    # price-trigger status and must surface the at-target action.
    assert (
        action_from_status_and_grade("BUY_CONFIRMED", "ACTIONABLE")
        == ACTION_REVIEW_AT_TARGET
    )


def test_action_buy_confirmed_avoid_is_pass():
    # The AVOID grade veto still applies even to a catalyst-confirmed buy.
    assert action_from_status_and_grade("BUY_CONFIRMED", "AVOID") == "Pass"


def test_action_active_actionable_waits_for_target():
    assert (
        action_from_status_and_grade(
            "ACTIVE", "ACTIONABLE", buy_price_target=32.19
        )
        == "Wait for $32.19"
    )


def test_action_price_data_suspect_is_not_a_buy():
    assert (
        action_from_status_and_grade(
            "PRICE_DATA_SUSPECT", "ACTIONABLE", buy_price_target=32.19
        )
        == "Wait for $32.19"
    )


def test_action_quarantine_is_not_a_buy():
    assert (
        action_from_status_and_grade(
            "QUARANTINE", "ACTIONABLE", buy_price_target=32.19
        )
        == "Wait for $32.19"
    )


def test_pct_to_target_below_target():
    assert round(pct_to_target(18.33, 32.19), 4) == -0.4306


def test_pct_to_target_above_target():
    assert round(pct_to_target(138.44, 136.60), 4) == 0.0135


def test_pct_to_target_zero_target_is_none():
    assert pct_to_target(50.0, 0.0) is None


def test_at_target_line_exact_format():
    assert (
        buy_now_imperative_line("CRTO", 18.33, 32.19, 0.34, "ACTIONABLE")
        == "AT TARGET CRTO: $18.33 <= target $32.19 (+34.0%/yr, ACTIONABLE)"
        " — review trigger, not a buy signal"
    )


def test_at_target_line_omits_return_when_none():
    assert (
        buy_now_imperative_line("CRTO", 18.33, 32.19, None, "ACTIONABLE")
        == "AT TARGET CRTO: $18.33 <= target $32.19 (ACTIONABLE)"
        " — review trigger, not a buy signal"
    )


def test_render_decision_block_markdown_header_and_action():
    block = DecisionBlock(
        action="Review at target",
        current_price=18.33,
        buy_price_target=32.19,
        pct_to_target=-0.4306,
        base_case_expected_return=None,
        conviction_grade="ACTIONABLE",
        confidence="HIGH",
        price_trigger_status="DEPLOY_READY",
        time_horizon="3-5 yr",
        what_would_change_my_mind=[],
        verdict_reconciliation_note=None,
    )
    lines = render_decision_block_markdown(block).splitlines()
    assert lines[0] == "## Decision"
    assert lines[1] == "- ACTION: Review at target"
