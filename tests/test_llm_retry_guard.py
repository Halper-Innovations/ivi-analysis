from __future__ import annotations

import time

import pytest

from app.llm.providers.retry_guard import (
    LLMCostBudgetExceeded,
    LLMMaxRetriesExceeded,
    LLMRetryBudgetExceeded,
    call_with_llm_retry_guard,
    llm_attempt_observer,
    llm_cost_budget,
    llm_physical_attempt_guard,
    llm_retry_budget,
)


def test_retry_guard_gives_up_after_two_retries_for_429(capsys):
    calls = {"count": 0}

    def always_rate_limited():
        calls["count"] += 1
        raise RuntimeError("status=429 rate limit")

    with llm_retry_budget(max_retries=15) as retry_context:
        with pytest.raises(LLMMaxRetriesExceeded, match="exceeded max retries \\(2\\)"):
            call_with_llm_retry_guard(
                provider_name="anthropic",
                schema_name="retry_test",
                call=always_rate_limited,
                timeout_seconds=1.0,
                max_retries=2,
                sleep_fn=lambda _seconds: None,
            )

    assert calls["count"] == 3
    assert retry_context.retry_count == 2
    assert [event.attempt for event in retry_context.events] == [1, 2]
    assert (
        "LLM retry 1/15: provider=anthropic schema=retry_test attempt=1" in capsys.readouterr().out
    )


def test_retry_guard_succeeds_after_one_retry_and_logs(capsys):
    calls = {"count": 0}

    def succeeds_after_retry():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("status=429 rate limit")
        return {"ok": True}

    with llm_retry_budget(max_retries=15) as retry_context:
        result = call_with_llm_retry_guard(
            provider_name="anthropic",
            schema_name="retry_success",
            call=succeeds_after_retry,
            timeout_seconds=1.0,
            max_retries=2,
            sleep_fn=lambda _seconds: None,
        )

    assert result == {"ok": True}
    assert calls["count"] == 2
    assert retry_context.retry_count == 1
    assert retry_context.events[0].reason == "status=429 rate limit"
    assert (
        "LLM retry 1/15: provider=anthropic schema=retry_success attempt=1"
        in capsys.readouterr().out
    )


def test_retry_guard_observer_sees_each_failed_physical_attempt():
    calls = {"count": 0}
    observed: list[dict] = []

    def succeeds_on_third_attempt():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("status=503 temporarily unavailable")
        return {"ok": True}

    with llm_attempt_observer(observed.append):
        result = call_with_llm_retry_guard(
            provider_name="openai",
            schema_name="physical_attempts",
            call=succeeds_on_third_attempt,
            timeout_seconds=1.0,
            max_retries=2,
            sleep_fn=lambda _seconds: None,
        )

    assert result == {"ok": True}
    assert [event["attempt"] for event in observed] == [1, 2]
    assert [event["will_retry"] for event in observed] == [True, True]
    assert all(event["retryable"] is True for event in observed)
    assert all(isinstance(event["error"], RuntimeError) for event in observed)


def test_physical_attempt_guard_runs_before_every_retry() -> None:
    calls = {"count": 0}
    guarded_attempts: list[int] = []

    def succeeds_after_retry():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("status=503 temporarily unavailable")
        return {"ok": True}

    with llm_physical_attempt_guard(lambda event: guarded_attempts.append(int(event["attempt"]))):
        result = call_with_llm_retry_guard(
            provider_name="openai",
            schema_name="physical_guard",
            call=succeeds_after_retry,
            timeout_seconds=1.0,
            max_retries=1,
            sleep_fn=lambda _seconds: None,
        )

    assert result == {"ok": True}
    assert calls["count"] == 2
    assert guarded_attempts == [1, 2]


def test_mutated_financial_scope_suppresses_second_physical_attempt() -> None:
    from app.autonomous.financial_integrity import (
        FinancialIntegrityScope,
        InvalidFinancialInputError,
        require_financial_integrity_scope,
        require_unchanged_financial_integrity_scope,
    )
    from app.autonomous.v1_financial_context import financial_input_scenario
    from tests.test_financial_integrity import _valid_packet

    packet = _valid_packet()
    scenario = financial_input_scenario(
        packet,
        financial_inputs={"provider_prompt": "original"},
    )
    scope = FinancialIntegrityScope(
        context="retry_mutation",
        run_as_of_date="2026-07-22",
        packets=(packet,),
        scenarios=(scenario,),
    )
    expected = require_financial_integrity_scope(scope).scope_fingerprint
    calls = {"count": 0}

    def retryable_then_mutate():
        calls["count"] += 1
        scenario["financial_inputs"]["provider_prompt"] = "mutated"
        raise RuntimeError("status=503 temporarily unavailable")

    def require_exact_scope(_event):
        require_unchanged_financial_integrity_scope(
            scope,
            expected_scope_fingerprint=expected,
        )

    with (
        llm_physical_attempt_guard(require_exact_scope),
        pytest.raises(InvalidFinancialInputError) as exc_info,
    ):
        call_with_llm_retry_guard(
            provider_name="openai",
            schema_name="retry_mutation",
            call=retryable_then_mutate,
            timeout_seconds=1.0,
            max_retries=1,
            sleep_fn=lambda _seconds: None,
        )

    assert calls["count"] == 1
    assert {item.code for item in exc_info.value.violations} == {"BOUND_FINANCIAL_INPUT_MUTATED"}


def test_strict_cost_context_disables_physical_provider_retries():
    calls = {"count": 0}
    observed: list[dict] = []

    def always_rate_limited():
        calls["count"] += 1
        raise RuntimeError("status=429 rate limit")

    with (
        llm_cost_budget(max_cost_usd=1.0, strict_first_call=True),
        llm_retry_budget(max_retries=15) as retry_context,
        llm_attempt_observer(observed.append),
        pytest.raises(LLMMaxRetriesExceeded, match="max retries \\(0\\)"),
    ):
        call_with_llm_retry_guard(
            provider_name="anthropic",
            schema_name="strict_no_retry",
            call=always_rate_limited,
            timeout_seconds=1.0,
            max_retries=2,
            sleep_fn=lambda _seconds: None,
        )

    assert calls["count"] == 1
    assert retry_context.retry_count == 0
    assert len(observed) == 1
    assert observed[0]["attempt"] == 1
    assert observed[0]["will_retry"] is False


def test_retry_guard_waits_for_definitive_slow_call_without_orphaning():
    calls = {"count": 0}
    completed = {"value": False}

    def hangs_longer_than_timeout():
        calls["count"] += 1
        time.sleep(0.20)
        completed["value"] = True
        return {"ok": True}

    started = time.monotonic()
    result = call_with_llm_retry_guard(
        provider_name="anthropic",
        schema_name="timeout_test",
        call=hangs_longer_than_timeout,
        timeout_seconds=0.01,
        max_retries=2,
        sleep_fn=lambda _seconds: None,
    )

    assert result == {"ok": True}
    assert calls["count"] == 1
    assert completed["value"] is True
    assert time.monotonic() - started >= 0.18


def test_retry_guard_run_budget_aborts_on_sixteenth_retry():
    def always_rate_limited():
        raise RuntimeError("status=429 rate limit")

    with llm_retry_budget(max_retries=15) as retry_context:
        with pytest.raises(LLMRetryBudgetExceeded, match="16>15"):
            for _call_number in range(8):
                try:
                    call_with_llm_retry_guard(
                        provider_name="anthropic",
                        schema_name="budget_test",
                        call=always_rate_limited,
                        timeout_seconds=1.0,
                        max_retries=2,
                        sleep_fn=lambda _seconds: None,
                    )
                except LLMMaxRetriesExceeded:
                    continue

    assert retry_context.retry_count == 16
    assert len(retry_context.events) == 16


def test_cost_budget_allows_oversized_first_call_then_blocks_second():
    with llm_cost_budget(max_cost_usd=0.10) as cost_context:
        first = cost_context.reserve_call(
            provider="openai",
            schema_name="first_call",
            estimated_cost_usd=0.25,
        )
        cost_context.complete_call(first, actual_cost_usd=0.25)

        with pytest.raises(LLMCostBudgetExceeded, match="projected \\$0.2600 > budget \\$0.1000"):
            cost_context.reserve_call(
                provider="openai",
                schema_name="second_call",
                estimated_cost_usd=0.01,
            )

    assert cost_context.call_count == 1
    assert cost_context.summary()["cumulative_cost_usd"] == 0.25


def test_strict_cost_budget_blocks_oversized_first_call_before_provider():
    with llm_cost_budget(max_cost_usd=0.10, strict_first_call=True) as cost_context:
        with pytest.raises(
            LLMCostBudgetExceeded,
            match="projected \\$0.2500 > budget \\$0.1000",
        ):
            cost_context.reserve_call(
                provider="anthropic",
                schema_name="strict_first_call",
                estimated_cost_usd=0.25,
            )

    assert cost_context.call_count == 0
    assert cost_context.summary()["cumulative_cost_usd"] == 0.0


def test_cost_budget_none_disables_budget_enforcement():
    with llm_cost_budget(max_cost_usd=None) as cost_context:
        first = cost_context.reserve_call(
            provider="openai",
            schema_name="first_call",
            estimated_cost_usd=999.0,
        )
        cost_context.complete_call(first, actual_cost_usd=0.75)
        second = cost_context.reserve_call(
            provider="openai",
            schema_name="second_call",
            estimated_cost_usd=999.0,
        )
        cost_context.complete_call(second, actual_cost_usd=1.25)

    assert first is not None
    assert first.call_number == 1
    assert second is not None
    assert second.call_number == 2
    assert cost_context.summary() == {
        "max_cost_usd": None,
        "call_count": 2,
        "cumulative_cost_usd": 2.0,
        "reserved_cost_usd": 0.0,
        "events": [
            {
                "provider": "openai",
                "schema_name": "first_call",
                "call_number": 1,
                "estimated_cost_usd": 999.0,
                "actual_cost_usd": 0.75,
                "cumulative_cost_usd": 0.75,
            },
            {
                "provider": "openai",
                "schema_name": "second_call",
                "call_number": 2,
                "estimated_cost_usd": 999.0,
                "actual_cost_usd": 1.25,
                "cumulative_cost_usd": 2.0,
            },
        ],
    }
