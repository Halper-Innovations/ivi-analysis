from types import SimpleNamespace

from app.llm.usage_capture import (
    attach_provider_usage_to_exception,
    attached_provider_usage_records,
    current_provider_usage_lane,
    failed_provider_usage_meta,
    provider_usage_capture,
    provider_usage_meta,
    provider_usage_records,
    provider_usage_records_from_exception,
    record_provider_usage,
)


def _provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="openai",
        model="gpt-5.5",
    )


def test_successful_provider_usage_preserves_actual_tokens_cache_cost_and_lane() -> None:
    result = SimpleNamespace(
        json_text='{"status":"ok"}',
        model="gpt-5.5",
        usage_input_tokens=1_000,
        usage_cached_input_tokens=250,
        usage_output_tokens=100,
    )

    with provider_usage_capture("parent_research") as captured:
        record_provider_usage(
            provider_usage_meta(
                provider=_provider(),
                result=result,
                prompt="test prompt",
                schema_name="sector_plan",
            )
        )

    assert captured == [
        {
            "cached_input_tokens": 250,
            "cost_estimate_usd": 0.006875,
            "estimated_tokens": False,
            "input_tokens": 1_000,
            "lane": "parent_research",
            "model": "gpt-5.5",
            "output_tokens": 100,
            "provider": "openai",
            "provider_call_id": "P1",
            "reserved_output_tokens": 0,
            "schema_name": "sector_plan",
            "status": "OK",
        }
    ]
    assert current_provider_usage_lane() is None


def test_failed_provider_attempt_is_counted_with_conservative_reserved_cost() -> None:
    with provider_usage_capture("selected_company_validation") as captured:
        record_provider_usage(
            failed_provider_usage_meta(
                provider=_provider(),
                prompt="x" * 4_000,
                schema_name="selected_challenge",
                estimated_output_tokens=2_000,
                error=TimeoutError("provider timeout"),
            )
        )

    assert len(captured) == 1
    row = captured[0]
    assert row["status"] == "ERROR"
    assert row["lane"] == "selected_company_validation"
    assert row["input_tokens"] == 1_000
    assert row["output_tokens"] == 0
    assert row["reserved_output_tokens"] == 2_000
    assert row["cost_basis"] == "CONSERVATIVE_FAILED_CALL_RESERVE"
    assert row["cost_estimate_usd"] == 0.065
    assert row["error"] == "TimeoutError: provider timeout"


def test_successful_output_retries_are_split_into_physical_response_records() -> None:
    first = {
        "model": "gpt-5.5",
        "status": "incomplete",
        "usage": {
            "input_tokens": 1_000,
            "output_tokens": 500,
            "input_tokens_details": {"cached_tokens": 100},
        },
    }
    second = {
        "model": "gpt-5.5",
        "status": "completed",
        "usage": {
            "input_tokens": 1_100,
            "output_tokens": 700,
            "input_tokens_details": {"cached_tokens": 200},
        },
    }
    result = SimpleNamespace(
        json_text='{"status":"ok"}',
        model="gpt-5.5",
        usage_input_tokens=2_100,
        usage_cached_input_tokens=300,
        usage_output_tokens=1_200,
        raw={**second, "_response_attempts": [first, second]},
    )

    with provider_usage_capture("company_underwriting"):
        records = provider_usage_records(
            provider=_provider(),
            result=result,
            prompt="test prompt",
            schema_name="child_decision",
        )

    assert [record["status"] for record in records] == ["INCOMPLETE", "OK"]
    assert [record["physical_response_sequence"] for record in records] == [1, 2]
    assert [record["input_tokens"] for record in records] == [1_000, 1_100]
    assert [record["cached_input_tokens"] for record in records] == [100, 200]
    assert [record["output_tokens"] for record in records] == [500, 700]
    assert sum(record["cost_estimate_usd"] for record in records) == 0.04515


def test_nested_capture_keeps_child_usage_out_of_parent_lane() -> None:
    result = SimpleNamespace(
        json_text="{}",
        model="gpt-5.5",
        usage_input_tokens=10,
        usage_cached_input_tokens=0,
        usage_output_tokens=5,
    )
    with provider_usage_capture("parent_research") as parent:
        record_provider_usage(
            provider_usage_meta(
                provider=_provider(),
                result=result,
                prompt="parent",
                schema_name="parent",
            )
        )
        with provider_usage_capture("company_underwriting") as child:
            record_provider_usage(
                provider_usage_meta(
                    provider=_provider(),
                    result=result,
                    prompt="child",
                    schema_name="child",
                )
            )
        record_provider_usage(
            provider_usage_meta(
                provider=_provider(),
                result=result,
                prompt="parent again",
                schema_name="parent_final",
            )
        )

    assert [row["schema_name"] for row in parent] == ["parent", "parent_final"]
    assert [row["provider_call_id"] for row in parent] == ["P1", "P2"]
    assert [row["lane"] for row in parent] == ["parent_research", "parent_research"]
    assert [row["schema_name"] for row in child] == ["child"]
    assert child[0]["provider_call_id"] == "P1"
    assert child[0]["lane"] == "company_underwriting"


def test_billed_response_usage_survives_provider_exception() -> None:
    error = RuntimeError("second expansion failed")
    error._provider_response_attempts = [
        {
            "model": "gpt-5.5",
            "status": "incomplete",
            "usage": {
                "input_tokens": 400,
                "output_tokens": 120,
                "input_tokens_details": {"cached_tokens": 50},
            },
        }
    ]

    with provider_usage_capture("parent_research"):
        records = provider_usage_records_from_exception(
            provider=_provider(),
            error=error,
            prompt="memo prompt",
            schema_name="memo_schema",
        )

    assert len(records) == 1
    assert records[0]["status"] == "INCOMPLETE"
    assert records[0]["input_tokens"] == 400
    assert records[0]["cached_input_tokens"] == 50
    assert records[0]["output_tokens"] == 120
    assert records[0]["lane"] == "parent_research"

    attach_provider_usage_to_exception(error, records)
    assert attached_provider_usage_records(error) == records
    attach_provider_usage_to_exception(
        error,
        [{**records[0], "provider_call_id": "P1"}],
    )
    propagated = attached_provider_usage_records(error)
    assert len(propagated) == 1
    assert propagated[0]["provider_call_id"] == "P1"


def test_deepseek_chat_usage_fields_are_normalized_from_exception() -> None:
    provider = SimpleNamespace(
        provider_name="deepseek",
        model="deepseek-v4-pro",
    )
    error = RuntimeError("invalid json")
    error._provider_response_attempts = [
        {
            "model": "deepseek-v4-pro",
            "usage": {
                "prompt_tokens": 300,
                "completion_tokens": 60,
                "prompt_cache_hit_tokens": 25,
            },
        }
    ]
    records = provider_usage_records_from_exception(
        provider=provider,
        error=error,
        prompt="memo prompt",
        schema_name="memo_schema",
    )
    assert len(records) == 1
    assert records[0]["provider"] == "deepseek"
    assert records[0]["input_tokens"] == 300
    assert records[0]["cached_input_tokens"] == 25
    assert records[0]["output_tokens"] == 60
    assert records[0]["estimated_tokens"] is False
    assert records[0]["cost_estimate_usd"] == 0.000172
