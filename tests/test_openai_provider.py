from __future__ import annotations

import json
import re

import pytest

from app.config import get_config
from app.llm.providers.openai_provider import (
    OpenAIOutputTruncatedError,
    OpenAIProvider,
)
from app.llm.providers.retry_guard import llm_cost_budget
from app.llm.execution_policy import LLMExecutionPolicy, llm_execution_policy


def _reset_cfg(monkeypatch):
    OpenAIProvider._reset_circuit_breaker()
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("VOE_OPENAI_MODEL", raising=False)
    # The transport is mocked in these tests; the network switch is on.
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    get_config.cache_clear()
    return get_config()


def test_openai_provider_builds_responses_payload_with_text_format_json_schema(monkeypatch):
    cfg = _reset_cfg(monkeypatch)
    captured: dict[str, object] = {}

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "output_text": json.dumps({"ok": True}),
                "usage": {"input_tokens": 11, "output_tokens": 7},
            }

    def _fake_post(url, *, headers, json, timeout):  # noqa: A002
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr("app.llm.providers.openai_provider.requests.post", _fake_post)
    provider = OpenAIProvider(cfg)
    assert provider.enabled()

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
    out = provider.synthesize_json(
        prompt="hello",
        schema=schema,
        schema_name="Synthesis Packet v1 !! @@@",
    )
    assert out.model == "gpt-5-mini"

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "gpt-5-mini"
    assert "input" in payload
    assert payload["input"] == "hello"
    assert "response_format" not in payload
    assert "tools" not in payload
    assert "include" not in payload
    assert "max_tool_calls" not in payload
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["name"] == "Synthesis_Packet_v1"
    assert len(payload["text"]["format"]["name"]) <= 64
    assert re.match(r"^[A-Za-z0-9_-]+$", payload["text"]["format"]["name"])
    assert payload["text"]["format"]["schema"] == schema
    assert payload["text"]["format"]["strict"] is True
    assert out.raw["_request_contract"] == {
        "initial_max_output_tokens": 1200,
        "final_max_output_tokens": 1200,
        "response_output_token_limits": [1200],
        "physical_response_count": 1,
        "allow_output_token_retry": True,
        "max_retries_per_request": 2,
        "strict_cost_cap": False,
    }


def test_openai_provider_additively_supports_bounded_web_search(monkeypatch):
    cfg = _reset_cfg(monkeypatch)
    captured: dict[str, object] = {}

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "id": "resp_search_1",
                "model": "gpt-5.5",
                "service_tier": "default",
                "output_text": json.dumps({"ok": True}),
                "output": [
                    {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "action": {
                            "sources": [
                                {"url": "https://example.com/market-cap"}
                            ]
                        },
                    }
                ],
                "usage": {
                    "input_tokens": 120,
                    "input_tokens_details": {"cached_tokens": 20},
                    "output_tokens": 30,
                },
            }

    def _fake_post(url, *, headers, json, timeout):  # noqa: A002
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(
        "app.llm.providers.openai_provider.requests.post",
        _fake_post,
    )
    result = OpenAIProvider(cfg).synthesize_json(
        prompt="find direct market cap",
        schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        schema_name="terminal_cap_search_v1",
        tools=[{"type": "web_search"}],
        include=["web_search_call.action.sources"],
        max_tool_calls=2,
        model="gpt-5.5",
        service_tier="default",
    )

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "gpt-5.5"
    assert payload["tools"] == [{"type": "web_search"}]
    assert payload["include"] == ["web_search_call.action.sources"]
    assert payload["max_tool_calls"] == 2
    assert payload["service_tier"] == "default"
    assert result.model == "gpt-5.5"
    assert result.usage_input_tokens == 120
    assert result.usage_cached_input_tokens == 20
    assert result.usage_output_tokens == 30


def test_openai_provider_rejects_service_tier_drift(monkeypatch):
    cfg = _reset_cfg(monkeypatch)

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "service_tier": "priority",
                "output_text": json.dumps({"ok": True}),
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

    monkeypatch.setattr(
        "app.llm.providers.openai_provider.requests.post",
        lambda *args, **kwargs: _Resp(),
    )

    with pytest.raises(RuntimeError, match="service tier mismatch") as exc_info:
        OpenAIProvider(cfg).synthesize_json(
            prompt="bounded standard request",
            schema={"type": "object"},
            service_tier="default",
            allow_output_token_retry=False,
            max_retries_per_request=0,
        )
    assert exc_info.value._provider_response_attempts == [
        {
            "service_tier": "priority",
            "output_text": json.dumps({"ok": True}),
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    ]


def test_openai_provider_rejects_unbounded_zero_tool_limit(monkeypatch):
    cfg = _reset_cfg(monkeypatch)
    provider = OpenAIProvider(cfg)
    with pytest.raises(ValueError, match="positive integer"):
        provider.synthesize_json(
            prompt="test",
            schema={"type": "object"},
            tools=[{"type": "web_search"}],
            max_tool_calls=0,
        )


def test_openai_provider_enforces_v2_model_and_request_bounds_before_network(
    monkeypatch,
) -> None:
    cfg = _reset_cfg(monkeypatch)
    network_calls: list[str] = []
    monkeypatch.setattr(
        "app.llm.providers.openai_provider.requests.post",
        lambda *args, **kwargs: network_calls.append("called"),
    )
    policy = LLMExecutionPolicy(
        provider="openai",
        model="gpt-5.5",
        max_serialized_request_bytes=500,
        max_output_tokens=100,
        max_retries_per_request=0,
    )

    with llm_execution_policy(policy):
        with pytest.raises(RuntimeError, match="model drift"):
            OpenAIProvider(cfg).synthesize_json(
                prompt="small",
                schema={"type": "object"},
                max_output_tokens=100,
            )

    monkeypatch.setenv("VOE_OPENAI_MODEL", "gpt-5.5")
    get_config.cache_clear()
    cfg = get_config()
    with llm_execution_policy(policy):
        with pytest.raises(RuntimeError, match="input bound"):
            OpenAIProvider(cfg).synthesize_json(
                prompt="x" * 1_000,
                schema={"type": "object"},
                max_output_tokens=100,
            )
        with pytest.raises(RuntimeError, match="output bound"):
            OpenAIProvider(cfg).synthesize_json(
                prompt="small",
                schema={"type": "object"},
                max_output_tokens=101,
            )

    assert network_calls == []


def test_openai_provider_includes_error_body_on_non_2xx(monkeypatch):
    OpenAIProvider._reset_circuit_breaker()
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_API_KEY", "sk-secret-token")
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    get_config.cache_clear()
    cfg = get_config()

    class _Resp:
        status_code = 400
        text = "{\"error\":\"bad request\"}"

        @staticmethod
        def json():
            return {"error": {"message": "bad model name"}}

    monkeypatch.setattr("app.llm.providers.openai_provider.requests.post", lambda *args, **kwargs: _Resp())
    provider = OpenAIProvider(cfg)

    with pytest.raises(RuntimeError) as exc:
        provider.synthesize_json(prompt="test", schema={"type": "object"}, schema_name="synthesis_packet_v1")
    msg = str(exc.value)
    assert "status=400" in msg
    assert "bad model name" in msg
    assert "sk-secret-token" not in msg
    OpenAIProvider._reset_circuit_breaker()


def test_openai_provider_opens_quota_circuit_breaker_after_insufficient_quota(monkeypatch):
    cfg = _reset_cfg(monkeypatch)
    calls: list[dict] = []

    class _Resp:
        status_code = 429
        text = ""

        @staticmethod
        def json():
            return {"error": {"message": "You exceeded your current quota.", "code": "insufficient_quota"}}

    def _fake_post(url, *, headers, json, timeout):  # noqa: A002
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _Resp()

    monkeypatch.setattr("app.llm.providers.openai_provider.requests.post", _fake_post)

    first_provider = OpenAIProvider(cfg)
    with pytest.raises(RuntimeError) as first_exc:
        first_provider.synthesize_json(prompt="test", schema={"type": "object"}, schema_name="quota_test")
    assert "status=429" in str(first_exc.value)
    assert "insufficient_quota" in str(first_exc.value)

    second_provider = OpenAIProvider(cfg)
    with pytest.raises(RuntimeError) as second_exc:
        second_provider.synthesize_json(prompt="test again", schema={"type": "object"}, schema_name="quota_test")
    assert "circuit breaker open" in str(second_exc.value)
    assert "insufficient_quota" in str(second_exc.value)
    assert len(calls) == 1

    OpenAIProvider._reset_circuit_breaker()


def test_openai_provider_extracts_structured_json_content_when_output_text_missing(monkeypatch):
    cfg = _reset_cfg(monkeypatch)

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_json",
                                "json": {"ticker": "AAPL", "ok": True},
                            }
                        ]
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 9},
            }

    monkeypatch.setattr("app.llm.providers.openai_provider.requests.post", lambda *args, **kwargs: _Resp())
    provider = OpenAIProvider(cfg)
    out = provider.synthesize_json(prompt="hello", schema={"type": "object"}, schema_name="synthesis_packet_v1")
    assert json.loads(out.json_text) == {"ticker": "AAPL", "ok": True}


def test_openai_provider_normalizes_fenced_json_output_text(monkeypatch):
    cfg = _reset_cfg(monkeypatch)

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "output_text": "```json\n{\"ok\": true}\n```",
                "usage": {"input_tokens": 5, "output_tokens": 5},
            }

    monkeypatch.setattr("app.llm.providers.openai_provider.requests.post", lambda *args, **kwargs: _Resp())
    provider = OpenAIProvider(cfg)
    out = provider.synthesize_json(prompt="hello", schema={"type": "object"}, schema_name="synthesis_packet_v1")
    assert json.loads(out.json_text) == {"ok": True}


def test_openai_provider_retries_invalid_json_once_at_max_tokens(monkeypatch):
    cfg = _reset_cfg(monkeypatch)
    calls: list[dict] = []

    class _Resp:
        status_code = 200
        text = ""

        def __init__(self, output_text: str) -> None:
            self._output_text = output_text

        def json(self):
            return {
                "output_text": self._output_text,
                "usage": {"input_tokens": 5, "output_tokens": 5},
            }

    def _fake_post(url, *, headers, json, timeout):  # noqa: A002
        _ = (url, headers, timeout)
        calls.append(json)
        if len(calls) == 1:
            return _Resp('{"ok": true')
        return _Resp('{"ok": true}')

    monkeypatch.setattr("app.llm.providers.openai_provider.requests.post", _fake_post)
    provider = OpenAIProvider(cfg)
    out = provider.synthesize_json(
        prompt="hello",
        schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False},
        schema_name="retry_schema",
        max_output_tokens=1000,
    )

    assert json.loads(out.json_text) == {"ok": True}
    assert [call["max_output_tokens"] for call in calls] == [1000, 8000]
    assert out.usage_input_tokens == 10
    assert out.usage_output_tokens == 10
    assert len(out.raw["_response_attempts"]) == 2


def test_openai_provider_can_disable_output_retry_for_preflight_bounded_call(
    monkeypatch,
):
    cfg = _reset_cfg(monkeypatch)
    calls: list[dict] = []

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "output_text": '{"ok": true',
                "usage": {"input_tokens": 5, "output_tokens": 5},
            }

    def _fake_post(url, *, headers, json, timeout):  # noqa: A002
        _ = (url, headers, timeout)
        calls.append(json)
        return _Resp()

    monkeypatch.setattr("app.llm.providers.openai_provider.requests.post", _fake_post)

    with pytest.raises(RuntimeError, match="not valid JSON"):
        OpenAIProvider(cfg).synthesize_json(
            prompt="hello",
            schema={"type": "object"},
            schema_name="bounded_schema",
            max_output_tokens=1000,
            allow_output_token_retry=False,
        )

    assert [call["max_output_tokens"] for call in calls] == [1000]


def test_openai_provider_preserves_billed_response_when_expansion_request_fails(
    monkeypatch,
):
    cfg = _reset_cfg(monkeypatch)
    calls: list[dict] = []

    class _Resp:
        text = ""

        def __init__(self, status_code: int, payload: dict) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    def _fake_post(url, *, headers, json, timeout):  # noqa: A002
        _ = (url, headers, timeout)
        calls.append(json)
        if len(calls) == 1:
            return _Resp(
                200,
                {
                    "model": "gpt-5-mini",
                    "output_text": '{"ok": true',
                    "usage": {"input_tokens": 17, "output_tokens": 9},
                },
            )
        return _Resp(500, {"error": {"message": "expansion unavailable"}})

    monkeypatch.setattr(
        "app.llm.providers.openai_provider.requests.post",
        _fake_post,
    )

    with pytest.raises(RuntimeError) as exc_info:
        OpenAIProvider(cfg).synthesize_json(
            prompt="hello",
            schema={"type": "object"},
            schema_name="expansion_failure_schema",
            max_output_tokens=1000,
            max_retries_per_request=0,
        )

    billed = exc_info.value._provider_response_attempts
    assert len(calls) == 2
    assert len(billed) == 1
    assert billed[0]["usage"] == {"input_tokens": 17, "output_tokens": 9}


def test_openai_provider_strict_cost_context_makes_incomplete_json_one_loud_attempt(
    monkeypatch,
) -> None:
    cfg = _reset_cfg(monkeypatch)
    calls: list[dict] = []

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output_text": '{"ok": true}',
                "usage": {"input_tokens": 100, "output_tokens": 1200},
            }

    def _fake_post(url, *, headers, json, timeout):  # noqa: A002
        _ = (url, headers, timeout)
        calls.append(json)
        return _Resp()

    monkeypatch.setattr("app.llm.providers.openai_provider.requests.post", _fake_post)
    with llm_cost_budget(max_cost_usd=1.0, strict_first_call=True):
        with pytest.raises(OpenAIOutputTruncatedError, match="incomplete") as exc_info:
            OpenAIProvider(cfg).synthesize_json(
                prompt="strict",
                schema={"type": "object"},
                schema_name="strict_incomplete",
            )

    assert [call["max_output_tokens"] for call in calls] == [1200]
    assert exc_info.value._provider_response_attempts[0]["usage"] == {
        "input_tokens": 100,
        "output_tokens": 1200,
    }


def test_openai_provider_strict_cost_context_rejects_output_cap_drift_before_network(
    monkeypatch,
) -> None:
    cfg = _reset_cfg(monkeypatch)
    network_calls: list[str] = []
    monkeypatch.setattr(
        "app.llm.providers.openai_provider.requests.post",
        lambda *args, **kwargs: network_calls.append("called"),
    )

    with llm_cost_budget(max_cost_usd=1.0, strict_first_call=True):
        with pytest.raises(RuntimeError, match="strict output-token cap exceeded"):
            OpenAIProvider(cfg).synthesize_json(
                prompt="strict",
                schema={"type": "object"},
                max_output_tokens=1201,
            )

    assert network_calls == []


def test_openai_provider_strict_cost_context_does_not_retry_malformed_output(
    monkeypatch,
) -> None:
    cfg = _reset_cfg(monkeypatch)
    calls: list[dict] = []

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "output_text": '{"ok": true',
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

    def _fake_post(url, *, headers, json, timeout):  # noqa: A002
        _ = (url, headers, timeout)
        calls.append(json)
        return _Resp()

    monkeypatch.setattr("app.llm.providers.openai_provider.requests.post", _fake_post)
    with llm_cost_budget(max_cost_usd=1.0, strict_first_call=True):
        with pytest.raises(RuntimeError, match="not valid JSON"):
            OpenAIProvider(cfg).synthesize_json(
                prompt="strict malformed",
                schema={"type": "object"},
            )

    assert len(calls) == 1


@pytest.mark.parametrize("bad_limit", [0, -1, True])
def test_openai_provider_rejects_nonpositive_or_boolean_output_limit(
    monkeypatch,
    bad_limit,
) -> None:
    cfg = _reset_cfg(monkeypatch)
    with pytest.raises(ValueError, match="positive integer"):
        OpenAIProvider(cfg).synthesize_json(
            prompt="invalid limit",
            schema={"type": "object"},
            max_output_tokens=bad_limit,
        )


def test_openai_provider_never_accepts_incomplete_valid_json_at_final_cap(
    monkeypatch,
) -> None:
    cfg = _reset_cfg(monkeypatch)
    calls: list[dict] = []

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output_text": '{"ok": true}',
                "usage": {"input_tokens": 20, "output_tokens": 8000},
            }

    def _fake_post(url, *, headers, json, timeout):  # noqa: A002
        _ = (url, headers, timeout)
        calls.append(json)
        return _Resp()

    monkeypatch.setattr("app.llm.providers.openai_provider.requests.post", _fake_post)
    with pytest.raises(OpenAIOutputTruncatedError, match="max_output_tokens"):
        OpenAIProvider(cfg).synthesize_json(
            prompt="at cap",
            schema={"type": "object"},
            max_output_tokens=8000,
        )

    assert len(calls) == 1
