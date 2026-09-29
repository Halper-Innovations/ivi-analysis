from __future__ import annotations

import json
import stat

import pytest

from app.config import get_config
from app.llm.providers import (
    DeepSeekOutputTruncatedError,
    DeepSeekProvider,
    get_llm_provider,
)
from app.llm.usage_capture import provider_usage_records_from_exception


def _reset_cfg(monkeypatch):
    monkeypatch.setenv("VOE_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("VOE_DEEPSEEK_API_KEY", "sk-deepseek-test-key")
    monkeypatch.setenv("VOE_DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("VOE_DEEPSEEK_MAX_OUTPUT_TOKENS", "1200")
    # The transport is mocked in these tests; the network switch is on.
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    get_config.cache_clear()
    return get_config()


def test_deepseek_provider_builds_strict_non_thinking_json_request(monkeypatch) -> None:
    cfg = _reset_cfg(monkeypatch)
    captured: dict[str, object] = {}

    class _Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "id": "chatcmpl-test",
                "model": "deepseek-v4-pro",
                "choices": [{"message": {"role": "assistant", "content": '{"ok":true}'}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "prompt_cache_hit_tokens": 40,
                    "prompt_cache_miss_tokens": 60,
                },
            }

    def _fake_post(url, *, headers, json, timeout):  # noqa: A002
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return _Response()

    monkeypatch.setattr("app.llm.providers.deepseek_provider.requests.post", _fake_post)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    result = DeepSeekProvider(cfg).synthesize_json(
        prompt="Return JSON for the company.",
        schema=schema,
        schema_name="candidate_memo",
    )

    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "deepseek-v4-pro"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["stream"] is False
    assert payload["max_tokens"] == 1200
    assert "temperature" not in payload
    assert payload["messages"][1] == {
        "role": "user",
        "content": "Return JSON for the company.",
    }
    assert "candidate_memo JSON Schema" in payload["messages"][0]["content"]
    assert (
        json.dumps(schema, sort_keys=True, separators=(",", ":"))
        in payload["messages"][0]["content"]
    )
    assert json.loads(result.json_text) == {"ok": True}
    assert result.model == "deepseek-v4-pro"
    assert result.usage_input_tokens == 100
    assert result.usage_cached_input_tokens == 40
    assert result.usage_output_tokens == 25


def test_deepseek_provider_reports_length_finish_as_first_class_truncation(
    monkeypatch,
) -> None:
    cfg = _reset_cfg(monkeypatch)

    class _Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": '{"research_questions":[{"question_id":"R1"',
                            "reasoning_content": None,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 16275,
                    "completion_tokens": 1800,
                    "prompt_cache_hit_tokens": 8064,
                },
            }

    monkeypatch.setattr(
        "app.llm.providers.deepseek_provider.requests.post",
        lambda *args, **kwargs: _Response(),
    )
    provider = DeepSeekProvider(cfg)
    with pytest.raises(DeepSeekOutputTruncatedError) as exc_info:
        provider.synthesize_json(
            prompt="compact automotive recovery",
            schema={"type": "object"},
            schema_name="autonomous_sector_minimum_tool_plan",
            max_output_tokens=1800,
        )

    assert str(exc_info.value) == (
        "LLM_PROVIDER_TRUNCATED: DeepSeek response finish_reason=length "
        "at requested max_output_tokens=1800"
    )
    records = provider_usage_records_from_exception(
        provider=provider,
        error=exc_info.value,
        prompt="compact automotive recovery",
        schema_name="autonomous_sector_minimum_tool_plan",
    )
    assert len(records) == 1
    assert records[0]["status"] == "OK"
    assert records[0]["provider"] == "deepseek"
    assert records[0]["model"] == "deepseek-v4-pro"
    assert records[0]["schema_name"] == "autonomous_sector_minimum_tool_plan"
    assert records[0]["input_tokens"] == 16275
    assert records[0]["cached_input_tokens"] == 8064
    assert records[0]["output_tokens"] == 1800
    assert records[0]["physical_response_sequence"] == 1
    assert records[0]["physical_response_count"] == 1


def test_deepseek_provider_is_selected_by_registry(monkeypatch) -> None:
    cfg = _reset_cfg(monkeypatch)
    provider = get_llm_provider()
    assert isinstance(provider, DeepSeekProvider)
    assert provider.cfg is cfg
    assert provider.enabled()


def test_deepseek_provider_rejects_model_and_feature_drift_before_network(
    monkeypatch,
) -> None:
    cfg = _reset_cfg(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        "app.llm.providers.deepseek_provider.requests.post",
        lambda *args, **kwargs: calls.append("called"),
    )
    provider = DeepSeekProvider(cfg)
    with pytest.raises(RuntimeError, match="model drift"):
        provider.synthesize_json(
            prompt="test", schema={"type": "object"}, model="deepseek-v4-flash"
        )
    with pytest.raises(ValueError, match="does not support tools or tiers"):
        provider.synthesize_json(
            prompt="test",
            schema={"type": "object"},
            tools=[{"type": "web_search"}],
        )
    with pytest.raises(ValueError, match="output retries are disabled"):
        provider.synthesize_json(
            prompt="test",
            schema={"type": "object"},
            allow_output_token_retry=True,
        )
    assert calls == []


def test_deepseek_provider_redacts_secret_from_error(monkeypatch) -> None:
    cfg = _reset_cfg(monkeypatch)

    class _Response:
        status_code = 400
        text = ""

        @staticmethod
        def json():
            return {"error": {"message": "bad sk-deepseek-test-key"}}

    monkeypatch.setattr(
        "app.llm.providers.deepseek_provider.requests.post",
        lambda *args, **kwargs: _Response(),
    )
    with pytest.raises(RuntimeError) as exc:
        DeepSeekProvider(cfg).synthesize_json(prompt="test", schema={"type": "object"})
    assert "status=400" in str(exc.value)
    assert "sk-deepseek-test-key" not in str(exc.value)
    assert "***REDACTED***" in str(exc.value)


def test_deepseek_billed_invalid_json_preserves_physical_usage(monkeypatch) -> None:
    cfg = _reset_cfg(monkeypatch)

    class _Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "model": "deepseek-v4-pro",
                "choices": [{"message": {"content": "not-json"}}],
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 50,
                    "prompt_cache_hit_tokens": 20,
                },
            }

    monkeypatch.setattr(
        "app.llm.providers.deepseek_provider.requests.post",
        lambda *args, **kwargs: _Response(),
    )
    provider = DeepSeekProvider(cfg)
    with pytest.raises(RuntimeError, match="not valid JSON") as exc:
        provider.synthesize_json(prompt="memo", schema={"type": "object"})
    records = provider_usage_records_from_exception(
        provider=provider,
        error=exc.value,
        prompt="memo",
        schema_name="candidate_memo",
    )
    assert len(records) == 1
    assert records[0]["provider"] == "deepseek"
    assert records[0]["model"] == "deepseek-v4-pro"
    assert records[0]["input_tokens"] == 200
    assert records[0]["cached_input_tokens"] == 20
    assert records[0]["output_tokens"] == 50
    assert records[0]["cost_estimate_usd"] == 0.000122


def test_deepseek_capture_fsyncs_exact_bytes_before_content_parse(
    monkeypatch,
    tmp_path,
) -> None:
    cfg = _reset_cfg(monkeypatch)
    raw_path = tmp_path / "compact_probe.response.bin"
    metadata_path = tmp_path / "compact_probe.response.bin.metadata.json"
    response_bytes = (
        b'{"id":"probe-live-shape","model":"deepseek-v4-pro",'
        b'"choices":[{"finish_reason":"stop","message":{"role":"assistant",'
        b'"content":"not-json","reasoning_content":"analysis preamble"}}],'
        b'"usage":{"prompt_tokens":12345,"completion_tokens":777,'
        b'"prompt_cache_hit_tokens":12000}}'
    )
    parse_observations: list[str] = []
    original_normalize = DeepSeekProvider._normalize_json_text

    class _Response:
        status_code = 200
        content = response_bytes
        text = response_bytes.decode("utf-8")

        @staticmethod
        def json():
            raise AssertionError("captured responses must parse from persisted bytes")

    def _inspect_before_content_parse(text: str) -> str:
        assert raw_path.read_bytes() == response_bytes
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["capture_status"] == "ENVELOPE_RECORDED_AFTER_RAW_PERSIST"
        assert metadata["capture_ordering"] == ("raw_response_bytes_fsynced_before_any_json_parse")
        parse_observations.append("raw-already-durable")
        return original_normalize(text)

    monkeypatch.setattr(
        "app.llm.providers.deepseek_provider.requests.post",
        lambda *args, **kwargs: _Response(),
    )
    monkeypatch.setattr(
        DeepSeekProvider,
        "_normalize_json_text",
        staticmethod(_inspect_before_content_parse),
    )

    with pytest.raises(RuntimeError, match="not valid JSON"):
        DeepSeekProvider(cfg).synthesize_json(
            prompt="compact automotive recovery",
            schema={"type": "object"},
            schema_name="autonomous_sector_minimum_tool_plan",
            max_output_tokens=1800,
            max_retries_per_request=0,
            raw_response_capture_path=raw_path,
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert parse_observations == ["raw-already-durable"]
    assert raw_path.read_bytes() == response_bytes
    assert metadata["response_body_sha256"] == (
        "870e55e233185cb4164a5ea8d14d3a5762d28abcc545b49067951b6ab9c5571f"
    )
    assert metadata["response_body_bytes"] == 268
    assert metadata["finish_reason"] == "stop"
    assert metadata["content"] == "not-json"
    assert metadata["reasoning_content"] == "analysis preamble"
    assert metadata["usage"] == {
        "prompt_tokens": 12345,
        "completion_tokens": 777,
        "prompt_cache_hit_tokens": 12000,
    }
    assert metadata["schema_name"] == "autonomous_sector_minimum_tool_plan"
    assert metadata["model"] == "deepseek-v4-pro"
    assert metadata["requested_max_output_tokens"] == 1800
    assert stat.S_IMODE(raw_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(metadata_path.stat().st_mode) == 0o600


def test_deepseek_capture_creation_failure_refuses_before_transport(
    monkeypatch,
    tmp_path,
) -> None:
    cfg = _reset_cfg(monkeypatch)
    raw_path = tmp_path / "already-exists.response.bin"
    raw_path.write_bytes(b"owner evidence")
    transport_calls: list[str] = []
    monkeypatch.setattr(
        "app.llm.providers.deepseek_provider.requests.post",
        lambda *args, **kwargs: transport_calls.append("called"),
    )

    with pytest.raises(FileExistsError):
        DeepSeekProvider(cfg).synthesize_json(
            prompt="compact automotive recovery",
            schema={"type": "object"},
            schema_name="autonomous_sector_minimum_tool_plan",
            max_output_tokens=1800,
            max_retries_per_request=0,
            raw_response_capture_path=raw_path,
        )

    assert transport_calls == []
    assert raw_path.read_bytes() == b"owner evidence"
    assert not (tmp_path / "already-exists.response.bin.metadata.json").exists()


def test_deepseek_capture_phases_increment_counter_at_requests_post(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    cfg = _reset_cfg(monkeypatch)
    raw_path = tmp_path / "phase-probe.response.bin"
    metadata_path = tmp_path / "phase-probe.response.bin.metadata.json"
    response_bytes = (
        b'{"model":"deepseek-v4-pro","choices":[{"finish_reason":"stop",'
        b'"message":{"content":"{\\"ok\\":true}","reasoning_content":null}}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":5}}'
    )
    phases: list[str] = []
    transport_counter = 0
    post_observations: list[tuple[int, str]] = []

    class _Response:
        status_code = 200
        content = response_bytes
        text = response_bytes.decode("utf-8")

    def _before_transport() -> None:
        nonlocal transport_counter
        transport_counter += 1

    def _fake_post(*args, **kwargs):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        post_observations.append((transport_counter, metadata["capture_status"]))
        return _Response()

    monkeypatch.setattr(
        "app.llm.providers.deepseek_provider.requests.post",
        _fake_post,
    )

    result = DeepSeekProvider(cfg).synthesize_json(
        prompt="compact automotive recovery",
        schema={"type": "object"},
        schema_name="autonomous_sector_minimum_tool_plan",
        max_output_tokens=1800,
        max_retries_per_request=0,
        raw_response_capture_path=raw_path,
        diagnostic_phase_callback=phases.append,
        before_transport_callback=_before_transport,
    )

    assert cfg.safe_mode is True
    assert json.loads(result.json_text) == {"ok": True}
    assert transport_counter == 1
    assert post_observations == [(1, "ARMED_BEFORE_TRANSPORT")]
    assert phases == [
        "provider_validation_started",
        "provider_validation_complete",
        "request_payload_built",
        "capture_arming_started",
        "capture_armed",
        "response_received",
    ]


def test_deepseek_missing_credential_fails_during_validation_before_capture(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("VOE_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("VOE_DEEPSEEK_API_KEY", "")
    get_config.cache_clear()
    cfg = get_config()
    provider = DeepSeekProvider(cfg)
    phases: list[str] = []
    transport_calls: list[str] = []
    raw_path = tmp_path / "missing-key.response.bin"
    monkeypatch.setattr(
        "app.llm.providers.deepseek_provider.requests.post",
        lambda *args, **kwargs: transport_calls.append("called"),
    )

    assert provider.enabled() is False
    with pytest.raises(RuntimeError, match="provider not enabled"):
        provider.synthesize_json(
            prompt="compact automotive recovery",
            schema={"type": "object"},
            max_retries_per_request=0,
            raw_response_capture_path=raw_path,
            diagnostic_phase_callback=phases.append,
            before_transport_callback=lambda: transport_calls.append("callback"),
        )

    assert phases == ["provider_validation_started"]
    assert transport_calls == []
    assert not raw_path.exists()


def test_deepseek_capture_refuses_retryable_transport_before_transport(
    monkeypatch,
    tmp_path,
) -> None:
    cfg = _reset_cfg(monkeypatch)
    raw_path = tmp_path / "must-not-exist.response.bin"
    transport_calls: list[str] = []
    monkeypatch.setattr(
        "app.llm.providers.deepseek_provider.requests.post",
        lambda *args, **kwargs: transport_calls.append("called"),
    )

    with pytest.raises(ValueError, match="exactly one transport attempt"):
        DeepSeekProvider(cfg).synthesize_json(
            prompt="compact automotive recovery",
            schema={"type": "object"},
            schema_name="autonomous_sector_minimum_tool_plan",
            max_output_tokens=1800,
            max_retries_per_request=1,
            raw_response_capture_path=raw_path,
        )

    assert transport_calls == []
    assert not raw_path.exists()
    assert not (tmp_path / "must-not-exist.response.bin.metadata.json").exists()
