"""Tests for Anthropic provider."""
import json
from unittest.mock import MagicMock, patch

from app.llm.providers.anthropic_provider import AnthropicProvider
from app.llm.providers.disabled_provider import LLMResult


class TestAnthropicProvider:
    def test_enabled_with_api_key(self):
        cfg = MagicMock()
        cfg.anthropic_api_key = "test-key"
        provider = AnthropicProvider(cfg)
        assert provider.enabled() is True

    def test_disabled_without_api_key(self):
        cfg = MagicMock()
        cfg.anthropic_api_key = None
        provider = AnthropicProvider(cfg)
        assert provider.enabled() is False

    def test_synthesize_json_returns_llm_result(self):
        cfg = MagicMock()
        cfg.anthropic_api_key = "test-key"
        cfg.anthropic_model = "claude-haiku-4-5"
        cfg.anthropic_max_output_tokens = 8000
        cfg.anthropic_request_timeout = 120

        mock_block = MagicMock()
        mock_block.type = "tool_use"
        mock_block.name = "test_schema"
        mock_block.input = {"result": "success"}

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.stop_reason = "end_turn"
        mock_response.model = "claude-haiku-4-5"
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 500

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("app.llm.providers.anthropic_provider._anthropic_sdk") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            provider = AnthropicProvider(cfg)
            result = provider.synthesize_json(
                prompt="test prompt",
                schema={"type": "object", "properties": {"result": {"type": "string"}}},
                schema_name="test_schema",
            )

        assert isinstance(result, LLMResult)
        assert json.loads(result.json_text) == {"result": "success"}
        assert result.model == "claude-haiku-4-5"
        assert result.usage_input_tokens == 1000
        assert result.usage_output_tokens == 500

    def test_raises_when_disabled(self):
        cfg = MagicMock()
        cfg.anthropic_api_key = None
        provider = AnthropicProvider(cfg)
        import pytest
        with pytest.raises(RuntimeError, match="not enabled"):
            provider.synthesize_json(
                prompt="test",
                schema={"type": "object"},
            )

    def test_raises_when_no_tool_use_in_response(self):
        cfg = MagicMock()
        cfg.anthropic_api_key = "test-key"
        cfg.anthropic_model = "claude-haiku-4-5"
        cfg.anthropic_max_output_tokens = 8000
        cfg.anthropic_request_timeout = 120

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.name = None

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("app.llm.providers.anthropic_provider._anthropic_sdk") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            provider = AnthropicProvider(cfg)
            import pytest
            with pytest.raises(RuntimeError, match="did not include tool use"):
                provider.synthesize_json(
                    prompt="test",
                    schema={"type": "object"},
                    schema_name="test_schema",
                )


class TestConfigDefaultModel:
    def test_default_anthropic_model_is_haiku(self, monkeypatch):
        from app.config import AppConfig, get_config
        monkeypatch.delenv("VOE_ANTHROPIC_MODEL", raising=False)
        get_config.cache_clear()
        cfg = get_config()
        assert cfg.anthropic_model == "claude-haiku-4-5"
        # The bare pydantic default must also be off opus.
        assert AppConfig().anthropic_model == "claude-haiku-4-5"
        get_config.cache_clear()


class TestOpusGuard:
    def test_constructing_opus_provider_raises_without_optin(self, monkeypatch):
        monkeypatch.delenv("VOE_ALLOW_OPUS", raising=False)
        cfg = MagicMock()
        cfg.anthropic_api_key = "test-key"
        cfg.anthropic_model = "claude-opus-4-6"
        import pytest
        with pytest.raises(RuntimeError, match="opus"):
            AnthropicProvider(cfg)

    def test_constructing_opus_provider_allowed_with_optin(self, monkeypatch):
        monkeypatch.setenv("VOE_ALLOW_OPUS", "1")
        cfg = MagicMock()
        cfg.anthropic_api_key = "test-key"
        cfg.anthropic_model = "claude-opus-4-6"
        provider = AnthropicProvider(cfg)
        assert provider.cfg.anthropic_model == "claude-opus-4-6"

    def test_non_opus_model_constructs_without_optin(self, monkeypatch):
        monkeypatch.delenv("VOE_ALLOW_OPUS", raising=False)
        cfg = MagicMock()
        cfg.anthropic_api_key = "test-key"
        cfg.anthropic_model = "claude-haiku-4-5"
        provider = AnthropicProvider(cfg)
        assert provider.enabled() is True

    def test_explicit_model_override_uses_cheap_model(self, monkeypatch):
        monkeypatch.delenv("VOE_ALLOW_OPUS", raising=False)
        cfg = MagicMock()
        cfg.anthropic_api_key = "test-key"
        cfg.anthropic_model = "claude-opus-4-6"
        # An explicit cheap model override must bypass the opus default safely.
        provider = AnthropicProvider(cfg, model="claude-haiku-4-5")
        assert provider.model == "claude-haiku-4-5"


class TestGetAnthropicProviderModelSafety:
    def test_fallback_provider_does_not_inherit_opus(self, monkeypatch):
        from app.config import get_config
        monkeypatch.setenv("VOE_ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("VOE_ANTHROPIC_MODEL", "claude-opus-4-6")
        monkeypatch.delenv("VOE_ALLOW_OPUS", raising=False)
        get_config.cache_clear()
        from app.llm.providers import get_anthropic_provider
        # Without opt-in, the helper must return a provider on a cheap model,
        # never silently constructing on opus.
        provider = get_anthropic_provider()
        assert provider is not None
        assert "opus" not in provider.model.lower()
        get_config.cache_clear()

    def test_get_anthropic_provider_allows_opus_with_optin(self, monkeypatch):
        from app.config import get_config
        monkeypatch.setenv("VOE_ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("VOE_ANTHROPIC_MODEL", "claude-opus-4-6")
        monkeypatch.setenv("VOE_ALLOW_OPUS", "1")
        get_config.cache_clear()
        from app.llm.providers import get_anthropic_provider
        provider = get_anthropic_provider()
        assert provider is not None
        assert provider.model == "claude-opus-4-6"
        get_config.cache_clear()


class TestGetAnthropicProvider:
    def test_returns_provider_with_key(self, monkeypatch):
        from app.config import get_config
        monkeypatch.setenv("VOE_ANTHROPIC_API_KEY", "test-key")
        get_config.cache_clear()
        from app.llm.providers import get_anthropic_provider
        provider = get_anthropic_provider()
        assert provider is not None
        assert provider.provider_name == "anthropic"
        get_config.cache_clear()

    def test_returns_none_without_key(self, monkeypatch):
        from app.config import get_config
        monkeypatch.delenv("VOE_ANTHROPIC_API_KEY", raising=False)
        get_config.cache_clear()
        from app.llm.providers import get_anthropic_provider
        provider = get_anthropic_provider()
        assert provider is None
        get_config.cache_clear()


class TestGlobalLLMProviderRouting:
    def test_routes_to_anthropic_when_configured(self, monkeypatch):
        from app.config import get_config

        monkeypatch.setenv("VOE_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("VOE_ANTHROPIC_API_KEY", "anthropic-test-key")
        monkeypatch.delenv("VOE_OPENAI_API_KEY", raising=False)
        get_config.cache_clear()

        from app.llm.providers import get_llm_provider

        provider = get_llm_provider()

        assert provider.provider_name == "anthropic"
        assert provider.enabled() is True
        get_config.cache_clear()

    def test_configured_anthropic_without_key_is_disabled(self, monkeypatch):
        from app.config import get_config

        monkeypatch.setenv("VOE_LLM_PROVIDER", "anthropic")
        monkeypatch.delenv("VOE_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("VOE_OPENAI_API_KEY", raising=False)
        get_config.cache_clear()

        from app.llm.providers import get_llm_provider

        provider = get_llm_provider()

        assert provider.provider_name == "anthropic"
        assert provider.enabled() is False
        get_config.cache_clear()
