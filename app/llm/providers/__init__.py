from __future__ import annotations

from app.config import get_config
from app.llm.providers.anthropic_provider import (
    AnthropicProvider,
    opus_invocation_allowed,
)
from app.llm.providers.disabled_provider import DisabledLLMProvider
from app.llm.providers.deepseek_provider import (
    DeepSeekOutputTruncatedError as DeepSeekOutputTruncatedError,
    DeepSeekProvider,
)
from app.llm.providers.openai_provider import OpenAIProvider
from app.llm.execution_policy import active_llm_execution_policy

# Cheap default for any implicit Anthropic usage (e.g. the OpenAI-quota
# fallback path). Opus is never selected implicitly.
CHEAP_ANTHROPIC_FALLBACK_MODEL = "claude-haiku-4-5"


def get_llm_provider():
    cfg = get_config()
    policy = active_llm_execution_policy()
    if policy is not None:
        # A paid v2 benchmark is pinned to OpenAI.  Returning the OpenAI
        # provider here also prevents a configured Anthropic provider from
        # being called while the context-local execution policy is active.
        return OpenAIProvider(cfg)
    provider = (cfg.llm_provider or "disabled").strip().lower()
    if provider == "openai":
        return OpenAIProvider(cfg)
    if provider == "anthropic":
        return AnthropicProvider(cfg)
    if provider == "deepseek":
        return DeepSeekProvider(cfg)
    return DisabledLLMProvider(cfg)


def get_anthropic_provider() -> AnthropicProvider | None:
    """Return an Anthropic provider for flows that require an explicit key check.

    Governance: never hand back a provider implicitly bound to an Opus model.
    If the configured model is Opus-class and the operator has not opted in
    via VOE_ALLOW_OPUS, fall back to a cheap model rather than risking an
    accidental Opus invocation through the OpenAI-quota fallback path.
    """
    policy = active_llm_execution_policy()
    if policy is not None and not policy.allow_provider_fallback:
        return None
    cfg = get_config()
    if not cfg.anthropic_api_key:
        return None
    configured_model = str(cfg.anthropic_model or "")
    if "opus" in configured_model.lower() and not opus_invocation_allowed():
        return AnthropicProvider(cfg, model=CHEAP_ANTHROPIC_FALLBACK_MODEL)
    return AnthropicProvider(cfg)
