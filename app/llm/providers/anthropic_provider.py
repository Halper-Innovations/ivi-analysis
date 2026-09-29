"""Anthropic API provider for structured LLM synthesis and validation flows."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from app.config import AppConfig, get_config
from app.util.http import require_network
from app.llm.providers.disabled_provider import LLMResult
from app.llm.providers.retry_guard import (
    DEFAULT_LLM_CALL_TIMEOUT_SECONDS,
    call_with_llm_retry_guard,
)

logger = logging.getLogger(__name__)

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None  # type: ignore[assignment]


def opus_invocation_allowed() -> bool:
    """Whether Opus-class models may be invoked.

    Hard governance guard: Opus is expensive and must never run implicitly.
    Callers must opt in explicitly via VOE_ALLOW_OPUS to use any model whose
    name contains 'opus'.
    """
    return os.getenv("VOE_ALLOW_OPUS", "").strip().lower() in {"1", "true", "yes", "enabled", "on"}


def assert_model_allowed(model: str | None) -> None:
    """Refuse Opus-class models unless explicitly opted in via VOE_ALLOW_OPUS.

    Defense-in-depth against the OpenAI-quota fallback path silently re-issuing
    on an Opus default if VOE_ANTHROPIC_MODEL is cleared.
    """
    name = str(model or "").lower()
    if "opus" in name and not opus_invocation_allowed():
        raise RuntimeError(
            f"Refusing to use Opus-class model '{model}': set VOE_ALLOW_OPUS=1 to "
            "explicitly opt in (Opus must never run implicitly)."
        )


class AnthropicProvider:
    provider_name = "anthropic"
    _handles_retry_guard = True

    def __init__(self, cfg: AppConfig | None = None, *, model: str | None = None) -> None:
        self.cfg = cfg or get_config()
        # Explicit override wins; otherwise inherit the configured model.
        self.model = str(model) if model else str(self.cfg.anthropic_model)
        # Hard guard: never construct a provider bound to an Opus model
        # unless the operator explicitly opted in.
        assert_model_allowed(self.model)

    def enabled(self) -> bool:
        return bool(self.cfg.anthropic_api_key)

    def synthesize_json(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        """Call Anthropic Messages API with tool use for structured JSON output."""
        if not self.enabled():
            raise RuntimeError("Anthropic provider not enabled (no API key)")
        if _anthropic_sdk is None:
            raise RuntimeError("anthropic package not installed (pip install anthropic)")

        tool_name = (schema_name or "structured_output").replace("-", "_")[:64]
        tool_def = {
            "name": tool_name,
            "description": "Return structured analysis results.",
            "input_schema": schema,
        }

        max_tokens = max_output_tokens or self.cfg.anthropic_max_output_tokens

        def _call_once() -> LLMResult:
            # VOE_NET_PROVIDER is the single switch for all outbound traffic.
            require_network(self.cfg.anthropic_base_url, self.cfg)
            client = _anthropic_sdk.Anthropic(
                api_key=self.cfg.anthropic_api_key,
                base_url=self.cfg.anthropic_base_url,
                timeout=min(float(self.cfg.anthropic_request_timeout), DEFAULT_LLM_CALL_TIMEOUT_SECONDS),
                max_retries=0,
            )

            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                tools=[tool_def],
                tool_choice={"type": "tool", "name": tool_name},
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract tool use result
            tool_input = None
            for block in response.content:
                if block.type == "tool_use" and block.name == tool_name:
                    tool_input = block.input
                    break

            if tool_input is None:
                raise RuntimeError(
                    f"Anthropic response did not include tool use for {tool_name}. "
                    f"Stop reason: {response.stop_reason}"
                )

            json_text = json.dumps(tool_input)
            return LLMResult(
                json_text=json_text,
                model=self.model,
                usage_input_tokens=response.usage.input_tokens if response.usage else None,
                usage_output_tokens=response.usage.output_tokens if response.usage else None,
                raw={"stop_reason": response.stop_reason, "model": response.model},
            )

        return call_with_llm_retry_guard(
            provider_name=self.provider_name,
            schema_name=tool_name,
            call=_call_once,
            timeout_seconds=min(float(self.cfg.anthropic_request_timeout), DEFAULT_LLM_CALL_TIMEOUT_SECONDS),
        )
