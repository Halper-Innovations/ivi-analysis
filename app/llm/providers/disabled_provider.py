from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import AppConfig, get_config


@dataclass
class LLMResult:
    json_text: str
    model: str
    usage_input_tokens: int | None
    usage_output_tokens: int | None
    raw: dict[str, Any]
    usage_cached_input_tokens: int | None = None


class DisabledLLMProvider:
    provider_name = "disabled"

    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or get_config()

    def enabled(self) -> bool:
        return False

    def synthesize_json(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str | None = None,
    ) -> LLMResult:
        _ = (prompt, schema, schema_name)
        raise RuntimeError("LLM provider disabled")
