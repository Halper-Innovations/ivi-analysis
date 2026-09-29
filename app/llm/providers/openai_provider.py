from __future__ import annotations

import json
import re
from typing import Any

import requests

from app.util.http import require_network

from app.config import AppConfig, get_config
from app.llm.providers.disabled_provider import LLMResult
from app.llm.providers.retry_guard import (
    DEFAULT_LLM_CALL_TIMEOUT_SECONDS,
    DEFAULT_LLM_MAX_RETRIES_PER_CALL,
    call_with_llm_retry_guard,
    current_cost_context,
)
from app.llm.execution_policy import enforce_openai_request_policy
from app.logging import get_logger


logger = get_logger(__name__)


_OPENAI_CIRCUIT_BREAKER_REASON: str | None = None
_PROVIDER_RESPONSE_ATTEMPTS_ATTR = "_provider_response_attempts"


class OpenAIOutputTruncatedError(RuntimeError):
    """A billed response ended incomplete and cannot satisfy its schema."""


def _attach_provider_response_attempts(
    error: BaseException,
    response_payloads: list[dict[str, Any]],
) -> None:
    """Preserve billed 2xx responses when later logical processing fails."""

    prior = getattr(error, _PROVIDER_RESPONSE_ATTEMPTS_ATTR, None)
    prior = (
        [payload for payload in prior if isinstance(payload, dict)]
        if isinstance(prior, list)
        else []
    )
    current = [payload for payload in response_payloads if isinstance(payload, dict)]
    if prior and current[-len(prior) :] == prior:
        merged = current
    elif current and prior[: len(current)] == current:
        merged = prior
    else:
        merged = [*current, *prior]
    if not merged:
        return
    try:
        setattr(error, _PROVIDER_RESPONSE_ATTEMPTS_ATTR, merged)
    except Exception:  # pragma: no cover - ordinary provider exceptions are mutable
        return


class OpenAIProvider:
    provider_name = "openai"
    _handles_retry_guard = True

    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or get_config()
        self.base_url = "https://api.openai.com/v1/responses"

    def enabled(self) -> bool:
        return bool(self.cfg.openai_api_key and (self.cfg.llm_provider or "").lower() == "openai")

    @staticmethod
    def _reset_circuit_breaker() -> None:
        global _OPENAI_CIRCUIT_BREAKER_REASON
        _OPENAI_CIRCUIT_BREAKER_REASON = None

    @staticmethod
    def _quota_failure_reason(status_code: int, error_body: Any) -> str | None:
        if status_code != 429:
            return None
        if not isinstance(error_body, dict):
            return None
        error = error_body.get("error")
        if not isinstance(error, dict):
            return None
        code = str(error.get("code") or "").strip().lower()
        message = str(error.get("message") or "").strip().lower()
        if code == "insufficient_quota" or "insufficient quota" in message:
            return "status=429 insufficient_quota"
        return None

    @staticmethod
    def _extract_structured_output(payload: dict[str, Any]) -> dict[str, Any] | list[Any] | None:
        top_output_json = payload.get("output_json")
        if isinstance(top_output_json, (dict, list)):
            return top_output_json
        output_parsed = payload.get("output_parsed")
        if isinstance(output_parsed, (dict, list)):
            return output_parsed

        output = payload.get("output")
        if not isinstance(output, list):
            return None
        for item in output:
            if not isinstance(item, dict):
                continue
            for key in ("json", "parsed"):
                structured = item.get(key)
                if isinstance(structured, (dict, list)):
                    return structured
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                for key in ("json", "parsed"):
                    structured = part.get(key)
                    if isinstance(structured, (dict, list)):
                        return structured
        return None

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        structured = OpenAIProvider._extract_structured_output(payload)
        if isinstance(structured, (dict, list)):
            return json.dumps(structured)

        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        if isinstance(output_text, list) and output_text:
            chunks = [str(x) for x in output_text if isinstance(x, str) and str(x).strip()]
            if chunks:
                return "\n".join(chunks).strip()

        output = payload.get("output") or []
        chunks: list[str] = []
        top_output_json = payload.get("output_json")
        if isinstance(top_output_json, (dict, list)):
            chunks.append(json.dumps(top_output_json))
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content") or []
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    for key in ("json", "parsed"):
                        structured = part.get(key)
                        if isinstance(structured, (dict, list)):
                            chunks.append(json.dumps(structured))
                            break
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
        return "\n".join(chunks).strip()

    @staticmethod
    def _stable_schema_name(schema_name: str | None) -> str:
        base = (schema_name or "").strip() or "synthesis_packet_v1"
        sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", base)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        if not sanitized:
            sanitized = "synthesis_packet_v1"
        return sanitized[:64]

    @staticmethod
    def _normalize_json_text(text: str) -> str:
        candidates: list[str] = []
        stripped = text.strip()
        if stripped:
            candidates.append(stripped)

        fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        for match in fence_matches:
            candidate = match.strip()
            if candidate:
                candidates.append(candidate)

        first_obj = stripped.find("{")
        last_obj = stripped.rfind("}")
        if first_obj != -1 and last_obj > first_obj:
            candidates.append(stripped[first_obj : last_obj + 1])

        first_arr = stripped.find("[")
        last_arr = stripped.rfind("]")
        if first_arr != -1 and last_arr > first_arr:
            candidates.append(stripped[first_arr : last_arr + 1])

        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                parsed = json.loads(candidate)
                return json.dumps(parsed)
            except Exception:  # noqa: BLE001
                continue
        raise RuntimeError("OpenAI output is not valid JSON")

    def synthesize_json(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str | None = None,
        max_output_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        include: list[str] | None = None,
        max_tool_calls: int | None = None,
        model: str | None = None,
        service_tier: str | None = None,
        allow_output_token_retry: bool = True,
        max_retries_per_request: int = DEFAULT_LLM_MAX_RETRIES_PER_CALL,
    ) -> LLMResult:
        if not self.enabled():
            raise RuntimeError("OpenAI provider not enabled")
        if _OPENAI_CIRCUIT_BREAKER_REASON:
            raise RuntimeError(
                "OpenAI provider circuit breaker open: "
                f"{_OPENAI_CIRCUIT_BREAKER_REASON}. "
                "Skipping request for the rest of this process."
            )

        headers = {
            "Authorization": f"Bearer {self.cfg.openai_api_key}",
            "Content-Type": "application/json",
        }
        format_name = self._stable_schema_name(schema_name)
        resolved_model = str(model or self.cfg.openai_model).strip()
        if not resolved_model:
            raise ValueError("OpenAI model must not be empty")
        configured_output_limit = self.cfg.openai_max_output_tokens
        if (
            isinstance(configured_output_limit, bool)
            or not isinstance(configured_output_limit, int)
            or configured_output_limit <= 0
        ):
            raise ValueError("configured OpenAI max_output_tokens must be a positive integer")
        if max_output_tokens is not None and (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        resolved_output_limit = (
            configured_output_limit
            if max_output_tokens is None
            else max_output_tokens
        )
        cost_context = current_cost_context()
        strict_cost_cap = bool(cost_context and cost_context.strict_first_call)
        if strict_cost_cap and resolved_output_limit > configured_output_limit:
            raise RuntimeError(
                "OpenAI strict output-token cap exceeded before request: "
                f"requested={resolved_output_limit}, configured={configured_output_limit}"
            )
        if max_tool_calls is not None and (
            isinstance(max_tool_calls, bool) or int(max_tool_calls) < 1
        ):
            raise ValueError("max_tool_calls must be a positive integer")
        if isinstance(max_retries_per_request, bool) or int(max_retries_per_request) < 0:
            raise ValueError("max_retries_per_request must be a non-negative integer")
        normalized_service_tier = (
            str(service_tier or "").strip().lower() or None
        )
        if normalized_service_tier not in {None, "auto", "default", "flex", "priority"}:
            raise ValueError("unsupported OpenAI service_tier")
        request_payload = {
            "model": resolved_model,
            "max_output_tokens": resolved_output_limit,
            "input": prompt,
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": format_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        if tools is not None:
            request_payload["tools"] = [dict(tool) for tool in tools]
        if include is not None:
            request_payload["include"] = [str(item) for item in include]
        if max_tool_calls is not None:
            request_payload["max_tool_calls"] = int(max_tool_calls)
        if normalized_service_tier is not None:
            request_payload["service_tier"] = normalized_service_tier

        execution_policy = enforce_openai_request_policy(
            model=resolved_model,
            request_payload=request_payload,
            max_output_tokens=int(request_payload["max_output_tokens"]),
        )
        if execution_policy is not None:
            allow_output_token_retry = bool(execution_policy.allow_output_token_retry)
            max_retries_per_request = min(
                int(max_retries_per_request),
                execution_policy.max_retries_per_request,
            )
        if strict_cost_cap:
            # A strict campaign reserves one frozen physical request.  Output
            # expansion and transport retry would each rebill the full input
            # outside that reservation, so both are prohibited here.
            allow_output_token_retry = False
            max_retries_per_request = 0

        request_timeout = min(float(self.cfg.openai_request_timeout), DEFAULT_LLM_CALL_TIMEOUT_SECONDS)

        def _request_json(payload: dict[str, Any]) -> dict[str, Any]:
            # VOE_NET_PROVIDER is the single switch for all outbound traffic.
            require_network(self.base_url, self.cfg)
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=request_timeout,
            )
            if response.status_code >= 200 and response.status_code < 300:
                body = response.json()
                normalized_body = body if isinstance(body, dict) else {}
                if normalized_service_tier is not None:
                    returned_tier = str(
                        normalized_body.get("service_tier") or ""
                    ).strip().lower()
                    if returned_tier != normalized_service_tier:
                        error = RuntimeError(
                            "OpenAI Responses API service tier mismatch: "
                            f"requested={normalized_service_tier}, "
                            f"returned={returned_tier or 'missing'}"
                        )
                        # This is a billed 2xx response even though the
                        # execution contract rejected it.  Carry its exact
                        # usage through the exception rather than replacing
                        # it with a conservative failed-call estimate.
                        _attach_provider_response_attempts(error, [normalized_body])
                        raise error
                return normalized_body

            error_body: Any
            try:
                error_body = response.json()
            except Exception:  # noqa: BLE001
                error_body = {"error_text": response.text[:2000]}
            error_text = json.dumps(error_body)
            if self.cfg.openai_api_key:
                error_text = error_text.replace(self.cfg.openai_api_key, "***REDACTED***")
            error_text = re.sub(r"sk-[A-Za-z0-9_-]{6,}", "sk-***REDACTED***", error_text)
            quota_failure_reason = self._quota_failure_reason(response.status_code, error_body)
            if quota_failure_reason:
                global _OPENAI_CIRCUIT_BREAKER_REASON
                _OPENAI_CIRCUIT_BREAKER_REASON = quota_failure_reason
            logger.error(
                "openai_responses_request_failed",
                extra={
                    "stage_name": "llm_provider",
                    "stage_status_code": response.status_code,
                    "stage_error": error_text[:4000],
                },
            )
            raise RuntimeError(
                f"OpenAI Responses API request failed (status={response.status_code}): {error_text}"
            )

        response_payloads: list[dict[str, Any]] = []
        response_output_token_limits: list[int] = []

        def _guarded_request(payload: dict[str, Any]) -> dict[str, Any]:
            try:
                response_payload = call_with_llm_retry_guard(
                    provider_name=self.provider_name,
                    schema_name=format_name,
                    call=lambda: _request_json(payload),
                    timeout_seconds=request_timeout,
                    max_retries=int(max_retries_per_request),
                )
            except Exception as exc:
                _attach_provider_response_attempts(exc, response_payloads)
                raise
            response_payloads.append(response_payload)
            response_output_token_limits.append(int(payload["max_output_tokens"]))
            return response_payload

        payload = _guarded_request(request_payload)
        current_tokens = int(request_payload["max_output_tokens"])
        while (
            allow_output_token_retry
            and payload.get("status") == "incomplete"
            and (payload.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
            and current_tokens < 8000
        ):
            next_tokens = min(current_tokens * 2, 8000)
            if next_tokens <= current_tokens:
                break
            retry_payload = dict(request_payload)
            retry_payload["max_output_tokens"] = next_tokens
            payload = _guarded_request(retry_payload)
            current_tokens = next_tokens

        if str(payload.get("status") or "").strip().lower() == "incomplete":
            reason = str(
                (payload.get("incomplete_details") or {}).get("reason") or "unknown"
            )
            error = OpenAIOutputTruncatedError(
                "OpenAI response is incomplete and cannot satisfy the JSON schema: "
                f"reason={reason}, max_output_tokens={current_tokens}"
            )
            _attach_provider_response_attempts(error, response_payloads)
            raise error

        def _normalized_text_from_payload(response_payload: dict[str, Any]) -> str:
            extracted_text = self._extract_text(response_payload)
            if not extracted_text:
                payload_preview = json.dumps(response_payload)
                payload_preview = re.sub(r"sk-[A-Za-z0-9_-]{6,}", "sk-***REDACTED***", payload_preview)
                raise RuntimeError(
                    f"OpenAI response did not include output text. payload={payload_preview[:4000]}"
                )
            try:
                return self._normalize_json_text(extracted_text)
            except Exception as exc:  # noqa: BLE001
                structured = self._extract_structured_output(response_payload)
                if isinstance(structured, (dict, list)):
                    return json.dumps(structured)
                snippet = extracted_text[:400].replace("\n", "\\n")
                raise RuntimeError(f"OpenAI output is not valid JSON: {exc}; snippet={snippet}") from exc

        try:
            text = _normalized_text_from_payload(payload)
        except RuntimeError as exc:
            if not allow_output_token_retry or current_tokens >= 8000:
                _attach_provider_response_attempts(exc, response_payloads)
                raise
            retry_payload = dict(request_payload)
            retry_payload["max_output_tokens"] = 8000
            payload = _guarded_request(retry_payload)
            current_tokens = 8000
            try:
                text = _normalized_text_from_payload(payload)
            except RuntimeError as retry_exc:
                _attach_provider_response_attempts(retry_exc, response_payloads)
                raise

        def _usage_total(field: str) -> int | None:
            values: list[int] = []
            for response_payload in response_payloads:
                usage = response_payload.get("usage")
                value = usage.get(field) if isinstance(usage, dict) else None
                if isinstance(value, int):
                    values.append(value)
            return sum(values) if values else None

        cached_values: list[int] = []
        for response_payload in response_payloads:
            usage = response_payload.get("usage")
            input_details = (
                usage.get("input_tokens_details")
                if isinstance(usage, dict)
                else None
            )
            value = (
                input_details.get("cached_tokens")
                if isinstance(input_details, dict)
                else None
            )
            if isinstance(value, int):
                cached_values.append(value)
        input_tokens = _usage_total("input_tokens")
        output_tokens = _usage_total("output_tokens")
        cached_input_tokens = sum(cached_values) if cached_values else None
        raw_payload = {
            **(payload if isinstance(payload, dict) else {}),
            "_request_contract": {
                "initial_max_output_tokens": int(request_payload["max_output_tokens"]),
                "final_max_output_tokens": current_tokens,
                "response_output_token_limits": response_output_token_limits,
                "physical_response_count": len(response_payloads),
                "allow_output_token_retry": bool(allow_output_token_retry),
                "max_retries_per_request": int(max_retries_per_request),
                "strict_cost_cap": strict_cost_cap,
            },
        }
        if len(response_payloads) > 1:
            raw_payload = {
                **raw_payload,
                "_response_attempts": response_payloads,
            }

        return LLMResult(
            json_text=text,
            model=resolved_model,
            usage_input_tokens=int(input_tokens) if isinstance(input_tokens, int) else None,
            usage_output_tokens=int(output_tokens) if isinstance(output_tokens, int) else None,
            raw=raw_payload,
            usage_cached_input_tokens=(
                int(cached_input_tokens)
                if isinstance(cached_input_tokens, int)
                else None
            ),
        )
