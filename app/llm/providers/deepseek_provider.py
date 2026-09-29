from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from app.util.http import require_network

from app.config import AppConfig, get_config
from app.llm.providers.disabled_provider import LLMResult
from app.llm.providers.retry_guard import (
    DEFAULT_LLM_CALL_TIMEOUT_SECONDS,
    call_with_llm_retry_guard,
)
from app.logging import get_logger


logger = get_logger(__name__)


_DEEPSEEK_V4_PRO_MODEL = "deepseek-v4-pro"
_PROVIDER_RESPONSE_ATTEMPTS_ATTR = "_provider_response_attempts"


class DeepSeekOutputTruncatedError(RuntimeError):
    """A billed response reached its output limit before completing its schema."""


class _RawResponseCapture:
    """Pre-armed, fail-closed capture for one physical DeepSeek response."""

    def __init__(
        self,
        raw_path: str | Path,
        *,
        schema_name: str,
        model: str,
        requested_max_output_tokens: int,
    ) -> None:
        self.raw_path = Path(raw_path).expanduser().resolve()
        self.metadata_path = self.raw_path.with_name(f"{self.raw_path.name}.metadata.json")
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        self._raw_file = None
        self._metadata_file = None
        raw_created = False
        metadata_created = False
        try:
            self._raw_file = self.raw_path.open("xb")
            raw_created = True
            os.chmod(self.raw_path, 0o600)
            self._metadata_file = self.metadata_path.open("x+b")
            metadata_created = True
            os.chmod(self.metadata_path, 0o600)
            self._write_metadata(
                {
                    "capture_status": "ARMED_BEFORE_TRANSPORT",
                    "schema_name": schema_name,
                    "model": model,
                    "requested_max_output_tokens": requested_max_output_tokens,
                    "raw_response_path": str(self.raw_path),
                    "response_body_sha256": None,
                    "response_body_bytes": 0,
                    "http_status_code": None,
                    "finish_reason": None,
                    "content": None,
                    "reasoning_content": None,
                    "usage": None,
                    "capture_ordering": ("raw_response_bytes_fsynced_before_any_json_parse"),
                }
            )
            self._fsync_parent()
        except Exception:
            self.close()
            for created, path in (
                (metadata_created, self.metadata_path),
                (raw_created, self.raw_path),
            ):
                if not created:
                    continue
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            raise

    def _fsync_parent(self) -> None:
        descriptor = os.open(self.raw_path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _write_metadata(self, payload: dict[str, Any]) -> None:
        if self._metadata_file is None:
            raise RuntimeError("DeepSeek response capture metadata is not writable")
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        self._metadata_file.seek(0)
        self._metadata_file.truncate()
        self._metadata_file.write(encoded)
        self._metadata_file.flush()
        os.fsync(self._metadata_file.fileno())

    def persist_raw(self, response_bytes: bytes, *, status_code: int) -> str:
        if self._raw_file is None:
            raise RuntimeError("DeepSeek raw response capture is not writable")
        exact_bytes = bytes(response_bytes)
        self._raw_file.write(exact_bytes)
        self._raw_file.flush()
        os.fsync(self._raw_file.fileno())
        digest = hashlib.sha256(exact_bytes).hexdigest()
        self._write_metadata(
            {
                **self.metadata(),
                "capture_status": "RAW_PERSISTED_BEFORE_PARSE",
                "response_body_sha256": digest,
                "response_body_bytes": len(exact_bytes),
                "http_status_code": int(status_code),
            }
        )
        return digest

    def metadata(self) -> dict[str, Any]:
        if self._metadata_file is None:
            raise RuntimeError("DeepSeek response capture metadata is not readable")
        self._metadata_file.flush()
        self._metadata_file.seek(0)
        raw = self._metadata_file.read()
        return json.loads(raw.decode("utf-8"))

    def record_envelope(
        self,
        body: Any,
        *,
        response_body_sha256: str,
        status_code: int,
    ) -> None:
        choice = None
        message = None
        if isinstance(body, dict):
            choices = body.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                choice = choices[0]
                if isinstance(choice.get("message"), dict):
                    message = choice["message"]
        self._write_metadata(
            {
                **self.metadata(),
                "capture_status": "ENVELOPE_RECORDED_AFTER_RAW_PERSIST",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "response_body_sha256": response_body_sha256,
                "http_status_code": int(status_code),
                "finish_reason": choice.get("finish_reason") if choice else None,
                "content": message.get("content") if message else None,
                "reasoning_content": (message.get("reasoning_content") if message else None),
                "usage": body.get("usage") if isinstance(body, dict) else None,
            }
        )

    def record_parse_error(
        self,
        error: BaseException,
        *,
        response_body_sha256: str,
        status_code: int,
    ) -> None:
        self._write_metadata(
            {
                **self.metadata(),
                "capture_status": "RESPONSE_JSON_PARSE_FAILED",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "response_body_sha256": response_body_sha256,
                "http_status_code": int(status_code),
                "parse_error": f"{type(error).__name__}: {error}",
            }
        )

    def close(self) -> None:
        for handle_name in ("_raw_file", "_metadata_file"):
            handle = getattr(self, handle_name, None)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)


def _attach_provider_response_attempts(
    error: BaseException,
    response_payloads: list[dict[str, Any]],
) -> None:
    current = [payload for payload in response_payloads if isinstance(payload, dict)]
    if not current:
        return
    try:
        setattr(error, _PROVIDER_RESPONSE_ATTEMPTS_ATTR, current)
    except Exception:  # pragma: no cover - ordinary provider exceptions are mutable
        return


class DeepSeekProvider:
    """Strict DeepSeek V4 Pro JSON provider for paid classic scans.

    DeepSeek enables thinking by default for V4 Pro.  This adapter always sends
    an explicit non-thinking request and rejects model or feature drift before
    any network call.
    """

    provider_name = "deepseek"
    _handles_retry_guard = True

    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or get_config()
        self.base_url = "https://api.deepseek.com/chat/completions"

    def enabled(self) -> bool:
        return bool(
            self.cfg.deepseek_api_key
            and str(self.cfg.llm_provider or "").strip().lower() == self.provider_name
        )

    @staticmethod
    def _normalize_json_text(text: str) -> str:
        stripped = str(text or "").strip()
        candidates = [stripped] if stripped else []
        candidates.extend(
            match.strip()
            for match in re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
            if match.strip()
        )
        first_obj, last_obj = stripped.find("{"), stripped.rfind("}")
        if first_obj != -1 and last_obj > first_obj:
            candidates.append(stripped[first_obj : last_obj + 1])
        for candidate in dict.fromkeys(candidates):
            try:
                parsed = json.loads(candidate)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(parsed, dict):
                raise RuntimeError("DeepSeek output must be a JSON object")
            return json.dumps(parsed)
        raise RuntimeError("DeepSeek output is not valid JSON")

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
        allow_output_token_retry: bool = False,
        max_retries_per_request: int = 0,
        raw_response_capture_path: str | Path | None = None,
        diagnostic_phase_callback: Callable[[str], None] | None = None,
        before_transport_callback: Callable[[], None] | None = None,
    ) -> LLMResult:
        def _record_phase(phase: str) -> None:
            if diagnostic_phase_callback is not None:
                diagnostic_phase_callback(phase)

        _record_phase("provider_validation_started")
        if not self.enabled():
            raise RuntimeError("DeepSeek provider not enabled")
        resolved_model = str(model or self.cfg.deepseek_model).strip()
        if resolved_model != _DEEPSEEK_V4_PRO_MODEL:
            raise RuntimeError(
                "DeepSeek model drift: "
                f"required={_DEEPSEEK_V4_PRO_MODEL}, configured={resolved_model or 'missing'}"
            )
        if any(value is not None for value in (tools, include, max_tool_calls, service_tier)):
            raise ValueError(
                "DeepSeek V4 Pro classic-scan provider does not support tools or tiers"
            )
        if allow_output_token_retry:
            raise ValueError("DeepSeek V4 Pro output retries are disabled for strict cost control")
        if isinstance(max_retries_per_request, bool) or int(max_retries_per_request) < 0:
            raise ValueError("max_retries_per_request must be a non-negative integer")
        if raw_response_capture_path is not None and int(max_retries_per_request) != 0:
            raise ValueError("DeepSeek raw response capture requires exactly one transport attempt")
        output_limit = int(max_output_tokens or self.cfg.deepseek_max_output_tokens)
        if output_limit < 1:
            raise ValueError("DeepSeek max output tokens must be positive")
        _record_phase("provider_validation_complete")

        schema_label = str(schema_name or "structured_output").strip() or "structured_output"
        system_prompt = (
            "Return only one valid JSON object. The response must conform exactly to "
            f"the {schema_label} JSON Schema below; do not add prose or markdown.\n"
            f"JSON Schema:\n{json.dumps(schema, sort_keys=True, separators=(',', ':'))}"
        )
        request_payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": output_limit,
            "thinking": {"type": "disabled"},
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.deepseek_api_key}",
            "Content-Type": "application/json",
        }
        request_timeout = min(
            float(self.cfg.deepseek_request_timeout),
            DEFAULT_LLM_CALL_TIMEOUT_SECONDS,
        )
        _record_phase("request_payload_built")
        response_capture = None
        if raw_response_capture_path is not None:
            _record_phase("capture_arming_started")
            response_capture = _RawResponseCapture(
                raw_response_capture_path,
                schema_name=schema_label,
                model=resolved_model,
                requested_max_output_tokens=output_limit,
            )
            _record_phase("capture_armed")

        def _request_json() -> dict[str, Any]:
            if before_transport_callback is not None:
                before_transport_callback()
            # VOE_NET_PROVIDER is the single switch for all outbound traffic.
            require_network(self.base_url, self.cfg)
            response = requests.post(
                self.base_url,
                headers=headers,
                json=request_payload,
                timeout=request_timeout,
            )
            _record_phase("response_received")
            captured_body: Any = None
            if response_capture is not None:
                response_bytes = bytes(response.content)
                response_digest = response_capture.persist_raw(
                    response_bytes,
                    status_code=response.status_code,
                )
                try:
                    captured_body = json.loads(response_bytes)
                except Exception as exc:
                    response_capture.record_parse_error(
                        exc,
                        response_body_sha256=response_digest,
                        status_code=response.status_code,
                    )
                    raise RuntimeError("DeepSeek API response body is not valid JSON") from exc
                response_capture.record_envelope(
                    captured_body,
                    response_body_sha256=response_digest,
                    status_code=response.status_code,
                )
            if 200 <= response.status_code < 300:
                body = captured_body if response_capture is not None else response.json()
                if not isinstance(body, dict):
                    raise RuntimeError("DeepSeek API returned a non-object response")
                return body
            if response_capture is not None:
                error_body = captured_body
            else:
                try:
                    error_body = response.json()
                except Exception:  # noqa: BLE001
                    error_body = {"error_text": str(response.text or "")[:2000]}
            error_text = json.dumps(error_body, default=str)
            if self.cfg.deepseek_api_key:
                error_text = error_text.replace(self.cfg.deepseek_api_key, "***REDACTED***")
            error_text = re.sub(r"sk-[A-Za-z0-9_-]{6,}", "sk-***REDACTED***", error_text)
            logger.error(
                "deepseek_chat_request_failed",
                extra={
                    "stage_name": "llm_provider",
                    "stage_status_code": response.status_code,
                    "stage_error": error_text[:4000],
                },
            )
            raise RuntimeError(
                f"DeepSeek Chat API request failed (status={response.status_code}): {error_text}"
            )

        response_payloads: list[dict[str, Any]] = []
        try:
            payload = call_with_llm_retry_guard(
                provider_name=self.provider_name,
                schema_name=schema_label,
                call=_request_json,
                timeout_seconds=request_timeout,
                max_retries=int(max_retries_per_request),
            )
            response_payloads.append(payload)
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("DeepSeek response did not include a choice")
            choice = choices[0] if isinstance(choices[0], dict) else {}
            finish_reason = str(choice.get("finish_reason") or "").strip().lower()
            if finish_reason == "length":
                raise DeepSeekOutputTruncatedError(
                    "LLM_PROVIDER_TRUNCATED: DeepSeek response finish_reason=length "
                    f"at requested max_output_tokens={output_limit}"
                )
            message = choice.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("DeepSeek response did not include output text")
            normalized_text = self._normalize_json_text(content)
        except Exception as exc:
            _attach_provider_response_attempts(exc, response_payloads)
            raise
        finally:
            if response_capture is not None:
                response_capture.close()

        usage = payload.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        cached_tokens = usage.get("prompt_cache_hit_tokens")
        return LLMResult(
            json_text=normalized_text,
            model=resolved_model,
            usage_input_tokens=(
                int(input_tokens)
                if isinstance(input_tokens, int) and not isinstance(input_tokens, bool)
                else None
            ),
            usage_output_tokens=(
                int(output_tokens)
                if isinstance(output_tokens, int) and not isinstance(output_tokens, bool)
                else None
            ),
            usage_cached_input_tokens=(
                int(cached_tokens)
                if isinstance(cached_tokens, int) and not isinstance(cached_tokens, bool)
                else None
            ),
            raw={**payload, "_response_attempts": response_payloads},
        )
