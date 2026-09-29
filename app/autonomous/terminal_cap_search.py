"""Explicit, bounded OpenAI web-search fallback for direct issuer market cap.

The low-level search function remains independently testable.  Production
scan code receives an :class:`AuthorizedTerminalCapSearch` callback, which can
only be constructed from a successful whole-run cost-preflight artifact.  The
callback keeps one atomic attempt/evidence ledger and exposes exact usage for
benchmark reconciliation; merely selecting pipeline v2 never enables spend.
"""

from __future__ import annotations

import fcntl
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.autonomous.cap_resolver import (
    TERMINAL_CAP_MAX_AGE_DAYS,
    SecurityIdentity,
    TerminalCapEvidence,
    derive_terminal_source_kind,
)
from app.llm.synthesis_agent import _estimate_cost_usd


TERMINAL_CAP_SEARCH_MODEL = "gpt-5.5"
TERMINAL_CAP_SEARCH_TOOL_COST_USD = 0.01
MAX_TERMINAL_CAP_SEARCH_TOOL_CALLS = 4
MAX_WHOLE_RUN_COST_USD = 100.0
TERMINAL_CAP_SEARCH_MAX_INPUT_TOKENS = 10_000
TERMINAL_CAP_SEARCH_MAX_OUTPUT_TOKENS = 1_200
# Responses does not expose a hard max-input-tokens control for web search.
# Bound the paid session against GPT-5.5's entire 1.05M context window minus
# our hard output allowance. Responses web-search content is billed at model
# input rates and has a smaller documented search-context limit, so this
# intentionally over-reserves both usage and cost.
TERMINAL_CAP_SEARCH_CONTEXT_WINDOW_TOKENS = 1_050_000
TERMINAL_CAP_SEARCH_MAX_BILLABLE_INPUT_TOKENS = (
    TERMINAL_CAP_SEARCH_CONTEXT_WINDOW_TOKENS - TERMINAL_CAP_SEARCH_MAX_OUTPUT_TOKENS
)
WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE = "autonomous_sector_whole_run_cost_preflight_v1"
V2_EXECUTION_PREFLIGHT_ARTIFACT_TYPE = "all_sector_execution_authorization_v1"
TERMINAL_CAP_SEARCH_LEDGER_ARTIFACT_TYPE = "terminal_cap_search_attempt_ledger_v1"
WHOLE_RUN_REQUIRED_COST_LANES = (
    "provider_preflight",
    "parent_research",
    "company_underwriting",
    "selected_company_validation",
    "repair_fallback",
)
_LEDGER_PROCESS_LOCKS: dict[str, Lock] = {}
_LEDGER_PROCESS_LOCKS_GUARD = Lock()

_RESULT_FIELDS = frozenset(
    {
        "status",
        "ticker",
        "issuer_name",
        "issuer_cik",
        "market_cap_basis",
        "market_cap_mm",
        "market_cap_currency",
        "market_cap_units",
        "as_of_date",
        "source_name",
        "source_url",
        "confidence",
        "detail",
    }
)

TERMINAL_CAP_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["RESOLVED", "NOT_FOUND"]},
        "ticker": {"type": "string"},
        "issuer_name": {"type": ["string", "null"]},
        "issuer_cik": {"type": ["string", "null"]},
        "market_cap_basis": {
            "type": "string",
            "enum": ["DIRECT_ISSUER_MARKET_CAP"],
        },
        "market_cap_mm": {"type": ["number", "null"]},
        "market_cap_currency": {"type": "string", "enum": ["USD"]},
        "market_cap_units": {
            "type": "string",
            "enum": ["USD_MILLIONS"],
        },
        "as_of_date": {"type": ["string", "null"]},
        "source_name": {"type": ["string", "null"]},
        "source_url": {"type": ["string", "null"]},
        "confidence": {
            "type": ["string", "null"],
            "enum": ["HIGH", "MEDIUM", "LOW", None],
        },
        "detail": {"type": ["string", "null"]},
    },
    "required": sorted(_RESULT_FIELDS),
    "additionalProperties": False,
}


@dataclass(frozen=True)
class TerminalCapSearchResult:
    status: Literal["RESOLVED", "NOT_FOUND", "REJECTED", "FAILED"]
    reason_code: str
    evidence: TerminalCapEvidence | None
    cited_source_urls: tuple[str, ...]
    web_search_call_count: int
    usage_records: tuple[dict[str, Any], ...]
    total_cost_usd: float
    response_id: str | None = None


@dataclass(frozen=True)
class TerminalCapSearchAuthorization:
    """Spend authority derived from a successful whole-run cost preflight."""

    run_id: str
    authorized_at: str
    max_cost_usd: float
    whole_run_worst_case_cost_usd: float
    terminal_cap_search_reserved_cost_usd: float
    max_attempts: int
    max_tool_calls_per_attempt: int
    lane_worst_case_costs_usd: dict[str, float]
    request_fingerprint: str
    ledger_fingerprint: str
    preflight_fingerprint: str
    allowed_tickers: tuple[str, ...] = ()
    execution_fingerprint: str | None = None
    model: str = TERMINAL_CAP_SEARCH_MODEL
    preflight_artifact_type: str = WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_tickers"] = list(self.allowed_tickers)
        return payload


def estimate_terminal_cap_search_worst_case_cost_usd(
    *,
    max_attempts: int,
    max_tool_calls_per_attempt: int,
) -> float:
    """Return the hard-bound reserve a whole-run preflight must include."""

    attempts = int(max_attempts)
    tool_calls = int(max_tool_calls_per_attempt)
    if attempts < 0:
        raise ValueError("max_attempts must be non-negative")
    if not 1 <= tool_calls <= MAX_TERMINAL_CAP_SEARCH_TOOL_CALLS:
        raise ValueError(
            f"max_tool_calls_per_attempt must be between 1 and {MAX_TERMINAL_CAP_SEARCH_TOOL_CALLS}"
        )
    model_cost = _estimate_cost_usd(
        TERMINAL_CAP_SEARCH_MODEL,
        TERMINAL_CAP_SEARCH_MAX_BILLABLE_INPUT_TOKENS,
        TERMINAL_CAP_SEARCH_MAX_OUTPUT_TOKENS,
        provider_name="openai",
        cached_input_tokens=0,
    )
    per_attempt = model_cost + (tool_calls * TERMINAL_CAP_SEARCH_TOOL_COST_USD)
    return round(attempts * per_attempt, 6)


def whole_run_preflight_request_fingerprint(
    *,
    sectors: list[str] | tuple[str, ...],
    objective: str,
    as_of_date: str,
    market_cap_focus: str,
    pipeline_version: str,
    budget: dict[str, Any],
    max_candidates: int | None,
    candidate_bindings: dict[str, Any] | None = None,
    provider_name: str = "openai",
    model: str = TERMINAL_CAP_SEARCH_MODEL,
    terminal_cap_search_max_attempts: int | None = None,
) -> str:
    """Bind paid authority to one exact benchmark request."""

    normalized_sectors = list(
        dict.fromkeys(str(item).strip() for item in sectors if str(item).strip())
    )
    payload = {
        "sectors": normalized_sectors,
        "objective": str(objective),
        "as_of_date": str(as_of_date),
        "market_cap_focus": str(market_cap_focus),
        "pipeline_version": str(pipeline_version).lower(),
        "budget": budget,
        "max_candidates": max_candidates,
        "candidate_bindings": candidate_bindings or {},
        "provider": str(provider_name).strip().lower(),
        "model": str(model).strip().lower(),
        "terminal_cap_search_max_attempts": terminal_cap_search_max_attempts,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def terminal_cap_search_ledger_fingerprint(path: str | Path) -> str:
    """Bind one spend authorization to one canonical durable ledger path."""

    canonical = str(Path(path).expanduser().resolve(strict=False))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    token = str(value or "").strip().lower()
    return len(token) == 64 and all(char in "0123456789abcdef" for char in token)


def _reject_diagnostic_spend_authority(preflight: dict[str, Any]) -> None:
    if bool(preflight.get("diagnostic_only")) or preflight.get("execution_requested") is False:
        raise ValueError("diagnostic-only cost preflight cannot authorize provider or search spend")


def bind_v2_cost_preflight_for_terminal_cap_search(
    preflight: dict[str, Any],
    *,
    run_id: str,
    authorized_at: str,
    request_fingerprint: str,
    ledger_path: str | Path,
    max_tool_calls_per_attempt: int = MAX_TERMINAL_CAP_SEARCH_TOOL_CALLS,
) -> dict[str, Any]:
    """Bind the canonical six-lane preflight to one terminal-search ledger.

    This adds identity/ledger metadata without changing the independently
    reconcilable cost estimate.  The returned artifact remains the v2 preflight
    authority; ``authorization_from_whole_run_preflight`` adapts it internally
    for legacy callers.
    """

    if str(preflight.get("artifact_type") or "") != V2_EXECUTION_PREFLIGHT_ARTIFACT_TYPE:
        raise ValueError("v2 terminal authorization requires the all-sector execution preflight")
    _reject_diagnostic_spend_authority(preflight)
    if str(preflight.get("status") or "").upper() != "AUTHORIZED" or not bool(
        preflight.get("spend_authorized")
    ):
        raise ValueError("v2 all-sector cost preflight did not authorize spend")
    bound = json.loads(json.dumps(preflight, sort_keys=True))
    bound["terminal_cap_search_binding"] = {
        "run_id": str(run_id).strip(),
        "authorized_at": str(authorized_at).strip(),
        "request_fingerprint": str(request_fingerprint).strip().lower(),
        "ledger_fingerprint": terminal_cap_search_ledger_fingerprint(ledger_path),
        "max_tool_calls_per_attempt": int(max_tool_calls_per_attempt),
    }
    # Validate immediately so a malformed binding is never persisted as an
    # apparent spend authority.
    _legacy_preflight_from_v2(bound)
    return bound


def _legacy_preflight_from_v2(preflight: dict[str, Any]) -> dict[str, Any]:
    from app.autonomous.all_sector_cost_preflight import AllSectorCostPreflight

    if str(preflight.get("artifact_type") or "") != V2_EXECUTION_PREFLIGHT_ARTIFACT_TYPE:
        raise ValueError("not an all-sector v2 execution preflight")
    _reject_diagnostic_spend_authority(preflight)
    if str(preflight.get("status") or "").upper() != "AUTHORIZED" or not bool(
        preflight.get("spend_authorized")
    ):
        raise ValueError("v2 all-sector cost preflight did not authorize spend")
    estimate_payload = preflight.get("estimate")
    if not isinstance(estimate_payload, dict):
        raise ValueError("v2 all-sector preflight is missing its exact cost estimate")
    estimate = AllSectorCostPreflight.from_dict(estimate_payload)
    if estimate.status != "AUTHORIZED":
        raise ValueError("v2 all-sector cost estimate did not authorize spend")
    binding = preflight.get("terminal_cap_search_binding")
    if not isinstance(binding, dict):
        raise ValueError("v2 all-sector preflight is not bound to a terminal-cap ledger")
    run_id = str(binding.get("run_id") or "").strip()
    authorized_at = str(binding.get("authorized_at") or "").strip()
    request_fingerprint = str(binding.get("request_fingerprint") or "").strip().lower()
    ledger_fingerprint = str(binding.get("ledger_fingerprint") or "").strip().lower()
    if not run_id or not authorized_at:
        raise ValueError("v2 terminal-cap binding requires run_id and authorized_at")
    try:
        datetime.fromisoformat(authorized_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("v2 terminal-cap binding authorized_at must be ISO-8601") from exc
    if not _valid_sha256(request_fingerprint) or not _valid_sha256(ledger_fingerprint):
        raise ValueError("v2 terminal-cap binding fingerprints must be SHA-256 values")
    max_attempts = int(estimate.request.terminal_cap_search_attempts)
    max_tool_calls = int(binding.get("max_tool_calls_per_attempt") or 0)
    if not 1 <= max_tool_calls <= MAX_TERMINAL_CAP_SEARCH_TOOL_CALLS:
        raise ValueError("v2 terminal-cap binding tool-call limit is invalid")
    lanes = estimate_payload.get("lane_costs")
    if not isinstance(lanes, dict) or set(lanes) != {
        "provider_preflight",
        "parent_research",
        "company_underwriting",
        "selected_company_validation",
        "repair_fallback",
        "terminal_cap_search",
    }:
        raise ValueError("v2 all-sector preflight must contain exactly six cost lanes")
    lane_costs = {lane: float((lanes.get(lane) or {}).get("cost_usd") or 0.0) for lane in lanes}
    terminal_reserve = lane_costs["terminal_cap_search"]
    # The legacy authorization shape carried terminal reserve inside the repair
    # bucket.  Preserve that representation only at this compatibility boundary;
    # the canonical artifact and benchmark accounting retain six separate lanes.
    legacy_lanes = {
        "provider_preflight": lane_costs["provider_preflight"]
        + float(estimate.request.prior_realized_cost_usd),
        "parent_research": lane_costs["parent_research"],
        "company_underwriting": lane_costs["company_underwriting"],
        "selected_company_validation": lane_costs["selected_company_validation"],
        "repair_fallback": lane_costs["repair_fallback"] + terminal_reserve,
    }
    aggregate_cost = float(estimate_payload["aggregate"]["cost_usd"])
    if abs(sum(legacy_lanes.values()) - aggregate_cost) > 1e-6:
        raise ValueError("v2 terminal compatibility lanes do not reconcile")
    return {
        "artifact_type": WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE,
        "status": "AUTHORIZED",
        "run_id": run_id,
        "authorized_at": authorized_at,
        "model": TERMINAL_CAP_SEARCH_MODEL,
        "request_fingerprint": request_fingerprint,
        "max_cost_usd": float(estimate.request.authorization_ceiling_usd),
        "worst_case_cost_usd": aggregate_cost,
        "terminal_cap_search_reserved_cost_usd": terminal_reserve,
        "lane_worst_case_costs_usd": legacy_lanes,
        "terminal_cap_search_max_attempts": max_attempts,
        "terminal_cap_search_max_tool_calls_per_attempt": max_tool_calls,
        "terminal_cap_search_ledger_fingerprint": ledger_fingerprint,
        "allowed_tickers": sorted(
            {
                str(ticker).strip().upper()
                for tickers in (preflight.get("frozen_candidate_tickers") or {}).values()
                for ticker in (tickers if isinstance(tickers, list) else [])
                if str(ticker).strip()
            }
        ),
        "execution_fingerprint": str(preflight.get("execution_set_fingerprint") or "")
        .strip()
        .lower()
        or None,
        "v2_preflight_fingerprint": sha256(
            json.dumps(preflight, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def authorization_from_whole_run_preflight(
    preflight: dict[str, Any],
) -> TerminalCapSearchAuthorization:
    """Validate and narrow one whole-run preflight into terminal-cap authority.

    A generic boolean or provider health check is deliberately insufficient.
    The caller must present the versioned whole-run artifact containing the
    aggregate worst case and an explicit terminal-cap reserve.
    """

    if not isinstance(preflight, dict):
        raise ValueError("whole-run cost preflight artifact is required")
    _reject_diagnostic_spend_authority(preflight)
    if str(preflight.get("artifact_type") or "") == V2_EXECUTION_PREFLIGHT_ARTIFACT_TYPE:
        preflight = _legacy_preflight_from_v2(preflight)
    if str(preflight.get("artifact_type") or "") != WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE:
        raise ValueError("terminal cap search requires a whole-run cost preflight artifact")
    if str(preflight.get("status") or "").strip().upper() != "AUTHORIZED":
        raise ValueError("whole-run cost preflight did not authorize spend")
    run_id = str(preflight.get("run_id") or "").strip()
    authorized_at = str(preflight.get("authorized_at") or "").strip()
    if not run_id or not authorized_at:
        raise ValueError("whole-run cost preflight must include run_id and authorized_at")
    try:
        datetime.fromisoformat(authorized_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("whole-run cost preflight authorized_at must be ISO-8601") from exc
    model = str(preflight.get("model") or "").strip().lower()
    if model != TERMINAL_CAP_SEARCH_MODEL:
        raise ValueError(f"terminal cap search is pinned to {TERMINAL_CAP_SEARCH_MODEL}")
    try:
        max_cost = float(preflight["max_cost_usd"])
        whole_run_worst_case = float(preflight["worst_case_cost_usd"])
        reserved = float(preflight["terminal_cap_search_reserved_cost_usd"])
        max_attempts = int(preflight["terminal_cap_search_max_attempts"])
        max_tool_calls = int(preflight["terminal_cap_search_max_tool_calls_per_attempt"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("whole-run cost preflight is missing terminal-cap budget fields") from exc
    if not all(math.isfinite(value) for value in (max_cost, whole_run_worst_case, reserved)):
        raise ValueError("whole-run cost preflight values must be finite")
    if max_cost <= 0 or max_cost > MAX_WHOLE_RUN_COST_USD:
        raise ValueError(f"whole-run cost ceiling must be within ${MAX_WHOLE_RUN_COST_USD:.2f}")
    if whole_run_worst_case < 0 or whole_run_worst_case > max_cost:
        raise ValueError("whole-run worst-case cost exceeds the authorized ceiling")
    required_reserve = estimate_terminal_cap_search_worst_case_cost_usd(
        max_attempts=max_attempts,
        max_tool_calls_per_attempt=max_tool_calls,
    )
    if reserved < required_reserve or reserved > whole_run_worst_case:
        raise ValueError("terminal-cap reserve is absent from the authorized whole-run worst case")
    raw_lanes = preflight.get("lane_worst_case_costs_usd")
    if not isinstance(raw_lanes, dict) or set(raw_lanes) != set(WHOLE_RUN_REQUIRED_COST_LANES):
        raise ValueError("whole-run cost preflight must include every canonical cost lane")
    try:
        lane_costs = {str(name): float(value) for name, value in raw_lanes.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("whole-run cost lane values must be numeric") from exc
    if any(not math.isfinite(value) or value < 0 for value in lane_costs.values()):
        raise ValueError("whole-run cost lane values must be finite and non-negative")
    if abs(sum(lane_costs.values()) - whole_run_worst_case) > 1e-6:
        raise ValueError("whole-run cost lane sum does not equal aggregate worst case")
    if lane_costs["repair_fallback"] + 1e-9 < reserved:
        raise ValueError("repair/fallback lane does not contain terminal-cap reserve")
    request_fingerprint = str(preflight.get("request_fingerprint") or "").strip().lower()
    if not _valid_sha256(request_fingerprint):
        raise ValueError("whole-run cost preflight request_fingerprint is required")
    ledger_fingerprint = (
        str(preflight.get("terminal_cap_search_ledger_fingerprint") or "").strip().lower()
    )
    if not _valid_sha256(ledger_fingerprint):
        raise ValueError("whole-run cost preflight must bind one terminal-cap ledger")
    allowed_tickers = tuple(
        dict.fromkeys(
            str(ticker).strip().upper()
            for ticker in preflight.get("allowed_tickers") or []
            if str(ticker).strip()
        )
    )
    execution_fingerprint = (
        str(preflight.get("execution_fingerprint") or "").strip().lower() or None
    )
    if allowed_tickers and not _valid_sha256(execution_fingerprint):
        raise ValueError("terminal-cap authority must bind the authorized execution set")
    return TerminalCapSearchAuthorization(
        run_id=run_id,
        authorized_at=authorized_at,
        max_cost_usd=max_cost,
        whole_run_worst_case_cost_usd=whole_run_worst_case,
        terminal_cap_search_reserved_cost_usd=reserved,
        max_attempts=max_attempts,
        max_tool_calls_per_attempt=max_tool_calls,
        lane_worst_case_costs_usd=lane_costs,
        request_fingerprint=request_fingerprint,
        ledger_fingerprint=ledger_fingerprint,
        preflight_fingerprint=sha256(
            json.dumps(preflight, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        allowed_tickers=allowed_tickers,
        execution_fingerprint=execution_fingerprint,
        model=model,
    )


def _terminal_evidence_ledger_row(evidence: TerminalCapEvidence) -> dict[str, Any]:
    return {
        "ticker": evidence.ticker,
        "issuer_cik": evidence.issuer_cik,
        "issuer_name": evidence.issuer_name,
        "security_name": evidence.security_name,
        "market_cap_basis": "DIRECT_ISSUER_MARKET_CAP",
        "market_cap_mm": evidence.market_cap_mm,
        "market_cap_currency": "USD",
        "market_cap_units": "USD_MILLIONS",
        "source_kind": evidence.source_kind,
        "source_name": evidence.source_name,
        "source_url": evidence.source_url,
        "as_of_date": evidence.as_of_date,
        "confidence": evidence.confidence,
        "detail": evidence.detail,
    }


def _empty_search_result(reason_code: str) -> TerminalCapSearchResult:
    return TerminalCapSearchResult(
        status="REJECTED",
        reason_code=reason_code,
        evidence=None,
        cited_source_urls=(),
        web_search_call_count=0,
        usage_records=(),
        total_cost_usd=0.0,
    )


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


@contextmanager
def _exclusive_ledger_lock(ledger_path: Path):
    """Serialize ledger refresh, reservation, provider call, and settlement."""

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_name(f"{ledger_path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _ledger_process_lock(ledger_path: Path) -> Lock:
    canonical = str(ledger_path.expanduser().resolve(strict=False))
    with _LEDGER_PROCESS_LOCKS_GUARD:
        return _LEDGER_PROCESS_LOCKS.setdefault(canonical, Lock())


def _reserved_attempt_usage_records(
    *,
    authorization: TerminalCapSearchAuthorization,
    attempt_number: int,
    ticker: str,
) -> list[dict[str, Any]]:
    model_cost = _estimate_cost_usd(
        authorization.model,
        TERMINAL_CAP_SEARCH_MAX_BILLABLE_INPUT_TOKENS,
        TERMINAL_CAP_SEARCH_MAX_OUTPUT_TOKENS,
        provider_name="openai",
        cached_input_tokens=0,
    )
    common = {
        "lane": "terminal_cap_search",
        "provider": "openai",
        "authorization_run_id": authorization.run_id,
        "attempt_number": attempt_number,
        "ticker": str(ticker).upper(),
        "attempt_status": "PENDING",
        "reason_code": "WORST_CASE_RESERVE_HELD",
        "billing_status": "WORST_CASE_RESERVED",
    }
    records: list[dict[str, Any]] = [
        {
            **common,
            "call_type": "responses_model",
            "provider_call_id": (
                f"terminal-response:{authorization.run_id}:{attempt_number}:{str(ticker).upper()}"
            ),
            "model": authorization.model,
            "response_model": None,
            "service_tier": "default",
            "input_tokens": TERMINAL_CAP_SEARCH_MAX_BILLABLE_INPUT_TOKENS,
            "cached_input_tokens": 0,
            "output_tokens": TERMINAL_CAP_SEARCH_MAX_OUTPUT_TOKENS,
            "reserved_output_tokens": 0,
            "estimated_tokens": True,
            "cost_estimate_usd": model_cost,
        }
    ]
    for index in range(authorization.max_tool_calls_per_attempt):
        records.append(
            {
                **common,
                "call_type": "web_search_call_reserve",
                "call_id": f"reserved_web_search_{index + 1}",
                "unit_cost_usd": TERMINAL_CAP_SEARCH_TOOL_COST_USD,
                "cost_estimate_usd": TERMINAL_CAP_SEARCH_TOOL_COST_USD,
            }
        )
    return records


def _has_authoritative_usage(result: TerminalCapSearchResult) -> bool:
    return any(
        record.get("call_type") == "responses_model"
        and record.get("estimated_tokens") is False
        and isinstance(record.get("input_tokens"), int)
        and isinstance(record.get("output_tokens"), int)
        for record in result.usage_records
        if isinstance(record, dict)
    )


class AuthorizedTerminalCapSearch:
    """Callable last-resort search lane with durable evidence and usage."""

    _voe_terminal_cap_search_authorized = True

    def __init__(
        self,
        *,
        authorization: TerminalCapSearchAuthorization,
        provider: Any,
        ledger_path: str | Path,
        preflight_artifact_path: str | Path | None = None,
    ) -> None:
        self.authorization = authorization
        self.provider = provider
        self.ledger_path = Path(ledger_path)
        if authorization.ledger_fingerprint != terminal_cap_search_ledger_fingerprint(
            self.ledger_path
        ):
            raise ValueError("whole-run cost preflight is bound to a different ledger")
        self.preflight_artifact_path = (
            Path(preflight_artifact_path) if preflight_artifact_path is not None else None
        )
        self._lock = Lock()
        self._attempts: list[dict[str, Any]] = []
        self._evidence: list[dict[str, Any]] = []
        with _ledger_process_lock(self.ledger_path), _exclusive_ledger_lock(self.ledger_path):
            self._load_existing_ledger()
            self._write_ledger()

    def _load_existing_ledger(self) -> None:
        if not self.ledger_path.exists():
            return
        try:
            payload = json.loads(
                self.ledger_path.read_text(encoding="utf-8"),
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unreadable terminal cap search ledger: {self.ledger_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid terminal cap search ledger: {self.ledger_path}")
        existing_auth = payload.get("authorization")
        if existing_auth != self.authorization.to_dict():
            raise RuntimeError("Terminal cap search ledger authorization does not match this run")
        self._attempts = [
            dict(row) for row in payload.get("attempts") or [] if isinstance(row, dict)
        ]
        self._evidence = [
            dict(row) for row in payload.get("evidence") or [] if isinstance(row, dict)
        ]

    def _write_ledger(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        usage_records = [
            dict(record)
            for attempt in self._attempts
            for record in attempt.get("usage_records") or []
            if isinstance(record, dict)
        ]
        payload = {
            "artifact_type": TERMINAL_CAP_SEARCH_LEDGER_ARTIFACT_TYPE,
            "authorization": self.authorization.to_dict(),
            "preflight_artifact_path": (
                str(self.preflight_artifact_path)
                if self.preflight_artifact_path is not None
                else None
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "attempt_count": len(self._attempts),
            "resolved_count": len(self._evidence),
            "usage": {
                "response_calls": sum(
                    1 for row in usage_records if row.get("call_type") == "responses_model"
                ),
                "web_search_calls": sum(
                    1 for row in usage_records if row.get("call_type") == "web_search_call"
                ),
                "reserved_web_search_calls": sum(
                    1 for row in usage_records if row.get("call_type") == "web_search_call_reserve"
                ),
                "worst_case_reserved_attempts": sum(
                    1
                    for row in usage_records
                    if row.get("call_type") == "responses_model"
                    and row.get("billing_status") == "WORST_CASE_RESERVED"
                ),
                "input_tokens": sum(int(row.get("input_tokens") or 0) for row in usage_records),
                "cached_input_tokens": sum(
                    int(row.get("cached_input_tokens") or 0) for row in usage_records
                ),
                "output_tokens": sum(int(row.get("output_tokens") or 0) for row in usage_records),
                "cost_estimate_usd": round(
                    sum(float(row.get("cost_estimate_usd") or 0.0) for row in usage_records),
                    6,
                ),
            },
            "attempts": self._attempts,
            # ``terminal_cap_evidence.load_terminal_cap_evidence`` consumes
            # this exact top-level collection on resume/offline replay.
            "evidence": self._evidence,
        }
        temporary = self.ledger_path.with_suffix(f"{self.ledger_path.suffix}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.ledger_path)

    @property
    def usage_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(record)
            for attempt in self._attempts
            for record in attempt.get("usage_records") or []
            if isinstance(record, dict)
        )

    @property
    def attempt_count(self) -> int:
        return len(self._attempts)

    @property
    def spent_cost_usd(self) -> float:
        return round(
            sum(float(record.get("cost_estimate_usd") or 0.0) for record in self.usage_records),
            6,
        )

    def _cached_evidence(
        self,
        ticker: str,
        as_of_date: str,
        identity: SecurityIdentity,
    ) -> TerminalCapEvidence | None:
        if not self._evidence:
            return None
        from app.autonomous.terminal_cap_evidence import terminal_cap_lookup_from_path

        target = _iso_date(as_of_date)
        if target is None:
            return None
        for evidence in terminal_cap_lookup_from_path(self.ledger_path)(
            ticker,
            target,
            identity,
        ):
            used = _iso_date(evidence.as_of_date)
            if used is None:
                continue
            age_days = (date.fromisoformat(target) - date.fromisoformat(used)).days
            if 0 <= age_days <= TERMINAL_CAP_MAX_AGE_DAYS:
                return evidence
        return None

    def __call__(
        self,
        ticker: str,
        as_of_date: str,
        identity: SecurityIdentity,
    ) -> TerminalCapSearchResult:
        normalized_ticker = str(ticker).strip().upper()
        if (
            self.authorization.allowed_tickers
            and normalized_ticker not in self.authorization.allowed_tickers
        ):
            return _empty_search_result("TICKER_OUTSIDE_AUTHORIZED_EXECUTION_SET")
        with (
            self._lock,
            _ledger_process_lock(self.ledger_path),
            _exclusive_ledger_lock(self.ledger_path),
        ):
            # Another process may have consumed authority since this object
            # was constructed. Refresh under the cross-process lock before
            # checking cache, attempt count, or remaining reserve.
            self._load_existing_ledger()
            cached = self._cached_evidence(ticker, as_of_date, identity)
            if cached is not None:
                return TerminalCapSearchResult(
                    status="RESOLVED",
                    reason_code="DIRECT_ISSUER_CAP_RESOLVED_FROM_LEDGER",
                    evidence=cached,
                    cited_source_urls=(cached.source_url,),
                    web_search_call_count=0,
                    usage_records=(),
                    total_cost_usd=0.0,
                )
            if self.attempt_count >= self.authorization.max_attempts:
                return _empty_search_result("AUTHORIZED_ATTEMPT_LIMIT_EXHAUSTED")
            one_attempt_reserve = estimate_terminal_cap_search_worst_case_cost_usd(
                max_attempts=1,
                max_tool_calls_per_attempt=(self.authorization.max_tool_calls_per_attempt),
            )
            if (
                self.spent_cost_usd + one_attempt_reserve
                > self.authorization.terminal_cap_search_reserved_cost_usd + 1e-9
            ):
                return _empty_search_result("AUTHORIZED_COST_RESERVE_EXHAUSTED")

            # Debit the complete attempt before crossing the API boundary. A
            # process crash, KeyboardInterrupt, missing usage payload, or
            # provider-side failure therefore consumes authority instead of
            # making the same budget silently reusable.
            attempt_number = self.attempt_count + 1
            reserved_usage = _reserved_attempt_usage_records(
                authorization=self.authorization,
                attempt_number=attempt_number,
                ticker=ticker,
            )
            attempt = {
                "attempt_number": attempt_number,
                "attempted_at": datetime.now(timezone.utc).isoformat(),
                "ticker": str(ticker).upper(),
                "as_of_date": str(as_of_date),
                "issuer_cik": _normalize_cik(identity.issuer_cik),
                "status": "PENDING",
                "reason_code": "WORST_CASE_RESERVE_HELD",
                "response_id": None,
                "cited_source_urls": [],
                "web_search_call_count": 0,
                "usage_records": reserved_usage,
                "total_cost_usd": one_attempt_reserve,
                "evidence": None,
            }
            self._attempts.append(attempt)
            self._write_ledger()

            result = search_terminal_cap_with_openai(
                ticker=ticker,
                as_of_date=as_of_date,
                identity=identity,
                provider=self.provider,
                model=self.authorization.model,
                max_tool_calls=self.authorization.max_tool_calls_per_attempt,
                authorization=self.authorization,
            )
            authoritative_usage = _has_authoritative_usage(result)
            if authoritative_usage:
                billed_usage = [
                    {
                        **dict(record),
                        **(
                            {
                                "provider_call_id": (
                                    f"terminal-response:{self.authorization.run_id}:"
                                    f"{attempt_number}:{str(ticker).upper()}:{index}"
                                )
                            }
                            if record.get("call_type") == "responses_model"
                            else {}
                        ),
                        "authorization_run_id": self.authorization.run_id,
                        "attempt_number": attempt_number,
                        "ticker": str(ticker).upper(),
                        "attempt_status": result.status,
                        "reason_code": result.reason_code,
                        "billing_status": "AUTHORITATIVE_USAGE",
                        "service_tier": "default",
                    }
                    for index, record in enumerate(result.usage_records, start=1)
                ]
                billed_cost = result.total_cost_usd
            else:
                billed_usage = reserved_usage
                billed_cost = one_attempt_reserve

            attempt.update(
                {
                    "settled_at": datetime.now(timezone.utc).isoformat(),
                    "status": result.status,
                    "reason_code": result.reason_code,
                    "response_id": result.response_id,
                    "cited_source_urls": list(result.cited_source_urls),
                    "web_search_call_count": result.web_search_call_count,
                    "usage_authoritative": authoritative_usage,
                    "reported_usage_records": [dict(record) for record in result.usage_records],
                    "usage_records": billed_usage,
                    "total_cost_usd": billed_cost,
                    "evidence": (
                        _terminal_evidence_ledger_row(result.evidence)
                        if result.evidence is not None
                        else None
                    ),
                }
            )
            if result.evidence is not None:
                evidence_row = _terminal_evidence_ledger_row(result.evidence)
                key = (
                    evidence_row["ticker"],
                    evidence_row["issuer_cik"],
                    evidence_row["as_of_date"],
                    evidence_row["source_url"],
                )
                existing_keys = {
                    (
                        row.get("ticker"),
                        row.get("issuer_cik"),
                        row.get("as_of_date"),
                        row.get("source_url"),
                    )
                    for row in self._evidence
                }
                if key not in existing_keys:
                    self._evidence.append(evidence_row)
            self._write_ledger()
            return replace(
                result,
                usage_records=tuple(dict(record) for record in billed_usage),
                total_cost_usd=billed_cost,
            )


def build_authorized_terminal_cap_search(
    *,
    preflight: dict[str, Any],
    provider: Any,
    ledger_path: str | Path,
    preflight_artifact_path: str | Path | None = None,
    expected_request_fingerprint: str | None = None,
) -> AuthorizedTerminalCapSearch:
    """Build the production callback only after validating whole-run authority."""

    authorization = authorization_from_whole_run_preflight(preflight)
    if (
        expected_request_fingerprint is not None
        and authorization.request_fingerprint != str(expected_request_fingerprint).lower()
    ):
        raise ValueError("whole-run cost preflight does not match this benchmark request")
    return AuthorizedTerminalCapSearch(
        authorization=authorization,
        provider=provider,
        ledger_path=ledger_path,
        preflight_artifact_path=preflight_artifact_path,
    )


def rebuild_authorized_terminal_cap_search(
    *,
    source: AuthorizedTerminalCapSearch,
    preflight: dict[str, Any],
    preflight_artifact_path: str | Path | None = None,
    expected_request_fingerprint: str | None = None,
) -> AuthorizedTerminalCapSearch:
    """Replace an unspent supplied callback's authority with the frozen preflight.

    A callback supplied to the benchmark may contribute its provider and ledger
    path, but its earlier authorization can never survive candidate discovery.
    Rebinding is allowed only while its ledger is empty, so prior spend cannot be
    erased or moved under a new whole-run authority.
    """

    if not isinstance(source, AuthorizedTerminalCapSearch):
        raise ValueError("supplied terminal cap search must be an AuthorizedTerminalCapSearch")
    authorization = authorization_from_whole_run_preflight(preflight)
    if (
        expected_request_fingerprint is not None
        and authorization.request_fingerprint != str(expected_request_fingerprint).strip().lower()
    ):
        raise ValueError("whole-run cost preflight does not match this benchmark request")
    if authorization.ledger_fingerprint != terminal_cap_search_ledger_fingerprint(
        source.ledger_path
    ):
        raise ValueError("frozen whole-run preflight is bound to a different ledger")

    with (
        source._lock,  # noqa: SLF001 - atomic authority replacement on this class
        _ledger_process_lock(source.ledger_path),
        _exclusive_ledger_lock(source.ledger_path),
    ):
        source._load_existing_ledger()  # noqa: SLF001 - verify original authority
        if (
            source._attempts  # noqa: SLF001 - spend history must remain immutable
            or source._evidence  # noqa: SLF001 - cached results belong to old authority
            or source.usage_records
        ):
            raise ValueError(
                "supplied terminal cap search already contains attempts or spend and cannot be rebound"
            )
        source.authorization = authorization
        source.preflight_artifact_path = (
            Path(preflight_artifact_path) if preflight_artifact_path is not None else None
        )
        source._write_ledger()  # noqa: SLF001 - persist the sole frozen authority
    return source


def _normalize_cik(value: Any) -> str | None:
    digits = "".join(char for char in str(value or "") if char.isdigit())
    return digits.zfill(10) if digits else None


def _iso_date(value: Any) -> str | None:
    token = str(value or "").strip()[:10]
    try:
        return date.fromisoformat(token).isoformat()
    except ValueError:
        return None


def _canonical_url(value: Any) -> str | None:
    token = str(value or "").strip()
    try:
        parsed = urlsplit(token)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


def _response_search_metadata(
    raw: dict[str, Any],
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    call_ids: list[str] = []
    urls: list[str] = []
    attempts = raw.get("_response_attempts")
    response_payloads = (
        [payload for payload in attempts if isinstance(payload, dict)]
        if isinstance(attempts, list)
        else [raw]
    )
    for response_payload in response_payloads:
        output = response_payload.get("output")
        if not isinstance(output, list):
            continue
        for item in output:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "") == "web_search_call":
                call_ids.append(str(item.get("id") or f"web_search_call_{len(call_ids) + 1}"))
                action = item.get("action")
                sources = action.get("sources") if isinstance(action, dict) else None
                if isinstance(sources, list):
                    for source in sources:
                        if isinstance(source, dict) and source.get("url"):
                            urls.append(str(source["url"]))
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                annotations = part.get("annotations")
                if not isinstance(annotations, list):
                    continue
                for annotation in annotations:
                    if (
                        isinstance(annotation, dict)
                        and str(annotation.get("type") or "") == "url_citation"
                        and annotation.get("url")
                    ):
                        urls.append(str(annotation["url"]))
    return (
        len(call_ids),
        tuple(dict.fromkeys(urls)),
        tuple(call_ids),
    )


def _usage_records(
    *,
    result: Any,
    prompt: str,
    web_search_call_ids: tuple[str, ...],
    requested_model: str,
) -> tuple[tuple[dict[str, Any], ...], float]:
    model = str(requested_model or TERMINAL_CAP_SEARCH_MODEL)
    response_model = str(getattr(result, "model", None) or model)
    json_text = str(getattr(result, "json_text", "") or "")
    raw_input_tokens = getattr(result, "usage_input_tokens", None)
    raw_output_tokens = getattr(result, "usage_output_tokens", None)
    raw_cached_tokens = getattr(result, "usage_cached_input_tokens", None)
    estimated = not isinstance(raw_input_tokens, int) or not isinstance(raw_output_tokens, int)
    input_tokens = (
        int(raw_input_tokens) if isinstance(raw_input_tokens, int) else max(1, len(prompt) // 4)
    )
    output_tokens = (
        int(raw_output_tokens)
        if isinstance(raw_output_tokens, int)
        else max(1, len(json_text) // 4)
    )
    cached_input_tokens = (
        min(max(0, int(raw_cached_tokens)), input_tokens)
        if isinstance(raw_cached_tokens, int)
        else 0
    )
    model_cost = _estimate_cost_usd(
        model,
        input_tokens,
        output_tokens,
        provider_name="openai",
        cached_input_tokens=cached_input_tokens,
    )
    records: list[dict[str, Any]] = [
        {
            "lane": "terminal_cap_search",
            "call_type": "responses_model",
            "provider_call_id": "terminal-response:direct:1",
            "provider": "openai",
            "model": model,
            "response_model": response_model,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "reserved_output_tokens": 0,
            "estimated_tokens": estimated,
            "cost_estimate_usd": model_cost,
        }
    ]
    for call_id in web_search_call_ids:
        records.append(
            {
                "lane": "terminal_cap_search",
                "call_type": "web_search_call",
                "provider": "openai",
                "call_id": call_id,
                "unit_cost_usd": TERMINAL_CAP_SEARCH_TOOL_COST_USD,
                "cost_estimate_usd": TERMINAL_CAP_SEARCH_TOOL_COST_USD,
            }
        )
    return tuple(records), round(sum(float(record["cost_estimate_usd"]) for record in records), 6)


def _prompt(
    *,
    ticker: str,
    as_of_date: str,
    identity: SecurityIdentity,
) -> str:
    identity_payload = {
        "requested_ticker": ticker,
        "issuer_cik": identity.issuer_cik,
        "issuer_primary_ticker": identity.issuer_primary_ticker,
        "issuer_listed_tickers": list(identity.issuer_listed_tickers),
        "security_role": identity.security_role,
        "is_adr": identity.is_adr,
        "is_secondary_class": identity.is_secondary_class,
    }
    return (
        "Find a source-backed direct total issuer market capitalization for the "
        f"following resolved security identity at or before {as_of_date}.\n"
        f"Identity: {json.dumps(identity_payload, sort_keys=True)}\n"
        "Use web search. Return RESOLVED only when a cited page explicitly reports "
        "the issuer-wide market capitalization and its effective date. You may "
        "convert a directly reported USD billion figure to USD millions. Never "
        "calculate market cap from shares, ADR ratios, class shares, or a quote. "
        "Never substitute enterprise value. The source_url must be a URL actually "
        "used by web search. If direct evidence is unavailable, return NOT_FOUND "
        "with null evidence fields."
    )


def _result(
    *,
    status: Literal["RESOLVED", "NOT_FOUND", "REJECTED", "FAILED"],
    reason_code: str,
    evidence: TerminalCapEvidence | None,
    cited_urls: tuple[str, ...],
    call_count: int,
    usage_records: tuple[dict[str, Any], ...],
    total_cost_usd: float,
    raw: dict[str, Any],
) -> TerminalCapSearchResult:
    return TerminalCapSearchResult(
        status=status,
        reason_code=reason_code,
        evidence=evidence,
        cited_source_urls=cited_urls,
        web_search_call_count=call_count,
        usage_records=usage_records,
        total_cost_usd=total_cost_usd,
        response_id=str(raw.get("id") or "").strip() or None,
    )


def search_terminal_cap_with_openai(
    *,
    ticker: str,
    as_of_date: str,
    identity: SecurityIdentity,
    provider: Any,
    model: str = TERMINAL_CAP_SEARCH_MODEL,
    max_tool_calls: int = 2,
    authorization: TerminalCapSearchAuthorization | None = None,
) -> TerminalCapSearchResult:
    """Run one explicitly authorized, bounded web-search cap lookup."""

    upper = str(ticker or "").strip().upper()
    target_as_of = _iso_date(as_of_date)
    if not upper or target_as_of is None:
        raise ValueError("ticker and ISO as_of_date are required")
    if not 1 <= int(max_tool_calls) <= MAX_TERMINAL_CAP_SEARCH_TOOL_CALLS:
        raise ValueError(
            f"max_tool_calls must be between 1 and {MAX_TERMINAL_CAP_SEARCH_TOOL_CALLS}"
        )
    if str(model).strip().lower() != TERMINAL_CAP_SEARCH_MODEL:
        raise ValueError(f"terminal cap search is pinned to {TERMINAL_CAP_SEARCH_MODEL}")
    if authorization is None:
        return _empty_search_result("DETERMINISTIC_PREFLIGHT_AUTHORIZATION_REQUIRED")
    if (
        str(authorization.model).strip().lower() != str(model).strip().lower()
        or int(authorization.max_tool_calls_per_attempt) != int(max_tool_calls)
        or (authorization.allowed_tickers and upper not in authorization.allowed_tickers)
    ):
        return _empty_search_result("DETERMINISTIC_PREFLIGHT_AUTHORIZATION_MISMATCH")
    if str(getattr(provider, "provider_name", "") or "").strip().lower() != "openai":
        return TerminalCapSearchResult(
            status="REJECTED",
            reason_code="OPENAI_PROVIDER_REQUIRED",
            evidence=None,
            cited_source_urls=(),
            web_search_call_count=0,
            usage_records=(),
            total_cost_usd=0.0,
        )
    identity_cik = _normalize_cik(identity.issuer_cik)
    if identity_cik is None:
        return TerminalCapSearchResult(
            status="REJECTED",
            reason_code="IDENTITY_CIK_REQUIRED",
            evidence=None,
            cited_source_urls=(),
            web_search_call_count=0,
            usage_records=(),
            total_cost_usd=0.0,
        )

    prompt = _prompt(ticker=upper, as_of_date=target_as_of, identity=identity)
    if max(1, len(prompt) // 4) > TERMINAL_CAP_SEARCH_MAX_INPUT_TOKENS:
        return _empty_search_result("PROMPT_TOKEN_BOUND_EXCEEDED")
    try:
        llm_result = provider.synthesize_json(
            prompt=prompt,
            schema=TERMINAL_CAP_SEARCH_SCHEMA,
            schema_name="terminal_cap_search_v1",
            max_output_tokens=1200,
            tools=[{"type": "web_search", "search_context_size": "low"}],
            include=["web_search_call.action.sources"],
            max_tool_calls=int(max_tool_calls),
            model=str(model),
            service_tier="default",
            allow_output_token_retry=False,
            max_retries_per_request=0,
        )
    except InvalidFinancialInputError:
        raise
    except Exception as exc:  # noqa: BLE001 - caller receives a truthful failed lane
        estimated_input_tokens = max(1, len(prompt) // 4)
        attempted_cost = _estimate_cost_usd(
            str(model),
            estimated_input_tokens,
            0,
            provider_name="openai",
            cached_input_tokens=0,
        )
        attempted_usage = (
            {
                "lane": "terminal_cap_search",
                "call_type": "responses_model",
                "provider_call_id": "terminal-response:direct:1",
                "provider": "openai",
                "model": str(model),
                "response_model": None,
                "input_tokens": estimated_input_tokens,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reserved_output_tokens": 0,
                "estimated_tokens": True,
                "billing_status": "UNKNOWN_PROVIDER_FAILURE",
                "cost_estimate_usd": attempted_cost,
            },
        )
        return TerminalCapSearchResult(
            status="FAILED",
            reason_code=f"PROVIDER_ERROR:{type(exc).__name__}",
            evidence=None,
            cited_source_urls=(),
            web_search_call_count=0,
            usage_records=attempted_usage,
            total_cost_usd=attempted_cost,
        )

    raw = getattr(llm_result, "raw", None)
    raw = raw if isinstance(raw, dict) else {}
    call_count, cited_urls, call_ids = _response_search_metadata(raw)
    usage_records, total_cost = _usage_records(
        result=llm_result,
        prompt=prompt,
        web_search_call_ids=call_ids,
        requested_model=str(model),
    )
    common = {
        "cited_urls": cited_urls,
        "call_count": call_count,
        "usage_records": usage_records,
        "total_cost_usd": total_cost,
        "raw": raw,
    }
    if call_count == 0:
        return _result(
            status="REJECTED",
            reason_code="WEB_SEARCH_NOT_USED",
            evidence=None,
            **common,
        )
    if call_count > int(max_tool_calls):
        return _result(
            status="REJECTED",
            reason_code="TOOL_CALL_LIMIT_EXCEEDED",
            evidence=None,
            **common,
        )

    try:
        payload = json.loads(str(getattr(llm_result, "json_text", "") or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None
    if not isinstance(payload, dict) or set(payload) != _RESULT_FIELDS:
        return _result(
            status="REJECTED",
            reason_code="INVALID_STRUCTURED_RESULT",
            evidence=None,
            **common,
        )
    if payload.get("status") == "NOT_FOUND":
        return _result(
            status="NOT_FOUND",
            reason_code="DIRECT_ISSUER_CAP_NOT_FOUND",
            evidence=None,
            **common,
        )
    if payload.get("status") != "RESOLVED":
        return _result(
            status="REJECTED",
            reason_code="INVALID_STATUS",
            evidence=None,
            **common,
        )
    if str(payload.get("ticker") or "").strip().upper() != upper:
        reason = "TICKER_MISMATCH"
    elif _normalize_cik(payload.get("issuer_cik")) != identity_cik:
        reason = "ISSUER_CIK_MISMATCH"
    elif payload.get("market_cap_basis") != "DIRECT_ISSUER_MARKET_CAP":
        reason = "NON_DIRECT_MARKET_CAP"
    elif (
        payload.get("market_cap_currency") != "USD"
        or payload.get("market_cap_units") != "USD_MILLIONS"
    ):
        reason = "UNSUPPORTED_MARKET_CAP_UNITS"
    elif (
        isinstance(payload.get("market_cap_mm"), bool)
        or not isinstance(payload.get("market_cap_mm"), (int, float))
        or float(payload["market_cap_mm"]) <= 0
    ):
        reason = "INVALID_MARKET_CAP"
    elif str(payload.get("confidence") or "") not in {"HIGH", "MEDIUM"}:
        reason = "INSUFFICIENT_CONFIDENCE"
    elif not str(payload.get("source_name") or "").strip():
        reason = "SOURCE_NAME_REQUIRED"
    else:
        used_as_of = _iso_date(payload.get("as_of_date"))
        target = date.fromisoformat(target_as_of)
        if used_as_of is None:
            reason = "INVALID_EVIDENCE_DATE"
        else:
            age_days = (target - date.fromisoformat(used_as_of)).days
            reason = (
                "EVIDENCE_DATE_OUT_OF_RANGE"
                if age_days < 0 or age_days > TERMINAL_CAP_MAX_AGE_DAYS
                else ""
            )
    source_url_text = str(payload.get("source_url") or "").strip()
    source_url = _canonical_url(source_url_text)
    if not reason and (source_url is None or source_url_text not in set(cited_urls)):
        reason = "SOURCE_URL_NOT_IN_RESPONSE_SOURCES"
    if reason:
        return _result(
            status="REJECTED",
            reason_code=reason,
            evidence=None,
            **common,
        )

    used_as_of = _iso_date(payload["as_of_date"])
    assert source_url is not None and used_as_of is not None
    source_kind = derive_terminal_source_kind(
        source_url=source_url_text,
        claimed_kind="SEARCH",
    )
    evidence = TerminalCapEvidence(
        ticker=upper,
        market_cap_mm=float(payload["market_cap_mm"]),
        source_kind=source_kind,  # type: ignore[arg-type]
        source_name=str(payload["source_name"]).strip(),
        source_url=source_url_text,
        as_of_date=used_as_of,
        confidence=str(payload["confidence"]),  # type: ignore[arg-type]
        issuer_cik=identity_cik,
        issuer_name=str(payload.get("issuer_name") or "").strip() or None,
        security_name=upper,
        detail=str(payload.get("detail") or "").strip() or None,
    )
    return _result(
        status="RESOLVED",
        reason_code="DIRECT_ISSUER_CAP_RESOLVED",
        evidence=evidence,
        **common,
    )


__all__ = [
    "AuthorizedTerminalCapSearch",
    "MAX_TERMINAL_CAP_SEARCH_TOOL_CALLS",
    "MAX_WHOLE_RUN_COST_USD",
    "TERMINAL_CAP_SEARCH_MODEL",
    "TERMINAL_CAP_SEARCH_SCHEMA",
    "TERMINAL_CAP_SEARCH_TOOL_COST_USD",
    "WHOLE_RUN_REQUIRED_COST_LANES",
    "TerminalCapSearchAuthorization",
    "TerminalCapSearchResult",
    "authorization_from_whole_run_preflight",
    "bind_v2_cost_preflight_for_terminal_cap_search",
    "build_authorized_terminal_cap_search",
    "rebuild_authorized_terminal_cap_search",
    "estimate_terminal_cap_search_worst_case_cost_usd",
    "search_terminal_cap_with_openai",
    "terminal_cap_search_ledger_fingerprint",
    "whole_run_preflight_request_fingerprint",
]
