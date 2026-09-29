"""Cross-sector benchmark harness for autonomous sector runs."""

from __future__ import annotations

from hashlib import sha256
import json
import secrets
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from app.autonomous.all_sector_cost_preflight import (
    DIAGNOSTIC_REPRICE_MODELS,
    PRODUCTION_MAX_OUTPUT_TOKENS,
    PRODUCTION_MAX_RETRIES_PER_REQUEST,
    PRODUCTION_MAX_SERIALIZED_REQUEST_BYTES,
    PRODUCTION_MODEL,
    PRODUCTION_PROVIDER,
    build_diagnostic_v2_model_reprice,
    build_production_v2_cost_preflight,
)
from app.ingest.cache_readiness import (
    cache_prewarm_limit as _benchmark_cache_prewarm_limit,
    cache_readiness_for_tickers as _cache_readiness_for_tickers,
    counter_dict as _counter_dict,
    refreshed_tickers_by_sector as _refreshed_tickers_by_sector,
)
from app.autonomous.output_store import (
    persist_autonomous_sector_attempt_diagnostic,
    persist_autonomous_sector_diagnostic,
    persist_autonomous_sector_run,
)
from app.autonomous.financial_integrity import (
    InvalidFinancialInputError,
    require_financial_integrity_scope,
    require_unchanged_financial_integrity_scope,
)
from app.autonomous.run_contract import AutonomousRunBudget
from app.autonomous.sector_candidates import (
    AcceptedCensusRunAuthority,
    freeze_v2_execution_bound,
    resolve_sector_candidate_tickers,
)
from app.autonomous.sector_contract import AutonomousSectorFinancialRunArtifact
from app.autonomous.sector_lane_budget import CANONICAL_LANES
from app.autonomous.sector_runtime import (
    DEFAULT_SECTOR_OBJECTIVE,
    run_sector_autonomous_financial_analysis,
    sector_artifact_summary,
)
from app.autonomous.v1_financial_context import build_canonical_v1_financial_context
from app.config import (
    canonical_market_cap_focus,
    ensure_directories,
    get_config,
    resolve_autonomous_sector_pipeline_version,
)
from app.llm.providers import get_anthropic_provider, get_llm_provider
from app.llm.providers.retry_guard import (
    llm_attempt_observer,
    llm_physical_attempt_guard,
)
from app.llm.execution_policy import LLMExecutionPolicy, llm_execution_policy
from app.llm.usage_capture import (
    failed_provider_usage_meta,
    provider_usage_records,
)
from app.research.source_quality import classify_source_quality
from app.util.dates import utc_now


DEFAULT_BENCHMARK_SECTORS = [
    "enterprise_software",
    "diversified_industrials",
    "semiconductors",
    "medical_devices",
]

EXECUTION_RAN = "RAN"
EXECUTION_REUSED = "REUSED_FROM_RESUME"
EXECUTION_RERAN = "RERAN_FROM_RESUME"
EXECUTION_SKIPPED_PROVIDER = "SKIPPED_PROVIDER_UNAVAILABLE"
EXECUTION_SKIPPED_CACHE = "SKIPPED_CACHE_NOT_READY"
EXECUTION_FAILED = "FAILED"
EXECUTION_STOPPED_COST_PREFLIGHT = "STOPPED_BEFORE_SPEND"
EXECUTION_MODE_FULL = "FULL_EXECUTION"
EXECUTION_MODE_COST_PREFLIGHT_ONLY = "COST_PREFLIGHT_ONLY"
EXECUTION_MODE_READINESS_PREFLIGHT_ONLY = "READINESS_PREFLIGHT_ONLY"
EXECUTION_MODE_FREE_DATA_REPAIR_ONLY = "FREE_DATA_REPAIR_ONLY"


def _file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _bind_sector_artifact_digests(results: list[dict[str, Any]]) -> None:
    """Bind every persisted sector-artifact reference to its exact bytes."""

    for row in results:
        raw_path = row.get("artifact_path")
        if not raw_path:
            continue
        row["artifact_sha256"] = _file_sha256(Path(str(raw_path)))


def _fixed_cohort_from_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist independently checkable v2 cohort identity and artifact bindings."""

    security_slots: set[tuple[str, str]] = set()
    issuer_keys: set[str] = set()
    unresolved_issuer_identity: list[str] = []
    artifact_digests: list[dict[str, Any]] = []
    for row in results:
        sector = str(row.get("sector") or "").strip().lower()
        raw_path = str(row.get("artifact_path") or "").strip()
        digest = str(row.get("artifact_sha256") or "").strip().lower()
        if raw_path:
            artifact_digests.append(
                {
                    "sector": sector,
                    "run_id": row.get("run_id"),
                    "artifact_path": raw_path,
                    "artifact_sha256": digest or None,
                }
            )
        if not raw_path:
            continue
        try:
            artifact = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        for disposition in artifact.get("candidate_dispositions") or []:
            if not isinstance(disposition, dict):
                continue
            ticker = str(disposition.get("ticker") or "").strip().upper()
            if not sector or not ticker:
                continue
            security_slots.add((sector, ticker))
            cik = "".join(
                char for char in str(disposition.get("issuer_cik") or "") if char.isdigit()
            ).lstrip("0")
            issuer_key = (
                str(disposition.get("issuer_key") or disposition.get("issuer_id") or "")
                .strip()
                .upper()
            )
            if cik:
                issuer_keys.add(f"CIK:{cik}")
            elif issuer_key:
                issuer_keys.add(f"ISSUER:{issuer_key}")
            else:
                unresolved_issuer_identity.append(f"{sector}:{ticker}")
    return {
        "schema_version": "all_sector_v2_fixed_cohort_v1",
        "security_count": len(security_slots),
        "issuer_count": len(issuer_keys),
        "unresolved_issuer_identity": sorted(set(unresolved_issuer_identity)),
        "sector_artifacts": artifact_digests,
        "trusted_manifest_policy": "CALLER_OR_CONFIG_PINNED_NOT_BENCHMARK_SELF_DECLARED",
    }


_FRESHNESS_ISSUE_BUCKETS = {"stale_over_90d", "future_dated", "undated"}
_LEGACY_LANE_ALIASES = {"selected_validation": "selected_company_validation"}
_LANE_USAGE_INTEGER_FIELDS = (
    "tool_call_attempts",
    "tool_calls_ok",
    "tool_calls_failed",
    "provider_call_attempts",
    "provider_calls_ok",
    "provider_calls_failed",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reserved_output_tokens",
    "cost_microdollars",
)
_MICRODOLLARS_PER_DOLLAR = 1_000_000
V2_EXECUTION_PREFLIGHT_ARTIFACT_TYPE = "all_sector_execution_authorization_v1"


@dataclass(frozen=True)
class AutonomousSectorBenchmarkPaths:
    summary_json: Path
    report_md: Path
    cost_preflight_json: Path | None = None
    readiness_preflight_json: Path | None = None
    free_data_repair_json: Path | None = None


def _normalize_sectors(sectors: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if sectors is None:
        raw_items = DEFAULT_BENCHMARK_SECTORS
    elif isinstance(sectors, str):
        raw_items = sectors.split(",")
    else:
        raw_items = list(sectors)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        sector = str(item or "").strip()
        if not sector or sector in seen:
            continue
        normalized.append(sector)
        seen.add(sector)
    return normalized


def _benchmark_run_id(created_at: str) -> str:
    date_part = str(created_at)[:10].replace("-", "")
    return f"autonomous_sector_benchmark_{date_part}_{secrets.token_hex(3)}"


def _classify_provider_failure(exc: BaseException | str | None) -> str:
    text = str(exc or "").lower()
    exc_type = type(exc).__name__.lower() if isinstance(exc, BaseException) else ""
    if "llm_provider_truncated" in text or "outputtruncated" in exc_type:
        return "LLM_PROVIDER_TRUNCATED"
    if "insufficient_quota" in text or "quota" in text or "status=429" in text or " 429" in text:
        return "LLM_PROVIDER_QUOTA_EXHAUSTED"
    if "timeout" in text or "timed out" in text or "timeout" in exc_type:
        return "LLM_PROVIDER_TIMEOUT"
    if "not valid json" in text or "invalid json" in text:
        return "LLM_PROVIDER_INVALID_JSON"
    if "did not include output text" in text or "empty output" in text:
        return "LLM_PROVIDER_EMPTY_OUTPUT"
    if "not enabled" in text or "disabled" in text or "missing" in text:
        return "LLM_PROVIDER_UNAVAILABLE"
    return "LLM_PROVIDER_ERROR"


def _run_provider_preflight_for_provider(
    provider: Any, *, fallback_from: str | None = None
) -> dict[str, Any]:
    provider_name = str(getattr(provider, "provider_name", "unknown") or "unknown")
    prompt = 'Return JSON exactly matching this schema: {"ok": true}.'
    schema_name = "benchmark_provider_preflight_v1"
    if not provider.enabled():
        return {
            "enabled": True,
            "status": "FAILED",
            "provider": provider_name,
            "failure_code": "LLM_PROVIDER_UNAVAILABLE",
            "error": f"LLM provider '{provider_name}' is not enabled for benchmark preflight.",
            "provider_calls": [],
        }
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    failed_attempts: list[dict[str, Any]] = []

    def observe_failed_attempt(event: dict[str, Any]) -> None:
        error = event.get("error")
        exc = error if isinstance(error, BaseException) else RuntimeError(str(error or "error"))
        failed_attempts.append(
            {
                **failed_provider_usage_meta(
                    provider=provider,
                    prompt=prompt,
                    schema_name=schema_name,
                    estimated_output_tokens=64,
                    error=exc,
                ),
                "lane": "provider_preflight",
                "physical_attempt": int(event.get("attempt") or len(failed_attempts) + 1),
                "retryable": bool(event.get("retryable")),
                "will_retry": bool(event.get("will_retry")),
            }
        )

    try:
        with llm_attempt_observer(observe_failed_attempt):
            result = provider.synthesize_json(
                prompt=prompt,
                schema=schema,
                schema_name=schema_name,
                max_output_tokens=64,
            )
        payload = json.loads(result.json_text)
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise RuntimeError(
                f"Provider preflight returned unexpected payload: {result.json_text[:200]}"
            )
        successful_attempts = provider_usage_records(
            provider=provider,
            result=result,
            prompt=prompt,
            schema_name=schema_name,
        )
        provider_calls = []
        for index, usage in enumerate([*failed_attempts, *successful_attempts], start=1):
            provider_calls.append(
                {
                    **usage,
                    "provider_call_id": f"provider-preflight-{index}",
                    "lane": "provider_preflight",
                }
            )
        return {
            "enabled": True,
            "status": "OK",
            "provider": provider_name,
            "model": result.model,
            "usage": {
                "input_tokens": result.usage_input_tokens,
                "output_tokens": result.usage_output_tokens,
            },
            "provider_calls": provider_calls,
            **({"fallback_from": fallback_from} if fallback_from else {}),
        }
    except InvalidFinancialInputError:
        raise
    except Exception as exc:  # noqa: BLE001
        if not failed_attempts:
            failed_attempts.append(
                {
                    **failed_provider_usage_meta(
                        provider=provider,
                        prompt=prompt,
                        schema_name=schema_name,
                        estimated_output_tokens=64,
                        error=exc,
                    ),
                    "lane": "provider_preflight",
                }
            )
        provider_calls = [
            {
                **usage,
                "provider_call_id": f"provider-preflight-{index}",
                "lane": "provider_preflight",
            }
            for index, usage in enumerate(failed_attempts, start=1)
        ]
        return {
            "enabled": True,
            "status": "FAILED",
            "provider": provider_name,
            "failure_code": _classify_provider_failure(exc),
            "error": str(exc),
            "provider_calls": provider_calls,
        }


def _run_provider_preflight(*, allow_fallback: bool = True) -> dict[str, Any]:
    """Run a tiny structured provider call so benchmark jobs fail cheaply."""

    provider = get_llm_provider()
    result = _run_provider_preflight_for_provider(provider)
    if (
        not allow_fallback
        or result.get("status") != "FAILED"
        or result.get("failure_code") != "LLM_PROVIDER_QUOTA_EXHAUSTED"
    ):
        return result
    if str(result.get("provider") or "").lower() == "anthropic":
        return result
    fallback = get_anthropic_provider()
    if fallback is None or not fallback.enabled():
        return result
    fallback_result = _run_provider_preflight_for_provider(
        fallback,
        fallback_from=str(result.get("provider") or "primary"),
    )
    combined_calls = [
        {
            **call,
            "provider_call_id": f"provider-preflight-{index}",
        }
        for index, call in enumerate(
            [
                *list(result.get("provider_calls") or []),
                *list(fallback_result.get("provider_calls") or []),
            ],
            start=1,
        )
        if isinstance(call, dict)
    ]
    if fallback_result.get("status") == "OK":
        fallback_result["primary_provider_failure"] = result
        fallback_result["provider_calls"] = combined_calls
        return fallback_result
    fallback_result["primary_provider_failure"] = result
    fallback_result["provider_calls"] = combined_calls
    return fallback_result


def _provider_preflight_skipped(reason: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "status": "SKIPPED",
        "reason": reason,
        "provider_calls": [],
    }


def _run_provider_preflight_in_v2_policy() -> dict[str, Any]:
    with _v2_execution_policy_context():
        return _run_provider_preflight(allow_fallback=False)


def _benchmark_summary_path(run_id: str) -> Path:
    return (
        get_config().runs_dir
        / "autonomous_sector_benchmark"
        / str(run_id)
        / "benchmark_summary.json"
    )


def _financial_cache_summary_path(run_id: str) -> Path:
    return get_config().runs_dir / "financial_cache_refresh" / str(run_id) / "refresh_summary.json"


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_benchmark_resume(run_id: str | None) -> dict[str, Any]:
    run_id_norm = str(run_id or "").strip()
    if not run_id_norm:
        return {}
    return _safe_json(_benchmark_summary_path(run_id_norm))


def _load_cache_refresh_summary(run_id: str | None) -> dict[str, Any]:
    run_id_norm = str(run_id or "").strip()
    if not run_id_norm:
        return {}
    return _safe_json(_financial_cache_summary_path(run_id_norm))


def _v2_execution_policy_context():
    return llm_execution_policy(
        LLMExecutionPolicy(
            provider=PRODUCTION_PROVIDER,
            model=PRODUCTION_MODEL,
            max_serialized_request_bytes=PRODUCTION_MAX_SERIALIZED_REQUEST_BYTES,
            max_output_tokens=PRODUCTION_MAX_OUTPUT_TOKENS,
            max_retries_per_request=PRODUCTION_MAX_RETRIES_PER_REQUEST,
            allow_provider_fallback=False,
            allow_output_token_retry=False,
        )
    )


def _configured_v2_provider_error() -> str | None:
    cfg = get_config()
    provider = str(cfg.llm_provider or "").strip().lower()
    model = str(cfg.openai_model or "").strip().lower()
    if provider != PRODUCTION_PROVIDER:
        return f"V2_PROVIDER_MUST_BE_OPENAI:configured={provider or 'disabled'}"
    if model != PRODUCTION_MODEL:
        return f"V2_MODEL_MUST_BE_GPT_5_5:configured={model or 'missing'}"
    return None


def _resume_realized_cost_microdollars(resume_artifact: dict[str, Any]) -> int:
    """Return every already-realized dollar from the resumed benchmark."""

    # Current artifacts explicitly separate this invocation from all prior
    # lineage. Prefer the reconciled cumulative value so a third-generation
    # resume does not forget its grandparent's spend or add the current ledger
    # twice.
    spend = (
        resume_artifact.get("spend_reconciliation") if isinstance(resume_artifact, dict) else None
    )
    if isinstance(spend, dict):
        current = _safe_non_negative_int(spend.get("current_run_actual_cost_microdollars"))
        prior = _safe_non_negative_int(spend.get("prior_lineage_realized_cost_microdollars"))
        combined = _safe_non_negative_int(spend.get("combined_realized_cost_microdollars"))
        if spend.get("combined_realized_reconciles") is True and combined == current + prior:
            return combined

    rollups = resume_artifact.get("rollups") if isinstance(resume_artifact, dict) else None
    usage = rollups.get("lane_usage_totals") if isinstance(rollups, dict) else None
    aggregate = usage.get("aggregate") if isinstance(usage, dict) else None
    total = 0
    if isinstance(aggregate, dict):
        declared = _safe_non_negative_int(aggregate.get("cost_microdollars"))
        if declared:
            total = declared
        else:
            total = _cost_microdollars(aggregate)
    if not total:
        for row in resume_artifact.get("sector_results") or []:
            if not isinstance(row, dict):
                continue
            lane_usage = row.get("lane_usage")
            row_aggregate = lane_usage.get("aggregate") if isinstance(lane_usage, dict) else None
            if isinstance(row_aggregate, dict):
                total += _safe_non_negative_int(row_aggregate.get("cost_microdollars"))
    raw_lanes = usage.get("lanes") if isinstance(usage, dict) else None
    preflight_lane = raw_lanes.get("provider_preflight") if isinstance(raw_lanes, dict) else None
    provider_preflight_already_included = isinstance(preflight_lane, dict) and (
        _safe_non_negative_int(preflight_lane.get("provider_call_attempts")) > 0
        or _safe_non_negative_int(preflight_lane.get("cost_microdollars")) > 0
    )
    provider_preflight = resume_artifact.get("provider_preflight")
    if not provider_preflight_already_included:
        for call in (
            provider_preflight.get("provider_calls") if isinstance(provider_preflight, dict) else []
        ) or []:
            if isinstance(call, dict):
                total += _cost_microdollars(call)
    return total


def _unknown_cap_candidates(
    candidate_payloads: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    unknown: set[tuple[str, str]] = set()
    for sector, payload in candidate_payloads.items():
        selected_values = payload.get("execution_tickers")
        if not isinstance(selected_values, list):
            selected_values = payload.get("selected_tickers") or []
        selected = {
            str(ticker).strip().upper() for ticker in selected_values if str(ticker).strip()
        }
        classifications = payload.get("cap_classifications")
        if not isinstance(classifications, dict):
            continue
        for ticker in selected:
            row = classifications.get(ticker)
            if not isinstance(row, dict):
                continue
            source = str(row.get("cap_source") or row.get("source") or "").lower()
            cap_value = row.get("market_cap_mm")
            if source == "unknown" or cap_value is None:
                unknown.add((sector, ticker))
    return [{"sector": sector, "ticker": ticker} for sector, ticker in sorted(unknown)]


def _v2_execution_preflight_payload(
    *,
    candidate_payloads: dict[str, dict[str, Any]],
    parent_max_turns: int,
    prior_realized_cost_microdollars: int,
    terminal_cap_search_max_attempts: int | None,
    diagnostic_reprice_model: str | None = None,
    resolution_errors: dict[str, str] | None = None,
    execution_sectors: list[str] | None = None,
    blocking_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Build the authoritative stop/continue record before provider spend."""

    execution_sector_set = set(
        candidate_payloads if execution_sectors is None else execution_sectors
    )
    execution_payloads = {
        sector: payload
        for sector, payload in candidate_payloads.items()
        if sector in execution_sector_set
    }
    membership_tickers = {
        sector: list(payload.get("membership_tickers") or payload.get("selected_tickers") or [])
        for sector, payload in candidate_payloads.items()
    }
    execution_tickers = {
        sector: list(payload.get("execution_tickers") or payload.get("selected_tickers") or [])
        for sector, payload in candidate_payloads.items()
    }
    deferred_tickers = {
        sector: list(payload.get("deferred_by_bound_tickers") or [])
        for sector, payload in candidate_payloads.items()
    }
    excluded_tickers = {
        sector: list(payload.get("excluded_tickers") or [])
        for sector, payload in candidate_payloads.items()
    }
    frozen_tickers = {
        sector: execution_tickers[sector]
        for sector in candidate_payloads
        if sector in execution_sector_set
    }
    frozen_counts = {sector: len(tickers) for sector, tickers in frozen_tickers.items()}
    unknown_candidates = _unknown_cap_candidates(execution_payloads)
    unknown_count = len(unknown_candidates)
    requested_terminal_attempts = (
        unknown_count
        if terminal_cap_search_max_attempts is None
        else max(0, int(terminal_cap_search_max_attempts))
    )
    terminal_attempts = min(unknown_count, requested_terminal_attempts)
    reasons: list[str] = list(blocking_reasons or [])
    estimate_payload: dict[str, Any] | None = None
    diagnostic_reprice_payload: dict[str, Any] | None = None
    if resolution_errors:
        reasons.append("CANDIDATE_COUNT_FREEZE_INCOMPLETE")
    elif not frozen_counts and not reasons:
        reasons.append("NO_SECTOR_EXECUTION_REQUIRED")
    elif not reasons:
        estimate = build_production_v2_cost_preflight(
            sector_candidate_counts=frozen_counts,
            terminal_cap_search_attempts=terminal_attempts,
            parent_max_turns=parent_max_turns,
            prior_realized_cost_usd=_cost_usd_string(prior_realized_cost_microdollars),
        )
        estimate_payload = estimate.to_dict()
        if diagnostic_reprice_model is not None:
            diagnostic_reprice_payload = build_diagnostic_v2_model_reprice(
                estimate,
                target_model=diagnostic_reprice_model,
            )
        reasons.extend(estimate.reason_codes)
        provider_error = _configured_v2_provider_error()
        if provider_error:
            reasons.append(provider_error)
    no_execution = reasons == ["NO_SECTOR_EXECUTION_REQUIRED"]
    spend_authorized = not reasons and estimate_payload is not None
    payload = {
        "artifact_type": V2_EXECUTION_PREFLIGHT_ARTIFACT_TYPE,
        "status": (
            "AUTHORIZED"
            if spend_authorized
            else "SKIPPED_NO_EXECUTION_REQUIRED"
            if no_execution
            else "STOP_BEFORE_SPEND"
        ),
        "spend_authorized": spend_authorized,
        "reason_codes": reasons,
        "provider_binding": {
            "provider": PRODUCTION_PROVIDER,
            "model": PRODUCTION_MODEL,
            "fallback_allowed": False,
        },
        "execution_bounds": {
            "max_serialized_request_bytes": PRODUCTION_MAX_SERIALIZED_REQUEST_BYTES,
            "max_output_tokens": PRODUCTION_MAX_OUTPUT_TOKENS,
            "max_retries_per_request": PRODUCTION_MAX_RETRIES_PER_REQUEST,
            "cached_input_tokens_assumed": 0,
            "filing_risk_llm_enabled": False,
            "filing_risk_classification": "DETERMINISTIC_KEYWORD_ONLY",
        },
        "frozen_candidate_counts": frozen_counts,
        "frozen_candidate_tickers": frozen_tickers,
        "membership_candidate_counts": {
            sector: len(tickers) for sector, tickers in membership_tickers.items()
        },
        "membership_candidate_tickers": membership_tickers,
        "execution_candidate_counts": {
            sector: len(tickers) for sector, tickers in execution_tickers.items()
        },
        "execution_candidate_tickers": execution_tickers,
        "deferred_by_bound_counts": {
            sector: len(tickers) for sector, tickers in deferred_tickers.items()
        },
        "deferred_by_bound_tickers": deferred_tickers,
        "excluded_candidate_tickers": excluded_tickers,
        "candidate_membership_fingerprints": {
            sector: payload.get("membership_fingerprint")
            for sector, payload in candidate_payloads.items()
        },
        "candidate_execution_fingerprints": {
            sector: payload.get("execution_fingerprint")
            for sector, payload in candidate_payloads.items()
        },
        "census_lineage_by_sector": {
            sector: dict(payload.get("census_lineage") or {})
            for sector, payload in candidate_payloads.items()
        },
        "unknown_cap_candidate_count": unknown_count,
        "unknown_cap_candidates": unknown_candidates,
        "terminal_cap_search_requested_max_attempts": requested_terminal_attempts,
        "terminal_cap_search_attempts": terminal_attempts,
        "prior_realized_cost_microdollars": prior_realized_cost_microdollars,
        "prior_realized_cost_usd": _cost_usd_string(prior_realized_cost_microdollars),
        "candidate_resolution_errors": dict(resolution_errors or {}),
        "estimate": estimate_payload,
    }
    if diagnostic_reprice_payload is not None:
        payload["diagnostic_model_reprice"] = diagnostic_reprice_payload
    return payload


def _copy_budget(budget: AutonomousRunBudget, *, max_candidates: int | None) -> AutonomousRunBudget:
    return AutonomousRunBudget(
        max_tool_calls=max(0, int(budget.max_tool_calls)),
        max_turns=max(1, int(budget.max_turns)),
        max_cost_usd=budget.max_cost_usd,
        timebox_seconds=budget.timebox_seconds,
        max_candidates=max_candidates,
    )


def _v2_candidate_request_bindings(
    *,
    sectors: list[str],
    candidate_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Canonical immutable census/membership/execution inputs for authority."""

    return {
        sector: {
            "membership_tickers": list(
                candidate_payloads.get(sector, {}).get("membership_tickers") or []
            ),
            "execution_tickers": list(
                candidate_payloads.get(sector, {}).get("execution_tickers") or []
            ),
            "deferred_by_bound_tickers": list(
                candidate_payloads.get(sector, {}).get("deferred_by_bound_tickers") or []
            ),
            "excluded_tickers": list(
                candidate_payloads.get(sector, {}).get("excluded_tickers") or []
            ),
            "membership_fingerprint": candidate_payloads.get(sector, {}).get(
                "membership_fingerprint"
            ),
            "execution_fingerprint": candidate_payloads.get(sector, {}).get(
                "execution_fingerprint"
            ),
            "execution_bound": candidate_payloads.get(sector, {}).get("execution_bound"),
            "census_lineage": dict(candidate_payloads.get(sector, {}).get("census_lineage") or {}),
        }
        for sector in sectors
    }


def _v2_execution_set_fingerprint(
    *,
    sectors: list[str],
    candidate_payloads: dict[str, dict[str, Any]],
) -> str:
    payload = [
        {
            "sector": sector,
            "execution_tickers": list(
                candidate_payloads.get(sector, {}).get("execution_tickers") or []
            ),
        }
        for sector in sectors
    ]
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _fmt_ratio_pct(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value) * 100:.1f}%"
    return "-"


def _audit_values(artifact: AutonomousSectorFinancialRunArtifact) -> dict[str, Any]:
    return artifact.selection_audit if isinstance(artifact.selection_audit, dict) else {}


def _final_decision_blockers(artifact: AutonomousSectorFinancialRunArtifact) -> list[str]:
    if artifact.final_decision is None:
        return []
    return list(artifact.final_decision.selection_blockers)


def _selected_evidence_count(artifact: AutonomousSectorFinancialRunArtifact) -> int:
    selected = str(artifact.selected_ticker or "").upper()
    if not selected:
        return 0
    return len([item for item in artifact.evidence if str(item.ticker or "").upper() == selected])


def _evidence_gap_candidates(artifact: AutonomousSectorFinancialRunArtifact) -> list[str]:
    audit = _audit_values(artifact)
    candidates: list[str] = []
    candidates.extend(str(item) for item in audit.get("confidence_caps", []) or [])
    candidates.extend(str(item) for item in audit.get("hard_blockers", []) or [])
    candidates.extend(str(item) for item in artifact.degraded_states)
    candidates.extend(str(item) for item in _final_decision_blockers(artifact))
    gap_markers = (
        "MISSING",
        "UNAVAILABLE",
        "UNKNOWN",
        "STALE",
        "UNREADABLE",
        "LOW_CONFIDENCE",
        "UNSUPPORTED",
        "NO_",
        "EVIDENCE",
    )
    gaps = [item for item in candidates if any(marker in item.upper() for marker in gap_markers)]
    return list(dict.fromkeys(gaps))


def _top_blockers(artifact: AutonomousSectorFinancialRunArtifact) -> list[str]:
    audit = _audit_values(artifact)
    blockers: list[str] = []
    blockers.extend(str(item) for item in audit.get("hard_blockers", []) or [])
    blockers.extend(str(item) for item in _final_decision_blockers(artifact))
    blockers.extend(str(item) for item in artifact.degraded_states if "LLM_PROVIDER_" in str(item))
    if artifact.no_selection_reason:
        blockers.append(str(artifact.no_selection_reason))
    return list(dict.fromkeys(item for item in blockers if item))


def _tool_failure_key(call: Any) -> str | None:
    status = str(getattr(call, "status", "") or "").strip().upper()
    if not status or status == "OK":
        return None
    tool_name = str(getattr(call, "tool_name", "") or "UNKNOWN_TOOL").strip() or "UNKNOWN_TOOL"
    return f"{tool_name}:{status}"


def _failed_tool_calls(artifact: AutonomousSectorFinancialRunArtifact) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for call in artifact.tool_calls:
        key = _tool_failure_key(call)
        if not key:
            continue
        failures.append(
            {
                "call_id": call.call_id,
                "tool_name": call.tool_name,
                "status": call.status,
                "question_id": call.question_id,
                "error": call.error,
                "failure_key": key,
            }
        )
    return failures


def _tool_call_counts(artifact: AutonomousSectorFinancialRunArtifact) -> dict[str, int]:
    return _counter_dict([str(call.tool_name or "UNKNOWN_TOOL") for call in artifact.tool_calls])


def _tool_call_value(call: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(call, dict):
        return call.get(field_name, default)
    return getattr(call, field_name, default)


def _canonical_benchmark_lane(value: Any, *, default: str) -> str:
    lane = str(value or "").strip()
    lane = _LEGACY_LANE_ALIASES.get(lane, lane)
    return lane if lane in CANONICAL_LANES else default


def _child_attempt_payloads(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Return each real child attempt once, without duplicating the final artifact."""

    attempts = run.get("attempts")
    if isinstance(attempts, list) and attempts:
        return [dict(item) for item in attempts if isinstance(item, dict)]
    nested = run.get("artifact")
    return [dict(nested)] if isinstance(nested, dict) else []


def _terminal_cap_usage_records(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> list[dict[str, Any]]:
    repair = (
        artifact.candidate_selection.get("data_gap_repair")
        if isinstance(artifact.candidate_selection, dict)
        else None
    )
    usage_records = (
        repair.get("terminal_cap_search_usage_records") if isinstance(repair, dict) else None
    )
    return [dict(item) for item in usage_records or [] if isinstance(item, dict)]


def _safe_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _cost_microdollars(record: dict[str, Any]) -> int:
    declared = record.get("cost_microdollars")
    if isinstance(declared, int) and not isinstance(declared, bool) and declared >= 0:
        return declared
    raw_cost = record.get("cost_estimate_usd", record.get("cost_usd", 0))
    try:
        amount = Decimal(str(raw_cost or 0))
    except (InvalidOperation, TypeError, ValueError):
        return 0
    if not amount.is_finite() or amount < 0:
        return 0
    return int(
        (amount * _MICRODOLLARS_PER_DOLLAR).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


def _cost_usd_string(microdollars: int) -> str:
    return f"{Decimal(max(0, microdollars)) / _MICRODOLLARS_PER_DOLLAR:.6f}"


def _empty_lane_usage_totals() -> dict[str, int]:
    return {field: 0 for field in _LANE_USAGE_INTEGER_FIELDS}


def _lane_usage_totals_payload(values: dict[str, int]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        field: int(values.get(field) or 0) for field in _LANE_USAGE_INTEGER_FIELDS
    }
    payload["cost_usd"] = _cost_usd_string(payload["cost_microdollars"])
    return payload


def _terminal_usage_status(record: dict[str, Any]) -> str:
    billing_status = str(record.get("billing_status") or "").upper()
    attempt_status = str(record.get("attempt_status") or "").upper()
    if billing_status == "WORST_CASE_RESERVED" or attempt_status in {"FAILED", "PENDING"}:
        return "ERROR"
    return "OK"


def _all_lane_tool_calls(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> list[dict[str, Any]]:
    """Flatten parent, repair, child, and validation calls with lane identity."""

    records: list[dict[str, Any]] = []
    physical_only_accounting = artifact.pipeline_version == "v2"

    def append(
        call: Any,
        *,
        lane: str,
        ticker: str | None = None,
        cost_microdollars: int = 0,
        dispatched: bool | None = None,
    ) -> None:
        status = str(_tool_call_value(call, "status", "UNKNOWN") or "UNKNOWN").upper()
        tool_name = str(_tool_call_value(call, "tool_name", "UNKNOWN_TOOL") or "UNKNOWN_TOOL")
        was_dispatched = status in {"OK", "ERROR"} if dispatched is None else bool(dispatched)
        record = {
            "call_id": _tool_call_value(call, "call_id"),
            "tool_name": tool_name,
            "status": status,
            "question_id": _tool_call_value(call, "question_id"),
            "error": _tool_call_value(call, "error"),
            "lane": _canonical_benchmark_lane(lane, default="parent_research"),
        }
        if physical_only_accounting:
            record["dispatched"] = was_dispatched
        if ticker:
            record["ticker"] = ticker
        if cost_microdollars:
            record["cost_microdollars"] = cost_microdollars
            record["cost_usd"] = _cost_usd_string(cost_microdollars)
        if was_dispatched and status == "ERROR" if physical_only_accounting else status != "OK":
            record["failure_key"] = f"{tool_name}:{status}"
        records.append(record)

    for call in artifact.tool_calls:
        question_id = str(getattr(call, "question_id", "") or "").upper()
        default_lane = (
            "repair_fallback" if question_id.startswith(("AGR", "WR")) else "parent_research"
        )
        append(
            call,
            lane=_canonical_benchmark_lane(getattr(call, "lane", None), default=default_lane),
        )
    for run_index, run in enumerate(artifact.company_autonomy_runs, start=1):
        if not isinstance(run, dict):
            continue
        ticker = str(run.get("ticker") or "").upper() or None
        attempts = _child_attempt_payloads(run)
        if attempts:
            for attempt in attempts:
                nested_calls = attempt.get("tool_calls")
                if not isinstance(nested_calls, list):
                    continue
                for call in nested_calls:
                    if isinstance(call, dict):
                        append(call, lane="company_underwriting", ticker=ticker)
            continue
        successful_calls = _safe_non_negative_int(run.get("tool_calls"))
        attempted_calls = max(
            successful_calls,
            _safe_non_negative_int(run.get("tool_call_attempts")),
        )
        for call_index in range(attempted_calls):
            status = (
                "OK"
                if call_index < successful_calls
                else "ERROR"
                if physical_only_accounting
                else "UNKNOWN"
            )
            append(
                {
                    "call_id": f"legacy-child-{run_index}-{call_index + 1}",
                    "tool_name": "UNKNOWN_CHILD_TOOL",
                    "status": status,
                    "error": (
                        None
                        if status == "OK" or not physical_only_accounting
                        else "Legacy compact record counted an attempted but unsuccessful call."
                    ),
                },
                lane="company_underwriting",
                ticker=ticker,
                dispatched=True if physical_only_accounting else None,
            )
    validation = getattr(artifact, "selection_validation", None)
    for call in list(getattr(validation, "tool_calls", []) or []):
        append(
            call,
            lane="selected_company_validation",
            ticker=str(getattr(validation, "selected_ticker", "") or "").upper() or None,
        )
    for index, usage in enumerate(_terminal_cap_usage_records(artifact), start=1):
        call_type = str(usage.get("call_type") or "")
        if call_type != "web_search_call":
            continue
        status = _terminal_usage_status(usage)
        append(
            {
                "call_id": usage.get("call_id") or f"terminal-cap-search-{index}",
                "tool_name": "web_search",
                "status": status,
                "question_id": f"TERMINAL_CAP:{str(usage.get('ticker') or '').upper()}",
                "error": usage.get("reason_code") if status != "OK" else None,
            },
            lane="terminal_cap_search",
            ticker=str(usage.get("ticker") or "").upper() or None,
            cost_microdollars=_cost_microdollars(usage),
        )
    return records


def _all_lane_provider_calls(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> list[dict[str, Any]]:
    """Flatten every persisted provider attempt without child-attempt duplication."""

    records: list[dict[str, Any]] = []

    def append(
        usage: dict[str, Any],
        *,
        lane: str,
        ticker: str | None = None,
        fallback_call_id: str,
    ) -> None:
        status = str(usage.get("status") or "OK").upper()
        record = {
            "provider_call_id": str(
                usage.get("provider_call_id") or usage.get("call_id") or fallback_call_id
            ),
            "lane": _canonical_benchmark_lane(lane, default="parent_research"),
            "provider": str(usage.get("provider") or "unknown"),
            "model": str(usage.get("model") or usage.get("response_model") or "unknown"),
            "status": status,
            "input_tokens": _safe_non_negative_int(usage.get("input_tokens")),
            "cached_input_tokens": _safe_non_negative_int(usage.get("cached_input_tokens")),
            "output_tokens": _safe_non_negative_int(usage.get("output_tokens")),
            "reserved_output_tokens": _safe_non_negative_int(usage.get("reserved_output_tokens")),
            "cost_microdollars": _cost_microdollars(usage),
        }
        record["cached_input_tokens"] = min(record["cached_input_tokens"], record["input_tokens"])
        record["cost_usd"] = _cost_usd_string(record["cost_microdollars"])
        if ticker:
            record["ticker"] = ticker
        records.append(record)

    for index, usage in enumerate(artifact.provider_usage, start=1):
        if not isinstance(usage, dict):
            continue
        append(
            usage,
            lane=_canonical_benchmark_lane(
                usage.get("lane"),
                default="parent_research",
            ),
            fallback_call_id=f"parent-provider-{index}",
        )
    for run_index, run in enumerate(artifact.company_autonomy_runs, start=1):
        if not isinstance(run, dict):
            continue
        ticker = str(run.get("ticker") or "").upper() or None
        attempts = _child_attempt_payloads(run)
        usage_groups = [
            attempt.get("provider_usage")
            for attempt in attempts
            if isinstance(attempt.get("provider_usage"), list)
        ]
        if not attempts and isinstance(run.get("provider_usage"), list):
            usage_groups = [run.get("provider_usage")]
        for attempt_index, usage_group in enumerate(usage_groups, start=1):
            for usage_index, usage in enumerate(usage_group or [], start=1):
                if isinstance(usage, dict):
                    append(
                        usage,
                        lane="company_underwriting",
                        ticker=ticker,
                        fallback_call_id=(
                            f"child-provider-{run_index}-{attempt_index}-{usage_index}"
                        ),
                    )
    validation = getattr(artifact, "selection_validation", None)
    for index, usage in enumerate(
        list(getattr(validation, "provider_usage", []) or []),
        start=1,
    ):
        if isinstance(usage, dict):
            append(
                usage,
                lane="selected_company_validation",
                ticker=str(getattr(validation, "selected_ticker", "") or "").upper() or None,
                fallback_call_id=f"selected-validation-provider-{index}",
            )
    terminal_usage_records = _terminal_cap_usage_records(artifact)
    for index, usage in enumerate(terminal_usage_records, start=1):
        if str(usage.get("call_type") or "") != "responses_model":
            continue
        attempt_key = (
            usage.get("authorization_run_id"),
            usage.get("attempt_number"),
            usage.get("ticker"),
        )
        reserved_web_cost = (
            sum(
                _cost_microdollars(record)
                for record in terminal_usage_records
                if str(record.get("call_type") or "") == "web_search_call_reserve"
                and (
                    record.get("authorization_run_id"),
                    record.get("attempt_number"),
                    record.get("ticker"),
                )
                == attempt_key
            )
            if any(value is not None for value in attempt_key)
            else 0
        )
        terminal_usage = {
            **usage,
            "status": _terminal_usage_status(usage),
            "cost_microdollars": _cost_microdollars(usage) + reserved_web_cost,
        }
        append(
            terminal_usage,
            lane="terminal_cap_search",
            ticker=str(usage.get("ticker") or "").upper() or None,
            fallback_call_id=f"terminal-cap-provider-{index}",
        )
    return records


def _lane_usage_summary_from_records(
    *,
    tool_calls: list[dict[str, Any]],
    provider_calls: list[dict[str, Any]],
    physical_only: bool = False,
) -> dict[str, Any]:
    lane_totals = {lane: _empty_lane_usage_totals() for lane in CANONICAL_LANES}
    for call in tool_calls:
        if physical_only and not bool(call.get("dispatched")):
            continue
        lane = _canonical_benchmark_lane(call.get("lane"), default="parent_research")
        totals = lane_totals[lane]
        totals["tool_call_attempts"] += 1
        if str(call.get("status") or "").upper() == "OK":
            totals["tool_calls_ok"] += 1
        else:
            totals["tool_calls_failed"] += 1
        totals["cost_microdollars"] += _safe_non_negative_int(call.get("cost_microdollars"))
    for call in provider_calls:
        lane = _canonical_benchmark_lane(call.get("lane"), default="parent_research")
        totals = lane_totals[lane]
        totals["provider_call_attempts"] += 1
        if str(call.get("status") or "").upper() == "OK":
            totals["provider_calls_ok"] += 1
        else:
            totals["provider_calls_failed"] += 1
        for field in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reserved_output_tokens",
        ):
            totals[field] += _safe_non_negative_int(call.get(field))
        totals["cost_microdollars"] += _safe_non_negative_int(call.get("cost_microdollars"))
    aggregate = _empty_lane_usage_totals()
    for totals in lane_totals.values():
        for field in _LANE_USAGE_INTEGER_FIELDS:
            aggregate[field] += totals[field]
    return {
        "currency": "USD",
        "cost_unit": "microdollars",
        "lanes": {lane: _lane_usage_totals_payload(lane_totals[lane]) for lane in CANONICAL_LANES},
        "aggregate": _lane_usage_totals_payload(aggregate),
        "aggregate_reconciles": True,
    }


def _normalized_declared_lane_usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    declared = value.get("summary") if isinstance(value.get("summary"), dict) else value
    raw_lanes = declared.get("lanes") if isinstance(declared, dict) else None
    if not isinstance(raw_lanes, dict):
        return None
    lane_totals: dict[str, dict[str, int]] = {}
    for lane in CANONICAL_LANES:
        raw = raw_lanes.get(lane)
        if raw is None and lane == "selected_company_validation":
            raw = raw_lanes.get("selected_validation")
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            return None
        totals = {
            field: _safe_non_negative_int(raw.get(field)) for field in _LANE_USAGE_INTEGER_FIELDS
        }
        if "cost_microdollars" not in raw:
            totals["cost_microdollars"] = _cost_microdollars(raw)
        if totals["tool_call_attempts"] != (totals["tool_calls_ok"] + totals["tool_calls_failed"]):
            return None
        if totals["provider_call_attempts"] != (
            totals["provider_calls_ok"] + totals["provider_calls_failed"]
        ):
            return None
        if totals["cached_input_tokens"] > totals["input_tokens"]:
            return None
        lane_totals[lane] = totals
    aggregate = _empty_lane_usage_totals()
    for totals in lane_totals.values():
        for field in _LANE_USAGE_INTEGER_FIELDS:
            aggregate[field] += totals[field]
    return {
        "currency": "USD",
        "cost_unit": "microdollars",
        "lanes": {lane: _lane_usage_totals_payload(lane_totals[lane]) for lane in CANONICAL_LANES},
        "aggregate": _lane_usage_totals_payload(aggregate),
        "aggregate_reconciles": True,
    }


def _sector_lane_usage(
    artifact: AutonomousSectorFinancialRunArtifact,
    *,
    tool_calls: list[dict[str, Any]],
    provider_calls: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    reconstructed = _lane_usage_summary_from_records(
        tool_calls=tool_calls,
        provider_calls=provider_calls,
        physical_only=artifact.pipeline_version == "v2",
    )
    declared = _normalized_declared_lane_usage(artifact.lane_usage)
    if declared is None:
        return reconstructed, "artifact_records_rebuilt"
    if declared != reconstructed:
        return reconstructed, "artifact_lane_usage_mismatch_rebuilt"
    return declared, "artifact_lane_usage_reconciled"


def _terminal_cap_search_accounting(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> dict[str, Any]:
    repair = (
        artifact.candidate_selection.get("data_gap_repair")
        if isinstance(artifact.candidate_selection, dict)
        else None
    )
    accounting = repair.get("terminal_cap_search_accounting") if isinstance(repair, dict) else None
    return (
        dict(accounting)
        if isinstance(accounting, dict)
        else {
            "lane_totals": {},
            "aggregate": {
                "response_calls": 0,
                "web_search_calls": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "cost_estimate_usd": 0.0,
            },
            "aggregate_reconciles": True,
            "ledger_path": None,
        }
    )


def _child_tool_call_status_counts(
    company_autonomy_runs: list[dict[str, Any]],
) -> Counter[str]:
    """Count every child call, falling back to the legacy compact OK count."""

    counts: Counter[str] = Counter()
    for run in company_autonomy_runs:
        if not isinstance(run, dict):
            continue
        attempts = _child_attempt_payloads(run)
        if attempts:
            for attempt in attempts:
                nested_calls = attempt.get("tool_calls")
                if not isinstance(nested_calls, list):
                    continue
                for call in nested_calls:
                    if isinstance(call, dict):
                        counts[str(call.get("status") or "UNKNOWN").upper()] += 1
            continue
        successful_calls = _safe_non_negative_int(run.get("tool_calls"))
        attempted_calls = max(
            successful_calls,
            _safe_non_negative_int(run.get("tool_call_attempts")),
        )
        counts["OK"] += successful_calls
        if attempted_calls > successful_calls:
            counts["UNKNOWN"] += attempted_calls - successful_calls
    return counts


def _parse_source_date(value: str | None) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date()


def _freshness_bucket_from_source_date(
    *, source_date: str | None, as_of_date: str | None
) -> str | None:
    published = _parse_source_date(source_date)
    anchor = _parse_source_date(as_of_date)
    if published is None or anchor is None:
        return None
    days = (anchor - published).days
    if days < 0:
        return "future_dated"
    if days == 0:
        return "same_day"
    if days <= 7:
        return "recent_7d"
    if days <= 30:
        return "current_30d"
    if days <= 90:
        return "current_90d"
    return "stale_over_90d"


def _json_from_excerpt(excerpt: str | None) -> Any:
    if not excerpt:
        return None
    try:
        return json.loads(excerpt)
    except (TypeError, json.JSONDecodeError):
        return None


def _raw_source_metadata(value: dict[str, Any]) -> dict[str, Any] | None:
    record: dict[str, Any] = {}
    for key in (
        "source_url",
        "published_at",
        "source_published_at",
        "source_date",
        "source_type",
        "title",
    ):
        if value.get(key):
            record[key] = value.get(key)
    return record or None


def _iter_source_quality_records(value: Any):
    if isinstance(value, dict):
        if isinstance(value.get("source_quality"), dict):
            record = dict(value["source_quality"])
            for meta_key in (
                "source_url",
                "published_at",
                "source_published_at",
                "source_type",
                "title",
            ):
                if meta_key in value and meta_key not in record:
                    record[meta_key] = value.get(meta_key)
            yield record
        elif "freshness_bucket" in value and (
            "source_family" in value or "source_quality_score" in value
        ):
            yield dict(value)
        else:
            raw_record = _raw_source_metadata(value)
            if raw_record is not None:
                yield raw_record
        for key, child in value.items():
            if key == "source_quality":
                continue
            yield from _iter_source_quality_records(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_source_quality_records(item)


def _source_freshness_fields(artifact: AutonomousSectorFinancialRunArtifact) -> dict[str, Any]:
    freshness_buckets: list[str] = []
    source_families: list[str] = []
    reputation_statuses: list[str] = []
    issue_sources: list[dict[str, Any]] = []
    reputation_issue_sources: list[dict[str, Any]] = []

    def add_source(
        *,
        evidence: Any,
        bucket: str,
        source_family: str | None = None,
        source_domain: str | None = None,
        source_url: str | None = None,
        source_date: str | None = None,
        reputation_status: str | None = None,
    ) -> None:
        freshness_buckets.append(bucket)
        if source_family:
            source_families.append(source_family)
        if reputation_status:
            reputation_statuses.append(reputation_status)
            if reputation_status == "missing":
                reputation_issue_sources.append(
                    {
                        "evidence_id": getattr(evidence, "evidence_id", None),
                        "source_label": getattr(evidence, "source_label", None),
                        "ticker": getattr(evidence, "ticker", None),
                        "source_family": source_family,
                        "source_domain": source_domain,
                        "source_url": source_url or getattr(evidence, "source_url", None),
                        "source_date": source_date or getattr(evidence, "source_date", None),
                        "source_reputation_status": reputation_status,
                    }
                )
        if bucket in _FRESHNESS_ISSUE_BUCKETS:
            issue_sources.append(
                {
                    "evidence_id": getattr(evidence, "evidence_id", None),
                    "source_label": getattr(evidence, "source_label", None),
                    "ticker": getattr(evidence, "ticker", None),
                    "freshness_bucket": bucket,
                    "source_family": source_family,
                    "source_domain": source_domain,
                    "source_url": source_url or getattr(evidence, "source_url", None),
                    "source_date": source_date or getattr(evidence, "source_date", None),
                }
            )

    for evidence in artifact.evidence:
        records = list(_iter_source_quality_records(_json_from_excerpt(evidence.excerpt)))
        if records:
            for record in records:
                source_url = str(record.get("source_url") or "") or getattr(
                    evidence, "source_url", None
                )
                source_date = str(
                    record.get("source_published_at")
                    or record.get("published_at")
                    or record.get("source_date")
                    or ""
                ) or getattr(evidence, "source_date", None)
                source_family = str(record.get("source_family") or "") or None
                source_domain = str(record.get("source_domain") or "") or None
                reputation_status = str(record.get("source_reputation_status") or "") or None
                bucket = str(record.get("freshness_bucket") or "").strip()
                if not bucket and (source_url or source_date):
                    quality = classify_source_quality(
                        source_type=str(record.get("source_type") or "") or None,
                        source_url=source_url,
                        published_at=source_date,
                        as_of_date=artifact.as_of_date,
                        source_reputation_path=get_config().research_source_reputation_path,
                    )
                    bucket = str(quality.get("freshness_bucket") or "").strip()
                    source_family = source_family or str(quality.get("source_family") or "") or None
                    source_domain = source_domain or str(quality.get("source_domain") or "") or None
                    reputation_status = (
                        reputation_status
                        or str(quality.get("source_reputation_status") or "")
                        or None
                    )
                if not bucket:
                    continue
                add_source(
                    evidence=evidence,
                    bucket=bucket,
                    source_family=source_family,
                    source_domain=source_domain,
                    source_url=source_url,
                    source_date=source_date,
                    reputation_status=reputation_status,
                )
            continue
        if evidence.source_url:
            quality = classify_source_quality(
                source_type=evidence.source_type,
                source_url=evidence.source_url,
                published_at=evidence.source_date,
                as_of_date=artifact.as_of_date,
                source_reputation_path=get_config().research_source_reputation_path,
            )
            add_source(
                evidence=evidence,
                bucket=str(quality.get("freshness_bucket") or "undated"),
                source_family=str(quality.get("source_family") or "") or None,
                source_domain=str(quality.get("source_domain") or "") or None,
                source_url=evidence.source_url,
                source_date=evidence.source_date,
                reputation_status=str(quality.get("source_reputation_status") or "") or None,
            )
            continue
        bucket = _freshness_bucket_from_source_date(
            source_date=evidence.source_date, as_of_date=artifact.as_of_date
        )
        if bucket:
            add_source(
                evidence=evidence,
                bucket=bucket,
                source_family=None,
                source_domain=None,
                source_url=evidence.source_url,
                source_date=evidence.source_date,
                reputation_status=None,
            )

    return {
        "source_freshness_bucket_counts": _counter_dict(freshness_buckets),
        "source_family_counts": _counter_dict(source_families),
        "source_reputation_status_counts": _counter_dict(reputation_statuses),
        "stale_evidence_count": len(
            [item for item in freshness_buckets if item == "stale_over_90d"]
        ),
        "freshness_issue_count": len(
            [item for item in freshness_buckets if item in _FRESHNESS_ISSUE_BUCKETS]
        ),
        "freshness_issue_sources": issue_sources[:8],
        "source_reputation_issue_count": len(reputation_issue_sources),
        "source_reputation_issue_sources": reputation_issue_sources[:8],
    }


def _framework_result_fields(
    artifact: AutonomousSectorFinancialRunArtifact,
    audit: dict[str, Any],
) -> dict[str, Any]:
    framework = artifact.framework
    required = [str(item) for item in audit.get("framework_required_evidence") or []]
    covered = [str(item) for item in audit.get("framework_required_evidence_covered") or []]
    missing = [str(item) for item in audit.get("framework_required_evidence_missing") or []]
    coverage_ratio = audit.get("framework_required_evidence_coverage_ratio")
    return {
        "framework_economic_model": framework.economic_model if framework else None,
        "framework_required_evidence": required,
        "framework_required_evidence_covered": covered,
        "framework_required_evidence_missing": missing,
        "framework_required_evidence_coverage_ratio": coverage_ratio,
        "framework_required_evidence_incomplete": bool(missing),
    }


def _framework_preflight_rows(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> list[dict[str, Any]]:
    return [dict(item) for item in artifact.framework_evidence_preflight if isinstance(item, dict)]


def _framework_preflight_status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return _counter_dict([str(row.get("packet_support_status") or "UNKNOWN") for row in rows])


def _framework_preflight_need_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return _counter_dict(
        [str(item) for row in rows for item in row.get("needs_tool_evidence", []) or []]
    )


def _framework_preflight_ratio_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(row.get("packet_support_ratio"))
        for row in rows
        if isinstance(row.get("packet_support_ratio"), (int, float))
        and not isinstance(row.get("packet_support_ratio"), bool)
    ]
    return {
        "preflight_candidate_count": len(rows),
        "average_packet_support_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
        "minimum_packet_support_ratio": round(min(ratios), 4) if ratios else None,
    }


def _framework_preflight_fields(artifact: AutonomousSectorFinancialRunArtifact) -> dict[str, Any]:
    rows = _framework_preflight_rows(artifact)
    return {
        "framework_evidence_preflight": rows,
        "framework_evidence_preflight_status_counts": _framework_preflight_status_counts(rows),
        "framework_evidence_preflight_need_counts": _framework_preflight_need_counts(rows),
        **_framework_preflight_ratio_summary(rows),
    }


def _framework_filter_fields(artifact: AutonomousSectorFinancialRunArtifact) -> dict[str, Any]:
    selection = (
        artifact.candidate_selection if isinstance(artifact.candidate_selection, dict) else {}
    )
    metadata = (
        selection.get("framework_evidence_filter")
        if isinstance(selection.get("framework_evidence_filter"), dict)
        else {}
    )
    excluded = [
        str(item).upper() for item in metadata.get("excluded_tickers") or [] if str(item).strip()
    ]
    return {
        "framework_evidence_filter_status": metadata.get("status"),
        "framework_evidence_filter_excluded_tickers": excluded,
        "framework_evidence_filter_excluded_count": len(excluded),
        "framework_evidence_filter_selected_before": [
            str(item).upper()
            for item in metadata.get("selected_tickers_before_filter") or []
            if str(item).strip()
        ],
        "framework_evidence_filter_selected_after": [
            str(item).upper()
            for item in metadata.get("selected_tickers_after_filter") or []
            if str(item).strip()
        ],
    }


def _sector_result_from_artifact(
    *,
    artifact: AutonomousSectorFinancialRunArtifact,
    artifact_path: Path | None,
    report_path: Path | None,
    cache_coverage: dict[str, Any] | None = None,
    benchmark_execution_status: str = EXECUTION_RAN,
) -> dict[str, Any]:
    summary = sector_artifact_summary(artifact)
    audit = _audit_values(artifact)
    parent_failed_tool_calls = _failed_tool_calls(artifact)
    all_lane_tool_calls = _all_lane_tool_calls(artifact)
    all_lane_provider_calls = _all_lane_provider_calls(artifact)
    lane_usage, lane_accounting_source = _sector_lane_usage(
        artifact,
        tool_calls=all_lane_tool_calls,
        provider_calls=all_lane_provider_calls,
    )
    physical_tool_calls = [
        record for record in all_lane_tool_calls if bool(record.get("dispatched"))
    ]
    reported_tool_calls = (
        physical_tool_calls if artifact.pipeline_version == "v2" else all_lane_tool_calls
    )
    failed_tool_calls = [
        record
        for record in reported_tool_calls
        if (
            record["status"] == "ERROR"
            if artifact.pipeline_version == "v2"
            else record["status"] != "OK"
        )
    ]
    framework_fields = _framework_result_fields(artifact, audit)
    preflight_fields = _framework_preflight_fields(artifact)
    filter_fields = _framework_filter_fields(artifact)
    source_freshness_fields = _source_freshness_fields(artifact)
    lane_counter = Counter(record["lane"] for record in reported_tool_calls)
    repair_call_count = lane_counter["repair_fallback"]
    parent_call_count = lane_counter["parent_research"]
    child_status_counts = _child_tool_call_status_counts(artifact.company_autonomy_runs)
    validation_calls = list(
        getattr(getattr(artifact, "selection_validation", None), "tool_calls", []) or []
    )
    validation_status_counts = Counter(
        str(getattr(call, "status", None) or "UNKNOWN").upper() for call in validation_calls
    )
    if artifact.pipeline_version == "v2":
        tool_call_lane_counts = {
            lane: int(lane_usage["lanes"][lane]["tool_call_attempts"]) for lane in CANONICAL_LANES
        }
    else:
        tool_call_lane_counts = {
            "parent_research": parent_call_count,
            "repair_fallback": repair_call_count,
            "company_underwriting": lane_counter["company_underwriting"],
            "selected_validation": lane_counter["selected_company_validation"],
        }
        if lane_counter["terminal_cap_search"]:
            tool_call_lane_counts["terminal_cap_search"] = lane_counter["terminal_cap_search"]
    tool_call_lane_counts["aggregate"] = sum(tool_call_lane_counts.values())
    provider_call_lane_counts = {
        lane: int(lane_usage["lanes"][lane]["provider_call_attempts"]) for lane in CANONICAL_LANES
    }
    provider_call_lane_counts["aggregate"] = sum(provider_call_lane_counts.values())
    lane_aggregate = lane_usage["aggregate"]
    terminal_cap_accounting = _terminal_cap_search_accounting(artifact)
    validation_status_field = (
        "selected_company_validation_tool_call_status_counts"
        if artifact.pipeline_version == "v2"
        else "selected_validation_tool_call_status_counts"
    )
    return {
        "sector": artifact.sector,
        "status": str(artifact.execution_status or artifact.status or "FAILED"),
        "pipeline_version": artifact.pipeline_version,
        "execution_status": artifact.execution_status,
        "decision_status": artifact.decision_status,
        "benchmark_execution_status": benchmark_execution_status,
        "run_id": artifact.run_id,
        "final_verdict": artifact.final_verdict,
        "selected_ticker": artifact.selected_ticker,
        "confidence": artifact.confidence,
        "selection_audit_status": summary.get("selection_audit_status"),
        "actionable": bool(summary.get("actionable")),
        "audit_gap_repair_status": artifact.audit_gap_repair_status,
        "watchlist_resolution_status": artifact.watchlist_resolution_status,
        "no_selection_finalist_audit_status": artifact.no_selection_finalist_audit_status,
        "no_selection_finalist_resolution_status": artifact.no_selection_finalist_resolution_status,
        "alternate_finalist_audit_status": artifact.alternate_finalist_audit_status,
        "company_autonomy_attempted": artifact.company_autonomy_attempted,
        "company_autonomy_status": artifact.company_autonomy_status,
        "company_autonomy_decision_trace": summary.get("company_autonomy_decision_trace") or {},
        "company_autonomy_runs": summary.get("company_autonomy_runs") or [],
        "relative_ranking": artifact.relative_ranking,
        "top_ranked_ticker": artifact.relative_ranking[0].get("ticker")
        if artifact.relative_ranking
        else None,
        "top_ranked_actionable": artifact.relative_ranking[0].get("actionable")
        if artifact.relative_ranking
        else None,
        "degraded_states": list(artifact.degraded_states),
        "top_blockers": _top_blockers(artifact),
        "evidence_gaps": _evidence_gap_candidates(artifact),
        "failed_tool_calls": failed_tool_calls,
        "evidence_counts": {
            "company_packets": len(artifact.company_packets),
            "expected_return_scenarios": len(artifact.expected_return_scenarios),
            "research_questions": len(artifact.research_questions),
            "tool_calls_ok": (
                int(lane_aggregate["tool_calls_ok"])
                if artifact.pipeline_version == "v2"
                else len([call for call in artifact.tool_calls if call.status == "OK"])
            ),
            "tool_calls_total": (
                int(lane_aggregate["tool_call_attempts"])
                if artifact.pipeline_version == "v2"
                else len(artifact.tool_calls)
            ),
            "tool_calls_failed": (
                int(lane_aggregate["tool_calls_failed"])
                if artifact.pipeline_version == "v2"
                else len(parent_failed_tool_calls)
            ),
            "evidence_references": len(artifact.evidence),
            "selected_ticker_evidence_references": _selected_evidence_count(artifact),
        },
        "tool_call_status_counts": _counter_dict(
            [str(record["status"]) for record in reported_tool_calls]
        ),
        "tool_call_counts": _counter_dict(
            [str(record["tool_name"]) for record in reported_tool_calls]
        ),
        **(
            {
                "tool_call_diagnostic_status_counts": _counter_dict(
                    [str(record["status"]) for record in all_lane_tool_calls]
                )
            }
            if artifact.pipeline_version == "v2"
            else {}
        ),
        "tool_call_lane_counts": tool_call_lane_counts,
        "provider_call_status_counts": _counter_dict(
            [str(record["status"]) for record in all_lane_provider_calls]
        ),
        "provider_call_counts": _counter_dict(
            [str(record["provider"]) for record in all_lane_provider_calls]
        ),
        "provider_call_lane_counts": provider_call_lane_counts,
        "lane_usage": lane_usage,
        "lane_accounting_source": lane_accounting_source,
        "terminal_cap_search_accounting": terminal_cap_accounting,
        "company_underwriting_tool_call_status_counts": dict(child_status_counts),
        validation_status_field: dict(validation_status_counts),
        "tool_failure_counts": _counter_dict(
            [str(item["failure_key"]) for item in failed_tool_calls]
        ),
        **source_freshness_fields,
        **framework_fields,
        **preflight_fields,
        **filter_fields,
        "candidate_source": summary.get("candidate_source"),
        "candidate_warnings": list(artifact.candidate_selection.get("warnings") or []),
        "cache_coverage": cache_coverage or {},
        "execution_candidate_tickers": list(
            (cache_coverage or {}).get("final_candidate_pool")
            or artifact.candidate_selection.get("selected_tickers")
            or []
        ),
        "no_selection_reason": artifact.no_selection_reason,
        "audit_confidence_caps": list(audit.get("confidence_caps", []) or []),
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "artifact_sha256": _file_sha256(artifact_path),
        "report_path": str(report_path) if report_path is not None else None,
        "summary": summary,
    }


def _sector_result_from_exception(*, sector: str, exc: Exception) -> dict[str, Any]:
    error = f"{type(exc).__name__}: {exc}"
    row = {
        "sector": sector,
        "status": "FAILED",
        "benchmark_execution_status": EXECUTION_FAILED,
        "run_id": None,
        "final_verdict": "ERROR",
        "selected_ticker": None,
        "confidence": None,
        "selection_audit_status": None,
        "actionable": False,
        "audit_gap_repair_status": None,
        "watchlist_resolution_status": None,
        "no_selection_finalist_audit_status": None,
        "no_selection_finalist_resolution_status": None,
        "alternate_finalist_audit_status": None,
        "company_autonomy_attempted": False,
        "company_autonomy_status": None,
        "company_autonomy_decision_trace": {},
        "company_autonomy_runs": [],
        "relative_ranking": [],
        "top_ranked_ticker": None,
        "top_ranked_actionable": None,
        "degraded_states": ["BENCHMARK_SECTOR_RUN_FAILED"],
        "top_blockers": [error],
        "evidence_gaps": [],
        "failed_tool_calls": [],
        "evidence_counts": {
            "company_packets": 0,
            "expected_return_scenarios": 0,
            "research_questions": 0,
            "tool_calls_ok": 0,
            "tool_calls_total": 0,
            "tool_calls_failed": 0,
            "evidence_references": 0,
            "selected_ticker_evidence_references": 0,
        },
        "tool_call_status_counts": {},
        "tool_call_counts": {},
        "tool_failure_counts": {},
        "source_freshness_bucket_counts": {},
        "source_family_counts": {},
        "source_reputation_status_counts": {},
        "stale_evidence_count": 0,
        "freshness_issue_count": 0,
        "freshness_issue_sources": [],
        "source_reputation_issue_count": 0,
        "source_reputation_issue_sources": [],
        "framework_economic_model": None,
        "framework_required_evidence": [],
        "framework_required_evidence_covered": [],
        "framework_required_evidence_missing": [],
        "framework_required_evidence_coverage_ratio": None,
        "framework_required_evidence_incomplete": False,
        "framework_evidence_preflight": [],
        "framework_evidence_preflight_status_counts": {},
        "framework_evidence_preflight_need_counts": {},
        "preflight_candidate_count": 0,
        "average_packet_support_ratio": None,
        "minimum_packet_support_ratio": None,
        "candidate_source": None,
        "candidate_warnings": [],
        "cache_coverage": {},
        "execution_candidate_tickers": [],
        "no_selection_reason": error,
        "audit_confidence_caps": [],
        "artifact_path": None,
        "report_path": None,
        "error": error,
    }
    if isinstance(exc, InvalidFinancialInputError):
        row["financial_integrity_status"] = exc.status
        row["financial_integrity"] = exc.result.to_dict()
    return row


def _sector_result_skipped_provider(
    *,
    sector: str,
    provider_failure: dict[str, Any],
    pipeline_version: str = "v1",
) -> dict[str, Any]:
    failure_code = str(provider_failure.get("failure_code") or "LLM_PROVIDER_UNAVAILABLE")
    error = str(provider_failure.get("error") or failure_code)
    row = {
        "sector": sector,
        "status": "SKIPPED",
        "benchmark_execution_status": EXECUTION_SKIPPED_PROVIDER,
        "run_id": None,
        "final_verdict": "NO_SELECTION",
        "selected_ticker": None,
        "confidence": None,
        "selection_audit_status": "NOT_APPLICABLE",
        "actionable": False,
        "audit_gap_repair_status": None,
        "watchlist_resolution_status": None,
        "no_selection_finalist_audit_status": None,
        "no_selection_finalist_resolution_status": None,
        "alternate_finalist_audit_status": None,
        "company_autonomy_attempted": False,
        "company_autonomy_status": None,
        "company_autonomy_decision_trace": {},
        "company_autonomy_runs": [],
        "relative_ranking": [],
        "top_ranked_ticker": None,
        "top_ranked_actionable": None,
        "degraded_states": [failure_code],
        "top_blockers": [failure_code],
        "evidence_gaps": [],
        "failed_tool_calls": [],
        "evidence_counts": {
            "company_packets": 0,
            "expected_return_scenarios": 0,
            "research_questions": 0,
            "tool_calls_ok": 0,
            "tool_calls_total": 0,
            "tool_calls_failed": 0,
            "evidence_references": 0,
            "selected_ticker_evidence_references": 0,
        },
        "tool_call_status_counts": {},
        "tool_call_counts": {},
        "tool_failure_counts": {},
        "source_freshness_bucket_counts": {},
        "source_family_counts": {},
        "source_reputation_status_counts": {},
        "stale_evidence_count": 0,
        "freshness_issue_count": 0,
        "freshness_issue_sources": [],
        "source_reputation_issue_count": 0,
        "source_reputation_issue_sources": [],
        "framework_economic_model": None,
        "framework_required_evidence": [],
        "framework_required_evidence_covered": [],
        "framework_required_evidence_missing": [],
        "framework_required_evidence_coverage_ratio": None,
        "framework_required_evidence_incomplete": False,
        "framework_evidence_preflight": [],
        "framework_evidence_preflight_status_counts": {},
        "framework_evidence_preflight_need_counts": {},
        "preflight_candidate_count": 0,
        "average_packet_support_ratio": None,
        "minimum_packet_support_ratio": None,
        "candidate_source": None,
        "candidate_warnings": [],
        "cache_coverage": {},
        "execution_candidate_tickers": [],
        "no_selection_reason": f"Sector skipped because benchmark provider was unavailable: {failure_code}.",
        "audit_confidence_caps": [],
        "artifact_path": None,
        "report_path": None,
        "error": error,
    }
    if pipeline_version == "v2":
        row.update(
            {
                "pipeline_version": "v2",
                "execution_status": "FAILED",
                "decision_status": "INCOMPLETE",
                "final_verdict": None,
            }
        )
    return row


def _sector_result_stopped_cost_preflight(
    *,
    sector: str,
    candidate_payload: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    reasons = [str(item) for item in preflight.get("reason_codes") or []]
    blocker = reasons[0] if reasons else "WHOLE_RUN_COST_PREFLIGHT_NOT_AUTHORIZED"
    row = _sector_result_skipped_provider(
        sector=sector,
        provider_failure={
            "failure_code": blocker,
            "error": "Whole-run v2 cost preflight stopped execution before provider spend.",
        },
        pipeline_version="v2",
    )
    row.update(
        {
            "status": "FAILED",
            "benchmark_execution_status": EXECUTION_STOPPED_COST_PREFLIGHT,
            "degraded_states": reasons or [blocker],
            "top_blockers": reasons or [blocker],
            "evidence_gaps": reasons or [blocker],
            "candidate_source": candidate_payload.get("source"),
            "candidate_warnings": list(candidate_payload.get("warnings") or []),
            "execution_candidate_tickers": list(candidate_payload.get("selected_tickers") or []),
            "no_selection_reason": (
                "Whole-run v2 cost preflight stopped execution before provider spend: "
                + ", ".join(reasons or [blocker])
            ),
            "error": blocker,
        }
    )
    return row


def _sector_result_skipped_cache(
    *,
    sector: str,
    cache_coverage: dict[str, Any],
    pipeline_version: str = "v1",
) -> dict[str, Any]:
    reasons = list(
        cache_coverage.get("cache_limited_reasons")
        or cache_coverage.get("cache_readiness_warnings")
        or []
    )
    blocker = reasons[0] if reasons else "CACHE_NOT_READY"
    row = {
        "sector": sector,
        "status": "SKIPPED",
        "benchmark_execution_status": EXECUTION_SKIPPED_CACHE,
        "run_id": None,
        "final_verdict": "NO_SELECTION",
        "selected_ticker": None,
        "confidence": None,
        "selection_audit_status": "NOT_APPLICABLE",
        "actionable": False,
        "audit_gap_repair_status": None,
        "watchlist_resolution_status": None,
        "no_selection_finalist_audit_status": None,
        "no_selection_finalist_resolution_status": None,
        "alternate_finalist_audit_status": None,
        "company_autonomy_attempted": False,
        "company_autonomy_status": None,
        "company_autonomy_decision_trace": {},
        "company_autonomy_runs": [],
        "relative_ranking": [],
        "top_ranked_ticker": None,
        "top_ranked_actionable": None,
        "degraded_states": ["CACHE_NOT_READY"],
        "top_blockers": [blocker],
        "evidence_gaps": reasons or ["CACHE_NOT_READY"],
        "failed_tool_calls": [],
        "evidence_counts": {
            "company_packets": 0,
            "expected_return_scenarios": 0,
            "research_questions": 0,
            "tool_calls_ok": 0,
            "tool_calls_total": 0,
            "tool_calls_failed": 0,
            "evidence_references": 0,
            "selected_ticker_evidence_references": 0,
        },
        "tool_call_status_counts": {},
        "tool_call_counts": {},
        "tool_failure_counts": {},
        "source_freshness_bucket_counts": {},
        "source_family_counts": {},
        "source_reputation_status_counts": {},
        "stale_evidence_count": 0,
        "freshness_issue_count": 0,
        "freshness_issue_sources": [],
        "source_reputation_issue_count": 0,
        "source_reputation_issue_sources": [],
        "framework_economic_model": None,
        "framework_required_evidence": [],
        "framework_required_evidence_covered": [],
        "framework_required_evidence_missing": [],
        "framework_required_evidence_coverage_ratio": None,
        "framework_required_evidence_incomplete": False,
        "framework_evidence_preflight": [],
        "framework_evidence_preflight_status_counts": {},
        "framework_evidence_preflight_need_counts": {},
        "preflight_candidate_count": 0,
        "average_packet_support_ratio": None,
        "minimum_packet_support_ratio": None,
        "candidate_source": None,
        "candidate_warnings": list(cache_coverage.get("cache_readiness_warnings") or []),
        "cache_coverage": cache_coverage,
        "execution_candidate_tickers": [],
        "no_selection_reason": f"Sector skipped because cache-ready candidates were unavailable: {blocker}.",
        "audit_confidence_caps": [],
        "artifact_path": None,
        "report_path": None,
        "error": blocker,
    }
    if pipeline_version == "v2":
        row.update(
            {
                "pipeline_version": "v2",
                "execution_status": "FAILED",
                "decision_status": "INCOMPLETE",
                "final_verdict": None,
            }
        )
    return row


def _provider_failure_from_result(row: dict[str, Any]) -> dict[str, Any] | None:
    for state in row.get("degraded_states") or []:
        state_text = str(state or "")
        if state_text.startswith("LLM_PROVIDER_"):
            return {
                "status": "FAILED",
                "failure_code": state_text,
                "error": row.get("no_selection_reason") or state_text,
            }
    error = str(row.get("error") or row.get("no_selection_reason") or "")
    if "provider" in error.lower() or "openai" in error.lower() or "quota" in error.lower():
        return {
            "status": "FAILED",
            "failure_code": _classify_provider_failure(error),
            "error": error,
        }
    return None


def _sum_count_dicts(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        values = row.get(field_name)
        if not isinstance(values, dict):
            continue
        for key, count in values.items():
            counter[str(key)] += int(count or 0)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _tool_call_totals(results: list[dict[str, Any]]) -> dict[str, Any]:
    lane_totals: Counter[str] = Counter()
    total = 0
    ok = 0
    failed = 0
    canonical_output = any(str(row.get("pipeline_version") or "") == "v2" for row in results)
    if canonical_output:
        for lane in CANONICAL_LANES:
            lane_totals[lane] = 0
    for row in results:
        if row.get("benchmark_execution_status") == EXECUTION_REUSED:
            continue
        counts = row.get("evidence_counts") if isinstance(row.get("evidence_counts"), dict) else {}
        lane_usage = row.get("lane_usage")
        if (
            str(row.get("pipeline_version") or "") == "v2"
            and isinstance(lane_usage, dict)
            and isinstance(lane_usage.get("lanes"), dict)
        ):
            row_total = 0
            for lane in CANONICAL_LANES:
                lane_values = lane_usage["lanes"].get(lane)
                attempts = (
                    int(lane_values.get("tool_call_attempts") or 0)
                    if isinstance(lane_values, dict)
                    else 0
                )
                lane_totals[lane] += attempts
                row_total += attempts
                if isinstance(lane_values, dict):
                    ok += int(lane_values.get("tool_calls_ok") or 0)
                    failed += int(lane_values.get("tool_calls_failed") or 0)
            total += row_total
            continue
        lanes = row.get("tool_call_lane_counts")
        if isinstance(lanes, dict):
            for lane in ("parent_research", "repair_fallback", "company_underwriting"):
                lane_totals[lane] += int(lanes.get(lane) or 0)
            selected_lane = (
                "selected_company_validation" if canonical_output else "selected_validation"
            )
            lane_totals[selected_lane] += int(
                lanes.get("selected_company_validation") or lanes.get("selected_validation") or 0
            )
            if "terminal_cap_search" in lanes:
                lane_totals["terminal_cap_search"] += int(lanes.get("terminal_cap_search") or 0)
            total += int(lanes.get("aggregate") or 0)
            status_counts = row.get("tool_call_status_counts")
            if isinstance(status_counts, dict):
                ok += int(status_counts.get("OK") or 0)
                failed += sum(
                    int(count or 0)
                    for status, count in status_counts.items()
                    if str(status).upper() != "OK"
                )
            else:
                ok += int(counts.get("tool_calls_ok") or 0)
                failed += int(counts.get("tool_calls_failed") or 0)
        else:
            parent_total = int(counts.get("tool_calls_total") or 0)
            lane_totals["parent_research"] += parent_total
            total += parent_total
            ok += int(counts.get("tool_calls_ok") or 0)
            failed += int(counts.get("tool_calls_failed") or 0)
    lane_sum = sum(lane_totals.values())
    return {
        "tool_calls_total": total,
        "tool_calls_ok": ok,
        "tool_calls_failed": failed,
        "tool_call_failure_rate": round(failed / total, 4) if total else 0.0,
        "lane_totals": dict(sorted(lane_totals.items())),
        "lane_sum": lane_sum,
        "aggregate_reconciles": total == lane_sum,
    }


def _lane_usage_totals(
    results: list[dict[str, Any]],
    *,
    include_reused: bool = False,
) -> dict[str, Any]:
    """Sum exact per-sector ledgers and derive the aggregate only from lanes."""

    lane_totals = {lane: _empty_lane_usage_totals() for lane in CANONICAL_LANES}
    accounted_sector_count = 0
    for row in results:
        if not include_reused and row.get("benchmark_execution_status") == EXECUTION_REUSED:
            continue
        usage = row.get("lane_usage")
        lanes = usage.get("lanes") if isinstance(usage, dict) else None
        if not isinstance(lanes, dict):
            continue
        accounted_sector_count += 1
        for lane in CANONICAL_LANES:
            values = lanes.get(lane)
            if not isinstance(values, dict):
                continue
            for field in _LANE_USAGE_INTEGER_FIELDS:
                lane_totals[lane][field] += _safe_non_negative_int(values.get(field))
    aggregate = _empty_lane_usage_totals()
    for values in lane_totals.values():
        for field in _LANE_USAGE_INTEGER_FIELDS:
            aggregate[field] += values[field]
    return {
        "currency": "USD",
        "cost_unit": "microdollars",
        "accounted_sector_count": accounted_sector_count,
        "lanes": {lane: _lane_usage_totals_payload(lane_totals[lane]) for lane in CANONICAL_LANES},
        "aggregate": _lane_usage_totals_payload(aggregate),
        "aggregate_reconciles": True,
    }


def _lane_usage_totals_with_provider_preflight(
    results: list[dict[str, Any]],
    provider_preflight: dict[str, Any] | None,
) -> dict[str, Any]:
    """Add top-level health-check attempts to the canonical six-lane ledger."""

    payload = _lane_usage_totals(results)
    lanes = payload["lanes"]
    provider_lane = lanes["provider_preflight"]
    for call in (
        provider_preflight.get("provider_calls") if isinstance(provider_preflight, dict) else []
    ) or []:
        if not isinstance(call, dict):
            continue
        provider_lane["provider_call_attempts"] += 1
        status = str(call.get("status") or "ERROR").upper()
        provider_lane["provider_calls_ok" if status == "OK" else "provider_calls_failed"] += 1
        for field in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reserved_output_tokens",
        ):
            provider_lane[field] += _safe_non_negative_int(call.get(field))
        provider_lane["cost_microdollars"] += _cost_microdollars(call)
    provider_lane["cached_input_tokens"] = min(
        provider_lane["cached_input_tokens"], provider_lane["input_tokens"]
    )
    provider_lane["cost_usd"] = _cost_usd_string(provider_lane["cost_microdollars"])
    aggregate = _empty_lane_usage_totals()
    for lane in CANONICAL_LANES:
        for field in _LANE_USAGE_INTEGER_FIELDS:
            aggregate[field] += _safe_non_negative_int(lanes[lane].get(field))
    payload["aggregate"] = _lane_usage_totals_payload(aggregate)
    payload["aggregate_reconciles"] = all(
        aggregate[field]
        == sum(_safe_non_negative_int(lanes[lane].get(field)) for lane in CANONICAL_LANES)
        for field in _LANE_USAGE_INTEGER_FIELDS
    )
    return payload


def _spend_reconciliation(
    *,
    rollups: dict[str, Any],
    all_sector_cost_preflight: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reconcile this invocation's ledger with realized spend from resume lineage."""

    lane_usage = rollups.get("lane_usage_totals") if isinstance(rollups, dict) else None
    aggregate = lane_usage.get("aggregate") if isinstance(lane_usage, dict) else None
    current_actual = _safe_non_negative_int(
        aggregate.get("cost_microdollars") if isinstance(aggregate, dict) else 0
    )
    preflight = all_sector_cost_preflight if isinstance(all_sector_cost_preflight, dict) else {}
    prior_lineage = _safe_non_negative_int(preflight.get("prior_realized_cost_microdollars"))
    combined_realized = prior_lineage + current_actual
    estimate = preflight.get("estimate") if isinstance(preflight.get("estimate"), dict) else {}
    estimate_aggregate = (
        estimate.get("aggregate") if isinstance(estimate.get("aggregate"), dict) else {}
    )
    preflight_total = (
        _safe_non_negative_int(estimate_aggregate.get("cost_microdollars"))
        if estimate_aggregate
        else None
    )
    return {
        "currency": "USD",
        "cost_unit": "microdollars",
        "current_run_actual_cost_microdollars": current_actual,
        "current_run_actual_cost_usd": _cost_usd_string(current_actual),
        "prior_lineage_realized_cost_microdollars": prior_lineage,
        "prior_lineage_realized_cost_usd": _cost_usd_string(prior_lineage),
        "combined_realized_cost_microdollars": combined_realized,
        "combined_realized_cost_usd": _cost_usd_string(combined_realized),
        "combined_realized_reconciles": (combined_realized == prior_lineage + current_actual),
        "preflight_total_worst_case_cost_microdollars": preflight_total,
        "preflight_total_worst_case_cost_usd": (
            _cost_usd_string(preflight_total) if preflight_total is not None else None
        ),
        "combined_realized_within_preflight_total_worst_case": (
            combined_realized <= preflight_total if preflight_total is not None else None
        ),
    }


def _terminal_cap_search_totals(results: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "response_calls",
        "web_search_calls",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
    )
    aggregate: dict[str, Any] = {field: 0 for field in fields}
    aggregate["cost_estimate_usd"] = 0.0
    ledger_paths: list[str] = []
    for row in results:
        if row.get("benchmark_execution_status") == EXECUTION_REUSED:
            continue
        accounting = row.get("terminal_cap_search_accounting")
        if not isinstance(accounting, dict):
            continue
        values = accounting.get("aggregate")
        if not isinstance(values, dict):
            continue
        for field in fields:
            aggregate[field] += int(values.get(field) or 0)
        aggregate["cost_estimate_usd"] += float(values.get("cost_estimate_usd") or 0.0)
        ledger_path = str(accounting.get("ledger_path") or "").strip()
        if ledger_path:
            ledger_paths.append(ledger_path)
    aggregate["cost_estimate_usd"] = round(aggregate["cost_estimate_usd"], 6)
    lane_total = dict(aggregate)
    return {
        "lane_totals": {"terminal_cap_search": lane_total},
        "aggregate": dict(aggregate),
        "aggregate_reconciles": lane_total == aggregate,
        "ledger_paths": list(dict.fromkeys(ledger_paths)),
    }


def _framework_required_evidence_coverage(results: list[dict[str, Any]]) -> dict[str, Any]:
    required_rows = [row for row in results if row.get("framework_required_evidence")]
    ratios = [
        float(row.get("framework_required_evidence_coverage_ratio"))
        for row in required_rows
        if isinstance(row.get("framework_required_evidence_coverage_ratio"), (int, float))
        and not isinstance(row.get("framework_required_evidence_coverage_ratio"), bool)
    ]
    incomplete = [row for row in required_rows if row.get("framework_required_evidence_missing")]
    return {
        "sectors_with_required_evidence": len(required_rows),
        "complete_sector_count": len(required_rows) - len(incomplete),
        "incomplete_sector_count": len(incomplete),
        "average_coverage_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
        "minimum_coverage_ratio": round(min(ratios), 4) if ratios else None,
    }


def _framework_evidence_preflight_coverage(results: list[dict[str, Any]]) -> dict[str, Any]:
    preflight_rows = [
        item
        for row in results
        for item in row.get("framework_evidence_preflight", []) or []
        if isinstance(item, dict)
    ]
    ratios = [
        float(row.get("packet_support_ratio"))
        for row in preflight_rows
        if isinstance(row.get("packet_support_ratio"), (int, float))
        and not isinstance(row.get("packet_support_ratio"), bool)
    ]
    sectors_with_preflight = [row for row in results if row.get("framework_evidence_preflight")]
    incomplete_rows = [row for row in preflight_rows if row.get("needs_tool_evidence")]
    return {
        "sectors_with_preflight": len(sectors_with_preflight),
        "preflight_candidate_count": len(preflight_rows),
        "packet_support_present_count": len(preflight_rows) - len(incomplete_rows),
        "needs_tool_evidence_count": len(incomplete_rows),
        "average_packet_support_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
        "minimum_packet_support_ratio": round(min(ratios), 4) if ratios else None,
    }


def _benchmark_recommendations(
    *,
    sector_count: int,
    failed_sector_count: int,
    actionable_selection_count: int,
    provider_failure_counts: dict[str, int],
    tool_failure_counts: dict[str, int],
    evidence_gap_counts: dict[str, int],
    source_freshness_bucket_counts: dict[str, int],
    source_reputation_status_counts: dict[str, int],
    framework_required_evidence_incomplete_sectors: list[dict[str, Any]],
    framework_evidence_preflight_need_counts: dict[str, int],
    framework_evidence_filter_excluded_count: int,
    cache_limited_sectors: list[dict[str, Any]],
    ranked_non_actionable_finalists: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []

    def add(priority: str, reason_code: str, recommendation: str, supporting_count: int) -> None:
        recommendations.append(
            {
                "priority": priority,
                "reason_code": reason_code,
                "recommendation": recommendation,
                "supporting_count": int(supporting_count),
            }
        )

    provider_count = sum(int(count) for count in provider_failure_counts.values())
    if provider_count:
        add(
            "P0",
            "LLM_PROVIDER_FAILURES",
            "Stabilize provider availability or resume the benchmark after provider recovery before interpreting sector outcomes.",
            provider_count,
        )
    if failed_sector_count:
        add(
            "P0",
            "SECTOR_RUNTIME_FAILURES",
            "Convert failed sector exceptions into targeted regression tests and rerun the affected sectors.",
            failed_sector_count,
        )
    tool_failure_count = sum(int(count) for count in tool_failure_counts.values())
    if tool_failure_count:
        add(
            "P1",
            "AUTONOMOUS_TOOL_FAILURES",
            "Fix or guard the most common failing autonomous tools before expanding benchmark breadth.",
            tool_failure_count,
        )
    evidence_gap_count = sum(int(count) for count in evidence_gap_counts.values())
    if evidence_gap_count:
        add(
            "P1",
            "RECURRING_EVIDENCE_GAPS",
            "Prioritize the most frequent evidence gaps as tool-quality or source-breadth work.",
            evidence_gap_count,
        )
    freshness_issue_count = sum(
        int(count)
        for bucket, count in source_freshness_bucket_counts.items()
        if str(bucket) in _FRESHNESS_ISSUE_BUCKETS
    )
    if freshness_issue_count:
        add(
            "P1",
            "STALE_OR_UNDATED_EVIDENCE",
            "Prioritize fresh current-event, transcript, or readable filing evidence for sectors with stale or undated sources.",
            freshness_issue_count,
        )
    missing_reputation_count = int(source_reputation_status_counts.get("missing") or 0)
    if missing_reputation_count:
        add(
            "P1",
            "MISSING_SOURCE_REPUTATION",
            "Populate source reputation history for secondary-source domains before relying on source calibration.",
            missing_reputation_count,
        )
    if framework_required_evidence_incomplete_sectors:
        add(
            "P1",
            "FRAMEWORK_REQUIRED_EVIDENCE_GAPS",
            "Use framework-required evidence gaps to reduce generic tool choices and steer sector-specific follow-up tests.",
            len(framework_required_evidence_incomplete_sectors),
        )
    preflight_need_count = sum(
        int(count) for count in framework_evidence_preflight_need_counts.values()
    )
    if preflight_need_count:
        add(
            "P1",
            "FRAMEWORK_PREFLIGHT_EVIDENCE_NEEDS",
            "Use framework preflight needs to plan sector-specific evidence tools before candidates reach final audit.",
            preflight_need_count,
        )
    if framework_evidence_filter_excluded_count:
        add(
            "P2",
            "FRAMEWORK_PREFLIGHT_FILTERED_CANDIDATES",
            "Review framework pre-provider filter exclusions to calibrate whether zero-support candidates need better packet data or should stay out of provider runs.",
            framework_evidence_filter_excluded_count,
        )
    if cache_limited_sectors:
        add(
            "P1",
            "CACHE_LIMITED_SECTORS",
            "Increase cache prewarm scope or repair missing cache inputs for sectors with thin executable candidate pools.",
            len(cache_limited_sectors),
        )
    if ranked_non_actionable_finalists:
        add(
            "P2",
            "RANKED_BUT_NON_ACTIONABLE_FINALISTS",
            "Use top-ranked non-actionable finalists to drive focused evidence repair and watchlist/no-selection resolution tests.",
            len(ranked_non_actionable_finalists),
        )
    if (
        sector_count
        and actionable_selection_count == 0
        and not provider_count
        and not failed_sector_count
    ):
        add(
            "P2",
            "NO_ACTIONABLE_SELECTIONS",
            "Inspect whether no-selection outcomes reflect true underwriting discipline or recurring missing evidence.",
            sector_count,
        )
    if not recommendations:
        add(
            "P3",
            "EXPAND_BENCHMARK_BREADTH",
            "No blocking benchmark pattern dominated this run; add sectors or compare against benchmark history.",
            sector_count,
        )
    return recommendations


def _build_rollups(
    results: list[dict[str, Any]],
    provider_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verdicts = [str(row.get("final_verdict") or "UNKNOWN") for row in results]
    audit_statuses = [str(row.get("selection_audit_status") or "NONE") for row in results]
    provider_failures = [
        state
        for row in results
        for state in row.get("degraded_states", [])
        if str(state).startswith("LLM_PROVIDER_")
    ]
    blockers = [item for row in results for item in row.get("top_blockers", [])]
    evidence_gaps = [item for row in results for item in row.get("evidence_gaps", [])]
    actionable = [row for row in results if row.get("actionable") is True]
    watchlist = [row for row in results if row.get("final_verdict") == "WATCHLIST"]
    no_selection = [row for row in results if row.get("final_verdict") == "NO_SELECTION"]
    failed = [row for row in results if row.get("status") == "FAILED"]
    skipped = [row for row in results if row.get("status") == "SKIPPED"]
    execution_statuses = [
        str(row.get("benchmark_execution_status") or "UNKNOWN") for row in results
    ]
    company_autonomy_statuses = [
        str(row.get("company_autonomy_status") or "NOT_ATTEMPTED") for row in results
    ]
    company_autonomy_decision_traces = [
        row.get("company_autonomy_decision_trace")
        for row in results
        if isinstance(row.get("company_autonomy_decision_trace"), dict)
    ]
    company_autonomy_decision_impact_counts = Counter()
    for trace in company_autonomy_decision_traces:
        company_autonomy_decision_impact_counts.update(
            {
                str(key): int(value or 0)
                for key, value in (trace.get("impact_counts") or {}).items()
                if int(value or 0) > 0
            }
        )
    company_autonomy_decision_impact_sectors = [
        {
            "sector": row.get("sector"),
            "status": trace.get("status"),
            "changed_sector_finalist_decision": bool(trace.get("changed_sector_finalist_decision")),
            "validated_sector_finalist_decision": bool(
                trace.get("validated_sector_finalist_decision")
            ),
            "contradicted_sector_finalist_decision": bool(
                trace.get("contradicted_sector_finalist_decision")
            ),
            "selected_ticker_child_verdict": trace.get("selected_ticker_child_verdict"),
            "top_ranked_child_verdict": trace.get("top_ranked_child_verdict"),
            "impact_counts": dict(trace.get("impact_counts") or {}),
        }
        for row in results
        for trace in [
            row.get("company_autonomy_decision_trace")
            if isinstance(row.get("company_autonomy_decision_trace"), dict)
            else {}
        ]
        if trace.get("impact_counts")
    ]
    ranked_non_actionable = [
        {"sector": row.get("sector"), "ticker": row.get("top_ranked_ticker")}
        for row in results
        if row.get("top_ranked_ticker") and row.get("top_ranked_actionable") is False
    ]
    needs_follow_up = [
        {
            "sector": row.get("sector"),
            "reason": (
                row.get("evidence_gaps")
                or row.get("top_blockers")
                or [row.get("no_selection_reason")]
            )[0],
        }
        for row in results
        if row.get("status") == "FAILED"
        or row.get("final_verdict") in {"NO_SELECTION", "WATCHLIST"}
    ]
    cache_coverages = [
        row.get("cache_coverage")
        for row in results
        if isinstance(row.get("cache_coverage"), dict) and row.get("cache_coverage")
    ]
    cache_limited = [
        {
            "sector": row.get("sector"),
            "reasons": row.get("cache_coverage", {}).get("cache_limited_reasons", []),
        }
        for row in results
        if isinstance(row.get("cache_coverage"), dict)
        and row.get("cache_coverage", {}).get("cache_limited_reasons")
    ]
    cache_readiness_counts = Counter()
    cache_warnings = [
        warning
        for row in results
        if isinstance(row.get("cache_coverage"), dict)
        for warning in row.get("cache_coverage", {}).get("cache_readiness_warnings", [])
    ]
    for row in results:
        cache = row.get("cache_coverage") if isinstance(row.get("cache_coverage"), dict) else {}
        cache_readiness_counts.update(cache.get("readiness_counts") or {})
    tool_failure_sectors = [
        {
            "sector": row.get("sector"),
            "failed_tool_calls": row.get("failed_tool_calls") or [],
        }
        for row in results
        if row.get("failed_tool_calls")
    ]
    provider_failure_counts = _counter_dict(provider_failures)
    top_blocker_counts = _counter_dict(blockers)
    evidence_gap_counts = _counter_dict(evidence_gaps)
    tool_call_status_counts = _sum_count_dicts(results, "tool_call_status_counts")
    tool_call_diagnostic_status_counts = _sum_count_dicts(
        results,
        "tool_call_diagnostic_status_counts",
    )
    tool_call_counts = _sum_count_dicts(results, "tool_call_counts")
    tool_failure_counts = _sum_count_dicts(results, "tool_failure_counts")
    source_freshness_bucket_counts = _sum_count_dicts(results, "source_freshness_bucket_counts")
    source_family_counts = _sum_count_dicts(results, "source_family_counts")
    source_reputation_status_counts = _sum_count_dicts(results, "source_reputation_status_counts")
    freshness_issue_count = sum(int(row.get("freshness_issue_count") or 0) for row in results)
    stale_evidence_count = sum(int(row.get("stale_evidence_count") or 0) for row in results)
    source_reputation_issue_count = sum(
        int(row.get("source_reputation_issue_count") or 0) for row in results
    )
    freshness_issue_sectors = [
        {
            "sector": row.get("sector"),
            "stale_evidence_count": int(row.get("stale_evidence_count") or 0),
            "freshness_issue_count": int(row.get("freshness_issue_count") or 0),
            "source_freshness_bucket_counts": dict(row.get("source_freshness_bucket_counts") or {}),
            "freshness_issue_sources": list(row.get("freshness_issue_sources") or []),
        }
        for row in results
        if int(row.get("freshness_issue_count") or 0) > 0
    ]
    source_reputation_issue_sectors = [
        {
            "sector": row.get("sector"),
            "source_reputation_issue_count": int(row.get("source_reputation_issue_count") or 0),
            "source_reputation_status_counts": dict(
                row.get("source_reputation_status_counts") or {}
            ),
            "source_reputation_issue_sources": list(
                row.get("source_reputation_issue_sources") or []
            ),
        }
        for row in results
        if int(row.get("source_reputation_issue_count") or 0) > 0
    ]
    framework_evidence_preflight_status_counts = _sum_count_dicts(
        results, "framework_evidence_preflight_status_counts"
    )
    framework_evidence_preflight_need_counts = _sum_count_dicts(
        results, "framework_evidence_preflight_need_counts"
    )
    framework_evidence_filter_status_counts = _counter_dict(
        [
            str(row.get("framework_evidence_filter_status"))
            for row in results
            if row.get("framework_evidence_filter_status")
        ]
    )
    framework_evidence_filter_excluded_count = sum(
        int(row.get("framework_evidence_filter_excluded_count") or 0) for row in results
    )
    framework_evidence_filter_excluded_sectors = [
        {
            "sector": row.get("sector"),
            "status": row.get("framework_evidence_filter_status"),
            "excluded_tickers": list(row.get("framework_evidence_filter_excluded_tickers") or []),
            "selected_tickers_before_filter": list(
                row.get("framework_evidence_filter_selected_before") or []
            ),
            "selected_tickers_after_filter": list(
                row.get("framework_evidence_filter_selected_after") or []
            ),
        }
        for row in results
        if int(row.get("framework_evidence_filter_excluded_count") or 0) > 0
    ]
    framework_required_evidence_gap_counts = _counter_dict(
        [
            str(item)
            for row in results
            for item in row.get("framework_required_evidence_missing", []) or []
        ]
    )
    framework_required_evidence_incomplete_sectors = [
        {
            "sector": row.get("sector"),
            "selected_ticker": row.get("selected_ticker") or row.get("top_ranked_ticker"),
            "missing": list(row.get("framework_required_evidence_missing") or []),
            "coverage_ratio": row.get("framework_required_evidence_coverage_ratio"),
        }
        for row in results
        if row.get("framework_required_evidence_missing")
    ]
    framework_evidence_preflight_incomplete_sectors = [
        {
            "sector": row.get("sector"),
            "needs": dict(row.get("framework_evidence_preflight_need_counts") or {}),
            "average_packet_support_ratio": row.get("average_packet_support_ratio"),
            "minimum_packet_support_ratio": row.get("minimum_packet_support_ratio"),
        }
        for row in results
        if row.get("framework_evidence_preflight_need_counts")
    ]
    sector_count = len(results)
    failed_sector_count = len(failed)
    actionable_selection_count = len(actionable)
    return {
        "sector_count": sector_count,
        "completed_sector_count": len([row for row in results if row.get("status") == "COMPLETED"]),
        "failed_sector_count": failed_sector_count,
        "skipped_sector_count": len(skipped),
        "benchmark_execution_status_counts": _counter_dict(execution_statuses),
        "verdict_counts": _counter_dict(verdicts),
        "selection_audit_status_counts": _counter_dict(audit_statuses),
        "actionable_selection_count": actionable_selection_count,
        "actionable_sectors": [row.get("sector") for row in actionable],
        "selected_tickers": [
            {"sector": row.get("sector"), "ticker": row.get("selected_ticker")}
            for row in actionable
            if row.get("selected_ticker")
        ],
        "watchlist_count": len(watchlist),
        "no_selection_count": len(no_selection),
        "provider_failure_counts": provider_failure_counts,
        "company_autonomy_status_counts": _counter_dict(company_autonomy_statuses),
        "company_autonomy_decision_impact_counts": dict(
            sorted(company_autonomy_decision_impact_counts.items())
        ),
        "company_autonomy_changed_sector_count": len(
            [
                row
                for row in company_autonomy_decision_impact_sectors
                if row.get("changed_sector_finalist_decision")
            ]
        ),
        "company_autonomy_validated_sector_count": len(
            [
                row
                for row in company_autonomy_decision_impact_sectors
                if row.get("validated_sector_finalist_decision")
            ]
        ),
        "company_autonomy_contradicted_sector_count": len(
            [
                row
                for row in company_autonomy_decision_impact_sectors
                if row.get("contradicted_sector_finalist_decision")
            ]
        ),
        "company_autonomy_decision_impact_sectors": company_autonomy_decision_impact_sectors,
        "ranked_non_actionable_finalists": ranked_non_actionable,
        "top_blocker_counts": top_blocker_counts,
        "evidence_gap_counts": evidence_gap_counts,
        "sectors_needing_follow_up": needs_follow_up,
        "tool_call_status_counts": tool_call_status_counts,
        "tool_call_diagnostic_status_counts": tool_call_diagnostic_status_counts,
        "tool_call_counts": tool_call_counts,
        "tool_failure_counts": tool_failure_counts,
        "tool_call_totals": _tool_call_totals(results),
        "lane_usage_totals": _lane_usage_totals_with_provider_preflight(
            results,
            provider_preflight,
        ),
        "prior_lineage_lane_usage_totals": _lane_usage_totals(
            [row for row in results if row.get("benchmark_execution_status") == EXECUTION_REUSED],
            include_reused=True,
        ),
        "terminal_cap_search_totals": _terminal_cap_search_totals(results),
        "tool_failure_sectors": tool_failure_sectors,
        "source_freshness_bucket_counts": source_freshness_bucket_counts,
        "source_family_counts": source_family_counts,
        "source_reputation_status_counts": source_reputation_status_counts,
        "stale_evidence_count": stale_evidence_count,
        "freshness_issue_count": freshness_issue_count,
        "freshness_issue_sectors": freshness_issue_sectors,
        "source_reputation_issue_count": source_reputation_issue_count,
        "source_reputation_issue_sectors": source_reputation_issue_sectors,
        "framework_model_counts": _counter_dict(
            [
                str(row.get("framework_economic_model"))
                for row in results
                if row.get("framework_economic_model")
            ]
        ),
        "framework_required_evidence_coverage": _framework_required_evidence_coverage(results),
        "framework_required_evidence_gap_counts": framework_required_evidence_gap_counts,
        "framework_required_evidence_incomplete_sectors": framework_required_evidence_incomplete_sectors,
        "framework_evidence_preflight_coverage": _framework_evidence_preflight_coverage(results),
        "framework_evidence_preflight_status_counts": framework_evidence_preflight_status_counts,
        "framework_evidence_preflight_need_counts": framework_evidence_preflight_need_counts,
        "framework_evidence_preflight_incomplete_sectors": framework_evidence_preflight_incomplete_sectors,
        "framework_evidence_filter_status_counts": framework_evidence_filter_status_counts,
        "framework_evidence_filter_excluded_count": framework_evidence_filter_excluded_count,
        "framework_evidence_filter_excluded_sectors": framework_evidence_filter_excluded_sectors,
        "cache_coverage_status_counts": _counter_dict(
            [
                str(item.get("coverage_status") or "UNKNOWN")
                for item in cache_coverages
                if isinstance(item, dict)
            ]
        ),
        "cache_limited_sectors": cache_limited,
        "cache_candidate_readiness_counts": dict(sorted(cache_readiness_counts.items())),
        "cache_readiness_warning_counts": _counter_dict([str(item) for item in cache_warnings]),
        "benchmark_recommendations": _benchmark_recommendations(
            sector_count=sector_count,
            failed_sector_count=failed_sector_count,
            actionable_selection_count=actionable_selection_count,
            provider_failure_counts=provider_failure_counts,
            tool_failure_counts=tool_failure_counts,
            evidence_gap_counts=evidence_gap_counts,
            source_freshness_bucket_counts=source_freshness_bucket_counts,
            source_reputation_status_counts=source_reputation_status_counts,
            framework_required_evidence_incomplete_sectors=framework_required_evidence_incomplete_sectors,
            framework_evidence_preflight_need_counts=framework_evidence_preflight_need_counts,
            framework_evidence_filter_excluded_count=framework_evidence_filter_excluded_count,
            cache_limited_sectors=cache_limited,
            ranked_non_actionable_finalists=ranked_non_actionable,
        ),
    }


def _cache_refresh_metadata(
    summary: dict[str, Any] | None, *, error: str | None = None
) -> dict[str, Any]:
    if error:
        return {"status": "FAILED", "error": error}
    if not isinstance(summary, dict) or not summary:
        return {}
    return {
        "run_id": summary.get("run_id"),
        "status": summary.get("status"),
        "ticker_count": summary.get("ticker_count"),
        "ticker_status_counts": summary.get("ticker_status_counts"),
        "step_status_counts": summary.get("step_status_counts"),
        "error_count": summary.get("error_count"),
        "warnings": summary.get("warnings", []),
        "summary_path": summary.get("summary_path"),
        "manifest_path": summary.get("manifest_path"),
        "report_path": summary.get("report_path"),
        "scope": summary.get("scope", {}),
        "cache_readiness": summary.get("cache_readiness", {}),
    }


def _resume_results_by_sector(resume_artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(resume_artifact, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in resume_artifact.get("sector_results") or []:
        if not isinstance(row, dict):
            continue
        sector = str(row.get("sector") or "").strip()
        if sector:
            out[sector] = row
    return out


def _resume_artifact_pipeline_version(resume_artifact: dict[str, Any]) -> str:
    value = str(resume_artifact.get("pipeline_version") or "").strip().lower()
    if value in {"v1", "v2"}:
        return value
    artifact_type = str(resume_artifact.get("artifact_type") or "").strip().lower()
    if artifact_type.endswith("_v2"):
        return "v2"
    return "v1"


def _canonical_resume_cap_band(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "micro": "micro_cap",
        "small": "small_cap",
        "mid": "mid_cap",
        "large": "large_cap",
        "mega": "mega_cap",
    }.get(normalized, normalized)


def _resume_artifact_compatibility(
    resume_artifact: dict[str, Any],
    *,
    pipeline_version: str,
    as_of_date: str | None,
    market_cap_focus: str,
    objective: str,
    max_candidates: int | None,
    budget: AutonomousRunBudget,
    expected_request_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Decide whether prior sector decisions are safe to reuse.

    V1 retains its legacy resume semantics. V2 requires a fixed-as-of,
    input-compatible benchmark so a resumed row cannot be presented under a
    different research request.
    """

    resume_pipeline_version = _resume_artifact_pipeline_version(resume_artifact)
    reasons: list[str] = []
    if resume_pipeline_version != pipeline_version:
        reasons.append("PIPELINE_VERSION_MISMATCH")

    strict_input_match = pipeline_version == "v2"
    if strict_input_match:
        requested_as_of = str(as_of_date or "").strip()
        resume_as_of = str(resume_artifact.get("as_of_date") or "").strip()
        if not requested_as_of or not resume_as_of:
            reasons.append("AS_OF_DATE_UNPINNED")
        elif resume_as_of != requested_as_of:
            reasons.append("AS_OF_DATE_MISMATCH")

        if _canonical_resume_cap_band(
            resume_artifact.get("market_cap_focus")
        ) != _canonical_resume_cap_band(market_cap_focus):
            reasons.append("MARKET_CAP_FOCUS_MISMATCH")

        if str(resume_artifact.get("objective") or "").strip() != str(objective or "").strip():
            reasons.append("OBJECTIVE_MISMATCH")

        if resume_artifact.get("max_candidates") != max_candidates:
            reasons.append("MAX_CANDIDATES_MISMATCH")

        resume_budget = resume_artifact.get("budget")
        if not isinstance(resume_budget, dict) or resume_budget != budget.to_dict():
            reasons.append("BUDGET_MISMATCH")

        if expected_request_fingerprint is not None:
            resume_request_fingerprint = (
                str(resume_artifact.get("request_fingerprint") or "").strip().lower()
            )
            if not resume_request_fingerprint:
                reasons.append("REQUEST_FINGERPRINT_UNPINNED")
            elif resume_request_fingerprint != expected_request_fingerprint:
                reasons.append("REQUEST_FINGERPRINT_MISMATCH")

    return {
        "compatible": not reasons,
        "strict_input_match": strict_input_match,
        "resume_pipeline_version": resume_pipeline_version,
        "reasons": reasons,
    }


def _can_reuse_resume_result(
    row: dict[str, Any],
    *,
    pipeline_version: str,
    resume_pipeline_version: str,
    rerun_completed_sectors: bool,
) -> bool:
    if rerun_completed_sectors:
        return False
    if resume_pipeline_version != pipeline_version:
        return False
    row_pipeline_version = str(row.get("pipeline_version") or "").strip().lower()
    if row_pipeline_version and row_pipeline_version != pipeline_version:
        return False
    if row.get("status") != "COMPLETED":
        return False
    if pipeline_version == "v2" and (
        row.get("execution_status") != "COMPLETED" or row.get("decision_status") != "COMPLETE"
    ):
        return False
    if row.get("benchmark_execution_status") == EXECUTION_SKIPPED_PROVIDER:
        return False
    return _provider_failure_from_result(row) is None


def _reuse_resume_result(row: dict[str, Any]) -> dict[str, Any]:
    reused = dict(row)
    reused["benchmark_execution_status"] = EXECUTION_REUSED
    reused["reused_from_resume"] = True
    return reused


def _benchmark_status(results: list[dict[str, Any]]) -> str:
    if any(
        row.get("benchmark_execution_status") == EXECUTION_STOPPED_COST_PREFLIGHT for row in results
    ):
        return "STOPPED_BEFORE_SPEND"
    if any(row.get("status") == "FAILED" for row in results):
        return "COMPLETED_WITH_FAILURES"
    if any(row.get("status") == "SKIPPED" for row in results):
        # Keep the legacy v1 status vocabulary stable. V2 makes skipped work an
        # explicit failed execution so a provider/budget/runtime skip cannot be
        # mistaken for a completed decision.
        if any(str(row.get("pipeline_version") or "v1").lower() == "v2" for row in results):
            return "FAILED_WITH_SKIPS"
        return "COMPLETED_WITH_SKIPS"
    if any(row.get("decision_status") == "INCOMPLETE" for row in results):
        return "COMPLETED_WITH_INCOMPLETE_DECISIONS"
    return "COMPLETED"


def _sector_execution_status(results: list[dict[str, Any]], sector: str) -> str | None:
    for row in results:
        if row.get("sector") == sector:
            status = row.get("benchmark_execution_status")
            return str(status) if status else None
    return None


def run_autonomous_sector_benchmark(
    *,
    sectors: list[str] | tuple[str, ...] | str | None = None,
    objective: str = DEFAULT_SECTOR_OBJECTIVE,
    as_of_date: str | None = None,
    market_cap_focus: str = "smid_cap",
    pipeline_version: str | None = None,
    budget: AutonomousRunBudget | None = None,
    max_candidates: int | None = None,
    prewarm_cache: bool = False,
    cache_refresh_run_id: str | None = None,
    cache_years: int = 10,
    force_cache_refresh: bool = False,
    cache_max_tickers_per_sector: int | None = None,
    provider_preflight: bool = True,
    continue_on_provider_unavailable: bool = False,
    resume_benchmark_run_id: str | None = None,
    rerun_completed_sectors: bool = False,
    terminal_cap_search: Any | None = None,
    terminal_cap_search_max_attempts: int | None = None,
    terminal_cap_search_ledger_path: str | Path | None = None,
    cost_preflight_only: bool = False,
    diagnostic_reprice_model: str | None = None,
    readiness_preflight_only: bool = False,
    free_data_repair_only: bool = False,
) -> dict[str, Any]:
    """Run autonomous sector analysis across multiple sectors and aggregate results."""

    created_at = utc_now().isoformat()
    # Allocate the benchmark identity before any sector work starts.  The same
    # identity is used by the completed artifact and by an interrupt checkpoint,
    # which makes a Ctrl-C checkpoint directly usable by --resume-benchmark-run-id.
    benchmark_run_id = _benchmark_run_id(created_at)
    sector_list = _normalize_sectors(sectors)
    resolved_pipeline_version = resolve_autonomous_sector_pipeline_version(
        market_cap_focus,
        pipeline_version,
    )
    if cost_preflight_only and resolved_pipeline_version != "v2":
        raise ValueError("cost-preflight-only is available only for pipeline v2")
    if readiness_preflight_only and resolved_pipeline_version != "v2":
        raise ValueError("readiness-preflight-only is available only for pipeline v2")
    if free_data_repair_only and resolved_pipeline_version != "v2":
        raise ValueError("free-data-repair-only is available only for pipeline v2")
    if readiness_preflight_only and not as_of_date:
        raise ValueError("readiness-preflight-only requires an explicit as-of date")
    if readiness_preflight_only and canonical_market_cap_focus(market_cap_focus) != (
        "large_and_mega"
    ):
        raise ValueError("readiness-preflight-only requires large_and_mega")
    if readiness_preflight_only and cost_preflight_only:
        raise ValueError("readiness-preflight-only and cost-preflight-only are mutually exclusive")
    if readiness_preflight_only and prewarm_cache:
        raise ValueError("readiness-preflight-only cannot run generic cache prewarming")
    if readiness_preflight_only and resume_benchmark_run_id:
        raise ValueError("readiness-preflight-only cannot reuse benchmark sector results")
    if readiness_preflight_only and terminal_cap_search is not None:
        raise ValueError("readiness-preflight-only cannot receive terminal search authority")
    if readiness_preflight_only and terminal_cap_search_max_attempts is not None:
        raise ValueError("readiness-preflight-only cannot reserve terminal search attempts")
    if readiness_preflight_only and terminal_cap_search_ledger_path is not None:
        raise ValueError("readiness-preflight-only cannot receive a terminal search ledger")
    if free_data_repair_only and not as_of_date:
        raise ValueError("free-data-repair-only requires an explicit as-of date")
    if free_data_repair_only and canonical_market_cap_focus(market_cap_focus) != ("large_and_mega"):
        raise ValueError("free-data-repair-only requires large_and_mega")
    if free_data_repair_only and (cost_preflight_only or readiness_preflight_only):
        raise ValueError(
            "free-data-repair-only, readiness-preflight-only, and "
            "cost-preflight-only are mutually exclusive"
        )
    if free_data_repair_only and (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or max_candidates < 1
        or max_candidates > 3
    ):
        raise ValueError("free-data-repair-only requires --max-candidates between 1 and 3")
    if free_data_repair_only and prewarm_cache:
        raise ValueError("free-data-repair-only cannot run generic cache prewarming")
    if free_data_repair_only and resume_benchmark_run_id:
        raise ValueError("free-data-repair-only cannot reuse benchmark sector results")
    if free_data_repair_only and rerun_completed_sectors:
        raise ValueError("free-data-repair-only cannot rerun benchmark sector results")
    if free_data_repair_only and terminal_cap_search is not None:
        raise ValueError("free-data-repair-only cannot receive terminal search authority")
    if free_data_repair_only and terminal_cap_search_max_attempts is not None:
        raise ValueError("free-data-repair-only cannot reserve terminal search attempts")
    if free_data_repair_only and terminal_cap_search_ledger_path is not None:
        raise ValueError("free-data-repair-only cannot receive a terminal search ledger")
    normalized_reprice_model = str(diagnostic_reprice_model or "").strip().lower() or None
    if normalized_reprice_model is not None and not cost_preflight_only:
        raise ValueError(
            "diagnostic-reprice-model requires cost-preflight-only; it cannot authorize execution"
        )
    if (
        normalized_reprice_model is not None
        and normalized_reprice_model not in DIAGNOSTIC_REPRICE_MODELS
    ):
        raise ValueError(
            "diagnostic reprice model must be one of " + ", ".join(DIAGNOSTIC_REPRICE_MODELS)
        )
    if resolved_pipeline_version == "v2" and prewarm_cache:
        raise ValueError(
            "v2 cache prewarming is disabled because cache refresh does not yet "
            "consume the frozen accepted-census execution authority"
        )
    if terminal_cap_search_max_attempts is not None and (
        isinstance(terminal_cap_search_max_attempts, bool)
        or int(terminal_cap_search_max_attempts) < 0
    ):
        raise ValueError("terminal_cap_search_max_attempts must be non-negative")
    if terminal_cap_search is not None:
        from app.autonomous.evidence_resolution import (
            _terminal_cap_search_is_authorized,
        )

        if resolved_pipeline_version != "v2":
            raise ValueError("terminal cap search is available only to explicit v2 runs")
        if not _terminal_cap_search_is_authorized(terminal_cap_search):
            raise ValueError("terminal cap search requires whole-run cost-preflight authorization")
    terminal_cap_search_authorization = (
        {
            **terminal_cap_search.authorization.to_dict(),
            "preflight_artifact_path": (
                str(getattr(terminal_cap_search, "preflight_artifact_path", "") or "") or None
            ),
            "ledger_path": str(getattr(terminal_cap_search, "ledger_path", "") or "") or None,
        }
        if terminal_cap_search is not None
        else None
    )
    run_budget = budget or AutonomousRunBudget(
        max_tool_calls=16,
        max_turns=6,
        max_cost_usd=None,
        timebox_seconds=None,
        max_candidates=max_candidates,
    )
    resume_source_run_id = str(resume_benchmark_run_id or "").strip() or None
    resume_artifact = _load_benchmark_resume(resume_source_run_id)
    resume_pipeline_version = _resume_artifact_pipeline_version(resume_artifact)
    resume_rows = _resume_results_by_sector(resume_artifact)

    # V2 freezes every requested sector before resume reuse, cost authorization,
    # provider health checks, repair, or any terminal lane.  Candidate discovery
    # is local-only here; unknown caps may enter the separately authorized
    # terminal lane only after this exact execution set is bound.
    frozen_candidate_selections: dict[str, Any] = {}
    frozen_candidate_payloads: dict[str, dict[str, Any]] = {}
    candidate_resolution_errors: dict[str, str] = {}
    accepted_census_authority = (
        AcceptedCensusRunAuthority()
        if resolved_pipeline_version == "v2"
        and canonical_market_cap_focus(market_cap_focus) == "large_and_mega"
        else None
    )
    if resolved_pipeline_version == "v2":
        from app.autonomous.evidence_resolution import v2_repair_checkpoint_path

        repair_as_of = as_of_date or date.today().isoformat()
        for sector in sector_list:
            try:
                selection = resolve_sector_candidate_tickers(
                    sector=sector,
                    explicit_tickers=[],
                    candidate_pool_tickers=[],
                    market_cap_focus=market_cap_focus,
                    max_candidates=max_candidates,
                    as_of_date=as_of_date,
                    pipeline_version="v2",
                    allow_live_market_data=False,
                    require_accepted_census=(
                        canonical_market_cap_focus(market_cap_focus) == "large_and_mega"
                    ),
                    accepted_census_authority=accepted_census_authority,
                )
                selection = freeze_v2_execution_bound(selection, max_candidates)
                payload = selection.to_dict()
                payload["execution_as_of_date"] = repair_as_of
                payload["data_gap_repair_checkpoint_path"] = str(
                    v2_repair_checkpoint_path(
                        tickers=selection.execution_tickers,
                        as_of_date=repair_as_of,
                        checkpoint_scope=f"{sector}:{market_cap_focus}",
                        candidate_context=payload,
                    )
                )
                frozen_candidate_selections[sector] = selection
                frozen_candidate_payloads[sector] = deepcopy(payload)
            except Exception as exc:  # noqa: BLE001 - fail closed before spend
                candidate_resolution_errors[sector] = f"{type(exc).__name__}: {exc}"
                frozen_candidate_payloads[sector] = {}

    expected_request_fingerprint: str | None = None
    execution_set_fingerprint: str | None = None
    terminal_attempt_ceiling = 0
    if resolved_pipeline_version == "v2":
        from app.autonomous.terminal_cap_search import (
            whole_run_preflight_request_fingerprint,
        )

        candidate_bindings = _v2_candidate_request_bindings(
            sectors=sector_list,
            candidate_payloads=frozen_candidate_payloads,
        )
        unknown_execution_count = len(_unknown_cap_candidates(frozen_candidate_payloads))
        requested_terminal_attempts = (
            unknown_execution_count
            if terminal_cap_search_max_attempts is None
            else int(terminal_cap_search_max_attempts)
        )
        terminal_attempt_ceiling = min(
            unknown_execution_count,
            requested_terminal_attempts,
        )
        execution_set_fingerprint = _v2_execution_set_fingerprint(
            sectors=sector_list,
            candidate_payloads=frozen_candidate_payloads,
        )
        expected_request_fingerprint = whole_run_preflight_request_fingerprint(
            sectors=sector_list,
            objective=objective,
            as_of_date=as_of_date or date.today().isoformat(),
            market_cap_focus=market_cap_focus,
            pipeline_version=resolved_pipeline_version,
            budget=run_budget.to_dict(),
            max_candidates=max_candidates,
            candidate_bindings=candidate_bindings,
            provider_name=PRODUCTION_PROVIDER,
            model=PRODUCTION_MODEL,
            terminal_cap_search_max_attempts=terminal_attempt_ceiling,
        )
        for payload in frozen_candidate_payloads.values():
            if payload:
                payload["request_fingerprint"] = expected_request_fingerprint
                payload["execution_set_fingerprint"] = execution_set_fingerprint
    if free_data_repair_only:
        from app.autonomous.accepted_census_free_data_repair import (
            run_accepted_census_free_data_repair,
        )

        free_data_repair = run_accepted_census_free_data_repair(
            benchmark_run_id=benchmark_run_id,
            sectors=sector_list,
            candidate_payloads=frozen_candidate_payloads,
            as_of_date=str(as_of_date),
            market_cap_focus=market_cap_focus,
            execution_set_fingerprint=str(execution_set_fingerprint or ""),
            request_fingerprint=str(expected_request_fingerprint or ""),
            candidate_resolution_errors=candidate_resolution_errors,
            accepted_census_authority=accepted_census_authority,
            cfg=get_config(),
        )
        readiness_preflight = dict(free_data_repair.get("after_readiness") or {})
        provider_preflight_result = _provider_preflight_skipped("free_data_repair_only")
        rollups = _build_rollups([], provider_preflight_result)
        completed_at = utc_now().isoformat()
        readiness_counts = (
            readiness_preflight.get("counts")
            if isinstance(readiness_preflight.get("counts"), dict)
            else {}
        )
        repair_completed = free_data_repair.get("status") == "COMPLETED"
        empty_preflight: dict[str, Any] = {}
        return {
            "run_id": benchmark_run_id,
            "created_at": created_at,
            "completed_at": completed_at,
            "status": (
                EXECUTION_MODE_FREE_DATA_REPAIR_ONLY
                if repair_completed
                else "FREE_DATA_REPAIR_INCOMPLETE"
            ),
            "execution_mode": EXECUTION_MODE_FREE_DATA_REPAIR_ONLY,
            "diagnostic_only": False,
            "maintenance_only": True,
            "spend_authorized": False,
            "objective": objective,
            "as_of_date": as_of_date,
            "market_cap_focus": market_cap_focus,
            "pipeline_version": resolved_pipeline_version,
            "max_candidates": max_candidates,
            "budget": run_budget.to_dict(),
            "sectors": sector_list,
            "request_fingerprint": expected_request_fingerprint,
            "execution_set_fingerprint": execution_set_fingerprint,
            "candidate_execution_ledger": _v2_candidate_request_bindings(
                sectors=sector_list,
                candidate_payloads=frozen_candidate_payloads,
            ),
            "fixed_cohort": {
                "schema_version": "all_sector_v2_free_data_repair_cohort_v1",
                "membership_security_count": int(
                    readiness_counts.get("membership_candidates") or 0
                ),
                "execution_security_count": int(readiness_counts.get("execution_candidates") or 0),
                "execution_issuer_count": int(readiness_counts.get("execution_candidates") or 0),
                "unresolved_issuer_identity": [],
                "census_lineage": dict(readiness_preflight.get("census_lineage") or {}),
            },
            "free_data_repair": free_data_repair,
            "readiness_preflight": readiness_preflight,
            "actual_usage": dict(free_data_repair.get("actual_usage") or {}),
            "production_sector_scan_exercised": False,
            "screened_candidate_count": 0,
            "underwritten_candidate_count": 0,
            "actionable_candidate_count": 0,
            "sector_results": [],
            "rollups": rollups,
            "spend_reconciliation": _spend_reconciliation(
                rollups=rollups,
                all_sector_cost_preflight=empty_preflight,
            ),
            "cache_refresh": {},
            "provider_preflight": provider_preflight_result,
            "all_sector_cost_preflight": empty_preflight,
            "terminal_cap_search_authorization": None,
            "resume_source_run_id": None,
            "resume_compatibility": {
                "requested": False,
                "compatible": None,
                "strict_input_match": True,
                "resume_pipeline_version": None,
                "reasons": [],
            },
            "resumed_sector_count": 0,
            "reused_sector_count": 0,
            "rerun_sector_count": 0,
            "skipped_sector_count": 0,
            "artifact_type": "autonomous_sector_benchmark_v2",
        }
    if readiness_preflight_only:
        from app.autonomous.readiness_preflight import run_v2_readiness_preflight

        readiness_preflight = run_v2_readiness_preflight(
            sectors=sector_list,
            candidate_payloads=frozen_candidate_payloads,
            as_of_date=str(as_of_date),
            market_cap_focus=market_cap_focus,
            execution_set_fingerprint=str(execution_set_fingerprint or ""),
            request_fingerprint=str(expected_request_fingerprint or ""),
            candidate_resolution_errors=candidate_resolution_errors,
            accepted_census_authority=accepted_census_authority,
            cfg=get_config(),
        )
        provider_preflight_result = _provider_preflight_skipped("readiness_preflight_only")
        rollups = _build_rollups([], provider_preflight_result)
        completed_at = utc_now().isoformat()
        readiness_counts = (
            readiness_preflight.get("counts")
            if isinstance(readiness_preflight.get("counts"), dict)
            else {}
        )
        readiness_completed = readiness_preflight.get("status") == "COMPLETED"
        empty_preflight: dict[str, Any] = {}
        return {
            "run_id": benchmark_run_id,
            "created_at": created_at,
            "completed_at": completed_at,
            "status": (
                EXECUTION_MODE_READINESS_PREFLIGHT_ONLY
                if readiness_completed
                else "READINESS_PREFLIGHT_INCOMPLETE"
            ),
            "execution_mode": EXECUTION_MODE_READINESS_PREFLIGHT_ONLY,
            "diagnostic_only": True,
            "objective": objective,
            "as_of_date": as_of_date,
            "market_cap_focus": market_cap_focus,
            "pipeline_version": resolved_pipeline_version,
            "max_candidates": max_candidates,
            "budget": run_budget.to_dict(),
            "sectors": sector_list,
            "request_fingerprint": expected_request_fingerprint,
            "execution_set_fingerprint": execution_set_fingerprint,
            "candidate_execution_ledger": _v2_candidate_request_bindings(
                sectors=sector_list,
                candidate_payloads=frozen_candidate_payloads,
            ),
            "fixed_cohort": {
                "schema_version": "all_sector_v2_readiness_cohort_v1",
                "membership_security_count": int(
                    readiness_counts.get("membership_candidates") or 0
                ),
                "execution_security_count": int(readiness_counts.get("execution_candidates") or 0),
                "execution_issuer_count": int(readiness_counts.get("execution_candidates") or 0),
                "unresolved_issuer_identity": [],
                "census_lineage": dict(readiness_preflight.get("census_lineage") or {}),
            },
            "readiness_preflight": readiness_preflight,
            "actual_usage": dict(readiness_preflight.get("actual_usage") or {}),
            "production_sector_scan_exercised": False,
            "screened_candidate_count": 0,
            "underwritten_candidate_count": 0,
            "actionable_candidate_count": 0,
            "sector_results": [],
            "rollups": rollups,
            "spend_reconciliation": _spend_reconciliation(
                rollups=rollups,
                all_sector_cost_preflight=empty_preflight,
            ),
            "cache_refresh": {},
            "provider_preflight": provider_preflight_result,
            "all_sector_cost_preflight": empty_preflight,
            "terminal_cap_search_authorization": None,
            "resume_source_run_id": None,
            "resume_compatibility": {
                "requested": False,
                "compatible": None,
                "strict_input_match": True,
                "resume_pipeline_version": None,
                "reasons": [],
            },
            "resumed_sector_count": 0,
            "reused_sector_count": 0,
            "rerun_sector_count": 0,
            "skipped_sector_count": 0,
            "artifact_type": "autonomous_sector_benchmark_v2",
        }
    if resume_source_run_id is None:
        resume_compatibility = {
            "requested": False,
            "compatible": None,
            "strict_input_match": resolved_pipeline_version == "v2",
            "resume_pipeline_version": None,
            "reasons": [],
        }
    elif not resume_artifact:
        resume_compatibility = {
            "requested": True,
            "compatible": False,
            "strict_input_match": resolved_pipeline_version == "v2",
            "resume_pipeline_version": None,
            "reasons": ["RESUME_ARTIFACT_NOT_FOUND"],
        }
    else:
        resume_compatibility = {
            "requested": True,
            **_resume_artifact_compatibility(
                resume_artifact,
                pipeline_version=resolved_pipeline_version,
                as_of_date=as_of_date,
                market_cap_focus=market_cap_focus,
                objective=objective,
                max_candidates=max_candidates,
                budget=run_budget,
                expected_request_fingerprint=expected_request_fingerprint,
            ),
        }
    resume_reuse_allowed = resume_compatibility["compatible"] is not False
    cache_refresh_summary: dict[str, Any] | None = None
    cache_refresh_error: str | None = None
    cache_tickers_by_sector: dict[str, list[str]] = {}
    results: list[dict[str, Any]] = []
    sectors_to_execute: list[str] = []
    for sector in sector_list:
        prior_row = resume_rows.get(sector)
        if (
            resume_reuse_allowed
            and prior_row
            and _can_reuse_resume_result(
                prior_row,
                pipeline_version=resolved_pipeline_version,
                resume_pipeline_version=resume_pipeline_version,
                rerun_completed_sectors=bool(rerun_completed_sectors),
            )
        ):
            results.append(_reuse_resume_result(prior_row))
        else:
            sectors_to_execute.append(sector)

    # A supplied callback contributes only its provider and ledger.  Its
    # pre-discovery authority must not constrain or survive the exact
    # candidate-bound whole-run preflight below.
    effective_terminal_max_attempts = terminal_cap_search_max_attempts
    resume_preflight_blockers = (
        ["RESUME_EXECUTION_FINGERPRINT_MISMATCH"]
        if (
            resolved_pipeline_version == "v2"
            and resume_source_run_id is not None
            and resume_compatibility.get("compatible") is False
        )
        else []
    )

    all_sector_cost_preflight = (
        _v2_execution_preflight_payload(
            candidate_payloads=frozen_candidate_payloads,
            parent_max_turns=run_budget.max_turns,
            prior_realized_cost_microdollars=(
                _resume_realized_cost_microdollars(resume_artifact) if resume_artifact else 0
            ),
            terminal_cap_search_max_attempts=(
                terminal_attempt_ceiling if effective_terminal_max_attempts is not None else None
            ),
            diagnostic_reprice_model=normalized_reprice_model,
            resolution_errors=candidate_resolution_errors,
            execution_sectors=sectors_to_execute,
            blocking_reasons=resume_preflight_blockers,
        )
        if resolved_pipeline_version == "v2"
        else {
            "artifact_type": V2_EXECUTION_PREFLIGHT_ARTIFACT_TYPE,
            "status": "NOT_APPLICABLE_V1",
            "spend_authorized": None,
            "reason_codes": [],
        }
    )
    execution_mode = (
        EXECUTION_MODE_COST_PREFLIGHT_ONLY if cost_preflight_only else EXECUTION_MODE_FULL
    )
    if resolved_pipeline_version == "v2":
        all_sector_cost_preflight["diagnostic_only"] = bool(cost_preflight_only)
        all_sector_cost_preflight["execution_requested"] = not bool(cost_preflight_only)
        all_sector_cost_preflight["request_fingerprint"] = expected_request_fingerprint
        all_sector_cost_preflight["execution_set_fingerprint"] = execution_set_fingerprint
        all_sector_cost_preflight["execution_bound"] = max_candidates
        if cost_preflight_only:
            if all_sector_cost_preflight.get("status") == "AUTHORIZED":
                all_sector_cost_preflight["status"] = "WITHIN_CEILING_AWAITING_USER_AUTHORIZATION"
                all_sector_cost_preflight["spend_authorized"] = False
                all_sector_cost_preflight.setdefault("reason_codes", []).append(
                    "COST_PREFLIGHT_ONLY_REQUIRES_SEPARATE_EXECUTION_REQUEST"
                )
            all_sector_cost_preflight.update(
                {
                    "actual_usage": {
                        "model_calls": 0,
                        "search_calls": 0,
                        "network_calls": 0,
                        "cost_microdollars": 0,
                        "cost_usd": "0.000000",
                    },
                    "production_sector_scan_exercised": False,
                    "screened_candidate_count": 0,
                    "underwritten_candidate_count": 0,
                    "actionable_candidate_count": 0,
                }
            )

    if (
        resolved_pipeline_version == "v2"
        and all_sector_cost_preflight.get("status") == "AUTHORIZED"
    ):
        from app.autonomous.terminal_cap_search import (
            bind_v2_cost_preflight_for_terminal_cap_search,
            build_authorized_terminal_cap_search,
            rebuild_authorized_terminal_cap_search,
        )

        if terminal_cap_search_ledger_path is not None:
            resolved_terminal_ledger_path = Path(terminal_cap_search_ledger_path)
        elif terminal_cap_search is not None:
            resolved_terminal_ledger_path = Path(terminal_cap_search.ledger_path)
        else:
            resolved_terminal_ledger_path = (
                get_config().runs_dir
                / "autonomous_sector_benchmark"
                / benchmark_run_id
                / "terminal_cap_search_ledger.json"
            )
        all_sector_cost_preflight = bind_v2_cost_preflight_for_terminal_cap_search(
            all_sector_cost_preflight,
            run_id=benchmark_run_id,
            authorized_at=created_at,
            request_fingerprint=str(expected_request_fingerprint or ""),
            ledger_path=resolved_terminal_ledger_path,
        )
        all_sector_cost_preflight["terminal_cap_search_ledger_path"] = str(
            resolved_terminal_ledger_path
        )
        if terminal_cap_search is not None:
            terminal_cap_search = rebuild_authorized_terminal_cap_search(
                source=terminal_cap_search,
                preflight=all_sector_cost_preflight,
                expected_request_fingerprint=expected_request_fingerprint,
            )
        elif (
            not cost_preflight_only
            and int(all_sector_cost_preflight.get("terminal_cap_search_attempts") or 0) > 0
        ):
            from app.llm.providers.openai_provider import OpenAIProvider

            terminal_cap_search = build_authorized_terminal_cap_search(
                preflight=all_sector_cost_preflight,
                provider=OpenAIProvider(get_config()),
                ledger_path=resolved_terminal_ledger_path,
                expected_request_fingerprint=expected_request_fingerprint,
            )
        if terminal_cap_search is not None:
            terminal_cap_search_authorization = {
                **terminal_cap_search.authorization.to_dict(),
                "preflight_artifact_path": (
                    str(getattr(terminal_cap_search, "preflight_artifact_path", "") or "") or None
                ),
                "ledger_path": (str(getattr(terminal_cap_search, "ledger_path", "") or "") or None),
            }
    elif resolved_pipeline_version == "v2":
        # A stopped or malformed fresh preflight revokes any callback supplied
        # under an earlier authority.  It is never exposed to sector execution
        # or persisted as if it remained authorized.
        terminal_cap_search = None
        terminal_cap_search_authorization = None

    if (
        resolved_pipeline_version == "v2"
        and all_sector_cost_preflight.get("status") == "STOP_BEFORE_SPEND"
        and not cost_preflight_only
    ):
        for sector in sectors_to_execute:
            results.append(
                _sector_result_stopped_cost_preflight(
                    sector=sector,
                    candidate_payload=frozen_candidate_payloads.get(sector, {}),
                    preflight=all_sector_cost_preflight,
                )
            )
        sectors_to_execute = []

    if cost_preflight_only:
        # Candidate membership and the exact estimate are the product of this
        # diagnostic mode. It intentionally creates no company decision rows.
        sectors_to_execute = []

    provider_preflight_result: dict[str, Any] | None = None
    if resolved_pipeline_version == "v2":
        provider_preflight_result = (
            _run_provider_preflight_in_v2_policy()
            if provider_preflight and sectors_to_execute
            else _provider_preflight_skipped(
                "cost_preflight_only"
                if cost_preflight_only
                else "disabled"
                if not provider_preflight
                else "no_sector_execution_required"
            )
        )
        if provider_preflight_result.get("status") == "FAILED":
            for sector in sectors_to_execute:
                results.append(
                    _sector_result_skipped_provider(
                        sector=sector,
                        provider_failure=provider_preflight_result,
                        pipeline_version=resolved_pipeline_version,
                    )
                )
            sectors_to_execute = []

    if prewarm_cache and sectors_to_execute:
        try:
            if cache_refresh_run_id and not force_cache_refresh:
                cache_refresh_summary = _load_cache_refresh_summary(cache_refresh_run_id)
                if not cache_refresh_summary:
                    cache_refresh_error = f"CACHE_REFRESH_RUN_NOT_FOUND:{cache_refresh_run_id}"
            if cache_refresh_summary is None and cache_refresh_error is None:
                from app.ingest.financial_cache_refresh import run_financial_cache_refresh

                cache_refresh_summary = run_financial_cache_refresh(
                    sectors=sectors_to_execute,
                    market_cap_focus=market_cap_focus,
                    max_tickers_per_sector=_benchmark_cache_prewarm_limit(
                        max_candidates, cache_max_tickers_per_sector
                    ),
                    sector_selection_mode="autonomous_candidates",
                    years=max(1, int(cache_years)),
                    as_of_date=as_of_date,
                    force=bool(force_cache_refresh),
                    resume_run_id=cache_refresh_run_id,
                )
            cache_tickers_by_sector = _refreshed_tickers_by_sector(cache_refresh_summary)
        except Exception as exc:  # noqa: BLE001
            cache_refresh_error = f"{type(exc).__name__}: {exc}"

    prepared_v1_sectors: dict[str, dict[str, Any]] = {}
    if resolved_pipeline_version != "v2" and sectors_to_execute:
        financially_ready_sectors: list[str] = []
        for sector in sectors_to_execute:
            try:
                cache_pool = cache_tickers_by_sector.get(sector, []) if prewarm_cache else []
                cache_coverage = (
                    _cache_readiness_for_tickers(
                        cache_refresh_summary,
                        cache_pool,
                        max_candidates=max_candidates,
                    )
                    if prewarm_cache
                    else {}
                )
                if prewarm_cache:
                    cache_pool = list(cache_coverage.get("ready_tickers") or [])
                    if not cache_pool:
                        results.append(
                            _sector_result_skipped_cache(
                                sector=sector,
                                cache_coverage=cache_coverage,
                                pipeline_version=resolved_pipeline_version,
                            )
                        )
                        continue
                candidate_selection = resolve_sector_candidate_tickers(
                    sector=sector,
                    explicit_tickers=[],
                    candidate_pool_tickers=cache_pool,
                    market_cap_focus=market_cap_focus,
                    max_candidates=max_candidates,
                    as_of_date=as_of_date,
                    pipeline_version=resolved_pipeline_version,
                    filing_risk_use_llm=False,
                    allow_live_market_data=False,
                )
                candidate_selection_payload = candidate_selection.to_dict()
                if prewarm_cache:
                    cache_coverage["final_candidate_pool"] = list(
                        candidate_selection.selected_tickers
                    )
                    cache_coverage["candidate_count"] = len(candidate_selection.selected_tickers)
                    if not candidate_selection.selected_tickers:
                        cache_coverage.setdefault("cache_limited_reasons", []).append(
                            "NO_SELECTABLE_CACHE_READY_CANDIDATES"
                        )
                        results.append(
                            _sector_result_skipped_cache(
                                sector=sector,
                                cache_coverage=cache_coverage,
                                pipeline_version=resolved_pipeline_version,
                            )
                        )
                        continue

                if provider_preflight:
                    # Provider health is itself a paid physical call.  Build
                    # and authorize the exact deterministic V1 candidate
                    # context first so an invalid/no-data sector incurs no
                    # provider attempt.
                    financial_as_of_date = str(as_of_date or date.today().isoformat())
                    financial_context = build_canonical_v1_financial_context(
                        tickers=candidate_selection.selected_tickers,
                        as_of_date=financial_as_of_date,
                        db_path=get_config().db_path,
                    )
                    integrity_scope = financial_context.scope(
                        context=(f"benchmark_provider_preflight:{sector}:{financial_as_of_date}")
                    )
                    integrity_result = require_financial_integrity_scope(integrity_scope)
                else:
                    integrity_scope = None
                    integrity_result = None
                prepared_v1_sectors[sector] = {
                    "candidate_selection": candidate_selection,
                    "candidate_selection_payload": candidate_selection_payload,
                    "cache_coverage": cache_coverage,
                    "integrity_scope": integrity_scope,
                    "integrity_scope_fingerprint": (
                        integrity_result.scope_fingerprint if integrity_result is not None else None
                    ),
                }
                financially_ready_sectors.append(sector)
            except Exception as exc:  # noqa: BLE001
                results.append(_sector_result_from_exception(sector=sector, exc=exc))
        sectors_to_execute = financially_ready_sectors

    if resolved_pipeline_version != "v2":
        if provider_preflight and sectors_to_execute:

            def _require_unchanged_v1_preflight_scopes(
                _attempt: dict[str, Any] | None = None,
            ) -> None:
                for prepared_sector in sectors_to_execute:
                    prepared = prepared_v1_sectors[prepared_sector]
                    scope = prepared.get("integrity_scope")
                    fingerprint = prepared.get("integrity_scope_fingerprint")
                    if scope is None or not fingerprint:
                        raise RuntimeError(
                            f"{prepared_sector} has no authorized V1 provider-preflight scope"
                        )
                    require_unchanged_financial_integrity_scope(
                        scope,
                        expected_scope_fingerprint=str(fingerprint),
                    )

            _require_unchanged_v1_preflight_scopes()
            try:
                with llm_physical_attempt_guard(_require_unchanged_v1_preflight_scopes):
                    provider_preflight_result = _run_provider_preflight(allow_fallback=True)
            except Exception as exc:
                try:
                    _require_unchanged_v1_preflight_scopes()
                except InvalidFinancialInputError as integrity_exc:
                    raise integrity_exc from exc
                raise
            _require_unchanged_v1_preflight_scopes()
        else:
            provider_preflight_result = _provider_preflight_skipped(
                "cost_preflight_only"
                if cost_preflight_only
                else "disabled"
                if not provider_preflight
                else "no_sector_execution_required"
            )
        if provider_preflight_result.get("status") == "FAILED":
            for sector in sectors_to_execute:
                results.append(
                    _sector_result_skipped_provider(
                        sector=sector,
                        provider_failure=provider_preflight_result,
                        pipeline_version=resolved_pipeline_version,
                    )
                )
            sectors_to_execute = []
    assert provider_preflight_result is not None

    provider_stop: dict[str, Any] | None = None
    for sector in sectors_to_execute:
        candidate_selection = None
        candidate_selection_payload: dict[str, Any] | None = None
        artifact = None
        try:
            cache_pool = cache_tickers_by_sector.get(sector, []) if prewarm_cache else []
            if resolved_pipeline_version == "v2":
                candidate_selection = frozen_candidate_selections.get(sector)
                candidate_selection_payload = deepcopy(frozen_candidate_payloads.get(sector, {}))
                cache_coverage = (
                    _cache_readiness_for_tickers(
                        cache_refresh_summary,
                        list(candidate_selection.execution_tickers)
                        if candidate_selection is not None
                        else cache_pool,
                        max_candidates=max_candidates,
                    )
                    if prewarm_cache
                    else {}
                )
            else:
                prepared = prepared_v1_sectors[sector]
                candidate_selection = prepared["candidate_selection"]
                candidate_selection_payload = deepcopy(prepared["candidate_selection_payload"])
                cache_coverage = dict(prepared["cache_coverage"])
            readiness_tickers = (
                list(candidate_selection.execution_tickers)
                if resolved_pipeline_version == "v2" and candidate_selection is not None
                else cache_pool
            )
            if resolved_pipeline_version == "v2" and prewarm_cache:
                cache_coverage = _cache_readiness_for_tickers(
                    cache_refresh_summary,
                    readiness_tickers,
                    max_candidates=max_candidates,
                )
            if candidate_selection is None or candidate_selection_payload is None:
                raise RuntimeError("Frozen v2 candidate selection is unavailable")
            if prewarm_cache:
                bounded_pool = (
                    candidate_selection.execution_tickers
                    if resolved_pipeline_version == "v2"
                    else candidate_selection.selected_tickers
                )
                cache_coverage["final_candidate_pool"] = list(bounded_pool)
                cache_coverage["candidate_count"] = len(bounded_pool)
                if resolved_pipeline_version != "v2" and not candidate_selection.selected_tickers:
                    cache_coverage.setdefault("cache_limited_reasons", []).append(
                        "NO_SELECTABLE_CACHE_READY_CANDIDATES"
                    )
                    results.append(
                        _sector_result_skipped_cache(
                            sector=sector,
                            cache_coverage=cache_coverage,
                            pipeline_version=resolved_pipeline_version,
                        )
                    )
                    continue
            execution_context = (
                _v2_execution_policy_context()
                if resolved_pipeline_version == "v2"
                else nullcontext()
            )
            with execution_context:
                artifact = run_sector_autonomous_financial_analysis(
                    sector=sector,
                    tickers=(
                        candidate_selection.execution_tickers
                        if resolved_pipeline_version == "v2"
                        else candidate_selection.selected_tickers
                    ),
                    objective=objective,
                    as_of_date=as_of_date,
                    market_cap_focus=market_cap_focus,
                    budget=_copy_budget(run_budget, max_candidates=max_candidates),
                    candidate_selection=candidate_selection_payload,
                    pipeline_version=resolved_pipeline_version,
                    terminal_cap_search=terminal_cap_search,
                )
            if prewarm_cache:
                cache_coverage["final_candidate_pool"] = list(
                    artifact.candidate_selection.get("execution_tickers")
                    or candidate_selection.execution_tickers
                    if resolved_pipeline_version == "v2"
                    else artifact.candidate_selection.get("selected_tickers")
                    or candidate_selection.selected_tickers
                )
                cache_coverage["candidate_count"] = len(cache_coverage["final_candidate_pool"])
            is_v2_diagnostic = artifact.pipeline_version == "v2" and (
                artifact.execution_status != "COMPLETED" or artifact.decision_status == "INCOMPLETE"
            )
            if is_v2_diagnostic:
                diagnostic_paths = persist_autonomous_sector_diagnostic(artifact)
                artifact_path = diagnostic_paths.artifact_json
                report_path = None
            else:
                paths = persist_autonomous_sector_run(artifact)
                artifact_path = paths.artifact_json
                report_path = paths.report_md
            row = _sector_result_from_artifact(
                artifact=artifact,
                artifact_path=artifact_path,
                report_path=report_path,
                cache_coverage=cache_coverage,
                benchmark_execution_status=(
                    EXECUTION_FAILED
                    if artifact.execution_status == "FAILED"
                    else EXECUTION_RERAN
                    if sector in resume_rows
                    else EXECUTION_RAN
                ),
            )
            results.append(row)
            provider_stop = _provider_failure_from_result(row)
            if provider_stop and not continue_on_provider_unavailable:
                break
        except KeyboardInterrupt as exc:
            diagnostic_path: Path | None = None
            if resolved_pipeline_version == "v2":
                diagnostic_paths = persist_autonomous_sector_attempt_diagnostic(
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    objective=objective,
                    as_of_date=as_of_date,
                    pipeline_version=resolved_pipeline_version,
                    candidate_selection=candidate_selection_payload or {},
                    error=exc,
                    last_completed_stage=(
                        "SECTOR_ANALYSIS" if artifact is not None else "CANDIDATE_SELECTION"
                    ),
                    artifact_snapshot=(artifact.to_dict() if artifact is not None else None),
                )
                diagnostic_path = diagnostic_paths.artifact_json

            # Preserve every already-finished sector in a resumable aggregate
            # checkpoint before propagating the interrupt.  Previously the CLI
            # only persisted the benchmark after this function returned, so a
            # Ctrl-C discarded the benchmark identity and made the completed
            # sector rows unusable by --resume-benchmark-run-id.
            interrupted_at = utc_now().isoformat()
            checkpoint_results = [
                row
                for checkpoint_sector in sector_list
                for row in results
                if row.get("sector") == checkpoint_sector
            ]
            resumed_sector_count = len(
                [
                    checkpoint_sector
                    for checkpoint_sector in sector_list
                    if checkpoint_sector in resume_rows
                ]
            )
            checkpoint_rollups = _build_rollups(
                checkpoint_results,
                provider_preflight_result,
            )
            _bind_sector_artifact_digests(checkpoint_results)
            checkpoint = {
                "run_id": benchmark_run_id,
                "created_at": created_at,
                "completed_at": None,
                "interrupted_at": interrupted_at,
                "status": "INTERRUPTED",
                "execution_mode": execution_mode,
                "objective": objective,
                "as_of_date": as_of_date,
                "market_cap_focus": market_cap_focus,
                "pipeline_version": resolved_pipeline_version,
                "max_candidates": max_candidates,
                "budget": run_budget.to_dict(),
                "sectors": sector_list,
                "request_fingerprint": expected_request_fingerprint,
                "execution_set_fingerprint": execution_set_fingerprint,
                "candidate_execution_ledger": _v2_candidate_request_bindings(
                    sectors=sector_list,
                    candidate_payloads=frozen_candidate_payloads,
                )
                if resolved_pipeline_version == "v2"
                else {},
                "sector_results": checkpoint_results,
                **(
                    {"fixed_cohort": _fixed_cohort_from_results(checkpoint_results)}
                    if resolved_pipeline_version == "v2"
                    else {}
                ),
                "rollups": checkpoint_rollups,
                "spend_reconciliation": _spend_reconciliation(
                    rollups=checkpoint_rollups,
                    all_sector_cost_preflight=all_sector_cost_preflight,
                ),
                "cache_refresh": _cache_refresh_metadata(
                    cache_refresh_summary, error=cache_refresh_error
                ),
                "provider_preflight": provider_preflight_result,
                "all_sector_cost_preflight": all_sector_cost_preflight,
                "terminal_cap_search_authorization": terminal_cap_search_authorization,
                "resume_source_run_id": resume_source_run_id,
                "resume_compatibility": resume_compatibility,
                "resumed_sector_count": resumed_sector_count,
                "reused_sector_count": len(
                    [
                        row
                        for row in checkpoint_results
                        if row.get("benchmark_execution_status") == EXECUTION_REUSED
                    ]
                ),
                "rerun_sector_count": len(
                    [
                        row
                        for row in checkpoint_results
                        if row.get("benchmark_execution_status") == EXECUTION_RERAN
                    ]
                ),
                "skipped_sector_count": len(
                    [
                        row
                        for row in checkpoint_results
                        if row.get("benchmark_execution_status")
                        in {EXECUTION_SKIPPED_PROVIDER, EXECUTION_SKIPPED_CACHE}
                    ]
                ),
                "interrupted_sector": sector,
                "interrupted_diagnostic_path": (
                    str(diagnostic_path) if diagnostic_path is not None else None
                ),
                "interrupted_diagnostic_sha256": _file_sha256(diagnostic_path),
                "interrupted_data_gap_repair_checkpoint_path": (
                    candidate_selection_payload.get("data_gap_repair_checkpoint_path")
                    if isinstance(candidate_selection_payload, dict)
                    else None
                ),
                "artifact_type": (f"autonomous_sector_benchmark_{resolved_pipeline_version}"),
                "artifact_class": "checkpoint",
            }
            checkpoint_paths = persist_autonomous_sector_benchmark(checkpoint)
            # Preserve discoverability for programmatic callers while retaining
            # the conventional KeyboardInterrupt behavior expected by Click.
            exc.benchmark_run_id = benchmark_run_id
            exc.benchmark_checkpoint_path = str(checkpoint_paths.summary_json)
            raise
        except Exception as exc:
            row = _sector_result_from_exception(sector=sector, exc=exc)
            if resolved_pipeline_version == "v2":
                diagnostic_paths = persist_autonomous_sector_attempt_diagnostic(
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    objective=objective,
                    as_of_date=as_of_date,
                    pipeline_version=resolved_pipeline_version,
                    candidate_selection=candidate_selection_payload or {},
                    error=exc,
                    last_completed_stage=(
                        "SECTOR_ANALYSIS" if artifact is not None else "CANDIDATE_SELECTION"
                    ),
                    artifact_snapshot=(artifact.to_dict() if artifact is not None else None),
                )
                row.update(
                    {
                        "pipeline_version": "v2",
                        "execution_status": "FAILED",
                        "decision_status": "INCOMPLETE",
                        "final_verdict": None,
                        "artifact_path": str(diagnostic_paths.artifact_json),
                        "artifact_sha256": _file_sha256(diagnostic_paths.artifact_json),
                        "report_path": None,
                    }
                )
            results.append(row)
            provider_stop = _provider_failure_from_result(row)
            if provider_stop and not continue_on_provider_unavailable:
                break

    if provider_stop and not continue_on_provider_unavailable:
        completed_sectors = {str(row.get("sector") or "") for row in results}
        for sector in sector_list:
            if sector not in completed_sectors:
                results.append(
                    _sector_result_skipped_provider(
                        sector=sector,
                        provider_failure=provider_stop,
                        pipeline_version=resolved_pipeline_version,
                    )
                )

    completed_at = utc_now().isoformat()
    ordered_results = []
    for sector in sector_list:
        match = next((row for row in results if row.get("sector") == sector), None)
        if match is not None:
            ordered_results.append(match)
    status = (
        EXECUTION_MODE_COST_PREFLIGHT_ONLY
        if cost_preflight_only
        else _benchmark_status(ordered_results)
    )
    resumed_sector_count = len([sector for sector in sector_list if sector in resume_rows])
    reused_sector_count = len(
        [
            row
            for row in ordered_results
            if row.get("benchmark_execution_status") == EXECUTION_REUSED
        ]
    )
    rerun_sector_count = len(
        [row for row in ordered_results if row.get("benchmark_execution_status") == EXECUTION_RERAN]
    )
    skipped_sector_count = len(
        [
            row
            for row in ordered_results
            if row.get("benchmark_execution_status")
            in {EXECUTION_SKIPPED_PROVIDER, EXECUTION_SKIPPED_CACHE}
        ]
    )
    _bind_sector_artifact_digests(ordered_results)
    rollups = _build_rollups(ordered_results, provider_preflight_result)
    artifact = {
        "run_id": benchmark_run_id,
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "execution_mode": execution_mode,
        "diagnostic_only": bool(cost_preflight_only),
        "objective": objective,
        "as_of_date": as_of_date,
        "market_cap_focus": market_cap_focus,
        "pipeline_version": resolved_pipeline_version,
        "max_candidates": max_candidates,
        "budget": run_budget.to_dict(),
        "sectors": sector_list,
        "request_fingerprint": expected_request_fingerprint,
        "execution_set_fingerprint": execution_set_fingerprint,
        "candidate_execution_ledger": _v2_candidate_request_bindings(
            sectors=sector_list,
            candidate_payloads=frozen_candidate_payloads,
        )
        if resolved_pipeline_version == "v2"
        else {},
        "sector_results": ordered_results,
        **(
            {"fixed_cohort": _fixed_cohort_from_results(ordered_results)}
            if resolved_pipeline_version == "v2"
            else {}
        ),
        "rollups": rollups,
        "spend_reconciliation": _spend_reconciliation(
            rollups=rollups,
            all_sector_cost_preflight=all_sector_cost_preflight,
        ),
        "cache_refresh": _cache_refresh_metadata(cache_refresh_summary, error=cache_refresh_error),
        "provider_preflight": provider_preflight_result,
        "all_sector_cost_preflight": all_sector_cost_preflight,
        "terminal_cap_search_authorization": terminal_cap_search_authorization,
        "resume_source_run_id": resume_source_run_id,
        "resume_compatibility": resume_compatibility,
        "resumed_sector_count": resumed_sector_count,
        "reused_sector_count": reused_sector_count,
        "rerun_sector_count": rerun_sector_count,
        "skipped_sector_count": skipped_sector_count,
        "artifact_type": f"autonomous_sector_benchmark_{resolved_pipeline_version}",
    }
    return artifact


def benchmark_compact_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return a concise JSON-safe benchmark summary for CLI output."""

    rollups = dict(artifact.get("rollups", {}))
    readiness = (
        artifact.get("readiness_preflight")
        if isinstance(artifact.get("readiness_preflight"), dict)
        else {}
    )
    free_data_repair = (
        artifact.get("free_data_repair")
        if isinstance(artifact.get("free_data_repair"), dict)
        else {}
    )
    return {
        "run_id": artifact.get("run_id"),
        "status": artifact.get("status"),
        "execution_mode": artifact.get("execution_mode", EXECUTION_MODE_FULL),
        "diagnostic_only": bool(artifact.get("diagnostic_only")),
        "readiness_status": readiness.get("readiness_status"),
        "readiness_counts": readiness.get("counts", {}),
        "readiness_missing_input_counts": readiness.get("missing_input_counts", {}),
        "readiness_actual_usage": readiness.get("actual_usage", {}),
        "free_data_repair_status": free_data_repair.get("status"),
        "free_data_repair_usage": free_data_repair.get("actual_usage", {}),
        "free_data_repair_network": free_data_repair.get("network", {}),
        "free_data_repair_transitions": free_data_repair.get("readiness_transition_counts", {}),
        "sector_count": rollups.get("sector_count"),
        "completed_sector_count": rollups.get("completed_sector_count"),
        "failed_sector_count": rollups.get("failed_sector_count"),
        "skipped_sector_count": artifact.get(
            "skipped_sector_count", rollups.get("skipped_sector_count")
        ),
        "benchmark_execution_status_counts": rollups.get("benchmark_execution_status_counts"),
        "verdict_counts": rollups.get("verdict_counts"),
        "selection_audit_status_counts": rollups.get("selection_audit_status_counts"),
        "actionable_selection_count": rollups.get("actionable_selection_count"),
        "actionable_sectors": rollups.get("actionable_sectors"),
        "selected_tickers": rollups.get("selected_tickers"),
        "watchlist_count": rollups.get("watchlist_count"),
        "no_selection_count": rollups.get("no_selection_count"),
        "provider_failure_counts": rollups.get("provider_failure_counts"),
        "company_autonomy_status_counts": rollups.get("company_autonomy_status_counts"),
        "company_autonomy_decision_impact_counts": rollups.get(
            "company_autonomy_decision_impact_counts"
        ),
        "company_autonomy_changed_sector_count": rollups.get(
            "company_autonomy_changed_sector_count"
        ),
        "company_autonomy_validated_sector_count": rollups.get(
            "company_autonomy_validated_sector_count"
        ),
        "company_autonomy_contradicted_sector_count": rollups.get(
            "company_autonomy_contradicted_sector_count"
        ),
        "company_autonomy_decision_impact_sectors": rollups.get(
            "company_autonomy_decision_impact_sectors"
        ),
        "ranked_non_actionable_finalists": rollups.get("ranked_non_actionable_finalists"),
        "tool_call_status_counts": rollups.get("tool_call_status_counts"),
        "tool_call_diagnostic_status_counts": rollups.get("tool_call_diagnostic_status_counts"),
        "tool_call_counts": rollups.get("tool_call_counts"),
        "tool_failure_counts": rollups.get("tool_failure_counts"),
        "tool_call_totals": rollups.get("tool_call_totals"),
        "tool_failure_sectors": rollups.get("tool_failure_sectors"),
        "source_freshness_bucket_counts": rollups.get("source_freshness_bucket_counts"),
        "source_family_counts": rollups.get("source_family_counts"),
        "source_reputation_status_counts": rollups.get("source_reputation_status_counts"),
        "stale_evidence_count": rollups.get("stale_evidence_count"),
        "freshness_issue_count": rollups.get("freshness_issue_count"),
        "freshness_issue_sectors": rollups.get("freshness_issue_sectors"),
        "source_reputation_issue_count": rollups.get("source_reputation_issue_count"),
        "source_reputation_issue_sectors": rollups.get("source_reputation_issue_sectors"),
        "framework_model_counts": rollups.get("framework_model_counts"),
        "framework_required_evidence_coverage": rollups.get("framework_required_evidence_coverage"),
        "framework_required_evidence_gap_counts": rollups.get(
            "framework_required_evidence_gap_counts"
        ),
        "framework_required_evidence_incomplete_sectors": rollups.get(
            "framework_required_evidence_incomplete_sectors"
        ),
        "framework_evidence_preflight_coverage": rollups.get(
            "framework_evidence_preflight_coverage"
        ),
        "framework_evidence_preflight_status_counts": rollups.get(
            "framework_evidence_preflight_status_counts"
        ),
        "framework_evidence_preflight_need_counts": rollups.get(
            "framework_evidence_preflight_need_counts"
        ),
        "framework_evidence_preflight_incomplete_sectors": rollups.get(
            "framework_evidence_preflight_incomplete_sectors"
        ),
        "framework_evidence_filter_status_counts": rollups.get(
            "framework_evidence_filter_status_counts"
        ),
        "framework_evidence_filter_excluded_count": rollups.get(
            "framework_evidence_filter_excluded_count"
        ),
        "framework_evidence_filter_excluded_sectors": rollups.get(
            "framework_evidence_filter_excluded_sectors"
        ),
        "benchmark_recommendations": rollups.get("benchmark_recommendations"),
        "provider_preflight": artifact.get("provider_preflight", {}),
        "all_sector_cost_preflight": artifact.get("all_sector_cost_preflight", {}),
        "spend_reconciliation": artifact.get("spend_reconciliation", {}),
        "lane_usage_totals": rollups.get("lane_usage_totals", {}),
        "resume_source_run_id": artifact.get("resume_source_run_id"),
        "resume_compatibility": artifact.get("resume_compatibility", {}),
        "resumed_sector_count": artifact.get("resumed_sector_count", 0),
        "reused_sector_count": artifact.get("reused_sector_count", 0),
        "rerun_sector_count": artifact.get("rerun_sector_count", 0),
        "cache_refresh": artifact.get("cache_refresh", {}),
        "cache_coverage_status_counts": rollups.get("cache_coverage_status_counts"),
        "cache_limited_sectors": rollups.get("cache_limited_sectors"),
        "cache_candidate_readiness_counts": rollups.get("cache_candidate_readiness_counts"),
        "cache_readiness_warning_counts": rollups.get("cache_readiness_warning_counts"),
        "top_blocker_counts": rollups.get("top_blocker_counts"),
        "evidence_gap_counts": rollups.get("evidence_gap_counts"),
        "sectors_needing_follow_up": rollups.get("sectors_needing_follow_up"),
        "sector_results": [
            {
                "sector": row.get("sector"),
                "status": row.get("status"),
                "benchmark_execution_status": row.get("benchmark_execution_status"),
                "final_verdict": row.get("final_verdict"),
                "selected_ticker": row.get("selected_ticker"),
                "selection_audit_status": row.get("selection_audit_status"),
                "actionable": row.get("actionable"),
                "company_autonomy_status": row.get("company_autonomy_status"),
                "company_autonomy_decision_trace": row.get("company_autonomy_decision_trace", {}),
                "top_ranked_ticker": row.get("top_ranked_ticker"),
                "cache_coverage": row.get("cache_coverage", {}),
                "framework_economic_model": row.get("framework_economic_model"),
                "framework_required_evidence_coverage_ratio": row.get(
                    "framework_required_evidence_coverage_ratio"
                ),
                "framework_required_evidence_missing": row.get(
                    "framework_required_evidence_missing", []
                ),
                "framework_evidence_preflight_need_counts": row.get(
                    "framework_evidence_preflight_need_counts", {}
                ),
                "framework_evidence_preflight_status_counts": row.get(
                    "framework_evidence_preflight_status_counts", {}
                ),
                "average_packet_support_ratio": row.get("average_packet_support_ratio"),
                "minimum_packet_support_ratio": row.get("minimum_packet_support_ratio"),
                "framework_evidence_filter_status": row.get("framework_evidence_filter_status"),
                "framework_evidence_filter_excluded_tickers": row.get(
                    "framework_evidence_filter_excluded_tickers", []
                ),
                "tool_call_status_counts": row.get("tool_call_status_counts", {}),
                "tool_call_diagnostic_status_counts": row.get(
                    "tool_call_diagnostic_status_counts", {}
                ),
                "tool_call_counts": row.get("tool_call_counts", {}),
                "tool_failure_counts": row.get("tool_failure_counts", {}),
                "failed_tool_calls": row.get("failed_tool_calls", []),
                "source_freshness_bucket_counts": row.get("source_freshness_bucket_counts", {}),
                "source_family_counts": row.get("source_family_counts", {}),
                "source_reputation_status_counts": row.get("source_reputation_status_counts", {}),
                "stale_evidence_count": row.get("stale_evidence_count", 0),
                "freshness_issue_count": row.get("freshness_issue_count", 0),
                "freshness_issue_sources": row.get("freshness_issue_sources", []),
                "source_reputation_issue_count": row.get("source_reputation_issue_count", 0),
                "source_reputation_issue_sources": row.get("source_reputation_issue_sources", []),
                "execution_candidate_tickers": row.get("execution_candidate_tickers", []),
                "artifact_path": row.get("artifact_path"),
                "report_path": row.get("report_path"),
            }
            for row in artifact.get("sector_results", [])
        ],
    }


def render_autonomous_sector_benchmark_report(artifact: dict[str, Any]) -> str:
    """Render a concise markdown report for benchmark validation."""

    rollups = artifact.get("rollups", {})
    lines = [
        f"# Autonomous Sector Benchmark: {artifact.get('run_id')}",
        "",
        f"**Status:** {artifact.get('status')}",
        f"**Execution mode:** {artifact.get('execution_mode', EXECUTION_MODE_FULL)}",
        f"**Created:** {artifact.get('created_at')}",
        f"**Market-cap focus:** {artifact.get('market_cap_focus')}",
        f"**Sectors:** {', '.join(artifact.get('sectors', []))}",
        "",
        "## Rollups",
        "",
        f"- Verdict counts: `{rollups.get('verdict_counts', {})}`",
        f"- Selection audit statuses: `{rollups.get('selection_audit_status_counts', {})}`",
        f"- Benchmark execution statuses: `{rollups.get('benchmark_execution_status_counts', {})}`",
        f"- Actionable selections: `{rollups.get('actionable_selection_count', 0)}`",
        f"- Provider failures: `{rollups.get('provider_failure_counts', {})}`",
        f"- Company autonomy statuses: `{rollups.get('company_autonomy_status_counts', {})}`",
        f"- Company autonomy decision impacts: `{rollups.get('company_autonomy_decision_impact_counts', {})}`",
        f"- Tool call statuses: `{rollups.get('tool_call_status_counts', {})}`",
        f"- Tool call diagnostic statuses: `{rollups.get('tool_call_diagnostic_status_counts', {})}`",
        f"- Tool calls by name: `{rollups.get('tool_call_counts', {})}`",
        f"- Tool failures: `{rollups.get('tool_failure_counts', {})}`",
        f"- Tool call totals: `{rollups.get('tool_call_totals', {})}`",
        f"- Source freshness buckets: `{rollups.get('source_freshness_bucket_counts', {})}`",
        f"- Source families: `{rollups.get('source_family_counts', {})}`",
        f"- Source reputation statuses: `{rollups.get('source_reputation_status_counts', {})}`",
        f"- Freshness issue count: `{rollups.get('freshness_issue_count', 0)}`",
        f"- Source reputation issue count: `{rollups.get('source_reputation_issue_count', 0)}`",
        f"- Framework models: `{rollups.get('framework_model_counts', {})}`",
        f"- Framework required evidence coverage: `{rollups.get('framework_required_evidence_coverage', {})}`",
        f"- Framework required evidence gaps: `{rollups.get('framework_required_evidence_gap_counts', {})}`",
        f"- Framework evidence preflight coverage: `{rollups.get('framework_evidence_preflight_coverage', {})}`",
        f"- Framework evidence preflight needs: `{rollups.get('framework_evidence_preflight_need_counts', {})}`",
        f"- Framework evidence filter statuses: `{rollups.get('framework_evidence_filter_status_counts', {})}`",
        f"- Framework evidence filter excluded candidates: `{rollups.get('framework_evidence_filter_excluded_count', 0)}`",
        f"- Cache coverage statuses: `{rollups.get('cache_coverage_status_counts', {})}`",
        f"- Cache candidate readiness: `{rollups.get('cache_candidate_readiness_counts', {})}`",
        f"- Cache readiness warnings: `{rollups.get('cache_readiness_warning_counts', {})}`",
        "",
    ]
    readiness = (
        artifact.get("readiness_preflight")
        if isinstance(artifact.get("readiness_preflight"), dict)
        else {}
    )
    if readiness:
        lines.extend(
            [
                "## Accepted-Census Readiness Preflight",
                "",
                f"- Status: `{readiness.get('status')}`",
                f"- Readiness: `{readiness.get('readiness_status')}`",
                f"- Authority validation: `{readiness.get('authority_validation_status')}`",
                f"- Census run: `{(readiness.get('census_lineage') or {}).get('run_id')}`",
                f"- Execution-set fingerprint: `{readiness.get('execution_set_fingerprint')}`",
                f"- Counts: `{readiness.get('counts', {})}`",
                f"- Missing inputs: `{readiness.get('missing_input_counts', {})}`",
                f"- Actual usage: `{readiness.get('actual_usage', {})}`",
                "- Packet materialization: `False`",
                "- Production sector scan exercised: `False`",
                "",
                "| Sector | Membership | Execution | Deferred | Ready | Needs Data | Incomplete |",
                "|--------|------------|-----------|----------|-------|------------|------------|",
            ]
        )
        readiness_sectors = (
            readiness.get("sector_results")
            if isinstance(readiness.get("sector_results"), dict)
            else {}
        )
        for sector in artifact.get("sectors") or []:
            row = readiness_sectors.get(sector)
            if not isinstance(row, dict):
                continue
            counts = (
                row.get("readiness_counts") if isinstance(row.get("readiness_counts"), dict) else {}
            )
            lines.append(
                "| {sector} | {membership} | {execution} | {deferred} | {ready} | {needs} | {incomplete} |".format(
                    sector=sector,
                    membership=len(row.get("membership_tickers") or []),
                    execution=len(row.get("execution_tickers") or []),
                    deferred=len(row.get("deferred_by_bound_tickers") or []),
                    ready=counts.get("READY", 0),
                    needs=counts.get("NEEDS_DATA", 0),
                    incomplete=counts.get("INCOMPLETE", 0),
                )
            )
        lines.append("")
    free_data_repair = (
        artifact.get("free_data_repair")
        if isinstance(artifact.get("free_data_repair"), dict)
        else {}
    )
    if free_data_repair:
        lines.extend(
            [
                "## Accepted-Census Free-Data Repair",
                "",
                f"- Status: `{free_data_repair.get('status')}`",
                f"- Maintenance only: `{free_data_repair.get('maintenance_only')}`",
                f"- Authority validation: `{free_data_repair.get('authority_validation_status')}`",
                f"- Scope preserved: `{free_data_repair.get('scope_preserved')}`",
                f"- Hard limits: `{free_data_repair.get('hard_limits', {})}`",
                f"- Readiness transitions: `{free_data_repair.get('readiness_transition_counts', {})}`",
                f"- Free network: `{free_data_repair.get('network', {})}`",
                f"- Actual usage: `{free_data_repair.get('actual_usage', {})}`",
                "- Paid providers: `[]`",
                "- Packet materialization: `False`",
                "- Production sector scan exercised: `False`",
                "",
            ]
        )
    cost_preflight = artifact.get("all_sector_cost_preflight")
    if isinstance(cost_preflight, dict) and cost_preflight:
        estimate = (
            cost_preflight.get("estimate")
            if isinstance(cost_preflight.get("estimate"), dict)
            else {}
        )
        aggregate = estimate.get("aggregate") if isinstance(estimate.get("aggregate"), dict) else {}
        lines.extend(
            [
                "## Whole-Run Cost Preflight",
                "",
                f"- Status: `{cost_preflight.get('status')}`",
                f"- Spend authorized: `{cost_preflight.get('spend_authorized')}`",
                f"- Reasons: `{cost_preflight.get('reason_codes', [])}`",
                f"- Frozen candidate counts: `{cost_preflight.get('frozen_candidate_counts', {})}`",
                f"- Prior realized cost: `${cost_preflight.get('prior_realized_cost_usd', '0.000000')}`",
                f"- Remaining worst case: `${aggregate.get('remaining_worst_case_cost_usd', '0.000000')}`",
                f"- Authorized total worst case: `${aggregate.get('cost_usd', '0.000000')}`",
                f"- Frozen candidates: `{cost_preflight.get('frozen_candidate_tickers', {})}`",
                f"- Unknown-cap candidates: `{cost_preflight.get('unknown_cap_candidates', [])}`",
                f"- Terminal-cap attempts reserved: `{cost_preflight.get('terminal_cap_search_attempts', 0)}`",
                f"- Terminal-cap ledger: `{cost_preflight.get('terminal_cap_search_ledger_path') or ''}`",
                "",
            ]
        )
        diagnostic_reprice = cost_preflight.get("diagnostic_model_reprice")
        if isinstance(diagnostic_reprice, dict):
            target = diagnostic_reprice.get("target_pricing_binding")
            envelope = diagnostic_reprice.get("cost_envelope")
            diagnostic_aggregate = envelope.get("aggregate") if isinstance(envelope, dict) else {}
            comparison = diagnostic_reprice.get("comparison")
            lines.extend(
                [
                    "## Diagnostic Model Reprice",
                    "",
                    f"- Target pricing binding: `{target or {}}`",
                    f"- Spend authorized: `{diagnostic_reprice.get('spend_authorized')}`",
                    f"- Execution binding unchanged: `{diagnostic_reprice.get('execution_binding_unchanged')}`",
                    f"- Worst-case comparison: `${diagnostic_aggregate.get('remaining_worst_case_cost_usd', '0.000000')}`",
                    f"- Estimated savings: `${(comparison or {}).get('estimated_savings_usd', '0.000000')}`",
                    f"- Rate card: `{diagnostic_reprice.get('rate_card', {})}`",
                    "",
                ]
            )
    spend = artifact.get("spend_reconciliation")
    if isinstance(spend, dict) and spend:
        lines.extend(
            [
                "## Spend Reconciliation",
                "",
                f"- Current run actual: `${spend.get('current_run_actual_cost_usd', '0.000000')}`",
                f"- Prior lineage realized: `${spend.get('prior_lineage_realized_cost_usd', '0.000000')}`",
                f"- Combined realized: `${spend.get('combined_realized_cost_usd', '0.000000')}`",
                f"- Exact reconciliation: `{spend.get('combined_realized_reconciles')}`",
                "",
            ]
        )
    provider_preflight = artifact.get("provider_preflight")
    if isinstance(provider_preflight, dict) and provider_preflight:
        lines.extend(
            [
                "## Provider Preflight",
                "",
                f"- Status: `{provider_preflight.get('status')}`",
                f"- Provider: `{provider_preflight.get('provider') or ''}`",
                f"- Failure code: `{provider_preflight.get('failure_code') or ''}`",
                f"- Error: `{provider_preflight.get('error') or ''}`",
                "",
            ]
        )
    if artifact.get("resume_source_run_id"):
        resume_compatibility = artifact.get("resume_compatibility")
        if not isinstance(resume_compatibility, dict):
            resume_compatibility = {}
        lines.extend(
            [
                "## Benchmark Resume",
                "",
                f"- Source run: `{artifact.get('resume_source_run_id')}`",
                f"- Input compatible: `{resume_compatibility.get('compatible')}`",
                f"- Compatibility reasons: `{resume_compatibility.get('reasons', [])}`",
                f"- Resumed sectors: `{artifact.get('resumed_sector_count', 0)}`",
                f"- Reused sectors: `{artifact.get('reused_sector_count', 0)}`",
                f"- Rerun sectors: `{artifact.get('rerun_sector_count', 0)}`",
                f"- Skipped sectors: `{artifact.get('skipped_sector_count', 0)}`",
                "",
            ]
        )
    cache_refresh = artifact.get("cache_refresh")
    if isinstance(cache_refresh, dict) and cache_refresh:
        refresh_readiness = (
            cache_refresh.get("cache_readiness")
            if isinstance(cache_refresh.get("cache_readiness"), dict)
            else {}
        )
        refresh_rollups = (
            refresh_readiness.get("rollups")
            if isinstance(refresh_readiness.get("rollups"), dict)
            else {}
        )
        lines.extend(
            [
                "## Cache Refresh",
                "",
                f"- Status: `{cache_refresh.get('status')}`",
                f"- Run ID: `{cache_refresh.get('run_id') or ''}`",
                f"- Tickers: `{cache_refresh.get('ticker_count')}`",
                f"- Step status counts: `{cache_refresh.get('step_status_counts', {})}`",
                f"- Readiness status: `{refresh_readiness.get('overall_status') or ''}`",
                f"- Readiness counts: `{refresh_rollups.get('candidate_readiness_counts') or {}}`",
                f"- Summary: `{cache_refresh.get('summary_path') or ''}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Sector Results",
            "",
            "| Sector | Execution | Verdict | Selected | Top Ranked | Audit | Framework Coverage | Company Autonomy | Cache | Actionable | Top Blocker | Artifact |",
            "|--------|-----------|---------|----------|------------|-------|--------------------|------------------|-------|------------|-------------|----------|",
        ]
    )
    for row in artifact.get("sector_results", []):
        blocker = (row.get("top_blockers") or row.get("evidence_gaps") or [""])[0]
        artifact_path = row.get("artifact_path") or ""
        cache = row.get("cache_coverage") if isinstance(row.get("cache_coverage"), dict) else {}
        lines.append(
            "| {sector} | {execution} | {verdict} | {selected} | {top_ranked} | {audit} | {framework_coverage} | {company_autonomy} | {cache} | {actionable} | {blocker} | {artifact_path} |".format(
                sector=row.get("sector") or "",
                execution=row.get("benchmark_execution_status") or "",
                verdict=row.get("final_verdict") or "",
                selected=row.get("selected_ticker") or "-",
                top_ranked=row.get("top_ranked_ticker") or "-",
                audit=row.get("selection_audit_status") or "-",
                framework_coverage=_fmt_ratio_pct(
                    row.get("framework_required_evidence_coverage_ratio")
                ),
                company_autonomy=row.get("company_autonomy_status") or "-",
                cache=cache.get("coverage_status") or "-",
                actionable=str(bool(row.get("actionable"))),
                blocker=str(blocker).replace("|", "\\|"),
                artifact_path=artifact_path,
            )
        )
    framework_coverage = rollups.get("framework_required_evidence_coverage") or {}
    lines.extend(
        [
            "",
            "## Framework Evidence Coverage",
            "",
            f"- Sectors with required evidence: `{framework_coverage.get('sectors_with_required_evidence', 0)}`",
            f"- Complete sectors: `{framework_coverage.get('complete_sector_count', 0)}`",
            f"- Incomplete sectors: `{framework_coverage.get('incomplete_sector_count', 0)}`",
            f"- Average coverage: `{_fmt_ratio_pct(framework_coverage.get('average_coverage_ratio'))}`",
            f"- Minimum coverage: `{_fmt_ratio_pct(framework_coverage.get('minimum_coverage_ratio'))}`",
            "",
        ]
    )
    for gap, count in (rollups.get("framework_required_evidence_gap_counts") or {}).items():
        lines.append(f"- `{gap}`: {count}")
    if not rollups.get("framework_required_evidence_gap_counts"):
        lines.append("- None")
    incomplete_framework_sectors = (
        rollups.get("framework_required_evidence_incomplete_sectors") or []
    )
    if incomplete_framework_sectors:
        lines.extend(
            [
                "",
                "| Sector | Selected/Focus | Coverage | Missing Framework Evidence |",
                "|--------|----------------|----------|----------------------------|",
            ]
        )
        for row in incomplete_framework_sectors:
            lines.append(
                "| {sector} | {ticker} | {coverage} | {missing} |".format(
                    sector=str(row.get("sector") or "").replace("|", "\\|"),
                    ticker=str(row.get("selected_ticker") or "-").replace("|", "\\|"),
                    coverage=_fmt_ratio_pct(row.get("coverage_ratio")),
                    missing=", ".join(str(item) for item in row.get("missing") or []).replace(
                        "|", "\\|"
                    )
                    or "-",
                )
            )
    lines.extend(
        [
            "",
            "## Framework Evidence Preflight",
            "",
        ]
    )
    preflight_coverage = rollups.get("framework_evidence_preflight_coverage") or {}
    lines.extend(
        [
            f"- Sectors with preflight: `{preflight_coverage.get('sectors_with_preflight', 0)}`",
            f"- Preflight candidates: `{preflight_coverage.get('preflight_candidate_count', 0)}`",
            f"- Packet-support present: `{preflight_coverage.get('packet_support_present_count', 0)}`",
            f"- Needs tool evidence: `{preflight_coverage.get('needs_tool_evidence_count', 0)}`",
            f"- Average packet support: `{_fmt_ratio_pct(preflight_coverage.get('average_packet_support_ratio'))}`",
            f"- Minimum packet support: `{_fmt_ratio_pct(preflight_coverage.get('minimum_packet_support_ratio'))}`",
            "",
        ]
    )
    for need, count in (rollups.get("framework_evidence_preflight_need_counts") or {}).items():
        lines.append(f"- `{need}`: {count}")
    if not rollups.get("framework_evidence_preflight_need_counts"):
        lines.append("- None")
    preflight_sectors = rollups.get("framework_evidence_preflight_incomplete_sectors") or []
    if preflight_sectors:
        lines.extend(
            [
                "",
                "| Sector | Avg Packet Support | Min Packet Support | Needs Tool Evidence |",
                "|--------|--------------------|--------------------|---------------------|",
            ]
        )
        for row in preflight_sectors:
            needs = row.get("needs") if isinstance(row.get("needs"), dict) else {}
            lines.append(
                "| {sector} | {avg} | {minimum} | {needs} |".format(
                    sector=str(row.get("sector") or "").replace("|", "\\|"),
                    avg=_fmt_ratio_pct(row.get("average_packet_support_ratio")),
                    minimum=_fmt_ratio_pct(row.get("minimum_packet_support_ratio")),
                    needs=", ".join(f"{key}: {value}" for key, value in needs.items()).replace(
                        "|", "\\|"
                    )
                    or "-",
                )
            )
    filter_sectors = rollups.get("framework_evidence_filter_excluded_sectors") or []
    lines.extend(
        [
            "",
            "## Framework Evidence Filter",
            "",
            f"- Filter statuses: `{rollups.get('framework_evidence_filter_status_counts', {})}`",
            f"- Excluded candidates: `{rollups.get('framework_evidence_filter_excluded_count', 0)}`",
        ]
    )
    if filter_sectors:
        lines.extend(
            [
                "",
                "| Sector | Status | Before | After | Excluded |",
                "|--------|--------|--------|-------|----------|",
            ]
        )
        for row in filter_sectors:
            lines.append(
                "| {sector} | {status} | {before} | {after} | {excluded} |".format(
                    sector=str(row.get("sector") or "").replace("|", "\\|"),
                    status=str(row.get("status") or "").replace("|", "\\|"),
                    before=", ".join(
                        str(item) for item in row.get("selected_tickers_before_filter") or []
                    ).replace("|", "\\|")
                    or "-",
                    after=", ".join(
                        str(item) for item in row.get("selected_tickers_after_filter") or []
                    ).replace("|", "\\|")
                    or "-",
                    excluded=", ".join(
                        str(item) for item in row.get("excluded_tickers") or []
                    ).replace("|", "\\|")
                    or "-",
                )
            )
    lines.extend(
        [
            "",
            "## Cache Readiness",
            "",
            "| Sector | Coverage | Ready | Partial | Not Usable | Execution Pool | Excluded | Warnings |",
            "|--------|----------|-------|---------|------------|----------------|----------|----------|",
        ]
    )
    for row in artifact.get("sector_results", []):
        cache = row.get("cache_coverage") if isinstance(row.get("cache_coverage"), dict) else {}
        lines.append(
            "| {sector} | {coverage} | {ready} | {partial} | {not_usable} | {pool} | {excluded} | {warnings} |".format(
                sector=str(row.get("sector") or "").replace("|", "\\|"),
                coverage=str(cache.get("coverage_status") or "-").replace("|", "\\|"),
                ready=", ".join(str(ticker) for ticker in cache.get("ready_tickers") or []).replace(
                    "|", "\\|"
                )
                or "-",
                partial=", ".join(
                    str(ticker) for ticker in cache.get("partial_tickers") or []
                ).replace("|", "\\|")
                or "-",
                not_usable=", ".join(
                    str(ticker) for ticker in cache.get("not_usable_tickers") or []
                ).replace("|", "\\|")
                or "-",
                pool=", ".join(
                    str(ticker)
                    for ticker in row.get("execution_candidate_tickers")
                    or cache.get("final_candidate_pool")
                    or []
                ).replace("|", "\\|")
                or "-",
                excluded=", ".join(
                    str(ticker) for ticker in cache.get("excluded_tickers") or []
                ).replace("|", "\\|")
                or "-",
                warnings=", ".join(
                    str(item)
                    for item in (
                        cache.get("cache_limited_reasons")
                        or cache.get("cache_readiness_warnings")
                        or []
                    )
                ).replace("|", "\\|")
                or "-",
            )
        )
    lines.extend(
        [
            "",
            "## Common Blockers",
            "",
        ]
    )
    for blocker, count in (rollups.get("top_blocker_counts") or {}).items():
        lines.append(f"- `{blocker}`: {count}")
    if not rollups.get("top_blocker_counts"):
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Common Tool Failures",
            "",
        ]
    )
    for failure, count in (rollups.get("tool_failure_counts") or {}).items():
        lines.append(f"- `{failure}`: {count}")
    if not rollups.get("tool_failure_counts"):
        lines.append("- None")
    tool_failure_sectors = rollups.get("tool_failure_sectors") or []
    if tool_failure_sectors:
        lines.extend(
            [
                "",
                "| Sector | Failed Tool Calls |",
                "|--------|-------------------|",
            ]
        )
        for row in tool_failure_sectors:
            failures = [
                f"{item.get('tool_name')}:{item.get('status')}"
                for item in row.get("failed_tool_calls") or []
                if isinstance(item, dict)
            ]
            lines.append(
                "| {sector} | {failures} |".format(
                    sector=str(row.get("sector") or "").replace("|", "\\|"),
                    failures=", ".join(failures).replace("|", "\\|") or "-",
                )
            )
    lines.extend(
        [
            "",
            "## Evidence Freshness Diagnostics",
            "",
            f"- Freshness buckets: `{rollups.get('source_freshness_bucket_counts', {})}`",
            f"- Source families: `{rollups.get('source_family_counts', {})}`",
            f"- Source reputation statuses: `{rollups.get('source_reputation_status_counts', {})}`",
            f"- Stale evidence count: `{rollups.get('stale_evidence_count', 0)}`",
            f"- Freshness issue count: `{rollups.get('freshness_issue_count', 0)}`",
            f"- Source reputation issue count: `{rollups.get('source_reputation_issue_count', 0)}`",
            "",
        ]
    )
    freshness_issue_sectors = rollups.get("freshness_issue_sectors") or []
    if freshness_issue_sectors:
        lines.extend(
            [
                "| Sector | Stale | Issues | Buckets | Example Sources |",
                "|--------|-------|--------|---------|-----------------|",
            ]
        )
        for row in freshness_issue_sectors:
            sources = [
                f"{item.get('source_label')}:{item.get('freshness_bucket')}"
                for item in row.get("freshness_issue_sources") or []
                if isinstance(item, dict)
            ]
            lines.append(
                "| {sector} | {stale} | {issues} | {buckets} | {sources} |".format(
                    sector=str(row.get("sector") or "").replace("|", "\\|"),
                    stale=int(row.get("stale_evidence_count") or 0),
                    issues=int(row.get("freshness_issue_count") or 0),
                    buckets=str(row.get("source_freshness_bucket_counts") or {}).replace(
                        "|", "\\|"
                    ),
                    sources=", ".join(sources).replace("|", "\\|") or "-",
                )
            )
    else:
        lines.append("- None")
    lines.extend(["", "### Source Reputation Issues", ""])
    reputation_issue_sectors = rollups.get("source_reputation_issue_sectors") or []
    if reputation_issue_sectors:
        lines.extend(
            [
                "| Sector | Missing Reputation | Statuses | Example Sources |",
                "|--------|--------------------|----------|-----------------|",
            ]
        )
        for row in reputation_issue_sectors:
            sources = [
                f"{item.get('source_label')}:{item.get('source_domain') or item.get('source_url') or 'unknown'}"
                for item in row.get("source_reputation_issue_sources") or []
                if isinstance(item, dict)
            ]
            lines.append(
                "| {sector} | {missing} | {statuses} | {sources} |".format(
                    sector=str(row.get("sector") or "").replace("|", "\\|"),
                    missing=int(row.get("source_reputation_issue_count") or 0),
                    statuses=str(row.get("source_reputation_status_counts") or {}).replace(
                        "|", "\\|"
                    ),
                    sources=", ".join(sources).replace("|", "\\|") or "-",
                )
            )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Evidence Gaps",
            "",
        ]
    )
    for gap, count in (rollups.get("evidence_gap_counts") or {}).items():
        lines.append(f"- `{gap}`: {count}")
    if not rollups.get("evidence_gap_counts"):
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Benchmark Recommendations",
            "",
        ]
    )
    for item in rollups.get("benchmark_recommendations") or []:
        if not isinstance(item, dict):
            continue
        lines.append(
            "- `{priority}` `{reason}` ({count}): {recommendation}".format(
                priority=item.get("priority") or "",
                reason=item.get("reason_code") or "",
                count=item.get("supporting_count", 0),
                recommendation=item.get("recommendation") or "",
            )
        )
    if not rollups.get("benchmark_recommendations"):
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def persist_autonomous_sector_benchmark(artifact: dict[str, Any]) -> AutonomousSectorBenchmarkPaths:
    """Write benchmark summary JSON and markdown report under data/outputs/runs."""

    cfg = get_config()
    ensure_directories(cfg)
    run_dir = cfg.runs_dir / "autonomous_sector_benchmark" / str(artifact["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "benchmark_summary.json"
    report_path = run_dir / "benchmark_report.md"
    cost_preflight_path: Path | None = None
    readiness_preflight_path: Path | None = None
    free_data_repair_path: Path | None = None
    summary_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(render_autonomous_sector_benchmark_report(artifact), encoding="utf-8")
    cost_preflight = artifact.get("all_sector_cost_preflight")
    if isinstance(cost_preflight, dict) and cost_preflight:
        cost_preflight_path = run_dir / "all_sector_cost_preflight.json"
        cost_preflight_path.write_text(
            json.dumps(cost_preflight, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    readiness_preflight = artifact.get("readiness_preflight")
    if isinstance(readiness_preflight, dict) and readiness_preflight:
        readiness_preflight_path = run_dir / "readiness_preflight.json"
        readiness_preflight_path.write_text(
            json.dumps(readiness_preflight, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    free_data_repair = artifact.get("free_data_repair")
    if isinstance(free_data_repair, dict) and free_data_repair:
        free_data_repair_path = run_dir / "accepted_census_free_data_repair.json"
        free_data_repair_path.write_text(
            json.dumps(free_data_repair, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return AutonomousSectorBenchmarkPaths(
        summary_json=summary_path,
        report_md=report_path,
        cost_preflight_json=cost_preflight_path,
        readiness_preflight_json=readiness_preflight_path,
        free_data_repair_json=free_data_repair_path,
    )


__all__ = [
    "AutonomousSectorBenchmarkPaths",
    "DEFAULT_BENCHMARK_SECTORS",
    "EXECUTION_MODE_FREE_DATA_REPAIR_ONLY",
    "EXECUTION_MODE_READINESS_PREFLIGHT_ONLY",
    "benchmark_compact_summary",
    "persist_autonomous_sector_benchmark",
    "render_autonomous_sector_benchmark_report",
    "run_autonomous_sector_benchmark",
]
