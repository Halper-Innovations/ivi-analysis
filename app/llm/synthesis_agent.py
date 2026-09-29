from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from app.analyst.ops_snapshot import (
    OpsAnalysisSnapshot,
    load_ops_analysis_snapshot,
    snapshot_to_synthesis_context,
)
from app.autonomous.financial_integrity import (
    INVALID_FINANCIAL_INPUT,
    NEEDS_DATA,
    FinancialIntegrityGateResult,
    FinancialIntegrityViolation,
    InvalidFinancialInputError,
    require_financial_integrity_scope,
)
from app.autonomous.v1_financial_context import (
    bind_v1_financial_scope,
    build_canonical_v1_financial_context,
    financial_input_scenario,
)
from app.config import get_config
from app.db import get_db, utc_now_iso
from app.evidence.enrichment import enrich_evidence_packet
from app.llm.providers import get_llm_provider
from app.llm.providers.retry_guard import llm_physical_attempt_guard
from app.llm.schemas import (
    SynthesisPacket,
    check_numeric_claim_trace,
    synthesis_schema_for_prompt,
    validate_synthesis_packet,
)
from app.logging import get_logger
from app.util.hashing import sha256_file, sha256_text
from app.util.issuer_classification import ISSUER_CLASS_FINANCIAL, resolve_issuer_classification
from app.valuation.lineage import latest_decision_eligible_valuation_row


logger = get_logger(__name__)

# Exact-match overrides for specific deployed models. Keyed lowercase.
_MODEL_PRICING_PER_1K = {
    "gpt-5-mini": (0.0010, 0.0030),
    # GPT-5.4 mini standard API rates: $0.75 input / $4.50 output per 1M
    # tokens, verified against the official OpenAI rate card on 2026-07-18.
    "gpt-5.4-mini": (0.00075, 0.00450),
    "gpt-4o-mini": (0.0003, 0.0012),
    "claude-haiku-4-5": (0.0010, 0.0050),
    # Full-size gpt-5.4 MUST have an exact entry: the "gpt-5" family fallback
    # below would bill it at mini rates (5x under on output) and break budget
    # enforcement. $2.50/$15 per 1M.
    "gpt-5.4": (0.0025, 0.0150),
    # GPT-5.5 standard API rates: $5 input / $30 output per 1M tokens.
    # Cached input is tracked separately below at $0.50 per 1M.
    "gpt-5.5": (0.0050, 0.0300),
    # Sonnet 5 sticker $3/$15 per 1M (intro discount through 2026-08-31 not
    # modeled — conservative overestimate is the safe direction for budgets).
    "claude-sonnet-5": (0.0030, 0.0150),
    # DeepSeek V4 Pro standard rates: $0.435 cache-miss input and $0.87
    # output per 1M tokens.  Thinking mode does not change the rate card.
    "deepseek-v4-pro": (0.000435, 0.00087),
}
# Family pricing (substring match on the lowercased model name), checked when
# no exact entry exists. Deployed models drift in their version suffix
# (gpt-5.4-mini, claude-haiku-4-5, ...), so we key on the family, not the exact
# string, to avoid silently falling back to a wildly different default rate.
_MODEL_FAMILY_PRICING_PER_1K: tuple[tuple[str, tuple[float, float]], ...] = (
    ("haiku", (0.0010, 0.0050)),  # ~ $1 / $5 per 1M
    ("sonnet", (0.0030, 0.0150)),  # ~ $3 / $15 per 1M
    ("opus", (0.0150, 0.0750)),  # ~ $15 / $75 per 1M
    ("gpt-5", (0.0010, 0.0030)),
    ("gpt-4o-mini", (0.0003, 0.0012)),
)
_DEFAULT_PRICING_PER_1K = (0.0015, 0.0040)
# Unknown Anthropic model: assume the cheapest tier rather than the most
# expensive (Opus) one, so an unrecognized model never inflates cost ~15x and
# trips a premature budget abort.
_ANTHROPIC_DEFAULT_PRICING_PER_1K = (0.0010, 0.0050)
_MODEL_CACHED_INPUT_PRICING_PER_1K = {
    # GPT-5.4 mini standard cached-input rate: $0.075 per 1M tokens.
    "gpt-5.4-mini": 0.000075,
    "gpt-5.5": 0.0005,
    # DeepSeek V4 Pro cache-hit input: $0.003625 per 1M tokens.
    "deepseek-v4-pro": 0.000003625,
}
_GPT_5_5_LONG_CONTEXT_THRESHOLD_INPUT_TOKENS = 272_000


def _pricing_per_1k(model: str, *, provider_name: str = "openai") -> tuple[float, float]:
    """Resolve (input_per_1k, output_per_1k) for a model by exact match, then family."""
    key = str(model or "").strip().lower()
    if key in _MODEL_PRICING_PER_1K:
        return _MODEL_PRICING_PER_1K[key]
    # Snapshot IDs retain the base model's pricing.  Resolve this before the
    # broad ``gpt-5`` family fallback, which intentionally reflects the older
    # mini-priced default and would materially undercount GPT-5.5 snapshots.
    if key.startswith("gpt-5.5-"):
        return _MODEL_PRICING_PER_1K["gpt-5.5"]
    for family, pricing in _MODEL_FAMILY_PRICING_PER_1K:
        if family in key:
            return pricing
    return (
        _ANTHROPIC_DEFAULT_PRICING_PER_1K
        if str(provider_name or "").strip().lower() == "anthropic"
        else _DEFAULT_PRICING_PER_1K
    )


def _cached_input_pricing_per_1k(
    model: str,
    *,
    provider_name: str = "openai",
) -> float:
    """Resolve cached-input pricing without changing the legacy 2-rate API."""

    key = str(model or "").strip().lower()
    if key in _MODEL_CACHED_INPUT_PRICING_PER_1K:
        return _MODEL_CACHED_INPUT_PRICING_PER_1K[key]
    if key.startswith("gpt-5.5-"):
        return _MODEL_CACHED_INPUT_PRICING_PER_1K["gpt-5.5"]
    # Unknown cached-input schedules are costed conservatively as ordinary
    # input instead of assuming a discount that may not exist.
    input_rate, _output_rate = _pricing_per_1k(
        model,
        provider_name=provider_name,
    )
    return input_rate


def _estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    provider_name: str = "openai",
    cached_input_tokens: int = 0,
) -> float:
    in_rate, out_rate = _pricing_per_1k(model, provider_name=provider_name)
    normalized_input_tokens = max(0, int(input_tokens))
    cached_tokens = min(max(0, int(cached_input_tokens)), normalized_input_tokens)
    uncached_tokens = normalized_input_tokens - cached_tokens
    cached_rate = _cached_input_pricing_per_1k(
        model,
        provider_name=provider_name,
    )
    model_key = str(model or "").strip().lower()
    if (
        str(provider_name or "").strip().lower() == "openai"
        and (model_key == "gpt-5.5" or model_key.startswith("gpt-5.5-"))
        and normalized_input_tokens > _GPT_5_5_LONG_CONTEXT_THRESHOLD_INPUT_TOKENS
    ):
        # OpenAI prices the full GPT-5.5 session at 2x input (including cached
        # input) and 1.5x output once the prompt exceeds 272K input tokens.
        in_rate *= 2.0
        cached_rate *= 2.0
        out_rate *= 1.5
    return round(
        ((uncached_tokens / 1000.0) * in_rate)
        + ((cached_tokens / 1000.0) * cached_rate)
        + ((output_tokens / 1000.0) * out_rate),
        6,
    )


def _estimate_tokens_from_text(text: str) -> int:
    return max(1, len(text) // 4)


def _run_spend_usd(conn, run_id: str) -> float:
    row = conn.execute(
        """
        SELECT
            (
                SELECT COALESCE(SUM(cost_estimate_usd), 0)
                FROM synthesis_paid_attempts
                WHERE run_id = ?
            )
            +
            (
                SELECT COALESCE(SUM(cost_estimate_usd), 0)
                FROM synthesis_packets
                WHERE run_id = ?
                  AND paid_invocation_id IS NULL
            ) AS spent
        """,
        (run_id, run_id),
    ).fetchone()
    return float(row["spent"] or 0.0)


def _configured_provider_model(cfg: Any, provider_name: str) -> str:
    if provider_name == "anthropic":
        return cfg.anthropic_model
    if provider_name == "deepseek":
        return cfg.deepseek_model
    return cfg.openai_model


def _provider_max_output_tokens(cfg: Any, provider_name: str) -> int:
    if provider_name == "anthropic":
        return int(cfg.anthropic_max_output_tokens)
    if provider_name == "deepseek":
        return int(cfg.deepseek_max_output_tokens)
    return int(cfg.openai_max_output_tokens)


def _provider_budget_usd(cfg: Any, provider_name: str) -> float:
    if provider_name == "anthropic":
        return float(cfg.anthropic_budget_usd)
    if provider_name == "deepseek":
        return float(cfg.deepseek_budget_usd_per_run)
    return float(cfg.openai_budget_usd_per_run)


def _provider_enablement_hint(provider_name: str) -> str:
    if provider_name == "openai":
        return "Set VOE_OPENAI_API_KEY or switch VOE_LLM_PROVIDER=disabled."
    if provider_name == "anthropic":
        return "Set VOE_ANTHROPIC_API_KEY or switch VOE_LLM_PROVIDER=disabled."
    if provider_name == "deepseek":
        return "Set VOE_DEEPSEEK_API_KEY or switch VOE_LLM_PROVIDER=disabled."
    return "Set the provider API key or switch VOE_LLM_PROVIDER=disabled."


def _latest_evidence_packet(conn, ticker: str, as_of_date: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT as_of_date, packet_path
        FROM evidence_packets
        WHERE ticker = ? AND as_of_date <= ?
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (ticker, as_of_date),
    ).fetchone()
    if not row:
        return None
    path = Path(row["packet_path"])
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    payload["as_of_date"] = row["as_of_date"]
    payload["_packet_path"] = str(path)
    return payload


def _is_iso_date_at_or_before(value: Any, cutoff: str) -> bool:
    from datetime import date

    try:
        observed = date.fromisoformat(str(value or "").strip()[:10])
        boundary = date.fromisoformat(str(cutoff or "").strip()[:10])
    except ValueError:
        return False
    return observed <= boundary


def _prompt_evidence_violation(
    violations: list[FinancialIntegrityViolation],
    *,
    code: str,
    ticker: str,
    field: str,
    observed: Any,
    expected: Any,
    reason: str,
    terminal_status: str = INVALID_FINANCIAL_INPUT,
) -> None:
    violations.append(
        FinancialIntegrityViolation(
            code=code,
            ticker=ticker,
            field=field,
            source_values={field: observed},
            expected_relationship=expected,
            observed_relationship=observed,
            reason=reason,
            terminal_status=terminal_status,
        )
    )


def _validate_prompt_provenance_record(
    record: Any,
    *,
    ticker: str,
    run_as_of_date: str,
    field: str,
    violations: list[FinancialIntegrityViolation],
    expected_value: Any = None,
) -> None:
    if not isinstance(record, dict):
        _prompt_evidence_violation(
            violations,
            code="PROMPT_FINANCIAL_PROVENANCE_MISSING",
            ticker=ticker,
            field=field,
            observed=record,
            expected=("value, unit, source, period_end, filed_date, and source_reference"),
            reason=(
                "Investment-relevant prompt evidence derived from "
                "companyfacts requires exact provenance."
            ),
            terminal_status=NEEDS_DATA,
        )
        return

    required = (
        "value",
        "unit",
        "source",
        "period_end",
        "filed_date",
        "source_reference",
    )
    missing = [
        name
        for name in required
        if record.get(name) is None
        or (isinstance(record.get(name), str) and not record.get(name).strip())
    ]
    value = record.get("value")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        if "value" not in missing:
            missing.append("value")
    if missing:
        _prompt_evidence_violation(
            violations,
            code="PROMPT_FINANCIAL_PROVENANCE_INCOMPLETE",
            ticker=ticker,
            field=field,
            observed={"missing_fields": sorted(set(missing))},
            expected=list(required),
            reason=("Investment-relevant prompt financial provenance is incomplete."),
            terminal_status=NEEDS_DATA,
        )
        return

    period_end = str(record["period_end"]).strip()[:10]
    filed_date = str(record["filed_date"]).strip()[:10]
    if (
        not _is_iso_date_at_or_before(period_end, run_as_of_date)
        or not _is_iso_date_at_or_before(filed_date, run_as_of_date)
        or period_end > filed_date
    ):
        _prompt_evidence_violation(
            violations,
            code="PROMPT_FINANCIAL_PROVENANCE_ASOF_INVALID",
            ticker=ticker,
            field=field,
            observed={
                "period_end": period_end,
                "filed_date": filed_date,
                "run_as_of_date": run_as_of_date,
            },
            expected=("period_end <= filed_date <= run_as_of_date"),
            reason=(
                "Prompt financial evidence must have been filed and visible "
                "on or before the effective as-of date."
            ),
        )
    if (
        expected_value is not None
        and not isinstance(expected_value, bool)
        and isinstance(expected_value, (int, float))
        and not math.isclose(
            float(value),
            float(expected_value),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        _prompt_evidence_violation(
            violations,
            code="PROMPT_FINANCIAL_PROVENANCE_VALUE_MISMATCH",
            ticker=ticker,
            field=field,
            observed={
                "prompt_value": expected_value,
                "provenance_value": value,
            },
            expected="prompt value == provenance value",
            reason=("Prompt financial evidence must reconcile exactly to its provenance record."),
        )

    input_provenance = record.get("input_provenance")
    is_derived = str(record.get("source") or "").startswith("derived:")
    if is_derived and not str(record.get("formula") or "").strip():
        _prompt_evidence_violation(
            violations,
            code="PROMPT_DERIVED_PROVENANCE_FORMULA_MISSING",
            ticker=ticker,
            field=f"{field}.formula",
            observed=record.get("formula"),
            expected="non-empty deterministic formula",
            reason=("Derived prompt financial evidence requires its exact deterministic formula."),
            terminal_status=NEEDS_DATA,
        )
    if is_derived and input_provenance is None:
        _prompt_evidence_violation(
            violations,
            code="PROMPT_DERIVED_PROVENANCE_INPUTS_MISSING",
            ticker=ticker,
            field=f"{field}.input_provenance",
            observed=None,
            expected="non-empty provenance mapping",
            reason=(
                "Derived prompt financial evidence requires provenance for every formula input."
            ),
            terminal_status=NEEDS_DATA,
        )
    if input_provenance is not None:
        if not isinstance(input_provenance, dict) or not input_provenance:
            _prompt_evidence_violation(
                violations,
                code="PROMPT_DERIVED_PROVENANCE_INPUTS_MISSING",
                ticker=ticker,
                field=f"{field}.input_provenance",
                observed=input_provenance,
                expected="non-empty provenance mapping",
                reason=(
                    "Derived prompt financial evidence requires provenance for every formula input."
                ),
                terminal_status=NEEDS_DATA,
            )
        else:
            for input_name, input_record in input_provenance.items():
                _validate_prompt_provenance_record(
                    input_record,
                    ticker=ticker,
                    run_as_of_date=run_as_of_date,
                    field=f"{field}.input_provenance.{input_name}",
                    violations=violations,
                )


def _contains_companyfacts_reference(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return any(str(item).strip().startswith("companyfacts_facts.") for item in value)


def _require_prompt_evidence_financial_integrity(
    *,
    ticker: str,
    run_as_of_date: str,
    evidence_packet: dict[str, Any],
) -> None:
    """Validate point-in-time provenance before cache, fallback, or spend."""

    violations: list[FinancialIntegrityViolation] = []

    for index, filing in enumerate(evidence_packet.get("filings_used") or []):
        if not isinstance(filing, dict):
            continue
        filing_date = filing.get("filing_date")
        if filing_date and not _is_iso_date_at_or_before(
            filing_date,
            run_as_of_date,
        ):
            _prompt_evidence_violation(
                violations,
                code="PROMPT_FILING_ASOF_INVALID",
                ticker=ticker,
                field=f"evidence_packet.filings_used[{index}].filing_date",
                observed=filing_date,
                expected=f"valid ISO date <= {run_as_of_date}",
                reason=(
                    "A filing used in the synthesis prompt was not visible "
                    "at the effective as-of date."
                ),
            )

    fundamentals = evidence_packet.get("fundamentals")
    fundamentals = fundamentals if isinstance(fundamentals, dict) else {}
    provenance_by_metric = evidence_packet.get("fundamentals_provenance")
    if provenance_by_metric is not None and not isinstance(
        provenance_by_metric,
        dict,
    ):
        _prompt_evidence_violation(
            violations,
            code="PROMPT_FUNDAMENTALS_PROVENANCE_INVALID",
            ticker=ticker,
            field="evidence_packet.fundamentals_provenance",
            observed=provenance_by_metric,
            expected="mapping keyed by financial metric",
            reason="Fundamentals provenance must be a mapping.",
            terminal_status=NEEDS_DATA,
        )
    elif isinstance(provenance_by_metric, dict):
        numeric_fundamentals = {
            metric: value
            for metric, value in fundamentals.items()
            if not isinstance(value, bool) and isinstance(value, (int, float))
        }
        for metric, value in numeric_fundamentals.items():
            _validate_prompt_provenance_record(
                provenance_by_metric.get(metric),
                ticker=ticker,
                run_as_of_date=run_as_of_date,
                field=f"evidence_packet.fundamentals_provenance.{metric}",
                violations=violations,
                expected_value=value,
            )
        for metric in set(provenance_by_metric) - set(numeric_fundamentals):
            _validate_prompt_provenance_record(
                provenance_by_metric[metric],
                ticker=ticker,
                run_as_of_date=run_as_of_date,
                field=f"evidence_packet.fundamentals_provenance.{metric}",
                violations=violations,
                expected_value=fundamentals.get(metric),
            )
    elif any(
        not isinstance(value, bool) and isinstance(value, (int, float))
        for value in fundamentals.values()
    ):
        for metric, value in fundamentals.items():
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                continue
            _validate_prompt_provenance_record(
                None,
                ticker=ticker,
                run_as_of_date=run_as_of_date,
                field=f"evidence_packet.fundamentals_provenance.{metric}",
                violations=violations,
                expected_value=value,
            )

    for index, fact in enumerate(evidence_packet.get("extracted_facts") or []):
        if not isinstance(fact, dict):
            continue
        fact_value = fact.get("value")
        if not isinstance(fact_value, dict):
            continue
        numeric_value = fact_value.get("value")
        requires_provenance = (
            not isinstance(numeric_value, bool) and isinstance(numeric_value, (int, float))
        ) or _contains_companyfacts_reference(fact_value.get("derived_from"))
        if not requires_provenance:
            continue
        _validate_prompt_provenance_record(
            fact.get("provenance"),
            ticker=ticker,
            run_as_of_date=run_as_of_date,
            field=f"evidence_packet.extracted_facts[{index}].provenance",
            violations=violations,
            expected_value=numeric_value,
        )

    for index, row in enumerate(evidence_packet.get("financials") or []):
        if not isinstance(row, dict):
            continue
        numeric_value = row.get("value")
        if isinstance(numeric_value, bool) or not isinstance(
            numeric_value,
            (int, float),
        ):
            continue
        _validate_prompt_provenance_record(
            row.get("provenance"),
            ticker=ticker,
            run_as_of_date=run_as_of_date,
            field=f"evidence_packet.financials[{index}].provenance",
            violations=violations,
            expected_value=numeric_value,
        )

    row_traces = fundamentals.get("row_traces")
    rows_by_year = {
        str(row.get("year") or ""): row
        for row in (fundamentals.get("rows") or [])
        if isinstance(row, dict) and str(row.get("year") or "")
    }
    if isinstance(row_traces, dict):
        for year, traces in row_traces.items():
            if not isinstance(traces, dict):
                continue
            for metric, trace in traces.items():
                if not isinstance(trace, dict) or not _contains_companyfacts_reference(
                    trace.get("derived_from")
                ):
                    continue
                row_value = (rows_by_year.get(str(year)) or {}).get(metric)
                if (
                    isinstance(row_value, bool)
                    or not isinstance(row_value, (int, float))
                    or not math.isfinite(float(row_value))
                ):
                    continue
                _validate_prompt_provenance_record(
                    trace.get("provenance"),
                    ticker=ticker,
                    run_as_of_date=run_as_of_date,
                    field=(f"evidence_packet.fundamentals.row_traces.{year}.{metric}.provenance"),
                    violations=violations,
                    expected_value=row_value,
                )

    derived_signals = fundamentals.get("derived_signals")
    if isinstance(derived_signals, dict):
        for signal, payload in derived_signals.items():
            if not isinstance(payload, dict) or not _contains_companyfacts_reference(
                payload.get("derived_from")
            ):
                continue
            signal_value = payload.get("value")
            if (
                isinstance(signal_value, bool)
                or not isinstance(signal_value, (int, float))
                or not math.isfinite(float(signal_value))
            ):
                continue
            _validate_prompt_provenance_record(
                payload.get("provenance"),
                ticker=ticker,
                run_as_of_date=run_as_of_date,
                field=(f"evidence_packet.fundamentals.derived_signals.{signal}.provenance"),
                violations=violations,
                expected_value=signal_value,
            )

    if not violations:
        return
    status = (
        INVALID_FINANCIAL_INPUT
        if any(violation.terminal_status == INVALID_FINANCIAL_INPUT for violation in violations)
        else NEEDS_DATA
    )
    raise InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context=f"synthesis_prompt_evidence:{ticker}",
            run_as_of_date=run_as_of_date,
            status=status,
            violations=tuple(violations),
        )
    )


def _require_prompt_point_in_time_dates(
    *,
    ticker: str,
    run_as_of_date: str,
    payload: Any,
) -> None:
    """Reject future-dated evidence anywhere in the exact provider input."""

    violations: list[FinancialIntegrityViolation] = []
    date_fields = {
        "filed_date",
        "filing_date",
        "source_date",
        "period_end",
    }

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                item_path = f"{path}.{key}" if path else str(key)
                if (
                    key in date_fields
                    and item is not None
                    and str(item).strip()
                    and not _is_iso_date_at_or_before(
                        item,
                        run_as_of_date,
                    )
                ):
                    _prompt_evidence_violation(
                        violations,
                        code="PROMPT_EVIDENCE_ASOF_INVALID",
                        ticker=ticker,
                        field=item_path,
                        observed=item,
                        expected=f"valid ISO date <= {run_as_of_date}",
                        reason=(
                            "Every dated source exposed to synthesis must be "
                            "visible on or before the effective as-of date."
                        ),
                    )
                visit(item, item_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(payload, "synthesis_inputs")
    if violations:
        raise InvalidFinancialInputError(
            FinancialIntegrityGateResult(
                context=f"synthesis_prompt_dates:{ticker}",
                run_as_of_date=run_as_of_date,
                status=INVALID_FINANCIAL_INPUT,
                violations=tuple(violations),
            )
        )


def _analysis_context_needs_refresh(
    *,
    evidence_packet: dict[str, Any],
    analysis_snapshot: OpsAnalysisSnapshot | None,
    run_id: str,
) -> bool:
    if analysis_snapshot is None:
        return True
    if analysis_snapshot.analysis_source == "analysis_report":
        return False
    payload = analysis_snapshot.legacy_research_payload
    if not isinstance(payload, dict):
        return True
    if str(payload.get("run_id") or "").strip() != str(run_id).strip():
        return True
    evidence_as_of = str(evidence_packet.get("as_of_date") or "").strip()
    context_as_of = str(payload.get("as_of_date") or "").strip()
    if evidence_as_of and context_as_of and context_as_of < evidence_as_of:
        return True
    evidence_path = Path(str(evidence_packet.get("_packet_path") or ""))
    legacy_path = analysis_snapshot.legacy_research_path
    try:
        if evidence_path.exists() and legacy_path is not None and legacy_path.exists():
            if legacy_path.stat().st_mtime < evidence_path.stat().st_mtime:
                return True
    except Exception:
        return False
    return False


def _refresh_analysis_context(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
) -> OpsAnalysisSnapshot | None:
    try:
        from app.research import run_research_agent_for_ticker
    except Exception:
        return None
    try:
        run_research_agent_for_ticker(
            ticker=ticker,
            as_of_date=as_of_date,
            run_id=run_id,
            build_legacy_packet=True,
        )
    except Exception:
        return None
    return load_ops_analysis_snapshot(ticker, as_of_date=as_of_date, run_id=run_id)


def _latest_delta(conn, ticker: str, run_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT delta_path
        FROM ticker_deltas
        WHERE ticker = ? AND run_id = ?
        LIMIT 1
        """,
        (ticker, run_id),
    ).fetchone()
    if not row:
        return None
    path = Path(row["delta_path"])
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _valuation_summary(packet: dict[str, Any]) -> dict[str, Any]:
    valuations = packet.get("valuations", {})
    if not isinstance(valuations, dict):
        return {}

    def _method_payload(name: str) -> dict[str, Any]:
        payload = valuations.get(name, {})
        if not isinstance(payload, dict):
            return {}
        outputs = payload.get("outputs")
        if isinstance(outputs, dict):
            normalized = dict(outputs)
            warnings = payload.get("warnings")
            if isinstance(warnings, list) and warnings and "warnings" not in normalized:
                normalized["warnings"] = warnings
            inputs = payload.get("inputs")
            if isinstance(inputs, dict) and inputs:
                normalized["inputs"] = inputs
            return normalized
        return dict(payload)

    summary = {
        "owner_earnings": _method_payload("owner_earnings"),
        "dcf": _method_payload("dcf"),
        "dcf_adjusted": _method_payload("dcf_adjusted"),
        "epv": _method_payload("epv"),
        "epv_adjusted": _method_payload("epv_adjusted"),
        "graham": _method_payload("graham"),
        "ncav": _method_payload("ncav"),
        "scorecard": _method_payload("scorecard"),
        "roic": _method_payload("roic"),
        "capital_structure": _method_payload("capital_structure"),
        "reverse_dcf": _method_payload("reverse_dcf"),
        "tech_adjustment": _method_payload("tech_adjustment"),
    }

    legacy_dcf = valuations.get("dcf_lite")
    if not summary["dcf"] and isinstance(legacy_dcf, dict):
        legacy_outputs = (
            legacy_dcf.get("outputs") if isinstance(legacy_dcf.get("outputs"), dict) else {}
        )
        per_share_range = (
            legacy_outputs.get("per_share_range")
            if isinstance(legacy_outputs.get("per_share_range"), dict)
            else {}
        )
        summary["dcf"] = {
            "status": str(legacy_outputs.get("status") or "LEGACY_COMPAT"),
            "low": per_share_range.get("low"),
            "base": per_share_range.get("base"),
            "high": per_share_range.get("high"),
        }

    legacy_reverse = valuations.get("reverse_dcf")
    if not summary["reverse_dcf"] and isinstance(legacy_reverse, dict):
        summary["reverse_dcf"] = _method_payload("reverse_dcf")

    legacy_multiples = valuations.get("multiples")
    if isinstance(legacy_multiples, dict):
        summary["legacy_multiples"] = _method_payload("multiples")

    legacy_warnings = valuations.get("sanity_checks")
    if isinstance(legacy_warnings, list) and legacy_warnings:
        summary["legacy_sanity_checks"] = legacy_warnings

    return {key: value for key, value in summary.items() if value not in ({}, [], None)}


def _dossier_focus(packet: dict[str, Any]) -> dict[str, Any]:
    fundamentals = packet.get("fundamentals", {})
    if not isinstance(fundamentals, dict):
        fundamentals = {}
    return {
        "reinvestment": {
            "r_and_d_total": fundamentals.get("r_and_d_total"),
            "sales_marketing_total": fundamentals.get("sales_marketing_total"),
            "g_and_a_total": fundamentals.get("g_and_a_total"),
        },
        "revenue_visibility": {
            "deferred_revenue_amount": fundamentals.get("deferred_revenue_amount"),
            "deferred_revenue_to_revenue_latest": fundamentals.get(
                "deferred_revenue_to_revenue_latest"
            ),
            "rpo_amount": fundamentals.get("rpo_amount"),
            "rpo_to_revenue_latest": fundamentals.get("rpo_to_revenue_latest"),
        },
        "capital_allocation": {
            "share_repurchases_amount": fundamentals.get("share_repurchases_amount"),
            "dividends_paid_amount": fundamentals.get("dividends_paid_amount"),
        },
        "concentration_and_segments": {
            "customer_concentration_present": fundamentals.get("customer_concentration_present"),
            "customer_concentration_pct": fundamentals.get("customer_concentration_pct"),
            "segment_count": fundamentals.get("segment_count"),
        },
        "trend_signals": {
            "revenue_cagr_3y": fundamentals.get("revenue_cagr_3y"),
            "revenue_cagr_5y": fundamentals.get("revenue_cagr_5y"),
            "gross_margin_trend_slope": fundamentals.get("gross_margin_trend_slope"),
            "operating_margin_trend_slope": fundamentals.get("operating_margin_trend_slope"),
            "fcf_margin_trend_slope": fundamentals.get("fcf_margin_trend_slope"),
            "r_and_d_intensity_latest": fundamentals.get("r_and_d_intensity_latest"),
            "r_and_d_intensity_delta": fundamentals.get("r_and_d_intensity_delta"),
            "segment_count_delta": fundamentals.get("segment_count_delta"),
            "customer_concentration_delta": fundamentals.get("customer_concentration_delta"),
            "dilution_rate_shares_cagr": fundamentals.get("dilution_rate_shares_cagr"),
        },
    }


def _compact_evidence(packet: dict[str, Any]) -> dict[str, Any]:
    enriched_packet = enrich_evidence_packet(packet)
    extracted = packet.get("extracted_facts", [])
    extracted_trim = extracted[:80] if isinstance(extracted, list) else []
    financials = packet.get("financials", [])
    financials_trim = financials[:120] if isinstance(financials, list) else []
    variant_report = (
        packet.get("variant_perceptions")
        if isinstance(packet.get("variant_perceptions"), dict)
        else {}
    )
    variant_perceptions = (
        variant_report.get("perceptions")
        if isinstance(variant_report.get("perceptions"), list)
        else []
    )
    return {
        "ticker": packet.get("ticker"),
        "as_of_date": packet.get("as_of_date"),
        "filings_used": packet.get("filings_used", []),
        "fundamentals": packet.get("fundamentals", {}),
        "fundamentals_provenance": packet.get(
            "fundamentals_provenance",
            {},
        ),
        "dossier_focus": _dossier_focus(packet),
        "valuations": _valuation_summary(packet),
        "variant_perceptions": {
            "signal_summary": variant_report.get("signal_summary")
            if isinstance(variant_report.get("signal_summary"), dict)
            else {},
            "data_quality": variant_report.get("data_quality")
            if isinstance(variant_report.get("data_quality"), dict)
            else {},
            "perceptions": [
                {
                    "perception_id": row.get("perception_id"),
                    "thesis": row.get("thesis"),
                    "direction": row.get("direction"),
                    "confidence": row.get("confidence"),
                    "implied_vs_estimated": row.get("implied_vs_estimated"),
                    "supporting_signals": (row.get("supporting_signals") or [])[:4]
                    if isinstance(row, dict)
                    else [],
                    "contradicting_signals": (row.get("contradicting_signals") or [])[:3]
                    if isinstance(row, dict)
                    else [],
                    "testable_prediction": row.get("testable_prediction"),
                    "time_horizon": row.get("time_horizon"),
                    "catalyst": row.get("catalyst"),
                    "risk": row.get("risk"),
                }
                for row in variant_perceptions[:3]
                if isinstance(row, dict)
            ],
        },
        "enrichment": enriched_packet.get("enrichment", {}),
        "deltas_vs_prior_period": packet.get("deltas_vs_prior_period", {}),
        "extracted_facts": extracted_trim,
        "financials": financials_trim,
    }


def _build_inputs(
    *,
    ticker: str,
    as_of_date: str,
    evidence_packet: dict[str, Any],
    canonical_financial_context: dict[str, Any],
    analysis_context: dict[str, Any] | None,
    delta_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "canonical_financial_context": canonical_financial_context,
        "evidence_packet": _compact_evidence(evidence_packet),
        "analysis_context": analysis_context or {"status": "MISSING"},
        "delta_context": delta_payload or {},
    }


def _build_prompt(inputs: dict[str, Any], schema: dict[str, Any]) -> str:
    instructions = """
You are the Valuation Engine Synthesis Agent.
Output ONLY valid JSON that matches the given JSON schema exactly.
Do not include markdown, prose wrappers, or comments.

Hard constraints:
1) Use ONLY facts from the provided inputs.
2) Do not invent data, dates, values, or sources.
3) Every outward-facing factual claim must be citeable or derived.
4) For claims with type="numeric", citations OR derived_from is mandatory.
5) If evidence is missing, state uncertainty in non-numeric form and add next_actions.
6) Keep ticker and as_of_date exactly as provided.
7) Fill the Layer 3 qualitative fields directly: business_quality_summary,
   valuation_interpretation, risk_frame, catalyst_frame, evidence_gaps,
   recommended_next_actions, confidence_notes.
8) Use canonical_financial_context as the sole authority for price, shares, market cap,
   split basis, quote identity, and related formula traces. Prefer canonical references
   rooted at canonical_financial_context.*, evidence_packet.*, or analysis_context.* in
   derived_from.
9) Prefer canonical valuation methods (dcf, epv, graham, ncav, scorecard, reverse_dcf).
   Mention legacy methods only if canonical methods are absent from the input.
10) Round user-facing numbers so they read naturally:
    - money/share values: usually 2 decimals max
    - percentages: usually 1 decimal max
    - avoid long machine-like decimals
11) Do not mention fallback extraction mechanics like "cover page extraction" unless the
    evidence clearly requires that distinction.
12) When present, prefer deterministic dossier facts for reinvestment and revenue quality:
    r_and_d_total, deferred_revenue_amount, rpo_amount, customer_concentration_pct,
    segment_count, share_repurchases_amount, dividends_paid_amount.
13) Use evidence_packet.dossier_focus as a first-pass map for what to discuss:
    - reinvestment: R&D / S&M / G&A and operating spend intensity
    - revenue_visibility: deferred revenue and RPO
    - capital_allocation: repurchases and dividends
    - concentration_and_segments: customer concentration and segment breadth
14) If customer_concentration_present = 0 and customer_concentration_pct is UNKNOWN,
    treat that as explicit evidence that no major customer exceeded the disclosed threshold,
    not as missing evidence.
15) In business_quality_summary and valuation_interpretation, explicitly use these fact families
    when they are present instead of defaulting to generic statements about quality or cash flow.
16) For reinvestment, prefer concrete interpretations such as:
    - whether R&D intensity appears meaningful or modest relative to revenue
    - whether S&M and G&A suggest scaled operating leverage or heavy commercial overhead
    - whether the operating expense mix supports or constrains durable margins
17) For revenue visibility, explicitly interpret deferred revenue and RPO as visibility signals
    when present. Explain whether they support durability, backlog, or forward revenue coverage,
    and avoid generic "recurring revenue" language unless the evidence supports it.
18) For capital allocation, explicitly mention repurchases and dividends when present and connect
    them to shareholder return, maturity, or capital discipline. Do not omit them in favor of a
    generic free-cash-flow comment if the factual amounts are available.
19) For concentration and segments, explicitly mention segment breadth when segment_count is present.
    If customer concentration is absent by disclosure, say that the filing did not identify a major
    customer above the stated threshold. Distinguish that from true missing evidence.
20) In valuation_interpretation, connect the deterministic dossier facts to the valuation frame:
    - reinvestment and margin structure should inform quality/durability judgments
    - deferred revenue / RPO should inform visibility judgments
    - repurchases / dividends should inform capital-allocation quality
    - customer concentration / segment breadth should inform risk and resilience
20b) For banks, insurers, brokers, and other financial institutions, do not force a software-style
    DCF framing if the packet is balance-sheet heavy. Instead discuss deposits, loans, securities,
    funding mix, capital returns, credit quality, and regulatory or liquidity constraints, while
    stating clearly which valuation inputs are still missing.
20c) For financial institutions, do not treat missing FCF as a generic defect by default. If cash
    flow data is noisy or sector-limited, say that traditional FCF-style framing is less informative
    here and prioritize operating profitability, capital returns, funding mix, regulatory capital,
    and balance-sheet quality.
21) Use evidence_packet.dossier_focus.trend_signals when available to say whether the business is
    improving, stable, or weakening across recent years. Prefer trend-aware language over one-year
    snapshots when the evidence supports it.
21b) When evidence_packet.variant_perceptions.perceptions is present, treat those as deterministic,
    pre-built thesis candidates. Use the strongest one to anchor valuation_interpretation or
    catalyst_frame when relevant, and preserve its stated support, contradiction, catalyst, risk,
    and testable prediction instead of inventing a disconnected thesis.
22) Writing style should read like a concise investor memo, not a checklist. Favor 2-4 clean
    sentences per major section instead of long parenthetical chains or numbered lists inside prose.
23) Keep citations and derived references in claims and supporting logic, but do not overload the
    narrative fields with repetitive path-like citations unless they materially improve clarity.
24) When trend data is mixed, say so plainly. Avoid overconfident language if, for example, margins
    are improving but free-cash-flow conversion or dilution trends are less favorable.
25) When evidence_packet.enrichment.quality_assessment is present, use it as follows:
    - If gate_verdict is BLOCK, your stance MUST be "avoid" and explain why the valuation
      framework does not apply (e.g., financial issuer, severe secular decline).
    - If signal_context is VALUE_TRAP_RISK, your valuation_interpretation MUST discuss
      why the apparent discount may be a trap (weak moat, declining business).
    - If signal_context is PREMIUM_JUSTIFIED, acknowledge in valuation_interpretation
      that the premium may be warranted by moat strength, rather than defaulting to "overvalued."
    - If downside_risk_class is SEVERE, your risk_frame MUST include the bear-case intrinsic
      value and the assumptions behind it.
26) Surface valuation_headwinds and valuation_supports in your analysis:
    - Headwinds should inform risk_frame and temper bullish valuation_interpretation.
    - Supports should inform business_quality_summary and reinforce conviction where warranted.
    - Do not simply list them — interpret what they mean for the investment thesis.
27) Use nonrecurring_flags, sbc_flags, and depreciation_flags to add specificity:
    - If RESTRUCTURING_CHARGE_DETECTED, note that reported earnings may understate normalized
      earning power.
    - If SBC_ACCELERATING, discuss whether SBC growth is proportional to value creation.
    - If CAPEX_BELOW_DEPRECIATION, flag potential underinvestment risk.
28) Incorporate moat_class and confidence_class into confidence_notes:
    - State the moat classification and what signals drove it.
    - State the confidence classification and any specific gate_reason_codes.
29) When evidence_packet.enrichment.filing_intelligence is present, use it:
    - If high_materiality_changes exist, these are year-over-year shifts in 10-K language
      (MD&A, Risk Factors) that may signal material business changes not yet reflected in
      financials. Discuss the most significant changes in your valuation_interpretation.
    - Change types include: NEW_DISCLOSURE, REMOVED_DISCLOSURE, LANGUAGE_SHIFT,
      COMPETITIVE_SIGNAL, RISK_SIGNAL, STRATEGIC_SIGNAL. Interpret what each means.
    - If a HIGH materiality filing change contradicts the quality gate verdict, flag the
      contradiction explicitly — this is where alpha lives.
30) When confirmed_patterns are present from the cross-sectional pattern scanner, cite them:
    - These are empirically validated patterns (e.g., "capex-to-depreciation divergence predicted
      margin expansion in X% of historical cases"). State the pattern, hit rate, and what it
      implies for this company's forward trajectory.
    - Confirmed patterns should strengthen or weaken your variant thesis, not just be listed.
"""
    _ = schema
    return instructions.strip() + "\n\nINPUT_PAYLOAD:\n" + json.dumps(inputs, sort_keys=True)


_LONG_FLOAT_RE = re.compile(r"(-?\d+\.\d{3,})(%?)")
_ANALYSIS_ALIAS_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("research_packet.quality.top_gaps", "analysis_context.research_quality.top_gaps"),
    ("research_packet.quality", "analysis_context.research_quality"),
    ("research_packet.evidence_items.", "analysis_context.citations."),
    ("research_packet.evidence_items", "analysis_context.citations"),
    ("research_packet.catalysts.", "analysis_context.recent_event_impacts."),
    ("research_packet.catalysts", "analysis_context.recent_event_impacts"),
    ("research_packet.findings.", "analysis_context.positives."),
    ("research_packet.findings", "analysis_context.positives"),
    ("research_packet.risks.", "analysis_context.risks."),
    ("research_packet.risks", "analysis_context.risks"),
    ("research_packet.next_actions.", "analysis_context.next_actions."),
    ("research_packet.next_actions", "analysis_context.next_actions"),
    ("research_packet.evidence_gaps", "analysis_context.open_questions"),
    ("research_packet.signals.", "analysis_context.research_quality."),
    ("research_packet.signals", "analysis_context.research_quality"),
)
_TEXT_PATH_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("analysis_context.next_actions.", "analysis_context.next_actions "),
    ("analysis_context.research_quality.top_gaps", "analysis_context.research_quality"),
    ("analysis_context.citations.", "analysis_context.citations "),
    ("analysis_context.recent_event_impacts.", "analysis_context.recent_event_impacts "),
    ("analysis_context.research_quality.", "analysis_context.research_quality "),
    ("evidence_packet.fundamentals.derived_signals.", "evidence_packet.fundamentals."),
    (
        "evidence_packet.extracted_facts.r_and_d_pct_revenue",
        "evidence_packet.fundamentals.r_and_d_intensity_latest",
    ),
    (
        "evidence_packet.extracted_facts.deferred_revenue_ratio",
        "evidence_packet.fundamentals.deferred_revenue_to_revenue_latest",
    ),
    (
        "evidence_packet.extracted_facts.rpo_ratio",
        "evidence_packet.fundamentals.rpo_to_revenue_latest",
    ),
)


def _rewrite_analysis_context_aliases(text: str) -> str:
    value = str(text or "")
    for old, new in _ANALYSIS_ALIAS_REPLACEMENTS:
        value = value.replace(old, new)
    return value


def _round_text_numbers(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        pct = match.group(2)
        try:
            value = float(raw)
        except Exception:
            return match.group(0)
        precision = 1 if pct == "%" else 2
        rounded = f"{value:.{precision}f}".rstrip("0").rstrip(".")
        return f"{rounded}{pct}"

    return _LONG_FLOAT_RE.sub(_replace, text)


def _polish_text(text: str) -> str:
    value = _rewrite_analysis_context_aliases(text)
    value = value.replace("cover page extraction", "fallback filing extraction")
    value = value.replace("segments_signal", "segment disclosures")
    value = value.replace("segments signals", "segment disclosures")
    value = value.replace("segment signals", "segment disclosures")
    value = value.replace("segments signal present", "segment disclosures are present")
    value = value.replace("segment disclosures present", "segment disclosures across filings")
    value = value.replace(
        "material_weakness flagged in filings", "material-weakness disclosure in filings"
    )
    value = value.replace("Research packet identifies", "Current research identifies")
    value = value.replace("research packet identifies", "current research identifies")
    value = value.replace("research packet signal", "research signal")
    value = value.replace("research packet signals", "research signals")
    value = value.replace("capital_allocation", "capital allocation")
    value = value.replace("fundamentals.revenue_cagr_3y", "multi-year revenue growth")
    value = value.replace("companyfacts loans", "loan series")
    value = value.replace("canonical canonical valuation methods", "canonical valuation methods")
    value = value.replace("debt_maturity", "debt maturity")
    value = value.replace("material_weakness", "material weakness")
    value = value.replace("cash_flow_quality", "cash-flow quality")
    value = value.replace("debt_liquidity", "liquidity and debt")
    value = value.replace("sbc_dilution_signal", "stock-based compensation dilution signal")
    value = value.replace("quality_signal", "quality signal")
    value = value.replace("&D are unavailable", "R&D details are unavailable")
    value = value.replace("&D are", "R&D are")
    value = value.replace("fundamentals.gaps", "current evidence gaps")
    for old, new in _TEXT_PATH_REPLACEMENTS:
        value = value.replace(old, new)
    value = re.sub(r"(?<!evidence_packet\.)\bvaluations\.", "evidence_packet.valuations.", value)
    value = value.replace("evidence_packet.evidence_packet.", "evidence_packet.")
    value = value.replace(
        "before per-share evidence_packet.valuations.", "before per-share valuation work."
    )
    value = re.sub(
        r"\bevidence_packet\.valuations\.\s*(?=[\.,;:]|\Z)", "canonical valuation methods", value
    )
    value = _round_text_numbers(value)
    value = re.sub(r"\s{2,}", " ", value).strip()
    return value


_TRACE_PAREN_RE = re.compile(
    r"\s*\((?=[^()]*\b(?:evidence_packet|analysis_context|research_packet|delta_context)\b)[^()]*\)"
)
_TRACE_INLINE_RE = re.compile(
    r"(?:\b(?:analysis_context|research_packet|delta_context)\.[A-Za-z0-9_\.\[\]'=@:\-?() ]+|"
    r"\bevidence_packet\.(?:fundamentals|valuations|enrichment|extracted_facts|dossier_focus)\.[A-Za-z0-9_\.\[\]'=@:\-?() ]+|"
    r"\b(?:extracted_facts|analysis_context|research_packet|evidence_packet)\.[A-Za-z0-9_\.\[\]'=@:\-?() ]+|"
    r"\bevidence_items\b\s+[A-Za-z0-9_]+)"
)
_RESEARCH_STEP_RE = re.compile(
    r"\b(?:(?:research_packet|analysis_context)\.)?next_actions(?:\[[0-9]+\])?\s*(?:step\s*)?A\d+\b",
    re.IGNORECASE,
)
_RESEARCH_GAP_RE = re.compile(
    r"\b(?:(?:research_packet|analysis_context)\.)?(?:evidence_gaps?|open_questions)\s+[A-Z0-9_]+\b",
    re.IGNORECASE,
)
_NUMERIC_TOKEN = r"-?\d(?:[\d,]*\d)?(?:\.\d+)?"
_LARGE_NUMERIC_TOKEN = r"-?(?:\d{1,3}(?:,\d{3}){1,}|\d{7,})(?:\.\d+)?"
_RAW_DOLLAR_RE = re.compile(
    rf"(?P<prefix>\$)\s?(?P<value>{_LARGE_NUMERIC_TOKEN})\s*(?P<suffix>USD|usd)?\b"
)
_THOUSAND_DOLLAR_RE = re.compile(rf"(?P<prefix>\$)\s?(?P<value>{_NUMERIC_TOKEN})k\b", re.IGNORECASE)
_BARE_USD_RE = re.compile(rf"(?<![\$0-9])(?P<value>{_LARGE_NUMERIC_TOKEN})\s*USD\b", re.IGNORECASE)


def _format_compact_currency(value: float) -> str:
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    if abs_value >= 1_000_000_000_000:
        return f"{sign}${abs_value / 1_000_000_000_000:.2f}t"
    if abs_value >= 1_000_000_000:
        return f"{sign}${abs_value / 1_000_000_000:.1f}b"
    if abs_value >= 1_000_000:
        return f"{sign}${abs_value / 1_000_000:.1f}m"
    return f"{sign}${abs_value:,.0f}"


def _normalize_money_mentions(text: str) -> str:
    value = str(text or "")

    def _replace_raw(match: re.Match[str]) -> str:
        try:
            amount = float(match.group("value").replace(",", ""))
        except Exception:
            return match.group(0)
        return _format_compact_currency(amount)

    def _replace_thousands(match: re.Match[str]) -> str:
        try:
            amount = float(match.group("value").replace(",", ""))
        except Exception:
            return match.group(0)
        return f"${amount:,.0f}k"

    def _replace_bare_usd(match: re.Match[str]) -> str:
        try:
            amount = float(match.group("value").replace(",", ""))
        except Exception:
            return match.group(0)
        return _format_compact_currency(amount)

    value = _THOUSAND_DOLLAR_RE.sub(_replace_thousands, value)
    value = _RAW_DOLLAR_RE.sub(_replace_raw, value)
    value = _BARE_USD_RE.sub(_replace_bare_usd, value)
    return value


def _sanitize_rendered_text(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"(-?\$[0-9,]+m)\.0\b", r"\1", value, flags=re.IGNORECASE)
    value = re.sub(r"(-?\$[0-9,]+m)\s+million\b", r"\1", value, flags=re.IGNORECASE)
    value = re.sub(r"(-?\$[0-9,]+m)\s*USD\b", r"\1", value, flags=re.IGNORECASE)
    value = re.sub(r"(-?\$[0-9,]+)million\b", r"\1m", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\b([0-9][0-9,]*)\.0\s*\((?:millions?|million)\s+USD\)",
        r"$\1m",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\b([0-9][0-9,]*)\.0\s+millions?\s+USD\b", r"$\1m", value, flags=re.IGNORECASE)
    value = re.sub(r"(-?\$[0-9,]+m)\.0\s+million\b", r"\1", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<!\$)(\b\d[\d,\.]*)\s*mm\b", r"\1m", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\bsegment disclosures across filings\b",
        "segment reporting across filings",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\bsegment disclosures\b", "segment reporting", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\bresearch packet flags\b", "current research notes", value, flags=re.IGNORECASE
    )
    value = re.sub(r"\bresearch packet\b", "current research", value, flags=re.IGNORECASE)
    value = re.sub(r"\bcompanyfacts feed\b", "companyfacts data", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\bshareholder_return_active\b", "shareholder-return active", value, flags=re.IGNORECASE
    )
    value = re.sub(
        r"\boperating_margin_trend_slope\b", "operating-margin trend", value, flags=re.IGNORECASE
    )
    value = re.sub(
        r"\ballowance_for_credit_losses\b",
        "allowance for credit losses",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\bcash-flow quality signals\b",
        "cash-flow quality disclosures",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\bcash-flow quality_signal\b", "cash-flow quality disclosure", value, flags=re.IGNORECASE
    )
    value = re.sub(
        r"\bdebt maturity and covenant signals\b",
        "debt maturity and covenant disclosures",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\bdebt maturity_signal; covenant_signal\b",
        "debt maturity and covenant disclosures",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\bdebt maturity_signal\b", "debt maturity disclosure", value, flags=re.IGNORECASE
    )
    value = re.sub(r"\bcovenant_signal\b", "covenant disclosure", value, flags=re.IGNORECASE)
    value = re.sub(r"\b([0-9]+\.[0-9]+)m[–-]([0-9]+\.[0-9]+)m USD\b", r"$\1tn-\2tn", value)
    value = value.replace("~$", "~$")
    value = re.sub(r"\s{2,}", " ", value).strip()
    return value


_EXACT_METRIC_UNITS = {
    "USD",
    "USD_millions",
    "USD_per_share",
    "shares",
    "shares_millions",
    "ratio",
    "percent",
}


def _claim_metric_from_citations(claim: dict[str, Any]) -> tuple[str | None, float | None]:
    aliases = {
        "price": "market_price",
        "market_price": "market_price",
    }
    for citation in claim.get("citations") or []:
        if not isinstance(citation, dict):
            continue
        snippet = str(citation.get("snippet") or "").strip().lower()
        match = re.match(r"([a-z_]+):\s*(-?\d+(?:\.\d+)?)", snippet)
        if not match:
            continue
        metric = match.group(1)
        try:
            value = float(match.group(2))
        except Exception:
            continue
        return aliases.get(metric, metric), value
    return None, None


def _metric_units_from_packet(evidence_packet: dict[str, Any] | None, metric: str) -> str | None:
    if not isinstance(evidence_packet, dict):
        return None
    financials = (
        evidence_packet.get("financials")
        if isinstance(evidence_packet.get("financials"), list)
        else []
    )
    matching_units: list[str] = []
    for row in financials:
        if not isinstance(row, dict):
            continue
        if str(row.get("line_item") or "") != metric:
            continue
        units = str(row.get("units") or "").strip()
        if not units or units not in _EXACT_METRIC_UNITS:
            return None
        matching_units.append(units)
    distinct_units = set(matching_units)
    return next(iter(distinct_units)) if len(distinct_units) == 1 else None


def _metric_value_from_packet(evidence_packet: dict[str, Any] | None, metric: str) -> float | None:
    if not isinstance(evidence_packet, dict):
        return None
    fundamentals = evidence_packet.get("fundamentals")
    if isinstance(fundamentals, dict):
        value = fundamentals.get(metric)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _format_metric_value_for_units(metric: str, value: float, units: str | None) -> str:
    normalized_units = str(units or "").strip()
    if normalized_units == "USD_per_share":
        return f"${value:,.2f}".rstrip("0").rstrip(".")
    if normalized_units == "USD_millions":
        sign = "-" if value < 0 else ""
        return f"{sign}${abs(value):,.0f}m"
    if normalized_units == "USD":
        return _format_compact_currency(value)
    if normalized_units == "shares_millions":
        return f"{value:,.2f}m shares"
    if normalized_units == "shares":
        return f"{value:,.0f} shares"
    if normalized_units == "percent":
        return f"{value:,.2f}%"
    if normalized_units == "ratio":
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    raise ValueError(f"Unsupported metric unit: {normalized_units or 'missing'}")


def _replace_first_numeric_token(text: str, replacements: tuple[str, ...], replacement: str) -> str:
    for target in sorted((token for token in replacements if token), key=len, reverse=True):
        if target and target in text:
            return text.replace(target, replacement, 1)
    return text


def _normalize_claim_text_units(
    claim: dict[str, Any],
    *,
    evidence_packet: dict[str, Any] | None = None,
) -> bool:
    text = str(claim.get("text") or "")
    metric, metric_value = _claim_metric_from_citations(claim)
    if not metric or metric_value is None:
        claim["text"] = _normalize_money_mentions(text)
        return True
    units = _metric_units_from_packet(evidence_packet, metric)
    if units is None:
        claim["unit_integrity_status"] = "INVALID_FINANCIAL_INPUT"
        claim["unit_integrity_reason"] = "MISSING_OR_AMBIGUOUS_UNIT"
        return False
    packet_metric_value = _metric_value_from_packet(evidence_packet, metric)
    if packet_metric_value is not None and not math.isclose(
        metric_value,
        packet_metric_value,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        # The citation and canonical packet disagree. Magnitude cannot decide
        # which unit was intended, and silently replacing one with the other
        # would launder a conflicting financial claim.
        claim["unit_integrity_status"] = "INVALID_FINANCIAL_INPUT"
        return False
    formatted = _format_metric_value_for_units(metric, metric_value, units)
    if metric == "net_debt":
        raw_target = f"${metric_value:,.0f}"
        text = text.replace(raw_target, formatted)
        text = text.replace(f"{metric_value:,.0f} USD", formatted)
        text = text.replace(f"{metric_value:,.0f}", formatted)
    elif metric == "market_price":
        text = _replace_first_numeric_token(
            text,
            (
                f"${metric_value:,.2f}".rstrip("0").rstrip("."),
                f"{metric_value:,.2f}".rstrip("0").rstrip("."),
                f"{metric_value}",
            ),
            formatted,
        )
    elif units == "USD_millions":
        raw_usd = metric_value * 1_000_000.0
        text = _replace_first_numeric_token(
            text,
            (
                f"${metric_value:,.0f}m",
                f"{metric_value:,.0f}m",
                f"${metric_value:,.1f}m",
                f"${metric_value:,.0f}",
                f"{metric_value:,.0f}",
                f"${raw_usd:,.0f}",
                f"{raw_usd:,.0f}",
            ),
            formatted,
        )
    else:
        text = _replace_first_numeric_token(
            text,
            (
                f"${metric_value:,.2f}".rstrip("0").rstrip("."),
                f"${metric_value:,.0f}",
                f"{metric_value:,.2f}".rstrip("0").rstrip("."),
                f"{metric_value:,.0f}",
                f"{metric_value}",
            ),
            formatted,
        )
    claim["text"] = _sanitize_rendered_text(_normalize_money_mentions(text))
    return True


def _strip_trace_refs_from_narrative(text: str) -> str:
    value = _rewrite_analysis_context_aliases(text)
    value = re.sub(r"\bderived_from:\s*[^.;]+", "", value)
    value = _TRACE_PAREN_RE.sub("", value)
    value = _TRACE_INLINE_RE.sub("", value)
    value = re.sub(r"\bextracted_facts\s+([A-Za-z0-9_ ]+?)\s+signals?\b", r"\1 signals", value)
    value = re.sub(r"\bextracted_facts\s+([A-Za-z0-9_]+)\b", r"\1", value)
    value = _RESEARCH_STEP_RE.sub("the research next step", value)
    value = _RESEARCH_GAP_RE.sub("the current research gap", value)
    value = re.sub(r"\bSources?\s*[:;]\s*[^.]*", "", value, flags=re.IGNORECASE)
    value = value.replace("[", "").replace("]", "")
    value = re.sub(r"(?:\s*[;,:]){2,}", "; ", value)
    value = re.sub(r"\s*;\s*([,.;:])", r"\1", value)
    value = re.sub(r"\(\s*[;,:]+\s*\)", "", value)
    value = re.sub(r"\s*\(\s*$", "", value)
    value = re.sub(r"\(\s*\)", "", value)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    value = re.sub(r"([,;:])([A-Za-z])", r"\1 \2", value)
    value = re.sub(r"\s{2,}", " ", value).strip()
    return value


def _is_generic_derived_ref(value: str) -> bool:
    token = str(value or "").strip()
    return token in {
        "evidence_packet.fundamentals",
        "evidence_packet.rows",
        "evidence_packet.extracted_facts",
        "evidence_packet.financials",
    } or bool(re.fullmatch(r"evidence_packet\.extracted_facts\[\d+\]\.value", token))


def _is_soft_derived_ref(value: str) -> bool:
    token = str(value or "").strip()
    return _is_generic_derived_ref(token) or token.startswith(
        "evidence_packet.dossier_focus.trend_signals."
    )


def _derived_from_from_citation(claim: dict[str, Any]) -> list[str]:
    citations = claim.get("citations")
    if not isinstance(citations, list):
        return []
    out: list[str] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        snippet = str(citation.get("snippet") or "").strip().lower()
        source_url = str(citation.get("source_url") or "").strip().lower()
        if "companyfacts" in source_url:
            metric_match = re.match(r"([a-z_]+):", snippet)
            if metric_match:
                metric = metric_match.group(1)
                for candidate in (
                    f"evidence_packet.fundamentals.{metric}",
                    f"evidence_packet.financials.{metric}",
                    f"evidence_packet.extracted_facts.{metric}",
                ):
                    canonical = _canonicalize_derived_path(candidate)
                    if canonical and canonical not in out:
                        out.append(canonical)
        if "price:" in snippet and "price_source:" in snippet:
            for candidate in (
                "evidence_packet.valuations.reverse_dcf.inputs.price",
                "evidence_packet.valuations.reverse_dcf.inputs.market_price",
            ):
                canonical = _canonicalize_derived_path(candidate)
                if canonical and canonical not in out:
                    out.append(canonical)
    return out


def _preferred_metric_paths(evidence_packet: dict[str, Any], metric: str) -> list[str]:
    out: list[str] = []
    fundamentals = (
        evidence_packet.get("fundamentals")
        if isinstance(evidence_packet.get("fundamentals"), dict)
        else {}
    )
    if metric in {"market_price", "price"}:
        for candidate in (
            "evidence_packet.valuations.reverse_dcf.inputs.price",
            "evidence_packet.valuations.legacy_multiples.inputs.price",
        ):
            canonical = _canonicalize_derived_path(candidate)
            if canonical and canonical not in out:
                out.append(canonical)
        return out
    if metric in {
        "revenue_cagr_3y",
        "revenue_cagr_5y",
        "revenue_cagr_10y",
        "dilution_rate_shares_cagr",
        "deposits_to_assets_latest",
        "loans_to_deposits_latest",
        "allowance_to_loans_latest",
    }:
        canonical = _canonicalize_derived_path(f"evidence_packet.fundamentals.{metric}")
        return [canonical] if canonical else []
    if isinstance(fundamentals, dict) and metric in fundamentals:
        canonical = _canonicalize_derived_path(f"evidence_packet.fundamentals.{metric}")
        if canonical and canonical not in out:
            return [canonical]
    extracted = (
        evidence_packet.get("extracted_facts")
        if isinstance(evidence_packet.get("extracted_facts"), list)
        else []
    )
    if any(
        isinstance(row, dict) and str(row.get("fact_type") or "") == metric for row in extracted
    ):
        canonical = _canonicalize_derived_path(f"evidence_packet.extracted_facts.{metric}")
        if canonical and canonical not in out:
            return [canonical]
    financials = (
        evidence_packet.get("financials")
        if isinstance(evidence_packet.get("financials"), list)
        else []
    )
    if any(
        isinstance(row, dict) and str(row.get("line_item") or "") == metric for row in financials
    ):
        canonical = _canonicalize_derived_path(f"evidence_packet.financials.{metric}")
        if canonical and canonical not in out:
            return [canonical]
    return out


def _repair_claim_derived_from(
    claim: dict[str, Any], *, evidence_packet: dict[str, Any] | None = None
) -> None:
    derived_from = claim.get("derived_from")
    current = [str(item).strip() for item in derived_from] if isinstance(derived_from, list) else []
    current = [item for item in current if item]
    inferred = _derived_from_from_citation(claim)
    metric_hint = _metric_hint_from_claim(claim)
    preferred = (
        _preferred_metric_paths(evidence_packet or {}, metric_hint)
        if metric_hint and isinstance(evidence_packet, dict)
        else []
    )
    if claim.get("type") != "numeric":
        if preferred and (not current or all(_is_soft_derived_ref(item) for item in current)):
            claim["derived_from"] = preferred
        else:
            claim["derived_from"] = current
        return
    if not current:
        claim["derived_from"] = preferred or inferred
        return
    specific_current = [item for item in current if not _is_soft_derived_ref(item)]
    if specific_current:
        claim["derived_from"] = preferred or specific_current
        return
    claim["derived_from"] = preferred or inferred or current


def _is_external_source_url(value: str) -> bool:
    token = str(value or "").strip().lower()
    return token.startswith("https://") or token.startswith("http://")


def _dedupe_citation_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        source_url = str(row.get("source_url") or "").strip()
        snippet = str(row.get("snippet") or "").strip()
        section_label = str(row.get("section_label") or "").strip()
        if not source_url:
            continue
        key = (source_url, snippet, section_label)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "source_url": source_url,
                "snippet": snippet,
                "section_label": section_label or None,
            }
        )
    return out


def _metric_citations_from_packet(
    evidence_packet: dict[str, Any], metric: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    financials = (
        evidence_packet.get("financials")
        if isinstance(evidence_packet.get("financials"), list)
        else []
    )
    for row in financials:
        if not isinstance(row, dict) or str(row.get("line_item") or "") != metric:
            continue
        citation = row.get("citation")
        if isinstance(citation, dict) and _is_external_source_url(
            str(citation.get("source_url") or "")
        ):
            out.append(citation)
    extracted = (
        evidence_packet.get("extracted_facts")
        if isinstance(evidence_packet.get("extracted_facts"), list)
        else []
    )
    for row in extracted:
        if not isinstance(row, dict) or str(row.get("fact_type") or "") != metric:
            continue
        citation = row.get("citation")
        if isinstance(citation, dict) and _is_external_source_url(
            str(citation.get("source_url") or "")
        ):
            out.append(citation)
    fundamentals = (
        evidence_packet.get("fundamentals")
        if isinstance(evidence_packet.get("fundamentals"), dict)
        else {}
    )
    row_traces = (
        fundamentals.get("row_traces")
        if isinstance(fundamentals, dict) and isinstance(fundamentals.get("row_traces"), dict)
        else {}
    )
    rows = (
        fundamentals.get("rows")
        if isinstance(fundamentals, dict) and isinstance(fundamentals.get("rows"), list)
        else []
    )
    latest_year = None
    if rows:
        latest = rows[-1]
        if isinstance(latest, dict):
            latest_year = str(latest.get("year") or "")
    if latest_year and isinstance(row_traces.get(latest_year), dict):
        trace_bucket = (row_traces.get(latest_year) or {}).get(metric)
        if isinstance(trace_bucket, dict):
            for citation in trace_bucket.get("citations") or []:
                if isinstance(citation, dict) and _is_external_source_url(
                    str(citation.get("source_url") or "")
                ):
                    out.append(citation)
    if not out:
        filings = (
            evidence_packet.get("filings_used")
            if isinstance(evidence_packet.get("filings_used"), list)
            else []
        )
        source_url = ""
        for filing in filings:
            if not isinstance(filing, dict):
                continue
            candidate = str(filing.get("primary_doc_url") or "").strip()
            if _is_external_source_url(candidate):
                source_url = candidate
                break
        if source_url:
            value = fundamentals.get(metric)
            snippet = (
                f"{metric}: {value}"
                if isinstance(value, (int, float, str)) and str(value).strip()
                else metric
            )
            out.append(
                {
                    "source_url": source_url,
                    "snippet": snippet,
                    "section_label": "financial_statements",
                }
            )
    return _dedupe_citation_dicts(out)


_TREND_SIGNAL_BASE_METRICS = {
    "revenue_cagr_3y": ["revenue"],
    "revenue_cagr_5y": ["revenue"],
    "revenue_cagr_10y": ["revenue"],
    "dilution_rate_shares_cagr": ["shares_outstanding"],
    "deposits_to_assets_latest": ["deposits", "total_assets"],
    "loans_to_deposits_latest": ["loans", "deposits"],
    "allowance_to_loans_latest": ["allowance_for_credit_losses", "loans"],
}


def _metric_hint_from_claim(claim: dict[str, Any]) -> str | None:
    blob = " ".join(
        [
            str(claim.get("id") or ""),
            str(claim.get("text") or ""),
            " ".join(
                str((citation or {}).get("snippet") or "")
                for citation in (claim.get("citations") or [])
                if isinstance(citation, dict)
            ),
        ]
    ).lower()
    hints = (
        ("3-year cagr", "revenue_cagr_3y"),
        ("3‑year cagr", "revenue_cagr_3y"),
        ("3 year cagr", "revenue_cagr_3y"),
        ("5-year cagr", "revenue_cagr_5y"),
        ("5‑year cagr", "revenue_cagr_5y"),
        ("5 year cagr", "revenue_cagr_5y"),
        ("10-year cagr", "revenue_cagr_10y"),
        ("10‑year cagr", "revenue_cagr_10y"),
        ("10 year cagr", "revenue_cagr_10y"),
        ("3y cagr", "revenue_cagr_3y"),
        ("5y cagr", "revenue_cagr_5y"),
        ("10y cagr", "revenue_cagr_10y"),
        ("share repurchases", "share_repurchases_amount"),
        ("repurchases", "share_repurchases_amount"),
        ("dividends", "dividends_paid_amount"),
        ("revenue cagr", "revenue_cagr_3y"),
        ("revenue", "revenue"),
        ("operating cash flow", "cfo"),
        ("cfo", "cfo"),
        ("deposits-to-assets", "deposits_to_assets_latest"),
        ("deposits to assets", "deposits_to_assets_latest"),
        ("deposits", "deposits"),
        ("loans-to-deposits", "loans_to_deposits_latest"),
        ("loans", "loans"),
        ("allowance", "allowance_for_credit_losses"),
        ("market price", "market_price"),
        ("price used", "market_price"),
        ("net debt", "net_debt"),
    )
    for token, metric in hints:
        if token in blob:
            return metric
    return None


def _valuation_price_citations(evidence_packet: dict[str, Any]) -> list[dict[str, Any]]:
    valuations = (
        evidence_packet.get("valuations")
        if isinstance(evidence_packet.get("valuations"), dict)
        else {}
    )
    out: list[dict[str, Any]] = []
    for method_payload in valuations.values():
        if not isinstance(method_payload, dict):
            continue
        inputs = method_payload.get("inputs")
        if not isinstance(inputs, dict):
            continue
        source_url = str(inputs.get("price_source_url") or "").strip()
        if not _is_external_source_url(source_url):
            continue
        price = inputs.get("market_price", inputs.get("price"))
        snippet = f"market_price: {price}" if isinstance(price, (int, float)) else ""
        out.append({"source_url": source_url, "snippet": snippet, "section_label": "valuations"})
    return _dedupe_citation_dicts(out)


def _fallback_external_citations_from_packet(
    evidence_packet: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in evidence_packet.get("extracted_facts") or []:
        if not isinstance(row, dict):
            continue
        citation = row.get("citation")
        if isinstance(citation, dict) and _is_external_source_url(
            str(citation.get("source_url") or "")
        ):
            out.append(citation)
    for row in evidence_packet.get("financials") or []:
        if not isinstance(row, dict):
            continue
        citation = row.get("citation")
        if isinstance(citation, dict) and _is_external_source_url(
            str(citation.get("source_url") or "")
        ):
            out.append(citation)
    return _dedupe_citation_dicts(out)


def _inferred_citations_for_claim(
    claim: dict[str, Any],
    *,
    evidence_packet: dict[str, Any],
    analysis_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    refs = [str(ref).strip() for ref in (claim.get("derived_from") or []) if str(ref).strip()]
    metric_hint = _metric_hint_from_claim(claim)
    candidates: list[dict[str, Any]] = []
    for ref in refs:
        canonical = _canonicalize_derived_path(ref)
        if canonical.startswith("evidence_packet.financials."):
            metric = canonical.split(".")[-1]
            candidates.extend(_metric_citations_from_packet(evidence_packet, metric))
        elif canonical.startswith("evidence_packet.extracted_facts."):
            metric = canonical.split(".")[-1]
            candidates.extend(_metric_citations_from_packet(evidence_packet, metric))
        elif canonical.startswith("evidence_packet.fundamentals."):
            metric = canonical.split(".")[-1]
            if metric in _TREND_SIGNAL_BASE_METRICS:
                for base_metric in _TREND_SIGNAL_BASE_METRICS[metric]:
                    candidates.extend(_metric_citations_from_packet(evidence_packet, base_metric))
            else:
                candidates.extend(_metric_citations_from_packet(evidence_packet, metric))
        elif canonical.startswith("evidence_packet.dossier_focus.capital_allocation"):
            if metric_hint in {"share_repurchases_amount", "dividends_paid_amount"}:
                candidates.extend(_metric_citations_from_packet(evidence_packet, metric_hint))
        elif canonical.startswith("evidence_packet.dossier_focus.trend_signals"):
            if metric_hint == "revenue":
                candidates.extend(_metric_citations_from_packet(evidence_packet, "revenue"))
            elif metric_hint == "dilution_rate_shares_cagr":
                candidates.extend(
                    _metric_citations_from_packet(evidence_packet, "shares_outstanding")
                )
        elif canonical.startswith("evidence_packet.valuations."):
            if (
                metric_hint == "market_price"
                or ".price" in canonical
                or ".market_price" in canonical
            ):
                candidates.extend(_valuation_price_citations(evidence_packet))
        elif canonical.startswith("analysis_context.") and isinstance(analysis_context, dict):
            for citation in (analysis_context.get("citations") or [])[:2]:
                if isinstance(citation, dict) and _is_external_source_url(
                    str(citation.get("source_url") or "")
                ):
                    candidates.append(citation)
    if metric_hint and not candidates:
        if metric_hint in _TREND_SIGNAL_BASE_METRICS:
            for base_metric in _TREND_SIGNAL_BASE_METRICS[metric_hint]:
                candidates.extend(_metric_citations_from_packet(evidence_packet, base_metric))
        elif metric_hint == "market_price":
            candidates.extend(_valuation_price_citations(evidence_packet))
        else:
            candidates.extend(_metric_citations_from_packet(evidence_packet, metric_hint))
    if not candidates:
        candidates.extend(_fallback_external_citations_from_packet(evidence_packet)[:1])
    return _dedupe_citation_dicts(candidates)


def _claim_prefers_inferred_citations(claim: dict[str, Any]) -> bool:
    metric_hint = _metric_hint_from_claim(claim)
    blob = " ".join([str(claim.get("id") or ""), str(claim.get("text") or "")]).lower()
    return (
        metric_hint in {"revenue_cagr_3y", "revenue_cagr_5y", "revenue_cagr_10y"} or "cagr" in blob
    )


def _claim_has_acceptable_support(claim: dict[str, Any]) -> bool:
    metric_hint = _metric_hint_from_claim(claim)
    citations = claim.get("citations") or []
    if metric_hint in {"revenue_cagr_3y", "revenue_cagr_5y", "revenue_cagr_10y"}:
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            snippet = str(citation.get("snippet") or "").lower()
            section_label = str(citation.get("section_label") or "").lower()
            if "cagr" in snippet or "cagr" in section_label or "fundamentals" in section_label:
                return True
        return False
    return True


def _repair_claim_citations(
    claim: dict[str, Any],
    *,
    evidence_packet: dict[str, Any],
    analysis_context: dict[str, Any] | None = None,
) -> None:
    citations = claim.get("citations")
    current = (
        [row for row in citations if isinstance(row, dict)] if isinstance(citations, list) else []
    )
    external = [row for row in current if _is_external_source_url(str(row.get("source_url") or ""))]
    inferred = _inferred_citations_for_claim(
        claim,
        evidence_packet=evidence_packet,
        analysis_context=analysis_context,
    )
    if inferred and _claim_prefers_inferred_citations(claim):
        claim["citations"] = inferred
        return
    claim["citations"] = external or inferred


def _polish_narrative_text(text: str, *, strip_trace_refs: bool = True) -> str:
    value = _polish_text(text)
    value = _normalize_money_mentions(value)
    value = _sanitize_rendered_text(value)
    if strip_trace_refs:
        value = _strip_trace_refs_from_narrative(value)
    return value


def _canonicalize_derived_path(path: str) -> str:
    value = _rewrite_analysis_context_aliases(path).strip()
    if not value:
        return value
    if re.fullmatch(r"evidence_packet\.[A-Za-z0-9_]+\[\d+\]", value):
        return ""
    financial_match = re.fullmatch(
        r"evidence_packet\.financials\[\?\(@\.line_item=='([A-Za-z0-9_]+)'(?:\s*&&\s*@\.period=='[^']+')?\)\]\.value",
        value,
    )
    if financial_match:
        return f"evidence_packet.financials.{financial_match.group(1)}"
    if value.startswith("evidence_packet.fundamentals.derived_signals."):
        suffix = value.split(".", 3)[3]
        return f"evidence_packet.fundamentals.{suffix}"
    if value == "evidence_packet.fundamentals.r_and_d_pct_revenue":
        return "evidence_packet.fundamentals.r_and_d_intensity_latest"
    if value.startswith(("evidence_packet.", "analysis_context.", "delta_context.", "config.")):
        return value
    if value.startswith("evidence_packet.extracted_facts (") and value.endswith(")"):
        metric = (
            value.removeprefix("evidence_packet.extracted_facts (").removesuffix(")").split()[0]
        )
        return f"evidence_packet.extracted_facts.{metric}"
    if value.startswith("valuations."):
        return f"evidence_packet.{value}"
    if value.startswith("fundamentals."):
        return f"evidence_packet.{value}"
    if value.startswith("financials."):
        suffix = value.split(".", 1)[1]
        return f"evidence_packet.financials.{suffix}"
    if value.startswith("extracted_facts."):
        suffix = value.split(".", 1)[1]
        return f"evidence_packet.extracted_facts.{suffix}"
    if value.startswith("dossier.extractors."):
        suffix = value.split(".", 2)[2]
        return f"evidence_packet.extracted_facts.{suffix}"
    if value.startswith("dossier.metrics."):
        suffix = value.split(".", 2)[2]
        return f"evidence_packet.fundamentals.{suffix}"
    if value.startswith("dossier.time_series."):
        suffix = value.split(".", 2)[2]
        return f"evidence_packet.fundamentals.{suffix}"
    if value.startswith("dossier."):
        suffix = value.split(".", 1)[1]
        return f"evidence_packet.fundamentals.{suffix}"
    return value


_NUMERIC_CLAIM_HINT_RE = re.compile(
    r"(\$\s?\d|\b\d+(?:\.\d+)?\s?(?:%|x)\b|\bcount\s*[:=]\s*\d|\bsegments?\s*[:=]\s*\d|\b=\s*\d)",
    re.IGNORECASE,
)


def _looks_numeric_claim(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    return bool(_NUMERIC_CLAIM_HINT_RE.search(value))


def _ensure_trend_language(field: str, text: str, payload: dict[str, Any]) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    if field not in {"business_quality_summary", "valuation_interpretation"}:
        return value
    lowered = value.lower()
    if any(
        token in lowered
        for token in (
            "trend",
            "cagr",
            "multi-year",
            "over time",
            "over the past",
            "improving",
            "stable",
            "weakening",
        )
    ):
        return value
    fundamentals = (
        payload.get("fundamentals") if isinstance(payload.get("fundamentals"), dict) else {}
    )
    if not isinstance(fundamentals, dict):
        fundamentals = {}
    trend_bits: list[str] = []
    revenue_cagr_3y = fundamentals.get("revenue_cagr_3y")
    if _is_num_like(revenue_cagr_3y):
        trend_bits.append(
            f"Revenue has compounded at roughly {float(revenue_cagr_3y) * 100:.1f}% over the last 3 years"
        )
    operating_margin_trend = fundamentals.get("operating_margin_trend_slope")
    if _is_num_like(operating_margin_trend):
        if float(operating_margin_trend) > 0:
            trend_bits.append("operating margins have improved over time")
        elif float(operating_margin_trend) < 0:
            trend_bits.append("operating margins have softened over time")
    dilution_cagr = fundamentals.get("dilution_rate_shares_cagr")
    if _is_num_like(dilution_cagr):
        if float(dilution_cagr) < 0:
            trend_bits.append("share count has trended modestly lower")
        elif float(dilution_cagr) > 0:
            trend_bits.append("share count has continued to drift upward")
    if not trend_bits:
        return value
    return f"{value} {'; '.join(trend_bits)}."


def _is_num_like(value: Any) -> bool:
    return isinstance(value, (int, float))


def _needs_deterministic_backfill(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    text = value.strip()
    if not text:
        return True
    lowered = text.lower()
    return lowered.startswith("unknown") or lowered in {
        "n/a",
        "na",
        "none",
        "null",
    }


def _polish_synthesis_payload(
    payload: dict[str, Any],
    *,
    evidence_packet: dict[str, Any] | None = None,
    analysis_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text_fields = [
        "business_quality_summary",
        "valuation_interpretation",
        "risk_frame",
        "catalyst_frame",
        "confidence_notes",
    ]
    for field in text_fields:
        value = payload.get(field)
        if isinstance(value, str):
            payload[field] = _ensure_trend_language(field, _polish_narrative_text(value), payload)

    for field in ("evidence_gaps", "recommended_next_actions"):
        rows = payload.get(field)
        if isinstance(rows, list):
            payload[field] = [_polish_narrative_text(str(item)) for item in rows]

    hypotheses = payload.get("hypotheses")
    if isinstance(hypotheses, list):
        for row in hypotheses:
            if not isinstance(row, dict):
                continue
            for key in ("statement", "why_it_might_be_true"):
                if isinstance(row.get(key), str):
                    row[key] = _polish_narrative_text(str(row[key]))
            for key in ("falsifiers", "required_evidence"):
                vals = row.get(key)
                if isinstance(vals, list):
                    row[key] = [_polish_narrative_text(str(item)) for item in vals]

    claims = payload.get("claims")
    if isinstance(claims, list):
        cleaned_claims: list[dict[str, Any]] = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            if isinstance(claim.get("text"), str):
                claim["text"] = _polish_text(str(claim["text"]))
            derived_from = claim.get("derived_from")
            if isinstance(derived_from, list):
                claim["derived_from"] = [
                    normalized
                    for item in derived_from
                    if str(item).strip()
                    for normalized in [_canonicalize_derived_path(str(item))]
                    if normalized
                ]
            if claim.get("type") == "non_numeric" and _looks_numeric_claim(
                str(claim.get("text") or "")
            ):
                claim["type"] = "numeric"
            _repair_claim_derived_from(claim, evidence_packet=evidence_packet)
            if isinstance(evidence_packet, dict):
                _repair_claim_citations(
                    claim,
                    evidence_packet=evidence_packet,
                    analysis_context=analysis_context,
                )
            if not _normalize_claim_text_units(
                claim,
                evidence_packet=evidence_packet,
            ):
                continue
            if not _claim_has_acceptable_support(claim):
                continue
            cleaned_claims.append(claim)
        payload["claims"] = cleaned_claims

    priced_in = payload.get("priced_in_assessment")
    if isinstance(priced_in, dict):
        for key in ("what_market_assumes", "what_is_not_priced", "uncertainty_notes"):
            if isinstance(priced_in.get(key), str):
                priced_in[key] = _polish_narrative_text(str(priced_in[key]))

    next_actions = payload.get("next_actions")
    if isinstance(next_actions, list):
        for action in next_actions:
            if not isinstance(action, dict):
                continue
            for key in ("query_or_url_hint", "why"):
                if isinstance(action.get(key), str):
                    action[key] = _polish_narrative_text(str(action[key]))

    decision_frame = payload.get("decision_frame")
    if isinstance(decision_frame, dict):
        for key in ("key_risks", "catalysts"):
            rows = decision_frame.get(key)
            if isinstance(rows, list):
                decision_frame[key] = [_polish_narrative_text(str(item)) for item in rows]
    return payload


def _issuer_classification_from_packet(evidence_packet: dict[str, Any]) -> str:
    fundamentals = (
        evidence_packet.get("fundamentals")
        if isinstance(evidence_packet.get("fundamentals"), dict)
        else {}
    )
    if isinstance(fundamentals, dict):
        classification = str(fundamentals.get("issuer_classification") or "").strip().lower()
        if classification:
            return classification
    financials = (
        evidence_packet.get("financials")
        if isinstance(evidence_packet.get("financials"), list)
        else []
    )
    texts = [
        str(((row.get("citation") or {}).get("snippet")) or "")
        for row in financials
        if isinstance(row, dict)
    ]
    line_items = [
        str(row.get("line_item") or "")
        for row in financials
        if isinstance(row, dict) and str(row.get("line_item") or "").strip()
    ]
    texts.extend(
        str(item)
        for item in (
            evidence_packet.get("ticker"),
            (
                (evidence_packet.get("fundamentals") or {})
                if isinstance(evidence_packet.get("fundamentals"), dict)
                else {}
            ).get("issuer_classification"),
        )
        if str(item or "").strip()
    )
    # SIC-first (VOE_ISSUER_CLASSIFICATION_BY_SIC); the substring rule is the fallback.
    return resolve_issuer_classification(
        ticker=evidence_packet.get("ticker"),
        texts=texts,
        line_items=line_items,
    )[0]


def _rewrite_financial_issuer_text(text: str) -> str:
    value = str(text or "")
    replacements = (
        (
            "fcf and fcf_margin are UNKNOWN — cash-flow bridge and working-capital adjustments need extraction.",
            "Traditional FCF-style metrics are sector-limited for bank-like issuers; prioritize balance-sheet mix, capital returns, and regulatory capital extraction instead.",
        ),
        (
            "Given missing FCF/operating-margin inputs, reverse-DCF and DCF outputs are currently UNKNOWN and multiples/DCF ranges are not actionable without further extraction.",
            "Traditional FCF-style DCF outputs are less informative for bank-like issuers; valuation should lean more on funding mix, capital returns, regulatory capital, and asset-liability quality until operating profitability and capital-allocation details are more complete.",
        ),
        (
            "To reconcile CFO to FCF and identify one-off working-capital movements flagged in cash-flow quality disclosures.",
            "To separate recurring funding and liquidity movements from one-off balance-sheet noise and assess capital-return capacity.",
        ),
        (
            "Extract cash flow bridge and working-capital adjustments from recent filings to compute CFO→FCF conversion",
            "Extract cash flow bridge and liquidity disclosures from recent filings to separate recurring funding flows, one-offs, and capital-return capacity",
        ),
        (
            "Valuation is constrained by missing FCF and operating-margin data; reverse-DCF and DCF are UNKNOWN in the packet and multiples guidance warns of insufficient inputs.",
            "Valuation is constrained by incomplete operating-profitability and capital-allocation detail; traditional FCF-style reverse-DCF checks are less informative for bank-like issuers until balance-sheet and capital-return extraction is richer.",
        ),
        (
            "Sustainable free-cash-flow conversion, operating-margin durability, and explicit capital-return amounts (repurchases/dividends) are not priced because FCF, operating_margin, and capital allocation amounts are UNKNOWN or absent from dossier_focus.",
            "Operating-profitability durability, capital-return capacity, and explicit repurchase/dividend amounts remain under-specified; those matter more than a generic FCF bridge for bank-like issuers.",
        ),
        (
            "fcf (UNKNOWN) — missing FCF prevents canonical DCF/multiples inputs",
            "Traditional FCF-style metrics are sector-limited for bank-like issuers; prioritize funding mix, capital returns, and balance-sheet quality instead of a generic FCF bridge.",
        ),
        (
            "operating_margin (UNKNOWN) required to judge margin durability and to run canonical DCF/EPV inputs",
            "Operating-profitability detail is still incomplete; rely more on segment earnings power, capital returns, and balance-sheet quality until bank-appropriate profitability views are richer.",
        ),
        (
            "Extract cash-flow bridge and working-capital adjustments from recent filings to reconcile CFO vs net income",
            "Extract cash-flow bridge and liquidity disclosures from recent filings to separate treasury timing items, recurring funding flows, and capital-return capacity",
        ),
        (
            "large negative CFO in 2025 increases uncertainty around sustainable free cash generation and the feasibility of repeatable repurchases at that scale.",
            "large negative CFO in 2025 increases uncertainty around recurring liquidity generation, funding mix, and the durability of repurchases at that scale.",
        ),
    )
    for old, new in replacements:
        value = value.replace(old, new)
    value = value.replace(
        "Canonical DCF-style per-share valuations are limited by missing FCF and operating-margin inputs",
        "For a large bank, valuation should start with deposits, loans, reserve coverage, funding mix, capital returns, and regulatory capital; traditional FCF-style per-share DCF is secondary here",
    )
    value = value.replace(
        "The dossier contains deterministic facts that guide framing:",
        "The dossier contains bank-native facts that guide framing:",
    )
    value = value.replace(
        "FCF/operating-profit",
        "bank-appropriate profitability and cash-generation",
    )
    value = value.replace(
        "missing FCF and operating-margin inputs",
        "missing bank-appropriate profitability, reserve, and cash-generation inputs",
    )
    return value


def _ensure_financial_field_priority(
    field: str, text: str, *, evidence_packet: dict[str, Any]
) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    lowered = value.lower()
    if field == "business_quality_summary" and not any(
        token in lowered
        for token in ("deposit", "loan", "funding", "reserve", "credit", "balance-sheet")
    ):
        return (
            "For a bank-like issuer, business quality is best judged through funding mix, deposit franchise strength, loan and reserve quality, and capital-return discipline. "
            + value
        )
    if field == "valuation_interpretation" and not any(
        token in lowered
        for token in ("deposit", "loan", "funding", "reserve", "capital ratio", "liquidity")
    ):
        return (
            "For a bank-like issuer, valuation should anchor on deposits, loans, reserve coverage, funding mix, liquidity, and capital-return capacity before treating FCF-style DCF as secondary. "
            + value
        )
    if field == "risk_frame" and not any(
        token in lowered
        for token in ("deposit", "loan", "funding", "reserve", "capital", "liquidity", "credit")
    ):
        return (
            "Bank-specific risk is driven first by funding mix, credit quality, reserve adequacy, liquidity headroom, and regulatory capital discipline. "
            + value
        )
    return value


def _financialize_synthesis_payload(
    payload: dict[str, Any], *, evidence_packet: dict[str, Any]
) -> dict[str, Any]:
    if _issuer_classification_from_packet(evidence_packet) != ISSUER_CLASS_FINANCIAL:
        return payload

    for field in (
        "business_quality_summary",
        "valuation_interpretation",
        "risk_frame",
        "catalyst_frame",
        "confidence_notes",
    ):
        if isinstance(payload.get(field), str):
            payload[field] = _ensure_financial_field_priority(
                field,
                _rewrite_financial_issuer_text(str(payload[field])),
                evidence_packet=evidence_packet,
            )

    for field in ("evidence_gaps", "recommended_next_actions"):
        rows = payload.get(field)
        if isinstance(rows, list):
            payload[field] = [_rewrite_financial_issuer_text(str(item)) for item in rows]

    priced_in = payload.get("priced_in_assessment")
    if isinstance(priced_in, dict):
        for key in ("what_market_assumes", "what_is_not_priced", "uncertainty_notes"):
            if isinstance(priced_in.get(key), str):
                priced_in[key] = _rewrite_financial_issuer_text(str(priced_in[key]))

    next_actions = payload.get("next_actions")
    if isinstance(next_actions, list):
        for action in next_actions:
            if not isinstance(action, dict):
                continue
            for key in ("query_or_url_hint", "why"):
                if isinstance(action.get(key), str):
                    action[key] = _rewrite_financial_issuer_text(str(action[key]))

    claims = payload.get("claims")
    fundamentals = (
        evidence_packet.get("fundamentals")
        if isinstance(evidence_packet.get("fundamentals"), dict)
        else {}
    )
    if isinstance(claims, list) and isinstance(fundamentals, dict):
        for metric, label in (
            ("allowance_for_credit_losses", "Allowance for credit losses"),
            ("provision_for_credit_losses", "Provision for credit losses"),
            ("net_charge_offs", "Net charge-offs"),
            ("nonaccrual_loans", "Nonaccrual loans"),
        ):
            has_metric_claim = any(
                isinstance(claim, dict)
                and (
                    label.lower() in str(claim.get("text") or "").lower()
                    or any(metric in str(ref) for ref in (claim.get("derived_from") or []))
                )
                for claim in claims
            )
            value = _get_num(fundamentals, metric)
            units = _metric_units_from_packet(evidence_packet, metric)
            if has_metric_claim or value is None or units is None:
                continue
            claims.append(
                {
                    "id": f"C_{metric}",
                    "text": f"{label} (latest) {_format_metric_value_for_units(metric, value, units)}",
                    "type": "numeric",
                    "citations": _metric_citations_from_packet(evidence_packet, metric),
                    "derived_from": [f"evidence_packet.fundamentals.{metric}"],
                }
            )

    return payload


def _get_num(mapping: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "UNKNOWN"
    return f"${value:,.0f}m"


def _fmt_pct(value: float | None, *, decimals: int = 1) -> str:
    if value is None:
        return "UNKNOWN"
    return f"{value * 100:.{decimals}f}%"


def _valuation_output_value(
    valuations: dict[str, Any] | None, method: str, key: str
) -> float | None:
    if not isinstance(valuations, dict):
        return None
    payload = valuations.get(method)
    if not isinstance(payload, dict):
        return None
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        return None
    value = outputs.get(key)
    if not isinstance(value, (int, float)):
        nested_outputs = outputs.get("outputs")
        if isinstance(nested_outputs, dict):
            value = nested_outputs.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _deterministic_business_quality_summary(evidence_packet: dict[str, Any]) -> str | None:
    fundamentals = evidence_packet.get("fundamentals")
    if not isinstance(fundamentals, dict):
        return None
    revenue = _get_num(fundamentals, "revenue")
    operating_income = _get_num(fundamentals, "operating_income")
    gross_margin = _get_num(fundamentals, "gross_margin")
    operating_margin = _get_num(fundamentals, "operating_margin") or _get_num(
        fundamentals, "op_margin"
    )
    roic_proxy = _get_num(fundamentals, "roic_proxy")
    r_and_d_total = _get_num(fundamentals, "r_and_d_total")
    sales_marketing_total = _get_num(fundamentals, "sales_marketing_total")
    g_and_a_total = _get_num(fundamentals, "g_and_a_total")
    deferred_revenue_amount = _get_num(fundamentals, "deferred_revenue_amount")
    deferred_revenue_ratio = _get_num(fundamentals, "deferred_revenue_to_revenue_latest")
    rpo_amount = _get_num(fundamentals, "rpo_amount")
    rpo_ratio = _get_num(fundamentals, "rpo_to_revenue_latest")
    repurchases = _get_num(fundamentals, "share_repurchases_amount")
    dividends = _get_num(fundamentals, "dividends_paid_amount")
    segment_count = _get_num(fundamentals, "segment_count")
    customer_concentration_present = fundamentals.get("customer_concentration_present")
    revenue_cagr_3y = _get_num(fundamentals, "revenue_cagr_3y")
    operating_margin_trend = _get_num(fundamentals, "operating_margin_trend_slope")
    dilution_trend = _get_num(fundamentals, "dilution_rate_shares_cagr")

    parts: list[str] = []
    if (
        revenue is not None
        and operating_income is not None
        and gross_margin is not None
        and operating_margin is not None
    ):
        lead = (
            f"{str(evidence_packet.get('ticker') or 'The company')} combines scale with strong profitability: revenue was {_fmt_money(revenue)} and "
            f"operating income was {_fmt_money(operating_income)}, with gross margin at {_fmt_pct(gross_margin)} "
            f"and operating margin at {_fmt_pct(operating_margin)}."
        )
        if roic_proxy is not None:
            lead += f" ROIC proxy remains high at {_fmt_pct(roic_proxy)}."
        parts.append(lead)
    reinvestment_bits = []
    if r_and_d_total is not None:
        reinvestment_bits.append(f"R&D {_fmt_money(r_and_d_total)}")
    if sales_marketing_total is not None:
        reinvestment_bits.append(f"S&M {_fmt_money(sales_marketing_total)}")
    if g_and_a_total is not None:
        reinvestment_bits.append(f"G&A {_fmt_money(g_and_a_total)}")
    if reinvestment_bits:
        parts.append(
            "The reinvestment mix remains balanced, with "
            + ", ".join(
                reinvestment_bits[:-1]
                + (
                    [f"and {reinvestment_bits[-1]}"]
                    if len(reinvestment_bits) > 1
                    else reinvestment_bits
                )
            )
            + ", which supports ongoing product investment without looking commercially bloated."
        )
    if deferred_revenue_amount is not None or rpo_amount is not None:
        visibility_bits = []
        if deferred_revenue_amount is not None:
            visibility_bits.append(
                f"deferred revenue of {_fmt_money(deferred_revenue_amount)} ({_fmt_pct(deferred_revenue_ratio)})"
                if deferred_revenue_ratio is not None
                else f"deferred revenue of {_fmt_money(deferred_revenue_amount)}"
            )
        if rpo_amount is not None:
            visibility_bits.append(
                f"RPO of {_fmt_money(rpo_amount)} ({rpo_ratio:.2f}x revenue)"
                if rpo_ratio is not None
                else f"RPO of {_fmt_money(rpo_amount)}"
            )
        parts.append(
            "Revenue visibility also looks strong, with " + " and ".join(visibility_bits) + "."
        )
    if repurchases is not None or dividends is not None:
        capital_bits = []
        if dividends is not None:
            capital_bits.append(f"dividends of {_fmt_money(dividends)}")
        if repurchases is not None:
            capital_bits.append(f"repurchases of {_fmt_money(repurchases)}")
        parts.append(
            "Capital allocation remains active through " + " and ".join(capital_bits) + "."
        )
    trend_bits: list[str] = []
    if revenue_cagr_3y is not None:
        trend_bits.append(
            f"revenue has compounded at roughly {_fmt_pct(revenue_cagr_3y)} over the last 3 years"
        )
    if operating_margin_trend is not None:
        if operating_margin_trend > 0:
            trend_bits.append("operating margin has improved over time")
        elif operating_margin_trend < 0:
            trend_bits.append("operating margin has softened over time")
    if dilution_trend is not None:
        if dilution_trend < 0:
            trend_bits.append("share count has trended modestly lower")
        elif dilution_trend > 0:
            trend_bits.append("share count has drifted upward")
    if segment_count is not None or customer_concentration_present == 0:
        seg_text = []
        if customer_concentration_present == 0:
            seg_text.append(
                "the filing does not identify a customer above the disclosed concentration threshold"
            )
        if segment_count is not None:
            seg_text.append(f"the business spans {int(segment_count)} reportable segments")
        if seg_text:
            trend_bits.append(" and ".join(seg_text))
    if trend_bits:
        parts.append("At a multi-year level, " + "; ".join(trend_bits) + ".")
    return " ".join(parts).strip() or None


def _deterministic_valuation_interpretation(evidence_packet: dict[str, Any]) -> str | None:
    fundamentals = evidence_packet.get("fundamentals")
    valuations = evidence_packet.get("valuations")
    if not isinstance(fundamentals, dict) or not isinstance(valuations, dict):
        return None
    dcf_base = _valuation_output_value(valuations, "dcf", "base")
    dcf_adjusted = _valuation_output_value(valuations, "dcf_adjusted", "base")
    epv_value = _valuation_output_value(valuations, "epv", "value_per_share")
    epv_adjusted = _valuation_output_value(valuations, "epv_adjusted", "value_per_share")
    graham_value = _valuation_output_value(valuations, "graham", "value_per_share")
    implied_growth = _valuation_output_value(valuations, "reverse_dcf", "implied_growth")
    if _valuation_output_value(valuations, "reverse_dcf", "implied_growth_saturated"):
        implied_growth = None  # clipped bound, not a solve — don't narrate it
    price = None
    reverse_dcf = valuations.get("reverse_dcf")
    if isinstance(reverse_dcf, dict):
        inputs = reverse_dcf.get("inputs")
        if isinstance(inputs, dict):
            price = (
                float(inputs["price"]) if isinstance(inputs.get("price"), (int, float)) else None
            )
    scorecard_signal = None
    scorecard = valuations.get("scorecard")
    if isinstance(scorecard, dict):
        outputs = scorecard.get("outputs")
        if isinstance(outputs, dict) and isinstance(outputs.get("signal"), str):
            scorecard_signal = outputs.get("signal")
    tech_adjustment = valuations.get("tech_adjustment")
    tech_adjustment_outputs = (
        tech_adjustment.get("outputs")
        if isinstance(tech_adjustment, dict) and isinstance(tech_adjustment.get("outputs"), dict)
        else {}
    )
    category_payload = (
        tech_adjustment_outputs.get("category_classification")
        if isinstance(tech_adjustment_outputs.get("category_classification"), dict)
        else {}
    )
    rnd_payload = (
        tech_adjustment_outputs.get("rnd_adjustment")
        if isinstance(tech_adjustment_outputs.get("rnd_adjustment"), dict)
        else {}
    )
    tech_divergence = tech_adjustment_outputs.get("tech_valuation_divergence")
    r_and_d_intensity = _get_num(fundamentals, "r_and_d_intensity_latest")
    deferred_revenue_ratio = _get_num(fundamentals, "deferred_revenue_to_revenue_latest")
    rpo_ratio = _get_num(fundamentals, "rpo_to_revenue_latest")
    repurchases = _get_num(fundamentals, "share_repurchases_amount")
    dividends = _get_num(fundamentals, "dividends_paid_amount")
    operating_margin_trend = _get_num(fundamentals, "operating_margin_trend_slope")

    parts: list[str] = []
    valuation_bits = []
    if dcf_base is not None:
        valuation_bits.append(f"DCF base value is about ${dcf_base:,.2f} per share")
    if dcf_adjusted is not None:
        valuation_bits.append(f"R&D-adjusted DCF is about ${dcf_adjusted:,.2f}")
    if epv_value is not None:
        valuation_bits.append(f"EPV is about ${epv_value:,.2f}")
    if epv_adjusted is not None:
        valuation_bits.append(f"R&D-adjusted EPV is about ${epv_adjusted:,.2f}")
    if graham_value is not None:
        valuation_bits.append(f"Graham value is about ${graham_value:,.2f}")
    if valuation_bits:
        sentence = ", ".join(
            valuation_bits[:-1]
            + ([f"and {valuation_bits[-1]}"] if len(valuation_bits) > 1 else valuation_bits)
        )
        if scorecard_signal:
            sentence += f", while the scorecard reads {scorecard_signal}"
        sentence += "."
        parts.append(sentence)
    if (
        dcf_base is not None
        and dcf_adjusted is not None
        and isinstance(tech_divergence, (int, float))
        and category_payload.get("category")
        and isinstance(rnd_payload.get("amortization_life"), int)
    ):
        parts.append(
            f"On the tech-adjusted track, capitalizing R&D with a {int(rnd_payload['amortization_life'])}-year life for the "
            f"{category_payload.get('category')} route lifts DCF from ${dcf_base:,.2f} to ${dcf_adjusted:,.2f} per share, "
            f"a {float(tech_divergence):+.1%} divergence."
        )
    if implied_growth is not None:
        growth_text = f"Reverse DCF still implies roughly {_fmt_pct(implied_growth)} growth"
        if price is not None:
            growth_text += f" to support the price input of ${price:,.2f}"
        growth_text += ", which leaves limited room for operational disappointment."
        parts.append(growth_text)
    support_bits = []
    if r_and_d_intensity is not None:
        support_bits.append(f"R&D intensity remains meaningful at {_fmt_pct(r_and_d_intensity)}")
    if deferred_revenue_ratio is not None:
        support_bits.append(
            f"deferred revenue covers about {_fmt_pct(deferred_revenue_ratio)} of annual revenue"
        )
    if rpo_ratio is not None:
        support_bits.append(f"RPO stands at roughly {rpo_ratio:.2f}x revenue")
    if repurchases is not None or dividends is not None:
        capital_return = []
        if dividends is not None:
            capital_return.append(f"dividends of {_fmt_money(dividends)}")
        if repurchases is not None:
            capital_return.append(f"repurchases of {_fmt_money(repurchases)}")
        support_bits.append("capital returns remain active through " + " and ".join(capital_return))
    if support_bits:
        parts.append(
            "Those premium-quality markers help justify a higher-quality multiple, but not necessarily the current market-implied growth bar: "
            + "; ".join(support_bits)
            + "."
        )
    if operating_margin_trend is not None:
        if operating_margin_trend > 0:
            parts.append(
                "Multi-year margin direction is favorable, which supports the durability case even if valuation still looks demanding."
            )
        elif operating_margin_trend < 0:
            parts.append(
                "Multi-year margin direction is less favorable, which makes the current valuation bar harder to underwrite."
            )
    return " ".join(parts).strip() or None


def _fallback_hypotheses(evidence_packet: dict[str, Any]) -> list[dict[str, Any]]:
    fundamentals = evidence_packet.get("fundamentals")
    if not isinstance(fundamentals, dict):
        fundamentals = {}

    deferred_revenue_amount = _get_num(fundamentals, "deferred_revenue_amount")
    deferred_revenue_ratio = _get_num(fundamentals, "deferred_revenue_to_revenue_latest")
    rpo_amount = _get_num(fundamentals, "rpo_amount")
    rpo_ratio = _get_num(fundamentals, "rpo_to_revenue_latest")
    r_and_d_intensity = _get_num(fundamentals, "r_and_d_intensity_latest")
    operating_margin = _get_num(fundamentals, "operating_margin") or _get_num(
        fundamentals, "op_margin"
    )
    operating_margin_trend = _get_num(fundamentals, "operating_margin_trend_slope")

    if deferred_revenue_amount is not None or rpo_amount is not None:
        support_bits: list[str] = []
        if deferred_revenue_amount is not None:
            support_bits.append(
                f"deferred revenue is {_fmt_money(deferred_revenue_amount)}"
                + (
                    f" ({_fmt_pct(deferred_revenue_ratio)} of annual revenue)"
                    if deferred_revenue_ratio is not None
                    else ""
                )
            )
        if rpo_amount is not None:
            support_bits.append(
                f"RPO is {_fmt_money(rpo_amount)}"
                + (f" ({rpo_ratio:.2f}x revenue)" if rpo_ratio is not None else "")
            )
        return [
            {
                "id": "H_fallback_visibility",
                "statement": "Reported backlog-style metrics support forward revenue visibility.",
                "why_it_might_be_true": " and ".join(support_bits)
                + " based on evidence_packet.dossier_focus.revenue_visibility.",
                "falsifiers": [
                    "Subsequent filings show a sharp drop in deferred revenue or RPO.",
                    "Management discloses cancellations or materially weaker conversion of backlog into revenue.",
                ],
                "required_evidence": [
                    "Confirm deferred revenue and RPO trends in the next 10-Q or 10-K.",
                    "Check contract footnotes for timing, cancellations, or non-convertible backlog.",
                ],
            }
        ]

    if r_and_d_intensity is not None or operating_margin is not None:
        support_bits = []
        if r_and_d_intensity is not None:
            support_bits.append(f"R&D intensity is {_fmt_pct(r_and_d_intensity)}")
        if operating_margin is not None:
            support_bits.append(f"operating margin is {_fmt_pct(operating_margin)}")
        if operating_margin_trend is not None:
            if operating_margin_trend > 0:
                support_bits.append("operating margin trend remains favorable")
            elif operating_margin_trend < 0:
                support_bits.append("operating margin trend has softened")
        return [
            {
                "id": "H_fallback_durability",
                "statement": "Current reinvestment levels support durable margins rather than one-off profitability.",
                "why_it_might_be_true": " and ".join(support_bits)
                + " based on evidence_packet.fundamentals.",
                "falsifiers": [
                    "Margins compress materially despite stable or higher reinvestment.",
                    "Future filings reveal one-off items inflated current profitability.",
                ],
                "required_evidence": [
                    "Review segment margin and expense disclosures in the next filing.",
                    "Check non-GAAP reconciliations and notes for one-time benefits.",
                ],
            }
        ]

    return [
        {
            "id": "H_fallback_validation",
            "statement": "Additional filing work is needed to validate the current valuation and quality frame.",
            "why_it_might_be_true": "The synthesis packet omitted model-generated hypotheses, so this fallback uses deterministic evidence only.",
            "falsifiers": [
                "A subsequent rerun produces grounded hypotheses directly from the model output.",
            ],
            "required_evidence": [
                "Rerun synthesis and compare against the deterministic dossier facts.",
            ],
        }
    ]


def _fallback_next_actions(evidence_packet: dict[str, Any]) -> list[dict[str, Any]]:
    fundamentals = evidence_packet.get("fundamentals")
    if not isinstance(fundamentals, dict):
        fundamentals = {}

    actions = [
        {
            "action_type": "pull_filings",
            "target_source": "EDGAR",
            "query_or_url_hint": "Pull the latest 10-Q/10-K and relevant footnotes for revenue, margins, and cash flow.",
            "why": "Validate whether recent filing detail supports the current quality and valuation frame.",
        }
    ]

    if (
        _get_num(fundamentals, "deferred_revenue_amount") is not None
        or _get_num(fundamentals, "rpo_amount") is not None
    ):
        actions.append(
            {
                "action_type": "check_revenue_visibility",
                "target_source": "EDGAR",
                "query_or_url_hint": "Review revenue recognition and contract liability footnotes for deferred revenue and RPO trends.",
                "why": "Confirm whether backlog and contract liabilities continue to support forward revenue visibility.",
            }
        )
    return actions


def _fallback_claims(evidence_packet: dict[str, Any]) -> list[dict[str, Any]]:
    fundamentals = evidence_packet.get("fundamentals")
    if not isinstance(fundamentals, dict):
        fundamentals = {}

    claims: list[dict[str, Any]] = []
    for metric, label, formatter in (
        ("revenue", "Revenue", "money"),
        ("fcf", "Free cash flow", "money"),
        ("operating_margin", "Operating margin", "percent"),
        ("market_price", "Market price", "price"),
    ):
        value = _get_num(fundamentals, metric)
        if value is None:
            continue
        if formatter == "percent":
            formatted_value = _fmt_pct(value)
        elif formatter == "price":
            formatted_value = f"${value:,.2f}".rstrip("0").rstrip(".")
        else:
            formatted_value = _fmt_money(value)
        claims.append(
            {
                "id": f"c_deterministic_{metric}",
                "text": f"{label} latest {formatted_value}",
                "type": "numeric",
                "citations": _metric_citations_from_packet(evidence_packet, metric),
                "derived_from": [f"evidence_packet.fundamentals.{metric}"],
            }
        )

    if claims:
        return claims

    return [
        {
            "id": "c_deterministic_packet_available",
            "text": "A deterministic evidence packet was available for synthesis.",
            "type": "non_numeric",
            "citations": [],
            "derived_from": ["evidence_packet"],
        }
    ]


def _fallback_priced_in_assessment(evidence_packet: dict[str, Any]) -> dict[str, str]:
    valuation_context = _deterministic_valuation_interpretation(evidence_packet)
    return {
        "what_market_assumes": "Market expectations require manual interpretation because the model omitted its priced-in assessment.",
        "what_is_not_priced": "Potential mispricing depends on validating deterministic valuation, quality, and filing evidence gaps.",
        "uncertainty_notes": valuation_context
        or "Model output was incomplete, so deterministic evidence remains the source of truth until synthesis is rerun.",
    }


def _fallback_decision_frame(evidence_packet: dict[str, Any]) -> dict[str, Any]:
    fundamentals = evidence_packet.get("fundamentals")
    if not isinstance(fundamentals, dict):
        fundamentals = {}
    risk_items = ["model_output_missing_decision_frame"]
    if _get_num(fundamentals, "fcf") is None:
        risk_items.append("missing_free_cash_flow")
    if (
        _get_num(fundamentals, "operating_margin") is None
        and _get_num(fundamentals, "op_margin") is None
    ):
        risk_items.append("missing_operating_margin")
    return {
        "stance": "watchlist",
        "key_risks": risk_items,
        "catalysts": ["next filing update", "complete model synthesis rerun"],
        "time_horizon_days": 90,
    }


def _finalize_synthesis_payload(
    payload: dict[str, Any],
    *,
    evidence_packet: dict[str, Any],
    analysis_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    polished = _polish_synthesis_payload(
        payload,
        evidence_packet=evidence_packet,
        analysis_context=analysis_context,
    )
    polished = _financialize_synthesis_payload(
        polished,
        evidence_packet=evidence_packet,
    )
    deterministic_business_quality = _deterministic_business_quality_summary(evidence_packet)
    if deterministic_business_quality and _needs_deterministic_backfill(
        polished.get("business_quality_summary")
    ):
        polished["business_quality_summary"] = deterministic_business_quality
    deterministic_valuation = _deterministic_valuation_interpretation(evidence_packet)
    if deterministic_valuation and _needs_deterministic_backfill(
        polished.get("valuation_interpretation")
    ):
        polished["valuation_interpretation"] = deterministic_valuation
    if not isinstance(polished.get("hypotheses"), list) or not polished.get("hypotheses"):
        polished["hypotheses"] = _fallback_hypotheses(evidence_packet)
    if not isinstance(polished.get("next_actions"), list) or not polished.get("next_actions"):
        polished["next_actions"] = _fallback_next_actions(evidence_packet)
    if not isinstance(polished.get("priced_in_assessment"), dict):
        polished["priced_in_assessment"] = _fallback_priced_in_assessment(evidence_packet)
        evidence_gaps = polished.get("evidence_gaps")
        if not isinstance(evidence_gaps, list):
            evidence_gaps = []
            polished["evidence_gaps"] = evidence_gaps
        if "model_output_missing_priced_in_assessment" not in evidence_gaps:
            evidence_gaps.append("model_output_missing_priced_in_assessment")
    if not isinstance(polished.get("decision_frame"), dict):
        polished["decision_frame"] = _fallback_decision_frame(evidence_packet)
        evidence_gaps = polished.get("evidence_gaps")
        if not isinstance(evidence_gaps, list):
            evidence_gaps = []
            polished["evidence_gaps"] = evidence_gaps
        if "model_output_missing_decision_frame" not in evidence_gaps:
            evidence_gaps.append("model_output_missing_decision_frame")
    if not isinstance(polished.get("claims"), list) or not polished.get("claims"):
        polished["claims"] = _fallback_claims(evidence_packet)
        evidence_gaps = polished.get("evidence_gaps")
        if not isinstance(evidence_gaps, list):
            evidence_gaps = []
            polished["evidence_gaps"] = evidence_gaps
        if "model_output_missing_claims" not in evidence_gaps:
            evidence_gaps.append("model_output_missing_claims")
    return polished


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _disabled_fallback_packet(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
    model: str,
    prompt_hash: str,
    input_hash: str,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "run_id": run_id,
        "business_quality_summary": "UNKNOWN (synthesis provider disabled; no model-based qualitative interpretation was generated).",
        "valuation_interpretation": "UNKNOWN (provider disabled); deterministic valuation outputs remain authoritative.",
        "risk_frame": "Primary current risk is interpretive incompleteness because the qualitative layer is disabled.",
        "catalyst_frame": "Catalyst framing deferred until model-based synthesis is enabled.",
        "evidence_gaps": ["LLM provider disabled"],
        "recommended_next_actions": [
            "Enable model provider and rerun synthesis for a grounded Layer 3 overlay."
        ],
        "confidence_notes": "Fallback packet only. Deterministic evidence and valuation remain the source of truth.",
        "hypotheses": [
            {
                "id": "h_disabled_provider",
                "statement": "Synthesis generated in deterministic fallback mode because LLM provider is disabled.",
                "why_it_might_be_true": "Configured provider is disabled, so no model inference call was made.",
                "falsifiers": [
                    "Enable a structured LLM provider and compare model-generated synthesis output."
                ],
                "required_evidence": [
                    "Set VOE_LLM_PROVIDER to openai or anthropic and rerun synthesis."
                ],
            }
        ],
        "claims": [
            {
                "id": "c_disabled_provider",
                "text": "Provider is disabled in local configuration.",
                "type": "non_numeric",
                "citations": [],
                "derived_from": ["config.llm_provider"],
            }
        ],
        "priced_in_assessment": {
            "what_market_assumes": "UNKNOWN (synthesis provider disabled)",
            "what_is_not_priced": "UNKNOWN (synthesis provider disabled)",
            "uncertainty_notes": "Deterministic fallback packet generated without live model call.",
        },
        "next_actions": [
            {
                "action_type": "config_update",
                "target_source": "local_config",
                "query_or_url_hint": "VOE_LLM_PROVIDER=openai or VOE_LLM_PROVIDER=anthropic",
                "why": "Enable model-based synthesis on next run.",
            }
        ],
        "decision_frame": {
            "stance": "watchlist",
            "key_risks": ["LLM synthesis provider disabled"],
            "catalysts": ["rerun synthesis after enabling provider"],
            "time_horizon_days": 90,
        },
        "llm_meta": {
            "model": model,
            "prompt_hash": prompt_hash,
            "input_hash": input_hash,
            "cost_estimate_usd": 0.0,
            "created_at": utc_now_iso(),
        },
    }


def _apply_as_of_resolution(
    payload: dict[str, Any],
    *,
    requested_as_of_date: str,
    effective_as_of_date: str,
) -> dict[str, Any]:
    resolved = dict(payload)
    resolution = "exact" if requested_as_of_date == effective_as_of_date else "fallback"
    resolved["requested_as_of_date"] = requested_as_of_date
    resolved["effective_as_of_date"] = effective_as_of_date
    resolved["as_of_resolution"] = resolution
    resolved["as_of_resolution_reason"] = (
        "Exact evidence packet matched requested as-of date."
        if resolution == "exact"
        else "Used the latest available evidence packet at or before the requested as-of date."
    )
    return resolved


def resolve_synthesis_as_of(ticker: str, *, as_of_date: str) -> dict[str, Any] | None:
    ticker = ticker.upper()
    requested_as_of_date = str(as_of_date or "")
    with get_db() as conn:
        evidence_packet = _latest_evidence_packet(conn, ticker, requested_as_of_date)
    if not evidence_packet:
        return None
    effective_as_of_date = str(evidence_packet.get("as_of_date") or requested_as_of_date)
    return {
        "ticker": ticker,
        "requested_as_of_date": requested_as_of_date,
        "effective_as_of_date": effective_as_of_date,
        "as_of_resolution": "exact" if requested_as_of_date == effective_as_of_date else "fallback",
        "as_of_resolution_reason": (
            "Exact evidence packet matched requested as-of date."
            if requested_as_of_date == effective_as_of_date
            else "Used the latest available evidence packet at or before the requested as-of date."
        ),
    }


def _existing_packet_for_run(
    conn,
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
    input_hash: str,
    prompt_hash: str,
    model: str,
    provider: str,
) -> Path | None:
    row = conn.execute(
        """
        SELECT packet_path, input_hash, prompt_hash, model, provider
        FROM synthesis_packets
        WHERE ticker = ? AND as_of_date = ? AND run_id = ?
        LIMIT 1
        """,
        (ticker, as_of_date, run_id),
    ).fetchone()
    if not row:
        return None
    if (
        row["input_hash"] == input_hash
        and row["prompt_hash"] == prompt_hash
        and row["model"] == model
        and row["provider"] == provider
    ):
        path = Path(row["packet_path"])
        if path.exists():
            return path
    return None


def _cached_packet_by_hash(
    conn,
    *,
    input_hash: str,
    prompt_hash: str,
    model: str,
    provider: str,
) -> tuple[dict[str, Any] | None, str | None]:
    row = conn.execute(
        """
        SELECT packet_json, run_id
        FROM synthesis_packets
        WHERE input_hash = ? AND prompt_hash = ? AND model = ? AND provider = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (input_hash, prompt_hash, model, provider),
    ).fetchone()
    if not row:
        return None, None
    payload = json.loads(row["packet_json"])
    if not isinstance(payload, dict):
        return None, None
    return payload, row["run_id"]


def _persist_synthesis_packet(
    *,
    payload: dict[str, Any],
    ticker: str,
    as_of_date: str,
    run_id: str,
    input_hash: str,
    prompt_hash: str,
    model: str,
    provider: str,
    usage: dict[str, Any],
    cost_estimate_usd: float,
    from_cache: bool,
    paid_invocation_id: str | None = None,
) -> Path:
    cfg = get_config()
    cfg.synthesis_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.synthesis_dir / f"{ticker}_{as_of_date}_run_{run_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    packet_hash = sha256_file(out_path)

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO synthesis_packets(
                ticker, as_of_date, run_id, packet_path, packet_hash, packet_json,
                prompt_hash, input_hash, provider, model, usage_json, cost_estimate_usd,
                paid_invocation_id, from_cache, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, as_of_date, run_id) DO UPDATE SET
                packet_path=excluded.packet_path,
                packet_hash=excluded.packet_hash,
                packet_json=excluded.packet_json,
                prompt_hash=excluded.prompt_hash,
                input_hash=excluded.input_hash,
                provider=excluded.provider,
                model=excluded.model,
                usage_json=excluded.usage_json,
                cost_estimate_usd=excluded.cost_estimate_usd,
                paid_invocation_id=excluded.paid_invocation_id,
                from_cache=excluded.from_cache,
                created_at=excluded.created_at
            """,
            (
                ticker,
                as_of_date,
                run_id,
                str(out_path),
                packet_hash,
                json.dumps(payload),
                prompt_hash,
                input_hash,
                provider,
                model,
                json.dumps(usage),
                float(cost_estimate_usd),
                paid_invocation_id,
                1 if from_cache else 0,
                utc_now_iso(),
            ),
        )
    return out_path


def _persist_synthesis_paid_attempts(
    *,
    invocation_id: str,
    ticker: str,
    as_of_date: str,
    run_id: str,
    prompt_hash: str,
    input_hash: str,
    provider: str,
    model: str,
    schema_name: str,
    usage_records: list[dict[str, Any]],
) -> None:
    """Persist each physical paid response exactly once before publication."""

    if not usage_records:
        return
    normalized_rows: list[tuple[Any, ...]] = []
    created_at = utc_now_iso()
    for physical_sequence, usage_record in enumerate(usage_records, start=1):
        raw_cost = usage_record.get("cost_estimate_usd")
        if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
            raise ValueError("synthesis paid-attempt cost must be numeric")
        cost_estimate_usd = float(raw_cost)
        if not math.isfinite(cost_estimate_usd) or cost_estimate_usd < 0.0:
            raise ValueError("synthesis paid-attempt cost must be finite and non-negative")
        status = str(usage_record.get("status") or "").strip().upper()
        if status not in {"OK", "INCOMPLETE", "ERROR"}:
            raise ValueError(f"unsupported synthesis paid-attempt status: {status!r}")
        normalized_rows.append(
            (
                f"{invocation_id}:{physical_sequence}",
                invocation_id,
                physical_sequence,
                ticker,
                as_of_date,
                run_id,
                prompt_hash,
                input_hash,
                str(usage_record.get("provider") or provider),
                str(usage_record.get("model") or model),
                str(usage_record.get("schema_name") or schema_name),
                status,
                json.dumps(usage_record, sort_keys=True),
                cost_estimate_usd,
                created_at,
            )
        )

    with get_db() as conn:
        conn.executemany(
            """
            INSERT INTO synthesis_paid_attempts(
                attempt_id, invocation_id, physical_sequence, ticker, as_of_date,
                run_id, prompt_hash, input_hash, provider, model, schema_name,
                status, usage_json, cost_estimate_usd, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            normalized_rows,
        )


def run_synthesis_for_ticker(
    ticker: str,
    *,
    as_of_date: str,
    run_id: str,
    strict_as_of: bool = False,
) -> Path | None:
    return _run_synthesis_for_ticker_impl(
        ticker=ticker,
        as_of_date=as_of_date,
        run_id=run_id,
        allow_cache=True,
        strict_as_of=strict_as_of,
    )


def _run_synthesis_for_ticker_impl(
    ticker: str,
    *,
    as_of_date: str,
    run_id: str,
    allow_cache: bool,
    strict_as_of: bool,
) -> Path | None:
    cfg = get_config()
    ticker = ticker.upper()
    run_as_of_date = as_of_date

    with get_db() as conn:
        evidence_packet = _latest_evidence_packet(conn, ticker, run_as_of_date)
        if not evidence_packet:
            return None
        effective_as_of_date = str(evidence_packet.get("as_of_date") or run_as_of_date)
        if strict_as_of and effective_as_of_date != run_as_of_date:
            logger.warning(
                "synthesis_strict_as_of_miss",
                extra={
                    "stage_name": "synthesis",
                    "stage_ticker": ticker,
                    "stage_requested_as_of": run_as_of_date,
                    "stage_effective_as_of": effective_as_of_date,
                },
            )
            return None
        delta_payload = _latest_delta(conn, ticker, run_id)
    _require_prompt_evidence_financial_integrity(
        ticker=ticker,
        run_as_of_date=effective_as_of_date,
        evidence_packet=evidence_packet,
    )
    financial_context = build_canonical_v1_financial_context(
        tickers=[ticker],
        as_of_date=effective_as_of_date,
        db_path=cfg.db_path,
    )
    integrity_scope = financial_context.scope(context=f"synthesis_agent:{ticker}:{run_id}")
    # Fail before cache publication, provider-disabled fallback prose, or any
    # paid attempt. Missing canonical quote/cap/share lineage is not an
    # ordinary synthesis miss and must never degrade into substantive output.
    require_financial_integrity_scope(integrity_scope)

    provider = get_llm_provider()
    provider_name = getattr(provider, "provider_name", "disabled")
    model = _configured_provider_model(cfg, provider_name)
    canonical_packet = financial_context.packets[ticker]
    analysis_snapshot = load_ops_analysis_snapshot(
        ticker, as_of_date=effective_as_of_date, run_id=run_id
    )
    if _analysis_context_needs_refresh(
        evidence_packet=evidence_packet,
        analysis_snapshot=analysis_snapshot,
        run_id=run_id,
    ):
        refreshed_snapshot = _refresh_analysis_context(
            ticker=ticker,
            as_of_date=effective_as_of_date,
            run_id=run_id,
        )
        if refreshed_snapshot is not None:
            analysis_snapshot = refreshed_snapshot
    analysis_context = snapshot_to_synthesis_context(analysis_snapshot)

    schema = synthesis_schema_for_prompt()
    inputs = _build_inputs(
        ticker=ticker,
        as_of_date=effective_as_of_date,
        evidence_packet=evidence_packet,
        canonical_financial_context=dict(vars(canonical_packet)),
        analysis_context=analysis_context,
        delta_payload=delta_payload,
    )
    _require_prompt_point_in_time_dates(
        ticker=ticker,
        run_as_of_date=effective_as_of_date,
        payload=inputs,
    )
    input_hash = sha256_text(json.dumps(inputs, sort_keys=True, separators=(",", ":")))
    prompt = _build_prompt(inputs, schema)
    prompt_hash = sha256_text(prompt)

    def current_financial_scenario() -> dict[str, Any]:
        return financial_input_scenario(
            canonical_packet,
            financial_inputs={
                "synthesis_inputs": inputs,
                "publication_evidence_packet": evidence_packet,
                "publication_analysis_context": analysis_context,
                "publication_delta_payload": delta_payload,
                "provider_prompt": prompt,
                "provider_schema": schema,
                "provider_schema_name": "synthesis_packet_v1",
                "provider_name": provider_name,
                "provider_model": model,
                "provider_max_output_tokens": (
                    _provider_max_output_tokens(
                        cfg,
                        provider_name,
                    )
                ),
            },
        )

    financial_scenario = current_financial_scenario()
    bound_financial_scope = bind_v1_financial_scope(
        context=f"synthesis_agent:{ticker}:{run_id}",
        run_as_of_date=effective_as_of_date,
        packets=(canonical_packet,),
        scenarios=(financial_scenario,),
    )

    def require_exact_scope(_attempt: dict[str, Any] | None = None) -> None:
        bound_financial_scope.require(
            scenarios=(current_financial_scenario(),),
        )

    with get_db() as conn:
        if allow_cache:
            existing = _existing_packet_for_run(
                conn,
                ticker=ticker,
                as_of_date=effective_as_of_date,
                run_id=run_id,
                input_hash=input_hash,
                prompt_hash=prompt_hash,
                model=model,
                provider=provider_name,
            )
            if existing:
                require_exact_scope()
                return existing

            cached_payload, cached_run_id = _cached_packet_by_hash(
                conn,
                input_hash=input_hash,
                prompt_hash=prompt_hash,
                model=model,
                provider=provider_name,
            )
            if cached_payload:
                cached_payload["ticker"] = ticker
                cached_payload["as_of_date"] = effective_as_of_date
                cached_payload["run_id"] = run_id
                llm_meta = cached_payload.get("llm_meta") or {}
                llm_meta["model"] = model
                llm_meta["prompt_hash"] = prompt_hash
                llm_meta["input_hash"] = input_hash
                llm_meta["cost_estimate_usd"] = 0.0
                llm_meta["created_at"] = utc_now_iso()
                cached_payload["llm_meta"] = llm_meta
                cached_payload = _apply_as_of_resolution(
                    cached_payload,
                    requested_as_of_date=run_as_of_date,
                    effective_as_of_date=effective_as_of_date,
                )
                cached_payload = _finalize_synthesis_payload(
                    cached_payload,
                    evidence_packet=evidence_packet,
                    analysis_context=analysis_context,
                )
                valid_packet = validate_synthesis_packet(cached_payload)
                require_exact_scope()
                return _persist_synthesis_packet(
                    payload=valid_packet.model_dump(mode="json"),
                    ticker=ticker,
                    as_of_date=effective_as_of_date,
                    run_id=run_id,
                    input_hash=input_hash,
                    prompt_hash=prompt_hash,
                    model=model,
                    provider=provider_name,
                    usage={"cached_from_run_id": cached_run_id},
                    cost_estimate_usd=0.0,
                    from_cache=True,
                )

    with get_db() as conn:
        spent = _run_spend_usd(conn, run_id)
        estimated_input_tokens = _estimate_tokens_from_text(prompt)
        estimated_max_cost = _estimate_cost_usd(
            model,
            estimated_input_tokens,
            _provider_max_output_tokens(cfg, provider_name),
            provider_name=provider_name,
        )
        provider_budget_usd = _provider_budget_usd(cfg, provider_name)
        if spent + estimated_max_cost > provider_budget_usd:
            logger.warning(
                "synthesis_budget_skip",
                extra={
                    "stage_name": "synthesis",
                    "stage_ticker": ticker,
                    "stage_run_id": run_id,
                    "stage_spent_usd": spent,
                    "stage_estimated_max_cost_usd": estimated_max_cost,
                    "stage_provider": provider_name,
                    "stage_budget_usd": provider_budget_usd,
                },
            )
            return None

    if not provider.enabled():
        if provider_name != "disabled":
            raise RuntimeError(
                f"LLM provider '{provider_name}' is configured but not enabled. "
                f"{_provider_enablement_hint(provider_name)}"
            )
        fallback_payload = _disabled_fallback_packet(
            ticker=ticker,
            as_of_date=effective_as_of_date,
            run_id=run_id,
            model=model,
            prompt_hash=prompt_hash,
            input_hash=input_hash,
        )
        fallback_payload = _apply_as_of_resolution(
            fallback_payload,
            requested_as_of_date=run_as_of_date,
            effective_as_of_date=effective_as_of_date,
        )
        packet: SynthesisPacket = validate_synthesis_packet(fallback_payload)
        require_exact_scope()
        return _persist_synthesis_packet(
            payload=packet.model_dump(mode="json"),
            ticker=ticker,
            as_of_date=effective_as_of_date,
            run_id=run_id,
            input_hash=input_hash,
            prompt_hash=prompt_hash,
            model=model,
            provider=provider_name,
            usage={"reason": "provider_disabled_fallback"},
            cost_estimate_usd=0.0,
            from_cache=False,
        )

    # Import lazily because usage_capture shares the canonical pricing helper
    # from this module.
    from app.llm.usage_capture import (
        attach_provider_usage_to_exception,
        provider_failed_attempt_capture,
        provider_usage_records,
        provider_usage_records_from_exception,
        provider_usage_request,
        record_provider_usage,
    )

    schema_name = "synthesis_packet_v1"
    max_output_tokens = _provider_max_output_tokens(cfg, provider_name)
    failed_attempts: list[dict[str, Any]] = []
    successful_attempts: list[dict[str, Any]] = []
    paid_invocation_id = uuid4().hex
    paid_attempts_persistence_started = False

    def persist_paid_attempts() -> None:
        nonlocal paid_attempts_persistence_started
        if paid_attempts_persistence_started:
            return
        usage_records = [*failed_attempts, *successful_attempts]
        if not usage_records:
            return
        _persist_synthesis_paid_attempts(
            invocation_id=paid_invocation_id,
            ticker=ticker,
            as_of_date=effective_as_of_date,
            run_id=run_id,
            prompt_hash=prompt_hash,
            input_hash=input_hash,
            provider=provider_name,
            model=model,
            schema_name=schema_name,
            usage_records=usage_records,
        )
        paid_attempts_persistence_started = True

    try:
        require_exact_scope()
        with provider_usage_request(
            provider=provider,
            prompt=prompt,
            schema=schema,
            schema_name=schema_name,
            max_output_tokens=max_output_tokens,
        ) as request_kwargs:
            try:
                with (
                    provider_failed_attempt_capture(
                        provider=provider,
                        prompt=prompt,
                        schema_name=schema_name,
                        estimated_output_tokens=max_output_tokens,
                    ) as failed_attempts,
                    llm_physical_attempt_guard(require_exact_scope),
                ):
                    result = provider.synthesize_json(
                        prompt=prompt,
                        schema=schema,
                        schema_name=schema_name,
                        **request_kwargs,
                    )
            except BaseException as exc:
                successful_attempts = provider_usage_records_from_exception(
                    provider=provider,
                    error=exc,
                    prompt=prompt,
                    schema_name=schema_name,
                )
                for usage_record in successful_attempts:
                    record_provider_usage(usage_record)
                attach_provider_usage_to_exception(exc, [*failed_attempts, *successful_attempts])
                try:
                    require_exact_scope()
                except InvalidFinancialInputError as integrity_exc:
                    attach_provider_usage_to_exception(
                        integrity_exc,
                        [*failed_attempts, *successful_attempts],
                    )
                    raise integrity_exc from exc
                raise
            successful_attempts = provider_usage_records(
                provider=provider,
                result=result,
                prompt=prompt,
                schema_name=schema_name,
            )
            for usage_record in successful_attempts:
                record_provider_usage(usage_record)

        persist_paid_attempts()
        payload = json.loads(result.json_text)
        if not isinstance(payload, dict):
            raise RuntimeError("synthesis output was not a JSON object")
        payload["ticker"] = ticker
        payload["as_of_date"] = effective_as_of_date
        payload["run_id"] = run_id
        payload = _apply_as_of_resolution(
            payload,
            requested_as_of_date=run_as_of_date,
            effective_as_of_date=effective_as_of_date,
        )

        usage_input_tokens = result.usage_input_tokens or _estimate_tokens_from_text(prompt)
        usage_output_tokens = result.usage_output_tokens or _estimate_tokens_from_text(
            result.json_text
        )
        raw_cached_input_tokens = getattr(result, "usage_cached_input_tokens", None)
        usage_cached_input_tokens = (
            min(max(0, int(raw_cached_input_tokens)), int(usage_input_tokens))
            if isinstance(raw_cached_input_tokens, int)
            else 0
        )
        cost_estimate_usd = round(
            sum(
                float(record.get("cost_estimate_usd") or 0.0)
                for record in [*failed_attempts, *successful_attempts]
            ),
            6,
        )
        payload["llm_meta"] = {
            "model": model,
            "prompt_hash": prompt_hash,
            "input_hash": input_hash,
            "cost_estimate_usd": cost_estimate_usd,
            "created_at": utc_now_iso(),
        }
        payload = _finalize_synthesis_payload(
            payload,
            evidence_packet=evidence_packet,
            analysis_context=analysis_context,
        )

        # Enforce numeric trace rule before strict schema parse for clearer errors.
        numeric_trace_failures = check_numeric_claim_trace(payload)
        if numeric_trace_failures:
            raise RuntimeError("; ".join(numeric_trace_failures))
        packet: SynthesisPacket = validate_synthesis_packet(payload)

        with get_db() as conn:
            spent = _run_spend_usd(conn, run_id)
            provider_budget_usd = _provider_budget_usd(cfg, provider_name)
            if spent > provider_budget_usd:
                logger.warning(
                    "synthesis_budget_post_call_exceeded",
                    extra={
                        "stage_name": "synthesis",
                        "stage_ticker": ticker,
                        "stage_run_id": run_id,
                        "stage_spent_usd": spent,
                        "stage_call_cost_usd": cost_estimate_usd,
                        "stage_provider": provider_name,
                        "stage_budget_usd": provider_budget_usd,
                    },
                )
                return None

        require_exact_scope()
        return _persist_synthesis_packet(
            payload=packet.model_dump(mode="json"),
            ticker=ticker,
            as_of_date=effective_as_of_date,
            run_id=run_id,
            input_hash=input_hash,
            prompt_hash=prompt_hash,
            model=model,
            provider=provider_name,
            usage={
                "input_tokens": usage_input_tokens,
                "cached_input_tokens": usage_cached_input_tokens,
                "output_tokens": usage_output_tokens,
            },
            cost_estimate_usd=cost_estimate_usd,
            from_cache=False,
            paid_invocation_id=paid_invocation_id,
        )
    except BaseException as exc:
        persist_paid_attempts()
        attach_provider_usage_to_exception(exc, [*failed_attempts, *successful_attempts])
        raise


def run_synthesis_for_ticker_no_cache(
    ticker: str,
    *,
    as_of_date: str,
    run_id: str,
    strict_as_of: bool = False,
) -> Path | None:
    return _run_synthesis_for_ticker_impl(
        ticker=ticker,
        as_of_date=as_of_date,
        run_id=run_id,
        allow_cache=False,
        strict_as_of=strict_as_of,
    )


def run_synthesis_for_scope(
    *,
    as_of_date: str,
    run_id: str,
    tickers: list[str] | None = None,
    limit: int | None = None,
    allow_cache: bool = True,
    strict_as_of: bool = False,
) -> dict[str, Any]:
    tickers = [t.upper() for t in (tickers or []) if t.strip()]
    if not tickers:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT ticker
                FROM scores
                WHERE run_id = ?
                ORDER BY is_candidate DESC, total_score DESC, ticker ASC
                """,
                (run_id,),
            ).fetchall()
            tickers = [row["ticker"] for row in rows]
    if limit is not None and limit > 0:
        tickers = tickers[:limit]

    built = 0
    paths: list[str] = []
    for ticker in tickers:
        path = _run_synthesis_for_ticker_impl(
            ticker,
            as_of_date=as_of_date,
            run_id=run_id,
            allow_cache=allow_cache,
            strict_as_of=strict_as_of,
        )
        if not path:
            continue
        built += 1
        paths.append(str(path))
    return {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "attempted": len(tickers),
        "built": built,
        "paths": paths,
    }


def latest_synthesis_path(conn, ticker: str, run_id: str) -> Path | None:
    row = conn.execute(
        """
        SELECT packet_path
        FROM synthesis_packets
        WHERE ticker = ? AND run_id = ?
        ORDER BY as_of_date DESC, created_at DESC
        LIMIT 1
        """,
        (ticker, run_id),
    ).fetchone()
    if not row:
        return None
    path = Path(row["packet_path"])
    if not path.exists():
        return None
    return path


def export_synthesis_packet(*, ticker: str, run_id: str, out: Path) -> Path:
    ticker = ticker.upper()
    with get_db() as conn:
        path = latest_synthesis_path(conn, ticker, run_id)
    if not path:
        raise ValueError(f"No synthesis packet found for ticker={ticker} run_id={run_id}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return out


def append_synthesis_section(*, ticker: str, run_id: str, dossier_md_path: str) -> None:
    dossier_path = Path(dossier_md_path)
    if not dossier_path.exists():
        return
    with get_db() as conn:
        synth_path = latest_synthesis_path(conn, ticker.upper(), run_id)
    if not synth_path:
        return
    payload = _load_json(synth_path)
    if not payload:
        return
    synth_as_of = str(payload.get("as_of_date") or "")
    evidence_packet = None
    if synth_as_of:
        with get_db() as conn:
            evidence_packet = _latest_evidence_packet(conn, ticker.upper(), synth_as_of)
    # Load quality signals from scorecard
    quality_ctx = {}
    moat_data = {}
    downside_data = {}
    signal_context_val = None
    if synth_as_of:
        with get_db() as conn:
            sc_row = latest_decision_eligible_valuation_row(
                conn,
                ticker=ticker,
                method="scorecard",
                as_of_date=synth_as_of,
                exact_as_of_date=True,
            )
            if sc_row:
                sc_outputs = json.loads(sc_row["outputs_json"] or "{}")
                quality_ctx = (
                    sc_outputs.get("quality_context")
                    if isinstance(sc_outputs.get("quality_context"), dict)
                    else {}
                )
                moat_data = (
                    sc_outputs.get("moat_strength")
                    if isinstance(sc_outputs.get("moat_strength"), dict)
                    else {}
                )
                downside_data = (
                    sc_outputs.get("downside_scenario")
                    if isinstance(sc_outputs.get("downside_scenario"), dict)
                    else {}
                )
                signal_context_val = sc_outputs.get("signal_context")
    decision_frame = (
        payload.get("decision_frame") if isinstance(payload.get("decision_frame"), dict) else {}
    )
    stance = str(decision_frame.get("stance") or "watchlist").upper()
    verdict_reason = str(
        payload.get("valuation_interpretation")
        or payload.get("business_quality_summary")
        or "UNKNOWN"
    )
    valuation = _valuation_summary(evidence_packet or {})
    reverse_dcf = (
        valuation.get("reverse_dcf") if isinstance(valuation.get("reverse_dcf"), dict) else {}
    )
    price = None
    reverse_inputs = (
        reverse_dcf.get("inputs") if isinstance(reverse_dcf.get("inputs"), dict) else {}
    )
    if isinstance(reverse_inputs.get("price"), (int, float)):
        price = float(reverse_inputs["price"])

    def _value_for(method: str, key: str) -> float | None:
        method_payload = valuation.get(method) if isinstance(valuation.get(method), dict) else {}
        value = method_payload.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    def _fmt_share(value: float | None) -> str:
        if value is None:
            return "UNKNOWN"
        return f"${value:,.2f}"

    def _fmt_gap(price_value: float | None, intrinsic_value: float | None) -> str:
        if price_value is None or intrinsic_value in (None, 0):
            return "UNKNOWN"
        gap_pct = ((intrinsic_value - price_value) / price_value) * 100.0
        return f"{gap_pct:+.0f}%"

    rows = [
        ("DCF", _value_for("dcf", "base")),
        ("EPV", _value_for("epv", "value_per_share")),
        ("Graham", _value_for("graham", "value_per_share")),
        ("NCAV", _value_for("ncav", "value_per_share")),
    ]
    lines = [
        "",
        "## Layer 3: Qualitative Synthesis",
        "",
        "### Verdict",
        f"{stance} - {verdict_reason}",
        "",
        "### Business Quality",
        str(payload.get("business_quality_summary") or "UNKNOWN"),
    ]
    supports = quality_ctx.get("valuation_supports") or []
    if supports:
        lines.append("")
        lines.append(f"**Valuation Supports:** {', '.join(supports)}")
    lines.extend(
        [
            "",
            "### Valuation",
            "| Method | Intrinsic Value | Current Price | Gap |",
            "| --- | --- | --- | --- |",
        ]
    )
    for label, intrinsic_value in rows:
        lines.append(
            f"| {label} | {_fmt_share(intrinsic_value)} | {_fmt_share(price)} | {_fmt_gap(price, intrinsic_value)} |"
        )
    implied_growth = _value_for("reverse_dcf", "implied_growth")
    if _value_for("reverse_dcf", "implied_growth_saturated"):
        implied_growth = None  # clipped bound, not a solve — don't narrate it
    if implied_growth is not None:
        lines.append("")
        lines.append(f"Implied growth rate: {_fmt_pct(implied_growth)}")
    # Quality Assessment sub-table
    gate_v = str(quality_ctx.get("gate_action") or "UNKNOWN")
    conf_v = str(quality_ctx.get("confidence_class") or "UNKNOWN")
    moat_v = str(moat_data.get("moat_class") or "UNKNOWN")
    moat_s = moat_data.get("moat_score")
    moat_display = f"{moat_v} ({moat_s})" if moat_s is not None else moat_v
    down_v = str(downside_data.get("downside_risk_class") or "UNKNOWN")
    sig_ctx = str(signal_context_val or "UNKNOWN")
    lines.extend(
        [
            "",
            "#### Quality Assessment",
            "| Signal | Value |",
            "| --- | --- |",
            f"| Gate Verdict | {gate_v} |",
            f"| Confidence | {conf_v} |",
            f"| Moat | {moat_display} |",
            f"| Downside Risk | {down_v} |",
            f"| Signal Context | {sig_ctx} |",
        ]
    )
    lines.extend(
        [
            str(payload.get("valuation_interpretation") or "UNKNOWN"),
            "",
            "### Key Risks",
        ]
    )
    risk_rows = (
        decision_frame.get("key_risks") if isinstance(decision_frame.get("key_risks"), list) else []
    )
    if risk_rows:
        lines.extend([f"- {item}" for item in risk_rows])
    else:
        lines.append(f"- {payload.get('risk_frame') or 'UNKNOWN'}")
    headwinds = quality_ctx.get("valuation_headwinds") or []
    if headwinds:
        lines.append("")
        lines.append(f"**Valuation Headwinds:** {', '.join(headwinds)}")
    lines.extend(["", "### Catalysts"])
    catalyst_rows = (
        decision_frame.get("catalysts") if isinstance(decision_frame.get("catalysts"), list) else []
    )
    if catalyst_rows:
        lines.extend([f"- {item}" for item in catalyst_rows])
    else:
        lines.append(f"- {payload.get('catalyst_frame') or 'UNKNOWN'}")
    lines.extend(["", "### Open Hypotheses"])
    hypotheses = payload.get("hypotheses") if isinstance(payload.get("hypotheses"), list) else []
    if hypotheses:
        for idx, item in enumerate(hypotheses, start=1):
            if not isinstance(item, dict):
                continue
            statement = str(item.get("statement") or "UNKNOWN")
            falsifiers = item.get("falsifiers") if isinstance(item.get("falsifiers"), list) else []
            falsifier_text = (
                "; ".join(str(x) for x in falsifiers[:2]) if falsifiers else "No falsifiers stated."
            )
            lines.append(f"{idx}. {statement} Falsifiers: {falsifier_text}")
    else:
        lines.append("1. No hypotheses stated.")
    lines.extend(
        ["", "### Confidence & Evidence Gaps", str(payload.get("confidence_notes") or "UNKNOWN")]
    )
    evidence_gaps = (
        payload.get("evidence_gaps") if isinstance(payload.get("evidence_gaps"), list) else []
    )
    if evidence_gaps:
        lines.extend([f"- {gap}" for gap in evidence_gaps])
    else:
        lines.append("- None stated.")
    lines.extend(["", "### Recommended Next Actions"])
    next_actions = (
        payload.get("recommended_next_actions")
        if isinstance(payload.get("recommended_next_actions"), list)
        else []
    )
    if next_actions:
        lines.extend([f"- {action}" for action in next_actions])
    else:
        lines.append("- None stated.")
    with dossier_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
