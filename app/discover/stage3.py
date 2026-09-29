"""Stage 3 Sonnet mid-depth research — production module.

Wraps the spike from experiments/stage3_research.py with:
- Injected Anthropic client and bundle_builder callable for testability
- Persistence to the discover session DB
- Configurable pricing and truncation via Stage3Config
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.autonomous.v1_financial_context import (
    bind_v1_financial_scope,
    financial_input_scenario,
)
from app.analyst.evidence_bundle import AnalysisEvidenceBundle, BundleFiling
from app.discover.persistence import (
    DiscoverCostBudgetExceeded,
    DurableDiscoverClient,
    _publish_discover_result,
    build_discover_publication_evidence,
    discover_paid_attempt_summary,
    require_no_prior_discover_paid_attempt,
    require_discover_financial_scope_unchanged,
    require_discover_result_financial_scope,
    stage_result_financial_fingerprints,
)

logger = logging.getLogger(__name__)


_DEFAULT_SECTION_LIMITS = {
    "mda": 8000,
    "risk_factors": 6000,
    "fin_notes": 4000,
    "business": 2000,
}


@dataclass
class Stage3Config:
    model: str = "claude-sonnet-4-6"
    input_usd_per_mtok: float = 3.00
    output_usd_per_mtok: float = 15.00
    thinking_budget_tokens: int = 4000
    max_output_tokens: int = 8192
    section_char_limits: dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_SECTION_LIMITS)
    )


_SYSTEM_PROMPT = """You are a fundamental-analysis researcher performing mid-depth triage.

You receive a compact evidence bundle: deterministic valuation context
(DCF, EPV, Graham, MOS, quality signals) plus excerpted sections from
the most recent 10-K filing (MD&A, Risk Factors, Financial Notes,
Business). Your job is NOT to do a full 4-hour deep dive. It is to
spend ~5 minutes of thinking on this ticker and decide whether it
deserves the next stage (a multi-turn deep research loop).

Form a view. Defend it. Do not just summarize the evidence.

Output via the stage3_analysis tool.

Rules:
1. Cite specific numbers from the evidence.
2. Be willing to PASS on famous companies if the evidence says so.
3. Be willing to BUY_CANDIDATE on obscure companies if the evidence is compelling.
4. Treat negative DCF/EPV as a warning, not an automatic PASS.
5. Your confidence should reflect how much the evidence supports the verdict."""


_TOOL_SCHEMA = {
    "name": "stage3_analysis",
    "description": "Return a structured mid-depth analysis decision for one ticker.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["BUY_CANDIDATE", "WATCH", "PASS"]},
            "confidence": {"type": "string", "enum": ["HIGH", "MODERATE", "LOW"]},
            "thesis_summary": {"type": "string"},
            "key_numbers": {"type": "array", "items": {"type": "string"}},
            "positives": {"type": "array", "items": {"type": "string"}},
            "risks": {"type": "array", "items": {"type": "string"}},
            "open_questions": {"type": "array", "items": {"type": "string"}},
            "reasoning_trace": {"type": "string"},
        },
        "required": [
            "verdict",
            "confidence",
            "thesis_summary",
            "key_numbers",
            "positives",
            "risks",
            "open_questions",
            "reasoning_trace",
        ],
    },
}


@dataclass
class Stage3Result:
    ticker: str
    as_of_date: str
    verdict: str
    confidence: str
    thesis_summary: str
    key_numbers: list[str] = field(default_factory=list)
    positives: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    reasoning_trace: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    wall_ms: int = 0
    error: str | None = None
    financial_scope_fingerprint: str | None = None
    financial_scope_publication_fingerprint: str | None = None


def _fmt_value(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _serialize_filing(f: BundleFiling, char_limits: dict[str, int]) -> str:
    lines = [
        f"form_type: {f.form_type}",
        f"filing_date: {f.filing_date}",
        f"accession: {f.accession}",
        f"sections_included: {f.sections_included}",
    ]
    for key, text in f.section_text.items():
        limit = char_limits.get(key, 3000)
        excerpt = text[:limit]
        if len(text) > limit:
            excerpt += f"\n... [truncated {len(text) - limit} chars]"
        lines.append(f"\n### {key} ({len(text)} chars, truncated to {limit}):\n{excerpt}")
    return "\n".join(lines)


def build_stage3_prompt(
    bundle: AnalysisEvidenceBundle,
    section_char_limits: dict[str, int] | None = None,
    financial_packet: Any | None = None,
) -> str:
    """Serialize an AnalysisEvidenceBundle into a Stage 3 user message."""
    limits = section_char_limits or _DEFAULT_SECTION_LIMITS
    snap = bundle.valuation
    packet = (
        dict(financial_packet)
        if isinstance(financial_packet, dict)
        else dict(vars(financial_packet))
        if financial_packet is not None
        else {}
    )
    financial_values = {
        "current_price": packet.get("current_price", snap.current_price),
        "market_cap": packet.get("market_cap_mm", snap.market_cap),
        "dcf_base": packet.get("dcf_value", snap.dcf_base),
        "epv_adjusted": packet.get("epv_value", snap.epv_adjusted),
        "graham_value": packet.get("graham_value", snap.graham_value),
        "methods_agree": packet.get("methods_agree", snap.methods_agree),
        "tension_type": packet.get(
            "method_tension_type",
            snap.tension_type,
        ),
        "gate_action": packet.get("gate_verdict", snap.gate_action),
        "solvency_status": packet.get(
            "solvency_risk",
            snap.solvency_status,
        ),
        "filing_risk_status": packet.get(
            "filing_risk_status",
            snap.filing_risk_status,
        ),
    }
    parts: list[str] = [
        f"=== TICKER: {bundle.ticker} ===",
        f"as_of_date: {bundle.as_of_date}",
        f"analysis_years: {bundle.analysis_years}",
    ]
    if bundle.warnings:
        parts.append(f"warnings: {bundle.warnings}")
    parts.append("")
    parts.append("=== VALUATION CONTEXT (deterministic) ===")
    for key, value in financial_values.items():
        parts.append(f"{key}: {_fmt_value(value)}")
    parts.append("")
    if bundle.filings:
        parts.append("=== MOST RECENT FILING (excerpted) ===")
        parts.append(_serialize_filing(bundle.filings[0], limits))
    else:
        parts.append("=== NO FILING TEXT AVAILABLE ===")
    parts.append("")
    parts.append(
        "Decide: is this a BUY_CANDIDATE, WATCH, or PASS? "
        "Use the stage3_analysis tool to return your structured decision."
    )
    return "\n".join(parts)


def _bind_stage3_financial_input(
    *,
    bundle: AnalysisEvidenceBundle,
    config: Stage3Config,
    financial_packet: Any | None,
):
    if financial_packet is None:
        missing_ticker = str(getattr(bundle, "ticker", "UNKNOWN")).strip().upper()
        missing_as_of = str(getattr(bundle, "as_of_date", "")).strip()[:10]
        bind_v1_financial_scope(
            context=f"discover_stage3:{missing_ticker}:{missing_as_of}",
            run_as_of_date=missing_as_of,
            packets=(),
            scenarios=(),
        )
    prompt = build_stage3_prompt(
        bundle,
        config.section_char_limits,
        financial_packet,
    )
    financial_scenario = financial_input_scenario(
        financial_packet,
        financial_inputs={
            "stage3_prompt": prompt,
            "provider_system_prompt": _SYSTEM_PROMPT,
            "provider_tool_schema": _TOOL_SCHEMA,
            "provider_model": config.model,
            "provider_max_output_tokens": config.max_output_tokens,
            "provider_thinking_budget_tokens": (config.thinking_budget_tokens),
        },
    )
    financial_scope = bind_v1_financial_scope(
        context=f"discover_stage3:{bundle.ticker}:{bundle.as_of_date}",
        run_as_of_date=bundle.as_of_date,
        packets=(financial_packet,),
        scenarios=(financial_scenario,),
    )
    return prompt, financial_scope, financial_scenario


def _require_current_stage3_scope(
    *,
    bundle: AnalysisEvidenceBundle,
    config: Stage3Config,
    financial_packet: Any | None,
    authorized_fingerprint: str | None,
    phase: str,
    sweep_id: str | None = None,
) -> str:
    _, current_scope, _ = _bind_stage3_financial_input(
        bundle=bundle,
        config=config,
        financial_packet=financial_packet,
    )
    current_fingerprint = current_scope.expected_scope_fingerprint
    require_discover_financial_scope_unchanged(
        stage=3,
        ticker=bundle.ticker,
        run_as_of_date=bundle.as_of_date,
        authorized_fingerprint=authorized_fingerprint,
        current_fingerprint=current_fingerprint,
        phase=phase,
        sweep_id=sweep_id,
    )
    return current_fingerprint


def research_ticker(
    client,
    ticker: str,
    config: Stage3Config,
    bundle_builder: Callable[..., AnalysisEvidenceBundle],
    financial_packet: Any | None = None,
) -> Stage3Result:
    """Run Stage 3 on one ticker. Builds the bundle via the injected callable."""
    t_start = time.perf_counter()
    try:
        bundle = bundle_builder(ticker, as_of_date=None)
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        return Stage3Result(
            ticker=ticker,
            as_of_date="unknown",
            verdict="ERROR",
            confidence="LOW",
            thesis_summary="",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            wall_ms=int((time.perf_counter() - t_start) * 1000),
            error=f"bundle build failed: {exc}",
        )

    prompt, financial_scope, financial_scenario = _bind_stage3_financial_input(
        bundle=bundle,
        config=config,
        financial_packet=financial_packet,
    )
    scope_fingerprint = financial_scope.expected_scope_fingerprint

    try:
        financial_scope.require(scenarios=(financial_scenario,))
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_output_tokens,
            thinking={"type": "enabled", "budget_tokens": config.thinking_budget_tokens},
            system=_SYSTEM_PROMPT,
            tools=[_TOOL_SCHEMA],
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": prompt}],
        )
        publication_scope_fingerprint = _require_current_stage3_scope(
            bundle=bundle,
            config=config,
            financial_packet=financial_packet,
            authorized_fingerprint=scope_fingerprint,
            phase="post_response",
        )
    except InvalidFinancialInputError:
        raise
    except DiscoverCostBudgetExceeded:
        raise
    except Exception as exc:
        try:
            _require_current_stage3_scope(
                bundle=bundle,
                config=config,
                financial_packet=financial_packet,
                authorized_fingerprint=scope_fingerprint,
                phase="failed_response",
            )
        except InvalidFinancialInputError as integrity_exc:
            raise integrity_exc from exc
        return Stage3Result(
            ticker=ticker,
            as_of_date=bundle.as_of_date,
            verdict="ERROR",
            confidence="LOW",
            thesis_summary="",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            wall_ms=int((time.perf_counter() - t_start) * 1000),
            error=f"api error: {exc}",
            financial_scope_fingerprint=scope_fingerprint,
        )

    wall = int((time.perf_counter() - t_start) * 1000)

    tool_input: dict[str, Any] | None = None
    for block in response.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "stage3_analysis"
        ):
            tool_input = block.input or {}
            break

    usage = response.usage
    cost = (
        usage.input_tokens / 1_000_000 * config.input_usd_per_mtok
        + usage.output_tokens / 1_000_000 * config.output_usd_per_mtok
    )

    if tool_input is None:
        return Stage3Result(
            ticker=ticker,
            as_of_date=bundle.as_of_date,
            verdict="ERROR",
            confidence="LOW",
            thesis_summary="",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost,
            wall_ms=wall,
            error=f"no tool_use block; stop_reason={response.stop_reason}",
            financial_scope_fingerprint=scope_fingerprint,
            financial_scope_publication_fingerprint=(publication_scope_fingerprint),
        )

    return Stage3Result(
        ticker=ticker,
        as_of_date=bundle.as_of_date,
        verdict=tool_input.get("verdict", "ERROR"),
        confidence=tool_input.get("confidence", "LOW"),
        thesis_summary=tool_input.get("thesis_summary", ""),
        key_numbers=list(tool_input.get("key_numbers", [])),
        positives=list(tool_input.get("positives", [])),
        risks=list(tool_input.get("risks", [])),
        open_questions=list(tool_input.get("open_questions", [])),
        reasoning_trace=tool_input.get("reasoning_trace", ""),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost,
        wall_ms=wall,
        financial_scope_fingerprint=scope_fingerprint,
        financial_scope_publication_fingerprint=publication_scope_fingerprint,
    )


def run_stage3(
    db_path: str | Path,
    sweep_id: str,
    tickers: Iterable[str],
    client,
    config: Stage3Config | None = None,
    bundle_builder: Callable[..., AnalysisEvidenceBundle] | None = None,
    financial_packets: dict[str, Any] | None = None,
) -> list[Stage3Result]:
    """Run Stage 3 over a list of tickers. Writes each result to the session DB."""
    cfg = config or Stage3Config()
    if bundle_builder is None:
        from app.analyst.bundle_builder import build_analysis_evidence_bundle as bundle_builder

    existing = stage_result_financial_fingerprints(
        db_path,
        stage=3,
        sweep_id=sweep_id,
    )
    results: list[Stage3Result] = []
    for ticker in tickers:
        normalized_ticker = str(ticker).strip().upper()
        if normalized_ticker in existing:
            try:
                cached_bundle = bundle_builder(
                    ticker,
                    as_of_date=None,
                )
            except InvalidFinancialInputError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"cannot authorize cached Discover Stage 3 result for {ticker}: {exc}"
                ) from exc
            _, cached_scope, _ = _bind_stage3_financial_input(
                bundle=cached_bundle,
                config=cfg,
                financial_packet=(financial_packets or {}).get(normalized_ticker),
            )
            require_discover_result_financial_scope(
                stage=3,
                sweep_id=sweep_id,
                ticker=ticker,
                run_as_of_date=cached_bundle.as_of_date,
                stored_fingerprint=existing[normalized_ticker],
                expected_fingerprint=(cached_scope.expected_scope_fingerprint),
            )
            continue
        require_no_prior_discover_paid_attempt(
            db_path,
            stage=3,
            sweep_id=sweep_id,
            ticker=normalized_ticker,
        )
        durable_client = DurableDiscoverClient(
            client,
            db_path=db_path,
            stage=3,
            sweep_id=sweep_id,
            ticker=normalized_ticker,
            input_usd_per_mtok=cfg.input_usd_per_mtok,
            output_usd_per_mtok=cfg.output_usd_per_mtok,
        )
        result = research_ticker(
            durable_client,
            ticker,
            cfg,
            bundle_builder,
            financial_packet=(financial_packets or {}).get(ticker.upper()),
        )
        attempt_summary = discover_paid_attempt_summary(
            db_path,
            stage=3,
            sweep_id=sweep_id,
            ticker=normalized_ticker,
        )
        result.input_tokens = int(attempt_summary["input_tokens"])
        result.output_tokens = int(attempt_summary["output_tokens"])
        result.cost_usd = float(attempt_summary["cost_usd"])
        results.append(result)

        publication_evidence = None
        publication_scope = None
        publication_scenario = None
        if result.financial_scope_fingerprint is not None:
            try:
                publication_bundle = bundle_builder(
                    ticker,
                    as_of_date=None,
                )
            except InvalidFinancialInputError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"cannot authorize Discover Stage 3 result for {ticker}: {exc}"
                ) from exc
            publication_prompt, publication_scope, publication_scenario = (
                _bind_stage3_financial_input(
                    bundle=publication_bundle,
                    config=cfg,
                    financial_packet=(financial_packets or {}).get(normalized_ticker),
                )
            )
            publication_scope.require(scenarios=(publication_scenario,))
            require_discover_financial_scope_unchanged(
                stage=3,
                ticker=ticker,
                run_as_of_date=publication_bundle.as_of_date,
                authorized_fingerprint=result.financial_scope_fingerprint,
                current_fingerprint=publication_scope.expected_scope_fingerprint,
                phase="pre_persistence",
                sweep_id=sweep_id,
            )
            result.financial_scope_publication_fingerprint = (
                publication_scope.expected_scope_fingerprint
            )
            publication_evidence = build_discover_publication_evidence(
                stage=3,
                ticker=ticker,
                scope_fingerprint=result.financial_scope_publication_fingerprint,
                primary_evidence={
                    "model": cfg.model,
                    "max_tokens": cfg.max_output_tokens,
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": cfg.thinking_budget_tokens,
                    },
                    "system": _SYSTEM_PROMPT,
                    "tools": [_TOOL_SCHEMA],
                    "tool_choice": {"type": "auto"},
                    "messages": [{"role": "user", "content": publication_prompt}],
                },
            )
        if publication_scope is None or publication_evidence is None:
            raise ValueError(
                f"Discover Stage 3 production result for {result.ticker} lacks authorization"
            )
        _publish_discover_result(
            db_path,
            stage=3,
            row={
                "sweep_id": sweep_id,
                "ticker": result.ticker,
                "verdict": result.verdict,
                "confidence": result.confidence,
                "thesis_summary": result.thesis_summary,
                "key_numbers_json": result.key_numbers,
                "positives_json": result.positives,
                "risks_json": result.risks,
                "open_questions_json": result.open_questions,
                "reasoning_trace": result.reasoning_trace,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
                "wall_ms": result.wall_ms,
                "error": result.error,
                "financial_scope_fingerprint": result.financial_scope_fingerprint,
                "financial_scope_publication_fingerprint": (
                    result.financial_scope_publication_fingerprint
                ),
                "publication_evidence_json": None,
            },
            financial_scope=publication_scope,
            publication_evidence=publication_evidence,
            financial_scenarios=(publication_scenario,),
        )

    return results
