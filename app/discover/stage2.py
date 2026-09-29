"""Stage 2 Haiku classifier — production module.

Wraps the spike from experiments/stage2_classifier.py with:
- Injected Anthropic client (for tests)
- Persistence to the discover session DB (not JSON artifacts)
- Configurable pricing via Stage2Config dataclass
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.autonomous.v1_financial_context import (
    bind_v1_financial_scope,
    financial_input_scenario,
)
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


@dataclass
class Stage2Config:
    model: str = "claude-haiku-4-5-20251001"
    input_usd_per_mtok: float = 1.00
    output_usd_per_mtok: float = 5.00
    max_output_tokens: int = 512


_SYSTEM_PROMPT = """You are a fundamental-analysis triage classifier.

You read a compact summary of a company's deterministic valuation
scorecard and decide whether the ticker is worth sending to a deeper
LLM-driven research loop. You do NOT do the deep analysis — you only
decide whether it would be worth the time and money to do so.

Your bar is not "is this a great investment." Your bar is:
"Is there enough signal here that a 30-minute deep read of the
filings would be a reasonable use of analyst time?"

You return one of:
- KEEP: worth sending to Stage 3. There is specific evidence of
  potential mispricing, operational turnaround, or undervalued quality.
- DROP: not worth Stage 3 effort. The scorecard is either obviously
  uninvestable (hard blocker, going concern, severe decline), obviously
  fairly-priced (no margin of safety, methods all agree), or too
  uncertain to bother (all metrics N/A).

Err on the side of KEEP when the case is genuinely ambiguous — the
cost of a wrong DROP (missing a winner) is higher than the cost of a
wrong KEEP (one extra deep dive)."""


_TOOL_SCHEMA = {
    "name": "classify_ticker",
    "description": "Return a KEEP/DROP classification with confidence and a short reason.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["KEEP", "DROP"]},
            "confidence": {"type": "string", "enum": ["HIGH", "MODERATE", "LOW"]},
            "reason": {"type": "string"},
        },
        "required": ["decision", "confidence", "reason"],
    },
}


@dataclass
class Stage2Result:
    ticker: str
    as_of_date: str
    decision: str
    confidence: str
    reason: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    wall_ms: int
    error: str | None = None
    financial_scope_fingerprint: str | None = None
    financial_scope_publication_fingerprint: str | None = None


def compact_scorecard(scorecard: dict[str, Any]) -> str:
    """Build a compact signal-dense summary from a raw scorecard dict.

    Target size: ~300-600 tokens for the Haiku budget.
    """
    import json as _json

    pzd = scorecard.get("pricing_zone_detail") or {}
    qc = scorecard.get("quality_context") or {}
    lines: list[str] = []
    lines.append(f"pricing_zone: {scorecard.get('pricing_zone', 'UNKNOWN')}")
    lines.append(f"signal: {scorecard.get('signal', 'UNKNOWN')}")

    def fmt(v: Any) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines.append("")
    lines.append("VALUATION:")
    # margin_of_safety_* below are TEXTBOOK (intrinsic-price)/intrinsic;
    # graham_dodd/scout surfaces use UPSIDE ratios (intrinsic/price - 1).
    lines.append("  (margin_of_safety_* convention: TEXTBOOK (intrinsic-price)/intrinsic)")
    for k in (
        "current_price",
        "dcf_base",
        "dcf_durable_base",
        "epv_adjusted",
        "epv_cash_adjusted",
        "margin_of_safety_vs_epv_adjusted",
        "margin_of_safety_vs_epv_cash_adjusted",
        "margin_of_safety_vs_dcf_durable",
        "gate_action",
    ):
        if k in pzd:
            lines.append(f"  {k}: {fmt(pzd[k])}")
    if pzd.get("nonrecurring_revenue_spike"):
        lines.append(
            f"  nonrecurring_revenue_spike: TRUE "
            f"(durable base ~${fmt(pzd.get('durable_revenue_base_m'))}M; "
            f"see dcf_durable_base for spike-corrected anchor)"
        )

    lines.append("")
    lines.append("GROWTH & QUALITY:")
    for k in (
        "revenue_cagr_5y",
        "revenue_cagr_3y",
        "earnings_quality",
        "epv_quality",
        "cycle_position",
    ):
        if k in pzd:
            lines.append(f"  {k}: {fmt(pzd[k])}")

    lines.append("")
    lines.append("FLAGS:")
    for k in (
        "confidence_class",
        "allocation_grade",
        "gate_reason",
        "leverage_stress",
        "decline_years_consecutive",
        "decline_magnitude_total",
        "nonrecurring_detection",
        "sbc_burden",
        "valuation_headwinds",
    ):
        if k in qc:
            v = qc[k]
            if isinstance(v, (list, dict)):
                v = _json.dumps(v, separators=(",", ":"))[:200]
            lines.append(f"  {k}: {fmt(v)}")
    # Compact one-line summaries of the new EPV / nonrecurring revenue blocks
    ia = qc.get("intangible_amort") or {}
    if ia.get("is_materially_distorted"):
        pct = ia.get("average_addback_pct_of_revenue")
        pct_s = f"{pct * 100:.1f}%" if isinstance(pct, (int, float)) else "?"
        lines.append(
            f"  intangible_amort_distortion: addback ~${fmt(ia.get('average_addback_m'))}M/yr ({pct_s} of rev)"
        )
    nr = qc.get("nonrecurring_revenue") or {}
    if nr.get("has_suspected_nonrecurring"):
        ratio = nr.get("spike_ratio")
        ratio_s = f"{ratio:.1f}x" if isinstance(ratio, (int, float)) else "?"
        lines.append(
            f"  nonrecurring_revenue: FY{nr.get('spike_year')} spike "
            f"${fmt(nr.get('spike_revenue_m'))}M vs prior ${fmt(nr.get('prior_year_revenue_m'))}M ({ratio_s})"
        )

    return "\n".join(lines)


def _prepare_stage2_financial_input(
    ticker: str,
    as_of_date: str,
    scorecard: dict[str, Any],
) -> str:
    summary = compact_scorecard(scorecard)
    return (
        f"Ticker: {ticker}\nAs-of date: {as_of_date}\n\n"
        f"Scorecard summary:\n{summary}\n\n"
        "Decide: KEEP or DROP this ticker for Stage 3 deep analysis?"
    )


def _bind_stage2_financial_input(
    *,
    ticker: str,
    as_of_date: str,
    scorecard: dict[str, Any],
    config: Stage2Config,
    financial_packet: Any | None,
):
    user_msg = _prepare_stage2_financial_input(
        ticker,
        as_of_date,
        scorecard,
    )
    if financial_packet is None:
        bind_v1_financial_scope(
            context=f"discover_stage2:{ticker}:{as_of_date}",
            run_as_of_date=as_of_date,
            packets=(),
            scenarios=(),
        )
    financial_scenario = financial_input_scenario(
        financial_packet,
        financial_inputs={
            "scorecard": scorecard,
            "provider_user_message": user_msg,
            "provider_system_prompt": _SYSTEM_PROMPT,
            "provider_tool_schema": _TOOL_SCHEMA,
            "provider_model": config.model,
            "provider_max_output_tokens": config.max_output_tokens,
        },
    )
    financial_scope = bind_v1_financial_scope(
        context=f"discover_stage2:{ticker}:{as_of_date}",
        run_as_of_date=as_of_date,
        packets=(financial_packet,),
        scenarios=(financial_scenario,),
    )
    return user_msg, financial_scope, financial_scenario


def _require_current_stage2_scope(
    *,
    ticker: str,
    as_of_date: str,
    scorecard: dict[str, Any],
    config: Stage2Config,
    financial_packet: Any | None,
    authorized_fingerprint: str | None,
    phase: str,
    sweep_id: str | None = None,
) -> str:
    _, current_scope, _ = _bind_stage2_financial_input(
        ticker=ticker,
        as_of_date=as_of_date,
        scorecard=scorecard,
        config=config,
        financial_packet=financial_packet,
    )
    current_fingerprint = current_scope.expected_scope_fingerprint
    require_discover_financial_scope_unchanged(
        stage=2,
        ticker=ticker,
        run_as_of_date=as_of_date,
        authorized_fingerprint=authorized_fingerprint,
        current_fingerprint=current_fingerprint,
        phase=phase,
        sweep_id=sweep_id,
    )
    return current_fingerprint


def classify_ticker(
    client,
    ticker: str,
    as_of_date: str,
    scorecard: dict[str, Any],
    config: Stage2Config,
    financial_packet: Any | None = None,
) -> Stage2Result:
    """Call the Haiku classifier for one ticker. Never raises — returns error Result."""
    user_msg, financial_scope, financial_scenario = _bind_stage2_financial_input(
        ticker=ticker,
        as_of_date=as_of_date,
        scorecard=scorecard,
        config=config,
        financial_packet=financial_packet,
    )
    scope_fingerprint = financial_scope.expected_scope_fingerprint

    t0 = time.perf_counter()
    try:
        financial_scope.require(scenarios=(financial_scenario,))
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_output_tokens,
            system=_SYSTEM_PROMPT,
            tools=[_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "classify_ticker"},
            messages=[{"role": "user", "content": user_msg}],
        )
        publication_scope_fingerprint = _require_current_stage2_scope(
            ticker=ticker,
            as_of_date=as_of_date,
            scorecard=scorecard,
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
            _require_current_stage2_scope(
                ticker=ticker,
                as_of_date=as_of_date,
                scorecard=scorecard,
                config=config,
                financial_packet=financial_packet,
                authorized_fingerprint=scope_fingerprint,
                phase="failed_response",
            )
        except InvalidFinancialInputError as integrity_exc:
            raise integrity_exc from exc
        wall = int((time.perf_counter() - t0) * 1000)
        return Stage2Result(
            ticker=ticker,
            as_of_date=as_of_date,
            decision="ERROR",
            confidence="LOW",
            reason=f"api error: {exc}",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            wall_ms=wall,
            error=str(exc),
            financial_scope_fingerprint=scope_fingerprint,
        )

    wall = int((time.perf_counter() - t0) * 1000)

    decision = "ERROR"
    confidence = "LOW"
    reason = "no tool_use in response"
    for block in response.content:
        btype = getattr(block, "type", None)
        if btype == "tool_use" and getattr(block, "name", None) == "classify_ticker":
            tool_input = block.input or {}
            decision = tool_input.get("decision", "ERROR")
            confidence = tool_input.get("confidence", "LOW")
            reason = tool_input.get("reason", "")
            break

    usage = response.usage
    cost = (
        usage.input_tokens / 1_000_000 * config.input_usd_per_mtok
        + usage.output_tokens / 1_000_000 * config.output_usd_per_mtok
    )

    return Stage2Result(
        ticker=ticker,
        as_of_date=as_of_date,
        decision=decision,
        confidence=confidence,
        reason=reason,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost,
        wall_ms=wall,
        financial_scope_fingerprint=scope_fingerprint,
        financial_scope_publication_fingerprint=publication_scope_fingerprint,
    )


def _overvaluation_sanity_check(
    result: Stage2Result,
    scorecard: dict[str, Any],
    threshold: float = 1.5,
) -> Stage2Result:
    """Override KEEP → DROP when price is materially above BOTH intrinsic anchors.

    Catches sign-inversion hallucinations where Haiku calls an overvalued
    stock "undervalued". The threshold of 1.5 means price must be ≥ 150%
    of BOTH DCF and EPV to trigger the override.
    """
    if result.decision != "KEEP":
        return result

    pzd = scorecard.get("pricing_zone_detail") or {}
    price = pzd.get("current_price")
    dcf = pzd.get("dcf_base")
    epv = pzd.get("epv_adjusted")

    if not isinstance(price, (int, float)) or price <= 0:
        return result

    price_above_dcf = isinstance(dcf, (int, float)) and dcf > 0 and price > dcf * threshold
    price_above_epv = isinstance(epv, (int, float)) and epv > 0 and price > epv * threshold

    if price_above_dcf and price_above_epv:
        logger.warning(
            "stage2 sanity check: %s KEEP overridden to DROP — price $%.2f is >%.0f%% "
            "above DCF $%.2f and EPV $%.2f",
            result.ticker,
            price,
            threshold * 100,
            dcf,
            epv,
        )
        return Stage2Result(
            ticker=result.ticker,
            as_of_date=result.as_of_date,
            decision="DROP",
            confidence=result.confidence,
            reason=f"SANITY_OVERRIDE: price ${price:.2f} > {threshold:.0f}x both DCF ${dcf:.2f} and EPV ${epv:.2f}. Original: {result.reason}",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            wall_ms=result.wall_ms,
            financial_scope_fingerprint=(result.financial_scope_fingerprint),
            financial_scope_publication_fingerprint=(
                result.financial_scope_publication_fingerprint
            ),
        )

    return result


def run_stage2(
    db_path: str | Path,
    sweep_id: str,
    tickers_with_scorecards: Iterable[tuple[str, str, dict[str, Any]]],
    client,
    config: Stage2Config | None = None,
    financial_packets: dict[str, Any] | None = None,
) -> list[Stage2Result]:
    """Run Stage 2 over a list of tickers. Writes each result to the session DB."""
    cfg = config or Stage2Config()
    existing = stage_result_financial_fingerprints(
        db_path,
        stage=2,
        sweep_id=sweep_id,
    )
    results: list[Stage2Result] = []
    for ticker, as_of_date, scorecard in tickers_with_scorecards:
        normalized_ticker = str(ticker).strip().upper()
        if normalized_ticker in existing:
            _, cached_scope, _ = _bind_stage2_financial_input(
                ticker=ticker,
                as_of_date=as_of_date,
                scorecard=scorecard,
                config=cfg,
                financial_packet=(financial_packets or {}).get(normalized_ticker),
            )
            require_discover_result_financial_scope(
                stage=2,
                sweep_id=sweep_id,
                ticker=ticker,
                run_as_of_date=as_of_date,
                stored_fingerprint=existing[normalized_ticker],
                expected_fingerprint=(cached_scope.expected_scope_fingerprint),
            )
            continue
        require_no_prior_discover_paid_attempt(
            db_path,
            stage=2,
            sweep_id=sweep_id,
            ticker=normalized_ticker,
        )
        durable_client = DurableDiscoverClient(
            client,
            db_path=db_path,
            stage=2,
            sweep_id=sweep_id,
            ticker=normalized_ticker,
            input_usd_per_mtok=cfg.input_usd_per_mtok,
            output_usd_per_mtok=cfg.output_usd_per_mtok,
        )
        result = classify_ticker(
            durable_client,
            ticker,
            as_of_date,
            scorecard,
            cfg,
            financial_packet=(financial_packets or {}).get(ticker.upper()),
        )
        result = _overvaluation_sanity_check(result, scorecard)
        attempt_summary = discover_paid_attempt_summary(
            db_path,
            stage=2,
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
            publication_user_msg, publication_scope, publication_scenario = (
                _bind_stage2_financial_input(
                    ticker=ticker,
                    as_of_date=as_of_date,
                    scorecard=scorecard,
                    config=cfg,
                    financial_packet=(financial_packets or {}).get(ticker.upper()),
                )
            )
            publication_scope.require(scenarios=(publication_scenario,))
            require_discover_financial_scope_unchanged(
                stage=2,
                ticker=ticker,
                run_as_of_date=as_of_date,
                authorized_fingerprint=result.financial_scope_fingerprint,
                current_fingerprint=publication_scope.expected_scope_fingerprint,
                phase="pre_persistence",
                sweep_id=sweep_id,
            )
            result.financial_scope_publication_fingerprint = (
                publication_scope.expected_scope_fingerprint
            )
            publication_evidence = build_discover_publication_evidence(
                stage=2,
                ticker=ticker,
                scope_fingerprint=result.financial_scope_publication_fingerprint,
                primary_evidence={
                    "model": cfg.model,
                    "max_tokens": cfg.max_output_tokens,
                    "system": _SYSTEM_PROMPT,
                    "tools": [_TOOL_SCHEMA],
                    "tool_choice": {"type": "tool", "name": "classify_ticker"},
                    "messages": [
                        {
                            "role": "user",
                            "content": publication_user_msg,
                        }
                    ],
                },
            )
        if publication_scope is None or publication_evidence is None:
            raise ValueError(
                f"Discover Stage 2 production result for {result.ticker} lacks authorization"
            )
        _publish_discover_result(
            db_path,
            stage=2,
            row={
                "sweep_id": sweep_id,
                "ticker": result.ticker,
                "decision": result.decision,
                "confidence": result.confidence,
                "reason": result.reason,
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
