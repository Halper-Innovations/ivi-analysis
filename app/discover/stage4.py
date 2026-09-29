"""Stage 4 Sonnet deep loop — production module.

Multi-turn reasoning loop with tool use, injected dependencies for tests.

Tools are dispatched via a callable `tool_dispatcher(name, input) -> str`
passed into run_stage4. This lets tests provide stub tool outputs
without needing the real DB / filesystem / companyfacts layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

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
    stage4_result_financial_manifests,
)

logger = logging.getLogger(__name__)


@dataclass
class Stage4Config:
    model: str = "claude-sonnet-4-6"
    input_usd_per_mtok: float = 3.00
    output_usd_per_mtok: float = 15.00
    max_output_tokens: int = 8192
    max_turns: int = 15
    max_cost_usd: float = 2.00


_SYSTEM_PROMPT = """You are a fundamental analyst performing a deep research dive on a single company.

You have four tools: fetch_current_price, fetch_filing_section, fetch_companyfacts,
and fetch_historical_scorecards. Use them selectively to close the most important
evidence gaps. Do not speculate about numbers you can fetch, and do not call tools
that are unlikely to change the verdict.

YOUR JOB:
- Turn the initial scorecard, filing metadata, and prior-stage context into a final
  BUY, WATCH, or PASS.
- Base the verdict on evidence available in the prompt and tools, not on training-data memory.
- Prefer concrete balance-sheet, cash-flow, profitability, dilution, and filing evidence
  over vague narrative.

WORKFLOW:
1. Check whether the prompt already contains enough evidence for a view.
2. If an important fact is missing, fetch only the evidence needed:
   - fetch_current_price when the price is missing, stale, or central to the margin-of-safety call
   - fetch_companyfacts for specific line items needed to test leverage, cash generation,
     dilution, reinvestment, or trend quality
   - fetch_filing_section for primary-source context from the most recent 10-K sections available
   - fetch_historical_scorecards when multi-year valuation or risk context matters
3. Stop once additional evidence is unlikely to change the verdict.

WHAT TO EVALUATE:
- Valuation: does the current price look attractive relative to DCF, EPV, Graham, and
  the quality of the business?
- Balance sheet and cash generation: is leverage manageable and is the business funding itself?
- Capital allocation: how does management deploy free cash flow? Use fetch_companyfacts to
  check buybacks (share_repurchases_amount), debt trajectory (total_debt over time), R&D
  reinvestment (r_and_d_total), and dilution (shares_outstanding trend, sbc). Buybacks
  below intrinsic value are accretive; above it, they are value-destructive. Rising
  goodwill faster than revenue growth suggests overpaid acquisitions.
- Trend quality: are revenue, margins, cash flow, or dilution improving, deteriorating, or unstable?
  Distinguish organic growth from acquisition-driven growth when the data allows it.
- Filing context: do the MD&A, business description, risk factors, or notes materially
  change the interpretation of the scorecard?
- Confidence: how much of the verdict rests on solid evidence versus unresolved uncertainty?

RULES:
- Keep key findings grounded in specific figures when available.
- When raw companyfacts or filing text conflict with an interpretation in the scorecard,
  prefer the raw facts and explain the conflict.
- Do not invent competitor-share claims, management judgments, or precise forecasts that
  the tools cannot support.
- You do not need a precise buy-below price to finalize; use the current price and the
  available valuation anchors.
- A WATCH means the business may be interesting but the current price or evidence is not
  attractive enough today. If the evidence supports a specific number, populate the
  optional buy_below_price field with the price at or below which the name would become
  interesting. Leave it null when the evidence does not support a precise anchor.
- A BUY means the current evidence supports upside at today's price with acceptable risk.
- A PASS means the business, balance sheet, valuation, or evidence quality is unattractive.
- It is acceptable to leave non-critical uncertainty in open_questions.
- Call finalize_analysis when further evidence would not materially change the verdict."""


_TOOL_SCHEMAS = [
    {
        "name": "fetch_filing_section",
        "description": (
            "Return the untruncated text of one canonical section from the most "
            "recent 10-K. section_key in mda / risk_factors / fin_notes / business."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "section_key": {
                    "type": "string",
                    "enum": ["mda", "risk_factors", "fin_notes", "business"],
                },
            },
            "required": ["ticker", "section_key"],
        },
    },
    {
        "name": "fetch_historical_scorecards",
        "description": "Return the N most recent cached scorecards for a ticker.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "n_years": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["ticker", "n_years"],
        },
    },
    {
        "name": "fetch_companyfacts",
        "description": (
            "Return time-series of normalized financial line items from the "
            "companyfacts cache. Use the NORMALIZED names below — these match "
            "what's stored in the database. Raw XBRL names like 'Revenues' or "
            "'NetIncomeLoss' will NOT match (they get translated server-side, "
            "but use normalized names directly to avoid translation gaps).\n\n"
            "Available normalized line_items:\n"
            "  Income statement: revenue, net_income, operating_income, gross_profit\n"
            "  Cash flow: cfo, capex, depreciation_amortization, sbc, "
            "share_repurchases_amount, dividends_paid_amount\n"
            "  Balance sheet: cash, total_debt, total_assets, total_liabilities, "
            "equity, current_assets, current_liabilities, accounts_receivable, "
            "accounts_payable, inventory, goodwill, intangible_assets, "
            "investment_securities, deferred_revenue, gross_ppe, "
            "operating_lease_liability\n"
            "  Other: shares_outstanding, interest_expense, r_and_d_total, "
            "restructuring_charges, depreciation\n\n"
            "Example: line_items=['revenue','cfo','total_debt','goodwill']"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "line_items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 10,
                },
            },
            "required": ["ticker", "line_items"],
        },
    },
    {
        "name": "fetch_current_price",
        "description": (
            "Fetch the current market price for a ticker via Yahoo Finance. "
            "Returns price, currency, and as-of date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "finalize_analysis",
        "description": (
            "Terminal tool. Call when enough evidence is gathered to commit to a final verdict."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["BUY", "WATCH", "PASS"]},
                "confidence": {"type": "string", "enum": ["HIGH", "MODERATE", "LOW"]},
                "thesis": {"type": "string"},
                "key_findings": {"type": "array", "items": {"type": "string"}},
                "open_questions": {"type": "array", "items": {"type": "string"}},
                "falsifiers": {"type": "array", "items": {"type": "string"}},
                "reasoning_trace": {"type": "string"},
                "buy_below_price": {
                    "type": ["number", "null"],
                    "description": (
                        "Optional. For WATCH verdicts, a price at or below which the "
                        "company would become interesting given the valuation anchors "
                        "and evidence found. Null if the evidence does not support a "
                        "specific number, or for BUY/PASS verdicts where it is not "
                        "applicable."
                    ),
                },
            },
            "required": [
                "verdict",
                "confidence",
                "thesis",
                "key_findings",
                "open_questions",
                "falsifiers",
                "reasoning_trace",
            ],
        },
    },
]


@dataclass
class Stage4Result:
    ticker: str
    verdict: str = ""
    confidence: str = ""
    thesis: str = ""
    key_findings: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    falsifiers: list[str] = field(default_factory=list)
    reasoning_trace: str = ""
    buy_below_price: float | None = None
    num_turns: int = 0
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    tool_transcript: list[dict[str, Any]] = field(default_factory=list)
    termination_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    error: str | None = None
    financial_scope_fingerprint: str | None = None
    financial_scope_publication_fingerprint: str | None = None
    financial_scope_manifest: list[dict[str, Any]] = field(default_factory=list)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_TOOL_NAMES = frozenset(
    schema["name"] for schema in _TOOL_SCHEMAS if schema["name"] != "finalize_analysis"
)


def _tool_output_sha256(output: str) -> str:
    return hashlib.sha256(output.encode("utf-8")).hexdigest()


def _stage4_cache_scenario(
    *,
    financial_packet: Any,
    context: dict[str, Any],
    config: Stage4Config,
    tool_evidence_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    return financial_input_scenario(
        financial_packet,
        financial_inputs={
            "stage4_context": context,
            "provider_system_prompt": _SYSTEM_PROMPT,
            "provider_tool_schemas": _TOOL_SCHEMAS,
            "provider_model": config.model,
            "provider_max_output_tokens": config.max_output_tokens,
            "stage4_max_turns": config.max_turns,
            "stage4_max_cost_usd": config.max_cost_usd,
            "stage4_tool_evidence_manifest": tool_evidence_manifest,
        },
    )


def _bind_stage4_cache_input(
    *,
    ticker: str,
    context: dict[str, Any],
    config: Stage4Config,
    financial_packet: Any | None,
    tool_evidence_manifest: list[dict[str, Any]],
):
    run_as_of_date = str(context.get("as_of_date") or "")
    if financial_packet is None:
        bind_v1_financial_scope(
            context=f"discover_stage4:{ticker}:missing_packet",
            run_as_of_date=run_as_of_date,
            packets=(),
            scenarios=(),
        )
    scenario = _stage4_cache_scenario(
        financial_packet=financial_packet,
        context=context,
        config=config,
        tool_evidence_manifest=tool_evidence_manifest,
    )
    scope = bind_v1_financial_scope(
        context=f"discover_stage4_cache:{ticker}:{run_as_of_date}",
        run_as_of_date=run_as_of_date,
        packets=(financial_packet,),
        scenarios=(scenario,),
    )
    return scope, scenario


def _bind_stage4_turn_input(
    *,
    ticker: str,
    context: dict[str, Any],
    config: Stage4Config,
    financial_packet: Any,
    messages: list[dict[str, Any]],
):
    scenario = financial_input_scenario(
        financial_packet,
        financial_inputs={
            "stage4_system_prompt": _SYSTEM_PROMPT,
            "stage4_tool_schemas": _TOOL_SCHEMAS,
            "stage4_messages": messages,
            "provider_model": config.model,
            "provider_max_output_tokens": config.max_output_tokens,
        },
    )
    scope = bind_v1_financial_scope(
        context=f"discover_stage4:{ticker}:{context.get('as_of_date')}",
        run_as_of_date=str(context.get("as_of_date") or ""),
        packets=(financial_packet,),
        scenarios=(scenario,),
    )
    return scope, scenario


def _rebuild_stage4_manifest(
    *,
    raw_manifest: str | None,
    ticker: str,
    tool_dispatcher: Callable[[str, dict[str, Any]], str],
) -> list[dict[str, Any]] | None:
    """Recompute the exact evidence hashes used by a cached Stage 4 result."""

    if raw_manifest is None:
        return None
    try:
        stored_manifest = json.loads(raw_manifest)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(stored_manifest, list):
        return None

    rebuilt: list[dict[str, Any]] = []
    normalized_ticker = str(ticker).strip().upper()
    for item in stored_manifest:
        if not isinstance(item, dict):
            return None
        turn = item.get("turn")
        tool_name = item.get("tool")
        tool_input = item.get("input")
        output_sha256 = str(item.get("output_sha256") or "").lower()
        if (
            not isinstance(turn, int)
            or isinstance(turn, bool)
            or turn < 1
            or tool_name not in _EVIDENCE_TOOL_NAMES
            or not isinstance(tool_input, dict)
            or _SHA256_RE.fullmatch(output_sha256) is None
        ):
            return None
        requested_ticker = str(tool_input.get("ticker") or "").strip().upper()
        if requested_ticker and requested_ticker != normalized_ticker:
            return None
        try:
            output = tool_dispatcher(tool_name, tool_input)
        except InvalidFinancialInputError:
            raise
        except Exception as exc:
            output = f"ERROR: {exc}"
        output_text = output if isinstance(output, str) else str(output)
        rebuilt.append(
            {
                "turn": turn,
                "tool": tool_name,
                "input": tool_input,
                "output_sha256": _tool_output_sha256(output_text),
            }
        )
    return rebuilt


def deep_research_ticker(
    client,
    ticker: str,
    config: Stage4Config,
    context_builder: Callable[[str], dict[str, Any]],
    tool_dispatcher: Callable[[str, dict[str, Any]], str],
    financial_packet: Any | None = None,
) -> Stage4Result:
    """Run the deep loop on one ticker.

    Parameters
    ----------
    client              : Anthropic client (or stub)
    ticker              : the ticker to research
    config              : Stage4Config with model + budget
    context_builder     : callable(ticker) -> {"ticker":..., "user_message":...}
    tool_dispatcher     : callable(name, tool_input) -> str (tool result text)
    """
    result = Stage4Result(ticker=ticker)
    t_start = time.perf_counter()

    if financial_packet is None:
        bind_v1_financial_scope(
            context=f"discover_stage4:{ticker}:missing_packet",
            run_as_of_date="",
            packets=(),
            scenarios=(),
        )

    try:
        ctx = context_builder(ticker)
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        result.error = f"context build failed: {exc}"
        result.termination_reason = "context_build_failed"
        return result

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": ctx["user_message"]},
    ]
    baseline_scope, baseline_scenario = _bind_stage4_cache_input(
        ticker=ticker,
        context=ctx,
        config=config,
        financial_packet=financial_packet,
        tool_evidence_manifest=[],
    )
    baseline_scope.require(scenarios=(baseline_scenario,))
    baseline_fingerprint = baseline_scope.expected_scope_fingerprint

    for turn_num in range(1, config.max_turns + 1):
        financial_scope, financial_scenario = _bind_stage4_turn_input(
            ticker=ticker,
            context=ctx,
            config=config,
            financial_packet=financial_packet,
            messages=messages,
        )
        turn_context_scope, turn_context_scenario = _bind_stage4_cache_input(
            ticker=ticker,
            context=ctx,
            config=config,
            financial_packet=financial_packet,
            tool_evidence_manifest=result.financial_scope_manifest,
        )
        try:
            financial_scope.require(scenarios=(financial_scenario,))
            turn_context_scope.require(scenarios=(turn_context_scenario,))
            response = client.messages.create(
                model=config.model,
                max_tokens=config.max_output_tokens,
                system=_SYSTEM_PROMPT,
                tools=_TOOL_SCHEMAS,
                tool_choice={"type": "auto"},
                messages=messages,
            )
            post_response_scope, post_response_scenario = _bind_stage4_turn_input(
                ticker=ticker,
                context=ctx,
                config=config,
                financial_packet=financial_packet,
                messages=messages,
            )
            post_response_scope.require(scenarios=(post_response_scenario,))
            require_discover_financial_scope_unchanged(
                stage=4,
                ticker=ticker,
                run_as_of_date=str(ctx.get("as_of_date") or ""),
                authorized_fingerprint=(financial_scope.expected_scope_fingerprint),
                current_fingerprint=(post_response_scope.expected_scope_fingerprint),
                phase=f"turn_{turn_num}_post_response",
            )
            post_response_context_scope, post_response_context_scenario = _bind_stage4_cache_input(
                ticker=ticker,
                context=ctx,
                config=config,
                financial_packet=financial_packet,
                tool_evidence_manifest=result.financial_scope_manifest,
            )
            post_response_context_scope.require(scenarios=(post_response_context_scenario,))
            require_discover_financial_scope_unchanged(
                stage=4,
                ticker=ticker,
                run_as_of_date=str(ctx.get("as_of_date") or ""),
                authorized_fingerprint=(turn_context_scope.expected_scope_fingerprint),
                current_fingerprint=(post_response_context_scope.expected_scope_fingerprint),
                phase=f"turn_{turn_num}_context_post_response",
            )
        except InvalidFinancialInputError:
            raise
        except DiscoverCostBudgetExceeded:
            raise
        except Exception as exc:
            try:
                failed_response_scope, failed_response_scenario = _bind_stage4_turn_input(
                    ticker=ticker,
                    context=ctx,
                    config=config,
                    financial_packet=financial_packet,
                    messages=messages,
                )
                failed_response_scope.require(scenarios=(failed_response_scenario,))
                require_discover_financial_scope_unchanged(
                    stage=4,
                    ticker=ticker,
                    run_as_of_date=str(ctx.get("as_of_date") or ""),
                    authorized_fingerprint=(financial_scope.expected_scope_fingerprint),
                    current_fingerprint=(failed_response_scope.expected_scope_fingerprint),
                    phase=f"turn_{turn_num}_failed_response",
                )
                failed_context_scope, failed_context_scenario = _bind_stage4_cache_input(
                    ticker=ticker,
                    context=ctx,
                    config=config,
                    financial_packet=financial_packet,
                    tool_evidence_manifest=result.financial_scope_manifest,
                )
                failed_context_scope.require(scenarios=(failed_context_scenario,))
                require_discover_financial_scope_unchanged(
                    stage=4,
                    ticker=ticker,
                    run_as_of_date=str(ctx.get("as_of_date") or ""),
                    authorized_fingerprint=(turn_context_scope.expected_scope_fingerprint),
                    current_fingerprint=(failed_context_scope.expected_scope_fingerprint),
                    phase=f"turn_{turn_num}_context_failed_response",
                )
            except InvalidFinancialInputError as integrity_exc:
                raise integrity_exc from exc
            result.error = f"turn {turn_num}: {exc}"
            result.termination_reason = "api_error"
            break

        usage = response.usage
        cost = (
            usage.input_tokens / 1_000_000 * config.input_usd_per_mtok
            + usage.output_tokens / 1_000_000 * config.output_usd_per_mtok
        )
        result.input_tokens += usage.input_tokens
        result.output_tokens += usage.output_tokens
        result.cost_usd += cost
        result.num_turns = turn_num

        assistant_blocks: list[dict[str, Any]] = []
        tool_uses: list[tuple[str, str, dict]] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                assistant_blocks.append({"type": "text", "text": getattr(block, "text", "")})
            elif btype == "tool_use":
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input or {},
                    }
                )
                tool_uses.append((block.id, block.name, block.input or {}))
                result.tool_call_counts[block.name] = result.tool_call_counts.get(block.name, 0) + 1

        messages.append({"role": "assistant", "content": assistant_blocks})

        if not tool_uses:
            result.termination_reason = "no_tool_call"
            break

        finalize_call = next(
            ((i, n, inp) for i, n, inp in tool_uses if n == "finalize_analysis"),
            None,
        )
        if finalize_call is not None:
            _, _, final_input = finalize_call
            result.verdict = final_input.get("verdict", "")
            result.confidence = final_input.get("confidence", "")
            result.thesis = final_input.get("thesis", "")
            result.key_findings = list(final_input.get("key_findings", []))
            result.open_questions = list(final_input.get("open_questions", []))
            result.falsifiers = list(final_input.get("falsifiers", []))
            result.reasoning_trace = final_input.get("reasoning_trace", "")
            bbp = final_input.get("buy_below_price")
            if isinstance(bbp, (int, float)) and bbp > 0:
                result.buy_below_price = float(bbp)
            result.termination_reason = "finalize_analysis"
            break

        tool_results: list[dict[str, Any]] = []
        for tu_id, tu_name, tu_input in tool_uses:
            requested_ticker = str(tu_input.get("ticker") or "").strip().upper()
            if requested_ticker and requested_ticker != ticker.strip().upper():
                mismatched_scenario = financial_input_scenario(
                    financial_packet,
                    financial_inputs={
                        "stage4_tool_name": tu_name,
                        "stage4_tool_input": tu_input,
                    },
                )
                mismatched_scenario["ticker"] = requested_ticker
                bind_v1_financial_scope(
                    context=(f"discover_stage4_tool_ticker:{ticker}:{requested_ticker}"),
                    run_as_of_date=str(ctx.get("as_of_date") or ""),
                    packets=(financial_packet,),
                    scenarios=(mismatched_scenario,),
                )
            try:
                tool_output = tool_dispatcher(tu_name, tu_input)
            except InvalidFinancialInputError:
                raise
            except Exception as exc:
                tool_output = f"ERROR: {exc}"
            tool_output = tool_output if isinstance(tool_output, str) else str(tool_output)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": tool_output,
                }
            )
            # Record in transcript for audit
            result.tool_transcript.append(
                {
                    "turn": turn_num,
                    "tool": tu_name,
                    "input": tu_input,
                    "output_preview": tool_output[:500]
                    if isinstance(tool_output, str)
                    else str(tool_output)[:500],
                    "output_chars": len(tool_output) if isinstance(tool_output, str) else 0,
                }
            )
            result.financial_scope_manifest.append(
                {
                    "turn": turn_num,
                    "tool": tu_name,
                    "input": tu_input,
                    "output_sha256": _tool_output_sha256(tool_output),
                }
            )
        messages.append({"role": "user", "content": tool_results})

        if result.cost_usd >= config.max_cost_usd:
            result.termination_reason = "budget_cap"
            break
    else:
        result.termination_reason = "max_turns"

    current_baseline_scope, current_baseline_scenario = _bind_stage4_cache_input(
        ticker=ticker,
        context=ctx,
        config=config,
        financial_packet=financial_packet,
        tool_evidence_manifest=[],
    )
    current_baseline_scope.require(scenarios=(current_baseline_scenario,))
    require_discover_financial_scope_unchanged(
        stage=4,
        ticker=ticker,
        run_as_of_date=str(ctx.get("as_of_date") or ""),
        authorized_fingerprint=baseline_fingerprint,
        current_fingerprint=current_baseline_scope.expected_scope_fingerprint,
        phase="post_execution_baseline",
    )
    cache_scope, cache_scenario = _bind_stage4_cache_input(
        ticker=ticker,
        context=ctx,
        config=config,
        financial_packet=financial_packet,
        tool_evidence_manifest=result.financial_scope_manifest,
    )
    cache_scope.require(scenarios=(cache_scenario,))
    result.financial_scope_fingerprint = cache_scope.expected_scope_fingerprint
    result.wall_seconds = time.perf_counter() - t_start
    return result


def run_stage4(
    db_path: str | Path,
    sweep_id: str,
    tickers: Iterable[str],
    client,
    config: Stage4Config | None = None,
    context_builder: Callable[[str], dict[str, Any]] | None = None,
    tool_dispatcher: Callable[[str, dict[str, Any]], str] | None = None,
    financial_packets: dict[str, Any] | None = None,
) -> list[Stage4Result]:
    """Run Stage 4 on a list of tickers. Writes each result to the session DB."""
    cfg = config or Stage4Config()

    if context_builder is None or tool_dispatcher is None:
        from app.discover.stage4_context import (
            default_context_builder,
            default_tool_dispatcher,
        )

        if context_builder is None:
            context_builder = default_context_builder
        if tool_dispatcher is None:
            tool_dispatcher = default_tool_dispatcher

    existing = stage_result_financial_fingerprints(
        db_path,
        stage=4,
        sweep_id=sweep_id,
    )
    stored_manifests = stage4_result_financial_manifests(
        db_path,
        sweep_id=sweep_id,
    )
    results: list[Stage4Result] = []
    for ticker in tickers:
        normalized_ticker = str(ticker).strip().upper()
        if normalized_ticker in existing:
            try:
                cached_context = context_builder(ticker)
            except InvalidFinancialInputError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"cannot authorize cached Discover Stage 4 result for {ticker}: {exc}"
                ) from exc
            baseline_scope, _ = _bind_stage4_cache_input(
                ticker=ticker,
                context=cached_context,
                config=cfg,
                financial_packet=(financial_packets or {}).get(normalized_ticker),
                tool_evidence_manifest=[],
            )
            stored_fingerprint = existing[normalized_ticker]
            if _SHA256_RE.fullmatch(str(stored_fingerprint or "").strip().lower()) is None:
                require_discover_result_financial_scope(
                    stage=4,
                    sweep_id=sweep_id,
                    ticker=ticker,
                    run_as_of_date=str(cached_context.get("as_of_date") or ""),
                    stored_fingerprint=stored_fingerprint,
                    expected_fingerprint=(baseline_scope.expected_scope_fingerprint),
                )
                raise AssertionError("unreachable")
            rebuilt_manifest = _rebuild_stage4_manifest(
                raw_manifest=stored_manifests.get(normalized_ticker),
                ticker=ticker,
                tool_dispatcher=tool_dispatcher,
            )
            if rebuilt_manifest is None:
                require_discover_result_financial_scope(
                    stage=4,
                    sweep_id=sweep_id,
                    ticker=ticker,
                    run_as_of_date=str(cached_context.get("as_of_date") or ""),
                    stored_fingerprint=None,
                    expected_fingerprint=(baseline_scope.expected_scope_fingerprint),
                )
                raise AssertionError("unreachable")
            cached_scope, _ = _bind_stage4_cache_input(
                ticker=ticker,
                context=cached_context,
                config=cfg,
                financial_packet=(financial_packets or {}).get(normalized_ticker),
                tool_evidence_manifest=rebuilt_manifest,
            )
            require_discover_result_financial_scope(
                stage=4,
                sweep_id=sweep_id,
                ticker=ticker,
                run_as_of_date=str(cached_context.get("as_of_date") or ""),
                stored_fingerprint=stored_fingerprint,
                expected_fingerprint=(cached_scope.expected_scope_fingerprint),
            )
            continue
        require_no_prior_discover_paid_attempt(
            db_path,
            stage=4,
            sweep_id=sweep_id,
            ticker=normalized_ticker,
        )
        durable_client = DurableDiscoverClient(
            client,
            db_path=db_path,
            stage=4,
            sweep_id=sweep_id,
            ticker=normalized_ticker,
            input_usd_per_mtok=cfg.input_usd_per_mtok,
            output_usd_per_mtok=cfg.output_usd_per_mtok,
            hard_ticker_cost_limit_usd=cfg.max_cost_usd,
        )
        result = deep_research_ticker(
            durable_client,
            ticker,
            cfg,
            context_builder,
            tool_dispatcher,
            financial_packet=(financial_packets or {}).get(ticker.upper()),
        )
        attempt_summary = discover_paid_attempt_summary(
            db_path,
            stage=4,
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
                publication_context = context_builder(ticker)
            except InvalidFinancialInputError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"cannot authorize Discover Stage 4 result for {ticker}: {exc}"
                ) from exc
            rebuilt_manifest = _rebuild_stage4_manifest(
                raw_manifest=json.dumps(result.financial_scope_manifest),
                ticker=ticker,
                tool_dispatcher=tool_dispatcher,
            )
            if rebuilt_manifest is None:
                require_discover_financial_scope_unchanged(
                    stage=4,
                    ticker=ticker,
                    run_as_of_date=str(publication_context.get("as_of_date") or ""),
                    authorized_fingerprint=result.financial_scope_fingerprint,
                    current_fingerprint=None,
                    phase="pre_persistence",
                    sweep_id=sweep_id,
                )
                raise AssertionError("unreachable")
            publication_scope, publication_scenario = _bind_stage4_cache_input(
                ticker=ticker,
                context=publication_context,
                config=cfg,
                financial_packet=(financial_packets or {}).get(normalized_ticker),
                tool_evidence_manifest=rebuilt_manifest,
            )
            publication_scope.require(scenarios=(publication_scenario,))
            require_discover_financial_scope_unchanged(
                stage=4,
                ticker=ticker,
                run_as_of_date=str(publication_context.get("as_of_date") or ""),
                authorized_fingerprint=result.financial_scope_fingerprint,
                current_fingerprint=publication_scope.expected_scope_fingerprint,
                phase="pre_persistence",
                sweep_id=sweep_id,
            )
            result.financial_scope_publication_fingerprint = (
                publication_scope.expected_scope_fingerprint
            )
            publication_evidence = build_discover_publication_evidence(
                stage=4,
                ticker=ticker,
                scope_fingerprint=result.financial_scope_publication_fingerprint,
                primary_evidence={
                    "context": publication_context,
                    "model": cfg.model,
                    "max_tokens": cfg.max_output_tokens,
                    "max_turns": cfg.max_turns,
                    "max_cost_usd": cfg.max_cost_usd,
                    "system": _SYSTEM_PROMPT,
                    "tools": _TOOL_SCHEMAS,
                },
                stage4_tool_manifest=rebuilt_manifest,
            )
        if publication_scope is None or publication_evidence is None:
            raise ValueError(
                f"Discover Stage 4 production result for {result.ticker} lacks authorization"
            )
        _publish_discover_result(
            db_path,
            stage=4,
            row={
                "sweep_id": sweep_id,
                "ticker": result.ticker,
                "verdict": result.verdict,
                "confidence": result.confidence,
                "thesis": result.thesis,
                "key_findings_json": result.key_findings,
                "open_questions_json": result.open_questions,
                "falsifiers_json": result.falsifiers,
                "reasoning_trace": result.reasoning_trace,
                "num_turns": result.num_turns,
                "tool_call_counts_json": result.tool_call_counts,
                "termination_reason": result.termination_reason,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
                "wall_seconds": result.wall_seconds,
                "error": result.error,
                "financial_scope_fingerprint": result.financial_scope_fingerprint,
                "financial_scope_publication_fingerprint": (
                    result.financial_scope_publication_fingerprint
                ),
                "financial_scope_manifest_json": result.financial_scope_manifest,
                "publication_evidence_json": None,
            },
            financial_scope=publication_scope,
            publication_evidence=publication_evidence,
            financial_scenarios=(publication_scenario,),
        )

    return results
