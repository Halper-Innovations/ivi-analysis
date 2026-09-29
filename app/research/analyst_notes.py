"""LLM-first analyst notes — Stage A of the parallel hybrid pipeline.

Reads curated filing sections (MDA, Risk Factors, Financial Notes) via LLM,
produces structured findings with verbatim citations, validates citations
against filing text.

Public API: generate_analyst_notes(filing_html, form_type, scorecard, ticker)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    InvalidFinancialInputError,
    require_financial_integrity_scope,
)
from app.autonomous.v1_financial_context import BoundV1FinancialScope
from app.llm.providers.retry_guard import llm_physical_attempt_guard
from app.llm.usage_capture import (
    attach_provider_usage_to_exception,
    provider_failed_attempt_capture,
    provider_usage_records,
    provider_usage_records_from_exception,
    provider_usage_request,
    record_provider_usage,
)

logger = logging.getLogger(__name__)

FinancialScenarioSource = Sequence[Any] | Callable[[], Sequence[Any]]


@dataclass
class AnalystCitation:
    """Citation from LLM analyst read."""

    section: str  # mda / risk_factors / fin_notes
    excerpt: str  # verbatim, under 300 chars
    block_id: str | None  # from filing parser, e.g., "mda_p12"


@dataclass
class AnalystNote:
    """Single finding from LLM filing read."""

    category: str  # POSITIVE / RISK / SURPRISE / ADJUSTMENT_TRIGGER
    claim: str
    direction: str | None  # BULLISH / BEARISH / NEUTRAL
    severity: str  # HIGH / MODERATE / LOW
    citations: list[AnalystCitation]
    suggested_adjustment: str | None
    validation_status: str  # VERIFIED / UNVERIFIED


@dataclass
class AnalystNotes:
    """Complete output of Stage A."""

    ticker: str
    positives: list[AnalystNote]
    risks: list[AnalystNote]
    surprises: list[AnalystNote]
    adjustment_triggers: list[AnalystNote]
    overall_assessment: str
    filing_sections_read: list[str]


# ---------------------------------------------------------------------------
# Citation Validation
# ---------------------------------------------------------------------------

_VALIDATION_THRESHOLD = 0.8  # 80% word overlap required


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _word_overlap_score(excerpt_tokens: list[str], reference_text: str) -> float:
    """Fraction of excerpt tokens found in a contiguous window of reference text."""
    if not excerpt_tokens:
        return 0.0
    ref_tokens = _tokenize(reference_text)
    if not ref_tokens:
        return 0.0
    window_size = len(excerpt_tokens) * 3  # generous window
    best_score = 0.0
    for start in range(max(1, len(ref_tokens) - window_size + 1)):
        window = set(ref_tokens[start : start + window_size])
        matched = sum(1 for t in excerpt_tokens if t in window)
        score = matched / len(excerpt_tokens)
        if score > best_score:
            best_score = score
            if best_score >= 1.0:
                break
    return best_score


def _validate_single_citation(
    citation: AnalystCitation,
    block_index: dict,
    section_text: dict[str, str],
) -> bool:
    """Validate one citation. Returns True if verified."""
    excerpt_tokens = _tokenize(citation.excerpt)
    if not excerpt_tokens:
        return False

    # Step 1: try block-level match
    if citation.block_id and citation.block_id in block_index:
        block = block_index[citation.block_id]
        score = _word_overlap_score(excerpt_tokens, block.text)
        if score >= _VALIDATION_THRESHOLD:
            return True

    # Step 2: fall back to entire section text (concatenated blocks).
    # This handles excerpts that span adjacent blocks within a section.
    full_section = section_text.get(citation.section, "")
    if full_section:
        score = _word_overlap_score(excerpt_tokens, full_section)
        if score >= _VALIDATION_THRESHOLD:
            return True

    return False


def validate_citations(
    notes: list[AnalystNote],
    blocks: list,
) -> list[AnalystNote]:
    """Validate citations on each note. Sets validation_status to VERIFIED or UNVERIFIED.

    Returns new list of AnalystNote with updated validation_status.
    Does not mutate input notes.
    """
    block_index = {b.block_id: b for b in blocks}
    # Concatenate all block texts per section for section-level fallback.
    # This allows validation of excerpts spanning adjacent blocks.
    section_blocks: dict[str, list] = {}
    for b in blocks:
        section_blocks.setdefault(b.section, []).append(b)
    section_text: dict[str, str] = {
        section: " ".join(b.text for b in blks) for section, blks in section_blocks.items()
    }

    result: list[AnalystNote] = []
    for note in notes:
        if not note.citations:
            result.append(
                AnalystNote(
                    category=note.category,
                    claim=note.claim,
                    direction=note.direction,
                    severity=note.severity,
                    citations=note.citations,
                    suggested_adjustment=note.suggested_adjustment,
                    validation_status="UNVERIFIED",
                )
            )
            continue

        all_valid = all(
            _validate_single_citation(c, block_index, section_text) for c in note.citations
        )
        result.append(
            AnalystNote(
                category=note.category,
                claim=note.claim,
                direction=note.direction,
                severity=note.severity,
                citations=note.citations,
                suggested_adjustment=note.suggested_adjustment,
                validation_status="VERIFIED" if all_valid else "UNVERIFIED",
            )
        )
    return result


# ---------------------------------------------------------------------------
# Curated Section Extraction
# ---------------------------------------------------------------------------

from app.llm.providers import get_llm_provider  # noqa: E402
from app.research.current_event_context import CurrentEventContext  # noqa: E402
from app.research.evidence_searcher import (  # noqa: E402
    FilingBlock,
    build_current_event_blocks,
    parse_filing_context_blocks,
)
from app.research.filing_context import FilingContext, build_inline_filing_context  # noqa: E402

_CURATED_SECTIONS = frozenset(
    {
        "mda",
        "risk_factors",
        "fin_notes",
        "results_guidance",
        "agreement_termination",
        "financing_liquidity",
        "distress_restructuring",
        "restatement_controls",
        "leadership_governance",
        "strategic_transaction",
        "legal_regulatory",
        "other_material",
    }
)


def extract_curated_sections(
    filing_html: str | None,
    form_type: str | None,
) -> tuple[list[str], list[FilingBlock]]:
    """Extract curated filing sections (MDA, Risk Factors, Financial Notes).

    Returns (section_labels, filtered_blocks). Reuses evidence_searcher's
    parsing infrastructure.
    """
    filing_context = build_inline_filing_context(
        filing_html,
        ticker="INLINE",
        form_type=form_type,
    )
    return extract_curated_sections_from_filing_context(filing_context)


# ---------------------------------------------------------------------------
# JSON Schema for OpenAI structured output
# ---------------------------------------------------------------------------

_NOTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["POSITIVE", "RISK", "SURPRISE", "ADJUSTMENT_TRIGGER"],
        },
        "claim": {"type": "string"},
        "direction": {
            "anyOf": [
                {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
                {"type": "null"},
            ]
        },
        "severity": {"type": "string", "enum": ["HIGH", "MODERATE", "LOW"]},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "excerpt": {"type": "string"},
                    "block_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["section", "excerpt", "block_id"],
                "additionalProperties": False,
            },
        },
        "suggested_adjustment": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["category", "claim", "direction", "severity", "citations", "suggested_adjustment"],
    "additionalProperties": False,
}

ANALYST_NOTES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_assessment": {"type": "string"},
        "positives": {"type": "array", "items": _NOTE_SCHEMA},
        "risks": {"type": "array", "items": _NOTE_SCHEMA},
        "surprises": {"type": "array", "items": _NOTE_SCHEMA},
        "adjustment_triggers": {"type": "array", "items": _NOTE_SCHEMA},
    },
    "required": ["overall_assessment", "positives", "risks", "surprises", "adjustment_triggers"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_ANALYST_PROMPT_TEMPLATE = """\
You are a senior equity research analyst reading a company's recent SEC filings.
The company trades under ticker {ticker}.

Valuation context:
- DCF fair value: {dcf}
- EPV fair value: {epv}
- Current price: {price}
- Market cap: {market_cap}

Read the filing sections below and produce structured analyst notes.
For each finding, provide a verbatim excerpt (under 300 characters) from the filings
as a citation, with the section name and block_id if visible.

Filing sections:
{sections_text}

Recent company-controlled current events:
{recent_events_text}

Produce findings in four categories:
1. POSITIVE — genuinely strong attributes (moats, capital allocation, durable demand)
2. RISK — factors that could impair value (competitive, regulatory, operational, accounting)
3. SURPRISE — unusual or counterintuitive observations
4. ADJUSTMENT_TRIGGER — specific findings that should modify the valuation (include direction and suggested adjustment)

Rules:
- Every finding MUST have at least one citation with a verbatim excerpt
- Limit to 3-5 findings per category (focus on the most material)
- Include an overall_assessment (2-3 sentences summarizing the filing)
- Be specific and quantitative where possible
"""


def _build_prompt(
    ticker: str,
    scorecard: dict[str, Any],
    sections_text: str,
    recent_events_text: str,
) -> str:
    pzd = scorecard.get("pricing_zone_detail") or {}
    dcf = pzd.get("dcf_base")
    epv = pzd.get("epv_adjusted")
    price = pzd.get("current_price")
    market_cap = pzd.get("market_cap")
    return _ANALYST_PROMPT_TEMPLATE.format(
        ticker=ticker,
        dcf=f"${dcf:.2f}" if dcf is not None else "N/A",
        epv=f"${epv:.2f}" if epv is not None else "N/A",
        price=f"${price:.2f}" if price is not None else "N/A",
        market_cap=f"${market_cap:,.0f}M" if market_cap is not None else "N/A",
        sections_text=sections_text,
        recent_events_text=recent_events_text or "None",
    )


def _format_sections_text(blocks: list[FilingBlock]) -> str:
    """Format filing blocks into labeled text for the prompt."""
    current_group: tuple[str, str] | None = None
    parts: list[str] = []
    for block in blocks:
        source_label = block.source_form_type or "FILING"
        source_date = block.source_filing_date or "undated"
        source_heading = f"{source_label} ({source_date})"
        section_heading = block.section.upper()
        if block.item_code:
            section_heading = f"ITEM {block.item_code} / {section_heading}"
        group = (source_heading, section_heading)
        if group != current_group:
            current_group = group
            parts.append(f"\n=== {source_heading} / {section_heading} ===\n")
        parts.append(f"[{block.block_id}] {block.text}\n")
    return "".join(parts)


def _format_recent_events_text(blocks: list[FilingBlock]) -> str:
    if not blocks:
        return "None"
    parts: list[str] = []
    for block in blocks[:10]:
        source_date = block.source_published_at or "undated"
        source_type = (block.source_type or "current_event").upper()
        title = block.source_title or block.source_url or block.block_id
        parts.append(f"\n=== {source_type} ({source_date}) ===\n")
        parts.append(f"[{block.block_id}] {title}\n")
        parts.append(f"{block.text}\n")
    return "".join(parts)


def _parse_llm_response(
    raw: dict[str, Any], ticker: str, sections_read: list[str], blocks: list[FilingBlock]
) -> AnalystNotes:
    """Parse raw LLM JSON into AnalystNotes with citation validation."""

    def _parse_notes(items: list[dict]) -> list[AnalystNote]:
        parsed = []
        for item in items:
            citations = [
                AnalystCitation(
                    section=c.get("section", ""),
                    excerpt=c.get("excerpt", ""),
                    block_id=c.get("block_id"),
                )
                for c in item.get("citations", [])
            ]
            parsed.append(
                AnalystNote(
                    category=item.get("category", ""),
                    claim=item.get("claim", ""),
                    direction=item.get("direction"),
                    severity=item.get("severity", "LOW"),
                    citations=citations,
                    suggested_adjustment=item.get("suggested_adjustment"),
                    validation_status="UNVERIFIED",  # set by validate_citations
                )
            )
        return parsed

    positives = _parse_notes(raw.get("positives", []))
    risks = _parse_notes(raw.get("risks", []))
    surprises = _parse_notes(raw.get("surprises", []))
    triggers = _parse_notes(raw.get("adjustment_triggers", []))

    # Validate all notes against filing blocks
    all_notes = positives + risks + surprises + triggers
    validated = validate_citations(all_notes, blocks)

    # Split back into categories (order preserved)
    idx = 0
    v_positives = validated[idx : idx + len(positives)]
    idx += len(positives)
    v_risks = validated[idx : idx + len(risks)]
    idx += len(risks)
    v_surprises = validated[idx : idx + len(surprises)]
    idx += len(surprises)
    v_triggers = validated[idx : idx + len(triggers)]

    return AnalystNotes(
        ticker=ticker,
        positives=v_positives,
        risks=v_risks,
        surprises=v_surprises,
        adjustment_triggers=v_triggers,
        overall_assessment=raw.get("overall_assessment", ""),
        filing_sections_read=sections_read,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_analyst_notes(
    filing_html: str | None,
    form_type: str | None,
    scorecard: dict[str, Any],
    ticker: str,
    *,
    current_event_context: CurrentEventContext | None = None,
    financial_integrity_scope: BoundV1FinancialScope | None = None,
    financial_scenarios: FinancialScenarioSource | None = None,
) -> AnalystNotes | None:
    """Generate LLM-first analyst notes from curated filing sections.

    Returns None if:
    - No filing text available
    - LLM provider is disabled
    - No curated sections found in filing
    - LLM call fails
    """
    filing_context = build_inline_filing_context(
        filing_html,
        ticker=ticker,
        form_type=form_type,
    )
    return generate_analyst_notes_from_filing_context(
        filing_context,
        scorecard=scorecard,
        ticker=ticker,
        current_event_context=current_event_context,
        financial_integrity_scope=financial_integrity_scope,
        financial_scenarios=financial_scenarios,
    )


def extract_curated_sections_from_filing_context(
    filing_context: FilingContext,
) -> tuple[list[str], list[FilingBlock]]:
    """Extract curated sections across one or more filings."""
    if not filing_context.documents:
        return [], []
    all_blocks = parse_filing_context_blocks(filing_context)
    filtered = [b for b in all_blocks if b.section in _CURATED_SECTIONS]
    sections = sorted(set(b.section for b in filtered))
    return sections, filtered


def generate_analyst_notes_from_filing_context(
    filing_context: FilingContext,
    *,
    scorecard: dict[str, Any],
    ticker: str,
    current_event_context: CurrentEventContext | None = None,
    financial_integrity_scope: BoundV1FinancialScope | None = None,
    financial_scenarios: FinancialScenarioSource | None = None,
) -> AnalystNotes | None:
    """Generate LLM-first analyst notes from curated sections across filings."""
    if not filing_context.documents:
        return None

    provider = get_llm_provider()
    if provider.provider_name == "disabled":
        return None

    sections, blocks = extract_curated_sections_from_filing_context(filing_context)
    current_event_blocks = build_current_event_blocks(current_event_context)
    if not blocks and not current_event_blocks:
        return None

    sections_text = _format_sections_text(blocks)
    recent_events_text = _format_recent_events_text(current_event_blocks)
    prompt = _build_prompt(ticker, scorecard, sections_text, recent_events_text)

    schema_name = "analyst_notes_v1"
    max_output_tokens = 4000
    failed_attempts: list[dict[str, Any]] = []
    successful_attempts: list[dict[str, Any]] = []
    try:

        def require_exact_scope(_attempt=None):
            if financial_integrity_scope is None:
                require_financial_integrity_scope(
                    FinancialIntegrityScope(
                        context=f"analyst_notes:{ticker}",
                        run_as_of_date="",
                    )
                )
            else:
                current_scenarios = (
                    financial_scenarios() if callable(financial_scenarios) else financial_scenarios
                )
                financial_integrity_scope.require(scenarios=current_scenarios)

        with provider_usage_request(
            provider=provider,
            prompt=prompt,
            schema=ANALYST_NOTES_SCHEMA,
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
                    require_exact_scope()
                    provider_options = {"max_output_tokens": max_output_tokens}
                    provider_options.update(request_kwargs)
                    result = provider.synthesize_json(
                        prompt=prompt,
                        schema=ANALYST_NOTES_SCHEMA,
                        schema_name=schema_name,
                        **provider_options,
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
                raise
            successful_attempts = provider_usage_records(
                provider=provider,
                result=result,
                prompt=prompt,
                schema_name=schema_name,
            )
            for usage_record in successful_attempts:
                record_provider_usage(usage_record)
            require_exact_scope()
        raw = json.loads(result.json_text)
    except InvalidFinancialInputError as exc:
        attach_provider_usage_to_exception(exc, [*failed_attempts, *successful_attempts])
        raise
    except Exception as exc:
        attach_provider_usage_to_exception(exc, [*failed_attempts, *successful_attempts])
        try:
            require_exact_scope()
        except InvalidFinancialInputError as integrity_exc:
            attach_provider_usage_to_exception(
                integrity_exc,
                [*failed_attempts, *successful_attempts],
            )
            raise
        logger.exception("analyst_notes: LLM call failed for %s", ticker)
        return None

    notes = _parse_llm_response(raw, ticker, sections, blocks + current_event_blocks)
    try:
        require_exact_scope()
    except InvalidFinancialInputError as exc:
        attach_provider_usage_to_exception(exc, [*failed_attempts, *successful_attempts])
        raise
    return notes
