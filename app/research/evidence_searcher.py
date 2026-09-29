"""Filing evidence engine — searches filing text for hypothesis evidence.

Layers:
1. Parse filing HTML into section-labeled text blocks
2. Score and rank blocks per evidence item (token matching + query expansion)
3. Adjudicate top candidates via LLM
4. Aggregate into hypothesis-level results

Public API: search_evidence(hypotheses, filing_html, form_type, max_hypotheses)
"""

from __future__ import annotations

import html as _html_module
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
from app.dossier.sections import event_category_for_8k_section, item_code_for_8k_section
from app.research.current_event_context import CurrentEventContext
from app.research.hypothesis_generator import Hypothesis, EvidenceNeed
from app.research.filing_context import FilingContext, build_inline_filing_context
from app.research.source_quality import classify_source_quality
from app.util.hashing import sha256_text

logger = logging.getLogger(__name__)

FinancialScenarioSource = Sequence[Any] | Callable[[], Sequence[Any]]


def _require_paid_financial_scope(
    scope: BoundV1FinancialScope | None,
    *,
    scenarios: FinancialScenarioSource | None,
    context: str,
) -> None:
    if scope is None:
        require_financial_integrity_scope(
            FinancialIntegrityScope(
                context=context,
                run_as_of_date="",
            )
        )
        return
    current_scenarios = scenarios() if callable(scenarios) else scenarios
    scope.require(scenarios=current_scenarios)


# ---------------------------------------------------------------------------
# Layer 1: Filing Block Parser
# ---------------------------------------------------------------------------


@dataclass
class FilingBlock:
    """A text block from a parsed filing, labeled with its section."""

    block_id: str  # "{section}_p{ordinal}"
    section: str  # "business" / "risk_factors" / "mda" / "fin_notes" / "other"
    ordinal: int  # position within section (0-indexed)
    text: str
    raw_section: str | None = None
    source_form_type: str | None = None
    source_filing_date: str | None = None
    source_accession: str | None = None
    source_role: str | None = None
    item_code: str | None = None
    event_category: str | None = None
    source_type: str | None = None
    source_title: str | None = None
    source_url: str | None = None
    source_published_at: str | None = None
    source_quality: dict[str, Any] | None = None


def _prepare_filing_text(html: str) -> str:
    """Convert raw filing HTML to text with paragraph boundaries preserved.

    Inserts structural boundaries for block-level tags, strips remaining HTML,
    decodes entities, collapses excessive newlines. Does NOT collapse spaces/tabs.
    """
    text = html
    # Remove script/style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Insert \n\n before block-level opening tags (not <div>) — lookahead preserves tag
    text = re.sub(r"<(?=(?:p|h[1-6]|tr|li|blockquote)[\s>/])", "\n\n<", text, flags=re.IGNORECASE)
    # Insert \n after block-level closing tags
    text = re.sub(r"(</(?:p|h[1-6]|tr|li|blockquote)>)", r"\1\n", text, flags=re.IGNORECASE)
    # <br> → \n
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    # Insert \t before <td>/<th> — lookahead preserves tag
    text = re.sub(r"<(?=(?:td|th)[\s>/])", "\t<", text, flags=re.IGNORECASE)
    # Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode HTML entities
    text = _html_module.unescape(text)
    # Collapse 3+ newlines to \n\n
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Section Detection
# ---------------------------------------------------------------------------

_10K_PATTERNS: dict[str, re.Pattern] = {
    "business": re.compile(r"item\s+1\.?\s+business", re.IGNORECASE),
    "risk_factors": re.compile(r"item\s+1a\.?\s+risk\s+factors", re.IGNORECASE),
    "mda": re.compile(r"item\s+7\.?\s+management", re.IGNORECASE),
    "fin_notes_8": re.compile(r"item\s+8\.?\s+financial\s+statements", re.IGNORECASE),
    "fin_notes_notes": re.compile(
        r"notes?\s+to\s+(?:consolidated\s+)?financial\s+statements", re.IGNORECASE
    ),
}

_20F_PATTERNS: dict[str, re.Pattern] = {
    "business": re.compile(r"item\s+4\.?\s+information\s+on\s+the\s+company", re.IGNORECASE),
    "risk_factors": re.compile(r"(?:item\s+3d\.?\s+|d\.?\s+)risk\s+factors", re.IGNORECASE),
    "mda": re.compile(r"item\s+5\.?\s+operating\s+and\s+financial\s+review", re.IGNORECASE),
    "fin_notes_notes": re.compile(
        r"notes?\s+to\s+(?:consolidated\s+)?financial\s+statements", re.IGNORECASE
    ),
}

_10Q_PATTERNS: dict[str, re.Pattern] = {
    "risk_factors": re.compile(r"item\s+1a\.?\s+risk\s+factors", re.IGNORECASE),
    "fin_notes_item1": re.compile(r"item\s+1\.?\s+financial\s+statements", re.IGNORECASE),
    "mda": re.compile(
        r"item\s+2\.?\s+management.{0,50}?discussion.{0,50}?analysis",
        re.IGNORECASE | re.DOTALL,
    ),
    "fin_notes_notes": re.compile(
        r"notes?\s+to\s+(?:condensed\s+)?(?:consolidated\s+)?financial\s+statements",
        re.IGNORECASE,
    ),
}

_8K_PATTERNS: dict[str, re.Pattern] = {
    "item_101": re.compile(r"item\s+1\.01\b", re.IGNORECASE),
    "item_102": re.compile(r"item\s+1\.02\b", re.IGNORECASE),
    "item_103": re.compile(r"item\s+1\.03\b", re.IGNORECASE),
    "item_202": re.compile(r"item\s+2\.02\b", re.IGNORECASE),
    "item_203": re.compile(r"item\s+2\.03\b", re.IGNORECASE),
    "item_204": re.compile(r"item\s+2\.04\b", re.IGNORECASE),
    "item_205": re.compile(r"item\s+2\.05\b", re.IGNORECASE),
    "item_206": re.compile(r"item\s+2\.06\b", re.IGNORECASE),
    "item_301": re.compile(r"item\s+3\.01\b", re.IGNORECASE),
    "item_402": re.compile(r"item\s+4\.02\b", re.IGNORECASE),
    "item_502": re.compile(r"item\s+5\.02\b", re.IGNORECASE),
    "item_701": re.compile(r"item\s+7\.01\b", re.IGNORECASE),
    "item_801": re.compile(r"item\s+8\.01\b", re.IGNORECASE),
    "item_901": re.compile(r"item\s+9\.01\b", re.IGNORECASE),
}

_LABEL_MAP = {
    "fin_notes_8": "fin_notes",
    "fin_notes_notes": "fin_notes",
    "fin_notes_item1": "fin_notes",
}

_MIN_SECTION_CHARS = 2000


def _detect_sections(text: str, form_type: str | None = None) -> list[tuple[str, str]]:
    """Detect section boundaries and return (label, text) pairs.

    Uses text-only regex patterns with TOC disambiguation (minimum content
    threshold + largest region tiebreaker). Returns sections in document order.
    "other" (preamble) appears at most once, always first.
    """
    upper_form_type = (form_type or "").upper()
    if upper_form_type.startswith("20"):
        patterns = _20F_PATTERNS
    elif upper_form_type.startswith("8-K"):
        patterns = _8K_PATTERNS
    elif upper_form_type.startswith("10-Q"):
        patterns = _10Q_PATTERNS
    else:
        patterns = _10K_PATTERNS

    # Find all matches for each pattern, mapped to canonical labels
    candidates: dict[str, list[int]] = {}
    for key, pattern in patterns.items():
        label = _LABEL_MAP.get(key, key)
        positions = [m.start() for m in pattern.finditer(text)]
        if label not in candidates:
            candidates[label] = []
        candidates[label].extend(positions)

    if not candidates or all(len(v) == 0 for v in candidates.values()):
        return [("other", text)]

    # Collect all candidate positions across all labels for boundary computation
    all_positions: list[tuple[int, str]] = []
    for label, positions in candidates.items():
        for pos in positions:
            all_positions.append((pos, label))
    all_positions.sort(key=lambda x: x[0])

    # TOC disambiguation per label
    final_boundaries: list[tuple[int, str]] = []
    for label in candidates:
        label_positions = sorted(candidates[label])
        if not label_positions:
            continue

        # Compute content region size for each candidate
        # Next boundary = next match of ANY label (including same label)
        scored: list[tuple[int, int]] = []  # (position, content_size)
        for pos in label_positions:
            next_boundary = len(text)
            for other_pos, _other_label in all_positions:
                if other_pos > pos:
                    next_boundary = other_pos
                    break
            scored.append((pos, next_boundary - pos))

        # Filter by minimum threshold
        surviving = [(pos, size) for pos, size in scored if size >= _MIN_SECTION_CHARS]

        if surviving:
            best = max(surviving, key=lambda x: x[1])
        else:
            # No candidate passed threshold — use largest anyway (short section)
            best = max(scored, key=lambda x: x[1])

        final_boundaries.append((best[0], label))

    if not final_boundaries:
        return [("other", text)]

    final_boundaries.sort(key=lambda x: x[0])

    # Build section slices
    sections: list[tuple[str, str]] = []
    first_pos = final_boundaries[0][0]

    # Preamble → "other"
    if first_pos > 0:
        preamble = text[:first_pos].strip()
        if preamble:
            sections.append(("other", preamble))

    # Section slices — last section gets everything to end of text
    for i, (pos, label) in enumerate(final_boundaries):
        if i + 1 < len(final_boundaries):
            end = final_boundaries[i + 1][0]
        else:
            end = len(text)
        section_text = text[pos:end].strip()
        if section_text:
            sections.append((label, section_text))

    return sections if sections else [("other", text)]


# ---------------------------------------------------------------------------
# Block Splitting
# ---------------------------------------------------------------------------

_MIN_BLOCK_CHARS = 50


def _split_into_blocks(section_label: str, section_text: str) -> list[FilingBlock]:
    """Split section text into blocks on double-newline boundaries."""
    chunks = re.split(r"\n\n+", section_text)
    blocks: list[FilingBlock] = []
    ordinal = 0
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < _MIN_BLOCK_CHARS:
            continue
        blocks.append(
            FilingBlock(
                block_id=f"{section_label}_p{ordinal}",
                section=section_label,
                ordinal=ordinal,
                text=chunk,
                raw_section=section_label,
            )
        )
        ordinal += 1
    return blocks


def parse_filing_blocks(
    filing_html: str,
    form_type: str | None = None,
    *,
    source_form_type: str | None = None,
    source_filing_date: str | None = None,
    source_accession: str | None = None,
    source_role: str | None = None,
) -> list[FilingBlock]:
    """Parse raw filing HTML into section-labeled text blocks."""
    if not filing_html or not filing_html.strip():
        return []
    prepared = _prepare_filing_text(filing_html)
    if not prepared.strip():
        return []
    sections = _detect_sections(prepared, form_type)
    blocks: list[FilingBlock] = []
    block_prefix = f"{source_accession}:" if source_accession else ""
    for raw_label, section_text in sections:
        event_category = event_category_for_8k_section(raw_label)
        normalized_section = _LABEL_MAP.get(raw_label, event_category or raw_label)
        item_code = item_code_for_8k_section(raw_label)
        section_blocks = _split_into_blocks(raw_label, section_text)
        for block in section_blocks:
            if block_prefix:
                block.block_id = f"{block_prefix}{block.block_id}"
            block.section = normalized_section
            block.raw_section = raw_label
            block.source_form_type = source_form_type or form_type
            block.source_filing_date = source_filing_date
            block.source_accession = source_accession
            block.source_role = source_role
            block.item_code = item_code
            block.event_category = event_category
            source_quality_type = source_form_type or form_type
            if source_quality_type:
                block.source_quality = classify_source_quality(
                    source_type=source_quality_type,
                    published_at=source_filing_date,
                )
        blocks.extend(section_blocks)
    return blocks


# ---------------------------------------------------------------------------
# Layer 2: Candidate Retrieval
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the of in for by on to and or is are was were be with from at as this that it its".split()
)

_QUERY_EXPANSIONS: dict[str, list[str]] = {
    "concentration": [
        "concentrated",
        "accounted for",
        "largest",
        "significant portion",
        "top 10 customers",
        "single customer",
        "revenue concentration",
    ],
    "retention": ["churn", "renewal", "attrition", "net revenue retention"],
    "guidance": ["outlook", "expects", "anticipates", "projects", "forecast"],
    "maturity": ["due", "repayment", "matures", "payable"],
    "covenant": ["compliance", "covenants", "restricted", "default"],
    "impairment": ["writedown", "write-down", "write down", "goodwill impairment"],
    "restructuring": ["reorganization", "severance", "exit costs"],
    "acquisition": ["acquired", "business combination", "purchase price"],
    "segment": ["reportable segment", "operating segment", "by region", "geographic"],
    "backlog": ["order", "bookings", "pipeline", "contracted"],
    "capex": ["capital expenditure", "capital spending", "purchases of property"],
    "profitability": ["path to profitability", "break even", "breakeven", "operating income"],
    "buyback": ["repurchase", "share repurchase", "treasury stock"],
    "pivot": ["transition", "transformation", "new markets", "strategic shift"],
    "disruption": ["disruptive", "displacement", "obsolete", "emerging technology"],
    "margin": ["gross margin", "operating margin", "gross profit", "cost of revenue"],
    "inventory": ["inventory turnover", "days inventory", "inventory obsolescence"],
    "receivables": ["accounts receivable", "days sales outstanding", "allowance for doubtful"],
    "solvency": ["going concern", "ability to continue", "substantial doubt"],
}

_SOURCE_SECTION_PRIORS: dict[str, list[str]] = {
    "GROWTH_VS_EARNINGS_POWER": ["mda", "risk_factors"],
    "ASSET_VS_EARNINGS": ["fin_notes", "mda"],
    "UNANIMOUS_UNDERVALUATION": ["risk_factors", "business"],
    "MARKET_PREMIUM": ["mda"],
    "Q4_EARNINGS_BOMB": ["mda", "fin_notes"],
    "MARGIN_COLLAPSE": ["mda", "risk_factors"],
    "REVENUE_DECLINE_FROM_PEAK": ["mda", "risk_factors"],
    "DEBT_SPIKE": ["fin_notes", "mda"],
    "PERSISTENT_CASH_BURN": ["mda", "fin_notes"],
    "NEGATIVE_EQUITY": ["fin_notes", "mda"],
    "INTANGIBLE_ASSET_JUMP": ["fin_notes", "mda"],
    "WORKING_CAPITAL_CRISIS": ["fin_notes", "mda"],
    "COMPETITIVE_DISRUPTION": ["risk_factors", "business"],
    "SECULAR_DECLINE": ["risk_factors", "mda"],
    "PROCEED_SEVERE_DOWNSIDE": ["risk_factors", "fin_notes"],
    "NEGATIVE_OWNER_EARNINGS": ["mda", "fin_notes"],
    "REVENUE_DECLINE_RD_INCREASE": ["mda", "business"],
}
_DEFAULT_SECTION_PRIORS = ["mda", "risk_factors"]
_HIGH_SIGNAL_EVENT_ITEM_PRIORS: dict[str, float] = {
    "4.02": 1.35,
    "5.02": 1.30,
    "2.03": 1.25,
    "2.04": 1.25,
    "2.06": 1.25,
    "1.03": 1.25,
    "3.01": 1.20,
    "2.05": 1.15,
    "1.01": 1.10,
    "2.02": 1.10,
    "7.01": 1.05,
    "9.01": 1.00,
    "8.01": 0.95,
}
_CURRENT_EVENT_SOURCE_PRIORS: dict[str, float] = {
    "TRANSCRIPT": 1.15,
    "ir_press": 1.10,
    "external_news": 1.05,
    "company_news": 1.00,
}


def _event_item_priority(item_code: str | None) -> float:
    if not item_code:
        return 1.0
    return _HIGH_SIGNAL_EVENT_ITEM_PRIORS.get(item_code, 1.0)


def _current_event_source_priority(source_type: str | None) -> float:
    if not source_type:
        return 1.0
    return _CURRENT_EVENT_SOURCE_PRIORS.get(source_type, 1.0)


def _sort_timestamp(value: str | None) -> float:
    if not value:
        return float("-inf")
    try:
        return (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            .astimezone(timezone.utc)
            .timestamp()
        )
    except ValueError:
        return float("-inf")


def _block_recency_priority(block: FilingBlock) -> float:
    if block.source_role == "current_event":
        return _sort_timestamp(block.source_published_at)
    return _sort_timestamp(block.source_filing_date)


@dataclass
class CandidateBlock:
    """A block scored for relevance to an evidence query."""

    block: FilingBlock
    score: float
    match_terms: list[str]


def _tokenize_query(query: str) -> list[str]:
    """Split query into lowercase tokens, removing stopwords and punctuation."""
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _token_matches_block(token: str, block_text_lower: str) -> bool:
    """Check if token appears in block text on word boundary.

    Uses \\b{token}s?\\b to handle basic English plurals
    (customer/customers, rate/rates) without full stemming.
    """
    return bool(re.search(rf"\b{re.escape(token)}s?\b", block_text_lower))


_NEGATION_WORDS = re.compile(
    r"\b(?:no|not|none|never|neither|without|zero|lack|absent|n't)\b", re.IGNORECASE
)
_NEGATION_WINDOW = 40  # characters before a matched term to check for negation


def _has_negation_near_matches(
    block_lower: str,
    matched_tokens: list[str],
    expansions: dict[str, list[str]],
) -> bool:
    """Check if negation words appear near the majority of matched terms."""
    if not matched_tokens:
        return False

    negated_count = 0
    for token in matched_tokens:
        # Find position of token (or its expansion) in block
        m = re.search(rf"\b{re.escape(token)}s?\b", block_lower)
        if not m:
            # Try expansions
            for phrase in expansions.get(token, []):
                idx = block_lower.find(phrase)
                if idx >= 0:
                    m = type("M", (), {"start": lambda self, i=idx: i})()
                    break
        if m:
            window_start = max(0, m.start() - _NEGATION_WINDOW)
            window = block_lower[window_start : m.start()]
            if _NEGATION_WORDS.search(window):
                negated_count += 1

    # Demote if majority of matched terms are negated
    return negated_count > len(matched_tokens) / 2


def retrieve_candidates(
    query: str,
    blocks: list[FilingBlock],
    source: str | None = None,
    top_n: int = 5,
) -> list[CandidateBlock]:
    """Retrieve top candidate blocks for an evidence query."""
    tokens = _tokenize_query(query)
    if not tokens or not blocks:
        return []

    # Build expansion map: token → list of expansion phrases (lowered)
    expansions: dict[str, list[str]] = {}
    for token in tokens:
        exps = _QUERY_EXPANSIONS.get(token)
        if exps:
            expansions[token] = [e.lower() for e in exps]

    priors = _SOURCE_SECTION_PRIORS.get(source or "", _DEFAULT_SECTION_PRIORS)
    total_tokens = len(tokens)

    # Determine thresholds
    if total_tokens == 1:
        min_fraction = 0.5
        min_absolute = 1
    else:
        min_fraction = 0.3
        min_absolute = 2

    candidates: list[CandidateBlock] = []
    for block in blocks:
        block_lower = block.text.lower()
        matched: list[str] = []

        for token in tokens:
            # Direct word-boundary match
            if _token_matches_block(token, block_lower):
                matched.append(token)
                continue
            # Expansion phrase match — full phrase, case-insensitive substring
            for phrase in expansions.get(token, []):
                if phrase in block_lower:
                    matched.append(token)
                    break

        matched_unique = list(dict.fromkeys(matched))  # deduplicate, preserve order
        matched_count = len(matched_unique)
        if matched_count < min_absolute:
            continue
        fraction = matched_count / total_tokens
        if fraction < min_fraction:
            continue

        score = fraction
        # Negation penalty: if negation words appear near matched terms, demote
        if _has_negation_near_matches(block_lower, matched_unique, expansions):
            score *= 0.5
        is_prior = block.section in priors
        if is_prior:
            score *= 1.2
        score *= _event_item_priority(block.item_code)
        if block.source_role == "current_event":
            score *= _current_event_source_priority(block.source_type)

        candidates.append(CandidateBlock(block=block, score=score, match_terms=matched_unique))

    # Sort: score desc, section-prior first, higher-signal 8-K items first,
    # newer filings first, ordinal asc.
    candidates.sort(
        key=lambda c: (
            c.score,
            c.block.section in priors,
            _event_item_priority(c.block.item_code),
            _current_event_source_priority(c.block.source_type),
            _block_recency_priority(c.block),
            -c.block.ordinal,
        ),
        reverse=True,
    )

    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Layer 3: LLM Adjudication
# ---------------------------------------------------------------------------

_ADJUDICATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["CONFIRMS", "CONTRADICTS", "INCONCLUSIVE", "NOT_FOUND"],
        },
        "cited_blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "block_id": {"type": "string"},
                    "excerpt": {"type": "string"},
                },
                "required": ["block_id", "excerpt"],
                "additionalProperties": False,
            },
        },
        "structured_fact": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        },
        "reasoning_short": {"type": "string"},
    },
    "required": ["status", "cited_blocks", "structured_fact", "reasoning_short"],
    "additionalProperties": False,
}


@dataclass
class Citation:
    """A reference to a specific block in the filing."""

    block_id: str
    section: str
    ordinal: int
    excerpt: str
    item_code: str | None = None
    event_category: str | None = None
    source_form_type: str | None = None
    source_filing_date: str | None = None
    source_accession: str | None = None
    source_role: str | None = None
    source_type: str | None = None
    source_title: str | None = None
    source_url: str | None = None
    source_published_at: str | None = None
    source_quality: dict[str, Any] | None = None
    is_synthetic: bool = False
    span_start: int | None = None
    span_end: int | None = None
    unverified_excerpt: bool = False


@dataclass
class EvidenceItemResult:
    """Result of searching for one evidence item."""

    need_id: str
    needed: str
    importance: str
    status: str  # CONFIRMS / CONTRADICTS / INCONCLUSIVE / NOT_FOUND / UNCLASSIFIED
    classification_method: str  # LLM / FALLBACK / NONE
    citations: list[Citation]
    excerpt: str
    structured_fact: str | None
    reasoning_short: str
    candidates_considered: int
    top_candidate_score: float
    candidate_rankings: list[CandidateBlock]
    warnings: list[str] = field(default_factory=list)


def _build_citations(
    cited_blocks: list[dict],
    block_index: dict[str, FilingBlock],
    is_synthetic: bool = False,
) -> tuple[list[Citation], list[str]]:
    """Build verified Citation objects from an LLM response.

    Two integrity guards are enforced so the citation layer cannot launder
    hallucinations into the packet:

    1. Excerpt verification — if the cited excerpt is not a verbatim substring
       of the block text, the citation is kept but flagged
       (``unverified_excerpt=True`` and ``is_synthetic=True``) with no span
       offsets, and a warning is surfaced.
    2. Block-id quarantine — only block_ids actually presented to the LLM
       (present in ``block_index``) are accepted. A cited block_id that was
       never presented is treated as hallucinated, dropped entirely, and a
       warning is surfaced.

    Returns the accepted citations plus a list of human-readable warnings.
    """
    citations: list[Citation] = []
    unverified_count = 0
    dropped_unknown_count = 0
    for cb in cited_blocks:
        block_id = cb.get("block_id", "")
        excerpt = cb.get("excerpt", "")
        block = block_index.get(block_id)
        if block:
            span_start = None
            span_end = None
            idx = block.text.find(excerpt) if excerpt else -1
            verbatim = idx >= 0
            if verbatim:
                span_start = idx
                span_end = idx + len(excerpt)
            else:
                # Excerpt was not found verbatim — quarantine as unverified.
                unverified_count += 1
            citations.append(
                Citation(
                    block_id=block_id,
                    section=block.section,
                    ordinal=block.ordinal,
                    item_code=block.item_code,
                    event_category=block.event_category,
                    source_form_type=block.source_form_type,
                    source_filing_date=block.source_filing_date,
                    source_accession=block.source_accession,
                    source_role=block.source_role,
                    source_type=block.source_type,
                    source_title=block.source_title,
                    source_url=block.source_url,
                    source_published_at=block.source_published_at,
                    source_quality=block.source_quality,
                    excerpt=excerpt,
                    is_synthetic=is_synthetic or not verbatim,
                    span_start=span_start,
                    span_end=span_end,
                    unverified_excerpt=not verbatim,
                )
            )
        elif block_id:
            # block_id was never presented to the LLM — hallucinated, drop it.
            dropped_unknown_count += 1

    warnings: list[str] = []
    if unverified_count:
        warnings.append(f"{unverified_count} cited excerpt(s) not found verbatim in their blocks")
    if dropped_unknown_count:
        warnings.append(
            f"{dropped_unknown_count} citation(s) dropped for unknown/hallucinated block_id"
        )
    return citations, warnings


def _adjudicate_evidence_item(
    need: EvidenceNeed,
    hypothesis: Hypothesis,
    candidates: list[CandidateBlock],
    provider: Any,
    block_index: dict[str, FilingBlock] | None = None,
    financial_integrity_scope: BoundV1FinancialScope | None = None,
    financial_scenarios: FinancialScenarioSource | None = None,
) -> EvidenceItemResult:
    """Adjudicate one evidence item against candidate blocks."""
    top_score = candidates[0].score if candidates else 0.0
    rankings = list(candidates)

    if not candidates:
        return EvidenceItemResult(
            need_id=need.need_id,
            needed=need.description,
            importance=need.importance,
            status="NOT_FOUND",
            classification_method="NONE",
            citations=[],
            excerpt="",
            structured_fact=None,
            reasoning_short="No candidate blocks found.",
            candidates_considered=0,
            top_candidate_score=0.0,
            candidate_rankings=[],
        )

    if provider is None:
        return EvidenceItemResult(
            need_id=need.need_id,
            needed=need.description,
            importance=need.importance,
            status="UNCLASSIFIED",
            classification_method="NONE",
            citations=[],
            excerpt="",
            structured_fact=None,
            reasoning_short="LLM provider disabled.",
            candidates_considered=len(candidates),
            top_candidate_score=top_score,
            candidate_rankings=rankings,
        )

    # Build prompt
    block_texts = []
    for c in candidates:
        block_texts.append(f"[{c.block.block_id}]\n{c.block.text}")
    formatted = "\n\n---\n\n".join(block_texts)

    prompt = (
        f"You are reviewing passages from a public company filing to determine whether "
        f"they contain evidence for a specific information need.\n\n"
        f"Evidence needed: {need.description}\n"
        f'Context: This evidence is needed to evaluate the hypothesis: "{hypothesis.claim}"\n\n'
        f"The following passages were retrieved from the filing. Each is labeled with its block_id.\n\n"
        f"{formatted}\n\n"
        f"Your job:\n"
        f"1. Determine whether these passages contain evidence that CONFIRMS, CONTRADICTS, "
        f"or is INCONCLUSIVE regarding the evidence need. If the passages do not address "
        f"this topic at all, return NOT_FOUND.\n"
        f"2. For each passage that contains relevant evidence, cite its block_id and quote "
        f"the most relevant 1-2 sentences verbatim (not the full paragraph). Keep excerpts under 200 characters.\n"
        f"3. Extract a structured fact (percentage, dollar amount, ratio) if one exists.\n"
        f"4. Explain your reasoning in 1-2 sentences."
    )

    schema_name = "evidence_adjudication_v1"
    max_output_tokens = 2500
    failed_attempts: list[dict[str, Any]] = []
    successful_attempts: list[dict[str, Any]] = []
    try:

        def require_exact_scope(_attempt=None):
            _require_paid_financial_scope(
                financial_integrity_scope,
                scenarios=financial_scenarios,
                context=f"evidence_searcher:{need.need_id}",
            )

        with provider_usage_request(
            provider=provider,
            prompt=prompt,
            schema=_ADJUDICATE_SCHEMA,
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
                        schema=_ADJUDICATE_SCHEMA,
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
        payload = json.loads(result.json_text)
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
        logger.warning("evidence_searcher: LLM call failed for %s: %s", need.need_id, exc)
        return EvidenceItemResult(
            need_id=need.need_id,
            needed=need.description,
            importance=need.importance,
            status="INCONCLUSIVE",
            classification_method="NONE",
            citations=[],
            excerpt="",
            structured_fact=None,
            reasoning_short="LLM call failed.",
            candidates_considered=len(candidates),
            top_candidate_score=top_score,
            candidate_rankings=rankings,
        )

    status = payload.get("status", "INCONCLUSIVE")
    cited = payload.get("cited_blocks", [])
    bindex = block_index or {c.block.block_id: c.block for c in candidates}
    citations, citation_warnings = _build_citations(cited, bindex)
    primary_excerpt = cited[0]["excerpt"] if cited else ""
    if citation_warnings:
        logger.warning(
            "evidence_searcher: citation integrity warnings for %s: %s",
            need.need_id,
            "; ".join(citation_warnings),
        )

    result = EvidenceItemResult(
        need_id=need.need_id,
        needed=need.description,
        importance=need.importance,
        status=status,
        classification_method="LLM",
        citations=citations,
        excerpt=primary_excerpt,
        structured_fact=payload.get("structured_fact"),
        reasoning_short=payload.get("reasoning_short", ""),
        candidates_considered=len(candidates),
        top_candidate_score=top_score,
        candidate_rankings=rankings,
        warnings=citation_warnings,
    )
    try:
        require_exact_scope()
    except InvalidFinancialInputError as exc:
        attach_provider_usage_to_exception(exc, [*failed_attempts, *successful_attempts])
        raise
    return result


# ---------------------------------------------------------------------------
# Layer 4: Aggregation
# ---------------------------------------------------------------------------


@dataclass
class EvidenceResult:
    """Aggregated evidence result for one hypothesis."""

    hypothesis: Hypothesis
    evidence_item_results: list[EvidenceItemResult]
    hypothesis_status: str
    coverage_score: float
    classification_method: str


def _derive_hypothesis_status(items: list[EvidenceItemResult]) -> str:
    """Derive hypothesis status from item-level results, weighted by importance."""
    if not items:
        return "INCONCLUSIVE"

    if all(item.status == "UNCLASSIFIED" for item in items):
        return "UNCLASSIFIED"

    req_imp = [i for i in items if i.importance in ("REQUIRED", "IMPORTANT")]
    if not req_imp:
        return "INCONCLUSIVE"

    # Rule 1: REQUIRED contradiction → CONTRADICTED
    if any(i.status == "CONTRADICTS" and i.importance == "REQUIRED" for i in items):
        return "CONTRADICTED"

    ri_confirms = [i for i in req_imp if i.status == "CONFIRMS"]
    ri_contradicts = [i for i in req_imp if i.status == "CONTRADICTS"]

    # Rule 2: All REQUIRED + IMPORTANT confirm → CONFIRMED
    if all(i.status == "CONFIRMS" for i in req_imp):
        return "CONFIRMED"

    has_ri_confirm = len(ri_confirms) > 0
    has_ri_contradict = len(ri_contradicts) > 0

    # Rule 3: Some confirm, none contradict → PARTIALLY_CONFIRMED
    if has_ri_confirm and not has_ri_contradict:
        return "PARTIALLY_CONFIRMED"

    # Rule 4: Some confirm AND some IMPORTANT contradict → PARTIALLY_CONFIRMED (mixed)
    if has_ri_confirm and has_ri_contradict:
        return "PARTIALLY_CONFIRMED"

    # Rule 5: IMPORTANT contradict, no confirms → INCONCLUSIVE
    if has_ri_contradict and not has_ri_confirm:
        return "INCONCLUSIVE"

    # Rule 6: All else → INCONCLUSIVE
    return "INCONCLUSIVE"


def _compute_coverage(items: list[EvidenceItemResult]) -> float:
    """Fraction of items that are resolved (CONFIRMS or CONTRADICTS)."""
    if not items:
        return 0.0
    resolved = sum(1 for i in items if i.status in ("CONFIRMS", "CONTRADICTS"))
    return resolved / len(items)


def _derive_classification_method(items: list[EvidenceItemResult]) -> str:
    """Derive overall classification method from item-level methods."""
    if not items:
        return "NONE"
    methods = set(i.classification_method for i in items)
    if len(methods) == 1:
        return methods.pop()
    return "MIXED"


# ---------------------------------------------------------------------------
# Public API + Fallback
# ---------------------------------------------------------------------------

_MAX_FALLBACK_CHARS = 30_000


def _build_fallback_block(
    sections: list[tuple[str, str]],
    source: str | None,
    prepared_text: str,
) -> FilingBlock:
    """Build a synthetic block for fallback adjudication."""
    priors = _SOURCE_SECTION_PRIORS.get(source or "", _DEFAULT_SECTION_PRIORS)
    # Order: prior-preferred sections first, then remaining
    prior_sections = [(l, t) for l, t in sections if l in priors]
    other_sections = [(l, t) for l, t in sections if l not in priors]
    ordered = prior_sections + other_sections
    combined = "\n\n".join(t for _, t in ordered)
    if combined:
        text = combined[:_MAX_FALLBACK_CHARS]
        return FilingBlock("all_sections_fallback", "other", 0, text)
    # Parser failure: sample evenly across prepared text
    if not prepared_text:
        return FilingBlock("raw_text_fallback", "other", 0, "")
    chunk_size = len(prepared_text) // 5 or 1
    chunks = []
    for i in range(5):
        start = i * chunk_size
        chunks.append(prepared_text[start : start + 6000])
    text = "\n\n[...]\n\n".join(chunks)
    return FilingBlock("raw_text_fallback", "other", 0, text[:_MAX_FALLBACK_CHARS])


def get_llm_provider():
    """Import and return the LLM provider. Separate function for monkeypatching."""
    from app.llm.providers import get_llm_provider as _get

    return _get()


def parse_filing_context_blocks(filing_context: FilingContext) -> list[FilingBlock]:
    """Parse every filing in a context into one source-aware block list."""
    blocks: list[FilingBlock] = []
    for document in filing_context.ordered_documents:
        blocks.extend(
            parse_filing_blocks(
                document.html,
                document.form_type,
                source_form_type=document.form_type,
                source_filing_date=document.filing_date,
                source_accession=document.accession,
                source_role=document.role,
            )
        )
    return blocks


def build_current_event_blocks(
    current_event_context: CurrentEventContext | None,
) -> list[FilingBlock]:
    """Build synthetic searchable blocks from current-event documents."""
    if current_event_context is None or not current_event_context.documents:
        return []

    blocks: list[FilingBlock] = []
    for ordinal, document in enumerate(current_event_context.ordered_documents):
        snippet = ""
        if document.citations:
            snippet = (document.citations[0].snippet or "").strip()
        text_parts = [document.title.strip()]
        if document.summary.strip():
            text_parts.append(document.summary.strip())
        if snippet and snippet not in text_parts[-1]:
            text_parts.append(snippet)
        block_text = "\n\n".join(part for part in text_parts if part)[:4000]
        digest = sha256_text(
            f"{document.source_type}|{document.source_url}|{document.published_at or ''}|{document.title}"
        )[:16]
        blocks.append(
            FilingBlock(
                block_id=f"event_{digest}",
                section=document.source_type,
                ordinal=ordinal,
                text=block_text,
                raw_section=document.source_type,
                source_role=document.source_role,
                source_type=document.source_type,
                source_title=document.title,
                source_url=document.source_url,
                source_published_at=document.published_at,
                source_quality=document.source_quality,
            )
        )
    return blocks


def _prepared_sections_from_context(
    filing_context: FilingContext,
) -> tuple[str, list[tuple[str, str]], dict[str, FilingBlock]]:
    prepared_parts: list[str] = []
    sections: list[tuple[str, str]] = []
    blocks = parse_filing_context_blocks(filing_context)
    block_index = {block.block_id: block for block in blocks}
    for document in filing_context.ordered_documents:
        prepared = _prepare_filing_text(document.html) if document.html else ""
        if not prepared:
            continue
        prepared_parts.append(prepared)
        sections.extend(_detect_sections(prepared, document.form_type))
    return "\n\n".join(prepared_parts), sections, block_index


def search_evidence(
    hypotheses: list[Hypothesis],
    filing_html: str,
    form_type: str | None = None,
    max_hypotheses: int = 10,
    enable_fallback: bool = True,
    financial_integrity_scope: BoundV1FinancialScope | None = None,
    financial_scenarios: FinancialScenarioSource | None = None,
) -> list[EvidenceResult]:
    """Search filing HTML for evidence to test each hypothesis."""
    filing_context = build_inline_filing_context(
        filing_html,
        ticker="INLINE",
        form_type=form_type,
    )
    return search_evidence_in_filing_context(
        hypotheses,
        filing_context,
        max_hypotheses=max_hypotheses,
        enable_fallback=enable_fallback,
        financial_integrity_scope=financial_integrity_scope,
        financial_scenarios=financial_scenarios,
    )


def search_evidence_in_filing_context(
    hypotheses: list[Hypothesis],
    filing_context: FilingContext,
    current_event_context: CurrentEventContext | None = None,
    max_hypotheses: int = 10,
    enable_fallback: bool = True,
    financial_integrity_scope: BoundV1FinancialScope | None = None,
    financial_scenarios: FinancialScenarioSource | None = None,
) -> list[EvidenceResult]:
    """Search one or more filings for evidence to test each hypothesis."""
    hypotheses = hypotheses[:max_hypotheses]
    if not hypotheses:
        return []

    # Layer 1: Parse
    prepared, sections, block_index = _prepared_sections_from_context(filing_context)
    current_event_blocks = build_current_event_blocks(current_event_context)
    for block in current_event_blocks:
        block_index[block.block_id] = block
    blocks = list(block_index.values())

    # Get LLM provider
    try:
        provider = get_llm_provider()
        if provider.provider_name == "disabled":
            provider = None
    except InvalidFinancialInputError:
        raise
    except Exception:
        provider = None

    results: list[EvidenceResult] = []
    for hypothesis in hypotheses:
        item_results: list[EvidenceItemResult] = []

        for need in hypothesis.evidence_needed:
            # Layer 2: Retrieve
            candidates = retrieve_candidates(
                need.description,
                blocks,
                source=hypothesis.source,
            )

            if candidates:
                # Layer 3: Adjudicate
                item_result = _adjudicate_evidence_item(
                    need,
                    hypothesis,
                    candidates,
                    provider,
                    block_index,
                    financial_integrity_scope=financial_integrity_scope,
                    financial_scenarios=financial_scenarios,
                )
            elif enable_fallback and (sections or prepared):
                # Fallback: build synthetic block
                fallback_block = _build_fallback_block(sections, hypothesis.source, prepared)
                if fallback_block.text:
                    fb_candidate = CandidateBlock(block=fallback_block, score=0.0, match_terms=[])
                    fb_index = {fallback_block.block_id: fallback_block}
                    item_result = _adjudicate_evidence_item(
                        need,
                        hypothesis,
                        [fb_candidate],
                        provider,
                        fb_index,
                        financial_integrity_scope=financial_integrity_scope,
                        financial_scenarios=financial_scenarios,
                    )
                    # Override classification to FALLBACK if LLM was used
                    if item_result.classification_method == "LLM":
                        item_result = EvidenceItemResult(
                            need_id=item_result.need_id,
                            needed=item_result.needed,
                            importance=item_result.importance,
                            status=item_result.status,
                            classification_method="FALLBACK",
                            citations=[
                                Citation(
                                    block_id=c.block_id,
                                    section=c.section,
                                    ordinal=c.ordinal,
                                    item_code=c.item_code,
                                    event_category=c.event_category,
                                    source_form_type=c.source_form_type,
                                    source_filing_date=c.source_filing_date,
                                    source_accession=c.source_accession,
                                    source_role=c.source_role,
                                    source_type=c.source_type,
                                    source_title=c.source_title,
                                    source_url=c.source_url,
                                    source_published_at=c.source_published_at,
                                    source_quality=c.source_quality,
                                    excerpt=c.excerpt,
                                    is_synthetic=True,
                                    span_start=c.span_start,
                                    span_end=c.span_end,
                                    unverified_excerpt=c.unverified_excerpt,
                                )
                                for c in item_result.citations
                            ],
                            excerpt=item_result.excerpt,
                            structured_fact=item_result.structured_fact,
                            reasoning_short=item_result.reasoning_short,
                            candidates_considered=item_result.candidates_considered,
                            top_candidate_score=item_result.top_candidate_score,
                            candidate_rankings=item_result.candidate_rankings,
                            warnings=item_result.warnings,
                        )
                else:
                    item_result = _adjudicate_evidence_item(need, hypothesis, [], None)
            else:
                item_result = _adjudicate_evidence_item(need, hypothesis, [], None)

            item_results.append(item_result)

        results.append(
            EvidenceResult(
                hypothesis=hypothesis,
                evidence_item_results=item_results,
                hypothesis_status=_derive_hypothesis_status(item_results),
                coverage_score=_compute_coverage(item_results),
                classification_method=_derive_classification_method(item_results),
            )
        )

    return results
