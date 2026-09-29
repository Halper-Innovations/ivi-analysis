"""LLM-powered filing investigator.

Takes anomaly questions and searches the most-recent cached annual filing text
for answers. Supports follow-up questions in a loop capped at _MAX_ITERATIONS.

Public API
----------
investigate_anomalies(ticker, anomalies, *, financial_integrity_scope) -> list[Investigation]
    Returns empty list when: no anomalies supplied, no filing cached, LLM disabled.
    A paid call requires a valid canonical BoundV1FinancialScope for ticker.
"""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from copy import deepcopy
from datetime import date
from typing import Any

from app.alpha.filing_risk_scan import _find_latest_annual_path, _strip_html
from app.alpha.schemas import Anomaly, Investigation
from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    InvalidFinancialInputError,
    require_financial_integrity_scope,
)
from app.autonomous.v1_financial_context import (
    BoundV1FinancialScope,
    bind_v1_financial_scope,
    financial_input_scenario,
)
from app.llm.providers import get_llm_provider
from app.llm.providers.retry_guard import llm_physical_attempt_guard

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 5
_MAX_FILING_CHARS = 50_000
_MAX_SECTION_CHARS = 30_000
_CONTEXT_WINDOW = 750  # chars around each keyword hit — enough to capture financial table rows
_MAX_OUTPUT_TOKENS = 2_500

# ---------------------------------------------------------------------------
# LLM schema
# ---------------------------------------------------------------------------

_INVESTIGATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Direct answer to the question based on filing text.",
        },
        "evidence_excerpt": {
            "type": "string",
            "description": "Verbatim or near-verbatim excerpt from the filing that supports the answer.",
        },
        "follow_up": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "A follow-up question if the answer raises a new material concern, or null.",
        },
    },
    "required": ["answer", "evidence_excerpt", "follow_up"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_full_text(ticker: str) -> tuple[str | None, str | None]:
    """Load and strip the full filing HTML. Returns (plain_text, form_type) or (None, None)."""
    local_path, form_type = _find_latest_annual_path(ticker.upper())
    if not local_path:
        return None, None
    try:
        with open(local_path, encoding="utf-8", errors="replace") as fh:
            html = fh.read()
    except OSError as exc:
        logger.warning("filing_investigator: cannot read %s: %s", local_path, exc)
        return None, None
    return _strip_html(html), form_type


def _extract_section(
    text: str, start_pattern: str, end_pattern: str, min_length: int = 2000
) -> str:
    """Extract a filing section by matching start/end patterns, picking the largest gap."""
    starts = list(re.finditer(start_pattern, text, re.IGNORECASE))
    ends = list(re.finditer(end_pattern, text, re.IGNORECASE))
    best_text = ""
    best_size = 0
    for s in starts:
        for e in ends:
            gap = e.start() - s.start()
            if min_length < gap > best_size:
                best_text = text[s.start() : e.start()]
                best_size = gap
    return best_text[:_MAX_SECTION_CHARS]


def _extract_targeted_passages(
    text: str, keywords: list[str], max_chars: int = _MAX_FILING_CHARS
) -> str:
    """Extract passages around keyword hits within a section.

    Instead of sending an entire 300K section, find the ~500-char windows
    around each keyword match and concatenate them. This gives the LLM
    the most relevant context.
    """
    lower = text.lower()
    passages: list[tuple[int, str]] = []  # (position, passage)
    seen_ranges: list[tuple[int, int]] = []

    for kw in keywords:
        kw_lower = kw.lower()
        start = 0
        while True:
            idx = lower.find(kw_lower, start)
            if idx < 0:
                break
            window_start = max(0, idx - _CONTEXT_WINDOW)
            window_end = min(len(text), idx + len(kw) + _CONTEXT_WINDOW)
            # Skip if overlaps with existing passage
            overlaps = any(
                ws <= window_start <= we or ws <= window_end <= we for ws, we in seen_ranges
            )
            if not overlaps:
                passages.append((idx, text[window_start:window_end]))
                seen_ranges.append((window_start, window_end))
            start = idx + len(kw)

    # Sort by position in document and concatenate
    passages.sort(key=lambda x: x[0])
    result = "\n\n[...]\n\n".join(p for _, p in passages)
    return result[:max_chars] if result else text[:max_chars]


# Anomaly-type to keyword mapping for targeted extraction
_ANOMALY_KEYWORDS: dict[str, list[str]] = {
    "Q4_EARNINGS_BOMB": [
        "impairment",
        "writedown",
        "write-down",
        "valuation allowance",
        "restructuring",
        "one-time",
        "non-cash",
        "non-recurring",
        "provision for income tax",
        "income tax provision",
        "deferred tax",
        "income tax benefit",
        "net loss",
        "net income (loss)",
        "total operating expenses",
        "loss from operations",
        "operating loss",
        "gross profit",
        "comprehensive loss",
    ],
    "NEGATIVE_EQUITY": [
        "accumulated deficit",
        "stockholders' equity",
        "shareholders' equity",
        "total stockholders",
        "total equity",
        "retained earnings",
        "share repurchase",
        "dividends",
        "net loss",
        "net income (loss)",
        "comprehensive loss",
        "treasury stock",
    ],
    "MARGIN_COLLAPSE": [
        "operating margin",
        "cost of revenue",
        "cost of revenues",
        "gross margin",
        "gross profit",
        "operating expenses:",
        "total operating expenses",
        "research and development",
        "selling and marketing",
        "general and administrative",
        "advertising and marketing",
        "cost increase",
    ],
    "INTANGIBLE_ASSET_JUMP": [
        "acquisition",
        "acquired",
        "business combination",
        "capitalized",
        "intangible assets",
        "purchase price",
        "internally developed software",
        "software development cost",
        "film cost",
        "content asset",
        "license agreement",
    ],
    "REVENUE_DECLINE_FROM_PEAK": [
        "revenue decline",
        "decrease in revenue",
        "total revenue",
        "customer loss",
        "market conditions",
        "competitive",
        "demand",
        "net revenue",
        "product revenue",
        "service revenue",
    ],
    "PERSISTENT_CASH_BURN": [
        "cash flow",
        "operating activities",
        "net cash",
        "cash used in operating",
        "liquidity",
        "financing activities",
        "credit facility",
        "line of credit",
        "additional capital",
        "ability to fund",
        "raise additional",
        "additional financing",
        "cash and cash equivalents",
    ],
    "WORKING_CAPITAL_CRISIS": [
        "working capital",
        "current liabilities",
        "current assets",
        "liquidity",
        "credit facility",
        "revolving",
        "short-term borrowing",
        "convertible note",
        "repayment",
        "maturity",
        "additional financing",
    ],
    "DEBT_SPIKE": [
        "convertible note",
        "credit facility",
        "term loan",
        "borrowing",
        "debt issuance",
        "refinanc",
        "maturity",
        "repayment",
        "principal amount",
        "interest rate",
        "promissory note",
    ],
}


def _build_investigation_context(
    full_text: str,
    form_type: str | None,
    anomaly_type: str,
) -> str:
    """Build targeted filing context for investigating a specific anomaly.

    Strategy:
    1. Extract MD&A section (where management explains results)
    2. Extract financial statement notes (where accounting details live)
    3. Within those sections, find keyword passages relevant to the anomaly
    4. If sections not found, fall back to keyword search on full text
    """
    form = form_type or "10-K"

    # Extract key sections — 10-K and 20-F have different structures
    if form.startswith("20-F"):
        # 20-F: MD&A is "Item 5. Operating and Financial Review" → "Item 6"
        mda = _extract_section(full_text, r"Item\s+5[\s\.\:].*Operating", r"Item\s+6[\s\.\:]")
        if not mda:
            mda = _extract_section(
                full_text, r"Operating\s+and\s+Financial\s+Review", r"Item\s+6[\s\.\:]"
            )
        # 20-F: Financial notes under Item 18, end at Item 19 or SIGNATURES
        fin_notes = _extract_section(
            full_text,
            r"Notes\s+to\s+(?:Consolidated\s+)?Financial\s+Statements",
            r"Item\s+19[\s\.\:]|SIGNATURES|Pursuant\s+to\s+the\s+requirements",
            min_length=5000,
        )
    else:
        # 10-K: MD&A is Item 7 → Item 7A or Item 8
        mda = _extract_section(
            full_text, r"Item\s+7\.?\s+Management", r"Item\s+7A[\s\.\:]|Item\s+8[\s\.\:]"
        )
        # 10-K: Financial notes end at Item 9
        fin_notes = _extract_section(
            full_text,
            r"Notes\s+to\s+(?:Consolidated\s+)?Financial\s+Statements",
            r"Item\s+9[\s\.\:]|SIGNATURES",
            min_length=5000,
        )

    # Get keywords for this anomaly type
    keywords = _ANOMALY_KEYWORDS.get(
        anomaly_type, ["impairment", "writedown", "restructuring", "non-cash"]
    )

    # Build context from sections with targeted keyword extraction
    parts: list[str] = []

    if mda:
        mda_passages = _extract_targeted_passages(mda, keywords, max_chars=20_000)
        if mda_passages:
            parts.append(f"=== MD&A RELEVANT PASSAGES ===\n{mda_passages}")

    if fin_notes:
        notes_passages = _extract_targeted_passages(fin_notes, keywords, max_chars=20_000)
        if notes_passages:
            parts.append(f"=== FINANCIAL NOTES RELEVANT PASSAGES ===\n{notes_passages}")

    # If no sections found, fall back to keyword search on full text
    if not parts:
        fallback = _extract_targeted_passages(full_text, keywords, max_chars=_MAX_FILING_CHARS)
        if fallback:
            parts.append(f"=== FILING RELEVANT PASSAGES ===\n{fallback}")

    # If still nothing, send first 50K of full text as last resort
    if not parts:
        return full_text[:_MAX_FILING_CHARS]

    return "\n\n".join(parts)


def _investigate_one(
    ticker: str,
    question: str,
    anomaly_type: str,
    filing_context: str,
    provider: Any,
    *,
    financial_integrity_scope: BoundV1FinancialScope,
) -> Investigation | None:
    """Single LLM call: ask the question against targeted filing passages."""
    packet = _require_bound_ticker_packet(
        financial_integrity_scope,
        ticker=ticker,
    )
    prompt = (
        f"You are an investigative financial analyst reviewing a public company filing "
        f"(ticker: {ticker}).\n\n"
        f"Question: {question}\n\n"
        f"The following are targeted passages extracted from the company's most recent "
        f"annual filing (10-K or 20-F), focused on sections most likely to contain the answer "
        f"(MD&A and financial statement notes). Search these passages and provide:\n"
        f"1. A direct answer citing specific dollar amounts, dates, and accounting items.\n"
        f"2. A verbatim or near-verbatim evidence excerpt from the text.\n"
        f"3. A follow-up question ONLY if the answer reveals a new material concern "
        f"(null if the question is fully resolved).\n\n"
        f"--- FILING PASSAGES ---\n{filing_context}\n--- END ---\n\n"
        f"Return a JSON object matching the required schema."
    )
    call_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "schema": deepcopy(_INVESTIGATE_SCHEMA),
        "schema_name": "filing_investigator_v1",
        "max_output_tokens": _MAX_OUTPUT_TOKENS,
    }

    def current_scenario() -> dict[str, Any]:
        return financial_input_scenario(
            packet,
            financial_inputs={
                "ticker": ticker,
                "anomaly_type": anomaly_type,
                "question": question,
                "provider_name": str(getattr(provider, "provider_name", "unknown")),
                "prompt": call_kwargs["prompt"],
                "schema": call_kwargs["schema"],
                "schema_name": call_kwargs["schema_name"],
                "max_output_tokens": call_kwargs["max_output_tokens"],
            },
        )

    exact_scope = bind_v1_financial_scope(
        context=f"filing_investigator:{ticker}:{anomaly_type}",
        run_as_of_date=financial_integrity_scope.run_as_of_date,
        packets=(packet,),
        scenarios=(current_scenario(),),
    )

    def require_exact_scope(_attempt: dict[str, Any] | None = None) -> None:
        exact_scope.require(scenarios=(current_scenario(),))

    try:
        with llm_physical_attempt_guard(require_exact_scope):
            require_exact_scope()
            result = provider.synthesize_json(**call_kwargs)
        # Do not turn a response into Investigation prose if the exact
        # prompt/schema/packet authorization changed during the call.
        require_exact_scope()
        payload = json.loads(result.json_text)
        return Investigation(
            anomaly_type=anomaly_type,
            question=question,
            answer=payload.get("answer", ""),
            evidence_excerpt=payload.get("evidence_excerpt", ""),
            follow_up=payload.get("follow_up") or None,
        )
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        try:
            require_exact_scope()
        except InvalidFinancialInputError as integrity_exc:
            raise integrity_exc from exc
        logger.warning(
            "filing_investigator: LLM call failed for %s / %s: %s",
            ticker,
            anomaly_type,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _raise_missing_bound_scope(
    *,
    ticker: str,
    run_as_of_date: str | None = None,
) -> None:
    """Raise the canonical terminal error for a missing ticker packet."""

    require_financial_integrity_scope(
        FinancialIntegrityScope(
            context=f"filing_investigator:{ticker}:missing_bound_scope",
            run_as_of_date=run_as_of_date or date.today().isoformat(),
            packets=(),
        )
    )
    raise AssertionError("empty financial-integrity scope unexpectedly passed")


def _require_bound_ticker_packet(
    scope: BoundV1FinancialScope | None,
    *,
    ticker: str,
) -> Any:
    """Return the one authorized packet for ``ticker`` or fail closed."""

    if scope is None:
        _raise_missing_bound_scope(ticker=ticker)

    scope.require()
    normalized_ticker = ticker.strip().upper()
    matches = [
        packet
        for packet in scope.packets
        if isinstance(packet, dict)
        and str(packet.get("ticker") or "").strip().upper() == normalized_ticker
    ]
    if len(matches) != 1:
        _raise_missing_bound_scope(
            ticker=normalized_ticker,
            run_as_of_date=scope.run_as_of_date,
        )
    return matches[0]


def investigate_anomalies(
    ticker: str,
    anomalies: list[Anomaly],
    *,
    financial_integrity_scope: BoundV1FinancialScope | None = None,
) -> list[Investigation]:
    """Investigate each anomaly against the latest cached annual filing.

    Uses a queue allowing follow-up questions from LLM responses, capped at
    _MAX_ITERATIONS total LLM calls.

    Returns empty list when:
    - anomalies is empty
    - no cached filing found for ticker
    - LLM provider is disabled

    Raises InvalidFinancialInputError before any provider attempt when a paid
    investigation lacks a valid bound ticker packet or exact call scenario.
    """
    if not anomalies:
        return []

    provider = get_llm_provider()
    if provider.provider_name == "disabled":
        return []
    enabled = getattr(provider, "enabled", None)
    if callable(enabled):
        try:
            if not enabled():
                return []
        except Exception:
            # Provider-state uncertainty cannot bypass financial authorization.
            pass

    full_text, form_type = _load_full_text(ticker)
    if not full_text:
        return []

    # Validate before creating any question queue or substantive output. The
    # exact per-call prompt/schema scenario is bound again in _investigate_one.
    _require_bound_ticker_packet(
        financial_integrity_scope,
        ticker=ticker,
    )

    # Queue entries: (question, anomaly_type)
    queue: deque[tuple[str, str]] = deque((a.question, a.anomaly_type) for a in anomalies)

    investigations: list[Investigation] = []
    iterations = 0

    while queue and iterations < _MAX_ITERATIONS:
        question, anomaly_type = queue.popleft()
        iterations += 1

        # Build targeted context for this specific anomaly type
        context = _build_investigation_context(full_text, form_type, anomaly_type)

        inv = _investigate_one(
            ticker,
            question,
            anomaly_type,
            context,
            provider,
            financial_integrity_scope=financial_integrity_scope,
        )
        if inv is None:
            continue

        investigations.append(inv)

        # If the LLM surfaced a follow-up, enqueue it (same anomaly_type for traceability)
        if inv.follow_up and iterations < _MAX_ITERATIONS:
            queue.append((inv.follow_up, anomaly_type))

    return investigations
