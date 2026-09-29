"""Grade-vs-narrative consistency checks for sector-run candidates.

The conviction grade is derived deterministically (audit/final-verdict
machinery) while the thesis narrative is LLM-generated; they are joined only
at watchlist write time. A repeated failure mode is a thesis that argues for
watchlist-only (or claims missing data) while the structured grade says
ACTIONABLE. These helpers detect that divergence so the writer can flag it
instead of persisting it silently. The post-hoc eval gate
(tests/evals/test_semantic_consistency.py) shares this logic.
"""

from __future__ import annotations

import json
import re
from typing import Any

ACTIONABLE_VERDICTS = {"ACTIONABLE", "SELECTED"}

FINAL_RECOMMENDATION_RE = re.compile(
    r"\b(?:(?:the\s+)?(?:correct|right|appropriate|final)\s+)?"
    r"(?:company-level\s+)?(?:verdict|recommendation|decision|positioning)\s*"
    r"(?:is|at|as|remains|should\s+be|:)\s*"
    r"(?P<verdict>watchlist[_ -]?only|watchlist\s+only|avoid)\b",
    re.IGNORECASE,
)

# A softer narrative pattern: the thesis argues the name "justifies watchlist
# positioning" / "caps conviction" without naming the enum. Kept narrow so an
# ACTIONABLE thesis that merely mentions the watchlist does not trip it.
WATCHLIST_ARGUMENT_RE = re.compile(
    r"\b(?:justif(?:y|ies|ying)|warrant(?:s|ing)?|support(?:s|ing)?|argues?\s+for)\s+"
    r"(?:a\s+|the\s+)?watchlist(?:[_ -]only)?\s+(?:positioning|treatment|stance|status)\b",
    re.IGNORECASE,
)

_DATA_GAP_WORDS_RE = re.compile(
    r"\b(?:missing|unavailable|not\s+available|absent|undisclosed|"
    r"not\s+computable|cannot\s+be\s+computed|lack(?:s|ing)?)\b",
    re.IGNORECASE,
)
_DATA_NOUN_RE = re.compile(
    r"\b(?:data|inputs?|cash[\s-]?flows?|working[\s-]?capital|disclosures?|"
    r"figures?|financials?|filings?|statements?)\b",
    re.IGNORECASE,
)
_MISSING_CODE_RE = re.compile(r"\b[A-Z][A-Z0-9_]*_(?:MISSING|UNAVAILABLE)\b")


def normalize_verdict(value: Any) -> str | None:
    if not value:
        return None
    normalized = str(value).upper().replace("-", "_").replace(" ", "_")
    if normalized in {"WATCHLIST", "WATCHLIST_ONLY"}:
        return "WATCHLIST_ONLY"
    if normalized in {"ACTIONABLE", "SELECTED", "AVOID", "NO_SELECTION", "NO_WINNER"}:
        return normalized
    return None


def recommended_verdicts(thesis: str) -> list[str]:
    """Explicit final-verdict recommendations stated inside the thesis text."""
    verdicts: list[str] = []
    for match in FINAL_RECOMMENDATION_RE.finditer(thesis):
        verdict = match.group("verdict").upper().replace("-", "_").replace(" ", "_")
        if verdict == "WATCHLIST":
            verdict = "WATCHLIST_ONLY"
        verdicts.append(verdict)
    if WATCHLIST_ARGUMENT_RE.search(thesis):
        verdicts.append("WATCHLIST_ONLY")
    return verdicts


def grade_narrative_mismatch(thesis: str | None, conviction_grade: str | None) -> str | None:
    """One-line mismatch description, or None when grade and narrative agree."""
    if not thesis or not conviction_grade:
        return None
    grade = normalize_verdict(conviction_grade)
    recommendations = recommended_verdicts(thesis)
    if not recommendations:
        return None
    if "WATCHLIST_ONLY" in recommendations and grade in ACTIONABLE_VERDICTS:
        return f"thesis argues WATCHLIST_ONLY but structured grade is {grade}"
    if "AVOID" in recommendations and grade != "AVOID":
        return f"thesis argues AVOID but structured grade is {grade}"
    return None


def missing_data_claims(thesis: str | None) -> list[str]:
    """Sentences in the thesis that assert some financial data is missing."""
    if not thesis:
        return []
    claims: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", thesis):
        if _DATA_GAP_WORDS_RE.search(sentence) and _DATA_NOUN_RE.search(sentence):
            claims.append(" ".join(sentence.split()))
    return claims


# Topic-level grounding: a claim about a SPECIFIC input (cash flow, working
# capital, filings) is only grounded by packet codes about that same input —
# an unrelated gap (e.g. a missing filing) must not license a false
# "missing cash-flow inputs" claim.
_CLAIM_TOPICS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (
        re.compile(r"\bcash[\s-]?flows?\b", re.IGNORECASE),
        ("CFO", "CASH_FLOW", "OPERATING_CASH"),
    ),
    (
        re.compile(r"\bworking[\s-]?capital\b", re.IGNORECASE),
        ("WORKING_CAPITAL", "CURRENT_ASSETS", "CURRENT_LIABILITIES"),
    ),
    (
        re.compile(r"\bfilings?\b", re.IGNORECASE),
        ("FILING", "NO_FILING"),
    ),
]


def _packet_payload_text(packet: Any) -> str:
    if packet is None:
        return ""
    try:
        return json.dumps(
            packet if isinstance(packet, dict) else getattr(packet, "__dict__", {}),
            default=str,
        )
    except (TypeError, ValueError):
        return ""


def packet_shows_data_gap(packet: Any) -> bool:
    """True when the deterministic packet itself records any data gap.

    Grounds are: a non-OK financial/model-fit/data-quality status, any
    blocker or confidence cap, or a ``*_MISSING`` / ``*_UNAVAILABLE`` code
    anywhere in the packet payload (per-metric not_computable_reasons roll up
    as such codes).
    """
    if packet is None:
        return True  # no packet at all is itself a data gap
    for status_field in ("financial_status", "model_fit_status", "data_quality_status"):
        status = getattr(packet, status_field, None)
        if status is not None and str(status).upper() not in {"OK", ""}:
            return True
    if getattr(packet, "blockers", None):
        return True
    if getattr(packet, "confidence_caps", None):
        return True
    return bool(_MISSING_CODE_RE.search(_packet_payload_text(packet)))


def ungrounded_missing_data_claims(thesis: str | None, packet: Any) -> list[str]:
    """Thesis missing-data claims with no deterministic data gap behind them.

    A claim naming a specific input (cash flow, working capital, filings) is
    grounded only by packet codes for that input; a claim with no specific
    topic is grounded by any recorded data gap.
    """
    claims = missing_data_claims(thesis)
    if not claims:
        return []
    if packet is None:
        return []
    payload = _packet_payload_text(packet).upper()
    any_gap = packet_shows_data_gap(packet)
    ungrounded: list[str] = []
    for claim in claims:
        topics = [codes for pattern, codes in _CLAIM_TOPICS if pattern.search(claim)]
        if not topics:
            if not any_gap:
                ungrounded.append(claim)
            continue
        for codes in topics:
            if not any(code in payload for code in codes):
                ungrounded.append(claim)
                break
    return ungrounded
