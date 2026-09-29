from __future__ import annotations

import re
from typing import Any

from app.util.text import extract_snippet, normalize_whitespace


SIGNALS = {
    "going_concern": ["going concern"],
    "covenant_pressure": ["covenant", "waiver"],
    "refinancing_need": ["maturity", "refinancing"],
    "dilution_risk": ["at-the-market", "shelf registration", "convertible"],
    "restatement": ["restatement"],
    "auditor_change": ["change in independent registered public accounting firm"],
    "material_weakness": ["material weakness"],
    "customer_concentration": ["customer concentration", "single customer"],
}

# Going concern and material weakness are risk flags, not topics: a keyword
# hit is not a finding. Every large filer's 10-K carries the auditor's
# "assessing the risk that a material weakness exists", and a plain substring
# test reads "ongoing concern over climate change" as going concern. These two
# count only when the text asserts them (see asserted_risk_excerpt).
_MATERIAL_WEAKNESS_RE = re.compile(r"\bmaterial\s+weakness(?:es)?\b", re.IGNORECASE)
_MW_BOILERPLATE = (
    r"\brisk\s+that\s+a\s+material\s+weakness\s+exist",
    r"\ba\s+material\s+weakness\s+is\s+a\s+deficiency",
)
_MW_NEGATED = (
    r"\bno\s+material\s+weakness",
    r"\bnot\s+(?:identify|identified|found|detected|aware\s+of)\b[^.;]{0,60}\bmaterial\s+weakness",
    r"\bmaterial\s+weakness(?:es)?\b[^.;]{0,60}\b(?:does|do|did)\s+not\s+exist",
    r"\bmaterial\s+weakness(?:es)?\b[^.;]{0,80}\b(?:has|have|was|were)\s+(?:been\s+)?(?:fully\s+)?remediated",
)
_MW_ASSERTED = (
    r"\b(?:identified|disclosed|reported)\b[^.;]{0,80}\bmaterial\s+weakness",
    r"\bmaterial\s+weakness(?:es)?\b[^.;]{0,120}\b(?:was|were|has\s+been|have\s+been)\s+identified",
    r"\b(?:because\s+of|due\s+to|as\s+a\s+result\s+of)\s+(?:the|this|these|a)\s+material\s+weakness",
    r"\bmaterial\s+weakness(?:es)?\b[^.;]{0,80}\b(?:continues?\s+to\s+exist|(?:has|have)\s+not\s+(?:yet\s+)?been\s+remediated)",
)
_MW_HYPOTHETICAL = (
    r"\b(?:if|could|may|might|would|should|potential)\b[^.;]{0,160}\bmaterial\s+weakness",
    r"\bmaterial\s+weakness(?:es)?\b[^.;]{0,80}\b(?:could|may|might|would)\b",
)


def _sentence(text: str, start: int, end: int, *, max_chars: int = 640) -> str:
    left = max(text.rfind(mark, 0, start) for mark in (".", "?", "!", ";")) + 1
    right_candidates = [p for mark in (".", "?", "!", ";") if (p := text.find(mark, end)) >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    left = max(left, start - max_chars // 2)
    right = min(right, end + max_chars // 2)
    return text[left:right].strip()


def _any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _asserted_material_weakness(content: str) -> str | None:
    for match in _MATERIAL_WEAKNESS_RE.finditer(content):
        sentence = _sentence(content, match.start(), match.end())
        if _any(_MW_BOILERPLATE, sentence) or _any(_MW_NEGATED, sentence):
            continue
        if _any(_MW_ASSERTED, sentence) or not _any(_MW_HYPOTHETICAL, sentence):
            return sentence
    return None


def _asserted_going_concern(content: str) -> str | None:
    # The structural gate's attributed detector: only a current, affirmative
    # assertion about the registrant (or its consolidated group) counts;
    # negations, accounting policy, hypotheticals and third parties do not.
    from app.alpha.solvency_scanner import detect_going_concern_assertions

    for assertion in detect_going_concern_assertions(content, ticker=""):
        if assertion.blockable:
            return assertion.excerpt
    return None


def asserted_risk_excerpt(fact_type: str, text: str) -> str | None:
    """Return the asserting sentence for a going-concern or material-weakness
    flag, or None when the text only mentions the phrase."""

    content = normalize_whitespace(text)
    if fact_type == "going_concern":
        return _asserted_going_concern(content)
    if fact_type == "material_weakness":
        return _asserted_material_weakness(content)
    raise ValueError(f"no assertion rule for {fact_type!r}")


def extract_footnote_signals(text: str, source_url: str) -> list[dict[str, Any]]:
    content = normalize_whitespace(text)
    lowered = content.lower()
    out: list[dict[str, Any]] = []
    for fact_type, keywords in SIGNALS.items():
        hit = None
        for keyword in keywords:
            idx = lowered.find(keyword)
            if idx >= 0:
                hit = keyword
                break
        if hit is None:
            continue
        snippet = extract_snippet(content, hit, width=500)
        if fact_type in {"going_concern", "material_weakness"}:
            asserted = asserted_risk_excerpt(fact_type, content)
            if asserted is None:
                continue
            snippet = asserted[:500]
        out.append(
            {
                "fact_type": fact_type,
                "value_json": {"present": True, "keyword": hit},
                "source_url": source_url,
                "snippet": snippet,
                "section_label": "footnotes_or_risk",
            }
        )
    return out
