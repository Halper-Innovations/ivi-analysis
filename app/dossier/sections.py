from __future__ import annotations

import re
from dataclasses import dataclass

from app.util.text import normalize_whitespace


@dataclass(frozen=True)
class SectionSpan:
    section_label: str
    start_offset: int
    end_offset: int
    text: str


@dataclass(frozen=True)
class SectionPattern:
    """A pattern for matching a section header in 10-K HTML.

    `prefers_first_match` distinguishes two classes:
      - First-match patterns: TOC-immune markers like anchor-id attributes
        (`id="item_1_business"`) and uppercase prose headers
        (`NOTES TO CONSOLIDATED FINANCIAL STATEMENTS`). These appear once
        at the section anchor and the first hit is the right one.
      - Last-match patterns (default): mixed-case prose like
        `Item 1. Business` or `Notes to Consolidated Financial Statements`
        which appear in TOC, body, AND cross-references. The body comes
        after the TOC so the last hit is usually right.

    Combining both pattern classes per label and picking the EARLIEST
    resulting position lets the more-specific marker win when present.
    """

    pattern: re.Pattern[str]
    prefers_first_match: bool = False


# Whitespace-or-HTML-entity gap. Many 10-K filings use `&#160;` or `&nbsp;`
# (non-breaking space) between "Item" and the number / title. The previous
# patterns used `\s+` which doesn't match these entities, so the body section
# header was missed and the parser fell back to the TOC entry instead.
_WS = r"(?:\s|&#160;|&nbsp;)"
_ITEM_TITLE_SEP = rf"(?:\.?{_WS}+|{_WS}*[-–—:]{_WS}*)"


SECTION_PATTERNS: dict[str, list[SectionPattern]] = {
    "business": [
        SectionPattern(re.compile(r'id\s*=\s*["\']item_1_business["\']', re.IGNORECASE), prefers_first_match=True),
        SectionPattern(re.compile(rf"\bitem{_WS}+1{_ITEM_TITLE_SEP}business\b", re.IGNORECASE)),
    ],
    "risk_factors": [
        SectionPattern(re.compile(r'id\s*=\s*["\']item_1a_risk_factors["\']', re.IGNORECASE), prefers_first_match=True),
        SectionPattern(re.compile(rf"\bitem{_WS}+1a{_ITEM_TITLE_SEP}risk{_WS}+factors\b", re.IGNORECASE)),
    ],
    "md_and_a": [
        SectionPattern(
            re.compile(r'id\s*=\s*["\']item_7_managements_discussion_analysis[^"\']*["\']', re.IGNORECASE),
            prefers_first_match=True,
        ),
        # The body header can be "Item 7. Management's Discussion and Analysis"
        # with the apostrophe written as plain `'`, smart quote `’`, or HTML
        # entity `&#8217;`/`&rsquo;`. Match "Item 7" + within ~30 chars of
        # arbitrary content + "discussion" + "and" + "analysis". DOTALL lets
        # `.` match the entity colons / semicolons across the apostrophe.
        SectionPattern(re.compile(
            rf"\bitem{_WS}+7{_ITEM_TITLE_SEP}management.{{0,30}}?discussion{_WS}+and{_WS}+analysis",
            re.IGNORECASE | re.DOTALL,
        )),
    ],
    "financial_statements": [
        SectionPattern(
            re.compile(r'id\s*=\s*["\']item_8_financial_statements[^"\']*["\']', re.IGNORECASE),
            prefers_first_match=True,
        ),
        SectionPattern(re.compile(rf"\bitem{_WS}+8{_ITEM_TITLE_SEP}financial{_WS}+statements", re.IGNORECASE)),
    ],
    # Notes has no standard anchor-id, and "Notes to Consolidated Financial
    # Statements" appears LITERALLY everywhere in 10-Ks: in TOC links, in the
    # actual section header, in cross-references ("see Notes to..."), in inline
    # XBRL block-tag labels, and in page-level "—(Continued)" footers.
    #
    # Strategy: prefer the case-sensitive UPPERCASE pattern as a TOC-immune
    # first-match marker — section headers are typically uppercase and rare
    # while cross-references use mixed case. Fall back to mixed-case last-match
    # for filings that don't uppercase the header.
    "notes": [
        SectionPattern(
            re.compile(
                rf"NOTES?{_WS}+TO{_WS}+(?:THE{_WS}+)?CONSOLIDATED{_WS}+FINANCIAL{_WS}+STATEMENTS",
                # NO re.IGNORECASE — case-sensitive uppercase only
            ),
            prefers_first_match=True,
        ),
        SectionPattern(re.compile(
            rf"\bnotes?{_WS}+to{_WS}+(?:the{_WS}+)?consolidated{_WS}+financial{_WS}+statements\b",
            re.IGNORECASE,
        )),
    ],
    "segment_info": [
        SectionPattern(re.compile(rf"\bsegment{_WS}+information\b", re.IGNORECASE)),
        SectionPattern(re.compile(rf"\breportable{_WS}+segments?\b", re.IGNORECASE)),
    ],
}

SECTION_PATTERNS_10Q: dict[str, list[SectionPattern]] = {
    "risk_factors": [
        SectionPattern(re.compile(r'id\s*=\s*["\']item_1a_risk_factors["\']', re.IGNORECASE), prefers_first_match=True),
        SectionPattern(re.compile(rf"\bitem{_WS}+1a{_ITEM_TITLE_SEP}risk{_WS}+factors\b", re.IGNORECASE)),
    ],
    "md_and_a": [
        SectionPattern(
            re.compile(r'id\s*=\s*["\']item_2_management[^"\']*discussion[^"\']*analysis[^"\']*["\']', re.IGNORECASE),
            prefers_first_match=True,
        ),
        SectionPattern(
            re.compile(
                rf"\bitem{_WS}+2{_ITEM_TITLE_SEP}management.{{0,50}}?discussion.{{0,50}}?analysis",
                re.IGNORECASE | re.DOTALL,
            )
        ),
    ],
    "financial_statements": [
        SectionPattern(
            re.compile(r'id\s*=\s*["\']item_1_financial_statements["\']', re.IGNORECASE),
            prefers_first_match=True,
        ),
        SectionPattern(re.compile(rf"\bitem{_WS}+1{_ITEM_TITLE_SEP}financial{_WS}+statements\b", re.IGNORECASE)),
    ],
    "notes": [
        SectionPattern(
            re.compile(
                rf"NOTES?{_WS}+TO{_WS}+(?:THE{_WS}+)?(?:CONDENSED{_WS}+)?(?:CONSOLIDATED{_WS}+)?FINANCIAL{_WS}+STATEMENTS"
            ),
            prefers_first_match=True,
        ),
        SectionPattern(
            re.compile(
                rf"\bnotes?{_WS}+to{_WS}+(?:the{_WS}+)?(?:condensed{_WS}+)?(?:consolidated{_WS}+)?financial{_WS}+statements\b",
                re.IGNORECASE,
            )
        ),
    ],
}

SECTION_PATTERNS_8K: dict[str, list[SectionPattern]] = {
    "item_101": [
        SectionPattern(re.compile(rf"\bitem{_WS}+1\.01\b", re.IGNORECASE)),
    ],
    "item_102": [
        SectionPattern(re.compile(rf"\bitem{_WS}+1\.02\b", re.IGNORECASE)),
    ],
    "item_103": [
        SectionPattern(re.compile(rf"\bitem{_WS}+1\.03\b", re.IGNORECASE)),
    ],
    "item_202": [
        SectionPattern(re.compile(rf"\bitem{_WS}+2\.02\b", re.IGNORECASE)),
    ],
    "item_203": [
        SectionPattern(re.compile(rf"\bitem{_WS}+2\.03\b", re.IGNORECASE)),
    ],
    "item_204": [
        SectionPattern(re.compile(rf"\bitem{_WS}+2\.04\b", re.IGNORECASE)),
    ],
    "item_205": [
        SectionPattern(re.compile(rf"\bitem{_WS}+2\.05\b", re.IGNORECASE)),
    ],
    "item_206": [
        SectionPattern(re.compile(rf"\bitem{_WS}+2\.06\b", re.IGNORECASE)),
    ],
    "item_301": [
        SectionPattern(re.compile(rf"\bitem{_WS}+3\.01\b", re.IGNORECASE)),
    ],
    "item_402": [
        SectionPattern(re.compile(rf"\bitem{_WS}+4\.02\b", re.IGNORECASE)),
    ],
    "item_502": [
        SectionPattern(re.compile(rf"\bitem{_WS}+5\.02\b", re.IGNORECASE)),
    ],
    "item_701": [
        SectionPattern(re.compile(rf"\bitem{_WS}+7\.01\b", re.IGNORECASE)),
    ],
    "item_801": [
        SectionPattern(re.compile(rf"\bitem{_WS}+8\.01\b", re.IGNORECASE)),
    ],
    "item_901": [
        SectionPattern(re.compile(rf"\bitem{_WS}+9\.01\b", re.IGNORECASE)),
    ],
}

SECTION_METADATA_8K: dict[str, dict[str, str]] = {
    "item_101": {"item_code": "1.01", "event_category": "strategic_transaction"},
    "item_102": {"item_code": "1.02", "event_category": "agreement_termination"},
    "item_103": {"item_code": "1.03", "event_category": "distress_restructuring"},
    "item_202": {"item_code": "2.02", "event_category": "results_guidance"},
    "item_203": {"item_code": "2.03", "event_category": "financing_liquidity"},
    "item_204": {"item_code": "2.04", "event_category": "financing_liquidity"},
    "item_205": {"item_code": "2.05", "event_category": "distress_restructuring"},
    "item_206": {"item_code": "2.06", "event_category": "distress_restructuring"},
    "item_301": {"item_code": "3.01", "event_category": "legal_regulatory"},
    "item_402": {"item_code": "4.02", "event_category": "restatement_controls"},
    "item_502": {"item_code": "5.02", "event_category": "leadership_governance"},
    "item_701": {"item_code": "7.01", "event_category": "results_guidance"},
    "item_801": {"item_code": "8.01", "event_category": "other_material"},
    "item_901": {"item_code": "9.01", "event_category": "results_guidance"},
}


# Risk-disclosure patterns for the cheapness-explanation pass
# (app/events/cheapness.py). Deliberately NOT registered in the shared
# SECTION_PATTERNS dicts: adding markers there would move existing span
# boundaries (e.g. `notes` would truncate at the commitments header) and
# change dossier/analyst extraction behavior. Callers compose them with the
# base dicts via segment_sections().
LEGAL_PROCEEDINGS_PATTERNS: list[SectionPattern] = [
    SectionPattern(
        re.compile(r'id\s*=\s*["\']item_3_legal_proceedings["\']', re.IGNORECASE),
        prefers_first_match=True,
    ),
    SectionPattern(re.compile(rf"\bitem{_WS}+3{_ITEM_TITLE_SEP}legal{_WS}+proceedings\b", re.IGNORECASE)),
    SectionPattern(re.compile(rf"LEGAL{_WS}+PROCEEDINGS"), prefers_first_match=True),
    SectionPattern(re.compile(rf"\blegal{_WS}+proceedings\b", re.IGNORECASE)),
]

# 10-Q legal proceedings is Part II, Item 1.
LEGAL_PROCEEDINGS_PATTERNS_10Q: list[SectionPattern] = [
    SectionPattern(
        re.compile(r'id\s*=\s*["\']item_1_legal_proceedings["\']', re.IGNORECASE),
        prefers_first_match=True,
    ),
    SectionPattern(re.compile(rf"\bitem{_WS}+1{_ITEM_TITLE_SEP}legal{_WS}+proceedings\b", re.IGNORECASE)),
    SectionPattern(re.compile(rf"LEGAL{_WS}+PROCEEDINGS"), prefers_first_match=True),
    SectionPattern(re.compile(rf"\blegal{_WS}+proceedings\b", re.IGNORECASE)),
]

COMMITMENTS_CONTINGENCIES_PATTERNS: list[SectionPattern] = [
    SectionPattern(
        re.compile(rf"COMMITMENTS{_WS}+AND{_WS}+CONTINGENCIES"),
        prefers_first_match=True,
    ),
    SectionPattern(
        re.compile(rf"\bcommitments{_WS}+and{_WS}+contingencies\b", re.IGNORECASE)
    ),
]


def segment_sections(text: str, patterns: dict[str, list[SectionPattern]]) -> list[SectionSpan]:
    """Public composer: segment with a caller-supplied pattern dict."""
    return _segment_sections(text, patterns)


# When the same pattern matches in both the table of contents and the body,
# the body match is what we want — the TOC text is just a hyperlink with no
# real content. The body match is always at a higher offset than the TOC
# match. So when a text pattern has multiple hits, we use the last one.
# First-match patterns (anchor IDs, uppercase headers) only ever appear at
# the section anchor itself, so for those we use the first match.
def _pick_best_match(matches: list[re.Match], *, prefers_first_match: bool) -> int | None:
    if not matches:
        return None
    if prefers_first_match:
        return int(matches[0].start())
    return int(matches[-1].start())


def _segment_sections(text: str, patterns: dict[str, list[SectionPattern]]) -> list[SectionSpan]:
    if not text.strip():
        return []
    markers: list[tuple[int, str]] = []
    for label, label_patterns in patterns.items():
        # For each label, evaluate each pattern. First-match patterns (anchor
        # IDs, uppercase headers) are TOC-immune; mixed-case text patterns
        # may collide with TOC entries so use last match. Pick the EARLIEST
        # resulting position so the more specific marker wins over a
        # downstream cross-reference or continuation footer.
        best: int | None = None
        for sp in label_patterns:
            pos = _pick_best_match(
                list(sp.pattern.finditer(text)),
                prefers_first_match=sp.prefers_first_match,
            )
            if pos is None:
                continue
            if best is None or pos < best:
                best = pos
        if best is not None:
            markers.append((best, label))
    if not markers:
        normalized = normalize_whitespace(text)
        return [
            SectionSpan(
                section_label="full_document",
                start_offset=0,
                end_offset=len(text),
                text=normalized,
            )
        ]
    dedup: dict[str, int] = {}
    for start, label in sorted(markers):
        if label not in dedup:
            dedup[label] = start
    ordered = sorted((idx, label) for label, idx in dedup.items())
    spans: list[SectionSpan] = []
    for idx, (start, label) in enumerate(ordered):
        end = ordered[idx + 1][0] if idx + 1 < len(ordered) else len(text)
        chunk = normalize_whitespace(text[start:end])
        if not chunk:
            continue
        spans.append(
            SectionSpan(
                section_label=label,
                start_offset=int(start),
                end_offset=int(end),
                text=chunk,
            )
        )
    return spans


def segment_10k_sections(text: str) -> list[SectionSpan]:
    return _segment_sections(text, SECTION_PATTERNS)


def segment_10q_sections(text: str) -> list[SectionSpan]:
    return _segment_sections(text, SECTION_PATTERNS_10Q)


def segment_8k_sections(text: str) -> list[SectionSpan]:
    return _segment_sections(text, SECTION_PATTERNS_8K)


def item_code_for_8k_section(label: str) -> str | None:
    metadata = SECTION_METADATA_8K.get(label)
    if metadata is None:
        return None
    return metadata["item_code"]


def event_category_for_8k_section(label: str) -> str | None:
    metadata = SECTION_METADATA_8K.get(label)
    if metadata is None:
        return None
    return metadata["event_category"]


def section_by_label(spans: list[SectionSpan], label: str) -> SectionSpan | None:
    for span in spans:
        if span.section_label == label:
            return span
    return None
