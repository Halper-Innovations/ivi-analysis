from __future__ import annotations

import re
from typing import Any

from app.util.text import extract_snippet, normalize_whitespace


SHARES_RE = re.compile(r"shares\s+outstanding[^\d]{0,60}([\d,]{3,})", re.IGNORECASE)
SHARES_RE_REVERSED = re.compile(r"([\d,]{3,})\s+shares\s+outstanding", re.IGNORECASE)
PERIOD_RE = re.compile(r"for\s+the\s+period\s+ended\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", re.IGNORECASE)


def extract_cover_page_facts(text: str, source_url: str) -> list[dict[str, Any]]:
    content = normalize_whitespace(text)
    out: list[dict[str, Any]] = []

    shares_match = SHARES_RE.search(content) or SHARES_RE_REVERSED.search(content)
    if shares_match:
        raw = shares_match.group(1)
        try:
            value = int(raw.replace(",", ""))
        except ValueError:
            value = None
        out.append(
            {
                "fact_type": "shares_outstanding",
                "value_json": {"value": value, "raw": raw},
                "source_url": source_url,
                "snippet": extract_snippet(content, shares_match.group(0), width=450),
                "section_label": "cover_page",
            }
        )

    period_match = PERIOD_RE.search(content)
    if period_match:
        raw = period_match.group(1)
        out.append(
            {
                "fact_type": "period_end_label",
                "value_json": {"value": raw},
                "source_url": source_url,
                "snippet": extract_snippet(content, period_match.group(0), width=450),
                "section_label": "cover_page",
            }
        )

    return out
