from __future__ import annotations

from app.util.text import extract_snippet, normalize_whitespace


DEBT_KEYWORDS = ["credit facility", "liquidity", "minimum cash", "debt covenant", "interest rate"]


def extract_debt_liquidity_signals(text: str, source_url: str) -> list[dict]:
    content = normalize_whitespace(text)
    lowered = content.lower()
    out: list[dict] = []
    for keyword in DEBT_KEYWORDS:
        if keyword in lowered:
            out.append(
                {
                    "fact_type": "debt_liquidity_signal",
                    "value_json": {"keyword": keyword, "present": True},
                    "source_url": source_url,
                    "snippet": extract_snippet(content, keyword, width=500),
                    "section_label": "debt_liquidity",
                }
            )
    return out
