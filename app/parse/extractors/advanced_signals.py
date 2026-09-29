from __future__ import annotations

import re
from typing import Any

from app.util.text import extract_snippet, normalize_whitespace


PERCENT_RE = re.compile(r"(\d{1,2}(?:\.\d+)?)%")
YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _signal(fact_type: str, keyword: str, content: str, source_url: str, section: str, value_json: dict[str, Any]) -> dict[str, Any]:
    return {
        "fact_type": fact_type,
        "value_json": value_json,
        "source_url": source_url,
        "snippet": extract_snippet(content, keyword, width=550),
        "section_label": section,
    }


def extract_segment_signals(text: str, source_url: str) -> list[dict[str, Any]]:
    content = normalize_whitespace(text)
    lowered = content.lower()
    out: list[dict[str, Any]] = []
    keywords = ["reportable segment", "segments", "segment performance"]
    for keyword in keywords:
        if keyword in lowered:
            segment_names = re.findall(r"([A-Z][A-Za-z&\-\s]{2,40})\s+segment", content)
            out.append(
                _signal(
                    "segments_signal",
                    keyword,
                    content,
                    source_url,
                    "segments",
                    {
                        "present": True,
                        "keyword": keyword,
                        "segment_names": sorted(list({s.strip() for s in segment_names}))[:8],
                    },
                )
            )
            break
    return out


def extract_sbc_dilution_signals(text: str, source_url: str) -> list[dict[str, Any]]:
    content = normalize_whitespace(text)
    lowered = content.lower()
    out: list[dict[str, Any]] = []
    keywords = ["stock-based compensation", "share-based compensation", "share issuance", "weighted-average shares"]
    for keyword in keywords:
        if keyword in lowered:
            out.append(
                _signal(
                    "sbc_dilution_signal",
                    keyword,
                    content,
                    source_url,
                    "equity_dilution",
                    {
                        "present": True,
                        "keyword": keyword,
                    },
                )
            )
    return out


def extract_debt_maturity_and_covenants(text: str, source_url: str) -> list[dict[str, Any]]:
    content = normalize_whitespace(text)
    lowered = content.lower()
    out: list[dict[str, Any]] = []

    maturity_keywords = ["maturity", "senior notes due", "term loan due", "debt due"]
    for keyword in maturity_keywords:
        if keyword in lowered:
            years = sorted(list({int(y) for y in YEAR_RE.findall(content)}))
            out.append(
                _signal(
                    "debt_maturity_signal",
                    keyword,
                    content,
                    source_url,
                    "debt_maturity",
                    {
                        "present": True,
                        "keyword": keyword,
                        "years_mentioned": years[:12],
                    },
                )
            )
            break

    covenant_keywords = ["covenant", "leverage ratio", "interest coverage", "minimum liquidity", "waiver"]
    for keyword in covenant_keywords:
        if keyword in lowered:
            out.append(
                _signal(
                    "covenant_signal",
                    keyword,
                    content,
                    source_url,
                    "covenants",
                    {
                        "present": True,
                        "keyword": keyword,
                    },
                )
            )
            break

    return out


def extract_customer_concentration_signal(text: str, source_url: str) -> list[dict[str, Any]]:
    content = normalize_whitespace(text)
    lowered = content.lower()
    keywords = ["customer concentration", "single customer", "major customer"]
    for keyword in keywords:
        if keyword not in lowered:
            continue
        snippet = extract_snippet(content, keyword, width=550)
        pct = None
        pct_match = PERCENT_RE.search(snippet)
        if pct_match:
            try:
                pct = float(pct_match.group(1))
            except Exception:
                pct = None
        return [
            _signal(
                "customer_concentration_signal",
                keyword,
                content,
                source_url,
                "customer_concentration",
                {
                    "present": True,
                    "keyword": keyword,
                    "customer_pct": pct,
                },
            )
        ]
    return []


def extract_cash_flow_quality_signals(text: str, source_url: str) -> list[dict[str, Any]]:
    content = normalize_whitespace(text)
    lowered = content.lower()
    out: list[dict[str, Any]] = []
    keywords = [
        "working capital",
        "accounts receivable",
        "inventory",
        "non-cash",
        "factoring",
        "deferred revenue",
    ]
    for keyword in keywords:
        if keyword in lowered:
            out.append(
                _signal(
                    "cash_flow_quality_signal",
                    keyword,
                    content,
                    source_url,
                    "cash_flow_quality",
                    {
                        "present": True,
                        "keyword": keyword,
                    },
                )
            )
    return out


def extract_non_gaap_reconciliation_signals(text: str, source_url: str) -> list[dict[str, Any]]:
    content = normalize_whitespace(text)
    lowered = content.lower()
    keywords = ["non-gaap", "adjusted ebitda", "adjusted earnings", "reconciliation"]
    for keyword in keywords:
        if keyword in lowered:
            return [
                _signal(
                    "non_gaap_reconciliation_signal",
                    keyword,
                    content,
                    source_url,
                    "non_gaap",
                    {
                        "present": True,
                        "keyword": keyword,
                    },
                )
            ]
    return []
