from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.alpha.filing_investigator import _load_full_text
from app.alpha.schemas import TickerSignalPacket
from app.alpha.solvency_scanner import going_concern_asserted
from app.config import get_config
from app.db import get_db
from app.insurance.peer_context import compute_insurance_subtype_peer_relative_metrics
from app.insurance.routing import ISSUER_INSURANCE_UNDERWRITER
from app.research.current_event_context import load_current_event_context
from app.research.adapters.base import AdapterContext
from app.research.adapters.transcripts import TranscriptAdapter
from app.research.filing_context import load_research_filing_context
from app.util.html_strip import strip_html
from app.util.companyfacts_aliases import normalize_companyfacts_line_items
from app.util.financial_data_access import ANNUAL_COMPANYFACTS_PERIOD_TYPES, companyfacts_rows
from app.valuation.peer_context import (
    PEER_METRIC_DISPLAY_NAMES,
    SUPPORTED_COMPARISON_METRICS,
    compute_peer_relative_metrics,
    normalize_peer_metric,
)

SUPPORTED_PEER_METRICS = set(SUPPORTED_COMPARISON_METRICS) | set(PEER_METRIC_DISPLAY_NAMES.values())


@dataclass(frozen=True)
class AlphaToolContext:
    sector: str
    ticker: str
    packet: TickerSignalPacket
    as_of_date: str
    consensus_rank: int | None = None
    consensus_score: float | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def alpha_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "name": "fetch_filing_section",
            "description": "Fetch targeted passages from the latest cached annual filing using keywords and an optional section focus.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "section_focus": {
                        "type": "string",
                        "description": "Optional filing section focus such as liquidity, risk, competition, margins, or capital allocation.",
                    },
                    "max_chars": {"type": "integer"},
                },
                "required": ["keywords"],
                "additionalProperties": False,
            },
        },
        {
            "name": "fetch_companyfacts_timeseries",
            "description": "Fetch annual companyfacts timeseries for one or more normalized line items.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "line_items": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "years": {"type": "integer"},
                },
                "required": ["line_items"],
                "additionalProperties": False,
            },
        },
        {
            "name": "fetch_kpi_trends",
            "description": "Fetch deterministic KPI trend and packet summary context for the candidate.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "fetch_insurance_evidence_packet",
            "description": "Fetch the deterministic insurance-specific routing, valuation, and evidence packet when present.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "compare_peer_metric",
            "description": "Compare the candidate against sector peers on a deterministic annual peer metric.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": [
                            "EV/EBITDA",
                            "EV/EBIT",
                            "EV/Sales",
                            "P/E",
                            "P/B",
                            "FCF yield",
                            "Dividend yield",
                            "roic",
                            "operating_margin",
                            "revenue_growth_5y",
                        ],
                    }
                },
                "required": ["metric"],
                "additionalProperties": False,
            },
        },
        {
            "name": "analyze_dilution",
            "description": "Analyze annual share-count trend and dilution direction from companyfacts.",
            "input_schema": {
                "type": "object",
                "properties": {"years": {"type": "integer"}},
                "additionalProperties": False,
            },
        },
        {
            "name": "analyze_liquidity_stress",
            "description": "Summarize deterministic solvency and liquidity stress signals already computed for the candidate.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "analyze_capital_structure_resolution",
            "description": "Resolve whether liquidity, refinancing, covenant, maturity, or no-assurance financing flags are active, cured, or unresolved.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "max_chars": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "analyze_capital_allocation",
            "description": "Summarize deterministic capital-allocation signals from scorecard context and share-count trend.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "fetch_current_events",
            "description": "Fetch canonical company-controlled current events for the candidate.",
            "input_schema": {
                "type": "object",
                "properties": {"max_items": {"type": "integer"}},
                "additionalProperties": False,
            },
        },
        {
            "name": "fetch_recent_filing_context",
            "description": "Fetch readable recent 10-Q and 8-K context for the candidate when cached SEC filings are available.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "quarters": {"type": "integer"},
                    "material_event_window_days": {"type": "integer"},
                    "max_documents": {"type": "integer"},
                    "max_chars": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "fetch_transcript_excerpt",
            "description": "Fetch transcript follow-up if transcript support is enabled; otherwise returns explicit unavailability.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "focus": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "finalize_candidate_investigation",
            "description": "Finalize the candidate investigation once further tool calls would not materially change the verdict.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["PROCEED", "WATCH", "AVOID", "NO_WINNER"],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["HIGH", "MODERATE", "LOW"],
                    },
                    "key_findings": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "open_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "key_risk": {"type": "string"},
                    "falsification_trigger": {"type": "string"},
                    "reasoning_trace": {"type": "string"},
                    "model_validity": {
                        "type": "string",
                        "enum": [
                            "VALID",
                            "INVALID_SECURITY_TYPE",
                            "INVALID_VALUATION_MODEL",
                            "INSUFFICIENT_SECURITY_IDENTITY",
                            "UNKNOWN",
                        ],
                    },
                    "selection_blockers": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "eligible_for_selection": {"type": "boolean"},
                },
                "required": [
                    "verdict",
                    "confidence",
                    "key_findings",
                    "open_questions",
                    "key_risk",
                    "falsification_trigger",
                    "reasoning_trace",
                    "model_validity",
                    "selection_blockers",
                    "eligible_for_selection",
                ],
                "additionalProperties": False,
            },
        },
    ]


def _companyfacts_series(
    ticker: str,
    *,
    line_items: list[str],
    years: int,
    as_of_date: str,
) -> dict[str, list[dict[str, Any]]]:
    years = max(1, min(int(years or 5), 10))
    with get_db() as conn:
        rows = companyfacts_rows(
            conn,
            ticker,
            columns=(
                "fiscal_year",
                "line_item",
                "value",
                "units",
                "period_end",
                "filed_date",
                "source_url",
                "accession",
            ),
            line_items=tuple(dict.fromkeys(line_items)),
            period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
            as_of_date=as_of_date,
            value_not_null=True,
            require_filed_asof=True,
            order_by="fiscal_year ASC",
        )
    series: dict[str, list[dict[str, Any]]] = {item: [] for item in line_items}
    for row in rows:
        item = str(row["line_item"])
        if item not in series:
            continue
        value = row["value"]
        units = str(row["units"] or "").strip()
        period_end = str(row["period_end"] or "").strip()
        filed_date = str(row["filed_date"] or "").strip()
        source_url = str(row["source_url"] or "").strip()
        accession = str(row["accession"] or "").strip()
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not units
            or not period_end
            or not filed_date
            or not source_url
        ):
            continue
        fact_row: dict[str, Any] = {
            "fiscal_year": int(row["fiscal_year"]),
            "value": float(value),
            "unit": units,
            "units": units,
            "period_end": period_end,
            "filed_date": filed_date,
            "source": "SEC_COMPANYFACTS",
            "source_url": source_url,
            "source_reference": accession or source_url,
            "accession": accession or None,
            "as_of_date": as_of_date,
        }
        if item == "shares_outstanding":
            fact_row.update(
                {
                    "raw_value": float(value),
                    "normalized_value": float(value),
                    "shares_basis": "UNADJUSTED",
                    "split_adjustment_factor": 1.0,
                    "split_effective_date": None,
                    "split_source_reference": accession or source_url,
                }
            )
        series[item].append(fact_row)
    for key in list(series.keys()):
        series[key] = series[key][-years:]
    return series


def _shares_cagr(rows: list[dict[str, Any]]) -> float | None:
    if len(rows) < 2:
        return None
    start = rows[0]
    end = rows[-1]
    start_value = float(start["value"])
    end_value = float(end["value"])
    years = int(end["fiscal_year"]) - int(start["fiscal_year"])
    if years <= 0 or start_value <= 0:
        return None
    return (end_value / start_value) ** (1.0 / years) - 1.0


def _split_adjusted_share_series(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Use only explicit source-to-normalized split lineage.

    Share-count jumps can be issuance, buybacks, M&A, or a stock split. Their
    magnitude is not split evidence. Callers must provide each row's raw and
    normalized values, literal basis, factor, and (for a non-unit factor)
    effective date before dilution arithmetic is decision-usable.
    """

    if not rows:
        return [], [], True

    adjusted_rows: list[dict[str, Any]] = []
    adjustments: list[dict[str, Any]] = []
    for row in rows:
        raw_value = row.get("raw_value")
        normalized_value = row.get("normalized_value")
        factor = row.get("split_adjustment_factor")
        basis = str(row.get("shares_basis") or "").strip().upper()
        effective_date = str(row.get("split_effective_date") or "").strip()
        if not (
            isinstance(raw_value, (int, float))
            and not isinstance(raw_value, bool)
            and float(raw_value) > 0
            and isinstance(normalized_value, (int, float))
            and not isinstance(normalized_value, bool)
            and float(normalized_value) > 0
            and isinstance(factor, (int, float))
            and not isinstance(factor, bool)
            and float(factor) > 0
            and basis in {"UNADJUSTED", "SPLIT_ADJUSTED"}
        ):
            return [dict(item) for item in rows], [], False
        raw_number = float(raw_value)
        normalized_number = float(normalized_value)
        factor_number = float(factor)
        if basis == "UNADJUSTED":
            if not math.isclose(factor_number, 1.0, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
                raw_number,
                normalized_number,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                return [dict(item) for item in rows], [], False
        elif not math.isclose(
            raw_number * factor_number,
            normalized_number,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ) or (
            not math.isclose(
                factor_number,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and not effective_date
        ):
            return [dict(item) for item in rows], [], False
        adjusted_row = dict(row)
        adjusted_row["raw_value"] = raw_number
        adjusted_row["value"] = normalized_number
        adjusted_rows.append(adjusted_row)
        if not math.isclose(
            factor_number,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            adjustments.append(
                {
                    "fiscal_year": int(row["fiscal_year"]),
                    "raw_value": raw_number,
                    "normalized_value": normalized_number,
                    "shares_basis": basis,
                    "split_adjustment_factor": factor_number,
                    "split_effective_date": effective_date,
                    "source_reference": row.get("split_source_reference"),
                }
            )
    return adjusted_rows, adjustments, True


def _pct_summary(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value) * 100:.1f}%"
    return "UNKNOWN"


def _text_or_unknown(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "UNKNOWN"


_NARRATIVE_SECTION_PATTERNS: dict[str, list[tuple[str, str, str]]] = {
    "quarterly": [
        (
            "quarterly_mdna",
            r"Item\s+2[\s\.\:\-–—]+Management(?:['’]s|s)?\s+Discussion",
            r"Item\s+3[\s\.\:\-–—]+Quantitative|Item\s+4[\s\.\:\-–—]+Controls|Part\s+II[\s\.\:\-–—]+Other",
        ),
        (
            "quarterly_risk_update",
            r"Item\s+1A[\s\.\:\-–—]+Risk\s+Factors",
            r"Item\s+2[\s\.\:\-–—]+Unregistered|Item\s+3[\s\.\:\-–—]+Defaults|Item\s+4[\s\.\:\-–—]+Mine|Item\s+5[\s\.\:\-–—]+Other|Item\s+6[\s\.\:\-–—]+Exhibits|SIGNATURES",
        ),
    ],
    "material_event": [
        (
            "8k_results",
            r"Item\s+2\.02[\s\.\:\-–—]+Results",
            r"Item\s+[1-9]\.\d{2}|SIGNATURES|EXHIBIT\s+INDEX|EXHIBITS",
        ),
        (
            "8k_agreement",
            r"Item\s+1\.01[\s\.\:\-–—]+Entry\s+Into",
            r"Item\s+[1-9]\.\d{2}|SIGNATURES|EXHIBIT\s+INDEX|EXHIBITS",
        ),
        (
            "8k_agreement_termination",
            r"Item\s+1\.02[\s\.\:\-–—]+Termination",
            r"Item\s+[1-9]\.\d{2}|SIGNATURES|EXHIBIT\s+INDEX|EXHIBITS",
        ),
        (
            "8k_financial_obligation",
            r"Item\s+2\.03[\s\.\:\-–—]+Creation",
            r"Item\s+[1-9]\.\d{2}|SIGNATURES|EXHIBIT\s+INDEX|EXHIBITS",
        ),
        (
            "8k_default",
            r"Item\s+2\.04[\s\.\:\-–—]+Triggering",
            r"Item\s+[1-9]\.\d{2}|SIGNATURES|EXHIBIT\s+INDEX|EXHIBITS",
        ),
        (
            "8k_restructuring",
            r"Item\s+2\.05[\s\.\:\-–—]+Costs\s+Associated",
            r"Item\s+[1-9]\.\d{2}|SIGNATURES|EXHIBIT\s+INDEX|EXHIBITS",
        ),
        (
            "8k_impairment",
            r"Item\s+2\.06[\s\.\:\-–—]+Material\s+Impairments",
            r"Item\s+[1-9]\.\d{2}|SIGNATURES|EXHIBIT\s+INDEX|EXHIBITS",
        ),
        (
            "8k_delisting",
            r"Item\s+3\.01[\s\.\:\-–—]+Notice",
            r"Item\s+[1-9]\.\d{2}|SIGNATURES|EXHIBIT\s+INDEX|EXHIBITS",
        ),
        (
            "8k_non_reliance",
            r"Item\s+4\.02[\s\.\:\-–—]+Non-Reliance",
            r"Item\s+[1-9]\.\d{2}|SIGNATURES|EXHIBIT\s+INDEX|EXHIBITS",
        ),
        (
            "8k_leadership",
            r"Item\s+5\.02[\s\.\:\-–—]+Departure",
            r"Item\s+[1-9]\.\d{2}|SIGNATURES|EXHIBIT\s+INDEX|EXHIBITS",
        ),
        (
            "8k_reg_fd",
            r"Item\s+7\.01[\s\.\:\-–—]+Regulation\s+FD",
            r"Item\s+[1-9]\.\d{2}|SIGNATURES|EXHIBIT\s+INDEX|EXHIBITS",
        ),
        (
            "8k_other_event",
            r"Item\s+8\.01[\s\.\:\-–—]+Other\s+Events",
            r"Item\s+[1-9]\.\d{2}|SIGNATURES|EXHIBIT\s+INDEX|EXHIBITS",
        ),
        (
            "material_event_body",
            r"Item\s+[1-9]\.\d{2}",
            r"Item\s+[1-9]\.\d{2}|SIGNATURES|EXHIBIT\s+INDEX|EXHIBITS",
        ),
    ],
    "risk": [
        (
            "risk_factors",
            r"Item\s+1A[\s\.\:\-–—]+Risk\s+Factors",
            r"Item\s+1B[\s\.\:\-–—]+Unresolved|Item\s+1C[\s\.\:\-–—]+Cybersecurity|Item\s+2[\s\.\:\-–—]+Properties",
        ),
        ("risk_factors", r"(?:D[\s\.\:\-–—]+)?Risk\s+Factors", r"Item\s+4[\s\.\:\-–—]"),
    ],
    "competition": [
        (
            "business",
            r"Item\s+1[\s\.\:\-–—]+Business",
            r"Item\s+1A[\s\.\:\-–—]+Risk\s+Factors|Item\s+2[\s\.\:\-–—]+Properties",
        ),
        (
            "risk_factors",
            r"Item\s+1A[\s\.\:\-–—]+Risk\s+Factors",
            r"Item\s+1B[\s\.\:\-–—]+Unresolved|Item\s+1C[\s\.\:\-–—]+Cybersecurity|Item\s+2[\s\.\:\-–—]+Properties",
        ),
    ],
    "liquidity": [
        (
            "mdna",
            r"Item\s+7[\s\.\:\-–—]+Management",
            r"Item\s+7A[\s\.\:\-–—]+Quantitative|Item\s+8[\s\.\:\-–—]+Financial",
        ),
        (
            "operating_review",
            r"Item\s+5[\s\.\:\-–—]+Operating",
            r"Item\s+6[\s\.\:\-–—]|Item\s+7[\s\.\:\-–—]",
        ),
    ],
    "capital allocation": [
        (
            "mdna",
            r"Item\s+7[\s\.\:\-–—]+Management",
            r"Item\s+7A[\s\.\:\-–—]+Quantitative|Item\s+8[\s\.\:\-–—]+Financial",
        ),
        (
            "business",
            r"Item\s+1[\s\.\:\-–—]+Business",
            r"Item\s+1A[\s\.\:\-–—]+Risk\s+Factors|Item\s+2[\s\.\:\-–—]+Properties",
        ),
    ],
    "margins": [
        (
            "mdna",
            r"Item\s+7[\s\.\:\-–—]+Management",
            r"Item\s+7A[\s\.\:\-–—]+Quantitative|Item\s+8[\s\.\:\-–—]+Financial",
        ),
        (
            "operating_review",
            r"Item\s+5[\s\.\:\-–—]+Operating",
            r"Item\s+6[\s\.\:\-–—]|Item\s+7[\s\.\:\-–—]",
        ),
    ],
}

_DEFAULT_NARRATIVE_PATTERNS: list[tuple[str, str, str]] = [
    (
        "mdna",
        r"Item\s+7[\s\.\:\-–—]+Management",
        r"Item\s+7A[\s\.\:\-–—]+Quantitative|Item\s+8[\s\.\:\-–—]+Financial",
    ),
    (
        "risk_factors",
        r"Item\s+1A[\s\.\:\-–—]+Risk\s+Factors",
        r"Item\s+1B[\s\.\:\-–—]+Unresolved|Item\s+1C[\s\.\:\-–—]+Cybersecurity|Item\s+2[\s\.\:\-–—]+Properties",
    ),
    (
        "business",
        r"Item\s+1[\s\.\:\-–—]+Business",
        r"Item\s+1A[\s\.\:\-–—]+Risk\s+Factors|Item\s+2[\s\.\:\-–—]+Properties",
    ),
    (
        "operating_review",
        r"Item\s+5[\s\.\:\-–—]+Operating",
        r"Item\s+6[\s\.\:\-–—]|Item\s+7[\s\.\:\-–—]",
    ),
]

_FILING_NOISE_RE = re.compile(
    r"\b(?:us-gaap|dei|srt|xbrli|iso4217|country|naics):|"
    r"contextref|decimals=|unitref|duration_|instant_|member\b|"
    r"\b[A-Za-z]+(?:Axis|Domain|Member|Table|LineItems)\b|"
    r"http://(?:fasb\.org|www\.sec\.gov|xbrl\.sec\.gov)/|"
    r"#[A-Za-z]+(?:Axis|Domain|Member|Table|LineItems)\b|"
    r"\b\d{10,}\b",
    re.IGNORECASE,
)


def _filing_excerpt_noise_count(text: str) -> int:
    return len(_FILING_NOISE_RE.findall(text or ""))


def _is_noisy_filing_excerpt(text: str) -> bool:
    if not text.strip():
        return True
    word_count = max(1, len(re.findall(r"[A-Za-z][A-Za-z\-']+", text)))
    noise_count = _filing_excerpt_noise_count(text)
    inline_xbrl_tokens = len(
        re.findall(
            r"(?:us-gaap|dei|srt|xbrli|iso4217|stpr|[a-z]{2,12}):|"
            r"http://(?:fasb\.org|www\.sec\.gov|xbrl\.sec\.gov)/|"
            r"#[A-Za-z]+(?:Axis|Domain|Member|Table|LineItems)\b|"
            r"\b\d{10,}\b",
            text,
            re.IGNORECASE,
        )
    )
    if inline_xbrl_tokens >= 8:
        return True
    if noise_count >= 6:
        return True
    if noise_count / word_count > 0.015:
        return True
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", text))
    statement_table_hits = len(
        re.findall(
            r"\b(?:revenue|cost of revenue|gross profit|operating income|net income|"
            r"earnings per share|weighted average shares|cash and cash equivalents|"
            r"accounts receivable|assets|liabilities|stockholders'? equity|"
            r"accumulated deficit|cash flows?)\b",
            text,
            re.IGNORECASE,
        )
    )
    if statement_table_hits >= 8 and sentence_count <= 2:
        return True
    # Glossaries are readable English, but they are usually poor answers to
    # requests for liquidity, risk, or operating evidence.
    short_definition_hits = len(re.findall(r"\b[A-Z]{2,8}\s+[A-Z][A-Za-z][^.;]{5,80}", text))
    return short_definition_hits >= 8 and word_count < 260


def _filing_section_patterns(section_focus: str | None) -> list[tuple[str, str, str]]:
    focus = str(section_focus or "").strip().lower()
    selected: list[tuple[str, str, str]] = []
    for key, patterns in _NARRATIVE_SECTION_PATTERNS.items():
        if key in focus:
            selected.extend(patterns)
    selected.extend(_DEFAULT_NARRATIVE_PATTERNS)

    deduped: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in selected:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _extract_narrative_section(
    text: str, start_pattern: str, end_pattern: str, *, min_length: int = 200
) -> str:
    starts = list(re.finditer(start_pattern, text, re.IGNORECASE))
    ends = list(re.finditer(end_pattern, text, re.IGNORECASE))
    candidates: list[tuple[int, int, str]] = []
    for start in starts:
        for end in ends:
            if end.start() <= start.end():
                continue
            gap = end.start() - start.start()
            if gap <= min_length:
                break
            candidates.append((start.start(), gap, text[start.start() : end.start()].strip()))
            break
    if not candidates:
        return ""
    # Prefer the later real section over table-of-contents references. The
    # imported filing investigator helper intentionally picks the widest span
    # for anomaly work, which can start at the TOC; tool calls need the body.
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][2][:30000]


def _candidate_filing_sections(text: str, section_focus: str | None) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    seen_text: set[str] = set()
    for label, start_pattern, end_pattern in _filing_section_patterns(section_focus):
        section = _extract_narrative_section(text, start_pattern, end_pattern, min_length=200)
        section = section.strip()
        if not section:
            continue
        fingerprint = section[:500]
        if fingerprint in seen_text:
            continue
        seen_text.add(fingerprint)
        sections.append((label, section))
    return sections


def _keyword_windows(text: str, keywords: list[str], *, max_chars: int) -> tuple[str, int, bool]:
    lower = text.lower()
    clean: list[tuple[int, str]] = []
    noisy: list[tuple[int, str]] = []
    seen_ranges: list[tuple[int, int]] = []

    for keyword in keywords:
        kw = str(keyword or "").strip()
        if not kw:
            continue
        start = 0
        kw_lower = kw.lower()
        while True:
            idx = lower.find(kw_lower, start)
            if idx < 0:
                break
            window_start = max(0, idx - 900)
            window_end = min(len(text), idx + len(kw) + 1200)
            overlaps = any(
                ws <= window_start <= we or ws <= window_end <= we for ws, we in seen_ranges
            )
            if not overlaps:
                passage = text[window_start:window_end].strip()
                if _is_noisy_filing_excerpt(passage):
                    noisy.append((idx, passage))
                else:
                    clean.append((idx, passage))
                seen_ranges.append((window_start, window_end))
            start = idx + len(kw)

    selected = clean if clean else noisy
    selected.sort(key=lambda item: item[0])
    result = "\n\n[...]\n\n".join(passage for _, passage in selected)
    return result[:max_chars], len(noisy), not clean and bool(noisy)


def _extract_readable_filing_passages(
    text: str,
    *,
    keywords: list[str],
    section_focus: str | None,
    max_chars: int,
) -> tuple[str, str, int, bool]:
    sections = _candidate_filing_sections(text, section_focus)
    for label, section in sections:
        excerpt, noisy_count, used_noisy = _keyword_windows(section, keywords, max_chars=max_chars)
        if excerpt and not used_noisy:
            return excerpt, f"section:{label}", noisy_count, False

    for label, section in sections:
        if not _is_noisy_filing_excerpt(section[:max_chars]):
            return section[:max_chars], f"section_start:{label}", 0, False

    excerpt, noisy_count, used_noisy = _keyword_windows(text, keywords, max_chars=max_chars)
    if excerpt:
        strategy = "full_text_noisy_fallback" if used_noisy else "full_text_keyword"
        return excerpt, strategy, noisy_count, used_noisy

    return text[:max_chars], "full_text_start", 0, _is_noisy_filing_excerpt(text[:max_chars])


def _recent_filing_extraction_plan(document: Any) -> tuple[str, list[str]]:
    form_type = str(getattr(document, "form_type", "") or "").upper()
    role = str(getattr(document, "role", "") or "").lower()
    if role == "quarterly" or form_type.startswith("10-Q"):
        return (
            "quarterly liquidity risk margins",
            [
                "management discussion",
                "liquidity",
                "revenue",
                "cash flow",
                "risk factors",
                "outlook",
            ],
        )
    if role == "material_event" or form_type.startswith("8-K"):
        return (
            "material_event",
            [
                "results",
                "guidance",
                "liquidity",
                "agreement",
                "acquisition",
                "material",
                "risk",
                "appointed",
                "appointment",
                "resigned",
                "officer",
                "compensation",
                "restructuring",
                "severance",
                "impairment",
                "impairments",
                "write-down",
                "writedown",
                "exit",
                "disposal",
                "termination",
                "terminated",
                "default",
                "accelerate",
                "acceleration",
                "delisting",
                "listing",
                "non-reliance",
                "restatement",
                "restated",
                "material weakness",
            ],
        )
    return (
        "liquidity risk capital allocation",
        [
            "management discussion",
            "liquidity",
            "risk factors",
            "capital allocation",
            "cash flow",
        ],
    )


def _extract_recent_filing_excerpt(document: Any, *, max_chars: int) -> tuple[str, str, int, bool]:
    clean_text = strip_html(getattr(document, "html", "") or "")
    section_focus, keywords = _recent_filing_extraction_plan(document)
    excerpt, strategy, noise_filtered, used_noisy = _extract_readable_filing_passages(
        clean_text,
        keywords=keywords,
        section_focus=section_focus,
        max_chars=max_chars,
    )
    return excerpt, strategy, noise_filtered, used_noisy


def _fetch_filing_section(
    ctx: AlphaToolContext, *, keywords: list[str], section_focus: str | None, max_chars: int | None
) -> dict[str, Any]:
    text, form_type = _load_full_text(ctx.ticker)
    if not text:
        return {
            "status": "unavailable",
            "ticker": ctx.ticker,
            "reason": "latest_annual_filing_not_cached",
            "keywords": keywords,
            "section_focus": section_focus,
            "usable_for_decision": False,
            "evidence_status": "NO_FILING_TEXT",
            "warnings": ["latest_annual_filing_not_cached"],
            "summary": "Latest annual filing text is not cached, so no filing evidence was available.",
        }
    limit = max(1000, min(int(max_chars or 8000), 20000))
    focus_tokens = [token for token in [section_focus] if isinstance(token, str) and token.strip()]
    keyword_list = [str(token).strip() for token in keywords if str(token).strip()]
    passages, source_strategy, noise_filtered, used_noisy_fallback = (
        _extract_readable_filing_passages(
            text,
            keywords=focus_tokens + keyword_list,
            section_focus=section_focus,
            max_chars=limit,
        )
    )
    warnings = []
    if used_noisy_fallback:
        warnings.append("readable_narrative_section_not_found")
    usable = bool(passages.strip()) and not used_noisy_fallback
    evidence_status = "READABLE_NARRATIVE" if usable else "NO_READABLE_NARRATIVE"
    return {
        "status": "ok",
        "ticker": ctx.ticker,
        "form_type": form_type,
        "section_focus": section_focus,
        "keywords": keyword_list,
        "excerpt": passages[:limit],
        "excerpt_chars": len(passages[:limit]),
        "source_strategy": source_strategy,
        "noise_filtered_passages": noise_filtered,
        "warnings": warnings,
        "usable_for_decision": usable,
        "evidence_status": evidence_status,
        "summary": (
            f"Readable filing narrative found via {source_strategy}."
            if usable
            else "Filing lookup did not find readable decision-usable narrative text."
        ),
    }


def _fetch_companyfacts_timeseries(
    ctx: AlphaToolContext, *, line_items: list[str], years: int | None
) -> dict[str, Any]:
    raw_items = [str(item).strip() for item in line_items if str(item).strip()]
    normalized_items, translated_items = normalize_companyfacts_line_items(raw_items)
    if not normalized_items:
        return {
            "status": "error",
            "ticker": ctx.ticker,
            "reason": "line_items_required",
            "usable_for_decision": False,
            "evidence_status": "LINE_ITEMS_REQUIRED",
            "warnings": ["line_items_required"],
            "summary": "Companyfacts timeseries requires at least one normalized line item.",
        }
    series = _companyfacts_series(
        ctx.ticker,
        line_items=normalized_items,
        years=int(years or 5),
        as_of_date=ctx.as_of_date,
    )
    populated = [item for item in normalized_items if series.get(item)]
    missing = [item for item in normalized_items if not series.get(item)]
    usable = bool(populated)
    warnings = []
    if translated_items:
        warnings.append(f"translated_line_items:{', '.join(translated_items)}")
    if missing:
        warnings.append(f"missing_line_items:{', '.join(missing)}")
    return {
        "status": "ok" if usable else "unavailable",
        "ticker": ctx.ticker,
        "as_of_date": ctx.as_of_date,
        "series": series,
        "requested_line_items": normalized_items,
        "raw_line_items": raw_items,
        "translated_line_items": translated_items,
        "populated_line_items": populated,
        "missing_line_items": missing,
        "usable_for_decision": usable,
        "evidence_status": "HAS_COMPANYFACTS_SERIES" if usable else "NO_COMPANYFACTS_SERIES",
        "warnings": warnings,
        "summary": (
            f"Companyfacts annual series available for {', '.join(populated)}."
            if usable
            else "No annual companyfacts series were available for the requested normalized line items."
        ),
    }


def _fetch_kpi_trends(ctx: AlphaToolContext) -> dict[str, Any]:
    packet = ctx.packet
    scorecard = packet.raw_valuation if isinstance(packet.raw_valuation, dict) else {}
    quality_ctx = packet.raw_quality_ctx if isinstance(packet.raw_quality_ctx, dict) else {}
    pzd = (
        scorecard.get("pricing_zone_detail")
        if isinstance(scorecard.get("pricing_zone_detail"), dict)
        else {}
    )
    method_tension = (
        scorecard.get("method_tension") if isinstance(scorecard.get("method_tension"), dict) else {}
    )
    insurance_operating_metrics = (
        packet.insurance_packet.get("operating_metrics")
        if isinstance(packet.insurance_packet, dict)
        and isinstance(packet.insurance_packet.get("operating_metrics"), dict)
        else {}
    )
    insurance_peer_context: dict[str, Any] = {}
    if packet.issuer_type == ISSUER_INSURANCE_UNDERWRITER:
        try:
            peer = compute_insurance_subtype_peer_relative_metrics(packet.ticker, ctx.as_of_date)
            insurance_peer_context = {
                "status": peer.get("status"),
                "peer_scope": peer.get("peer_scope"),
                "peer_group": peer.get("peer_group"),
                "insurance_subtype": peer.get("insurance_subtype"),
                "peer_count": peer.get("peer_count"),
                "relative_position": peer.get("relative_position"),
                "relative_ratios": peer.get("relative_ratios"),
                "sector_medians": peer.get("sector_medians"),
                "fallback_reason": peer.get("fallback_reason"),
            }
        except Exception:
            insurance_peer_context = {"status": "ERROR"}
    insurance_operating_metrics_payload = {
        "status": insurance_operating_metrics.get("status"),
        "confidence": insurance_operating_metrics.get("confidence"),
        "combined_ratio": insurance_operating_metrics.get("combined_ratio"),
        "loss_ratio": insurance_operating_metrics.get("loss_ratio"),
        "expense_ratio": insurance_operating_metrics.get("expense_ratio"),
        "combined_ratio_assessment": insurance_operating_metrics.get("combined_ratio_assessment"),
        "reserve_development": insurance_operating_metrics.get("reserve_development"),
        "reinsurance_program": insurance_operating_metrics.get("reinsurance_program"),
        "catastrophe_exposure": insurance_operating_metrics.get("catastrophe_exposure"),
        "missing_components": list(insurance_operating_metrics.get("missing_components") or []),
    }
    for key in (
        "metric_family",
        "pmier_excess_ratio",
        "pmier_available_to_required_ratio",
        "primary_iif_billion",
        "primary_rif_billion",
        "new_insurance_written_billion",
        "customer_count",
        "policies_in_force",
        "loans_in_default",
        "default_rate",
        "rif_on_defaulted_loans_billion",
        "annual_persistency",
        "quarterly_runoff",
        "claims_paid_count",
        "claims_paid_million",
        "credit_capital_assessment",
    ):
        if key in insurance_operating_metrics:
            insurance_operating_metrics_payload[key] = insurance_operating_metrics.get(key)
    kpi_summary = (
        f"{ctx.ticker} KPI context: 5Y revenue CAGR {_pct_summary(quality_ctx.get('revenue_cagr_5y'))}, "
        f"earnings quality {_text_or_unknown(quality_ctx.get('earnings_quality'))}, "
        f"quarterly revenue trend {_text_or_unknown(packet.quarterly_revenue_trend)}, "
        f"method tension {_text_or_unknown(packet.method_tension_type)}."
    )
    return {
        "status": "ok",
        "ticker": ctx.ticker,
        "consensus_rank": ctx.consensus_rank,
        "consensus_score": ctx.consensus_score,
        "pricing_zone_detail": {
            "current_price": pzd.get("current_price"),
            "dcf_base": pzd.get("dcf_base"),
            "epv_adjusted": pzd.get("epv_adjusted"),
            "gate_action": pzd.get("gate_action"),
            "signal": scorecard.get("signal"),
        },
        "quality_context": {
            "revenue_cagr_5y": quality_ctx.get("revenue_cagr_5y"),
            "revenue_cagr_3y": quality_ctx.get("revenue_cagr_3y"),
            "earnings_quality": quality_ctx.get("earnings_quality"),
            "valuation_supports": list(quality_ctx.get("valuation_supports") or []),
            "valuation_headwinds": list(quality_ctx.get("valuation_headwinds") or []),
        },
        "peer_context": {
            "position": packet.peer_position,
            "roic_vs_median": packet.roic_vs_median,
            "operating_margin_vs_median": packet.op_margin_vs_median,
            "revenue_growth_vs_median": packet.revenue_growth_vs_median,
        },
        "quarterly_freshness": {
            "latest_quarterly_revenue": packet.latest_quarterly_revenue,
            "latest_quarterly_period": packet.latest_quarterly_period,
            "quarterly_revenue_trend": packet.quarterly_revenue_trend,
        },
        "method_tension": {
            "method_tension_type": packet.method_tension_type,
            "growth_dependency_ratio": packet.growth_dependency_ratio,
            "methods_agree": packet.methods_agree,
            "consensus_direction": packet.consensus_direction,
            "intrinsic_range_low": packet.intrinsic_range_low,
            "intrinsic_range_high": packet.intrinsic_range_high,
            "detail": method_tension,
        },
        "insurance_context": {
            "security_type": packet.security_type,
            "issuer_type": packet.issuer_type,
            "insurance_subtype": packet.insurance_subtype,
            "insurance_method": packet.insurance_method,
            "insurance_value": packet.insurance_value,
            "model_status": packet.model_status,
            "model_blockers": list(packet.model_blockers or []),
            "model_fit_warnings": list(packet.model_fit_warnings or []),
            "generic_valuation_valid": (
                packet.insurance_packet.get("generic_valuation_valid")
                if isinstance(packet.insurance_packet, dict)
                else None
            ),
            "operating_metrics": insurance_operating_metrics_payload,
            "peer_context": insurance_peer_context,
        },
        "usable_for_decision": True,
        "evidence_status": "KPI_TRENDS_AVAILABLE",
        "warnings": [],
        "summary": kpi_summary,
    }


def _fetch_insurance_evidence_packet(ctx: AlphaToolContext) -> dict[str, Any]:
    packet = ctx.packet.insurance_packet if isinstance(ctx.packet.insurance_packet, dict) else {}
    if not packet:
        return {
            "status": "unavailable",
            "ticker": ctx.ticker,
            "reason": "insurance_packet_not_present",
        }
    return {
        "status": "ok",
        "ticker": ctx.ticker,
        "insurance_packet": packet,
    }


def _compare_peer_metric(ctx: AlphaToolContext, *, metric: str) -> dict[str, Any]:
    requested_metric = str(metric or "").strip()
    metric_key = normalize_peer_metric(requested_metric)
    if metric_key not in SUPPORTED_COMPARISON_METRICS:
        return {
            "status": "unavailable",
            "ticker": ctx.ticker,
            "metric": requested_metric,
            "metric_key": metric_key,
            "reason": f"unsupported_peer_metric:{requested_metric or 'missing'}",
            "supported_metrics": sorted(SUPPORTED_PEER_METRICS),
            "usable_for_decision": False,
            "evidence_status": "UNSUPPORTED_PEER_METRIC",
            "warnings": [f"unsupported_peer_metric:{requested_metric or 'missing'}"],
            "summary": (
                f"Peer metric '{requested_metric}' is not supported by the deterministic peer comparison tool."
                if requested_metric
                else "Peer metric was missing, so no deterministic peer comparison was run."
            ),
        }
    if ctx.packet.issuer_type == ISSUER_INSURANCE_UNDERWRITER:
        peer = compute_insurance_subtype_peer_relative_metrics(ctx.ticker, ctx.as_of_date)
    else:
        peer = compute_peer_relative_metrics(ctx.ticker, ctx.as_of_date, sector=ctx.sector)
    metric_key_map = {
        "roic": "roic_vs_median",
        "operating_margin": "operating_margin_vs_median",
        "revenue_growth_5y": "revenue_growth_vs_median",
        "ev_ebitda": "ev_ebitda_vs_median",
        "ev_ebit": "ev_ebit_vs_median",
        "ev_sales": "ev_sales_vs_median",
        "p_e": "p_e_vs_median",
        "p_b": "p_b_vs_median",
        "fcf_yield": "fcf_yield_vs_median",
        "dividend_yield": "dividend_yield_vs_median",
    }
    ratio_key = metric_key_map.get(metric_key)
    relative_ratios = (
        peer.get("relative_ratios") if isinstance(peer.get("relative_ratios"), dict) else {}
    )
    ratio = relative_ratios.get(ratio_key) if ratio_key else None
    ticker_metrics = (
        peer.get("ticker_metrics") if isinstance(peer.get("ticker_metrics"), dict) else {}
    )
    sector_medians = (
        peer.get("sector_medians") if isinstance(peer.get("sector_medians"), dict) else {}
    )
    sector_q1 = peer.get("sector_q1") if isinstance(peer.get("sector_q1"), dict) else {}
    sector_q3 = peer.get("sector_q3") if isinstance(peer.get("sector_q3"), dict) else {}
    percentile_ranks = (
        peer.get("percentile_ranks") if isinstance(peer.get("percentile_ranks"), dict) else {}
    )
    stock_value = ticker_metrics.get(metric_key)
    sector_median = sector_medians.get(metric_key)
    usable = (
        peer.get("status") == "OK"
        and isinstance(stock_value, (int, float))
        and isinstance(sector_median, (int, float))
    )
    display_metric = PEER_METRIC_DISPLAY_NAMES.get(metric_key, requested_metric or metric_key)
    return {
        "status": "ok" if usable else "unavailable",
        "ticker": ctx.ticker,
        "metric": display_metric,
        "metric_key": metric_key,
        "sector": peer.get("sector"),
        "peer_scope": peer.get("peer_scope") or "broad_sector",
        "peer_group": peer.get("peer_group") or peer.get("sector"),
        "insurance_subtype": peer.get("insurance_subtype"),
        "fallback_reason": peer.get("fallback_reason"),
        "peer_status": peer.get("status"),
        "peer_count": peer.get("peer_count"),
        "relative_position": peer.get("relative_position"),
        "stock_value": stock_value,
        "sector_median": sector_median,
        "sector_q1": sector_q1.get(metric_key),
        "sector_q3": sector_q3.get(metric_key),
        "percentile_rank": percentile_ranks.get(metric_key),
        "peer_set_used": list(peer.get("peer_set_used") or []),
        "ticker_metrics": ticker_metrics,
        "sector_medians": sector_medians,
        "sector_q1_by_metric": sector_q1,
        "sector_q3_by_metric": sector_q3,
        "percentile_ranks": percentile_ranks,
        "ratio": ratio,
        "usable_for_decision": usable,
        "evidence_status": "PEER_METRIC_AVAILABLE" if usable else "PEER_METRIC_UNAVAILABLE",
        "warnings": [] if usable else ["peer_metric_ratio_unavailable"],
        "summary": (
            f"{display_metric} peer comparison available for {ctx.ticker}: stock {stock_value:.2f}, "
            f"sector median {sector_median:.2f}, percentile {percentile_ranks.get(metric_key):.1f}."
            if usable and isinstance(percentile_ranks.get(metric_key), (int, float))
            else f"{display_metric} peer comparison did not produce a decision-usable value."
        ),
    }


def _analyze_dilution(ctx: AlphaToolContext, *, years: int | None) -> dict[str, Any]:
    series = _companyfacts_series(
        ctx.ticker,
        line_items=["shares_outstanding"],
        years=int(years or 5),
        as_of_date=ctx.as_of_date,
    ).get("shares_outstanding", [])
    adjusted_series, split_adjustments, split_lineage_complete = _split_adjusted_share_series(
        series
    )
    raw_cagr = _shares_cagr(series)
    cagr = _shares_cagr(adjusted_series) if split_lineage_complete else None
    latest = series[-1]["value"] if series else None
    prior = series[0]["value"] if series else None
    adjusted_latest = adjusted_series[-1]["value"] if adjusted_series else None
    adjusted_prior = adjusted_series[0]["value"] if adjusted_series else None
    direction = "UNKNOWN"
    if cagr is not None:
        if cagr <= -0.01:
            direction = "BUYBACKS_OR_SHARE_REDUCTION"
        elif cagr <= 0.02:
            direction = "LOW_DILUTION"
        elif cagr <= 0.06:
            direction = "MODERATE_DILUTION"
        else:
            direction = "HIGH_DILUTION"
    usable = cagr is not None
    warnings = [] if usable else ["share_count_series_unavailable"]
    evidence_status = "DILUTION_SERIES_AVAILABLE" if usable else "NO_SHARE_COUNT_SERIES"
    if series and not split_lineage_complete:
        warnings = ["share_count_split_lineage_missing"]
        evidence_status = "SHARE_SPLIT_LINEAGE_MISSING"
    if split_adjustments:
        warnings.append("share_count_split_adjustment_applied")
        evidence_status = "DILUTION_SERIES_SPLIT_ADJUSTED"
    if usable and split_adjustments:
        summary = (
            f"{ctx.ticker} split-adjusted share-count CAGR {_pct_summary(cagr)} "
            f"over the available annual series; raw CAGR {_pct_summary(raw_cagr)} was adjusted "
            f"for probable stock-split artifact(s); dilution direction {direction}."
        )
    elif usable:
        summary = (
            f"{ctx.ticker} share-count CAGR {_pct_summary(cagr)} over the available annual series; "
            f"dilution direction {direction}."
        )
    elif series and not split_lineage_complete:
        summary = (
            f"{ctx.ticker} share-count history lacks exact split-basis lineage; "
            "dilution direction is unavailable."
        )
    else:
        summary = f"No annual share-count series was available for {ctx.ticker}; dilution direction is UNKNOWN."
    return {
        "status": "ok" if usable else "unavailable",
        "ticker": ctx.ticker,
        "as_of_date": ctx.as_of_date,
        "share_series": series,
        "split_adjusted_share_series": adjusted_series if split_adjustments else [],
        "share_count_split_adjustments": split_adjustments,
        "share_count_latest": latest,
        "share_count_oldest": prior,
        "share_count_latest_adjusted": adjusted_latest if split_adjustments else None,
        "share_count_oldest_adjusted": adjusted_prior if split_adjustments else None,
        "share_count_cagr_raw": raw_cagr if split_adjustments else None,
        "share_count_cagr": cagr,
        "dilution_direction": direction,
        "usable_for_decision": usable,
        "evidence_status": evidence_status,
        "warnings": warnings,
        "summary": summary,
    }


def _analyze_liquidity_stress(ctx: AlphaToolContext) -> dict[str, Any]:
    research = ctx.packet.research_report if isinstance(ctx.packet.research_report, dict) else {}
    solvency = research.get("solvency") if isinstance(research.get("solvency"), dict) else {}
    solvency_risk = ctx.packet.solvency_risk or solvency.get("risk")
    usable = bool(solvency_risk or solvency)
    return {
        "status": "ok" if usable else "unavailable",
        "ticker": ctx.ticker,
        "solvency_risk": ctx.packet.solvency_risk,
        "solvency": {
            "risk": solvency.get("risk"),
            "signals": list(solvency.get("signals") or []),
            "details": solvency.get("details"),
            "negative_equity": solvency.get("negative_equity"),
            "current_ratio": solvency.get("current_ratio"),
            "cash_runway_quarters": solvency.get("cash_runway_quarters"),
            "going_concern_language": solvency.get("going_concern_language"),
            "going_concern_asserted": going_concern_asserted(solvency),
            "no_assurance_financing": solvency.get("no_assurance_financing"),
            "debt_due_within_12mo": solvency.get("debt_due_within_12mo"),
        },
        "filing_risk_signals": dict(ctx.packet.filing_risk_signals or {}),
        "usable_for_decision": usable,
        "evidence_status": "LIQUIDITY_STRESS_AVAILABLE"
        if usable
        else "LIQUIDITY_STRESS_UNAVAILABLE",
        "warnings": [] if usable else ["liquidity_stress_context_unavailable"],
        "summary": (
            f"{ctx.ticker} liquidity/solvency risk is {_text_or_unknown(solvency_risk)}; "
            f"signals: {', '.join(str(item) for item in solvency.get('signals') or []) or 'none'}."
            if usable
            else f"No deterministic liquidity or solvency stress context was available for {ctx.ticker}."
        ),
    }


_CAPITAL_STRUCTURE_KEYWORDS = [
    "liquidity",
    "debt",
    "credit facility",
    "covenant",
    "default",
    "waiver",
    "amendment",
    "refinancing",
    "maturity",
    "going concern",
    "no assurance",
    "financing",
    "capital resources",
]

_CAPITAL_STRUCTURE_ACTIVE_PATTERNS = [
    r"substantial\s+doubt\s+about\s+.*going\s+concern",
    r"going\s+concern",
    r"event\s+of\s+default",
    r"in\s+default\s+under",
    r"not\s+in\s+compliance\s+with\s+.*covenant",
    r"failed\s+to\s+comply\s+with\s+.*covenant",
    r"no\s+assurance\s+(?:can\s+be\s+given|that\s+we\s+will).*(?:financ|capital|fund)",
    r"may\s+not\s+be\s+able\s+to\s+(?:raise|obtain|secure).*(?:financ|capital|fund)",
]

_CAPITAL_STRUCTURE_RESOLUTION_PATTERNS = [
    r"waiv(?:ed|er)",
    r"amend(?:ed|ment)",
    r"refinanced",
    r"completed\s+.*refinancing",
    r"repaid",
    r"entered\s+into\s+.*(?:credit|loan|financing)",
    r"completed\s+.*(?:financing|offering|refinancing)",
    r"in\s+compliance\s+with\s+.*covenant",
    r"no\s+(?:event\s+of\s+)?defaults?",
    r"no\s+covenant\s+defaults?",
]


def _compact_sentence(text: str, *, max_chars: int = 240) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def _capital_structure_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[\.\!\?])\s+|\n+", str(text or ""))
        if sentence.strip()
    ]


def _normalize_debt_amount(
    amount_text: str, unit_text: str | None
) -> tuple[float | None, str | None]:
    try:
        amount = float(str(amount_text or "").replace("$", "").replace(",", "").strip())
    except ValueError:
        return None, None
    unit = str(unit_text or "").strip().lower()
    if unit in {"m", "mm"}:
        unit = "million"
    elif unit in {"b", "bn"}:
        unit = "billion"
    elif unit == "%":
        unit = "percent"
    elif not unit:
        unit = "unspecified"
    return amount, unit


def _infer_table_amount_unit(text: str) -> str | None:
    compact = re.sub(r"\s+", " ", str(text or "").strip().lower())
    parenthetical_match = re.search(
        r"\(\s*(?:\$?\s*)?(?:amounts?\s+)?(?:in\s+)?(?P<unit>millions?|billions?|thousands?)\s*\)",
        compact,
    )
    text_match = parenthetical_match or re.search(
        r"\b(?:amounts?\s+)?(?:\$?\s*)?in\s+(?P<unit>millions?|billions?|thousands?)\b",
        compact,
    )
    if not text_match:
        return None
    unit = text_match.group("unit")
    if unit.startswith("million"):
        return "million"
    if unit.startswith("billion"):
        return "billion"
    if unit.startswith("thousand"):
        return "thousand"
    return None


def _normalize_maturity_period(value: str | None) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    text = text.replace("–", "-").replace("—", "-")
    if text.startswith("in "):
        text = text[3:]
    if text in {
        "less than one year",
        "within one year",
        "one year or less",
        "1 year or less",
        "within 12 months",
    }:
        return "less than one year"
    if text in {"one to three years", "1 to 3 years", "1-3 years", "one through three years"}:
        return "1-3 years"
    if text in {"three to five years", "3 to 5 years", "3-5 years", "three through five years"}:
        return "3-5 years"
    if text in {
        "more than five years",
        "over five years",
        "after five years",
        "greater than five years",
    }:
        return "more than five years"
    return text


_MATURITY_DATE_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+20\d{2}|20\d{2}[-/]\d{1,2}[-/]\d{1,2}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+20\d{2}"
)

_MATURITY_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _normalize_maturity_date(value: str | None) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "").replace(".", "").strip())
    text = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)
    numeric_match = re.fullmatch(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", text)
    if numeric_match:
        year = int(numeric_match.group(1))
        month = int(numeric_match.group(2))
        day = int(numeric_match.group(3))
    else:
        named_match = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(20\d{2})", text)
        named_month_match = re.fullmatch(r"([A-Za-z]+)\s+(20\d{2})", text)
        if named_match:
            month = _MATURITY_MONTHS.get(named_match.group(1).lower()[:3])
            day = int(named_match.group(2))
            year = int(named_match.group(3))
        elif named_month_match:
            month = _MATURITY_MONTHS.get(named_month_match.group(1).lower()[:3])
            day = 1
            year = int(named_month_match.group(2))
        else:
            return None
        if month is None:
            return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _extract_maturity_schedule(text: str, *, max_rows: int = 6) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, float | None, str]] = set()
    amount_date_pattern = re.compile(
        r"\$?\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?"
        r"(?:\s+(?:of\s+)?(?:principal|debt|borrowings|notes|senior\s+notes|term\s+loan|maturities))?"
        r"[^;\.\n]{0,80}?"
        r"(?:due|matur(?:e|es|ing|ity|ities)|repay(?:able|ment)?)"
        r"[^;\.\n]{0,40}?"
        rf"(?P<maturity_date>{_MATURITY_DATE_PATTERN})",
        flags=re.IGNORECASE,
    )
    date_amount_pattern = re.compile(
        rf"(?P<maturity_date>{_MATURITY_DATE_PATTERN})"
        r"[^;\.\n]{0,40}?"
        r"\$?\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?",
        flags=re.IGNORECASE,
    )
    amount_year_pattern = re.compile(
        r"\$?\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?"
        r"(?:\s+(?:of\s+)?(?:principal|debt|borrowings|notes|senior\s+notes|term\s+loan|maturities))?"
        r"[^;,\.\n]{0,45}?"
        r"(?:due\s+in|matur(?:e|es|ing)?\s+in|repay(?:able|ment)?\s+in|in|during)\s+"
        r"(?P<year>20\d{2})",
        flags=re.IGNORECASE,
    )
    amount_year_pair_pattern = re.compile(
        r"\$?\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?"
        r"(?:\s+(?:of\s+)?(?:principal|debt|borrowings|notes|senior\s+notes|term\s+loan|maturities))?"
        r"[^;,\.\n]{0,45}?"
        r"(?:due\s+in|matur(?:e|es|ing)?\s+in|repay(?:able|ment)?\s+in|in|during)\s+"
        r"(?P<start_year>20\d{2})\s*(?:and|&)\s*(?P<end_year>20\d{2})",
        flags=re.IGNORECASE,
    )
    pair_pattern = re.compile(
        r"(?:\$?\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*(?P<unit>million|billion|thousand|mm|m|bn|b)?"
        r"[^\.]{0,90}?"
        r"(?:due|matur(?:e|es|ing|ity|ities)|repay(?:able|ment)?|principal)"
        r"[^\.]{0,60}?"
        r"(?P<year>20\d{2}))",
        flags=re.IGNORECASE,
    )
    year_amount_pattern = re.compile(
        r"(?P<year>20\d{2})"
        r"[^\.]{0,40}?"
        r"\$?\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?",
        flags=re.IGNORECASE,
    )
    year_range_amount_pattern = re.compile(
        r"(?P<start_year>20\d{2})\s*(?:-|–|—|to|through|thru)\s*(?P<end_year>20\d{2})"
        r"[^\.]{0,40}?"
        r"\$?\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?",
        flags=re.IGNORECASE,
    )
    year_pair_amount_pattern = re.compile(
        r"(?P<start_year>20\d{2})\s*(?:and|&)\s*(?P<end_year>20\d{2})"
        r"[^\.]{0,40}?"
        r"\$?\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?",
        flags=re.IGNORECASE,
    )
    year_and_thereafter_amount_pattern = re.compile(
        r"(?P<year>20\d{2})\s+(?:and\s+)?thereafter"
        r"[^\.]{0,40}?"
        r"\$?\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?",
        flags=re.IGNORECASE,
    )
    duration_amount_pattern = re.compile(
        r"(?P<period>less\s+than\s+one\s+year|within\s+one\s+year|one\s+year\s+or\s+less|"
        r"1\s+year\s+or\s+less|within\s+12\s+months|one\s+to\s+three\s+years|"
        r"1\s*(?:-|–|—|to)\s*3\s+years|one\s+through\s+three\s+years|"
        r"three\s+to\s+five\s+years|3\s*(?:-|–|—|to)\s*5\s+years|"
        r"three\s+through\s+five\s+years|more\s+than\s+five\s+years|"
        r"over\s+five\s+years|after\s+five\s+years|greater\s+than\s+five\s+years)"
        r"[^;,\.\n]{0,40}?"
        r"\$?\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?",
        flags=re.IGNORECASE,
    )
    amount_duration_pattern = re.compile(
        r"\$?\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?"
        r"(?:\s+(?:of\s+)?(?:principal|debt|borrowings|notes|maturities))?"
        r"[^;,\.\n]{0,35}?"
        r"(?P<period>less\s+than\s+one\s+year|within\s+one\s+year|one\s+year\s+or\s+less|"
        r"1\s+year\s+or\s+less|within\s+12\s+months|(?:in\s+)?one\s+to\s+three\s+years|"
        r"(?:in\s+)?1\s*(?:-|–|—|to)\s*3\s+years|(?:in\s+)?one\s+through\s+three\s+years|"
        r"(?:in\s+)?three\s+to\s+five\s+years|(?:in\s+)?3\s*(?:-|–|—|to)\s*5\s+years|"
        r"(?:in\s+)?three\s+through\s+five\s+years|(?:in\s+)?more\s+than\s+five\s+years|"
        r"(?:in\s+)?over\s+five\s+years|(?:in\s+)?after\s+five\s+years|"
        r"(?:in\s+)?greater\s+than\s+five\s+years)",
        flags=re.IGNORECASE,
    )
    thereafter_amount_pattern = re.compile(
        r"(?:thereafter|after\s+20\d{2}|subsequent\s+to\s+20\d{2})"
        r"[^\.]{0,40}?"
        r"\$?\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?",
        flags=re.IGNORECASE,
    )
    amount_thereafter_pattern = re.compile(
        r"\$?\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?"
        r"[^;,\.\n]{0,40}?"
        r"(?:thereafter|after\s+20\d{2}|subsequent\s+to\s+20\d{2})",
        flags=re.IGNORECASE,
    )

    def append_thereafter_rows(sentence: str) -> bool:
        appended = False
        for pattern in (thereafter_amount_pattern, amount_thereafter_pattern):
            for match in pattern.finditer(sentence):
                prefix = sentence[max(0, match.start() - 15) : match.start()].lower()
                if re.search(r"20\d{2}\s+(?:and\s+)?$", prefix):
                    continue
                if pattern is amount_thereafter_pattern and re.fullmatch(
                    r"20\d{2}", str(match.group("amount") or "")
                ):
                    continue
                amount, unit = normalize_maturity_amount(match, sentence)
                if amount is None:
                    continue
                key = ("thereafter", amount, sentence)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "year": None,
                        "period": "thereafter",
                        "amount": amount,
                        "unit": unit,
                        "context": _compact_sentence(sentence),
                    }
                )
                appended = True
                if len(rows) >= max_rows:
                    return appended
        return appended

    def normalize_maturity_amount(
        match: re.Match[str], sentence: str
    ) -> tuple[float | None, str | None]:
        amount, unit = _normalize_debt_amount(match.group("amount"), match.group("unit"))
        if amount is not None and unit == "unspecified":
            unit = _infer_table_amount_unit(sentence) or unit
        return amount, unit

    def append_year_pair(match: re.Match[str], sentence: str) -> bool:
        amount, unit = normalize_maturity_amount(match, sentence)
        if amount is None:
            return False
        start_year = int(match.group("start_year"))
        end_year = int(match.group("end_year"))
        period = f"{start_year} and {end_year}"
        key = (period, amount, sentence)
        if key in seen:
            return False
        seen.add(key)
        rows.append(
            {
                "year": None,
                "period": period,
                "start_year": start_year,
                "end_year": end_year,
                "amount": amount,
                "unit": unit,
                "context": _compact_sentence(sentence),
            }
        )
        return True

    year_pattern = re.compile(r"\b(20\d{2})\b")
    for sentence in _capital_structure_sentences(text):
        sentence_lower = sentence.lower()
        has_maturity_language = any(
            term in sentence_lower for term in ("matur", "due", "repay", "principal")
        )
        has_debt_payment_schedule = any(
            term in sentence_lower
            for term in (
                "debt payments",
                "debt payment",
                "scheduled payments",
                "contractual obligations",
                "long-term debt payments",
                "long term debt payments",
            )
        )
        if not has_maturity_language and not has_debt_payment_schedule:
            continue
        matched = False
        for pattern in (amount_date_pattern, date_amount_pattern):
            for match in pattern.finditer(sentence):
                maturity_date = _normalize_maturity_date(match.group("maturity_date"))
                if maturity_date is None:
                    continue
                amount, unit = normalize_maturity_amount(match, sentence)
                if amount is None:
                    continue
                key = (maturity_date, amount, sentence)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "year": int(maturity_date[:4]),
                        "maturity_date": maturity_date,
                        "amount": amount,
                        "unit": unit,
                        "context": _compact_sentence(sentence),
                    }
                )
                matched = True
                if len(rows) >= max_rows:
                    return rows
        if matched:
            continue
        matched_amount_year = False
        for match in amount_year_pair_pattern.finditer(sentence):
            matched_amount_year = append_year_pair(match, sentence) or matched_amount_year
            matched = matched_amount_year or matched
            if len(rows) >= max_rows:
                return rows
        for match in amount_year_pattern.finditer(sentence):
            suffix = sentence[match.end() : match.end() + 12]
            if re.match(r"\s*(?:and|&)\s*20\d{2}", suffix, flags=re.IGNORECASE):
                continue
            amount, unit = normalize_maturity_amount(match, sentence)
            if amount is None:
                continue
            key = (int(match.group("year")), amount, sentence)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "year": int(match.group("year")),
                    "amount": amount,
                    "unit": unit,
                    "context": _compact_sentence(sentence),
                }
            )
            matched = True
            matched_amount_year = True
            if len(rows) >= max_rows:
                return rows
        if matched_amount_year:
            matched = append_thereafter_rows(sentence) or matched
            if len(rows) >= max_rows:
                return rows
            continue
        for match in pair_pattern.finditer(sentence):
            amount, unit = normalize_maturity_amount(match, sentence)
            if amount is None:
                continue
            key = (int(match.group("year")), amount, sentence)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "year": int(match.group("year")),
                    "amount": amount,
                    "unit": unit,
                    "context": _compact_sentence(sentence),
                }
            )
            matched = True
            if len(rows) >= max_rows:
                return rows
        if matched:
            continue
        for match in year_pair_amount_pattern.finditer(sentence):
            matched = append_year_pair(match, sentence) or matched
            if len(rows) >= max_rows:
                return rows
        for match in year_amount_pattern.finditer(sentence):
            if "thereafter" in match.group(0).lower() or re.search(
                r"20\d{2}\s*(?:-|–|—|to|through|thru|and|&)\s*20\d{2}",
                match.group(0),
                re.IGNORECASE,
            ):
                continue
            amount, unit = normalize_maturity_amount(match, sentence)
            if amount is None:
                continue
            key = (int(match.group("year")), amount, sentence)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "year": int(match.group("year")),
                    "amount": amount,
                    "unit": unit,
                    "context": _compact_sentence(sentence),
                }
            )
            matched = True
            if len(rows) >= max_rows:
                return rows
        for match in year_range_amount_pattern.finditer(sentence):
            amount, unit = normalize_maturity_amount(match, sentence)
            if amount is None:
                continue
            start_year = int(match.group("start_year"))
            end_year = int(match.group("end_year"))
            period = f"{start_year}-{end_year}"
            key = (period, amount, sentence)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "year": None,
                    "period": period,
                    "start_year": start_year,
                    "end_year": end_year,
                    "amount": amount,
                    "unit": unit,
                    "context": _compact_sentence(sentence),
                }
            )
            matched = True
            if len(rows) >= max_rows:
                return rows
        for match in year_and_thereafter_amount_pattern.finditer(sentence):
            amount, unit = normalize_maturity_amount(match, sentence)
            if amount is None:
                continue
            year = int(match.group("year"))
            period = f"{year} and thereafter"
            key = (period, amount, sentence)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "year": None,
                    "period": period,
                    "amount": amount,
                    "unit": unit,
                    "context": _compact_sentence(sentence),
                }
            )
            matched = True
            if len(rows) >= max_rows:
                return rows
        for match in duration_amount_pattern.finditer(sentence):
            amount, unit = normalize_maturity_amount(match, sentence)
            if amount is None:
                continue
            period = _normalize_maturity_period(match.group("period"))
            key = (period, amount, sentence)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "year": None,
                    "period": period,
                    "amount": amount,
                    "unit": unit,
                    "context": _compact_sentence(sentence),
                }
            )
            matched = True
            if len(rows) >= max_rows:
                return rows
        for match in amount_duration_pattern.finditer(sentence):
            amount, unit = normalize_maturity_amount(match, sentence)
            if amount is None:
                continue
            period = _normalize_maturity_period(match.group("period"))
            key = (period, amount, sentence)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "year": None,
                    "period": period,
                    "amount": amount,
                    "unit": unit,
                    "context": _compact_sentence(sentence),
                }
            )
            matched = True
            if len(rows) >= max_rows:
                return rows
        matched = append_thereafter_rows(sentence) or matched
        if len(rows) >= max_rows:
            return rows
        if matched:
            continue
        for year_match in year_pattern.finditer(sentence):
            key = (int(year_match.group(1)), None, sentence)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "year": int(year_match.group(1)),
                    "amount": None,
                    "unit": None,
                    "context": _compact_sentence(sentence),
                }
            )
            if len(rows) >= max_rows:
                return rows
    return rows


def _extract_covenant_status(text: str) -> tuple[str, str | None]:
    covenant_sentences = [
        sentence
        for sentence in _capital_structure_sentences(text)
        if any(term in sentence.lower() for term in ("covenant", "default", "compliance", "waiver"))
    ]
    evidence = _compact_sentence(covenant_sentences[0]) if covenant_sentences else None
    covenant_text = " ".join(covenant_sentences)
    if _matches_any_pattern(covenant_text, _CAPITAL_STRUCTURE_ACTIVE_PATTERNS):
        return "DEFAULT_OR_NONCOMPLIANCE_EVIDENCED", evidence
    if _matches_any_pattern(covenant_text, _CAPITAL_STRUCTURE_RESOLUTION_PATTERNS):
        return "COMPLIANCE_EVIDENCED", evidence
    if covenant_sentences:
        return "COVENANT_MENTIONED", evidence
    return "NOT_FOUND", None


def _normalize_covenant_condition(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text in {">=", "≥"}:
        return "MINIMUM"
    if text in {"<=", "≤"}:
        return "MAXIMUM"
    if any(
        token in text
        for token in (
            "minimum",
            "at least",
            "not less than",
            "no less than",
            "greater than or equal to",
            "equal to or greater than",
        )
    ):
        return "MINIMUM"
    if any(
        token in text
        for token in (
            "maximum",
            "not to exceed",
            "not exceed",
            "no greater than",
            "no more than",
            "not greater than",
            "less than or equal to",
            "equal to or less than",
        )
    ):
        return "MAXIMUM"
    if "greater than" in text:
        return "MINIMUM"
    if "less than" in text:
        return "MAXIMUM"
    return "UNSPECIFIED"


def _extract_covenant_terms(text: str, *, max_rows: int = 6) -> list[dict[str, Any]]:
    covenant_condition_terms = (
        r">=|<=|≥|≤|"
        r"greater\s+than\s+or\s+equal\s+to|equal\s+to\s+or\s+greater\s+than|"
        r"less\s+than\s+or\s+equal\s+to|equal\s+to\s+or\s+less\s+than|"
        r"maximum|minimum|not\s+to\s+exceed|not\s+exceed|no\s+greater\s+than|no\s+more\s+than|"
        r"not\s+greater\s+than|not\s+less\s+than|no\s+less\s+than|"
        r"at\s+least|greater\s+than|less\s+than"
    )
    monetary_condition_terms = covenant_condition_terms
    liquidity_metric_terms = (
        r"liquidity|cash|availability|unused\s+availability|cash\s+and\s+availability"
    )
    monetary_metric_terms = (
        r"(?:consolidated\s+)?(?:tangible\s+)?net\s+worth|"
        r"shareholders'? equity|stockholders'? equity|"
        r"(?:consolidated\s+)?(?:adjusted\s+)?ebitda|"
        r"(?:annual\s+)?capital\s+expenditures?|(?:annual\s+)?capex"
    )
    ratio_pattern = re.compile(
        rf"(?P<condition_before>{covenant_condition_terms})?"
        r"\s*(?P<metric>(?:(?:consolidated|total|senior\s+secured|senior|secured|unsecured|first\s+lien)\s+)*(?:net\s+)?leverage\s+ratio|"
        r"debt\s+to\s+ebitda|(?:unencumbered\s+)?fixed\s+charge\s+coverage\s+ratio|(?:unencumbered\s+)?interest\s+coverage\s+ratio|"
        r"debt\s+service\s+coverage\s+ratio|asset\s+coverage\s+ratio|"
        r"(?:total\s+)?debt\s+to\s+capital(?:ization)?\s+ratio|current\s+ratio|"
        r"(?:secured|unsecured)\s+debt\s+to\s+(?:total\s+)?assets?\s+ratio|"
        r"unsecured\s+debt\s+to\s+unencumbered\s+assets?\s+ratio|"
        r"loan[-\s]+to[-\s]+value\s+ratio|ltv\s+ratio|(?:unencumbered\s+)?debt\s+yield)"
        rf"(?:[^\.]{{0,40}}?(?P<condition_after>{covenant_condition_terms}))?"
        r"[^\.]{0,80}?"
        r"(?P<threshold>\d+(?:\.\d+)?)\s*(?P<unit>x|to\s+1(?:\.00)?|:1|%|percent)?",
        flags=re.IGNORECASE,
    )
    liquidity_pattern = re.compile(
        r"(?P<condition>minimum|at\s+least|greater\s+than|not\s+less\s+than)\s+"
        rf"(?P<metric>{liquidity_metric_terms})"
        r"[^\.]{0,50}?\$?\s*(?P<threshold>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?",
        flags=re.IGNORECASE,
    )
    liquidity_after_pattern = re.compile(
        rf"(?P<metric>{liquidity_metric_terms})"
        rf"[^;\.\d$]{{0,50}}?(?P<condition>{monetary_condition_terms})"
        r"[^\.]{0,50}?\$?\s*(?P<threshold>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?",
        flags=re.IGNORECASE,
    )
    monetary_before_pattern = re.compile(
        rf"(?P<condition>{monetary_condition_terms})\s+"
        rf"(?P<metric>{monetary_metric_terms})"
        r"[^\.]{0,50}?\$?\s*(?P<threshold>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?",
        flags=re.IGNORECASE,
    )
    monetary_after_pattern = re.compile(
        rf"(?P<metric>{monetary_metric_terms})"
        rf"[^;\.\d$]{{0,50}}?(?P<condition>{monetary_condition_terms})"
        r"[^\.]{0,30}?\$?\s*(?P<threshold>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(?P<unit>million|billion|thousand|mm|m|bn|b)?",
        flags=re.IGNORECASE,
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float, str]] = set()
    for sentence in _capital_structure_sentences(text):
        sentence_lower = sentence.lower()
        if not any(
            term in sentence_lower
            for term in (
                "covenant",
                "ratio",
                "loan-to-value",
                "loan to value",
                "ltv",
                "debt yield",
                "liquidity",
                "availability",
                "net worth",
                "equity",
                "ebitda",
                "capital expenditure",
                "capex",
            )
        ):
            continue
        for pattern in (
            ratio_pattern,
            liquidity_pattern,
            liquidity_after_pattern,
            monetary_before_pattern,
            monetary_after_pattern,
        ):
            for match in pattern.finditer(sentence):
                metric = re.sub(r"\s+", " ", match.group("metric")).strip().lower()
                threshold, raw_unit = _normalize_debt_amount(
                    match.group("threshold"), match.group("unit")
                )
                if threshold is None:
                    continue
                unit = (
                    "percent"
                    if pattern is ratio_pattern and raw_unit == "percent"
                    else "ratio"
                    if pattern is ratio_pattern
                    else raw_unit
                )
                condition = (
                    match.groupdict().get("condition_before")
                    or match.groupdict().get("condition_after")
                    or match.groupdict().get("condition")
                )
                key = (metric, threshold, sentence)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "metric": metric,
                        "condition": _normalize_covenant_condition(condition),
                        "threshold": threshold,
                        "unit": unit,
                        "context": _compact_sentence(sentence),
                    }
                )
                if len(rows) >= max_rows:
                    return rows
    return rows


def _extract_capital_structure_terms(text: str) -> dict[str, Any]:
    maturity_schedule = _extract_maturity_schedule(text)
    covenant_status, covenant_evidence = _extract_covenant_status(text)
    covenant_terms = _extract_covenant_terms(text)
    extracted = bool(maturity_schedule) or covenant_status != "NOT_FOUND" or bool(covenant_terms)
    return {
        "extraction_status": "STRUCTURED_TERMS_EXTRACTED"
        if extracted
        else "NO_STRUCTURED_TERMS_EXTRACTED",
        "maturity_schedule_status": (
            "MATURITY_SCHEDULE_EXTRACTED" if maturity_schedule else "NO_MATURITY_SCHEDULE_EXTRACTED"
        ),
        "maturity_schedule": maturity_schedule,
        "covenant_status": covenant_status,
        "covenant_evidence": covenant_evidence,
        "covenant_terms_status": "COVENANT_TERMS_EXTRACTED"
        if covenant_terms
        else "NO_COVENANT_TERMS_EXTRACTED",
        "covenant_terms": covenant_terms,
    }


def _latest_companyfact_value(
    series: dict[str, list[dict[str, Any]]], line_item: str
) -> float | None:
    values = series.get(line_item) or []
    if not values:
        return None
    value = values[-1].get("value")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _latest_companyfact_row(
    series: dict[str, list[dict[str, Any]]],
    line_item: str,
) -> dict[str, Any] | None:
    values = series.get(line_item) or []
    return dict(values[-1]) if values and isinstance(values[-1], dict) else None


def _companyfact_input_trace(row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {
            "value": None,
            "unit": None,
            "period_end": None,
            "filed_date": None,
            "source": None,
            "source_reference": None,
        }
    return {
        "value": row.get("value"),
        "unit": row.get("unit") or row.get("units"),
        "period_end": row.get("period_end"),
        "filed_date": row.get("filed_date"),
        "source": row.get("source"),
        "source_reference": row.get("source_reference"),
    }


def _capital_structure_filing_evidence(
    ctx: AlphaToolContext, *, max_chars: int
) -> tuple[list[dict[str, Any]], str]:
    try:
        filing_context = load_research_filing_context(
            ctx.ticker,
            as_of_date=ctx.as_of_date,
            quarters=1,
            include_material_events=True,
            material_event_window_days=730,
            material_event_limit=8,
        )
    except Exception:
        return [], "FILING_CONTEXT_UNAVAILABLE"

    documents: list[dict[str, Any]] = []
    for document in filing_context.ordered_documents[:8]:
        clean_text = strip_html(getattr(document, "html", "") or "")
        excerpt, source_strategy, noise_filtered, used_noisy = _extract_readable_filing_passages(
            clean_text,
            keywords=_CAPITAL_STRUCTURE_KEYWORDS,
            section_focus="liquidity risk capital allocation material_event",
            max_chars=max_chars,
        )
        readable = (
            bool(excerpt.strip()) and not used_noisy and not _is_noisy_filing_excerpt(excerpt)
        )
        documents.append(
            {
                "form_type": getattr(document, "form_type", None),
                "filing_date": getattr(document, "filing_date", None),
                "accession": getattr(document, "accession", None),
                "role": getattr(document, "role", None),
                "source_strategy": source_strategy,
                "excerpt": excerpt,
                "excerpt_chars": len(excerpt),
                "noise_filtered_passages": noise_filtered,
                "readable_for_decision": readable,
                "materialized_from": getattr(document, "materialized_from", None),
            }
        )

    status = str(getattr(filing_context, "recent_filing_status", "") or "UNKNOWN")
    return documents, status


def _matches_any_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


def _analyze_capital_structure_resolution(
    ctx: AlphaToolContext, *, max_chars: int | None = None
) -> dict[str, Any]:
    research = ctx.packet.research_report if isinstance(ctx.packet.research_report, dict) else {}
    solvency = research.get("solvency") if isinstance(research.get("solvency"), dict) else {}
    solvency_risk = str(ctx.packet.solvency_risk or solvency.get("risk") or "UNKNOWN").upper()
    signals = [str(item) for item in solvency.get("signals") or []]

    facts = _companyfacts_series(
        ctx.ticker,
        line_items=["cash", "total_debt", "current_assets", "current_liabilities", "cfo"],
        years=2,
        as_of_date=ctx.as_of_date,
    )
    cash = _latest_companyfact_value(facts, "cash")
    debt = _latest_companyfact_value(facts, "total_debt")
    current_assets = _latest_companyfact_value(facts, "current_assets")
    current_liabilities = _latest_companyfact_value(facts, "current_liabilities")
    cfo = _latest_companyfact_value(facts, "cfo")
    cash_row = _latest_companyfact_row(facts, "cash")
    debt_row = _latest_companyfact_row(facts, "total_debt")
    balance_sheet_complete = debt is not None and cash is not None
    net_debt = debt - cash if balance_sheet_complete else None
    current_ratio = (
        current_assets / current_liabilities
        if current_assets is not None and current_liabilities not in {None, 0}
        else solvency.get("current_ratio")
    )

    char_limit = max(500, min(int(max_chars if max_chars is not None else 1400), 5000))
    filing_documents, filing_status = _capital_structure_filing_evidence(ctx, max_chars=char_limit)
    readable_documents = [doc for doc in filing_documents if doc.get("readable_for_decision")]
    combined_filing_text = "\n\n".join(str(doc.get("excerpt") or "") for doc in readable_documents)
    active_language = _matches_any_pattern(combined_filing_text, _CAPITAL_STRUCTURE_ACTIVE_PATTERNS)
    resolution_language = _matches_any_pattern(
        combined_filing_text, _CAPITAL_STRUCTURE_RESOLUTION_PATTERNS
    )
    capital_structure_terms = _extract_capital_structure_terms(combined_filing_text)

    # Going concern here means a stored, blockable filed assertion, not the bare flag.
    going_concern = going_concern_asserted(solvency)
    no_assurance = bool(solvency.get("no_assurance_financing"))
    debt_due = bool(solvency.get("debt_due_within_12mo"))
    critical = solvency_risk == "CRITICAL"
    elevated = solvency_risk == "ELEVATED"
    net_cash = isinstance(net_debt, (int, float)) and net_debt < 0
    ample_cash_against_debt = (
        cash is not None
        and debt is not None
        and cash >= debt
        and not going_concern
        and not critical
    )

    warnings: list[str] = []
    if not readable_documents:
        warnings.append("capital_structure_filing_context_unreadable")
    companyfacts_unavailable = all(not values for values in facts.values())
    if companyfacts_unavailable:
        warnings.append("capital_structure_companyfacts_unavailable")
    else:
        if debt is None:
            warnings.append("capital_structure_total_debt_unavailable")
        if cash is None:
            warnings.append("capital_structure_cash_unavailable")

    source_strategy = (
        ", ".join(
            dict.fromkeys(
                str(doc.get("source_strategy") or "")
                for doc in readable_documents
                if doc.get("source_strategy")
            )
        )
        or "none"
    )
    filing_summary = f"source_strategy={source_strategy}, recent_filing_status={filing_status}"

    if critical or going_concern or (no_assurance and not resolution_language) or active_language:
        evidence_status = "CAPITAL_STRUCTURE_ACTIVE_DISTRESS"
        usable = True
        status = "ok"
        summary = (
            f"{ctx.ticker} capital-structure evidence indicates active distress or unresolved financing risk; "
            f"solvency risk {solvency_risk}, signals {', '.join(signals) or 'none'} ({filing_summary})."
        )
    elif resolution_language and (no_assurance or debt_due or elevated):
        evidence_status = "CAPITAL_STRUCTURE_RESOLVED_CLEAR"
        usable = True
        status = "ok"
        summary = (
            f"{ctx.ticker} capital-structure evidence found waiver/refinancing/compliance language that resolves "
            f"the prior financing flag; solvency risk {solvency_risk} ({filing_summary})."
        )
    elif (
        readable_documents
        and capital_structure_terms["extraction_status"] == "STRUCTURED_TERMS_EXTRACTED"
    ):
        evidence_status = (
            "CAPITAL_STRUCTURE_RESOLVED_CLEAR"
            if capital_structure_terms["covenant_status"] == "COMPLIANCE_EVIDENCED"
            else "CAPITAL_STRUCTURE_RESOLVED_WATCHLIST"
        )
        usable = True
        status = "ok"
        summary = (
            (
                f"{ctx.ticker} capital structure appears clear: solvency risk "
                f"LOW, net debt {_text_or_unknown(net_debt)}, signals "
                f"{', '.join(signals) or 'none'} ({filing_summary})."
            )
            if balance_sheet_complete and solvency_risk == "LOW"
            else (
                f"{ctx.ticker} has decision-usable filing evidence for debt "
                f"maturities or covenant terms; solvency risk {solvency_risk} "
                f"({filing_summary})."
            )
        )
    elif (debt_due or elevated) and readable_documents:
        evidence_status = "CAPITAL_STRUCTURE_RESOLVED_WATCHLIST"
        usable = True
        status = "ok"
        summary = (
            f"{ctx.ticker} has decision-usable capital-structure evidence but near-term debt/liquidity risk "
            f"remains a watchlist cap; solvency risk {solvency_risk}, signals {', '.join(signals) or 'none'} "
            f"({filing_summary})."
        )
    elif (
        solvency_risk == "LOW"
        and balance_sheet_complete
        and (readable_documents or any(facts.values()))
        and (net_cash or ample_cash_against_debt or not signals)
    ):
        evidence_status = "CAPITAL_STRUCTURE_RESOLVED_CLEAR"
        usable = True
        status = "ok"
        summary = (
            f"{ctx.ticker} capital structure appears clear: solvency risk LOW, "
            f"net debt {_text_or_unknown(net_debt)}, signals {', '.join(signals) or 'none'} ({filing_summary})."
        )
    else:
        evidence_status = "CAPITAL_STRUCTURE_UNRESOLVED"
        usable = False
        status = "unavailable"
        summary = (
            f"{ctx.ticker} capital-structure resolution is unresolved; deterministic facts or readable filing "
            f"context were insufficient ({filing_summary})."
        )

    return {
        "status": status,
        "ticker": ctx.ticker,
        "as_of_date": ctx.as_of_date,
        "usable_for_decision": usable,
        "evidence_status": evidence_status,
        "warnings": warnings,
        "summary": summary,
        "source_strategy": source_strategy,
        "recent_filing_status": filing_status,
        "solvency_risk": solvency_risk,
        "signals": signals,
        "liquidity": {
            "cash": cash,
            "total_debt": debt,
            "net_debt": net_debt,
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "current_ratio": current_ratio,
            "cfo": cfo,
            "provenance": {
                line_item: _companyfact_input_trace(_latest_companyfact_row(facts, line_item))
                for line_item in (
                    "cash",
                    "total_debt",
                    "current_assets",
                    "current_liabilities",
                    "cfo",
                )
            },
            "net_debt_trace": {
                "status": "OK" if balance_sheet_complete else "NEEDS_DATA",
                "formula": "total_debt - cash",
                "value": net_debt,
                "unit": (
                    debt_row.get("unit") or debt_row.get("units")
                    if isinstance(debt_row, dict)
                    else None
                ),
                "inputs": {
                    "total_debt": _companyfact_input_trace(debt_row),
                    "cash": _companyfact_input_trace(cash_row),
                },
            },
        },
        "stress_flags": {
            "going_concern_language": going_concern,
            "going_concern_asserted": going_concern,
            "no_assurance_language": no_assurance,
            "debt_due_within_12mo": debt_due,
            "active_distress_language_found": active_language,
            "waiver_refinancing_or_compliance_language_found": resolution_language,
        },
        "capital_structure_terms": capital_structure_terms,
        "maturity_schedule": capital_structure_terms["maturity_schedule"],
        "covenant_status": capital_structure_terms["covenant_status"],
        "covenant_terms": capital_structure_terms["covenant_terms"],
        "filing_documents": filing_documents,
    }


def _analyze_capital_allocation(ctx: AlphaToolContext) -> dict[str, Any]:
    quality_ctx = ctx.packet.raw_quality_ctx if isinstance(ctx.packet.raw_quality_ctx, dict) else {}
    dilution = _analyze_dilution(ctx, years=5)
    supports = list(ctx.packet.valuation_supports or [])
    headwinds = list(ctx.packet.valuation_headwinds or [])
    usable = bool(
        supports
        or headwinds
        or quality_ctx.get("earnings_quality")
        or quality_ctx.get("revenue_cagr_5y") is not None
        or dilution.get("usable_for_decision")
    )
    return {
        "status": "ok" if usable else "unavailable",
        "ticker": ctx.ticker,
        "valuation_supports": supports,
        "valuation_headwinds": headwinds,
        "earnings_quality": quality_ctx.get("earnings_quality"),
        "revenue_cagr_5y": quality_ctx.get("revenue_cagr_5y"),
        "revenue_cagr_3y": quality_ctx.get("revenue_cagr_3y"),
        "share_count_analysis": dilution,
        "usable_for_decision": usable,
        "evidence_status": "CAPITAL_ALLOCATION_CONTEXT_AVAILABLE"
        if usable
        else "CAPITAL_ALLOCATION_CONTEXT_UNAVAILABLE",
        "warnings": [] if usable else ["capital_allocation_context_unavailable"],
        "summary": (
            f"{ctx.ticker} capital allocation context: earnings quality "
            f"{_text_or_unknown(quality_ctx.get('earnings_quality'))}, 5Y revenue CAGR "
            f"{_pct_summary(quality_ctx.get('revenue_cagr_5y'))}, dilution "
            f"{_text_or_unknown(dilution.get('dilution_direction'))}, supports "
            f"{', '.join(supports) if supports else 'none'}, headwinds "
            f"{', '.join(headwinds) if headwinds else 'none'}."
            if usable
            else f"No deterministic capital-allocation context was available for {ctx.ticker}."
        ),
    }


def _fetch_current_events(ctx: AlphaToolContext, *, max_items: int | None) -> dict[str, Any]:
    event_context = load_current_event_context(ctx.ticker, as_of_date=ctx.as_of_date)
    limit = max(1, min(int(max_items or 5), 10))
    documents = []
    for document in event_context.ordered_documents[:limit]:
        documents.append(
            {
                "ticker": document.ticker,
                "source_type": document.source_type,
                "published_at": document.published_at,
                "title": document.title,
                "source_url": document.source_url,
                "summary": document.summary,
                "source_quality": document.source_quality,
            }
        )
    usable = bool(documents)
    metadata_source = str(getattr(event_context, "metadata_source", "unknown") or "unknown")
    evidence_status = "CURRENT_EVENTS_AVAILABLE" if usable else "NO_CURRENT_EVENTS"
    if not usable and metadata_source == "missing":
        evidence_status = "CURRENT_EVENT_METADATA_MISSING"
    homepage_present = bool(getattr(event_context, "homepage_url_present", False))
    ir_present = bool(getattr(event_context, "ir_rss_url_present", False))
    allowlist_domains = list(getattr(event_context, "allowlist_domains", []) or [])
    allowlist_source = str(getattr(event_context, "allowlist_source", "none") or "none")
    source_summary = (
        f"metadata={metadata_source}, homepage={'present' if homepage_present else 'missing'}, "
        f"ir_rss={'present' if ir_present else 'missing'}, allowlist={allowlist_source}"
    )
    has_explicit_diagnostics = (
        metadata_source != "unknown" or homepage_present or ir_present or bool(allowlist_domains)
    )
    return {
        "status": "ok" if usable else "unavailable",
        "ticker": ctx.ticker,
        "documents": documents,
        "document_count": len(documents),
        "warnings": list(event_context.warnings),
        "source_warnings": list(event_context.warnings),
        "metadata_source": metadata_source,
        "homepage_url_present": homepage_present,
        "ir_rss_url_present": ir_present,
        "allowlist_domains": allowlist_domains,
        "allowlist_source": allowlist_source,
        "usable_for_decision": usable,
        "evidence_status": evidence_status,
        "summary": (
            (
                f"{len(documents)} current-event document(s) available from configured company-controlled sources "
                f"({source_summary})."
            )
            if usable
            else (
                (
                    "No current-event documents were available from configured company-controlled sources "
                    f"({source_summary})."
                )
                if has_explicit_diagnostics
                else "No current-event documents were available from configured company-controlled sources."
            )
        ),
    }


def _date_age_days(as_of_date: str, filing_date: str | None) -> int | None:
    if not filing_date:
        return None
    try:
        return (date.fromisoformat(as_of_date) - date.fromisoformat(str(filing_date))).days
    except ValueError:
        return None


def _fetch_recent_filing_context(
    ctx: AlphaToolContext,
    *,
    quarters: int | None,
    material_event_window_days: int | None,
    max_documents: int | None,
    max_chars: int | None,
) -> dict[str, Any]:
    quarter_count = max(0, min(int(quarters if quarters is not None else 1), 4))
    event_window = max(
        0,
        min(
            int(material_event_window_days if material_event_window_days is not None else 365), 1095
        ),
    )
    document_limit = max(1, min(int(max_documents if max_documents is not None else 4), 12))
    char_limit = max(500, min(int(max_chars if max_chars is not None else 1800), 5000))
    filing_context = load_research_filing_context(
        ctx.ticker,
        as_of_date=ctx.as_of_date,
        quarters=quarter_count,
        include_material_events=True,
        material_event_window_days=event_window,
        material_event_limit=document_limit,
    )
    documents = []
    fresh_document_count = 0
    fresh_annual_document_count = 0
    for document in filing_context.ordered_documents[:document_limit]:
        excerpt, source_strategy, noise_filtered, used_noisy_fallback = (
            _extract_recent_filing_excerpt(
                document,
                max_chars=char_limit,
            )
        )
        readable_for_decision = (
            bool(excerpt.strip())
            and not used_noisy_fallback
            and not _is_noisy_filing_excerpt(excerpt)
        )
        is_fresh_role = document.role in {"quarterly", "material_event"}
        if is_fresh_role and readable_for_decision:
            fresh_document_count += 1
        annual_age_days = (
            _date_age_days(ctx.as_of_date, document.filing_date)
            if document.role == "annual"
            else None
        )
        is_readable_latest_annual = (
            document.role == "annual"
            and annual_age_days is not None
            and 0 <= annual_age_days <= 455
            and readable_for_decision
        )
        if is_readable_latest_annual:
            fresh_annual_document_count += 1
        documents.append(
            {
                "ticker": document.ticker,
                "form_type": document.form_type,
                "filing_date": document.filing_date,
                "period_end": document.period_end,
                "accession": document.accession,
                "role": document.role,
                "primary_doc_url": document.primary_doc_url,
                "excerpt": excerpt,
                "excerpt_chars": len(excerpt),
                "source_strategy": source_strategy,
                "narrative_chars": len(excerpt) if readable_for_decision else 0,
                "noise_filtered_passages": noise_filtered,
                "filing_age_days": annual_age_days,
                "materialized_from": document.materialized_from,
                "readable_for_decision": readable_for_decision,
            }
        )
    usable = (fresh_document_count + fresh_annual_document_count) > 0
    any_readable_document = any(
        bool(document.get("readable_for_decision")) for document in documents
    )
    readable_strategies = list(
        dict.fromkeys(
            str(document.get("source_strategy") or "")
            for document in documents
            if document.get("readable_for_decision") and document.get("source_strategy")
        )
    )
    strategy_summary = ", ".join(readable_strategies) if readable_strategies else "none"
    evidence_status = (
        "RECENT_FILING_CONTEXT_AVAILABLE"
        if usable
        else (
            "RECENT_FILINGS_UNREADABLE"
            if documents and not any_readable_document
            else str(
                getattr(filing_context, "recent_filing_status", "") or "NO_RECENT_FILING_CONTEXT"
            )
        )
    )
    if not usable and evidence_status == "UNKNOWN":
        evidence_status = "NO_RECENT_FILING_CONTEXT"
    return {
        "status": "ok" if usable else "unavailable",
        "ticker": ctx.ticker,
        "as_of_date": ctx.as_of_date,
        "documents": documents,
        "document_count": len(documents),
        "fresh_document_count": fresh_document_count,
        "fresh_annual_document_count": fresh_annual_document_count,
        "recent_filing_status": getattr(filing_context, "recent_filing_status", "UNKNOWN"),
        "recovered_document_count": getattr(filing_context, "recovered_document_count", 0),
        "warnings": list(filing_context.warnings),
        "usable_for_decision": usable,
        "evidence_status": evidence_status,
        "summary": (
            f"{fresh_document_count} recent quarterly/material-event filing document(s) available with readable narrative "
            f"(source_strategy={strategy_summary})."
            if fresh_document_count > 0 and fresh_annual_document_count == 0
            else (
                f"{fresh_annual_document_count} readable latest annual filing document(s) available as fresh filing evidence "
                f"(source_strategy={strategy_summary})."
                if usable
                else f"No readable recent filing context was available ({evidence_status})."
            )
        ),
    }


def _focused_excerpt(text: str, *, focus: str | None, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return ""
    focus_terms = [
        token.lower() for token in re.split(r"[^a-zA-Z0-9]+", str(focus or "")) if len(token) >= 3
    ]
    if focus_terms:
        lower = clean.lower()
        positions = [lower.find(term) for term in focus_terms if lower.find(term) >= 0]
        if positions:
            start = max(0, min(positions) - max_chars // 3)
            end = min(len(clean), start + max_chars)
            return clean[start:end]
    return clean[:max_chars]


def _fetch_transcript_excerpt(ctx: AlphaToolContext, *, focus: str | None) -> dict[str, Any]:
    cfg = get_config()
    if (
        cfg.safe_mode
        or not cfg.research_enable_transcripts
        or str(cfg.research_transcript_provider or "").lower() == "disabled"
    ):
        return {
            "status": "unavailable",
            "ticker": ctx.ticker,
            "reason": "transcripts_disabled",
            "focus": focus,
        }
    adapter = TranscriptAdapter(cfg)
    result = adapter.collect(
        AdapterContext(
            ticker=ctx.ticker.upper(),
            as_of_date=ctx.as_of_date,
            company_name=getattr(ctx.packet, "name", None),
            packet={},
        )
    )
    warnings = [gap.gap_id for gap in result.evidence_gaps]
    if not result.evidence_items:
        return {
            "status": "unavailable",
            "ticker": ctx.ticker,
            "reason": "transcript_not_found",
            "focus": focus,
            "warnings": warnings,
        }
    item = result.evidence_items[0]
    excerpt = _focused_excerpt(item.excerpt_text, focus=focus, max_chars=1500)
    return {
        "status": "ok",
        "ticker": ctx.ticker,
        "source_type": item.source_type,
        "source_title": item.source_title,
        "source_url": item.source_url,
        "source_published_at": item.source_published_at,
        "excerpt": excerpt,
        "focus": focus,
        "warnings": warnings,
    }


def dispatch_alpha_tool(
    name: str, tool_input: dict[str, Any], ctx: AlphaToolContext
) -> dict[str, Any]:
    name = str(name or "").strip()
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    if name == "fetch_filing_section":
        return _fetch_filing_section(
            ctx,
            keywords=[str(item) for item in (tool_input.get("keywords") or [])],
            section_focus=str(tool_input.get("section_focus") or "").strip() or None,
            max_chars=tool_input.get("max_chars"),
        )
    if name == "fetch_companyfacts_timeseries":
        return _fetch_companyfacts_timeseries(
            ctx,
            line_items=[str(item) for item in (tool_input.get("line_items") or [])],
            years=tool_input.get("years"),
        )
    if name == "fetch_kpi_trends":
        return _fetch_kpi_trends(ctx)
    if name == "fetch_insurance_evidence_packet":
        return _fetch_insurance_evidence_packet(ctx)
    if name == "compare_peer_metric":
        return _compare_peer_metric(ctx, metric=str(tool_input.get("metric") or ""))
    if name == "analyze_dilution":
        return _analyze_dilution(ctx, years=tool_input.get("years"))
    if name == "analyze_liquidity_stress":
        return _analyze_liquidity_stress(ctx)
    if name == "analyze_capital_structure_resolution":
        return _analyze_capital_structure_resolution(ctx, max_chars=tool_input.get("max_chars"))
    if name == "analyze_capital_allocation":
        return _analyze_capital_allocation(ctx)
    if name == "fetch_current_events":
        return _fetch_current_events(ctx, max_items=tool_input.get("max_items"))
    if name == "fetch_recent_filing_context":
        return _fetch_recent_filing_context(
            ctx,
            quarters=tool_input.get("quarters"),
            material_event_window_days=tool_input.get("material_event_window_days"),
            max_documents=tool_input.get("max_documents"),
            max_chars=tool_input.get("max_chars"),
        )
    if name == "fetch_transcript_excerpt":
        return _fetch_transcript_excerpt(
            ctx, focus=str(tool_input.get("focus") or "").strip() or None
        )
    if name == "finalize_candidate_investigation":
        return {"status": "finalize_passthrough"}
    return {"status": "error", "ticker": ctx.ticker, "reason": f"unknown_tool:{name}"}


def dispatch_alpha_tool_json(name: str, tool_input: dict[str, Any], ctx: AlphaToolContext) -> str:
    return json.dumps(_jsonable(dispatch_alpha_tool(name, tool_input, ctx)))


def suggested_tool_calls_for_gaps(evidence_gaps: list[str]) -> list[tuple[str, dict[str, Any]]]:
    gap_text = " ".join(str(item).lower() for item in evidence_gaps)
    suggested: list[tuple[str, dict[str, Any]]] = [("fetch_kpi_trends", {})]
    if any(token in gap_text for token in ("dilution", "share", "per-share")):
        suggested.append(("analyze_dilution", {"years": 5}))
        suggested.append(("analyze_capital_allocation", {}))
    if any(token in gap_text for token in ("liquidity", "debt", "balance sheet", "refinancing")):
        suggested.append(("analyze_liquidity_stress", {}))
        suggested.append(("analyze_capital_structure_resolution", {}))
        suggested.append(
            (
                "fetch_companyfacts_timeseries",
                {"line_items": ["cash", "total_debt", "cfo"], "years": 5},
            )
        )
    if any(
        token in gap_text
        for token in ("insurance", "preferred", "depositary", "book", "roe", "yield", "liquidation")
    ):
        suggested.append(("fetch_insurance_evidence_packet", {}))
    if any(token in gap_text for token in ("peer", "margin", "roic", "growth")):
        suggested.append(("compare_peer_metric", {"metric": "roic"}))
        suggested.append(("compare_peer_metric", {"metric": "operating_margin"}))
    if any(
        token in gap_text
        for token in ("filing", "risk", "competition", "capital allocation", "guidance")
    ):
        suggested.append(
            (
                "fetch_filing_section",
                {
                    "keywords": ["risk", "competition", "liquidity", "capital allocation"],
                    "max_chars": 6000,
                },
            )
        )
    if any(token in gap_text for token in ("event", "press", "news")):
        suggested.append(("fetch_current_events", {"max_items": 5}))
    if any(token in gap_text for token in ("transcript", "call", "management")):
        suggested.append(("fetch_transcript_excerpt", {"focus": "management commentary"}))
    deduped: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for tool_name, payload in suggested:
        token = json.dumps({"tool": tool_name, "payload": payload}, sort_keys=True)
        if token in seen:
            continue
        seen.add(token)
        deduped.append((tool_name, payload))
    return deduped


def default_as_of_date() -> str:
    return date.today().isoformat()
