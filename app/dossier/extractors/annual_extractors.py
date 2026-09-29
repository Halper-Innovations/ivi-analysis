from __future__ import annotations

from html import unescape
import json
import re
from typing import Any
import logging

from app.db import get_db
from app.dossier.collector import DossierFiling
from app.dossier.sections import SectionSpan, section_by_label
from app.util.text import extract_snippet, try_parse_number


UNKNOWN = "UNKNOWN"
logger = logging.getLogger(__name__)
PCT_RE = re.compile(r"(\d{1,2}(?:\.\d+)?)%")
ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
TABLE_RE = re.compile(r"<table\b[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
NUMERIC_RE = re.compile(r"\(?-?\$?[\d,]{1,15}(?:\.\d+)?\)?")
IX_NON_FRACTION_RE = re.compile(
    r"<ix:nonFraction\b[^>]*name=\"(?P<name>[^\"]+)\"[^>]*>(?P<value>.*?)</ix:nonFraction>",
    re.IGNORECASE | re.DOTALL,
)
_TABLE_UNIT_RE = re.compile(
    r"\b(?:amounts?\s+)?in\s+(?P<unit>millions?|thousands?|dollars?)\b",
    re.IGNORECASE,
)


def _year_for_filing(filing: DossierFiling) -> int:
    if filing.period_end:
        try:
            return int(filing.period_end[:4])
        except Exception:
            pass
    return int(filing.filing_date[:4])


def _load_companyfacts_rows(filing: DossierFiling) -> dict[str, dict[str, Any]]:
    """Load only facts published by the exact selected annual filing.

    The live CompanyFacts table is mutable: a later amendment can replace the
    row for the same ticker/year/line item. The append-only vintage table
    preserves the earlier value, so both sources are searched under the
    selected filing's accession, filing date, and period end.
    """

    accession = str(filing.accession or "").strip()
    filing_date = str(filing.filing_date or "").strip()
    period_end = str(filing.period_end or "").strip()
    if not accession or not filing_date or not period_end:
        return {}
    with get_db() as conn:
        rows = conn.execute(
            """
            WITH candidates AS (
                SELECT
                    id AS source_id,
                    line_item,
                    value,
                    units,
                    source_url,
                    filed_date,
                    accession,
                    period_end,
                    0 AS source_rank
                FROM companyfacts_facts
                WHERE UPPER(ticker) = ?
                  AND fiscal_year = ?
                  AND period_type = 'FY'
                  AND period_end = ?
                  AND filed_date = ?
                  AND accession = ?
                  AND source_url IS NOT NULL
                  AND TRIM(source_url) != ''
                UNION ALL
                SELECT
                    id AS source_id,
                    line_item,
                    value,
                    units,
                    source_url,
                    filed_date,
                    accession,
                    period_end,
                    1 AS source_rank
                FROM companyfacts_vintages
                WHERE UPPER(ticker) = ?
                  AND fiscal_year = ?
                  AND period_type = 'FY'
                  AND period_end = ?
                  AND filed_date = ?
                  AND accession = ?
                  AND source_url IS NOT NULL
                  AND TRIM(source_url) != ''
            ),
            ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY line_item
                        ORDER BY source_rank ASC, source_id DESC
                    ) AS row_rank
                FROM candidates
            )
            SELECT
                line_item,
                value,
                units,
                source_url,
                filed_date,
                accession,
                period_end
            FROM ranked
            WHERE row_rank = 1
            ORDER BY line_item
            """,
            (
                filing.ticker.upper(),
                _year_for_filing(filing),
                period_end,
                filing_date,
                accession,
                filing.ticker.upper(),
                _year_for_filing(filing),
                period_end,
                filing_date,
                accession,
            ),
        ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        line = str(row["line_item"] or "")
        if not line or line in out:
            continue
        out[line] = {
            "value": row["value"],
            "units": row["units"],
            "source_url": row["source_url"] or "",
            "filed_date": row["filed_date"],
            "accession": row["accession"],
            "period_end": row["period_end"],
            "snippet": "",
        }
    return out


def _load_fact_rows(filing_id: int) -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT fact_type, value_json, source_url, snippet, section_label
            FROM extracted_facts
            WHERE filing_id = ?
            """,
            (filing_id,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["value_json"] or "{}")
        except Exception:
            payload = {}
        out.append(
            {
                "fact_type": row["fact_type"],
                "value_json": payload,
                "source_url": row["source_url"] or "",
                "snippet": row["snippet"] or "",
                "section_label": row["section_label"] or "",
            }
        )
    return out


def _build_item(
    *,
    filing: DossierFiling,
    metric: str,
    value: float | int | str,
    section_label: str,
    source_url: str,
    snippet: str,
    derived_from: list[str] | None = None,
    citations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "ticker": filing.ticker,
        "year": _year_for_filing(filing),
        "metric": metric,
        "value": value,
        "section_label": section_label,
        "source_url": source_url,
        "snippet": snippet[:650],
        "derived_from": list(derived_from or []),
        "citations": list(citations or []),
        "filing_accession": filing.accession,
        "filing_date": filing.filing_date,
        "period_end": filing.period_end,
    }


def _citation(source_url: str, snippet: str, section_label: str) -> dict[str, Any]:
    return {
        "source_url": source_url,
        "snippet": snippet[:650],
        "section_label": section_label,
    }


def _strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(TAG_RE.sub(" ", text or ""))).strip()


def _explicit_table_unit(raw_table: str) -> str | None:
    match = _TABLE_UNIT_RE.search(_strip_tags(raw_table))
    if match is None:
        return None
    token = match.group("unit").lower()
    if token.startswith("million"):
        return "USD_millions"
    if token.startswith("thousand"):
        return "USD_thousands"
    if token.startswith("dollar"):
        return "USD"
    return None


def _extract_table_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tables = list(TABLE_RE.finditer(text or ""))
    table_payloads = [(match.group(0), _explicit_table_unit(match.group(0))) for match in tables]
    if not table_payloads:
        table_payloads = [(text or "", None)]
    for raw_table, table_unit in table_payloads:
        for match in ROW_RE.finditer(raw_table):
            raw_row = match.group(1)
            raw_cells = CELL_RE.findall(raw_row)
            if not raw_cells:
                continue
            cells = [_strip_tags(cell) for cell in raw_cells]
            cells = [cell for cell in cells if cell]
            if len(cells) < 2:
                continue
            label = cells[0]
            numeric_values: list[float] = []
            for cell in cells[1:]:
                for raw_num in NUMERIC_RE.findall(cell):
                    value = try_parse_number(raw_num.replace("$", ""))
                    if isinstance(value, (int, float)):
                        numeric_values.append(float(value))
                        break
            if not numeric_values:
                continue
            snippet = _strip_tags(match.group(0))
            rows.append(
                {
                    "label": label,
                    "label_norm": normalize_label(label),
                    "value": numeric_values[0],
                    "unit": table_unit,
                    "snippet": snippet[:650],
                }
            )
    return rows


def _read_raw_filing_html(filing: DossierFiling) -> str:
    if not filing.local_path:
        return ""
    try:
        return open(filing.local_path, encoding="utf-8", errors="ignore").read()
    except Exception:
        return ""


def _section_html(raw_html: str, sections: list[SectionSpan], label: str) -> str:
    section = section_by_label(sections, label)
    if not section or not raw_html:
        return ""
    start = max(0, int(section.start_offset))
    end = min(len(raw_html), int(section.end_offset))
    if end <= start:
        return ""
    return raw_html[start:end]


_SECTION_FALLBACK_MARKERS: dict[str, tuple[list[str], list[str]]] = {
    "business": (
        [r'id="item_1_business"', r">\s*ITEM\s*1\.\s*B(?:USINESS|usiness)\b"],
        [r'id="item_1a_risk_factors"', r">\s*ITEM\s*1A\.\s*RISK\s*FACTORS\b"],
    ),
    "risk_factors": (
        [r'id="item_1a_risk_factors"', r">\s*ITEM\s*1A\.\s*RISK\s*FACTORS\b"],
        [
            r'id="item_1b_unresolved_staff_comments"',
            r'id="item_2_properties"',
            r">\s*ITEM\s*1B\.",
            r">\s*ITEM\s*2\.",
        ],
    ),
    "md_and_a": (
        [
            r'id="item_7_management_s_discussion_and_analysis"',
            r'id="item_7_managements_discussion_and_analysis"',
            r">\s*ITEM\s*7\.\s*MANAGEMENT",
        ],
        [
            r'id="item_7a_quantitative_and_qualitative_disclosures_about_market_risk"',
            r'id="item_8_financial_statements"',
            r">\s*ITEM\s*7A\.",
            r">\s*ITEM\s*8\.",
        ],
    ),
}


def _fallback_section_html(raw_html: str, label: str) -> str:
    if not raw_html:
        return ""
    markers = _SECTION_FALLBACK_MARKERS.get(label)
    if not markers:
        return ""
    start_patterns, end_patterns = markers
    start = None
    for pattern in start_patterns:
        match = re.search(pattern, raw_html, re.IGNORECASE)
        if match:
            start = match.start()
            break
    if start is None:
        return ""
    end = len(raw_html)
    for pattern in end_patterns:
        match = re.search(pattern, raw_html[start + 1 :], re.IGNORECASE)
        if match:
            end = min(end, start + 1 + match.start())
    if end <= start:
        return ""
    return raw_html[start:end]


def _ix_values_by_name(raw_html: str, names: list[str]) -> list[tuple[str, float, str]]:
    wanted = {name.lower(): name for name in names}
    out: list[tuple[str, float, str]] = []
    for match in IX_NON_FRACTION_RE.finditer(raw_html or ""):
        full_tag = match.group(0)
        tag_name = str(match.group("name") or "")
        if tag_name.lower() not in wanted:
            continue
        value = try_parse_number(_strip_tags(match.group("value") or "").replace("$", ""))
        if not isinstance(value, (int, float)):
            continue
        unit_match = re.search(r'\bunitRef="([^"]+)"', full_tag, re.IGNORECASE)
        if unit_match is None or "usd" not in unit_match.group(1).lower():
            continue
        scale_match = re.search(r'\bscale="(-?\d+)"', full_tag, re.IGNORECASE)
        try:
            scale = int(scale_match.group(1)) if scale_match is not None else 0
            value = float(value) * (10 ** (scale - 6))
        except (OverflowError, ValueError):
            continue
        snippet = _strip_tags(match.group(0))
        out.append((tag_name, float(value), snippet[:650]))
    return out


def _xbrl_value(raw_html: str, names: list[str]) -> tuple[float | str, str]:
    values = _ix_values_by_name(raw_html, names)
    if not values:
        return UNKNOWN, ""
    tag_name, value, snippet = values[0]
    return float(value), f"{tag_name}: {snippet}"[:650]


def _normalize_operating_expense_value(
    *,
    filing: DossierFiling,
    metric: str,
    value: float | str,
    source_unit: str | None,
) -> float | str:
    if not isinstance(value, (int, float)):
        return value
    raw_value = float(value)
    unit = str(source_unit or "").strip()
    scale = {
        "USD_millions": 1.0,
        "USD_thousands": 1_000.0,
        "USD": 1_000_000.0,
    }.get(unit)
    if scale is None:
        logger.warning(
            "filing_operating_expense_unit_unavailable",
            extra={
                "ticker": filing.ticker,
                "metric": metric,
                "year": _year_for_filing(filing),
                "raw_value": raw_value,
            },
        )
        return UNKNOWN
    normalized = raw_value / scale
    if scale != 1.0:
        logger.warning(
            "normalized_filing_operating_expense_units",
            extra={
                "ticker": filing.ticker,
                "metric": metric,
                "year": _year_for_filing(filing),
                "raw_value": raw_value,
                "normalized_value": normalized,
                "source_unit": unit,
                "output_unit": "USD_millions",
            },
        )
    return float(normalized)


def _customer_concentration_from_raw(raw_html: str) -> tuple[int | str, float | str, str]:
    plain = _strip_tags(raw_html)
    if not plain:
        return UNKNOWN, UNKNOWN, ""

    explicit_no_customer = re.search(
        r"no sales to an individual customer[^.]{0,300}accounted for more than (\d{1,2}(?:\.\d+)?)%",
        plain,
        re.IGNORECASE,
    )
    if explicit_no_customer:
        snippet = explicit_no_customer.group(0)[:650]
        return 0, UNKNOWN, snippet

    explicit_customer = re.search(
        r"(?:major customer|customer)[^.]{0,200}(?:represented|accounted for)\s+(\d{1,2}(?:\.\d+)?)%\s+of revenue",
        plain,
        re.IGNORECASE,
    )
    if explicit_customer:
        snippet = explicit_customer.group(0)[:650]
        try:
            return 1, float(explicit_customer.group(1)) / 100.0, snippet
        except Exception:
            return 1, UNKNOWN, snippet

    return UNKNOWN, UNKNOWN, ""


def normalize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (label or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _row_matches(label_norm: str, keywords: list[str]) -> bool:
    for keyword in keywords:
        normalized = normalize_label(keyword)
        if not normalized:
            continue
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])")
        if pattern.search(label_norm):
            return True
    return False


def _table_value_for_keywords(
    rows: list[dict[str, Any]],
    keywords: list[str],
    *,
    disallowed_keywords: list[str] | None = None,
) -> tuple[float | str, str]:
    value, snippet, _unit = _table_value_with_unit_for_keywords(
        rows,
        keywords,
        disallowed_keywords=disallowed_keywords,
    )
    return value, snippet


def _table_value_with_unit_for_keywords(
    rows: list[dict[str, Any]],
    keywords: list[str],
    *,
    disallowed_keywords: list[str] | None = None,
) -> tuple[float | str, str, str | None]:
    matches = []
    for row in rows:
        label_norm = str(row.get("label_norm") or "")
        if not _row_matches(label_norm, keywords):
            continue
        if disallowed_keywords and any(
            _row_matches(label_norm, [keyword]) for keyword in disallowed_keywords
        ):
            continue
        matches.append(row)
    if len(matches) != 1:
        return UNKNOWN, "", None
    match = matches[0]
    value = match.get("value")
    if not isinstance(value, (int, float)):
        return UNKNOWN, "", None
    return (
        float(value),
        str(match.get("snippet") or ""),
        str(match["unit"]) if match.get("unit") else None,
    )


def _sentence_candidates(text: str) -> list[str]:
    normalized = _strip_tags(text)
    if not normalized:
        return []
    chunks = re.split(r"(?<=[\.;])\s+", normalized)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _sentence_with_keywords(text: str, keywords: list[str]) -> str:
    for sentence in _sentence_candidates(text):
        lowered = sentence.lower()
        if any(keyword.lower() in lowered for keyword in keywords):
            return sentence
    return ""


def _amount_from_text(text: str, keywords: list[str]) -> tuple[float | str, str]:
    sentence = _sentence_with_keywords(text, keywords)
    if not sentence:
        return UNKNOWN, ""
    raw_numbers = NUMERIC_RE.findall(sentence)
    parsed: list[float] = []
    for raw_num in raw_numbers:
        value = try_parse_number(raw_num.replace("$", ""))
        if isinstance(value, (int, float)):
            parsed.append(float(value))
    if len(parsed) != 1:
        return UNKNOWN, sentence[:650]
    return parsed[0], sentence[:650]


def _pct_from_text(text: str, keywords: list[str]) -> tuple[float | str, str]:
    sentence = _sentence_with_keywords(text, keywords)
    if not sentence:
        return UNKNOWN, ""
    match = PCT_RE.search(sentence)
    if not match:
        return UNKNOWN, sentence[:650]
    try:
        return float(match.group(1)) / 100.0, sentence[:650]
    except Exception:
        return UNKNOWN, sentence[:650]


def _value_near_keywords(text: str, keywords: list[str]) -> tuple[float | str, str]:
    for keyword in keywords:
        pattern = re.compile(
            rf"({keyword})[^\d\(\)\$-]{{0,80}}(\(?-?\$?[\d,]{{1,15}}(?:\.\d+)?\)?)",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            continue
        raw = match.group(2).replace("$", "")
        value = try_parse_number(raw)
        if isinstance(value, (int, float)):
            return float(value), extract_snippet(text, keyword, width=540)
    return UNKNOWN, ""


def _pct_near_keywords(text: str, keywords: list[str]) -> tuple[float | str, str]:
    for keyword in keywords:
        idx = text.lower().find(keyword.lower())
        if idx < 0:
            continue
        window = text[max(0, idx - 100) : min(len(text), idx + 220)]
        match = PCT_RE.search(window)
        if not match:
            continue
        try:
            return float(match.group(1)) / 100.0, extract_snippet(text, keyword, width=540)
        except Exception:
            continue
    return UNKNOWN, ""


def _keyword_mentions(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(k.lower()) for k in keywords)


def _keyword_sentence_count(text: str, keywords: list[str]) -> tuple[int | str, str]:
    sentences = _sentence_candidates(text)
    if not sentences:
        return UNKNOWN, ""
    matching = [
        sentence
        for sentence in sentences
        if any(keyword.lower() in sentence.lower() for keyword in keywords)
    ]
    if not matching:
        return 0, ""
    return len(matching), matching[0][:650]


def _filtered_sentence_count(
    text: str,
    keywords: list[str],
    *,
    excluded_phrases: list[str] | None = None,
) -> tuple[int | str, str]:
    sentences = _sentence_candidates(text)
    if not sentences:
        return UNKNOWN, ""
    excluded = [phrase.lower() for phrase in (excluded_phrases or []) if phrase.strip()]
    matching: list[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if not any(keyword.lower() in lowered for keyword in keywords):
            continue
        if any(phrase in lowered for phrase in excluded):
            continue
        matching.append(sentence)
    if not matching:
        return 0, ""
    return len(matching), matching[0][:650]


def _segment_names_from_text(text: str) -> tuple[list[str], str]:
    plain = _strip_tags(text)
    if not plain:
        return [], ""
    candidates: list[str] = []
    snippet = ""

    sentence = _sentence_with_keywords(
        plain,
        ["reportable segments", "operating segments", "segments include", "segments are"],
    )
    if sentence:
        snippet = sentence[:650]
        colon_match = re.search(
            r"(?:reportable|operating)\s+segments?[:\s]+(.+?)(?:\.|;|$)", sentence, re.IGNORECASE
        )
        if colon_match:
            tail = colon_match.group(1)
            tail = re.sub(r"\band\b", ",", tail, flags=re.IGNORECASE)
            parts = [part.strip(" .;:") for part in tail.split(",") if part.strip(" .;:")]
            for part in parts:
                part = re.sub(r"\bsegment\b", "", part, flags=re.IGNORECASE).strip(" .;:")
                if part and len(part) >= 3:
                    candidates.append(part)

        for match in re.finditer(r"([A-Z][A-Za-z&/\-\s]{2,60}?)\s+segment\b", sentence):
            name = match.group(1).strip(" .;:")
            if name:
                candidates.append(name)

    cleaned: list[str] = []
    for name in candidates:
        normalized = re.sub(r"\s+", " ", name).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in {"our", "two", "three", "four", "reportable", "operating"}:
            continue
        if lowered.startswith("we operate"):
            continue
        cleaned.append(normalized)
    deduped = sorted(set(cleaned))
    return deduped, snippet


def extract_annual_items(
    *,
    filing: DossierFiling,
    sections: list[SectionSpan],
) -> list[dict[str, Any]]:
    financials = _load_companyfacts_rows(filing)

    # Prefer companyfacts shares; extracted_facts cover_page is the fallback (handled in the facts loop)
    shares = UNKNOWN
    shares_section_label = "cover_page"
    shares_source_url = source_url = filing.primary_doc_url
    shares_snippet = "shares outstanding extracted from cover page"
    shares_derived_from = ["extracted_facts.shares_outstanding"]
    shares_citations: list[dict[str, Any]] = []
    _cf_shares = financials.get("shares_outstanding")
    if _cf_shares and isinstance(_cf_shares.get("value"), (int, float)):
        shares = float(_cf_shares["value"])
        shares_section_label = "financial_statements"
        shares_source_url = _cf_shares.get("source_url") or source_url
        shares_snippet = _cf_shares.get("snippet") or ""
        shares_derived_from = ["financials.shares_outstanding"]
        shares_citations = (
            [_citation(shares_source_url, shares_snippet, "financial_statements")]
            if shares_source_url
            else []
        )

    facts = _load_fact_rows(filing.filing_id)

    revenue = financials.get("revenue", {}).get("value")
    gross_profit = financials.get("gross_profit", {}).get("value")
    operating_income = financials.get("operating_income", {}).get("value")
    net_income = financials.get("net_income", {}).get("value")
    cfo = financials.get("cfo", {}).get("value")
    capex = financials.get("capex", {}).get("value")
    cash = financials.get("cash", {}).get("value")
    total_debt = financials.get("total_debt", {}).get("value")
    equity = financials.get("equity", {}).get("value")

    out: list[dict[str, Any]] = []
    raw_html = _read_raw_filing_html(filing)
    fin_section = section_by_label(sections, "financial_statements")
    fin_text = fin_section.text if fin_section else ""
    fin_html = _section_html(raw_html, sections, "financial_statements") or fin_text
    fin_rows = _extract_table_rows(fin_html)
    notes = section_by_label(sections, "notes")
    notes_text = notes.text if notes else ""
    notes_html = _section_html(raw_html, sections, "notes") or notes_text
    notes_rows = _extract_table_rows(notes_html)

    def _item_from_line(
        metric: str,
        line_item: str,
        value: float | int | str,
        *,
        derived_from: list[str],
        citations_override: list[dict[str, Any]] | None = None,
        snippet_override: str | None = None,
        source_override: str | None = None,
    ) -> None:
        row = financials.get(line_item, {})
        row_source = (
            source_override
            if source_override is not None
            else (row.get("source_url") or source_url)
        )
        row_snippet = (
            snippet_override if snippet_override is not None else (row.get("snippet") or "")
        )
        if citations_override is None:
            citations = (
                [_citation(row_source, row_snippet, "financial_statements")] if row_source else []
            )
        else:
            citations = citations_override
        out.append(
            _build_item(
                filing=filing,
                metric=metric,
                value=value,
                section_label="financial_statements",
                source_url=row_source,
                snippet=row_snippet,
                derived_from=derived_from,
                citations=citations,
            )
        )

    _item_from_line(
        "revenue",
        "revenue",
        revenue if isinstance(revenue, (int, float)) else UNKNOWN,
        derived_from=["financials.revenue"],
    )
    _item_from_line(
        "gross_profit",
        "gross_profit",
        gross_profit if isinstance(gross_profit, (int, float)) else UNKNOWN,
        derived_from=["financials.gross_profit"],
    )
    _item_from_line(
        "gross_profit_dollars",
        "gross_profit",
        gross_profit if isinstance(gross_profit, (int, float)) else UNKNOWN,
        derived_from=["financials.gross_profit"],
    )
    _item_from_line(
        "operating_income",
        "operating_income",
        operating_income if isinstance(operating_income, (int, float)) else UNKNOWN,
        derived_from=["financials.operating_income"],
    )
    _item_from_line(
        "operating_income_dollars",
        "operating_income",
        operating_income if isinstance(operating_income, (int, float)) else UNKNOWN,
        derived_from=["financials.operating_income"],
    )
    _item_from_line(
        "net_income",
        "net_income",
        net_income if isinstance(net_income, (int, float)) else UNKNOWN,
        derived_from=["financials.net_income"],
    )
    _item_from_line(
        "cfo",
        "cfo",
        cfo if isinstance(cfo, (int, float)) else UNKNOWN,
        derived_from=["financials.cfo"],
    )
    _item_from_line(
        "capex",
        "capex",
        capex if isinstance(capex, (int, float)) else UNKNOWN,
        derived_from=["financials.capex"],
    )
    _item_from_line(
        "cash",
        "cash",
        cash if isinstance(cash, (int, float)) else UNKNOWN,
        derived_from=["financials.cash"],
    )
    _item_from_line(
        "total_debt",
        "total_debt",
        total_debt if isinstance(total_debt, (int, float)) else UNKNOWN,
        derived_from=["financials.total_debt"],
    )
    _item_from_line(
        "equity",
        "equity",
        equity if isinstance(equity, (int, float)) else UNKNOWN,
        derived_from=["financials.equity"],
    )

    if isinstance(cfo, (int, float)) and isinstance(capex, (int, float)):
        cfo_row = financials.get("cfo", {})
        capex_row = financials.get("capex", {})
        cits = []
        if cfo_row.get("source_url") or cfo_row.get("snippet"):
            cits.append(
                _citation(
                    cfo_row.get("source_url") or source_url,
                    cfo_row.get("snippet") or "",
                    "financial_statements",
                )
            )
        if capex_row.get("source_url") or capex_row.get("snippet"):
            cits.append(
                _citation(
                    capex_row.get("source_url") or source_url,
                    capex_row.get("snippet") or "",
                    "financial_statements",
                )
            )
        _item_from_line(
            "fcf_dollars",
            "cfo",
            float(cfo) - float(capex),
            derived_from=["financials.cfo", "financials.capex"],
            citations_override=cits,
        )
        _item_from_line(
            "fcf",
            "cfo",
            float(cfo) - float(capex),
            derived_from=["financials.cfo", "financials.capex"],
            citations_override=cits,
        )
    else:
        _item_from_line(
            "fcf_dollars",
            "cfo",
            UNKNOWN,
            derived_from=["financials.cfo", "financials.capex"],
        )
        _item_from_line("fcf", "cfo", UNKNOWN, derived_from=["financials.cfo", "financials.capex"])

    if isinstance(total_debt, (int, float)) and isinstance(cash, (int, float)):
        debt_row = financials.get("total_debt", {})
        cash_row = financials.get("cash", {})
        cits = []
        if debt_row.get("source_url") or debt_row.get("snippet"):
            cits.append(
                _citation(
                    debt_row.get("source_url") or source_url,
                    debt_row.get("snippet") or "",
                    "financial_statements",
                )
            )
        if cash_row.get("source_url") or cash_row.get("snippet"):
            cits.append(
                _citation(
                    cash_row.get("source_url") or source_url,
                    cash_row.get("snippet") or "",
                    "financial_statements",
                )
            )
        _item_from_line(
            "net_debt",
            "total_debt",
            float(total_debt) - float(cash),
            derived_from=["financials.total_debt", "financials.cash"],
            citations_override=cits,
        )
    else:
        _item_from_line(
            "net_debt",
            "total_debt",
            UNKNOWN,
            derived_from=["financials.total_debt", "financials.cash"],
        )

    if (
        isinstance(gross_profit, (int, float))
        and isinstance(revenue, (int, float))
        and float(revenue) != 0
    ):
        _item_from_line(
            "gross_margin",
            "gross_profit",
            float(gross_profit) / float(revenue),
            derived_from=["financials.gross_profit", "financials.revenue"],
        )
    else:
        _item_from_line(
            "gross_margin",
            "gross_profit",
            UNKNOWN,
            derived_from=["financials.gross_profit", "financials.revenue"],
        )

    if (
        isinstance(operating_income, (int, float))
        and isinstance(revenue, (int, float))
        and float(revenue) != 0
    ):
        _item_from_line(
            "operating_margin",
            "operating_income",
            float(operating_income) / float(revenue),
            derived_from=["financials.operating_income", "financials.revenue"],
        )
    else:
        _item_from_line(
            "operating_margin",
            "operating_income",
            UNKNOWN,
            derived_from=["financials.operating_income", "financials.revenue"],
        )

    if (
        isinstance(cfo, (int, float))
        and isinstance(capex, (int, float))
        and isinstance(revenue, (int, float))
        and float(revenue) != 0
    ):
        fcf = float(cfo) - float(capex)
        _item_from_line(
            "fcf_margin",
            "cfo",
            fcf / float(revenue),
            derived_from=["financials.cfo", "financials.capex", "financials.revenue"],
        )
    else:
        _item_from_line(
            "fcf_margin",
            "cfo",
            UNKNOWN,
            derived_from=["financials.cfo", "financials.capex", "financials.revenue"],
        )

    for metric, keywords, disallowed in [
        (
            "r_and_d_total",
            ["research and development", "r&d"],
            ["sales and marketing", "selling and marketing", "general and administrative", "g&a"],
        ),
        (
            "sales_marketing_total",
            ["sales and marketing", "selling and marketing"],
            ["research and development", "r&d", "general and administrative", "g&a"],
        ),
        ("g_and_a_total", ["general and administrative", "g&a"], []),
    ]:
        value, snippet, source_unit = _table_value_with_unit_for_keywords(
            fin_rows,
            keywords,
            disallowed_keywords=disallowed,
        )
        if value is UNKNOWN and metric == "r_and_d_total":
            value, snippet = _xbrl_value(fin_html, ["us-gaap:ResearchAndDevelopmentExpense"])
            if value is not UNKNOWN:
                source_unit = "USD_millions"
        if value is UNKNOWN and metric == "r_and_d_total":
            value, snippet = _xbrl_value(raw_html, ["us-gaap:ResearchAndDevelopmentExpense"])
            if value is not UNKNOWN:
                source_unit = "USD_millions"
        if value is UNKNOWN and metric == "sales_marketing_total":
            value, snippet = _xbrl_value(raw_html, ["us-gaap:SellingAndMarketingExpense"])
            if value is not UNKNOWN:
                source_unit = "USD_millions"
        if value is UNKNOWN and metric == "g_and_a_total":
            value, snippet = _xbrl_value(raw_html, ["us-gaap:GeneralAndAdministrativeExpense"])
            if value is not UNKNOWN:
                source_unit = "USD_millions"
        value = _normalize_operating_expense_value(
            filing=filing,
            metric=metric,
            value=value,
            source_unit=source_unit,
        )
        out.append(
            _build_item(
                filing=filing,
                metric=metric,
                value=value,
                section_label="financial_statements",
                source_url=source_url,
                snippet=snippet,
                derived_from=[f"dossier.extractors.{metric}"],
                citations=[_citation(source_url, snippet, "financial_statements")]
                if snippet
                else [],
            )
        )
        pct_metric = metric.replace("_total", "_pct_revenue")
        if (
            isinstance(value, (int, float))
            and isinstance(revenue, (int, float))
            and float(revenue) != 0
        ):
            pct_value = float(value) / float(revenue)
            derived = [f"dossier.metrics.{metric}", "financials.revenue"]
        else:
            pct_value = UNKNOWN
            derived = [f"dossier.metrics.{metric}", "financials.revenue"]
        out.append(
            _build_item(
                filing=filing,
                metric=pct_metric,
                value=pct_value,
                section_label="financial_statements",
                source_url=source_url,
                snippet=snippet,
                derived_from=derived,
                citations=[_citation(source_url, snippet, "financial_statements")]
                if snippet
                else [],
            )
        )

    for metric, keywords in [
        (
            "share_repurchases_amount",
            ["common stock repurchased", "share repurchases", "repurchases of common stock"],
        ),
        ("dividends_paid_amount", ["dividends paid", "cash dividends"]),
    ]:
        value, snippet = _table_value_for_keywords(fin_rows, keywords)
        if value is UNKNOWN:
            value, snippet = _table_value_for_keywords(notes_rows, keywords)
        if value is UNKNOWN:
            value, snippet = _amount_from_text(
                notes_html or notes_text or fin_html or fin_text, keywords
            )
        if value is UNKNOWN and metric == "share_repurchases_amount":
            value, snippet = _xbrl_value(raw_html, ["us-gaap:PaymentsForRepurchaseOfCommonStock"])
        if value is UNKNOWN and metric == "dividends_paid_amount":
            value, snippet = _xbrl_value(raw_html, ["us-gaap:PaymentsOfDividendsCommonStock"])
        section_label = (
            "financial_statements"
            if snippet and (snippet in fin_text or snippet in fin_html)
            else "notes"
        )
        out.append(
            _build_item(
                filing=filing,
                metric=metric,
                value=value,
                section_label=section_label,
                source_url=source_url,
                snippet=snippet,
                derived_from=[f"dossier.extractors.{metric}"],
                citations=[_citation(source_url, snippet, section_label)] if snippet else [],
            )
        )

    risk_section = section_by_label(sections, "risk_factors")
    risk_text = risk_section.text if risk_section else ""
    risk_html = (
        _section_html(raw_html, sections, "risk_factors")
        or _fallback_section_html(raw_html, "risk_factors")
        or risk_text
    )
    risk_keywords = [
        "competition",
        "cybersecurity",
        "supply chain",
        "regulation",
        "macroeconomic",
        "litigation",
        "geopolitical",
    ]
    risk_count, risk_snippet = _keyword_sentence_count(risk_html or risk_text, risk_keywords)
    out.append(
        _build_item(
            filing=filing,
            metric="risk_factor_keyword_count",
            value=int(risk_count) if isinstance(risk_count, (int, float)) else UNKNOWN,
            section_label="risk_factors",
            source_url=source_url,
            snippet=risk_snippet,
            derived_from=["dossier.extractors.risk_factor_keyword_count"],
            citations=[_citation(source_url, risk_snippet, "risk_factors")] if risk_snippet else [],
        )
    )

    mdna = section_by_label(sections, "md_and_a")
    mdna_text = mdna.text if mdna else ""
    mdna_html = (
        _section_html(raw_html, sections, "md_and_a")
        or _fallback_section_html(raw_html, "md_and_a")
        or mdna_text
    )
    acq_keywords = ["acquisition", "acquired", "business combination", "acquiree", "merger"]
    acq_excluded_phrases = [
        "traffic acquisition",
        "traffic acquisition costs",
        "customer acquisition",
        "cost of acquisition",
        "acquisition cost",
    ]
    acq_mentions, acq_snippet = _filtered_sentence_count(
        mdna_html or mdna_text,
        acq_keywords,
        excluded_phrases=acq_excluded_phrases,
    )
    out.append(
        _build_item(
            filing=filing,
            metric="acquisition_mentions_count",
            value=int(acq_mentions) if isinstance(acq_mentions, (int, float)) else UNKNOWN,
            section_label="md_and_a",
            source_url=source_url,
            snippet=acq_snippet,
            derived_from=["dossier.extractors.acquisition_mentions_count"],
            citations=[_citation(source_url, acq_snippet, "md_and_a")] if acq_snippet else [],
        )
    )

    for metric, amount_metric, keywords in [
        ("deferred_revenue_mention", "deferred_revenue_amount", ["deferred revenue"]),
        ("rpo_mention", "rpo_amount", ["remaining performance obligation", "rpo"]),
    ]:
        amount_value, amount_snippet = _table_value_for_keywords(notes_rows, keywords)
        if amount_value is UNKNOWN:
            amount_value, amount_snippet = _amount_from_text(notes_html or notes_text, keywords)
        if amount_value is UNKNOWN and amount_metric == "deferred_revenue_amount":
            amount_value, amount_snippet = _xbrl_value(
                raw_html,
                [
                    "us-gaap:ContractWithCustomerLiabilityCurrent",
                    "us-gaap:ContractWithCustomerLiability",
                    "msft:ContractWithCustomerLiabilityRevenueDeferred",
                ],
            )
        if amount_value is UNKNOWN and amount_metric == "rpo_amount":
            amount_value, amount_snippet = _xbrl_value(
                raw_html,
                [
                    "us-gaap:RevenueRemainingPerformanceObligation",
                ],
            )
        out.append(
            _build_item(
                filing=filing,
                metric=amount_metric,
                value=amount_value,
                section_label="notes",
                source_url=source_url,
                snippet=amount_snippet,
                derived_from=[f"dossier.extractors.{amount_metric}"],
                citations=[_citation(source_url, amount_snippet, "notes")]
                if amount_snippet
                else [],
            )
        )
        note_blob = notes_html or notes_text
        mention_count = _keyword_mentions(note_blob, keywords)
        if mention_count > 0 or isinstance(amount_value, (int, float)):
            value = 1
        else:
            value = 0 if note_blob else UNKNOWN
        mention_snippet = amount_snippet or (
            _sentence_with_keywords(note_blob, keywords)[:650] if note_blob else ""
        )
        out.append(
            _build_item(
                filing=filing,
                metric=metric,
                value=value,
                section_label="notes",
                source_url=source_url,
                snippet=mention_snippet,
                derived_from=[f"dossier.extractors.{metric}"],
                citations=[_citation(source_url, mention_snippet, "notes")]
                if mention_snippet
                else [],
            )
        )

    segment_names: list[str] = []
    sbc_signal = 0
    customer_pct: float | str = UNKNOWN
    customer_flag = 0
    segment_source_url = source_url
    segment_snippet = ""
    for fact in facts:
        fact_type = str(fact.get("fact_type") or "")
        value_json = fact.get("value_json") or {}
        if fact_type == "segments_signal":
            segment_names.extend(
                [str(x) for x in (value_json.get("segment_names") or []) if str(x).strip()]
            )
            segment_source_url = fact.get("source_url") or source_url
            segment_snippet = fact.get("snippet") or segment_snippet
        if fact_type in {"sbc_dilution_signal", "dilution_risk"}:
            sbc_signal = 1
        if fact_type == "customer_concentration_signal":
            customer_flag = 1
            pct = value_json.get("customer_pct")
            if isinstance(pct, (int, float)):
                customer_pct = float(pct) / 100.0 if float(pct) > 1 else float(pct)
        if fact_type == "shares_outstanding" and shares is UNKNOWN:
            val = value_json.get("value")
            if isinstance(val, (int, float)):
                shares = float(val)
                shares_section_label = "cover_page"
                shares_source_url = fact.get("source_url") or source_url
                shares_snippet = (
                    fact.get("snippet") or "shares outstanding extracted from cover page"
                )
                shares_derived_from = ["extracted_facts.shares_outstanding"]
                shares_citations = (
                    [_citation(shares_source_url, shares_snippet, "cover_page")]
                    if shares_source_url
                    else []
                )

    raw_customer_present, raw_customer_pct, raw_customer_snippet = _customer_concentration_from_raw(
        raw_html
    )

    if customer_flag == 0 and raw_customer_present == 0:
        customer_flag = 0
        customer_pct = UNKNOWN
        customer_snippet = raw_customer_snippet
    elif customer_pct is UNKNOWN:
        pct_value, pct_snippet = _pct_from_text(
            notes_text, ["customer concentration", "major customer", "represented"]
        )
        if isinstance(pct_value, (int, float)):
            customer_pct = pct_value
            customer_flag = 1
            customer_snippet = pct_snippet
        else:
            customer_snippet = ""
    else:
        customer_snippet = "customer concentration signal present"

    if customer_flag == 0:
        if raw_customer_present in (0, 1):
            customer_flag = int(raw_customer_present)
            customer_snippet = raw_customer_snippet
            if isinstance(raw_customer_pct, (int, float)):
                customer_pct = float(raw_customer_pct)

    if not segment_names:
        business_section = section_by_label(sections, "business")
        business_text = business_section.text if business_section else ""
        fallback_segment_names, fallback_segment_snippet = _segment_names_from_text(business_text)
        if not fallback_segment_names:
            business_html = _section_html(raw_html, sections, "business") or _fallback_section_html(
                raw_html, "business"
            )
            fallback_segment_names, fallback_segment_snippet = _segment_names_from_text(
                business_html
            )
        if not fallback_segment_names:
            fallback_segment_names, fallback_segment_snippet = _segment_names_from_text(raw_html)
        if fallback_segment_names:
            segment_names = fallback_segment_names
            segment_snippet = fallback_segment_snippet or segment_snippet

    segment_names = sorted(set(name.strip() for name in segment_names if name.strip()))
    out.append(
        _build_item(
            filing=filing,
            metric="segment_names",
            value=segment_names if segment_names else UNKNOWN,
            section_label="segment_info",
            source_url=segment_source_url,
            snippet=segment_snippet or "; ".join(segment_names[:6]),
            derived_from=["extracted_facts.segments_signal"],
            citations=[
                _citation(
                    segment_source_url,
                    segment_snippet or "; ".join(segment_names[:6]),
                    "segment_info",
                )
            ]
            if segment_names
            else [],
        )
    )

    out.append(
        _build_item(
            filing=filing,
            metric="segment_count",
            value=len(segment_names) if segment_names else UNKNOWN,
            section_label="segment_info",
            source_url=segment_source_url,
            snippet=segment_snippet or "; ".join(segment_names[:6]) if segment_names else "",
            derived_from=["extracted_facts.segments_signal"],
            citations=[
                _citation(
                    segment_source_url,
                    segment_snippet or "; ".join(segment_names[:6]),
                    "segment_info",
                )
            ]
            if segment_names
            else [],
        )
    )
    out.append(
        _build_item(
            filing=filing,
            metric="sbc_dilution_indicator",
            value=sbc_signal if sbc_signal else 0,
            section_label="equity_dilution",
            source_url=source_url,
            snippet="sbc/dilution signal present"
            if sbc_signal
            else "no explicit sbc/dilution signal parsed",
            derived_from=["extracted_facts.sbc_dilution_signal", "extracted_facts.dilution_risk"],
            citations=[],
        )
    )
    out.append(
        _build_item(
            filing=filing,
            metric="customer_concentration_present",
            value=customer_flag if customer_flag in (0, 1) else UNKNOWN,
            section_label="customer_concentration",
            source_url=source_url,
            snippet=customer_snippet,
            derived_from=[
                "extracted_facts.customer_concentration_signal",
                "dossier.extractors.customer_concentration_present",
            ],
            citations=[_citation(source_url, customer_snippet, "notes")]
            if customer_snippet
            else [],
        )
    )
    out.append(
        _build_item(
            filing=filing,
            metric="customer_concentration_pct",
            value=customer_pct if customer_flag else UNKNOWN,
            section_label="customer_concentration",
            source_url=source_url,
            snippet=customer_snippet,
            derived_from=["extracted_facts.customer_concentration_signal"],
            citations=[_citation(source_url, customer_snippet, "notes")]
            if customer_snippet
            else [],
        )
    )
    out.append(
        _build_item(
            filing=filing,
            metric="shares_outstanding",
            value=shares,
            section_label=shares_section_label,
            source_url=shares_source_url,
            snippet=shares_snippet,
            derived_from=shares_derived_from,
            citations=shares_citations,
        )
    )

    return out
