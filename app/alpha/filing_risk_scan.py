"""Filing risk scanner — extracts and classifies risk narratives from 10-K/20-F filings.

Public API
----------
scan_filing_risks(ticker) -> dict
    Returns risk classification for the most recent cached annual filing.
    Always returns a dict with keys:
        competitive_disruption, secular_decline, regulatory_legal,
        customer_concentration, summary, status
    status values: OK | KEYWORD_FALLBACK | NO_FILING | ERROR
"""

from __future__ import annotations

import html as _html_module
import json
import re
from datetime import date
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    InvalidFinancialInputError,
    require_financial_integrity_scope,
    require_unchanged_financial_integrity_scope,
)
from app.config import AppConfig, get_config
from app.db import get_db
from app.llm.providers import get_llm_provider
from app.llm.providers.retry_guard import LLMCostBudgetExceeded, llm_physical_attempt_guard
from app.llm.usage_capture import (
    attach_provider_usage_to_exception,
    attached_provider_usage_records,
    provider_failed_attempt_capture,
    provider_usage_lane,
    provider_usage_records,
    provider_usage_records_from_exception,
    provider_usage_request,
    record_provider_usage,
)
from app.logging import get_logger
from app.research.filing_context import FilingDocument, load_research_filing_context
from app.util.financial_data_access import ANNUAL_CACHED_FILING_FORM_TYPES, latest_filing_row

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RISK_TEXT = 30_000
_MIN_RISK_TEXT = 200
_STALE_ANNUAL_DAYS = 548

# Module-level result cache — keyed by issuer, filing revision, as-of, and LLM
# mode.  A ticker-only cache silently reused an old classification after a new
# annual filing landed and duplicated classifications across security aliases.
# Cleared in tests via _RISK_CACHE.clear().
_RISK_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}

_DIMENSIONS = (
    "competitive_disruption",
    "secular_decline",
    "regulatory_legal",
    "customer_concentration",
)

_UNKNOWN_RESULT = {
    "competitive_disruption": "UNKNOWN",
    "secular_decline": "UNKNOWN",
    "regulatory_legal": "UNKNOWN",
    "customer_concentration": "UNKNOWN",
    "summary": "",
    "status": "NO_FILING",
    "evidence_status": "NO_READABLE_ANNUAL_FILING",
    "warnings": [],
    "source_accession": None,
    "source_form_type": None,
    "source_filing_date": None,
    "source_filing_age_days": None,
    "source_issuer_cik": None,
    "source_content_revision": None,
    "analysis_as_of_date": None,
    "risk_text_chars": 0,
}

# ---------------------------------------------------------------------------
# Keyword lists for fallback classification
# ---------------------------------------------------------------------------

_KEYWORDS: dict[str, list[str]] = {
    "competitive_disruption": [
        "artificial intelligence",
        "generative ai",
        "large language model",
        "machine learning",
        "automat",
        "chatbot",
        "disrupt",
    ],
    "secular_decline": [
        "secular decline",
        "market shrink",
        "obsolete",
        "legacy",
        "declining demand",
        "sunset",
        "end of life",
    ],
    "regulatory_legal": [
        "regulatory",
        "antitrust",
        "litigation",
        "compliance",
        "investigation",
        "consent decree",
        "enforcement action",
    ],
    "customer_concentration": [
        "customer concentration",
        "top 10 customers",
        "significant portion of revenue",
        "single customer",
        "largest customer",
        "revenue concentration",
    ],
}

# ---------------------------------------------------------------------------
# LLM schema
# ---------------------------------------------------------------------------

_RISK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "competitive_disruption": {
            "type": "string",
            "enum": ["HIGH", "MODERATE", "LOW", "NOT_MENTIONED"],
            "description": (
                "HIGH = existential threat from AI/ML/automation explicitly described; "
                "MODERATE = meaningful risk acknowledged; "
                "LOW = minor mention; "
                "NOT_MENTIONED = not discussed."
            ),
        },
        "secular_decline": {
            "type": "string",
            "enum": ["HIGH", "MODERATE", "LOW", "NOT_MENTIONED"],
            "description": (
                "HIGH = core revenue stream in structural decline explicitly stated; "
                "MODERATE = declining segment acknowledged; "
                "LOW = minor concern; "
                "NOT_MENTIONED = not discussed."
            ),
        },
        "regulatory_legal": {
            "type": "string",
            "enum": ["HIGH", "MODERATE", "LOW", "NOT_MENTIONED"],
            "description": (
                "HIGH = existential legal/regulatory overhang (DOJ, SEC, GDPR fines, etc.); "
                "MODERATE = meaningful compliance burden; "
                "LOW = routine disclosures; "
                "NOT_MENTIONED = not discussed."
            ),
        },
        "customer_concentration": {
            "type": "string",
            "enum": ["HIGH", "MODERATE", "LOW", "NOT_MENTIONED"],
            "description": (
                "HIGH = >30% revenue from top 5 customers explicitly stated; "
                "MODERATE = concentration acknowledged; "
                "LOW = minor mention; "
                "NOT_MENTIONED = not discussed."
            ),
        },
        "summary": {
            "type": "string",
            "description": "1–2 sentence plain-English summary of the most material risk factors.",
        },
    },
    "required": [
        "competitive_disruption",
        "secular_decline",
        "regulatory_legal",
        "customer_concentration",
        "summary",
    ],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _latest_readable_annual_document(
    ticker: str,
    *,
    as_of_date: str | None = None,
    issuer_cik: str | None = None,
    aliases: tuple[str, ...] = (),
    issuer_aware: bool = False,
    allow_network_materialization: bool = True,
    connection: Connection | None = None,
    cfg: AppConfig | None = None,
    allowed_filing_roots: tuple[str | Path, ...] = (),
) -> tuple[FilingDocument | None, list[str]]:
    """Return the newest readable annual filing document plus materialization warnings."""
    documents, warnings = _readable_annual_documents(
        ticker,
        as_of_date=as_of_date,
        issuer_cik=issuer_cik,
        aliases=aliases,
        issuer_aware=issuer_aware,
        limit=1,
        allow_network_materialization=allow_network_materialization,
        connection=connection,
        cfg=cfg,
        allowed_filing_roots=allowed_filing_roots,
    )
    return (documents[0] if documents else None), warnings


def _readable_annual_documents(
    ticker: str,
    *,
    as_of_date: str | None = None,
    issuer_cik: str | None = None,
    aliases: tuple[str, ...] = (),
    issuer_aware: bool = False,
    limit: int = 12,
    allow_network_materialization: bool = True,
    connection: Connection | None = None,
    cfg: AppConfig | None = None,
    allowed_filing_roots: tuple[str | Path, ...] = (),
) -> tuple[list[FilingDocument], list[str]]:
    """Return readable annual filings newest first for amendment fallback."""
    context = load_research_filing_context(
        ticker,
        as_of_date=as_of_date or date.today().isoformat(),
        issuer_cik=issuer_cik,
        aliases=aliases,
        issuer_aware=issuer_aware,
        annual_filing_limit=max(1, int(limit)),
        quarters=0,
        include_material_events=False,
        allow_annual_download=allow_network_materialization,
        connection=connection,
        cfg=cfg,
        allowed_filing_roots=allowed_filing_roots,
    )
    documents = [doc for doc in context.ordered_documents if doc.role == "annual"]
    return documents, list(context.warnings)


def _find_latest_annual_path(ticker: str) -> tuple[str | None, str | None]:
    """Return (local_path, form_type) for the most recent readable cached annual filing."""
    try:
        document, _warnings = _latest_readable_annual_document(ticker)
        if document and document.local_path:
            return document.local_path, document.form_type
    except Exception as exc:
        logger.warning(
            "filing_risk_scan: readable annual materialization failed for %s: %s", ticker, exc
        )

    # Compatibility fallback for callers/tests that only need the original
    # cheap lookup behavior and cannot materialize from cache metadata.
    try:
        with get_db() as conn:
            row = latest_filing_row(
                conn,
                ticker,
                columns=("local_path", "form_type"),
                form_types=ANNUAL_CACHED_FILING_FORM_TYPES,
                require_local_path=True,
            )
        if row:
            return row[0], row[1]
    except Exception as exc:
        logger.warning("filing_risk_scan: DB query failed for %s: %s", ticker, exc)
    return None, None


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode entities, collapse whitespace."""
    # Remove script/style blocks
    text = re.sub(
        r"<(script|style)[^>]*>.*?</(script|style)>", " ", html, flags=re.DOTALL | re.IGNORECASE
    )
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities (&amp; &lt; &#160; etc.)
    text = _html_module.unescape(text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_RISK_PHRASE_PATTERN = r"R\s*I\s*S\s*K\s+Factors?"
_TOC_HEADING_RE = re.compile(r"\bItem\s+\d+[A-Z]?\b", re.IGNORECASE)
_REFERENCE_LEAD_RE = re.compile(
    r"^[\s\"“”'’.,;:\-–—]*(?:and|of|included|elsewhere|in)\b",
    re.IGNORECASE,
)


def _candidate_section(raw: str, *, start: int, body_start: int, end_pat: re.Pattern[str]) -> str:
    segment = raw[body_start:]
    end_match = end_pat.search(segment)
    if end_match:
        segment = segment[: end_match.start()]
    return segment.strip()


def _looks_like_toc_segment(segment: str) -> bool:
    if len(segment) >= _MIN_RISK_TEXT:
        return False
    return len(_TOC_HEADING_RE.findall(segment)) >= 1


def _extract_risk_text(html: str, form_type: str) -> str:
    """Extract the risk factors section text from filing HTML.

    10-K: Item 1A → Item 1B / Item 1C / Item 2
    20-F (and 40-F): "Risk Factors" (Item 3.D) → Item 4
    """
    raw = _strip_html(html)

    if form_type.startswith("10-K"):
        start_pat = re.compile(
            rf"\bItem\s+1A\b[\s\.\:\-–—]*[\"“”']?\s*{_RISK_PHRASE_PATTERN}",
            re.IGNORECASE,
        )
        end_pat = re.compile(
            r"\bItem\s+1[BC]\b[\s\.\:\-–—]|\bItem\s+2\b[\s\.\:\-–—]", re.IGNORECASE
        )
    else:
        # 20-F: "Risk Factors" section (may appear as "D. Risk Factors" or just "Risk Factors")
        start_pat = re.compile(
            rf"(?:\bItem\s+3\b[\s\.\:\-–—]*D[\s\.\:\-–—]*)?(?:\bD[\s\.\:\-–—]+)?{_RISK_PHRASE_PATTERN}",
            re.IGNORECASE,
        )
        end_pat = re.compile(r"\bItem\s+4\b[\s\.\:\-–—]", re.IGNORECASE)

    matches = list(start_pat.finditer(raw))
    if not matches:
        logger.debug("filing_risk_scan: risk section start not found (form_type=%s)", form_type)
        return ""

    fallback = ""
    best_narrative = ""
    for start_match in matches:
        segment = _candidate_section(
            raw, start=start_match.start(), body_start=start_match.end(), end_pat=end_pat
        )
        if len(segment) > len(fallback):
            fallback = segment
        if _looks_like_toc_segment(segment):
            continue
        if _REFERENCE_LEAD_RE.search(segment[:180]):
            continue
        if len(segment) > len(best_narrative):
            best_narrative = segment
        if len(segment) >= _MIN_RISK_TEXT:
            return segment[:_MAX_RISK_TEXT]

    if best_narrative:
        return best_narrative[:_MAX_RISK_TEXT]
    return fallback[:_MAX_RISK_TEXT] if len(fallback) >= _MIN_RISK_TEXT else ""


def _metadata_from_document(
    *,
    document: FilingDocument | None,
    warnings: list[str],
    evidence_status: str,
    risk_text_chars: int = 0,
    analysis_as_of_date: str | None = None,
) -> dict[str, Any]:
    normalized_warnings = list(dict.fromkeys(warnings))
    source_age_days: int | None = None
    effective_as_of = analysis_as_of_date or date.today().isoformat()
    if document and document.filing_date:
        try:
            source_age_days = (
                date.fromisoformat(effective_as_of) - date.fromisoformat(str(document.filing_date))
            ).days
        except ValueError:
            source_age_days = None
    if (
        evidence_status == "READABLE_RISK_SECTION"
        and source_age_days is not None
        and source_age_days > _STALE_ANNUAL_DAYS
    ):
        evidence_status = "STALE_READABLE_RISK_SECTION"
        normalized_warnings.append(f"stale_annual_filing:{source_age_days}d")
    return {
        "evidence_status": evidence_status,
        "warnings": list(dict.fromkeys(normalized_warnings)),
        "source_accession": document.accession if document else None,
        "source_form_type": document.form_type if document else None,
        "source_filing_date": document.filing_date if document else None,
        "source_filing_age_days": source_age_days,
        "source_issuer_cik": document.cik if document else None,
        "source_content_revision": document.content_revision if document else None,
        "analysis_as_of_date": effective_as_of,
        "risk_text_chars": int(risk_text_chars),
    }


def _count_keyword_hits(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    return sum(lower.count(kw.lower()) for kw in keywords)


def _classify_count(count: int) -> str:
    if count >= 10:
        return "HIGH"
    if count >= 3:
        return "MODERATE"
    if count >= 1:
        return "LOW"
    return "NOT_MENTIONED"


def _keyword_fallback(risk_text: str) -> dict[str, Any]:
    """Keyword-based heuristic fallback when LLM is unavailable."""
    result: dict[str, Any] = {}
    for dim, keywords in _KEYWORDS.items():
        hits = _count_keyword_hits(risk_text, keywords)
        result[dim] = _classify_count(hits)

    # Build a terse summary from the highest-rated dimensions
    high_dims = [d for d in _DIMENSIONS if result.get(d) == "HIGH"]
    moderate_dims = [d for d in _DIMENSIONS if result.get(d) == "MODERATE"]
    if high_dims:
        result["summary"] = f"High-risk dimensions (keyword scan): {', '.join(high_dims)}."
    elif moderate_dims:
        result["summary"] = f"Moderate-risk dimensions (keyword scan): {', '.join(moderate_dims)}."
    else:
        result["summary"] = "No prominent risk keywords detected."

    result["status"] = "KEYWORD_FALLBACK"
    return result


def _database_cache_scope(
    connection: Connection | None,
    *,
    cfg: AppConfig | None = None,
) -> str:
    """Return a stable database identity for process-local risk caching."""

    if connection is not None:
        try:
            rows = connection.execute("PRAGMA database_list").fetchall()
            for row in rows:
                if str(row[1] or "") == "main" and str(row[2] or "").strip():
                    return str(Path(str(row[2])).expanduser().resolve())
        except Exception:
            pass
        return f"connection:{id(connection)}"
    return str(Path((cfg or get_config()).db_path).expanduser().resolve())


def _provider_max_output_tokens(provider: Any) -> int:
    cfg = getattr(provider, "cfg", None)
    provider_name = str(getattr(provider, "provider_name", "") or "").strip().lower()
    field_name = (
        "anthropic_max_output_tokens"
        if provider_name == "anthropic"
        else "openai_max_output_tokens"
    )
    configured = getattr(cfg, field_name, None)
    if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
        return configured
    return 4000


def _llm_classify_risks(
    risk_text: str,
    ticker: str,
    *,
    provider: Any,
    integrity_scope: FinancialIntegrityScope | None,
    run_as_of_date: str,
    expected_scope_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Send risk text to LLM for structured classification."""
    prompt = (
        f"You are a financial risk analyst. Analyze the following Risk Factors section from a "
        f"public company filing (ticker: {ticker}) and classify each risk dimension.\n\n"
        "Definitions:\n"
        "- competitive_disruption HIGH: existential threat from AI/ML/automation explicitly described\n"
        "- secular_decline HIGH: core revenue stream in structural decline\n"
        "- regulatory_legal HIGH: existential legal/regulatory overhang\n"
        "- customer_concentration HIGH: >30% revenue from top 5 customers stated\n\n"
        "For each dimension choose: HIGH, MODERATE, LOW, or NOT_MENTIONED.\n"
        "Then write a 1-2 sentence summary of the most material risks.\n\n"
        f"--- RISK FACTORS TEXT ---\n{risk_text}\n--- END ---\n\n"
        "Return a JSON object matching the required schema."
    )

    scope = integrity_scope or FinancialIntegrityScope(
        context=f"filing_risk_scan:{ticker}",
        run_as_of_date=run_as_of_date,
        packets=(),
    )
    initial = require_financial_integrity_scope(scope)
    expected_fingerprint = (
        str(expected_scope_fingerprint) if expected_scope_fingerprint else initial.scope_fingerprint
    )

    def require_exact_scope(_attempt: dict[str, Any] | None = None) -> None:
        require_unchanged_financial_integrity_scope(
            scope,
            expected_scope_fingerprint=expected_fingerprint,
        )

    schema_name = "filing_risk_scan_v1"
    max_output_tokens = _provider_max_output_tokens(provider)
    failed_attempts: list[dict[str, Any]] = []
    successful_attempts: list[dict[str, Any]] = []
    try:
        with (
            provider_usage_lane(f"filing_risk:{ticker}"),
            provider_usage_request(
                provider=provider,
                prompt=prompt,
                schema=_RISK_SCHEMA,
                schema_name=schema_name,
                max_output_tokens=max_output_tokens,
            ) as request_kwargs,
        ):
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
                        schema=_RISK_SCHEMA,
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
                attach_provider_usage_to_exception(
                    exc,
                    [*failed_attempts, *successful_attempts],
                )
                try:
                    require_exact_scope()
                except InvalidFinancialInputError as integrity_exc:
                    attach_provider_usage_to_exception(
                        integrity_exc,
                        [*failed_attempts, *successful_attempts],
                    )
                    raise integrity_exc from exc
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
    except Exception as exc:
        attach_provider_usage_to_exception(
            exc,
            [*failed_attempts, *successful_attempts],
        )
        try:
            require_exact_scope()
        except InvalidFinancialInputError as integrity_exc:
            attach_provider_usage_to_exception(
                integrity_exc,
                [*failed_attempts, *successful_attempts],
            )
            raise integrity_exc from exc
        raise
    payload["status"] = "OK"
    usage_records = [*failed_attempts, *successful_attempts]
    payload["provider_usage"] = usage_records
    payload["cost_usd"] = round(
        sum(float(item.get("cost_estimate_usd") or 0.0) for item in usage_records),
        6,
    )
    return payload


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_filing_risks(
    ticker: str,
    *,
    use_llm: bool = True,
    as_of_date: str | None = None,
    issuer_cik: str | None = None,
    aliases: tuple[str, ...] = (),
    issuer_aware: bool = False,
    allow_network_materialization: bool = True,
    integrity_scope: FinancialIntegrityScope | None = None,
    connection: Connection | None = None,
    cfg: AppConfig | None = None,
    allowed_filing_roots: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Classify filing risk dimensions for *ticker*.

    Returns a dict with keys:
        competitive_disruption, secular_decline, regulatory_legal,
        customer_concentration, summary, status

    status values:
        OK               — LLM classified successfully
        KEYWORD_FALLBACK — LLM disabled or failed; keyword heuristics used
        NO_FILING        — No cached annual filing found (or risk text too short)
        ERROR            — Unexpected error during processing
    """
    ticker_key = ticker.upper()
    effective_as_of = as_of_date or date.today().isoformat()
    authorized_scope: FinancialIntegrityScope | None = None
    authorized_scope_fingerprint = ""

    # --- 1. Find/read filing --------------------------------------------------
    try:
        documents, filing_warnings = _readable_annual_documents(
            ticker_key,
            as_of_date=effective_as_of,
            issuer_cik=issuer_cik,
            aliases=aliases,
            issuer_aware=issuer_aware,
            allow_network_materialization=allow_network_materialization,
            connection=connection,
            cfg=cfg,
            allowed_filing_roots=allowed_filing_roots,
        )
    except Exception as exc:
        logger.warning(
            "filing_risk_scan: annual filing materialization failed for %s: %s", ticker_key, exc
        )
        documents, filing_warnings = [], [f"annual_filing_materialization_failed:{exc}"]

    if not documents:
        result = {
            **_UNKNOWN_RESULT,
            "status": "NO_FILING",
            **_metadata_from_document(
                document=None,
                warnings=filing_warnings,
                evidence_status="NO_READABLE_ANNUAL_FILING",
                analysis_as_of_date=effective_as_of,
            ),
        }
        # Missing evidence is deliberately not cached: a resumed repair can
        # ingest the filing later in the same process and must see it.
        return result

    newest_document = documents[0]
    candidate_documents = list(documents)
    if issuer_aware and str(newest_document.form_type or "").upper().endswith("/A"):
        full_filings = [
            item for item in documents[1:] if not str(item.form_type or "").upper().endswith("/A")
        ]
        other_amendments = [item for item in documents[1:] if item not in full_filings]
        candidate_documents = [newest_document, *full_filings, *other_amendments]

    document = newest_document
    risk_text = ""
    extraction_warnings: list[str] = []
    for candidate in candidate_documents:
        candidate_text = _extract_risk_text(candidate.html, candidate.form_type or "10-K")
        if len(candidate_text) >= _MIN_RISK_TEXT:
            document = candidate
            risk_text = candidate_text
            if candidate.accession != newest_document.accession:
                extraction_warnings.append(
                    "annual_risk_section_fallback:"
                    f"{newest_document.accession}->{candidate.accession}"
                )
            break
        extraction_warnings.append(f"risk_section_unavailable:{candidate.accession}")
    filing_warnings = list(dict.fromkeys([*filing_warnings, *extraction_warnings]))

    content_revision = str(document.content_revision or "").strip()
    if not content_revision:
        content_revision = "unversioned"
    issuer_key = str(document.cik or issuer_cik or ticker_key).strip().lstrip("0") or "0"
    database_scope = _database_cache_scope(connection, cfg=cfg)
    base_cache_key = (
        database_scope,
        issuer_key,
        str(document.accession or ""),
        effective_as_of,
        content_revision,
    )

    # --- 2. Extract risk section ----------------------------------------------
    if len(risk_text) < _MIN_RISK_TEXT:
        metadata = _metadata_from_document(
            document=document,
            warnings=filing_warnings,
            evidence_status="RISK_SECTION_NOT_FOUND",
            risk_text_chars=len(risk_text),
            analysis_as_of_date=effective_as_of,
        )
        result = {
            **_UNKNOWN_RESULT,
            "status": "NO_FILING",
            "summary": "Risk section too short or not found.",
            **metadata,
        }
        _RISK_CACHE[(*base_cache_key, "no_risk_section")] = result
        return result

    metadata = _metadata_from_document(
        document=document,
        warnings=filing_warnings,
        evidence_status="READABLE_RISK_SECTION",
        risk_text_chars=len(risk_text),
        analysis_as_of_date=effective_as_of,
    )

    # --- 3. Classify ----------------------------------------------------------
    try:
        provider = get_llm_provider()
        if not use_llm or provider.provider_name == "disabled":
            cache_key = (*base_cache_key, "deterministic")
            if cache_key in _RISK_CACHE:
                return _RISK_CACHE[cache_key]
            result = _keyword_fallback(risk_text)
        else:
            scope = integrity_scope or FinancialIntegrityScope(
                context=f"filing_risk_scan:{ticker_key}",
                run_as_of_date=effective_as_of,
                packets=(),
            )
            gate_result = require_financial_integrity_scope(scope)
            authorized_scope = scope
            authorized_scope_fingerprint = gate_result.scope_fingerprint
            cache_key = (
                *base_cache_key,
                "llm",
                gate_result.scope_fingerprint,
            )
            if cache_key in _RISK_CACHE:
                return _RISK_CACHE[cache_key]
            result = _llm_classify_risks(
                risk_text,
                ticker_key,
                provider=provider,
                integrity_scope=scope,
                run_as_of_date=effective_as_of,
                expected_scope_fingerprint=gate_result.scope_fingerprint,
            )
    except (InvalidFinancialInputError, LLMCostBudgetExceeded):
        raise
    except Exception as exc:
        failed_provider_usage = attached_provider_usage_records(exc)
        if authorized_scope is not None:
            require_unchanged_financial_integrity_scope(
                authorized_scope,
                expected_scope_fingerprint=authorized_scope_fingerprint,
            )
        logger.warning(
            "filing_risk_scan: classification failed for %s (%s); using keyword fallback",
            ticker_key,
            exc,
        )
        try:
            cache_key = (*base_cache_key, "provider_error")
            result = _keyword_fallback(risk_text)
            if failed_provider_usage:
                result["provider_usage"] = failed_provider_usage
                result["cost_usd"] = round(
                    sum(
                        float(item.get("cost_estimate_usd") or 0.0)
                        for item in failed_provider_usage
                    ),
                    6,
                )
        except Exception as exc2:
            logger.error(
                "filing_risk_scan: keyword fallback also failed for %s: %s", ticker_key, exc2
            )
            result = {**_UNKNOWN_RESULT, "status": "ERROR", "summary": str(exc2)}

    if authorized_scope is not None:
        require_unchanged_financial_integrity_scope(
            authorized_scope,
            expected_scope_fingerprint=authorized_scope_fingerprint,
        )
    result.update(metadata)
    _RISK_CACHE[cache_key] = result
    return result
