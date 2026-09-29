"""Analyst evidence bundle builder (Redirect Task 2).

Populates the AnalysisEvidenceBundle contract from the same sources
run_deep_research currently uses, including annual, quarterly, and
material-event SEC filings from the canonical filing context.

Temporal coherence rule: effective_as_of_date is frozen at the start
of the call and never updated from any data source. Every IO step
that accepts a date argument takes effective_as_of_date, with one
deliberate exception: _load_scorecard receives the caller's original
as_of_date so its "None → latest row" semantic keeps working. The
scorecard's returned resolved_date is not used as the bundle anchor;
it is only consulted for the scorecard_stale diagnostic.

API contract: malformed as_of_date input and financial-integrity authorization
failures raise terminally. Other data-level failures (DB exceptions, JSON
parse errors, missing files, network errors) degrade gracefully by appending
to bundle.warnings and continuing.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Callable

from app.alpha.filing_risk_scan import scan_filing_risks
from app.alpha.solvency_scanner import assess_solvency
from app.analyst.evidence_bundle import (
    AnalysisEvidenceBundle,
    BundleEvent,
    BundleFiling,
    ValuationSnapshot,
)
from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    InvalidFinancialInputError,
    require_financial_integrity_scope,
)
from app.autonomous.v1_financial_context import bind_v1_financial_scope
from app.dossier.collector import DossierFiling, collect_10k_docket, read_filing_text
from app.dossier.sections import (
    event_category_for_8k_section,
    item_code_for_8k_section,
    segment_10k_sections,
    segment_10q_sections,
    segment_8k_sections,
)
from app.util.html_strip import strip_html
from app.ingest.facts_writer import ensure_all_facts
from app.llm.providers import get_llm_provider
from app.llm.providers.retry_guard import LLMCostBudgetExceeded
from app.research.filing_context import FilingDocument, load_research_filing_context
from app.research.current_event_context import CurrentEventDocument, load_current_event_context
from app.research.adapters.sec_exhibits import (  # noqa: F401
    SecExhibitsAdapter,  # compatibility alias for monkeypatch-based tests
)
from app.research.deep_research import (
    _compute_tensions_from_scorecard,
    _load_scorecard,
)
from app.research.source_quality import classify_source_quality
from app.valuation.valuation_writer import ensure_valuation
from app.valuation.mos_conventions import graham_value_from_textbook_discount

logger = logging.getLogger(__name__)

_ANNUAL_FORMS = frozenset({"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"})
_QUARTERLY_FORMS = frozenset({"10-Q", "10-Q/A"})
_MATERIAL_EVENT_FORMS = frozenset({"8-K", "8-K/A"})


# Dossier section labels (from app.dossier.sections.SECTION_PATTERNS) mapped
# to the canonical keys defined in the Task 1 analyst contract spec.
# Dropped labels return None.
_SECTION_MAP: dict[str, str] = {
    "md_and_a": "mda",
    "risk_factors": "risk_factors",
    "notes": "fin_notes",
    "business": "business",
    # "financial_statements" and "segment_info" are intentionally dropped
    # per spec — no canonical key consumes them in Task 2.
}


def _validate_as_of_date(as_of_date: str | None) -> None:
    """Validate as_of_date input. Raises ValueError on malformed non-None strings.

    None is allowed. Valid ISO-8601 date strings (YYYY-MM-DD) are allowed.
    Any other string raises ValueError with a message containing the offending
    value. Called once at the top of the builder before any IO — malformed
    dates are programmer errors, not runtime degradation.
    """
    if as_of_date is None:
        return
    if not isinstance(as_of_date, str) or not as_of_date:
        raise ValueError(f"as_of_date must be None or an ISO date string, got: {as_of_date!r}")
    try:
        date.fromisoformat(as_of_date)
    except ValueError as exc:
        raise ValueError(
            f"as_of_date must be a valid ISO-8601 date (YYYY-MM-DD), got: {as_of_date!r}"
        ) from exc


def _resolve_as_of_date(as_of_date: str | None) -> str:
    """Return as_of_date when provided, otherwise today's ISO string.

    Assumes as_of_date has already been validated by _validate_as_of_date.
    Called once at the top of the builder; the returned value is frozen
    for the rest of the call (the temporal coherence rule).
    """
    if as_of_date is None:
        return date.today().isoformat()
    return as_of_date


def _is_live_as_of_date(as_of_date: str | None) -> bool:
    """Return True when the caller is asking for live-today analysis.

    None is treated as live. An explicit date equal to today's ISO
    string is also live. Any other valid-but-not-today string is
    historical. Malformed strings return False defensively (the
    builder should catch these via _validate_as_of_date first).

    Gate input is the caller's ORIGINAL as_of_date, not effective_as_of_date.
    """
    if as_of_date is None:
        return True
    try:
        date.fromisoformat(as_of_date)
    except (ValueError, TypeError):
        return False
    return as_of_date == date.today().isoformat()


def _filing_risk_provider_enabled() -> bool:
    """Return whether the filing-risk path could make a paid LLM call."""

    try:
        provider = get_llm_provider()
    except Exception:
        # Provider discovery uncertainty must not authorize an unscoped path.
        return True
    if str(getattr(provider, "provider_name", "")).strip().lower() == "disabled":
        return False
    enabled = getattr(provider, "enabled", None)
    if not callable(enabled):
        return True
    try:
        return bool(enabled())
    except Exception:
        return True


def _reject_missing_filing_risk_packet(
    *,
    ticker: str,
    as_of_date: str,
) -> None:
    """Fail closed before filing-risk analysis can make a paid call."""

    require_financial_integrity_scope(
        FinancialIntegrityScope(
            context=f"analysis_bundle_filing_risk:{ticker}:missing_packet",
            run_as_of_date=as_of_date,
            packets=(),
        )
    )
    raise AssertionError("empty financial-integrity scope unexpectedly passed")


def _canonical_section(dossier_label: str) -> str | None:
    """Map a dossier section label to a canonical analyst key.

    Unknown or dropped labels return None.
    """
    event_category = event_category_for_8k_section(dossier_label)
    if event_category is not None:
        return event_category
    return _SECTION_MAP.get(dossier_label)


def _build_valuation_snapshot(
    scorecard: dict[str, Any] | None,
    tensions: dict[str, Any],
    solvency: Any | None,
    filing_risk: dict[str, Any] | None,
) -> ValuationSnapshot:
    """Map scorecard + tensions + solvency + filing_risk into a ValuationSnapshot.

    Every field is optional. Missing keys or None inputs produce None fields.

    Graham value is back-computed from price and the TEXTBOOK Graham discount
    stored at scorecard["discounts"]["graham"]: d = (iv - price)/iv, so
    iv = price/(1 - d) (audit: graham-discount-inversion). The -1 sentinel
    ("not computable") is handled by the shared helper.
    """
    pricing_detail: dict[str, Any] = {}
    quality_ctx: dict[str, Any] = {}
    discounts: dict[str, Any] = {}
    if scorecard is not None:
        pricing_detail = scorecard.get("pricing_zone_detail") or {}
        quality_ctx = scorecard.get("quality_context") or {}
        discounts = scorecard.get("discounts") or {}

    current_price = pricing_detail.get("current_price")
    graham_value = graham_value_from_textbook_discount(
        current_price if isinstance(current_price, (int, float)) else None,
        discounts.get("graham") if isinstance(discounts.get("graham"), (int, float)) else None,
    )

    gate_action = pricing_detail.get("gate_action") or quality_ctx.get("gate_action")

    solvency_status: str | None = None
    if solvency is not None:
        solvency_status = getattr(solvency, "solvency_risk", None)

    filing_risk_status: str | None = None
    if filing_risk is not None:
        filing_risk_status = filing_risk.get("status")

    return ValuationSnapshot(
        current_price=current_price,
        market_cap=pricing_detail.get("market_cap"),
        dcf_base=pricing_detail.get("dcf_base"),
        epv_adjusted=pricing_detail.get("epv_adjusted"),
        graham_value=graham_value,
        methods_agree=tensions.get("methods_agree"),
        tension_type=tensions.get("tension_type"),
        gate_action=gate_action,
        solvency_status=solvency_status,
        filing_risk_status=filing_risk_status,
    )


def _build_bundle_filings(
    dossier_filings: list[DossierFiling],
    reader: Callable[[DossierFiling], str | None] = read_filing_text,
) -> tuple[list[BundleFiling], list[str]]:
    """Map annual-form DossierFilings to BundleFilings with canonical sections.

    Returns (filings, warnings). The helper owns its warning channel because
    read_filing_text signals failure via None, not an exception.

    - Reader returning None emits "filing_text_unavailable:{accession}".
    - Filings whose sections all map to None (dropped) emit
      "filing_sections_empty:{accession}".
    """
    filings: list[BundleFiling] = []
    warnings: list[str] = []

    for dossier in dossier_filings:
        form_type = (dossier.form_type or "").upper()
        if form_type not in _ANNUAL_FORMS:
            continue

        text = reader(dossier)
        if text is None:
            warnings.append(f"filing_text_unavailable:{dossier.accession}")
            continue

        bundle_filing, warning = _build_bundle_filing_record(
            text=text,
            form_type=dossier.form_type,
            filing_date=dossier.filing_date,
            accession=dossier.accession,
            role="annual",
        )
        if bundle_filing is not None:
            filings.append(bundle_filing)
        if warning is not None:
            warnings.append(warning)

    return filings, warnings


def _build_bundle_filing_record(
    *,
    text: str,
    form_type: str,
    filing_date: str,
    accession: str | None,
    role: str,
) -> tuple[BundleFiling | None, str | None]:
    upper_form_type = str(form_type or "").upper()
    if upper_form_type in _QUARTERLY_FORMS:
        spans = segment_10q_sections(text)
    elif upper_form_type in _MATERIAL_EVENT_FORMS:
        spans = segment_8k_sections(text)
    else:
        spans = segment_10k_sections(text)

    sections_included: list[str] = []
    section_text: dict[str, str] = {}
    for span in spans:
        canonical = _canonical_section(getattr(span, "section_label", ""))
        if canonical is None:
            continue
        text_value = strip_html(getattr(span, "text", ""))
        if canonical in section_text:
            if text_value and text_value not in section_text[canonical]:
                section_text[canonical] = f"{section_text[canonical]}\n\n{text_value}".strip()
            continue
        sections_included.append(canonical)
        section_text[canonical] = text_value

    if not sections_included:
        return None, f"filing_sections_empty:{accession}"

    return (
        BundleFiling(
            form_type=form_type,
            filing_date=filing_date,
            accession=accession,
            role=role,
            sections_included=sections_included,
            section_text=section_text,
        ),
        None,
    )


def _build_bundle_context_filings(
    documents: list[FilingDocument],
) -> tuple[list[BundleFiling], list[str]]:
    filings: list[BundleFiling] = []
    warnings: list[str] = []

    for document in documents:
        bundle_filing, warning = _build_bundle_filing_record(
            text=document.html,
            form_type=document.form_type,
            filing_date=document.filing_date,
            accession=document.accession,
            role=document.role,
        )
        if bundle_filing is not None:
            filings.append(bundle_filing)
        if warning is not None:
            warnings.append(warning)

    return filings, warnings


def _build_recent_events(
    documents: list[FilingDocument],
) -> list[BundleEvent]:
    """Map parsed material-event filings into BundleEvent compatibility entries."""
    events: list[BundleEvent] = []
    for document in documents:
        spans = segment_8k_sections(document.html)
        for span in spans:
            label = getattr(span, "section_label", "")
            item_code = item_code_for_8k_section(label)
            event_category = event_category_for_8k_section(label)
            if item_code is None or event_category is None:
                continue
            title = f"8-K Item {item_code} — {event_category.replace('_', ' ')}"
            summary = strip_html(getattr(span, "text", ""))[:500]
            events.append(
                BundleEvent(
                    source_type="8-K",
                    published_at=document.filing_date,
                    title=title,
                    summary=summary,
                    source_url=document.primary_doc_url,
                    materiality=None,
                    accession=document.accession,
                    item_code=item_code,
                    event_category=event_category,
                    source_quality=classify_source_quality(
                        source_type="8-K",
                        source_url=document.primary_doc_url,
                        published_at=document.filing_date,
                    ),
                )
            )

    return events


def _build_current_event_bundle_events(
    documents: list[CurrentEventDocument],
) -> list[BundleEvent]:
    events: list[BundleEvent] = []
    for document in documents:
        events.append(
            BundleEvent(
                source_type=document.source_type,
                published_at=document.published_at,
                title=document.title,
                summary=document.summary[:500],
                source_url=document.source_url,
                materiality=None,
                source_quality=document.source_quality,
            )
        )
    return events


def _assemble_bundle_from_scorecard(
    *,
    ticker: str,
    as_of_date: str | None,
    effective_as_of_date: str,
    is_live: bool,
    scorecard: dict[str, Any] | None,
    financial_packet: Any | None,
    years: int,
    quarters: int,
    freshness_window_days: int,
    warnings: list[str],
) -> AnalysisEvidenceBundle:
    """Given a scorecard (or None) and frozen temporal context, assemble
    the rest of the bundle: docket → scanners → tensions → filings → events.

    This helper is called by both ``build_analysis_evidence_bundle`` (after
    it runs the slow ensure_all_facts + ensure_valuation + _load_scorecard
    chain) and ``build_analysis_evidence_bundle_from_cached_scorecard``
    (when the caller passes the scorecard directly and wants to skip the
    slow chain).

    The ``warnings`` list is mutated in place to accumulate additional
    warnings from the IO this helper runs.

    Parameters
    ----------
    ticker:
        Ticker symbol (case-insensitive; upper-cased internally).
    as_of_date:
        The caller's original ``as_of_date`` (may be ``None``). Used only
        to decide the live-vs-historical gate via ``is_live`` — the actual
        date used for all IO is ``effective_as_of_date``.
    effective_as_of_date:
        The frozen temporal anchor. Every IO step that accepts a date
        uses this value. Per the temporal coherence rule, it is never
        mutated after the caller computes it.
    is_live:
        Result of ``_is_live_as_of_date(as_of_date)``. Drives the
        live-vs-historical gate for the lookahead-leaky scanners.
    scorecard:
        Scorecard dict (from the valuations.outputs_json column), or
        ``None`` if the caller couldn't load one. Tensions and the
        valuation snapshot derive from this.
    financial_packet:
        The canonical V1 packet authorized by the parent run. Required before
        a live filing-risk provider call; deterministic provider-disabled
        scans remain available without it.
    years, quarters, freshness_window_days:
        Same semantics as the public API.
    warnings:
        Mutable list; this helper appends to it.
    """
    upper = ticker.upper()

    # Docket (reads filings from local dossier cache; does NOT call SEC)
    docket: list[DossierFiling] = []
    try:
        docket = collect_10k_docket(
            ticker=upper,
            as_of_date=effective_as_of_date,
            years_back=years,
        )
        if not docket:
            warnings.append("no_annual_filings_available")
    except Exception as exc:
        logger.warning("bundle_builder: collect_10k_docket failed for %s: %s", upper, exc)
        warnings.append("filing_docket_failed")

    quarterly_documents: list[FilingDocument] = []
    material_event_documents: list[FilingDocument] = []
    current_event_documents: list[CurrentEventDocument] = []
    try:
        filing_context = load_research_filing_context(
            upper,
            as_of_date=effective_as_of_date,
            quarters=quarters,
        )
        quarterly_documents = [
            document for document in filing_context.documents if document.role == "quarterly"
        ]
        material_event_documents = [
            document for document in filing_context.documents if document.role == "material_event"
        ]
        warnings.extend(warning for warning in filing_context.warnings if warning not in warnings)
    except Exception as exc:
        logger.warning("bundle_builder: filing context failed for %s: %s", upper, exc)
        warnings.append("filing_context_failed")

    try:
        current_event_context = load_current_event_context(
            upper,
            as_of_date=effective_as_of_date,
        )
        current_event_documents = list(current_event_context.documents)
        warnings.extend(
            warning for warning in current_event_context.warnings if warning not in warnings
        )
    except Exception as exc:
        logger.warning("bundle_builder: current event context failed for %s: %s", upper, exc)
        warnings.append("current_event_context_failed")

    # Solvency scanner (live-gated)
    solvency: Any | None = None
    if is_live:
        try:
            solvency = assess_solvency(upper)
        except Exception as exc:
            logger.warning("bundle_builder: assess_solvency failed for %s: %s", upper, exc)
            warnings.append("solvency_scan_failed")
    else:
        warnings.append("solvency_gated_on_live_as_of_date")

    # Filing-risk scanner (live-gated)
    filing_risk: dict[str, Any] | None = None
    if is_live:
        try:
            if not _filing_risk_provider_enabled():
                filing_risk = scan_filing_risks(
                    upper,
                    use_llm=False,
                    as_of_date=effective_as_of_date,
                    allow_network_materialization=False,
                )
            else:
                if financial_packet is None:
                    _reject_missing_filing_risk_packet(
                        ticker=upper,
                        as_of_date=effective_as_of_date,
                    )
                bound_scope = bind_v1_financial_scope(
                    context=f"analysis_bundle_filing_risk:{upper}",
                    run_as_of_date=effective_as_of_date,
                    packets=(financial_packet,),
                    scenarios=(),
                )
                bound_scope.require()
                bound_tickers = [
                    str(packet.get("ticker") or "").strip().upper()
                    for packet in bound_scope.packets
                    if isinstance(packet, dict)
                ]
                if bound_tickers != [upper]:
                    _reject_missing_filing_risk_packet(
                        ticker=upper,
                        as_of_date=effective_as_of_date,
                    )
                filing_risk = scan_filing_risks(
                    upper,
                    use_llm=True,
                    as_of_date=effective_as_of_date,
                    allow_network_materialization=False,
                    integrity_scope=FinancialIntegrityScope(
                        context=bound_scope.context,
                        run_as_of_date=bound_scope.run_as_of_date,
                        packets=bound_scope.packets,
                    ),
                )
        except (InvalidFinancialInputError, LLMCostBudgetExceeded):
            raise
        except Exception as exc:
            logger.warning("bundle_builder: scan_filing_risks failed for %s: %s", upper, exc)
            warnings.append("filing_risk_scan_failed")
    else:
        warnings.append("filing_risk_gated_on_live_as_of_date")

    # Tensions (only if scorecard exists)
    tensions: dict[str, Any] = {}
    if scorecard is not None:
        try:
            quality_ctx = scorecard.get("quality_context") or {}
            tensions = _compute_tensions_from_scorecard(scorecard, quality_ctx)
        except Exception as exc:
            logger.warning("bundle_builder: tension analysis failed for %s: %s", upper, exc)
            warnings.append("tension_analysis_failed")

    # Filing text + sections (helper owns its own per-filing warnings)
    filings, filing_warnings = _build_bundle_filings(docket, reader=read_filing_text)
    warnings.extend(filing_warnings)
    quarterly_filings, quarterly_warnings = _build_bundle_context_filings(quarterly_documents)
    filings.extend(quarterly_filings)
    warnings.extend(quarterly_warnings)
    material_event_filings, material_event_warnings = _build_bundle_context_filings(
        material_event_documents
    )
    filings.extend(material_event_filings)
    warnings.extend(material_event_warnings)

    # Canonical material-event compatibility summaries derived from parsed 8-K filings.
    recent_events = _build_recent_events(material_event_documents)
    recent_events.extend(_build_current_event_bundle_events(current_event_documents))
    recent_events.sort(key=lambda event: str(event.published_at or ""), reverse=True)

    # Build the ValuationSnapshot
    valuation = _build_valuation_snapshot(scorecard, tensions, solvency, filing_risk)

    # Assemble final bundle. built_at is UTC ISO timestamp.
    built_at = datetime.now(timezone.utc).isoformat()

    return AnalysisEvidenceBundle(
        ticker=upper,
        as_of_date=effective_as_of_date,
        built_at=built_at,
        analysis_years=years,
        analysis_quarters=quarters,
        freshness_window_days=freshness_window_days,
        valuation=valuation,
        filings=filings,
        recent_events=recent_events,
        prior_thesis=None,
        warnings=warnings,
    )


def build_analysis_evidence_bundle(
    ticker: str,
    as_of_date: str | None = None,
    years: int = 5,
    quarters: int = 0,
    freshness_window_days: int = 90,
    financial_packet: Any | None = None,
) -> AnalysisEvidenceBundle:
    """Build an AnalysisEvidenceBundle for the given ticker.

    Runs the same ingestion / valuation / filing-load chain that
    run_deep_research uses, including canonical material-event filings.
    Maps the outputs into the Task 1 contract and returns it.

    Raises ValueError if as_of_date is a non-None string that does not parse
    as ISO-8601. Raises InvalidFinancialInputError when live filing-risk
    analysis lacks a valid canonical packet. Other data-level failures
    degrade by populating bundle.warnings.

    Prefer ``build_analysis_evidence_bundle_from_cached_scorecard`` in
    sweep contexts where scorecards are already loaded in bulk and the
    slow ensure_valuation recomputation is prohibitive.
    """
    upper = ticker.upper()
    warnings: list[str] = []

    # Pre-flight: validate, resolve, freeze the temporal anchor
    _validate_as_of_date(as_of_date)
    effective_as_of_date = _resolve_as_of_date(as_of_date)
    is_live = _is_live_as_of_date(as_of_date)

    # Slow-chain step 1: ensure companyfacts data is fresh
    try:
        ensure_all_facts(upper, years_back=years)
    except Exception as exc:
        logger.warning("bundle_builder: ensure_all_facts failed for %s: %s", upper, exc)
        warnings.append("facts_ingestion_failed")

    # Slow-chain step 2: ensure valuation is computed for the anchor date
    try:
        ensure_valuation(
            upper,
            effective_as_of_date,
            require_filed_asof=True,
        )
    except Exception as exc:
        logger.warning("bundle_builder: ensure_valuation failed for %s: %s", upper, exc)
        warnings.append("valuation_failed")

    # Slow-chain step 3: load the scorecard from the DB.
    #
    # NOTE: _load_scorecard receives the caller's original as_of_date (not
    # effective_as_of_date) so its "None → most-recent row" semantic still
    # works. This is a deliberate exception to the "every IO step uses the
    # frozen date" rule — callers who passed None are asking for the latest
    # stored scorecard, and the loader's SQL branches on None vs explicit.
    #
    # _load_scorecard does DB access and json.loads internally with no
    # internal guard. Any DB or JSON-parse exception must not propagate
    # out of the builder.
    scorecard: dict[str, Any] | None = None
    resolved_date: str | None = None
    try:
        scorecard, resolved_date = _load_scorecard(upper, as_of_date)
    except Exception as exc:
        logger.warning("bundle_builder: _load_scorecard failed for %s: %s", upper, exc)
        warnings.append("scorecard_load_failed")

    if scorecard is None:
        # Missing (no row found) is separate from "load raised". If the
        # try block above already appended scorecard_load_failed, don't
        # also append scorecard_missing — they are disjoint failure modes.
        if "scorecard_load_failed" not in warnings:
            warnings.append("scorecard_missing")
    elif resolved_date and resolved_date != effective_as_of_date:
        # The scorecard came back with a different date than the bundle's
        # anchor. Surface this so Task 3 can reason about freshness instead
        # of silently consuming mixed-date inputs.
        warnings.append(f"scorecard_stale:{resolved_date}")

    # Fast-chain (shared): docket + scanners + tensions + filings + events
    return _assemble_bundle_from_scorecard(
        ticker=ticker,
        as_of_date=as_of_date,
        effective_as_of_date=effective_as_of_date,
        is_live=is_live,
        scorecard=scorecard,
        financial_packet=financial_packet,
        years=years,
        quarters=quarters,
        freshness_window_days=freshness_window_days,
        warnings=warnings,
    )


def build_analysis_evidence_bundle_from_cached_scorecard(
    ticker: str,
    scorecard: dict[str, Any],
    scorecard_as_of_date: str | None = None,
    as_of_date: str | None = None,
    years: int = 5,
    quarters: int = 0,
    freshness_window_days: int = 90,
    financial_packet: Any | None = None,
) -> AnalysisEvidenceBundle:
    """Fast-path: build an AnalysisEvidenceBundle from an already-loaded scorecard.

    Skips the slow ``ensure_all_facts → ensure_valuation → _load_scorecard``
    chain that the main ``build_analysis_evidence_bundle`` runs. Designed
    for sweep contexts (e.g. ``ivi discover``) where the caller has already
    loaded every ticker's scorecard in bulk and per-ticker valuation
    recomputation would be catastrophically slow.

    The caller is responsible for providing a sufficiently fresh scorecard.
    This function trusts the scorecard as-is without re-running the
    valuation pipeline.

    Skipped (vs. ``build_analysis_evidence_bundle``):

    - ``ensure_all_facts`` — caller's scorecard is assumed to already
      reflect current companyfacts state.
    - ``ensure_valuation`` — caller provides the scorecard directly.
    - ``_load_scorecard`` — no DB round-trip.

    Still performed (same as the main function):

    - ``collect_10k_docket`` (reads local dossier cache; fast)
    - ``assess_solvency`` and ``scan_filing_risks`` (live-gated)
    - ``_compute_tensions_from_scorecard``
    - ``_build_bundle_filings`` (reads filing text from dossier)
    - ``load_research_filing_context`` (reads quarterly + material-event filings from DB)

    Parameters
    ----------
    ticker:
        Ticker symbol.
    scorecard:
        Pre-loaded scorecard dict (the decoded JSON payload from the
        ``valuations.outputs_json`` column). Must not be ``None`` — if
        you don't have a scorecard, call ``build_analysis_evidence_bundle``
        instead and let it try to load one.
    scorecard_as_of_date:
        The ``as_of_date`` the scorecard was stamped with (typically
        ``valuations.as_of_date``). If this differs from the bundle's
        computed ``effective_as_of_date``, a ``scorecard_stale:{date}``
        warning is emitted so Task 3 / downstream consumers can reason
        about mixed-freshness inputs. Pass ``None`` to skip the staleness
        check entirely.
    as_of_date:
        Optional historical pin. Same semantics as the main function —
        ``None`` and today's ISO string are live; any other valid date
        gates the leaky scanners.
    years, quarters, freshness_window_days:
        Same semantics as the main function.
    financial_packet:
        Canonical packet from the parent run. Required when a live paid
        filing-risk scan is enabled.

    Raises
    ------
    ValueError
        If ``as_of_date`` is a non-None string that does not parse as
        ISO-8601.
    InvalidFinancialInputError
        If a live paid filing-risk scan lacks a valid canonical packet or its
        authorization fails. Other data-level failures degrade by populating
        ``bundle.warnings``.
    """
    warnings: list[str] = []

    # Pre-flight: validate, resolve, freeze
    _validate_as_of_date(as_of_date)
    effective_as_of_date = _resolve_as_of_date(as_of_date)
    is_live = _is_live_as_of_date(as_of_date)

    # Staleness check against the caller-provided scorecard date.
    # We do NOT update effective_as_of_date — the temporal coherence rule
    # still applies. The warning is a diagnostic only.
    if scorecard_as_of_date and scorecard_as_of_date != effective_as_of_date:
        warnings.append(f"scorecard_stale:{scorecard_as_of_date}")

    return _assemble_bundle_from_scorecard(
        ticker=ticker,
        as_of_date=as_of_date,
        effective_as_of_date=effective_as_of_date,
        is_live=is_live,
        scorecard=scorecard,
        financial_packet=financial_packet,
        years=years,
        quarters=quarters,
        freshness_window_days=freshness_window_days,
        warnings=warnings,
    )


def build_analysis_evidence_bundle_from_latest_scorecard(
    ticker: str,
    as_of_date: str | None = None,
    years: int = 5,
    quarters: int = 0,
    freshness_window_days: int = 90,
    financial_packet: Any | None = None,
) -> AnalysisEvidenceBundle:
    """Build a bundle from the latest cached scorecard without re-running valuation.

    This is the runtime-friendly seam for callers like `analyze` that already
    forced the upstream valuation/deep-research pipeline and want the analyst
    bundle without paying the full ensure/recompute chain twice.

    Falls back to the full builder only when the cached scorecard cannot be
    loaded or is missing entirely.
    """
    upper = ticker.upper()
    try:
        scorecard, scorecard_as_of_date = _load_scorecard(upper, as_of_date)
    except Exception as exc:
        logger.warning(
            "bundle_builder: latest scorecard fast-path failed for %s: %s",
            upper,
            exc,
        )
        return build_analysis_evidence_bundle(
            ticker=ticker,
            as_of_date=as_of_date,
            years=years,
            quarters=quarters,
            freshness_window_days=freshness_window_days,
            financial_packet=financial_packet,
        )

    if scorecard is None:
        return build_analysis_evidence_bundle(
            ticker=ticker,
            as_of_date=as_of_date,
            years=years,
            quarters=quarters,
            freshness_window_days=freshness_window_days,
            financial_packet=financial_packet,
        )

    return build_analysis_evidence_bundle_from_cached_scorecard(
        ticker=ticker,
        scorecard=scorecard,
        scorecard_as_of_date=scorecard_as_of_date,
        as_of_date=as_of_date,
        years=years,
        quarters=quarters,
        freshness_window_days=freshness_window_days,
        financial_packet=financial_packet,
    )
