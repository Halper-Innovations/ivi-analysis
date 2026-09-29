"""Per-ticker research report renderer.

Pure presentation layer. No DB, no file I/O, no LLM calls.
Takes a ResearchReport, maps it to a ReportView, renders markdown or terminal summary.

Public API:
    build_view(report) -> ReportView
    render_report(view) -> str (markdown)
    render_summary(view) -> str (terminal, 6 lines max)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.research.source_quality import source_quality_label


# ---------------------------------------------------------------------------
# View Model
# ---------------------------------------------------------------------------

@dataclass
class MethodView:
    name: str
    original: float | None
    adjusted: float | None
    discount: float | None  # (value - price) / value, None if no value or no price
    usability: str          # OK / OVERRIDDEN / N/A


@dataclass
class AdjustmentView:
    claim: str
    direction: str
    status: str
    method: str
    magnitude: float
    confidence: str
    detail: str
    citation_handles: list[str]


@dataclass
class WhyWrongEntry:
    assumption: str
    invalidation: str
    impact: str


@dataclass
class UnresolvedView:
    description: str
    hypothesis_source: str
    importance: str
    blocking_reason: str | None  # only for hard blockers


@dataclass
class CitationView:
    citation_id: str
    section: str
    relevance: str
    excerpt: str
    source_label: str | None = None
    source_date: str | None = None
    source_type: str | None = None
    source_url: str | None = None
    item_code: str | None = None
    event_category: str | None = None
    source_accession: str | None = None
    source_role: str | None = None
    source_quality: dict[str, Any] | None = None


@dataclass
class AnalystFindingView:
    """Presentation model for a single analyst note finding."""
    category: str
    claim: str
    severity: str
    direction: str | None
    validation_status: str
    citation_handles: list[str]
    suggested_adjustment: str | None


@dataclass
class AnalystReadView:
    """Presentation model for the Analyst Read section."""
    overall_assessment: str
    positives: list[AnalystFindingView]
    risks: list[AnalystFindingView]
    surprises: list[AnalystFindingView]
    adjustment_triggers: list[AnalystFindingView]
    sections_read: list[str]


@dataclass
class ReconciliationView:
    """Presentation model for the Findings Reconciliation section."""
    agreement_score: float
    both_paths: list[str]
    llm_only: list[str]
    pipeline_only: list[str]


@dataclass
class ReportView:
    # Verdict block
    ticker: str | None = None
    as_of_date: str | None = None
    filing_form: str | None = None
    filing_date: str | None = None
    latest_evidence_date: str | None = None
    latest_evidence_source_type: str | None = None
    verdict: str | None = None
    conviction_class: str | None = None
    conviction_score: int | None = None
    adjusted_margin_of_safety: float | None = None
    investigation_status: str | None = None
    artifact_quality: str | None = None
    artifact_quality_reason: str | None = None
    gate_action: str | None = None
    solvency_status: str | None = None

    # Expectations gap (leads the memo when reliable)
    implied_growth: float | None = None
    supportable_growth: float | None = None
    expectations_gap: float | None = None
    expectations_gap_bucket: str | None = None
    expectations_gap_line: str | None = None

    # Valuation summary
    methods: list[MethodView] = field(default_factory=list)
    current_price: float | None = None
    tension_type: str | None = None
    tension_explanation: str | None = None
    tension_citation_handles: list[str] = field(default_factory=list)

    # Thesis adjustments
    adjustments: list[AdjustmentView] = field(default_factory=list)

    # Why this could be wrong
    why_wrong: list[WhyWrongEntry] = field(default_factory=list)

    # Unresolved
    open_questions: list[UnresolvedView] = field(default_factory=list)
    hard_blockers: list[UnresolvedView] = field(default_factory=list)

    # Conviction breakdown
    conviction_components: dict[str, Any] | None = None

    # Citations
    citations: list[CitationView] = field(default_factory=list)

    # Analyst notes (Stage A) — None when not run
    analyst_read: AnalystReadView | None = None
    # Findings reconciliation (Stage C) — None when not run
    reconciliation: ReconciliationView | None = None
    # Analyst citations (separate namespace from deterministic [C1])
    analyst_citations: list[CitationView] = field(default_factory=list)

    # Pipeline warnings (surfaced prominently when non-empty)
    warnings: list[str] = field(default_factory=list)

    # Analysis scope
    analysis_years: int = 5
    analysis_quarters: int = 0

    # Artifact path
    report_path: str | None = None


# ---------------------------------------------------------------------------
# Blocker classification
# ---------------------------------------------------------------------------

def _classify_blocker(item: Any, top_adjustment_sources: set[str]) -> bool:
    """Determine if an unresolved item is a hard blocker.

    Rules (any match -> blocker):
    1. importance == "REQUIRED"
    2. hypothesis_source in top 3 adjustment sources by magnitude
    3. importance == "IMPORTANT" AND hypothesis_priority == "P1" AND hypothesis_direction == "BEARISH"

    Fallback: missing fields disable the rules that depend on them.
    If no rules can be evaluated, classify as open question (fail safe).
    """
    importance = getattr(item, "importance", "") or ""
    hyp_source = getattr(item, "hypothesis_source", "") or ""
    hyp_priority = getattr(item, "hypothesis_priority", "") or ""
    hyp_direction = getattr(item, "hypothesis_direction", "") or ""

    if importance == "REQUIRED":
        return True
    if hyp_source and hyp_source in top_adjustment_sources:
        return True
    if importance == "IMPORTANT" and hyp_priority == "P1" and hyp_direction == "BEARISH":
        return True

    return False


def _blocking_reason(item: Any, top_adjustment_sources: set[str]) -> str:
    """Return human-readable reason why this item is a blocker."""
    importance = getattr(item, "importance", "") or ""
    hyp_source = getattr(item, "hypothesis_source", "") or ""
    hyp_priority = getattr(item, "hypothesis_priority", "") or ""
    hyp_direction = getattr(item, "hypothesis_direction", "") or ""

    if importance == "REQUIRED":
        return "Required evidence for hypothesis resolution"
    if hyp_source and hyp_source in top_adjustment_sources:
        return f"Tied to thesis-critical assumption ({hyp_source})"
    if importance == "IMPORTANT" and hyp_priority == "P1" and hyp_direction == "BEARISH":
        return "High-priority bearish hypothesis with material downside uncertainty"
    return "Unspecified"


# ---------------------------------------------------------------------------
# Method usability
# ---------------------------------------------------------------------------

def _method_usability(
    method_name: str,
    original_value: float | None,
    adjustments: list[Any],
) -> str:
    """Derive usability label: N/A, OVERRIDDEN, or OK."""
    if original_value is None:
        return "N/A"

    total_abs_adj = sum(
        abs(a.adjustment_magnitude)
        for a in adjustments
        if a.affected_method == method_name
    )
    if abs(original_value) > 0 and total_abs_adj / abs(original_value) > 0.20:
        return "OVERRIDDEN"

    return "OK"


# ---------------------------------------------------------------------------
# Tension explanation
# ---------------------------------------------------------------------------

def _derive_tension_explanation(
    tension_type: str | None,
    methods: list[MethodView],
    price: float | None,
) -> str | None:
    """Generate a two-line tension explanation from method values."""
    if not tension_type or tension_type == "NONE":
        return None

    method_map = {m.name: m for m in methods}
    dcf = method_map.get("DCF")
    epv = method_map.get("EPV")

    if tension_type == "GROWTH_VS_EARNINGS_POWER" and dcf and epv:
        dcf_val = dcf.adjusted if dcf.adjusted is not None else dcf.original
        epv_val = epv.adjusted if epv.adjusted is not None else epv.original
        if dcf_val is not None and epv_val is not None:
            gap = dcf_val - epv_val
            return (
                f"DCF (${dcf_val:.0f}) values growth; EPV (${epv_val:.0f}) values current earnings only. "
                f"${gap:.0f}/share of the gap is growth premium — if growth stalls, intrinsic value falls to EPV."
            )

    if tension_type == "ASSET_VS_EARNINGS":
        ncav = method_map.get("NCAV")
        if ncav and epv:
            ncav_val = ncav.adjusted if ncav.adjusted is not None else ncav.original
            epv_val = epv.adjusted if epv.adjusted is not None else epv.original
            if ncav_val is not None and epv_val is not None:
                return (
                    f"Asset value (${ncav_val:.0f}) exceeds earnings-derived estimates (${epv_val:.0f}). "
                    f"Company may be worth more in liquidation, or earnings are temporarily depressed."
                )

    if tension_type == "INSUFFICIENT_METHODS":
        return "Too few valuation methods produced values to assess consensus."

    return f"Methods disagree ({tension_type}). Review individual valuations for detail."


# ---------------------------------------------------------------------------
# Verdict derivation (canonical, single authority)
# ---------------------------------------------------------------------------

def derive_verdict(report: Any) -> str:
    """Canonical pipeline verdict. Single authority — used by the renderer.

    Returns "PROCEED" / "WATCH" / "DO NOT ACT".
    """
    conv = report.conviction_score
    if report.gate_action == "BLOCK":
        return "DO NOT ACT"
    if conv is None or conv < 25:
        return "DO NOT ACT"
    if not report.investigation_ran:
        return "WATCH"

    # Hard blocker check — reuse existing classification logic
    thesis = report.thesis
    adjustments = thesis.adjustments if thesis else []
    all_unresolved = list(report.researchable_items) + list(report.not_researchable_items)

    source_magnitudes: dict[str, float] = {}
    for adj in adjustments:
        src = adj.hypothesis_source
        source_magnitudes[src] = source_magnitudes.get(src, 0.0) + abs(adj.adjustment_magnitude)
    top_sources = set(
        sorted(source_magnitudes, key=source_magnitudes.get, reverse=True)[:3]
    ) if source_magnitudes else set()

    has_blockers = any(_classify_blocker(item, top_sources) for item in all_unresolved)
    if has_blockers:
        return "WATCH"

    if report.gate_action == "ADJUST":
        return "WATCH"
    return "PROCEED"


# ---------------------------------------------------------------------------
# build_view
# ---------------------------------------------------------------------------

def build_view(report: Any) -> ReportView:
    """Map a ResearchReport to a ReportView for rendering.

    This is the only function that reads ResearchReport internals.
    All rendering operates on the returned ReportView.
    """
    view = ReportView()

    # --- Verdict block ---
    view.ticker = report.ticker
    view.as_of_date = report.as_of_date
    view.filing_form = report.form_type
    view.filing_date = report.filing_date
    view.latest_evidence_date = getattr(report, "latest_evidence_date", None)
    view.latest_evidence_source_type = getattr(report, "latest_evidence_source_type", None)
    view.conviction_class = report.conviction_class
    view.conviction_score = report.conviction_score
    view.gate_action = report.gate_action
    view.solvency_status = report.solvency_status
    view.report_path = report.report_path

    # Investigation status
    if not report.investigation_ran:
        view.investigation_status = f"Investigation did not run ({report.status})"
    else:
        view.investigation_status = "Investigation complete"

    warning_codes = getattr(report, "warnings", []) or []
    if not report.investigation_ran:
        view.artifact_quality = "DEGRADED_DIAGNOSTIC"
        if "llm_provider_disabled" in warning_codes:
            view.artifact_quality_reason = (
                "LLM provider unavailable; investigation did not run. "
                "Use this as a pipeline diagnostic, not an investment research output."
            )
        else:
            view.artifact_quality_reason = (
                "Investigation did not run. Use this as a pipeline diagnostic, "
                "not an investment research output."
            )
    elif any(str(warning).startswith(("facts_ingestion_failed", "valuation_failed")) for warning in warning_codes):
        view.artifact_quality = "DEGRADED_DIAGNOSTIC"
        view.artifact_quality_reason = (
            "Critical upstream data collection or valuation failed. "
            "Review warnings before using this research output."
        )

    # Thesis data
    thesis = report.thesis
    adjustments = thesis.adjustments if thesis else []
    all_unresolved = list(report.researchable_items) + list(report.not_researchable_items)

    # Adjusted MOS
    if thesis and thesis.adjusted_margin_of_safety is not None:
        view.adjusted_margin_of_safety = thesis.adjusted_margin_of_safety

    # Current price — from thesis if available, otherwise from scorecard
    if thesis and thesis.current_price is not None:
        view.current_price = thesis.current_price
    elif getattr(report, "scorecard_price", None) is not None:
        view.current_price = report.scorecard_price

    # --- Blocker classification ---
    source_magnitudes: dict[str, float] = {}
    for adj in adjustments:
        src = adj.hypothesis_source
        source_magnitudes[src] = source_magnitudes.get(src, 0.0) + abs(adj.adjustment_magnitude)
    top_sources = set(
        sorted(source_magnitudes, key=source_magnitudes.get, reverse=True)[:3]  # type: ignore[arg-type]
    ) if source_magnitudes else set()

    hard_blockers: list[UnresolvedView] = []
    open_questions: list[UnresolvedView] = []
    for item in all_unresolved:
        if _classify_blocker(item, top_sources):
            hard_blockers.append(UnresolvedView(
                description=item.description,
                hypothesis_source=item.hypothesis_source,
                importance=item.importance,
                blocking_reason=_blocking_reason(item, top_sources),
            ))
        else:
            open_questions.append(UnresolvedView(
                description=item.description,
                hypothesis_source=item.hypothesis_source,
                importance=item.importance,
                blocking_reason=None,
            ))
    view.hard_blockers = hard_blockers
    view.open_questions = open_questions

    # --- Verdict derivation (after blocker classification) ---
    view.verdict = derive_verdict(report)

    # --- Expectations gap (passthrough from scorecard_* fields) ---
    view.implied_growth = getattr(report, "scorecard_implied_growth", None)
    view.supportable_growth = getattr(report, "scorecard_supportable_growth", None)
    view.expectations_gap = getattr(report, "scorecard_expectations_gap", None)
    view.expectations_gap_bucket = getattr(
        report, "scorecard_expectations_gap_bucket", None
    )
    # Prefer the canonical one-liner threaded from the expectations_gap authority.
    # When absent (legacy reports), fall back to recomputing it from the present
    # legs so the SINGLE authority (app.valuation.expectations_gap) still formats
    # the line rather than the renderer reconstructing a divergent phrasing.
    view.expectations_gap_line = getattr(
        report, "scorecard_expectations_gap_line", None
    )
    if (
        view.expectations_gap_line is None
        and view.implied_growth is not None
        and view.supportable_growth is not None
        and view.expectations_gap_bucket
        and view.expectations_gap_bucket != "EXPECTATIONS_GAP_UNRELIABLE"
    ):
        from app.valuation.expectations_gap import compute_expectations_gap

        recomputed = compute_expectations_gap(
            view.implied_growth, view.supportable_growth, False
        )
        view.expectations_gap_line = recomputed.get("line")

    # --- Valuation summary ---
    # Use thesis values when available, fall back to scorecard values
    def _val_or_fallback(thesis_attr: str, scorecard_attr: str) -> float | None:
        v = getattr(thesis, thesis_attr, None) if thesis else None
        if v is not None:
            return v
        return getattr(report, scorecard_attr, None)

    method_configs = [
        ("DCF", _val_or_fallback("original_dcf", "scorecard_dcf"), getattr(thesis, "adjusted_dcf", None)),
        ("EPV", _val_or_fallback("original_epv", "scorecard_epv"), getattr(thesis, "adjusted_epv", None)),
        ("Graham", _val_or_fallback("original_graham", "scorecard_graham"), None),
    ]
    price = view.current_price
    for name, orig, adj in method_configs:
        discount = None
        display_value = adj if adj is not None else orig
        if display_value is not None and price is not None and price > 0 and display_value != 0:
            discount = (display_value - price) / display_value
        view.methods.append(MethodView(
            name=name,
            original=orig,
            adjusted=adj,
            discount=discount,
            usability=_method_usability(name.lower(), orig, adjustments),
        ))

    view.tension_type = report.tension_type
    view.tension_explanation = _derive_tension_explanation(view.tension_type, view.methods, price)

    # --- Citation handle mapping ---
    # Map ALL need_ids (primary + additional) to citation handles so that
    # adjustments referencing any need_id for a shared citation get handles.
    need_to_cite: dict[str, list[str]] = {}
    for cite in getattr(report, "citations", []):
        need_to_cite.setdefault(cite.need_id, []).append(cite.citation_id)
        for extra_nid in getattr(cite, "additional_need_ids", []):
            need_to_cite.setdefault(extra_nid, []).append(cite.citation_id)

    # Tension citation handles: from adjustments matching tension type
    if report.tension_type and report.tension_type != "NONE":
        tension_handles: list[str] = []
        for adj in adjustments:
            if adj.hypothesis_source == report.tension_type:
                for eid in adj.evidence_item_ids:
                    tension_handles.extend(need_to_cite.get(eid, []))
        view.tension_citation_handles = sorted(set(tension_handles))

    # --- Thesis adjustments ---
    for adj in adjustments:
        handles: list[str] = []
        for eid in adj.evidence_item_ids:
            handles.extend(need_to_cite.get(eid, []))
        view.adjustments.append(AdjustmentView(
            claim=adj.hypothesis_claim,
            direction=adj.hypothesis_direction,
            status=adj.hypothesis_status,
            method=adj.affected_method.upper(),
            magnitude=adj.adjustment_magnitude,
            confidence=adj.adjustment_confidence,
            detail=adj.calibration_detail,
            citation_handles=sorted(set(handles)),
        ))

    # --- Why this could be wrong (top 3, max 4) ---
    candidates: list[WhyWrongEntry] = []

    sorted_adjs = sorted(adjustments, key=lambda a: abs(a.adjustment_magnitude), reverse=True)
    for adj in sorted_adjs[:3]:
        candidates.append(WhyWrongEntry(
            assumption=adj.hypothesis_claim,
            invalidation=f"If {adj.hypothesis_direction.lower()} thesis is wrong, the "
                         f"{adj.affected_method.upper()} adjustment of ${adj.adjustment_magnitude:+.2f} reverses.",
            impact=f"${abs(adj.adjustment_magnitude):.2f}/share on {adj.affected_method.upper()}",
        ))

    for hb in hard_blockers:
        if len(candidates) >= 4:
            break
        candidates.append(WhyWrongEntry(
            assumption=f"Unknown: {hb.description}",
            invalidation=f"If resolved negatively ({hb.blocking_reason})",
            impact="Indeterminate — thesis-critical evidence missing",
        ))

    view.why_wrong = candidates[:4]

    # --- Conviction breakdown ---
    from app.valuation.conviction import compute_conviction
    conviction = compute_conviction(report)
    view.conviction_components = {
        "method_agreement": conviction.method_agreement_score,
        "evidence_coverage": conviction.evidence_coverage_score,
        "gate_quality": conviction.gate_quality_score,
        "investigation_resolution": conviction.investigation_resolution_score,
    }

    # --- Citations ---
    for cite in getattr(report, "citations", []):
        view.citations.append(CitationView(
            citation_id=cite.citation_id,
            section=cite.section,
            relevance=cite.relevance,
            excerpt=cite.excerpt,
            source_label=(
                getattr(cite, "source_title", None)
                or getattr(cite, "source_form_type", None)
            ),
            source_date=(
                getattr(cite, "source_published_at", None)
                or getattr(cite, "source_filing_date", None)
            ),
            source_type=getattr(cite, "source_type", None),
            source_url=getattr(cite, "source_url", None),
            item_code=getattr(cite, "item_code", None),
            event_category=getattr(cite, "event_category", None),
            source_accession=getattr(cite, "source_accession", None),
            source_role=getattr(cite, "source_role", None),
            source_quality=getattr(cite, "source_quality", None),
        ))

    # --- Analyst Read (Stage A) ---
    analyst_notes = getattr(report, "analyst_notes", None)
    merged_findings = getattr(report, "merged_findings", None)

    if analyst_notes is not None:
        analyst_cite_counter = 0

        def _map_findings(notes: list) -> list[AnalystFindingView]:
            nonlocal analyst_cite_counter
            mapped = []
            for note in notes:
                handles = []
                for citation in getattr(note, "citations", []):
                    analyst_cite_counter += 1
                    handle = f"A{analyst_cite_counter}"
                    handles.append(f"[{handle}]")
                    view.analyst_citations.append(CitationView(
                        citation_id=handle,
                        section=citation.section,
                        relevance=note.claim,
                        excerpt=citation.excerpt,
                    ))
                mapped.append(AnalystFindingView(
                    category=note.category,
                    claim=note.claim,
                    severity=note.severity,
                    direction=note.direction,
                    validation_status=note.validation_status,
                    citation_handles=handles,
                    suggested_adjustment=note.suggested_adjustment,
                ))
            return mapped

        view.analyst_read = AnalystReadView(
            overall_assessment=analyst_notes.overall_assessment,
            positives=_map_findings(analyst_notes.positives),
            risks=_map_findings(analyst_notes.risks),
            surprises=_map_findings(analyst_notes.surprises),
            adjustment_triggers=_map_findings(analyst_notes.adjustment_triggers),
            sections_read=analyst_notes.filing_sections_read,
        )

    if merged_findings is not None:
        both_descs = []
        for m in merged_findings.both_paths:
            both_descs.append(f"{m.llm_note.claim} (pipeline: {m.pipeline_finding.source_hypothesis})")
        llm_descs = [n.claim for n in merged_findings.llm_only]
        pipe_descs = [f"{f.claim} ({f.source_hypothesis})" for f in merged_findings.pipeline_only]
        view.reconciliation = ReconciliationView(
            agreement_score=merged_findings.agreement_score,
            both_paths=both_descs,
            llm_only=llm_descs,
            pipeline_only=pipe_descs,
        )

    # --- Pipeline warnings ---
    view.warnings = getattr(report, "warnings", []) or []

    # --- Analysis scope ---
    view.analysis_years = getattr(report, "analysis_years", 5)
    view.analysis_quarters = getattr(report, "analysis_quarters", 0)

    return view


# ---------------------------------------------------------------------------
# Markdown Renderer
# ---------------------------------------------------------------------------

def render_report(view: ReportView) -> str:
    """Render a full markdown report from a ReportView."""
    sections: list[str] = []

    sections.append(_render_verdict_block(view))
    # Expectations gap leads the memo (above Valuation Summary) when reliable.
    if (
        view.expectations_gap_bucket
        and view.expectations_gap_bucket != "EXPECTATIONS_GAP_UNRELIABLE"
    ):
        sections.append(_render_expectations_gap(view))
    sections.append(_render_valuation_summary(view))

    if view.adjustments:
        sections.append(_render_adjustments(view))
    if view.why_wrong:
        sections.append(_render_why_wrong(view))

    # Analyst notes sections (after Why Wrong, before Open Questions)
    if view.analyst_read:
        sections.append(_render_analyst_read(view))
    if view.reconciliation:
        sections.append(_render_reconciliation(view))

    if view.open_questions:
        sections.append(_render_open_questions(view))
    if view.hard_blockers:
        sections.append(_render_hard_blockers(view))
    if view.conviction_components:
        sections.append(_render_conviction(view))

    # Citations: deterministic first, then analyst
    if view.citations or view.analyst_citations:
        sections.append(_render_all_citations(view))

    return "\n\n---\n\n".join(sections) + "\n"


def _render_verdict_block(view: ReportView) -> str:
    report_title = (
        "Deep Research Diagnostic Report"
        if view.artifact_quality == "DEGRADED_DIAGNOSTIC"
        else "Deep Research Report"
    )
    lines = [f"# {view.ticker} — {report_title}"]
    lines.append("")
    lines.append(f"**As of:** {view.as_of_date or 'N/A'}")
    if view.filing_form or view.filing_date:
        lines.append(f"**Filing:** {view.filing_form or 'N/A'} ({view.filing_date or 'N/A'})")
    if view.latest_evidence_source_type or view.latest_evidence_date:
        lines.append(
            f"**Latest Evidence:** {view.latest_evidence_source_type or 'N/A'}"
            f" ({view.latest_evidence_date or 'N/A'})"
        )
    lines.append("")
    lines.append(f"**Verdict: {view.verdict}**")
    if view.conviction_class and view.conviction_score is not None:
        lines.append(f"**Conviction:** {view.conviction_class} ({view.conviction_score}/100)")
    if view.adjusted_margin_of_safety is not None:
        lines.append(f"**Adjusted Margin of Safety:** {view.adjusted_margin_of_safety:.1%}")
    lines.append(f"**Status:** {view.investigation_status}")
    if view.artifact_quality:
        reason = f" — {view.artifact_quality_reason}" if view.artifact_quality_reason else ""
        lines.append(f"**Artifact Quality:** {view.artifact_quality}{reason}")
    if "llm_provider_disabled" in view.warnings:
        lines.append("")
        lines.append("**WARNING: LLM provider was unavailable during this run. "
                      "Evidence adjudication and analyst notes did not execute. "
                      "Conviction scores are DEGRADED and do not reflect true analytical quality. "
                      "Set VOE_LLM_PROVIDER and VOE_OPENAI_API_KEY in your environment.**")
    return "\n".join(lines)


def _render_expectations_gap(view: ReportView) -> str:
    """Render the lead Expectations Gap section (Mauboussin/Rappaport frame)."""
    lines = ["## Expectations Gap"]
    lines.append("")
    if view.expectations_gap_line:
        line = view.expectations_gap_line
    else:
        implied_str = (
            f"{view.implied_growth:.0%}" if view.implied_growth is not None else "N/A"
        )
        supportable_str = (
            f"{view.supportable_growth:.0%}"
            if view.supportable_growth is not None
            else "N/A"
        )
        gap_str = (
            f"{view.expectations_gap:.0%}" if view.expectations_gap is not None else "N/A"
        )
        line = (
            f"Market implies {implied_str} growth; supportable {supportable_str}; "
            f"gap {gap_str} ({view.expectations_gap_bucket})"
        )
    lines.append(line)
    lines.append("")
    bucket = view.expectations_gap_bucket
    if bucket == "EXPENSIVE_VS_EXPECTATIONS":
        interpretation = (
            "A positive gap means the market is paying for MORE growth than the "
            "business has historically supported — expensive vs expectations; "
            "wait for a lower price or avoid."
        )
    elif bucket == "FAIRLY_PRICED_EXPECTATIONS":
        interpretation = (
            "The market's implied growth is roughly in line with what the business "
            "has historically supported — fairly priced on expectations."
        )
    else:  # CHEAP_VS_EXPECTATIONS
        interpretation = (
            "A negative gap means the market is paying for LESS growth than the "
            "business has historically supported — cheap vs expectations. This "
            "is a candidate signal under measurement, not an established edge."
        )
    lines.append(interpretation)
    return "\n".join(lines)


def _render_valuation_summary(view: ReportView) -> str:
    lines = ["## Valuation Summary"]
    lines.append("")
    price_str = f"${view.current_price:.2f}" if view.current_price is not None else "N/A"
    lines.append(f"**Current Price:** {price_str}")
    lines.append("")

    lines.append("| Method | Original | Adjusted | Discount | Usability |")
    lines.append("|--------|----------|----------|----------|-----------|")
    for m in view.methods:
        orig = f"${m.original:.2f}" if m.original is not None else "\u2014"
        adj = f"${m.adjusted:.2f}" if m.adjusted is not None else "\u2014"
        disc = f"{m.discount:.1%}" if m.discount is not None else "\u2014"
        lines.append(f"| {m.name} | {orig} | {adj} | {disc} | {m.usability} |")

    if view.tension_type and view.tension_type != "NONE":
        lines.append("")
        handle_str = ""
        if view.tension_citation_handles:
            handle_str = " " + " ".join(f"[{h}]" for h in view.tension_citation_handles)
        lines.append(f"**Tension:** {view.tension_type}{handle_str}")
        if view.tension_explanation:
            lines.append(f"> {view.tension_explanation}")

    return "\n".join(lines)


def _render_adjustments(view: ReportView) -> str:
    lines = ["## Thesis Adjustments"]
    for adj in view.adjustments:
        lines.append("")
        handle_str = ""
        if adj.citation_handles:
            handle_str = " " + " ".join(f"[{h}]" for h in adj.citation_handles)
        lines.append(f"**{adj.direction}** \u2014 {adj.claim}{handle_str}")
        lines.append(f"- Status: {adj.status}")
        lines.append(f"- {adj.method}: ${adj.magnitude:+.2f}/share ({adj.confidence})")
        lines.append(f"- {adj.detail}")
    return "\n".join(lines)


def _render_why_wrong(view: ReportView) -> str:
    lines = ["## Why This Could Be Wrong"]
    for i, entry in enumerate(view.why_wrong, 1):
        lines.append("")
        lines.append(f"**{i}. {entry.assumption}**")
        lines.append(f"- Invalidation: {entry.invalidation}")
        lines.append(f"- Impact: {entry.impact}")
    return "\n".join(lines)


def _render_open_questions(view: ReportView) -> str:
    lines = ["## Open Questions"]
    for q in view.open_questions:
        lines.append(f"- **{q.description}** ({q.hypothesis_source}, {q.importance})")
    return "\n".join(lines)


def _render_hard_blockers(view: ReportView) -> str:
    lines = ["## Hard Blockers"]
    for b in view.hard_blockers:
        lines.append("")
        lines.append(f"- **{b.description}** ({b.hypothesis_source})")
        lines.append(f"  - Why: {b.blocking_reason}")
    return "\n".join(lines)


_CONVICTION_RATIONALES = {
    "method_agreement": {
        (0, 5): "Single method or no agreement",
        (6, 15): "Partial method agreement",
        (16, 24): "Most methods agree on direction",
        (25, 25): "All methods agree on direction",
    },
    "evidence_coverage": {
        (0, 0): "No investigation ran",
        (1, 12): "Low evidence coverage — many hypotheses unresolved",
        (13, 20): "Moderate evidence coverage",
        (21, 25): "High evidence coverage — most hypotheses resolved",
    },
    "gate_quality": {
        (0, 0): "No gate verdict available",
        (5, 5): "BLOCK — quality gate flagged critical issues",
        (15, 15): "ADJUST — quality gate flagged concerns",
        (25, 25): "PROCEED — quality gate passed",
    },
    "investigation_resolution": {
        (0, 0): "No investigation ran",
        (1, 10): "Few hypotheses resolved",
        (11, 20): "Most hypotheses resolved",
        (21, 25): "Hypotheses well-resolved, tensions explained",
    },
}


def _conviction_rationale(component: str, score: int) -> str:
    """Return brief rationale for a conviction component score."""
    ranges = _CONVICTION_RATIONALES.get(component, {})
    for (lo, hi), rationale in ranges.items():
        if lo <= score <= hi:
            return rationale
    return ""


def _render_conviction(view: ReportView) -> str:
    llm_disabled = "llm_provider_disabled" in view.warnings
    lines = ["## Conviction Breakdown"]
    if llm_disabled:
        lines.append("**WARNING: LLM provider was unavailable — scores below are degraded.**")
        lines.append("")
    if view.conviction_components:
        llm_dependent = {"evidence_coverage", "investigation_resolution"}
        for key, score in view.conviction_components.items():
            label = key.replace("_", " ").title()
            rationale = _conviction_rationale(key, score)
            if llm_disabled and key in llm_dependent and score == 0:
                rationale = "LLM UNAVAILABLE — score is 0 due to missing API key, not weak fundamentals"
            suffix = f" — {rationale}" if rationale else ""
            lines.append(f"- {label}: {score}/25{suffix}")
    return "\n".join(lines)


def _render_citations(view: ReportView) -> str:
    lines = ["## Citations"]
    for c in view.citations:
        lines.append("")
        source_bits = [
            bit
            for bit in [
                c.source_label,
                f"Item {c.item_code}" if c.item_code else None,
                c.source_date,
                c.source_type if c.source_type and c.source_label is None else None,
            ]
            if bit
        ]
        source_label = " / ".join(source_bits)
        if source_label:
            lines.append(f"**[{c.citation_id}]** {source_label} - {c.section}")
        else:
            lines.append(f"**[{c.citation_id}]** {c.section}")
        lines.append(f"- Relevance: {c.relevance}")
        lines.append(f"- > {c.excerpt}")
        if c.source_url:
            lines.append(f"- Source: {c.source_url}")
        quality_label = source_quality_label(c.source_quality)
        if quality_label:
            lines.append(f"- Source Quality: {quality_label}")
    return "\n".join(lines)


def _render_analyst_read(view: ReportView) -> str:
    lines = ["## Analyst Read"]
    ar = view.analyst_read
    lines.append("")
    lines.append(f"**Overall:** {ar.overall_assessment}")
    lines.append(f"**Sections read:** {', '.join(ar.sections_read)}")

    for label, findings in [
        ("Key Positives", ar.positives),
        ("Key Risks", ar.risks),
        ("Surprises", ar.surprises),
        ("Adjustment Triggers", ar.adjustment_triggers),
    ]:
        if findings:
            lines.append("")
            lines.append(f"### {label}")
            for f in findings:
                handle_str = " ".join(f.citation_handles)
                status_tag = " *[unverified]*" if f.validation_status == "UNVERIFIED" else ""
                lines.append(f"- **[{f.severity}]** {f.claim}{status_tag} {handle_str}")
                if f.suggested_adjustment:
                    lines.append(f"  - Suggested: {f.suggested_adjustment}")

    return "\n".join(lines)


def _render_reconciliation(view: ReportView) -> str:
    lines = ["## Findings Reconciliation"]
    rc = view.reconciliation
    lines.append("")
    lines.append(f"**Agreement score:** {rc.agreement_score:.0%}")

    if rc.both_paths:
        lines.append("")
        lines.append("**Both paths found:**")
        for desc in rc.both_paths:
            lines.append(f"- {desc}")
    if rc.llm_only:
        lines.append("")
        lines.append("**Analyst read only:**")
        for desc in rc.llm_only:
            lines.append(f"- {desc}")
    if rc.pipeline_only:
        lines.append("")
        lines.append("**Pipeline only:**")
        for desc in rc.pipeline_only:
            lines.append(f"- {desc}")

    return "\n".join(lines)


def _render_all_citations(view: ReportView) -> str:
    """Render both deterministic and analyst citations in one appendix."""
    lines = ["## Citations"]

    if view.citations:
        lines.append("")
        lines.append("### Deterministic Pipeline")
        for c in view.citations:
            lines.append("")
            source_bits = [
                bit
                for bit in [
                    c.source_label,
                    f"Item {c.item_code}" if c.item_code else None,
                    c.source_date,
                    c.source_type if c.source_type and c.source_label is None else None,
                ]
                if bit
            ]
            source_label = " / ".join(source_bits)
            if source_label:
                lines.append(f"**[{c.citation_id}]** {source_label} - {c.section}")
            else:
                lines.append(f"**[{c.citation_id}]** {c.section}")
            lines.append(f"- Relevance: {c.relevance}")
            lines.append(f"- > {c.excerpt}")
            if c.source_url:
                lines.append(f"- Source: {c.source_url}")
            quality_label = source_quality_label(c.source_quality)
            if quality_label:
                lines.append(f"- Source Quality: {quality_label}")

    if view.analyst_citations:
        lines.append("")
        lines.append("### Analyst Read")
        for c in view.analyst_citations:
            lines.append("")
            lines.append(f"**[{c.citation_id}]** {c.section}")
            lines.append(f"- Relevance: {c.relevance}")
            lines.append(f"- > {c.excerpt}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Terminal Summary
# ---------------------------------------------------------------------------

def render_summary(view: ReportView) -> str:
    """Render terminal summary for completed analysis."""
    ticker = view.ticker or "???"
    verdict = view.verdict or "UNKNOWN"
    conv = f"{view.conviction_score}/100" if view.conviction_score is not None else "N/A"

    # Scope description
    scope_parts = [f"{view.analysis_years}Y"]
    if view.analysis_quarters:
        scope_parts.append(f"{view.analysis_quarters}Q")
    scope = "+".join(scope_parts)

    lines = [f"=== {ticker} — {verdict} ({conv}) [{scope}] ==="]

    # Warning line (before metrics so it's impossible to miss)
    if "llm_provider_disabled" in view.warnings:
        lines.append("WARNING: LLM provider unavailable — conviction scores are DEGRADED")

    gate = view.gate_action or "N/A"
    mos = f"{view.adjusted_margin_of_safety:.1%}" if view.adjusted_margin_of_safety is not None else "N/A"
    solv = view.solvency_status or "N/A"
    lines.append(f"Gate: {gate} | MOS: {mos} | Solvency: {solv}")

    values = [
        m.adjusted if m.adjusted is not None else m.original
        for m in view.methods
        if (m.adjusted is not None or m.original is not None)
    ]
    if values:
        lo, hi = min(values), max(values)
        lines.append(f"Adjusted range: ${lo:.2f} — ${hi:.2f}")
    else:
        lines.append("Adjusted range: N/A")

    tension = view.tension_type or "NONE"
    lines.append(f"Tension: {tension}")

    confirmed = sum(1 for a in view.adjustments if a.status in ("CONFIRMED", "PARTIALLY_CONFIRMED"))
    n_questions = len(view.open_questions)
    n_blockers = len(view.hard_blockers)
    lines.append(f"Findings: {confirmed} confirmed, {n_questions} open questions, {n_blockers} blocker{'s' if n_blockers != 1 else ''}")

    path = view.report_path or "N/A"
    lines.append(f"-> {path}")

    return "\n".join(lines) + "\n"
