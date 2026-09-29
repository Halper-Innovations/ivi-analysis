"""Deep research orchestrator — single-pass per-ticker investigation pipeline.

Wires Tasks 4 (hypothesis generation), 5 (evidence search), and 6 (thesis update)
into a complete research pipeline. Two entry points:

- assemble_research(): No DB or file I/O. Takes pre-loaded inputs, returns ResearchReport.
- run_deep_research(): Loads inputs from DB/filesystem, calls assemble_research, persists result.

Public API: run_deep_research(ticker, as_of_date) for CLI/callers,
            assemble_research(...) for testing and batch callers.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from app.alpha.schemas import Anomaly, SolvencyAssessment
from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.autonomous.v1_financial_context import (
    BoundV1FinancialScope,
    bind_v1_financial_scope,
    build_canonical_v1_financial_context,
    financial_input_scenario,
)
from app.research.filing_context import (
    FilingContext,
    FilingDocument,
    build_inline_filing_context,
    load_research_filing_context,
)
from app.research.current_event_context import CurrentEventContext, load_current_event_context

if TYPE_CHECKING:
    from app.config import AppConfig
    from app.research.analyst_notes import AnalystNotes
    from app.research.findings_reconciliation import MergedFindings
from app.research.thesis_updater import ThesisResult, ThesisAdjustment, UnresolvedEvidence
from app.valuation.lineage import latest_decision_eligible_valuation_row
from app.valuation.mos_conventions import graham_value_from_textbook_discount

logger = logging.getLogger(__name__)

STANDARD_REPORT_SUFFIX = "_report.md"
DIAGNOSTIC_REPORT_SUFFIX = "_diagnostic_report.md"

FinancialScenarioSource = Sequence[Any] | Callable[[], Sequence[Any]]


# ---------------------------------------------------------------------------
# Output Schema
# ---------------------------------------------------------------------------


@dataclass
class ReportCitation:
    """Lightweight citation for report rendering — extracted from EvidenceItemResult."""

    citation_id: str  # "C1", "C2", etc. — assigned during assembly
    need_id: str  # primary need_id (first encountered during dedup)
    section: str  # filing section (e.g., "Item 1A Risk Factors")
    excerpt: str  # verbatim filing text
    relevance: str  # what was being sought (from EvidenceItemResult.needed)
    hypothesis_source: str  # which hypothesis this supports
    item_code: str | None = None
    event_category: str | None = None
    source_form_type: str | None = None
    source_filing_date: str | None = None
    source_accession: str | None = None
    source_role: str | None = None
    source_type: str | None = None
    source_title: str | None = None
    source_url: str | None = None
    source_published_at: str | None = None
    source_quality: dict[str, Any] | None = None
    additional_need_ids: list[str] = field(
        default_factory=list
    )  # other need_ids sharing this citation


@dataclass
class ResearchReport:
    ticker: str
    as_of_date: str
    status: str  # OK / NO_SCORECARD / NO_FILING / NO_HYPOTHESES
    started_at: str
    completed_at: str

    # Upstream inputs summary
    scorecard_present: bool
    filing_present: bool
    filing_date: str | None
    form_type: str | None
    anomaly_count: int
    solvency_status: str | None
    filing_risk_status: str | None  # best-effort, NOT date-aligned
    gate_action: str | None

    # Investigation
    investigation_ran: bool
    hypotheses_generated: int
    thesis: ThesisResult | None

    # Unresolved evidence items (item-level, metadata only — no re-search in v1)
    researchable_items: list[UnresolvedEvidence]
    not_researchable_items: list[UnresolvedEvidence]

    # Summary counts (0 when thesis is None)
    total_adjustments: int
    fact_calibrated_count: int
    heuristic_count: int

    # Tension summary (from analyze_method_tensions — carried for conviction scoring)
    # NO dataclass defaults — every constructor must set explicitly.
    methods_agree: bool | None
    consensus_strength: int | None
    method_count: int | None
    tension_type: str | None

    # Scorecard-level values (always populated when scorecard exists, independent of thesis)
    scorecard_dcf: float | None = None
    scorecard_epv: float | None = None
    scorecard_graham: float | None = None
    scorecard_price: float | None = None

    # Expectations gap (mirrors scorecard_dcf passthrough pattern)
    scorecard_implied_growth: float | None = None
    scorecard_supportable_growth: float | None = None
    scorecard_expectations_gap: float | None = None
    scorecard_expectations_gap_bucket: str | None = None
    scorecard_expectations_gap_line: str | None = None

    # Citations (populated during assembly from evidence results)
    citations: list[ReportCitation] = field(default_factory=list)

    # Latest evidence across filings and current events
    latest_evidence_date: str | None = None
    latest_evidence_source_type: str | None = None

    # Conviction (populated post-construction by compute_conviction)
    conviction_score: int | None = None
    conviction_class: str | None = None

    # Report artifact path (set after markdown write in run_deep_research)
    run_id: str | None = None
    report_path: str | None = None
    artifact_path: str | None = None

    # Analyst notes (Stage A) — None when disabled or skipped
    analyst_notes: AnalystNotes | None = None
    # Merged findings (Stage C) — None when Stage A didn't run
    merged_findings: MergedFindings | None = None

    # Pipeline warnings — surfaced in report when non-empty
    warnings: list[str] = field(default_factory=list)

    # Analysis scope
    analysis_years: int = 5
    analysis_quarters: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchReport":
        """Reconstruct from a plain dict (e.g., deserialized JSON from DB).

        Handles nested dataclasses: ThesisResult, ThesisAdjustment, UnresolvedEvidence.
        """
        d = dict(data)

        # Reconstruct thesis
        thesis_data = d.get("thesis")
        if thesis_data is not None and isinstance(thesis_data, dict):
            td = dict(thesis_data)
            td["adjustments"] = [ThesisAdjustment(**a) for a in td.get("adjustments", [])]
            td["unresolved"] = [UnresolvedEvidence(**u) for u in td.get("unresolved", [])]
            d["thesis"] = ThesisResult(**td)

        # Reconstruct unresolved lists
        d["researchable_items"] = [
            UnresolvedEvidence(**item) if isinstance(item, dict) else item
            for item in d.get("researchable_items", [])
        ]
        d["not_researchable_items"] = [
            UnresolvedEvidence(**item) if isinstance(item, dict) else item
            for item in d.get("not_researchable_items", [])
        ]

        # Reconstruct citations
        d["citations"] = [
            ReportCitation(**c) if isinstance(c, dict) else c for c in d.get("citations", [])
        ]

        # Reconstruct analyst_notes
        an_data = d.get("analyst_notes")
        if an_data is not None and isinstance(an_data, dict):
            from app.research.analyst_notes import AnalystNotes, AnalystNote, AnalystCitation

            def _rebuild_notes(items: list) -> list[AnalystNote]:
                rebuilt = []
                for item in items:
                    if isinstance(item, dict):
                        citations = [
                            AnalystCitation(**c) if isinstance(c, dict) else c
                            for c in item.get("citations", [])
                        ]
                        rebuilt.append(
                            AnalystNote(
                                category=item["category"],
                                claim=item["claim"],
                                direction=item.get("direction"),
                                severity=item["severity"],
                                citations=citations,
                                suggested_adjustment=item.get("suggested_adjustment"),
                                validation_status=item.get("validation_status", "UNVERIFIED"),
                            )
                        )
                    else:
                        rebuilt.append(item)
                return rebuilt

            d["analyst_notes"] = AnalystNotes(
                ticker=an_data["ticker"],
                positives=_rebuild_notes(an_data.get("positives", [])),
                risks=_rebuild_notes(an_data.get("risks", [])),
                surprises=_rebuild_notes(an_data.get("surprises", [])),
                adjustment_triggers=_rebuild_notes(an_data.get("adjustment_triggers", [])),
                overall_assessment=an_data.get("overall_assessment", ""),
                filing_sections_read=an_data.get("filing_sections_read", []),
            )

        # Reconstruct merged_findings
        mf_data = d.get("merged_findings")
        if mf_data is not None and isinstance(mf_data, dict):
            from app.research.findings_reconciliation import (
                MergedFindings,
                MatchedFinding,
                PipelineFinding,
            )
            from app.research.analyst_notes import (
                AnalystNote as _AnalystNote,
                AnalystCitation as _AnalystCitation,
            )

            def _rebuild_note(item: dict) -> _AnalystNote:
                citations = [
                    _AnalystCitation(**c) if isinstance(c, dict) else c
                    for c in item.get("citations", [])
                ]
                return _AnalystNote(
                    category=item["category"],
                    claim=item["claim"],
                    direction=item.get("direction"),
                    severity=item["severity"],
                    citations=citations,
                    suggested_adjustment=item.get("suggested_adjustment"),
                    validation_status=item.get("validation_status", "UNVERIFIED"),
                )

            def _rebuild_pf(item: dict) -> PipelineFinding:
                cw = item.get("content_words", [])
                return PipelineFinding(
                    claim=item["claim"],
                    direction=item["direction"],
                    section=item.get("section"),
                    content_words=set(cw) if isinstance(cw, list) else cw,
                    source_hypothesis=item["source_hypothesis"],
                    evidence_status=item["evidence_status"],
                )

            both = []
            for m in mf_data.get("both_paths", []):
                if isinstance(m, dict):
                    both.append(
                        MatchedFinding(
                            llm_note=_rebuild_note(m["llm_note"])
                            if isinstance(m["llm_note"], dict)
                            else m["llm_note"],
                            pipeline_finding=_rebuild_pf(m["pipeline_finding"])
                            if isinstance(m["pipeline_finding"], dict)
                            else m["pipeline_finding"],
                            shared_words=m.get("shared_words", []),
                        )
                    )
                else:
                    both.append(m)

            llm_only = [
                _rebuild_note(n) if isinstance(n, dict) else n for n in mf_data.get("llm_only", [])
            ]
            pipe_only = [
                _rebuild_pf(p) if isinstance(p, dict) else p
                for p in mf_data.get("pipeline_only", [])
            ]

            d["merged_findings"] = MergedFindings(
                both_paths=both,
                llm_only=llm_only,
                pipeline_only=pipe_only,
                agreement_score=mf_data.get("agreement_score", 0.0),
            )

        # Drop legacy opus_validation payloads from persisted artifacts (removed feature)
        d.pop("opus_validation", None)

        return cls(**d)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify_unresolved(
    unresolved: list[UnresolvedEvidence],
) -> tuple[list[UnresolvedEvidence], list[UnresolvedEvidence]]:
    """Split unresolved items into researchable vs not_researchable."""
    researchable = []
    not_researchable = []
    for item in unresolved:
        if item.unresolved_reason == "NOT_FOUND":
            not_researchable.append(item)
        else:
            researchable.append(item)
    return researchable, not_researchable


def _latest_filing_from_context(filing_context: FilingContext) -> FilingDocument | None:
    return filing_context.latest_document


def _latest_evidence_metadata(
    filing_context: FilingContext,
    current_event_context: CurrentEventContext | None,
) -> tuple[str | None, str | None]:
    latest_filing = filing_context.latest_document
    latest_event = (
        current_event_context.latest_document if current_event_context is not None else None
    )

    filing_date = latest_filing.filing_date if latest_filing is not None else None
    event_date = latest_event.published_at if latest_event is not None else None

    def _as_sortable(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            if "T" in value:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
            return datetime.fromisoformat(f"{value}T00:00:00+00:00").astimezone(timezone.utc)
        except ValueError:
            return None

    filing_dt = _as_sortable(filing_date)
    event_dt = _as_sortable(event_date)

    if filing_dt is not None and event_dt is not None:
        return (
            (event_date, latest_event.source_type)
            if event_dt > filing_dt
            else (filing_date, latest_filing.form_type)
        )
    if event_date:
        return event_date, latest_event.source_type
    if filing_date:
        return filing_date, latest_filing.form_type
    return None, None


def _run_analyst_stage_from_context(
    filing_context: FilingContext,
    current_event_context: CurrentEventContext | None,
    scorecard: dict[str, Any],
    ticker: str,
    hypotheses: list | None,
    evidence_results: list | None,
    report: ResearchReport,
    financial_integrity_scope: BoundV1FinancialScope | None = None,
    financial_scenarios: FinancialScenarioSource | None = None,
) -> None:
    """Run Stage A (analyst notes) + Stage C (reconciliation) on a report.

    Mutates report in place. Stage A runs whenever filing text is available.
    Stage C runs only when both Stage A and Stage B produced output.
    """
    from app.config import get_config

    cfg = get_config()
    if not cfg.analyst_notes_enabled:
        return

    try:
        from app.research.analyst_notes import generate_analyst_notes_from_filing_context
        from app.research.findings_reconciliation import reconcile_findings

        analyst_notes = generate_analyst_notes_from_filing_context(
            filing_context,
            scorecard=scorecard,
            ticker=ticker,
            current_event_context=current_event_context,
            financial_integrity_scope=financial_integrity_scope,
            financial_scenarios=financial_scenarios,
        )
        if analyst_notes is None:
            return
        report.analyst_notes = analyst_notes

        # Stage C: reconciliation only when both paths produced output.
        # Spec: "Either path missing → no reconciliation."
        if hypotheses and evidence_results:
            merged = reconcile_findings(
                analyst_notes,
                hypotheses,
                evidence_results,
            )
            report.merged_findings = merged
    except InvalidFinancialInputError:
        raise
    except Exception:
        logger.exception("deep_research: analyst notes failed for %s", ticker)


def _run_analyst_stage(
    filing_html: str,
    form_type: str | None,
    scorecard: dict[str, Any],
    ticker: str,
    hypotheses: list | None,
    evidence_results: list | None,
    report: ResearchReport,
    financial_integrity_scope: BoundV1FinancialScope | None = None,
    financial_scenarios: FinancialScenarioSource | None = None,
) -> None:
    filing_context = build_inline_filing_context(
        filing_html,
        ticker=ticker,
        form_type=form_type,
    )
    _run_analyst_stage_from_context(
        filing_context,
        None,
        scorecard,
        ticker,
        hypotheses,
        evidence_results,
        report,
        financial_integrity_scope,
        financial_scenarios,
    )


def _summary_counts(thesis: ThesisResult | None) -> tuple[int, int, int]:
    """Extract (total_adjustments, fact_calibrated_count, heuristic_count)."""
    if thesis is None:
        return 0, 0, 0
    total = len(thesis.adjustments)
    fact = sum(1 for a in thesis.adjustments if a.adjustment_confidence == "FACT_CALIBRATED")
    heuristic = sum(1 for a in thesis.adjustments if a.adjustment_confidence == "HEURISTIC")
    return total, fact, heuristic


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _expectations_gap_passthrough(
    scorecard: dict[str, Any],
) -> tuple[float | None, float | None, float | None, str | None, str | None]:
    """Derive the expectations-gap fields from the reverse-DCF valuation row.

    Returns (implied_growth, supportable_growth, gap, bucket, line). All None
    when the inputs are missing; the gap is bucketed by the single canonical
    authority (app.valuation.expectations_gap) so the verdict path and the memo
    agree. A missing/saturated/UNKNOWN leg yields a silent (None bucket) signal
    rather than a fabricated one.

    DATA SOURCE: the implied-growth and supportable legs live in the
    method='reverse_dcf' valuation row, NOT the method='scorecard' row. The
    orchestration layer (run_deep_research -> _load_reverse_dcf) merges that row
    under scorecard['reverse_dcf'] before this pure helper runs, so the live
    memo/verdict paths read real data rather than always-missing scorecard keys.

    When the reverse_dcf row already carries a persisted expectations_gap dict
    (written by valuation_writer via the same canonical module, at the
    TOP LEVEL of outputs_json — sibling to "outputs"), that dict is the
    authority and is used verbatim. Legacy rows lacking it are recomputed here
    from implied_growth + revenue_cagr_5y_used so the signal activates
    incrementally without a backfill.
    """
    from app.valuation.expectations_gap import (
        BUCKET_UNRELIABLE,
        compute_expectations_gap,
        estimate_supportable_growth,
    )

    reverse = scorecard.get("reverse_dcf") or {}
    reverse_outputs = reverse.get("outputs") or {}

    # Prefer the persisted expectations_gap dict — single authority.
    # The writer persists it at the TOP LEVEL of the reverse_dcf outputs_json
    # (sibling to "outputs"/"revenue_cagr_5y_used"), NOT nested under "outputs"
    # — matching the reader in sector_financial_packets. The
    # implied-growth leg, however, lives under "outputs".
    persisted = reverse.get("expectations_gap")
    if isinstance(persisted, dict) and persisted.get("bucket"):
        implied = reverse_outputs.get("implied_growth")
        supportable = persisted.get("supportable_growth")
        gap = persisted.get("gap")
        bucket = persisted.get("bucket")
        line = persisted.get("line")
        if bucket == BUCKET_UNRELIABLE:
            return (None, supportable, None, None, None)
        implied_out = implied if _is_real_number(implied) else None
        return (implied_out, supportable, gap, bucket, line)

    # Legacy / not-yet-recomputed row: rebuild via the canonical authority.
    implied = reverse_outputs.get("implied_growth")
    saturated = bool(reverse_outputs.get("implied_growth_saturated", False))

    revenue_cagr = reverse.get("revenue_cagr_5y_used")
    if not _is_real_number(revenue_cagr):
        revenue_cagr = None

    supportable, _basis = estimate_supportable_growth(
        revenue_cagr_5y=revenue_cagr,
        owner_earnings_cagr_5y=None,
        quality_flags=[],
    )
    gap_dict = compute_expectations_gap(implied, supportable, saturated)
    bucket = gap_dict.get("bucket")
    if bucket == BUCKET_UNRELIABLE:
        return (None, supportable, None, None, None)

    implied_out = implied if _is_real_number(implied) else None
    return (
        implied_out,
        supportable,
        gap_dict.get("gap"),
        bucket,
        gap_dict.get("line"),
    )


def _extract_citations(evidence_results: list[Any]) -> list[ReportCitation]:
    """Extract deduplicated citations from evidence results for report rendering.

    Iterates over EvidenceResult -> EvidenceItemResult -> Citation objects.
    Skips items with status NOT_FOUND. Deduplicates by (block_id, excerpt).
    First need_id and hypothesis_source encountered win for the primary fields.

    When two evidence items cite the same filing block, the deduplicated
    citation records ALL need_ids in additional_need_ids so that citation
    handle mapping works for every adjustment, not just the first one.

    Typed as list[Any] to avoid importing evidence_searcher types at module level,
    but expects list[EvidenceResult] from app.research.evidence_searcher.
    """
    seen: dict[tuple[str, str, str | None], int] = {}  # key -> index into citations list
    citations: list[ReportCitation] = []
    counter = 0

    for er in evidence_results:
        hyp_source = er.hypothesis.source
        for item in er.evidence_item_results:
            if item.status == "NOT_FOUND":
                continue
            for cite in item.citations:
                key = (cite.block_id, cite.excerpt, getattr(cite, "source_accession", None))
                if key in seen:
                    # Track additional need_id for handle mapping
                    existing = citations[seen[key]]
                    if item.need_id != existing.need_id:
                        existing.additional_need_ids.append(item.need_id)
                    continue
                counter += 1
                rc = ReportCitation(
                    citation_id=f"C{counter}",
                    need_id=item.need_id,
                    section=cite.section,
                    excerpt=cite.excerpt,
                    relevance=item.needed,
                    hypothesis_source=hyp_source,
                    item_code=getattr(cite, "item_code", None),
                    event_category=getattr(cite, "event_category", None),
                    source_form_type=getattr(cite, "source_form_type", None),
                    source_filing_date=getattr(cite, "source_filing_date", None),
                    source_accession=getattr(cite, "source_accession", None),
                    source_role=getattr(cite, "source_role", None),
                    source_type=getattr(cite, "source_type", None),
                    source_title=getattr(cite, "source_title", None),
                    source_url=getattr(cite, "source_url", None),
                    source_published_at=getattr(cite, "source_published_at", None),
                    source_quality=getattr(cite, "source_quality", None),
                )
                seen[key] = len(citations)
                citations.append(rc)

    return citations


def _empty_report(
    ticker: str,
    as_of_date: str,
    status: str,
    started_at: str,
    *,
    scorecard_present: bool = False,
    filing_present: bool = False,
    filing_date: str | None = None,
    form_type: str | None = None,
    anomaly_count: int = 0,
    solvency_status: str | None = None,
    filing_risk_status: str | None = None,
    gate_action: str | None = None,
    latest_evidence_date: str | None = None,
    latest_evidence_source_type: str | None = None,
    hypotheses_generated: int = 0,
    methods_agree: bool | None,
    consensus_strength: int | None,
    method_count: int | None,
    tension_type: str | None,
    scorecard_dcf: float | None = None,
    scorecard_epv: float | None = None,
    scorecard_graham: float | None = None,
    scorecard_price: float | None = None,
    scorecard_implied_growth: float | None = None,
    scorecard_supportable_growth: float | None = None,
    scorecard_expectations_gap: float | None = None,
    scorecard_expectations_gap_bucket: str | None = None,
    scorecard_expectations_gap_line: str | None = None,
) -> ResearchReport:
    """Build a ResearchReport for non-OK statuses."""
    return ResearchReport(
        ticker=ticker,
        as_of_date=as_of_date,
        status=status,
        started_at=started_at,
        completed_at=_utc_now_iso(),
        scorecard_present=scorecard_present,
        filing_present=filing_present,
        filing_date=filing_date,
        form_type=form_type,
        latest_evidence_date=latest_evidence_date,
        latest_evidence_source_type=latest_evidence_source_type,
        anomaly_count=anomaly_count,
        solvency_status=solvency_status,
        filing_risk_status=filing_risk_status,
        gate_action=gate_action,
        investigation_ran=False,
        hypotheses_generated=hypotheses_generated,
        thesis=None,
        researchable_items=[],
        not_researchable_items=[],
        total_adjustments=0,
        fact_calibrated_count=0,
        heuristic_count=0,
        methods_agree=methods_agree,
        consensus_strength=consensus_strength,
        method_count=method_count,
        tension_type=tension_type,
        scorecard_dcf=scorecard_dcf,
        scorecard_epv=scorecard_epv,
        scorecard_graham=scorecard_graham,
        scorecard_price=scorecard_price,
        scorecard_implied_growth=scorecard_implied_growth,
        scorecard_supportable_growth=scorecard_supportable_growth,
        scorecard_expectations_gap=scorecard_expectations_gap,
        scorecard_expectations_gap_bucket=scorecard_expectations_gap_bucket,
        scorecard_expectations_gap_line=scorecard_expectations_gap_line,
    )


# ---------------------------------------------------------------------------
# Pure Pipeline
# ---------------------------------------------------------------------------


def assemble_research(
    ticker: str,
    as_of_date: str,
    scorecard: dict[str, Any],
    tensions: dict[str, Any],
    anomalies: list[Anomaly],
    quality_ctx: dict[str, Any],
    solvency: SolvencyAssessment | None,
    filing_risk: dict[str, Any] | None,
    filing_html: str | None,
    form_type: str | None,
    current_event_context: CurrentEventContext | None = None,
    financial_integrity_scope: BoundV1FinancialScope | None = None,
) -> ResearchReport:
    filing_context = build_inline_filing_context(
        filing_html,
        ticker=ticker,
        form_type=form_type,
    )
    return assemble_research_from_filing_context(
        ticker=ticker,
        as_of_date=as_of_date,
        scorecard=scorecard,
        tensions=tensions,
        anomalies=anomalies,
        quality_ctx=quality_ctx,
        solvency=solvency,
        filing_risk=filing_risk,
        filing_context=filing_context,
        current_event_context=current_event_context,
        financial_integrity_scope=financial_integrity_scope,
    )


def _deep_research_financial_inputs(
    *,
    scorecard: dict[str, Any],
    anomalies: Sequence[Anomaly],
    solvency: SolvencyAssessment | None,
) -> dict[str, Any]:
    """Return the literal deterministic inputs used to form research claims."""

    return {
        "scorecard": scorecard,
        "anomalies": [asdict(anomaly) for anomaly in anomalies],
        "solvency": asdict(solvency) if solvency is not None else None,
    }


def assemble_research_from_filing_context(
    ticker: str,
    as_of_date: str,
    scorecard: dict[str, Any],
    tensions: dict[str, Any],
    anomalies: list[Anomaly],
    quality_ctx: dict[str, Any],
    solvency: SolvencyAssessment | None,
    filing_risk: dict[str, Any] | None,
    filing_context: FilingContext,
    current_event_context: CurrentEventContext | None = None,
    financial_integrity_scope: BoundV1FinancialScope | None = None,
) -> ResearchReport:
    """Assemble a complete research report from upstream outputs.

    No DB access or file I/O. Task 5 may make LLM network calls
    internally for evidence adjudication.
    Calls generate_hypotheses, search_evidence, update_thesis in sequence.
    """
    from app.research.hypothesis_generator import generate_hypotheses
    from app.research.evidence_searcher import search_evidence_in_filing_context
    from app.research.thesis_updater import update_thesis

    started_at = _utc_now_iso()
    gate_action = quality_ctx.get("gate_action")
    solvency_status = getattr(solvency, "status", None) if solvency else None
    filing_risk_status = filing_risk.get("status") if filing_risk else None
    latest_filing = _latest_filing_from_context(filing_context)
    latest_evidence_date, latest_evidence_source_type = _latest_evidence_metadata(
        filing_context,
        current_event_context,
    )

    # Extract scorecard-level values (available even without investigation)
    pzd = scorecard.get("pricing_zone_detail") or {}
    discounts = scorecard.get("discounts") or {}
    sc_dcf = pzd.get("dcf_base")
    sc_epv = pzd.get("epv_adjusted")
    sc_price = pzd.get("current_price")
    # Graham: invert the TEXTBOOK discount d=(iv-price)/iv => iv = price/(1-d)
    # (audit: graham-discount-inversion)
    sc_graham = graham_value_from_textbook_discount(
        sc_price if isinstance(sc_price, (int, float)) else None,
        discounts.get("graham") if isinstance(discounts.get("graham"), (int, float)) else None,
    )

    # Expectations gap: source implied growth from the reverse-DCF valuation
    # row (merged under scorecard['reverse_dcf'] by the orchestration layer) and
    # bucket it via the canonical authority. Defaults to a silent (None) signal
    # when any leg is missing/saturated, so legacy/incomplete scorecards never
    # crash and never fabricate a bucket.
    (
        sc_implied,
        sc_supportable,
        sc_gap,
        sc_gap_bucket,
        sc_gap_line,
    ) = _expectations_gap_passthrough(scorecard)
    financial_scenarios: FinancialScenarioSource | None = None
    if financial_integrity_scope is not None:

        def current_financial_scenarios() -> tuple[Any, ...]:
            return (
                financial_input_scenario(
                    financial_integrity_scope.packets[0],
                    financial_inputs=_deep_research_financial_inputs(
                        scorecard=scorecard,
                        anomalies=anomalies,
                        solvency=solvency,
                    ),
                ),
            )

        financial_scenarios = current_financial_scenarios

    def require_exact_scope() -> None:
        if financial_integrity_scope is None:
            return
        current_scenarios = (
            financial_scenarios() if callable(financial_scenarios) else financial_scenarios
        )
        financial_integrity_scope.require(scenarios=current_scenarios)

    require_exact_scope()

    common = dict(
        scorecard_present=True,
        filing_present=latest_filing is not None,
        filing_date=latest_filing.filing_date if latest_filing is not None else None,
        form_type=latest_filing.form_type if latest_filing is not None else None,
        latest_evidence_date=latest_evidence_date,
        latest_evidence_source_type=latest_evidence_source_type,
        anomaly_count=len(anomalies),
        solvency_status=solvency_status,
        filing_risk_status=filing_risk_status,
        gate_action=gate_action,
        methods_agree=tensions.get("methods_agree"),
        consensus_strength=tensions.get("consensus_strength"),
        method_count=tensions.get("method_count"),
        tension_type=tensions.get("tension_type"),
        scorecard_dcf=sc_dcf,
        scorecard_epv=sc_epv,
        scorecard_graham=sc_graham,
        scorecard_price=sc_price,
        scorecard_implied_growth=sc_implied,
        scorecard_supportable_growth=sc_supportable,
        scorecard_expectations_gap=sc_gap,
        scorecard_expectations_gap_bucket=sc_gap_bucket,
        scorecard_expectations_gap_line=sc_gap_line,
    )

    # Stage 1: Hypothesis generation
    hypotheses = generate_hypotheses(
        ticker,
        valuation=scorecard,
        tensions=tensions,
        anomalies=anomalies,
        quality_ctx=quality_ctx,
        solvency=solvency,
        filing_risk=filing_risk,
    )

    if not hypotheses:
        report = _empty_report(
            ticker, as_of_date, "NO_HYPOTHESES", started_at, hypotheses_generated=0, **common
        )
        if latest_filing is not None:
            _run_analyst_stage_from_context(
                filing_context,
                current_event_context,
                scorecard,
                ticker,
                None,
                None,
                report,
                financial_integrity_scope,
                financial_scenarios,
            )
        require_exact_scope()
        return report

    # Stage 2: Evidence search
    if latest_filing is None:
        report = _empty_report(
            ticker,
            as_of_date,
            "NO_FILING",
            started_at,
            hypotheses_generated=len(hypotheses),
            **common,
        )
        require_exact_scope()
        return report

    evidence_results = search_evidence_in_filing_context(
        hypotheses,
        filing_context,
        current_event_context=current_event_context,
        max_hypotheses=10,
        financial_integrity_scope=financial_integrity_scope,
        financial_scenarios=financial_scenarios,
    )

    # Extract citations before evidence_results fall out of scope
    citations = _extract_citations(evidence_results)

    # Stage 3: Thesis update
    thesis = update_thesis(ticker, scorecard, tensions, evidence_results, iteration=0)

    # Stage 4: Unresolved classification
    researchable, not_researchable = _classify_unresolved(thesis.unresolved)
    total_adj, fact_count, heuristic_count = _summary_counts(thesis)

    report = ResearchReport(
        ticker=ticker,
        as_of_date=as_of_date,
        status="OK",
        started_at=started_at,
        completed_at=_utc_now_iso(),
        scorecard_present=True,
        filing_present=True,
        filing_date=latest_filing.filing_date if latest_filing is not None else None,
        form_type=latest_filing.form_type if latest_filing is not None else None,
        latest_evidence_date=latest_evidence_date,
        latest_evidence_source_type=latest_evidence_source_type,
        anomaly_count=len(anomalies),
        solvency_status=solvency_status,
        filing_risk_status=filing_risk_status,
        gate_action=gate_action,
        investigation_ran=True,
        hypotheses_generated=len(hypotheses),
        thesis=thesis,
        researchable_items=researchable,
        not_researchable_items=not_researchable,
        total_adjustments=total_adj,
        fact_calibrated_count=fact_count,
        heuristic_count=heuristic_count,
        citations=citations,
        methods_agree=tensions.get("methods_agree"),
        consensus_strength=tensions.get("consensus_strength"),
        method_count=tensions.get("method_count"),
        tension_type=tensions.get("tension_type"),
        scorecard_dcf=sc_dcf,
        scorecard_epv=sc_epv,
        scorecard_graham=sc_graham,
        scorecard_price=sc_price,
        scorecard_implied_growth=sc_implied,
        scorecard_supportable_growth=sc_supportable,
        scorecard_expectations_gap=sc_gap,
        scorecard_expectations_gap_bucket=sc_gap_bucket,
        scorecard_expectations_gap_line=sc_gap_line,
    )
    if latest_filing is not None:
        _run_analyst_stage_from_context(
            filing_context,
            current_event_context,
            scorecard,
            ticker,
            hypotheses,
            evidence_results,
            report,
            financial_integrity_scope,
            financial_scenarios,
        )
    require_exact_scope()
    return report


# ---------------------------------------------------------------------------
# I/O Helpers (used by run_deep_research only)
# ---------------------------------------------------------------------------


def _load_scorecard(
    ticker: str, as_of_date: str | None
) -> tuple[dict[str, Any] | None, str | None]:
    """Load the newest exact-source scorecard or return ``(None, None)``."""
    from app.db import get_db

    with get_db() as conn:
        row = latest_decision_eligible_valuation_row(
            conn,
            ticker=ticker,
            method="scorecard",
            as_of_date=as_of_date,
            exact_as_of_date=as_of_date is not None,
        )

    if not row:
        return None, None

    scorecard = json.loads(row["outputs_json"] or "{}")
    return scorecard, row["as_of_date"]


def _load_reverse_dcf(ticker: str, as_of_date: str | None) -> dict[str, Any] | None:
    """Load the reverse-DCF valuation row (method='reverse_dcf') from the DB.

    The expectations-gap implied/supportable legs live in THIS row, not the
    method='scorecard' row. Returns the parsed outputs_json dict (carrying
    revenue_cagr_5y_used, feasibility, the nested outputs.implied_growth, and a
    TOP-LEVEL expectations_gap dict — sibling to "outputs"), or None when no row
    exists. Pinned to as_of_date when supplied so it date-aligns with the
    scorecard.
    """
    from app.db import get_db

    with get_db() as conn:
        row = latest_decision_eligible_valuation_row(
            conn,
            ticker=ticker,
            method="reverse_dcf",
            as_of_date=as_of_date,
            exact_as_of_date=as_of_date is not None,
        )

    if not row:
        return None
    return json.loads(row["outputs_json"] or "{}")


def _load_filing_context(
    ticker: str,
    as_of_date: str,
    quarters: int = 0,
    *,
    include_material_events: bool = True,
    allow_network_materialization: bool = True,
    cfg: AppConfig | None = None,
    allowed_filing_roots: tuple[str | Path, ...] = (),
) -> FilingContext:
    """Load the canonical filing context pinned to as_of_date."""
    return load_research_filing_context(
        ticker,
        as_of_date=as_of_date,
        quarters=quarters,
        include_material_events=include_material_events,
        allow_network_materialization=allow_network_materialization,
        cfg=cfg,
        allowed_filing_roots=allowed_filing_roots,
    )


def _load_current_event_context(
    ticker: str,
    as_of_date: str,
) -> CurrentEventContext:
    return load_current_event_context(ticker, as_of_date=as_of_date)


def _ensure_recent_filing_context_cache(ticker: str, as_of_date: str, quarters: int) -> list[str]:
    if int(quarters) <= 0:
        return []
    upper = ticker.upper()
    try:
        from app.config import get_config
        from app.ingest.filings import ingest_filings_between
        from app.parse.filing_parser import parse_pending_filings

        cfg = get_config()
        as_of = date.fromisoformat(as_of_date)
        since = as_of - timedelta(days=max(1, int(cfg.filing_lookback_days_10q)))
        ingested = ingest_filings_between(
            since,
            as_of,
            ["10-Q", "10-Q/A", "8-K", "8-K/A"],
            as_of_date=as_of_date,
            tickers=[upper],
            limit=1,
        )
        parse_limit = max(10, min(40, int(ingested) + int(quarters) + 4))
        parse_pending_filings(limit=parse_limit, tickers=[upper])
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        logger.warning("deep_research: recent filing context refresh failed for %s: %s", upper, exc)
        return [f"recent_filing_context_refresh_failed: {exc}"]
    return []


def _load_filing_html(ticker: str, as_of_date: str) -> tuple[str | None, str | None, str | None]:
    """Backwards-compatible single-filing loader for annual-only callers."""
    filing_context = _load_filing_context(
        ticker,
        as_of_date,
        quarters=0,
        include_material_events=False,
    )
    document = _latest_filing_from_context(filing_context)
    if document is None:
        return None, None, None
    return document.html, document.form_type, document.filing_date


def _persist_to_db(
    ticker: str, as_of_date: str, report: ResearchReport, warnings: list[str]
) -> None:
    """Persist ResearchReport to valuations table."""
    from dataclasses import asdict

    from app.db import get_db
    from app.valuation.lineage import (
        sha256_file,
        valuation_integrity_fingerprint,
    )
    from app.valuation.valuation_writer import _archive_valuation_row

    inputs = {
        "scorecard_as_of_date": as_of_date,
        "form_type": report.form_type,
        "filing_present": report.filing_present,
        "hypotheses_generated": report.hypotheses_generated,
    }
    inputs_json = json.dumps(inputs)
    outputs_json = json.dumps(asdict(report), default=str)
    warnings_json = json.dumps(warnings)
    created_at = _utc_now_iso()
    source_run_id = str(report.run_id or "").strip() or None
    source_path: Path | None = None
    source_sha256: str | None = None
    if source_run_id and report.artifact_path:
        candidate = Path(report.artifact_path).expanduser()
        try:
            resolved = candidate.resolve(strict=True)
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if (
            isinstance(payload, dict)
            and str(resolved) == str(candidate)
            and payload == json.loads(outputs_json)
            and str(payload.get("run_id") or "") == source_run_id
            and str(payload.get("ticker") or "").strip().upper() == ticker.upper()
        ):
            source_path = resolved
            source_sha256 = sha256_file(resolved)
    lineage_values = {
        "ticker": ticker.upper(),
        "as_of_date": as_of_date,
        "method": "deep_research",
        "inputs_json": inputs_json,
        "outputs_json": outputs_json,
        "warnings_json": warnings_json,
        "created_at": created_at,
        "valuation_writer_version": "deep_research_v1",
        "quality_gate_verdict": None,
        "confidence_class": None,
        "gate_reason_codes": None,
        "valuation_headwinds": None,
        "valuation_supports": None,
        "source_run_id": source_run_id,
        "source_artifact_path": str(source_path) if source_path is not None else None,
        "source_artifact_sha256": source_sha256,
    }
    fingerprint = valuation_integrity_fingerprint(lineage_values)

    with get_db() as conn:
        _archive_valuation_row(
            conn,
            ticker=ticker,
            as_of_date=as_of_date,
            method="deep_research",
            new_outputs_json=outputs_json,
            new_source_run_id=source_run_id,
            new_source_artifact_path=(str(source_path) if source_path is not None else None),
            new_source_artifact_sha256=source_sha256,
            new_financial_integrity_fingerprint=fingerprint,
        )
        conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at, valuation_writer_version,
                source_run_id, source_artifact_path, source_artifact_sha256,
                financial_integrity_fingerprint
            ) VALUES (?, ?, 'deep_research', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, as_of_date, method) DO UPDATE SET
                inputs_json = excluded.inputs_json,
                outputs_json = excluded.outputs_json,
                warnings_json = excluded.warnings_json,
                created_at = excluded.created_at,
                valuation_writer_version = excluded.valuation_writer_version,
                quality_gate_verdict = NULL,
                confidence_class = NULL,
                gate_reason_codes = NULL,
                valuation_headwinds = NULL,
                valuation_supports = NULL,
                source_run_id = excluded.source_run_id,
                source_artifact_path = excluded.source_artifact_path,
                source_artifact_sha256 = excluded.source_artifact_sha256,
                financial_integrity_fingerprint =
                    excluded.financial_integrity_fingerprint
            """,
            (
                ticker.upper(),
                as_of_date,
                inputs_json,
                outputs_json,
                warnings_json,
                created_at,
                "deep_research_v1",
                source_run_id,
                str(source_path) if source_path is not None else None,
                source_sha256,
                fingerprint,
            ),
        )


def _artifact_base_path(ticker: str, as_of_date: str) -> str:
    """Generate shared base path for JSON and markdown artifacts.

    Both artifact types derive filenames from this base to ensure
    they share a timestamp and never drift.
    """
    from app.config import get_config

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return str(get_config().research_dir / f"{ticker}_{as_of_date}_{ts}")


def _write_artifact(base_path: str, report: ResearchReport) -> str | None:
    """Write JSON filesystem artifact. Returns path on success, None on failure."""
    from dataclasses import asdict

    json_path = f"{base_path}.json"
    try:
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")
        return json_path
    except OSError as exc:
        logger.warning("deep_research: artifact write failed: %s", exc)
        return None


def _write_markdown_artifact(
    base_path: str, markdown: str, *, diagnostic: bool = False
) -> str | None:
    """Write markdown report artifact. Returns path on success, None on failure."""
    suffix = DIAGNOSTIC_REPORT_SUFFIX if diagnostic else STANDARD_REPORT_SUFFIX
    md_path = f"{base_path}{suffix}"
    try:
        p = Path(md_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown, encoding="utf-8")
        return md_path
    except OSError as exc:
        logger.warning("deep_research: markdown artifact write failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Module-level wrappers (thin shims for easy patching in tests)
# ---------------------------------------------------------------------------


def detect_anomalies(
    ticker: str,
    *,
    as_of_date: str | None = None,
    require_filed_asof: bool = False,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
) -> list[Anomaly]:
    """Import and call anomaly detector."""
    from app.alpha.anomaly_detector import detect_anomalies as _detect

    if as_of_date is None and not require_filed_asof and issuer_cik is None and not aliases:
        return _detect(ticker)
    return _detect(
        ticker,
        as_of_date=as_of_date,
        require_filed_asof=require_filed_asof,
        issuer_cik=issuer_cik,
        aliases=tuple(aliases),
    )


def assess_solvency(
    ticker: str,
    *,
    as_of_date: str | None = None,
    require_filed_asof: bool = False,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
) -> SolvencyAssessment:
    """Import and call solvency scanner."""
    from app.alpha.solvency_scanner import assess_solvency as _assess

    if as_of_date is None and not require_filed_asof and issuer_cik is None and not aliases:
        return _assess(ticker)
    return _assess(
        ticker,
        as_of_date=as_of_date,
        require_filed_asof=require_filed_asof,
        issuer_cik=issuer_cik,
        aliases=tuple(aliases),
    )


def scan_filing_risks(ticker: str, **kwargs: Any) -> dict[str, Any]:
    """Import and call filing risk scanner."""
    from app.alpha.filing_risk_scan import scan_filing_risks as _scan

    return _scan(ticker, **kwargs)


# ---------------------------------------------------------------------------
# Orchestration Entry Point
# ---------------------------------------------------------------------------


def run_deep_research(
    ticker: str,
    as_of_date: str | None = None,
    years: int = 5,
    quarters: int = 0,
) -> ResearchReport:
    """Full fundamental analysis pipeline for a single ticker.

    Ensures upstream data (facts, filing, scorecard) exists before running.
    Calls assemble_research() for the pure pipeline.
    Persists result to valuations table (method='deep_research') — canonical.
    Writes filesystem artifact to configured research output dir — derived, for human inspection.

    Args:
        years: Number of years of annual data to analyze (default 5).
        quarters: Number of recent quarters to include (default 0 = annual only).
    """
    started_at = _utc_now_iso()
    warnings: list[str] = []
    upper = ticker.upper()
    effective_date = as_of_date or date.today().isoformat()

    # --- Check LLM provider availability ---
    try:
        from app.llm.providers import get_llm_provider as _get_llm

        _llm = _get_llm()
        if _llm.provider_name == "disabled":
            warnings.append("llm_provider_disabled")
            logger.warning(
                "deep_research: LLM provider is DISABLED for %s — "
                "evidence adjudication and analyst notes will not run. "
                "Conviction scores will be degraded. "
                "Set VOE_LLM_PROVIDER and VOE_OPENAI_API_KEY in your environment.",
                upper,
            )
    except InvalidFinancialInputError:
        raise
    except Exception:
        warnings.append("llm_provider_disabled")

    # --- Ensure upstream data exists (no-ops if already cached) ---
    try:
        from app.ingest.facts_writer import ensure_all_facts

        ensure_all_facts(upper, years_back=years)
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        logger.warning("deep_research: facts ingestion failed for %s: %s", upper, exc)
        warnings.append(f"facts_ingestion_failed: {exc}")

    try:
        from app.dossier.collector import collect_10k_docket

        collect_10k_docket(ticker=upper, as_of_date=effective_date, years_back=years)
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        logger.warning("deep_research: filing collection failed for %s: %s", upper, exc)
        warnings.append(f"filing_collection_failed: {exc}")

    try:
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation(
            upper,
            effective_date,
            require_filed_asof=True,
        )
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        logger.warning("deep_research: valuation failed for %s: %s", upper, exc)
        warnings.append(f"valuation_failed: {exc}")

    # --- Load scorecard (should now exist after ensure_valuation) ---
    scorecard, resolved_date = _load_scorecard(upper, as_of_date)
    if scorecard is None:
        report = _empty_report(
            upper,
            as_of_date or date.today().isoformat(),
            "NO_SCORECARD",
            started_at,
            methods_agree=None,
            consensus_strength=None,
            method_count=None,
            tension_type=None,
        )
        report.warnings = warnings
        report.analysis_years = years
        report.analysis_quarters = quarters
        from app.valuation.conviction import compute_conviction

        conviction = compute_conviction(report)
        report.conviction_score = conviction.conviction_score
        report.conviction_class = conviction.conviction_class
        _persist_to_db(upper, report.as_of_date, report, warnings)
        return report

    effective_date = resolved_date or as_of_date or date.today().isoformat()

    # Merge the reverse-DCF row for deterministic, provider-free reporting.
    # The provider-bound scorecard is replaced below by the exact canonical
    # packet's raw valuation so a mutable latest-row read cannot cross the paid
    # boundary.
    try:
        reverse_dcf_row = _load_reverse_dcf(upper, resolved_date)
        if reverse_dcf_row is not None:
            scorecard["reverse_dcf"] = reverse_dcf_row
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        logger.warning("deep_research: reverse_dcf load failed for %s: %s", upper, exc)
        warnings.append(f"reverse_dcf_load_failed: {exc}")

    financial_context = build_canonical_v1_financial_context(
        tickers=[upper],
        as_of_date=effective_date,
        scorecard_evidence={upper: (resolved_date, scorecard)},
    )
    canonical_packet = financial_context.packets[upper]
    scorecard = copy.deepcopy(canonical_packet.raw_valuation)
    pricing_zone_detail = dict(scorecard.get("pricing_zone_detail") or {})
    pricing_zone_detail.update(
        {
            "current_price": canonical_packet.current_price,
            "dcf_base": canonical_packet.dcf_value,
            "epv_adjusted": canonical_packet.epv_value,
            "graham_value_per_share": canonical_packet.graham_value,
        }
    )
    scorecard["pricing_zone_detail"] = pricing_zone_detail
    issuer_aliases = tuple(
        dict.fromkeys(
            str(item).strip().upper()
            for item in (
                upper,
                canonical_packet.issuer_primary_ticker,
                *(canonical_packet.issuer_listed_tickers or ()),
            )
            if str(item or "").strip()
        )
    )

    # These deterministic research inputs can change hypotheses and report
    # status, so resolve them from the same issuer-bound filing cutoff before
    # authorizing any provider scenario. Strict solvency mode also disables
    # filing-network materialization.
    anomalies = detect_anomalies(
        upper,
        as_of_date=effective_date,
        require_filed_asof=True,
        issuer_cik=canonical_packet.issuer_cik,
        aliases=issuer_aliases,
    )
    try:
        solvency = assess_solvency(
            upper,
            as_of_date=effective_date,
            require_filed_asof=True,
            issuer_cik=canonical_packet.issuer_cik,
            aliases=issuer_aliases,
        )
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        logger.warning("deep_research: solvency scanner failed for %s: %s", upper, exc)
        warnings.append(f"solvency_scanner_failed: {exc}")
        solvency = None

    financial_scenario = financial_input_scenario(
        canonical_packet,
        financial_inputs=_deep_research_financial_inputs(
            scorecard=scorecard,
            anomalies=anomalies,
            solvency=solvency,
        ),
    )
    financial_integrity_scope = bind_v1_financial_scope(
        context=f"deep_research:{upper}:{effective_date}",
        run_as_of_date=effective_date,
        packets=(canonical_packet,),
        scenarios=(financial_scenario,),
    )
    from app.config import get_config

    research_cfg = get_config()
    allowed_filing_roots = (
        research_cfg.raw_filings_dir,
        research_cfg.cache_dir,
    )

    if quarters > 0:
        warnings.extend(
            warning
            for warning in _ensure_recent_filing_context_cache(upper, effective_date, quarters)
            if warning not in warnings
        )

    # Load remaining upstream data (filing risk is best-effort).
    quality_ctx = scorecard.get("quality_context") or {}

    try:
        filing_risk = scan_filing_risks(
            upper,
            as_of_date=effective_date,
            integrity_scope=financial_integrity_scope,
            allow_network_materialization=False,
            cfg=research_cfg,
            allowed_filing_roots=allowed_filing_roots,
        )
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        logger.warning("deep_research: filing risk scan failed for %s: %s", upper, exc)
        warnings.append(f"filing_risk_scan_failed: {exc}")
        filing_risk = None

    # Load filing context (pinned to as_of_date)
    filing_context = _load_filing_context(
        upper,
        effective_date,
        quarters=quarters,
        allow_network_materialization=False,
        cfg=research_cfg,
        allowed_filing_roots=allowed_filing_roots,
    )
    warnings.extend(warning for warning in filing_context.warnings if warning not in warnings)
    try:
        current_event_context = _load_current_event_context(upper, effective_date)
        warnings.extend(
            warning for warning in current_event_context.warnings if warning not in warnings
        )
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        logger.warning("deep_research: current event context failed for %s: %s", upper, exc)
        warnings.append(f"current_event_context_failed: {exc}")
        current_event_context = CurrentEventContext()
    latest_filing = _latest_filing_from_context(filing_context)

    # Compute method tensions from scorecard
    tensions = _compute_tensions_from_scorecard(scorecard, quality_ctx)

    # Run pure pipeline
    report = assemble_research_from_filing_context(
        ticker=upper,
        as_of_date=effective_date,
        scorecard=scorecard,
        tensions=tensions,
        anomalies=anomalies,
        quality_ctx=quality_ctx,
        solvency=solvency,
        filing_risk=filing_risk,
        filing_context=filing_context,
        current_event_context=current_event_context,
        financial_integrity_scope=financial_integrity_scope,
    )

    # Patch in latest filing metadata from orchestration layer.
    if latest_filing is not None:
        report.filing_date = latest_filing.filing_date
        report.form_type = latest_filing.form_type
    report.warnings = warnings
    report.analysis_years = years
    report.analysis_quarters = quarters

    # Task 8: Conviction scoring (post-pipeline, before persistence)
    from app.valuation.conviction import compute_conviction

    conviction = compute_conviction(report)
    report.conviction_score = conviction.conviction_score
    report.conviction_class = conviction.conviction_class

    # The exact financial inputs authorized for provider reasoning must still
    # match immediately before any investor-facing artifact or canonical row is
    # published.
    financial_integrity_scope.require(
        scenarios=(
            financial_input_scenario(
                canonical_packet,
                financial_inputs=_deep_research_financial_inputs(
                    scorecard=scorecard,
                    anomalies=anomalies,
                    solvency=solvency,
                ),
            ),
        )
    )

    # Artifacts: shared base path ensures JSON and markdown timestamps match
    base_path = _artifact_base_path(upper, effective_date)
    report.run_id = Path(base_path).name
    report.artifact_path = f"{base_path}.json"

    # Render Markdown first so the canonical JSON artifact and the DB row bind
    # the same final report object, including report_path and warnings.
    try:
        from app.research.report_renderer import build_view, render_report

        view = build_view(report)
        markdown = render_report(view)
        md_path = _write_markdown_artifact(
            base_path,
            markdown,
            diagnostic=view.artifact_quality == "DEGRADED_DIAGNOSTIC",
        )
        if md_path:
            report.report_path = md_path
        else:
            warnings.append("markdown_write_failed")
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        logger.warning("deep_research: markdown render/write failed: %s", exc)
        warnings.append(f"markdown_render_failed: {exc}")

    artifact_path = _write_artifact(base_path, report)
    if artifact_path is None:
        warnings.append("artifact_write_failed")
        report.artifact_path = None

    # DB persist (canonical) — happens regardless of artifact success
    _persist_to_db(upper, effective_date, report, warnings)

    return report


def _compute_tensions_from_scorecard(
    scorecard: dict[str, Any], quality_ctx: dict[str, Any]
) -> dict[str, Any]:
    """Extract method values from scorecard and compute tensions."""
    from app.valuation.method_tension import analyze_method_tensions

    pzd = scorecard.get("pricing_zone_detail") or {}
    discounts = scorecard.get("discounts") or {}
    wacc_detail = scorecard.get("wacc_detail") or {}

    dcf = pzd.get("dcf_base")
    epv = pzd.get("epv_adjusted")
    price = pzd.get("current_price")
    wacc = wacc_detail.get("adjusted_wacc") or 0.10
    terminal_growth = pzd.get("terminal_growth_used") or 0.015

    # Graham: invert the TEXTBOOK discount d=(iv-price)/iv => iv = price/(1-d)
    graham = graham_value_from_textbook_discount(
        price if isinstance(price, (int, float)) else None,
        discounts.get("graham") if isinstance(discounts.get("graham"), (int, float)) else None,
    )

    # Check for method_tension already in scorecard (valuation_writer computes it)
    existing = scorecard.get("method_tension")
    if existing and isinstance(existing, dict) and "tension_type" in existing:
        return existing

    return analyze_method_tensions(
        dcf_value=dcf,
        epv_value=epv,
        graham_value=graham,
        ncav_value=None,
        current_price=price,
        revenue_cagr_5y=quality_ctx.get("revenue_cagr_5y"),
        wacc=wacc,
        terminal_growth=terminal_growth,
    )
