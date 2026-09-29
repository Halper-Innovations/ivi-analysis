"""Adapters from the current deep-research runtime into AnalysisReport.

This is an interim bridge while the full analyst orchestrator is still being
landed. It lets the runtime emit the analyst-facing output contract today,
using the existing deep-research report as the underlying evidence.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from app.analyst.evidence_bundle import AnalysisEvidenceBundle
from app.analyst.thesis_contract import (
    AnalysisCitation,
    AnalysisFinding,
    AnalysisReport,
    ExpectationsGap,
    Falsifier,
    OpenQuestion,
    ValuationConclusion,
)
from app.research.report_renderer import build_view, derive_verdict


_VERDICT_MAP = {
    "PROCEED": "BUY",
    "WATCH": "WATCH",
    "DO NOT ACT": "STAY_AWAY",
}

_BUY_POSTURE_PREFIX = "BUY_POSTURE_BLOCKED:"
_RESEARCH_QUALITY_WARNING_PREFIXES = (
    "research_quality_unavailable",
    "llm_provider_disabled",
)
_CURRENT_CONTEXT_WARNING_PREFIXES = (
    "annual_filing_missing",
    "annual_filing_unreadable:",
    "current_event_context_failed",
    "current_event_gap:",
    "current_event_source_disabled:",
    "facts_ingestion_failed",
    "filing_context_failed",
    "filing_docket_failed",
    "filing_text_unavailable:",
    "no_annual_filings_available",
    "no_recent_quarterly_filings_cached",
    "quarterly_filings_partial:",
    "reverse_dcf_load_failed",
    "scorecard_load_failed",
    "scorecard_missing",
    "scorecard_missing_for_analysis_bundle",
    "scorecard_stale:",
    "valuation_failed",
)


def _finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def apply_investment_posture(
    report: AnalysisReport,
    *,
    has_unresolved_hard_blocker: bool = False,
) -> AnalysisReport:
    """Fail closed when a PROCEED research gate lacks investment inputs.

    The research gate remains explicit provenance. This function only limits
    the stronger BUY posture; it never promotes WATCH/STAY_AWAY or fabricates
    missing investment evidence. It is safe to call again after late-bound
    research-quality metadata has been attached.
    """

    if str(report.research_gate or "").upper() != "PROCEED":
        return report

    reasons: list[str] = []
    if not _finite_positive(report.valuation.price):
        reasons.append(f"{_BUY_POSTURE_PREFIX}MISSING_PRICE")
    if not _finite_positive(report.valuation.margin_of_safety):
        reasons.append(f"{_BUY_POSTURE_PREFIX}MISSING_MARGIN_OF_SAFETY")
    if not report.risks:
        reasons.append(f"{_BUY_POSTURE_PREFIX}MISSING_RISKS")
    if not report.falsifiers:
        reasons.append(f"{_BUY_POSTURE_PREFIX}MISSING_FALSIFIERS")

    warning_tokens = [str(item).strip().lower() for item in report.warnings]
    if any(
        token.startswith(_RESEARCH_QUALITY_WARNING_PREFIXES)
        for token in warning_tokens
    ):
        reasons.append(f"{_BUY_POSTURE_PREFIX}RESEARCH_QUALITY_UNAVAILABLE")
    if (
        has_unresolved_hard_blocker
        or (
            report.research_quality is not None
            and report.research_quality.incomplete is True
        )
        or any(
            token.startswith(_CURRENT_CONTEXT_WARNING_PREFIXES)
            for token in warning_tokens
        )
    ):
        reasons.append(f"{_BUY_POSTURE_PREFIX}CURRENT_CONTEXT_INCOMPLETE")

    if reasons:
        report.verdict = "WATCH"
        buy_prefix = f"{report.ticker} screens as BUY."
        if report.thesis_summary.startswith(buy_prefix):
            report.thesis_summary = report.thesis_summary.replace(
                buy_prefix,
                f"{report.ticker} screens as WATCH.",
                1,
            )
        report.warnings = list(dict.fromkeys([*report.warnings, *reasons]))
    return report


def _report_identity(report: Any, bundle: AnalysisEvidenceBundle) -> str:
    completed_at = str(getattr(report, "completed_at", "") or "")
    token = completed_at.replace(":", "").replace("-", "").replace("+", "_")
    token = token.replace(".", "").replace("T", "T").replace("Z", "Z")
    if not token:
        token = bundle.built_at.replace(":", "").replace("-", "").replace("+", "_").replace(".", "")
    return f"{bundle.ticker}_{bundle.as_of_date}_{token}"


def _strip_handles(handles: list[str]) -> list[str]:
    return [handle.strip().strip("[]") for handle in handles if handle]


def _display_values(view: Any) -> list[float]:
    values: list[float] = []
    for method in getattr(view, "methods", []) or []:
        value = getattr(method, "adjusted", None)
        if value is None:
            value = getattr(method, "original", None)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _expectations_gap(view: Any) -> ExpectationsGap | None:
    tension = getattr(view, "tension_type", None)
    explanation = getattr(view, "tension_explanation", None)
    if not tension or tension == "NONE":
        return None
    return ExpectationsGap(
        market_implied_view=f"Current price reflects a {tension.lower()} framing.",
        analyst_view=explanation or "Adjusted thesis differs from the market-implied setup.",
        key_mismatch=explanation or tension,
    )


def _thesis_summary(bundle: AnalysisEvidenceBundle, report: Any, view: Any) -> str:
    verdict = _VERDICT_MAP.get(derive_verdict(report), "WATCH")
    mos = getattr(view, "adjusted_margin_of_safety", None)
    positives = getattr(view, "adjustments", []) or []
    risks = getattr(view, "hard_blockers", []) or getattr(view, "open_questions", []) or []

    parts = [f"{bundle.ticker} screens as {verdict}."]
    if isinstance(mos, (int, float)):
        parts.append(f"Adjusted margin of safety is {mos:.1%}.")
    if positives:
        parts.append(f"Primary support: {positives[0].claim}.")
    if risks:
        risk_obj = risks[0]
        risk_text = getattr(risk_obj, "description", None) or getattr(risk_obj, "claim", None)
        if risk_text:
            parts.append(f"Main uncertainty: {risk_text}.")
    return " ".join(parts)


def _top_positive_findings(view: Any) -> list[AnalysisFinding]:
    findings: list[AnalysisFinding] = []
    for idx, adjustment in enumerate(getattr(view, "adjustments", []) or [], start=1):
        direction = str(getattr(adjustment, "direction", "") or "").upper()
        if direction not in {"BULLISH", "POSITIVE"}:
            continue
        findings.append(
            AnalysisFinding(
                finding_id=f"P{idx}",
                category="POSITIVE",
                claim=adjustment.claim,
                direction=direction,
                severity="HIGH" if abs(getattr(adjustment, "magnitude", 0.0)) >= 5 else "MODERATE",
                source_basis="filings",
                citation_ids=_strip_handles(getattr(adjustment, "citation_handles", [])),
            )
        )
        if len(findings) >= 3:
            break
    return findings


def _risk_findings(view: Any) -> list[AnalysisFinding]:
    findings: list[AnalysisFinding] = []
    counter = 1
    for adjustment in getattr(view, "adjustments", []) or []:
        direction = str(getattr(adjustment, "direction", "") or "").upper()
        if direction not in {"BEARISH", "NEGATIVE"}:
            continue
        findings.append(
            AnalysisFinding(
                finding_id=f"R{counter}",
                category="RISK",
                claim=adjustment.claim,
                direction=direction,
                severity="HIGH" if abs(getattr(adjustment, "magnitude", 0.0)) >= 5 else "MODERATE",
                source_basis="filings",
                citation_ids=_strip_handles(getattr(adjustment, "citation_handles", [])),
            )
        )
        counter += 1
        if len(findings) >= 3:
            break

    for blocker in getattr(view, "hard_blockers", []) or []:
        findings.append(
            AnalysisFinding(
                finding_id=f"R{counter}",
                category="RISK",
                claim=getattr(blocker, "description", "Thesis-critical blocker"),
                direction="BEARISH",
                severity="HIGH",
                source_basis="filings",
                citation_ids=[],
            )
        )
        counter += 1
        if len(findings) >= 5:
            break
    return findings


def _recent_event_findings(bundle: AnalysisEvidenceBundle) -> list[AnalysisFinding]:
    findings: list[AnalysisFinding] = []
    for idx, event in enumerate(bundle.recent_events[:5], start=1):
        claim = event.title or event.summary
        findings.append(
            AnalysisFinding(
                finding_id=f"E{idx}",
                category="RECENT_EVENT",
                claim=claim,
                direction=None,
                severity=event.materiality,
                source_basis="recent_events",
                citation_ids=[f"E{idx}"],
            )
        )
    return findings


def _open_questions(view: Any) -> list[OpenQuestion]:
    questions: list[OpenQuestion] = []
    for item in getattr(view, "hard_blockers", []) or []:
        questions.append(
            OpenQuestion(
                question=getattr(item, "description", "Unresolved blocker"),
                importance="HIGH",
                next_step=getattr(item, "blocking_reason", None),
            )
        )
    for item in getattr(view, "open_questions", []) or []:
        questions.append(
            OpenQuestion(
                question=getattr(item, "description", "Open question"),
                importance=getattr(item, "importance", "MEDIUM"),
                next_step=None,
            )
        )
    return questions[:6]


def _falsifiers(view: Any) -> list[Falsifier]:
    falsifiers: list[Falsifier] = []
    for entry in getattr(view, "why_wrong", []) or []:
        falsifiers.append(
            Falsifier(
                description=getattr(entry, "invalidation", getattr(entry, "assumption", "Thesis invalidated")),
                trigger_type="THESIS_BREAK",
                monitoring_hint=getattr(entry, "impact", None),
            )
        )
    return falsifiers[:4]


def _citations(report: Any, view: Any, bundle: AnalysisEvidenceBundle) -> list[AnalysisCitation]:
    citations: list[AnalysisCitation] = []
    filing_citations = getattr(report, "citations", None)
    if not filing_citations:
        filing_citations = getattr(view, "citations", []) or []
    for cite in filing_citations:
        cite_source_type = getattr(cite, "source_type", None)
        if cite_source_type in {"ir_press", "company_news", "external_news", "TRANSCRIPT"}:
            citations.append(
                AnalysisCitation(
                    citation_id=str(getattr(cite, "citation_id", "")),
                    source_type=cite_source_type,
                    source_label=(
                        getattr(cite, "source_title", None)
                        or cite_source_type
                    ),
                    source_date=getattr(cite, "source_published_at", None),
                    section=getattr(cite, "section", None),
                    excerpt=str(getattr(cite, "excerpt", "")),
                    source_url=getattr(cite, "source_url", None),
                    source_quality=getattr(cite, "source_quality", None),
                )
            )
            continue
        source_form_type = getattr(cite, "source_form_type", None)
        item_code = getattr(cite, "item_code", None)
        source_label = source_form_type or getattr(view, "filing_form", None) or "Filing"
        if source_form_type == "8-K" and item_code:
            source_label = f"{source_form_type} Item {item_code}"
        citations.append(
            AnalysisCitation(
                citation_id=str(getattr(cite, "citation_id", "")),
                source_type="filing",
                source_label=source_label,
                source_date=(
                    getattr(cite, "source_filing_date", None)
                    or getattr(view, "filing_date", None)
                ),
                section=getattr(cite, "section", None),
                excerpt=str(getattr(cite, "excerpt", "")),
                source_form_type=getattr(cite, "source_form_type", None),
                source_accession=getattr(cite, "source_accession", None),
                source_role=getattr(cite, "source_role", None),
                item_code=item_code,
                event_category=getattr(cite, "event_category", None),
                source_url=None,
                source_quality=getattr(cite, "source_quality", None),
            )
        )

    for idx, event in enumerate(bundle.recent_events[:5], start=1):
        source_label = event.title or event.source_type
        source_role = "material_event" if str(event.source_type or "").upper().startswith("8-K") else None
        source_form_type = event.source_type if source_role == "material_event" else None
        if source_form_type == "8-K" and event.item_code:
            source_label = f"8-K Item {event.item_code}"
        citations.append(
            AnalysisCitation(
                citation_id=f"E{idx}",
                source_type=event.source_type,
                source_label=source_label,
                source_date=event.published_at,
                section=event.event_category,
                excerpt=event.summary,
                source_url=event.source_url,
                source_form_type=source_form_type,
                source_accession=event.accession,
                source_role=source_role,
                item_code=event.item_code,
                event_category=event.event_category,
                source_quality=event.source_quality,
            )
        )
    return citations


def _evidence_sort_key(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if "T" in value:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        return datetime.fromisoformat(f"{value}T00:00:00+00:00").astimezone(timezone.utc)
    except ValueError:
        return None


def _latest_evidence_metadata(
    bundle: AnalysisEvidenceBundle,
    report: Any,
) -> tuple[str | None, str | None]:
    candidates: list[tuple[datetime, str, str]] = []

    def add_candidate(source_date: str | None, source_type: str | None) -> None:
        sort_key = _evidence_sort_key(source_date)
        if sort_key is not None and source_type:
            candidates.append((sort_key, source_date or "", source_type))

    add_candidate(
        getattr(report, "latest_evidence_date", None),
        getattr(report, "latest_evidence_source_type", None),
    )
    for filing in bundle.filings:
        add_candidate(filing.filing_date, filing.form_type)
    for event in bundle.recent_events:
        add_candidate(event.published_at, event.source_type)

    if not candidates:
        return None, None
    _, source_date, source_type = max(candidates, key=lambda item: item[0])
    return source_date, source_type


def build_analysis_report_from_research(
    bundle: AnalysisEvidenceBundle,
    report: Any,
) -> AnalysisReport:
    """Adapt the current deep-research runtime into AnalysisReport."""
    view = build_view(report)
    verdict_code = derive_verdict(report)
    verdict = _VERDICT_MAP.get(verdict_code, "WATCH")

    values = _display_values(view)
    price = getattr(view, "current_price", None)
    thesis = getattr(report, "thesis", None)
    adjusted_mid = getattr(thesis, "adjusted_intrinsic_mid", None) if thesis is not None else None

    valuation = ValuationConclusion(
        price=price,
        base_case_value=adjusted_mid if isinstance(adjusted_mid, (int, float)) else (sum(values) / len(values) if values else None),
        bear_case_value=min(values) if values else None,
        bull_case_value=max(values) if values else None,
        margin_of_safety=getattr(view, "adjusted_margin_of_safety", None),
        expectations_gap=_expectations_gap(view),
    )

    warnings = list(dict.fromkeys((bundle.warnings or []) + (getattr(report, "warnings", []) or [])))
    sources_used = sorted({
        "valuation_snapshot",
        *[f"filing:{filing.form_type}" for filing in bundle.filings],
        *[f"event:{event.source_type}" for event in bundle.recent_events],
    })
    latest_evidence_date, latest_evidence_source_type = _latest_evidence_metadata(bundle, report)

    analysis_report = AnalysisReport(
        analysis_id=_report_identity(report, bundle),
        ticker=bundle.ticker,
        as_of_date=bundle.as_of_date,
        generated_at=str(getattr(report, "completed_at", None) or bundle.built_at),
        verdict=verdict,
        research_gate=verdict_code,
        confidence_label=str(getattr(report, "conviction_class", None) or "LOW"),
        confidence_score=getattr(report, "conviction_score", None),
        thesis_summary=_thesis_summary(bundle, report, view),
        valuation=valuation,
        positives=_top_positive_findings(view),
        risks=_risk_findings(view),
        recent_event_impacts=_recent_event_findings(bundle),
        open_questions=_open_questions(view),
        falsifiers=_falsifiers(view),
        citations=_citations(report, view, bundle),
        prior_thesis_change_summary=None,
        latest_evidence_date=latest_evidence_date,
        latest_evidence_source_type=latest_evidence_source_type,
        sources_used=sources_used,
        warnings=warnings,
    )
    return apply_investment_posture(
        analysis_report,
        has_unresolved_hard_blocker=bool(getattr(view, "hard_blockers", []) or []),
    )
