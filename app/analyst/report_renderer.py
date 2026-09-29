"""Presentation helpers for the analyst-facing report contract."""
from __future__ import annotations

from app.analyst.thesis_contract import AnalysisFinding, AnalysisReport
from app.decision.decision_block import (
    DecisionBlock,
    render_decision_block_markdown,
)
from app.research.source_quality import source_quality_label

# Verdict-to-action mapping for the analyst memo. The
# analyst posture (BUY / WATCH / STAY_AWAY) drives the action verb here; the
# price-trigger STATUS reconciliation lives in the watchlist-aware surfaces.
_VERDICT_ACTION = {
    "BUY": "Buy now",
    "WATCH": "Wait",
    "STAY_AWAY": "Pass",
}


def _action_for_verdict(verdict: str | None) -> str:
    return _VERDICT_ACTION.get(verdict or "", "Wait")


def _decision_block_for_report(report: AnalysisReport) -> DecisionBlock:
    return DecisionBlock(
        action=_action_for_verdict(report.verdict),
        current_price=report.valuation.price,
        buy_price_target=None,
        pct_to_target=None,
        base_case_expected_return=None,
        conviction_grade=None,
        confidence=report.confidence_label,
        price_trigger_status=None,
        time_horizon="3-5 yr",
        what_would_change_my_mind=[f.description for f in report.falsifiers[:3]],
        verdict_reconciliation_note=None,
    )


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:.2f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def _finding_suffix(finding: AnalysisFinding) -> str:
    suffix_parts: list[str] = []
    if finding.direction:
        suffix_parts.append(finding.direction)
    if finding.severity:
        suffix_parts.append(finding.severity)
    if finding.citation_ids:
        suffix_parts.append(" ".join(f"[{cid}]" for cid in finding.citation_ids))
    if not suffix_parts:
        return ""
    return f" ({'; '.join(suffix_parts)})"


def _render_findings(title: str, findings: list[AnalysisFinding]) -> str:
    lines = [f"## {title}"]
    for finding in findings:
        lines.append(f"- {finding.claim}{_finding_suffix(finding)}")
    return "\n".join(lines)


def render_summary(report: AnalysisReport) -> str:
    """Render a compact terminal summary for the analyst contract."""
    valuation = report.valuation
    lines = [f"{report.ticker} Analyst Report ({report.as_of_date})"]
    lines.append(f"Action: {_action_for_verdict(report.verdict)}")
    confidence = report.confidence_label
    if report.confidence_score is not None:
        confidence = f"{confidence} ({report.confidence_score}/100)"
    lines.append(f"Verdict: {report.verdict} | Confidence: {confidence}")
    if report.research_gate:
        lines.append(f"Research gate: {report.research_gate} (pipeline provenance)")
    lines.append(f"Thesis: {report.thesis_summary}")
    lines.append(
        "Valuation: "
        f"Price {_fmt_money(valuation.price)} | "
        f"Base {_fmt_money(valuation.base_case_value)} | "
        f"MOS {_fmt_pct(valuation.margin_of_safety)}"
    )
    if report.research_quality is not None:
        quality = report.research_quality
        overall = "N/A" if quality.overall_score is None else f"{quality.overall_score:.1f}"
        coverage = "N/A" if quality.coverage_score is None else f"{quality.coverage_score:.1f}"
        freshness = "N/A" if quality.freshness_score is None else f"{quality.freshness_score:.1f}"
        gap = "N/A" if quality.gap_score is None else f"{quality.gap_score:.1f}"
        lines.append(
            "Research Quality: "
            f"overall={overall} (coverage={coverage}, freshness={freshness}, gaps={gap})"
        )
    if report.positives:
        lines.append(f"Top positive: {report.positives[0].claim}")
    if report.risks:
        lines.append(f"Top risk: {report.risks[0].claim}")
    if report.recent_event_impacts:
        lines.append(f"Recent event: {report.recent_event_impacts[0].claim}")
    if report.latest_evidence_source_type or report.latest_evidence_date:
        lines.append(
            "Latest evidence: "
            f"{report.latest_evidence_source_type or 'N/A'}"
            f" ({report.latest_evidence_date or 'N/A'})"
        )
    if report.warnings:
        lines.append(f"Warnings: {', '.join(report.warnings[:3])}")
    return "\n".join(lines)


def render_report(report: AnalysisReport) -> str:
    """Render a markdown report from an AnalysisReport."""
    valuation = report.valuation
    header_lines = [
        f"# {report.ticker} - Analyst Report",
        "",
        f"**As of:** {report.as_of_date}",
        f"**Generated:** {report.generated_at}",
        f"**Verdict:** {report.verdict}",
        f"**Research Gate:** {report.research_gate or 'N/A'} (pipeline provenance)",
        f"**Confidence:** {report.confidence_label}"
        + (
            f" ({report.confidence_score}/100)"
            if report.confidence_score is not None
            else ""
        ),
    ]
    if report.latest_evidence_source_type or report.latest_evidence_date:
        header_lines.append(
            f"**Latest Evidence:** {report.latest_evidence_source_type or 'N/A'}"
            f" ({report.latest_evidence_date or 'N/A'})"
        )
    sections = [
        "\n".join(header_lines),
        render_decision_block_markdown(_decision_block_for_report(report)),
        "\n".join(
            [
                "## Thesis",
                report.thesis_summary,
            ]
        ),
        "\n".join(
            [
                "## Valuation",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Price | {_fmt_money(valuation.price)} |",
                f"| Base Case | {_fmt_money(valuation.base_case_value)} |",
                f"| Bear Case | {_fmt_money(valuation.bear_case_value)} |",
                f"| Bull Case | {_fmt_money(valuation.bull_case_value)} |",
                f"| Margin of Safety | {_fmt_pct(valuation.margin_of_safety)} |",
            ]
        ),
    ]

    if report.research_quality is not None:
        quality = report.research_quality
        sections.append(
            "\n".join(
                [
                    "## Research Quality",
                    f"- Overall: {'N/A' if quality.overall_score is None else f'{quality.overall_score:.1f}'}",
                    f"- Coverage: {'N/A' if quality.coverage_score is None else f'{quality.coverage_score:.1f}'}",
                    f"- Freshness: {'N/A' if quality.freshness_score is None else f'{quality.freshness_score:.1f}'}",
                    f"- Gaps: {'N/A' if quality.gap_score is None else f'{quality.gap_score:.1f}'}",
                    f"- Evidence Count: {quality.evidence_count if quality.evidence_count is not None else 'N/A'}",
                ]
            )
        )

    if valuation.expectations_gap is not None:
        sections.append(
            "\n".join(
                [
                    "## Expectations Gap",
                    f"- Market-implied view: {valuation.expectations_gap.market_implied_view}",
                    f"- Analyst view: {valuation.expectations_gap.analyst_view}",
                    f"- Key mismatch: {valuation.expectations_gap.key_mismatch}",
                ]
            )
        )

    if report.positives:
        sections.append(_render_findings("Positives", report.positives))
    if report.risks:
        sections.append(_render_findings("Risks", report.risks))
    if report.recent_event_impacts:
        sections.append(_render_findings("Recent Events", report.recent_event_impacts))

    if report.open_questions:
        sections.append(
            "\n".join(
                ["## Open Questions"]
                + [
                    f"- {item.question} ({item.importance})"
                    + (f" - Next step: {item.next_step}" if item.next_step else "")
                    for item in report.open_questions
                ]
            )
        )

    if report.falsifiers:
        sections.append(
            "\n".join(
                ["## Falsifiers"]
                + [
                    f"- {item.description} ({item.trigger_type})"
                    + (
                        f" - Monitor: {item.monitoring_hint}"
                        if item.monitoring_hint
                        else ""
                    )
                    for item in report.falsifiers
                ]
            )
        )

    if report.citations:
        citation_lines = ["## Citations"]
        for citation in report.citations:
            header = (
                f"- [{citation.citation_id}] {citation.source_label}"
                + (f" ({citation.source_date})" if citation.source_date else "")
            )
            if citation.section:
                header += f" - {citation.section}"
            citation_lines.append(header)
            citation_lines.append(f"  {citation.excerpt}")
            if citation.source_url:
                citation_lines.append(f"  Source: {citation.source_url}")
            quality_label = source_quality_label(citation.source_quality)
            if quality_label:
                citation_lines.append(f"  Source Quality: {quality_label}")
        sections.append("\n".join(citation_lines))

    if report.sources_used:
        sections.append(
            "\n".join(
                [
                    "## Sources Used",
                    ", ".join(report.sources_used),
                ]
            )
        )

    if report.warnings:
        sections.append(
            "\n".join(
                [
                    "## Warnings",
                    *[f"- {warning}" for warning in report.warnings],
                ]
            )
        )

    return "\n\n---\n\n".join(sections) + "\n"
