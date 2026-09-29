"""Markdown renderer for single-candidate autonomous analyst runs."""
from __future__ import annotations

import json
from typing import Any

from app.autonomous.run_contract import AutonomousRunArtifact, EvidenceReference


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _fmt_pct(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "N/A"
    return f"{value:.1%}"


def _pipe(value: Any) -> str:
    return _fmt(value).replace("|", "\\|").replace("\n", " ")


def _truncate(value: Any, limit: int = 220) -> str:
    text = _fmt(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _analyst_context(artifact: AutonomousRunArtifact) -> dict[str, Any]:
    for item in artifact.evidence:
        if item.source_type == "analysis_report" and item.source_label == "analyst_context" and item.excerpt:
            try:
                payload = json.loads(item.excerpt)
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}
    return {}


def _ticker(artifact: AutonomousRunArtifact) -> str:
    tickers = artifact.request.candidate_scope.get("tickers")
    if isinstance(tickers, list) and tickers:
        return str(tickers[0]).upper()
    return str(artifact.selected_ticker or "UNKNOWN").upper()


def _render_header(artifact: AutonomousRunArtifact) -> str:
    lines = [
        f"# {_ticker(artifact)} - Autonomous Research Report",
        "",
        f"**As of:** {artifact.request.as_of_date}",
        f"**Generated:** {artifact.completed_at or artifact.started_at}",
        f"**Status:** {artifact.status}",
        f"**Final Verdict:** {artifact.final_verdict}",
        f"**Selected Ticker:** {_fmt(artifact.selected_ticker)}",
        f"**Confidence:** {_fmt(artifact.confidence)}",
    ]
    if artifact.no_winner_reason:
        lines.append(f"**No Winner Reason:** {artifact.no_winner_reason}")
    return "\n".join(lines)


def _render_company_identity(context: dict[str, Any], artifact: AutonomousRunArtifact) -> str:
    identity = context.get("company_identity") if isinstance(context.get("company_identity"), dict) else {}
    lines = [
        "## Company Identity",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Ticker | {_pipe(identity.get('ticker') or _ticker(artifact))} |",
        f"| Company | {_pipe(identity.get('company_name'))} |",
        f"| CIK | {_pipe(identity.get('cik'))} |",
        f"| Homepage | {_pipe(identity.get('homepage_url'))} |",
        f"| Sector | {_pipe(identity.get('sector'))} |",
        f"| Issuer Type | {_pipe(identity.get('issuer_type'))} |",
        f"| Security Type | {_pipe(identity.get('security_type'))} |",
        f"| Model Status | {_pipe(identity.get('model_status'))} |",
    ]
    return "\n".join(lines)


def _render_valuation(context: dict[str, Any]) -> str:
    valuation = context.get("valuation") if isinstance(context.get("valuation"), dict) else {}
    quality = context.get("research_quality") if isinstance(context.get("research_quality"), dict) else {}
    lines = [
        "## Valuation Context",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Price | {_pipe(valuation.get('price'))} |",
        f"| Base Case Value | {_pipe(valuation.get('base_case_value'))} |",
        f"| Bear Case Value | {_pipe(valuation.get('bear_case_value'))} |",
        f"| Bull Case Value | {_pipe(valuation.get('bull_case_value'))} |",
        f"| Margin of Safety | {_pipe(_fmt_pct(valuation.get('margin_of_safety')))} |",
        f"| Analyst Verdict | {_pipe(context.get('verdict'))} |",
        f"| Analyst Confidence | {_pipe(context.get('confidence_label'))} |",
        f"| Latest Evidence | {_pipe(context.get('latest_evidence_date'))} |",
        f"| Research Quality | {_pipe(quality.get('overall_score'))} |",
    ]
    return "\n".join(lines)


def _render_analyst_context(context: dict[str, Any]) -> str:
    paths = context.get("paths") if isinstance(context.get("paths"), dict) else {}
    lines = [
        "## Seeded Analyst Context",
        "",
        _fmt(context.get("thesis_summary")),
        "",
        "| Artifact | Path |",
        "|---|---|",
        f"| Evidence Bundle | {_pipe(paths.get('analysis_evidence_bundle'))} |",
        f"| Analyst JSON | {_pipe(paths.get('analysis_report_json'))} |",
        f"| Analyst Markdown | {_pipe(paths.get('analysis_report_markdown'))} |",
    ]
    return "\n".join(lines)


def _render_questions(artifact: AutonomousRunArtifact) -> str:
    lines = [
        "## Research Questions",
        "",
        "| ID | Status | Priority | Question | Selected Tools |",
        "|---|---|---|---|---|",
    ]
    if not artifact.questions:
        lines.append("| N/A | N/A | N/A | No research questions recorded. | N/A |")
    for question in artifact.questions:
        lines.append(
            f"| {_pipe(question.question_id)} | {_pipe(question.status)} | {_pipe(question.priority)} | "
            f"{_pipe(question.question)} | {_pipe(', '.join(question.selected_tools))} |"
        )
    return "\n".join(lines)


def _render_tool_calls(artifact: AutonomousRunArtifact) -> str:
    lines = [
        "## Executed Tools",
        "",
        "| Call | Status | Tool | Question | Evidence | Rationale/Error |",
        "|---|---|---|---|---|---|",
    ]
    if not artifact.tool_calls:
        lines.append("| N/A | N/A | No tool calls recorded. | N/A | N/A | N/A |")
    for call in artifact.tool_calls:
        rationale = call.error or call.rationale
        lines.append(
            f"| {_pipe(call.call_id)} | {_pipe(call.status)} | {_pipe(call.tool_name)} | "
            f"{_pipe(call.question_id)} | {_pipe(', '.join(call.evidence_ref_ids))} | "
            f"{_pipe(_truncate(rationale))} |"
        )
    return "\n".join(lines)


def _render_evidence_item(item: EvidenceReference) -> str:
    return (
        f"| {_pipe(item.evidence_id)} | {_pipe(item.confidence)} | {_pipe(item.source_type)} | "
        f"{_pipe(item.source_label)} | {_pipe(item.source_date)} | {_pipe(_truncate(item.summary))} |"
    )


def _render_evidence(artifact: AutonomousRunArtifact) -> str:
    lines = [
        "## Evidence",
        "",
        "| ID | Confidence | Type | Source | Date | Summary |",
        "|---|---|---|---|---|---|",
    ]
    if not artifact.evidence:
        lines.append("| N/A | N/A | N/A | N/A | N/A | No evidence recorded. |")
    for item in artifact.evidence:
        lines.append(_render_evidence_item(item))
    return "\n".join(lines)


def _render_belief_updates(artifact: AutonomousRunArtifact) -> str:
    lines = [
        "## Belief Updates",
        "",
        "| ID | Direction | Confidence | Summary | Evidence | Remaining Uncertainty |",
        "|---|---|---|---|---|---|",
    ]
    if not artifact.belief_updates:
        lines.append("| N/A | N/A | N/A | No belief updates recorded. | N/A | N/A |")
    for update in artifact.belief_updates:
        lines.append(
            f"| {_pipe(update.update_id)} | {_pipe(update.direction)} | {_pipe(update.confidence_after)} | "
            f"{_pipe(_truncate(update.summary))} | {_pipe(', '.join(update.evidence_ref_ids))} | "
            f"{_pipe('; '.join(update.remaining_uncertainty))} |"
        )
    return "\n".join(lines)


def _render_candidate_decision(artifact: AutonomousRunArtifact) -> str:
    lines = ["## Candidate Decision"]
    if not artifact.candidate_decisions:
        lines.append("No candidate decision was recorded.")
        return "\n\n".join(lines)
    decision = artifact.candidate_decisions[0]
    lines.extend(
        [
            "",
            f"**Ticker:** {decision.ticker}",
            f"**Verdict:** {decision.verdict}",
            f"**Confidence:** {decision.confidence}",
            f"**Eligible For Selection:** {decision.eligible_for_selection}",
            "",
            f"**Thesis:** {decision.thesis or 'N/A'}",
            "",
            f"**Key Risk:** {decision.key_risk or 'N/A'}",
            "",
            "**Selection Blockers:** "
            + (", ".join(decision.selection_blockers) if decision.selection_blockers else "None"),
            "",
            "**Falsifiers:** "
            + (", ".join(decision.falsifiers) if decision.falsifiers else "None"),
            "",
            "**Confidence Caps:** "
            + (", ".join(decision.confidence_cap_reasons) if decision.confidence_cap_reasons else "None"),
        ]
    )
    return "\n".join(lines)


def _render_warnings(artifact: AutonomousRunArtifact, context: dict[str, Any]) -> str:
    context_warnings = context.get("warnings") if isinstance(context.get("warnings"), list) else []
    warnings = list(dict.fromkeys([*artifact.degraded_states, *context_warnings]))
    lines = ["## Warnings And Audit Notes"]
    if warnings:
        lines.append("")
        lines.extend(f"- {item}" for item in warnings)
    if artifact.audit_notes:
        lines.append("")
        lines.append("### Audit Notes")
        lines.extend(f"- {item}" for item in artifact.audit_notes)
    if len(lines) == 1:
        lines.append("No warnings or audit notes recorded.")
    return "\n".join(lines)


def render_autonomous_run_report(artifact: AutonomousRunArtifact) -> str:
    """Render a human-readable Markdown report from an autonomous run artifact."""

    context = _analyst_context(artifact)
    sections = [
        _render_header(artifact),
        _render_company_identity(context, artifact),
        _render_valuation(context),
        _render_analyst_context(context),
        _render_questions(artifact),
        _render_tool_calls(artifact),
        _render_evidence(artifact),
        _render_belief_updates(artifact),
        _render_candidate_decision(artifact),
        _render_warnings(artifact, context),
    ]
    return "\n\n".join(sections) + "\n"


__all__ = ["render_autonomous_run_report"]
