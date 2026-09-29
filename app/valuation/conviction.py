"""Deterministic conviction scoring from a completed research report.

Pure function. No DB. No file I/O. No LLM calls.
Takes a ResearchReport, returns a ConvictionResult.

Public API: compute_conviction(report) -> ConvictionResult
"""
from __future__ import annotations

from dataclasses import dataclass

from app.research.deep_research import ResearchReport


# Highest composite a BLOCKED gate allows: the top of the LOW band.
_BLOCKED_GATE_SCORE_CAP = 49


@dataclass
class ConvictionResult:
    conviction_score: int          # 0-100 composite score
    conviction_class: str          # HIGH / MODERATE / LOW / INSUFFICIENT

    # Component scores (each 0-25)
    method_agreement_score: int
    evidence_coverage_score: int
    gate_quality_score: int
    investigation_resolution_score: int

    # Metadata for auditability
    detail: str                    # human-readable breakdown


def compute_conviction(report: ResearchReport) -> ConvictionResult:
    """Compute deterministic conviction score from a completed research report.

    Pure function. No DB, no file I/O, no LLM.
    All inputs come from ResearchReport fields.
    """
    # Short-circuit: NO_SCORECARD
    if report.status == "NO_SCORECARD":
        return ConvictionResult(
            conviction_score=0,
            conviction_class="INSUFFICIENT",
            method_agreement_score=0,
            evidence_coverage_score=0,
            gate_quality_score=0,
            investigation_resolution_score=0,
            detail="NO_SCORECARD — no valuation data available",
        )

    # Component 1: Method Agreement (0-25)
    if report.method_count is None or report.method_count <= 1:
        method_agreement_score = 5
    elif report.methods_agree:
        method_agreement_score = 25
    else:
        if report.consensus_strength is None:
            method_agreement_score = 5
        else:
            agreement_ratio = report.consensus_strength / report.method_count
            method_agreement_score = round(5 + 20 * agreement_ratio)

    # Component 2: Evidence Coverage (0-25)
    if not report.investigation_ran or report.thesis is None:
        evidence_coverage_score = 0
    else:
        evidence_coverage_score = round(25 * report.thesis.average_coverage)

    # Component 3: Gate Quality (0-25). A BLOCKED gate earns nothing: it used
    # to score 5, more than an unknown gate's 0, so a name the quality gate
    # refused to value could still reach HIGH conviction.
    if report.gate_action == "PROCEED":
        gate_quality_score = 25
    elif report.gate_action == "ADJUST":
        gate_quality_score = 15
    else:
        gate_quality_score = 0

    # Component 4: Investigation Resolution (0-25)
    if not report.investigation_ran or report.thesis is None:
        investigation_resolution_score = 0
    else:
        total_hypotheses = (
            report.thesis.hypotheses_confirmed
            + report.thesis.hypotheses_contradicted
            + report.thesis.hypotheses_partially_confirmed
            + report.thesis.hypotheses_inconclusive
            + report.thesis.hypotheses_unclassified
        )
        resolved = (
            report.thesis.hypotheses_confirmed
            + report.thesis.hypotheses_contradicted
            + report.thesis.hypotheses_partially_confirmed
        )
        if total_hypotheses == 0:
            investigation_resolution_score = 0
        else:
            resolution_ratio = resolved / total_hypotheses
            base_score = round(20 * resolution_ratio)
            if report.tension_type == "NONE":
                base_score = min(base_score + 5, 25)
            investigation_resolution_score = base_score

    # Composite
    conviction_score = (
        method_agreement_score
        + evidence_coverage_score
        + gate_quality_score
        + investigation_resolution_score
    )

    # Cap at INSUFFICIENT when investigation didn't run.
    # Method agreement and gate quality alone are not sufficient basis
    # for conviction — they become harmful when verdict uses the score.
    if not report.investigation_ran:
        conviction_score = min(conviction_score, 25)

    # Cap at LOW when the quality gate BLOCKED valuation: the gate refused to
    # value the company (going concern, insolvency, no owner earnings, ...),
    # so agreement, coverage and resolved hypotheses about its value cannot
    # add up to more than low conviction.
    gate_blocked = report.gate_action == "BLOCK"
    if gate_blocked:
        conviction_score = min(conviction_score, _BLOCKED_GATE_SCORE_CAP)

    # Class
    if conviction_score >= 75:
        conviction_class = "HIGH"
    elif conviction_score >= 50:
        conviction_class = "MODERATE"
    elif conviction_score >= 25:
        conviction_class = "LOW"
    else:
        conviction_class = "INSUFFICIENT"

    detail = (
        f"method_agreement={method_agreement_score}/25, "
        f"evidence_coverage={evidence_coverage_score}/25, "
        f"gate_quality={gate_quality_score}/25, "
        f"investigation_resolution={investigation_resolution_score}/25"
    )
    if not report.investigation_ran:
        detail += " (capped: uninvestigated)"
    if gate_blocked:
        detail += " (capped: gate blocked)"

    return ConvictionResult(
        conviction_score=conviction_score,
        conviction_class=conviction_class,
        method_agreement_score=method_agreement_score,
        evidence_coverage_score=evidence_coverage_score,
        gate_quality_score=gate_quality_score,
        investigation_resolution_score=investigation_resolution_score,
        detail=detail,
    )
