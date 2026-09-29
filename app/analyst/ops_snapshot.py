"""Shared analyst-contract loader for ops-facing consumers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.analyst.output_store import latest_analysis_output_path, latest_analysis_report
from app.analyst.thesis_contract import (
    AnalysisCitation,
    AnalysisFinding,
    AnalysisReport,
    Falsifier,
    OpenQuestion,
    ResearchGapSummary,
    ResearchQualitySummary,
)
from app.db import get_db


@dataclass
class OpsAnalysisSnapshot:
    analysis_report: AnalysisReport | None
    analysis_report_path: Path | None
    analysis_report_markdown_path: Path | None
    analysis_source: str
    research_quality: ResearchQualitySummary | None
    positives: list[AnalysisFinding] = field(default_factory=list)
    risks: list[AnalysisFinding] = field(default_factory=list)
    recent_event_impacts: list[AnalysisFinding] = field(default_factory=list)
    open_questions: list[OpenQuestion] = field(default_factory=list)
    falsifiers: list[Falsifier] = field(default_factory=list)
    citations: list[AnalysisCitation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verdict: str = ""
    confidence_label: str = ""
    confidence_score: int | None = None
    thesis_summary: str = ""
    legacy_research_path: Path | None = None
    legacy_research_payload: dict[str, Any] | None = None


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _coerce_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _dedupe_strs(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _latest_legacy_research_packet_row(
    ticker: str,
    *,
    as_of_date: str,
    run_id: str | None,
) -> tuple[Path | None, str | None]:
    ticker = ticker.upper()
    with get_db() as conn:
        if run_id:
            row = conn.execute(
                """
                SELECT packet_path, run_id
                FROM research_packets
                WHERE ticker = ? AND run_id = ? AND as_of_date <= ?
                ORDER BY as_of_date DESC, created_at DESC
                LIMIT 1
                """,
                (ticker, run_id, as_of_date),
            ).fetchone()
            if row:
                path = Path(row["packet_path"])
                if path.exists():
                    return path, row["run_id"]
        row = conn.execute(
            """
            SELECT packet_path, run_id
            FROM research_packets
            WHERE ticker = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            (ticker, as_of_date),
        ).fetchone()
    if not row:
        return None, None
    path = Path(row["packet_path"])
    if not path.exists():
        return None, None
    return path, row["run_id"]


def _quality_from_legacy_payload(payload: dict[str, Any]) -> ResearchQualitySummary | None:
    quality = payload.get("quality")
    if not isinstance(quality, dict):
        return None
    top_gaps = [
        ResearchGapSummary(
            severity=str(item.get("severity") or "UNKNOWN"),
            summary=str(item.get("summary") or ""),
            recommended_action=(
                str(item.get("recommended_action"))
                if item.get("recommended_action") is not None
                else None
            ),
        )
        for item in quality.get("top_gaps", [])
        if isinstance(item, dict) and str(item.get("summary") or "").strip()
    ]
    return ResearchQualitySummary(
        coverage_score=_coerce_float(quality.get("coverage_score")),
        freshness_score=_coerce_float(quality.get("freshness_score")),
        gap_score=_coerce_float(quality.get("gap_score")),
        overall_score=_coerce_float(quality.get("overall_research_score")),
        incomplete=bool(quality.get("incomplete")) if quality.get("incomplete") is not None else None,
        evidence_count=len(payload.get("evidence_items") or []),
        top_gaps=top_gaps,
    )


def _entry_text(item: dict[str, Any], *, preferred_keys: tuple[str, ...]) -> str:
    for key in preferred_keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _legacy_findings(
    items: list[Any],
    *,
    prefix: str,
    category: str,
    default_direction: str | None,
) -> list[AnalysisFinding]:
    findings: list[AnalysisFinding] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        claim = _entry_text(item, preferred_keys=("claim", "summary", "question", "action"))
        if not claim:
            continue
        findings.append(
            AnalysisFinding(
                finding_id=str(item.get("entry_id") or item.get("question_id") or f"{prefix}{index}"),
                category=category,
                claim=claim,
                direction=str(item.get("direction")) if item.get("direction") is not None else default_direction,
                severity=str(item.get("severity")) if item.get("severity") is not None else None,
                source_basis="legacy_research_packet",
                citation_ids=[],
            )
        )
    return findings


def _legacy_questions(items: list[Any]) -> list[OpenQuestion]:
    questions: list[OpenQuestion] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = _entry_text(item, preferred_keys=("question", "summary"))
        if not question:
            continue
        questions.append(
            OpenQuestion(
                question=question,
                importance=str(item.get("importance") or "UNKNOWN"),
                next_step=str(item.get("next_step") or item.get("recommended_action") or "") or None,
            )
        )
    return questions


def _legacy_falsifiers(items: list[Any]) -> list[Falsifier]:
    falsifiers: list[Falsifier] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        description = _entry_text(item, preferred_keys=("description", "summary", "claim"))
        if not description:
            continue
        falsifiers.append(
            Falsifier(
                description=description,
                trigger_type=str(item.get("trigger_type") or "LEGACY_RESEARCH"),
                monitoring_hint=str(item.get("monitoring_hint") or "") or None,
            )
        )
    return falsifiers


def _legacy_citations(items: list[Any]) -> list[AnalysisCitation]:
    citations: list[AnalysisCitation] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        excerpt = _entry_text(item, preferred_keys=("excerpt_text", "summary", "title"))
        if not excerpt:
            continue
        citations.append(
            AnalysisCitation(
                citation_id=f"L{index}",
                source_type=str(item.get("source_type") or "legacy_research"),
                source_label=str(item.get("source_type") or item.get("title") or "Legacy research"),
                source_date=str(item.get("published_at")) if item.get("published_at") is not None else None,
                section=str(item.get("section")) if item.get("section") is not None else None,
                excerpt=excerpt,
                source_url=str(item.get("source_url")) if item.get("source_url") is not None else None,
            )
        )
    return citations


def _snapshot_from_analysis_report(
    report: AnalysisReport,
    *,
    report_path: Path | None,
    report_markdown_path: Path | None,
) -> OpsAnalysisSnapshot:
    return OpsAnalysisSnapshot(
        analysis_report=report,
        analysis_report_path=report_path,
        analysis_report_markdown_path=report_markdown_path,
        analysis_source="analysis_report",
        research_quality=report.research_quality,
        positives=list(report.positives),
        risks=list(report.risks),
        recent_event_impacts=list(report.recent_event_impacts),
        open_questions=list(report.open_questions),
        falsifiers=list(report.falsifiers),
        citations=list(report.citations),
        warnings=list(report.warnings),
        verdict=report.verdict,
        confidence_label=report.confidence_label,
        confidence_score=report.confidence_score,
        thesis_summary=report.thesis_summary,
    )


def _snapshot_from_legacy_payload(
    payload: dict[str, Any],
    *,
    legacy_path: Path,
) -> OpsAnalysisSnapshot:
    warnings = _dedupe_strs(
        [
            "legacy_research_packet_fallback",
            *[str(item) for item in payload.get("warnings", []) if isinstance(item, str)],
        ]
    )
    return OpsAnalysisSnapshot(
        analysis_report=None,
        analysis_report_path=None,
        analysis_report_markdown_path=None,
        analysis_source="legacy_research_packet_fallback",
        research_quality=_quality_from_legacy_payload(payload),
        positives=_legacy_findings(
            payload.get("findings", []),
            prefix="P",
            category="POSITIVE",
            default_direction="BULLISH",
        ),
        risks=_legacy_findings(
            payload.get("risks", []),
            prefix="R",
            category="RISK",
            default_direction="BEARISH",
        ),
        recent_event_impacts=_legacy_findings(
            payload.get("catalysts", []),
            prefix="E",
            category="RECENT_EVENT",
            default_direction=None,
        ),
        open_questions=_legacy_questions(payload.get("key_questions", [])),
        falsifiers=_legacy_falsifiers(payload.get("disconfirming_evidence", [])),
        citations=_legacy_citations(payload.get("evidence_items", [])),
        warnings=warnings,
        legacy_research_path=legacy_path,
        legacy_research_payload=payload,
    )


def load_ops_analysis_snapshot(
    ticker: str,
    *,
    as_of_date: str,
    run_id: str | None = None,
) -> OpsAnalysisSnapshot | None:
    """Load the canonical analyst snapshot, with staged legacy fallback."""
    report_path = latest_analysis_output_path(ticker, "analysis_report", as_of_date=as_of_date)
    if report_path is not None:
        report = latest_analysis_report(ticker, as_of_date=as_of_date)
        if report is not None:
            report_markdown_path = latest_analysis_output_path(
                ticker,
                "analysis_report_markdown",
                as_of_date=as_of_date,
            )
            return _snapshot_from_analysis_report(
                report,
                report_path=report_path,
                report_markdown_path=report_markdown_path,
            )

    legacy_path, _matched_run_id = _latest_legacy_research_packet_row(
        ticker,
        as_of_date=as_of_date,
        run_id=run_id,
    )
    payload = _load_json(legacy_path)
    if legacy_path is None or payload is None:
        return None
    return _snapshot_from_legacy_payload(payload, legacy_path=legacy_path)


def snapshot_to_synthesis_context(
    snapshot: OpsAnalysisSnapshot | None,
) -> dict[str, Any]:
    """Return a JSON-safe analyst context for synthesis prompts."""
    if snapshot is None:
        return {"status": "MISSING"}

    next_actions: list[dict[str, Any]] = []
    seen_actions: set[str] = set()
    for question in snapshot.open_questions:
        step = str(question.next_step or "").strip()
        if not step or step in seen_actions:
            continue
        seen_actions.add(step)
        next_actions.append(
            {
                "step_id": f"A{len(next_actions) + 1}",
                "action": step,
                "source_basis": "open_question",
            }
        )
    for gap in snapshot.research_quality.top_gaps if snapshot.research_quality else []:
        action = str(gap.recommended_action or "").strip()
        if not action or action in seen_actions:
            continue
        seen_actions.add(action)
        next_actions.append(
            {
                "step_id": f"A{len(next_actions) + 1}",
                "action": action,
                "source_basis": "research_gap",
            }
        )

    return {
        "status": "OK",
        "source": snapshot.analysis_source,
        "warnings": list(snapshot.warnings[:6]),
        "verdict": snapshot.verdict,
        "confidence_label": snapshot.confidence_label,
        "confidence_score": snapshot.confidence_score,
        "thesis_summary": snapshot.thesis_summary,
        "positives": [
            {
                "finding_id": finding.finding_id,
                "claim": finding.claim,
                "severity": finding.severity,
                "source_basis": finding.source_basis,
            }
            for finding in snapshot.positives[:10]
        ],
        "risks": [
            {
                "finding_id": finding.finding_id,
                "claim": finding.claim,
                "severity": finding.severity,
                "source_basis": finding.source_basis,
            }
            for finding in snapshot.risks[:10]
        ],
        "recent_event_impacts": [
            {
                "finding_id": finding.finding_id,
                "claim": finding.claim,
                "severity": finding.severity,
                "source_basis": finding.source_basis,
            }
            for finding in snapshot.recent_event_impacts[:10]
        ],
        "open_questions": [
            {
                "question_id": f"Q{index}",
                "question": question.question,
                "importance": question.importance,
                "next_step": question.next_step,
            }
            for index, question in enumerate(snapshot.open_questions[:6], start=1)
        ],
        "next_actions": next_actions[:12],
        "falsifiers": [
            {
                "description": falsifier.description,
                "trigger_type": falsifier.trigger_type,
                "monitoring_hint": falsifier.monitoring_hint,
            }
            for falsifier in snapshot.falsifiers[:6]
        ],
        "research_quality": (
            {
                "coverage_score": snapshot.research_quality.coverage_score,
                "freshness_score": snapshot.research_quality.freshness_score,
                "gap_score": snapshot.research_quality.gap_score,
                "overall_score": snapshot.research_quality.overall_score,
                "incomplete": snapshot.research_quality.incomplete,
                "evidence_count": snapshot.research_quality.evidence_count,
                "top_gaps": [
                    {
                        "severity": gap.severity,
                        "summary": gap.summary,
                        "recommended_action": gap.recommended_action,
                    }
                    for gap in snapshot.research_quality.top_gaps[:6]
                ],
            }
            if snapshot.research_quality is not None
            else {}
        ),
        "citations": [
            {
                "citation_id": citation.citation_id,
                "source_type": citation.source_type,
                "source_label": citation.source_label,
                "source_date": citation.source_date,
                "section": citation.section,
                "excerpt": citation.excerpt,
                "source_url": citation.source_url,
                "source_quality": citation.source_quality,
            }
            for citation in snapshot.citations[:20]
        ],
    }


def research_quality_to_rubric_dict(
    research_quality: ResearchQualitySummary | None,
) -> dict[str, Any] | None:
    if research_quality is None:
        return None
    return {
        "coverage_score": research_quality.coverage_score,
        "freshness_score": research_quality.freshness_score,
        "gap_score": research_quality.gap_score,
        "overall_research_score": research_quality.overall_score,
        "incomplete": research_quality.incomplete,
        "top_gaps": [
            {
                "severity": gap.severity,
                "summary": gap.summary,
                "recommended_action": gap.recommended_action,
            }
            for gap in research_quality.top_gaps
        ],
    }


def snapshot_to_research_payload(snapshot: OpsAnalysisSnapshot | None) -> dict[str, Any] | None:
    """Return a legacy-shaped research payload from the normalized ops snapshot."""
    if snapshot is None:
        return None
    if snapshot.legacy_research_payload is not None:
        return snapshot.legacy_research_payload

    next_actions = _dedupe_strs(
        [
            *[
                question.next_step or ""
                for question in snapshot.open_questions
            ],
            *[
                gap.recommended_action or ""
                for gap in (snapshot.research_quality.top_gaps if snapshot.research_quality else [])
            ],
        ]
    )

    return {
        "quality": research_quality_to_rubric_dict(snapshot.research_quality),
        "key_questions": [
            {
                "question_id": f"Q{index}",
                "question": item.question,
                "importance": item.importance,
                "next_step": item.next_step,
            }
            for index, item in enumerate(snapshot.open_questions, start=1)
        ],
        "findings": [
            {"entry_id": item.finding_id, "summary": item.claim, "severity": item.severity}
            for item in snapshot.positives
        ],
        "risks": [
            {"entry_id": item.finding_id, "summary": item.claim, "severity": item.severity}
            for item in snapshot.risks
        ],
        "catalysts": [
            {"entry_id": item.finding_id, "summary": item.claim, "severity": item.severity}
            for item in snapshot.recent_event_impacts
        ],
        "disconfirming_evidence": [
            {"entry_id": f"F{index}", "summary": item.description, "trigger_type": item.trigger_type}
            for index, item in enumerate(snapshot.falsifiers, start=1)
        ],
        "evidence_items": [
            {
                "source_type": item.source_type,
                "source_url": item.source_url,
                "excerpt_text": item.excerpt,
                "section": item.section,
                "published_at": item.source_date,
                "source_quality": item.source_quality,
            }
            for item in snapshot.citations
        ],
        "next_actions": [
            {"step_id": f"S{index}", "action": action}
            for index, action in enumerate(next_actions, start=1)
        ],
        "warnings": list(snapshot.warnings),
    }
