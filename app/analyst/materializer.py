"""Shared runtime helpers for producing analyst-contract artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from app.analyst.bundle_builder import build_analysis_evidence_bundle_from_cached_scorecard
from app.analyst.evidence_bundle import AnalysisEvidenceBundle, ValuationSnapshot
from app.analyst.output_store import AnalysisOutputPaths, persist_analysis_outputs
from app.analyst.report_adapter import (
    apply_investment_posture,
    build_analysis_report_from_research,
)
from app.analyst.thesis_contract import AnalysisReport, ResearchGapSummary, ResearchQualitySummary
from app.db import get_db, utc_now_iso
from app.research.deep_research import _load_scorecard


@dataclass
class MaterializedAnalysisOutputs:
    bundle: AnalysisEvidenceBundle
    report: AnalysisReport
    paths: AnalysisOutputPaths


def _load_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _latest_legacy_research_packet_payload(ticker: str, as_of_date: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT packet_path
            FROM research_packets
            WHERE ticker = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            (ticker.upper(), as_of_date),
        ).fetchone()
    if not row:
        return None
    return _load_json(Path(row["packet_path"]))


def _coerce_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _research_quality_from_legacy_packet(payload: dict | None) -> ResearchQualitySummary | None:
    if not isinstance(payload, dict):
        return None
    quality = payload.get("quality")
    if not isinstance(quality, dict):
        return None
    return ResearchQualitySummary(
        coverage_score=_coerce_float(quality.get("coverage_score")),
        freshness_score=_coerce_float(quality.get("freshness_score")),
        gap_score=_coerce_float(quality.get("gap_score")),
        overall_score=_coerce_float(quality.get("overall_research_score")),
        incomplete=bool(quality.get("incomplete")) if quality.get("incomplete") is not None else None,
        evidence_count=len(payload.get("evidence_items") or []),
        top_gaps=[
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
        ],
    )


def _fallback_bundle_from_report(
    report: object,
    *,
    years: int,
    quarters: int,
    freshness_window_days: int,
) -> AnalysisEvidenceBundle:
    warnings = list(dict.fromkeys(["scorecard_missing_for_analysis_bundle", *(getattr(report, "warnings", []) or [])]))
    return AnalysisEvidenceBundle(
        ticker=str(getattr(report, "ticker", "")).upper(),
        as_of_date=str(getattr(report, "as_of_date", "")),
        built_at=utc_now_iso(),
        analysis_years=years,
        analysis_quarters=quarters,
        freshness_window_days=freshness_window_days,
        valuation=ValuationSnapshot(
            current_price=getattr(report, "scorecard_price", None),
            market_cap=None,
            dcf_base=getattr(report, "scorecard_dcf", None),
            epv_adjusted=getattr(report, "scorecard_epv", None),
            graham_value=getattr(report, "scorecard_graham", None),
            methods_agree=getattr(report, "methods_agree", None),
            tension_type=getattr(report, "tension_type", None),
            gate_action=getattr(report, "gate_action", None),
            solvency_status=getattr(report, "solvency_status", None),
            filing_risk_status=getattr(report, "filing_risk_status", None),
        ),
        warnings=warnings,
    )


def materialize_analysis_outputs_from_research(
    report: object,
    *,
    years: int = 5,
    quarters: int = 0,
    freshness_window_days: int = 90,
) -> MaterializedAnalysisOutputs:
    """Build and persist analyst-contract artifacts from a canonical ResearchReport."""
    ticker = str(getattr(report, "ticker", "")).upper()
    as_of_date = str(getattr(report, "as_of_date", ""))

    scorecard, resolved_date = _load_scorecard(ticker, as_of_date or None)
    if scorecard is not None:
        bundle = build_analysis_evidence_bundle_from_cached_scorecard(
            ticker=ticker,
            scorecard=scorecard,
            scorecard_as_of_date=resolved_date,
            as_of_date=as_of_date or None,
            years=years,
            quarters=quarters,
            freshness_window_days=freshness_window_days,
        )
    else:
        bundle = _fallback_bundle_from_report(
            report,
            years=years,
            quarters=quarters,
            freshness_window_days=freshness_window_days,
        )

    analysis_report = build_analysis_report_from_research(bundle, report)
    research_payload = _latest_legacy_research_packet_payload(ticker, as_of_date)
    analysis_report.research_quality = _research_quality_from_legacy_packet(research_payload)
    if analysis_report.research_quality is None and "research_quality_unavailable" not in analysis_report.warnings:
        analysis_report.warnings.append("research_quality_unavailable")
    apply_investment_posture(analysis_report)
    paths = persist_analysis_outputs(bundle, analysis_report)
    return MaterializedAnalysisOutputs(
        bundle=bundle,
        report=analysis_report,
        paths=paths,
    )
