"""Analyst report contract (output side).

Defines the normalized "what the analyst concluded" object. Later tasks will
migrate the renderer, persistence, and monitoring flows onto this contract
instead of depending directly on ResearchReport.

Rules:
- pure dataclasses, no I/O
- repeated fields default to empty lists, never None
- to_dict returns plain JSON-safe dicts
- from_dict reconstructs nested dataclasses exactly
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AnalysisCitation:
    """Citation record backing an analyst finding."""

    citation_id: str
    source_type: str
    source_label: str
    source_date: str | None
    section: str | None
    excerpt: str
    source_url: str | None = None
    source_form_type: str | None = None
    source_accession: str | None = None
    source_role: str | None = None
    item_code: str | None = None
    event_category: str | None = None
    source_quality: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisCitation":
        return cls(
            citation_id=data["citation_id"],
            source_type=data["source_type"],
            source_label=data["source_label"],
            source_date=data.get("source_date"),
            section=data.get("section"),
            excerpt=data["excerpt"],
            source_url=data.get("source_url"),
            source_form_type=data.get("source_form_type"),
            source_accession=data.get("source_accession"),
            source_role=data.get("source_role"),
            item_code=data.get("item_code"),
            event_category=data.get("event_category"),
            source_quality=data.get("source_quality"),
        )


@dataclass
class AnalysisFinding:
    """Structured analyst finding.

    ``category`` is ``POSITIVE`` / ``RISK`` / ``RECENT_EVENT`` / ``ADJUSTMENT_DRIVER``.
    ``source_basis`` is ``filings`` / ``recent_events`` / ``both`` / ``prior_thesis_diff``.
    """

    finding_id: str
    category: str
    claim: str
    direction: str | None
    severity: str | None
    source_basis: str
    citation_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisFinding":
        return cls(
            finding_id=data["finding_id"],
            category=data["category"],
            claim=data["claim"],
            direction=data.get("direction"),
            severity=data.get("severity"),
            source_basis=data["source_basis"],
            citation_ids=list(data.get("citation_ids", [])),
        )


@dataclass
class ExpectationsGap:
    """Market-implied view versus analyst view, with the key mismatch."""

    market_implied_view: str
    analyst_view: str
    key_mismatch: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectationsGap":
        return cls(
            market_implied_view=data["market_implied_view"],
            analyst_view=data["analyst_view"],
            key_mismatch=data["key_mismatch"],
        )


@dataclass
class OpenQuestion:
    """Unresolved item the analyst could not answer from current evidence."""

    question: str
    importance: str
    next_step: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenQuestion":
        return cls(
            question=data["question"],
            importance=data["importance"],
            next_step=data.get("next_step"),
        )


@dataclass
class Falsifier:
    """Observable condition that would invalidate the thesis."""

    description: str
    trigger_type: str
    monitoring_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Falsifier":
        return cls(
            description=data["description"],
            trigger_type=data["trigger_type"],
            monitoring_hint=data.get("monitoring_hint"),
        )


@dataclass
class ResearchGapSummary:
    """Compact research-gap summary used by ops-facing consumers."""

    severity: str
    summary: str
    recommended_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchGapSummary":
        return cls(
            severity=data["severity"],
            summary=data["summary"],
            recommended_action=data.get("recommended_action"),
        )


@dataclass
class ResearchQualitySummary:
    """Normalized research-quality metadata carried alongside AnalysisReport."""

    coverage_score: float | None
    freshness_score: float | None
    gap_score: float | None
    overall_score: float | None
    incomplete: bool | None
    evidence_count: int | None
    top_gaps: list[ResearchGapSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchQualitySummary":
        return cls(
            coverage_score=data.get("coverage_score"),
            freshness_score=data.get("freshness_score"),
            gap_score=data.get("gap_score"),
            overall_score=data.get("overall_score"),
            incomplete=data.get("incomplete"),
            evidence_count=data.get("evidence_count"),
            top_gaps=[ResearchGapSummary.from_dict(item) for item in data.get("top_gaps", [])],
        )


@dataclass
class ValuationConclusion:
    """Analyst-side valuation framing with optional expectations gap."""

    price: float | None
    base_case_value: float | None
    bear_case_value: float | None
    bull_case_value: float | None
    margin_of_safety: float | None
    expectations_gap: ExpectationsGap | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValuationConclusion":
        gap_data = data.get("expectations_gap")
        gap = ExpectationsGap.from_dict(gap_data) if gap_data is not None else None
        return cls(
            price=data.get("price"),
            base_case_value=data.get("base_case_value"),
            bear_case_value=data.get("bear_case_value"),
            bull_case_value=data.get("bull_case_value"),
            margin_of_safety=data.get("margin_of_safety"),
            expectations_gap=gap,
        )


@dataclass
class AnalysisReport:
    """Normalized analyst-facing output contract.

    ``verdict`` is a user-facing investment posture (``BUY`` / ``WATCH`` /
    ``STAY_AWAY``), not a pipeline status code. ``confidence_label`` captures
    confidence in the thesis, not whether a deterministic gate passed.
    """

    analysis_id: str
    ticker: str
    as_of_date: str
    generated_at: str
    verdict: str
    confidence_label: str
    confidence_score: int | None
    thesis_summary: str
    valuation: ValuationConclusion
    research_gate: str | None = None
    research_quality: ResearchQualitySummary | None = None
    positives: list[AnalysisFinding] = field(default_factory=list)
    risks: list[AnalysisFinding] = field(default_factory=list)
    recent_event_impacts: list[AnalysisFinding] = field(default_factory=list)
    open_questions: list[OpenQuestion] = field(default_factory=list)
    falsifiers: list[Falsifier] = field(default_factory=list)
    citations: list[AnalysisCitation] = field(default_factory=list)
    prior_thesis_change_summary: str | None = None
    latest_evidence_date: str | None = None
    latest_evidence_source_type: str | None = None
    sources_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisReport":
        return cls(
            analysis_id=data["analysis_id"],
            ticker=data["ticker"],
            as_of_date=data["as_of_date"],
            generated_at=data["generated_at"],
            verdict=data["verdict"],
            research_gate=data.get("research_gate"),
            confidence_label=data["confidence_label"],
            confidence_score=data.get("confidence_score"),
            thesis_summary=data["thesis_summary"],
            valuation=ValuationConclusion.from_dict(data["valuation"]),
            research_quality=(
                ResearchQualitySummary.from_dict(data["research_quality"])
                if data.get("research_quality") is not None
                else None
            ),
            positives=[AnalysisFinding.from_dict(f) for f in data.get("positives", [])],
            risks=[AnalysisFinding.from_dict(f) for f in data.get("risks", [])],
            recent_event_impacts=[
                AnalysisFinding.from_dict(f) for f in data.get("recent_event_impacts", [])
            ],
            open_questions=[OpenQuestion.from_dict(q) for q in data.get("open_questions", [])],
            falsifiers=[Falsifier.from_dict(f) for f in data.get("falsifiers", [])],
            citations=[AnalysisCitation.from_dict(c) for c in data.get("citations", [])],
            prior_thesis_change_summary=data.get("prior_thesis_change_summary"),
            latest_evidence_date=data.get("latest_evidence_date"),
            latest_evidence_source_type=data.get("latest_evidence_source_type"),
            sources_used=list(data.get("sources_used", [])),
            warnings=list(data.get("warnings", [])),
        )
