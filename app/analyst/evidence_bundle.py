"""Analyst evidence bundle contract (input side).

Defines the normalized "what the analyst can see" object that later tasks
will populate from valuation state, filing text, recent events, and prior
thesis memory. Task 1 defines the shape only — no population logic lives
here.

Rules:
- pure dataclasses, no I/O
- repeated fields default to empty collections, never None
- to_dict returns plain JSON-safe dicts
- from_dict reconstructs nested dataclasses exactly
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ValuationSnapshot:
    """Deterministic valuation context the analyst treats as a calculator."""

    current_price: float | None
    market_cap: float | None
    dcf_base: float | None
    epv_adjusted: float | None
    graham_value: float | None
    methods_agree: bool | None
    tension_type: str | None
    gate_action: str | None
    solvency_status: str | None
    filing_risk_status: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValuationSnapshot":
        return cls(
            current_price=data.get("current_price"),
            market_cap=data.get("market_cap"),
            dcf_base=data.get("dcf_base"),
            epv_adjusted=data.get("epv_adjusted"),
            graham_value=data.get("graham_value"),
            methods_agree=data.get("methods_agree"),
            tension_type=data.get("tension_type"),
            gate_action=data.get("gate_action"),
            solvency_status=data.get("solvency_status"),
            filing_risk_status=data.get("filing_risk_status"),
        )


@dataclass
class BundleFiling:
    """Normalized filing evidence. ``role`` is ``annual`` / ``quarterly`` / ``material_event``."""

    form_type: str
    filing_date: str
    accession: str | None
    role: str
    sections_included: list[str]
    section_text: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BundleFiling":
        return cls(
            form_type=data["form_type"],
            filing_date=data["filing_date"],
            accession=data.get("accession"),
            role=data["role"],
            sections_included=list(data.get("sections_included", [])),
            section_text=dict(data.get("section_text", {})),
        )


@dataclass
class BundleEvent:
    """Event-layer evidence: SEC exhibit, IR press, company news, transcript."""

    source_type: str
    published_at: str | None
    title: str
    summary: str
    source_url: str | None
    materiality: str | None
    accession: str | None = None
    item_code: str | None = None
    event_category: str | None = None
    source_quality: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BundleEvent":
        return cls(
            source_type=data["source_type"],
            published_at=data.get("published_at"),
            title=data["title"],
            summary=data["summary"],
            source_url=data.get("source_url"),
            materiality=data.get("materiality"),
            accession=data.get("accession"),
            item_code=data.get("item_code"),
            event_category=data.get("event_category"),
            source_quality=data.get("source_quality"),
        )


@dataclass
class PriorThesisSnapshot:
    """Snapshot of the last persisted thesis. Absent means fresh analysis."""

    as_of_date: str
    verdict: str
    thesis_summary: str
    key_risks: list[str]
    falsifiers: list[str]
    report_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PriorThesisSnapshot":
        return cls(
            as_of_date=data["as_of_date"],
            verdict=data["verdict"],
            thesis_summary=data["thesis_summary"],
            key_risks=list(data.get("key_risks", [])),
            falsifiers=list(data.get("falsifiers", [])),
            report_path=data.get("report_path"),
        )


@dataclass
class AnalysisEvidenceBundle:
    """Normalized input to the analyst orchestrator.

    The valuation snapshot is nested as one field among filings, recent events,
    and prior thesis memory. It cannot act as a gate on reasoning.
    """

    ticker: str
    as_of_date: str
    built_at: str
    analysis_years: int
    analysis_quarters: int
    freshness_window_days: int
    valuation: ValuationSnapshot
    filings: list[BundleFiling] = field(default_factory=list)
    recent_events: list[BundleEvent] = field(default_factory=list)
    prior_thesis: PriorThesisSnapshot | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisEvidenceBundle":
        prior_thesis_data = data.get("prior_thesis")
        prior_thesis = (
            PriorThesisSnapshot.from_dict(prior_thesis_data)
            if prior_thesis_data is not None
            else None
        )
        return cls(
            ticker=data["ticker"],
            as_of_date=data["as_of_date"],
            built_at=data["built_at"],
            analysis_years=data["analysis_years"],
            analysis_quarters=data["analysis_quarters"],
            freshness_window_days=data["freshness_window_days"],
            valuation=ValuationSnapshot.from_dict(data["valuation"]),
            filings=[BundleFiling.from_dict(f) for f in data.get("filings", [])],
            recent_events=[BundleEvent.from_dict(e) for e in data.get("recent_events", [])],
            prior_thesis=prior_thesis,
            warnings=list(data.get("warnings", [])),
        )
