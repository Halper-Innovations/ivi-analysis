from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class CitationRef(BaseModel):
    source_url: str
    snippet: str
    section_label: str | None = None


class EvidenceItem(BaseModel):
    id: str
    ticker: str
    as_of_date: str
    source_type: Literal["EDGAR", "TRANSCRIPT", "WIKIPEDIA", "ir_press", "company_news", "external_news", "sec_exhibit"]
    source_url: str
    source_title: str | None = None
    source_published_at: str | None = None
    retrieved_at: str
    excerpt_text: str
    citations: list[CitationRef] = Field(default_factory=list)
    hash: str
    content_hash: str | None = None
    dedupe_key: str | None = None
    adapter_run_id: str | None = None
    source_quality: dict[str, Any] | None = None


class KeyQuestion(BaseModel):
    question_id: str
    question: str


class EvidenceLinkedEntry(BaseModel):
    entry_id: str
    summary: str
    evidence_item_ids: list[str] = Field(default_factory=list)
    citations: list[CitationRef] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_evidence_links(self) -> "EvidenceLinkedEntry":
        if not self.evidence_item_ids:
            raise ValueError(f"{self.entry_id}: missing evidence_item_ids")
        return self


class ResearchPlanStep(BaseModel):
    step_id: str
    question_id: str
    action: str
    section_targets: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    disconfirmation_check: str
    evidence_gap: str
    tied_metrics: list[str] = Field(default_factory=list)
    allowed_source: Literal["EDGAR"] = "EDGAR"


class EvidenceGap(BaseModel):
    gap_id: str
    severity: Literal["low", "medium", "high"]
    summary: str
    source_type: str
    recommended_action: str


class ClaimTrace(BaseModel):
    claim_id: str
    label: str
    value: float
    unit: str | None = None
    citations: list[CitationRef] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_trace(self) -> "ClaimTrace":
        if not self.citations and not self.derived_from:
            raise ValueError(f"{self.claim_id}: numeric claim missing citations/derived_from")
        return self


class ResearchQuality(BaseModel):
    coverage_score: float
    freshness_score: float
    gap_score: float
    overall_research_score: float
    incomplete: bool
    top_gaps: list[EvidenceGap] = Field(default_factory=list)


class ResearchPacket(BaseModel):
    run_id: str
    ticker: str
    as_of_date: str
    generated_at: str
    key_questions: list[KeyQuestion] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    findings: list[EvidenceLinkedEntry] = Field(default_factory=list)
    risks: list[EvidenceLinkedEntry] = Field(default_factory=list)
    catalysts: list[EvidenceLinkedEntry] = Field(default_factory=list)
    disconfirming_evidence: list[EvidenceLinkedEntry] = Field(default_factory=list)
    next_actions: list[ResearchPlanStep] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    quality: ResearchQuality | None = None
    claims: list[ClaimTrace] = Field(default_factory=list)
    signals: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _strict_validate(self) -> "ResearchPacket":
        if not 5 <= len(self.key_questions) <= 10:
            raise ValueError("key_questions must contain 5 to 10 entries")

        evidence_ids = {item.id for item in self.evidence_items}
        if not evidence_ids:
            raise ValueError("evidence_items cannot be empty")

        if not self.disconfirming_evidence:
            raise ValueError("disconfirming_evidence is required")

        question_ids = {q.question_id for q in self.key_questions}
        for step in self.next_actions:
            if step.question_id not in question_ids:
                raise ValueError(f"next action {step.step_id} references unknown question_id={step.question_id}")

        for bucket_name in ["findings", "risks", "catalysts", "disconfirming_evidence"]:
            bucket: list[EvidenceLinkedEntry] = getattr(self, bucket_name)
            for entry in bucket:
                missing = [item_id for item_id in entry.evidence_item_ids if item_id not in evidence_ids]
                if missing:
                    raise ValueError(f"{bucket_name}:{entry.entry_id} references unknown evidence ids: {missing}")
                if bucket_name == "findings" and not entry.citations and not entry.derived_from:
                    raise ValueError(f"{bucket_name}:{entry.entry_id} requires citations or derived_from")
        for gap in self.evidence_gaps:
            if not gap.recommended_action:
                raise ValueError(f"evidence gap {gap.gap_id} missing recommended_action")

        return self


def validate_research_packet(payload: dict[str, Any]) -> ResearchPacket:
    return ResearchPacket.model_validate(payload)
