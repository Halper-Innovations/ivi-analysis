"""Autonomous analyst run contract.

This module defines the durable artifact shape for the future autonomous
analyst harness. It is deliberately pure:

- no I/O
- no config access
- no DB access
- no LLM/provider imports
- repeated fields default to empty lists
- ``to_dict`` returns JSON-safe dictionaries
- ``from_dict`` reconstructs nested dataclasses exactly

The contract is the first step toward an AI-led research loop where the model
chooses questions, calls tools, records belief updates, and can stop with
``NO_WINNER`` when evidence is insufficient.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


CONTRACT_VERSION = "autonomous_analyst_run_v1"


@dataclass
class AutonomousRunBudget:
    """Budget and stopping envelope for a bounded autonomous run."""

    max_tool_calls: int
    max_turns: int
    max_cost_usd: float | None
    timebox_seconds: int | None
    max_candidates: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutonomousRunBudget":
        return cls(
            max_tool_calls=data["max_tool_calls"],
            max_turns=data["max_turns"],
            max_cost_usd=data.get("max_cost_usd"),
            timebox_seconds=data.get("timebox_seconds"),
            max_candidates=data.get("max_candidates"),
        )


@dataclass
class AutonomousRunRequest:
    """User objective and allowed operating envelope for one run."""

    run_id: str
    objective: str
    as_of_date: str
    created_at: str
    candidate_scope: dict[str, Any]
    allowed_tools: list[str]
    budget: AutonomousRunBudget
    stop_rules: list[str] = field(default_factory=list)
    user_constraints: list[str] = field(default_factory=list)
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutonomousRunRequest":
        return cls(
            run_id=data["run_id"],
            objective=data["objective"],
            as_of_date=data["as_of_date"],
            created_at=data["created_at"],
            candidate_scope=dict(data.get("candidate_scope", {})),
            allowed_tools=list(data.get("allowed_tools", [])),
            budget=AutonomousRunBudget.from_dict(data["budget"]),
            stop_rules=list(data.get("stop_rules", [])),
            user_constraints=list(data.get("user_constraints", [])),
            contract_version=data.get("contract_version", CONTRACT_VERSION),
        )


@dataclass
class ResearchQuestion:
    """Question chosen by the AI because it may change the verdict."""

    question_id: str
    question: str
    rationale: str
    priority: str
    status: str
    target_tickers: list[str] = field(default_factory=list)
    selected_tools: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchQuestion":
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            rationale=data["rationale"],
            priority=data["priority"],
            status=data["status"],
            target_tickers=list(data.get("target_tickers", [])),
            selected_tools=list(data.get("selected_tools", [])),
            depends_on=list(data.get("depends_on", [])),
        )


@dataclass
class ToolCallRecord:
    """Auditable record of one tool call the AI chose to make."""

    call_id: str
    tool_name: str
    tool_input: dict[str, Any]
    rationale: str
    status: str
    question_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    output_preview: str | None = None
    output_path: str | None = None
    error: str | None = None
    evidence_ref_ids: list[str] = field(default_factory=list)
    lane: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCallRecord":
        return cls(
            call_id=data["call_id"],
            tool_name=data["tool_name"],
            tool_input=dict(data.get("tool_input", {})),
            rationale=data["rationale"],
            status=data["status"],
            question_id=data.get("question_id"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            output_preview=data.get("output_preview"),
            output_path=data.get("output_path"),
            error=data.get("error"),
            evidence_ref_ids=list(data.get("evidence_ref_ids", [])),
            lane=data.get("lane"),
        )


@dataclass
class EvidenceReference:
    """Evidence atom produced or cited by an autonomous run."""

    evidence_id: str
    source_type: str
    source_label: str
    summary: str
    ticker: str | None = None
    source_date: str | None = None
    source_url: str | None = None
    excerpt: str | None = None
    tool_call_id: str | None = None
    confidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceReference":
        return cls(
            evidence_id=data["evidence_id"],
            source_type=data["source_type"],
            source_label=data["source_label"],
            summary=data["summary"],
            ticker=data.get("ticker"),
            source_date=data.get("source_date"),
            source_url=data.get("source_url"),
            excerpt=data.get("excerpt"),
            tool_call_id=data.get("tool_call_id"),
            confidence=data.get("confidence"),
        )


@dataclass
class BeliefUpdate:
    """Change in the AI analyst's view after evidence is gathered."""

    update_id: str
    question_id: str | None
    ticker: str | None
    prior_belief: str
    updated_belief: str
    direction: str
    confidence_after: str
    summary: str
    evidence_ref_ids: list[str] = field(default_factory=list)
    remaining_uncertainty: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BeliefUpdate":
        return cls(
            update_id=data["update_id"],
            question_id=data.get("question_id"),
            ticker=data.get("ticker"),
            prior_belief=data["prior_belief"],
            updated_belief=data["updated_belief"],
            direction=data["direction"],
            confidence_after=data["confidence_after"],
            summary=data["summary"],
            evidence_ref_ids=list(data.get("evidence_ref_ids", [])),
            remaining_uncertainty=list(data.get("remaining_uncertainty", [])),
        )


@dataclass
class CandidateDecision:
    """Candidate-level verdict produced by the autonomous analyst."""

    ticker: str
    verdict: str
    confidence: str
    thesis: str
    key_risk: str
    eligible_for_selection: bool
    selection_blockers: list[str] = field(default_factory=list)
    falsifiers: list[str] = field(default_factory=list)
    evidence_ref_ids: list[str] = field(default_factory=list)
    confidence_cap_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateDecision":
        return cls(
            ticker=data["ticker"],
            verdict=data["verdict"],
            confidence=data["confidence"],
            thesis=data["thesis"],
            key_risk=data["key_risk"],
            eligible_for_selection=bool(data.get("eligible_for_selection", False)),
            selection_blockers=list(data.get("selection_blockers", [])),
            falsifiers=list(data.get("falsifiers", [])),
            evidence_ref_ids=list(data.get("evidence_ref_ids", [])),
            confidence_cap_reasons=list(data.get("confidence_cap_reasons", [])),
        )


@dataclass
class AutonomousRunArtifact:
    """Complete audit artifact for a bounded autonomous analyst run."""

    request: AutonomousRunRequest
    status: str
    started_at: str
    completed_at: str | None
    final_verdict: str
    selected_ticker: str | None
    confidence: str | None
    questions: list[ResearchQuestion] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    evidence: list[EvidenceReference] = field(default_factory=list)
    belief_updates: list[BeliefUpdate] = field(default_factory=list)
    candidate_decisions: list[CandidateDecision] = field(default_factory=list)
    no_winner_reason: str | None = None
    degraded_states: list[str] = field(default_factory=list)
    audit_notes: list[str] = field(default_factory=list)
    provider_usage: list[dict[str, Any]] = field(default_factory=list)
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutonomousRunArtifact":
        return cls(
            request=AutonomousRunRequest.from_dict(data["request"]),
            status=data["status"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            final_verdict=data["final_verdict"],
            selected_ticker=data.get("selected_ticker"),
            confidence=data.get("confidence"),
            questions=[ResearchQuestion.from_dict(item) for item in data.get("questions", [])],
            tool_calls=[ToolCallRecord.from_dict(item) for item in data.get("tool_calls", [])],
            evidence=[EvidenceReference.from_dict(item) for item in data.get("evidence", [])],
            belief_updates=[BeliefUpdate.from_dict(item) for item in data.get("belief_updates", [])],
            candidate_decisions=[CandidateDecision.from_dict(item) for item in data.get("candidate_decisions", [])],
            no_winner_reason=data.get("no_winner_reason"),
            degraded_states=list(data.get("degraded_states", [])),
            audit_notes=list(data.get("audit_notes", [])),
            provider_usage=[
                dict(item) for item in data.get("provider_usage", []) if isinstance(item, dict)
            ],
            contract_version=data.get("contract_version", CONTRACT_VERSION),
        )
