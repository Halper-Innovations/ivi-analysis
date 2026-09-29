from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DecisionPackNumericClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    value: float
    derived_from: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_trace(self) -> "DecisionPackNumericClaim":
        if not self.derived_from:
            raise ValueError("numeric claim requires derived_from trace")
        return self


class DecisionPackPeerRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    ticker: str
    overall_score: float
    future_whale_rank: int | None = None
    whale_signature_rank: int | None = None
    quality_rank: int | None = None
    valuation_rank: int | None = None
    risk_rank: int | None = None
    value_first_rank: int | None = None
    quality_score: float | None = None
    growth_score: float | None = None
    capital_discipline_score: float | None = None
    valuation_score: float | None = None
    risk_penalty: float | None = None
    score_total: float | None = None
    implied_return_base: float | None = None
    derived_from: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_trace(self) -> "DecisionPackPeerRow":
        if not self.derived_from:
            raise ValueError("peer row requires derived_from trace")
        return self


class DecisionPackWhaleSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal: str
    score_contribution: float
    status: str


class DecisionPackWhaleLeader(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    whale_signature_score: float
    top_signals: list[DecisionPackWhaleSignal] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_trace(self) -> "DecisionPackWhaleLeader":
        if not self.derived_from:
            raise ValueError("whale leader requires derived_from trace")
        return self


class DecisionPackCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    what_would_change_my_mind: list[str] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_trace(self) -> "DecisionPackCandidate":
        if not self.derived_from:
            raise ValueError("candidate requires derived_from trace")
        return self


class SectorDecisionPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    sector: str
    as_of_date: str
    created_at: str
    ranking_mode: str | None = None
    top_peers: list[DecisionPackPeerRow] = Field(default_factory=list)
    whale_signature_leaders: list[DecisionPackWhaleLeader] = Field(default_factory=list)
    top_candidates_to_deepen: list[DecisionPackCandidate] = Field(default_factory=list)
    value_quality_scatter_summary: dict[str, Any] = Field(default_factory=dict)
    peer_comparison: list[dict[str, Any]] = Field(default_factory=list)
    what_would_change_my_mind: list[str] = Field(default_factory=list)
    numeric_claims: list[DecisionPackNumericClaim] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_counts(self) -> "SectorDecisionPack":
        if not self.top_peers:
            raise ValueError("top_peers is required")
        if not self.top_candidates_to_deepen:
            raise ValueError("top_candidates_to_deepen is required")
        if len(self.top_candidates_to_deepen) > 5:
            raise ValueError("top_candidates_to_deepen must be <= 5")
        return self


def validate_sector_decision_pack(payload: dict[str, Any]) -> SectorDecisionPack:
    return SectorDecisionPack.model_validate(payload)
