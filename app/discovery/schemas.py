from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


UNKNOWN = "UNKNOWN"


class DiscoveryClaimCitation(BaseModel):
    source_url: str
    snippet: str = ""
    section_label: str | None = None


class DiscoveryClaim(BaseModel):
    claim_id: str
    label: str
    value: float | str
    unit: str | None = None
    citations: list[DiscoveryClaimCitation] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)

    @property
    def is_numeric(self) -> bool:
        return isinstance(self.value, (int, float))

    @model_validator(mode="after")
    def validate_numeric_trace(self) -> "DiscoveryClaim":
        if self.is_numeric and not self.citations and not self.derived_from:
            raise ValueError("numeric claim requires citations or derived_from")
        return self


class DiscoveryReasonEvidence(BaseModel):
    reason: str
    citations: list[DiscoveryClaimCitation] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)


class DiscoverySubscoreReason(BaseModel):
    reason_code: str
    summary: str
    severity: Literal["boost", "penalty", "info"]
    derived_from: list[str] = Field(default_factory=list)
    notes: str | None = None


class DiscoveryArtifacts(BaseModel):
    evidence_packet_path: str | None = None
    filing_accessions_used: list[str] = Field(default_factory=list)


class DiscoveryCandidate(BaseModel):
    ticker: str
    cik: str
    company_name: str | None = None
    run_id: str
    run_as_of_date: str
    effective_as_of_date: str
    market_cap: float | str
    discovery_score: float
    stage: Literal["REJECT", "WATCHLIST_ONLY", "ADVANCE_TO_DEEP"] = "WATCHLIST_ONLY"
    whale_fit_score: float = 0.0
    evidence_strength_score: float = 0.0
    explainability_score: float = 0.0
    subscores_json: dict[str, Any] = Field(default_factory=dict)
    key_reasons: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    reason_evidence: list[DiscoveryReasonEvidence] = Field(default_factory=list)
    subscore_reasons: list[DiscoverySubscoreReason] = Field(default_factory=list)
    numeric_claims: list[DiscoveryClaim] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    newly_surfaced: bool = False
    repeat_surfaced: bool = False
    recommended_action: Literal["ADD_TO_UNIVERSE", "WATCHLIST_ONLY", "SKIP"]
    suggested_next_pipeline: Literal["FULL_RESEARCH", "NONE"]
    artifacts: DiscoveryArtifacts = Field(default_factory=DiscoveryArtifacts)

    @model_validator(mode="after")
    def validate_market_cap_flag(self) -> "DiscoveryCandidate":
        if isinstance(self.market_cap, str) and self.market_cap == UNKNOWN and "MARKET_CAP_UNKNOWN" not in self.flags:
            self.flags.append("MARKET_CAP_UNKNOWN")
        return self


class DiscoveryRunStats(BaseModel):
    run_id: str
    run_as_of_date: str
    seed_hash: str
    config_hash: str
    tickers_targeted: int
    tickers_processed: int
    missing_cik_count: int
    skipped_due_to_throttle: bool = False
    errors_count: int = 0
    market_cap_known_count: int = 0
    market_cap_in_band_count: int = 0
    filtered_market_cap_count: int = 0
    unknown_market_cap_count: int = 0
    shortlisted_count: int = 0
    shadow_count: int = 0
    suppressed_counts: dict[str, int] = Field(default_factory=dict)
    stage_counts: dict[str, int] = Field(default_factory=dict)
    top_k: int
