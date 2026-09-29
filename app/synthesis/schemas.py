from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


SignalSource = Literal["VALUATION", "FILING_DIFF", "PATTERN", "INTANGIBLE_ECONOMICS"]
SignalDirection = Literal["SUPPORTS_UNDERVALUED", "SUPPORTS_OVERVALUED", "NEUTRAL"]
SignalStrength = Literal["HIGH", "MEDIUM", "LOW"]
PerceptionDirection = Literal["UNDERVALUED", "OVERVALUED"]
PerceptionConfidence = Literal["HIGH", "MEDIUM", "LOW"]
PerceptionHorizon = Literal["SHORT", "MEDIUM", "LONG"]


class SignalEvidence(BaseModel):
    source: SignalSource
    signal_type: str
    direction: SignalDirection
    strength: SignalStrength
    summary: str
    derived_from: list[str] = Field(default_factory=list)


class VariantPerception(BaseModel):
    perception_id: str
    ticker: str
    as_of_date: str
    thesis: str
    direction: PerceptionDirection
    confidence: PerceptionConfidence
    implied_vs_estimated: dict[str, float | None]
    supporting_signals: list[SignalEvidence] = Field(default_factory=list)
    contradicting_signals: list[SignalEvidence] = Field(default_factory=list)
    testable_prediction: str
    time_horizon: PerceptionHorizon
    catalyst: str
    risk: str
    derived_from: list[str] = Field(default_factory=list)
    generated_at: str

    @model_validator(mode="after")
    def _validate_variant(self) -> "VariantPerception":
        datetime.fromisoformat(self.generated_at)
        if not self.supporting_signals:
            raise ValueError("variant perception requires at least one supporting signal")
        required = {"market_implied_growth", "estimated_fair_growth", "gap_pct"}
        if not required.issubset(set(self.implied_vs_estimated.keys())):
            raise ValueError("implied_vs_estimated must include market_implied_growth, estimated_fair_growth, and gap_pct")
        return self


class VariantPerceptionReport(BaseModel):
    run_id: str
    ticker: str
    as_of_date: str
    perceptions: list[VariantPerception] = Field(default_factory=list)
    signal_summary: dict[str, Any] = Field(default_factory=dict)
    data_quality: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_report(self) -> "VariantPerceptionReport":
        if not isinstance(self.signal_summary, dict):
            raise ValueError("signal_summary must be a dict")
        if not isinstance(self.data_quality, dict):
            raise ValueError("data_quality must be a dict")
        return self
