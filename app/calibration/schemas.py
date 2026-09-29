from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


TrackingStatus = Literal["PENDING", "CONFIRMED", "DISCONFIRMED", "INCONCLUSIVE", "INSUFFICIENT_DATA"]
PerceptionDirection = Literal["UNDERVALUED", "OVERVALUED"]
PerceptionConfidence = Literal["HIGH", "MEDIUM", "LOW"]
PerceptionHorizon = Literal["SHORT", "MEDIUM", "LONG"]
ResolutionStatus = Literal["CONFIRMED", "DISCONFIRMED", "INCONCLUSIVE", "INSUFFICIENT_DATA"]


class PerceptionResolution(BaseModel):
    outcome_status: ResolutionStatus
    price_change_pct: float | None = None
    resolution_method: str
    notes: str
    resolved_at: str

    @model_validator(mode="after")
    def _validate_resolution(self) -> "PerceptionResolution":
        datetime.fromisoformat(self.resolved_at)
        return self


class PerceptionTrackingRecord(BaseModel):
    perception_id: str
    ticker: str
    as_of_date: str
    thesis: str = ""
    direction: PerceptionDirection
    confidence: PerceptionConfidence
    testable_prediction: str = ""
    falsification_trigger: str = ""
    time_horizon: PerceptionHorizon
    expected_resolution_date: str
    supporting_signal_sources: list[str] = Field(default_factory=list)
    pattern_ids_involved: list[str] = Field(default_factory=list)
    diff_signal_types_involved: list[str] = Field(default_factory=list)
    status: TrackingStatus = "PENDING"
    registered_at: str
    resolved_at: str | None = None
    resolution: PerceptionResolution | None = None
    derived_from: list[str] = Field(default_factory=list)
    registered_market_price: float | None = None

    @model_validator(mode="after")
    def _validate_tracking(self) -> "PerceptionTrackingRecord":
        datetime.fromisoformat(self.registered_at)
        datetime.fromisoformat(self.expected_resolution_date)
        if self.resolved_at:
            datetime.fromisoformat(self.resolved_at)
        return self


class PatternWeight(BaseModel):
    pattern_id: str
    hit_rate: float = Field(ge=0.0, le=1.0)
    sample_size: int = Field(ge=0)
    last_calibrated: str


class DiffSignalWeight(BaseModel):
    signal_type: str
    predictive_rate: float = Field(ge=0.0, le=1.0)
    sample_size: int = Field(ge=0)
    last_calibrated: str


class SectorAccuracy(BaseModel):
    sector_id: str
    accuracy_rate: float = Field(ge=0.0, le=1.0)
    sample_size: int = Field(ge=0)
    last_calibrated: str


class CalibrationWeights(BaseModel):
    pattern_weights: dict[str, PatternWeight] = Field(default_factory=dict)
    diff_signal_weights: dict[str, DiffSignalWeight] = Field(default_factory=dict)
    sector_accuracy: dict[str, SectorAccuracy] = Field(default_factory=dict)
    overall_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    total_resolved: int = Field(default=0, ge=0)
    total_confirmed: int = Field(default=0, ge=0)
    total_disconfirmed: int = Field(default=0, ge=0)
    total_inconclusive: int = Field(default=0, ge=0)
    last_calibrated: str

    @model_validator(mode="after")
    def _validate_weights(self) -> "CalibrationWeights":
        datetime.fromisoformat(self.last_calibrated)
        return self
