from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class PatternHit(BaseModel):
    ticker: str
    pattern_id: str
    years_detected: list[int] = Field(default_factory=list)
    detection_strength: float = Field(ge=0.0, le=1.0)
    recency_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    outcome_confirmed: bool | None = None
    outcome_value: float | None = None
    outcome_details: str | None = None
    derived_from: list[str] = Field(default_factory=list)


class PatternResult(BaseModel):
    pattern_id: str
    hypothesis: str
    hit_count: int = 0
    confirmed_count: int = 0
    unconfirmed_count: int = 0
    hit_rate: float | None = None
    sample_size: int = 0
    hits: list[PatternHit] = Field(default_factory=list)


class PatternScanReport(BaseModel):
    run_id: str
    scan_date: str
    peer_set_size: int = 0
    peer_set_tickers: list[str] = Field(default_factory=list)
    pattern_results: list[PatternResult] = Field(default_factory=list)
    patterns_with_signal: list[str] = Field(default_factory=list)
    recency_weighted_hit_count: float = 0.0

    @model_validator(mode="after")
    def _validate_scan_date(self) -> "PatternScanReport":
        datetime.fromisoformat(self.scan_date)
        return self
