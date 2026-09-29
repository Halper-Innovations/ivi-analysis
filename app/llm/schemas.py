from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator
import re


class ClaimCitation(BaseModel):
    source_url: str
    snippet: str = ""
    section_label: str | None = None


class SynthesisHypothesis(BaseModel):
    id: str
    statement: str
    why_it_might_be_true: str
    falsifiers: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)


class SynthesisClaim(BaseModel):
    id: str
    text: str
    type: Literal["numeric", "non_numeric"]
    citations: list[ClaimCitation] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_numeric_trace(self) -> "SynthesisClaim":
        if self.type == "numeric" and not self.citations and not self.derived_from:
            raise ValueError("numeric claim requires citations or derived_from trace")
        return self


class PricedInAssessment(BaseModel):
    what_market_assumes: str
    what_is_not_priced: str
    uncertainty_notes: str


class SynthesisNextAction(BaseModel):
    action_type: str
    target_source: str
    query_or_url_hint: str
    why: str


class SynthesisDecisionFrame(BaseModel):
    stance: Literal["long", "short", "watchlist", "avoid"]
    key_risks: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    time_horizon_days: int


class SynthesisLLMMeta(BaseModel):
    model: str
    prompt_hash: str
    input_hash: str
    cost_estimate_usd: float
    created_at: str

    @model_validator(mode="after")
    def validate_created_at(self) -> "SynthesisLLMMeta":
        datetime.fromisoformat(self.created_at)
        return self


class SynthesisPacket(BaseModel):
    ticker: str
    as_of_date: str
    run_id: str
    requested_as_of_date: str = ""
    effective_as_of_date: str = ""
    as_of_resolution: Literal["exact", "fallback"] = "exact"
    as_of_resolution_reason: str = ""
    business_quality_summary: str = ""
    valuation_interpretation: str = ""
    risk_frame: str = ""
    catalyst_frame: str = ""
    evidence_gaps: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    confidence_notes: str = ""
    hypotheses: list[SynthesisHypothesis] = Field(default_factory=list)
    claims: list[SynthesisClaim] = Field(default_factory=list)
    priced_in_assessment: PricedInAssessment
    next_actions: list[SynthesisNextAction] = Field(default_factory=list)
    decision_frame: SynthesisDecisionFrame
    llm_meta: SynthesisLLMMeta

    @model_validator(mode="after")
    def validate_required_sections(self) -> "SynthesisPacket":
        if not self.hypotheses:
            raise ValueError("at least one hypothesis is required")
        if not self.next_actions:
            raise ValueError("at least one next_action is required")
        return self


def validate_synthesis_packet(payload: dict) -> SynthesisPacket:
    return SynthesisPacket.model_validate(payload)


def _enforce_additional_properties_false(node: object) -> None:
    if isinstance(node, dict):
        node_type = node.get("type")
        if node_type == "object":
            if "additionalProperties" not in node:
                node["additionalProperties"] = False
            properties = node.get("properties")
            if isinstance(properties, dict):
                # OpenAI Structured Outputs expects object schemas to list every property in `required`.
                node["required"] = list(properties.keys())

        for value in node.values():
            _enforce_additional_properties_false(value)
    elif isinstance(node, list):
        for item in node:
            _enforce_additional_properties_false(item)


def synthesis_schema_for_prompt() -> dict:
    schema = SynthesisPacket.model_json_schema()
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    for field in [
        "requested_as_of_date",
        "effective_as_of_date",
        "as_of_resolution",
        "as_of_resolution_reason",
    ]:
        properties.pop(field, None)
        if field in required:
            required.remove(field)
    _enforce_additional_properties_false(schema)
    return schema


def check_numeric_claim_trace(payload: dict) -> list[str]:
    failures: list[str] = []
    claims = payload.get("claims") if isinstance(payload, dict) else None
    if not isinstance(claims, list):
        return ["claims missing"]
    for claim in claims:
        if not isinstance(claim, dict):
            failures.append("invalid claim payload")
            continue
        if claim.get("type") != "numeric":
            continue
        citations = claim.get("citations") or []
        derived_from = claim.get("derived_from") or []
        if not citations and not derived_from:
            failures.append(f"numeric claim missing trace: {claim.get('id', 'UNKNOWN')}")
    return failures


def can_parse_synthesis_packet(payload: dict) -> tuple[bool, str]:
    try:
        validate_synthesis_packet(payload)
    except ValidationError as exc:
        return False, str(exc)
    return True, ""


class SectorSynthesisClaim(BaseModel):
    id: str
    text: str
    type: Literal["numeric", "non_numeric"]
    citations: list[ClaimCitation] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_numeric_trace(self) -> "SectorSynthesisClaim":
        if self.type == "numeric" and not self.citations and not self.derived_from:
            raise ValueError("numeric claim requires citations or derived_from trace")
        return self


class SectorTickerNarrative(BaseModel):
    ticker: str
    moat_indicators: list[str] = Field(default_factory=list)
    operating_leverage: str
    capital_allocation: str
    reinvestment_runway: str
    citations: list[ClaimCitation] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)


class SectorSynthesisLLMMeta(BaseModel):
    model: str
    prompt_hash: str
    input_hash: str
    cost_estimate_usd: float
    created_at: str

    @model_validator(mode="after")
    def validate_created_at(self) -> "SectorSynthesisLLMMeta":
        datetime.fromisoformat(self.created_at)
        return self


class SectorSynthesisPacket(BaseModel):
    sector: str
    as_of_date: str
    run_id: str
    sector_summary: str = ""
    valuation_interpretation: str = ""
    risk_frame: str = ""
    catalyst_frame: str = ""
    evidence_gaps: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    confidence_notes: str = ""
    peer_tickers: list[str] = Field(default_factory=list)
    top_pick: str
    runner_ups: list[str] = Field(default_factory=list)
    avoid_list: list[str] = Field(default_factory=list)
    whale_checklist: list[str] = Field(default_factory=list)
    per_ticker_narrative: list[SectorTickerNarrative] = Field(default_factory=list)
    falsifiers: list[str] = Field(default_factory=list)
    what_to_read_next: list[str] = Field(default_factory=list)
    claims: list[SectorSynthesisClaim] = Field(default_factory=list)
    llm_meta: SectorSynthesisLLMMeta

    @model_validator(mode="after")
    def validate_sector_fields(self) -> "SectorSynthesisPacket":
        if not self.peer_tickers:
            raise ValueError("peer_tickers is required")
        if self.top_pick not in self.peer_tickers:
            raise ValueError("top_pick must be one of peer_tickers")
        for ticker in self.runner_ups:
            if ticker not in self.peer_tickers:
                raise ValueError("runner_ups must be subset of peer_tickers")
        for ticker in self.avoid_list:
            if ticker not in self.peer_tickers:
                raise ValueError("avoid_list must be subset of peer_tickers")
        if not self.per_ticker_narrative:
            raise ValueError("at least one per_ticker_narrative entry is required")
        # Enforce numeric-claim discipline by disallowing raw numeric strings in narrative text fields.
        # Numeric assertions should be represented in `claims` with citations/derived_from traces.
        digit_re = re.compile(r"\d")
        text_fields: list[str] = []
        text_fields.extend(self.whale_checklist)
        text_fields.extend(self.falsifiers)
        text_fields.extend(self.what_to_read_next)
        for row in self.per_ticker_narrative:
            text_fields.extend(row.moat_indicators)
            text_fields.append(row.operating_leverage)
            text_fields.append(row.capital_allocation)
            text_fields.append(row.reinvestment_runway)
        if any(isinstance(text, str) and digit_re.search(text or "") for text in text_fields):
            raise ValueError("numeric content must be placed in claims[] with citations or derived_from")
        return self


def validate_sector_synthesis_packet(payload: dict) -> SectorSynthesisPacket:
    return SectorSynthesisPacket.model_validate(payload)


def sector_synthesis_schema_for_prompt() -> dict:
    schema = SectorSynthesisPacket.model_json_schema()
    _enforce_additional_properties_false(schema)
    return schema
