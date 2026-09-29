from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ACTION_TYPES = (
    "REFINE_PEER_SET",
    "BUILD_DOSSIERS",
    "RUN_WHALE_SIGNALS",
    "BUILD_SCOREBOARD",
    "RUN_RESEARCH_GAP_CLOSER",
    "RUN_SYNTHESIS",
    "UPDATE_DECISION_PACK",
    "BUILD_FUNDAMENTALS",
    "VALUE_TICKER",
    "PREWARM_PRICES",
    "HYDRATE_PRICE_SNAPSHOT",
    "RESOLVE_SHARES",
    "HYDRATE_SHARES",
    "HYDRATE_FCF",
    "HYDRATE_FINANCIAL_FACTS",
    "RECOMPUTE_VALUATION",
    "REWEIGHT_RUBRIC",
    "ACTION_BUILD_FUNDAMENTALS",
    "ACTION_VALUE_TICKER",
    "ACTION_REWEIGHT_RUBRIC",
    "ACTION_NARROW_PEER_SET",
    "ACTION_STOP",
    "NO_OP",
    "CLOSE_GAPS",
    "DEEPEN_DOSSIER",
    "REBUILD_SCOREBOARD",
    "UPDATE_SCOREBOARD",
    "NARROW_PEER_SET",
    "STOP",
)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal[
        "REFINE_PEER_SET",
        "BUILD_DOSSIERS",
        "RUN_WHALE_SIGNALS",
        "BUILD_SCOREBOARD",
        "RUN_RESEARCH_GAP_CLOSER",
        "RUN_SYNTHESIS",
        "UPDATE_DECISION_PACK",
        "BUILD_FUNDAMENTALS",
        "VALUE_TICKER",
        "PREWARM_PRICES",
        "HYDRATE_PRICE_SNAPSHOT",
        "RESOLVE_SHARES",
        "HYDRATE_SHARES",
        "HYDRATE_FCF",
        "HYDRATE_FINANCIAL_FACTS",
        "RECOMPUTE_VALUATION",
        "REWEIGHT_RUBRIC",
        "ACTION_BUILD_FUNDAMENTALS",
        "ACTION_VALUE_TICKER",
        "ACTION_REWEIGHT_RUBRIC",
        "ACTION_NARROW_PEER_SET",
        "ACTION_STOP",
        "NO_OP",
        "CLOSE_GAPS",
        "DEEPEN_DOSSIER",
        "REBUILD_SCOREBOARD",
        "UPDATE_SCOREBOARD",
        "NARROW_PEER_SET",
        "STOP",
    ]
    tickers: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    gap_types: list[str] = Field(default_factory=list)
    limit: int | None = None
    years_back: int | None = None
    sections: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    target: dict[str, str] | None = None
    filter_rules: dict[str, Any] = Field(default_factory=dict)
    budget_notes: str | None = None
    reason_code: str | None = None
    summary: str | None = None

    peer_mode: Literal["taxonomy", "filings", "hybrid"] | None = None
    min_peers_dossierable: int | None = None
    max_peer_scan: int | None = None
    min_annual_filings: int | None = None
    include_foreign: bool | None = None
    include_otc: bool | None = None
    top_n: int | None = None
    fallback_days: int | None = None
    weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_by_action(self) -> "Action":
        self.tickers = sorted({str(t).strip().upper() for t in self.tickers if str(t).strip()})
        self.sources = sorted({str(s).strip().lower() for s in self.sources if str(s).strip()})
        self.gap_types = sorted({str(g).strip().lower() for g in self.gap_types if str(g).strip()})
        self.sections = sorted({str(s).strip().lower() for s in self.sections if str(s).strip()})
        self.metrics = sorted({str(m).strip() for m in self.metrics if str(m).strip()})

        if self.limit is not None and int(self.limit) <= 0:
            raise ValueError("limit must be > 0 when provided")
        if self.years_back is not None and int(self.years_back) <= 0:
            raise ValueError("years_back must be > 0 when provided")
        if self.min_peers_dossierable is not None and int(self.min_peers_dossierable) <= 0:
            raise ValueError("min_peers_dossierable must be > 0 when provided")
        if self.max_peer_scan is not None and int(self.max_peer_scan) <= 0:
            raise ValueError("max_peer_scan must be > 0 when provided")
        if self.min_annual_filings is not None and int(self.min_annual_filings) <= 0:
            raise ValueError("min_annual_filings must be > 0 when provided")
        if self.top_n is not None and int(self.top_n) <= 0:
            raise ValueError("top_n must be > 0 when provided")
        if self.fallback_days is not None and int(self.fallback_days) < 0:
            raise ValueError("fallback_days must be >= 0 when provided")

        if self.action_type in {"CLOSE_GAPS", "RUN_RESEARCH_GAP_CLOSER"}:
            if self.limit is None:
                self.limit = max(1, len(self.tickers)) if self.tickers else 10
            return self

        if self.action_type in {"DEEPEN_DOSSIER", "BUILD_DOSSIERS"}:
            if self.years_back is None:
                self.years_back = 10
            if self.limit is None and self.tickers:
                self.limit = len(self.tickers)
            return self

        if self.action_type in {
            "BUILD_FUNDAMENTALS",
            "ACTION_BUILD_FUNDAMENTALS",
            "VALUE_TICKER",
            "ACTION_VALUE_TICKER",
            "PREWARM_PRICES",
            "HYDRATE_PRICE_SNAPSHOT",
            "RESOLVE_SHARES",
            "HYDRATE_SHARES",
            "HYDRATE_FCF",
            "HYDRATE_FINANCIAL_FACTS",
            "RECOMPUTE_VALUATION",
        }:
            if self.limit is None and self.tickers:
                self.limit = len(self.tickers)
            return self

        if self.action_type in {"REWEIGHT_RUBRIC", "ACTION_REWEIGHT_RUBRIC"}:
            if not self.weights and isinstance(self.filter_rules.get("weights"), dict):
                self.weights = {
                    str(key): float(value)
                    for key, value in self.filter_rules.get("weights", {}).items()
                    if isinstance(value, (int, float))
                }
            return self

        if self.action_type in {
            "REBUILD_SCOREBOARD",
            "BUILD_SCOREBOARD",
            "UPDATE_SCOREBOARD",
            "RUN_WHALE_SIGNALS",
            "UPDATE_DECISION_PACK",
        }:
            return self

        if self.action_type == "RUN_SYNTHESIS":
            target = self.target or {}
            scope = str(target.get("scope") or "").strip().lower()
            value = str(target.get("value") or "").strip()
            if scope not in {"sector", "ticker"}:
                raise ValueError("RUN_SYNTHESIS target.scope must be 'sector' or 'ticker'")
            if not value:
                raise ValueError("RUN_SYNTHESIS target.value is required")
            self.target = {"scope": scope, "value": value.upper() if scope == "ticker" else value}
            return self

        if self.action_type in {"NARROW_PEER_SET", "ACTION_NARROW_PEER_SET"}:
            if not isinstance(self.filter_rules, dict) or not self.filter_rules:
                raise ValueError("NARROW_PEER_SET requires non-empty filter_rules")
            return self

        if self.action_type == "REFINE_PEER_SET":
            return self

        if self.action_type == "NO_OP":
            if not str(self.summary or "").strip():
                raise ValueError("NO_OP requires summary")
            return self

        if self.action_type in {"STOP", "ACTION_STOP"}:
            if not str(self.reason_code or "").strip():
                raise ValueError("STOP requires reason_code")
            if not str(self.summary or "").strip():
                raise ValueError("STOP requires summary")
            return self

        raise ValueError(f"unsupported action_type={self.action_type}")


class PlannerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rlm_version: Literal["v0", "v1.1"] = "v0"
    iteration: int
    objective: str
    actions: list[Action] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_actions(self) -> "PlannerOutput":
        if self.iteration < 0:
            raise ValueError("iteration must be >= 0")
        if not self.actions:
            raise ValueError("actions must not be empty")
        if len(self.actions) > 6:
            raise ValueError("actions must contain at most 6 entries")
        stop_idxs = [
            idx for idx, action in enumerate(self.actions)
            if action.action_type in {"STOP", "ACTION_STOP"}
        ]
        if len(stop_idxs) > 1:
            raise ValueError("at most one STOP action is allowed")
        if stop_idxs and stop_idxs[0] != len(self.actions) - 1:
            raise ValueError("STOP action must be the final action")

        no_op_idxs = [idx for idx, action in enumerate(self.actions) if action.action_type == "NO_OP"]
        if len(no_op_idxs) > 1:
            raise ValueError("at most one NO_OP action is allowed")
        if no_op_idxs:
            non_terminal = [
                action.action_type for idx, action in enumerate(self.actions)
                if idx != no_op_idxs[0] and action.action_type != "STOP"
            ]
            if non_terminal:
                raise ValueError("NO_OP cannot be combined with non-STOP actions")
        return self


class CriticReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rlm_version: Literal["v0"] = "v0"
    iteration: int
    current_top_k: list[str] = Field(default_factory=list)
    previous_top_k: list[str] = Field(default_factory=list)
    evidence_count_topk: int = 0
    evidence_delta_topk: int = 0
    gap_count_topk: int = 0
    gap_reduction_topk: float = 0.0
    ranking_overlap_topk: float = 0.0
    ranking_stable: bool = False
    no_new_evidence_topk: bool = False
    trace_compliance_rate: float = 1.0
    confidence: Literal["LOW", "MED", "HIGH"] = "LOW"
    contradictions: list[str] = Field(default_factory=list)
    delta_artifact_path: str | None = None
    delta_metrics_improved: int = 0
    delta_metrics_worsened: int = 0
    delta_metrics_unknown: int = 0
    improvement_score: float = 0.0
    watch_to_pass_count: int = 0
    fail_confirmed_count: int = 0
    value_gate_status_top_k: dict[str, str] = Field(default_factory=dict)
    calibration_required: dict[str, Any] | None = None
    derived_from: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_fields(self) -> "CriticReport":
        self.current_top_k = [str(t).upper() for t in self.current_top_k if str(t).strip()]
        self.previous_top_k = [str(t).upper() for t in self.previous_top_k if str(t).strip()]
        self.value_gate_status_top_k = {
            str(ticker).upper(): str(status).upper()
            for ticker, status in (self.value_gate_status_top_k or {}).items()
            if str(ticker).strip()
        }
        self.trace_compliance_rate = max(0.0, min(1.0, float(self.trace_compliance_rate)))
        self.ranking_overlap_topk = max(0.0, min(1.0, float(self.ranking_overlap_topk)))
        return self


class StopDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_stop: bool
    status: Literal["RUNNING", "DONE", "STOPPED", "NEEDS_HUMAN"]
    reason_code: str
    summary: str
    derived_from: list[str] = Field(default_factory=list)


def _enforce_additional_properties_false(node: object) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            if "additionalProperties" not in node:
                node["additionalProperties"] = False
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties.keys())
        for value in node.values():
            _enforce_additional_properties_false(value)
    elif isinstance(node, list):
        for item in node:
            _enforce_additional_properties_false(item)


def planner_schema_for_prompt() -> dict[str, Any]:
    schema = PlannerOutput.model_json_schema()
    _enforce_additional_properties_false(schema)
    return schema
