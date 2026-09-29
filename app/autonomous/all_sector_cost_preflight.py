"""Offline worst-case cost preflight for an explicit autonomous v2 sector scan.

The module is intentionally pure: it never resolves a provider, opens a network
connection, reads config, or writes an artifact. Callers supply every sector,
candidate count, lane call coefficient, token bound, and search bound explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Mapping, Sequence

from app.autonomous.sector_lane_budget import CANONICAL_LANES, CanonicalLane
from app.llm.synthesis_agent import _estimate_cost_usd


PREFLIGHT_INPUT_ARTIFACT_TYPE = "all_sector_cost_preflight_input_v1"
PREFLIGHT_ARTIFACT_TYPE = "all_sector_cost_preflight_v1"
DIAGNOSTIC_REPRICE_ARTIFACT_TYPE = "all_sector_cost_diagnostic_reprice_v1"
DIAGNOSTIC_REPRICE_MODELS = ("gpt-5.4-mini",)
DIAGNOSTIC_REPRICE_REASON = "DIAGNOSTIC_REPRICE_CANNOT_AUTHORIZE_EXECUTION"
MAX_AUTHORIZED_COST_USD = Decimal("100.00")
MAX_AUTHORIZED_COST_MICRODOLLARS = 100_000_000
MICRODOLLARS_PER_DOLLAR = 1_000_000
MICRO_QUANTUM = Decimal("0.000001")
CENT_QUANTUM = Decimal("0.01")

# These are execution bounds, not observed averages.  The matching provider
# policy rejects a request before network I/O when its complete serialized
# request exceeds the input bound or its requested output exceeds the output
# bound.  One logical OpenAI call may make at most two physical attempts (the
# initial request plus one retry); output-expansion retries are disabled.
PRODUCTION_MODEL = "gpt-5.5"
PRODUCTION_PROVIDER = "openai"
PRODUCTION_MAX_SERIALIZED_REQUEST_BYTES = 120_000
PRODUCTION_MAX_OUTPUT_TOKENS = 5_000
PRODUCTION_MAX_RETRIES_PER_REQUEST = 1
PRODUCTION_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL = 2
PRODUCTION_TERMINAL_SEARCH_CALLS_PER_ATTEMPT = 4
PRODUCTION_WEB_SEARCH_COST_USD_PER_CALL = Decimal("0.010000")

# Price-only comparisons are deliberately narrower than the provider model
# registry.  Each target needs an explicit current rate card and compatibility
# bounds before it can be compared with the frozen production envelope.  These
# values are standard API rates per 1M tokens.
_DIAGNOSTIC_REPRICE_RATE_CARDS: dict[str, dict[str, Any]] = {
    "gpt-5.4-mini": {
        "input_usd_per_million_tokens": Decimal("0.750000"),
        "cached_input_usd_per_million_tokens": Decimal("0.075000"),
        "output_usd_per_million_tokens": Decimal("4.500000"),
        "max_input_tokens": 272_000,
        "max_output_tokens": 128_000,
        "service_tier": "standard",
        "regional_processing_uplift_included": False,
        "verified_on": "2026-07-18",
        "source_url": "https://developers.openai.com/api/docs/pricing",
    }
}


class CostPreflightDriftError(ValueError):
    """Persisted fields do not reconcile with the explicit preflight input."""


def _count(value: Any, *, field_name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    if positive and value == 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _text(value: Any, *, field_name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field_name} must be non-empty")
    return result


def _lane(value: Any) -> CanonicalLane:
    if value not in CANONICAL_LANES:
        raise ValueError(f"lane must be one of {', '.join(CANONICAL_LANES)}")
    return value


def _money(value: Any, *, field_name: str, cents: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite non-negative USD amount")
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite non-negative USD amount") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{field_name} must be a finite non-negative USD amount")
    quantum = CENT_QUANTUM if cents else MICRO_QUANTUM
    try:
        quantized = amount.quantize(quantum)
    except InvalidOperation as exc:
        raise ValueError(f"{field_name} is outside supported precision") from exc
    if quantized != amount:
        precision = "cent" if cents else "microdollar"
        raise ValueError(f"{field_name} must be {precision}-safe")
    return quantized


def _microdollars(value: Any, *, field_name: str) -> int:
    return int(_money(value, field_name=field_name) * MICRODOLLARS_PER_DOLLAR)


def _usd_string(microdollars: int) -> str:
    value = _count(microdollars, field_name="microdollars")
    return f"{Decimal(value) / MICRODOLLARS_PER_DOLLAR:.6f}"


@dataclass(frozen=True, slots=True)
class SectorCandidateCount:
    sector: str
    candidate_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "sector", _text(self.sector, field_name="sector"))
        object.__setattr__(
            self,
            "candidate_count",
            _count(self.candidate_count, field_name="candidate_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"sector": self.sector, "candidate_count": self.candidate_count}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SectorCandidateCount":
        return cls(sector=data.get("sector"), candidate_count=data.get("candidate_count"))


@dataclass(frozen=True, slots=True)
class LaneCostConfiguration:
    """Explicit model/search bounds and call-count coefficients for one lane."""

    lane: CanonicalLane | str
    provider_name: str
    model: str
    input_tokens_per_model_call: int
    cached_input_tokens_per_model_call: int
    output_tokens_per_model_call: int
    fixed_model_calls: int = 0
    model_calls_per_sector: int = 0
    model_calls_per_candidate: int = 0
    model_calls_per_terminal_cap_attempt: int = 0
    search_calls_per_terminal_cap_attempt: int = 0
    search_cost_usd_per_call: Decimal | str | int | float = Decimal("0")

    def __post_init__(self) -> None:
        canonical = _lane(self.lane)
        object.__setattr__(self, "lane", canonical)
        provider = _text(self.provider_name, field_name="provider_name").lower()
        model = _text(self.model, field_name="model").lower()
        if provider != "openai":
            raise ValueError("GPT-5.5 cost preflight requires provider_name=openai")
        if model != "gpt-5.5" and not model.startswith("gpt-5.5-"):
            raise ValueError("cost preflight model must be GPT-5.5 or a GPT-5.5 snapshot")
        object.__setattr__(self, "provider_name", provider)
        object.__setattr__(self, "model", model)
        for field_name in (
            "input_tokens_per_model_call",
            "cached_input_tokens_per_model_call",
            "output_tokens_per_model_call",
            "fixed_model_calls",
            "model_calls_per_sector",
            "model_calls_per_candidate",
            "model_calls_per_terminal_cap_attempt",
            "search_calls_per_terminal_cap_attempt",
        ):
            object.__setattr__(
                self,
                field_name,
                _count(getattr(self, field_name), field_name=field_name),
            )
        if self.cached_input_tokens_per_model_call > self.input_tokens_per_model_call:
            raise ValueError("cached input tokens cannot exceed input tokens")
        search_cost = _money(
            self.search_cost_usd_per_call,
            field_name="search_cost_usd_per_call",
        )
        object.__setattr__(self, "search_cost_usd_per_call", search_cost)
        if canonical == "terminal_cap_search":
            if (
                self.fixed_model_calls
                or self.model_calls_per_sector
                or self.model_calls_per_candidate
            ):
                raise ValueError("terminal_cap_search calls must be driven by explicit attempts")
        elif (
            self.model_calls_per_terminal_cap_attempt
            or self.search_calls_per_terminal_cap_attempt
            or search_cost
        ):
            raise ValueError(
                "terminal-attempt and search fields belong only to terminal_cap_search"
            )

    @property
    def search_cost_microdollars_per_call(self) -> int:
        return int(Decimal(self.search_cost_usd_per_call) * MICRODOLLARS_PER_DOLLAR)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "provider_name": self.provider_name,
            "model": self.model,
            "input_tokens_per_model_call": self.input_tokens_per_model_call,
            "cached_input_tokens_per_model_call": self.cached_input_tokens_per_model_call,
            "output_tokens_per_model_call": self.output_tokens_per_model_call,
            "fixed_model_calls": self.fixed_model_calls,
            "model_calls_per_sector": self.model_calls_per_sector,
            "model_calls_per_candidate": self.model_calls_per_candidate,
            "model_calls_per_terminal_cap_attempt": self.model_calls_per_terminal_cap_attempt,
            "search_calls_per_terminal_cap_attempt": self.search_calls_per_terminal_cap_attempt,
            "search_cost_microdollars_per_call": self.search_cost_microdollars_per_call,
            "search_cost_usd_per_call": _usd_string(self.search_cost_microdollars_per_call),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LaneCostConfiguration":
        config = cls(
            lane=data.get("lane"),
            provider_name=data.get("provider_name"),
            model=data.get("model"),
            input_tokens_per_model_call=data.get("input_tokens_per_model_call"),
            cached_input_tokens_per_model_call=data.get("cached_input_tokens_per_model_call"),
            output_tokens_per_model_call=data.get("output_tokens_per_model_call"),
            fixed_model_calls=data.get("fixed_model_calls", 0),
            model_calls_per_sector=data.get("model_calls_per_sector", 0),
            model_calls_per_candidate=data.get("model_calls_per_candidate", 0),
            model_calls_per_terminal_cap_attempt=data.get(
                "model_calls_per_terminal_cap_attempt", 0
            ),
            search_calls_per_terminal_cap_attempt=data.get(
                "search_calls_per_terminal_cap_attempt", 0
            ),
            search_cost_usd_per_call=data.get("search_cost_usd_per_call", "0"),
        )
        if data.get("search_cost_microdollars_per_call") != (
            config.search_cost_microdollars_per_call
        ):
            raise CostPreflightDriftError("search cost USD and microdollars do not reconcile")
        return config


@dataclass(frozen=True, slots=True)
class AllSectorCostPreflightInput:
    pipeline_version: Literal["v2"]
    sector_candidates: tuple[SectorCandidateCount, ...]
    lane_configurations: tuple[LaneCostConfiguration, ...]
    terminal_cap_search_attempts: int
    authorization_ceiling_usd: Decimal | str | int | float = MAX_AUTHORIZED_COST_USD
    prior_realized_cost_usd: Decimal | str | int | float = Decimal("0")
    artifact_type: str = PREFLIGHT_INPUT_ARTIFACT_TYPE

    def __post_init__(self) -> None:
        if self.pipeline_version != "v2":
            raise ValueError("all-sector cost preflight is available only for pipeline v2")
        if self.artifact_type != PREFLIGHT_INPUT_ARTIFACT_TYPE:
            raise ValueError("invalid all-sector preflight input artifact_type")
        if not self.sector_candidates:
            raise ValueError("sector_candidates must be explicit and non-empty")
        names = [item.sector for item in self.sector_candidates]
        if len(names) != len(set(names)):
            raise ValueError("sector candidate counts must have unique sector names")
        if tuple(item.lane for item in self.lane_configurations) != CANONICAL_LANES:
            raise ValueError("lane configurations must use canonical lanes in canonical order")
        object.__setattr__(
            self,
            "terminal_cap_search_attempts",
            _count(
                self.terminal_cap_search_attempts,
                field_name="terminal_cap_search_attempts",
            ),
        )
        ceiling = _money(
            self.authorization_ceiling_usd,
            field_name="authorization_ceiling_usd",
            cents=True,
        )
        if ceiling > MAX_AUTHORIZED_COST_USD:
            raise ValueError("authorization ceiling cannot exceed $100.00")
        object.__setattr__(self, "authorization_ceiling_usd", ceiling)
        prior_realized = _money(
            self.prior_realized_cost_usd,
            field_name="prior_realized_cost_usd",
        )
        object.__setattr__(self, "prior_realized_cost_usd", prior_realized)
        terminal = self.lane_configurations[-1]
        if self.terminal_cap_search_attempts and (
            terminal.model_calls_per_terminal_cap_attempt <= 0
            or terminal.search_calls_per_terminal_cap_attempt <= 0
            or terminal.search_cost_microdollars_per_call <= 0
        ):
            raise ValueError("terminal cap attempts require explicit model and search bounds")

    @property
    def sector_count(self) -> int:
        return len(self.sector_candidates)

    @property
    def candidate_count(self) -> int:
        return sum(item.candidate_count for item in self.sector_candidates)

    @property
    def authorization_ceiling_microdollars(self) -> int:
        return int(Decimal(self.authorization_ceiling_usd) * MICRODOLLARS_PER_DOLLAR)

    @property
    def prior_realized_cost_microdollars(self) -> int:
        return int(Decimal(self.prior_realized_cost_usd) * MICRODOLLARS_PER_DOLLAR)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "pipeline_version": self.pipeline_version,
            "sector_candidates": [item.to_dict() for item in self.sector_candidates],
            "lane_configurations": [item.to_dict() for item in self.lane_configurations],
            "terminal_cap_search_attempts": self.terminal_cap_search_attempts,
            "authorization_ceiling_cents": (self.authorization_ceiling_microdollars // 10_000),
            "authorization_ceiling_microdollars": (self.authorization_ceiling_microdollars),
            "authorization_ceiling_usd": _usd_string(self.authorization_ceiling_microdollars),
            "prior_realized_cost_microdollars": self.prior_realized_cost_microdollars,
            "prior_realized_cost_usd": _usd_string(self.prior_realized_cost_microdollars),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AllSectorCostPreflightInput":
        raw_sectors = data.get("sector_candidates")
        raw_lanes = data.get("lane_configurations")
        if not isinstance(raw_sectors, list) or not isinstance(raw_lanes, list):
            raise ValueError("sector_candidates and lane_configurations must be lists")
        result = cls(
            artifact_type=data.get("artifact_type"),
            pipeline_version=data.get("pipeline_version"),
            sector_candidates=tuple(SectorCandidateCount.from_dict(item) for item in raw_sectors),
            lane_configurations=tuple(LaneCostConfiguration.from_dict(item) for item in raw_lanes),
            terminal_cap_search_attempts=data.get("terminal_cap_search_attempts"),
            authorization_ceiling_usd=data.get("authorization_ceiling_usd"),
            prior_realized_cost_usd=data.get("prior_realized_cost_usd", "0.000000"),
        )
        if dict(data) != result.to_dict():
            raise CostPreflightDriftError("serialized preflight input does not reconcile")
        return result


@dataclass(frozen=True, slots=True)
class LaneCostEstimate:
    lane: CanonicalLane
    model: str
    model_calls: int
    search_calls: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    model_cost_microdollars: int
    search_cost_microdollars: int

    @property
    def cost_microdollars(self) -> int:
        return self.model_cost_microdollars + self.search_cost_microdollars

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_calls": self.model_calls,
            "search_calls": self.search_calls,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "model_cost_microdollars": self.model_cost_microdollars,
            "model_cost_usd": _usd_string(self.model_cost_microdollars),
            "search_cost_microdollars": self.search_cost_microdollars,
            "search_cost_usd": _usd_string(self.search_cost_microdollars),
            "cost_microdollars": self.cost_microdollars,
            "cost_usd": _usd_string(self.cost_microdollars),
        }


@dataclass(frozen=True, slots=True)
class AllSectorCostPreflight:
    request: AllSectorCostPreflightInput
    lanes: tuple[LaneCostEstimate, ...]
    status: Literal["AUTHORIZED", "STOP_BEFORE_SPEND"]
    reason_codes: tuple[str, ...]

    @property
    def cost_microdollars(self) -> int:
        return self.request.prior_realized_cost_microdollars + sum(
            item.cost_microdollars for item in self.lanes
        )

    def to_dict(self) -> dict[str, Any]:
        model_calls = sum(item.model_calls for item in self.lanes)
        search_calls = sum(item.search_calls for item in self.lanes)
        input_tokens = sum(item.input_tokens for item in self.lanes)
        cached_tokens = sum(item.cached_input_tokens for item in self.lanes)
        output_tokens = sum(item.output_tokens for item in self.lanes)
        model_cost = sum(item.model_cost_microdollars for item in self.lanes)
        search_cost = sum(item.search_cost_microdollars for item in self.lanes)
        lane_sum = sum(item.cost_microdollars for item in self.lanes)
        prior_realized = self.request.prior_realized_cost_microdollars
        return {
            "artifact_type": PREFLIGHT_ARTIFACT_TYPE,
            "status": self.status,
            "spend_authorized": self.status == "AUTHORIZED",
            "reason_codes": list(self.reason_codes),
            "input": self.request.to_dict(),
            "lane_costs": {item.lane: item.to_dict() for item in self.lanes},
            "aggregate": {
                "sector_count": self.request.sector_count,
                "candidate_count": self.request.candidate_count,
                "model_calls": model_calls,
                "search_calls": search_calls,
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_tokens,
                "output_tokens": output_tokens,
                "model_cost_microdollars": model_cost,
                "model_cost_usd": _usd_string(model_cost),
                "search_cost_microdollars": search_cost,
                "search_cost_usd": _usd_string(search_cost),
                "lane_sum_microdollars": lane_sum,
                "remaining_worst_case_cost_microdollars": lane_sum,
                "remaining_worst_case_cost_usd": _usd_string(lane_sum),
                "prior_realized_cost_microdollars": prior_realized,
                "prior_realized_cost_usd": _usd_string(prior_realized),
                "cost_microdollars": self.cost_microdollars,
                "cost_usd": _usd_string(self.cost_microdollars),
            },
            "aggregate_reconciles": (
                lane_sum + prior_realized == self.cost_microdollars
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AllSectorCostPreflight":
        raw_input = data.get("input")
        if not isinstance(raw_input, Mapping):
            raise ValueError("preflight input must be an object")
        rebuilt = estimate_all_sector_cost_preflight(
            AllSectorCostPreflightInput.from_dict(raw_input)
        )
        if dict(data) != rebuilt.to_dict():
            raise CostPreflightDriftError("serialized preflight result does not reconcile")
        return rebuilt


def estimate_all_sector_cost_preflight(
    request: AllSectorCostPreflightInput,
) -> AllSectorCostPreflight:
    """Return an offline exact estimate and an authorization/stop decision."""

    lane_estimates: list[LaneCostEstimate] = []
    for config in request.lane_configurations:
        model_calls = (
            config.fixed_model_calls
            + config.model_calls_per_sector * request.sector_count
            + config.model_calls_per_candidate * request.candidate_count
            + config.model_calls_per_terminal_cap_attempt * request.terminal_cap_search_attempts
        )
        search_calls = (
            config.search_calls_per_terminal_cap_attempt * request.terminal_cap_search_attempts
        )
        per_call_cost = _estimate_cost_usd(
            config.model,
            config.input_tokens_per_model_call,
            config.output_tokens_per_model_call,
            provider_name=config.provider_name,
            cached_input_tokens=config.cached_input_tokens_per_model_call,
        )
        per_call_microdollars = _microdollars(
            per_call_cost,
            field_name=f"{config.lane}.model_cost_per_call",
        )
        lane_estimates.append(
            LaneCostEstimate(
                lane=config.lane,
                model=config.model,
                model_calls=model_calls,
                search_calls=search_calls,
                input_tokens=config.input_tokens_per_model_call * model_calls,
                cached_input_tokens=(config.cached_input_tokens_per_model_call * model_calls),
                output_tokens=config.output_tokens_per_model_call * model_calls,
                model_cost_microdollars=per_call_microdollars * model_calls,
                search_cost_microdollars=(config.search_cost_microdollars_per_call * search_calls),
            )
        )
    aggregate = request.prior_realized_cost_microdollars + sum(
        item.cost_microdollars for item in lane_estimates
    )
    authorized = aggregate <= request.authorization_ceiling_microdollars
    return AllSectorCostPreflight(
        request=request,
        lanes=tuple(lane_estimates),
        status="AUTHORIZED" if authorized else "STOP_BEFORE_SPEND",
        reason_codes=() if authorized else ("WHOLE_RUN_WORST_CASE_EXCEEDS_AUTHORIZED_CEILING",),
    )


def build_diagnostic_v2_model_reprice(
    production_estimate: AllSectorCostPreflight,
    *,
    target_model: str,
) -> dict[str, Any]:
    """Reprice one frozen production envelope without creating spend authority.

    The production request, call coefficients, retry reserves, token bounds, and
    search costs remain unchanged.  Only the model token rate is substituted.
    The returned artifact is diagnostic by construction and cannot be consumed
    as a whole-run execution authorization.
    """

    model = _text(target_model, field_name="target_model").lower()
    if model not in DIAGNOSTIC_REPRICE_MODELS:
        raise ValueError(
            "diagnostic reprice model must be one of "
            + ", ".join(DIAGNOSTIC_REPRICE_MODELS)
        )
    if production_estimate.request.pipeline_version != "v2":
        raise ValueError("diagnostic model reprice requires a pipeline v2 estimate")
    for config in production_estimate.request.lane_configurations:
        if config.provider_name != PRODUCTION_PROVIDER or (
            config.model != PRODUCTION_MODEL
            and not config.model.startswith(f"{PRODUCTION_MODEL}-")
        ):
            raise ValueError("diagnostic reprice source must be the production model envelope")

    card = _DIAGNOSTIC_REPRICE_RATE_CARDS[model]
    max_input_tokens = int(card["max_input_tokens"])
    max_output_tokens = int(card["max_output_tokens"])
    lane_payloads: dict[str, dict[str, Any]] = {}
    model_cost_microdollars = 0
    search_cost_microdollars = 0
    for config, source_lane in zip(
        production_estimate.request.lane_configurations,
        production_estimate.lanes,
        strict=True,
    ):
        if source_lane.model_calls and (
            config.input_tokens_per_model_call > max_input_tokens
            or config.output_tokens_per_model_call > max_output_tokens
        ):
            raise ValueError(
                f"{model} cannot price {config.lane}: per-call token bound exceeds "
                "the target model input/output limit"
            )
        uncached_per_call = (
            config.input_tokens_per_model_call
            - config.cached_input_tokens_per_model_call
        )
        per_call_cost = (
            Decimal(uncached_per_call)
            * Decimal(card["input_usd_per_million_tokens"])
            + Decimal(config.cached_input_tokens_per_model_call)
            * Decimal(card["cached_input_usd_per_million_tokens"])
            + Decimal(config.output_tokens_per_model_call)
            * Decimal(card["output_usd_per_million_tokens"])
        ) / Decimal("1000000")
        per_call_microdollars = _microdollars(
            per_call_cost.quantize(MICRO_QUANTUM),
            field_name=f"{config.lane}.diagnostic_model_cost_per_call",
        )
        lane_model_cost = per_call_microdollars * source_lane.model_calls
        lane_search_cost = source_lane.search_cost_microdollars
        lane_total = lane_model_cost + lane_search_cost
        model_cost_microdollars += lane_model_cost
        search_cost_microdollars += lane_search_cost
        lane_payloads[config.lane] = {
            "model": model,
            "model_calls": source_lane.model_calls,
            "search_calls": source_lane.search_calls,
            "input_tokens": source_lane.input_tokens,
            "cached_input_tokens": source_lane.cached_input_tokens,
            "output_tokens": source_lane.output_tokens,
            "model_cost_microdollars": lane_model_cost,
            "model_cost_usd": _usd_string(lane_model_cost),
            "search_cost_microdollars": lane_search_cost,
            "search_cost_usd": _usd_string(lane_search_cost),
            "cost_microdollars": lane_total,
            "cost_usd": _usd_string(lane_total),
        }

    source = production_estimate.to_dict()
    source_aggregate = source["aggregate"]
    prior_realized = production_estimate.request.prior_realized_cost_microdollars
    lane_sum = model_cost_microdollars + search_cost_microdollars
    total = prior_realized + lane_sum
    source_remaining = int(source_aggregate["remaining_worst_case_cost_microdollars"])
    savings = source_remaining - lane_sum
    source_decimal = Decimal(source_remaining)
    target_pct = (
        (Decimal(lane_sum) / source_decimal * Decimal("100")).quantize(MICRO_QUANTUM)
        if source_remaining
        else Decimal("0.000000")
    )
    savings_pct = (
        (Decimal(savings) / source_decimal * Decimal("100")).quantize(MICRO_QUANTUM)
        if source_remaining
        else Decimal("0.000000")
    )
    return {
        "artifact_type": DIAGNOSTIC_REPRICE_ARTIFACT_TYPE,
        "status": "DIAGNOSTIC_ONLY",
        "diagnostic_only": True,
        "execution_requested": False,
        "spend_authorized": False,
        "reason_codes": [DIAGNOSTIC_REPRICE_REASON],
        "execution_binding_unchanged": True,
        "execution_compatible_with_current_v2_policy": False,
        "source_execution_binding": {
            "provider": PRODUCTION_PROVIDER,
            "model": PRODUCTION_MODEL,
        },
        "target_pricing_binding": {
            "provider": PRODUCTION_PROVIDER,
            "model": model,
            "service_tier": card["service_tier"],
        },
        "rate_card": {
            "input_usd_per_million_tokens": str(
                card["input_usd_per_million_tokens"]
            ),
            "cached_input_usd_per_million_tokens": str(
                card["cached_input_usd_per_million_tokens"]
            ),
            "output_usd_per_million_tokens": str(
                card["output_usd_per_million_tokens"]
            ),
            "max_input_tokens": max_input_tokens,
            "max_output_tokens": max_output_tokens,
            "service_tier": card["service_tier"],
            "regional_processing_uplift_included": card[
                "regional_processing_uplift_included"
            ],
            "verified_on": card["verified_on"],
            "source_url": card["source_url"],
        },
        "cost_envelope": {
            "lane_costs": lane_payloads,
            "aggregate": {
                "sector_count": source_aggregate["sector_count"],
                "candidate_count": source_aggregate["candidate_count"],
                "model_calls": source_aggregate["model_calls"],
                "search_calls": source_aggregate["search_calls"],
                "input_tokens": source_aggregate["input_tokens"],
                "cached_input_tokens": source_aggregate["cached_input_tokens"],
                "output_tokens": source_aggregate["output_tokens"],
                "model_cost_microdollars": model_cost_microdollars,
                "model_cost_usd": _usd_string(model_cost_microdollars),
                "search_cost_microdollars": search_cost_microdollars,
                "search_cost_usd": _usd_string(search_cost_microdollars),
                "lane_sum_microdollars": lane_sum,
                "remaining_worst_case_cost_microdollars": lane_sum,
                "remaining_worst_case_cost_usd": _usd_string(lane_sum),
                "prior_realized_cost_microdollars": prior_realized,
                "prior_realized_cost_usd": _usd_string(prior_realized),
                "cost_microdollars": total,
                "cost_usd": _usd_string(total),
            },
            "aggregate_reconciles": (
                sum(row["cost_microdollars"] for row in lane_payloads.values())
                + prior_realized
                == total
            ),
        },
        "comparison": {
            "source_remaining_worst_case_cost_microdollars": source_remaining,
            "source_remaining_worst_case_cost_usd": _usd_string(source_remaining),
            "diagnostic_remaining_worst_case_cost_microdollars": lane_sum,
            "diagnostic_remaining_worst_case_cost_usd": _usd_string(lane_sum),
            "estimated_savings_microdollars": savings,
            "estimated_savings_usd": f"{Decimal(savings) / MICRODOLLARS_PER_DOLLAR:.6f}",
            "diagnostic_cost_as_pct_of_source": f"{target_pct:.6f}",
            "estimated_savings_pct": f"{savings_pct:.6f}",
        },
        "actual_usage": {
            "model_calls": 0,
            "search_calls": 0,
            "network_calls": 0,
            "cost_microdollars": 0,
            "cost_usd": "0.000000",
        },
    }


def production_v2_lane_configurations(
    *,
    parent_max_turns: int,
) -> tuple[LaneCostConfiguration, ...]:
    """Return conservative, provider-enforced GPT-5.5 production bounds.

    Coefficients count maximum *physical* requests.  Parent research reserves
    every configured turn, compact recovery/finalization, the two shared memo
    sections, and one candidate memo per admitted company.  Company and repair
    lanes reserve the full three-turn child path for every candidate; selected
    validation reserves the same path once per sector.
    """

    turns = _count(parent_max_turns, field_name="parent_max_turns", positive=True)
    physical = PRODUCTION_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
    common = {
        "provider_name": PRODUCTION_PROVIDER,
        "model": PRODUCTION_MODEL,
        "input_tokens_per_model_call": PRODUCTION_MAX_SERIALIZED_REQUEST_BYTES,
        "cached_input_tokens_per_model_call": 0,
        "output_tokens_per_model_call": PRODUCTION_MAX_OUTPUT_TOKENS,
    }
    # The terminal web-search call has a documented 1.05M model context and no
    # API max-input control.  It is therefore reserved at the full billable
    # context bound used by terminal_cap_search rather than the ordinary prompt
    # byte guard.
    from app.autonomous.terminal_cap_search import (
        TERMINAL_CAP_SEARCH_MAX_BILLABLE_INPUT_TOKENS,
        TERMINAL_CAP_SEARCH_MAX_OUTPUT_TOKENS,
    )

    return (
        LaneCostConfiguration(
            lane="provider_preflight",
            fixed_model_calls=physical,
            **common,
        ),
        LaneCostConfiguration(
            lane="parent_research",
            model_calls_per_sector=(turns + 4) * physical,
            model_calls_per_candidate=physical,
            **common,
        ),
        LaneCostConfiguration(
            lane="company_underwriting",
            model_calls_per_candidate=3 * physical,
            **common,
        ),
        LaneCostConfiguration(
            lane="selected_company_validation",
            model_calls_per_sector=3 * physical,
            **common,
        ),
        LaneCostConfiguration(
            lane="repair_fallback",
            model_calls_per_sector=2 * physical,
            model_calls_per_candidate=3 * physical,
            **common,
        ),
        LaneCostConfiguration(
            lane="terminal_cap_search",
            provider_name=PRODUCTION_PROVIDER,
            model=PRODUCTION_MODEL,
            input_tokens_per_model_call=TERMINAL_CAP_SEARCH_MAX_BILLABLE_INPUT_TOKENS,
            cached_input_tokens_per_model_call=0,
            output_tokens_per_model_call=TERMINAL_CAP_SEARCH_MAX_OUTPUT_TOKENS,
            model_calls_per_terminal_cap_attempt=1,
            search_calls_per_terminal_cap_attempt=(
                PRODUCTION_TERMINAL_SEARCH_CALLS_PER_ATTEMPT
            ),
            search_cost_usd_per_call=PRODUCTION_WEB_SEARCH_COST_USD_PER_CALL,
        ),
    )


def build_production_v2_cost_preflight(
    *,
    sector_candidate_counts: Mapping[str, int] | Sequence[SectorCandidateCount],
    terminal_cap_search_attempts: int,
    parent_max_turns: int,
    prior_realized_cost_usd: Decimal | str | int | float = Decimal("0"),
    authorization_ceiling_usd: Decimal | str | int | float = MAX_AUTHORIZED_COST_USD,
) -> AllSectorCostPreflight:
    """Build the single production authorization used by v2 benchmark runs."""

    if isinstance(sector_candidate_counts, Mapping):
        counts = tuple(
            SectorCandidateCount(str(sector), int(count))
            for sector, count in sector_candidate_counts.items()
        )
    else:
        counts = tuple(sector_candidate_counts)
    request = AllSectorCostPreflightInput(
        pipeline_version="v2",
        sector_candidates=counts,
        lane_configurations=production_v2_lane_configurations(
            parent_max_turns=parent_max_turns,
        ),
        terminal_cap_search_attempts=terminal_cap_search_attempts,
        authorization_ceiling_usd=authorization_ceiling_usd,
        prior_realized_cost_usd=prior_realized_cost_usd,
    )
    return estimate_all_sector_cost_preflight(request)


__all__ = [
    "AllSectorCostPreflight",
    "AllSectorCostPreflightInput",
    "CostPreflightDriftError",
    "DIAGNOSTIC_REPRICE_ARTIFACT_TYPE",
    "DIAGNOSTIC_REPRICE_MODELS",
    "DIAGNOSTIC_REPRICE_REASON",
    "LaneCostConfiguration",
    "LaneCostEstimate",
    "MAX_AUTHORIZED_COST_MICRODOLLARS",
    "MAX_AUTHORIZED_COST_USD",
    "PREFLIGHT_ARTIFACT_TYPE",
    "PREFLIGHT_INPUT_ARTIFACT_TYPE",
    "SectorCandidateCount",
    "PRODUCTION_MAX_OUTPUT_TOKENS",
    "PRODUCTION_MAX_RETRIES_PER_REQUEST",
    "PRODUCTION_MAX_SERIALIZED_REQUEST_BYTES",
    "PRODUCTION_MODEL",
    "PRODUCTION_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL",
    "PRODUCTION_PROVIDER",
    "build_production_v2_cost_preflight",
    "build_diagnostic_v2_model_reprice",
    "estimate_all_sector_cost_preflight",
    "production_v2_lane_configurations",
]
