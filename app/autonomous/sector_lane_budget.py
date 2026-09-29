"""Pure lane budgets and exact usage accounting for autonomous sector runs.

This module deliberately has no provider, network, database, config, or file I/O.
It defines the bounded contract that runtime orchestration can consume later:

* every research lane owns an independent tool, turn, and cost reserve;
* every attempted call consumes capacity, including failed calls;
* money is represented as integer microdollars after exact ``Decimal`` validation;
* persisted summaries must reconcile exactly with their underlying call records; and
* whole-run authorization stops before a worst case above the hard $100 ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Mapping, TypeAlias


CanonicalLane: TypeAlias = Literal[
    "provider_preflight",
    "parent_research",
    "company_underwriting",
    "selected_company_validation",
    "repair_fallback",
    "terminal_cap_search",
]

CANONICAL_LANES: tuple[CanonicalLane, ...] = (
    "provider_preflight",
    "parent_research",
    "company_underwriting",
    "selected_company_validation",
    "repair_fallback",
    "terminal_cap_search",
)
PER_SECTOR_LANES: frozenset[CanonicalLane] = frozenset(
    {
        "parent_research",
        "company_underwriting",
        "selected_company_validation",
        "repair_fallback",
    }
)
WHOLE_RUN_LANES: frozenset[CanonicalLane] = frozenset({"provider_preflight", "terminal_cap_search"})

POLICY_ARTIFACT_TYPE = "autonomous_sector_lane_budget_policy_v1"
USAGE_LEDGER_ARTIFACT_TYPE = "autonomous_sector_lane_usage_ledger_v1"
WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE = "autonomous_sector_lane_cost_preflight_v1"

MICRODOLLARS_PER_DOLLAR = 1_000_000
CENT_MICRODOLLARS = 10_000
USD_MICRO_QUANTUM = Decimal("0.000001")
USD_CENT_QUANTUM = Decimal("0.01")
MAX_AUTHORIZED_WHOLE_RUN_USD = Decimal("100.00")
MAX_AUTHORIZED_WHOLE_RUN_MICRODOLLARS = 100_000_000


class LaneBudgetExceededError(RuntimeError):
    """Raised before a call would cross its lane's independent reserve."""


class LaneAccountingDriftError(ValueError):
    """Raised when declared persisted accounting differs from exact records."""


class WholeRunAuthorizationError(RuntimeError):
    """Raised before a whole run whose worst case is not authorized."""


MoneyInput: TypeAlias = Decimal | str | int | float


def _non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _positive_int(value: Any, *, field_name: str) -> int:
    result = _non_negative_int(value, field_name=field_name)
    if result == 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _non_empty(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _canonical_lane(value: Any) -> CanonicalLane:
    if value not in CANONICAL_LANES:
        raise ValueError(f"lane must be one of {', '.join(CANONICAL_LANES)}")
    return value


def _exact_usd_decimal(
    value: MoneyInput,
    *,
    field_name: str,
    quantum: Decimal = USD_MICRO_QUANTUM,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite non-negative USD amount")
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite non-negative USD amount") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{field_name} must be a finite non-negative USD amount")
    try:
        exact = amount.quantize(quantum)
    except InvalidOperation as exc:
        raise ValueError(f"{field_name} is outside the supported USD precision") from exc
    if exact != amount:
        places = 2 if quantum == USD_CENT_QUANTUM else 6
        raise ValueError(f"{field_name} must be exact to at most {places} decimal places")
    return exact


def _usd_to_microdollars(value: MoneyInput, *, field_name: str) -> int:
    amount = _exact_usd_decimal(value, field_name=field_name)
    return int(amount * MICRODOLLARS_PER_DOLLAR)


def _microdollars_to_usd_string(value: int) -> str:
    microdollars = _non_negative_int(value, field_name="microdollars")
    return f"{Decimal(microdollars) / MICRODOLLARS_PER_DOLLAR:.6f}"


def _payload_cost_microdollars(
    payload: Mapping[str, Any],
    *,
    field_prefix: str,
    usd_key: str = "cost_usd",
    microdollars_key: str = "cost_microdollars",
) -> int:
    if usd_key not in payload or microdollars_key not in payload:
        raise LaneAccountingDriftError(
            f"{field_prefix} must declare both {usd_key} and {microdollars_key}"
        )
    try:
        declared_microdollars = _non_negative_int(
            payload[microdollars_key],
            field_name=f"{field_prefix}.{microdollars_key}",
        )
        derived_microdollars = _usd_to_microdollars(
            payload[usd_key],
            field_name=f"{field_prefix}.{usd_key}",
        )
    except ValueError as exc:
        raise LaneAccountingDriftError(str(exc)) from exc
    if declared_microdollars != derived_microdollars:
        raise LaneAccountingDriftError(
            f"{field_prefix} {usd_key} does not match {microdollars_key}"
        )
    return declared_microdollars


@dataclass(frozen=True, slots=True)
class LaneBudget:
    """Independent bounded reserve for one canonical lane."""

    max_tool_calls: int
    max_turns: int
    max_cost_usd: Decimal | str | int | float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_tool_calls",
            _non_negative_int(self.max_tool_calls, field_name="max_tool_calls"),
        )
        object.__setattr__(
            self,
            "max_turns",
            _non_negative_int(self.max_turns, field_name="max_turns"),
        )
        object.__setattr__(
            self,
            "max_cost_usd",
            _exact_usd_decimal(self.max_cost_usd, field_name="max_cost_usd"),
        )

    @property
    def max_cost_microdollars(self) -> int:
        return int(Decimal(self.max_cost_usd) * MICRODOLLARS_PER_DOLLAR)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tool_calls": self.max_tool_calls,
            "max_turns": self.max_turns,
            "max_cost_microdollars": self.max_cost_microdollars,
            "max_cost_usd": _microdollars_to_usd_string(self.max_cost_microdollars),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LaneBudget":
        cost_microdollars = _payload_cost_microdollars(
            data,
            field_prefix="lane_budget",
            usd_key="max_cost_usd",
            microdollars_key="max_cost_microdollars",
        )
        budget = cls(
            max_tool_calls=data.get("max_tool_calls"),
            max_turns=data.get("max_turns"),
            max_cost_usd=data.get("max_cost_usd"),
        )
        if budget.max_cost_microdollars != cost_microdollars:
            raise LaneAccountingDriftError("lane budget cost fields do not reconcile")
        return budget


@dataclass(frozen=True, slots=True)
class CompanyChildBudget:
    """Per-child call profile drawing from the company-underwriting lane."""

    profile: Literal["initial", "extended"]
    max_tool_calls: int
    max_turns: int

    def __post_init__(self) -> None:
        expected = {
            "initial": (4, 2),
            "extended": (8, 3),
        }.get(self.profile)
        if expected is None:
            raise ValueError("company child profile must be initial or extended")
        tool_calls = _non_negative_int(
            self.max_tool_calls,
            field_name="max_tool_calls",
        )
        turns = _non_negative_int(
            self.max_turns,
            field_name="max_turns",
        )
        if (tool_calls, turns) != expected:
            raise ValueError(
                f"{self.profile} company child budget must be {expected[0]} tools/{expected[1]} turns"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "max_tool_calls": self.max_tool_calls,
            "max_turns": self.max_turns,
        }


INITIAL_COMPANY_CHILD_BUDGET = CompanyChildBudget(
    profile="initial",
    max_tool_calls=4,
    max_turns=2,
)
EXTENDED_COMPANY_CHILD_BUDGET = CompanyChildBudget(
    profile="extended",
    max_tool_calls=8,
    max_turns=3,
)


@dataclass(frozen=True, slots=True)
class SectorLaneBudgetPolicy:
    """Immutable six-lane policy with no shared or fungible capacity."""

    provider_preflight: LaneBudget
    parent_research: LaneBudget
    company_underwriting: LaneBudget
    selected_company_validation: LaneBudget
    repair_fallback: LaneBudget
    terminal_cap_search: LaneBudget

    def __post_init__(self) -> None:
        for lane in CANONICAL_LANES:
            if not isinstance(getattr(self, lane), LaneBudget):
                raise TypeError(f"{lane} must be a LaneBudget")

    def lane_budget(self, lane: CanonicalLane | str) -> LaneBudget:
        canonical = _canonical_lane(lane)
        return getattr(self, canonical)

    def lane_budgets(self) -> tuple[tuple[CanonicalLane, LaneBudget], ...]:
        return tuple((lane, self.lane_budget(lane)) for lane in CANONICAL_LANES)

    def company_child_budget(self, profile: Literal["initial", "extended"]) -> CompanyChildBudget:
        if profile == "initial":
            return INITIAL_COMPANY_CHILD_BUDGET
        if profile == "extended":
            return EXTENDED_COMPANY_CHILD_BUDGET
        raise ValueError("company child profile must be initial or extended")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": POLICY_ARTIFACT_TYPE,
            "lanes": {lane: budget.to_dict() for lane, budget in self.lane_budgets()},
            "company_child_profiles": {
                "initial": INITIAL_COMPANY_CHILD_BUDGET.to_dict(),
                "extended": EXTENDED_COMPANY_CHILD_BUDGET.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SectorLaneBudgetPolicy":
        if data.get("artifact_type") != POLICY_ARTIFACT_TYPE:
            raise ValueError("invalid sector lane budget policy artifact_type")
        raw_lanes = data.get("lanes")
        if not isinstance(raw_lanes, Mapping) or set(raw_lanes) != set(CANONICAL_LANES):
            raise ValueError("policy must contain exactly the six canonical lanes")
        expected_profiles = {
            "initial": INITIAL_COMPANY_CHILD_BUDGET.to_dict(),
            "extended": EXTENDED_COMPANY_CHILD_BUDGET.to_dict(),
        }
        if data.get("company_child_profiles") != expected_profiles:
            raise ValueError("company child budget profiles have drifted")
        return cls(**{lane: LaneBudget.from_dict(raw_lanes[lane]) for lane in CANONICAL_LANES})


@dataclass(frozen=True, slots=True)
class ToolCallUsage:
    call_id: str
    lane: CanonicalLane
    tool_name: str
    status: str
    cost_microdollars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "lane": self.lane,
            "tool_name": self.tool_name,
            "status": self.status,
            "cost_microdollars": self.cost_microdollars,
            "cost_usd": _microdollars_to_usd_string(self.cost_microdollars),
        }


@dataclass(frozen=True, slots=True)
class ProviderCallUsage:
    call_id: str
    lane: CanonicalLane
    provider: str
    model: str
    status: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_microdollars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "lane": self.lane,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "cost_microdollars": self.cost_microdollars,
            "cost_usd": _microdollars_to_usd_string(self.cost_microdollars),
        }


def _empty_usage_totals() -> dict[str, int]:
    return {
        "tool_call_attempts": 0,
        "tool_calls_ok": 0,
        "tool_calls_failed": 0,
        "provider_call_attempts": 0,
        "provider_calls_ok": 0,
        "provider_calls_failed": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "cost_microdollars": 0,
    }


def _usage_totals_dict(totals: Mapping[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = dict(totals)
    result["cost_usd"] = _microdollars_to_usd_string(totals["cost_microdollars"])
    return result


class SectorLaneUsageLedger:
    """In-memory exact ledger; callers decide if and where to persist it."""

    def __init__(self, policy: SectorLaneBudgetPolicy) -> None:
        if not isinstance(policy, SectorLaneBudgetPolicy):
            raise TypeError("policy must be a SectorLaneBudgetPolicy")
        self._policy = policy
        self._tool_calls: list[ToolCallUsage] = []
        self._provider_calls: list[ProviderCallUsage] = []

    @property
    def policy(self) -> SectorLaneBudgetPolicy:
        return self._policy

    @property
    def tool_calls(self) -> tuple[ToolCallUsage, ...]:
        return tuple(self._tool_calls)

    @property
    def provider_calls(self) -> tuple[ProviderCallUsage, ...]:
        return tuple(self._provider_calls)

    def _lane_cost_microdollars(self, lane: CanonicalLane) -> int:
        return sum(call.cost_microdollars for call in self._tool_calls if call.lane == lane) + sum(
            call.cost_microdollars for call in self._provider_calls if call.lane == lane
        )

    def _assert_cost_capacity(self, lane: CanonicalLane, additional_cost: int) -> None:
        reserve = self._policy.lane_budget(lane)
        if self._lane_cost_microdollars(lane) + additional_cost > reserve.max_cost_microdollars:
            raise LaneBudgetExceededError(f"{lane} cost reserve exhausted")

    def record_tool_call(
        self,
        *,
        call_id: str,
        lane: CanonicalLane | str,
        tool_name: str,
        status: str,
        cost_usd: MoneyInput = Decimal("0"),
    ) -> ToolCallUsage:
        canonical = _canonical_lane(lane)
        normalized_call_id = _non_empty(call_id, field_name="call_id")
        if any(call.call_id == normalized_call_id for call in self._tool_calls):
            raise ValueError(f"duplicate tool call_id: {normalized_call_id}")
        reserve = self._policy.lane_budget(canonical)
        attempts = sum(1 for call in self._tool_calls if call.lane == canonical)
        if attempts >= reserve.max_tool_calls:
            raise LaneBudgetExceededError(f"{canonical} tool-call reserve exhausted")
        cost_microdollars = _usd_to_microdollars(cost_usd, field_name="cost_usd")
        self._assert_cost_capacity(canonical, cost_microdollars)
        record = ToolCallUsage(
            call_id=normalized_call_id,
            lane=canonical,
            tool_name=_non_empty(tool_name, field_name="tool_name"),
            status=_non_empty(status, field_name="status").upper(),
            cost_microdollars=cost_microdollars,
        )
        self._tool_calls.append(record)
        return record

    def record_provider_call(
        self,
        *,
        call_id: str,
        lane: CanonicalLane | str,
        provider: str,
        model: str,
        status: str,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
        cost_usd: MoneyInput,
    ) -> ProviderCallUsage:
        canonical = _canonical_lane(lane)
        normalized_call_id = _non_empty(call_id, field_name="call_id")
        if any(call.call_id == normalized_call_id for call in self._provider_calls):
            raise ValueError(f"duplicate provider call_id: {normalized_call_id}")
        reserve = self._policy.lane_budget(canonical)
        attempts = sum(1 for call in self._provider_calls if call.lane == canonical)
        if attempts >= reserve.max_turns:
            raise LaneBudgetExceededError(f"{canonical} turn reserve exhausted")
        normalized_input_tokens = _non_negative_int(input_tokens, field_name="input_tokens")
        normalized_cached_tokens = _non_negative_int(
            cached_input_tokens,
            field_name="cached_input_tokens",
        )
        if normalized_cached_tokens > normalized_input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        normalized_output_tokens = _non_negative_int(
            output_tokens,
            field_name="output_tokens",
        )
        cost_microdollars = _usd_to_microdollars(cost_usd, field_name="cost_usd")
        self._assert_cost_capacity(canonical, cost_microdollars)
        record = ProviderCallUsage(
            call_id=normalized_call_id,
            lane=canonical,
            provider=_non_empty(provider, field_name="provider"),
            model=_non_empty(model, field_name="model"),
            status=_non_empty(status, field_name="status").upper(),
            input_tokens=normalized_input_tokens,
            cached_input_tokens=normalized_cached_tokens,
            output_tokens=normalized_output_tokens,
            cost_microdollars=cost_microdollars,
        )
        self._provider_calls.append(record)
        return record

    def remaining(self, lane: CanonicalLane | str) -> dict[str, Any]:
        canonical = _canonical_lane(lane)
        reserve = self._policy.lane_budget(canonical)
        tool_attempts = sum(1 for call in self._tool_calls if call.lane == canonical)
        provider_attempts = sum(1 for call in self._provider_calls if call.lane == canonical)
        cost_remaining = reserve.max_cost_microdollars - self._lane_cost_microdollars(canonical)
        return {
            "tool_calls": reserve.max_tool_calls - tool_attempts,
            "turns": reserve.max_turns - provider_attempts,
            "cost_microdollars": cost_remaining,
            "cost_usd": _microdollars_to_usd_string(cost_remaining),
        }

    def summary(self) -> dict[str, Any]:
        per_lane = {lane: _empty_usage_totals() for lane in CANONICAL_LANES}
        for call in self._tool_calls:
            totals = per_lane[call.lane]
            totals["tool_call_attempts"] += 1
            if call.status == "OK":
                totals["tool_calls_ok"] += 1
            else:
                totals["tool_calls_failed"] += 1
            totals["cost_microdollars"] += call.cost_microdollars
        for call in self._provider_calls:
            totals = per_lane[call.lane]
            totals["provider_call_attempts"] += 1
            if call.status == "OK":
                totals["provider_calls_ok"] += 1
            else:
                totals["provider_calls_failed"] += 1
            totals["input_tokens"] += call.input_tokens
            totals["cached_input_tokens"] += call.cached_input_tokens
            totals["output_tokens"] += call.output_tokens
            totals["cost_microdollars"] += call.cost_microdollars

        aggregate = _empty_usage_totals()
        for totals in per_lane.values():
            for field_name in aggregate:
                aggregate[field_name] += totals[field_name]
        return {
            "currency": "USD",
            "cost_unit": "microdollars",
            "lanes": {lane: _usage_totals_dict(per_lane[lane]) for lane in CANONICAL_LANES},
            "aggregate": _usage_totals_dict(aggregate),
            "aggregate_reconciles": True,
        }

    def reconcile(self, declared_summary: Mapping[str, Any]) -> None:
        if dict(declared_summary) != self.summary():
            raise LaneAccountingDriftError(
                "declared lane accounting does not reconcile with exact call records"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": USAGE_LEDGER_ARTIFACT_TYPE,
            "policy": self._policy.to_dict(),
            "tool_calls": [call.to_dict() for call in self._tool_calls],
            "provider_calls": [call.to_dict() for call in self._provider_calls],
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(
        cls,
        policy: SectorLaneBudgetPolicy,
        data: Mapping[str, Any],
    ) -> "SectorLaneUsageLedger":
        if data.get("artifact_type") != USAGE_LEDGER_ARTIFACT_TYPE:
            raise ValueError("invalid sector lane usage ledger artifact_type")
        if data.get("policy") != policy.to_dict():
            raise LaneAccountingDriftError("usage ledger policy does not match active policy")
        raw_tools = data.get("tool_calls")
        raw_providers = data.get("provider_calls")
        if not isinstance(raw_tools, list) or not isinstance(raw_providers, list):
            raise ValueError("usage ledger calls must be lists")
        ledger = cls(policy)
        for index, raw in enumerate(raw_tools):
            if not isinstance(raw, Mapping):
                raise ValueError("tool call usage must be an object")
            cost_microdollars = _payload_cost_microdollars(
                raw,
                field_prefix=f"tool_calls[{index}]",
            )
            record = ledger.record_tool_call(
                call_id=raw.get("call_id"),
                lane=raw.get("lane"),
                tool_name=raw.get("tool_name"),
                status=raw.get("status"),
                cost_usd=raw.get("cost_usd"),
            )
            if record.cost_microdollars != cost_microdollars:
                raise LaneAccountingDriftError("tool call cost fields do not reconcile")
        for index, raw in enumerate(raw_providers):
            if not isinstance(raw, Mapping):
                raise ValueError("provider call usage must be an object")
            cost_microdollars = _payload_cost_microdollars(
                raw,
                field_prefix=f"provider_calls[{index}]",
            )
            record = ledger.record_provider_call(
                call_id=raw.get("call_id"),
                lane=raw.get("lane"),
                provider=raw.get("provider"),
                model=raw.get("model"),
                status=raw.get("status"),
                input_tokens=raw.get("input_tokens"),
                cached_input_tokens=raw.get("cached_input_tokens"),
                output_tokens=raw.get("output_tokens"),
                cost_usd=raw.get("cost_usd"),
            )
            if record.cost_microdollars != cost_microdollars:
                raise LaneAccountingDriftError("provider call cost fields do not reconcile")
        declared_summary = data.get("summary")
        if not isinstance(declared_summary, Mapping):
            raise ValueError("usage ledger summary must be an object")
        ledger.reconcile(declared_summary)
        return ledger


@dataclass(frozen=True, slots=True)
class WholeRunLaneWorstCase:
    lane: CanonicalLane
    scope: Literal["whole_run", "per_sector"]
    multiplier: int
    tool_call_attempts: int
    provider_call_attempts: int
    cost_microdollars: int

    def __post_init__(self) -> None:
        canonical = _canonical_lane(self.lane)
        expected_scope = "per_sector" if canonical in PER_SECTOR_LANES else "whole_run"
        if self.scope != expected_scope:
            raise ValueError(f"{canonical} must use {expected_scope} scope")
        _positive_int(self.multiplier, field_name="multiplier")
        _non_negative_int(self.tool_call_attempts, field_name="tool_call_attempts")
        _non_negative_int(self.provider_call_attempts, field_name="provider_call_attempts")
        _non_negative_int(self.cost_microdollars, field_name="cost_microdollars")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "multiplier": self.multiplier,
            "tool_call_attempts": self.tool_call_attempts,
            "provider_call_attempts": self.provider_call_attempts,
            "cost_microdollars": self.cost_microdollars,
            "cost_usd": _microdollars_to_usd_string(self.cost_microdollars),
        }


@dataclass(frozen=True, slots=True)
class WholeRunWorstCaseEstimate:
    sector_count: int
    lanes: tuple[WholeRunLaneWorstCase, ...]

    def __post_init__(self) -> None:
        sectors = _positive_int(self.sector_count, field_name="sector_count")
        if tuple(item.lane for item in self.lanes) != CANONICAL_LANES:
            raise ValueError("whole-run estimate must contain canonical lanes in canonical order")
        for item in self.lanes:
            expected_multiplier = sectors if item.lane in PER_SECTOR_LANES else 1
            if item.multiplier != expected_multiplier:
                raise ValueError(f"{item.lane} multiplier does not match its scope")

    @property
    def tool_call_attempts(self) -> int:
        return sum(item.tool_call_attempts for item in self.lanes)

    @property
    def provider_call_attempts(self) -> int:
        return sum(item.provider_call_attempts for item in self.lanes)

    @property
    def cost_microdollars(self) -> int:
        return sum(item.cost_microdollars for item in self.lanes)

    def to_dict(self) -> dict[str, Any]:
        tool_call_attempts = self.tool_call_attempts
        provider_call_attempts = self.provider_call_attempts
        cost_microdollars = self.cost_microdollars
        return {
            "sector_count": self.sector_count,
            "lanes": {item.lane: item.to_dict() for item in self.lanes},
            "aggregate": {
                "tool_call_attempts": tool_call_attempts,
                "provider_call_attempts": provider_call_attempts,
                "cost_microdollars": cost_microdollars,
                "cost_usd": _microdollars_to_usd_string(cost_microdollars),
            },
            "aggregate_reconciles": (
                tool_call_attempts == sum(item.tool_call_attempts for item in self.lanes)
                and provider_call_attempts
                == sum(item.provider_call_attempts for item in self.lanes)
                and cost_microdollars == sum(item.cost_microdollars for item in self.lanes)
            ),
        }


@dataclass(frozen=True, slots=True)
class WholeRunCostAuthorization:
    estimate: WholeRunWorstCaseEstimate
    ceiling_microdollars: int
    status: Literal["AUTHORIZED"] = "AUTHORIZED"
    artifact_type: str = WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE

    def __post_init__(self) -> None:
        ceiling = _non_negative_int(
            self.ceiling_microdollars,
            field_name="ceiling_microdollars",
        )
        if ceiling % CENT_MICRODOLLARS != 0:
            raise ValueError("whole-run authorization ceiling must be cent-safe")
        if ceiling > MAX_AUTHORIZED_WHOLE_RUN_MICRODOLLARS:
            raise WholeRunAuthorizationError(
                "whole-run authorization ceiling cannot exceed $100.00"
            )
        if self.estimate.cost_microdollars > ceiling:
            raise WholeRunAuthorizationError(
                "whole-run worst-case cost exceeds the authorized ceiling"
            )
        if self.status != "AUTHORIZED":
            raise ValueError("whole-run authorization status must be AUTHORIZED")
        if self.artifact_type != WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE:
            raise ValueError("invalid whole-run authorization artifact_type")

    def to_dict(self) -> dict[str, Any]:
        estimate = self.estimate.to_dict()
        return {
            "artifact_type": self.artifact_type,
            "status": self.status,
            "sector_count": self.estimate.sector_count,
            "ceiling_cents": self.ceiling_microdollars // CENT_MICRODOLLARS,
            "ceiling_microdollars": self.ceiling_microdollars,
            "ceiling_usd": _microdollars_to_usd_string(self.ceiling_microdollars),
            "lane_worst_cases": estimate["lanes"],
            "worst_case": estimate["aggregate"],
            "aggregate_reconciles": estimate["aggregate_reconciles"],
        }

    @classmethod
    def from_dict(
        cls,
        policy: SectorLaneBudgetPolicy,
        data: Mapping[str, Any],
    ) -> "WholeRunCostAuthorization":
        if data.get("artifact_type") != WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE:
            raise ValueError("invalid whole-run authorization artifact_type")
        if data.get("status") != "AUTHORIZED":
            raise ValueError("whole-run authorization status must be AUTHORIZED")
        try:
            authorization = authorize_whole_run_cost(
                policy,
                sector_count=data.get("sector_count"),
                ceiling_usd=data.get("ceiling_usd"),
            )
        except (TypeError, ValueError, WholeRunAuthorizationError) as exc:
            raise LaneAccountingDriftError(
                "whole-run authorization contains invalid budget values"
            ) from exc
        if dict(data) != authorization.to_dict():
            raise LaneAccountingDriftError(
                "whole-run authorization does not reconcile with its policy"
            )
        return authorization


def estimate_whole_run_worst_case(
    policy: SectorLaneBudgetPolicy,
    *,
    sector_count: int,
) -> WholeRunWorstCaseEstimate:
    """Expand per-sector reserves and retain whole-run lanes exactly once."""

    sectors = _positive_int(sector_count, field_name="sector_count")
    lanes: list[WholeRunLaneWorstCase] = []
    for lane in CANONICAL_LANES:
        budget = policy.lane_budget(lane)
        per_sector = lane in PER_SECTOR_LANES
        multiplier = sectors if per_sector else 1
        lanes.append(
            WholeRunLaneWorstCase(
                lane=lane,
                scope="per_sector" if per_sector else "whole_run",
                multiplier=multiplier,
                tool_call_attempts=budget.max_tool_calls * multiplier,
                provider_call_attempts=budget.max_turns * multiplier,
                cost_microdollars=budget.max_cost_microdollars * multiplier,
            )
        )
    return WholeRunWorstCaseEstimate(sector_count=sectors, lanes=tuple(lanes))


def authorize_whole_run_cost(
    policy: SectorLaneBudgetPolicy,
    *,
    sector_count: int,
    ceiling_usd: MoneyInput = MAX_AUTHORIZED_WHOLE_RUN_USD,
) -> WholeRunCostAuthorization:
    """Authorize only when the exact worst case fits a cent-safe ceiling at or below $100."""

    ceiling = _exact_usd_decimal(
        ceiling_usd,
        field_name="ceiling_usd",
        quantum=USD_CENT_QUANTUM,
    )
    ceiling_microdollars = int(ceiling * MICRODOLLARS_PER_DOLLAR)
    if ceiling_microdollars > MAX_AUTHORIZED_WHOLE_RUN_MICRODOLLARS:
        raise WholeRunAuthorizationError("whole-run authorization ceiling cannot exceed $100.00")
    estimate = estimate_whole_run_worst_case(policy, sector_count=sector_count)
    if estimate.cost_microdollars > ceiling_microdollars:
        raise WholeRunAuthorizationError("whole-run worst-case cost exceeds the authorized ceiling")
    return WholeRunCostAuthorization(
        estimate=estimate,
        ceiling_microdollars=ceiling_microdollars,
    )


__all__ = [
    "CANONICAL_LANES",
    "CENT_MICRODOLLARS",
    "CanonicalLane",
    "CompanyChildBudget",
    "EXTENDED_COMPANY_CHILD_BUDGET",
    "INITIAL_COMPANY_CHILD_BUDGET",
    "LaneAccountingDriftError",
    "LaneBudget",
    "LaneBudgetExceededError",
    "MAX_AUTHORIZED_WHOLE_RUN_MICRODOLLARS",
    "MAX_AUTHORIZED_WHOLE_RUN_USD",
    "MICRODOLLARS_PER_DOLLAR",
    "PER_SECTOR_LANES",
    "ProviderCallUsage",
    "SectorLaneBudgetPolicy",
    "SectorLaneUsageLedger",
    "ToolCallUsage",
    "WHOLE_RUN_LANES",
    "WholeRunAuthorizationError",
    "WholeRunCostAuthorization",
    "WholeRunLaneWorstCase",
    "WholeRunWorstCaseEstimate",
    "authorize_whole_run_cost",
    "estimate_whole_run_worst_case",
]
