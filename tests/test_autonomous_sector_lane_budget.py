from copy import deepcopy
from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from app.autonomous.sector_lane_budget import (
    CANONICAL_LANES,
    EXTENDED_COMPANY_CHILD_BUDGET,
    INITIAL_COMPANY_CHILD_BUDGET,
    LaneAccountingDriftError,
    LaneBudget,
    LaneBudgetExceededError,
    SectorLaneBudgetPolicy,
    SectorLaneUsageLedger,
    WholeRunCostAuthorization,
    WholeRunAuthorizationError,
    authorize_whole_run_cost,
    estimate_whole_run_worst_case,
)


def _policy() -> SectorLaneBudgetPolicy:
    return SectorLaneBudgetPolicy(
        provider_preflight=LaneBudget(0, 1, "0.500000"),
        parent_research=LaneBudget(2, 2, "1.000001"),
        company_underwriting=LaneBudget(4, 2, "2.000000"),
        selected_company_validation=LaneBudget(1, 1, "0.750000"),
        repair_fallback=LaneBudget(2, 1, "0.250000"),
        terminal_cap_search=LaneBudget(1, 1, "1.500000"),
    )


def test_policy_is_immutable_and_uses_exact_canonical_lanes_and_child_profiles() -> None:
    policy = _policy()

    assert CANONICAL_LANES == (
        "provider_preflight",
        "parent_research",
        "company_underwriting",
        "selected_company_validation",
        "repair_fallback",
        "terminal_cap_search",
    )
    assert INITIAL_COMPANY_CHILD_BUDGET.to_dict() == {
        "profile": "initial",
        "max_tool_calls": 4,
        "max_turns": 2,
    }
    assert EXTENDED_COMPANY_CHILD_BUDGET.to_dict() == {
        "profile": "extended",
        "max_tool_calls": 8,
        "max_turns": 3,
    }
    assert policy.company_child_budget("initial") == INITIAL_COMPANY_CHILD_BUDGET
    assert policy.company_child_budget("extended") == EXTENDED_COMPANY_CHILD_BUDGET
    assert policy.lane_budget("parent_research").to_dict() == {
        "max_tool_calls": 2,
        "max_turns": 2,
        "max_cost_microdollars": 1_000_001,
        "max_cost_usd": "1.000001",
    }
    with pytest.raises(FrozenInstanceError):
        policy.parent_research = LaneBudget(9, 9, "9.000000")
    with pytest.raises(FrozenInstanceError):
        policy.parent_research.max_turns = 9


def test_policy_round_trip_rejects_noncanonical_or_profile_drift() -> None:
    policy = _policy()
    payload = policy.to_dict()

    assert SectorLaneBudgetPolicy.from_dict(payload) == policy

    extra_lane = deepcopy(payload)
    extra_lane["lanes"]["other"] = {
        "max_tool_calls": 0,
        "max_turns": 0,
        "max_cost_microdollars": 0,
        "max_cost_usd": "0.000000",
    }
    with pytest.raises(ValueError, match="exactly the six canonical lanes"):
        SectorLaneBudgetPolicy.from_dict(extra_lane)

    profile_drift = deepcopy(payload)
    profile_drift["company_child_profiles"]["extended"]["max_tool_calls"] = 7
    with pytest.raises(ValueError, match="profiles have drifted"):
        SectorLaneBudgetPolicy.from_dict(profile_drift)

    with pytest.raises(ValueError, match="initial company child budget must be 4 tools/2 turns"):
        type(INITIAL_COMPANY_CHILD_BUDGET)("initial", 8, 3)


def test_failed_tool_attempts_consume_only_their_independent_lane_budget() -> None:
    ledger = SectorLaneUsageLedger(_policy())

    ledger.record_tool_call(
        call_id="P1",
        lane="parent_research",
        tool_name="fetch_filing",
        status="ERROR",
    )
    ledger.record_tool_call(
        call_id="P2",
        lane="parent_research",
        tool_name="fetch_filing",
        status="ERROR",
    )

    assert ledger.remaining("parent_research") == {
        "tool_calls": 0,
        "turns": 2,
        "cost_microdollars": 1_000_001,
        "cost_usd": "1.000001",
    }
    with pytest.raises(LaneBudgetExceededError, match="tool-call reserve exhausted"):
        ledger.record_tool_call(
            call_id="P3",
            lane="parent_research",
            tool_name="fetch_filing",
            status="ERROR",
        )

    ledger.record_tool_call(
        call_id="C1",
        lane="company_underwriting",
        tool_name="fetch_companyfacts",
        status="OK",
    )
    assert ledger.remaining("company_underwriting") == {
        "tool_calls": 3,
        "turns": 2,
        "cost_microdollars": 2_000_000,
        "cost_usd": "2.000000",
    }


def test_failed_provider_attempts_consume_turns_and_cost_checks_are_atomic() -> None:
    ledger = SectorLaneUsageLedger(_policy())

    ledger.record_tool_call(
        call_id="T1",
        lane="parent_research",
        tool_name="web_search",
        status="ERROR",
        cost_usd="0.700001",
    )
    ledger.record_provider_call(
        call_id="L1",
        lane="parent_research",
        provider="openai",
        model="gpt-test",
        status="ERROR",
        input_tokens=100,
        cached_input_tokens=20,
        output_tokens=10,
        cost_usd="0.300000",
    )

    assert ledger.remaining("parent_research") == {
        "tool_calls": 1,
        "turns": 1,
        "cost_microdollars": 0,
        "cost_usd": "0.000000",
    }
    with pytest.raises(LaneBudgetExceededError, match="cost reserve exhausted"):
        ledger.record_provider_call(
            call_id="L2",
            lane="parent_research",
            provider="openai",
            model="gpt-test",
            status="ERROR",
            input_tokens=1,
            cached_input_tokens=0,
            output_tokens=1,
            cost_usd="0.000001",
        )
    assert ledger.remaining("parent_research")["turns"] == 1

    zero_cost_ledger = SectorLaneUsageLedger(_policy())
    for call_id in ("L1", "L2"):
        zero_cost_ledger.record_provider_call(
            call_id=call_id,
            lane="parent_research",
            provider="openai",
            model="gpt-test",
            status="ERROR",
            input_tokens=0,
            cached_input_tokens=0,
            output_tokens=0,
            cost_usd="0.000000",
        )
    with pytest.raises(LaneBudgetExceededError, match="turn reserve exhausted"):
        zero_cost_ledger.record_provider_call(
            call_id="L3",
            lane="parent_research",
            provider="openai",
            model="gpt-test",
            status="ERROR",
            input_tokens=0,
            cached_input_tokens=0,
            output_tokens=0,
            cost_usd="0.000000",
        )


def test_usage_ledger_reconciles_all_lanes_tokens_attempts_and_microdollars_exactly() -> None:
    ledger = SectorLaneUsageLedger(_policy())
    ledger.record_tool_call(
        call_id="T1",
        lane="parent_research",
        tool_name="fetch_filing",
        status="OK",
        cost_usd="0.010001",
    )
    ledger.record_tool_call(
        call_id="T2",
        lane="company_underwriting",
        tool_name="fetch_companyfacts",
        status="ERROR",
        cost_usd="0.020002",
    )
    ledger.record_provider_call(
        call_id="L1",
        lane="parent_research",
        provider="openai",
        model="gpt-test",
        status="ERROR",
        input_tokens=100,
        cached_input_tokens=20,
        output_tokens=10,
        cost_usd="0.030003",
    )
    ledger.record_provider_call(
        call_id="L2",
        lane="selected_company_validation",
        provider="openai",
        model="gpt-test",
        status="OK",
        input_tokens=200,
        cached_input_tokens=50,
        output_tokens=30,
        cost_usd="0.040004",
    )

    zero = {
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
        "cost_usd": "0.000000",
    }
    assert ledger.summary() == {
        "currency": "USD",
        "cost_unit": "microdollars",
        "lanes": {
            "provider_preflight": zero,
            "parent_research": {
                "tool_call_attempts": 1,
                "tool_calls_ok": 1,
                "tool_calls_failed": 0,
                "provider_call_attempts": 1,
                "provider_calls_ok": 0,
                "provider_calls_failed": 1,
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "output_tokens": 10,
                "cost_microdollars": 40_004,
                "cost_usd": "0.040004",
            },
            "company_underwriting": {
                "tool_call_attempts": 1,
                "tool_calls_ok": 0,
                "tool_calls_failed": 1,
                "provider_call_attempts": 0,
                "provider_calls_ok": 0,
                "provider_calls_failed": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "cost_microdollars": 20_002,
                "cost_usd": "0.020002",
            },
            "selected_company_validation": {
                "tool_call_attempts": 0,
                "tool_calls_ok": 0,
                "tool_calls_failed": 0,
                "provider_call_attempts": 1,
                "provider_calls_ok": 1,
                "provider_calls_failed": 0,
                "input_tokens": 200,
                "cached_input_tokens": 50,
                "output_tokens": 30,
                "cost_microdollars": 40_004,
                "cost_usd": "0.040004",
            },
            "repair_fallback": zero,
            "terminal_cap_search": zero,
        },
        "aggregate": {
            "tool_call_attempts": 2,
            "tool_calls_ok": 1,
            "tool_calls_failed": 1,
            "provider_call_attempts": 2,
            "provider_calls_ok": 1,
            "provider_calls_failed": 1,
            "input_tokens": 300,
            "cached_input_tokens": 70,
            "output_tokens": 40,
            "cost_microdollars": 100_010,
            "cost_usd": "0.100010",
        },
        "aggregate_reconciles": True,
    }


def test_usage_ledger_round_trip_rejects_record_or_summary_drift() -> None:
    ledger = SectorLaneUsageLedger(_policy())
    ledger.record_tool_call(
        call_id="T1",
        lane="terminal_cap_search",
        tool_name="web_search",
        status="OK",
        cost_usd="0.010000",
    )
    ledger.record_provider_call(
        call_id="L1",
        lane="terminal_cap_search",
        provider="openai",
        model="gpt-5.5",
        status="OK",
        input_tokens=1_000,
        cached_input_tokens=250,
        output_tokens=100,
        cost_usd="0.020000",
    )
    payload = ledger.to_dict()

    rebuilt = SectorLaneUsageLedger.from_dict(_policy(), payload)
    assert rebuilt.to_dict() == payload

    record_drift = deepcopy(payload)
    record_drift["tool_calls"][0]["cost_microdollars"] = 10_001
    with pytest.raises(LaneAccountingDriftError, match="does not match"):
        SectorLaneUsageLedger.from_dict(_policy(), record_drift)

    summary_drift = deepcopy(payload)
    summary_drift["summary"]["aggregate"]["cost_microdollars"] = 30_001
    with pytest.raises(LaneAccountingDriftError, match="does not reconcile"):
        SectorLaneUsageLedger.from_dict(_policy(), summary_drift)


@pytest.mark.parametrize(
    "value",
    ["-0.000001", "NaN", "Infinity", "0.0000001", float("nan"), float("inf")],
)
def test_money_inputs_reject_negative_nonfinite_or_sub_microdollar_values(value: object) -> None:
    with pytest.raises(ValueError):
        LaneBudget(1, 1, value)


def test_counts_tokens_lanes_and_cent_ceiling_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LaneBudget(-1, 1, "0.000000")
    with pytest.raises(ValueError, match="lane must be one of"):
        SectorLaneUsageLedger(_policy()).record_tool_call(
            call_id="T1",
            lane="selected_validation",
            tool_name="fetch_filing",
            status="OK",
        )
    with pytest.raises(ValueError, match="cached_input_tokens cannot exceed"):
        SectorLaneUsageLedger(_policy()).record_provider_call(
            call_id="L1",
            lane="provider_preflight",
            provider="openai",
            model="gpt-test",
            status="OK",
            input_tokens=10,
            cached_input_tokens=11,
            output_tokens=1,
            cost_usd="0.010000",
        )
    with pytest.raises(ValueError, match="at most 2 decimal places"):
        authorize_whole_run_cost(_policy(), sector_count=1, ceiling_usd="100.001")


def test_whole_run_worst_case_multiplies_only_per_sector_lanes_exactly() -> None:
    estimate = estimate_whole_run_worst_case(_policy(), sector_count=2)

    assert estimate.to_dict() == {
        "sector_count": 2,
        "lanes": {
            "provider_preflight": {
                "scope": "whole_run",
                "multiplier": 1,
                "tool_call_attempts": 0,
                "provider_call_attempts": 1,
                "cost_microdollars": 500_000,
                "cost_usd": "0.500000",
            },
            "parent_research": {
                "scope": "per_sector",
                "multiplier": 2,
                "tool_call_attempts": 4,
                "provider_call_attempts": 4,
                "cost_microdollars": 2_000_002,
                "cost_usd": "2.000002",
            },
            "company_underwriting": {
                "scope": "per_sector",
                "multiplier": 2,
                "tool_call_attempts": 8,
                "provider_call_attempts": 4,
                "cost_microdollars": 4_000_000,
                "cost_usd": "4.000000",
            },
            "selected_company_validation": {
                "scope": "per_sector",
                "multiplier": 2,
                "tool_call_attempts": 2,
                "provider_call_attempts": 2,
                "cost_microdollars": 1_500_000,
                "cost_usd": "1.500000",
            },
            "repair_fallback": {
                "scope": "per_sector",
                "multiplier": 2,
                "tool_call_attempts": 4,
                "provider_call_attempts": 2,
                "cost_microdollars": 500_000,
                "cost_usd": "0.500000",
            },
            "terminal_cap_search": {
                "scope": "whole_run",
                "multiplier": 1,
                "tool_call_attempts": 1,
                "provider_call_attempts": 1,
                "cost_microdollars": 1_500_000,
                "cost_usd": "1.500000",
            },
        },
        "aggregate": {
            "tool_call_attempts": 19,
            "provider_call_attempts": 14,
            "cost_microdollars": 10_000_002,
            "cost_usd": "10.000002",
        },
        "aggregate_reconciles": True,
    }


def test_whole_run_authorization_is_cent_safe_and_stops_above_one_hundred_dollars() -> None:
    authorization = authorize_whole_run_cost(
        _policy(),
        sector_count=2,
        ceiling_usd=Decimal("100.00"),
    )
    payload = authorization.to_dict()
    assert payload["artifact_type"] == "autonomous_sector_lane_cost_preflight_v1"
    assert payload["status"] == "AUTHORIZED"
    assert payload["sector_count"] == 2
    assert payload["ceiling_cents"] == 10_000
    assert payload["ceiling_microdollars"] == 100_000_000
    assert payload["ceiling_usd"] == "100.000000"
    assert payload["lane_worst_cases"]["provider_preflight"] == {
        "scope": "whole_run",
        "multiplier": 1,
        "tool_call_attempts": 0,
        "provider_call_attempts": 1,
        "cost_microdollars": 500_000,
        "cost_usd": "0.500000",
    }
    assert payload["lane_worst_cases"]["terminal_cap_search"] == {
        "scope": "whole_run",
        "multiplier": 1,
        "tool_call_attempts": 1,
        "provider_call_attempts": 1,
        "cost_microdollars": 1_500_000,
        "cost_usd": "1.500000",
    }
    assert payload["worst_case"] == {
        "tool_call_attempts": 19,
        "provider_call_attempts": 14,
        "cost_microdollars": 10_000_002,
        "cost_usd": "10.000002",
    }
    assert payload["aggregate_reconciles"] is True
    assert WholeRunCostAuthorization.from_dict(_policy(), payload) == authorization

    preflight_drift = deepcopy(payload)
    preflight_drift["worst_case"]["cost_microdollars"] = 10_000_003
    with pytest.raises(LaneAccountingDriftError, match="does not reconcile"):
        WholeRunCostAuthorization.from_dict(_policy(), preflight_drift)

    exactly_one_hundred = SectorLaneBudgetPolicy(
        provider_preflight=LaneBudget(0, 0, "0.500000"),
        parent_research=LaneBudget(0, 0, "49.500000"),
        company_underwriting=LaneBudget(0, 0, "0.000000"),
        selected_company_validation=LaneBudget(0, 0, "0.000000"),
        repair_fallback=LaneBudget(0, 0, "0.000000"),
        terminal_cap_search=LaneBudget(0, 0, "0.500000"),
    )
    exact_authorization = authorize_whole_run_cost(exactly_one_hundred, sector_count=2)
    assert exact_authorization.to_dict()["worst_case"] == {
        "tool_call_attempts": 0,
        "provider_call_attempts": 0,
        "cost_microdollars": 100_000_000,
        "cost_usd": "100.000000",
    }

    above_one_hundred = SectorLaneBudgetPolicy(
        provider_preflight=LaneBudget(0, 0, "0.000000"),
        parent_research=LaneBudget(0, 0, "50.000001"),
        company_underwriting=LaneBudget(0, 0, "0.000000"),
        selected_company_validation=LaneBudget(0, 0, "0.000000"),
        repair_fallback=LaneBudget(0, 0, "0.000000"),
        terminal_cap_search=LaneBudget(0, 0, "0.000000"),
    )
    with pytest.raises(WholeRunAuthorizationError, match="worst-case cost exceeds"):
        authorize_whole_run_cost(above_one_hundred, sector_count=2)
    with pytest.raises(WholeRunAuthorizationError, match=r"cannot exceed \$100.00"):
        authorize_whole_run_cost(_policy(), sector_count=1, ceiling_usd="100.01")
