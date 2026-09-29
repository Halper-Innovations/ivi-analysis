from copy import deepcopy

import pytest

from app.autonomous.all_sector_cost_preflight import (
    AllSectorCostPreflight,
    AllSectorCostPreflightInput,
    CostPreflightDriftError,
    LaneCostConfiguration,
    SectorCandidateCount,
    build_diagnostic_v2_model_reprice,
    build_production_v2_cost_preflight,
    estimate_all_sector_cost_preflight,
)


SECTOR_CANDIDATES = (
    SectorCandidateCount("sector_01", 1),
    SectorCandidateCount("sector_02", 2),
    SectorCandidateCount("sector_03", 3),
    SectorCandidateCount("sector_04", 4),
    SectorCandidateCount("sector_05", 5),
    SectorCandidateCount("sector_06", 6),
    SectorCandidateCount("sector_07", 7),
    SectorCandidateCount("sector_08", 8),
    SectorCandidateCount("sector_09", 9),
    SectorCandidateCount("sector_10", 10),
    SectorCandidateCount("sector_11", 11),
    SectorCandidateCount("sector_12", 12),
    SectorCandidateCount("sector_13", 13),
    SectorCandidateCount("sector_14", 14),
    SectorCandidateCount("sector_15", 15),
    SectorCandidateCount("sector_16", 16),
    SectorCandidateCount("sector_17", 17),
    SectorCandidateCount("sector_18", 18),
    SectorCandidateCount("sector_19", 19),
    SectorCandidateCount("sector_20", 20),
    SectorCandidateCount("sector_21", 21),
    SectorCandidateCount("sector_22", 22),
)


def _lane(
    lane: str,
    *,
    input_tokens: int = 0,
    cached_tokens: int = 0,
    output_tokens: int = 0,
    fixed: int = 0,
    per_sector: int = 0,
    per_candidate: int = 0,
    per_terminal_attempt: int = 0,
    search_per_terminal_attempt: int = 0,
    search_cost: str = "0.000000",
) -> LaneCostConfiguration:
    return LaneCostConfiguration(
        lane=lane,
        provider_name="openai",
        model="gpt-5.5",
        input_tokens_per_model_call=input_tokens,
        cached_input_tokens_per_model_call=cached_tokens,
        output_tokens_per_model_call=output_tokens,
        fixed_model_calls=fixed,
        model_calls_per_sector=per_sector,
        model_calls_per_candidate=per_candidate,
        model_calls_per_terminal_cap_attempt=per_terminal_attempt,
        search_calls_per_terminal_cap_attempt=search_per_terminal_attempt,
        search_cost_usd_per_call=search_cost,
    )


def _normal_request() -> AllSectorCostPreflightInput:
    return AllSectorCostPreflightInput(
        pipeline_version="v2",
        sector_candidates=SECTOR_CANDIDATES,
        lane_configurations=(
            _lane("provider_preflight", input_tokens=1_000, fixed=1),
            _lane("parent_research", input_tokens=1_000, per_sector=1),
            _lane(
                "company_underwriting",
                input_tokens=1_000,
                cached_tokens=1_000,
                per_candidate=1,
            ),
            _lane(
                "selected_company_validation",
                output_tokens=1_000,
                per_sector=1,
            ),
            _lane("repair_fallback", input_tokens=1_000, fixed=1),
            _lane(
                "terminal_cap_search",
                input_tokens=1_000,
                per_terminal_attempt=1,
                search_per_terminal_attempt=4,
                search_cost="0.010000",
            ),
        ),
        terminal_cap_search_attempts=4,
        authorization_ceiling_usd="100.00",
    )


def test_explicit_22_sector_input_round_trips_without_count_or_lane_drift() -> None:
    request = _normal_request()
    payload = request.to_dict()

    assert request.sector_count == 22
    assert request.candidate_count == 253
    assert payload["pipeline_version"] == "v2"
    assert payload["sector_candidates"][0] == {
        "sector": "sector_01",
        "candidate_count": 1,
    }
    assert payload["sector_candidates"][21] == {
        "sector": "sector_22",
        "candidate_count": 22,
    }
    assert [row["lane"] for row in payload["lane_configurations"]] == [
        "provider_preflight",
        "parent_research",
        "company_underwriting",
        "selected_company_validation",
        "repair_fallback",
        "terminal_cap_search",
    ]
    assert payload["authorization_ceiling_cents"] == 10_000
    assert payload["authorization_ceiling_microdollars"] == 100_000_000
    assert payload["authorization_ceiling_usd"] == "100.000000"
    assert AllSectorCostPreflightInput.from_dict(payload) == request


def test_gpt_5_5_cached_lane_terminal_search_and_aggregate_reconcile_exactly() -> None:
    result = estimate_all_sector_cost_preflight(_normal_request())
    payload = result.to_dict()

    assert payload["status"] == "AUTHORIZED"
    assert payload["spend_authorized"] is True
    assert payload["reason_codes"] == []
    assert payload["lane_costs"]["company_underwriting"] == {
        "model": "gpt-5.5",
        "model_calls": 253,
        "search_calls": 0,
        "input_tokens": 253_000,
        "cached_input_tokens": 253_000,
        "output_tokens": 0,
        "model_cost_microdollars": 126_500,
        "model_cost_usd": "0.126500",
        "search_cost_microdollars": 0,
        "search_cost_usd": "0.000000",
        "cost_microdollars": 126_500,
        "cost_usd": "0.126500",
    }
    assert payload["lane_costs"]["terminal_cap_search"] == {
        "model": "gpt-5.5",
        "model_calls": 4,
        "search_calls": 16,
        "input_tokens": 4_000,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "model_cost_microdollars": 20_000,
        "model_cost_usd": "0.020000",
        "search_cost_microdollars": 160_000,
        "search_cost_usd": "0.160000",
        "cost_microdollars": 180_000,
        "cost_usd": "0.180000",
    }
    assert payload["aggregate"] == {
        "sector_count": 22,
        "candidate_count": 253,
        "model_calls": 303,
        "search_calls": 16,
        "input_tokens": 281_000,
        "cached_input_tokens": 253_000,
        "output_tokens": 22_000,
        "model_cost_microdollars": 926_500,
        "model_cost_usd": "0.926500",
        "search_cost_microdollars": 160_000,
        "search_cost_usd": "0.160000",
        "lane_sum_microdollars": 1_086_500,
        "remaining_worst_case_cost_microdollars": 1_086_500,
        "remaining_worst_case_cost_usd": "1.086500",
        "prior_realized_cost_microdollars": 0,
        "prior_realized_cost_usd": "0.000000",
        "cost_microdollars": 1_086_500,
        "cost_usd": "1.086500",
    }
    assert payload["aggregate_reconciles"] is True
    assert AllSectorCostPreflight.from_dict(payload) == result

    drifted = deepcopy(payload)
    drifted["aggregate"]["lane_sum_microdollars"] = 1_086_501
    with pytest.raises(CostPreflightDriftError, match="does not reconcile"):
        AllSectorCostPreflight.from_dict(drifted)


def test_ten_long_context_terminal_attempts_stop_before_spend_above_100() -> None:
    request = AllSectorCostPreflightInput(
        pipeline_version="v2",
        sector_candidates=SECTOR_CANDIDATES,
        lane_configurations=(
            _lane("provider_preflight"),
            _lane("parent_research"),
            _lane("company_underwriting"),
            _lane("selected_company_validation"),
            _lane("repair_fallback"),
            _lane(
                "terminal_cap_search",
                input_tokens=1_048_800,
                output_tokens=1_200,
                per_terminal_attempt=1,
                search_per_terminal_attempt=4,
                search_cost="0.010000",
            ),
        ),
        terminal_cap_search_attempts=10,
        authorization_ceiling_usd="100.00",
    )

    payload = estimate_all_sector_cost_preflight(request).to_dict()

    assert payload["status"] == "STOP_BEFORE_SPEND"
    assert payload["spend_authorized"] is False
    assert payload["reason_codes"] == ["WHOLE_RUN_WORST_CASE_EXCEEDS_AUTHORIZED_CEILING"]
    assert payload["lane_costs"]["terminal_cap_search"] == {
        "model": "gpt-5.5",
        "model_calls": 10,
        "search_calls": 40,
        "input_tokens": 10_488_000,
        "cached_input_tokens": 0,
        "output_tokens": 12_000,
        "model_cost_microdollars": 105_420_000,
        "model_cost_usd": "105.420000",
        "search_cost_microdollars": 400_000,
        "search_cost_usd": "0.400000",
        "cost_microdollars": 105_820_000,
        "cost_usd": "105.820000",
    }
    assert payload["aggregate"]["lane_sum_microdollars"] == 105_820_000
    assert payload["aggregate"]["cost_microdollars"] == 105_820_000
    assert payload["aggregate_reconciles"] is True


def test_preflight_is_offline_and_never_calls_provider_or_network(monkeypatch) -> None:
    crossings: list[str] = []

    def forbidden_provider(*args, **kwargs):
        crossings.append("provider")
        raise AssertionError("provider call forbidden in preflight")

    def forbidden_network(*args, **kwargs):
        crossings.append("network")
        raise AssertionError("network call forbidden in preflight")

    monkeypatch.setattr(
        "app.llm.providers.openai_provider.OpenAIProvider.synthesize_json",
        forbidden_provider,
    )
    monkeypatch.setattr("requests.sessions.Session.request", forbidden_network)

    result = estimate_all_sector_cost_preflight(_normal_request())

    assert result.status == "AUTHORIZED"
    assert crossings == []


def test_preflight_rejects_non_v2_non_gpt_5_5_and_ceiling_above_100() -> None:
    with pytest.raises(ValueError, match="pipeline v2"):
        AllSectorCostPreflightInput(
            pipeline_version="v1",
            sector_candidates=SECTOR_CANDIDATES,
            lane_configurations=_normal_request().lane_configurations,
            terminal_cap_search_attempts=0,
        )
    with pytest.raises(ValueError, match="model must be GPT-5.5"):
        LaneCostConfiguration(
            lane="parent_research",
            provider_name="openai",
            model="gpt-5.4",
            input_tokens_per_model_call=1_000,
            cached_input_tokens_per_model_call=0,
            output_tokens_per_model_call=0,
        )
    with pytest.raises(ValueError, match=r"cannot exceed \$100.00"):
        AllSectorCostPreflightInput(
            pipeline_version="v2",
            sector_candidates=SECTOR_CANDIDATES,
            lane_configurations=_normal_request().lane_configurations,
            terminal_cap_search_attempts=0,
            authorization_ceiling_usd="100.01",
        )


def test_production_preflight_is_uncached_and_adds_prior_resume_spend() -> None:
    authorized = build_production_v2_cost_preflight(
        sector_candidate_counts={"energy": 1},
        terminal_cap_search_attempts=0,
        parent_max_turns=1,
        prior_realized_cost_usd="70.000000",
    ).to_dict()
    stopped = build_production_v2_cost_preflight(
        sector_candidate_counts={"energy": 1},
        terminal_cap_search_attempts=0,
        parent_max_turns=1,
        prior_realized_cost_usd="74.000000",
    ).to_dict()

    assert authorized["status"] == "AUTHORIZED"
    assert authorized["aggregate"]["cached_input_tokens"] == 0
    assert authorized["aggregate"]["remaining_worst_case_cost_usd"] == "27.000000"
    assert authorized["aggregate"]["prior_realized_cost_usd"] == "70.000000"
    assert authorized["aggregate"]["cost_usd"] == "97.000000"
    assert stopped["status"] == "STOP_BEFORE_SPEND"
    assert stopped["aggregate"]["prior_realized_cost_usd"] == "74.000000"
    assert stopped["aggregate"]["cost_usd"] == "101.000000"


def test_production_22_sector_worst_case_stops_before_provider_spend() -> None:
    payload = build_production_v2_cost_preflight(
        sector_candidate_counts={f"sector_{index:02d}": 1 for index in range(1, 23)},
        terminal_cap_search_attempts=0,
        parent_max_turns=6,
    ).to_dict()

    assert payload["status"] == "STOP_BEFORE_SPEND"
    assert payload["spend_authorized"] is False
    assert payload["aggregate"]["sector_count"] == 22
    assert payload["aggregate"]["candidate_count"] == 22
    assert payload["aggregate"]["cached_input_tokens"] == 0
    assert payload["aggregate"]["cost_usd"] == "727.500000"


def test_one_sector_one_name_gpt_5_4_mini_reprice_is_diagnostic_and_offline(
    monkeypatch,
) -> None:
    crossings: list[str] = []

    def forbidden_provider(*args, **kwargs):
        crossings.append("provider")
        raise AssertionError("provider call forbidden in diagnostic reprice")

    def forbidden_network(*args, **kwargs):
        crossings.append("network")
        raise AssertionError("network call forbidden in diagnostic reprice")

    monkeypatch.setattr(
        "app.llm.providers.openai_provider.OpenAIProvider.synthesize_json",
        forbidden_provider,
    )
    monkeypatch.setattr("requests.sessions.Session.request", forbidden_network)
    production = build_production_v2_cost_preflight(
        sector_candidate_counts={"consumer_services": 1},
        terminal_cap_search_attempts=0,
        parent_max_turns=6,
    )

    payload = build_diagnostic_v2_model_reprice(
        production,
        target_model="gpt-5.4-mini",
    )

    assert crossings == []
    assert payload["status"] == "DIAGNOSTIC_ONLY"
    assert payload["spend_authorized"] is False
    assert payload["execution_requested"] is False
    assert payload["execution_binding_unchanged"] is True
    assert payload["execution_compatible_with_current_v2_policy"] is False
    assert payload["reason_codes"] == [
        "DIAGNOSTIC_REPRICE_CANNOT_AUTHORIZE_EXECUTION"
    ]
    assert payload["target_pricing_binding"] == {
        "provider": "openai",
        "model": "gpt-5.4-mini",
        "service_tier": "standard",
    }
    assert payload["rate_card"]["input_usd_per_million_tokens"] == "0.750000"
    assert payload["rate_card"]["cached_input_usd_per_million_tokens"] == (
        "0.075000"
    )
    assert payload["rate_card"]["output_usd_per_million_tokens"] == "4.500000"
    assert payload["rate_card"]["verified_on"] == "2026-07-18"
    aggregate = payload["cost_envelope"]["aggregate"]
    assert aggregate["sector_count"] == 1
    assert aggregate["candidate_count"] == 1
    assert aggregate["model_calls"] == 46
    assert aggregate["search_calls"] == 0
    assert aggregate["input_tokens"] == 5_520_000
    assert aggregate["cached_input_tokens"] == 0
    assert aggregate["output_tokens"] == 230_000
    assert aggregate["remaining_worst_case_cost_usd"] == "5.175000"
    assert payload["comparison"] == {
        "source_remaining_worst_case_cost_microdollars": 34_500_000,
        "source_remaining_worst_case_cost_usd": "34.500000",
        "diagnostic_remaining_worst_case_cost_microdollars": 5_175_000,
        "diagnostic_remaining_worst_case_cost_usd": "5.175000",
        "estimated_savings_microdollars": 29_325_000,
        "estimated_savings_usd": "29.325000",
        "diagnostic_cost_as_pct_of_source": "15.000000",
        "estimated_savings_pct": "85.000000",
    }
    assert payload["actual_usage"] == {
        "model_calls": 0,
        "search_calls": 0,
        "network_calls": 0,
        "cost_microdollars": 0,
        "cost_usd": "0.000000",
    }


def test_current_34_sector_700_name_gpt_5_4_mini_reprice_is_exact() -> None:
    sector_counts = {"sector_01": 667}
    sector_counts.update(
        {f"sector_{index:02d}": 1 for index in range(2, 35)}
    )
    production = build_production_v2_cost_preflight(
        sector_candidate_counts=sector_counts,
        terminal_cap_search_attempts=0,
        parent_max_turns=6,
    )

    payload = build_diagnostic_v2_model_reprice(
        production,
        target_model="gpt-5.4-mini",
    )

    aggregate = payload["cost_envelope"]["aggregate"]
    assert aggregate["sector_count"] == 34
    assert aggregate["candidate_count"] == 700
    assert aggregate["model_calls"] == 10_822
    assert aggregate["search_calls"] == 0
    assert aggregate["input_tokens"] == 1_298_640_000
    assert aggregate["cached_input_tokens"] == 0
    assert aggregate["output_tokens"] == 54_110_000
    assert aggregate["model_cost_usd"] == "1217.475000"
    assert aggregate["search_cost_usd"] == "0.000000"
    assert aggregate["remaining_worst_case_cost_usd"] == "1217.475000"
    assert payload["cost_envelope"]["aggregate_reconciles"] is True
    assert payload["comparison"]["source_remaining_worst_case_cost_usd"] == (
        "8116.500000"
    )
    assert payload["comparison"]["estimated_savings_usd"] == "6899.025000"
    assert payload["comparison"]["diagnostic_cost_as_pct_of_source"] == (
        "15.000000"
    )
    assert payload["comparison"]["estimated_savings_pct"] == "85.000000"
    assert payload["cost_envelope"]["lane_costs"]["provider_preflight"][
        "cost_usd"
    ] == "0.225000"
    assert payload["cost_envelope"]["lane_costs"]["parent_research"][
        "cost_usd"
    ] == "234.000000"
    assert payload["cost_envelope"]["lane_costs"]["company_underwriting"][
        "cost_usd"
    ] == "472.500000"
    assert payload["cost_envelope"]["lane_costs"][
        "selected_company_validation"
    ]["cost_usd"] == "22.950000"
    assert payload["cost_envelope"]["lane_costs"]["repair_fallback"][
        "cost_usd"
    ] == "487.800000"


def test_diagnostic_reprice_rejects_unknown_model_and_incompatible_context() -> None:
    production = build_production_v2_cost_preflight(
        sector_candidate_counts={"energy": 1},
        terminal_cap_search_attempts=0,
        parent_max_turns=6,
    )
    terminal_production = build_production_v2_cost_preflight(
        sector_candidate_counts={"energy": 1},
        terminal_cap_search_attempts=1,
        parent_max_turns=6,
    )

    with pytest.raises(ValueError, match="diagnostic reprice model must be one of"):
        build_diagnostic_v2_model_reprice(
            production,
            target_model="unpriced-model",
        )
    with pytest.raises(ValueError, match="per-call token bound exceeds"):
        build_diagnostic_v2_model_reprice(
            terminal_production,
            target_model="gpt-5.4-mini",
        )
