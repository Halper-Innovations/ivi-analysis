from __future__ import annotations

from app.autonomous.sector_contract import SectorCompanyFinancialPacket, SectorExpectedReturnScenario
from app.autonomous.sector_scenarios import (
    build_expected_return_scenarios,
    build_expected_return_scenarios_for_packets,
)


def _packet() -> SectorCompanyFinancialPacket:
    return SectorCompanyFinancialPacket(
        ticker="AAA",
        financial_status="Financially Viable",
        model_fit_status="VALID_GENERIC",
        data_quality_status="OK",
        current_price=50.0,
        business_quality={"revenue_cagr_5y": 0.10, "operating_margin": 0.21},
        capital_allocation={"dilution_rate_shares_cagr": 0.02},
        cash_conversion={"owner_earnings_per_share": 4.5},
        valuation={"anchor_method": "dcf", "valuation_anchor": 100.0},
        evidence_ref_ids=["packet:AAA"],
    )


def test_build_expected_return_scenarios_creates_downside_base_upside_cases():
    scenarios = build_expected_return_scenarios(_packet(), horizons=[5])

    assert [scenario.scenario_id for scenario in scenarios] == [
        "AAA_downside_5Y",
        "AAA_base_5Y",
        "AAA_upside_5Y",
    ]
    assert [scenario.scenario_name for scenario in scenarios] == ["downside", "base", "upside"]
    assert [scenario.estimated_future_value_per_share for scenario in scenarios] == [
        58.754852,
        146.932808,
        176.319369,
    ]
    assert [scenario.annualized_return for scenario in scenarios] == [
        0.032796,
        0.240594,
        0.286667,
    ]
    assert [scenario.revenue_cagr for scenario in scenarios] == [-0.02, 0.08, 0.08]
    assert [scenario.terminal_multiple for scenario in scenarios] == [0.65, 1.0, 1.2]
    assert scenarios[0].downside_value_per_share == 58.754852
    assert scenarios[1].downside_value_per_share is None
    assert scenarios[2].downside_value_per_share is None
    assert scenarios[0].normalized_operating_margin == 0.21
    assert scenarios[0].owner_earnings_per_share == 4.5
    assert scenarios[0].share_count_cagr == 0.02
    assert scenarios[0].evidence_ref_ids == ["packet:AAA"]


def test_scenario_assumptions_are_auditable_and_round_trip_through_contract():
    scenario = build_expected_return_scenarios(_packet(), horizons=[5])[1]

    assert scenario.assumptions == {
        "valuation_anchor": 100.0,
        "anchor_method": "dcf",
        "source_revenue_cagr": 0.1,
        "growth_source": "historical_revenue_cagr",
        "scenario_growth_rate": 0.08,
        "terminal_anchor_multiplier": 1.0,
        "share_count_cagr": 0.02,
        "formula": "valuation_anchor * terminal_anchor_multiplier * (1 + scenario_growth_rate) ** horizon_years",
    }
    assert scenario.key_sensitivities == [
        "valuation_anchor",
        "scenario_growth_rate",
        "terminal_anchor_multiplier",
        "share_count_cagr",
    ]
    assert scenario.unsupported_assumptions == []

    reconstructed = SectorExpectedReturnScenario.from_dict(scenario.to_dict())

    assert reconstructed == scenario


def test_sector_specific_anchor_method_flows_into_scenario_assumptions():
    packet = _packet()
    packet.ticker = "TECH"
    packet.model_fit_status = "VALID_SECTOR_SPECIFIC"
    packet.valuation = {
        "anchor_method": "technology_adjusted_dcf",
        "valuation_anchor": 120.0,
        "generic_anchor_method": "dcf",
        "generic_anchor_value": 100.0,
        "sector_specific_anchor_method": "technology_adjusted_dcf",
        "sector_specific_anchor_value": 120.0,
    }

    scenario = build_expected_return_scenarios(packet, horizons=[5])[1]

    assert scenario.assumptions["anchor_method"] == "technology_adjusted_dcf"
    assert scenario.assumptions["valuation_anchor"] == 120.0
    assert scenario.estimated_future_value_per_share == 176.319369
    assert scenario.annualized_return == 0.286667
    assert scenario.unsupported_assumptions == []


def test_build_expected_return_scenarios_returns_empty_when_required_inputs_are_missing():
    missing_price = _packet()
    missing_price.current_price = None
    missing_anchor = _packet()
    missing_anchor.valuation = {"anchor_method": "dcf", "valuation_anchor": None}

    assert build_expected_return_scenarios(missing_price, horizons=[5]) == []
    assert build_expected_return_scenarios(missing_anchor, horizons=[5]) == []


def test_missing_growth_defaults_to_zero_and_records_unsupported_assumption():
    packet = _packet()
    packet.business_quality = {"operating_margin": 0.18}
    packet.reinvestment = {}
    packet.capital_allocation = {}

    scenarios = build_expected_return_scenarios(packet, horizons=[5])

    assert [scenario.estimated_future_value_per_share for scenario in scenarios] == [
        65.0,
        100.0,
        120.0,
    ]
    assert [scenario.annualized_return for scenario in scenarios] == [
        0.053874,
        0.148698,
        0.191358,
    ]
    assert [scenario.revenue_cagr for scenario in scenarios] == [0.0, 0.0, 0.0]
    assert scenarios[0].assumptions["growth_source"] == "default_zero"
    assert scenarios[0].unsupported_assumptions == ["GROWTH_ASSUMPTION_DEFAULTED_TO_ZERO"]


def test_buybacks_help_upside_case_without_inflating_base_case():
    packet = _packet()
    packet.business_quality = {"revenue_cagr_5y": 0.08}
    packet.capital_allocation = {"dilution_rate_shares_cagr": -0.02}

    scenarios = build_expected_return_scenarios(packet, horizons=[5])

    assert [scenario.revenue_cagr for scenario in scenarios] == [0.0, 0.08, 0.1]
    assert [scenario.estimated_future_value_per_share for scenario in scenarios] == [
        65.0,
        146.932808,
        193.2612,
    ]
    assert [scenario.annualized_return for scenario in scenarios] == [
        0.053874,
        0.240594,
        0.310494,
    ]


def test_blockers_and_non_ok_data_quality_are_recorded_as_unsupported_assumptions():
    packet = _packet()
    packet.model_fit_status = "BLOCKED"
    packet.data_quality_status = "MODEL_BLOCKED"
    packet.blockers = ["SECURITY_IDENTITY_UNVERIFIED"]

    scenarios = build_expected_return_scenarios(packet, horizons=[5])

    assert scenarios[0].unsupported_assumptions == [
        "BLOCKERS_PRESENT",
        "MODEL_FIT_NOT_FULLY_VALIDATED",
        "DATA_QUALITY_MODEL_BLOCKED",
    ]


def test_build_expected_return_scenarios_for_packets_keys_results_by_ticker():
    aaa = _packet()
    bbb = _packet()
    bbb.ticker = "BBB"
    bbb.valuation = {"anchor_method": "epv", "valuation_anchor": 90.0}

    scenarios_by_ticker = build_expected_return_scenarios_for_packets([bbb, aaa], horizons=[5])

    assert list(scenarios_by_ticker.keys()) == ["BBB", "AAA"]
    assert [scenario.scenario_id for scenario in scenarios_by_ticker["BBB"]] == [
        "BBB_downside_5Y",
        "BBB_base_5Y",
        "BBB_upside_5Y",
    ]
    assert scenarios_by_ticker["AAA"][1].assumptions["anchor_method"] == "dcf"
    assert scenarios_by_ticker["BBB"][1].assumptions["anchor_method"] == "epv"
