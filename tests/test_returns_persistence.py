from __future__ import annotations

import json

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.escalation import build_escalation_plan
from app.universe.memo_pack import _memo_markdown
from app.universe.promotion import LANE_4_DEPRIORITIZED, build_promotion_state
from app.universe.ranking import ranking_sort_key
from app.valuation.investment_readiness import (
    BLOCKER_LOW_RETURNS_PERSISTENCE,
    compute_investment_readiness,
)
from app.valuation.returns_persistence import (
    HIGH_RETURNS_PERSISTENCE,
    LOW_RETURNS_PERSISTENCE,
    MODERATE_RETURNS_PERSISTENCE,
    RETURNS_PERSISTENCE_UNKNOWN,
    compute_returns_persistence,
    write_returns_persistence_for_run,
)
from app.valuation.value_type import compute_value_type


runner = CliRunner()


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "sample_universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker\nAAA\nBBB\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _fundamentals_rows(
    *,
    revenue: list[float],
    invested_capital: list[float],
    roic: list[float] | None = None,
    roe: list[float] | None = None,
    roa: list[float] | None = None,
) -> dict:
    years = [2021, 2022, 2023, 2024, 2025]
    roic = roic or [0.12] * 5
    roe = roe or [0.14] * 5
    roa = roa or [0.07] * 5
    rows = []
    for idx, year in enumerate(years):
        rows.append(
            {
                "year": year,
                "revenue": revenue[idx],
                "invested_capital": invested_capital[idx],
                "roic_proxy": roic[idx],
                "roe_proxy": roe[idx],
                "roa_proxy": roa[idx],
            }
        )
    return {"rows": rows, "derived_from": ["returns.fixture"]}


def _intrinsic_payload(*, mos_to_floor: float | str, mos_classification: str) -> dict:
    return {
        "mos_to_floor": mos_to_floor,
        "mos_to_base": 0.35 if isinstance(mos_to_floor, (int, float)) else "UNKNOWN",
        "mos_classification": mos_classification,
        "downside_support_type": "EARNINGS_POWER_SUPPORT",
        "normalized_earnings_power_status": "OK",
        "normalized_earnings_power_reason_codes": ["FCF_SELECTED"],
        "derived_from": ["intrinsic.fixture"],
        "claims": {
            "mos_to_floor": {"value": mos_to_floor, "derived_from": ["intrinsic.fixture.floor"]},
        },
    }


def _confidence_payload(*, support_count: int = 2, confidence_class: str = "HIGH_CONFIDENCE") -> dict:
    return {
        "valuation_support_count": support_count,
        "valuation_support_types_present": (
            ["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"] if support_count >= 2 else ["EPV_SUPPORT"]
        ),
        "valuation_fragility_status": "LOW_FRAGILITY" if support_count >= 2 else "HIGH_FRAGILITY",
        "valuation_fragility_reason_codes": [] if support_count >= 2 else ["SINGLE_SUPPORT_ONLY"],
        "valuation_confidence_class": confidence_class,
        "derived_from": ["confidence.fixture"],
        "claims": {
            "valuation_support_count": {"value": support_count, "derived_from": ["confidence.fixture.supports"]},
            "valuation_confidence_class": {"value": confidence_class, "derived_from": ["confidence.fixture.class"]},
        },
    }


def _integrity_payload() -> dict:
    return {
        "valuation_integrity_class": "INTEGRITY_OK",
        "valuation_integrity_reason_codes": [],
        "derived_from": ["integrity.fixture"],
    }


def test_high_returns_persistence_for_strong_stable_returns():
    payload = compute_returns_persistence(
        "AAA",
        "2026-03-08",
        fundamentals=_fundamentals_rows(
            revenue=[100, 109, 119, 130, 142],
            invested_capital=[80, 87, 95, 103, 112],
            roic=[0.17, 0.18, 0.18, 0.19, 0.18],
            roe=[0.18, 0.19, 0.19, 0.20, 0.19],
            roa=[0.09, 0.09, 0.10, 0.10, 0.09],
        ),
        intangible_payload={
            "gross_margin_durability_score": 4.5,
            "cycle_resilience_score": 4.0,
            "owner_value_capture_score": 4.0,
        },
        reinvestment_efficiency_payload={"reinvestment_efficiency_class": "HIGH_REINVESTMENT_EFFICIENCY"},
        capital_allocation_discipline_payload={"capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED"},
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    assert payload["returns_persistence_class"] == HIGH_RETURNS_PERSISTENCE
    assert payload["primary_returns_caution"] == "RETURNS_DURABILITY_SUPPORTIVE"
    assert "HIGH_RETURNS_PERSISTENCE_SUPPORT" in payload["returns_persistence_reason_codes"]


def test_moderate_returns_persistence_for_mixed_case():
    payload = compute_returns_persistence(
        "AAA",
        "2026-03-08",
        fundamentals=_fundamentals_rows(
            revenue=[100, 104, 109, 113, 118],
            invested_capital=[80, 85, 91, 97, 104],
            roic=[0.11, 0.10, 0.12, 0.09, 0.11],
            roe=[0.13, 0.12, 0.13, 0.11, 0.12],
            roa=[0.06, 0.06, 0.07, 0.05, 0.06],
        ),
        intangible_payload={
            "gross_margin_durability_score": 3.0,
            "cycle_resilience_score": 3.0,
            "owner_value_capture_score": 2.5,
        },
        reinvestment_efficiency_payload={"reinvestment_efficiency_class": "MODERATE_REINVESTMENT_EFFICIENCY"},
        capital_allocation_discipline_payload={"capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION"},
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    assert payload["returns_persistence_class"] == MODERATE_RETURNS_PERSISTENCE
    assert payload["primary_returns_caution"] == "RETURNS_DURABILITY_MIXED"


def test_low_returns_persistence_for_weak_unstable_deteriorating_case():
    payload = compute_returns_persistence(
        "AAA",
        "2026-03-08",
        fundamentals=_fundamentals_rows(
            revenue=[100, 102, 104, 105, 106],
            invested_capital=[80, 90, 102, 116, 132],
            roic=[0.11, 0.08, 0.06, 0.04, 0.03],
            roe=[0.13, 0.10, 0.08, 0.06, 0.05],
            roa=[0.07, 0.05, 0.04, 0.03, 0.02],
        ),
        intangible_payload={
            "gross_margin_durability_score": 2.0,
            "cycle_resilience_score": 1.5,
            "owner_value_capture_score": 1.0,
            "owner_value_capture_reason_codes": ["WEAK_PER_SHARE_CAPTURE"],
        },
        reinvestment_efficiency_payload={
            "reinvestment_efficiency_class": "LOW_REINVESTMENT_EFFICIENCY",
            "reinvestment_efficiency_reason_codes": ["CAPITAL_HUNGRY_GROWTH_HEADWIND"],
        },
        capital_allocation_discipline_payload={"capital_allocation_discipline_class": "OWNER_DILUTIVE_OR_DESTRUCTIVE"},
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    assert payload["returns_persistence_class"] == LOW_RETURNS_PERSISTENCE
    assert payload["primary_returns_caution"] == "RETURNS_DURABILITY_HEADWIND"
    assert "LOW_RETURNS_PERSISTENCE_HEADWIND" in payload["returns_persistence_reason_codes"]
    assert "INCREMENTAL_RETURNS_DETERIORATING" in payload["returns_headwind_signals"]


def test_unknown_returns_persistence_when_evidence_thin():
    payload = compute_returns_persistence(
        "AAA",
        "2026-03-08",
        fundamentals={"rows": [{"year": 2025, "revenue": 100.0}]},
        price_status="UNKNOWN",
        facts_status="UNKNOWN",
        shares_status="UNKNOWN",
        fcf_status="UNKNOWN",
    )
    assert payload["returns_persistence_class"] == RETURNS_PERSISTENCE_UNKNOWN
    assert "RETURNS_EVIDENCE_THIN" in payload["returns_persistence_reason_codes"]


def test_readiness_applies_returns_durability_headwind_honestly():
    returns_payload = {
        "returns_persistence_class": LOW_RETURNS_PERSISTENCE,
        "returns_persistence_reason_codes": ["LOW_RETURNS_PERSISTENCE_HEADWIND", "INCREMENTAL_RETURNS_DETERIORATION"],
        "returns_support_signals": [],
        "returns_headwind_signals": ["INCREMENTAL_RETURNS_DETERIORATING"],
        "primary_returns_caution": "RETURNS_DURABILITY_HEADWIND",
        "economic_durability_summary": "returns are weakening as capital is added",
        "derived_from": ["returns.fixture"],
    }
    readiness = compute_investment_readiness(
        "AAA",
        "2026-03-08",
        value_gate_status="WATCH",
        primary_blocker="NONE",
        intrinsic_payload=_intrinsic_payload(mos_to_floor=0.35, mos_classification="ADEQUATE_MARGIN_OF_SAFETY"),
        evidence_sufficiency_payload={
            "evidence_sufficiency_class": "SUFFICIENT_FOR_MOS",
            "evidence_sufficiency_reason_codes": ["PRICE_AVAILABLE", "SHARES_AVAILABLE", "FACTS_AVAILABLE"],
            "mos_assessment_status": "MOS_CONFIRMED_PRESENT",
            "mos_guardrail_reason_codes": ["MOS_PRESENT_WITH_SUFFICIENT_EVIDENCE"],
        },
        valuation_confidence_payload=_confidence_payload(support_count=2, confidence_class="MEDIUM_CONFIDENCE"),
        valuation_integrity_payload=_integrity_payload(),
        value_type_payload={"value_type_primary": "EARNINGS_POWER_VALUE", "value_type_reason_codes": ["EPV_DRIVEN"]},
        owner_quality_payload={"oe_quality_total": 8.0, "capital_allocation_score": 3.0, "oe_quality_reason_codes": []},
        intangible_payload={"intangible_economics_total": 7.0, "owner_value_capture_score": 3.0, "owner_value_capture_reason_codes": []},
        returns_persistence_payload=returns_payload,
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
    )
    assert BLOCKER_LOW_RETURNS_PERSISTENCE in readiness["blocker_stack_all"]
    assert "LOW_RETURNS_PERSISTENCE_HEADWIND" in readiness["readiness_support_headwinds"]


def test_value_type_uses_returns_persistence_as_refinement_not_override():
    high_returns = {
        "returns_persistence_class": HIGH_RETURNS_PERSISTENCE,
        "returns_persistence_reason_codes": ["HIGH_RETURNS_PERSISTENCE_SUPPORT"],
        "returns_support_signals": ["HIGH_RETURN_ON_CAPITAL_PRESENT", "RETURNS_STABILITY_PRESENT"],
        "returns_headwind_signals": [],
        "primary_returns_caution": "RETURNS_DURABILITY_SUPPORTIVE",
        "derived_from": ["returns.high"],
    }
    low_returns = {
        "returns_persistence_class": LOW_RETURNS_PERSISTENCE,
        "returns_persistence_reason_codes": ["LOW_RETURNS_PERSISTENCE_HEADWIND"],
        "returns_support_signals": [],
        "returns_headwind_signals": ["INCREMENTAL_RETURNS_DETERIORATING"],
        "primary_returns_caution": "RETURNS_DURABILITY_HEADWIND",
        "derived_from": ["returns.low"],
    }

    payload_high = compute_value_type(
        "AAA",
        "2026-03-08",
        intrinsic_payload={
            "mos_classification": "ADEQUATE_MARGIN_OF_SAFETY",
            "downside_support_type": "EARNINGS_POWER_SUPPORT",
            "normalized_earnings_power_status": "OK",
            "normalized_earnings_power_method_used": "FCF_SELECTED",
            "normalized_earnings_power_reason_codes": ["FCF_SELECTED"],
        },
        valuation_confidence_payload={
            "valuation_support_count": 3,
            "valuation_support_types_present": ["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT", "OWNER_EARNINGS_VALUE_SUPPORT"],
            "valuation_confidence_class": "HIGH_CONFIDENCE",
            "valuation_fragility_status": "LOW_FRAGILITY",
        },
        owner_quality_payload={"oe_quality_total": 9.0},
        intangible_payload={"intangible_economics_total": 9.0, "owner_value_capture_score": 4.0},
        reinvestment_efficiency_payload={"reinvestment_efficiency_class": "HIGH_REINVESTMENT_EFFICIENCY"},
        returns_persistence_payload=high_returns,
    )
    assert payload_high["value_type_primary"] == "QUALITY_VALUE"

    payload_low = compute_value_type(
        "AAA",
        "2026-03-08",
        intrinsic_payload={
            "mos_classification": "ADEQUATE_MARGIN_OF_SAFETY",
            "downside_support_type": "EARNINGS_POWER_SUPPORT",
            "normalized_earnings_power_status": "OK",
            "normalized_earnings_power_method_used": "FCF_SELECTED",
            "normalized_earnings_power_reason_codes": ["FCF_SELECTED"],
        },
        valuation_confidence_payload={
            "valuation_support_count": 3,
            "valuation_support_types_present": ["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT", "OWNER_EARNINGS_VALUE_SUPPORT"],
            "valuation_confidence_class": "HIGH_CONFIDENCE",
            "valuation_fragility_status": "LOW_FRAGILITY",
        },
        owner_quality_payload={"oe_quality_total": 9.0},
        intangible_payload={"intangible_economics_total": 9.0, "owner_value_capture_score": 4.0},
        reinvestment_efficiency_payload={"reinvestment_efficiency_class": "HIGH_REINVESTMENT_EFFICIENCY"},
        returns_persistence_payload=low_returns,
    )
    assert payload_low["value_type_primary"] != "QUALITY_VALUE"
    assert "LOW_RETURNS_PERSISTENCE_HEADWIND" in payload_low["value_type_reason_codes"]


def test_memo_pack_includes_returns_persistence_section():
    memo = _memo_markdown(
        {
            "header": {"ticker": "AAA", "as_of_date": "2026-03-08"},
            "returns_on_capital_persistence_economic_durability": {
                "returns_persistence_class": HIGH_RETURNS_PERSISTENCE,
                "primary_returns_caution": "RETURNS_DURABILITY_SUPPORTIVE",
                "returns_persistence_reason_codes": ["HIGH_RETURNS_PERSISTENCE_SUPPORT"],
                "returns_support_signals": ["HIGH_RETURN_ON_CAPITAL_PRESENT"],
                "returns_headwind_signals": [],
                "economic_durability_summary": "returns appear durable and repeatable",
                "derived_from": ["returns.fixture"],
            },
        }
    )
    assert "## Returns on Capital Persistence / Economic Durability" in memo
    assert HIGH_RETURNS_PERSISTENCE in memo


def test_promotion_and_escalation_surface_returns_persistence_without_overriding_fail():
    master_shortlist = {
        "campaign_run_id": "camp_returns",
        "rows": [
            {
                "ticker": "AAA",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "camp_returns__core"}],
                "best_rank_seen": 1,
                "value_gate_status": "WATCH",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": 0.18,
                "primary_blocker": "MISSING_FCF",
                "latest_primary_blocker": "MISSING_FCF",
                "returns_persistence_class": HIGH_RETURNS_PERSISTENCE,
                "returns_persistence_reason_codes": ["HIGH_RETURNS_PERSISTENCE_SUPPORT"],
                "returns_support_signals": ["HIGH_RETURN_ON_CAPITAL_PRESENT"],
                "returns_headwind_signals": [],
                "primary_returns_caution": "RETURNS_DURABILITY_SUPPORTIVE",
                "economic_durability_summary": "supportive",
                "memo_path": "memo/AAA.md",
            },
            {
                "ticker": "BBB",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "camp_returns__core"}],
                "best_rank_seen": 2,
                "value_gate_status": "FAIL",
                "latest_value_gate_status": "FAIL",
                "implied_return_base": 0.01,
                "primary_blocker": "PRICE_UNKNOWN",
                "latest_primary_blocker": "PRICE_UNKNOWN",
                "returns_persistence_class": LOW_RETURNS_PERSISTENCE,
                "returns_persistence_reason_codes": ["LOW_RETURNS_PERSISTENCE_HEADWIND"],
                "returns_support_signals": [],
                "returns_headwind_signals": ["INCREMENTAL_RETURNS_DETERIORATING"],
                "primary_returns_caution": "RETURNS_DURABILITY_HEADWIND",
                "economic_durability_summary": "headwind",
                "memo_path": "memo/BBB.md",
            },
        ],
    }
    master_watchlist_state = {
        "campaign_run_id": "camp_returns",
        "tickers": {
            "AAA": {"appearances_count": 2, "latest_value_gate_status": "WATCH", "latest_primary_blocker": "MISSING_FCF", "history": []},
            "BBB": {"appearances_count": 2, "latest_value_gate_status": "FAIL", "latest_primary_blocker": "PRICE_UNKNOWN", "history": []},
        },
    }
    promotion_state = build_promotion_state("camp_returns", master_watchlist_state, master_shortlist)
    row_bbb = next(row for row in promotion_state["rows"] if row["ticker"] == "BBB")
    assert row_bbb["returns_persistence_class"] == LOW_RETURNS_PERSISTENCE
    assert row_bbb["priority_lane"] == LANE_4_DEPRIORITIZED

    lanes = {
        "lane_1_high_priority": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_1_HIGH_PRIORITY"],
        "lane_2_research_queue": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_2_RESEARCH_QUEUE"],
        "lane_3_monitor": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_3_MONITOR"],
        "lane_4_deprioritized": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_4_DEPRIORITIZED"],
    }
    escalation = build_escalation_plan("camp_returns", promotion_state, lanes)
    assert any("returns_persistence_class" in entry for entry in escalation["queue"])


def test_returns_persistence_cli_open_and_value_first_durable_returns_sort(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    detail_high = compute_returns_persistence(
        "AAA",
        "2026-03-08",
        fundamentals=_fundamentals_rows(
            revenue=[100, 109, 119, 130, 142],
            invested_capital=[80, 87, 95, 103, 112],
            roic=[0.17, 0.18, 0.18, 0.19, 0.18],
        ),
        intangible_payload={"gross_margin_durability_score": 4.5, "cycle_resilience_score": 4.0, "owner_value_capture_score": 4.0},
        reinvestment_efficiency_payload={"reinvestment_efficiency_class": "HIGH_REINVESTMENT_EFFICIENCY"},
        capital_allocation_discipline_payload={"capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED"},
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    detail_low = compute_returns_persistence(
        "BBB",
        "2026-03-08",
        fundamentals=_fundamentals_rows(
            revenue=[100, 102, 104, 105, 106],
            invested_capital=[80, 90, 102, 116, 132],
            roic=[0.11, 0.08, 0.06, 0.04, 0.03],
        ),
        intangible_payload={"gross_margin_durability_score": 2.0, "cycle_resilience_score": 1.5, "owner_value_capture_score": 1.0},
        reinvestment_efficiency_payload={"reinvestment_efficiency_class": "LOW_REINVESTMENT_EFFICIENCY"},
        capital_allocation_discipline_payload={"capital_allocation_discipline_class": "OWNER_DILUTIVE_OR_DESTRUCTIVE"},
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    run_id = "returns_persistence_test_open"
    output_path = cfg.outputs_dir / "universe" / run_id / "returns_persistence.json"
    write_returns_persistence_for_run(
        run_id=run_id,
        as_of_date="2026-03-08",
        tickers=["AAA", "BBB"],
        output_path=output_path,
        scoreboard_rows=[
            {"ticker": "AAA", "returns_persistence_detail": detail_high},
            {"ticker": "BBB", "returns_persistence_detail": detail_low},
        ],
    )

    result = runner.invoke(app, ["universe-returns-persistence-open", "--run-id", run_id])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "OK"
    assert payload["counts_by_returns_persistence_class"][HIGH_RETURNS_PERSISTENCE] == 1
    assert payload["counts_by_returns_persistence_class"][LOW_RETURNS_PERSISTENCE] == 1

    rows = [
        {
            "ticker": "AAA",
            "scout_status": "WATCH",
            "investment_readiness_class": "RESEARCH_WORTHY_NOT_READY",
            "mos_to_floor": 0.20,
            "valuation_confidence_class": "MEDIUM_CONFIDENCE",
            "valuation_integrity_class": "INTEGRITY_OK",
            "returns_persistence_class": HIGH_RETURNS_PERSISTENCE,
            "reinvestment_efficiency_class": "HIGH_REINVESTMENT_EFFICIENCY",
            "capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED",
            "accounting_quality_class": "HIGH_ACCOUNTING_QUALITY",
            "value_type_primary": "QUALITY_VALUE",
            "normalization_credibility_class": "MODERATE_NORMALIZATION_CREDIBILITY",
            "oe_quality_total": 8.0,
            "intangible_economics_total": 7.0,
            "capital_allocation_score": 4.0,
        },
        {
            "ticker": "BBB",
            "scout_status": "WATCH",
            "investment_readiness_class": "RESEARCH_WORTHY_NOT_READY",
            "mos_to_floor": 0.20,
            "valuation_confidence_class": "MEDIUM_CONFIDENCE",
            "valuation_integrity_class": "INTEGRITY_OK",
            "returns_persistence_class": LOW_RETURNS_PERSISTENCE,
            "reinvestment_efficiency_class": "LOW_REINVESTMENT_EFFICIENCY",
            "capital_allocation_discipline_class": "OWNER_DILUTIVE_OR_DESTRUCTIVE",
            "accounting_quality_class": "MODERATE_ACCOUNTING_QUALITY",
            "value_type_primary": "EARNINGS_POWER_VALUE",
            "normalization_credibility_class": "MODERATE_NORMALIZATION_CREDIBILITY",
            "oe_quality_total": 6.0,
            "intangible_economics_total": 5.0,
            "capital_allocation_score": 1.0,
        },
    ]
    ordered = sorted(rows, key=lambda row: ranking_sort_key(row, policy="value_first_durable_returns"))
    assert [row["ticker"] for row in ordered] == ["AAA", "BBB"]


def test_no_return_on_capital_equity_or_assets_is_unknown_even_with_supportive_sub_scores():
    """Growth, capital-spread and intangible sub-scores are context, not a measured return:
    with ROIC, ROE and ROA all unknown the class is UNKNOWN with a named reason."""
    payload = compute_returns_persistence(
        "AAA",
        "2026-03-08",
        fundamentals={"rows": [{"year": 2024, "revenue": 100.0}, {"year": 2025, "revenue": 105.0}]},
        intangible_payload={
            "gross_margin_durability_score": 4.0,
            "cycle_resilience_score": 4.0,
            "owner_value_capture_score": 3.0,
        },
        reinvestment_efficiency_payload={"reinvestment_efficiency_class": "HIGH_REINVESTMENT_EFFICIENCY"},
        capital_allocation_discipline_payload={"capital_allocation_class": "OWNER_FRIENDLY_DISCIPLINED"},
        return_on_retained_earnings=0.20,
        revenue_cagr_proxy=0.05,
        invested_capital_cagr_proxy=0.05,
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    assert payload["returns_persistence_class"] == RETURNS_PERSISTENCE_UNKNOWN
    assert payload["primary_returns_caution"] == "RETURNS_DURABILITY_UNCLEAR"
    codes = payload["returns_persistence_reason_codes"]
    assert "RETURN_ON_CAPITAL_EQUITY_ASSETS_ALL_UNKNOWN" in codes
    assert "MISSING_RETURNS_INPUTS" in codes
    assert "HIGH_RETURNS_PERSISTENCE_SUPPORT" not in codes
    assert "RETURNS_EVIDENCE_THIN" in payload["returns_headwind_signals"]


def test_a_single_measured_return_is_enough_to_grade():
    """ROA alone (no ROIC, no ROE) is a measured return; the class is not forced UNKNOWN."""
    payload = compute_returns_persistence(
        "AAA",
        "2026-03-08",
        roa_proxy=0.02,
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    assert payload["returns_persistence_class"] != RETURNS_PERSISTENCE_UNKNOWN
    assert "RETURN_ON_CAPITAL_EQUITY_ASSETS_ALL_UNKNOWN" not in payload["returns_persistence_reason_codes"]
