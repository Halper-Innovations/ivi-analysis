from __future__ import annotations

import json

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.escalation import build_escalation_plan
from app.universe.memo_pack import _memo_markdown
from app.universe.promotion import LANE_4_DEPRIORITIZED, build_promotion_state
from app.universe.ranking import ranking_sort_key
from app.valuation.investment_readiness import BLOCKER_LOW_REINVESTMENT_EFFICIENCY, compute_investment_readiness
from app.valuation.reinvestment_efficiency import (
    HIGH_REINVESTMENT_EFFICIENCY,
    LOW_REINVESTMENT_EFFICIENCY,
    MODERATE_REINVESTMENT_EFFICIENCY,
    REINVESTMENT_EFFICIENCY_UNKNOWN,
    compute_reinvestment_efficiency,
    write_reinvestment_efficiency_for_run,
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
    cfo: list[float],
    fcf: list[float],
    capex: list[float],
    shares: list[float],
) -> dict:
    years = [2021, 2022, 2023, 2024, 2025]
    rows = []
    for idx, year in enumerate(years):
        rows.append(
            {
                "year": year,
                "revenue": revenue[idx],
                "cfo": cfo[idx],
                "fcf": fcf[idx],
                "capex": capex[idx],
                "shares_outstanding": shares[idx],
            }
        )
    return {"rows": rows, "derived_from": ["fundamentals.fixture"]}


def _intrinsic_payload(*, mos_to_floor: float, mos_classification: str) -> dict:
    return {
        "mos_to_floor": mos_to_floor,
        "mos_classification": mos_classification,
        "downside_support_type": "EARNINGS_POWER_SUPPORT",
        "normalized_earnings_power_status": "OK",
        "normalized_earnings_power_reason_codes": ["FCF_SELECTED"],
        "derived_from": ["intrinsic.fixture"],
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
            "valuation_support_count": {"derived_from": ["confidence.supports"]},
            "valuation_confidence_class": {"derived_from": ["confidence.class"]},
        },
    }


def _integrity_payload() -> dict:
    return {
        "valuation_integrity_class": "INTEGRITY_OK",
        "valuation_integrity_reason_codes": [],
        "derived_from": ["integrity.fixture"],
    }


def test_high_reinvestment_efficiency_for_productive_growth():
    payload = compute_reinvestment_efficiency(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_rows(
            revenue=[100, 110, 124, 141, 160],
            cfo=[20, 23, 27, 31, 36],
            fcf=[12, 14, 17, 21, 25],
            capex=[4, 5, 5, 6, 6],
            shares=[100, 100, 99, 99, 98],
        ),
        owner_quality_payload={
            "oe_quality_total": 9.0,
            "owner_earnings_stability_score": 4.0,
            "cash_conversion_score": 3.0,
            "derived_from": ["oe_quality.fixture"],
        },
        intangible_payload={
            "gross_margin_durability_score": 4.0,
            "rnd_productivity_score": 4.0,
            "sga_leverage_score": 4.0,
            "owner_value_capture_score": 4.0,
            "owner_value_capture_reason_codes": ["STRONG_OWNER_VALUE_CAPTURE"],
            "derived_from": ["intangible.fixture"],
        },
        capital_allocation_discipline_payload={
            "capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED",
            "per_share_support_signals": ["OWNER_VALUE_CAPTURE_PRESENT"],
            "derived_from": ["cap_alloc.fixture"],
        },
        evidence_sufficiency_payload={"evidence_sufficiency_class": "SUFFICIENT_FOR_MOS"},
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
    )
    assert payload["reinvestment_efficiency_class"] == HIGH_REINVESTMENT_EFFICIENCY
    assert payload["primary_reinvestment_caution"] == "REINVESTMENT_APPEARS_PRODUCTIVE"
    assert "PRODUCTIVE_REINVESTMENT_SUPPORT" in payload["reinvestment_efficiency_reason_codes"]


def test_moderate_reinvestment_efficiency_for_mixed_support():
    payload = compute_reinvestment_efficiency(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_rows(
            revenue=[100, 105, 110, 115, 120],
            cfo=[20, 20.5, 21, 21.5, 22],
            fcf=[10, 10.1, 10.2, 10.3, 10.4],
            capex=[5.5, 6.0, 6.0, 6.0, 6.5],
            shares=[100, 100, 100, 100, 100],
        ),
        owner_quality_payload={
            "oe_quality_total": 6.0,
            "owner_earnings_stability_score": 3.0,
            "cash_conversion_score": 2.0,
        },
        intangible_payload={
            "gross_margin_durability_score": 3.0,
            "rnd_productivity_score": 2.0,
            "sga_leverage_score": 2.0,
            "owner_value_capture_score": 2.0,
        },
        capital_allocation_discipline_payload={
            "capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION",
        },
        evidence_sufficiency_payload={"evidence_sufficiency_class": "SUFFICIENT_FOR_MOS"},
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
    )
    assert payload["reinvestment_efficiency_class"] == MODERATE_REINVESTMENT_EFFICIENCY
    assert payload["primary_reinvestment_caution"] == "REINVESTMENT_MIXED"
    assert "REINVESTMENT_MIXED" in payload["reinvestment_efficiency_reason_codes"]


def test_low_reinvestment_efficiency_for_capital_hungry_growth():
    payload = compute_reinvestment_efficiency(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_rows(
            revenue=[100, 112, 125, 138, 152],
            cfo=[20, 19, 18, 17, 16],
            fcf=[12, 11, 10, 9, 8],
            capex=[14, 15, 16, 18, 20],
            shares=[100, 102, 104, 107, 110],
        ),
        owner_quality_payload={
            "oe_quality_total": 4.0,
            "owner_earnings_stability_score": 1.0,
            "cash_conversion_score": 1.0,
        },
        intangible_payload={
            "gross_margin_durability_score": 2.0,
            "rnd_productivity_score": 1.0,
            "sga_leverage_score": 1.0,
            "owner_value_capture_score": 0.5,
            "owner_value_capture_reason_codes": ["WEAK_PER_SHARE_CAPTURE"],
        },
        capital_allocation_discipline_payload={
            "capital_allocation_discipline_class": "OWNER_DILUTIVE_OR_DESTRUCTIVE",
            "per_share_headwind_signals": ["HIGH_DILUTION", "DILUTION_DESTROYS_OWNER_VALUE"],
        },
        evidence_sufficiency_payload={"evidence_sufficiency_class": "SUFFICIENT_FOR_MOS"},
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
    )
    assert payload["reinvestment_efficiency_class"] == LOW_REINVESTMENT_EFFICIENCY
    assert payload["primary_reinvestment_caution"] == "REINVESTMENT_HEADWIND"
    assert "CAPITAL_HUNGRY_GROWTH_HEADWIND" in payload["reinvestment_efficiency_reason_codes"]
    assert "GROWTH_WITHOUT_OWNER_OUTCOME" in payload["reinvestment_efficiency_reason_codes"]


def test_unknown_reinvestment_efficiency_when_evidence_thin():
    payload = compute_reinvestment_efficiency(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_rows(
            revenue=[100, 101, 102, 103, 104],
            cfo=[20, 20, 20, 20, 20],
            fcf=[10, 10, 10, 10, 10],
            capex=[6, 6, 6, 6, 6],
            shares=[100, 100, 100, 100, 100],
        ),
        evidence_sufficiency_payload={"evidence_sufficiency_class": "INSUFFICIENT_FOR_MOS"},
        price_status="UNKNOWN",
        shares_status="UNKNOWN",
        fcf_status="UNKNOWN",
        facts_status="UNKNOWN",
    )
    assert payload["reinvestment_efficiency_class"] == REINVESTMENT_EFFICIENCY_UNKNOWN
    assert "REINVESTMENT_EVIDENCE_THIN" in payload["reinvestment_efficiency_reason_codes"]


def test_readiness_adds_low_reinvestment_headwind():
    readiness = compute_investment_readiness(
        "AAA",
        "2026-02-14",
        value_gate_status="WATCH",
        primary_blocker="NONE",
        intrinsic_payload=_intrinsic_payload(mos_to_floor=0.25, mos_classification="ADEQUATE_MARGIN_OF_SAFETY"),
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
        intangible_payload={"intangible_economics_total": 7.0, "owner_value_capture_score": 2.0, "owner_value_capture_reason_codes": []},
        reinvestment_efficiency_payload={
            "reinvestment_efficiency_class": LOW_REINVESTMENT_EFFICIENCY,
            "reinvestment_efficiency_reason_codes": ["CAPITAL_HUNGRY_GROWTH_HEADWIND"],
            "reinvestment_headwind_signals": ["CAPITAL_HUNGRY_GROWTH"],
        },
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
    )
    assert BLOCKER_LOW_REINVESTMENT_EFFICIENCY in readiness["blocker_stack_all"]
    assert "CAPITAL_HUNGRY_GROWTH_HEADWIND" in readiness["readiness_support_headwinds"]


def test_value_type_uses_reinvestment_as_refinement_not_override():
    payload = compute_value_type(
        "AAA",
        "2026-02-14",
        intrinsic_payload={
            "mos_classification": "MOS_UNKNOWN",
            "downside_support_type": "UNKNOWN_SUPPORT",
            "normalized_earnings_power_status": "UNKNOWN",
            "normalized_earnings_power_method_used": "UNKNOWN",
            "normalized_earnings_power_reason_codes": ["INSUFFICIENT_NORMALIZED_INPUTS"],
        },
        valuation_confidence_payload={
            "valuation_support_count": 0,
            "valuation_support_types_present": [],
            "valuation_confidence_class": "CONFIDENCE_UNKNOWN",
            "valuation_fragility_status": "FRAGILITY_UNKNOWN",
        },
        owner_quality_payload={"oe_quality_total": 9.0},
        intangible_payload={"intangible_economics_total": 9.0, "owner_value_capture_score": 4.0},
        reinvestment_efficiency_payload={"reinvestment_efficiency_class": HIGH_REINVESTMENT_EFFICIENCY},
        fail_due_to_missing_evidence=True,
        primary_fail_domain="EVIDENCE",
    )
    assert payload["value_type_primary"] == "UNKNOWN_VALUE_TYPE"
    assert "PRODUCTIVE_REINVESTMENT_SUPPORT" not in payload["value_type_reason_codes"]


def test_memo_pack_includes_reinvestment_efficiency_section():
    memo = _memo_markdown(
        {
            "header": {"ticker": "AAA", "as_of_date": "2026-02-14"},
            "incremental_reinvestment_efficiency": {
                "reinvestment_efficiency_class": HIGH_REINVESTMENT_EFFICIENCY,
                "primary_reinvestment_caution": "REINVESTMENT_APPEARS_PRODUCTIVE",
                "reinvestment_efficiency_reason_codes": ["PRODUCTIVE_REINVESTMENT_SUPPORT"],
                "reinvestment_support_signals": ["FCF_GROWTH_PRESENT"],
                "reinvestment_headwind_signals": [],
                "reinvestment_efficiency_summary": "productive reinvestment signals are present",
                "derived_from": ["reinvestment.fixture"],
            },
        }
    )
    assert "## Incremental Reinvestment Efficiency" in memo
    assert "HIGH_REINVESTMENT_EFFICIENCY" in memo


def test_promotion_and_escalation_surface_reinvestment_without_overriding_fail():
    master_shortlist = {
        "campaign_run_id": "camp_reinv",
        "rows": [
            {
                "ticker": "AAA",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "camp_reinv__core"}],
                "best_rank_seen": 1,
                "value_gate_status": "WATCH",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": 0.20,
                "primary_blocker": "MISSING_FCF",
                "latest_primary_blocker": "MISSING_FCF",
                "reinvestment_efficiency_class": HIGH_REINVESTMENT_EFFICIENCY,
                "reinvestment_efficiency_reason_codes": ["PRODUCTIVE_REINVESTMENT_SUPPORT"],
                "reinvestment_support_signals": ["FCF_GROWTH_PRESENT"],
                "reinvestment_headwind_signals": [],
                "primary_reinvestment_caution": "REINVESTMENT_APPEARS_PRODUCTIVE",
                "reinvestment_efficiency_summary": "productive",
                "memo_path": "memo/AAA.md",
            },
            {
                "ticker": "BBB",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "camp_reinv__core"}],
                "best_rank_seen": 2,
                "value_gate_status": "FAIL",
                "latest_value_gate_status": "FAIL",
                "implied_return_base": 0.02,
                "primary_blocker": "PRICE_UNKNOWN",
                "latest_primary_blocker": "PRICE_UNKNOWN",
                "reinvestment_efficiency_class": HIGH_REINVESTMENT_EFFICIENCY,
                "reinvestment_efficiency_reason_codes": ["PRODUCTIVE_REINVESTMENT_SUPPORT"],
                "reinvestment_support_signals": ["FCF_GROWTH_PRESENT"],
                "reinvestment_headwind_signals": [],
                "primary_reinvestment_caution": "REINVESTMENT_APPEARS_PRODUCTIVE",
                "reinvestment_efficiency_summary": "productive",
                "memo_path": "memo/BBB.md",
            },
        ],
    }
    master_watchlist_state = {
        "campaign_run_id": "camp_reinv",
        "tickers": {
            "AAA": {"appearances_count": 2, "latest_value_gate_status": "WATCH", "latest_primary_blocker": "MISSING_FCF", "history": []},
            "BBB": {"appearances_count": 2, "latest_value_gate_status": "FAIL", "latest_primary_blocker": "PRICE_UNKNOWN", "history": []},
        },
    }
    promotion_state = build_promotion_state("camp_reinv", master_watchlist_state, master_shortlist)
    row_bbb = next(row for row in promotion_state["rows"] if row["ticker"] == "BBB")
    assert row_bbb["reinvestment_efficiency_class"] == HIGH_REINVESTMENT_EFFICIENCY
    assert row_bbb["priority_lane"] == LANE_4_DEPRIORITIZED

    lanes = {
        "lane_1_high_priority": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_1_HIGH_PRIORITY"],
        "lane_2_research_queue": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_2_RESEARCH_QUEUE"],
        "lane_3_monitor": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_3_MONITOR"],
        "lane_4_deprioritized": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_4_DEPRIORITIZED"],
    }
    escalation = build_escalation_plan("camp_reinv", promotion_state, lanes)
    assert any("reinvestment_efficiency_class" in entry for entry in escalation["queue"])
    assert any(
        "PRODUCTIVE_REINVESTMENT_SUPPORT" in (entry.get("priority_support_codes") or [])
        for entry in escalation["queue"]
    )


def test_reinvestment_cli_open_and_value_first_reinvestment_sort(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    detail_high = compute_reinvestment_efficiency(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_rows(
            revenue=[100, 110, 124, 141, 160],
            cfo=[20, 23, 27, 31, 36],
            fcf=[12, 14, 17, 21, 25],
            capex=[4, 5, 5, 6, 6],
            shares=[100, 100, 99, 99, 98],
        ),
        owner_quality_payload={"oe_quality_total": 9.0, "owner_earnings_stability_score": 4.0, "cash_conversion_score": 3.0},
        intangible_payload={"gross_margin_durability_score": 4.0, "owner_value_capture_score": 4.0},
        capital_allocation_discipline_payload={"capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED"},
        evidence_sufficiency_payload={"evidence_sufficiency_class": "SUFFICIENT_FOR_MOS"},
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
    )
    detail_low = compute_reinvestment_efficiency(
        "BBB",
        "2026-02-14",
        fundamentals=_fundamentals_rows(
            revenue=[100, 112, 125, 138, 152],
            cfo=[20, 19, 18, 17, 16],
            fcf=[12, 11, 10, 9, 8],
            capex=[14, 15, 16, 18, 20],
            shares=[100, 102, 104, 107, 110],
        ),
        owner_quality_payload={"oe_quality_total": 4.0, "owner_earnings_stability_score": 1.0, "cash_conversion_score": 1.0},
        intangible_payload={"owner_value_capture_score": 0.5, "owner_value_capture_reason_codes": ["WEAK_PER_SHARE_CAPTURE"]},
        capital_allocation_discipline_payload={"capital_allocation_discipline_class": "OWNER_DILUTIVE_OR_DESTRUCTIVE"},
        evidence_sufficiency_payload={"evidence_sufficiency_class": "SUFFICIENT_FOR_MOS"},
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
    )
    run_id = "reinvestment_test_open"
    output_path = cfg.outputs_dir / "universe" / run_id / "reinvestment_efficiency.json"
    write_reinvestment_efficiency_for_run(
        run_id=run_id,
        as_of_date="2026-02-14",
        tickers=["AAA", "BBB"],
        output_path=output_path,
        scoreboard_rows=[
            {"ticker": "AAA", "reinvestment_efficiency_detail": detail_high},
            {"ticker": "BBB", "reinvestment_efficiency_detail": detail_low},
        ],
    )

    result = runner.invoke(app, ["universe-reinvestment-efficiency-open", "--run-id", run_id])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "OK"
    assert payload["counts_by_reinvestment_efficiency_class"][HIGH_REINVESTMENT_EFFICIENCY] == 1
    assert payload["counts_by_reinvestment_efficiency_class"][LOW_REINVESTMENT_EFFICIENCY] == 1

    rows = [
        {
            "ticker": "AAA",
            "scout_status": "WATCH",
            "investment_readiness_class": "RESEARCH_WORTHY_NOT_READY",
            "mos_to_floor": 0.20,
            "valuation_confidence_class": "MEDIUM_CONFIDENCE",
            "valuation_integrity_class": "INTEGRITY_OK",
            "reinvestment_efficiency_class": HIGH_REINVESTMENT_EFFICIENCY,
            "capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION",
            "value_type_primary": "EARNINGS_POWER_VALUE",
            "normalization_credibility_class": "MODERATE_NORMALIZATION_CREDIBILITY",
            "oe_quality_total": 7.0,
            "intangible_economics_total": 6.0,
            "owner_earnings_yield_ev_3y": 0.05,
        },
        {
            "ticker": "BBB",
            "scout_status": "WATCH",
            "investment_readiness_class": "RESEARCH_WORTHY_NOT_READY",
            "mos_to_floor": 0.20,
            "valuation_confidence_class": "MEDIUM_CONFIDENCE",
            "valuation_integrity_class": "INTEGRITY_OK",
            "reinvestment_efficiency_class": LOW_REINVESTMENT_EFFICIENCY,
            "capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION",
            "value_type_primary": "EARNINGS_POWER_VALUE",
            "normalization_credibility_class": "MODERATE_NORMALIZATION_CREDIBILITY",
            "oe_quality_total": 7.0,
            "intangible_economics_total": 6.0,
            "owner_earnings_yield_ev_3y": 0.05,
        },
    ]
    ordered = sorted(rows, key=lambda row: ranking_sort_key(row, policy="value_first_reinvestment"))
    assert [row["ticker"] for row in ordered] == ["AAA", "BBB"]
