from __future__ import annotations

import json

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.escalation import build_escalation_plan
from app.universe.memo_pack import _memo_markdown
from app.universe.promotion import LANE_4_DEPRIORITIZED, build_promotion_state
from app.universe.ranking import ranking_sort_key
from app.valuation.balance_sheet_stress import (
    BALANCE_SHEET_STRESS_UNKNOWN,
    CASHFLOW_UNITS_USD,
    CASHFLOW_UNITS_USD_MILLIONS,
    HIGH_BALANCE_SHEET_STRESS,
    HIGH_REFINANCING_RISK,
    LOW_BALANCE_SHEET_STRESS,
    MODERATE_BALANCE_SHEET_STRESS,
    compute_balance_sheet_stress,
    write_balance_sheet_stress_for_run,
)
from app.valuation.investment_readiness import (
    BLOCKER_HIGH_BALANCE_SHEET_STRESS,
    BLOCKER_HIGH_REFINANCING_RISK,
    compute_investment_readiness,
)
from app.valuation.valuation_confidence import compute_valuation_confidence


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


def _balance_rows(
    *,
    net_debt: list[float] | None = None,
    total_debt: list[float] | None = None,
    cash: list[float] | None = None,
    cfo: list[float] | None = None,
    owner_earnings: list[float] | None = None,
) -> dict:
    years = [2021, 2022, 2023, 2024, 2025]
    net_debt = net_debt or [10.0] * 5
    total_debt = total_debt or [50.0] * 5
    cash = cash or [40.0] * 5
    cfo = cfo or [20.0] * 5
    owner_earnings = owner_earnings or [18.0] * 5
    rows = []
    for idx, year in enumerate(years):
        rows.append(
            {
                "year": year,
                "net_debt": net_debt[idx],
                "total_debt": total_debt[idx],
                "cash_equivalents": cash[idx],
                "cfo": cfo[idx],
                "owner_earnings": owner_earnings[idx],
                "fcf": owner_earnings[idx],
            }
        )
    return {"rows": rows, "derived_from": ["balance.fixture"]}


def _intrinsic_payload(*, mos_to_floor: float | str, mos_classification: str) -> dict:
    return {
        "mos_to_floor": mos_to_floor,
        "mos_to_base": 0.30 if isinstance(mos_to_floor, (int, float)) else "UNKNOWN",
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


def test_low_balance_sheet_stress_for_strong_liquidity_case():
    payload = compute_balance_sheet_stress(
        "AAA",
        "2026-03-01",
        fundamentals=_balance_rows(
            net_debt=[8, 8, 9, 10, 10],
            total_debt=[45, 46, 47, 48, 50],
            cash=[38, 38, 39, 39, 40],
            cfo=[18, 19, 20, 21, 22],
            owner_earnings=[16, 17, 18, 19, 20],
        ),
        intangible_payload={"balance_sheet_optionality_score": 4.5, "derived_from": ["intangible.fixture"]},
        facts_status="OK",
    )
    assert payload["balance_sheet_stress_class"] == LOW_BALANCE_SHEET_STRESS
    assert payload["refinancing_risk_class"] == "LOW_REFINANCING_RISK"
    assert payload["primary_balance_sheet_caution"] == "BALANCE_SHEET_SUPPORTIVE"


def test_moderate_balance_sheet_stress_classification():
    payload = compute_balance_sheet_stress(
        "AAA",
        "2026-03-01",
        fundamentals=_balance_rows(
            net_debt=[35, 36, 38, 39, 40],
            total_debt=[38, 39, 40, 40, 40],
            cash=[8, 8, 9, 9, 10],
            cfo=[18, 18, 19, 19, 20],
            owner_earnings=[16, 16, 17, 17, 18],
        ),
        intangible_payload={"balance_sheet_optionality_score": 3.0},
        facts_status="OK",
    )
    assert payload["balance_sheet_stress_class"] == MODERATE_BALANCE_SHEET_STRESS
    assert payload["primary_balance_sheet_caution"] == "BALANCE_SHEET_MIXED"


def test_high_balance_sheet_stress_for_leverage_heavy_weak_cushion_case():
    payload = compute_balance_sheet_stress(
        "AAA",
        "2026-03-01",
        fundamentals=_balance_rows(
            net_debt=[90, 100, 108, 115, 120],
            total_debt=[95, 105, 113, 120, 125],
            cash=[6, 6, 6, 5.5, 5],
            cfo=[18, 18, 19, 19, 20],
            owner_earnings=[15, 15, 16, 17, 18],
        ),
        intangible_payload={
            "balance_sheet_optionality_score": 1.5,
            "balance_sheet_optionality_reason_codes": ["BALANCE_SHEET_OPTIONALITY_UNKNOWN"],
        },
        facts_status="OK",
    )
    assert payload["balance_sheet_stress_class"] == HIGH_BALANCE_SHEET_STRESS
    assert payload["primary_balance_sheet_caution"] == "BALANCE_SHEET_HEADWIND"
    assert "HIGH_BALANCE_SHEET_STRESS_HEADWIND" in payload["balance_sheet_stress_reason_codes"]


def test_high_refinancing_risk_classification():
    payload = compute_balance_sheet_stress(
        "AAA",
        "2026-03-01",
        fundamentals=_balance_rows(
            net_debt=[100, 105, 110, 115, 120],
            total_debt=[110, 115, 120, 125, 130],
            cash=[4, 4, 4, 4, 4],
            cfo=[18, 18, 19, 19, 20],
            owner_earnings=[14, 15, 15, 16, 16],
        ),
        owner_quality_payload={"oe_quality_reason_codes": ["EXCESS_DILUTION"]},
        intangible_payload={"balance_sheet_optionality_score": 1.0},
        facts_status="OK",
    )
    assert payload["refinancing_risk_class"] == HIGH_REFINANCING_RISK
    assert "HIGH_REFINANCING_RISK_HEADWIND" in payload["refinancing_risk_reason_codes"]


def test_unknown_balance_sheet_stress_when_evidence_thin():
    payload = compute_balance_sheet_stress(
        "AAA",
        "2026-03-01",
        fundamentals={"rows": [{"year": 2025, "revenue": 100.0}]},
        facts_status="UNKNOWN",
    )
    assert payload["balance_sheet_stress_class"] == BALANCE_SHEET_STRESS_UNKNOWN
    assert "BALANCE_SHEET_EVIDENCE_THIN" in payload["balance_sheet_stress_reason_codes"]


def test_blocked_facts_withhold_the_class_even_when_signals_fired():
    """Blocked facts (a status other than OK) on an input this module reads make both
    classes UNKNOWN whatever fired; it used to apply only when no signal fired, so a
    strong-liquidity series on blocked facts came back LOW. Signals stay listed."""
    payload = compute_balance_sheet_stress(
        "AAA",
        "2026-03-01",
        fundamentals=_balance_rows(
            net_debt=[8, 8, 9, 10, 10],
            total_debt=[45, 46, 47, 48, 50],
            cash=[38, 38, 39, 39, 40],
            cfo=[18, 19, 20, 21, 22],
            owner_earnings=[16, 17, 18, 19, 20],
        ),
        intangible_payload={"balance_sheet_optionality_score": 4.5},
        facts_status="BLOCKED",
    )
    assert payload["balance_sheet_stress_class"] == BALANCE_SHEET_STRESS_UNKNOWN
    assert payload["refinancing_risk_class"] == "REFINANCING_RISK_UNKNOWN"
    assert payload["primary_balance_sheet_caution"] == "BALANCE_SHEET_UNCLEAR"
    assert "BALANCE_SHEET_EVIDENCE_THIN" in payload["balance_sheet_stress_reason_codes"]
    assert "LOW_NET_DEBT_TO_CFO" in payload["balance_sheet_support_signals"]


def test_readiness_applies_balance_sheet_headwind_honestly():
    balance_payload = {
        "balance_sheet_stress_class": HIGH_BALANCE_SHEET_STRESS,
        "balance_sheet_stress_reason_codes": [
            "HIGH_BALANCE_SHEET_STRESS_HEADWIND",
            "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE",
        ],
        "refinancing_risk_class": HIGH_REFINANCING_RISK,
        "refinancing_risk_reason_codes": ["HIGH_REFINANCING_RISK_HEADWIND"],
        "balance_sheet_headwind_signals": [
            "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE",
            "REFINANCING_DEPENDENCE_HEADWIND",
        ],
        "primary_balance_sheet_caution": "BALANCE_SHEET_HEADWIND",
        "balance_sheet_discipline_summary": "capital structure dominates downside",
        "derived_from": ["balance.fixture"],
    }
    readiness = compute_investment_readiness(
        "AAA",
        "2026-03-01",
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
        owner_quality_payload={"oe_quality_total": 7.0, "capital_allocation_score": 3.0, "oe_quality_reason_codes": []},
        intangible_payload={"intangible_economics_total": 6.0, "owner_value_capture_score": 3.0, "owner_value_capture_reason_codes": []},
        balance_sheet_stress_payload=balance_payload,
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
    )
    assert BLOCKER_HIGH_BALANCE_SHEET_STRESS in readiness["blocker_stack_all"]
    assert BLOCKER_HIGH_REFINANCING_RISK in readiness["blocker_stack_all"]
    assert "HIGH_BALANCE_SHEET_STRESS_HEADWIND" in readiness["readiness_support_headwinds"]


def test_valuation_confidence_degrades_when_capital_structure_dominates_downside():
    balance_payload = {
        "balance_sheet_stress_class": HIGH_BALANCE_SHEET_STRESS,
        "balance_sheet_stress_reason_codes": [
            "HIGH_BALANCE_SHEET_STRESS_HEADWIND",
            "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE",
        ],
        "refinancing_risk_class": HIGH_REFINANCING_RISK,
        "refinancing_risk_reason_codes": ["HIGH_REFINANCING_RISK_HEADWIND"],
        "balance_sheet_headwind_signals": ["CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE"],
        "derived_from": ["balance.fixture"],
    }
    confidence = compute_valuation_confidence(
        "AAA",
        "2026-03-01",
        intrinsic_payload=_intrinsic_payload(mos_to_floor=0.35, mos_classification="ADEQUATE_MARGIN_OF_SAFETY"),
        balance_sheet_stress_payload=balance_payload,
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
        epv_per_share=80.0,
        epv_refs=["epv.fixture"],
        netnet_per_share=60.0,
        netnet_refs=["netnet.fixture"],
        existing_intrinsic_base=90.0,
        existing_intrinsic_base_refs=["intrinsic.fixture"],
        existing_intrinsic_conservative=70.0,
        existing_intrinsic_conservative_refs=["intrinsic.fixture.conservative"],
    )
    assert confidence["valuation_confidence_class"] == "LOW_CONFIDENCE"
    assert "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE" in confidence["valuation_confidence_reason_codes"]


def test_memo_pack_includes_balance_sheet_stress_section():
    memo = _memo_markdown(
        {
            "header": {"ticker": "AAA", "as_of_date": "2026-03-01"},
            "balance_sheet_stress_refinancing_risk": {
                "balance_sheet_stress_class": LOW_BALANCE_SHEET_STRESS,
                "refinancing_risk_class": "LOW_REFINANCING_RISK",
                "primary_balance_sheet_caution": "BALANCE_SHEET_SUPPORTIVE",
                "balance_sheet_stress_reason_codes": ["LOW_BALANCE_SHEET_STRESS_SUPPORT"],
                "refinancing_risk_reason_codes": ["LOW_NET_DEBT_TO_CFO"],
                "balance_sheet_support_signals": ["LOW_NET_DEBT_TO_CFO"],
                "balance_sheet_headwind_signals": [],
                "balance_sheet_discipline_summary": "capital structure is supportive",
                "derived_from": ["balance.fixture"],
            },
        }
    )
    assert "## Balance Sheet Stress / Refinancing Risk" in memo
    assert LOW_BALANCE_SHEET_STRESS in memo


def test_promotion_and_escalation_surface_balance_sheet_without_overriding_fail():
    master_shortlist = {
        "campaign_run_id": "camp_balance",
        "rows": [
            {
                "ticker": "AAA",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "camp_balance__core"}],
                "best_rank_seen": 1,
                "value_gate_status": "WATCH",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": 0.18,
                "primary_blocker": "MISSING_FCF",
                "latest_primary_blocker": "MISSING_FCF",
                "balance_sheet_stress_class": LOW_BALANCE_SHEET_STRESS,
                "balance_sheet_stress_reason_codes": ["LOW_BALANCE_SHEET_STRESS_SUPPORT"],
                "refinancing_risk_class": "LOW_REFINANCING_RISK",
                "refinancing_risk_reason_codes": ["LOW_NET_DEBT_TO_CFO"],
                "balance_sheet_support_signals": ["LOW_NET_DEBT_TO_CFO"],
                "balance_sheet_headwind_signals": [],
                "primary_balance_sheet_caution": "BALANCE_SHEET_SUPPORTIVE",
                "balance_sheet_discipline_summary": "supportive",
                "memo_path": "memo/AAA.md",
            },
            {
                "ticker": "BBB",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "camp_balance__core"}],
                "best_rank_seen": 2,
                "value_gate_status": "FAIL",
                "latest_value_gate_status": "FAIL",
                "implied_return_base": 0.01,
                "primary_blocker": "PRICE_UNKNOWN",
                "latest_primary_blocker": "PRICE_UNKNOWN",
                "balance_sheet_stress_class": HIGH_BALANCE_SHEET_STRESS,
                "balance_sheet_stress_reason_codes": ["HIGH_BALANCE_SHEET_STRESS_HEADWIND"],
                "refinancing_risk_class": HIGH_REFINANCING_RISK,
                "refinancing_risk_reason_codes": ["HIGH_REFINANCING_RISK_HEADWIND"],
                "balance_sheet_support_signals": [],
                "balance_sheet_headwind_signals": ["REFINANCING_DEPENDENCE_HEADWIND"],
                "primary_balance_sheet_caution": "BALANCE_SHEET_HEADWIND",
                "balance_sheet_discipline_summary": "headwind",
                "memo_path": "memo/BBB.md",
            },
        ],
    }
    master_watchlist_state = {
        "campaign_run_id": "camp_balance",
        "tickers": {
            "AAA": {"appearances_count": 2, "latest_value_gate_status": "WATCH", "latest_primary_blocker": "MISSING_FCF", "history": []},
            "BBB": {"appearances_count": 2, "latest_value_gate_status": "FAIL", "latest_primary_blocker": "PRICE_UNKNOWN", "history": []},
        },
    }
    promotion_state = build_promotion_state("camp_balance", master_watchlist_state, master_shortlist)
    row_bbb = next(row for row in promotion_state["rows"] if row["ticker"] == "BBB")
    assert row_bbb["balance_sheet_stress_class"] == HIGH_BALANCE_SHEET_STRESS
    assert row_bbb["priority_lane"] == LANE_4_DEPRIORITIZED

    lanes = {
        "lane_1_high_priority": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_1_HIGH_PRIORITY"],
        "lane_2_research_queue": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_2_RESEARCH_QUEUE"],
        "lane_3_monitor": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_3_MONITOR"],
        "lane_4_deprioritized": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_4_DEPRIORITIZED"],
    }
    escalation = build_escalation_plan("camp_balance", promotion_state, lanes)
    assert any("balance_sheet_stress_class" in entry for entry in escalation["queue"])


def test_balance_sheet_cli_open_and_value_first_balance_sheet_sort(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    detail_low = compute_balance_sheet_stress(
        "AAA",
        "2026-03-01",
        fundamentals=_balance_rows(),
        intangible_payload={"balance_sheet_optionality_score": 4.5},
        facts_status="OK",
    )
    detail_high = compute_balance_sheet_stress(
        "BBB",
        "2026-03-01",
        fundamentals=_balance_rows(
            net_debt=[100, 105, 110, 115, 120],
            total_debt=[110, 115, 120, 125, 130],
            cash=[4, 4, 4, 4, 4],
            cfo=[18, 18, 19, 19, 20],
            owner_earnings=[14, 15, 15, 16, 16],
        ),
        intangible_payload={"balance_sheet_optionality_score": 1.0},
        facts_status="OK",
    )
    run_id = "balance_sheet_stress_test_open"
    output_path = cfg.outputs_dir / "universe" / run_id / "balance_sheet_stress.json"
    write_balance_sheet_stress_for_run(
        run_id=run_id,
        as_of_date="2026-03-01",
        tickers=["AAA", "BBB"],
        output_path=output_path,
        scoreboard_rows=[
            {"ticker": "AAA", "balance_sheet_stress_detail": detail_low},
            {"ticker": "BBB", "balance_sheet_stress_detail": detail_high},
        ],
    )

    result = runner.invoke(app, ["universe-balance-sheet-stress-open", "--run-id", run_id])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "OK"
    assert payload["counts_by_balance_sheet_stress_class"][LOW_BALANCE_SHEET_STRESS] == 1
    assert payload["counts_by_balance_sheet_stress_class"][HIGH_BALANCE_SHEET_STRESS] == 1

    rows = [
        {
            "ticker": "AAA",
            "scout_status": "WATCH",
            "investment_readiness_class": "RESEARCH_WORTHY_NOT_READY",
            "mos_to_floor": 0.20,
            "valuation_confidence_class": "MEDIUM_CONFIDENCE",
            "valuation_integrity_class": "INTEGRITY_OK",
            "balance_sheet_stress_class": LOW_BALANCE_SHEET_STRESS,
            "refinancing_risk_class": "LOW_REFINANCING_RISK",
            "accounting_quality_class": "MODERATE_ACCOUNTING_QUALITY",
            "reinvestment_efficiency_class": "MODERATE_REINVESTMENT_EFFICIENCY",
            "capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION",
            "value_type_primary": "EARNINGS_POWER_VALUE",
            "normalization_credibility_class": "MODERATE_NORMALIZATION_CREDIBILITY",
            "oe_quality_total": 7.0,
            "intangible_economics_total": 6.0,
        },
        {
            "ticker": "BBB",
            "scout_status": "WATCH",
            "investment_readiness_class": "RESEARCH_WORTHY_NOT_READY",
            "mos_to_floor": 0.20,
            "valuation_confidence_class": "MEDIUM_CONFIDENCE",
            "valuation_integrity_class": "INTEGRITY_OK",
            "balance_sheet_stress_class": HIGH_BALANCE_SHEET_STRESS,
            "refinancing_risk_class": HIGH_REFINANCING_RISK,
            "accounting_quality_class": "MODERATE_ACCOUNTING_QUALITY",
            "reinvestment_efficiency_class": "MODERATE_REINVESTMENT_EFFICIENCY",
            "capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION",
            "value_type_primary": "EARNINGS_POWER_VALUE",
            "normalization_credibility_class": "MODERATE_NORMALIZATION_CREDIBILITY",
            "oe_quality_total": 7.0,
            "intangible_economics_total": 6.0,
        },
    ]
    ordered = sorted(rows, key=lambda row: ranking_sort_key(row, policy="value_first_balance_sheet"))
    assert [row["ticker"] for row in ordered] == ["AAA", "BBB"]


# ── units: net debt in $millions against a cash-flow series in whole dollars ──
# The universe scout hands
# in rows from app/valuation/owner_earnings.py, which reports companyfacts
# values unscaled, while net debt arrives from app/valuation/net_debt.py in
# $millions. The ratio came out a millionfold too small, so a levered issuer
# collected the LOW_NET_DEBT_TO_CFO *support* signal. The caller now declares
# the units of the rows it holds.


def _levered_scout_rows() -> dict:
    """Net debt $2.0bn against CFO $500m and owner earnings $380m — 4.0x."""
    years = [2021, 2022, 2023, 2024, 2025]
    return {
        "ticker": "AAA",
        "rows": [
            {"year": year, "cfo": 500_000_000.0, "owner_earnings": 380_000_000.0}
            for year in years
        ],
        "derived_from": ["owner_earnings.fixture"],
    }


def test_dollar_cashflow_rows_are_scaled_onto_the_net_debt_units():
    payload = compute_balance_sheet_stress(
        "AAA",
        "2026-03-01",
        fundamentals=_levered_scout_rows(),
        net_debt_proxy=2_000.0,
        total_debt=2_200.0,
        cash_equivalents=200.0,
        fundamentals_cashflow_units=CASHFLOW_UNITS_USD,
        facts_status="OK",
    )
    assert payload["net_debt_to_cfo"] == 4.0
    assert payload["net_debt_to_owner_earnings"] == 2_000.0 / 380.0
    assert "LOW_NET_DEBT_TO_CFO" not in payload["balance_sheet_support_signals"]
    assert "HIGH_NET_DEBT_TO_CFO" in payload["balance_sheet_headwind_signals"]
    assert payload["balance_sheet_stress_class"] != LOW_BALANCE_SHEET_STRESS


def test_the_same_company_classifies_the_same_in_either_unit():
    """Declaring dollars or millions must not change the verdict."""
    dollars = compute_balance_sheet_stress(
        "AAA",
        "2026-03-01",
        fundamentals=_levered_scout_rows(),
        net_debt_proxy=2_000.0,
        total_debt=2_200.0,
        cash_equivalents=200.0,
        fundamentals_cashflow_units=CASHFLOW_UNITS_USD,
        facts_status="OK",
    )
    millions = compute_balance_sheet_stress(
        "AAA",
        "2026-03-01",
        fundamentals={
            "ticker": "AAA",
            "rows": [
                {"year": year, "cfo": 500.0, "owner_earnings": 380.0}
                for year in [2021, 2022, 2023, 2024, 2025]
            ],
            "derived_from": ["owner_earnings.fixture"],
        },
        net_debt_proxy=2_000.0,
        total_debt=2_200.0,
        cash_equivalents=200.0,
        fundamentals_cashflow_units=CASHFLOW_UNITS_USD_MILLIONS,
        facts_status="OK",
    )
    assert dollars["net_debt_to_cfo"] == millions["net_debt_to_cfo"]
    assert dollars["balance_sheet_stress_class"] == millions["balance_sheet_stress_class"]
    assert dollars["refinancing_risk_class"] == millions["refinancing_risk_class"]


def test_millions_rows_are_untouched_by_default():
    """The deep-dive path passes rows already in $millions and must not move."""
    payload = compute_balance_sheet_stress(
        "AAA",
        "2026-03-01",
        fundamentals=_balance_rows(
            net_debt=[8, 8, 9, 10, 10],
            total_debt=[45, 46, 47, 48, 50],
            cash=[38, 38, 39, 39, 40],
            cfo=[18, 19, 20, 21, 22],
            owner_earnings=[16, 17, 18, 19, 20],
        ),
        intangible_payload={"balance_sheet_optionality_score": 4.5},
        facts_status="OK",
    )
    assert payload["net_debt_to_cfo"] == 10.0 / 22.0
    assert payload["balance_sheet_stress_class"] == LOW_BALANCE_SHEET_STRESS


def _burner_rows(*, cfo: float, owner_earnings: float, net_debt: float = 500.0) -> dict:
    return {
        "rows": [
            {
                "year": 2025,
                "net_debt": net_debt,
                "cfo": cfo,
                "owner_earnings": owner_earnings,
                "total_debt": 800.0,
                "cash": 300.0,
            }
        ]
    }


def test_positive_net_debt_against_negative_cash_flow_is_high_stress_with_a_reason():
    """Net debt 500 against cash from operations -50: no ratio exists, and that is the
    worst case, not an unformable one (was MODERATE with no leverage headwind)."""
    payload = compute_balance_sheet_stress(
        "BURN",
        "2026-01-01",
        fundamentals=_burner_rows(cfo=-50.0, owner_earnings=-60.0),
        facts_status="OK",
    )
    assert payload["balance_sheet_stress_class"] == HIGH_BALANCE_SHEET_STRESS
    assert payload["primary_balance_sheet_caution"] == "BALANCE_SHEET_HEADWIND"
    assert payload["net_debt_to_cfo"] == "UNKNOWN"  # still no invented multiple
    assert payload["net_debt_to_owner_earnings"] == "UNKNOWN"
    headwinds = payload["balance_sheet_headwind_signals"]
    assert "NET_DEBT_WITH_NON_POSITIVE_CASH_FLOW" in headwinds
    assert "HIGH_NET_DEBT_TO_CFO" in headwinds
    assert "HIGH_LEVERAGE_TO_OWNER_EARNINGS" in headwinds
    assert "NET_DEBT_WITH_NON_POSITIVE_CASH_FLOW" in payload["balance_sheet_stress_reason_codes"]
    assert "LOW_NET_DEBT_TO_CFO" not in payload["balance_sheet_support_signals"]


def test_zero_cash_flow_is_treated_like_negative_cash_flow():
    payload = compute_balance_sheet_stress(
        "ZERO",
        "2026-01-01",
        fundamentals=_burner_rows(cfo=0.0, owner_earnings=0.0),
        facts_status="OK",
    )
    assert payload["balance_sheet_stress_class"] == HIGH_BALANCE_SHEET_STRESS


def test_net_cash_with_negative_cash_flow_is_not_leverage_stress():
    """No net debt to service: a cash burn alone is not this module's leverage headwind."""
    payload = compute_balance_sheet_stress(
        "CASHRICH",
        "2026-01-01",
        fundamentals=_burner_rows(cfo=-50.0, owner_earnings=-60.0, net_debt=-400.0),
        facts_status="OK",
    )
    assert payload["balance_sheet_stress_class"] != HIGH_BALANCE_SHEET_STRESS
    assert "NET_DEBT_WITH_NON_POSITIVE_CASH_FLOW" not in payload["balance_sheet_headwind_signals"]


def test_a_negative_ratio_from_the_caller_is_not_read_as_low_leverage():
    """A caller that divides net debt by a negative cash flow hands in -10.0; that is not
    a leverage multiple below 1.5, so it must not collect the low-leverage support."""
    payload = compute_balance_sheet_stress(
        "NEGRATIO",
        "2026-01-01",
        net_debt_proxy=500.0,
        total_debt=800.0,
        cash_equivalents=300.0,
        net_debt_to_cfo=-10.0,
        facts_status="OK",
    )
    assert "LOW_NET_DEBT_TO_CFO" not in payload["balance_sheet_support_signals"]
    assert payload["balance_sheet_stress_class"] == HIGH_BALANCE_SHEET_STRESS


def test_unknown_cash_flow_stays_unknown_not_high():
    """No cash-flow row at all is missing evidence, not evidence of a burn."""
    payload = compute_balance_sheet_stress(
        "NOCFO",
        "2026-01-01",
        net_debt_proxy=500.0,
        total_debt=800.0,
        cash_equivalents=300.0,
        facts_status="OK",
    )
    assert "NET_DEBT_WITH_NON_POSITIVE_CASH_FLOW" not in payload["balance_sheet_headwind_signals"]
    assert payload["balance_sheet_stress_class"] != HIGH_BALANCE_SHEET_STRESS
