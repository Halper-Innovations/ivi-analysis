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
    BLOCKER_HIGH_REVENUE_DEPENDENCE_RISK,
    compute_investment_readiness,
)
from app.valuation.revenue_dependence import (
    HIGH_REVENUE_DEPENDENCE_RISK,
    LOW_REVENUE_DEPENDENCE_RISK,
    MODERATE_REVENUE_DEPENDENCE_RISK,
    REVENUE_DEPENDENCE_UNKNOWN,
    compute_revenue_dependence,
    write_revenue_dependence_for_run,
)
from app.valuation.returns_persistence import compute_returns_persistence
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


def _revenue_rows(
    *,
    customer_pct: float | None = None,
    channel_pct: float | None = None,
    end_market_pct: float | None = None,
    segment_count: int | None = None,
    channel_count: int | None = None,
    end_market_count: int | None = None,
) -> dict:
    row = {"year": 2025, "revenue": 100.0}
    if customer_pct is not None:
        row["customer_concentration_pct"] = customer_pct
    if channel_pct is not None:
        row["top_channel_pct"] = channel_pct
    if end_market_pct is not None:
        row["top_end_market_pct"] = end_market_pct
    if segment_count is not None:
        row["segment_count"] = segment_count
    if channel_count is not None:
        row["channel_count"] = channel_count
    if end_market_count is not None:
        row["end_market_count"] = end_market_count
    return {"rows": [row], "derived_from": ["revenue.fixture"]}


def _intrinsic_payload(*, mos_to_floor: float | str, mos_classification: str) -> dict:
    return {
        "mos_to_floor": mos_to_floor,
        "mos_to_base": 0.30 if isinstance(mos_to_floor, (int, float)) else "UNKNOWN",
        "mos_classification": mos_classification,
        "downside_support_type": "EARNINGS_POWER_SUPPORT",
        "normalized_earnings_power_status": "OK",
        "normalized_earnings_power_method_used": "FCF_SELECTED",
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
    }


def _integrity_payload() -> dict:
    return {
        "valuation_integrity_class": "INTEGRITY_OK",
        "valuation_integrity_reason_codes": [],
        "derived_from": ["integrity.fixture"],
    }


def test_low_revenue_dependence_for_diversified_case():
    payload = compute_revenue_dependence(
        "AAA",
        "2026-03-08",
        fundamentals=_revenue_rows(
            customer_pct=0.05,
            channel_pct=0.18,
            end_market_pct=0.22,
            segment_count=4,
            channel_count=3,
            end_market_count=4,
        ),
        facts_status="OK",
        shares_status="OK",
        price_status="OK",
    )
    assert payload["revenue_dependence_risk_class"] == LOW_REVENUE_DEPENDENCE_RISK
    assert payload["primary_revenue_dependence_caution"] == "REVENUE_BASE_SUPPORTIVE"


def test_moderate_revenue_dependence_for_mixed_case():
    payload = compute_revenue_dependence(
        "AAA",
        "2026-03-08",
        fundamentals=_revenue_rows(customer_pct=0.15, segment_count=2, channel_count=2, end_market_count=2),
        facts_status="OK",
        shares_status="OK",
        price_status="OK",
    )
    assert payload["revenue_dependence_risk_class"] == MODERATE_REVENUE_DEPENDENCE_RISK
    assert payload["primary_revenue_dependence_caution"] == "REVENUE_BASE_MIXED"


def test_high_revenue_dependence_for_single_customer_case():
    payload = compute_revenue_dependence(
        "AAA",
        "2026-03-08",
        fundamentals=_revenue_rows(customer_pct=0.36, channel_pct=0.52, end_market_pct=0.55, segment_count=1),
        facts_status="OK",
        shares_status="OK",
        price_status="OK",
    )
    assert payload["revenue_dependence_risk_class"] == HIGH_REVENUE_DEPENDENCE_RISK
    assert "CUSTOMER_CONCENTRATION_HEADWIND" in payload["revenue_dependence_risk_reason_codes"]
    assert "NARROW_CHANNEL_DEPENDENCE" in payload["revenue_dependence_headwind_signals"]


def test_unknown_revenue_dependence_when_evidence_thin():
    payload = compute_revenue_dependence(
        "AAA",
        "2026-03-08",
        fundamentals={"rows": [{"year": 2025, "revenue": 100.0}]},
        facts_status="UNKNOWN",
        shares_status="UNKNOWN",
        price_status="UNKNOWN",
    )
    assert payload["revenue_dependence_risk_class"] == REVENUE_DEPENDENCE_UNKNOWN
    assert "REVENUE_DEPENDENCE_UNKNOWN" in payload["revenue_dependence_risk_reason_codes"]


def test_readiness_applies_revenue_dependence_headwind_honestly():
    revenue_payload = {
        "revenue_dependence_risk_class": HIGH_REVENUE_DEPENDENCE_RISK,
        "revenue_dependence_risk_reason_codes": ["HIGH_REVENUE_DEPENDENCE_HEADWIND", "CUSTOMER_CONCENTRATION_HEADWIND"],
        "revenue_dependence_support_signals": [],
        "revenue_dependence_headwind_signals": ["SINGLE_CUSTOMER_CONCENTRATION", "REVENUE_BASE_FRAGILITY"],
        "primary_revenue_dependence_caution": "REVENUE_BASE_HEADWIND",
        "revenue_fragility_summary": "revenue depends too heavily on one customer",
        "derived_from": ["revenue.fixture"],
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
        revenue_dependence_payload=revenue_payload,
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
    )
    assert BLOCKER_HIGH_REVENUE_DEPENDENCE_RISK in readiness["blocker_stack_all"]
    assert "HIGH_REVENUE_DEPENDENCE_HEADWIND" in readiness["readiness_support_headwinds"]


def test_value_type_and_returns_use_revenue_dependence_as_refinement_not_override():
    high_revenue_dependence = {
        "revenue_dependence_risk_class": HIGH_REVENUE_DEPENDENCE_RISK,
        "revenue_dependence_risk_reason_codes": ["HIGH_REVENUE_DEPENDENCE_HEADWIND"],
        "revenue_dependence_support_signals": [],
        "revenue_dependence_headwind_signals": ["SINGLE_CUSTOMER_CONCENTRATION"],
        "primary_revenue_dependence_caution": "REVENUE_BASE_HEADWIND",
        "revenue_fragility_summary": "fragile",
        "derived_from": ["revenue.high"],
    }
    value_type = compute_value_type(
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
        returns_persistence_payload={
            "returns_persistence_class": "HIGH_RETURNS_PERSISTENCE",
            "returns_persistence_reason_codes": ["HIGH_RETURNS_PERSISTENCE_SUPPORT"],
            "returns_support_signals": ["HIGH_RETURN_ON_CAPITAL_PRESENT"],
            "returns_headwind_signals": [],
            "primary_returns_caution": "RETURNS_DURABILITY_SUPPORTIVE",
        },
        revenue_dependence_payload=high_revenue_dependence,
    )
    assert value_type["value_type_primary"] != "QUALITY_VALUE"
    assert "HIGH_REVENUE_DEPENDENCE_HEADWIND" in value_type["value_type_reason_codes"]

    returns_payload = compute_returns_persistence(
        "AAA",
        "2026-03-08",
        fundamentals={
            "rows": [
                {"year": 2021, "revenue": 100, "invested_capital": 80, "roic_proxy": 0.17, "roe_proxy": 0.18, "roa_proxy": 0.09},
                {"year": 2022, "revenue": 110, "invested_capital": 87, "roic_proxy": 0.18, "roe_proxy": 0.19, "roa_proxy": 0.09},
                {"year": 2023, "revenue": 121, "invested_capital": 95, "roic_proxy": 0.18, "roe_proxy": 0.19, "roa_proxy": 0.10},
                {"year": 2024, "revenue": 133, "invested_capital": 104, "roic_proxy": 0.19, "roe_proxy": 0.20, "roa_proxy": 0.10},
                {"year": 2025, "revenue": 146, "invested_capital": 113, "roic_proxy": 0.18, "roe_proxy": 0.19, "roa_proxy": 0.09},
            ],
            "derived_from": ["returns.fixture"],
        },
        intangible_payload={"gross_margin_durability_score": 4.5, "cycle_resilience_score": 4.0, "owner_value_capture_score": 4.0},
        reinvestment_efficiency_payload={"reinvestment_efficiency_class": "HIGH_REINVESTMENT_EFFICIENCY"},
        capital_allocation_discipline_payload={"capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED"},
        revenue_dependence_payload=high_revenue_dependence,
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    assert returns_payload["returns_persistence_class"] == "MODERATE_RETURNS_PERSISTENCE"
    assert "HIGH_REVENUE_DEPENDENCE_HEADWIND" in returns_payload["returns_persistence_reason_codes"]


def test_memo_pack_includes_revenue_dependence_section():
    memo = _memo_markdown(
        {
            "header": {"ticker": "AAA", "as_of_date": "2026-03-08"},
            "customer_concentration_revenue_dependence_risk": {
                "revenue_dependence_risk_class": HIGH_REVENUE_DEPENDENCE_RISK,
                "primary_revenue_dependence_caution": "REVENUE_BASE_HEADWIND",
                "revenue_dependence_risk_reason_codes": ["HIGH_REVENUE_DEPENDENCE_HEADWIND"],
                "revenue_dependence_support_signals": [],
                "revenue_dependence_headwind_signals": ["SINGLE_CUSTOMER_CONCENTRATION"],
                "revenue_fragility_summary": "revenue depends heavily on one customer",
                "derived_from": ["revenue.fixture"],
            },
        }
    )
    assert "## Customer Concentration / Revenue Dependence Risk" in memo
    assert HIGH_REVENUE_DEPENDENCE_RISK in memo


def test_promotion_and_escalation_surface_revenue_dependence_without_overriding_fail():
    master_shortlist = {
        "campaign_run_id": "camp_revenue",
        "rows": [
            {
                "ticker": "AAA",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "camp_revenue__core"}],
                "best_rank_seen": 1,
                "value_gate_status": "WATCH",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": 0.20,
                "primary_blocker": "MISSING_FCF",
                "latest_primary_blocker": "MISSING_FCF",
                "revenue_dependence_risk_class": LOW_REVENUE_DEPENDENCE_RISK,
                "revenue_dependence_risk_reason_codes": ["LOW_REVENUE_DEPENDENCE_SUPPORT"],
                "revenue_dependence_support_signals": ["DIVERSIFIED_REVENUE_BASE"],
                "revenue_dependence_headwind_signals": [],
                "primary_revenue_dependence_caution": "REVENUE_BASE_SUPPORTIVE",
                "revenue_fragility_summary": "diversified",
                "memo_path": "memo/AAA.md",
            },
            {
                "ticker": "BBB",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "camp_revenue__core"}],
                "best_rank_seen": 2,
                "value_gate_status": "FAIL",
                "latest_value_gate_status": "FAIL",
                "implied_return_base": 0.01,
                "primary_blocker": "PRICE_UNKNOWN",
                "latest_primary_blocker": "PRICE_UNKNOWN",
                "revenue_dependence_risk_class": HIGH_REVENUE_DEPENDENCE_RISK,
                "revenue_dependence_risk_reason_codes": ["HIGH_REVENUE_DEPENDENCE_HEADWIND", "CUSTOMER_CONCENTRATION_HEADWIND"],
                "revenue_dependence_support_signals": [],
                "revenue_dependence_headwind_signals": ["SINGLE_CUSTOMER_CONCENTRATION"],
                "primary_revenue_dependence_caution": "REVENUE_BASE_HEADWIND",
                "revenue_fragility_summary": "fragile",
                "memo_path": "memo/BBB.md",
            },
        ],
    }
    master_watchlist_state = {
        "campaign_run_id": "camp_revenue",
        "tickers": {
            "AAA": {"appearances_count": 2, "latest_value_gate_status": "WATCH", "latest_primary_blocker": "MISSING_FCF", "history": []},
            "BBB": {"appearances_count": 2, "latest_value_gate_status": "FAIL", "latest_primary_blocker": "PRICE_UNKNOWN", "history": []},
        },
    }
    promotion_state = build_promotion_state("camp_revenue", master_watchlist_state, master_shortlist)
    row_bbb = next(row for row in promotion_state["rows"] if row["ticker"] == "BBB")
    assert row_bbb["revenue_dependence_risk_class"] == HIGH_REVENUE_DEPENDENCE_RISK
    assert row_bbb["priority_lane"] == LANE_4_DEPRIORITIZED

    lanes = {
        "lane_1_high_priority": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_1_HIGH_PRIORITY"],
        "lane_2_research_queue": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_2_RESEARCH_QUEUE"],
        "lane_3_monitor": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_3_MONITOR"],
        "lane_4_deprioritized": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_4_DEPRIORITIZED"],
    }
    escalation = build_escalation_plan("camp_revenue", promotion_state, lanes)
    assert any("revenue_dependence_risk_class" in entry for entry in escalation["queue"])


def test_revenue_dependence_cli_open_and_value_first_revenue_resilience_sort(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    detail_low = compute_revenue_dependence(
        "AAA",
        "2026-03-08",
        fundamentals=_revenue_rows(customer_pct=0.05, segment_count=4, channel_count=3, end_market_count=4),
        facts_status="OK",
        shares_status="OK",
        price_status="OK",
    )
    detail_high = compute_revenue_dependence(
        "BBB",
        "2026-03-08",
        fundamentals=_revenue_rows(customer_pct=0.35, channel_pct=0.45, end_market_pct=0.60, segment_count=1),
        facts_status="OK",
        shares_status="OK",
        price_status="OK",
    )
    output_path = cfg.outputs_dir / "universe" / "revdep_run" / "revenue_dependence.json"
    write_revenue_dependence_for_run(
        run_id="revdep_run",
        as_of_date="2026-03-08",
        tickers=["AAA", "BBB"],
        output_path=output_path,
        scoreboard_rows=[
            {"ticker": "AAA", "revenue_dependence_detail": detail_low},
            {"ticker": "BBB", "revenue_dependence_detail": detail_high},
        ],
    )
    result = runner.invoke(app, ["universe-revenue-dependence-open", "--run-id", "revdep_run"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["counts_by_revenue_dependence_risk_class"][LOW_REVENUE_DEPENDENCE_RISK] == 1
    assert payload["counts_by_revenue_dependence_risk_class"][HIGH_REVENUE_DEPENDENCE_RISK] == 1

    better = {
        "scout_status": "WATCH",
        "investment_readiness_class": "RESEARCH_WORTHY_NOT_READY",
        "mos_to_floor": 0.30,
        "valuation_confidence_class": "MEDIUM_CONFIDENCE",
        "valuation_integrity_class": "INTEGRITY_OK",
        "returns_persistence_class": "HIGH_RETURNS_PERSISTENCE",
        "revenue_dependence_risk_class": LOW_REVENUE_DEPENDENCE_RISK,
        "accounting_quality_class": "HIGH_ACCOUNTING_QUALITY",
        "reinvestment_efficiency_class": "HIGH_REINVESTMENT_EFFICIENCY",
        "capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED",
        "value_type_primary": "QUALITY_VALUE",
        "normalization_credibility_class": "HIGH_NORMALIZATION_CREDIBILITY",
        "oe_quality_total": 8.0,
        "intangible_economics_total": 7.0,
        "ticker": "AAA",
    }
    worse = {**better, "ticker": "BBB", "revenue_dependence_risk_class": HIGH_REVENUE_DEPENDENCE_RISK}
    assert ranking_sort_key(better, policy="value_first_revenue_resilience") < ranking_sort_key(
        worse,
        policy="value_first_revenue_resilience",
    )
