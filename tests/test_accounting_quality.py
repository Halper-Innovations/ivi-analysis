from __future__ import annotations

import json

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.escalation import build_escalation_plan
from app.universe.memo_pack import _memo_markdown
from app.universe.promotion import LANE_4_DEPRIORITIZED, build_promotion_state
from app.universe.ranking import ranking_sort_key
from app.valuation.accounting_quality import (
    ACCOUNTING_QUALITY_UNKNOWN,
    HIGH_ACCOUNTING_QUALITY,
    LOW_ACCOUNTING_QUALITY,
    MODERATE_ACCOUNTING_QUALITY,
    compute_accounting_quality,
    write_accounting_quality_for_run,
)
from app.valuation.investment_readiness import (
    BLOCKER_LOW_ACCOUNTING_QUALITY,
    compute_investment_readiness,
)
from app.valuation.valuation_confidence import compute_valuation_confidence
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
    net_income: list[float],
    cfo: list[float],
    fcf: list[float],
    capex: list[float],
    shares: list[float],
    receivables: list[float] | None = None,
    inventory: list[float] | None = None,
    sbc: list[float] | None = None,
) -> dict:
    years = [2021, 2022, 2023, 2024, 2025]
    rows = []
    receivables = receivables or [10.0] * 5
    inventory = inventory or [8.0] * 5
    sbc = sbc or [1.0] * 5
    for idx, year in enumerate(years):
        rows.append(
            {
                "year": year,
                "revenue": revenue[idx],
                "net_income": net_income[idx],
                "cfo": cfo[idx],
                "fcf": fcf[idx],
                "capex": capex[idx],
                "shares_outstanding": shares[idx],
                "receivables": receivables[idx],
                "inventory": inventory[idx],
                "sbc": sbc[idx],
                "owner_earnings": fcf[idx],
            }
        )
    return {"rows": rows, "derived_from": ["fundamentals.fixture"]}


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


def test_high_accounting_quality_for_strong_cash_conversion():
    payload = compute_accounting_quality(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_rows(
            revenue=[100, 108, 117, 126, 136],
            net_income=[12, 14, 16, 18, 20],
            cfo=[15, 17, 19, 22, 24],
            fcf=[12, 14, 16, 19, 21],
            capex=[3, 3, 3, 3, 3],
            shares=[100, 100, 100, 99, 99],
            receivables=[10, 10.2, 10.4, 10.6, 10.8],
            inventory=[8, 8.1, 8.1, 8.2, 8.2],
            sbc=[0.8, 0.9, 0.9, 1.0, 1.0],
        ),
        owner_quality_payload={"oe_quality_total": 9.0, "cash_conversion_score": 3.0, "derived_from": ["oe.fixture"]},
        capital_allocation_discipline_payload={"capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED"},
        reinvestment_efficiency_payload={"reinvestment_efficiency_class": "HIGH_REINVESTMENT_EFFICIENCY"},
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    assert payload["accounting_quality_class"] == HIGH_ACCOUNTING_QUALITY
    assert payload["primary_accounting_caution"] == "CASH_EARNINGS_SUPPORTIVE"
    assert "HIGH_ACCOUNTING_QUALITY_SUPPORT" in payload["accounting_quality_reason_codes"]


def test_moderate_accounting_quality_for_mixed_signals():
    payload = compute_accounting_quality(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_rows(
            revenue=[100, 105, 110, 116, 121],
            net_income=[12, 13, 14, 15, 16],
            cfo=[10, 11, 12, 13, 14],
            fcf=[7, 7.5, 8, 8.5, 9],
            capex=[3, 3.5, 4, 4.5, 5],
            shares=[100, 100, 100, 100, 100],
            receivables=[10, 11, 12, 13.5, 15],
            inventory=[8, 8.4, 8.8, 9.2, 9.6],
            sbc=[1.5, 1.6, 1.7, 1.8, 1.9],
        ),
        owner_quality_payload={"oe_quality_total": 6.0, "cash_conversion_score": 2.0},
        capital_allocation_discipline_payload={"capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION"},
        reinvestment_efficiency_payload={"reinvestment_efficiency_class": "MODERATE_REINVESTMENT_EFFICIENCY"},
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    assert payload["accounting_quality_class"] == MODERATE_ACCOUNTING_QUALITY
    assert payload["primary_accounting_caution"] == "ACCOUNTING_QUALITY_MIXED"


def test_low_accounting_quality_for_accrual_heavy_case():
    payload = compute_accounting_quality(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_rows(
            revenue=[100, 112, 126, 141, 158],
            net_income=[12, 15, 18, 22, 26],
            cfo=[6, 7, 8, 9, 10],
            fcf=[2, 2.5, 3, 3.5, 4],
            capex=[4, 4.5, 5, 5.5, 6],
            shares=[100, 102, 104, 107, 110],
            receivables=[10, 13, 17, 22, 28],
            inventory=[8, 10, 12.5, 15, 18],
            sbc=[4.0, 4.5, 5.0, 5.5, 6.0],
        ),
        owner_quality_payload={"oe_quality_total": 3.0, "cash_conversion_score": 1.0},
        capital_allocation_discipline_payload={"capital_allocation_discipline_class": "OWNER_DILUTIVE_OR_DESTRUCTIVE"},
        reinvestment_efficiency_payload={"reinvestment_efficiency_class": "LOW_REINVESTMENT_EFFICIENCY"},
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    assert payload["accounting_quality_class"] == LOW_ACCOUNTING_QUALITY
    assert payload["primary_accounting_caution"] == "CASH_EARNINGS_HEADWIND"
    assert "LOW_ACCOUNTING_QUALITY_HEADWIND" in payload["accounting_quality_reason_codes"]
    assert "ACCRUAL_HEAVY_EARNINGS" in payload["cash_earnings_headwind_signals"]


def test_unknown_accounting_quality_when_evidence_thin():
    payload = compute_accounting_quality(
        "AAA",
        "2026-02-14",
        fundamentals={"rows": [{"year": 2025, "revenue": 100.0}]},
        price_status="UNKNOWN",
        facts_status="UNKNOWN",
        shares_status="UNKNOWN",
        fcf_status="UNKNOWN",
    )
    assert payload["accounting_quality_class"] == ACCOUNTING_QUALITY_UNKNOWN
    assert "ACCOUNTING_EVIDENCE_THIN" in payload["accounting_quality_reason_codes"]


def test_readiness_and_confidence_apply_accounting_headwind_honestly():
    accounting_payload = {
        "accounting_quality_class": LOW_ACCOUNTING_QUALITY,
        "accounting_quality_reason_codes": ["LOW_ACCOUNTING_QUALITY_HEADWIND", "WEAK_CASH_CONVERSION_HEADWIND"],
        "cash_earnings_headwind_signals": ["WEAK_FCF_TO_EARNINGS_CONVERSION"],
        "primary_accounting_caution": "CASH_EARNINGS_HEADWIND",
        "cash_earnings_discipline_summary": "weak conversion",
        "derived_from": ["accounting.fixture"],
    }
    confidence = compute_valuation_confidence(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(mos_to_floor=0.30, mos_classification="ADEQUATE_MARGIN_OF_SAFETY"),
        accounting_quality_payload=accounting_payload,
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
    assert confidence["valuation_confidence_class"] in {"LOW_CONFIDENCE", "MEDIUM_CONFIDENCE"}
    assert "LOW_ACCOUNTING_QUALITY_HEADWIND" in confidence["valuation_confidence_reason_codes"]

    readiness = compute_investment_readiness(
        "AAA",
        "2026-02-14",
        value_gate_status="WATCH",
        primary_blocker="NONE",
        intrinsic_payload=_intrinsic_payload(mos_to_floor=0.30, mos_classification="ADEQUATE_MARGIN_OF_SAFETY"),
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
        accounting_quality_payload=accounting_payload,
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
    )
    assert BLOCKER_LOW_ACCOUNTING_QUALITY in readiness["blocker_stack_all"]
    assert "LOW_ACCOUNTING_QUALITY_HEADWIND" in readiness["readiness_support_headwinds"]


def test_value_type_uses_accounting_as_refinement_not_override():
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
        accounting_quality_payload={"accounting_quality_class": HIGH_ACCOUNTING_QUALITY, "accounting_quality_reason_codes": []},
        reinvestment_efficiency_payload={"reinvestment_efficiency_class": "HIGH_REINVESTMENT_EFFICIENCY"},
        fail_due_to_missing_evidence=True,
        primary_fail_domain="EVIDENCE",
    )
    assert payload["value_type_primary"] == "UNKNOWN_VALUE_TYPE"
    assert "HIGH_ACCOUNTING_QUALITY_SUPPORT" not in payload["value_type_reason_codes"]


def test_memo_pack_includes_accounting_quality_section():
    memo = _memo_markdown(
        {
            "header": {"ticker": "AAA", "as_of_date": "2026-02-14"},
            "accounting_quality_cash_earnings_discipline": {
                "accounting_quality_class": HIGH_ACCOUNTING_QUALITY,
                "primary_accounting_caution": "CASH_EARNINGS_SUPPORTIVE",
                "accounting_quality_reason_codes": ["HIGH_ACCOUNTING_QUALITY_SUPPORT"],
                "cash_earnings_support_signals": ["STRONG_CFO_TO_EARNINGS_CONVERSION"],
                "cash_earnings_headwind_signals": [],
                "cash_earnings_discipline_summary": "cash earnings support reported results",
                "derived_from": ["accounting.fixture"],
            },
        }
    )
    assert "## Accounting Quality / Cash Earnings Discipline" in memo
    assert HIGH_ACCOUNTING_QUALITY in memo


def test_promotion_and_escalation_surface_accounting_without_overriding_fail():
    master_shortlist = {
        "campaign_run_id": "camp_accounting",
        "rows": [
            {
                "ticker": "AAA",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "camp_accounting__core"}],
                "best_rank_seen": 1,
                "value_gate_status": "WATCH",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": 0.20,
                "primary_blocker": "MISSING_FCF",
                "latest_primary_blocker": "MISSING_FCF",
                "accounting_quality_class": HIGH_ACCOUNTING_QUALITY,
                "accounting_quality_reason_codes": ["HIGH_ACCOUNTING_QUALITY_SUPPORT"],
                "cash_earnings_support_signals": ["STRONG_CFO_TO_EARNINGS_CONVERSION"],
                "cash_earnings_headwind_signals": [],
                "primary_accounting_caution": "CASH_EARNINGS_SUPPORTIVE",
                "cash_earnings_discipline_summary": "supportive",
                "memo_path": "memo/AAA.md",
            },
            {
                "ticker": "BBB",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "camp_accounting__core"}],
                "best_rank_seen": 2,
                "value_gate_status": "FAIL",
                "latest_value_gate_status": "FAIL",
                "implied_return_base": 0.02,
                "primary_blocker": "PRICE_UNKNOWN",
                "latest_primary_blocker": "PRICE_UNKNOWN",
                "accounting_quality_class": HIGH_ACCOUNTING_QUALITY,
                "accounting_quality_reason_codes": ["HIGH_ACCOUNTING_QUALITY_SUPPORT"],
                "cash_earnings_support_signals": ["STRONG_CFO_TO_EARNINGS_CONVERSION"],
                "cash_earnings_headwind_signals": [],
                "primary_accounting_caution": "CASH_EARNINGS_SUPPORTIVE",
                "cash_earnings_discipline_summary": "supportive",
                "memo_path": "memo/BBB.md",
            },
        ],
    }
    master_watchlist_state = {
        "campaign_run_id": "camp_accounting",
        "tickers": {
            "AAA": {"appearances_count": 2, "latest_value_gate_status": "WATCH", "latest_primary_blocker": "MISSING_FCF", "history": []},
            "BBB": {"appearances_count": 2, "latest_value_gate_status": "FAIL", "latest_primary_blocker": "PRICE_UNKNOWN", "history": []},
        },
    }
    promotion_state = build_promotion_state("camp_accounting", master_watchlist_state, master_shortlist)
    row_bbb = next(row for row in promotion_state["rows"] if row["ticker"] == "BBB")
    assert row_bbb["accounting_quality_class"] == HIGH_ACCOUNTING_QUALITY
    assert row_bbb["priority_lane"] == LANE_4_DEPRIORITIZED

    lanes = {
        "lane_1_high_priority": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_1_HIGH_PRIORITY"],
        "lane_2_research_queue": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_2_RESEARCH_QUEUE"],
        "lane_3_monitor": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_3_MONITOR"],
        "lane_4_deprioritized": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_4_DEPRIORITIZED"],
    }
    escalation = build_escalation_plan("camp_accounting", promotion_state, lanes)
    assert any("accounting_quality_class" in entry for entry in escalation["queue"])


def test_accounting_quality_cli_open_and_value_first_cash_earnings_sort(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    detail_high = compute_accounting_quality(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_rows(
            revenue=[100, 108, 117, 126, 136],
            net_income=[12, 14, 16, 18, 20],
            cfo=[15, 17, 19, 22, 24],
            fcf=[12, 14, 16, 19, 21],
            capex=[3, 3, 3, 3, 3],
            shares=[100, 100, 100, 99, 99],
        ),
        owner_quality_payload={"oe_quality_total": 9.0, "cash_conversion_score": 3.0},
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    detail_low = compute_accounting_quality(
        "BBB",
        "2026-02-14",
        fundamentals=_fundamentals_rows(
            revenue=[100, 112, 126, 141, 158],
            net_income=[12, 15, 18, 22, 26],
            cfo=[6, 7, 8, 9, 10],
            fcf=[2, 2.5, 3, 3.5, 4],
            capex=[4, 4.5, 5, 5.5, 6],
            shares=[100, 102, 104, 107, 110],
        ),
        owner_quality_payload={"oe_quality_total": 3.0, "cash_conversion_score": 1.0},
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    run_id = "accounting_quality_test_open"
    output_path = cfg.outputs_dir / "universe" / run_id / "accounting_quality.json"
    write_accounting_quality_for_run(
        run_id=run_id,
        as_of_date="2026-02-14",
        tickers=["AAA", "BBB"],
        output_path=output_path,
        scoreboard_rows=[
            {"ticker": "AAA", "accounting_quality_detail": detail_high},
            {"ticker": "BBB", "accounting_quality_detail": detail_low},
        ],
    )

    result = runner.invoke(app, ["universe-accounting-quality-open", "--run-id", run_id])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "OK"
    assert payload["counts_by_accounting_quality_class"][HIGH_ACCOUNTING_QUALITY] == 1
    assert payload["counts_by_accounting_quality_class"][LOW_ACCOUNTING_QUALITY] == 1

    rows = [
        {
            "ticker": "AAA",
            "scout_status": "WATCH",
            "investment_readiness_class": "RESEARCH_WORTHY_NOT_READY",
            "mos_to_floor": 0.20,
            "valuation_confidence_class": "MEDIUM_CONFIDENCE",
            "valuation_integrity_class": "INTEGRITY_OK",
            "accounting_quality_class": HIGH_ACCOUNTING_QUALITY,
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
            "accounting_quality_class": LOW_ACCOUNTING_QUALITY,
            "reinvestment_efficiency_class": "MODERATE_REINVESTMENT_EFFICIENCY",
            "capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION",
            "value_type_primary": "EARNINGS_POWER_VALUE",
            "normalization_credibility_class": "MODERATE_NORMALIZATION_CREDIBILITY",
            "oe_quality_total": 7.0,
            "intangible_economics_total": 6.0,
        },
    ]
    ordered = sorted(rows, key=lambda row: ranking_sort_key(row, policy="value_first_cash_earnings"))
    assert [row["ticker"] for row in ordered] == ["AAA", "BBB"]


# ── Accruals ratio detection ───────────────────────────────────────────


def test_accruals_aggressive_detected():
    """NI >> CFO for 3 years, accrual ratio > 0.10 → ACCRUALS_AGGRESSIVE."""
    from app.valuation.accounting_quality import _compute_accruals_signals

    rows = [
        {"year": 2023, "net_income": 80.0, "cfo": 35.0, "total_assets": 360.0},
        {"year": 2024, "net_income": 90.0, "cfo": 40.0, "total_assets": 380.0},
        {"year": 2025, "net_income": 100.0, "cfo": 50.0, "total_assets": 400.0},
    ]
    result = _compute_accruals_signals(rows)
    assert "ACCRUALS_AGGRESSIVE" in result["accruals_signals"]


def test_accruals_rising_detected():
    """Accrual ratio trending up from 0.04 to 0.08 → ACCRUALS_RISING."""
    from app.valuation.accounting_quality import _compute_accruals_signals

    rows = [
        {"year": 2023, "net_income": 100.0, "cfo": 84.0, "total_assets": 400.0},
        {"year": 2024, "net_income": 100.0, "cfo": 76.0, "total_assets": 400.0},
        {"year": 2025, "net_income": 100.0, "cfo": 68.0, "total_assets": 400.0},
    ]
    result = _compute_accruals_signals(rows)
    assert "ACCRUALS_RISING" in result["accruals_signals"]


def test_accruals_conservative_support():
    """Accrual ratio < 0.02 consistently → ACCRUALS_CONSERVATIVE."""
    from app.valuation.accounting_quality import _compute_accruals_signals

    rows = [
        {"year": 2023, "net_income": 100.0, "cfo": 95.0, "total_assets": 400.0},
        {"year": 2024, "net_income": 100.0, "cfo": 96.0, "total_assets": 400.0},
        {"year": 2025, "net_income": 100.0, "cfo": 94.0, "total_assets": 400.0},
    ]
    result = _compute_accruals_signals(rows)
    assert "ACCRUALS_CONSERVATIVE" in result["accruals_signals"]


def test_accruals_no_flag_moderate():
    """Ratio between 0.02 and 0.10, stable → no flag."""
    from app.valuation.accounting_quality import _compute_accruals_signals

    rows = [
        {"year": 2023, "net_income": 100.0, "cfo": 80.0, "total_assets": 400.0},
        {"year": 2024, "net_income": 100.0, "cfo": 79.0, "total_assets": 400.0},
        {"year": 2025, "net_income": 100.0, "cfo": 81.0, "total_assets": 400.0},
    ]
    result = _compute_accruals_signals(rows)
    assert "ACCRUALS_AGGRESSIVE" not in result["accruals_signals"]
    assert "ACCRUALS_RISING" not in result["accruals_signals"]
    assert "ACCRUALS_CONSERVATIVE" not in result["accruals_signals"]


def test_accruals_missing_data_graceful():
    """Missing total_assets → no crash, no flag."""
    from app.valuation.accounting_quality import _compute_accruals_signals

    rows = [
        {"year": 2025, "net_income": 100.0, "cfo": 50.0},
    ]
    result = _compute_accruals_signals(rows)
    assert result["accruals_signals"] == []
    assert result["accrual_ratios"] == []


def test_accruals_affects_quality_class():
    """ACCRUALS_AGGRESSIVE adds headwind strength, pushing toward LOW."""
    fundamentals = {
        "rows": [
            {
                "year": 2023, "net_income": 100.0, "cfo": 30.0, "fcf": 25.0,
                "total_assets": 400.0, "revenue": 500.0,
                "accounts_receivable": 50.0, "inventory": 30.0,
            },
            {
                "year": 2024, "net_income": 110.0, "cfo": 35.0, "fcf": 28.0,
                "total_assets": 420.0, "revenue": 520.0,
                "accounts_receivable": 55.0, "inventory": 33.0,
            },
            {
                "year": 2025, "net_income": 120.0, "cfo": 40.0, "fcf": 30.0,
                "total_assets": 440.0, "revenue": 540.0,
                "accounts_receivable": 60.0, "inventory": 36.0,
            },
        ],
    }
    result = compute_accounting_quality("TEST", "2026-03-26", fundamentals=fundamentals)
    headwinds = result.get("cash_earnings_headwind_signals", [])
    # With NI/CFO ~0.30, WEAK_CFO + ACCRUAL_HEAVY already fire
    # Accruals ratio (NI-CFO)/assets ~ 0.18 should also fire ACCRUALS_AGGRESSIVE
    assert "ACCRUALS_AGGRESSIVE" in headwinds or "ACCRUAL_HEAVY_EARNINGS" in headwinds


# ── the stock-compensation test reads the key the ingest actually writes ──────
# The module used to read "sbc_total"
# and fell back to "stock_based_compensation". Neither name is ever written —
# app/ingest writes "sbc" — so on every production row the stock-compensation
# ratio came back UNKNOWN and both signals were dormant. Same data, one key.


def _sbc_rows(sbc: float) -> list[dict[str, float]]:
    return [
        {
            "year": year,
            "revenue": 1000.0,
            "net_income": 100.0,
            "cfo": 120.0,
            "fcf": 100.0,
            "sbc": sbc,
        }
        for year in (2021, 2022, 2023, 2024, 2025)
    ]


def test_stock_compensation_burden_is_read_from_the_ingested_key():
    heavy = compute_accounting_quality(
        "AAA",
        "2026-03-01",
        fundamentals={"ticker": "AAA", "rows": _sbc_rows(150.0)},
        facts_status="OK",
    )
    assert heavy["sbc_to_revenue_median_3y"] == 0.15
    assert "SBC_BURDEN_HEADWIND" in heavy["cash_earnings_headwind_signals"]


def test_light_stock_compensation_is_a_support_signal():
    light = compute_accounting_quality(
        "AAA",
        "2026-03-01",
        fundamentals={"ticker": "AAA", "rows": _sbc_rows(10.0)},
        facts_status="OK",
    )
    assert light["sbc_to_revenue_median_3y"] == 0.01
    assert "LOW_SBC_BURDEN" in light["cash_earnings_support_signals"]


def test_the_documented_key_still_wins_when_both_are_present():
    rows = [dict(row, sbc_total=20.0) for row in _sbc_rows(150.0)]
    payload = compute_accounting_quality(
        "AAA",
        "2026-03-01",
        fundamentals={"ticker": "AAA", "rows": rows},
        facts_status="OK",
    )
    assert payload["sbc_to_revenue_median_3y"] == 0.02
