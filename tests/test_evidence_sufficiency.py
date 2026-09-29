from __future__ import annotations

import json

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.escalation import build_escalation_plan
from app.universe.memo_pack import _memo_markdown, build_investment_memo
from app.universe.promotion import LANE_2_RESEARCH_QUEUE, LANE_4_DEPRIORITIZED, build_promotion_state
from app.valuation.evidence_sufficiency import (
    MOS_CONFIRMED_ABSENT,
    MOS_UNASSESSABLE,
    REASON_MOS_BLOCKED_BY_MISSING_FACTS,
    REASON_MOS_BLOCKED_BY_MISSING_PRICE,
    REASON_MOS_BLOCKED_BY_MISSING_SHARES,
    REASON_NO_MOS_WITH_SUFFICIENT_EVIDENCE,
    SUFFICIENCY_INSUFFICIENT,
    SUFFICIENCY_SUFFICIENT,
    compute_evidence_sufficiency,
    write_evidence_sufficiency_for_run,
)
from app.valuation.intrinsic_discipline import MOS_ADEQUATE, MOS_NONE
from app.valuation.investment_readiness import (
    READY_NOT_INVESTABLE,
    READY_RESEARCH_WORTHY,
    READY_UNKNOWN,
    compute_investment_readiness,
)


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


def _claim(value, ref):
    status = "OK" if value != "UNKNOWN" else "UNKNOWN"
    reason = "OK" if value != "UNKNOWN" else "UNKNOWN"
    return {"value": value, "status": status, "reason_code": reason, "derived_from": [ref]}


def _intrinsic_payload(
    *,
    ticker: str = "AAA",
    mos_classification: str = MOS_ADEQUATE,
    mos_to_floor=0.30,
    downside_support_type: str = "EARNINGS_POWER_SUPPORT",
) -> dict:
    return {
        "normalized_earnings_power_value": 90.0 if mos_to_floor != "UNKNOWN" else "UNKNOWN",
        "normalized_earnings_power_method_used": "FCF_SELECTED",
        "normalized_earnings_power_status": "OK" if mos_to_floor != "UNKNOWN" else "UNKNOWN",
        "normalized_earnings_power_reason_codes": ["FCF_SELECTED"],
        "intrinsic_floor": 70.0 if mos_to_floor != "UNKNOWN" else "UNKNOWN",
        "intrinsic_base": 85.0 if mos_to_floor != "UNKNOWN" else "UNKNOWN",
        "intrinsic_ceiling": 95.0 if mos_to_floor != "UNKNOWN" else "UNKNOWN",
        "mos_to_floor": mos_to_floor,
        "mos_to_base": 0.45 if mos_to_floor != "UNKNOWN" else "UNKNOWN",
        "mos_classification": mos_classification,
        "downside_support_type": downside_support_type,
        "downside_support_status": "OK" if downside_support_type != "UNKNOWN_SUPPORT" else "UNKNOWN",
        "valuation_range_reason_codes": ["BASE_FROM_NORMALIZED_EARNINGS_POWER"],
        "downside_support_reason_codes": ["EARNINGS_POWER_SUPPORT"],
        "derived_from": [f"intrinsic.{ticker}"],
        "claims": {
            "intrinsic_floor": _claim(70.0 if mos_to_floor != "UNKNOWN" else "UNKNOWN", f"intrinsic.{ticker}.floor"),
            "mos_to_floor": _claim(mos_to_floor, f"intrinsic.{ticker}.mos_to_floor"),
        },
    }


def _valuation_confidence_payload(
    *,
    ticker: str = "AAA",
    support_count: int = 2,
    confidence_class: str = "HIGH_CONFIDENCE",
) -> dict:
    return {
        "valuation_support_count": support_count,
        "valuation_support_types_present": ["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"][:support_count],
        "valuation_support_count_reason_codes": ["EPV_SUPPORT"][:support_count] or ["NO_VALUATION_SUPPORTS_PRESENT"],
        "valuation_convergence_status": "STRONG_CONVERGENCE" if support_count >= 2 else "CONVERGENCE_UNKNOWN",
        "valuation_convergence_band_pct": 0.15 if support_count >= 2 else "UNKNOWN",
        "valuation_convergence_reason_codes": [],
        "valuation_fragility_status": "LOW_FRAGILITY" if support_count >= 2 else "MODERATE_FRAGILITY",
        "valuation_fragility_reason_codes": [] if support_count >= 2 else ["SINGLE_SUPPORT_ONLY"],
        "valuation_confidence_class": confidence_class,
        "valuation_confidence_reason_codes": ["MULTI_SUPPORT_VALUE_CASE"] if support_count >= 2 else ["SINGLE_SUPPORT_FRAGILE"],
        "derived_from": [f"confidence.{ticker}"],
        "claims": {
            "valuation_support_count": _claim(support_count, f"confidence.{ticker}.supports"),
            "valuation_confidence_class": _claim(confidence_class, f"confidence.{ticker}.class"),
        },
    }


def _valuation_integrity_payload(*, ticker: str = "AAA", integrity_class: str = "INTEGRITY_OK") -> dict:
    return {
        "valuation_integrity_class": integrity_class,
        "valuation_integrity_reason_codes": [] if integrity_class == "INTEGRITY_OK" else [integrity_class],
        "valuation_consistency_status": "CONSISTENT",
        "valuation_consistency_reason_codes": [],
        "valuation_input_fingerprint": f"fingerprint.{ticker}",
        "valuation_input_provenance_summary": {"ticker": ticker},
        "derived_from": [f"integrity.{ticker}"],
    }


def _value_type_payload(*, ticker: str = "AAA") -> dict:
    return {
        "value_type_primary": "EARNINGS_POWER_VALUE",
        "value_type_secondary": None,
        "value_type_reason_codes": ["NORMALIZED_EARNINGS_DRIVEN"],
        "value_type_support_summary": "Anchored to earnings power support.",
        "value_type_derived_from": [f"value_type.{ticker}"],
    }


def _owner_quality_payload(*, ticker: str = "AAA") -> dict:
    return {
        "oe_quality_total": 8.0,
        "owner_earnings_stability_score": 4.0,
        "capital_allocation_score": 3.0,
        "cash_conversion_score": 3.0,
        "oe_quality_reason_codes": [],
        "derived_from": [f"oe_quality.{ticker}"],
    }


def _intangible_payload(*, ticker: str = "AAA") -> dict:
    return {
        "intangible_economics_total": 7.0,
        "cycle_resilience_score": 2.0,
        "owner_value_capture_score": 4.0,
        "owner_value_capture_reason_codes": [],
        "derived_from": [f"intangible.{ticker}"],
    }


def _memo_sources(row: dict) -> dict:
    ticker = row["ticker"]
    return {
        "shortlist_row": row,
        "score_row": {
            "ticker": ticker,
            "metric_values": {
                "implied_return_base": row.get("implied_return_base", "UNKNOWN"),
                "intrinsic_per_share_base": row.get("intrinsic_base", 82.0),
                "current_price": 63.0,
                "mos_epv": row.get("mos_epv", "UNKNOWN"),
                "mos_netnet": row.get("mos_netnet", "UNKNOWN"),
                "epv_per_share": 80.0,
                "netnet_per_share": 70.0,
                "owner_earnings_yield_ev_3y": row.get("owner_earnings_yield_ev_3y", "UNKNOWN"),
                "fcf_yield_ev_3y": 0.05,
                "revenue_cagr_5y": 0.10,
                "revenue_cagr_10y": 0.09,
                "operating_margin_trend_slope": 0.01,
                "gross_margin_trend_slope": 0.01,
                "fcf_margin_trend_slope": 0.005,
                "roic_proxy": 0.14,
                "dilution_rate_shares_cagr": 0.01,
                "net_debt_proxy": 120.0,
                "risk_factor_keyword_delta": 1.0,
                "quality_score": 16.0,
                "risk_penalty": -2.0,
            },
            "metric_traces": {
                "implied_return_base": {"derived_from": [f"score.{ticker}.implied_return"]},
                "intrinsic_per_share_base": {"derived_from": [f"score.{ticker}.intrinsic_base"]},
            },
            "derived_from": [f"score.{ticker}"],
        },
        "gate_row": {
            "ticker": ticker,
            "gate_status": row.get("value_gate_status", "WATCH"),
            "gate_reasons": [row.get("primary_blocker", "NONE")],
            "primary_blocker": row.get("primary_blocker", "NONE"),
            "inputs_used": {
                "current_price": {"value": 63.0, "derived_from": [f"price.{ticker}"]},
                "net_debt_proxy": {"value": 120.0, "derived_from": [f"net_debt.{ticker}"]},
                "dilution_rate_shares_cagr": {"value": 0.01, "derived_from": [f"dilution.{ticker}"]},
            },
            "net_debt_to_cfo": 1.2,
        },
        "valuation_row": {
            "ticker": ticker,
            "price_status": row.get("price_status", "OK"),
            "price_reason_code": "OK",
            "valuation_status": row.get("valuation_status", "OK"),
            "valuation_reason_code": "OK",
            "derived_from": [f"valuation.{ticker}"],
        },
        "shares_row": {"ticker": ticker, "shares_status": row.get("shares_status", "OK"), "shares_reason_code": "OK"},
        "fcf_row": {"ticker": ticker, "fcf_status": row.get("fcf_status", "OK"), "fcf_reason_code": "OK"},
        "facts_row": {"ticker": ticker, "status": row.get("facts_status", "OK"), "fetch_reason_code": row.get("facts_reason_code", "OK")},
    }


def test_sufficient_evidence_and_no_mos_yields_confirmed_absent():
    payload = compute_evidence_sufficiency(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(mos_classification=MOS_NONE, mos_to_floor=-0.05),
        valuation_confidence_payload=_valuation_confidence_payload(),
        valuation_integrity_payload=_valuation_integrity_payload(),
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
    )

    assert payload["evidence_sufficiency_class"] == SUFFICIENCY_SUFFICIENT
    assert payload["mos_assessment_status"] == MOS_CONFIRMED_ABSENT
    assert REASON_NO_MOS_WITH_SUFFICIENT_EVIDENCE in payload["mos_guardrail_reason_codes"]


def test_missing_price_makes_mos_unassessable():
    payload = compute_evidence_sufficiency(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(),
        valuation_confidence_payload=_valuation_confidence_payload(),
        valuation_integrity_payload=_valuation_integrity_payload(),
        price_status="UNKNOWN",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="UNKNOWN",
    )

    assert payload["evidence_sufficiency_class"] == SUFFICIENCY_INSUFFICIENT
    assert payload["mos_assessment_status"] == MOS_UNASSESSABLE
    assert REASON_MOS_BLOCKED_BY_MISSING_PRICE in payload["mos_guardrail_reason_codes"]


def test_missing_facts_makes_mos_unassessable():
    payload = compute_evidence_sufficiency(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(),
        valuation_confidence_payload=_valuation_confidence_payload(),
        valuation_integrity_payload=_valuation_integrity_payload(),
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="UNKNOWN",
        valuation_status="OK",
        facts_blocker_class="FACTS_RETRYABLE_TIMEOUT",
        fail_due_to_missing_evidence=True,
        primary_fail_domain="EVIDENCE",
    )

    assert payload["mos_assessment_status"] == MOS_UNASSESSABLE
    assert REASON_MOS_BLOCKED_BY_MISSING_FACTS in payload["mos_guardrail_reason_codes"]


def test_missing_shares_makes_mos_unassessable():
    payload = compute_evidence_sufficiency(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(),
        valuation_confidence_payload=_valuation_confidence_payload(),
        valuation_integrity_payload=_valuation_integrity_payload(),
        price_status="OK",
        shares_status="UNKNOWN",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="UNKNOWN",
    )

    assert payload["mos_assessment_status"] == MOS_UNASSESSABLE
    assert REASON_MOS_BLOCKED_BY_MISSING_SHARES in payload["mos_guardrail_reason_codes"]


def test_evidence_blocked_name_does_not_collapse_to_not_investable_no_mos():
    payload = compute_investment_readiness(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(),
        valuation_confidence_payload=_valuation_confidence_payload(
            support_count=2,
            confidence_class="MEDIUM_CONFIDENCE",
        ),
        valuation_integrity_payload=_valuation_integrity_payload(),
        value_type_payload=_value_type_payload(),
        owner_quality_payload=_owner_quality_payload(),
        intangible_payload=_intangible_payload(),
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="UNKNOWN",
        valuation_status="OK",
        facts_blocker_class="FACTS_RETRYABLE_TIMEOUT",
        facts_blocker_retryable=True,
        facts_retry_recommended=True,
        fail_due_to_missing_evidence=True,
        primary_fail_domain="EVIDENCE",
        value_gate_status="WATCH",
        primary_blocker="FACTS_RETRYABLE_TIMEOUT",
        row_derived_from=["row.AAA"],
    )

    assert payload["investment_readiness_class"] == READY_RESEARCH_WORTHY
    assert payload["mos_assessment_status"] == MOS_UNASSESSABLE
    assert payload["blocker_stack_primary"] == "MOS_UNASSESSABLE_MISSING_FACTS"
    assert "VALUATION_NO_MOS_CONFIRMED" not in payload["blocker_stack_all"]


def test_evidence_degraded_name_without_support_becomes_readiness_unknown():
    payload = compute_investment_readiness(
        "BBB",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(
            ticker="BBB",
            mos_to_floor="UNKNOWN",
            mos_classification="MOS_UNKNOWN",
            downside_support_type="UNKNOWN_SUPPORT",
        ),
        valuation_confidence_payload=_valuation_confidence_payload(
            ticker="BBB",
            support_count=1,
            confidence_class="LOW_CONFIDENCE",
        ),
        valuation_integrity_payload=_valuation_integrity_payload(ticker="BBB", integrity_class="INTEGRITY_UNKNOWN"),
        value_type_payload={
            "value_type_primary": "UNKNOWN_VALUE_TYPE",
            "value_type_secondary": None,
            "value_type_reason_codes": ["INSUFFICIENT_VALUE_TYPE_EVIDENCE"],
            "value_type_support_summary": "",
            "value_type_derived_from": ["value_type.BBB"],
        },
        owner_quality_payload=_owner_quality_payload(ticker="BBB"),
        intangible_payload=_intangible_payload(ticker="BBB"),
        price_status="UNKNOWN",
        shares_status="OK",
        fcf_status="UNKNOWN",
        facts_status="UNKNOWN",
        valuation_status="UNKNOWN",
        facts_blocker_class="FACTS_RETRYABLE_TIMEOUT",
        facts_blocker_retryable=True,
        facts_retry_recommended=True,
        fail_due_to_missing_evidence=True,
        primary_fail_domain="EVIDENCE",
        value_gate_status="WATCH",
        primary_blocker="FACTS_RETRYABLE_TIMEOUT",
        row_derived_from=["row.BBB"],
    )

    assert payload["investment_readiness_class"] == READY_UNKNOWN
    assert payload["mos_assessment_status"] == MOS_UNASSESSABLE
    assert payload["primary_next_step"] == "CLEAR_EVIDENCE_BLOCKERS"


def test_memo_pack_includes_evidence_sufficiency_section():
    row = {
        "ticker": "AAA",
        "sector": "Software",
        "as_of_date": "2026-02-14",
        "source_depth_runs": [{"run_id": "depth_run", "sector": "Software", "as_of_date": "2026-02-14"}],
        "value_gate_status": "WATCH",
        "value_gate_reasons": ["FACTS_RETRYABLE_TIMEOUT"],
        "primary_blocker": "FACTS_RETRYABLE_TIMEOUT",
        "implied_return_base": 0.22,
        "mos_epv": 0.18,
        "mos_netnet": 0.05,
        "owner_earnings_yield_ev_3y": 0.06,
        "fcf_yield_ev_3y": 0.05,
        "yield_metric_used": "owner_earnings_yield_ev_3y",
        "price_status": "OK",
        "valuation_status": "OK",
        "shares_status": "OK",
        "fcf_status": "OK",
        "facts_status": "UNKNOWN",
        "facts_reason_code": "FACTS_RETRYABLE_TIMEOUT",
        "normalized_earnings_power_value": 90.0,
        "normalized_earnings_power_method_used": "FCF_SELECTED",
        "normalized_earnings_power_status": "OK",
        "normalized_earnings_power_reason_codes": ["FCF_SELECTED"],
        "intrinsic_floor": 70.0,
        "intrinsic_base": 82.0,
        "intrinsic_ceiling": 95.0,
        "mos_to_floor": 0.30,
        "mos_to_base": 0.45,
        "mos_classification": "ADEQUATE_MARGIN_OF_SAFETY",
        "downside_support_type": "EARNINGS_POWER_SUPPORT",
        "downside_support_status": "OK",
        "valuation_range_reason_codes": ["BASE_FROM_NORMALIZED_EARNINGS_POWER"],
        "downside_support_reason_codes": ["EARNINGS_POWER_SUPPORT"],
        "evidence_sufficiency_class": "PARTIAL_FOR_MOS",
        "evidence_sufficiency_reason_codes": ["PRICE_AVAILABLE", "SHARES_AVAILABLE", "MISSING_FACTS"],
        "mos_assessment_status": "MOS_UNASSESSABLE",
        "mos_guardrail_reason_codes": ["MOS_BLOCKED_BY_MISSING_FACTS"],
        "valuation_support_count": 2,
        "valuation_support_types_present": ["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"],
        "valuation_support_count_reason_codes": ["EPV_SUPPORT"],
        "valuation_convergence_status": "MODERATE_CONVERGENCE",
        "valuation_convergence_band_pct": 0.20,
        "valuation_convergence_reason_codes": [],
        "valuation_fragility_status": "MODERATE_FRAGILITY",
        "valuation_fragility_reason_codes": [],
        "valuation_confidence_class": "MEDIUM_CONFIDENCE",
        "valuation_confidence_reason_codes": ["MULTI_SUPPORT_VALUE_CASE"],
        "valuation_integrity_class": "INTEGRITY_OK",
        "valuation_integrity_reason_codes": [],
        "investment_readiness_class": READY_RESEARCH_WORTHY,
        "investment_readiness_reason_codes": ["BLOCKED_BUT_RESEARCH_WORTHY"],
        "blocker_stack_primary": "FACTS_RETRYABLE_TIMEOUT",
        "blocker_stack_secondary": "EVIDENCE_INSUFFICIENT_FOR_MOS",
        "blocker_stack_all": ["FACTS_RETRYABLE_TIMEOUT", "EVIDENCE_INSUFFICIENT_FOR_MOS"],
        "blocker_stack_retryable": True,
        "blocker_stack_structural": False,
        "readiness_support_present": ["ADEQUATE_MOS", "MULTI_SUPPORT_VALUE_CASE", "INTEGRITY_OK"],
        "readiness_support_missing": ["MOS_UNASSESSABLE_EVIDENCE_GAP", "FACTS_MISSING"],
        "readiness_support_headwinds": ["EVIDENCE_INSUFFICIENT_FOR_MOS"],
        "primary_next_step": "CLEAR_EVIDENCE_BLOCKERS",
        "primary_next_step_reason": "FACTS_RETRYABLE_TIMEOUT",
        "value_type_primary": "EARNINGS_POWER_VALUE",
        "value_type_reason_codes": ["EPV_DRIVEN", "NORMALIZED_EARNINGS_DRIVEN"],
        "value_type_support_summary": "Anchored to earnings power support.",
        "owner_earnings_stability_score": 4.0,
        "capital_allocation_score": 3.0,
        "cash_conversion_score": 3.0,
        "oe_quality_total": 9.0,
        "oe_quality_reason_codes": [],
        "gross_margin_durability_score": 4.0,
        "balance_sheet_optionality_score": 4.0,
        "cycle_resilience_score": 3.0,
        "rnd_productivity_score": 3.0,
        "sga_leverage_score": 3.0,
        "owner_value_capture_score": 4.0,
        "intangible_economics_total": 10.0,
        "rnd_productivity_reason_codes": [],
        "sga_leverage_reason_codes": [],
        "owner_value_capture_reason_codes": [],
        "intangible_economics_reason_codes": [],
        "derived_from": ["shortlist.AAA"],
    }

    memo = build_investment_memo(
        "AAA",
        universe_run_id="universe_run",
        batch_run_id="batch_run",
        sources=_memo_sources(row),
    )
    markdown = _memo_markdown(memo)

    assert "## Evidence Sufficiency for Margin of Safety" in markdown
    assert "mos_assessment_status: `MOS_UNASSESSABLE`" in markdown
    assert "MOS_BLOCKED_BY_MISSING_FACTS" in markdown


def test_promotion_and_escalation_surface_evidence_blocked_mos_without_overriding_fail():
    master_shortlist = {
        "campaign_run_id": "campaign_evidence",
        "rows": [
            {
                "ticker": "AAA",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "campaign_evidence__aaa", "batch_run_id": "campaign_evidence__aaa_depth_batch"}],
                "best_rank_seen": 1,
                "value_gate_status": "WATCH",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": 0.20,
                "mos_epv": 0.18,
                "owner_earnings_yield_ev_3y": 0.06,
                "yield_metric_used": "owner_earnings_yield_ev_3y",
                "primary_blocker": "FACTS_RETRYABLE_TIMEOUT",
                "latest_primary_blocker": "FACTS_RETRYABLE_TIMEOUT",
                "composite_score_total": 60.0,
                "memo_path": "memo/AAA.md",
                "mos_to_floor": 0.30,
                "mos_classification": "ADEQUATE_MARGIN_OF_SAFETY",
                "downside_support_type": "EARNINGS_POWER_SUPPORT",
                "valuation_support_count": 2,
                "valuation_support_types_present": ["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"],
                "valuation_support_count_reason_codes": ["EPV_SUPPORT"],
                "valuation_convergence_status": "MODERATE_CONVERGENCE",
                "valuation_convergence_band_pct": 0.20,
                "valuation_convergence_reason_codes": [],
                "valuation_fragility_status": "MODERATE_FRAGILITY",
                "valuation_fragility_reason_codes": [],
                "valuation_confidence_class": "MEDIUM_CONFIDENCE",
                "valuation_confidence_reason_codes": ["MULTI_SUPPORT_VALUE_CASE"],
                "valuation_integrity_class": "INTEGRITY_OK",
                "valuation_integrity_reason_codes": [],
                "investment_readiness_class": READY_RESEARCH_WORTHY,
                "investment_readiness_reason_codes": ["BLOCKED_BUT_RESEARCH_WORTHY"],
                "evidence_sufficiency_class": "INSUFFICIENT_FOR_MOS",
                "evidence_sufficiency_reason_codes": ["MISSING_FACTS"],
                "mos_assessment_status": "MOS_UNASSESSABLE",
                "mos_guardrail_reason_codes": ["MOS_BLOCKED_BY_MISSING_FACTS"],
                "blocker_stack_primary": "FACTS_RETRYABLE_TIMEOUT",
                "primary_next_step": "CLEAR_EVIDENCE_BLOCKERS",
                "value_type_primary": "EARNINGS_POWER_VALUE",
                "value_type_reason_codes": ["NORMALIZED_EARNINGS_DRIVEN"],
                "facts_blocker_class": "FACTS_RETRYABLE_TIMEOUT",
                "facts_blocker_retryable": True,
                "facts_blocker_terminal": False,
                "facts_blocker_partial_usable": False,
                "facts_missing_key_inputs": ["ev"],
                "facts_retry_recommended": True,
                "fail_due_to_missing_evidence": True,
                "fail_due_to_economic_weakness": False,
                "primary_fail_domain": "EVIDENCE",
                "oe_quality_total": 9.0,
                "oe_quality_reason_codes": [],
                "intangible_economics_total": 7.0,
                "intangible_economics_reason_codes": [],
                "owner_value_capture_score": 4.0,
                "owner_value_capture_reason_codes": [],
                "derived_from": ["shortlist.AAA"],
            },
            {
                "ticker": "BBB",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "campaign_evidence__bbb", "batch_run_id": "campaign_evidence__bbb_depth_batch"}],
                "best_rank_seen": 3,
                "value_gate_status": "FAIL",
                "latest_value_gate_status": "FAIL",
                "implied_return_base": 0.05,
                "mos_epv": -0.05,
                "owner_earnings_yield_ev_3y": 0.02,
                "yield_metric_used": "owner_earnings_yield_ev_3y",
                "primary_blocker": "STRUCTURAL_WEAK_ECONOMICS",
                "latest_primary_blocker": "STRUCTURAL_WEAK_ECONOMICS",
                "composite_score_total": 20.0,
                "memo_path": "memo/BBB.md",
                "mos_to_floor": -0.10,
                "mos_classification": "NO_MARGIN_OF_SAFETY",
                "downside_support_type": "LIMITED_SUPPORT",
                "valuation_support_count": 1,
                "valuation_support_types_present": ["EPV_SUPPORT"],
                "valuation_support_count_reason_codes": ["EPV_SUPPORT"],
                "valuation_convergence_status": "CONVERGENCE_UNKNOWN",
                "valuation_convergence_band_pct": "UNKNOWN",
                "valuation_convergence_reason_codes": [],
                "valuation_fragility_status": "HIGH_FRAGILITY",
                "valuation_fragility_reason_codes": ["SINGLE_SUPPORT_ONLY"],
                "valuation_confidence_class": "LOW_CONFIDENCE",
                "valuation_confidence_reason_codes": ["SINGLE_SUPPORT_FRAGILE"],
                "valuation_integrity_class": "INTEGRITY_OK",
                "valuation_integrity_reason_codes": [],
                "investment_readiness_class": READY_NOT_INVESTABLE,
                "investment_readiness_reason_codes": ["NOT_INVESTABLE_STRUCTURAL"],
                "evidence_sufficiency_class": "SUFFICIENT_FOR_MOS",
                "evidence_sufficiency_reason_codes": ["PRICE_AVAILABLE", "SHARES_AVAILABLE", "FACTS_AVAILABLE"],
                "mos_assessment_status": "MOS_CONFIRMED_ABSENT",
                "mos_guardrail_reason_codes": ["NO_MOS_WITH_SUFFICIENT_EVIDENCE"],
                "blocker_stack_primary": "STRUCTURAL_WEAK_ECONOMICS",
                "primary_next_step": "DO_NOT_ADVANCE",
                "value_type_primary": "FRAGILE_VALUE",
                "value_type_reason_codes": ["HIGH_FRAGILITY_CASE"],
                "facts_blocker_class": "FACTS_OK",
                "facts_blocker_retryable": False,
                "facts_blocker_terminal": False,
                "facts_blocker_partial_usable": False,
                "facts_missing_key_inputs": [],
                "facts_retry_recommended": False,
                "fail_due_to_missing_evidence": False,
                "fail_due_to_economic_weakness": True,
                "primary_fail_domain": "ECONOMICS",
                "oe_quality_total": 2.0,
                "oe_quality_reason_codes": ["EXCESS_DILUTION"],
                "intangible_economics_total": 2.0,
                "intangible_economics_reason_codes": ["WEAK_PER_SHARE_CAPTURE"],
                "owner_value_capture_score": 1.0,
                "owner_value_capture_reason_codes": ["WEAK_PER_SHARE_CAPTURE"],
                "derived_from": ["shortlist.BBB"],
            },
        ],
    }
    master_watchlist = {
        "campaign_run_id": "campaign_evidence",
        "tickers": {
            "AAA": {
                "latest_value_gate_status": "WATCH",
                "latest_implied_return_base": 0.20,
                "latest_primary_blocker": "FACTS_RETRYABLE_TIMEOUT",
                "appearances_count": 2,
                "history": [{"campaign_item": "core", "universe_run_id": "campaign_evidence__aaa", "value_gate_status": "WATCH", "implied_return_base": 0.20, "primary_blocker": "FACTS_RETRYABLE_TIMEOUT", "last_rank": 1}],
            },
            "BBB": {
                "latest_value_gate_status": "FAIL",
                "latest_implied_return_base": 0.05,
                "latest_primary_blocker": "STRUCTURAL_WEAK_ECONOMICS",
                "appearances_count": 2,
                "history": [{"campaign_item": "core", "universe_run_id": "campaign_evidence__bbb", "value_gate_status": "FAIL", "implied_return_base": 0.05, "primary_blocker": "STRUCTURAL_WEAK_ECONOMICS", "last_rank": 3}],
            },
        },
    }

    promotion = build_promotion_state("campaign_evidence", master_watchlist, master_shortlist)
    rows = {row["ticker"]: row for row in promotion["rows"]}
    assert rows["AAA"]["priority_lane"] == LANE_2_RESEARCH_QUEUE
    assert rows["AAA"]["evidence_sufficiency_class"] == "INSUFFICIENT_FOR_MOS"
    assert "MOS_UNASSESSABLE_EVIDENCE_GAP" in rows["AAA"]["promotion_reason_codes"]
    assert "NO_MOS_CONFIRMED" not in rows["AAA"]["promotion_reason_codes"]
    assert rows["BBB"]["priority_lane"] == LANE_4_DEPRIORITIZED
    assert "NO_MOS_CONFIRMED" in rows["BBB"]["promotion_reason_codes"]

    escalation = build_escalation_plan(
        "campaign_evidence",
        promotion,
        {
            "lane_1_high_priority": [],
            "lane_2_research_queue": [rows["AAA"]],
            "lane_3_monitor": [],
            "lane_4_deprioritized": [rows["BBB"]],
        },
        config={
            "as_of_date": "2026-02-14",
            "source_campaign_file": "data/universe/sample_campaign.json",
            "top_n": 10,
            "policy": "value_first_ready",
        },
    )
    aaa_queue = [row for row in escalation["queue"] if row["ticker"] == "AAA"]
    bbb_queue = [row for row in escalation["queue"] if row["ticker"] == "BBB"]
    assert aaa_queue[0]["evidence_sufficiency_class"] == "INSUFFICIENT_FOR_MOS"
    assert aaa_queue[0]["mos_assessment_status"] == "MOS_UNASSESSABLE"
    assert "MOS_UNASSESSABLE_EVIDENCE_GAP" in aaa_queue[0]["priority_support_codes"]
    assert "NO_MOS_CONFIRMED" in bbb_queue[0]["priority_support_codes"]


def test_evidence_sufficiency_cli_open(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "evidence_sufficiency_cli"

    aaa = compute_evidence_sufficiency(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(mos_classification=MOS_NONE, mos_to_floor=-0.05),
        valuation_confidence_payload=_valuation_confidence_payload(),
        valuation_integrity_payload=_valuation_integrity_payload(),
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
    )
    bbb = compute_evidence_sufficiency(
        "BBB",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(ticker="BBB"),
        valuation_confidence_payload=_valuation_confidence_payload(ticker="BBB"),
        valuation_integrity_payload=_valuation_integrity_payload(ticker="BBB"),
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="UNKNOWN",
        valuation_status="OK",
        facts_blocker_class="FACTS_RETRYABLE_TIMEOUT",
        fail_due_to_missing_evidence=True,
        primary_fail_domain="EVIDENCE",
    )

    write_evidence_sufficiency_for_run(
        run_id=run_id,
        as_of_date="2026-02-14",
        tickers=["AAA", "BBB"],
        output_path=cfg.outputs_dir / "universe" / run_id / "evidence_sufficiency.json",
        scoreboard_rows=[
            {"ticker": "AAA", "evidence_sufficiency_detail": aaa},
            {"ticker": "BBB", "evidence_sufficiency_detail": bbb},
        ],
    )

    result = runner.invoke(app, ["universe-evidence-sufficiency-open", "--run-id", run_id])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "OK"
    assert payload["counts_by_evidence_sufficiency_class"][SUFFICIENCY_SUFFICIENT] == 1
    assert payload["counts_by_mos_assessment_status"][MOS_UNASSESSABLE] == 1
    assert payload["top_mos_confirmed_absent"][0]["ticker"] == "AAA"
    assert payload["top_mos_unassessable"][0]["ticker"] == "BBB"
