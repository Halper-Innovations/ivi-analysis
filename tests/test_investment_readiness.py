from __future__ import annotations

import json

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.escalation import build_escalation_plan
from app.universe.memo_pack import _memo_markdown, build_investment_memo
from app.universe.promotion import LANE_2_RESEARCH_QUEUE, LANE_4_DEPRIORITIZED, build_promotion_state
from app.universe.ranking import ranking_sort_key
from app.valuation.investment_readiness import (
    NEXT_CLEAR_EVIDENCE,
    NEXT_DO_NOT_ADVANCE,
    READY_INVESTABLE_NOW,
    READY_NOT_INVESTABLE,
    READY_RESEARCH_WORTHY,
    READY_WATCH_ONLY,
    compute_investment_readiness,
    write_investment_readiness_for_run,
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
    mos_to_floor: float | str = 0.30,
    mos_classification: str = "ADEQUATE_MARGIN_OF_SAFETY",
    downside_support_type: str = "EARNINGS_POWER_SUPPORT",
    normalized_status: str = "OK",
    normalized_reason_codes: list[str] | None = None,
) -> dict:
    reason_codes = list(normalized_reason_codes or ["FCF_SELECTED"])
    return {
        "normalized_earnings_power_value": 90.0 if mos_to_floor != "UNKNOWN" else "UNKNOWN",
        "normalized_earnings_power_method_used": "FCF_SELECTED",
        "normalized_earnings_power_status": normalized_status,
        "normalized_earnings_power_reason_codes": reason_codes,
        "intrinsic_floor": 70.0 if mos_to_floor != "UNKNOWN" else "UNKNOWN",
        "intrinsic_base": 85.0 if mos_to_floor != "UNKNOWN" else "UNKNOWN",
        "intrinsic_ceiling": 95.0 if mos_to_floor != "UNKNOWN" else "UNKNOWN",
        "mos_to_floor": mos_to_floor,
        "mos_to_base": 0.45 if mos_to_floor != "UNKNOWN" else "UNKNOWN",
        "mos_classification": mos_classification,
        "downside_support_type": downside_support_type,
        "downside_support_status": "OK" if downside_support_type != "UNKNOWN_SUPPORT" else "UNKNOWN",
        "valuation_range_reason_codes": ["BASE_FROM_NORMALIZED_EARNINGS_POWER"],
        "downside_support_reason_codes": [downside_support_type] if downside_support_type != "UNKNOWN_SUPPORT" else [],
        "derived_from": [f"intrinsic.{ticker}"],
        "claims": {
            "normalized_earnings_power_value": _claim(
                90.0 if mos_to_floor != "UNKNOWN" else "UNKNOWN",
                f"intrinsic.{ticker}.normalized",
            ),
            "intrinsic_floor": _claim(70.0 if mos_to_floor != "UNKNOWN" else "UNKNOWN", f"intrinsic.{ticker}.floor"),
            "intrinsic_base": _claim(85.0 if mos_to_floor != "UNKNOWN" else "UNKNOWN", f"intrinsic.{ticker}.base"),
            "intrinsic_ceiling": _claim(
                95.0 if mos_to_floor != "UNKNOWN" else "UNKNOWN",
                f"intrinsic.{ticker}.ceiling",
            ),
            "mos_to_floor": _claim(mos_to_floor, f"intrinsic.{ticker}.mos_to_floor"),
            "mos_to_base": _claim(
                0.45 if mos_to_floor != "UNKNOWN" else "UNKNOWN",
                f"intrinsic.{ticker}.mos_to_base",
            ),
        },
    }


def _valuation_confidence_payload(
    *,
    ticker: str = "AAA",
    support_count: int = 2,
    support_types: list[str] | None = None,
    fragility_status: str = "LOW_FRAGILITY",
    confidence_class: str = "HIGH_CONFIDENCE",
    convergence_status: str = "STRONG_CONVERGENCE",
    fragility_reason_codes: list[str] | None = None,
) -> dict:
    support_types = list(support_types or ["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"])
    return {
        "valuation_support_count": support_count,
        "valuation_support_types_present": support_types[:support_count],
        "valuation_support_count_reason_codes": support_types[:support_count] or ["NO_VALUATION_SUPPORTS_PRESENT"],
        "valuation_convergence_status": convergence_status,
        "valuation_convergence_band_pct": 0.15 if support_count >= 2 else "UNKNOWN",
        "valuation_convergence_reason_codes": [] if convergence_status != "WEAK_CONVERGENCE" else ["SUPPORTS_CONFLICT"],
        "valuation_fragility_status": fragility_status,
        "valuation_fragility_reason_codes": list(fragility_reason_codes or ([] if support_count >= 2 else ["SINGLE_SUPPORT_ONLY"])),
        "valuation_confidence_class": confidence_class,
        "valuation_confidence_reason_codes": ["MULTI_SUPPORT_VALUE_CASE"] if support_count >= 2 else ["SINGLE_SUPPORT_FRAGILE"],
        "derived_from": [f"confidence.{ticker}"],
        "claims": {
            "valuation_support_count": _claim(support_count, f"confidence.{ticker}.supports"),
            "valuation_confidence_class": _claim(confidence_class, f"confidence.{ticker}.class"),
        },
    }


def _valuation_integrity_payload(
    *,
    ticker: str = "AAA",
    integrity_class: str = "INTEGRITY_OK",
    integrity_reason_codes: list[str] | None = None,
    consistency_status: str = "CONSISTENT",
) -> dict:
    return {
        "valuation_integrity_class": integrity_class,
        "valuation_integrity_reason_codes": list(integrity_reason_codes or ([] if integrity_class == "INTEGRITY_OK" else [integrity_class])),
        "valuation_consistency_status": consistency_status,
        "valuation_consistency_reason_codes": [] if consistency_status == "CONSISTENT" else ["METHOD_SUPPORT_INCONSISTENT"],
        "valuation_uniformity_group_id": None,
        "valuation_uniformity_reason_codes": [],
        "valuation_input_fingerprint": f"fingerprint.{ticker}",
        "valuation_input_provenance_summary": {"ticker": ticker},
        "derived_from": [f"integrity.{ticker}"],
    }


def _value_type_payload(*, ticker: str = "AAA", primary: str = "EARNINGS_POWER_VALUE") -> dict:
    return {
        "value_type_primary": primary,
        "value_type_secondary": None,
        "value_type_reason_codes": ["NORMALIZED_EARNINGS_DRIVEN"],
        "value_type_support_summary": "Anchored to normalized earnings support.",
        "value_type_derived_from": [f"value_type.{ticker}"],
    }


def _owner_quality_payload(
    *,
    ticker: str = "AAA",
    total: float | str = 8.0,
    capital_allocation_score: float | str = 3.0,
    reason_codes: list[str] | None = None,
) -> dict:
    return {
        "oe_quality_total": total,
        "owner_earnings_stability_score": 4.0 if total != "UNKNOWN" else "UNKNOWN",
        "capital_allocation_score": capital_allocation_score,
        "cash_conversion_score": 3.0 if total != "UNKNOWN" else "UNKNOWN",
        "oe_quality_reason_codes": list(reason_codes or []),
        "derived_from": [f"oe_quality.{ticker}"],
    }


def _intangible_payload(
    *,
    ticker: str = "AAA",
    total: float | str = 7.0,
    cycle_resilience_score: float | str = 2.0,
    owner_value_capture_score: float | str = 3.0,
    owner_value_capture_reason_codes: list[str] | None = None,
) -> dict:
    return {
        "intangible_economics_total": total,
        "cycle_resilience_score": cycle_resilience_score,
        "owner_value_capture_score": owner_value_capture_score,
        "owner_value_capture_reason_codes": list(owner_value_capture_reason_codes or []),
        "derived_from": [f"intangible.{ticker}"],
    }


def _readiness_row(
    *,
    ticker: str,
    readiness_class: str,
    blocker_primary: str,
    next_step: str,
    gate_status: str = "WATCH",
    facts_blocker_class: str = "FACTS_OK",
    facts_blocker_retryable: bool = False,
) -> dict:
    evidence_blocked = facts_blocker_class != "FACTS_OK"
    return {
        "ticker": ticker,
        "source_runs": [
            {
                "campaign_item": "core",
                "universe_run_id": f"campaign_ready__{ticker.lower()}",
                "batch_run_id": f"campaign_ready__{ticker.lower()}_depth_batch",
            }
        ],
        "best_rank_seen": 1 if ticker == "AAA" else 3,
        "value_gate_status": gate_status,
        "latest_value_gate_status": gate_status,
        "implied_return_base": 0.20 if ticker == "AAA" else 0.08,
        "mos_epv": 0.18 if ticker == "AAA" else -0.05,
        "owner_earnings_yield_ev_3y": 0.06 if ticker == "AAA" else 0.02,
        "yield_metric_used": "owner_earnings_yield_ev_3y",
        "primary_blocker": blocker_primary,
        "latest_primary_blocker": blocker_primary,
        "memo_path": f"memo/{ticker}.md",
        "composite_score_total": 60.0 if ticker == "AAA" else 25.0,
        "mos_to_floor": 0.30 if ticker == "AAA" else -0.05,
        "mos_classification": "ADEQUATE_MARGIN_OF_SAFETY" if ticker == "AAA" else "NO_MARGIN_OF_SAFETY",
        "downside_support_type": "EARNINGS_POWER_SUPPORT" if ticker == "AAA" else "LIMITED_SUPPORT",
        "valuation_support_count": 2 if ticker == "AAA" else 1,
        "valuation_support_types_present": ["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"] if ticker == "AAA" else ["EPV_SUPPORT"],
        "valuation_support_count_reason_codes": ["EPV_SUPPORT"],
        "valuation_convergence_status": "MODERATE_CONVERGENCE" if ticker == "AAA" else "CONVERGENCE_UNKNOWN",
        "valuation_convergence_band_pct": 0.20 if ticker == "AAA" else "UNKNOWN",
        "valuation_convergence_reason_codes": [],
        "valuation_fragility_status": "MODERATE_FRAGILITY" if ticker == "AAA" else "HIGH_FRAGILITY",
        "valuation_fragility_reason_codes": [] if ticker == "AAA" else ["SINGLE_SUPPORT_ONLY"],
        "valuation_confidence_class": "MEDIUM_CONFIDENCE" if ticker == "AAA" else "LOW_CONFIDENCE",
        "valuation_confidence_reason_codes": ["MULTI_SUPPORT_VALUE_CASE"] if ticker == "AAA" else ["SINGLE_SUPPORT_FRAGILE"],
        "valuation_integrity_class": "INTEGRITY_OK" if ticker == "AAA" else "INTEGRITY_SUSPECT",
        "valuation_integrity_reason_codes": [] if ticker == "AAA" else ["INTEGRITY_SUSPECT"],
        "investment_readiness_class": readiness_class,
        "investment_readiness_reason_codes": [readiness_class],
        "evidence_sufficiency_class": "INSUFFICIENT_FOR_MOS" if evidence_blocked else "SUFFICIENT_FOR_MOS",
        "evidence_sufficiency_reason_codes": ["MISSING_FACTS"] if evidence_blocked else ["PRICE_AVAILABLE", "SHARES_AVAILABLE", "FACTS_AVAILABLE"],
        "mos_assessment_status": "MOS_UNASSESSABLE" if evidence_blocked else ("MOS_CONFIRMED_PRESENT" if ticker == "AAA" else "MOS_CONFIRMED_ABSENT"),
        "mos_guardrail_reason_codes": ["MOS_BLOCKED_BY_MISSING_FACTS"] if evidence_blocked else (["MOS_PRESENT_WITH_SUFFICIENT_EVIDENCE"] if ticker == "AAA" else ["NO_MOS_WITH_SUFFICIENT_EVIDENCE"]),
        "blocker_stack_primary": blocker_primary,
        "primary_next_step": next_step,
        "value_type_primary": "ASSET_BACKED_VALUE" if ticker == "AAA" else "FRAGILE_VALUE",
        "value_type_reason_codes": ["NETNET_DRIVEN"] if ticker == "AAA" else ["HIGH_FRAGILITY_CASE"],
        "facts_blocker_class": facts_blocker_class,
        "facts_blocker_retryable": facts_blocker_retryable,
        "facts_blocker_terminal": False,
        "facts_blocker_partial_usable": False,
        "facts_missing_key_inputs": ["ev"] if facts_blocker_class != "FACTS_OK" else [],
        "facts_retry_recommended": facts_blocker_retryable,
        "fail_due_to_missing_evidence": facts_blocker_class != "FACTS_OK",
        "fail_due_to_economic_weakness": ticker != "AAA",
        "primary_fail_domain": "EVIDENCE" if ticker == "AAA" and facts_blocker_class != "FACTS_OK" else "ECONOMICS",
        "oe_quality_total": 9.0 if ticker == "AAA" else 2.0,
        "oe_quality_reason_codes": [] if ticker == "AAA" else ["EXCESS_DILUTION"],
        "intangible_economics_total": 7.0 if ticker == "AAA" else 2.0,
        "intangible_economics_reason_codes": [] if ticker == "AAA" else ["WEAK_PER_SHARE_CAPTURE"],
        "owner_value_capture_score": 4.0 if ticker == "AAA" else 1.0,
        "owner_value_capture_reason_codes": [] if ticker == "AAA" else ["WEAK_PER_SHARE_CAPTURE"],
        "derived_from": [f"shortlist.{ticker}"],
    }


def _memo_sources(row: dict) -> dict:
    ticker = row["ticker"]
    return {
        "shortlist_row": row,
        "score_row": {
            "ticker": ticker,
            "metric_values": {
                "implied_return_base": row.get("implied_return_base", "UNKNOWN"),
                "intrinsic_per_share_base": row.get("intrinsic_base", row.get("intrinsic_per_share_base", 82.0)),
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
            "gate_reasons": row.get("value_gate_reasons", ["MISSING_EV"]),
            "primary_blocker": row.get("primary_blocker", "MISSING_EV"),
            "inputs_used": {
                "current_price": {"value": 63.0, "derived_from": [f"price.{ticker}"]},
                "net_debt_proxy": {"value": 120.0, "derived_from": [f"net_debt.{ticker}"]},
                "dilution_rate_shares_cagr": {"value": 0.01, "derived_from": [f"dilution.{ticker}"]},
            },
            "net_debt_to_cfo": 1.2,
        },
        "valuation_row": {
            "ticker": ticker,
            "price_status": "OK",
            "price_reason_code": "OK",
            "valuation_status": "OK",
            "valuation_reason_code": "OK",
            "derived_from": [f"valuation.{ticker}"],
        },
        "shares_row": {"ticker": ticker, "shares_status": "OK", "shares_reason_code": "OK"},
        "fcf_row": {"ticker": ticker, "fcf_status": "OK", "fcf_reason_code": "OK"},
        "facts_row": {"ticker": ticker, "status": "OK", "fetch_reason_code": "OK"},
    }


def test_investable_now_requires_real_support_and_clean_underwriting():
    payload = compute_investment_readiness(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(),
        valuation_confidence_payload=_valuation_confidence_payload(),
        valuation_integrity_payload=_valuation_integrity_payload(),
        value_type_payload=_value_type_payload(),
        owner_quality_payload=_owner_quality_payload(),
        intangible_payload=_intangible_payload(),
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
        value_gate_status="PASS",
        primary_blocker="NONE",
        row_derived_from=["row.AAA"],
    )

    assert payload["investment_readiness_class"] == READY_INVESTABLE_NOW
    assert payload["blocker_stack_primary"] == "UNKNOWN"
    assert payload["primary_next_step"] == "DEEPER_UNDERWRITING"


def test_research_worthy_not_ready_classification_with_retryable_blockers():
    payload = compute_investment_readiness(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(),
        valuation_confidence_payload=_valuation_confidence_payload(
            fragility_status="MODERATE_FRAGILITY",
            confidence_class="MEDIUM_CONFIDENCE",
        ),
        valuation_integrity_payload=_valuation_integrity_payload(),
        value_type_payload=_value_type_payload(primary="ASSET_BACKED_VALUE"),
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
    assert payload["blocker_stack_primary"] == "MOS_UNASSESSABLE_MISSING_FACTS"
    assert payload["blocker_stack_retryable"] is True
    assert payload["primary_next_step"] == NEXT_CLEAR_EVIDENCE
    assert payload["primary_next_step_reason"] == "MOS_UNASSESSABLE_MISSING_FACTS"


def test_watch_only_classification_stays_distinct_from_ready_and_not_investable():
    payload = compute_investment_readiness(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(
            mos_to_floor=0.10,
            mos_classification="MODEST_MARGIN_OF_SAFETY",
            downside_support_type="LIMITED_SUPPORT",
        ),
        valuation_confidence_payload=_valuation_confidence_payload(
            support_count=1,
            support_types=["EPV_SUPPORT"],
            confidence_class="MEDIUM_CONFIDENCE",
            fragility_status="LOW_FRAGILITY",
            fragility_reason_codes=[],
        ),
        valuation_integrity_payload=_valuation_integrity_payload(),
        value_type_payload=_value_type_payload(primary="UNKNOWN_VALUE_TYPE"),
        owner_quality_payload=_owner_quality_payload(total=5.0),
        intangible_payload=_intangible_payload(total=4.0),
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
        value_gate_status="WATCH",
        primary_blocker="THIN_EDGE",
        row_derived_from=["row.AAA"],
    )

    assert payload["investment_readiness_class"] == READY_WATCH_ONLY
    assert payload["primary_next_step"] == "MONITOR_ONLY"
    assert payload["blocker_stack_primary"] == "VALUATION_FRAGILE"


def test_not_investable_classification_for_structural_weakness_and_no_mos():
    payload = compute_investment_readiness(
        "BBB",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(
            ticker="BBB",
            mos_to_floor=-0.10,
            mos_classification="NO_MARGIN_OF_SAFETY",
            downside_support_type="LIMITED_SUPPORT",
        ),
        valuation_confidence_payload=_valuation_confidence_payload(
            ticker="BBB",
            support_count=1,
            support_types=["EPV_SUPPORT"],
            confidence_class="LOW_CONFIDENCE",
            fragility_status="HIGH_FRAGILITY",
        ),
        valuation_integrity_payload=_valuation_integrity_payload(
            ticker="BBB",
            integrity_class="INTEGRITY_SUSPECT",
            integrity_reason_codes=["IDENTICAL_INTRINSIC_RANGE_CLUSTER"],
            consistency_status="INCONSISTENT",
        ),
        value_type_payload=_value_type_payload(ticker="BBB", primary="FRAGILE_VALUE"),
        owner_quality_payload=_owner_quality_payload(
            ticker="BBB",
            total=2.0,
            capital_allocation_score=1.0,
            reason_codes=["EXCESS_DILUTION", "DEBT_ACCUMULATION"],
        ),
        intangible_payload=_intangible_payload(
            ticker="BBB",
            total=2.0,
            owner_value_capture_score=1.0,
            owner_value_capture_reason_codes=["WEAK_PER_SHARE_CAPTURE"],
        ),
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
        fail_due_to_economic_weakness=True,
        primary_fail_domain="ECONOMICS",
        value_gate_status="FAIL",
        primary_blocker="STRUCTURAL_WEAK_ECONOMICS",
        row_derived_from=["row.BBB"],
    )

    assert payload["investment_readiness_class"] == READY_NOT_INVESTABLE
    assert payload["blocker_stack_primary"] == "STRUCTURAL_WEAK_ECONOMICS"
    assert payload["blocker_stack_secondary"] == "VALUATION_INTEGRITY_SUSPECT"
    assert payload["primary_next_step"] == NEXT_DO_NOT_ADVANCE


def test_blocker_stack_ordering_is_deterministic():
    payload = compute_investment_readiness(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(
            mos_to_floor=-0.05,
            mos_classification="NO_MARGIN_OF_SAFETY",
            downside_support_type="LIMITED_SUPPORT",
        ),
        valuation_confidence_payload=_valuation_confidence_payload(
            support_count=1,
            support_types=["EPV_SUPPORT"],
            confidence_class="LOW_CONFIDENCE",
            fragility_status="HIGH_FRAGILITY",
        ),
        valuation_integrity_payload=_valuation_integrity_payload(
            integrity_class="INTEGRITY_WARNING",
            integrity_reason_codes=["INTEGRITY_WARNING"],
        ),
        value_type_payload=_value_type_payload(primary="UNKNOWN_VALUE_TYPE"),
        owner_quality_payload=_owner_quality_payload(
            total=2.0,
            capital_allocation_score=1.0,
            reason_codes=["EXCESS_DILUTION"],
        ),
        intangible_payload=_intangible_payload(owner_value_capture_score=1.0, owner_value_capture_reason_codes=["WEAK_PER_SHARE_CAPTURE"]),
        price_status="UNKNOWN",
        shares_status="UNKNOWN",
        fcf_status="UNKNOWN",
        facts_status="UNKNOWN",
        valuation_status="UNKNOWN",
        facts_blocker_class="FACTS_RETRYABLE_TIMEOUT",
        facts_blocker_retryable=True,
        fail_due_to_economic_weakness=True,
        primary_fail_domain="ECONOMICS",
        value_gate_status="FAIL",
        primary_blocker="STRUCTURAL_WEAK_ECONOMICS",
        row_derived_from=["row.AAA"],
    )

    assert payload["blocker_stack_all"][:5] == [
        "STRUCTURAL_WEAK_ECONOMICS",
        "HIGH_DILUTION",
        "HIGH_LEVERAGE",
        "VALUATION_FRAGILE",
        "LOW_CONFIDENCE_VALUE",
    ]


def test_memo_pack_includes_investment_readiness_section():
    row = {
        "ticker": "AAA",
        "sector": "Software",
        "as_of_date": "2026-02-14",
        "source_depth_runs": [{"run_id": "depth_run", "sector": "Software", "as_of_date": "2026-02-14"}],
        "value_gate_status": "WATCH",
        "value_gate_reasons": ["FACTS_RETRYABLE_TIMEOUT"],
        "primary_blocker": "FACTS_RETRYABLE_TIMEOUT",
        "implied_return_base": 0.22,
        "implied_return_base_derived_from": ["trace.AAA.implied_return_base"],
        "intrinsic_per_share_base": 82.0,
        "intrinsic_per_share_base_derived_from": ["trace.AAA.intrinsic_per_share_base"],
        "mos_epv": 0.18,
        "mos_epv_derived_from": ["trace.AAA.mos_epv"],
        "mos_netnet": 0.05,
        "mos_netnet_derived_from": ["trace.AAA.mos_netnet"],
        "owner_earnings_yield_ev_3y": 0.06,
        "owner_earnings_yield_ev_3y_derived_from": ["trace.AAA.owner_yield"],
        "fcf_yield_ev_3y": 0.05,
        "yield_metric_used": "owner_earnings_yield_ev_3y",
        "price_status": "OK",
        "valuation_status": "OK",
        "shares_status": "OK",
        "fcf_status": "OK",
        "facts_status": "UNKNOWN",
        "price_reason_code": "OK",
        "valuation_reason_code": "OK",
        "shares_reason_code": "OK",
        "fcf_reason_code": "OK",
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
        "evidence_sufficiency_class": "PARTIAL_FOR_MOS",
        "evidence_sufficiency_reason_codes": ["PRICE_AVAILABLE", "SHARES_AVAILABLE", "MISSING_FACTS"],
        "mos_assessment_status": "MOS_UNASSESSABLE",
        "mos_guardrail_reason_codes": ["MOS_BLOCKED_BY_MISSING_FACTS"],
        "blocker_stack_primary": "FACTS_RETRYABLE_TIMEOUT",
        "blocker_stack_secondary": "LOW_CONFIDENCE_VALUE",
        "blocker_stack_all": ["FACTS_RETRYABLE_TIMEOUT", "LOW_CONFIDENCE_VALUE"],
        "blocker_stack_retryable": True,
        "blocker_stack_structural": False,
        "readiness_support_present": ["ADEQUATE_MOS", "MULTI_SUPPORT_VALUE_CASE", "INTEGRITY_OK"],
        "readiness_support_missing": ["FACTS_MISSING"],
        "readiness_support_headwinds": ["LOW_CONFIDENCE_VALUE"],
        "primary_next_step": NEXT_CLEAR_EVIDENCE,
        "primary_next_step_reason": "FACTS_RETRYABLE_TIMEOUT",
        "value_type_primary": "EARNINGS_POWER_VALUE",
        "value_type_reason_codes": ["EPV_DRIVEN", "NORMALIZED_EARNINGS_DRIVEN"],
        "value_type_support_summary": "Anchored to earnings power support.",
        "owner_earnings_stability_score": 4.0,
        "capital_allocation_score": 3.0,
        "cash_conversion_score": 3.0,
        "oe_quality_total": 9.0,
        "oe_quality_reason_codes": ["SHAREHOLDER_FRIENDLY"],
        "gross_margin_durability_score": 4.0,
        "balance_sheet_optionality_score": 4.0,
        "cycle_resilience_score": 3.0,
        "rnd_productivity_score": 3.0,
        "sga_leverage_score": 3.0,
        "owner_value_capture_score": 4.0,
        "intangible_economics_total": 10.0,
        "rnd_productivity_reason_codes": [],
        "sga_leverage_reason_codes": [],
        "owner_value_capture_reason_codes": ["STRONG_OWNER_VALUE_CAPTURE"],
        "intangible_economics_reason_codes": ["STRONG_OWNER_VALUE_CAPTURE"],
        "derived_from": ["shortlist.AAA"],
    }
    memo = build_investment_memo(
        "AAA",
        universe_run_id="universe_run",
        batch_run_id="batch_run",
        sources=_memo_sources(row),
    )
    markdown = _memo_markdown(memo)

    assert memo["investment_readiness_blocker_stack"]["investment_readiness_class"] == READY_RESEARCH_WORTHY
    assert "## Investment Readiness / Blocker Stack" in markdown
    assert "## Evidence Sufficiency for Margin of Safety" in markdown
    assert "investment_readiness_class: `RESEARCH_WORTHY_NOT_READY`" in markdown
    assert "primary_next_step: `CLEAR_EVIDENCE_BLOCKERS`" in markdown


def test_value_first_ready_ranking_is_deterministic():
    rows = [
        {
            "ticker": "AAA",
            "scout_status": "PASS",
            "investment_readiness_class": READY_INVESTABLE_NOW,
            "mos_to_floor": 0.18,
            "valuation_confidence_class": "MEDIUM_CONFIDENCE",
            "valuation_integrity_class": "INTEGRITY_OK",
            "value_type_primary": "EARNINGS_POWER_VALUE",
            "owner_earnings_yield_ev_3y": 0.05,
            "oe_quality_total": 7.0,
            "intangible_economics_total": 6.0,
            "memory_priority_total": 2,
        },
        {
            "ticker": "BBB",
            "scout_status": "PASS",
            "investment_readiness_class": READY_RESEARCH_WORTHY,
            "mos_to_floor": 0.35,
            "valuation_confidence_class": "HIGH_CONFIDENCE",
            "valuation_integrity_class": "INTEGRITY_OK",
            "value_type_primary": "EARNINGS_POWER_VALUE",
            "owner_earnings_yield_ev_3y": 0.06,
            "oe_quality_total": 8.0,
            "intangible_economics_total": 7.0,
            "memory_priority_total": 3,
        },
    ]

    ordered = sorted(rows, key=lambda row: ranking_sort_key(row, policy="value_first_ready"))
    assert [row["ticker"] for row in ordered] == ["AAA", "BBB"]


def test_promotion_and_escalation_visibility_fields_appear_without_overriding_fail():
    master_shortlist = {
        "campaign_run_id": "campaign_ready",
        "rows": [
            _readiness_row(
                ticker="AAA",
                readiness_class=READY_RESEARCH_WORTHY,
                blocker_primary="FACTS_RETRYABLE_TIMEOUT",
                next_step=NEXT_CLEAR_EVIDENCE,
                gate_status="WATCH",
                facts_blocker_class="FACTS_RETRYABLE_TIMEOUT",
                facts_blocker_retryable=True,
            ),
            _readiness_row(
                ticker="BBB",
                readiness_class=READY_NOT_INVESTABLE,
                blocker_primary="STRUCTURAL_WEAK_ECONOMICS",
                next_step=NEXT_DO_NOT_ADVANCE,
                gate_status="FAIL",
            ),
        ],
    }
    master_watchlist = {
        "campaign_run_id": "campaign_ready",
        "tickers": {
            "AAA": {
                "latest_value_gate_status": "WATCH",
                "latest_implied_return_base": 0.20,
                "latest_primary_blocker": "FACTS_RETRYABLE_TIMEOUT",
                "appearances_count": 2,
                "history": [
                    {
                        "campaign_item": "core",
                        "universe_run_id": "campaign_ready__aaa",
                        "value_gate_status": "WATCH",
                        "implied_return_base": 0.20,
                        "primary_blocker": "FACTS_RETRYABLE_TIMEOUT",
                        "last_rank": 1,
                    }
                ],
            },
            "BBB": {
                "latest_value_gate_status": "FAIL",
                "latest_implied_return_base": 0.08,
                "latest_primary_blocker": "STRUCTURAL_WEAK_ECONOMICS",
                "appearances_count": 2,
                "history": [
                    {
                        "campaign_item": "core",
                        "universe_run_id": "campaign_ready__bbb",
                        "value_gate_status": "FAIL",
                        "implied_return_base": 0.08,
                        "primary_blocker": "STRUCTURAL_WEAK_ECONOMICS",
                        "last_rank": 3,
                    }
                ],
            },
        },
    }

    promotion = build_promotion_state("campaign_ready", master_watchlist, master_shortlist)
    rows = {row["ticker"]: row for row in promotion["rows"]}

    assert rows["AAA"]["priority_lane"] == LANE_2_RESEARCH_QUEUE
    assert rows["AAA"]["investment_readiness_class"] == READY_RESEARCH_WORTHY
    assert rows["AAA"]["evidence_sufficiency_class"] == "INSUFFICIENT_FOR_MOS"
    assert rows["AAA"]["mos_assessment_status"] == "MOS_UNASSESSABLE"
    assert "BLOCKED_BUT_RESEARCH_WORTHY" in rows["AAA"]["promotion_reason_codes"]
    assert "MOS_UNASSESSABLE_EVIDENCE_GAP" in rows["AAA"]["promotion_reason_codes"]
    assert rows["BBB"]["priority_lane"] == LANE_4_DEPRIORITIZED
    assert "NOT_INVESTABLE_STRUCTURAL" in rows["BBB"]["promotion_reason_codes"]

    escalation = build_escalation_plan(
        "campaign_ready",
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

    assert aaa_queue[0]["investment_readiness_class"] == READY_RESEARCH_WORTHY
    assert aaa_queue[0]["evidence_sufficiency_class"] == "INSUFFICIENT_FOR_MOS"
    assert aaa_queue[0]["mos_assessment_status"] == "MOS_UNASSESSABLE"
    assert aaa_queue[0]["blocker_stack_primary"] == "FACTS_RETRYABLE_TIMEOUT"
    assert "BLOCKED_BUT_RESEARCH_WORTHY" in aaa_queue[0]["priority_support_codes"]
    assert "MOS_UNASSESSABLE_EVIDENCE_GAP" in aaa_queue[0]["priority_support_codes"]
    assert bbb_queue[0]["priority_lane"] == LANE_4_DEPRIORITIZED
    assert "NOT_INVESTABLE_STRUCTURAL" in bbb_queue[0]["priority_support_codes"]


def test_investment_readiness_cli_open(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "investment_readiness_cli"
    aaa = compute_investment_readiness(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(),
        valuation_confidence_payload=_valuation_confidence_payload(),
        valuation_integrity_payload=_valuation_integrity_payload(),
        value_type_payload=_value_type_payload(),
        owner_quality_payload=_owner_quality_payload(),
        intangible_payload=_intangible_payload(),
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
        value_gate_status="PASS",
        primary_blocker="NONE",
        row_derived_from=["row.AAA"],
    )
    bbb = compute_investment_readiness(
        "BBB",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(ticker="BBB"),
        valuation_confidence_payload=_valuation_confidence_payload(
            ticker="BBB",
            fragility_status="MODERATE_FRAGILITY",
            confidence_class="MEDIUM_CONFIDENCE",
        ),
        valuation_integrity_payload=_valuation_integrity_payload(ticker="BBB"),
        value_type_payload=_value_type_payload(ticker="BBB", primary="ASSET_BACKED_VALUE"),
        owner_quality_payload=_owner_quality_payload(ticker="BBB"),
        intangible_payload=_intangible_payload(ticker="BBB"),
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
        row_derived_from=["row.BBB"],
    )

    write_investment_readiness_for_run(
        run_id=run_id,
        as_of_date="2026-02-14",
        tickers=["AAA", "BBB"],
        output_path=cfg.outputs_dir / "universe" / run_id / "investment_readiness.json",
        scoreboard_rows=[
            {"ticker": "AAA", "investment_readiness_detail": aaa},
            {"ticker": "BBB", "investment_readiness_detail": bbb},
        ],
    )

    result = runner.invoke(app, ["universe-investment-readiness-open", "--run-id", run_id])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "OK"
    assert payload["counts_by_readiness_class"][READY_INVESTABLE_NOW] == 1
    assert payload["counts_by_readiness_class"][READY_RESEARCH_WORTHY] == 1
    assert payload["retryable_blocker_count"] == 1
