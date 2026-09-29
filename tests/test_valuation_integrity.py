from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.escalation import build_escalation_plan
from app.universe.memo_pack import _memo_markdown, build_investment_memo
from app.universe.promotion import LANE_2_RESEARCH_QUEUE, LANE_4_DEPRIORITIZED, build_promotion_state
from app.universe.ranking import ranking_sort_key
from app.valuation.valuation_confidence import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    apply_valuation_integrity_headwind,
    compute_valuation_confidence,
)
from app.valuation.valuation_integrity import (
    CONSISTENCY_CONSISTENT,
    CONSISTENCY_INCONSISTENT,
    INTEGRITY_OK,
    INTEGRITY_SUSPECT,
    INTEGRITY_WARNING,
    REASON_HIGH_CONFIDENCE_WITH_HIGH_FRAGILITY,
    REASON_IDENTICAL_INTRINSIC_RANGE_CLUSTER,
    REASON_IDENTICAL_NORMALIZED_EARNINGS_CLUSTER,
    REASON_METHOD_OWNER_EARNINGS_WITHOUT_OWNER_SUPPORT,
    REASON_MOS_WITHOUT_PRICE_OR_VALUE_SUPPORT,
    REASON_NETNET_FLOOR_WITHOUT_NETNET_SUPPORT,
    compute_valuation_integrity,
    write_valuation_integrity_for_run,
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
    ticker: str,
    normalized: float | str = 100.0,
    floor: float | str = 70.0,
    base: float | str = 85.0,
    ceiling: float | str = 95.0,
    method: str = "FCF_SELECTED",
    status: str = "OK",
    downside_support_type: str = "EARNINGS_POWER_SUPPORT",
    mos_classification: str = "ADEQUATE_MARGIN_OF_SAFETY",
    range_reason_codes: list[str] | None = None,
) -> dict:
    return {
        "normalized_earnings_power_value": normalized,
        "normalized_earnings_power_method_used": method,
        "normalized_earnings_power_status": status,
        "normalized_earnings_power_reason_codes": [method] if normalized != "UNKNOWN" else ["INSUFFICIENT_NORMALIZED_INPUTS"],
        "intrinsic_floor": floor,
        "intrinsic_base": base,
        "intrinsic_ceiling": ceiling,
        "mos_to_floor": 0.30 if floor != "UNKNOWN" else "UNKNOWN",
        "mos_to_base": 0.45 if base != "UNKNOWN" else "UNKNOWN",
        "mos_classification": mos_classification,
        "downside_support_type": downside_support_type,
        "downside_support_status": "OK" if downside_support_type != "UNKNOWN_SUPPORT" else "UNKNOWN",
        "valuation_range_reason_codes": list(range_reason_codes or ["BASE_FROM_NORMALIZED_EARNINGS_POWER"]),
        "downside_support_reason_codes": [downside_support_type] if downside_support_type != "UNKNOWN_SUPPORT" else [],
        "derived_from": [f"intrinsic.{ticker}"],
        "claims": {
            "normalized_earnings_power_value": _claim(normalized, f"intrinsic.{ticker}.normalized"),
            "intrinsic_floor": _claim(floor, f"intrinsic.{ticker}.floor"),
            "intrinsic_base": _claim(base, f"intrinsic.{ticker}.base"),
            "intrinsic_ceiling": _claim(ceiling, f"intrinsic.{ticker}.ceiling"),
            "mos_to_floor": _claim(0.30 if floor != "UNKNOWN" else "UNKNOWN", f"intrinsic.{ticker}.mos_floor"),
            "mos_to_base": _claim(0.45 if base != "UNKNOWN" else "UNKNOWN", f"intrinsic.{ticker}.mos_base"),
        },
    }


def _valuation_confidence_payload(
    *,
    ticker: str,
    support_types: list[str],
    support_count: int,
    fragility_status: str,
    confidence_class: str,
    fragility_reason_codes: list[str] | None = None,
    convergence_status: str = "STRONG_CONVERGENCE",
) -> dict:
    return {
        "valuation_support_count": support_count,
        "valuation_support_types_present": list(support_types),
        "valuation_support_count_reason_codes": list(support_types or ["NO_VALUATION_SUPPORTS_PRESENT"]),
        "valuation_convergence_status": convergence_status,
        "valuation_convergence_band_pct": 0.18 if support_count >= 2 else "UNKNOWN",
        "valuation_convergence_reason_codes": ["SUPPORTS_CONFLICT"] if convergence_status == "WEAK_CONVERGENCE" else [],
        "valuation_fragility_status": fragility_status,
        "valuation_fragility_reason_codes": list(fragility_reason_codes or []),
        "valuation_confidence_class": confidence_class,
        "valuation_confidence_reason_codes": ["MULTI_SUPPORT_VALUE_CASE"] if support_count >= 2 else ["SINGLE_SUPPORT_FRAGILE"],
        "derived_from": [f"confidence.{ticker}"],
        "claims": {
            "valuation_support_count": _claim(support_count, f"confidence.{ticker}.supports"),
            "valuation_confidence_class": _claim(confidence_class, f"confidence.{ticker}.class"),
            "valuation_fragility_status": _claim(fragility_status, f"confidence.{ticker}.fragility"),
        },
    }


def _value_type_payload(*, ticker: str, primary: str = "EARNINGS_POWER_VALUE") -> dict:
    return {
        "value_type_primary": primary,
        "value_type_secondary": None,
        "value_type_reason_codes": ["NORMALIZED_EARNINGS_DRIVEN"],
        "value_type_support_summary": "Anchored to normalized earnings power or EPV support.",
        "value_type_derived_from": [f"value_type.{ticker}"],
        "claims": {
            "value_type_primary": _claim(primary, f"value_type.{ticker}.primary"),
        },
    }


def _score_row(
    *,
    ticker: str,
    intrinsic_payload: dict,
    confidence_payload: dict,
    value_type_payload: dict,
    price_status: str = "OK",
    shares_status: str = "OK",
    fcf_status: str = "OK",
    facts_status: str = "OK",
    valuation_status: str = "OK",
) -> dict:
    return {
        "ticker": ticker,
        "intrinsic_discipline_detail": intrinsic_payload,
        "valuation_confidence_detail": confidence_payload,
        "value_type_detail": value_type_payload,
        "price_status": price_status,
        "shares_status": shares_status,
        "fcf_status": fcf_status,
        "facts_status": facts_status,
        "valuation_status": valuation_status,
    }


def test_identical_intrinsic_ranges_are_flagged():
    aaa = _score_row(
        ticker="AAA",
        intrinsic_payload=_intrinsic_payload(ticker="AAA", floor=70.0, base=85.0, ceiling=95.0),
        confidence_payload=_valuation_confidence_payload(
            ticker="AAA",
            support_types=["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"],
            support_count=2,
            fragility_status="LOW_FRAGILITY",
            confidence_class="MEDIUM_CONFIDENCE",
        ),
        value_type_payload=_value_type_payload(ticker="AAA"),
    )
    bbb = _score_row(
        ticker="BBB",
        intrinsic_payload=_intrinsic_payload(ticker="BBB", floor=70.0, base=85.0, ceiling=95.0),
        confidence_payload=_valuation_confidence_payload(
            ticker="BBB",
            support_types=["NETNET_ASSET_SUPPORT", "EPV_SUPPORT"],
            support_count=2,
            fragility_status="LOW_FRAGILITY",
            confidence_class="MEDIUM_CONFIDENCE",
        ),
        value_type_payload=_value_type_payload(ticker="BBB", primary="ASSET_BACKED_VALUE"),
    )

    payload = write_valuation_integrity_for_run(
        run_id="integrity_range_cluster",
        as_of_date="2026-02-14",
        tickers=["AAA", "BBB"],
        output_path=Path("/tmp/valuation_integrity_range.json"),
        scoreboard_rows=[aaa, bbb],
    )
    rows = {row["ticker"]: row for row in payload["rows"]}

    assert payload["exact_uniformity_cluster_count"] >= 1
    assert REASON_IDENTICAL_INTRINSIC_RANGE_CLUSTER in rows["AAA"]["valuation_uniformity_reason_codes"]
    assert REASON_IDENTICAL_INTRINSIC_RANGE_CLUSTER in rows["BBB"]["valuation_uniformity_reason_codes"]


def test_identical_normalized_earnings_cluster_is_flagged_when_supports_differ():
    aaa = _score_row(
        ticker="AAA",
        intrinsic_payload=_intrinsic_payload(ticker="AAA", normalized=100.0, floor=72.0, base=88.0, ceiling=99.0),
        confidence_payload=_valuation_confidence_payload(
            ticker="AAA",
            support_types=["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"],
            support_count=2,
            fragility_status="LOW_FRAGILITY",
            confidence_class="MEDIUM_CONFIDENCE",
        ),
        value_type_payload=_value_type_payload(ticker="AAA"),
    )
    bbb = _score_row(
        ticker="BBB",
        intrinsic_payload=_intrinsic_payload(ticker="BBB", normalized=100.0, floor=60.0, base=90.0, ceiling=120.0),
        confidence_payload=_valuation_confidence_payload(
            ticker="BBB",
            support_types=["OWNER_EARNINGS_VALUE_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"],
            support_count=2,
            fragility_status="LOW_FRAGILITY",
            confidence_class="MEDIUM_CONFIDENCE",
        ),
        value_type_payload=_value_type_payload(ticker="BBB"),
    )

    payload = write_valuation_integrity_for_run(
        run_id="integrity_norm_cluster",
        as_of_date="2026-02-14",
        tickers=["AAA", "BBB"],
        output_path=Path("/tmp/valuation_integrity_norm.json"),
        scoreboard_rows=[aaa, bbb],
    )
    rows = {row["ticker"]: row for row in payload["rows"]}

    assert REASON_IDENTICAL_NORMALIZED_EARNINGS_CLUSTER in rows["AAA"]["valuation_uniformity_reason_codes"]
    assert REASON_IDENTICAL_NORMALIZED_EARNINGS_CLUSTER in rows["BBB"]["valuation_uniformity_reason_codes"]


def test_method_support_inconsistencies_are_flagged():
    payload = compute_valuation_integrity(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(
            ticker="AAA",
            method="OWNER_EARNINGS_SELECTED",
            downside_support_type="UNKNOWN_SUPPORT",
            mos_classification="ADEQUATE_MARGIN_OF_SAFETY",
            range_reason_codes=["FLOOR_FROM_NETNET"],
        ),
        valuation_confidence_payload=_valuation_confidence_payload(
            ticker="AAA",
            support_types=["EPV_SUPPORT"],
            support_count=1,
            fragility_status="HIGH_FRAGILITY",
            confidence_class="HIGH_CONFIDENCE",
            fragility_reason_codes=["SINGLE_SUPPORT_ONLY"],
        ),
        value_type_payload=_value_type_payload(ticker="AAA", primary="EARNINGS_POWER_VALUE"),
        price_status="UNKNOWN",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
    )

    assert payload["valuation_consistency_status"] == CONSISTENCY_INCONSISTENT
    assert REASON_METHOD_OWNER_EARNINGS_WITHOUT_OWNER_SUPPORT in payload["valuation_consistency_reason_codes"]
    assert REASON_NETNET_FLOOR_WITHOUT_NETNET_SUPPORT in payload["valuation_consistency_reason_codes"]
    assert REASON_MOS_WITHOUT_PRICE_OR_VALUE_SUPPORT in payload["valuation_consistency_reason_codes"]
    assert REASON_HIGH_CONFIDENCE_WITH_HIGH_FRAGILITY in payload["valuation_consistency_reason_codes"]
    assert payload["valuation_integrity_class"] == INTEGRITY_SUSPECT


def test_clean_isolated_rows_remain_integrity_ok():
    payload = compute_valuation_integrity(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(ticker="AAA", range_reason_codes=["BASE_FROM_NORMALIZED_EARNINGS_POWER"]),
        valuation_confidence_payload=_valuation_confidence_payload(
            ticker="AAA",
            support_types=["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"],
            support_count=2,
            fragility_status="LOW_FRAGILITY",
            confidence_class="MEDIUM_CONFIDENCE",
        ),
        value_type_payload=_value_type_payload(ticker="AAA"),
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
    )

    assert payload["valuation_consistency_status"] == CONSISTENCY_CONSISTENT
    assert payload["valuation_integrity_class"] == INTEGRITY_OK
    assert payload["valuation_uniformity_reason_codes"] == []


def test_valuation_confidence_degrades_when_integrity_is_questionable():
    high = compute_valuation_confidence(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(ticker="AAA", normalized=96.0, floor=72.0, base=84.0, ceiling=92.0),
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
        epv_per_share=82.0,
        epv_refs=["gd.epv"],
        netnet_per_share=74.0,
        netnet_refs=["gd.netnet"],
    )

    warning = apply_valuation_integrity_headwind(
        high,
        integrity_class=INTEGRITY_WARNING,
        integrity_reason_codes=[REASON_IDENTICAL_INTRINSIC_RANGE_CLUSTER],
        derived_from=["integrity.warning"],
    )
    suspect = apply_valuation_integrity_headwind(
        high,
        integrity_class=INTEGRITY_SUSPECT,
        integrity_reason_codes=[REASON_POSSIBLE_STALE_ARTIFACT_REUSE := "POSSIBLE_STALE_ARTIFACT_REUSE"],
        derived_from=["integrity.suspect"],
    )

    assert high["valuation_confidence_class"] == CONFIDENCE_HIGH
    assert warning["valuation_confidence_class"] == CONFIDENCE_MEDIUM
    assert suspect["valuation_confidence_class"] == CONFIDENCE_LOW
    assert "INTEGRITY_WARNING_HEADWIND" in warning["valuation_confidence_reason_codes"]
    assert "INTEGRITY_SUSPECT_HEADWIND" in suspect["valuation_confidence_reason_codes"]


def test_memo_pack_includes_valuation_integrity_section():
    row = {
        "ticker": "AAA",
        "sector": "Software",
        "as_of_date": "2026-02-14",
        "source_depth_runs": [{"run_id": "depth_run", "sector": "Software", "as_of_date": "2026-02-14"}],
        "value_gate_status": "WATCH",
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
        "facts_status": "OK",
        "price_reason_code": "OK",
        "valuation_reason_code": "OK",
        "shares_reason_code": "OK",
        "fcf_reason_code": "OK",
        "facts_reason_code": "OK",
        "primary_blocker": "MISSING_EV",
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
        "valuation_support_count": 3,
        "valuation_support_types_present": ["NETNET_ASSET_SUPPORT", "EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"],
        "valuation_support_count_reason_codes": ["NETNET_ASSET_SUPPORT"],
        "valuation_convergence_status": "STRONG_CONVERGENCE",
        "valuation_convergence_band_pct": 0.14,
        "valuation_convergence_reason_codes": [],
        "valuation_fragility_status": "LOW_FRAGILITY",
        "valuation_fragility_reason_codes": ["FRAGILITY_REDUCED_BY_MULTI_SUPPORT"],
        "valuation_confidence_class": "MEDIUM_CONFIDENCE",
        "valuation_confidence_reason_codes": ["MULTI_SUPPORT_VALUE_CASE", "INTEGRITY_WARNING_HEADWIND"],
        "valuation_integrity_class": "INTEGRITY_WARNING",
        "valuation_integrity_reason_codes": [REASON_IDENTICAL_INTRINSIC_RANGE_CLUSTER],
        "valuation_consistency_status": "CONSISTENT",
        "valuation_consistency_reason_codes": [],
        "valuation_uniformity_group_id": "RANGE_001",
        "valuation_uniformity_reason_codes": [REASON_IDENTICAL_INTRINSIC_RANGE_CLUSTER],
        "valuation_input_fingerprint": "abc123",
        "valuation_input_provenance_summary": {"normalized_earnings_power_method_used": "FCF_SELECTED"},
        "valuation_integrity_class_derived_from": ["integrity.AAA"],
        "value_type_primary": "EARNINGS_POWER_VALUE",
        "value_type_secondary": None,
        "value_type_reason_codes": ["NORMALIZED_EARNINGS_DRIVEN"],
        "value_type_support_summary": "Anchored to normalized earnings power or EPV support.",
        "owner_earnings_stability_score": 4.0,
        "capital_allocation_score": 3.0,
        "cash_conversion_score": 3.0,
        "oe_quality_total": 10.0,
        "oe_quality_reason_codes": ["SHAREHOLDER_FRIENDLY"],
        "gross_margin_durability_score": 4.0,
        "balance_sheet_optionality_score": 4.0,
        "cycle_resilience_score": 3.0,
        "rnd_productivity_score": 3.0,
        "sga_leverage_score": 3.0,
        "owner_value_capture_score": 4.0,
        "intangible_economics_total": 12.0,
        "rnd_productivity_reason_codes": [],
        "sga_leverage_reason_codes": [],
        "owner_value_capture_reason_codes": ["STRONG_OWNER_VALUE_CAPTURE"],
        "intangible_economics_reason_codes": ["STRONG_OWNER_VALUE_CAPTURE"],
        "composite_score_total": 66.0,
        "derived_from": ["shortlist.AAA"],
        "value_type_derived_from": ["value_type.AAA"],
    }
    sources = {
        "shortlist_row": row,
        "score_row": {
            "ticker": "AAA",
            "metric_values": {
                "implied_return_base": 0.22,
                "intrinsic_per_share_base": 82.0,
                "current_price": 63.0,
                "mos_epv": 0.18,
                "mos_netnet": 0.05,
                "owner_earnings_yield_ev_3y": 0.06,
                "fcf_yield_ev_3y": 0.05,
                "risk_factor_keyword_delta": 1.0,
                "quality_score": 16.0,
                "risk_penalty": -2.0,
            },
            "metric_traces": {"implied_return_base": {"derived_from": ["score.implied_return"]}},
            "derived_from": ["score.AAA"],
        },
        "gate_row": {
            "ticker": "AAA",
            "gate_status": "WATCH",
            "gate_reasons": ["MISSING_EV"],
            "primary_blocker": "MISSING_EV",
            "inputs_used": {"current_price": {"value": 63.0, "derived_from": ["price.AAA"]}},
            "net_debt_to_cfo": 1.2,
        },
        "valuation_row": {"ticker": "AAA", "price_status": "OK", "price_reason_code": "OK", "valuation_status": "OK", "valuation_reason_code": "OK", "derived_from": ["valuation.AAA"]},
        "shares_row": {"ticker": "AAA", "shares_status": "OK", "shares_reason_code": "OK"},
        "fcf_row": {"ticker": "AAA", "fcf_status": "OK", "fcf_reason_code": "OK"},
        "facts_row": {"ticker": "AAA", "status": "OK", "fetch_reason_code": "OK"},
    }

    memo = build_investment_memo("AAA", universe_run_id="u1", batch_run_id="b1", sources=sources)
    markdown = _memo_markdown(memo)

    assert "## Valuation Integrity Audit" in markdown
    assert "valuation_integrity_class: `INTEGRITY_WARNING`" in markdown
    assert memo["valuation_integrity_audit"]["valuation_uniformity_group_id"] == "RANGE_001"


def test_value_first_trustworthy_ranking_is_deterministic():
    rows = [
        {
            "ticker": "AAA",
            "scout_status": "PASS",
            "mos_to_floor": 0.20,
            "implied_return_base": 0.25,
            "valuation_confidence_class": "HIGH_CONFIDENCE",
            "valuation_integrity_class": "INTEGRITY_SUSPECT",
            "value_type_primary": "ASSET_BACKED_VALUE",
            "owner_earnings_yield_ev_3y": 0.05,
            "oe_quality_total": 8.0,
            "intangible_economics_total": 7.0,
            "memory_priority_total": 1,
        },
        {
            "ticker": "BBB",
            "scout_status": "PASS",
            "mos_to_floor": 0.20,
            "implied_return_base": 0.25,
            "valuation_confidence_class": "HIGH_CONFIDENCE",
            "valuation_integrity_class": "INTEGRITY_OK",
            "value_type_primary": "ASSET_BACKED_VALUE",
            "owner_earnings_yield_ev_3y": 0.05,
            "oe_quality_total": 8.0,
            "intangible_economics_total": 7.0,
            "memory_priority_total": 1,
        },
    ]

    ordered = sorted(rows, key=lambda row: ranking_sort_key(row, policy="value_first_trustworthy"))
    assert [row["ticker"] for row in ordered] == ["BBB", "AAA"]


def test_promotion_and_escalation_visibility_include_integrity_without_overriding_fail():
    master_shortlist = {
        "campaign_run_id": "campaign_integrity",
        "rows": [
            {
                "ticker": "AAA",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "campaign_integrity__core", "batch_run_id": "campaign_integrity__core_depth_batch"}],
                "best_rank_seen": 1,
                "value_gate_status": "WATCH",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": 0.20,
                "mos_epv": 0.16,
                "owner_earnings_yield_ev_3y": 0.05,
                "yield_metric_used": "owner_earnings_yield_ev_3y",
                "primary_blocker": "MISSING_EV",
                "latest_primary_blocker": "MISSING_EV",
                "memo_path": "memo/AAA.md",
                "composite_score_total": 60.0,
                "mos_to_floor": 0.28,
                "mos_classification": "ADEQUATE_MARGIN_OF_SAFETY",
                "downside_support_type": "EARNINGS_POWER_SUPPORT",
                "valuation_support_count": 3,
                "valuation_support_types_present": ["NETNET_ASSET_SUPPORT", "EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"],
                "valuation_support_count_reason_codes": ["NETNET_ASSET_SUPPORT"],
                "valuation_convergence_status": "STRONG_CONVERGENCE",
                "valuation_convergence_band_pct": 0.18,
                "valuation_convergence_reason_codes": [],
                "valuation_fragility_status": "LOW_FRAGILITY",
                "valuation_fragility_reason_codes": ["FRAGILITY_REDUCED_BY_MULTI_SUPPORT"],
                "valuation_confidence_class": "MEDIUM_CONFIDENCE",
                "valuation_confidence_reason_codes": ["MULTI_SUPPORT_VALUE_CASE", "INTEGRITY_WARNING_HEADWIND"],
                "valuation_integrity_class": "INTEGRITY_WARNING",
                "valuation_integrity_reason_codes": [REASON_IDENTICAL_INTRINSIC_RANGE_CLUSTER],
                "value_type_primary": "EARNINGS_POWER_VALUE",
                "value_type_reason_codes": ["NORMALIZED_EARNINGS_DRIVEN"],
                "value_type_support_summary": "Anchored to normalized earnings power or EPV support.",
                "derived_from": ["shortlist.AAA"],
            },
            {
                "ticker": "BBB",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "campaign_integrity__core", "batch_run_id": "campaign_integrity__core_depth_batch"}],
                "best_rank_seen": 3,
                "value_gate_status": "FAIL",
                "latest_value_gate_status": "FAIL",
                "implied_return_base": 0.25,
                "mos_epv": 0.20,
                "owner_earnings_yield_ev_3y": 0.06,
                "yield_metric_used": "owner_earnings_yield_ev_3y",
                "primary_blocker": "PRICE_UNKNOWN",
                "latest_primary_blocker": "PRICE_UNKNOWN",
                "memo_path": "memo/BBB.md",
                "composite_score_total": 65.0,
                "mos_to_floor": 0.40,
                "mos_classification": "ADEQUATE_MARGIN_OF_SAFETY",
                "downside_support_type": "ASSET_SUPPORT",
                "valuation_support_count": 1,
                "valuation_support_types_present": ["EPV_SUPPORT"],
                "valuation_support_count_reason_codes": ["EPV_SUPPORT"],
                "valuation_convergence_status": "CONVERGENCE_UNKNOWN",
                "valuation_convergence_band_pct": "UNKNOWN",
                "valuation_convergence_reason_codes": ["SINGLE_SUPPORT_ONLY"],
                "valuation_fragility_status": "HIGH_FRAGILITY",
                "valuation_fragility_reason_codes": ["SINGLE_SUPPORT_ONLY"],
                "valuation_confidence_class": "LOW_CONFIDENCE",
                "valuation_confidence_reason_codes": ["SINGLE_SUPPORT_FRAGILE", "INTEGRITY_SUSPECT_HEADWIND"],
                "valuation_integrity_class": "INTEGRITY_SUSPECT",
                "valuation_integrity_reason_codes": [REASON_METHOD_OWNER_EARNINGS_WITHOUT_OWNER_SUPPORT],
                "value_type_primary": "FRAGILE_VALUE",
                "value_type_reason_codes": ["HIGH_FRAGILITY_CASE"],
                "value_type_support_summary": "Appears cheap, but support is thin, conflicted, or fragile.",
                "derived_from": ["shortlist.BBB"],
            },
        ],
    }
    master_watchlist = {
        "campaign_run_id": "campaign_integrity",
        "tickers": {
            "AAA": {
                "latest_value_gate_status": "WATCH",
                "latest_implied_return_base": 0.20,
                "latest_primary_blocker": "MISSING_EV",
                "appearances_count": 2,
                "history": [
                    {"campaign_item": "core", "universe_run_id": "campaign_integrity__core", "value_gate_status": "WATCH", "implied_return_base": 0.20, "primary_blocker": "MISSING_EV", "last_rank": 1},
                ],
            },
            "BBB": {
                "latest_value_gate_status": "FAIL",
                "latest_implied_return_base": 0.25,
                "latest_primary_blocker": "PRICE_UNKNOWN",
                "appearances_count": 2,
                "history": [
                    {"campaign_item": "core", "universe_run_id": "campaign_integrity__core", "value_gate_status": "FAIL", "implied_return_base": 0.25, "primary_blocker": "PRICE_UNKNOWN", "last_rank": 3},
                ],
            },
        },
    }

    promotion = build_promotion_state("campaign_integrity", master_watchlist, master_shortlist)
    rows = {row["ticker"]: row for row in promotion["rows"]}
    assert rows["AAA"]["priority_lane"] == LANE_2_RESEARCH_QUEUE
    assert rows["AAA"]["valuation_integrity_class"] == "INTEGRITY_WARNING"
    assert rows["BBB"]["priority_lane"] == LANE_4_DEPRIORITIZED
    assert "INTEGRITY_SUSPECT_HEADWIND" in rows["BBB"]["promotion_reason_codes"]

    escalation = build_escalation_plan(
        "campaign_integrity",
        promotion,
        {
            "lane_1_high_priority": [],
            "lane_2_research_queue": [rows["AAA"]],
            "lane_3_monitor": [],
            "lane_4_deprioritized": [rows["BBB"]],
        },
        config={"as_of_date": "2026-02-14", "source_campaign_file": "data/universe/sample_campaign.json", "top_n": 10, "policy": "value_first_trustworthy"},
    )
    aaa_queue = [row for row in escalation["queue"] if row["ticker"] == "AAA"]
    bbb_queue = [row for row in escalation["queue"] if row["ticker"] == "BBB"]

    assert aaa_queue[0]["valuation_integrity_class"] == "INTEGRITY_WARNING"
    assert bbb_queue[0]["priority_lane"] == LANE_4_DEPRIORITIZED
    assert "INTEGRITY_SUSPECT_HEADWIND" in bbb_queue[0]["priority_support_codes"]


def test_valuation_integrity_cli_open(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "valuation_integrity_cli"
    aaa = _score_row(
        ticker="AAA",
        intrinsic_payload=_intrinsic_payload(ticker="AAA"),
        confidence_payload=_valuation_confidence_payload(
            ticker="AAA",
            support_types=["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"],
            support_count=2,
            fragility_status="LOW_FRAGILITY",
            confidence_class="MEDIUM_CONFIDENCE",
        ),
        value_type_payload=_value_type_payload(ticker="AAA"),
    )
    bbb = _score_row(
        ticker="BBB",
        intrinsic_payload=_intrinsic_payload(ticker="BBB"),
        confidence_payload=_valuation_confidence_payload(
            ticker="BBB",
            support_types=["OWNER_EARNINGS_VALUE_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"],
            support_count=2,
            fragility_status="LOW_FRAGILITY",
            confidence_class="MEDIUM_CONFIDENCE",
        ),
        value_type_payload=_value_type_payload(ticker="BBB"),
    )
    write_valuation_integrity_for_run(
        run_id=run_id,
        as_of_date="2026-02-14",
        tickers=["AAA", "BBB"],
        output_path=cfg.outputs_dir / "universe" / run_id / "valuation_integrity.json",
        scoreboard_rows=[aaa, bbb],
    )

    result = runner.invoke(app, ["universe-valuation-integrity-open", "--run-id", run_id])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "OK"
    assert payload["exact_uniformity_cluster_count"] >= 1
