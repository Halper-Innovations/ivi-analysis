from __future__ import annotations

from app.db import init_db
from app.universe.promotion import (
    LANE_1_HIGH_PRIORITY,
    LANE_2_RESEARCH_QUEUE,
    LANE_3_MONITOR,
    LANE_4_DEPRIORITIZED,
    _build_risk_flags,
    _build_strength_flags,
    _default_l4_signal_fields,
    _lane_sort_key,
    build_promotion_state,
    classify_priority_lane,
)


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "sample_universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker\nAAA\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _base_row(**overrides):
    row = {
        "ticker": "AAA",
        "thresholds_effective": {
            "promotion_min_appearances_high_priority": 2,
            "promotion_min_implied_return": 0.20,
            "promotion_terminal_blocker_codes": ["PRICE_UNKNOWN"],
            "promotion_lane2_min_score": 60.0,
        },
        "appearances_count": 1,
        "latest_value_gate_status": "WATCH",
        "value_gate_status": "WATCH",
        "latest_primary_blocker": "NONE",
        "primary_blocker": "NONE",
        "implied_return_base": "UNKNOWN",
        "latest_implied_return_base": "UNKNOWN",
        "mos_epv": "UNKNOWN",
        "mos_netnet": "UNKNOWN",
        "mos_to_floor": "UNKNOWN",
        "owner_earnings_yield_ev_3y": "UNKNOWN",
        "composite_score_total": 20.0,
        "valuation_support_count": 0,
        "valuation_convergence_status": "UNKNOWN",
        "valuation_fragility_status": "UNKNOWN",
        "valuation_confidence_class": "CONFIDENCE_UNKNOWN",
        "valuation_integrity_class": "INTEGRITY_OK",
        "investment_readiness_class": "WATCH_ONLY",
        "reinvestment_efficiency_class": "REINVESTMENT_EFFICIENCY_UNKNOWN",
        "returns_persistence_class": "RETURNS_PERSISTENCE_UNKNOWN",
        "revenue_dependence_risk_class": "REVENUE_DEPENDENCE_UNKNOWN",
        "maintenance_capex_credibility_class": "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN",
        "asset_intensity_class": "ASSET_INTENSITY_UNKNOWN",
        "accounting_quality_class": "ACCOUNTING_QUALITY_UNKNOWN",
        "balance_sheet_stress_class": "BALANCE_SHEET_STRESS_UNKNOWN",
        "refinancing_risk_class": "REFINANCING_RISK_UNKNOWN",
        "value_type_primary": "UNKNOWN_VALUE_TYPE",
        "oe_quality_total": "UNKNOWN",
        "intangible_economics_total": "UNKNOWN",
        "owner_value_capture_score": "UNKNOWN",
        "memory_priority_total": 0,
        "facts_blocker_class": "FACTS_OK",
        "facts_blocker_retryable": False,
        "facts_blocker_terminal": False,
        "facts_blocker_partial_usable": False,
        "fail_due_to_economic_weakness": False,
        "mos_assessment_status": "UNKNOWN",
        "evidence_sufficiency_class": "UNKNOWN",
        "downside_support_type": "UNKNOWN",
        "mos_classification": "MOS_UNKNOWN",
        "normalization_credibility_class": "",
        "primary_normalization_caution": "",
        "capital_allocation_discipline_class": "",
        "impairment_class_primary": "",
        **_default_l4_signal_fields(),
    }
    row.update(overrides)
    row["strength_flags"] = _build_strength_flags(row)
    row["risk_flags"] = _build_risk_flags(row)
    row["priority_lane"] = classify_priority_lane(row)
    return row


def test_lane_boost_with_high_perception():
    row = _base_row(
        variant_perception_count=1,
        variant_perception_max_confidence="HIGH",
        variant_perception_direction="UNDERVALUED",
        variant_signal_source_count=3,
        _variant_supporting_sources=["VALUATION", "PATTERN", "FILING_DIFF"],
    )
    assert row["priority_lane"] == LANE_2_RESEARCH_QUEUE
    assert "PERCEPTION_BOOST_HIGH" in row["strength_flags"]


def test_medium_perception_does_not_change_lane():
    row = _base_row(
        variant_perception_count=1,
        variant_perception_max_confidence="MEDIUM",
        variant_perception_direction="UNDERVALUED",
        variant_signal_source_count=2,
        _variant_supporting_sources=["VALUATION", "PATTERN"],
    )
    assert row["priority_lane"] == LANE_3_MONITOR
    assert "PERCEPTION_SUPPORT_MEDIUM" in row["strength_flags"]


def test_high_overvalued_blocks_promotion():
    row = _base_row(
        appearances_count=2,
        latest_value_gate_status="PASS",
        value_gate_status="PASS",
        latest_implied_return_base=0.35,
        implied_return_base=0.35,
        latest_primary_blocker="NONE",
        primary_blocker="NONE",
        variant_perception_count=1,
        variant_perception_max_confidence="HIGH",
        variant_perception_direction="OVERVALUED",
        variant_signal_source_count=3,
        _variant_supporting_sources=["VALUATION", "PATTERN", "FILING_DIFF"],
    )
    assert row["priority_lane"] == LANE_4_DEPRIORITIZED
    assert "OVERVALUATION_BLOCK_HIGH" in row["risk_flags"]


def test_medium_overvalued_adds_risk_flag_only():
    row = _base_row(
        appearances_count=2,
        latest_value_gate_status="PASS",
        value_gate_status="PASS",
        latest_implied_return_base=0.35,
        implied_return_base=0.35,
        latest_primary_blocker="NONE",
        primary_blocker="NONE",
        variant_perception_count=1,
        variant_perception_max_confidence="MEDIUM",
        variant_perception_direction="OVERVALUED",
        variant_signal_source_count=2,
        _variant_supporting_sources=["VALUATION", "PATTERN"],
    )
    assert row["priority_lane"] == LANE_1_HIGH_PRIORITY
    assert "OVERVALUATION_RISK" in row["risk_flags"]


def test_convergent_signal_flag():
    row = _base_row(
        variant_perception_count=1,
        variant_perception_max_confidence="HIGH",
        variant_perception_direction="UNDERVALUED",
        variant_signal_source_count=3,
        pattern_confirmed_count=2,
        filing_diff_high_materiality_count=1,
        _variant_supporting_sources=["PATTERN", "FILING_DIFF", "VALUATION"],
    )
    assert "CONVERGENT_SIGNAL" in row["strength_flags"]


def test_tech_adjusted_value_flag():
    row = _base_row(
        tech_category="ENTERPRISE_SOFTWARE",
        tech_valuation_divergence=45.0,
    )
    assert "TECH_ADJUSTED_VALUE" in row["strength_flags"]


def test_graceful_degradation_without_l4_data():
    row = _base_row()
    assert row["priority_lane"] == LANE_3_MONITOR
    assert all(not flag.startswith("PERCEPTION_") for flag in row["strength_flags"])
    assert "CONVERGENT_SIGNAL" not in row["strength_flags"]
    assert "TECH_ADJUSTED_VALUE" not in row["strength_flags"]
    assert "OVERVALUATION_BLOCK_HIGH" not in row["risk_flags"]
    assert "OVERVALUATION_RISK" not in row["risk_flags"]


def test_sort_key_orders_by_l4_confidence_within_lane():
    high = _base_row(
        ticker="AAA",
        variant_perception_count=1,
        variant_perception_max_confidence="HIGH",
        variant_perception_direction="MIXED",
    )
    medium = _base_row(
        ticker="BBB",
        variant_perception_count=1,
        variant_perception_max_confidence="MEDIUM",
        variant_perception_direction="MIXED",
    )
    none = _base_row(ticker="CCC")

    ordered = [row["ticker"] for row in sorted([none, medium, high], key=_lane_sort_key)]
    assert ordered == ["AAA", "BBB", "CCC"]


def test_build_promotion_state_surfaces_l4_fields(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.universe.promotion._load_memory_lookup", lambda: {})
    monkeypatch.setattr(
        "app.universe.promotion._load_l4_signals_for_ticker",
        lambda ticker, run_id, campaign_run_id, **kwargs: {
            **_default_l4_signal_fields(),
            "variant_perception_count": 1,
            "variant_perception_max_confidence": "HIGH",
            "variant_perception_direction": "UNDERVALUED",
            "variant_signal_source_count": 3,
            "tech_category": "ENTERPRISE_SOFTWARE",
            "tech_valuation_divergence": 42.0,
            "filing_diff_high_materiality_count": 1,
            "pattern_hit_count": 2,
            "pattern_confirmed_count": 2,
            "_variant_supporting_sources": ["VALUATION", "PATTERN", "FILING_DIFF"],
        },
    )

    payload = build_promotion_state(
        "campaign_l4",
        {"tickers": {"AAA": {"appearances_count": 1, "latest_value_gate_status": "WATCH", "latest_primary_blocker": "NONE", "history": []}}},
        {
            "rows": [
                {
                    "ticker": "AAA",
                    "source_runs": [{"universe_run_id": "campaign_l4__software"}],
                    "value_gate_status": "WATCH",
                    "latest_value_gate_status": "WATCH",
                    "primary_blocker": "NONE",
                    "latest_primary_blocker": "NONE",
                    "implied_return_base": "UNKNOWN",
                    "composite_score_total": 20.0,
                }
            ]
        },
    )

    row = payload["rows"][0]
    assert row["variant_perception_max_confidence"] == "HIGH"
    assert row["tech_category"] == "ENTERPRISE_SOFTWARE"
    assert row["pattern_confirmed_count"] == 2
    assert row["filing_diff_high_materiality_count"] == 1
    assert "PERCEPTION_BOOST_HIGH" in row["strength_flags"]
