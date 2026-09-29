from __future__ import annotations

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.escalation import build_escalation_plan
from app.universe.memo_pack import _memo_markdown, build_investment_memo
from app.universe.promotion import LANE_2_RESEARCH_QUEUE, LANE_4_DEPRIORITIZED, build_promotion_state
from app.universe.ranking import ranking_sort_key
from app.valuation.returns_persistence import HIGH_RETURNS_PERSISTENCE
from app.valuation.value_type import compute_value_type, write_value_type_for_run


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
    downside_support_type: str,
    mos_classification: str = "ADEQUATE_MARGIN_OF_SAFETY",
    normalized_status: str = "OK",
    normalized_reason_codes: list[str] | None = None,
) -> dict:
    normalized_reason_codes = list(normalized_reason_codes or ["FCF_SELECTED"])
    return {
        "normalized_earnings_power_value": 100.0,
        "normalized_earnings_power_method_used": "FCF_SELECTED",
        "normalized_earnings_power_status": normalized_status,
        "normalized_earnings_power_reason_codes": normalized_reason_codes,
        "intrinsic_floor": 70.0,
        "intrinsic_base": 85.0,
        "intrinsic_ceiling": 95.0,
        "mos_to_floor": 0.25,
        "mos_to_base": 0.40,
        "mos_classification": mos_classification,
        "downside_support_type": downside_support_type,
        "derived_from": ["intrinsic.payload"],
        "claims": {
            "normalized_earnings_power_value": _claim(100.0, "intrinsic.normalized"),
            "intrinsic_floor": _claim(70.0, "intrinsic.floor"),
            "intrinsic_base": _claim(85.0, "intrinsic.base"),
            "intrinsic_ceiling": _claim(95.0, "intrinsic.ceiling"),
        },
    }


def _valuation_confidence_payload(
    *,
    support_types: list[str],
    support_count: int,
    fragility_status: str,
    confidence_class: str,
    convergence_status: str = "STRONG_CONVERGENCE",
    fragility_reason_codes: list[str] | None = None,
) -> dict:
    fragility_reason_codes = list(fragility_reason_codes or ([] if fragility_status != "HIGH_FRAGILITY" else ["SINGLE_SUPPORT_ONLY"]))
    return {
        "valuation_support_count": support_count,
        "valuation_support_types_present": support_types,
        "valuation_support_count_reason_codes": support_types or ["NO_VALUATION_SUPPORTS_PRESENT"],
        "valuation_convergence_status": convergence_status,
        "valuation_convergence_band_pct": 0.18,
        "valuation_convergence_reason_codes": [] if convergence_status != "WEAK_CONVERGENCE" else ["SUPPORTS_CONFLICT"],
        "valuation_fragility_status": fragility_status,
        "valuation_fragility_reason_codes": fragility_reason_codes,
        "valuation_confidence_class": confidence_class,
        "valuation_confidence_reason_codes": ["MULTI_SUPPORT_VALUE_CASE"] if support_count >= 2 else ["SINGLE_SUPPORT_FRAGILE"],
        "derived_from": ["confidence.payload"],
        "claims": {
            "valuation_support_count": _claim(support_count, "confidence.supports"),
            "valuation_confidence_class": _claim(confidence_class, "confidence.class"),
        },
    }


def _owner_quality_payload(total: float | str) -> dict:
    return {
        "oe_quality_total": total,
        "owner_earnings_stability_score": 4.0 if total != "UNKNOWN" else "UNKNOWN",
        "capital_allocation_score": 3.0 if total != "UNKNOWN" else "UNKNOWN",
        "cash_conversion_score": 3.0 if total != "UNKNOWN" else "UNKNOWN",
        "derived_from": ["oe_quality.payload"],
    }


def _intangible_payload(
    *,
    total: float | str,
    cycle_resilience_score: float | str = 2.0,
    owner_value_capture_score: float | str = 2.0,
) -> dict:
    return {
        "intangible_economics_total": total,
        "cycle_resilience_score": cycle_resilience_score,
        "owner_value_capture_score": owner_value_capture_score,
        "derived_from": ["intangible.payload"],
    }


def test_asset_backed_classification_from_netnet_and_downside_support():
    payload = compute_value_type(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(downside_support_type="ASSET_SUPPORT"),
        valuation_confidence_payload=_valuation_confidence_payload(
            support_types=["NETNET_ASSET_SUPPORT", "EPV_SUPPORT"],
            support_count=2,
            fragility_status="LOW_FRAGILITY",
            confidence_class="MEDIUM_CONFIDENCE",
        ),
        owner_quality_payload=_owner_quality_payload(5.0),
        intangible_payload=_intangible_payload(total=4.0),
    )

    assert payload["value_type_primary"] == "ASSET_BACKED_VALUE"
    assert "NETNET_DRIVEN" in payload["value_type_reason_codes"]
    assert "BALANCE_SHEET_SUPPORT_PRESENT" in payload["value_type_reason_codes"]


def test_earnings_power_classification_from_normalized_earnings_and_epv():
    payload = compute_value_type(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(downside_support_type="EARNINGS_POWER_SUPPORT"),
        valuation_confidence_payload=_valuation_confidence_payload(
            support_types=["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"],
            support_count=2,
            fragility_status="LOW_FRAGILITY",
            confidence_class="MEDIUM_CONFIDENCE",
        ),
        owner_quality_payload=_owner_quality_payload(6.0),
        intangible_payload=_intangible_payload(total=5.0),
    )

    assert payload["value_type_primary"] == "EARNINGS_POWER_VALUE"
    assert "EPV_DRIVEN" in payload["value_type_reason_codes"]
    assert "NORMALIZED_EARNINGS_DRIVEN" in payload["value_type_reason_codes"]


def test_quality_value_requires_real_support_and_quality_overlay():
    payload = compute_value_type(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(downside_support_type="EARNINGS_POWER_SUPPORT"),
        valuation_confidence_payload=_valuation_confidence_payload(
            support_types=["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT", "OWNER_EARNINGS_VALUE_SUPPORT"],
            support_count=3,
            fragility_status="LOW_FRAGILITY",
            confidence_class="HIGH_CONFIDENCE",
        ),
        owner_quality_payload=_owner_quality_payload(10.0),
        intangible_payload=_intangible_payload(total=9.0, owner_value_capture_score=4.0),
        returns_persistence_payload={
            "returns_persistence_class": HIGH_RETURNS_PERSISTENCE,
            "returns_persistence_reason_codes": ["HIGH_RETURNS_PERSISTENCE_SUPPORT"],
            "returns_support_signals": ["HIGH_RETURN_ON_CAPITAL_PRESENT", "RETURNS_STABILITY_PRESENT"],
            "returns_headwind_signals": [],
        },
    )

    assert payload["value_type_primary"] == "QUALITY_VALUE"
    assert payload["value_type_secondary"] == "EARNINGS_POWER_VALUE"
    assert "QUALITY_OVERLAY_SUPPORTED" in payload["value_type_reason_codes"]


def test_cyclical_value_classification_with_resilience_and_normalization_caution():
    payload = compute_value_type(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(
            downside_support_type="EARNINGS_POWER_SUPPORT",
            normalized_status="LOW_CONFIDENCE",
            normalized_reason_codes=["CYCLICAL_NORMALIZATION_LOW_CONFIDENCE"],
        ),
        valuation_confidence_payload=_valuation_confidence_payload(
            support_types=["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"],
            support_count=2,
            fragility_status="MODERATE_FRAGILITY",
            confidence_class="MEDIUM_CONFIDENCE",
            convergence_status="MODERATE_CONVERGENCE",
        ),
        owner_quality_payload=_owner_quality_payload(7.0),
        intangible_payload=_intangible_payload(total=6.0, cycle_resilience_score=4.0, owner_value_capture_score=3.0),
    )

    assert payload["value_type_primary"] == "CYCLICAL_VALUE"
    assert payload["value_type_secondary"] == "EARNINGS_POWER_VALUE"
    assert "CYCLICAL_NORMALIZATION_CASE" in payload["value_type_reason_codes"]


def test_fragile_value_classification_for_thin_high_fragility_support():
    payload = compute_value_type(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(downside_support_type="EARNINGS_POWER_SUPPORT"),
        valuation_confidence_payload=_valuation_confidence_payload(
            support_types=["EPV_SUPPORT"],
            support_count=1,
            fragility_status="HIGH_FRAGILITY",
            confidence_class="LOW_CONFIDENCE",
            fragility_reason_codes=["SINGLE_SUPPORT_ONLY"],
        ),
        owner_quality_payload=_owner_quality_payload(4.0),
        intangible_payload=_intangible_payload(total=3.0),
    )

    assert payload["value_type_primary"] == "FRAGILE_VALUE"
    assert "HIGH_FRAGILITY_CASE" in payload["value_type_reason_codes"]
    assert "SINGLE_SUPPORT_CASE" in payload["value_type_reason_codes"]


def test_unknown_value_type_when_support_is_insufficient():
    payload = compute_value_type(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(
            downside_support_type="UNKNOWN_SUPPORT",
            mos_classification="MOS_UNKNOWN",
            normalized_status="UNKNOWN",
            normalized_reason_codes=["INSUFFICIENT_NORMALIZED_INPUTS"],
        ),
        valuation_confidence_payload=_valuation_confidence_payload(
            support_types=[],
            support_count=0,
            fragility_status="FRAGILITY_UNKNOWN",
            confidence_class="CONFIDENCE_UNKNOWN",
            fragility_reason_codes=[],
        ),
        owner_quality_payload=_owner_quality_payload("UNKNOWN"),
        intangible_payload=_intangible_payload(total="UNKNOWN", cycle_resilience_score="UNKNOWN", owner_value_capture_score="UNKNOWN"),
        fail_due_to_missing_evidence=True,
        primary_fail_domain="EVIDENCE",
    )

    assert payload["value_type_primary"] == "UNKNOWN_VALUE_TYPE"
    assert "INSUFFICIENT_VALUE_TYPE_EVIDENCE" in payload["value_type_reason_codes"]
    assert "EVIDENCE_DEGRADED" in payload["value_type_reason_codes"]


def test_memo_pack_includes_value_type_section():
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
        "valuation_support_types_present": ["EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT", "OWNER_EARNINGS_VALUE_SUPPORT"],
        "valuation_support_count_reason_codes": ["EPV_SUPPORT"],
        "valuation_convergence_status": "STRONG_CONVERGENCE",
        "valuation_convergence_band_pct": 0.14,
        "valuation_convergence_reason_codes": [],
        "valuation_fragility_status": "LOW_FRAGILITY",
        "valuation_fragility_reason_codes": ["FRAGILITY_REDUCED_BY_MULTI_SUPPORT"],
        "valuation_confidence_class": "HIGH_CONFIDENCE",
        "valuation_confidence_reason_codes": ["MULTI_SUPPORT_VALUE_CASE", "LOW_FRAGILITY_UNDERWRITING"],
        "value_type_primary": "QUALITY_VALUE",
        "value_type_secondary": "EARNINGS_POWER_VALUE",
        "value_type_reason_codes": ["QUALITY_OVERLAY_SUPPORTED", "EARNINGS_POWER_SUPPORT_PRESENT"],
        "value_type_support_summary": "Anchored to valuation support with strong owner-economics overlays.",
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
                "epv_per_share": 80.0,
                "netnet_per_share": 70.0,
                "owner_earnings_yield_ev_3y": 0.06,
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

    memo = build_investment_memo(
        "AAA",
        universe_run_id="universe_run",
        batch_run_id="batch_run",
        sources=sources,
    )
    markdown = _memo_markdown(memo)

    assert "## Value Type Classification" in markdown
    assert "value_type_primary: `QUALITY_VALUE`" in markdown
    assert memo["value_type_classification"]["value_type_primary"] == "QUALITY_VALUE"


def test_value_first_typed_ranking_is_deterministic():
    rows = [
        {
            "ticker": "AAA",
            "scout_status": "PASS",
            "mos_to_floor": 0.20,
            "implied_return_base": 0.25,
            "valuation_confidence_class": "HIGH_CONFIDENCE",
            "value_type_primary": "FRAGILE_VALUE",
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
            "value_type_primary": "ASSET_BACKED_VALUE",
            "owner_earnings_yield_ev_3y": 0.05,
            "oe_quality_total": 8.0,
            "intangible_economics_total": 7.0,
            "memory_priority_total": 1,
        },
    ]

    ordered = sorted(rows, key=lambda row: ranking_sort_key(row, policy="value_first_typed"))
    assert [row["ticker"] for row in ordered] == ["BBB", "AAA"]


def test_promotion_and_escalation_visibility_fields_appear_without_overriding_fail():
    master_shortlist = {
        "campaign_run_id": "campaign_value_type",
        "rows": [
            {
                "ticker": "AAA",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "campaign_value_type__core", "batch_run_id": "campaign_value_type__core_depth_batch"}],
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
                "downside_support_type": "ASSET_SUPPORT",
                "valuation_support_count": 3,
                "valuation_support_types_present": ["NETNET_ASSET_SUPPORT", "EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"],
                "valuation_support_count_reason_codes": ["NETNET_ASSET_SUPPORT"],
                "valuation_convergence_status": "STRONG_CONVERGENCE",
                "valuation_convergence_band_pct": 0.18,
                "valuation_convergence_reason_codes": [],
                "valuation_fragility_status": "LOW_FRAGILITY",
                "valuation_fragility_reason_codes": ["FRAGILITY_REDUCED_BY_MULTI_SUPPORT"],
                "valuation_confidence_class": "HIGH_CONFIDENCE",
                "valuation_confidence_reason_codes": ["MULTI_SUPPORT_VALUE_CASE", "LOW_FRAGILITY_UNDERWRITING"],
                "value_type_primary": "ASSET_BACKED_VALUE",
                "value_type_reason_codes": ["NETNET_DRIVEN", "BALANCE_SHEET_SUPPORT_PRESENT"],
                "value_type_support_summary": "Anchored to asset or balance-sheet support.",
                "derived_from": ["shortlist.AAA"],
            },
            {
                "ticker": "BBB",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "campaign_value_type__core", "batch_run_id": "campaign_value_type__core_depth_batch"}],
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
                "downside_support_type": "EARNINGS_POWER_SUPPORT",
                "valuation_support_count": 1,
                "valuation_support_types_present": ["EPV_SUPPORT"],
                "valuation_support_count_reason_codes": ["EPV_SUPPORT"],
                "valuation_convergence_status": "CONVERGENCE_UNKNOWN",
                "valuation_convergence_band_pct": "UNKNOWN",
                "valuation_convergence_reason_codes": ["SINGLE_SUPPORT_ONLY"],
                "valuation_fragility_status": "HIGH_FRAGILITY",
                "valuation_fragility_reason_codes": ["SINGLE_SUPPORT_ONLY"],
                "valuation_confidence_class": "LOW_CONFIDENCE",
                "valuation_confidence_reason_codes": ["SINGLE_SUPPORT_FRAGILE"],
                "value_type_primary": "FRAGILE_VALUE",
                "value_type_reason_codes": ["HIGH_FRAGILITY_CASE", "SINGLE_SUPPORT_CASE"],
                "value_type_support_summary": "Appears cheap, but support is thin, conflicted, or fragile.",
                "derived_from": ["shortlist.BBB"],
            },
        ],
    }
    master_watchlist = {
        "campaign_run_id": "campaign_value_type",
        "tickers": {
            "AAA": {
                "latest_value_gate_status": "WATCH",
                "latest_implied_return_base": 0.20,
                "latest_primary_blocker": "MISSING_EV",
                "appearances_count": 2,
                "history": [
                    {"campaign_item": "core", "universe_run_id": "campaign_value_type__core", "value_gate_status": "WATCH", "implied_return_base": 0.20, "primary_blocker": "MISSING_EV", "last_rank": 1},
                ],
            },
            "BBB": {
                "latest_value_gate_status": "FAIL",
                "latest_implied_return_base": 0.25,
                "latest_primary_blocker": "PRICE_UNKNOWN",
                "appearances_count": 2,
                "history": [
                    {"campaign_item": "core", "universe_run_id": "campaign_value_type__core", "value_gate_status": "FAIL", "implied_return_base": 0.25, "primary_blocker": "PRICE_UNKNOWN", "last_rank": 3},
                ],
            },
        },
    }

    promotion = build_promotion_state("campaign_value_type", master_watchlist, master_shortlist)
    rows = {row["ticker"]: row for row in promotion["rows"]}

    assert rows["AAA"]["priority_lane"] == LANE_2_RESEARCH_QUEUE
    assert rows["AAA"]["value_type_primary"] == "ASSET_BACKED_VALUE"
    assert "ASSET_SUPPORT_CASE" in rows["AAA"]["promotion_reason_codes"]
    assert rows["BBB"]["priority_lane"] == LANE_4_DEPRIORITIZED
    assert "FRAGILE_VALUE_HEADWIND" in rows["BBB"]["promotion_reason_codes"]

    escalation = build_escalation_plan(
        "campaign_value_type",
        promotion,
        {
            "lane_1_high_priority": [],
            "lane_2_research_queue": [rows["AAA"]],
            "lane_3_monitor": [],
            "lane_4_deprioritized": [rows["BBB"]],
        },
        config={"as_of_date": "2026-02-14", "source_campaign_file": "data/universe/sample_campaign.json", "top_n": 10, "policy": "value_first_typed"},
    )
    aaa_queue = [row for row in escalation["queue"] if row["ticker"] == "AAA"]
    bbb_queue = [row for row in escalation["queue"] if row["ticker"] == "BBB"]

    assert aaa_queue[0]["value_type_primary"] == "ASSET_BACKED_VALUE"
    assert "ASSET_SUPPORT_CASE" in aaa_queue[0]["priority_support_codes"]
    assert bbb_queue[0]["priority_lane"] == LANE_4_DEPRIORITIZED
    assert "FRAGILE_VALUE_HEADWIND" in bbb_queue[0]["priority_support_codes"]


def test_value_type_cli_open(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "value_type_cli"
    detail = compute_value_type(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(downside_support_type="ASSET_SUPPORT"),
        valuation_confidence_payload=_valuation_confidence_payload(
            support_types=["NETNET_ASSET_SUPPORT", "EPV_SUPPORT"],
            support_count=2,
            fragility_status="LOW_FRAGILITY",
            confidence_class="MEDIUM_CONFIDENCE",
        ),
        owner_quality_payload=_owner_quality_payload(5.0),
        intangible_payload=_intangible_payload(total=4.0),
    )
    write_value_type_for_run(
        run_id=run_id,
        as_of_date="2026-02-14",
        tickers=["AAA"],
        output_path=cfg.outputs_dir / "universe" / run_id / "value_type.json",
        scoreboard_rows=[{"ticker": "AAA", "value_type_detail": detail}],
    )

    result = runner.invoke(app, ["universe-value-type-open", "--run-id", run_id])
    assert result.exit_code == 0, result.stdout
    assert '"status": "OK"' in result.stdout
    assert '"ASSET_BACKED_VALUE"' in result.stdout
