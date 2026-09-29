from __future__ import annotations

import json

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.escalation import build_escalation_plan
from app.universe.memo_pack import _memo_markdown, build_investment_memo
from app.universe.promotion import LANE_2_RESEARCH_QUEUE, LANE_4_DEPRIORITIZED, build_promotion_state
from app.universe.ranking import ranking_sort_key
from app.valuation.valuation_confidence import (
    compute_valuation_confidence,
    write_valuation_confidence_for_run,
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
    normalized: float | str,
    floor: float | str,
    base: float | str,
    ceiling: float | str,
    method: str = "FCF_SELECTED",
    status: str = "OK",
) -> dict:
    return {
        "normalized_earnings_power_value": normalized,
        "normalized_earnings_power_method_used": method,
        "normalized_earnings_power_status": status,
        "normalized_earnings_power_reason_codes": [method] if normalized != "UNKNOWN" else ["INSUFFICIENT_NORMALIZED_INPUTS"],
        "intrinsic_floor": floor,
        "intrinsic_base": base,
        "intrinsic_ceiling": ceiling,
        "derived_from": ["intrinsic.payload"],
        "claims": {
            "normalized_earnings_power_value": _claim(normalized, "intrinsic.normalized"),
            "intrinsic_floor": _claim(floor, "intrinsic.floor"),
            "intrinsic_base": _claim(base, "intrinsic.base"),
            "intrinsic_ceiling": _claim(ceiling, "intrinsic.ceiling"),
        },
    }


def _score_row_with_confidence(ticker: str, confidence_payload: dict) -> dict:
    return {
        "ticker": ticker,
        "valuation_confidence_detail": confidence_payload,
        "intrinsic_discipline_detail": confidence_payload.get("intrinsic_payload") or {},
        "derived_from": [f"score.{ticker}"],
        "metric_values": {
            "epv_per_share": 80.0,
            "netnet_per_share": 70.0,
            "intrinsic_per_share_base": 82.0,
            "valuation_confidence_class": confidence_payload.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN"),
        },
    }


def test_support_count_is_deterministic():
    payload = compute_valuation_confidence(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(normalized=100.0, floor=70.0, base=82.0, ceiling=95.0),
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
        epv_per_share=80.0,
        epv_refs=["gd.epv"],
        netnet_per_share=70.0,
        netnet_refs=["gd.netnet"],
    )

    assert payload["valuation_support_count"] == 3
    assert payload["valuation_support_types_present"] == [
        "NETNET_ASSET_SUPPORT",
        "EPV_SUPPORT",
        "NORMALIZED_EARNINGS_POWER_SUPPORT",
    ]


def test_convergence_classification_for_aligned_vs_conflicting_values():
    aligned = compute_valuation_confidence(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(normalized=95.0, floor=70.0, base=82.0, ceiling=90.0),
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
        epv_per_share=80.0,
        epv_refs=["gd.epv"],
        netnet_per_share=70.0,
        netnet_refs=["gd.netnet"],
    )
    conflicting = compute_valuation_confidence(
        "BBB",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(normalized=200.0, floor=40.0, base=180.0, ceiling=220.0),
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
        epv_per_share=120.0,
        epv_refs=["gd.epv"],
        netnet_per_share=40.0,
        netnet_refs=["gd.netnet"],
    )

    assert aligned["valuation_convergence_status"] == "STRONG_CONVERGENCE"
    assert conflicting["valuation_convergence_status"] == "WEAK_CONVERGENCE"
    assert "SUPPORTS_CONFLICT" in conflicting["valuation_convergence_reason_codes"]


def test_fragility_classification_responds_to_missing_inputs_and_single_support():
    payload = compute_valuation_confidence(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(normalized="UNKNOWN", floor="UNKNOWN", base="UNKNOWN", ceiling="UNKNOWN", status="UNKNOWN"),
        price_status="UNKNOWN",
        shares_status="UNKNOWN",
        fcf_status="UNKNOWN",
        facts_status="UNKNOWN",
        valuation_status="UNKNOWN",
        epv_per_share=85.0,
        epv_refs=["gd.epv"],
        netnet_per_share="UNKNOWN",
        netnet_refs=[],
    )

    assert payload["valuation_support_count"] == 1
    assert payload["valuation_fragility_status"] == "HIGH_FRAGILITY"
    assert "SINGLE_SUPPORT_ONLY" in payload["valuation_fragility_reason_codes"]
    assert "MISSING_PRICE" in payload["valuation_fragility_reason_codes"]
    assert "MISSING_SHARES" in payload["valuation_fragility_reason_codes"]
    assert "MISSING_FACTS" in payload["valuation_fragility_reason_codes"]


def test_confidence_class_combines_conservatively():
    high = compute_valuation_confidence(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(normalized=96.0, floor=72.0, base=84.0, ceiling=92.0),
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
    low = compute_valuation_confidence(
        "BBB",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(normalized="UNKNOWN", floor="UNKNOWN", base="UNKNOWN", ceiling="UNKNOWN", status="LOW_CONFIDENCE"),
        price_status="UNKNOWN",
        shares_status="UNKNOWN",
        fcf_status="UNKNOWN",
        facts_status="UNKNOWN",
        valuation_status="UNKNOWN",
        epv_per_share=85.0,
        epv_refs=["gd.epv"],
    )

    assert high["valuation_confidence_class"] == "HIGH_CONFIDENCE"
    assert low["valuation_confidence_class"] == "LOW_CONFIDENCE"


def test_memo_pack_includes_valuation_confidence_section():
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
        "valuation_support_count_reason_codes": ["NETNET_ASSET_SUPPORT", "EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"],
        "valuation_convergence_status": "STRONG_CONVERGENCE",
        "valuation_convergence_band_pct": 0.14,
        "valuation_convergence_reason_codes": [],
        "valuation_fragility_status": "LOW_FRAGILITY",
        "valuation_fragility_reason_codes": ["FRAGILITY_REDUCED_BY_MULTI_SUPPORT"],
        "valuation_confidence_class": "HIGH_CONFIDENCE",
        "valuation_confidence_reason_codes": ["MULTI_SUPPORT_VALUE_CASE", "LOW_FRAGILITY_UNDERWRITING"],
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
            "metric_traces": {
                "implied_return_base": {"derived_from": ["score.implied_return"]},
                "intrinsic_per_share_base": {"derived_from": ["score.intrinsic_base"]},
                "current_price": {"derived_from": ["score.price"]},
                "mos_epv": {"derived_from": ["score.mos_epv"]},
                "mos_netnet": {"derived_from": ["score.mos_netnet"]},
                "owner_earnings_yield_ev_3y": {"derived_from": ["score.owner_yield"]},
                "fcf_yield_ev_3y": {"derived_from": ["score.fcf_yield"]},
                "revenue_cagr_5y": {"derived_from": ["score.revenue5y"]},
                "revenue_cagr_10y": {"derived_from": ["score.revenue10y"]},
                "operating_margin_trend_slope": {"derived_from": ["score.op_margin"]},
                "gross_margin_trend_slope": {"derived_from": ["score.gross_margin"]},
                "fcf_margin_trend_slope": {"derived_from": ["score.fcf_margin"]},
                "roic_proxy": {"derived_from": ["score.roic"]},
                "dilution_rate_shares_cagr": {"derived_from": ["score.dilution"]},
                "net_debt_proxy": {"derived_from": ["score.net_debt"]},
                "risk_factor_keyword_delta": {"derived_from": ["score.risk_delta"]},
                "quality_score": {"derived_from": ["score.quality_score"]},
                "risk_penalty": {"derived_from": ["score.risk_penalty"]},
            },
            "derived_from": ["score.AAA"],
        },
        "gate_row": {
            "ticker": "AAA",
            "gate_status": "WATCH",
            "gate_reasons": ["MISSING_EV"],
            "primary_blocker": "MISSING_EV",
            "inputs_used": {
                "current_price": {"value": 63.0, "derived_from": ["price.AAA"]},
                "net_debt_proxy": {"value": 120.0, "derived_from": ["net_debt.AAA"]},
                "dilution_rate_shares_cagr": {"value": 0.01, "derived_from": ["dilution.AAA"]},
            },
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

    assert "## Valuation Confidence / Fragility" in markdown
    assert "valuation_confidence_class: `HIGH_CONFIDENCE`" in markdown
    assert memo["valuation_confidence_fragility"]["valuation_support_count"] == 3


def test_value_first_confidence_ranking_is_deterministic():
    rows = [
        {
            "ticker": "AAA",
            "scout_status": "PASS",
            "mos_to_floor": 0.30,
            "implied_return_base": 0.20,
            "valuation_confidence_class": "LOW_CONFIDENCE",
            "mos_epv": 0.10,
            "owner_earnings_yield_ev_3y": 0.04,
            "oe_quality_total": 5.0,
            "intangible_economics_total": 4.0,
            "memory_priority_total": 1,
        },
        {
            "ticker": "BBB",
            "scout_status": "PASS",
            "mos_to_floor": 0.10,
            "implied_return_base": 0.40,
            "valuation_confidence_class": "HIGH_CONFIDENCE",
            "mos_epv": 0.20,
            "owner_earnings_yield_ev_3y": 0.05,
            "oe_quality_total": 8.0,
            "intangible_economics_total": 7.0,
            "memory_priority_total": 3,
        },
    ]

    ordered = sorted(rows, key=lambda row: ranking_sort_key(row, policy="value_first_confidence"))
    assert [row["ticker"] for row in ordered] == ["AAA", "BBB"]


def test_promotion_and_escalation_visibility_fields_appear_without_overriding_fail():
    master_shortlist = {
        "campaign_run_id": "campaign_confidence",
        "rows": [
            {
                "ticker": "AAA",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "campaign_confidence__core", "batch_run_id": "campaign_confidence__core_depth_batch"}],
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
                "valuation_confidence_class": "HIGH_CONFIDENCE",
                "valuation_confidence_reason_codes": ["MULTI_SUPPORT_VALUE_CASE", "LOW_FRAGILITY_UNDERWRITING"],
                "derived_from": ["shortlist.AAA"],
            },
            {
                "ticker": "BBB",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "campaign_confidence__core", "batch_run_id": "campaign_confidence__core_depth_batch"}],
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
                "valuation_support_count": 3,
                "valuation_support_types_present": ["NETNET_ASSET_SUPPORT", "EPV_SUPPORT", "NORMALIZED_EARNINGS_POWER_SUPPORT"],
                "valuation_support_count_reason_codes": ["NETNET_ASSET_SUPPORT"],
                "valuation_convergence_status": "STRONG_CONVERGENCE",
                "valuation_convergence_band_pct": 0.10,
                "valuation_convergence_reason_codes": [],
                "valuation_fragility_status": "LOW_FRAGILITY",
                "valuation_fragility_reason_codes": ["FRAGILITY_REDUCED_BY_MULTI_SUPPORT"],
                "valuation_confidence_class": "HIGH_CONFIDENCE",
                "valuation_confidence_reason_codes": ["MULTI_SUPPORT_VALUE_CASE", "LOW_FRAGILITY_UNDERWRITING"],
                "derived_from": ["shortlist.BBB"],
            },
        ],
    }
    master_watchlist = {
        "campaign_run_id": "campaign_confidence",
        "tickers": {
            "AAA": {
                "latest_value_gate_status": "WATCH",
                "latest_implied_return_base": 0.20,
                "latest_primary_blocker": "MISSING_EV",
                "appearances_count": 2,
                "history": [
                    {"campaign_item": "core", "universe_run_id": "campaign_confidence__core", "value_gate_status": "WATCH", "implied_return_base": 0.20, "primary_blocker": "MISSING_EV", "last_rank": 1},
                ],
            },
            "BBB": {
                "latest_value_gate_status": "FAIL",
                "latest_implied_return_base": 0.25,
                "latest_primary_blocker": "PRICE_UNKNOWN",
                "appearances_count": 2,
                "history": [
                    {"campaign_item": "core", "universe_run_id": "campaign_confidence__core", "value_gate_status": "FAIL", "implied_return_base": 0.25, "primary_blocker": "PRICE_UNKNOWN", "last_rank": 3},
                ],
            },
        },
    }

    promotion = build_promotion_state("campaign_confidence", master_watchlist, master_shortlist)
    rows = {row["ticker"]: row for row in promotion["rows"]}

    assert rows["AAA"]["priority_lane"] == LANE_2_RESEARCH_QUEUE
    assert "MULTI_SUPPORT_VALUE_CASE" in rows["AAA"]["promotion_reason_codes"]
    assert rows["AAA"]["valuation_confidence_class"] == "HIGH_CONFIDENCE"
    assert rows["BBB"]["priority_lane"] == LANE_4_DEPRIORITIZED

    escalation = build_escalation_plan(
        "campaign_confidence",
        promotion,
        {
            "lane_1_high_priority": [],
            "lane_2_research_queue": [rows["AAA"]],
            "lane_3_monitor": [],
            "lane_4_deprioritized": [rows["BBB"]],
        },
        config={"as_of_date": "2026-02-14", "source_campaign_file": "data/universe/sample_campaign.json", "top_n": 10, "policy": "value_first_confidence"},
    )
    aaa_queue = [row for row in escalation["queue"] if row["ticker"] == "AAA"]
    bbb_queue = [row for row in escalation["queue"] if row["ticker"] == "BBB"]

    assert aaa_queue[0]["valuation_confidence_class"] == "HIGH_CONFIDENCE"
    assert "MULTI_SUPPORT_VALUE_CASE" in aaa_queue[0]["priority_support_codes"]
    assert bbb_queue[0]["priority_lane"] == LANE_4_DEPRIORITIZED


def test_valuation_confidence_cli_open(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "valuation_confidence_cli"
    detail = compute_valuation_confidence(
        "AAA",
        "2026-02-14",
        intrinsic_payload=_intrinsic_payload(normalized=96.0, floor=72.0, base=84.0, ceiling=92.0),
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
    write_valuation_confidence_for_run(
        run_id=run_id,
        as_of_date="2026-02-14",
        tickers=["AAA"],
        output_path=cfg.outputs_dir / "universe" / run_id / "valuation_confidence.json",
        scoreboard_rows=[{"ticker": "AAA", "valuation_confidence_detail": detail}],
        coverage_rows_by_ticker={"AAA": {"price_status": "OK", "shares_status": "OK", "fcf_status": "OK", "facts_status": "OK", "valuation_status": "OK"}},
    )

    result = runner.invoke(app, ["universe-valuation-confidence-open", "--run-id", run_id])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "OK"
    assert payload["high_confidence_count"] == 1
    assert payload["single_support_only_count"] == 0


def test_a_corroborating_estimate_inside_the_range_cannot_widen_the_convergence_band():
    """Two supports at 100 and 126 span 26 around a midpoint of
    113: band 26 / 113 = 0.230088, STRONG. A third estimate of 101 lies inside
    that range and corroborates it, yet scaling by the MEDIAN (now 101) widened
    the band to 26 / 101 = 0.257426 and downgraded convergence to MODERATE. The
    band is scaled by the midpoint of the range, which an interior estimate
    cannot move."""

    def _confidence(**extra):
        return compute_valuation_confidence(
            "AAA",
            "2026-02-14",
            intrinsic_payload={},
            price_status="OK",
            shares_status="OK",
            fcf_status="OK",
            facts_status="OK",
            valuation_status="OK",
            netnet_per_share=100.0,
            epv_per_share=126.0,
            **extra,
        )

    two = _confidence()
    three = _confidence(existing_intrinsic_conservative=101.0)

    assert two["valuation_support_count"] == 2
    assert three["valuation_support_count"] == 3
    assert two["valuation_convergence_band_pct"] == 0.230088
    assert three["valuation_convergence_band_pct"] == 0.230088
    assert two["valuation_convergence_status"] == "STRONG_CONVERGENCE"
    assert three["valuation_convergence_status"] == "STRONG_CONVERGENCE"
