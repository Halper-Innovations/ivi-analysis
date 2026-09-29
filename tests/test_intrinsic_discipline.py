from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.escalation import build_escalation_plan
from app.universe.memo_pack import _memo_markdown, build_investment_memo
from app.universe.promotion import (
    LANE_2_RESEARCH_QUEUE,
    LANE_4_DEPRIORITIZED,
    build_promotion_state,
)
from app.universe.ranking import ranking_sort_key
from app.valuation.intrinsic_discipline import compute_intrinsic_discipline


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fundamentals_payload(*, fcf_values: list[float], cfo_values: list[float], shares: float = 10.0, net_debt: float = 0.0) -> dict:
    rows = []
    traces: dict[str, dict] = {}
    for idx, year in enumerate([2021, 2022, 2023], start=0):
        fcf = float(fcf_values[idx])
        cfo = float(cfo_values[idx])
        row = {
            "year": year,
            "cfo": cfo,
            "capex": max(cfo - fcf, 0.0),
            "fcf": fcf,
            "shares_outstanding": shares,
            "net_debt": net_debt,
        }
        rows.append(row)
        traces[str(year)] = {
            "cfo": {"derived_from": [f"fundamentals.{year}.cfo"]},
            "capex": {"derived_from": [f"fundamentals.{year}.capex"]},
            "fcf": {"derived_from": [f"fundamentals.{year}.fcf"]},
            "shares_outstanding": {"derived_from": [f"fundamentals.{year}.shares"]},
            "net_debt": {"derived_from": [f"fundamentals.{year}.net_debt"]},
        }
    return {"rows": rows, "row_traces": traces}


def _owner_payload(*, normalized_3y: float, years: list[int] | None = None) -> dict:
    year_list = years or [2021, 2022, 2023]
    series = [
        {
            "year": year,
            "owner_earnings": normalized_3y,
            "derived_from": [f"owner.{year}"],
        }
        for year in year_list
    ]
    return {
        "summary": {
            "owner_earnings_normalized_3y": float(normalized_3y),
            "owner_earnings_normalized_method": "MEDIAN_POSITIVE_3Y",
            "owner_earnings_points": len(series),
        },
        "series": series,
        "derived_from": ["owner.summary"],
    }


def test_normalized_earnings_power_selection_is_conservative_and_deterministic():
    payload = compute_intrinsic_discipline(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_payload(fcf_values=[90.0, 90.0, 90.0], cfo_values=[120.0, 120.0, 120.0]),
        owner_payload=_owner_payload(normalized_3y=130.0),
        owner_quality_payload={"owner_earnings_stability_score": 4.0, "oe_quality_total": 8.0},
        intangible_payload={"cycle_resilience_score": 3.0},
        price_value=50.0,
        price_refs=["price.AAA"],
    )

    assert payload["normalized_earnings_power_method_used"] == "FCF_SELECTED"
    assert payload["normalized_earnings_power_value"] == 90.0
    assert payload["normalized_earnings_power_status"] == "OK"
    assert payload["intrinsic_base"] == 90.0
    assert payload["intrinsic_ceiling"] == 108.0


def test_valuation_range_and_mos_classification_use_conservative_supports():
    payload = compute_intrinsic_discipline(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_payload(fcf_values=[90.0, 90.0, 90.0], cfo_values=[100.0, 100.0, 100.0]),
        owner_payload=_owner_payload(normalized_3y=90.0),
        owner_quality_payload={"owner_earnings_stability_score": 4.0, "oe_quality_total": 7.0},
        intangible_payload={"cycle_resilience_score": 3.0},
        netnet_per_share=60.0,
        netnet_refs=["gd.AAA.netnet"],
        epv_per_share=80.0,
        epv_refs=["gd.AAA.epv"],
        price_value=40.0,
        price_refs=["price.AAA"],
    )

    assert payload["intrinsic_floor"] == 60.0
    assert payload["intrinsic_base"] == 90.0
    assert payload["intrinsic_ceiling"] == 108.0
    assert payload["mos_to_floor"] == 0.5
    assert payload["mos_classification"] == "DEEP_VALUE_SUPPORT"
    assert "FLOOR_FROM_NETNET" in payload["valuation_range_reason_codes"]
    assert payload["downside_support_type"] == "ASSET_SUPPORT"


def test_unknowns_and_downside_support_classification_remain_explicit():
    unknown_payload = compute_intrinsic_discipline(
        "AAA",
        "2026-02-14",
        fundamentals={"rows": [], "row_traces": {}},
        owner_payload={"summary": {}, "series": [], "derived_from": []},
        price_value="UNKNOWN",
    )
    balance_sheet_payload = compute_intrinsic_discipline(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_payload(fcf_values=[0.0, 0.0, 0.0], cfo_values=[0.0, 0.0, 0.0], net_debt=-15.0),
        owner_payload={"summary": {}, "series": [], "derived_from": []},
        price_value=25.0,
        price_refs=["price.AAA"],
    )

    assert unknown_payload["normalized_earnings_power_value"] == "UNKNOWN"
    assert unknown_payload["valuation_range_status"] == "UNKNOWN"
    assert unknown_payload["mos_classification"] == "MOS_UNKNOWN"
    assert "INSUFFICIENT_NORMALIZED_INPUTS" in unknown_payload["normalized_earnings_power_reason_codes"]
    assert "RANGE_UNKNOWN_MISSING_SHARES" in unknown_payload["valuation_range_reason_codes"]
    assert balance_sheet_payload["downside_support_type"] == "BALANCE_SHEET_SUPPORT"


def test_value_first_discipline_ranking_is_deterministic():
    rows = [
        {
            "ticker": "AAA",
            "scout_status": "PASS",
            "mos_to_floor": 0.30,
            "implied_return_base": 0.20,
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
            "implied_return_base": 0.50,
            "mos_epv": 0.20,
            "owner_earnings_yield_ev_3y": 0.05,
            "oe_quality_total": 9.0,
            "intangible_economics_total": 8.0,
            "memory_priority_total": 5,
        },
    ]

    ordered = sorted(rows, key=lambda row: ranking_sort_key(row, policy="value_first_discipline"))
    assert [row["ticker"] for row in ordered] == ["AAA", "BBB"]


def test_promotion_and_escalation_surface_intrinsic_fields_without_overriding_fail():
    master_shortlist = {
        "campaign_run_id": "campaign_intrinsic",
        "rows": [
            {
                "ticker": "AAA",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "campaign_intrinsic__core", "batch_run_id": "campaign_intrinsic__core_depth_batch"}],
                "best_rank_seen": 1,
                "value_gate_status": "WATCH",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": 0.18,
                "mos_epv": 0.12,
                "mos_netnet": "UNKNOWN",
                "owner_earnings_yield_ev_3y": "UNKNOWN",
                "yield_metric_used": "UNKNOWN",
                "primary_blocker": "MISSING_EV",
                "latest_primary_blocker": "MISSING_EV",
                "memo_path": "memo/AAA.md",
                "composite_score_total": 55.0,
                "mos_to_floor": 0.35,
                "mos_to_base": 0.60,
                "mos_classification": "ADEQUATE_MARGIN_OF_SAFETY",
                "downside_support_type": "EARNINGS_POWER_SUPPORT",
                "intrinsic_floor": 81.0,
                "intrinsic_base": 96.0,
                "intrinsic_ceiling": 110.0,
                "normalized_earnings_power_value": 90.0,
                "normalized_earnings_power_method_used": "FCF_SELECTED",
                "normalized_earnings_power_status": "OK",
                "normalized_earnings_power_reason_codes": ["FCF_SELECTED"],
                "valuation_range_reason_codes": ["BASE_FROM_NORMALIZED_EARNINGS_POWER"],
                "downside_support_reason_codes": ["EARNINGS_POWER_SUPPORT"],
                "derived_from": ["shortlist.AAA"],
            },
            {
                "ticker": "BBB",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "campaign_intrinsic__core", "batch_run_id": "campaign_intrinsic__core_depth_batch"}],
                "best_rank_seen": 2,
                "value_gate_status": "FAIL",
                "latest_value_gate_status": "FAIL",
                "implied_return_base": 0.12,
                "mos_epv": 0.05,
                "mos_netnet": "UNKNOWN",
                "owner_earnings_yield_ev_3y": "UNKNOWN",
                "yield_metric_used": "UNKNOWN",
                "primary_blocker": "MISSING_EV",
                "latest_primary_blocker": "MISSING_EV",
                "memo_path": "memo/BBB.md",
                "composite_score_total": 40.0,
                "mos_to_floor": 0.70,
                "mos_to_base": 1.00,
                "mos_classification": "DEEP_VALUE_SUPPORT",
                "downside_support_type": "ASSET_SUPPORT",
                "intrinsic_floor": 85.0,
                "intrinsic_base": 120.0,
                "intrinsic_ceiling": 150.0,
                "normalized_earnings_power_value": 100.0,
                "normalized_earnings_power_method_used": "FCF_SELECTED",
                "normalized_earnings_power_status": "OK",
                "normalized_earnings_power_reason_codes": ["FCF_SELECTED"],
                "valuation_range_reason_codes": ["BASE_FROM_NORMALIZED_EARNINGS_POWER"],
                "downside_support_reason_codes": ["ASSET_SUPPORT"],
                "derived_from": ["shortlist.BBB"],
            },
        ],
    }
    master_watchlist = {
        "campaign_run_id": "campaign_intrinsic",
        "tickers": {
            "AAA": {
                "latest_value_gate_status": "WATCH",
                "latest_implied_return_base": 0.18,
                "latest_primary_blocker": "MISSING_EV",
                "appearances_count": 2,
                "history": [
                    {"campaign_item": "core", "universe_run_id": "campaign_intrinsic__core", "value_gate_status": "WATCH", "implied_return_base": 0.16, "primary_blocker": "MISSING_EV", "last_rank": 1},
                    {"campaign_item": "core", "universe_run_id": "campaign_intrinsic__core", "value_gate_status": "WATCH", "implied_return_base": 0.18, "primary_blocker": "MISSING_EV", "last_rank": 1},
                ],
            },
            "BBB": {
                "latest_value_gate_status": "FAIL",
                "latest_implied_return_base": 0.12,
                "latest_primary_blocker": "MISSING_EV",
                "appearances_count": 2,
                "history": [
                    {"campaign_item": "core", "universe_run_id": "campaign_intrinsic__core", "value_gate_status": "FAIL", "implied_return_base": 0.14, "primary_blocker": "MISSING_EV", "last_rank": 2},
                    {"campaign_item": "core", "universe_run_id": "campaign_intrinsic__core", "value_gate_status": "FAIL", "implied_return_base": 0.12, "primary_blocker": "MISSING_EV", "last_rank": 2},
                ],
            },
        },
    }

    promotion_state = build_promotion_state("campaign_intrinsic", master_watchlist, master_shortlist)
    rows = {str(row.get("ticker")): row for row in promotion_state["rows"]}
    assert rows["AAA"]["priority_lane"] == LANE_2_RESEARCH_QUEUE
    assert rows["AAA"]["mos_to_floor"] == 0.35
    assert rows["AAA"]["downside_support_type"] == "EARNINGS_POWER_SUPPORT"
    assert "REAL_DOWNSIDE_SUPPORT_PRESENT" in rows["AAA"]["promotion_reason_codes"]
    assert rows["BBB"]["priority_lane"] == LANE_4_DEPRIORITIZED

    priority_lanes = {
        "lane_1_high_priority": [],
        "lane_2_research_queue": [rows["AAA"]],
        "lane_3_monitor": [],
        "lane_4_deprioritized": [rows["BBB"]],
    }
    escalation_plan = build_escalation_plan(
        "campaign_intrinsic",
        promotion_state,
        priority_lanes,
        config={"as_of_date": "2026-02-14", "source_campaign_file": "campaign.json", "policy": "value_first_discipline", "top_n": 10},
    )
    aaa_entries = [row for row in escalation_plan["queue"] if row.get("ticker") == "AAA"]
    assert aaa_entries
    assert aaa_entries[0]["mos_to_floor"] == 0.35
    assert aaa_entries[0]["downside_support_type"] == "EARNINGS_POWER_SUPPORT"
    assert "REAL_DOWNSIDE_SUPPORT_PRESENT" in aaa_entries[0]["priority_support_codes"]
    assert "MOS_TO_FLOOR_ATTRACTIVE" in aaa_entries[0]["priority_support_codes"]


def test_memo_pack_includes_intrinsic_value_discipline_section():
    row = {
        "ticker": "AAA",
        "sector": "Software",
        "as_of_date": "2026-02-14",
        "source_depth_runs": [{"run_id": "depth_run_1", "sector": "Software", "as_of_date": "2026-02-14"}],
        "value_gate_status": "WATCH",
        "value_gate_reasons": ["MISSING_EV"],
        "primary_blocker": "MISSING_EV",
        "implied_return_base": 0.22,
        "implied_return_base_derived_from": ["trace.AAA.implied_return_base"],
        "intrinsic_per_share_base": 95.0,
        "intrinsic_per_share_base_derived_from": ["trace.AAA.intrinsic_per_share_base"],
        "mos_epv": 0.25,
        "mos_epv_derived_from": ["trace.AAA.mos_epv"],
        "mos_netnet": 0.05,
        "mos_netnet_derived_from": ["trace.AAA.mos_netnet"],
        "owner_earnings_yield_ev_3y": 0.06,
        "owner_earnings_yield_ev_3y_derived_from": ["trace.AAA.owner_yield"],
        "fcf_yield_ev_3y": 0.05,
        "normalized_earnings_power_value": 90.0,
        "normalized_earnings_power_value_derived_from": ["trace.AAA.normalized_power"],
        "normalized_earnings_power_method_used": "FCF_SELECTED",
        "normalized_earnings_power_status": "OK",
        "normalized_earnings_power_reason_codes": ["FCF_SELECTED"],
        "intrinsic_floor": 72.0,
        "intrinsic_floor_derived_from": ["trace.AAA.intrinsic_floor"],
        "intrinsic_base": 90.0,
        "intrinsic_base_derived_from": ["trace.AAA.intrinsic_base"],
        "intrinsic_ceiling": 108.0,
        "intrinsic_ceiling_derived_from": ["trace.AAA.intrinsic_ceiling"],
        "mos_to_floor": 0.20,
        "mos_to_floor_derived_from": ["trace.AAA.mos_to_floor"],
        "mos_to_base": 0.50,
        "mos_to_base_derived_from": ["trace.AAA.mos_to_base"],
        "mos_classification": "MODEST_MARGIN_OF_SAFETY",
        "downside_support_type": "EARNINGS_POWER_SUPPORT",
        "downside_support_status": "OK",
        "valuation_range_reason_codes": ["BASE_FROM_NORMALIZED_EARNINGS_POWER"],
        "downside_support_reason_codes": ["EARNINGS_POWER_SUPPORT"],
        "oe_quality_reason_codes": ["SHAREHOLDER_FRIENDLY"],
        "intangible_economics_reason_codes": ["RESILIENT_CYCLICAL_PROFILE"],
        "derived_from": ["shortlist.AAA"],
    }
    sources = {
        "shortlist_row": row,
        "run_id": "depth_run_1",
        "score_row": {
            "metric_values": {
                "implied_return_base": 0.22,
                "intrinsic_per_share_base": 95.0,
                "current_price": 60.0,
            },
            "metric_traces": {
                "implied_return_base": {"derived_from": ["trace.AAA.implied_return_base"]},
                "intrinsic_per_share_base": {"derived_from": ["trace.AAA.intrinsic_per_share_base"]},
            },
        },
        "gate_row": {
            "gate_status": "WATCH",
            "gate_reasons": ["MISSING_EV"],
            "primary_blocker": "MISSING_EV",
            "inputs_used": {
                "current_price": {"value": 60.0, "derived_from": ["price.AAA"]},
                "net_debt_proxy": {"value": 100.0, "derived_from": ["netdebt.AAA"]},
                "dilution_rate_shares_cagr": {"value": 0.01, "derived_from": ["dilution.AAA"]},
            },
            "net_debt_to_cfo": 1.5,
        },
        "valuation_row": {
            "price_status": "OK",
            "price_reason_code": "OK",
            "valuation_status": "OK",
            "valuation_reason_code": "OK",
            "current_price": 60.0,
            "intrinsic_per_share_base": 95.0,
            "implied_return_base": 0.22,
            "derived_from": ["valuation.AAA"],
        },
        "shares_row": {"shares_status": "OK", "shares_reason_code": "OK"},
        "fcf_row": {"fcf_status": "OK", "fcf_reason_code": "OK"},
        "facts_row": {"status": "OK", "fetch_reason_code": "OK", "derived_from": ["facts.AAA"]},
    }

    memo = build_investment_memo("AAA", universe_run_id="u1", batch_run_id="b1", sources=sources)
    markdown = _memo_markdown(memo)

    assert memo["intrinsic_value_discipline"]["normalized_earnings_power_method_used"] == "FCF_SELECTED"
    assert memo["intrinsic_value_discipline"]["mos_to_floor"] == 0.20
    assert memo["intrinsic_value_discipline"]["downside_support_type"] == "EARNINGS_POWER_SUPPORT"
    assert "## Intrinsic Value Discipline" in markdown
    assert "- mos_to_floor: `0.2`" in markdown
    assert "- downside_support_type: `EARNINGS_POWER_SUPPORT`" in markdown


def test_intrinsic_discipline_open_cli(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "intrinsic_cli_run"
    payload = {
        "run_id": run_id,
        "ticker_count": 1,
        "known_count": 1,
        "unknown_count": 0,
        "top_10_by_mos_to_floor": [{"ticker": "AAA", "mos_to_floor": 0.25}],
        "top_10_with_margin_of_safety": [{"ticker": "AAA", "mos_classification": "ADEQUATE_MARGIN_OF_SAFETY"}],
        "negative_reason_counts": {"RANGE_LOW_CONFIDENCE": 1},
        "downside_support_counts": {"EARNINGS_POWER_SUPPORT": 1},
        "limited_support_count": 0,
        "unknown_support_count": 0,
        "rows": [],
    }
    _write_json(cfg.outputs_dir / "universe" / run_id / "intrinsic_discipline.json", payload)

    result = runner.invoke(app, ["universe-intrinsic-discipline-open", "--run-id", run_id])

    assert result.exit_code == 0
    opened = json.loads(result.stdout)
    assert opened["status"] == "OK"
    assert opened["known_count"] == 1
    assert opened["top_10_by_mos_to_floor"][0]["ticker"] == "AAA"


def test_intrinsic_discipline_mos_uses_upside_ratio_convention():
    """FIX 6: _margin_of_safety uses the UPSIDE-RATIO convention (intrinsic/price - 1).

    For intrinsic_floor=150, price=100 the upside-ratio MoS is 0.50
    (NOT the textbook (intrinsic-price)/intrinsic = 0.333...).
    """
    from app.valuation.intrinsic_discipline import _margin_of_safety

    result = _margin_of_safety(
        intrinsic_floor=150.0,
        intrinsic_base=150.0,
        price_value=100.0,
        price_refs=["prices_summary.rows[TEST]"],
    )
    assert result["mos_to_floor"] == 0.50
    assert result["mos_to_base"] == 0.50
