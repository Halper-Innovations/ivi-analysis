from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.depth_rollup import build_global_shortlist
from app.universe.escalation import ACTION_CLEAR_BLOCKERS, build_escalation_plan
from app.universe.memo_pack import _memo_markdown, build_investment_memo
from app.universe.promotion import LANE_2_RESEARCH_QUEUE, LANE_4_DEPRIORITIZED, build_promotion_state
from app.valuation.intangible_economics import (
    compute_intangible_economics,
    open_intangible_economics,
    write_intangible_economics_for_run,
)
from app.valuation.owner_earnings_quality import compute_owner_earnings_quality


runner = CliRunner()


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker,cik,name\nAAA,1,AAA\nBBB,2,BBB\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _fundamentals_payload(ticker: str, rows: list[dict[str, float | int | str]]) -> dict:
    row_traces: dict[str, dict[str, dict[str, list[str]]]] = {}
    for row in rows:
        year = int(row["year"])
        row_traces[str(year)] = {
            str(field): {"derived_from": [f"fundamentals.{ticker}.{year}.{field}"]}
            for field in row.keys()
            if field != "year"
        }
    return {
        "ticker": ticker,
        "run_id": f"{ticker.lower()}_run",
        "as_of_date": "2026-02-14",
        "rows": rows,
        "row_traces": row_traces,
        "derived_signals": {},
    }


def _owner_quality(ticker: str, rows: list[dict[str, float | int | str]]) -> dict:
    return compute_owner_earnings_quality(
        ticker,
        "2026-02-14",
        fundamentals=_fundamentals_payload(ticker, rows),
    )


def _value_row(
    ticker: str,
    *,
    intangible_total,
    rnd_productivity_score: float | str = 4.0,
    sga_leverage_score: float | str = 3.0,
    owner_value_capture_score: float | str = 4.0,
    owner_value_capture_reason_codes: list[str] | None = None,
    gate: str = "WATCH",
    blocker: str = "MISSING_EV",
    oe_quality_total: float = 9.0,
    implied_return: float = 0.20,
    mos_epv: float = 0.20,
    owner_yield: float = 0.05,
) -> dict:
    return {
        "ticker": ticker,
        "run_id": f"run_{ticker.lower()}",
        "sector": "Software",
        "as_of_date": "2026-02-14",
        "source_depth_runs": [{"run_id": f"run_{ticker.lower()}", "sector": "Software", "as_of_date": "2026-02-14"}],
        "value_gate_status": gate,
        "value_gate_reasons": [],
        "primary_blocker": blocker,
        "implied_return_base": implied_return,
        "implied_return_base_derived_from": [f"value.{ticker}.implied_return_base"],
        "intrinsic_per_share_base": 100.0,
        "intrinsic_per_share_base_derived_from": [f"value.{ticker}.intrinsic_per_share_base"],
        "mos_epv": mos_epv,
        "mos_epv_derived_from": [f"value.{ticker}.mos_epv"],
        "mos_netnet": "UNKNOWN",
        "mos_netnet_derived_from": [f"value.{ticker}.mos_netnet"],
        "owner_earnings_yield_ev_3y": owner_yield,
        "owner_earnings_yield_ev_3y_derived_from": [f"value.{ticker}.owner_earnings_yield_ev_3y"],
        "fcf_yield_ev_3y": 0.04,
        "fcf_yield_ev_3y_derived_from": [f"value.{ticker}.fcf_yield_ev_3y"],
        "yield_metric_used": "owner_earnings_yield_ev_3y",
        "yield_denominator_used": "EV",
        "owner_earnings_stability_score": 4.0,
        "owner_earnings_stability_score_derived_from": [f"oe.{ticker}.owner_earnings_stability_score"],
        "capital_allocation_score": 4.0,
        "capital_allocation_score_derived_from": [f"oe.{ticker}.capital_allocation_score"],
        "cash_conversion_score": 3.0,
        "cash_conversion_score_derived_from": [f"oe.{ticker}.cash_conversion_score"],
        "oe_quality_total": oe_quality_total,
        "oe_quality_total_derived_from": [f"oe.{ticker}.oe_quality_total"],
        "oe_quality_reason_codes": ["SHAREHOLDER_FRIENDLY"],
        "gross_margin_durability_score": 4.0 if intangible_total != "UNKNOWN" else "UNKNOWN",
        "gross_margin_durability_score_derived_from": [f"intangible.{ticker}.gross_margin_durability_score"],
        "balance_sheet_optionality_score": 3.0 if intangible_total != "UNKNOWN" else "UNKNOWN",
        "balance_sheet_optionality_score_derived_from": [f"intangible.{ticker}.balance_sheet_optionality_score"],
        "cycle_resilience_score": 3.0 if intangible_total != "UNKNOWN" else "UNKNOWN",
        "cycle_resilience_score_derived_from": [f"intangible.{ticker}.cycle_resilience_score"],
        "rnd_productivity_score": rnd_productivity_score if intangible_total != "UNKNOWN" else "UNKNOWN",
        "rnd_productivity_score_derived_from": [f"intangible.{ticker}.rnd_productivity_score"],
        "sga_leverage_score": sga_leverage_score if intangible_total != "UNKNOWN" else "UNKNOWN",
        "sga_leverage_score_derived_from": [f"intangible.{ticker}.sga_leverage_score"],
        "owner_value_capture_score": owner_value_capture_score if intangible_total != "UNKNOWN" else "UNKNOWN",
        "owner_value_capture_score_derived_from": [f"intangible.{ticker}.owner_value_capture_score"],
        "intangible_economics_total": intangible_total,
        "intangible_economics_total_derived_from": [f"intangible.{ticker}.intangible_economics_total"],
        "rnd_productivity_reason_codes": (
            ["RND_PRODUCTIVITY_UNKNOWN"] if intangible_total == "UNKNOWN" else []
        ),
        "sga_leverage_reason_codes": (
            ["SGA_LEVERAGE_UNKNOWN"] if intangible_total == "UNKNOWN" else []
        ),
        "owner_value_capture_reason_codes": (
            owner_value_capture_reason_codes
            if owner_value_capture_reason_codes is not None
            else (["STRONG_OWNER_VALUE_CAPTURE"] if intangible_total != "UNKNOWN" else ["OWNER_VALUE_CAPTURE_UNKNOWN"])
        ),
        "intangible_economics_reason_codes": (
            ["BALANCE_SHEET_OPTIONALITY_STRONG", "RESILIENT_CYCLICAL_PROFILE"]
            if intangible_total != "UNKNOWN"
            else ["GROSS_MARGIN_DURABILITY_UNKNOWN"]
        ),
        "quality_score": 12.0,
        "risk_penalty": -2.0,
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
        "derived_from": [f"shortlist.{ticker}"],
    }


def test_gross_margin_durability_scoring_is_deterministic():
    rows = [
        {"year": 2021, "revenue": 100.0, "gross_profit": 62.0, "cfo": 30.0, "capex": 6.0, "fcf": 24.0, "shares_outstanding": 100.0, "net_debt": -5.0},
        {"year": 2022, "revenue": 105.0, "gross_profit": 64.05, "cfo": 32.0, "capex": 6.0, "fcf": 26.0, "shares_outstanding": 99.0, "net_debt": -6.0},
        {"year": 2023, "revenue": 110.0, "gross_profit": 66.0, "cfo": 34.0, "capex": 7.0, "fcf": 27.0, "shares_outstanding": 98.0, "net_debt": -7.0},
        {"year": 2024, "revenue": 116.0, "gross_profit": 73.08, "cfo": 36.0, "capex": 7.0, "fcf": 29.0, "shares_outstanding": 97.0, "net_debt": -8.0},
        {"year": 2025, "revenue": 121.0, "gross_profit": 73.81, "cfo": 38.0, "capex": 7.0, "fcf": 31.0, "shares_outstanding": 96.0, "net_debt": -10.0},
    ]
    payload = compute_intangible_economics(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_payload("AAA", rows),
        owner_quality_payload=_owner_quality("AAA", rows),
    )

    assert payload["gross_margin_durability_score"] == 5.0
    assert payload["gross_margin_avg_5y"] > 0.60
    assert "VOLATILE_GROSS_MARGIN" not in payload["intangible_economics_reason_codes"]


def test_balance_sheet_optionality_scoring_is_deterministic():
    rows = [
        {"year": 2024, "revenue": 100.0, "gross_profit": 55.0, "cfo": 24.0, "capex": 4.0, "fcf": 20.0, "shares_outstanding": 100.0, "net_debt": -20.0},
        {"year": 2025, "revenue": 110.0, "gross_profit": 61.6, "cfo": 28.0, "capex": 4.0, "fcf": 24.0, "shares_outstanding": 99.0, "net_debt": -25.0},
    ]
    payload = compute_intangible_economics(
        "BBB",
        "2026-02-14",
        fundamentals=_fundamentals_payload("BBB", rows),
        owner_quality_payload=_owner_quality("BBB", rows),
    )

    assert payload["balance_sheet_optionality_score"] == 5.0
    assert payload["net_debt_to_cfo_proxy"] < 0
    assert "BALANCE_SHEET_OPTIONALITY_STRONG" in payload["balance_sheet_optionality_reason_codes"]


def test_cycle_resilience_scoring_is_deterministic():
    rows = [
        {"year": 2021, "revenue": 100.0, "gross_profit": 50.0, "cfo": 30.0, "capex": 6.0, "fcf": 21.0, "shares_outstanding": 100.0, "net_debt": 10.0},
        {"year": 2022, "revenue": 103.0, "gross_profit": 51.5, "cfo": 31.0, "capex": 6.0, "fcf": 22.0, "shares_outstanding": 99.5, "net_debt": 9.0},
        {"year": 2023, "revenue": 106.0, "gross_profit": 53.0, "cfo": 32.0, "capex": 6.0, "fcf": 23.0, "shares_outstanding": 99.0, "net_debt": 8.0},
        {"year": 2024, "revenue": 109.0, "gross_profit": 54.5, "cfo": 33.0, "capex": 6.0, "fcf": 24.0, "shares_outstanding": 98.5, "net_debt": 7.0},
        {"year": 2025, "revenue": 112.0, "gross_profit": 56.0, "cfo": 34.0, "capex": 6.0, "fcf": 25.0, "shares_outstanding": 98.0, "net_debt": 6.0},
    ]
    payload = compute_intangible_economics(
        "CCC",
        "2026-02-14",
        fundamentals=_fundamentals_payload("CCC", rows),
        owner_quality_payload=_owner_quality("CCC", rows),
    )

    assert payload["cycle_resilience_score"] >= 4.0
    assert "RESILIENT_CYCLICAL_PROFILE" in payload["cycle_resilience_reason_codes"]


def test_unknown_handling_and_reason_codes_are_explicit():
    rows = [
        {"year": 2024, "revenue": "UNKNOWN", "cfo": "UNKNOWN", "fcf": "UNKNOWN", "net_debt": "UNKNOWN"},
        {"year": 2025, "revenue": "UNKNOWN", "cfo": "UNKNOWN", "fcf": "UNKNOWN", "net_debt": "UNKNOWN"},
    ]
    payload = compute_intangible_economics(
        "DDD",
        "2026-02-14",
        fundamentals=_fundamentals_payload("DDD", rows),
        owner_quality_payload=_owner_quality("DDD", rows),
    )

    assert payload["gross_margin_durability_score"] == "UNKNOWN"
    assert payload["balance_sheet_optionality_score"] == "UNKNOWN"
    assert payload["cycle_resilience_score"] == "UNKNOWN"
    assert payload["rnd_productivity_score"] == "UNKNOWN"
    assert payload["sga_leverage_score"] == "UNKNOWN"
    assert payload["owner_value_capture_score"] == "UNKNOWN"
    assert "MISSING_GROSS_MARGIN_HISTORY" in payload["intangible_economics_reason_codes"]
    assert "MISSING_BALANCE_SHEET_OPTIONALITY_INPUTS" in payload["intangible_economics_reason_codes"]
    assert "INSUFFICIENT_CYCLE_HISTORY" in payload["intangible_economics_reason_codes"]
    assert "MISSING_RND_HISTORY" in payload["intangible_economics_reason_codes"]
    assert "MISSING_SGA_HISTORY" in payload["intangible_economics_reason_codes"]
    assert "MISSING_PER_SHARE_INPUTS" in payload["intangible_economics_reason_codes"]


def test_rnd_productivity_scoring_is_deterministic():
    rows = [
        {"year": 2021, "revenue": 100.0, "gross_profit": 64.0, "cfo": 28.0, "capex": 4.0, "fcf": 24.0, "shares_outstanding": 100.0, "net_debt": -5.0, "r_and_d_total": 10.0},
        {"year": 2022, "revenue": 110.0, "gross_profit": 71.5, "cfo": 31.0, "capex": 4.0, "fcf": 27.0, "shares_outstanding": 99.0, "net_debt": -6.0, "r_and_d_total": 10.5},
        {"year": 2023, "revenue": 122.0, "gross_profit": 80.5, "cfo": 35.0, "capex": 5.0, "fcf": 30.0, "shares_outstanding": 98.0, "net_debt": -7.0, "r_and_d_total": 11.0},
        {"year": 2024, "revenue": 135.0, "gross_profit": 90.5, "cfo": 39.0, "capex": 5.0, "fcf": 34.0, "shares_outstanding": 97.0, "net_debt": -8.0, "r_and_d_total": 11.5},
        {"year": 2025, "revenue": 149.0, "gross_profit": 100.6, "cfo": 43.0, "capex": 5.0, "fcf": 38.0, "shares_outstanding": 96.0, "net_debt": -10.0, "r_and_d_total": 12.0},
    ]
    payload = compute_intangible_economics(
        "RND",
        "2026-02-14",
        fundamentals=_fundamentals_payload("RND", rows),
        owner_quality_payload=_owner_quality("RND", rows),
    )

    assert payload["rnd_productivity_score"] >= 4.0
    assert payload["revenue_per_rnd_proxy"] > 10.0
    assert "LOW_RND_PRODUCTIVITY" not in payload["rnd_productivity_reason_codes"]


def test_sga_leverage_scoring_is_deterministic():
    rows = [
        {"year": 2021, "revenue": 100.0, "gross_profit": 68.0, "operating_income": 18.0, "cfo": 24.0, "capex": 4.0, "fcf": 20.0, "shares_outstanding": 100.0, "net_debt": -2.0, "sga_total": 32.0},
        {"year": 2022, "revenue": 111.0, "gross_profit": 76.6, "operating_income": 22.0, "cfo": 27.0, "capex": 4.0, "fcf": 23.0, "shares_outstanding": 99.0, "net_debt": -3.0, "sga_total": 33.0},
        {"year": 2023, "revenue": 123.0, "gross_profit": 85.0, "operating_income": 27.0, "cfo": 30.0, "capex": 4.0, "fcf": 26.0, "shares_outstanding": 98.0, "net_debt": -4.0, "sga_total": 34.0},
        {"year": 2024, "revenue": 136.0, "gross_profit": 95.2, "operating_income": 31.0, "cfo": 33.0, "capex": 4.0, "fcf": 29.0, "shares_outstanding": 97.0, "net_debt": -5.0, "sga_total": 35.0},
        {"year": 2025, "revenue": 150.0, "gross_profit": 105.0, "operating_income": 36.0, "cfo": 36.0, "capex": 4.0, "fcf": 32.0, "shares_outstanding": 96.0, "net_debt": -6.0, "sga_total": 36.0},
    ]
    payload = compute_intangible_economics(
        "SGA",
        "2026-02-14",
        fundamentals=_fundamentals_payload("SGA", rows),
        owner_quality_payload=_owner_quality("SGA", rows),
    )

    assert payload["sga_leverage_score"] >= 4.0
    assert payload["sga_growth_vs_revenue_growth_proxy"] > 0
    assert "WEAK_SGA_LEVERAGE" not in payload["sga_leverage_reason_codes"]


def test_owner_value_capture_penalizes_dilution_and_weak_per_share_capture():
    rows = [
        {"year": 2021, "revenue": 100.0, "gross_profit": 65.0, "cfo": 28.0, "capex": 4.0, "fcf": 20.0, "shares_outstanding": 100.0, "net_debt": -2.0},
        {"year": 2022, "revenue": 112.0, "gross_profit": 72.0, "cfo": 29.0, "capex": 4.0, "fcf": 20.5, "shares_outstanding": 108.0, "net_debt": -2.0},
        {"year": 2023, "revenue": 125.0, "gross_profit": 81.0, "cfo": 30.0, "capex": 5.0, "fcf": 21.0, "shares_outstanding": 117.0, "net_debt": -1.0},
        {"year": 2024, "revenue": 139.0, "gross_profit": 90.0, "cfo": 31.0, "capex": 5.0, "fcf": 21.5, "shares_outstanding": 127.0, "net_debt": -1.0},
        {"year": 2025, "revenue": 154.0, "gross_profit": 100.0, "cfo": 32.0, "capex": 5.0, "fcf": 22.0, "shares_outstanding": 138.0, "net_debt": 0.0},
    ]
    payload = compute_intangible_economics(
        "OWN",
        "2026-02-14",
        fundamentals=_fundamentals_payload("OWN", rows),
        owner_quality_payload=_owner_quality("OWN", rows),
    )

    assert payload["owner_value_capture_score"] <= 2.0
    assert "EXCESS_DILUTION" in payload["owner_value_capture_reason_codes"]
    assert "WEAK_PER_SHARE_CAPTURE" in payload["owner_value_capture_reason_codes"]


def test_value_first_intangible_shortlist_prefers_higher_intangible_total():
    payload = build_global_shortlist(
        [
            _value_row("AAA", intangible_total=10.0),
            _value_row("BBB", intangible_total=4.0),
        ],
        top_n=2,
        policy="value_first_intangible",
    )

    assert [row["ticker"] for row in payload["rows_top_n"]] == ["AAA", "BBB"]


def test_value_first_intangible_does_not_override_better_value_signal():
    payload = build_global_shortlist(
        [
            _value_row("AAA", intangible_total=14.0, implied_return=0.18),
            _value_row("BBB", intangible_total=6.0, implied_return=0.28),
        ],
        top_n=2,
        policy="value_first_intangible",
    )

    assert [row["ticker"] for row in payload["rows_top_n"]] == ["BBB", "AAA"]


def test_promotion_and_escalation_use_intangible_support_without_overriding_fail(monkeypatch, tmp_path):
    _cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.universe.promotion._load_memory_lookup", lambda: {})
    monkeypatch.setattr("app.universe.escalation._load_memory_lookup", lambda: {})

    master_watchlist = {
        "campaign_run_id": "campaign_intangible",
        "tickers": {
            "AAA": {
                "latest_value_gate_status": "FAIL",
                "latest_implied_return_base": 0.05,
                "latest_primary_blocker": "MISSING_EV",
                "appearances_count": 1,
                "history": [],
            },
            "BBB": {
                "latest_value_gate_status": "WATCH",
                "latest_implied_return_base": "UNKNOWN",
                "latest_primary_blocker": "MISSING_EV",
                "appearances_count": 2,
                "history": [],
            },
            "CCC": {
                "latest_value_gate_status": "WATCH",
                "latest_implied_return_base": "UNKNOWN",
                "latest_primary_blocker": "MISSING_EV",
                "appearances_count": 2,
                "history": [],
            },
        },
    }
    master_shortlist = {
        "campaign_run_id": "campaign_intangible",
        "rows": [
            {
                **_value_row("AAA", gate="FAIL", blocker="MISSING_EV", intangible_total=12.0),
                "source_runs": [{"campaign_item": "item_a", "universe_run_id": "run_a", "batch_run_id": "run_a_depth_batch"}],
                "best_rank_seen": 3,
                "latest_value_gate_status": "FAIL",
                "implied_return_base": "UNKNOWN",
                "mos_epv": "UNKNOWN",
                "owner_earnings_yield_ev_3y": "UNKNOWN",
                "yield_metric_used": "UNKNOWN",
                "latest_primary_blocker": "MISSING_EV",
                "memo_path": "",
                "composite_score_total": 4.0,
            },
            {
                **_value_row("BBB", gate="WATCH", blocker="MISSING_EV", intangible_total=11.0),
                "source_runs": [{"campaign_item": "item_b", "universe_run_id": "run_b", "batch_run_id": "run_b_depth_batch"}],
                "best_rank_seen": 2,
                "latest_value_gate_status": "WATCH",
                "latest_primary_blocker": "MISSING_EV",
                "memo_path": "",
                "composite_score_total": 6.0,
            },
            {
                **_value_row(
                    "CCC",
                    gate="WATCH",
                    blocker="MISSING_EV",
                    intangible_total=10.0,
                    owner_value_capture_score=1.0,
                    owner_value_capture_reason_codes=["EXCESS_DILUTION", "WEAK_PER_SHARE_CAPTURE"],
                ),
                "source_runs": [{"campaign_item": "item_c", "universe_run_id": "run_c", "batch_run_id": "run_c_depth_batch"}],
                "best_rank_seen": 4,
                "latest_value_gate_status": "WATCH",
                "latest_primary_blocker": "MISSING_EV",
                "memo_path": "",
                "composite_score_total": 6.0,
            },
        ],
    }

    promotion_payload = build_promotion_state("campaign_intangible", master_watchlist, master_shortlist)
    rows_by_ticker = {row["ticker"]: row for row in promotion_payload["rows"]}
    assert rows_by_ticker["AAA"]["priority_lane"] == LANE_4_DEPRIORITIZED
    assert "STRONG_INTANGIBLE_ECONOMICS_SUPPORT" not in rows_by_ticker["AAA"]["promotion_reason_codes"]
    assert rows_by_ticker["BBB"]["priority_lane"] == LANE_2_RESEARCH_QUEUE
    assert "STRONG_INTANGIBLE_ECONOMICS_SUPPORT" in rows_by_ticker["BBB"]["promotion_reason_codes"]
    assert "STRONG_OWNER_VALUE_CAPTURE_SUPPORT" in rows_by_ticker["BBB"]["promotion_reason_codes"]
    assert "WEAK_OWNER_VALUE_CAPTURE_HEADWIND" in rows_by_ticker["CCC"]["promotion_reason_codes"]

    queue_payload = build_escalation_plan(
        "campaign_intangible",
        promotion_payload,
        {"campaign_run_id": "campaign_intangible", "lane_2_research_queue": promotion_payload["rows"]},
        config={"as_of_date": "2026-02-14", "source_campaign_file": "data/universe/sample_campaign.json", "top_n": 5, "policy": "value_first_intangible"},
    )
    queue_by_ticker = {row["ticker"]: row for row in queue_payload["queue"] if row["action_type"] == ACTION_CLEAR_BLOCKERS}
    assert "BBB" in queue_by_ticker
    assert "STRONG_INTANGIBLE_ECONOMICS_SUPPORT" in queue_by_ticker["BBB"]["priority_support_codes"]
    assert "STRONG_OWNER_VALUE_CAPTURE_SUPPORT" in queue_by_ticker["BBB"]["priority_support_codes"]
    assert "WEAK_OWNER_VALUE_CAPTURE_HEADWIND" in queue_by_ticker["CCC"]["priority_support_codes"]
    assert "AAA" not in queue_by_ticker


def test_memo_pack_includes_modern_intangible_section():
    memo = build_investment_memo(
        "AAA",
        universe_run_id="u_intangible",
        batch_run_id="b_intangible",
        sources={
            "shortlist_row": _value_row("AAA", intangible_total=10.0),
            "score_row": {
                "ticker": "AAA",
                "metric_values": {
                    "implied_return_base": 0.20,
                    "intrinsic_per_share_base": 120.0,
                    "current_price": 100.0,
                    "mos_epv": 0.20,
                    "mos_netnet": "UNKNOWN",
                    "epv_per_share": 130.0,
                    "netnet_per_share": "UNKNOWN",
                    "owner_earnings_yield_ev_3y": 0.05,
                    "fcf_yield_ev_3y": 0.04,
                    "owner_earnings_yield_3y": 0.05,
                    "fcf_yield_3y": 0.04,
                    "revenue_cagr_5y": 0.10,
                    "revenue_cagr_10y": 0.08,
                    "operating_margin_trend_slope": 0.01,
                    "gross_margin_trend_slope": 0.01,
                    "fcf_margin_trend_slope": 0.01,
                    "roic_proxy": 0.15,
                    "dilution_rate_shares_cagr": 0.00,
                    "net_debt_proxy": 20.0,
                    "risk_factor_keyword_delta": 0.0,
                    "quality_score": 12.0,
                    "risk_penalty": -1.0,
                    "owner_earnings_stability_score": 4.0,
                    "capital_allocation_score": 4.0,
                    "cash_conversion_score": 3.0,
                    "oe_quality_total": 10.0,
                    "gross_margin_durability_score": 4.0,
                    "balance_sheet_optionality_score": 3.0,
                    "cycle_resilience_score": 3.0,
                    "rnd_productivity_score": 4.0,
                    "sga_leverage_score": 3.0,
                    "owner_value_capture_score": 4.0,
                    "intangible_economics_total": 10.0,
                },
                "metric_traces": {
                    key: {"derived_from": [f"trace.AAA.{key}"]}
                    for key in [
                        "implied_return_base",
                        "intrinsic_per_share_base",
                        "current_price",
                        "mos_epv",
                        "mos_netnet",
                        "epv_per_share",
                        "netnet_per_share",
                        "owner_earnings_yield_ev_3y",
                        "fcf_yield_ev_3y",
                        "owner_earnings_yield_3y",
                        "fcf_yield_3y",
                        "revenue_cagr_5y",
                        "revenue_cagr_10y",
                        "operating_margin_trend_slope",
                        "gross_margin_trend_slope",
                        "fcf_margin_trend_slope",
                        "roic_proxy",
                        "dilution_rate_shares_cagr",
                        "net_debt_proxy",
                        "risk_factor_keyword_delta",
                        "quality_score",
                        "risk_penalty",
                        "owner_earnings_stability_score",
                        "capital_allocation_score",
                        "cash_conversion_score",
                        "oe_quality_total",
                        "gross_margin_durability_score",
                        "balance_sheet_optionality_score",
                        "cycle_resilience_score",
                        "rnd_productivity_score",
                        "sga_leverage_score",
                        "owner_value_capture_score",
                        "intangible_economics_total",
                    ]
                },
            },
            "gate_row": {
                "ticker": "AAA",
                "gate_status": "WATCH",
                "gate_reasons": [],
                "primary_blocker": "MISSING_EV",
                "inputs_used": {
                    "current_price": {"value": 100.0, "derived_from": ["price.AAA"]},
                    "net_debt_proxy": {"value": 20.0, "derived_from": ["netdebt.AAA"]},
                    "dilution_rate_shares_cagr": {"value": 0.0, "derived_from": ["dilution.AAA"]},
                },
                "net_debt_to_cfo": 1.0,
            },
            "valuation_row": {
                "ticker": "AAA",
                "price_status": "OK",
                "valuation_status": "OK",
                "price_reason_code": "OK",
                "valuation_reason_code": "OK",
                "current_price": 100.0,
                "implied_return_base": 0.20,
                "intrinsic_per_share_base": 120.0,
                "derived_from": ["valuation.AAA"],
            },
            "shares_row": {"ticker": "AAA", "shares_status": "OK", "shares_reason_code": "OK", "derived_from": ["shares.AAA"]},
            "fcf_row": {"ticker": "AAA", "fcf_status": "OK", "fcf_reason_code": "OK", "derived_from": ["fcf.AAA"]},
            "facts_row": {"ticker": "AAA", "status": "OK", "fetch_reason_code": "OK", "derived_from": ["facts.AAA"]},
        },
    )

    assert memo["modern_intangible_economics"]["intangible_economics_total"] == 10.0
    markdown = _memo_markdown(memo)
    assert "## Modern Intangible Economics Overlay" in markdown
    assert "- rnd_productivity_score: `4.0`" in markdown
    assert "- sga_leverage_score: `3.0`" in markdown
    assert "- owner_value_capture_score: `4.0`" in markdown
    assert "- intangible_economics_total: `10.0`" in markdown
    assert "- derived_from: `" in markdown
    assert "trace.AAA.gross_margin_durability_score" in markdown


def test_write_and_open_intangible_economics_artifact(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "intangible_run"
    rows = [
        {"year": 2024, "revenue": 100.0, "gross_profit": 55.0, "cfo": 24.0, "capex": 4.0, "fcf": 20.0, "shares_outstanding": 100.0, "net_debt": -20.0},
        {"year": 2025, "revenue": 110.0, "gross_profit": 61.6, "cfo": 28.0, "capex": 4.0, "fcf": 24.0, "shares_outstanding": 99.0, "net_debt": -25.0},
    ]
    fundamentals = _fundamentals_payload("AAA", rows)
    owner_quality_payload = _owner_quality("AAA", rows)
    output_path = cfg.outputs_dir / "universe" / run_id / "intangible_economics.json"
    payload = write_intangible_economics_for_run(
        run_id=run_id,
        as_of_date="2026-02-14",
        tickers=["AAA"],
        output_path=output_path,
        scoreboard_rows=[
            {
                "ticker": "AAA",
                "intangible_economics_detail": compute_intangible_economics(
                    "AAA",
                    "2026-02-14",
                    fundamentals=fundamentals,
                    owner_quality_payload=owner_quality_payload,
                ),
            }
        ],
    )

    assert Path(payload["intangible_economics_path"]).exists()
    opened = open_intangible_economics(run_id=run_id)
    assert opened["status"] == "OK"
    assert opened["known_count"] == 1
    assert "top_owner_value_capture" in opened
    assert "rnd_unknown_reason_counts" in opened
    assert "sga_unknown_reason_counts" in opened

    cli = runner.invoke(app, ["universe-intangible-economics-open", "--run-id", run_id])
    assert cli.exit_code == 0
    assert "top_10_by_intangible_economics_total" in cli.stdout
    assert "top_owner_value_capture" in cli.stdout
