from __future__ import annotations

import json

from app.db import init_db
from app.universe.ranking import compute_composite_score, ranking_sort_key
from app.universe.scout import run_universe_scout, universe_depth_queue_to_runs


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "universe.csv"
    taxonomy = data_dir / "universe" / "sector_taxonomy.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker,cik,name\nAAA,1,AAA\nBBB,2,BBB\nCCC,3,CCC\nDDD,4,DDD\n", encoding="utf-8")
    taxonomy.write_text("ticker,sector\nAAA,Software\nBBB,Software\nCCC,Healthcare\nDDD,Healthcare\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe))
    monkeypatch.setenv("VOE_SECTOR_TAXONOMY_PATH", str(taxonomy))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_composite_score_determinism():
    row = {
        "ticker": "AAA",
        "scout_status": "WATCH",
        "mos_epv": 0.35,
        "mos_netnet": 0.10,
        "yield_gate_value_used": 0.06,
        "metric_values": {
            "cfo_value": 100.0,
            "fcf_value": 70.0,
            "dilution_rate": 0.02,
            "net_debt_to_cfo": 1.2,
            "roic_proxy": 0.14,
            "fcf_margin_trend_slope": 0.01,
        },
        "derived_from": ["row.AAA"],
    }
    gd = {"mos_epv": 0.35, "mos_netnet": 0.10, "derived_from": ["gd.AAA"]}
    yld = {"yield_gate_value_used": 0.06, "derived_from": ["yield.AAA"]}
    first = compute_composite_score(row, gd=gd, yield_data=yld)
    second = compute_composite_score(row, gd=gd, yield_data=yld)
    assert first == second
    assert first["status"] == "OK"
    assert first["composite_score_total"] > 0


def test_ranking_sort_tie_break_deterministic():
    rows = [
        {
            "ticker": "BBB",
            "scout_status": "WATCH",
            "composite_score_total": 60.0,
            "mos_epv": 0.20,
            "mos_netnet": 0.05,
            "yield_gate_value_used": 0.04,
        },
        {
            "ticker": "AAA",
            "scout_status": "WATCH",
            "composite_score_total": 60.0,
            "mos_epv": 0.20,
            "mos_netnet": 0.05,
            "yield_gate_value_used": 0.04,
        },
        {
            "ticker": "CCC",
            "scout_status": "PASS",
            "composite_score_total": 55.0,
            "mos_epv": 0.05,
            "mos_netnet": 0.02,
            "yield_gate_value_used": 0.03,
        },
        {
            "ticker": "DDD",
            "scout_status": "WATCH",
            "composite_score_total": 60.0,
            "mos_epv": 0.25,
            "mos_netnet": 0.05,
            "yield_gate_value_used": 0.04,
        },
    ]
    ordered = sorted(rows, key=ranking_sort_key)
    assert [row["ticker"] for row in ordered] == ["CCC", "DDD", "AAA", "BBB"]


def test_shortlist_diversification_respects_max_per_sector(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setenv("VOE_UNIVERSE_SHORTLIST_MAX_PER_SECTOR", "1")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    _ = _get_config()

    monkeypatch.setattr(
        "app.universe.scout.write_prices_for_run",
        lambda **kwargs: {
            "summary_path": "stub://prices",
            "rows": [
                {"ticker": str(t).upper(), "status": "OK", "price": 10.0, "reason_code": "CACHE_HIT"}
                for t in (kwargs.get("tickers") or [])
            ],
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.resolve_financial_facts_asof",
        lambda **kwargs: {
            "ticker": str(kwargs.get("ticker") or "").upper(),
            "status": "OK",
            "shares_status": "OK",
            "shares_reason": "OK",
            "shares_value": 10.0,
            "cfo_status": "OK",
            "cfo_reason": "OK",
            "cfo_value": 100.0,
            "capex_status": "OK",
            "capex_reason": "OK",
            "capex_value": 20.0,
            "fcf_status": "OK",
            "fcf_reason": "OK",
            "fcf_value": 70.0,
            "fetch_reason_code": "CACHE_HIT",
            "cache_path": None,
            "derived_from": [f"facts.{str(kwargs.get('ticker') or '').upper()}"],
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.resolve_net_debt_proxy",
        lambda **kwargs: {
            "ticker": str(kwargs.get("ticker") or "").upper(),
            "status": "OK",
            "reason_code": "OK",
            "net_debt_proxy": 50.0,
            "total_debt": {"value": 80.0},
            "cash_equivalents": {"value": 30.0},
            "derived_from": [f"netdebt.{str(kwargs.get('ticker') or '').upper()}"],
        },
    )

    def _build_record_stub(**kwargs):
        ticker = str(kwargs.get("ticker") or "").upper()
        composite_by_ticker = {"AAA": 80.0, "BBB": 79.0, "CCC": 78.0, "DDD": 77.0}
        mos_by_ticker = {"AAA": 0.5, "BBB": 0.4, "CCC": 0.3, "DDD": 0.2}
        score_row = {
            "ticker": ticker,
            "scout_status": "PASS",
            "score_total": composite_by_ticker[ticker],
            "composite_score_total": composite_by_ticker[ticker],
            "gd_score": 30.0,
            "yield_score": 18.0,
            "quality_score": 12.0,
            "risk_penalty": -2.0,
            "composite_status": "OK",
            "composite_reason_code": "OK",
            "primary_blocker": "NONE",
            "primary_blocker_category": "NONE",
            "blocker_categories": [],
            "delta_to_pass": 0.0,
            "yield_metric_used": "OWNER_EARNINGS_EV_MEDIAN_3Y",
            "yield_gate_value_used": 0.06,
            "yield_status": "OK",
            "yield_reason_code": "OWNER_EARNINGS_YIELD_EV_3Y",
            "yield_blocker_subreason": "YIELD_ABOVE_THRESHOLD",
            "yield_denominator_used": "EV",
            "yield_calibration_note": "",
            "yield_delta_to_pass": 0.0,
            "mos_epv": mos_by_ticker[ticker],
            "mos_netnet": -0.1,
            "epv_per_share": 20.0,
            "netnet_per_share": 9.0,
            "gd_value_status": "OK",
            "gd_primary_reason_code": "OK",
            "reasons": ["OK"],
            "metric_values": {
                "valuation_gap": 0.2,
                "fcf_yield": 0.06,
                "fcf_yield_3y": 0.06,
                "owner_earnings_yield_3y": 0.06,
                "net_debt_to_cfo": 0.5,
                "dilution_rate": 0.01,
                "mos_epv": mos_by_ticker[ticker],
                "mos_netnet": -0.1,
            },
            "near_miss_fields": [],
            "recommendation": "none",
            "inputs_used": {},
            "derived_from": [f"stub.{ticker}"],
            "graham_dodd_detail": {},
            "composite_detail": {},
        }
        cov_row = {
            "ticker": ticker,
            "scout_status": "PASS",
            "primary_blocker": "NONE",
            "primary_blocker_category": "NONE",
            "blocker_categories": [],
            "near_miss_fields": [],
            "recommendation": "none",
            "yield_metric_used": score_row["yield_metric_used"],
            "yield_status": "OK",
            "yield_reason_code": score_row["yield_reason_code"],
            "yield_blocker_subreason": score_row["yield_blocker_subreason"],
            "yield_denominator_used": "EV",
            "yield_calibration_note": "",
            "yield_delta_to_pass": 0.0,
            "price_status": "OK",
            "price_reason_code": "CACHE_HIT",
            "facts_status": "OK",
            "fetch_reason_code": "CACHE_HIT",
            "shares_status": "OK",
            "shares_reason_code": "OK",
            "cfo_status": "OK",
            "cfo_reason_code": "OK",
            "capex_status": "OK",
            "capex_reason_code": "OK",
            "fcf_status": "OK",
            "fcf_reason_code": "OK",
            "net_debt_status": "OK",
            "net_debt_reason_code": "OK",
            "market_cap_status": "OK",
            "market_cap_reason_code": "PRICE_X_SHARES",
            "ev_status": "OK",
            "ev_reason_code": "OK",
            "require_ev_yield": False,
            "dilution_status": "OK",
            "dilution_reason_code": "OK",
            "fcf_stability_status": "OK",
            "fcf_stability_reason_code": "OK",
            "valuation_gap_status": "OK",
            "valuation_gap_reason_code": "OK",
            "unknown_reasons": [],
            "fcf_history_values": [1.0, 1.0, 1.0],
            "mos_epv": score_row["mos_epv"],
            "mos_netnet": score_row["mos_netnet"],
            "gd_value_status": "OK",
            "gd_primary_reason_code": "OK",
            "composite_score_total": score_row["composite_score_total"],
            "gd_score": score_row["gd_score"],
            "yield_score": score_row["yield_score"],
            "quality_score": score_row["quality_score"],
            "risk_penalty": score_row["risk_penalty"],
            "composite_status": "OK",
            "composite_reason_code": "OK",
            "graham_dodd_detail": {},
            "derived_from": [f"stub.{ticker}"],
        }
        yld_row = {
            "ticker": ticker,
            "scout_status": "PASS",
            "yield_status": "OK",
            "yield_reason_code": score_row["yield_reason_code"],
            "yield_blocker_subreason": "YIELD_ABOVE_THRESHOLD",
            "yield_metric_used": score_row["yield_metric_used"],
            "yield_metric_type": "OWNER_EARNINGS",
            "yield_denominator_used": "EV",
            "yield_calibration_note": "",
            "yield_gate_value_used": 0.06,
            "yield_delta_to_pass": 0.0,
            "owner_earnings_yield_3y": 0.06,
            "fcf_yield_3y": 0.04,
            "owner_earnings_yield_ev_3y": 0.06,
            "fcf_yield_ev_3y": 0.04,
            "ev": 100.0,
            "ev_value": 100.0,
            "ev_status": "OK",
            "ev_reason_code": "OK",
            "ev_used": True,
            "net_debt_proxy_used": 50.0,
            "net_debt_proxy_reason_code": "OK",
            "denominator_used": "EV",
            "primary_blocker_category": "NONE",
            "epv_per_share": score_row["epv_per_share"],
            "mos_epv": score_row["mos_epv"],
            "netnet_per_share": score_row["netnet_per_share"],
            "mos_netnet": score_row["mos_netnet"],
            "gd_value_status": "OK",
            "gd_primary_reason_code": "OK",
            "composite_score_total": score_row["composite_score_total"],
            "gd_score": score_row["gd_score"],
            "yield_score": score_row["yield_score"],
            "quality_score": score_row["quality_score"],
            "risk_penalty": score_row["risk_penalty"],
            "composite_status": "OK",
            "composite_reason_code": "OK",
            "owner_earnings_summary": {},
            "maintenance_capex_ratio": 0.6,
            "proxy_flags": {"maintenance_capex_proxy": True, "owner_earnings_proxy": True},
            "inputs_used": {},
            "derived_from": [f"stub.{ticker}"],
        }
        return score_row, cov_row, yld_row

    monkeypatch.setattr("app.universe.scout._build_scout_record", _build_record_stub)
    run_universe_scout(run_id="diversify_test", as_of_date="2026-02-14", top_n=4, tickers=["AAA", "BBB", "CCC", "DDD"])

    shortlist = json.loads((cfg.sectors_dir / "diversify_test" / "universe_shortlist.json").read_text(encoding="utf-8"))
    by_sector = shortlist["top_candidates_by_sector"]
    assert by_sector
    for bucket in by_sector:
        assert len(bucket["candidates"]) <= 1


def test_depth_queue_is_deterministic_and_commands_stable(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setenv("VOE_UNIVERSE_DEPTH_QUEUE_CHUNK_SIZE", "2")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    _ = _get_config()
    monkeypatch.setattr(
        "app.universe.scout.write_prices_for_run",
        lambda **kwargs: {
            "summary_path": "stub://prices",
            "rows": [
                {"ticker": str(t).upper(), "status": "OK", "price": 10.0, "reason_code": "CACHE_HIT"}
                for t in (kwargs.get("tickers") or [])
            ],
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.resolve_financial_facts_asof",
        lambda **kwargs: {
            "ticker": str(kwargs.get("ticker") or "").upper(),
            "status": "UNKNOWN",
            "shares_status": "UNKNOWN",
            "shares_reason": "CIK_MISSING",
            "shares_value": "UNKNOWN",
            "cfo_status": "UNKNOWN",
            "cfo_reason": "CIK_MISSING",
            "cfo_value": "UNKNOWN",
            "capex_status": "UNKNOWN",
            "capex_reason": "CIK_MISSING",
            "capex_value": "UNKNOWN",
            "fcf_status": "UNKNOWN",
            "fcf_reason": "CIK_MISSING",
            "fcf_value": "UNKNOWN",
            "fetch_reason_code": "CIK_MISSING",
            "cache_path": None,
            "derived_from": [f"facts.{str(kwargs.get('ticker') or '').upper()}"],
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.resolve_net_debt_proxy",
        lambda **kwargs: {
            "ticker": str(kwargs.get("ticker") or "").upper(),
            "status": "UNKNOWN",
            "reason_code": "NO_FACTS",
            "net_debt_proxy": "UNKNOWN",
            "total_debt": {"value": "UNKNOWN"},
            "cash_equivalents": {"value": "UNKNOWN"},
            "derived_from": [f"netdebt.{str(kwargs.get('ticker') or '').upper()}"],
        },
    )
    run_universe_scout(run_id="queue_det", as_of_date="2026-02-14", top_n=4, tickers=["AAA", "BBB", "CCC", "DDD"])
    first = json.loads((cfg.outputs_dir / "universe" / "queue_det" / "depth_queue.json").read_text(encoding="utf-8"))
    run_universe_scout(
        run_id="queue_det",
        as_of_date="2026-02-14",
        top_n=4,
        tickers=["AAA", "BBB", "CCC", "DDD"],
        force_restart=True,
    )
    second = json.loads((cfg.outputs_dir / "universe" / "queue_det" / "depth_queue.json").read_text(encoding="utf-8"))
    assert first["entries"] == second["entries"]
    commands_payload = universe_depth_queue_to_runs(run_id="queue_det", max_runs=3)
    assert commands_payload["status"] == "OK"
    assert commands_payload["selected_count"] >= 1
    assert all(cmd.startswith(".venv/bin/python -m app.cli sector-rlm --mode depth") for cmd in commands_payload["commands"])
