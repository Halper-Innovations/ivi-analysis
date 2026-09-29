from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.memo_pack import (
    _decision_header_lines,
    _memo_markdown,
    diff_watchlist_states,
    open_investment_memo_pack,
    open_watchlist_state,
    write_investment_memo_pack,
    write_watchlist_state,
)
from app.watchlist.contract import WatchlistEntry


runner = CliRunner()


def test_v2_screen_row_memo_decision_header_is_research_only(monkeypatch):
    entry = WatchlistEntry(
        ticker="AAA",
        status="ACTIVE",
        conviction_grade="WATCHLIST_ONLY",
        confidence="LOW",
        conviction_source="sector_screen",
        buy_price_target=75.0,
        current_price_at_addition=90.0,
        source_run_id="autonomous_sector_v2_screen_memo",
        source_sector="energy",
        added_at="2026-07-16T12:00:00Z",
        pipeline_version="v2",
        candidate_disposition="READY_FOR_UNDERWRITING",
        decision_basis="SCREEN",
    )
    monkeypatch.setattr(
        "app.universe.memo_pack._watchlist_entry_for", lambda _ticker: entry
    )

    rendered = "\n".join(
        _decision_header_lines(
            {"ticker": "AAA"},
            {
                "price_claim": {"value": 90.0},
                "intrinsic_per_share_base": 100.0,
                "value_gate_status": "PASS",
            },
        )
    )

    assert "- ACTION: Research only" in rendered
    assert "BUY PRICE TARGET" not in rendered
    assert "% TO TARGET" not in rendered
    assert "$75.00" not in rendered
    assert "Wait for $" not in rendered
    assert "AT TARGET" not in rendered


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker,cik,name\nAAA,1,AAA\nBBB,2,BBB\nCCC,3,CCC\n", encoding="utf-8")
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


def _score_row(ticker: str, *, implied, status_price: str, status_valuation: str) -> dict:
    metric_values = {
        "implied_return_base": implied,
        "intrinsic_per_share_base": 100.0 if isinstance(implied, (int, float)) else "UNKNOWN",
        "mos_epv": 0.35 if ticker == "AAA" else ("UNKNOWN" if ticker == "BBB" else -0.10),
        "mos_netnet": 0.12 if ticker == "AAA" else ("UNKNOWN" if ticker == "BBB" else 0.01),
        "epv_per_share": 140.0 if ticker == "AAA" else "UNKNOWN",
        "netnet_per_share": 40.0 if ticker == "AAA" else "UNKNOWN",
        "owner_earnings_yield_ev_3y": 0.08 if ticker == "AAA" else ("UNKNOWN" if ticker == "BBB" else 0.01),
        "fcf_yield_ev_3y": 0.06 if ticker == "AAA" else 0.03,
        "owner_earnings_yield_3y": 0.07 if ticker == "AAA" else "UNKNOWN",
        "fcf_yield_3y": 0.05 if ticker == "AAA" else 0.02,
        "revenue_cagr_5y": 0.12 if ticker == "AAA" else 0.04,
        "revenue_cagr_10y": 0.10 if ticker == "AAA" else 0.03,
        "operating_margin_trend_slope": 0.01,
        "gross_margin_trend_slope": 0.008,
        "fcf_margin_trend_slope": 0.004,
        "roic_proxy": 0.15 if ticker == "AAA" else 0.08,
        "dilution_rate_shares_cagr": 0.01 if ticker == "AAA" else 0.07,
        "net_debt_proxy": 500.0 if ticker == "AAA" else 1200.0,
        "risk_factor_keyword_delta": 1.0 if ticker == "AAA" else 3.0,
        "quality_score": 17.0 if ticker == "AAA" else 10.0,
        "risk_penalty": -2.0 if ticker == "AAA" else -6.0,
        "price_status": status_price,
        "valuation_status": status_valuation,
    }
    metric_traces = {key: {"derived_from": [f"trace.{ticker}.{key}"]} for key in metric_values}
    return {
        "ticker": ticker,
        "metric_values": metric_values,
        "metric_traces": metric_traces,
    }


def _coverage_row(
    ticker: str,
    *,
    price_status: str,
    valuation_status: str,
    valuation_reason: str,
) -> dict:
    return {
        "ticker": ticker,
        "price_status": price_status,
        "price_reason_code": "OK" if price_status == "OK" else "PRICE_UNKNOWN",
        "price_source_resolution": "stooq_cache",
        "shares_status": "OK",
        "shares_reason_code": "OK",
        "fcf_status": "OK",
        "fcf_reason_code": "OK",
        "valuation_status": valuation_status,
        "valuation_reason_code": valuation_reason,
        "intrinsic_per_share_base": 100.0 if valuation_status == "OK" else "UNKNOWN",
        "implied_return_base": 0.20 if valuation_status == "OK" else "UNKNOWN",
        "derived_from": [f"coverage.{ticker}"],
    }


def _setup_fixture(cfg, *, universe_run_id: str, batch_run_id: str) -> None:
    run_id = f"{batch_run_id}__Software__001"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "peer_scoreboard.json",
        {
            "run_id": run_id,
            "rows": [
                _score_row("AAA", implied=0.30, status_price="OK", status_valuation="OK"),
                _score_row("BBB", implied="UNKNOWN", status_price="UNKNOWN", status_valuation="UNKNOWN"),
                _score_row("CCC", implied=0.05, status_price="OK", status_valuation="OK"),
            ],
        },
    )
    _write_json(
        run_dir / "value_gates.json",
        {
            "run_id": run_id,
            "entries": [
                {
                    "ticker": "AAA",
                    "gate_status": "PASS",
                    "gate_reasons": ["MOS_PASS", "YIELD_PASS"],
                    "primary_blocker": "NONE",
                    "inputs_used": {
                        "current_price": {"value": 100.0, "derived_from": ["price.AAA"]},
                        "net_debt_proxy": {"value": 500.0, "derived_from": ["netdebt.AAA"]},
                        "dilution_rate_shares_cagr": {"value": 0.01, "derived_from": ["dilution.AAA"]},
                    },
                    "net_debt_to_cfo": 1.8,
                },
                {
                    "ticker": "BBB",
                    "gate_status": "WATCH",
                    "gate_reasons": ["PRICE_UNKNOWN"],
                    "primary_blocker": "PRICE_UNKNOWN",
                    "inputs_used": {
                        "current_price": {"value": "UNKNOWN", "derived_from": ["price.BBB"]},
                        "net_debt_proxy": {"value": "UNKNOWN", "derived_from": ["netdebt.BBB"]},
                        "dilution_rate_shares_cagr": {"value": 0.07, "derived_from": ["dilution.BBB"]},
                    },
                    "net_debt_to_cfo": "UNKNOWN",
                },
                {
                    "ticker": "CCC",
                    "gate_status": "FAIL",
                    "gate_reasons": ["LOW_YIELD_OWNER_EARNINGS_EV"],
                    "primary_blocker": "LOW_YIELD_OWNER_EARNINGS_EV",
                    "inputs_used": {
                        "current_price": {"value": 55.0, "derived_from": ["price.CCC"]},
                        "net_debt_proxy": {"value": 1200.0, "derived_from": ["netdebt.CCC"]},
                        "dilution_rate_shares_cagr": {"value": 0.07, "derived_from": ["dilution.CCC"]},
                    },
                    "net_debt_to_cfo": 3.8,
                },
            ],
        },
    )
    _write_json(
        run_dir / "valuation_coverage.json",
        {
            "run_id": run_id,
            "entries": [
                _coverage_row("AAA", price_status="OK", valuation_status="OK", valuation_reason="OK"),
                _coverage_row("BBB", price_status="UNKNOWN", valuation_status="UNKNOWN", valuation_reason="PRICE_UNKNOWN"),
                _coverage_row("CCC", price_status="OK", valuation_status="OK", valuation_reason="OK"),
            ],
        },
    )
    _write_json(
        run_dir / "shares_coverage.json",
        {"run_id": run_id, "entries": [{"ticker": "AAA", "shares_status": "OK", "shares_reason_code": "OK"}]},
    )
    _write_json(
        run_dir / "fcf_coverage.json",
        {"run_id": run_id, "entries": [{"ticker": "AAA", "fcf_status": "OK", "fcf_reason_code": "OK"}]},
    )
    _write_json(
        run_dir / "facts_coverage.json",
        {
            "run_id": run_id,
            "entries": [
                {"ticker": "AAA", "status": "OK", "fetch_reason_code": "OK", "derived_from": ["facts.AAA"]},
                {"ticker": "BBB", "status": "UNKNOWN", "fetch_reason_code": "NO_FACTS", "derived_from": ["facts.BBB"]},
                {"ticker": "CCC", "status": "OK", "fetch_reason_code": "OK", "derived_from": ["facts.CCC"]},
            ],
        },
    )

    shortlist_path = cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "global_shortlist.json"
    _write_json(
        shortlist_path,
        {
            "universe_run_id": universe_run_id,
            "batch_run_id": batch_run_id,
            "policy": "value_first",
            "rows": [
                {
                    "ticker": "AAA",
                    "run_id": run_id,
                    "sector": "Software",
                    "as_of_date": "2026-02-14",
                    "source_depth_runs": [{"run_id": run_id, "sector": "Software", "as_of_date": "2026-02-14"}],
                    "value_gate_status": "PASS",
                    "value_gate_reasons": ["MOS_PASS", "YIELD_PASS"],
                    "primary_blocker": "NONE",
                    "implied_return_base": 0.30,
                    "implied_return_base_derived_from": ["trace.AAA.implied_return_base"],
                    "intrinsic_per_share_base": 100.0,
                    "intrinsic_per_share_base_derived_from": ["trace.AAA.intrinsic_per_share_base"],
                    "mos_epv": 0.35,
                    "mos_epv_derived_from": ["trace.AAA.mos_epv"],
                    "mos_netnet": 0.12,
                    "mos_netnet_derived_from": ["trace.AAA.mos_netnet"],
                    "owner_earnings_yield_ev_3y": 0.08,
                    "owner_earnings_yield_ev_3y_derived_from": ["trace.AAA.owner_earnings_yield_ev_3y"],
                    "fcf_yield_ev_3y": 0.06,
                    "fcf_yield_ev_3y_derived_from": ["trace.AAA.fcf_yield_ev_3y"],
                    "yield_metric_used": "owner_earnings_yield_ev_3y",
                    "yield_denominator_used": "EV",
                    "yield_reason_code": "OK",
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
                    "quality_score": 17.0,
                    "risk_penalty": -2.0,
                    "derived_from": ["global.AAA"],
                },
                {
                    "ticker": "BBB",
                    "run_id": run_id,
                    "sector": "Software",
                    "as_of_date": "2026-02-14",
                    "source_depth_runs": [{"run_id": run_id, "sector": "Software", "as_of_date": "2026-02-14"}],
                    "value_gate_status": "WATCH",
                    "value_gate_reasons": ["PRICE_UNKNOWN"],
                    "primary_blocker": "PRICE_UNKNOWN",
                    "implied_return_base": "UNKNOWN",
                    "implied_return_base_derived_from": ["trace.BBB.implied_return_base"],
                    "intrinsic_per_share_base": "UNKNOWN",
                    "intrinsic_per_share_base_derived_from": ["trace.BBB.intrinsic_per_share_base"],
                    "mos_epv": "UNKNOWN",
                    "mos_epv_derived_from": ["trace.BBB.mos_epv"],
                    "mos_netnet": "UNKNOWN",
                    "mos_netnet_derived_from": ["trace.BBB.mos_netnet"],
                    "owner_earnings_yield_ev_3y": "UNKNOWN",
                    "owner_earnings_yield_ev_3y_derived_from": ["trace.BBB.owner_earnings_yield_ev_3y"],
                    "fcf_yield_ev_3y": 0.03,
                    "fcf_yield_ev_3y_derived_from": ["trace.BBB.fcf_yield_ev_3y"],
                    "yield_metric_used": "fcf_yield_ev_3y",
                    "yield_denominator_used": "MARKET_CAP",
                    "yield_reason_code": "MISSING_OWNER_EARNINGS",
                    "price_status": "UNKNOWN",
                    "valuation_status": "UNKNOWN",
                    "shares_status": "OK",
                    "fcf_status": "OK",
                    "facts_status": "UNKNOWN",
                    "price_reason_code": "PRICE_UNKNOWN",
                    "valuation_reason_code": "PRICE_UNKNOWN",
                    "shares_reason_code": "OK",
                    "fcf_reason_code": "OK",
                    "facts_reason_code": "NO_FACTS",
                    "quality_score": 10.0,
                    "risk_penalty": -4.0,
                    "derived_from": ["global.BBB"],
                },
                {
                    "ticker": "CCC",
                    "run_id": run_id,
                    "sector": "Software",
                    "as_of_date": "2026-02-14",
                    "source_depth_runs": [{"run_id": run_id, "sector": "Software", "as_of_date": "2026-02-14"}],
                    "value_gate_status": "FAIL",
                    "value_gate_reasons": ["LOW_YIELD_OWNER_EARNINGS_EV"],
                    "primary_blocker": "LOW_YIELD_OWNER_EARNINGS_EV",
                    "implied_return_base": 0.05,
                    "implied_return_base_derived_from": ["trace.CCC.implied_return_base"],
                    "intrinsic_per_share_base": 60.0,
                    "intrinsic_per_share_base_derived_from": ["trace.CCC.intrinsic_per_share_base"],
                    "mos_epv": -0.10,
                    "mos_epv_derived_from": ["trace.CCC.mos_epv"],
                    "mos_netnet": 0.01,
                    "mos_netnet_derived_from": ["trace.CCC.mos_netnet"],
                    "owner_earnings_yield_ev_3y": 0.01,
                    "owner_earnings_yield_ev_3y_derived_from": ["trace.CCC.owner_earnings_yield_ev_3y"],
                    "fcf_yield_ev_3y": "UNKNOWN",
                    "fcf_yield_ev_3y_derived_from": ["trace.CCC.fcf_yield_ev_3y"],
                    "yield_metric_used": "owner_earnings_yield_ev_3y",
                    "yield_denominator_used": "EV",
                    "yield_reason_code": "LOW_YIELD_OWNER_EARNINGS_EV",
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
                    "quality_score": 9.0,
                    "risk_penalty": -6.0,
                    "derived_from": ["global.CCC"],
                },
            ],
        },
    )


def _assert_numeric_claims_have_refs(payload: Any) -> None:
    if isinstance(payload, dict):
        if "value" in payload and "derived_from" in payload:
            if isinstance(payload.get("value"), (int, float)):
                refs = payload.get("derived_from")
                assert isinstance(refs, list)
                assert refs
        for value in payload.values():
            _assert_numeric_claims_have_refs(value)
    elif isinstance(payload, list):
        for item in payload:
            _assert_numeric_claims_have_refs(item)


def test_memo_pack_outputs_are_deterministic_and_traceable(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    universe_run_id = "u_memo"
    batch_run_id = "b_memo"
    _setup_fixture(cfg, universe_run_id=universe_run_id, batch_run_id=batch_run_id)

    monkeypatch.setattr(
        "app.universe.memo_pack.write_depth_batch_rollup",
        lambda **_kwargs: {"status": "OK"},
    )

    payload = write_investment_memo_pack(
        universe_run_id=universe_run_id,
        batch_run_id=batch_run_id,
        top_n=3,
        policy="value_first",
    )
    assert payload["status"] == "OK"
    assert payload["memo_count"] == 3

    open_payload = open_investment_memo_pack(universe_run_id=universe_run_id, batch_run_id=batch_run_id)
    assert open_payload["status"] == "OK"
    assert open_payload["memo_count"] == 3

    pack_dir = cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "memo_pack"
    md_path = pack_dir / "memos" / "BBB.md"
    json_path = pack_dir / "memos" / "AAA.json"
    assert md_path.exists()
    assert json_path.exists()

    memo = json.loads(json_path.read_text(encoding="utf-8"))
    _assert_numeric_claims_have_refs(memo)

    md = md_path.read_text(encoding="utf-8")
    headings = [
        "## Header",
        "## Decision Snapshot",
        "## Graham/Dodd",
        "## Owner Earnings and Yield",
        "## Fundamentals Highlights",
        "## Risk Section",
        "## Unknowns and Blockers",
        "## Derived From Index",
    ]
    indices = [md.index(token) for token in headings]
    assert indices == sorted(indices)
    assert "PRICE_UNKNOWN" in md


def test_watchlist_state_and_diff_are_stable(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    prev_run_id = "u_prev"
    cur_run_id = "u_cur"

    prev_manifest = {"batch_run_id": "b_prev"}
    prev_state_payload = write_watchlist_state(
        prev_run_id,
        [
            {"ticker": "AAA", "rank_global": 1, "value_gate_status": "WATCH", "implied_return_base": 0.10, "primary_blocker": "PRICE_UNKNOWN"},
            {"ticker": "BBB", "rank_global": 2, "value_gate_status": "PASS", "implied_return_base": 0.30, "primary_blocker": "NONE"},
        ],
        prev_manifest,
        cfg.outputs_dir / "universe" / prev_run_id / "autopilot" / "watchlist_state.json",
    )
    prev_state_path = Path(prev_state_payload["watchlist_state_path"])
    assert prev_state_path.exists()

    cur_manifest = {"batch_run_id": "b_cur"}
    cur_state_payload = write_watchlist_state(
        cur_run_id,
        [
            {"ticker": "AAA", "rank_global": 1, "value_gate_status": "FAIL", "implied_return_base": 0.02, "primary_blocker": "LOW_YIELD_OWNER_EARNINGS_EV"},
            {"ticker": "CCC", "rank_global": 2, "value_gate_status": "WATCH", "implied_return_base": "UNKNOWN", "primary_blocker": "MISSING_EV"},
        ],
        cur_manifest,
        cfg.outputs_dir / "universe" / cur_run_id / "autopilot" / "watchlist_state.json",
        prev_state_path=prev_state_path,
    )
    assert Path(cur_state_payload["watchlist_state_path"]).exists()

    diff_payload = diff_watchlist_states(prev_run_id=prev_run_id, run_id=cur_run_id)
    assert diff_payload["status"] == "OK"
    assert diff_payload["additions"] == ["CCC"]
    assert diff_payload["removals"] == ["BBB"]
    assert diff_payload["downgrades"] == [{"ticker": "AAA", "from": "WATCH", "to": "FAIL"}]

    open_state = open_watchlist_state(run_id=cur_run_id)
    assert open_state["status"] == "OK"
    assert open_state["ticker_count"] == 2


def test_memo_pack_cli_and_watchlist_diff(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    universe_run_id = "u_memo_cli"
    batch_run_id = "b_memo_cli"
    _setup_fixture(cfg, universe_run_id=universe_run_id, batch_run_id=batch_run_id)
    monkeypatch.setattr(
        "app.universe.memo_pack.write_depth_batch_rollup",
        lambda **_kwargs: {"status": "OK"},
    )

    build_cmd = runner.invoke(
        app,
        [
            "universe-memo-pack",
            "--universe-run-id",
            universe_run_id,
            "--batch-run-id",
            batch_run_id,
            "--top-n",
            "3",
            "--policy",
            "value_first",
        ],
    )
    assert build_cmd.exit_code == 0, build_cmd.output
    build_payload = json.loads(build_cmd.output)
    assert build_payload["status"] == "OK"
    assert build_payload["memo_count"] == 3

    open_cmd = runner.invoke(
        app,
        [
            "universe-memo-pack-open",
            "--universe-run-id",
            universe_run_id,
            "--batch-run-id",
            batch_run_id,
        ],
    )
    assert open_cmd.exit_code == 0, open_cmd.output
    open_payload = json.loads(open_cmd.output)
    assert open_payload["status"] == "OK"
    assert len(open_payload["top_10"]) == 3

    state_cmd = runner.invoke(app, ["universe-watchlist-state-open", "--run-id", universe_run_id])
    assert state_cmd.exit_code == 0, state_cmd.output
    state_payload = json.loads(state_cmd.output)
    assert state_payload["status"] == "OK"

    diff_cmd = runner.invoke(
        app,
        [
            "universe-watchlist-diff",
            "--prev-run-id",
            universe_run_id,
            "--run-id",
            universe_run_id,
        ],
    )
    assert diff_cmd.exit_code == 0, diff_cmd.output
    diff_payload = json.loads(diff_cmd.output)
    assert diff_payload["status"] == "OK"
    assert diff_payload["summary"]["addition_count"] == 0


def test_write_watchlist_state_keeps_bak_of_prior(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "u_bak"
    out_path = cfg.outputs_dir / "universe" / run_id / "autopilot" / "watchlist_state.json"

    first = write_watchlist_state(
        run_id,
        [{"ticker": "AAA", "rank_global": 1, "value_gate_status": "WATCH", "implied_return_base": 0.10, "primary_blocker": "NONE"}],
        {"batch_run_id": "b1"},
        out_path,
    )
    first_path = Path(first["watchlist_state_path"])
    assert first_path.exists()
    bak_path = first_path.with_suffix(first_path.suffix + ".bak")
    assert not bak_path.exists()

    # Capture the exact prior content to assert the .bak preserves it verbatim.
    prior_text = first_path.read_text(encoding="utf-8")
    prior_ticker_count = json.loads(prior_text)["ticker_count"]
    assert prior_ticker_count == 1

    write_watchlist_state(
        run_id,
        [
            {"ticker": "AAA", "rank_global": 1, "value_gate_status": "PASS", "implied_return_base": 0.20, "primary_blocker": "NONE"},
            {"ticker": "BBB", "rank_global": 2, "value_gate_status": "PASS", "implied_return_base": 0.25, "primary_blocker": "NONE"},
        ],
        {"batch_run_id": "b2"},
        out_path,
        prev_state_path=first_path,
    )

    assert bak_path.exists()
    assert bak_path.read_text(encoding="utf-8") == prior_text
    assert json.loads(bak_path.read_text(encoding="utf-8"))["ticker_count"] == 1
    assert json.loads(first_path.read_text(encoding="utf-8"))["ticker_count"] == 2


def test_write_watchlist_state_leaves_no_tmp_file(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "u_atomic"
    out_path = cfg.outputs_dir / "universe" / run_id / "autopilot" / "watchlist_state.json"
    write_watchlist_state(
        run_id,
        [{"ticker": "AAA", "rank_global": 1, "value_gate_status": "WATCH", "implied_return_base": 0.10, "primary_blocker": "NONE"}],
        {"batch_run_id": "b1"},
        out_path,
    )
    tmp_siblings = [p.name for p in out_path.parent.iterdir() if p.name.endswith(".tmp")]
    assert tmp_siblings == []


# --- decision-first header in the sector memo pack ----------------------


def _memo_for(ticker: str, *, value_gate_status: str, current_price, intrinsic) -> dict:
    return {
        "header": {"ticker": ticker},
        "decision_snapshot": {
            "value_gate_status": value_gate_status,
            "price_claim": {"value": current_price},
            "intrinsic_per_share_base": intrinsic,
        },
    }


def _entry(ticker: str, *, status: str, grade: str, buy_price_target: float) -> WatchlistEntry:
    return WatchlistEntry(
        ticker=ticker,
        status=status,
        source_run_id="run",
        added_at="2026-01-01T00:00:00Z",
        conviction_grade=grade,
        confidence="HIGH",
        buy_price_target=buy_price_target,
    )


def test_memo_decision_block_deploy_ready_renders_at_target_review(monkeypatch):
    entry = _entry("CRTO", status="DEPLOY_READY", grade="ACTIONABLE", buy_price_target=32.19)
    monkeypatch.setattr("app.watchlist.store.get_latest", lambda ticker, **_: entry)

    md = _memo_markdown(_memo_for("CRTO", value_gate_status="PASS", current_price=18.33, intrinsic=100.0))

    decision_index = md.index("## Decision")
    snapshot_index = md.index("## Decision Snapshot")
    block = md[decision_index:snapshot_index]
    assert "- ACTION: Review at target" in block.splitlines()
    assert "AT TARGET CRTO: $18.33 <= target $32.19" in block


def test_memo_decision_block_not_on_watchlist_uses_intrinsic_fallback(monkeypatch):
    monkeypatch.setattr("app.watchlist.store.get_latest", lambda ticker, **_: None)

    md = _memo_markdown(_memo_for("AAA", value_gate_status="WATCH", current_price=150.0, intrinsic=100.0))

    decision_index = md.index("## Decision")
    snapshot_index = md.index("## Decision Snapshot")
    block = md[decision_index:snapshot_index]
    # intrinsic 100.0 * 0.75 == 75.0 fallback target
    assert "- ACTION: Wait for $75.00" in block.splitlines()


def test_memo_decision_block_precedes_decision_snapshot(monkeypatch):
    monkeypatch.setattr("app.watchlist.store.get_latest", lambda ticker, **_: None)

    md = _memo_markdown(_memo_for("AAA", value_gate_status="WATCH", current_price=150.0, intrinsic=100.0))

    assert md.index("## Decision") < md.index("## Decision Snapshot")


def test_memo_decision_block_reconciliation_note_on_grade_divergence(monkeypatch):
    entry = _entry("CRTO", status="ACTIVE", grade="WATCHLIST_ONLY", buy_price_target=75.0)
    monkeypatch.setattr("app.watchlist.store.get_latest", lambda ticker, **_: entry)

    md = _memo_markdown(_memo_for("CRTO", value_gate_status="PASS", current_price=150.0, intrinsic=100.0))

    decision_index = md.index("## Decision")
    snapshot_index = md.index("## Decision Snapshot")
    block = md[decision_index:snapshot_index]
    note_lines = [line for line in block.splitlines() if line.startswith("- NOTE:")]
    assert note_lines
    assert "grade" in note_lines[0]
