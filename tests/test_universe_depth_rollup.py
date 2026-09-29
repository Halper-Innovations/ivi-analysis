from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.depth_rollup import open_depth_batch_rollup, write_depth_batch_rollup


runner = CliRunner()


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker,cik,name\nAAA,1,AAA\n", encoding="utf-8")
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


def _score_row(
    ticker: str,
    *,
    implied_return,
    mos_epv,
    mos_netnet,
    owner_yield,
    fcf_yield,
    quality,
    risk,
):
    metric_values = {
        "implied_return_base": implied_return,
        "intrinsic_per_share_base": 123.0 if isinstance(implied_return, (int, float)) else "UNKNOWN",
        "mos_epv": mos_epv,
        "mos_netnet": mos_netnet,
        "owner_earnings_yield_ev_3y": owner_yield,
        "fcf_yield_ev_3y": fcf_yield,
        "quality_score": quality,
        "risk_penalty": risk,
    }
    metric_traces = {}
    for key in metric_values:
        metric_traces[key] = {"derived_from": [f"trace.{ticker}.{key}"]}
    return {
        "ticker": ticker,
        "overall_rank": 1,
        "overall_score": 10.0,
        "metric_values": metric_values,
        "metric_traces": metric_traces,
    }


def _write_run_artifacts(
    cfg,
    *,
    run_id: str,
    score_rows: list[dict],
    gate_rows: list[dict],
    valuation_entries: list[dict],
    shares_entries: list[dict],
    fcf_entries: list[dict],
    facts_entries: list[dict],
) -> Path:
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "peer_scoreboard.json",
        {"run_id": run_id, "rows": score_rows},
    )
    _write_json(
        run_dir / "value_gates.json",
        {
            "run_id": run_id,
            "summary": {
                "counts": {
                    "PASS": len([r for r in gate_rows if str(r.get("gate_status")).upper() == "PASS"]),
                    "WATCH": len([r for r in gate_rows if str(r.get("gate_status")).upper() == "WATCH"]),
                    "FAIL": len([r for r in gate_rows if str(r.get("gate_status")).upper() == "FAIL"]),
                }
            },
            "entries": gate_rows,
        },
    )
    _write_json(run_dir / "valuation_coverage.json", {"run_id": run_id, "entries": valuation_entries})
    _write_json(
        run_dir / "shares_coverage.json",
        {
            "run_id": run_id,
            "entries": shares_entries,
            "reason_counts": {"OK": len(shares_entries)},
        },
    )
    _write_json(
        run_dir / "fcf_coverage.json",
        {
            "run_id": run_id,
            "entries": fcf_entries,
            "reason_counts": {"OK": len(fcf_entries)},
        },
    )
    _write_json(
        run_dir / "facts_coverage.json",
        {
            "run_id": run_id,
            "entries": facts_entries,
            "reason_counts": {"shares_reason": {"OK": len(facts_entries)}},
        },
    )
    _write_json(
        run_dir / "rlm_state.json",
        {
            "run_id": run_id,
            "gate_thresholds_effective": {
                "mos_min": 0.3,
                "valuation_gap_min": 0.1,
            },
        },
    )
    return run_dir


def _gate_row(ticker: str, gate_status: str, blocker: str = "NONE") -> dict:
    return {
        "ticker": ticker,
        "gate_status": gate_status,
        "gate_reasons": [blocker] if blocker != "NONE" else ["OK"],
        "primary_blocker": blocker,
    }


def _coverage_row(ticker: str, *, price_status: str = "OK", valuation_status: str = "OK", valuation_reason: str = "OK") -> dict:
    return {
        "ticker": ticker,
        "price_status": price_status,
        "price_reason_code": "OK" if price_status == "OK" else "PRICE_UNKNOWN",
        "valuation_status": valuation_status,
        "valuation_reason_code": valuation_reason,
        "shares_status": "OK",
        "shares_reason_code": "OK",
        "fcf_status": "OK",
        "fcf_reason_code": "OK",
        "derived_from": [f"coverage.{ticker}"],
    }


def _simple_status_row(ticker: str, key: str) -> dict:
    return {key: "OK", "ticker": ticker, f"{key.replace('_status', '')}_reason_code": "OK", "derived_from": [f"{key}.{ticker}"]}


def _setup_batch_fixture(cfg):
    universe_run_id = "u_rollup"
    batch_run_id = "b_rollup"
    run1 = "depth_batch_rollup__Software__001"
    run2 = "depth_batch_rollup__Software__002"

    run1_dir = _write_run_artifacts(
        cfg,
        run_id=run1,
        score_rows=[
            _score_row("AAA", implied_return=0.5, mos_epv=0.4, mos_netnet=0.2, owner_yield=0.08, fcf_yield=0.07, quality=15, risk=-3),
            _score_row("BBB", implied_return="UNKNOWN", mos_epv=0.3, mos_netnet=0.1, owner_yield=0.05, fcf_yield=0.04, quality=12, risk=-2),
            _score_row("DDD", implied_return=0.1, mos_epv=0.05, mos_netnet=0.01, owner_yield=0.03, fcf_yield=0.02, quality=8, risk=-1),
        ],
        gate_rows=[
            _gate_row("AAA", "WATCH", "MISSING_PRICE"),
            _gate_row("BBB", "PASS"),
            _gate_row("DDD", "WATCH", "LOW_YIELD"),
        ],
        valuation_entries=[
            _coverage_row("AAA", price_status="UNKNOWN", valuation_status="UNKNOWN", valuation_reason="PRICE_UNKNOWN"),
            _coverage_row("BBB"),
            _coverage_row("DDD"),
        ],
        shares_entries=[_simple_status_row("AAA", "shares_status"), _simple_status_row("BBB", "shares_status"), _simple_status_row("DDD", "shares_status")],
        fcf_entries=[_simple_status_row("AAA", "fcf_status"), _simple_status_row("BBB", "fcf_status"), _simple_status_row("DDD", "fcf_status")],
        facts_entries=[{"ticker": "AAA", "status": "OK", "fetch_reason_code": "OK", "derived_from": ["facts.AAA"]}, {"ticker": "BBB", "status": "OK", "fetch_reason_code": "OK", "derived_from": ["facts.BBB"]}, {"ticker": "DDD", "status": "OK", "fetch_reason_code": "OK", "derived_from": ["facts.DDD"]}],
    )
    run2_dir = _write_run_artifacts(
        cfg,
        run_id=run2,
        score_rows=[
            _score_row("AAA", implied_return=0.6, mos_epv=0.45, mos_netnet=0.25, owner_yield=0.09, fcf_yield=0.08, quality=16, risk=-4),
            _score_row("CCC", implied_return=0.2, mos_epv=0.2, mos_netnet=0.05, owner_yield=0.06, fcf_yield=0.05, quality=10, risk=-2),
            _score_row("EEE", implied_return=0.1, mos_epv=0.05, mos_netnet=0.01, owner_yield=0.03, fcf_yield=0.02, quality=8, risk=-1),
        ],
        gate_rows=[
            _gate_row("AAA", "PASS"),
            _gate_row("CCC", "PASS"),
            _gate_row("EEE", "WATCH", "LOW_YIELD"),
        ],
        valuation_entries=[_coverage_row("AAA"), _coverage_row("CCC"), _coverage_row("EEE")],
        shares_entries=[_simple_status_row("AAA", "shares_status"), _simple_status_row("CCC", "shares_status"), _simple_status_row("EEE", "shares_status")],
        fcf_entries=[_simple_status_row("AAA", "fcf_status"), _simple_status_row("CCC", "fcf_status"), _simple_status_row("EEE", "fcf_status")],
        facts_entries=[{"ticker": "AAA", "status": "OK", "fetch_reason_code": "OK", "derived_from": ["facts.AAA"]}, {"ticker": "CCC", "status": "OK", "fetch_reason_code": "OK", "derived_from": ["facts.CCC"]}, {"ticker": "EEE", "status": "OK", "fetch_reason_code": "OK", "derived_from": ["facts.EEE"]}],
    )

    batch_dir = cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    state_payload = {
        "batch_run_id": batch_run_id,
        "universe_run_id": universe_run_id,
        "status": "DONE",
        "selection_policy": "QUEUE_ORDER",
        "planned_runs": [
            {
                "idx": 0,
                "run_id": run1,
                "sector": "Software",
                "as_of_date": "2026-02-14",
                "tickers": ["AAA", "BBB", "DDD"],
                "status": "DONE",
            },
            {
                "idx": 1,
                "run_id": run2,
                "sector": "Software",
                "as_of_date": "2026-02-14",
                "tickers": ["AAA", "CCC", "EEE"],
                "status": "DONE",
            },
        ],
        "completed_runs": [
            {
                "idx": 0,
                "run_id": run1,
                "status": "DONE",
                "sector": "Software",
                "tickers": ["AAA", "BBB", "DDD"],
                "artifacts_paths": {"run_dir": str(run1_dir)},
            },
            {
                "idx": 1,
                "run_id": run2,
                "status": "DONE",
                "sector": "Software",
                "tickers": ["AAA", "CCC", "EEE"],
                "artifacts_paths": {"run_dir": str(run2_dir)},
            },
        ],
        "execution_defaults": {"mode": "depth", "iterations": 1, "top_k": 5},
    }
    _write_json(batch_dir / "batch_state.json", state_payload)
    _write_json(
        batch_dir / "batch_summary.json",
        {
            "batch_run_id": batch_run_id,
            "universe_run_id": universe_run_id,
            "status": "DONE",
            "run_count_total": 2,
            "done_count": 2,
        },
    )
    return universe_run_id, batch_run_id, batch_dir


def test_depth_rollup_writes_artifacts_and_ranking(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    universe_run_id, batch_run_id, batch_dir = _setup_batch_fixture(cfg)

    payload = write_depth_batch_rollup(
        universe_run_id=universe_run_id,
        batch_run_id=batch_run_id,
        top_n=10,
        policy="value_first",
    )
    assert payload["status"] == "OK"
    shortlist_path = Path(payload["global_shortlist_json_path"])
    rollup_path = Path(payload["global_rollup_json_path"])
    md_path = Path(payload["global_shortlist_md_path"])
    assert shortlist_path.exists()
    assert rollup_path.exists()
    assert md_path.exists()

    shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
    tickers = [str(row.get("ticker") or "") for row in shortlist["rows"]]
    assert tickers[:5] == ["AAA", "CCC", "BBB", "DDD", "EEE"]
    aaa = next(row for row in shortlist["rows"] if row["ticker"] == "AAA")
    assert aaa["value_gate_status"] == "PASS"
    assert aaa["implied_return_base"] == 0.6
    assert len(aaa["source_depth_runs"]) == 2

    rollup = json.loads(rollup_path.read_text(encoding="utf-8"))
    assert rollup["run_count_total"] == 2
    assert rollup["run_count_done"] == 2
    assert rollup["candidate_count_ranked"] == 5
    assert rollup["value_gate_counts"]["PASS"] == 3
    assert rollup["coverage_breakdown"]["price_reason_counts"]["OK"] >= 4

    md = md_path.read_text(encoding="utf-8")
    assert "| Rank | Ticker | Gate |" in md
    assert "AAA" in md

    open_payload = open_depth_batch_rollup(universe_run_id, batch_run_id, top_n=3)
    assert open_payload["status"] == "OK"
    assert open_payload["run_counts"]["run_count_done"] == 2
    assert len(open_payload["top_shortlist"]) == 3


def test_depth_rollup_cli_open_summary(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    universe_run_id, batch_run_id, _batch_dir = _setup_batch_fixture(cfg)

    build_cmd = runner.invoke(
        app,
        [
            "universe-depth-batch-rollup",
            "--universe-run-id",
            universe_run_id,
            "--batch-run-id",
            batch_run_id,
            "--top-n",
            "10",
            "--policy",
            "value_first",
        ],
    )
    assert build_cmd.exit_code == 0, build_cmd.output
    build_payload = json.loads(build_cmd.output)
    assert build_payload["status"] == "OK"

    open_cmd = runner.invoke(
        app,
        [
            "universe-depth-batch-rollup-open",
            "--universe-run-id",
            universe_run_id,
            "--batch-run-id",
            batch_run_id,
        ],
    )
    assert open_cmd.exit_code == 0, open_cmd.output
    open_payload = json.loads(open_cmd.output)
    assert open_payload["status"] == "OK"
    assert "run_counts" in open_payload
    assert "candidate_counts" in open_payload
    assert "top_shortlist" in open_payload


def test_owner_earnings_hardness_policy_ranks_capital_allocation_without_crashing():
    from app.universe.depth_rollup import _sort_key_value_first_owner_earnings_hardness

    rows = [
        {"ticker": "DIL", "capital_allocation_discipline_class": "OWNER_DILUTIVE_OR_DESTRUCTIVE"},
        {"ticker": "DSC", "capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED"},
        {"ticker": "MIX", "capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION"},
    ]
    ranked = sorted(rows, key=_sort_key_value_first_owner_earnings_hardness)
    assert [row["ticker"] for row in ranked] == ["DSC", "MIX", "DIL"]
