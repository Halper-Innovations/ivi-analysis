from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.dossier_pack import open_dossier_pack, write_dossier_pack


runner = CliRunner()


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


def _write_fake_global_shortlist(cfg, *, universe_run_id: str, batch_run_id: str) -> Path:
    batch_dir = cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id
    shortlist_path = batch_dir / "global_shortlist.json"
    rows = [
        {
            "ticker": "AAA",
            "source_depth_runs": [{"run_id": "r1", "sector": "Software", "as_of_date": "2026-02-14"}],
            "value_gate_status": "PASS",
            "value_gate_reasons": ["BALANCE_PASS"],
            "primary_blocker": "NONE",
            "primary_blocker_reason_code": "NONE",
            "implied_return_base": 0.42,
            "intrinsic_per_share_base": 150.0,
            "mos_epv": 0.55,
            "mos_netnet": 0.10,
            "owner_earnings_yield_ev_3y": 0.08,
            "fcf_yield_ev_3y": 0.06,
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
            "quality_score": 18.0,
            "risk_penalty": -2.0,
            "notes": "PASS | implied_return_base=0.42 | blocker=NONE",
            "derived_from": ["trace.AAA.1", "trace.AAA.2"],
        },
        {
            "ticker": "BBB",
            "source_depth_runs": [{"run_id": "r2", "sector": "Software", "as_of_date": "2026-02-14"}],
            "value_gate_status": "WATCH",
            "value_gate_reasons": ["PRICE_UNKNOWN"],
            "primary_blocker": "PRICE_UNKNOWN",
            "primary_blocker_reason_code": "PRICE_UNKNOWN",
            "implied_return_base": "UNKNOWN",
            "intrinsic_per_share_base": "UNKNOWN",
            "mos_epv": "UNKNOWN",
            "mos_netnet": 0.05,
            "owner_earnings_yield_ev_3y": "UNKNOWN",
            "fcf_yield_ev_3y": 0.03,
            "yield_metric_used": "fcf_yield_ev_3y",
            "yield_denominator_used": "MARKET_CAP",
            "yield_reason_code": "MISSING_OWNER_EARNINGS",
            "price_status": "UNKNOWN",
            "valuation_status": "UNKNOWN",
            "shares_status": "OK",
            "fcf_status": "OK",
            "facts_status": "OK",
            "price_reason_code": "PRICE_UNKNOWN",
            "valuation_reason_code": "MODEL_PRECONDITION_FAILED",
            "shares_reason_code": "OK",
            "fcf_reason_code": "OK",
            "facts_reason_code": "OK",
            "quality_score": 12.0,
            "risk_penalty": -1.0,
            "notes": "WATCH | implied_return_base=UNKNOWN | blocker=PRICE_UNKNOWN",
            "derived_from": ["trace.BBB.1"],
        },
        {
            "ticker": "CCC",
            "source_depth_runs": [{"run_id": "r3", "sector": "Healthcare", "as_of_date": "2026-02-14"}],
            "value_gate_status": "FAIL",
            "value_gate_reasons": ["LOW_YIELD"],
            "primary_blocker": "LOW_YIELD",
            "primary_blocker_reason_code": "LOW_YIELD",
            "implied_return_base": 0.05,
            "intrinsic_per_share_base": 22.0,
            "mos_epv": -0.10,
            "mos_netnet": "UNKNOWN",
            "owner_earnings_yield_ev_3y": 0.01,
            "fcf_yield_ev_3y": "UNKNOWN",
            "yield_metric_used": "owner_earnings_yield_ev_3y",
            "yield_denominator_used": "EV",
            "yield_reason_code": "LOW_YIELD",
            "price_status": "OK",
            "valuation_status": "OK",
            "shares_status": "UNKNOWN",
            "fcf_status": "UNKNOWN",
            "facts_status": "UNKNOWN",
            "price_reason_code": "OK",
            "valuation_reason_code": "OK",
            "shares_reason_code": "MISSING_SHARES",
            "fcf_reason_code": "MISSING_FCF",
            "facts_reason_code": "NO_FACTS",
            "quality_score": 8.0,
            "risk_penalty": -4.0,
            "notes": "FAIL | implied_return_base=0.05 | blocker=LOW_YIELD",
            "derived_from": ["trace.CCC.1"],
        },
    ]
    _write_json(
        shortlist_path,
        {
            "universe_run_id": universe_run_id,
            "batch_run_id": batch_run_id,
            "policy": "value_first",
            "top_n": 10,
            "candidate_count_total": 3,
            "candidate_count_selected": 3,
            "rows": rows,
        },
    )
    return shortlist_path


def test_dossier_pack_outputs_and_ordering(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    universe_run_id = "u_dossier"
    batch_run_id = "b_dossier"
    _write_fake_global_shortlist(cfg, universe_run_id=universe_run_id, batch_run_id=batch_run_id)

    monkeypatch.setattr(
        "app.universe.dossier_pack.write_depth_batch_rollup",
        lambda **_kwargs: {"status": "OK"},
    )

    payload = write_dossier_pack(
        universe_run_id=universe_run_id,
        batch_run_id=batch_run_id,
        top_n=3,
        policy="value_first",
    )
    assert payload["status"] == "OK"
    pack_dir = cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "dossier_pack"
    manifest_path = pack_dir / "dossier_pack_manifest.json"
    watchlist_csv = pack_dir / "watchlist.csv"
    watchlist_json = pack_dir / "watchlist.json"
    assert manifest_path.exists()
    assert watchlist_csv.exists()
    assert watchlist_json.exists()

    with watchlist_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == [
        "ticker",
        "rank",
        "value_gate_status",
        "implied_return_base",
        "mos_epv",
        "mos_netnet",
        "yield_metric_used",
        "yield_denominator_used",
        "owner_earnings_yield_ev_3y",
        "fcf_yield_ev_3y",
        "price_status",
        "valuation_status",
        "primary_blocker",
    ]
    assert [row["ticker"] for row in rows] == ["AAA", "BBB", "CCC"]
    assert [row["rank"] for row in rows] == ["1", "2", "3"]

    packet_json_path = pack_dir / "candidates" / "BBB" / "packet.json"
    packet_md_path = pack_dir / "candidates" / "BBB" / "packet.md"
    assert packet_json_path.exists()
    assert packet_md_path.exists()
    packet = json.loads(packet_json_path.read_text(encoding="utf-8"))
    assert packet["derived_from"] == ["trace.BBB.1"]
    md = packet_md_path.read_text(encoding="utf-8")
    assert "## Blockers / Unknowns" in md
    assert "implied_return_base" in md
    assert "price_status" in md


def test_dossier_pack_cli_open(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    universe_run_id = "u_dossier_cli"
    batch_run_id = "b_dossier_cli"
    _write_fake_global_shortlist(cfg, universe_run_id=universe_run_id, batch_run_id=batch_run_id)
    monkeypatch.setattr(
        "app.universe.dossier_pack.write_depth_batch_rollup",
        lambda **_kwargs: {"status": "OK"},
    )

    build_cmd = runner.invoke(
        app,
        [
            "universe-dossier-pack",
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

    open_cmd = runner.invoke(
        app,
        [
            "universe-dossier-pack-open",
            "--universe-run-id",
            universe_run_id,
            "--batch-run-id",
            batch_run_id,
        ],
    )
    assert open_cmd.exit_code == 0, open_cmd.output
    open_payload = json.loads(open_cmd.output)
    assert open_payload["status"] == "OK"
    assert open_payload["candidate_count"] == 3
    assert len(open_payload["top_10"]) == 3
    assert "unknown_counts" in open_payload

    open_fn_payload = open_dossier_pack(universe_run_id=universe_run_id, batch_run_id=batch_run_id)
    assert open_fn_payload["status"] == "OK"
    assert open_fn_payload["candidate_count"] == 3
