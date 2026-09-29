from __future__ import annotations

import json
import os
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.autopilot import (
    AUTOPILOT_CANCELLED,
    AUTOPILOT_DONE,
    AUTOPILOT_PLANNED,
    cancel_universe_autopilot,
    open_universe_autopilot,
    run_universe_autopilot,
    universe_autopilot_status,
)


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _autopilot_expected_paths(cfg, run_id: str) -> dict[str, Path]:
    batch_run_id = f"{run_id}_depth_batch"
    universe_root = cfg.outputs_dir / "universe" / run_id
    sectors_root = cfg.sectors_dir / run_id
    batch_root = universe_root / "depth_batches" / batch_run_id
    dossier_root = batch_root / "dossier_pack"
    memo_root = batch_root / "memo_pack"
    autopilot_root = universe_root / "autopilot"
    filing_diff_root = universe_root / "filing_diffs"
    pattern_root = universe_root / "pattern_scan"
    variant_root = universe_root / "variant_perceptions"
    return {
        "state": autopilot_root / "autopilot_state.json",
        "summary": autopilot_root / "autopilot_summary.json",
        "universe_summary": sectors_root / "universe_summary.json",
        "depth_queue": universe_root / "depth_queue.json",
        "batch_state": batch_root / "batch_state.json",
        "batch_summary": batch_root / "batch_summary.json",
        "batch_log": batch_root / "batch_log.jsonl",
        "global_shortlist": batch_root / "global_shortlist.json",
        "global_rollup": batch_root / "global_rollup.json",
        "global_md": batch_root / "global_shortlist.md",
        "filing_diff_summary": filing_diff_root / "filing_diff_summary.json",
        "pattern_scan_report": pattern_root / "pattern_scan_report.json",
        "pattern_scan_summary": pattern_root / "pattern_scan_summary.json",
        "variant_summary": variant_root / "variant_synthesis_summary.json",
        "manifest": dossier_root / "dossier_pack_manifest.json",
        "watchlist_csv": dossier_root / "watchlist.csv",
        "watchlist_json": dossier_root / "watchlist.json",
        "memo_manifest": memo_root / "memo_pack_manifest.json",
        "watchlist_state": autopilot_root / "watchlist_state.json",
    }


def test_universe_autopilot_dry_run_planned_state(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "auto_dry"
    cmd = runner.invoke(
        app,
        [
            "universe-autopilot",
            "--run-id",
            run_id,
            "--as-of",
            "2026-02-14",
            "--dry-run",
        ],
    )
    assert cmd.exit_code == 0, cmd.output
    payload = json.loads(cmd.output)
    assert payload["status"] == AUTOPILOT_PLANNED
    assert len(payload["plan"]) == 9
    paths = _autopilot_expected_paths(cfg, run_id)
    assert paths["state"].exists()
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    assert state["status"] == AUTOPILOT_PLANNED
    assert state["depth"] == "fundamentals"
    assert state["stages"]["SCOUT"]["status"] == "NOT_STARTED"


def test_universe_autopilot_resume_skips_completed_stages(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "auto_resume_skip"
    batch_run_id = f"{run_id}_depth_batch"
    paths = _autopilot_expected_paths(cfg, run_id)

    _write_json(paths["universe_summary"], {"counts": {"PASS": 1, "WATCH": 1, "FAIL": 0}})
    _write_json(paths["depth_queue"], {"entries": [{"rank": 1, "tickers": ["AAA"], "sector_suggested": "Software"}]})
    _write_json(paths["batch_state"], {"status": "DONE", "cursor_next_idx": 1, "planned_runs": [{}], "completed_runs": [{}]})
    _write_json(paths["batch_summary"], {"status": "DONE", "total_planned": 1, "done_count": 1})
    paths["batch_log"].parent.mkdir(parents=True, exist_ok=True)
    paths["batch_log"].write_text("", encoding="utf-8")
    _write_json(paths["global_shortlist"], {"rows": [{"ticker": "AAA", "implied_return_base": 0.2, "mos_epv": 0.3, "yield_metric_used": "owner_earnings_yield_ev_3y"}]})
    _write_json(paths["global_rollup"], {"run_count_total": 1, "run_count_done": 1, "run_count_failed": 0, "run_count_cancelled": 0, "candidate_count_total": 1, "candidate_count_ranked": 1, "coverage_breakdown": {}})
    paths["global_md"].write_text("# md\n", encoding="utf-8")
    _write_json(paths["manifest"], {"watchlist_csv_path": str(paths["watchlist_csv"]), "watchlist_json_path": str(paths["watchlist_json"]), "candidate_count": 1, "unknown_counts": {}})
    paths["watchlist_csv"].parent.mkdir(parents=True, exist_ok=True)
    paths["watchlist_csv"].write_text("ticker,rank\nAAA,1\n", encoding="utf-8")
    _write_json(paths["watchlist_json"], {"rows": [{"ticker": "AAA", "rank": 1}]})
    _write_json(paths["memo_manifest"], {"memo_count": 1, "watchlist_state_path": str(paths["watchlist_state"])})
    _write_json(paths["watchlist_state"], {"ticker_count": 1, "tickers": {"AAA": {"last_rank": 1}}})

    older = paths["batch_state"].stat().st_mtime - 10
    newer = paths["batch_state"].stat().st_mtime + 10
    os.utime(paths["batch_state"], (older, older))
    os.utime(paths["batch_summary"], (older, older))
    os.utime(paths["global_shortlist"], (newer, newer))
    os.utime(paths["global_rollup"], (newer, newer))
    os.utime(paths["manifest"], (newer + 5, newer + 5))
    os.utime(paths["watchlist_csv"], (newer + 5, newer + 5))
    os.utime(paths["watchlist_json"], (newer + 5, newer + 5))
    os.utime(paths["memo_manifest"], (newer + 10, newer + 10))
    os.utime(paths["watchlist_state"], (newer + 10, newer + 10))

    calls: dict[str, int] = {"scout": 0, "batch": 0, "resume_batch": 0, "rollup": 0, "dossier": 0, "memo": 0}
    monkeypatch.setattr("app.universe.autopilot._run_scout", lambda **_kwargs: calls.__setitem__("scout", calls["scout"] + 1))
    monkeypatch.setattr("app.universe.autopilot._run_depth_batch", lambda **_kwargs: calls.__setitem__("batch", calls["batch"] + 1))
    monkeypatch.setattr("app.universe.autopilot._resume_depth_batch", lambda *_args, **_kwargs: calls.__setitem__("resume_batch", calls["resume_batch"] + 1))
    monkeypatch.setattr("app.universe.autopilot._run_rollup", lambda **_kwargs: calls.__setitem__("rollup", calls["rollup"] + 1))
    monkeypatch.setattr("app.universe.autopilot._run_dossier_pack", lambda **_kwargs: calls.__setitem__("dossier", calls["dossier"] + 1))
    monkeypatch.setattr("app.universe.autopilot._run_memo_pack", lambda **_kwargs: calls.__setitem__("memo", calls["memo"] + 1))

    payload = run_universe_autopilot(
        universe_run_id=run_id,
        as_of_date="2026-02-14",
        scout_params={},
        depth_batch_params={"max_runs": 3},
        rollup_params={"top_n": 10, "policy": "value_first"},
        dossier_pack_params={"top_n": 10, "policy": "value_first"},
        memo_pack_params={"top_n": 10, "policy": "value_first"},
        resume=True,
    )
    assert payload["status"] == AUTOPILOT_DONE
    assert calls == {"scout": 0, "batch": 0, "resume_batch": 0, "rollup": 0, "dossier": 0, "memo": 0}
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    assert state["batch_run_id"] == batch_run_id


def test_universe_autopilot_deterministic_batch_id_and_status_open(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "auto_full"
    paths = _autopilot_expected_paths(cfg, run_id)
    batch_run_id = f"{run_id}_depth_batch"
    calls: dict[str, int] = {"batch": 0}

    def _fake_scout(**_kwargs):
        _write_json(paths["universe_summary"], {"counts": {"PASS": 0, "WATCH": 1, "FAIL": 1}})
        _write_json(paths["depth_queue"], {"entries": [{"rank": 1, "tickers": ["AAA"], "sector_suggested": "Software"}]})
        return {"status": "OK", "counts": {"PASS": 0, "WATCH": 1, "FAIL": 1}}

    def _fake_batch(**kwargs):
        assert kwargs["batch_run_id"] == batch_run_id
        calls["batch"] += 1
        _write_json(paths["batch_state"], {"status": "DONE", "cursor_next_idx": 1, "planned_runs": [{}], "completed_runs": [{}]})
        _write_json(paths["batch_summary"], {"status": "DONE", "total_planned": 1, "done_count": 1})
        paths["batch_log"].parent.mkdir(parents=True, exist_ok=True)
        paths["batch_log"].write_text("", encoding="utf-8")
        return {"status": "OK", "run_status": "DONE", "planned_count": 1, "completed_count": 1}

    def _fake_rollup(**_kwargs):
        _write_json(paths["global_shortlist"], {"rows": [{"ticker": "AAA", "implied_return_base": 0.25, "mos_epv": 0.3, "yield_metric_used": "owner_earnings_yield_ev_3y"}]})
        _write_json(paths["global_rollup"], {"run_count_total": 1, "run_count_done": 1, "run_count_failed": 0, "run_count_cancelled": 0, "candidate_count_total": 1, "candidate_count_ranked": 1, "coverage_breakdown": {}})
        paths["global_md"].write_text("# md\n", encoding="utf-8")
        return {
            "status": "OK",
            "global_shortlist_json_path": str(paths["global_shortlist"]),
            "global_shortlist_md_path": str(paths["global_md"]),
            "global_rollup_json_path": str(paths["global_rollup"]),
        }

    def _fake_dossier(**_kwargs):
        _write_json(
            paths["manifest"],
            {
                "watchlist_csv_path": str(paths["watchlist_csv"]),
                "watchlist_json_path": str(paths["watchlist_json"]),
                "candidate_count": 1,
                "unknown_counts": {},
            },
        )
        paths["watchlist_csv"].parent.mkdir(parents=True, exist_ok=True)
        paths["watchlist_csv"].write_text("ticker,rank\nAAA,1\n", encoding="utf-8")
        _write_json(paths["watchlist_json"], {"rows": [{"ticker": "AAA", "rank": 1}]})
        return {
            "status": "OK",
            "manifest_path": str(paths["manifest"]),
            "watchlist_csv_path": str(paths["watchlist_csv"]),
            "watchlist_json_path": str(paths["watchlist_json"]),
        }

    def _fake_memo(**_kwargs):
        _write_json(
            paths["memo_manifest"],
            {
                "memo_count": 1,
                "watchlist_state_path": str(paths["watchlist_state"]),
            },
        )
        _write_json(paths["watchlist_state"], {"ticker_count": 1, "tickers": {"AAA": {"last_rank": 1}}})
        return {
            "status": "OK",
            "manifest_path": str(paths["memo_manifest"]),
            "watchlist_state_path": str(paths["watchlist_state"]),
        }

    monkeypatch.setattr("app.universe.autopilot._run_scout", _fake_scout)
    monkeypatch.setattr("app.universe.autopilot._run_depth_batch", _fake_batch)
    monkeypatch.setattr("app.universe.autopilot._run_rollup", _fake_rollup)
    monkeypatch.setattr("app.universe.autopilot._run_dossier_pack", _fake_dossier)
    monkeypatch.setattr("app.universe.autopilot._run_memo_pack", _fake_memo)

    payload = run_universe_autopilot(
        universe_run_id=run_id,
        as_of_date="2026-02-14",
        scout_params={},
        depth_batch_params={"max_runs": 3},
        rollup_params={"top_n": 10, "policy": "value_first"},
        dossier_pack_params={"top_n": 10, "policy": "value_first"},
        memo_pack_params={"top_n": 10, "policy": "value_first"},
        resume=True,
    )
    assert payload["status"] == AUTOPILOT_DONE
    assert calls["batch"] == 1

    status_payload = universe_autopilot_status(run_id)
    assert status_payload["status"] == "OK"
    assert status_payload["batch_run_id"] == batch_run_id
    assert status_payload["depth_batch_progress"]["run_status"] == "DONE"

    open_payload = open_universe_autopilot(run_id)
    assert open_payload["status"] == "OK"
    assert open_payload["summary"]["watchlist_paths"]["watchlist_csv_path"] == str(paths["watchlist_csv"])
    assert open_payload["summary"]["memo_pack_manifest_path"] == str(paths["memo_manifest"])


def test_universe_autopilot_cancel_prevents_execution(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    run_id = "auto_cancel"
    run_universe_autopilot(
        universe_run_id=run_id,
        as_of_date="2026-02-14",
        scout_params={},
        depth_batch_params={"max_runs": 1},
        rollup_params={"top_n": 5, "policy": "value_first"},
        dossier_pack_params={"top_n": 5, "policy": "value_first"},
        memo_pack_params={"top_n": 5, "policy": "value_first"},
        resume=True,
        dry_run=True,
    )
    cancel_payload = cancel_universe_autopilot(run_id, "operator cancel")
    assert cancel_payload["status"] == "OK"
    assert cancel_payload["run_status"] == AUTOPILOT_CANCELLED

    called = {"scout": 0}

    def _forbid(**_kwargs):
        called["scout"] += 1
        raise AssertionError("scout should not run after autopilot cancel")

    monkeypatch.setattr("app.universe.autopilot._run_scout", _forbid)
    payload = run_universe_autopilot(
        universe_run_id=run_id,
        as_of_date="2026-02-14",
        scout_params={},
        depth_batch_params={"max_runs": 1},
        rollup_params={"top_n": 5, "policy": "value_first"},
        dossier_pack_params={"top_n": 5, "policy": "value_first"},
        memo_pack_params={"top_n": 5, "policy": "value_first"},
        resume=True,
    )
    assert payload["status"] == "CANCELLED"
    assert called["scout"] == 0


def test_universe_autopilot_surfaces_scout_stall_reason(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "auto_scout_stall"
    paths = _autopilot_expected_paths(cfg, run_id)

    def _fake_scout(**_kwargs):
        _write_json(
            paths["universe_summary"],
            {
                "counts": {"PASS": 0, "WATCH": 0, "FAIL": 0},
                "run_status": "PARTIAL",
                "stop_reason_code": "STALE_SCOUT",
                "stop_summary": "Scout stalled during SCOUT_SCORING for ticker=AAA after 8.0s.",
                "hydration_status": "STALE",
                "primary_scout_blocker": "SCOUT_SCORING_TIMEOUT",
                "last_progress_phase": "SCOUT_SCORING",
                "stalled_reason_code": "SCOUT_SCORING_TIMEOUT",
                "hydration_progress": {"current_ticker": "AAA"},
                "facts_blockers": {
                    "retryable_facts_blocker_count": 2,
                    "terminal_facts_blocker_count": 1,
                    "partial_usable_facts_count": 0,
                    "top_retryable_facts_blockers": [
                        {
                            "ticker": "AAA",
                            "facts_blocker_class": "FACTS_RETRYABLE_TIMEOUT",
                            "facts_recommended_action": "RETRY_COMPANYFACTS_HYDRATION",
                        }
                    ],
                    "economic_fail_count_vs_evidence_fail_count": {
                        "evidence_fail_count": 2,
                        "economic_fail_count": 0,
                        "mixed_fail_count": 0,
                        "other_fail_count": 0,
                    },
                },
            },
        )
        _write_json(paths["depth_queue"], {"entries": []})
        return {
            "status": "OK",
            "run_status": "PARTIAL",
            "stop_reason_code": "STALE_SCOUT",
            "stop_summary": "Scout stalled during SCOUT_SCORING for ticker=AAA after 8.0s.",
            "hydration_status": "STALE",
            "primary_scout_blocker": "SCOUT_SCORING_TIMEOUT",
            "last_progress_phase": "SCOUT_SCORING",
            "stalled_reason_code": "SCOUT_SCORING_TIMEOUT",
        }

    monkeypatch.setattr("app.universe.autopilot._run_scout", _fake_scout)

    payload = run_universe_autopilot(
        universe_run_id=run_id,
        as_of_date="2026-02-14",
        scout_params={},
        depth_batch_params={"max_runs": 1},
        rollup_params={"top_n": 5, "policy": "value_first"},
        dossier_pack_params={"top_n": 5, "policy": "value_first"},
        memo_pack_params={"top_n": 5, "policy": "value_first"},
        resume=True,
    )
    assert payload["status"] == "PARTIAL"

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["scout_hydration_status"] == "STALE"
    assert summary["primary_scout_blocker"] == "SCOUT_SCORING_TIMEOUT"
    assert summary["scout_last_progress_phase"] == "SCOUT_SCORING"
    assert summary["scout_stalled_reason_code"] == "SCOUT_SCORING_TIMEOUT"
    assert summary["retryable_facts_blocker_count"] == 2
    assert summary["terminal_facts_blocker_count"] == 1
    assert summary["top_retryable_facts_blockers"][0]["facts_blocker_class"] == "FACTS_RETRYABLE_TIMEOUT"
