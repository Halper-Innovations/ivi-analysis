from __future__ import annotations

import json
from pathlib import Path

from app.db import init_db
from app.universe.autopilot import (
    STAGE_DOSSIER_PACK,
    STAGE_FILING_DIFF,
    STAGE_MEMO_PACK,
    STAGE_PATTERN_SCAN,
    STAGE_VARIANT_SYNTHESIS,
    _run_filing_diff,
    _run_pattern_scan,
    _run_variant_synthesis,
    run_universe_autopilot,
)


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


def _paths(cfg, run_id: str) -> dict[str, Path]:
    batch_run_id = f"{run_id}_depth_batch"
    universe_root = cfg.outputs_dir / "universe" / run_id
    batch_root = universe_root / "depth_batches" / batch_run_id
    return {
        "state": universe_root / "autopilot" / "autopilot_state.json",
        "summary": universe_root / "autopilot" / "autopilot_summary.json",
        "batch_state": batch_root / "batch_state.json",
        "batch_summary": batch_root / "batch_summary.json",
        "batch_log": batch_root / "batch_log.jsonl",
        "shortlist": batch_root / "global_shortlist.json",
        "rollup": batch_root / "global_rollup.json",
        "shortlist_md": batch_root / "global_shortlist.md",
        "dossier_manifest": batch_root / "dossier_pack" / "dossier_pack_manifest.json",
        "watchlist_csv": batch_root / "dossier_pack" / "watchlist.csv",
        "watchlist_json": batch_root / "dossier_pack" / "watchlist.json",
        "memo_manifest": batch_root / "memo_pack" / "memo_pack_manifest.json",
        "watchlist_state": universe_root / "autopilot" / "watchlist_state.json",
        "universe_summary": cfg.sectors_dir / run_id / "universe_summary.json",
        "depth_queue": universe_root / "depth_queue.json",
    }


def _install_base_stage_fakes(monkeypatch, cfg, run_id: str, calls: list[str]) -> None:
    paths = _paths(cfg, run_id)

    def _fake_scout(**_kwargs):
        calls.append("SCOUT")
        _write_json(paths["universe_summary"], {"counts": {"PASS": 1, "WATCH": 0, "FAIL": 0}})
        _write_json(paths["depth_queue"], {"entries": [{"rank": 1, "tickers": ["AAA"], "sector_suggested": "Software"}]})
        return {"status": "OK"}

    def _fake_batch(**_kwargs):
        calls.append("DEPTH_BATCH")
        _write_json(
            paths["batch_state"],
            {
                "status": "DONE",
                "cursor_next_idx": 1,
                "planned_runs": [{"run_id": f"{run_id}_depth_batch__Software__001"}],
                "completed_runs": [
                    {
                        "run_id": f"{run_id}_depth_batch__Software__001",
                        "status": "DONE",
                        "tickers": ["AAA"],
                    }
                ],
            },
        )
        _write_json(paths["batch_summary"], {"status": "DONE", "total_planned": 1, "done_count": 1})
        paths["batch_log"].parent.mkdir(parents=True, exist_ok=True)
        paths["batch_log"].write_text("", encoding="utf-8")
        return {"status": "OK", "run_status": "DONE", "planned_count": 1, "completed_count": 1}

    def _fake_rollup(**_kwargs):
        calls.append("ROLLUP")
        _write_json(
            paths["shortlist"],
            {"rows": [{"ticker": "AAA", "implied_return_base": 0.2, "mos_epv": 0.3, "yield_metric_used": "owner"}]},
        )
        _write_json(paths["rollup"], {"candidate_count_ranked": 1})
        paths["shortlist_md"].write_text("# shortlist\n", encoding="utf-8")
        return {
            "status": "OK",
            "global_shortlist_json_path": str(paths["shortlist"]),
            "global_shortlist_md_path": str(paths["shortlist_md"]),
            "global_rollup_json_path": str(paths["rollup"]),
        }

    def _fake_dossier(**_kwargs):
        calls.append("DOSSIER_PACK")
        _write_json(paths["dossier_manifest"], {"candidate_count": 1})
        paths["watchlist_csv"].parent.mkdir(parents=True, exist_ok=True)
        paths["watchlist_csv"].write_text("ticker,rank\nAAA,1\n", encoding="utf-8")
        _write_json(paths["watchlist_json"], {"rows": [{"ticker": "AAA", "rank": 1}]})
        return {
            "status": "OK",
            "manifest_path": str(paths["dossier_manifest"]),
            "watchlist_csv_path": str(paths["watchlist_csv"]),
            "watchlist_json_path": str(paths["watchlist_json"]),
        }

    def _fake_memo(**_kwargs):
        calls.append("MEMO_PACK")
        _write_json(paths["memo_manifest"], {"memo_count": 1})
        _write_json(paths["watchlist_state"], {"ticker_count": 1})
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


def test_depth_fundamentals_skips_l4_stages(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    calls: list[str] = []
    _install_base_stage_fakes(monkeypatch, cfg, "l5_fundamentals", calls)

    payload = run_universe_autopilot(
        universe_run_id="l5_fundamentals",
        as_of_date="2026-03-22",
        depth="fundamentals",
        scout_params={},
        depth_batch_params={"max_runs": 1},
        rollup_params={"top_n": 5, "policy": "value_first"},
        dossier_pack_params={"top_n": 5, "policy": "value_first"},
        memo_pack_params={"top_n": 5, "policy": "value_first"},
    )

    assert payload["status"] == "DONE"
    assert calls == ["SCOUT", "DEPTH_BATCH", "ROLLUP", "DOSSIER_PACK", "MEMO_PACK"]
    state = json.loads(_paths(cfg, "l5_fundamentals")["state"].read_text(encoding="utf-8"))
    for stage in [STAGE_FILING_DIFF, STAGE_PATTERN_SCAN, STAGE_VARIANT_SYNTHESIS]:
        assert state["stages"][stage]["status"] == "DONE"
        assert state["stages"][stage]["result"]["skipped_by_depth"] is True


def test_depth_full_runs_all_stages(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    calls: list[str] = []
    run_id = "l5_full"
    _install_base_stage_fakes(monkeypatch, cfg, run_id, calls)

    monkeypatch.setattr(
        "app.universe.autopilot._run_filing_diff",
        lambda **_kwargs: calls.append("FILING_DIFF") or {"status": "OK", "filing_diff_summary_path": str(_paths(cfg, run_id)["state"].parent / "fd.json"), "filing_diff_dir_path": str(_paths(cfg, run_id)["state"].parent)},
    )
    monkeypatch.setattr(
        "app.universe.autopilot._run_pattern_scan",
        lambda **_kwargs: calls.append("PATTERN_SCAN") or {"status": "OK", "pattern_scan_report_path": str(_paths(cfg, run_id)["state"].parent / "ps_report.json"), "pattern_scan_summary_path": str(_paths(cfg, run_id)["state"].parent / "ps_summary.json")},
    )
    monkeypatch.setattr(
        "app.universe.autopilot._run_variant_synthesis",
        lambda **_kwargs: calls.append("VARIANT_SYNTHESIS") or {"status": "OK", "variant_synthesis_summary_path": str(_paths(cfg, run_id)["state"].parent / "vs_summary.json"), "variant_perceptions_dir_path": str(_paths(cfg, run_id)["state"].parent)},
    )

    payload = run_universe_autopilot(
        universe_run_id=run_id,
        as_of_date="2026-03-22",
        depth="full",
        scout_params={},
        depth_batch_params={"max_runs": 1},
        rollup_params={"top_n": 5, "policy": "value_first"},
        dossier_pack_params={"top_n": 5, "policy": "value_first"},
        memo_pack_params={"top_n": 5, "policy": "value_first"},
    )

    assert payload["status"] == "DONE"
    assert calls == [
        "SCOUT",
        "DEPTH_BATCH",
        "ROLLUP",
        "FILING_DIFF",
        "PATTERN_SCAN",
        "VARIANT_SYNTHESIS",
        "DOSSIER_PACK",
        "MEMO_PACK",
    ]


def test_depth_alpha_only_skips_dossier_and_memo(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    calls: list[str] = []
    run_id = "l5_alpha"
    _install_base_stage_fakes(monkeypatch, cfg, run_id, calls)

    monkeypatch.setattr("app.universe.autopilot._run_filing_diff", lambda **_kwargs: calls.append("FILING_DIFF") or {"status": "OK"})
    monkeypatch.setattr("app.universe.autopilot._run_pattern_scan", lambda **_kwargs: calls.append("PATTERN_SCAN") or {"status": "OK"})
    monkeypatch.setattr("app.universe.autopilot._run_variant_synthesis", lambda **_kwargs: calls.append("VARIANT_SYNTHESIS") or {"status": "OK"})

    payload = run_universe_autopilot(
        universe_run_id=run_id,
        as_of_date="2026-03-22",
        depth="alpha-only",
        scout_params={},
        depth_batch_params={"max_runs": 1},
        rollup_params={"top_n": 5, "policy": "value_first"},
        dossier_pack_params={"top_n": 5, "policy": "value_first"},
        memo_pack_params={"top_n": 5, "policy": "value_first"},
    )

    assert payload["status"] == "DONE"
    assert calls == ["SCOUT", "DEPTH_BATCH", "ROLLUP", "FILING_DIFF", "PATTERN_SCAN", "VARIANT_SYNTHESIS"]
    state = json.loads(_paths(cfg, run_id)["state"].read_text(encoding="utf-8"))
    for stage in [STAGE_DOSSIER_PACK, STAGE_MEMO_PACK]:
        assert state["stages"][stage]["status"] == "DONE"
        assert state["stages"][stage]["result"]["skipped_by_depth"] is True


def test_stage_dependency_skips(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "l5_dep"
    batch_run_id = f"{run_id}_depth_batch"
    paths = _paths(cfg, run_id)

    _write_json(paths["batch_state"], {"status": "FAILED", "completed_runs": []})
    skipped_diff = _run_filing_diff(
        universe_run_id=run_id,
        batch_run_id=batch_run_id,
        as_of_date="2026-03-22",
        filing_diff_params={},
    )
    assert skipped_diff["status"] == "SKIPPED"
    assert skipped_diff["reason"] == "DEPTH_BATCH_NOT_READY"

    _write_json(paths["batch_state"], {"status": "PARTIAL", "completed_runs": []})
    skipped_pattern = _run_pattern_scan(
        universe_run_id=run_id,
        batch_run_id=batch_run_id,
        as_of_date="2026-03-22",
        pattern_scan_params={},
    )
    assert skipped_pattern["status"] == "SKIPPED"
    assert skipped_pattern["reason"] == "DEPTH_BATCH_INCOMPLETE"

    _write_json(
        paths["batch_state"],
        {"status": "DONE", "completed_runs": [{"run_id": "child", "status": "DONE", "tickers": ["AAA"]}]},
    )
    skipped_variant = _run_variant_synthesis(
        universe_run_id=run_id,
        batch_run_id=batch_run_id,
        as_of_date="2026-03-22",
        variant_synthesis_params={},
    )
    assert skipped_variant["status"] == "SKIPPED"
    assert skipped_variant["reason"] == "NO_L4_SIGNALS_AVAILABLE"


def test_resume_preserves_depth(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)

    first = run_universe_autopilot(
        universe_run_id="l5_resume_depth",
        as_of_date="2026-03-22",
        depth="full",
        scout_params={},
        depth_batch_params={"max_runs": 1},
        rollup_params={"top_n": 5, "policy": "value_first"},
        dossier_pack_params={"top_n": 5, "policy": "value_first"},
        memo_pack_params={"top_n": 5, "policy": "value_first"},
        dry_run=True,
    )
    assert first["depth"] == "full"

    second = run_universe_autopilot(
        universe_run_id="l5_resume_depth",
        as_of_date="2026-03-22",
        depth="fundamentals",
        scout_params={},
        depth_batch_params={"max_runs": 1},
        rollup_params={"top_n": 5, "policy": "value_first"},
        dossier_pack_params={"top_n": 5, "policy": "value_first"},
        memo_pack_params={"top_n": 5, "policy": "value_first"},
        dry_run=True,
        resume=True,
    )

    assert second["depth"] == "full"
    state = json.loads(_paths(cfg, "l5_resume_depth")["state"].read_text(encoding="utf-8"))
    assert state["depth"] == "full"


def test_default_depth_is_backward_compatible(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    calls: list[str] = []
    _install_base_stage_fakes(monkeypatch, cfg, "l5_default_depth", calls)

    payload = run_universe_autopilot(
        universe_run_id="l5_default_depth",
        as_of_date="2026-03-22",
        scout_params={},
        depth_batch_params={"max_runs": 1},
        rollup_params={"top_n": 5, "policy": "value_first"},
        dossier_pack_params={"top_n": 5, "policy": "value_first"},
        memo_pack_params={"top_n": 5, "policy": "value_first"},
    )

    assert payload["status"] == "DONE"
    assert payload["depth"] == "fundamentals"
    state = json.loads(_paths(cfg, "l5_default_depth")["state"].read_text(encoding="utf-8"))
    assert state["depth"] == "fundamentals"
    assert state["stages"][STAGE_FILING_DIFF]["result"]["skipped_by_depth"] is True
