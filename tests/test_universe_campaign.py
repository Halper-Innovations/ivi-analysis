from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.campaign import (
    CAMPAIGN_CANCELLED,
    CAMPAIGN_DONE,
    CAMPAIGN_PARTIAL,
    build_campaign_plan,
    cancel_campaign,
    campaign_status,
    open_campaign,
    run_campaign,
)


runner = CliRunner()


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "sample_universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker\nAAA\nBBB\nCCC\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg, universe


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _campaign_spec(path: Path) -> dict:
    return {
        "as_of_date": "2026-02-14",
        "items": [
            {
                "label": "software_core",
                "universe_file": str(path),
                "max_runs": 2,
                "top_n": 10,
                "policy": "value_first",
            },
            {
                "label": "software_growth",
                "universe_file": str(path),
                "max_runs": 2,
                "top_n": 10,
                "policy": "value_first",
            },
        ],
    }


def _write_child_outputs(cfg, *, universe_run_id: str, label: str) -> None:
    batch_run_id = f"{universe_run_id}_depth_batch"
    autopilot_dir = cfg.outputs_dir / "universe" / universe_run_id / "autopilot"
    batch_dir = cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id
    memo_dir = batch_dir / "memo_pack" / "memos"
    dossier_dir = batch_dir / "dossier_pack"
    memo_dir.mkdir(parents=True, exist_ok=True)
    dossier_dir.mkdir(parents=True, exist_ok=True)

    if label == "software_core":
        candidates = [
            {
                "ticker": "AAA",
                "rank": 1,
                "value_gate_status": "PASS",
                "implied_return_base": 0.35,
                "mos_epv": 0.40,
                "mos_netnet": 0.10,
                "owner_yield": 0.08,
                "yield_metric_used": "owner_earnings_yield_ev_3y",
                "primary_blocker": "NONE",
                "composite_score_total": 82.0,
            },
            {
                "ticker": "BBB",
                "rank": 2,
                "value_gate_status": "WATCH",
                "implied_return_base": "UNKNOWN",
                "mos_epv": "UNKNOWN",
                "mos_netnet": "UNKNOWN",
                "owner_yield": "UNKNOWN",
                "yield_metric_used": "UNKNOWN",
                "primary_blocker": "PRICE_UNKNOWN",
                "composite_score_total": 55.0,
            },
        ]
    else:
        candidates = [
            {
                "ticker": "AAA",
                "rank": 1,
                "value_gate_status": "WATCH",
                "implied_return_base": 0.20,
                "mos_epv": 0.22,
                "mos_netnet": 0.05,
                "owner_yield": 0.05,
                "yield_metric_used": "owner_earnings_yield_ev_3y",
                "primary_blocker": "MISSING_EV",
                "composite_score_total": 68.0,
            },
            {
                "ticker": "CCC",
                "rank": 2,
                "value_gate_status": "PASS",
                "implied_return_base": 0.28,
                "mos_epv": 0.30,
                "mos_netnet": 0.08,
                "owner_yield": 0.07,
                "yield_metric_used": "owner_earnings_yield_ev_3y",
                "primary_blocker": "NONE",
                "composite_score_total": 79.0,
            },
        ]

    memo_entries = []
    watch_tickers = {}
    shortlist_rows = []
    for candidate in candidates:
        ticker = candidate["ticker"]
        memo_json_path = memo_dir / f"{ticker}.json"
        memo_md_path = memo_dir / f"{ticker}.md"
        memo_payload = {
            "header": {
                "ticker": ticker,
                "sector": "Software",
                "as_of_date": "2026-02-14",
                "run_ids": {
                    "universe_run_id": universe_run_id,
                    "batch_run_id": batch_run_id,
                    "depth_run_ids": [f"{batch_run_id}__Software__001"],
                },
                "generated_at": "2026-03-06T00:00:00+00:00",
            },
            "decision_snapshot": {
                "value_gate_status": candidate["value_gate_status"],
                "implied_return_base": candidate["implied_return_base"],
            },
            "graham_dodd": {
                "mos_epv": candidate["mos_epv"],
                "mos_netnet": candidate["mos_netnet"],
            },
            "owner_earnings_and_yield": {
                "owner_earnings_yield_ev_3y": candidate["owner_yield"],
                "yield_metric_used": candidate["yield_metric_used"],
            },
            "derived_from_index": [f"memo.{label}.{ticker}"],
            "rank_global": candidate["rank"],
            "value_gate_status": candidate["value_gate_status"],
            "implied_return_base": candidate["implied_return_base"],
            "primary_blocker": candidate["primary_blocker"],
        }
        _write_json(memo_json_path, memo_payload)
        memo_md_path.write_text(f"# {ticker}\n", encoding="utf-8")
        memo_entries.append(
            {
                "ticker": ticker,
                "rank_global": candidate["rank"],
                "memo_json_path": str(memo_json_path),
                "memo_md_path": str(memo_md_path),
            }
        )
        watch_tickers[ticker] = {
            "first_seen_run_id": universe_run_id,
            "last_seen_run_id": universe_run_id,
            "last_value_gate_status": candidate["value_gate_status"],
            "last_implied_return_base": candidate["implied_return_base"],
            "last_primary_blocker_code": candidate["primary_blocker"],
            "last_rank": candidate["rank"],
            "history": [],
        }
        shortlist_rows.append(
            {
                "ticker": ticker,
                "value_gate_status": candidate["value_gate_status"],
                "implied_return_base": candidate["implied_return_base"],
                "mos_epv": candidate["mos_epv"],
                "mos_netnet": candidate["mos_netnet"],
                "owner_earnings_yield_ev_3y": candidate["owner_yield"],
                "yield_metric_used": candidate["yield_metric_used"],
                "primary_blocker": candidate["primary_blocker"],
                "composite_score_total": candidate["composite_score_total"],
                "derived_from": [f"shortlist.{label}.{ticker}"],
            }
        )

    memo_manifest_path = batch_dir / "memo_pack" / "memo_pack_manifest.json"
    _write_json(
        memo_manifest_path,
        {
            "universe_run_id": universe_run_id,
            "batch_run_id": batch_run_id,
            "memo_count": len(memo_entries),
            "memos": memo_entries,
        },
    )
    _write_json(
        batch_dir / "global_shortlist.json",
        {
            "universe_run_id": universe_run_id,
            "batch_run_id": batch_run_id,
            "rows": shortlist_rows,
        },
    )
    _write_json(
        batch_dir / "global_rollup.json",
        {"run_count_total": 1, "run_count_done": 1, "candidate_count_ranked": len(shortlist_rows)},
    )
    (batch_dir / "global_shortlist.md").write_text("# shortlist\n", encoding="utf-8")
    _write_json(dossier_dir / "dossier_pack_manifest.json", {"candidate_count": len(shortlist_rows)})
    _write_json(
        autopilot_dir / "watchlist_state.json",
        {
            "universe_run_id": universe_run_id,
            "ticker_count": len(watch_tickers),
            "tickers": watch_tickers,
        },
    )
    _write_json(
        autopilot_dir / "autopilot_state.json",
        {
            "universe_run_id": universe_run_id,
            "status": "DONE",
        },
    )
    _write_json(
        autopilot_dir / "autopilot_summary.json",
        {
            "status": "DONE",
            "memo_pack_manifest_path": str(memo_manifest_path),
        },
    )


def _write_scout_coverage(cfg, *, universe_run_id: str, rows: list[dict]) -> None:
    _write_json(
        cfg.outputs_dir / "sectors" / universe_run_id / "universe_coverage.json",
        {
            "run_id": universe_run_id,
            "rows": rows,
        },
    )


def test_campaign_plan_is_deterministic(monkeypatch, tmp_path):
    _cfg, universe = _init_cfg(monkeypatch, tmp_path)
    spec = _campaign_spec(universe)
    spec["campaign_run_id"] = "campaign_test"

    plan = build_campaign_plan(spec)
    assert [item["label"] for item in plan] == ["software_core", "software_growth"]
    assert [item["child_universe_run_id"] for item in plan] == [
        "campaign_test__software_core",
        "campaign_test__software_growth",
    ]


def test_campaign_resume_skips_completed_items_and_merges_master(monkeypatch, tmp_path):
    cfg, universe = _init_cfg(monkeypatch, tmp_path)
    spec = _campaign_spec(universe)
    calls: list[str] = []

    def _fake_run_autopilot(**kwargs):
        universe_run_id = kwargs["universe_run_id"]
        label = universe_run_id.split("__")[-1]
        calls.append(universe_run_id)
        _write_child_outputs(cfg, universe_run_id=universe_run_id, label=label)
        return {"status": CAMPAIGN_DONE, "universe_run_id": universe_run_id}

    def _fake_open_autopilot(universe_run_id: str):
        batch_run_id = f"{universe_run_id}_depth_batch"
        return {
            "run_status": "DONE",
            "autopilot_state_path": str(cfg.outputs_dir / "universe" / universe_run_id / "autopilot" / "autopilot_state.json"),
            "autopilot_summary_path": str(cfg.outputs_dir / "universe" / universe_run_id / "autopilot" / "autopilot_summary.json"),
            "stages": {
                "ROLLUP": {
                    "artifact_paths": {
                        "global_shortlist_json_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "global_shortlist.json"),
                        "global_rollup_json_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "global_rollup.json"),
                    }
                },
                "DOSSIER_PACK": {
                    "artifact_paths": {
                        "manifest_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "dossier_pack" / "dossier_pack_manifest.json"),
                    }
                },
                "MEMO_PACK": {
                    "artifact_paths": {
                        "manifest_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "memo_pack" / "memo_pack_manifest.json"),
                        "watchlist_state_path": str(cfg.outputs_dir / "universe" / universe_run_id / "autopilot" / "watchlist_state.json"),
                    }
                },
            },
        }

    monkeypatch.setattr("app.universe.campaign._run_autopilot", _fake_run_autopilot)
    monkeypatch.setattr("app.universe.campaign._open_autopilot", _fake_open_autopilot)

    first = run_campaign(spec, campaign_run_id="campaign_resume", resume=True, max_items=1)
    assert first["status"] == CAMPAIGN_PARTIAL
    assert calls == ["campaign_resume__software_core"]

    second = run_campaign(spec, campaign_run_id="campaign_resume", resume=True)
    assert second["status"] == CAMPAIGN_DONE
    assert calls == ["campaign_resume__software_core", "campaign_resume__software_growth"]

    master_shortlist = json.loads(Path(second["master_shortlist_json_path"]).read_text(encoding="utf-8"))
    tickers = [row["ticker"] for row in master_shortlist["rows"]]
    assert tickers == ["AAA", "CCC", "BBB"]
    aaa = next(row for row in master_shortlist["rows"] if row["ticker"] == "AAA")
    assert aaa["best_rank_seen"] == 1
    assert len(aaa["source_runs"]) == 2
    assert aaa["value_gate_status"] == "PASS"

    master_watchlist = json.loads(Path(second["master_watchlist_state_path"]).read_text(encoding="utf-8"))
    assert master_watchlist["tickers"]["AAA"]["appearances_count"] == 2
    assert master_watchlist["tickers"]["AAA"]["latest_value_gate_status"] == "WATCH"


def test_campaign_rehydrates_scout_facts_blockers_into_promotion_and_escalation(monkeypatch, tmp_path):
    cfg, universe = _init_cfg(monkeypatch, tmp_path)
    spec = {
        "as_of_date": "2026-02-14",
        "items": [
            {
                "label": "software_core",
                "universe_file": str(universe),
                "max_runs": 1,
                "top_n": 10,
                "policy": "value_first",
            }
        ],
    }

    def _fake_run_autopilot(**kwargs):
        universe_run_id = kwargs["universe_run_id"]
        _write_child_outputs(cfg, universe_run_id=universe_run_id, label="software_core")
        _write_scout_coverage(
            cfg,
            universe_run_id=universe_run_id,
            rows=[
                {
                    "ticker": "AAA",
                    "scout_status": "FAIL",
                    "primary_blocker_category": "MISSING_FACTS",
                    "facts_blocker_class": "FACTS_RETRYABLE_TIMEOUT",
                    "facts_blocker_retryable": True,
                    "facts_blocker_terminal": False,
                    "facts_blocker_partial_usable": False,
                    "facts_missing_key_inputs": ["SHARES", "CFO", "CAPEX", "FCF"],
                    "facts_retry_recommended": True,
                    "facts_blocker_reason_codes": ["SCOUT_FACTS_TIMEOUT"],
                    "facts_recommended_action": "RETRY_COMPANYFACTS_HYDRATION",
                    "fail_due_to_missing_evidence": True,
                    "fail_due_to_economic_weakness": False,
                    "primary_fail_domain": "EVIDENCE",
                },
                {
                    "ticker": "BBB",
                    "scout_status": "FAIL",
                    "primary_blocker_category": "MISSING_FACTS",
                    "facts_blocker_class": "FACTS_RETRYABLE_TIMEOUT",
                    "facts_blocker_retryable": True,
                    "facts_blocker_terminal": False,
                    "facts_blocker_partial_usable": False,
                    "facts_missing_key_inputs": ["SHARES", "CFO", "CAPEX", "FCF"],
                    "facts_retry_recommended": True,
                    "facts_blocker_reason_codes": ["SCOUT_FACTS_TIMEOUT"],
                    "facts_recommended_action": "RETRY_COMPANYFACTS_HYDRATION",
                    "fail_due_to_missing_evidence": True,
                    "fail_due_to_economic_weakness": False,
                    "primary_fail_domain": "EVIDENCE",
                },
            ],
        )
        return {"status": CAMPAIGN_DONE, "universe_run_id": universe_run_id}

    def _fake_open_autopilot(universe_run_id: str):
        batch_run_id = f"{universe_run_id}_depth_batch"
        return {
            "run_status": "DONE",
            "autopilot_state_path": str(cfg.outputs_dir / "universe" / universe_run_id / "autopilot" / "autopilot_state.json"),
            "autopilot_summary_path": str(cfg.outputs_dir / "universe" / universe_run_id / "autopilot" / "autopilot_summary.json"),
            "stages": {
                "ROLLUP": {
                    "artifact_paths": {
                        "global_shortlist_json_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "global_shortlist.json"),
                        "global_rollup_json_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "global_rollup.json"),
                    }
                },
                "DOSSIER_PACK": {
                    "artifact_paths": {
                        "manifest_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "dossier_pack" / "dossier_pack_manifest.json"),
                    }
                },
                "MEMO_PACK": {
                    "artifact_paths": {
                        "manifest_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "memo_pack" / "memo_pack_manifest.json"),
                        "watchlist_state_path": str(cfg.outputs_dir / "universe" / universe_run_id / "autopilot" / "watchlist_state.json"),
                    }
                },
            },
        }

    monkeypatch.setattr("app.universe.campaign._run_autopilot", _fake_run_autopilot)
    monkeypatch.setattr("app.universe.campaign._open_autopilot", _fake_open_autopilot)

    payload = run_campaign(spec, campaign_run_id="campaign_facts_overlay", resume=True)
    assert payload["status"] == CAMPAIGN_DONE

    promotion_state = json.loads((cfg.campaigns_dir / "campaign_facts_overlay" / "promotion_state.json").read_text(encoding="utf-8"))
    aaa = next(row for row in promotion_state["rows"] if row["ticker"] == "AAA")
    bbb = next(row for row in promotion_state["rows"] if row["ticker"] == "BBB")
    assert aaa["facts_blocker_class"] == "FACTS_RETRYABLE_TIMEOUT"
    assert aaa["facts_retry_recommended"] is True
    assert bbb["facts_blocker_class"] == "FACTS_RETRYABLE_TIMEOUT"
    assert "RETRYABLE_FACTS_BLOCKER" in bbb["risk_flags"]

    escalation_queue = json.loads((cfg.campaigns_dir / "campaign_facts_overlay" / "escalation_queue.json").read_text(encoding="utf-8"))
    aaa_retry = next(row for row in escalation_queue["rows"] if row["ticker"] == "AAA")
    assert aaa_retry["facts_blocker_class"] == "FACTS_RETRYABLE_TIMEOUT"
    assert aaa_retry["facts_recommended_action"] == "RETRY_COMPANYFACTS_HYDRATION"
    assert aaa_retry["action_type"] == "RETRY_COMPANYFACTS_HYDRATION"


def test_campaign_cancel_prevents_remaining_items(monkeypatch, tmp_path):
    cfg, universe = _init_cfg(monkeypatch, tmp_path)
    spec = _campaign_spec(universe)

    def _fake_run_autopilot(**kwargs):
        universe_run_id = kwargs["universe_run_id"]
        _write_child_outputs(cfg, universe_run_id=universe_run_id, label=universe_run_id.split("__")[-1])
        return {"status": CAMPAIGN_DONE, "universe_run_id": universe_run_id}

    def _fake_open_autopilot(universe_run_id: str):
        batch_run_id = f"{universe_run_id}_depth_batch"
        return {
            "run_status": "DONE",
            "autopilot_state_path": str(cfg.outputs_dir / "universe" / universe_run_id / "autopilot" / "autopilot_state.json"),
            "autopilot_summary_path": str(cfg.outputs_dir / "universe" / universe_run_id / "autopilot" / "autopilot_summary.json"),
            "stages": {
                "ROLLUP": {"artifact_paths": {"global_shortlist_json_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "global_shortlist.json"), "global_rollup_json_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "global_rollup.json")}},
                "DOSSIER_PACK": {"artifact_paths": {"manifest_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "dossier_pack" / "dossier_pack_manifest.json")}},
                "MEMO_PACK": {"artifact_paths": {"manifest_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "memo_pack" / "memo_pack_manifest.json"), "watchlist_state_path": str(cfg.outputs_dir / "universe" / universe_run_id / "autopilot" / "watchlist_state.json")}},
            },
        }

    monkeypatch.setattr("app.universe.campaign._run_autopilot", _fake_run_autopilot)
    monkeypatch.setattr("app.universe.campaign._open_autopilot", _fake_open_autopilot)

    run_campaign(spec, campaign_run_id="campaign_cancel", resume=True, max_items=1)
    cancel_payload = cancel_campaign("campaign_cancel", "operator cancel")
    assert cancel_payload["run_status"] == CAMPAIGN_CANCELLED

    def _forbid(**_kwargs):
        raise AssertionError("campaign should not run more autopilot items after cancel")

    monkeypatch.setattr("app.universe.campaign._run_autopilot", _forbid)
    payload = run_campaign(spec, campaign_run_id="campaign_cancel", resume=True)
    assert payload["status"] == CAMPAIGN_CANCELLED


def test_campaign_cli_open_and_status(monkeypatch, tmp_path):
    cfg, universe = _init_cfg(monkeypatch, tmp_path)
    spec = _campaign_spec(universe)
    spec_path = cfg.data_dir / "universe" / "sample_campaign.json"
    _write_json(spec_path, spec)

    def _fake_run_autopilot(**kwargs):
        universe_run_id = kwargs["universe_run_id"]
        _write_child_outputs(cfg, universe_run_id=universe_run_id, label=universe_run_id.split("__")[-1])
        return {"status": CAMPAIGN_DONE, "universe_run_id": universe_run_id}

    def _fake_open_autopilot(universe_run_id: str):
        batch_run_id = f"{universe_run_id}_depth_batch"
        return {
            "run_status": "DONE",
            "autopilot_state_path": str(cfg.outputs_dir / "universe" / universe_run_id / "autopilot" / "autopilot_state.json"),
            "autopilot_summary_path": str(cfg.outputs_dir / "universe" / universe_run_id / "autopilot" / "autopilot_summary.json"),
            "stages": {
                "ROLLUP": {"artifact_paths": {"global_shortlist_json_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "global_shortlist.json"), "global_rollup_json_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "global_rollup.json")}},
                "DOSSIER_PACK": {"artifact_paths": {"manifest_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "dossier_pack" / "dossier_pack_manifest.json")}},
                "MEMO_PACK": {"artifact_paths": {"manifest_path": str(cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "memo_pack" / "memo_pack_manifest.json"), "watchlist_state_path": str(cfg.outputs_dir / "universe" / universe_run_id / "autopilot" / "watchlist_state.json")}},
            },
        }

    monkeypatch.setattr("app.universe.campaign._run_autopilot", _fake_run_autopilot)
    monkeypatch.setattr("app.universe.campaign._open_autopilot", _fake_open_autopilot)

    build_cmd = runner.invoke(
        app,
        [
            "universe-campaign",
            "--campaign-file",
            str(spec_path),
            "--campaign-run-id",
            "campaign_cli",
            "--resume",
        ],
    )
    assert build_cmd.exit_code == 0, build_cmd.output
    build_payload = json.loads(build_cmd.output)
    assert build_payload["status"] == CAMPAIGN_DONE

    status_cmd = runner.invoke(app, ["universe-campaign-status", "--campaign-run-id", "campaign_cli"])
    assert status_cmd.exit_code == 0, status_cmd.output
    status_payload = json.loads(status_cmd.output)
    assert status_payload["status"] == "OK"
    assert status_payload["item_counts"]["done"] == 2

    open_cmd = runner.invoke(app, ["universe-campaign-open", "--campaign-run-id", "campaign_cli"])
    assert open_cmd.exit_code == 0, open_cmd.output
    open_payload = json.loads(open_cmd.output)
    assert open_payload["status"] == "OK"
    assert len(open_payload["top_10_master_shortlist"]) == 3
    assert open_payload["master_watchlist_state_path"].endswith("master_watchlist_state.json")


def test_campaign_summary_surfaces_child_scout_stall_reason(monkeypatch, tmp_path):
    cfg, universe = _init_cfg(monkeypatch, tmp_path)
    spec = {
        "as_of_date": "2026-02-14",
        "items": [
            {
                "label": "software_core",
                "universe_file": str(universe),
                "max_runs": 1,
                "top_n": 5,
                "policy": "value_first",
            }
        ],
    }

    def _fake_run_autopilot(**kwargs):
        universe_run_id = kwargs["universe_run_id"]
        autopilot_dir = cfg.outputs_dir / "universe" / universe_run_id / "autopilot"
        _write_json(autopilot_dir / "autopilot_state.json", {"universe_run_id": universe_run_id, "status": "PARTIAL"})
        _write_json(
            autopilot_dir / "autopilot_summary.json",
            {
                "status": "PARTIAL",
                "stop_reason_code": "STAGE_FAILED",
                "stop_summary": "Scout stalled during SCOUT_SCORING for ticker=AAA after 8.0s.",
                "primary_scout_blocker": "SCOUT_SCORING_TIMEOUT",
                "scout_hydration_status": "STALE",
                "scout_last_progress_phase": "SCOUT_SCORING",
                "scout_stalled_reason_code": "SCOUT_SCORING_TIMEOUT",
                "retryable_facts_blocker_count": 3,
                "terminal_facts_blocker_count": 1,
                "partial_usable_facts_count": 0,
                "top_retryable_facts_blockers": [
                    {
                        "ticker": "AAA",
                        "facts_blocker_class": "FACTS_RETRYABLE_TIMEOUT",
                        "facts_recommended_action": "RETRY_COMPANYFACTS_HYDRATION",
                        "primary_fail_domain": "EVIDENCE",
                    }
                ],
                "economic_fail_count_vs_evidence_fail_count": {
                    "evidence_fail_count": 3,
                    "economic_fail_count": 0,
                    "mixed_fail_count": 0,
                    "other_fail_count": 0,
                },
            },
        )
        return {
            "status": CAMPAIGN_PARTIAL,
            "stop_summary": "Scout stalled during SCOUT_SCORING for ticker=AAA after 8.0s.",
        }

    def _fake_open_autopilot(universe_run_id: str):
        autopilot_dir = cfg.outputs_dir / "universe" / universe_run_id / "autopilot"
        return {
            "run_status": "PARTIAL",
            "stop_summary": "Scout stalled during SCOUT_SCORING for ticker=AAA after 8.0s.",
            "autopilot_state_path": str(autopilot_dir / "autopilot_state.json"),
            "autopilot_summary_path": str(autopilot_dir / "autopilot_summary.json"),
            "stages": {},
        }

    monkeypatch.setattr("app.universe.campaign._run_autopilot", _fake_run_autopilot)
    monkeypatch.setattr("app.universe.campaign._open_autopilot", _fake_open_autopilot)

    payload = run_campaign(spec, campaign_run_id="campaign_scout_stall", resume=True)
    assert payload["status"] == CAMPAIGN_PARTIAL

    summary = json.loads((cfg.campaigns_dir / "campaign_scout_stall" / "campaign_summary.json").read_text(encoding="utf-8"))
    assert summary["item_blockers"][0]["primary_scout_blocker"] == "SCOUT_SCORING_TIMEOUT"
    assert summary["item_blockers"][0]["scout_hydration_status"] == "STALE"
    assert summary["item_blockers"][0]["scout_last_progress_phase"] == "SCOUT_SCORING"
    assert summary["retryable_facts_blocker_count"] == 3
    assert summary["top_retryable_facts_blockers"][0]["facts_blocker_class"] == "FACTS_RETRYABLE_TIMEOUT"
    assert summary["child_runs_with_facts_degradation"][0]["child_universe_run_id"] == "campaign_scout_stall__software_core"

    status_payload = campaign_status("campaign_scout_stall")
    assert status_payload["item_blockers"][0]["scout_stalled_reason_code"] == "SCOUT_SCORING_TIMEOUT"
    assert status_payload["economic_fail_count_vs_evidence_fail_count"]["evidence_fail_count"] == 3

    open_payload = open_campaign("campaign_scout_stall")
    assert open_payload["item_blockers"][0]["primary_scout_blocker"] == "SCOUT_SCORING_TIMEOUT"
