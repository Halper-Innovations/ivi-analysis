from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.campaign import CAMPAIGN_DONE, run_campaign
from app.universe.promotion import (
    LANE_1_HIGH_PRIORITY,
    LANE_2_RESEARCH_QUEUE,
    LANE_3_MONITOR,
    LANE_4_DEPRIORITIZED,
    build_promotion_state,
    open_priority_lanes,
    open_promotion_state,
    write_promotion_artifacts,
)


runner = CliRunner()


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "sample_universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker\nAAA\nBBB\nCCC\nDDD\nEEE\nFFF\n", encoding="utf-8")
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


def _promotion_payloads() -> tuple[dict, dict]:
    master_shortlist = {
        "campaign_run_id": "campaign_promote",
        "rows": [
            {
                "ticker": "AAA",
                "source_runs": [
                    {"campaign_item": "software_core", "universe_run_id": "campaign_promote__software_core", "batch_run_id": "campaign_promote__software_core_depth_batch"},
                    {"campaign_item": "software_growth", "universe_run_id": "campaign_promote__software_growth", "batch_run_id": "campaign_promote__software_growth_depth_batch"},
                ],
                "best_rank_seen": 1,
                "value_gate_status": "PASS",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": 0.27,
                "mos_epv": 0.35,
                "mos_netnet": 0.04,
                "owner_earnings_yield_ev_3y": 0.06,
                "yield_metric_used": "owner_earnings_yield_ev_3y",
                "primary_blocker": "NONE",
                "latest_primary_blocker": "NONE",
                "composite_score_total": 82.0,
                "memo_path": "memo/AAA.md",
                "derived_from": ["shortlist.AAA"],
            },
            {
                "ticker": "BBB",
                "source_runs": [
                    {"campaign_item": "software_core", "universe_run_id": "campaign_promote__software_core", "batch_run_id": "campaign_promote__software_core_depth_batch"},
                    {"campaign_item": "software_growth", "universe_run_id": "campaign_promote__software_growth", "batch_run_id": "campaign_promote__software_growth_depth_batch"},
                ],
                "best_rank_seen": 2,
                "value_gate_status": "WATCH",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": 0.18,
                "mos_epv": 0.16,
                "mos_netnet": "UNKNOWN",
                "owner_earnings_yield_ev_3y": "UNKNOWN",
                "yield_metric_used": "UNKNOWN",
                "primary_blocker": "MISSING_EV",
                "latest_primary_blocker": "MISSING_EV",
                "composite_score_total": 61.0,
                "memo_path": "memo/BBB.md",
                "derived_from": ["shortlist.BBB"],
            },
            {
                "ticker": "CCC",
                "source_runs": [
                    {"campaign_item": "software_core", "universe_run_id": "campaign_promote__software_core", "batch_run_id": "campaign_promote__software_core_depth_batch"},
                    {"campaign_item": "software_growth", "universe_run_id": "campaign_promote__software_growth", "batch_run_id": "campaign_promote__software_growth_depth_batch"},
                ],
                "best_rank_seen": 6,
                "value_gate_status": "FAIL",
                "latest_value_gate_status": "FAIL",
                "implied_return_base": 0.01,
                "mos_epv": "UNKNOWN",
                "mos_netnet": "UNKNOWN",
                "owner_earnings_yield_ev_3y": "UNKNOWN",
                "yield_metric_used": "UNKNOWN",
                "primary_blocker": "PRICE_UNKNOWN",
                "latest_primary_blocker": "PRICE_UNKNOWN",
                "composite_score_total": 22.0,
                "memo_path": "memo/CCC.md",
                "derived_from": ["shortlist.CCC"],
            },
            {
                "ticker": "DDD",
                "source_runs": [
                    {"campaign_item": "software_core", "universe_run_id": "campaign_promote__software_core", "batch_run_id": "campaign_promote__software_core_depth_batch"},
                ],
                "best_rank_seen": 4,
                "value_gate_status": "WATCH",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": "UNKNOWN",
                "mos_epv": "UNKNOWN",
                "mos_netnet": "UNKNOWN",
                "owner_earnings_yield_ev_3y": "UNKNOWN",
                "yield_metric_used": "UNKNOWN",
                "primary_blocker": "NONE",
                "latest_primary_blocker": "NONE",
                "composite_score_total": 30.0,
                "memo_path": "memo/DDD.md",
                "derived_from": ["shortlist.DDD"],
            },
            {
                "ticker": "EEE",
                "source_runs": [
                    {"campaign_item": "software_growth", "universe_run_id": "campaign_promote__software_growth", "batch_run_id": "campaign_promote__software_growth_depth_batch"},
                ],
                "best_rank_seen": 5,
                "value_gate_status": "WATCH",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": 0.11,
                "mos_epv": 0.08,
                "mos_netnet": "UNKNOWN",
                "owner_earnings_yield_ev_3y": "UNKNOWN",
                "yield_metric_used": "UNKNOWN",
                "primary_blocker": "MISSING_FCF",
                "latest_primary_blocker": "MISSING_FCF",
                "composite_score_total": 59.0,
                "memo_path": "memo/EEE.md",
                "derived_from": ["shortlist.EEE"],
            },
            {
                "ticker": "FFF",
                "source_runs": [
                    {"campaign_item": "software_growth", "universe_run_id": "campaign_promote__software_growth", "batch_run_id": "campaign_promote__software_growth_depth_batch"},
                ],
                "best_rank_seen": 5,
                "value_gate_status": "WATCH",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": 0.11,
                "mos_epv": 0.08,
                "mos_netnet": "UNKNOWN",
                "owner_earnings_yield_ev_3y": "UNKNOWN",
                "yield_metric_used": "UNKNOWN",
                "primary_blocker": "MISSING_FCF",
                "latest_primary_blocker": "MISSING_FCF",
                "composite_score_total": 59.0,
                "memo_path": "memo/FFF.md",
                "derived_from": ["shortlist.FFF"],
            },
        ],
    }
    master_watchlist = {
        "campaign_run_id": "campaign_promote",
        "tickers": {
            "AAA": {
                "latest_value_gate_status": "WATCH",
                "latest_implied_return_base": 0.27,
                "latest_primary_blocker": "NONE",
                "appearances_count": 2,
                "history": [
                    {"campaign_item": "software_core", "universe_run_id": "campaign_promote__software_core", "value_gate_status": "PASS", "implied_return_base": 0.30, "primary_blocker": "NONE", "last_rank": 1},
                    {"campaign_item": "software_growth", "universe_run_id": "campaign_promote__software_growth", "value_gate_status": "WATCH", "implied_return_base": 0.27, "primary_blocker": "NONE", "last_rank": 1},
                ],
            },
            "BBB": {
                "latest_value_gate_status": "WATCH",
                "latest_implied_return_base": 0.18,
                "latest_primary_blocker": "MISSING_EV",
                "appearances_count": 2,
                "history": [
                    {"campaign_item": "software_core", "universe_run_id": "campaign_promote__software_core", "value_gate_status": "WATCH", "implied_return_base": 0.17, "primary_blocker": "MISSING_EV", "last_rank": 2},
                    {"campaign_item": "software_growth", "universe_run_id": "campaign_promote__software_growth", "value_gate_status": "WATCH", "implied_return_base": 0.18, "primary_blocker": "MISSING_EV", "last_rank": 2},
                ],
            },
            "CCC": {
                "latest_value_gate_status": "FAIL",
                "latest_implied_return_base": 0.01,
                "latest_primary_blocker": "PRICE_UNKNOWN",
                "appearances_count": 2,
                "history": [
                    {"campaign_item": "software_core", "universe_run_id": "campaign_promote__software_core", "value_gate_status": "FAIL", "implied_return_base": 0.02, "primary_blocker": "PRICE_UNKNOWN", "last_rank": 5},
                    {"campaign_item": "software_growth", "universe_run_id": "campaign_promote__software_growth", "value_gate_status": "FAIL", "implied_return_base": 0.01, "primary_blocker": "PRICE_UNKNOWN", "last_rank": 6},
                ],
            },
            "DDD": {
                "latest_value_gate_status": "WATCH",
                "latest_implied_return_base": "UNKNOWN",
                "latest_primary_blocker": "NONE",
                "appearances_count": 1,
                "history": [
                    {"campaign_item": "software_core", "universe_run_id": "campaign_promote__software_core", "value_gate_status": "WATCH", "implied_return_base": "UNKNOWN", "primary_blocker": "NONE", "last_rank": 4},
                ],
            },
            "EEE": {
                "latest_value_gate_status": "WATCH",
                "latest_implied_return_base": 0.11,
                "latest_primary_blocker": "MISSING_FCF",
                "appearances_count": 1,
                "history": [
                    {"campaign_item": "software_growth", "universe_run_id": "campaign_promote__software_growth", "value_gate_status": "WATCH", "implied_return_base": 0.11, "primary_blocker": "MISSING_FCF", "last_rank": 5},
                ],
            },
            "FFF": {
                "latest_value_gate_status": "WATCH",
                "latest_implied_return_base": 0.11,
                "latest_primary_blocker": "MISSING_FCF",
                "appearances_count": 1,
                "history": [
                    {"campaign_item": "software_growth", "universe_run_id": "campaign_promote__software_growth", "value_gate_status": "WATCH", "implied_return_base": 0.11, "primary_blocker": "MISSING_FCF", "last_rank": 5},
                ],
            },
        },
    }
    return master_watchlist, master_shortlist


def _campaign_spec(path: Path) -> dict:
    return {
        "as_of_date": "2026-02-14",
        "items": [
            {"label": "software_core", "universe_file": str(path), "max_runs": 2, "top_n": 10, "policy": "value_first"},
            {"label": "software_growth", "universe_file": str(path), "max_runs": 2, "top_n": 10, "policy": "value_first"},
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
            {"ticker": "AAA", "rank": 1, "value_gate_status": "PASS", "implied_return_base": 0.30, "mos_epv": 0.35, "owner_yield": 0.06, "primary_blocker": "NONE", "composite_score_total": 80.0},
            {"ticker": "BBB", "rank": 2, "value_gate_status": "WATCH", "implied_return_base": 0.18, "mos_epv": 0.16, "owner_yield": "UNKNOWN", "primary_blocker": "MISSING_EV", "composite_score_total": 61.0},
        ]
    else:
        candidates = [
            {"ticker": "AAA", "rank": 1, "value_gate_status": "WATCH", "implied_return_base": 0.27, "mos_epv": 0.34, "owner_yield": 0.06, "primary_blocker": "NONE", "composite_score_total": 82.0},
            {"ticker": "CCC", "rank": 2, "value_gate_status": "FAIL", "implied_return_base": 0.01, "mos_epv": "UNKNOWN", "owner_yield": "UNKNOWN", "primary_blocker": "PRICE_UNKNOWN", "composite_score_total": 20.0},
        ]

    memo_entries = []
    watch_tickers = {}
    shortlist_rows = []
    for candidate in candidates:
        ticker = candidate["ticker"]
        memo_json_path = memo_dir / f"{ticker}.json"
        memo_md_path = memo_dir / f"{ticker}.md"
        _write_json(
            memo_json_path,
            {
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
                    "mos_netnet": "UNKNOWN",
                },
                "owner_earnings_and_yield": {
                    "owner_earnings_yield_ev_3y": candidate["owner_yield"],
                    "yield_metric_used": "owner_earnings_yield_ev_3y" if candidate["owner_yield"] != "UNKNOWN" else "UNKNOWN",
                },
                "derived_from_index": [f"memo.{label}.{ticker}"],
                "rank_global": candidate["rank"],
                "value_gate_status": candidate["value_gate_status"],
                "implied_return_base": candidate["implied_return_base"],
                "primary_blocker": candidate["primary_blocker"],
            },
        )
        memo_md_path.write_text(f"# {ticker}\n", encoding="utf-8")
        memo_entries.append({"ticker": ticker, "rank_global": candidate["rank"], "memo_json_path": str(memo_json_path), "memo_md_path": str(memo_md_path)})
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
                "mos_netnet": "UNKNOWN",
                "owner_earnings_yield_ev_3y": candidate["owner_yield"],
                "yield_metric_used": "owner_earnings_yield_ev_3y" if candidate["owner_yield"] != "UNKNOWN" else "UNKNOWN",
                "primary_blocker": candidate["primary_blocker"],
                "composite_score_total": candidate["composite_score_total"],
                "derived_from": [f"shortlist.{label}.{ticker}"],
            }
        )

    _write_json(batch_dir / "memo_pack" / "memo_pack_manifest.json", {"universe_run_id": universe_run_id, "batch_run_id": batch_run_id, "memo_count": len(memo_entries), "memos": memo_entries})
    _write_json(batch_dir / "global_shortlist.json", {"universe_run_id": universe_run_id, "batch_run_id": batch_run_id, "rows": shortlist_rows})
    _write_json(batch_dir / "global_rollup.json", {"run_count_total": 1, "run_count_done": 1, "candidate_count_ranked": len(shortlist_rows)})
    _write_json(dossier_dir / "dossier_pack_manifest.json", {"candidate_count": len(shortlist_rows)})
    _write_json(autopilot_dir / "watchlist_state.json", {"universe_run_id": universe_run_id, "ticker_count": len(watch_tickers), "tickers": watch_tickers})
    _write_json(autopilot_dir / "autopilot_state.json", {"universe_run_id": universe_run_id, "status": "DONE"})
    _write_json(autopilot_dir / "autopilot_summary.json", {"status": "DONE"})


def test_build_promotion_state_classifies_lanes_deterministically(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    master_watchlist, master_shortlist = _promotion_payloads()

    payload = build_promotion_state("campaign_promote", master_watchlist, master_shortlist)
    rows = {row["ticker"]: row for row in payload["rows"]}

    assert rows["AAA"]["priority_lane"] == LANE_1_HIGH_PRIORITY
    assert "REPEATED_SURVIVOR" in rows["AAA"]["strength_flags"]
    assert rows["BBB"]["priority_lane"] == LANE_2_RESEARCH_QUEUE
    assert "HYDRABLE_BLOCKER" in rows["BBB"]["promotion_reason_codes"]
    assert rows["DDD"]["priority_lane"] == LANE_3_MONITOR
    assert rows["CCC"]["priority_lane"] == LANE_4_DEPRIORITIZED
    assert "TERMINAL_BLOCKER" in rows["CCC"]["risk_flags"]
    assert payload["lane_counts"][LANE_1_HIGH_PRIORITY] == 1


def test_promotion_state_includes_facts_blocker_support_fields(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    master_watchlist, master_shortlist = _promotion_payloads()
    bbb = next(row for row in master_shortlist["rows"] if row["ticker"] == "BBB")
    bbb["facts_blocker_class"] = "FACTS_RETRYABLE_TIMEOUT"
    bbb["facts_blocker_retryable"] = True
    bbb["facts_blocker_terminal"] = False
    bbb["facts_blocker_partial_usable"] = False
    bbb["facts_missing_key_inputs"] = ["SHARES", "CFO", "CAPEX", "FCF"]
    bbb["facts_retry_recommended"] = True
    bbb["facts_blocker_reason_codes"] = ["SCOUT_FACTS_TIMEOUT"]
    bbb["facts_recommended_action"] = "RETRY_COMPANYFACTS_HYDRATION"
    bbb["fail_due_to_missing_evidence"] = True
    bbb["fail_due_to_economic_weakness"] = False
    bbb["primary_fail_domain"] = "EVIDENCE"

    payload = build_promotion_state("campaign_promote_facts", master_watchlist, master_shortlist)
    row = next(candidate for candidate in payload["rows"] if candidate["ticker"] == "BBB")

    assert row["facts_blocker_class"] == "FACTS_RETRYABLE_TIMEOUT"
    assert row["facts_blocker_retryable"] is True
    assert row["facts_recommended_action"] == "RETRY_COMPANYFACTS_HYDRATION"
    assert "RETRYABLE_FACTS_BLOCKER" in row["promotion_reason_codes"]


def test_promotion_candidate_order_uses_deterministic_tie_break(monkeypatch, tmp_path):
    cfg, _universe = _init_cfg(monkeypatch, tmp_path)
    master_watchlist, master_shortlist = _promotion_payloads()
    root = cfg.campaigns_dir / "campaign_promote"
    _write_json(root / "master_watchlist_state.json", master_watchlist)
    _write_json(root / "master_shortlist.json", master_shortlist)

    paths = write_promotion_artifacts("campaign_promote")
    candidates = json.loads(Path(paths["promotion_candidates_path"]).read_text(encoding="utf-8"))
    tickers = [row["ticker"] for row in candidates["rows"]]

    assert tickers[:4] == ["AAA", "BBB", "EEE", "FFF"]


def test_campaign_integration_writes_promotion_artifacts(monkeypatch, tmp_path):
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

    payload = run_campaign(spec, campaign_run_id="campaign_promotion_integration", resume=True)
    assert payload["status"] == CAMPAIGN_DONE
    assert Path(payload["promotion_state_path"]).exists()
    assert Path(payload["promotion_candidates_path"]).exists()
    assert Path(payload["priority_lanes_path"]).exists()

    promotion_open = open_promotion_state("campaign_promotion_integration")
    assert promotion_open["status"] == "OK"
    assert promotion_open["lane_counts"][LANE_1_HIGH_PRIORITY] >= 1
    lanes_open = open_priority_lanes("campaign_promotion_integration")
    assert lanes_open["status"] == "OK"
    assert lanes_open["top_by_lane"]["lane_1_high_priority"][0]["ticker"] == "AAA"


def test_promotion_cli_openers_print_expected_fields(monkeypatch, tmp_path):
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
        ["universe-campaign", "--campaign-file", str(spec_path), "--campaign-run-id", "campaign_promotion_cli", "--resume"],
    )
    assert build_cmd.exit_code == 0, build_cmd.output

    promotion_cmd = runner.invoke(
        app,
        ["universe-campaign-promotion-open", "--campaign-run-id", "campaign_promotion_cli"],
    )
    assert promotion_cmd.exit_code == 0, promotion_cmd.output
    promotion_payload = json.loads(promotion_cmd.output)
    assert promotion_payload["status"] == "OK"
    assert "lane_counts" in promotion_payload
    assert "top_blockers_preventing_lane_1" in promotion_payload

    lanes_cmd = runner.invoke(
        app,
        ["universe-campaign-priority-lanes-open", "--campaign-run-id", "campaign_promotion_cli"],
    )
    assert lanes_cmd.exit_code == 0, lanes_cmd.output
    lanes_payload = json.loads(lanes_cmd.output)
    assert lanes_payload["status"] == "OK"
    assert "top_by_lane" in lanes_payload
    assert "lane_1_high_priority" in lanes_payload["top_by_lane"]
