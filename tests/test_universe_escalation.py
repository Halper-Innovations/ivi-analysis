from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.campaign import CAMPAIGN_DONE, run_campaign
from app.universe.escalation import (
    ACTION_ADD_TO_ACTIVE_WATCHLIST,
    ACTION_BUILD_REFRESHED_MEMO,
    ACTION_CLEAR_BLOCKERS,
    ACTION_DEFER_UNTIL_EVIDENCE_REFRESH,
    ACTION_NO_ACTION,
    ACTION_REBUILD_DEPTH_RUN,
    ACTION_RECHECK_PROMOTION,
    ACTION_RETRY_COMPANYFACTS_HYDRATION,
    ACTION_SCHEDULE_LIGHT_REFRESH,
    build_escalation_plan,
    open_escalation_plan,
    write_escalation_artifacts,
)
from app.universe.promotion import (
    LANE_1_HIGH_PRIORITY,
    LANE_2_RESEARCH_QUEUE,
    LANE_3_MONITOR,
    LANE_4_DEPRIORITIZED,
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


def _promotion_state_fixture() -> tuple[dict, dict]:
    rows = [
        {
            "ticker": "AAA",
            "priority_lane": LANE_1_HIGH_PRIORITY,
            "appearances_count": 2,
            "best_rank_seen": 1,
            "latest_value_gate_status": "PASS",
            "latest_implied_return_base": 0.32,
            "latest_primary_blocker": "NONE",
            "primary_blocker": "NONE",
            "mos_epv": 0.38,
            "yield_metric_used": "owner_earnings_yield_ev_3y",
            "owner_earnings_yield_ev_3y": 0.07,
            "memo_path": "memo/AAA.md",
            "source_runs": [
                {
                    "campaign_item": "software_core",
                    "universe_run_id": "campaign_escalation__software_core",
                    "batch_run_id": "campaign_escalation__software_core_depth_batch",
                }
            ],
            "risk_flags": [],
            "history": [
                {"campaign_item": "software_core", "universe_run_id": "campaign_escalation__software_core", "value_gate_status": "WATCH", "implied_return_base": 0.24, "primary_blocker": "NONE", "last_rank": 2},
                {"campaign_item": "software_core", "universe_run_id": "campaign_escalation__software_core", "value_gate_status": "PASS", "implied_return_base": 0.32, "primary_blocker": "NONE", "last_rank": 1},
            ],
        },
        {
            "ticker": "BBB",
            "priority_lane": LANE_2_RESEARCH_QUEUE,
            "appearances_count": 2,
            "best_rank_seen": 2,
            "latest_value_gate_status": "WATCH",
            "latest_implied_return_base": 0.21,
            "latest_primary_blocker": "PRICE_UNKNOWN",
            "primary_blocker": "PRICE_UNKNOWN",
            "mos_epv": 0.19,
            "yield_metric_used": "UNKNOWN",
            "owner_earnings_yield_ev_3y": "UNKNOWN",
            "memo_path": "memo/BBB.md",
            "source_runs": [
                {
                    "campaign_item": "software_core",
                    "universe_run_id": "campaign_escalation__software_core",
                    "batch_run_id": "campaign_escalation__software_core_depth_batch",
                }
            ],
            "risk_flags": ["PRICE_UNKNOWN"],
            "history": [
                {"campaign_item": "software_core", "universe_run_id": "campaign_escalation__software_core", "value_gate_status": "WATCH", "implied_return_base": 0.21, "primary_blocker": "PRICE_UNKNOWN", "last_rank": 2},
            ],
        },
        {
            "ticker": "CCC",
            "priority_lane": LANE_3_MONITOR,
            "appearances_count": 2,
            "best_rank_seen": 3,
            "latest_value_gate_status": "WATCH",
            "latest_implied_return_base": "UNKNOWN",
            "latest_primary_blocker": "NONE",
            "primary_blocker": "NONE",
            "mos_epv": "UNKNOWN",
            "yield_metric_used": "UNKNOWN",
            "owner_earnings_yield_ev_3y": "UNKNOWN",
            "memo_path": "memo/CCC.md",
            "source_runs": [
                {
                    "campaign_item": "software_growth",
                    "universe_run_id": "campaign_escalation__software_growth",
                    "batch_run_id": "campaign_escalation__software_growth_depth_batch",
                }
            ],
            "risk_flags": [],
            "history": [
                {"campaign_item": "software_growth", "universe_run_id": "campaign_escalation__software_growth", "value_gate_status": "WATCH", "implied_return_base": "UNKNOWN", "primary_blocker": "NONE", "last_rank": 3},
            ],
        },
        {
            "ticker": "DDD",
            "priority_lane": LANE_4_DEPRIORITIZED,
            "appearances_count": 2,
            "best_rank_seen": 7,
            "latest_value_gate_status": "FAIL",
            "latest_implied_return_base": 0.01,
            "latest_primary_blocker": "PRICE_UNKNOWN",
            "primary_blocker": "PRICE_UNKNOWN",
            "mos_epv": "UNKNOWN",
            "yield_metric_used": "UNKNOWN",
            "owner_earnings_yield_ev_3y": "UNKNOWN",
            "memo_path": "memo/DDD.md",
            "source_runs": [
                {
                    "campaign_item": "software_growth",
                    "universe_run_id": "campaign_escalation__software_growth",
                    "batch_run_id": "campaign_escalation__software_growth_depth_batch",
                }
            ],
            "risk_flags": ["PRICE_UNKNOWN", "TERMINAL_BLOCKER", "REPEATED_FAIL"],
            "history": [
                {"campaign_item": "software_growth", "universe_run_id": "campaign_escalation__software_growth", "value_gate_status": "FAIL", "implied_return_base": 0.01, "primary_blocker": "PRICE_UNKNOWN", "last_rank": 7},
            ],
        },
        {
            "ticker": "EEE",
            "priority_lane": LANE_3_MONITOR,
            "appearances_count": 1,
            "best_rank_seen": 4,
            "latest_value_gate_status": "WATCH",
            "latest_implied_return_base": "UNKNOWN",
            "latest_primary_blocker": "NONE",
            "primary_blocker": "NONE",
            "mos_epv": "UNKNOWN",
            "yield_metric_used": "UNKNOWN",
            "owner_earnings_yield_ev_3y": "UNKNOWN",
            "memo_path": "memo/EEE.md",
            "source_runs": [
                {
                    "campaign_item": "software_growth",
                    "universe_run_id": "campaign_escalation__software_growth",
                    "batch_run_id": "campaign_escalation__software_growth_depth_batch",
                }
            ],
            "risk_flags": [],
            "history": [],
        },
        {
            "ticker": "FFF",
            "priority_lane": LANE_3_MONITOR,
            "appearances_count": 1,
            "best_rank_seen": 4,
            "latest_value_gate_status": "WATCH",
            "latest_implied_return_base": "UNKNOWN",
            "latest_primary_blocker": "NONE",
            "primary_blocker": "NONE",
            "mos_epv": "UNKNOWN",
            "yield_metric_used": "UNKNOWN",
            "owner_earnings_yield_ev_3y": "UNKNOWN",
            "memo_path": "memo/FFF.md",
            "source_runs": [
                {
                    "campaign_item": "software_growth",
                    "universe_run_id": "campaign_escalation__software_growth",
                    "batch_run_id": "campaign_escalation__software_growth_depth_batch",
                }
            ],
            "risk_flags": [],
            "history": [],
        },
    ]
    promotion_state = {
        "campaign_run_id": "campaign_escalation",
        "generated_at": "2026-03-06T00:00:00+00:00",
        "lane_counts": {
            LANE_1_HIGH_PRIORITY: 1,
            LANE_2_RESEARCH_QUEUE: 1,
            LANE_3_MONITOR: 3,
            LANE_4_DEPRIORITIZED: 1,
        },
        "rows": rows,
    }
    priority_lanes = {
        "campaign_run_id": "campaign_escalation",
        "generated_at": "2026-03-06T00:00:00+00:00",
        "lane_1_high_priority": [rows[0]],
        "lane_2_research_queue": [rows[1]],
        "lane_3_monitor": [rows[2], rows[4], rows[5]],
        "lane_4_deprioritized": [rows[3]],
    }
    return promotion_state, priority_lanes


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
            {"ticker": "AAA", "rank": 1, "value_gate_status": "PASS", "implied_return_base": 0.31, "mos_epv": 0.35, "owner_yield": 0.07, "primary_blocker": "NONE", "composite_score_total": 81.0},
            {"ticker": "BBB", "rank": 2, "value_gate_status": "WATCH", "implied_return_base": "UNKNOWN", "mos_epv": "UNKNOWN", "owner_yield": "UNKNOWN", "primary_blocker": "PRICE_UNKNOWN", "composite_score_total": 45.0},
        ]
    else:
        candidates = [
            {"ticker": "CCC", "rank": 1, "value_gate_status": "WATCH", "implied_return_base": "UNKNOWN", "mos_epv": "UNKNOWN", "owner_yield": "UNKNOWN", "primary_blocker": "NONE", "composite_score_total": 35.0},
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


def test_lane_1_generates_rebuild_memo_and_watchlist_actions(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    promotion_state, priority_lanes = _promotion_state_fixture()

    payload = build_escalation_plan(
        "campaign_escalation",
        promotion_state,
        priority_lanes,
        config={"as_of_date": "2026-02-14", "source_campaign_file": "data/universe/sample_campaign.json", "top_n": 10, "policy": "value_first"},
    )
    aaa_actions = [row for row in payload["queue"] if row["ticker"] == "AAA"]

    assert [row["action_type"] for row in aaa_actions] == [
        ACTION_REBUILD_DEPTH_RUN,
        ACTION_BUILD_REFRESHED_MEMO,
        ACTION_ADD_TO_ACTIVE_WATCHLIST,
    ]
    assert aaa_actions[0]["recommended_command"].startswith("python -m app.cli universe-autopilot")


def test_lane_2_price_unknown_generates_clear_blocker_and_recheck(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    promotion_state, priority_lanes = _promotion_state_fixture()

    payload = build_escalation_plan(
        "campaign_escalation",
        promotion_state,
        priority_lanes,
        config={"as_of_date": "2026-02-14", "source_campaign_file": "data/universe/sample_campaign.json", "top_n": 10, "policy": "value_first"},
    )
    bbb_actions = [row for row in payload["queue"] if row["ticker"] == "BBB"]

    assert [row["action_type"] for row in bbb_actions] == [ACTION_CLEAR_BLOCKERS, ACTION_RECHECK_PROMOTION]
    assert bbb_actions[0]["action_reason"] == "HYDRATE_PRICE_SNAPSHOT"
    assert "--run-id campaign_escalation__lane2__bbb" in bbb_actions[0]["recommended_command"]


def test_retryable_facts_blocker_generates_deterministic_retry_action(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    promotion_state, priority_lanes = _promotion_state_fixture()
    bbb = next(row for row in promotion_state["rows"] if row["ticker"] == "BBB")
    bbb["facts_blocker_class"] = "FACTS_RETRYABLE_TIMEOUT"
    bbb["facts_blocker_retryable"] = True
    bbb["facts_blocker_terminal"] = False
    bbb["facts_blocker_partial_usable"] = False
    bbb["facts_missing_key_inputs"] = ["SHARES", "CFO", "CAPEX", "FCF"]
    bbb["facts_retry_recommended"] = True
    bbb["facts_blocker_reason_codes"] = ["SCOUT_FACTS_TIMEOUT"]
    bbb["facts_recommended_action"] = ACTION_RETRY_COMPANYFACTS_HYDRATION
    bbb["primary_fail_domain"] = "EVIDENCE"

    payload = build_escalation_plan(
        "campaign_escalation",
        promotion_state,
        priority_lanes,
        config={"as_of_date": "2026-02-14", "source_campaign_file": "data/universe/sample_campaign.json", "top_n": 10, "policy": "value_first"},
    )
    bbb_actions = [row for row in payload["queue"] if row["ticker"] == "BBB"]

    assert [row["action_type"] for row in bbb_actions] == [ACTION_RETRY_COMPANYFACTS_HYDRATION, ACTION_RECHECK_PROMOTION]
    assert bbb_actions[0]["facts_blocker_class"] == "FACTS_RETRYABLE_TIMEOUT"
    assert "companyfacts-fetch" in bbb_actions[0]["recommended_command"]
    assert "FACTS_BLOCKER_RETRY_PATH" in bbb_actions[0]["priority_support_codes"]


def test_lane_3_generates_light_refresh_only(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    promotion_state, priority_lanes = _promotion_state_fixture()

    payload = build_escalation_plan(
        "campaign_escalation",
        promotion_state,
        priority_lanes,
        config={"as_of_date": "2026-02-14", "source_campaign_file": "data/universe/sample_campaign.json", "top_n": 10, "policy": "value_first"},
    )
    ccc_actions = [row for row in payload["queue"] if row["ticker"] == "CCC"]

    assert [row["action_type"] for row in ccc_actions] == [ACTION_SCHEDULE_LIGHT_REFRESH]
    assert "universe-campaign" in ccc_actions[0]["recommended_command"]


def test_partial_usable_facts_monitor_name_defers_until_refresh(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    promotion_state, priority_lanes = _promotion_state_fixture()
    ccc = next(row for row in promotion_state["rows"] if row["ticker"] == "CCC")
    ccc["facts_blocker_class"] = "FACTS_PARTIAL_COVERAGE"
    ccc["facts_blocker_retryable"] = False
    ccc["facts_blocker_terminal"] = False
    ccc["facts_blocker_partial_usable"] = True
    ccc["facts_missing_key_inputs"] = ["CAPEX", "FCF"]
    ccc["facts_retry_recommended"] = True
    ccc["facts_blocker_reason_codes"] = ["TAG_MISS"]
    ccc["facts_recommended_action"] = ACTION_DEFER_UNTIL_EVIDENCE_REFRESH
    ccc["primary_fail_domain"] = "NONE"

    payload = build_escalation_plan(
        "campaign_escalation",
        promotion_state,
        priority_lanes,
        config={"as_of_date": "2026-02-14", "source_campaign_file": "data/universe/sample_campaign.json", "top_n": 10, "policy": "value_first"},
    )
    ccc_actions = [row for row in payload["queue"] if row["ticker"] == "CCC"]

    assert [row["action_type"] for row in ccc_actions] == [ACTION_DEFER_UNTIL_EVIDENCE_REFRESH]
    assert "universe-scout-open" in ccc_actions[0]["recommended_command"]


def test_lane_4_defaults_to_no_action(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    promotion_state, priority_lanes = _promotion_state_fixture()

    payload = build_escalation_plan(
        "campaign_escalation",
        promotion_state,
        priority_lanes,
        config={"as_of_date": "2026-02-14", "source_campaign_file": "data/universe/sample_campaign.json", "top_n": 10, "policy": "value_first"},
    )
    ddd_actions = [row for row in payload["queue"] if row["ticker"] == "DDD"]

    assert [row["action_type"] for row in ddd_actions] == [ACTION_NO_ACTION]
    assert "universe-campaign-open" in ddd_actions[0]["recommended_command"]


def test_escalation_queue_order_is_deterministic(monkeypatch, tmp_path):
    cfg, _universe = _init_cfg(monkeypatch, tmp_path)
    promotion_state, priority_lanes = _promotion_state_fixture()
    root = cfg.campaigns_dir / "campaign_escalation"
    _write_json(root / "promotion_state.json", promotion_state)
    _write_json(root / "priority_lanes.json", priority_lanes)
    _write_json(root / "campaign_state.json", {"campaign_run_id": "campaign_escalation", "as_of_date": "2026-02-14", "source_campaign_file": "data/universe/sample_campaign.json"})

    paths = write_escalation_artifacts("campaign_escalation")
    queue_payload = json.loads(Path(paths["escalation_queue_path"]).read_text(encoding="utf-8"))
    queue_pairs = [(row["ticker"], row["action_type"]) for row in queue_payload["rows"][:8]]

    assert queue_pairs[:8] == [
        ("AAA", ACTION_REBUILD_DEPTH_RUN),
        ("AAA", ACTION_BUILD_REFRESHED_MEMO),
        ("AAA", ACTION_ADD_TO_ACTIVE_WATCHLIST),
        ("BBB", ACTION_CLEAR_BLOCKERS),
        ("BBB", ACTION_RECHECK_PROMOTION),
        ("CCC", ACTION_SCHEDULE_LIGHT_REFRESH),
        ("EEE", ACTION_SCHEDULE_LIGHT_REFRESH),
        ("FFF", ACTION_SCHEDULE_LIGHT_REFRESH),
    ]


def test_campaign_integration_writes_escalation_artifacts_and_summary(monkeypatch, tmp_path):
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

    payload = run_campaign(spec, campaign_run_id="campaign_escalation_integration", resume=True)
    assert payload["status"] == CAMPAIGN_DONE
    assert Path(payload["escalation_plan_path"]).exists()
    assert Path(payload["escalation_queue_path"]).exists()
    assert Path(payload["escalation_summary_path"]).exists()

    summary = json.loads(Path(payload["campaign_summary_path"]).read_text(encoding="utf-8"))
    assert summary["escalation_queue_count"] >= 1
    assert "lane_to_action_counts" in summary
    assert "top_10_escalation_actions" in summary

    open_payload = open_escalation_plan("campaign_escalation_integration")
    assert open_payload["status"] == "OK"
    assert open_payload["escalation_queue_count"] >= 1


def test_escalation_cli_opener(monkeypatch, tmp_path):
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
        ["universe-campaign", "--campaign-file", str(spec_path), "--campaign-run-id", "campaign_escalation_cli", "--resume"],
    )
    assert build_cmd.exit_code == 0, build_cmd.output

    open_cmd = runner.invoke(
        app,
        ["universe-campaign-escalation-open", "--campaign-run-id", "campaign_escalation_cli"],
    )
    assert open_cmd.exit_code == 0, open_cmd.output
    payload = json.loads(open_cmd.output)
    assert payload["status"] == "OK"
    assert "top_10_escalation_actions" in payload
    assert "lane_to_action_counts" in payload
