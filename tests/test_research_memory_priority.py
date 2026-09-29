from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.campaign import CAMPAIGN_DONE, run_campaign
from app.universe.escalation import ACTION_SCHEDULE_LIGHT_REFRESH, build_escalation_plan
from app.universe.promotion import LANE_2_RESEARCH_QUEUE, LANE_3_MONITOR, build_promotion_state
from app.universe.research_memory import compute_memory_priority, write_research_memory


runner = CliRunner()


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "sample_universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker\nAAA\nBBB\n", encoding="utf-8")
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


def _memory_entry(
    ticker: str,
    *,
    appearances: int,
    gate: str,
    lane: str,
    blocker: str,
    implied_return_base="UNKNOWN",
    mos_epv="UNKNOWN",
    delta_latest: dict | None = None,
) -> dict:
    history = []
    for idx in range(appearances):
        history.append(
            {
                "campaign_run_id": f"{ticker.lower()}_run_{idx + 1:03d}",
                "timestamp": f"2026-03-{idx + 1:02d}T00:00:00+00:00",
                "value_gate_status": gate,
                "priority_lane": lane,
                "implied_return_base": implied_return_base,
                "mos_epv": mos_epv,
                "yield_metric_used": "UNKNOWN",
                "primary_blocker": blocker,
                "blocker_reason_code": blocker,
                "memo_path": "",
                "derived_from": [],
                "source_paths": {},
            }
        )
    return {
        "ticker": ticker,
        "first_seen_run_id": history[0]["campaign_run_id"],
        "last_seen_run_id": history[-1]["campaign_run_id"],
        "appearances_count": appearances,
        "latest": {
            "campaign_run_id": history[-1]["campaign_run_id"],
            "value_gate_status": gate,
            "priority_lane": lane,
            "implied_return_base": implied_return_base,
            "mos_epv": mos_epv,
            "yield_metric_used": "UNKNOWN",
            "primary_blocker": blocker,
            "blocker_reason_code": blocker,
            "memo_path": "",
            "derived_from": [],
            "source_paths": {},
        },
        "history": history,
        "delta_latest": delta_latest
        or {
            "gate_change": "UNCHANGED",
            "lane_change": "UNCHANGED",
            "implied_return_change": {"from": implied_return_base, "to": implied_return_base, "direction": "UNCHANGED"},
            "mos_epv_change": {"from": mos_epv, "to": mos_epv, "direction": "UNCHANGED"},
            "blocker_change": {"from": blocker, "to": blocker, "changed": False, "classification": "UNCHANGED"},
            "escalation_effect": "NOT_APPLICABLE",
        },
    }


def _seed_memory(entries: dict[str, dict]) -> None:
    write_research_memory(
        {
            "generated_at": "2026-03-06T00:00:00+00:00",
            "last_updated_campaign_run_id": "memory_seed",
            "updated_tickers": sorted(entries.keys()),
            "tickers": entries,
        }
    )


def _campaign_spec(path: Path, *, policy: str) -> dict:
    return {
        "as_of_date": "2026-02-14",
        "items": [
            {
                "label": "memory_priority_core",
                "universe_file": str(path),
                "max_runs": 1,
                "top_n": 5,
                "policy": policy,
            }
        ],
    }


def _write_child_outputs(cfg, *, universe_run_id: str) -> None:
    batch_run_id = f"{universe_run_id}_depth_batch"
    autopilot_dir = cfg.outputs_dir / "universe" / universe_run_id / "autopilot"
    batch_dir = cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id
    memo_dir = batch_dir / "memo_pack" / "memos"
    dossier_dir = batch_dir / "dossier_pack"
    memo_dir.mkdir(parents=True, exist_ok=True)
    dossier_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        {
            "ticker": "AAA",
            "rank": 1,
            "value_gate_status": "WATCH",
            "implied_return_base": 0.20,
            "mos_epv": 0.10,
            "owner_yield": 0.05,
            "primary_blocker": "NONE",
            "composite_score_total": 60.0,
        },
        {
            "ticker": "BBB",
            "rank": 2,
            "value_gate_status": "WATCH",
            "implied_return_base": 0.20,
            "mos_epv": 0.30,
            "owner_yield": 0.05,
            "primary_blocker": "NONE",
            "composite_score_total": 60.0,
        },
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
                    "yield_metric_used": "owner_earnings_yield_ev_3y",
                },
                "derived_from_index": [f"memo.{ticker}"],
                "rank_global": candidate["rank"],
                "value_gate_status": candidate["value_gate_status"],
                "implied_return_base": candidate["implied_return_base"],
                "primary_blocker": candidate["primary_blocker"],
            },
        )
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
                "mos_netnet": "UNKNOWN",
                "owner_earnings_yield_ev_3y": candidate["owner_yield"],
                "yield_metric_used": "owner_earnings_yield_ev_3y",
                "primary_blocker": candidate["primary_blocker"],
                "composite_score_total": candidate["composite_score_total"],
                "derived_from": [f"shortlist.{ticker}"],
            }
        )

    _write_json(batch_dir / "memo_pack" / "memo_pack_manifest.json", {"universe_run_id": universe_run_id, "batch_run_id": batch_run_id, "memo_count": len(memo_entries), "memos": memo_entries})
    _write_json(batch_dir / "global_shortlist.json", {"universe_run_id": universe_run_id, "batch_run_id": batch_run_id, "rows": shortlist_rows})
    _write_json(batch_dir / "global_rollup.json", {"run_count_total": 1, "run_count_done": 1, "candidate_count_ranked": len(shortlist_rows)})
    _write_json(dossier_dir / "dossier_pack_manifest.json", {"candidate_count": len(shortlist_rows)})
    _write_json(autopilot_dir / "watchlist_state.json", {"universe_run_id": universe_run_id, "ticker_count": len(watch_tickers), "tickers": watch_tickers})
    _write_json(autopilot_dir / "autopilot_state.json", {"universe_run_id": universe_run_id, "status": "DONE"})
    _write_json(autopilot_dir / "autopilot_summary.json", {"status": "DONE"})


def test_repeated_survivor_scoring_is_deterministic():
    priority = compute_memory_priority(
        _memory_entry(
            "AAA",
            appearances=3,
            gate="WATCH",
            lane=LANE_3_MONITOR,
            blocker="NONE",
        )
    )

    assert priority["repeated_survivor_score"] == 2
    assert priority["memory_priority_total"] == 2
    assert "REPEATED_SURVIVOR_3_PLUS" in priority["memory_priority_reason_codes"]


def test_unknown_to_known_increases_memory_priority():
    priority = compute_memory_priority(
        _memory_entry(
            "AAA",
            appearances=2,
            gate="WATCH",
            lane=LANE_3_MONITOR,
            blocker="NONE",
            delta_latest={
                "gate_change": "UNCHANGED",
                "lane_change": "UNCHANGED",
                "implied_return_change": {"from": "UNKNOWN", "to": 0.18, "direction": "UNKNOWN_TO_KNOWN"},
                "mos_epv_change": {"from": "UNKNOWN", "to": "UNKNOWN", "direction": "UNCHANGED"},
                "blocker_change": {"from": "NONE", "to": "NONE", "changed": False, "classification": "UNCHANGED"},
                "escalation_effect": "NOT_APPLICABLE",
            },
        )
    )

    assert priority["improvement_score"] == 2
    assert priority["memory_priority_total"] == 3
    assert "IMPLIED_RETURN_UNKNOWN_TO_KNOWN" in priority["memory_priority_reason_codes"]


def test_recurring_unchanged_blocker_reduces_priority():
    priority = compute_memory_priority(
        _memory_entry(
            "AAA",
            appearances=3,
            gate="FAIL",
            lane=LANE_3_MONITOR,
            blocker="FAIL_CONFIRMED",
        )
    )

    assert priority["deterioration_penalty"] == -1
    assert priority["memory_priority_total"] == -1
    assert "RECURRING_TERMINAL_BLOCKER" in priority["memory_priority_reason_codes"]


def test_gate_and_lane_upgrades_and_downgrades_are_scored_deterministically():
    improved = compute_memory_priority(
        _memory_entry(
            "AAA",
            appearances=2,
            gate="PASS",
            lane=LANE_2_RESEARCH_QUEUE,
            blocker="NONE",
            delta_latest={
                "gate_change": "UPGRADE",
                "lane_change": "UPGRADE",
                "implied_return_change": {"from": 0.10, "to": 0.10, "direction": "UNCHANGED"},
                "mos_epv_change": {"from": 0.12, "to": 0.12, "direction": "UNCHANGED"},
                "blocker_change": {"from": "NONE", "to": "NONE", "changed": False, "classification": "UNCHANGED"},
                "escalation_effect": "NOT_APPLICABLE",
            },
        )
    )
    deteriorated = compute_memory_priority(
        _memory_entry(
            "BBB",
            appearances=2,
            gate="FAIL",
            lane=LANE_3_MONITOR,
            blocker="NONE",
            delta_latest={
                "gate_change": "DOWNGRADE",
                "lane_change": "DOWNGRADE",
                "implied_return_change": {"from": 0.20, "to": "UNKNOWN", "direction": "KNOWN_TO_UNKNOWN"},
                "mos_epv_change": {"from": 0.18, "to": 0.18, "direction": "UNCHANGED"},
                "blocker_change": {"from": "NONE", "to": "NONE", "changed": False, "classification": "UNCHANGED"},
                "escalation_effect": "NOT_APPLICABLE",
            },
        )
    )

    assert improved["memory_priority_total"] == 5
    assert "GATE_UPGRADE" in improved["memory_priority_reason_codes"]
    assert "LANE_UPGRADE" in improved["memory_priority_reason_codes"]
    assert deteriorated["memory_priority_total"] == -6
    assert "GATE_DOWNGRADE" in deteriorated["memory_priority_reason_codes"]
    assert "LANE_DOWNGRADE" in deteriorated["memory_priority_reason_codes"]
    assert "IMPLIED_RETURN_KNOWN_TO_UNKNOWN" in deteriorated["memory_priority_reason_codes"]


def test_promotion_can_use_memory_support_for_lane_2(monkeypatch, tmp_path):
    _cfg, _universe = _init_cfg(monkeypatch, tmp_path)
    _seed_memory(
        {
            "AAA": _memory_entry(
                "AAA",
                appearances=3,
                gate="WATCH",
                lane=LANE_3_MONITOR,
                blocker="NONE",
                delta_latest={
                    "gate_change": "UPGRADE",
                    "lane_change": "UNCHANGED",
                    "implied_return_change": {"from": "UNKNOWN", "to": 0.16, "direction": "UNKNOWN_TO_KNOWN"},
                    "mos_epv_change": {"from": "UNKNOWN", "to": "UNKNOWN", "direction": "UNCHANGED"},
                    "blocker_change": {"from": "MISSING_EV", "to": "NONE", "changed": True, "classification": "IMPROVED"},
                    "escalation_effect": "IMPROVED",
                },
            )
        }
    )
    master_watchlist = {
        "campaign_run_id": "campaign_promote_memory",
        "tickers": {
            "AAA": {
                "latest_value_gate_status": "WATCH",
                "latest_implied_return_base": "UNKNOWN",
                "latest_primary_blocker": "NONE",
                "appearances_count": 3,
                "history": [
                    {"campaign_item": "item_a", "universe_run_id": "run_a", "value_gate_status": "WATCH", "implied_return_base": "UNKNOWN", "primary_blocker": "NONE", "last_rank": 3},
                    {"campaign_item": "item_b", "universe_run_id": "run_b", "value_gate_status": "WATCH", "implied_return_base": "UNKNOWN", "primary_blocker": "NONE", "last_rank": 3},
                    {"campaign_item": "item_c", "universe_run_id": "run_c", "value_gate_status": "WATCH", "implied_return_base": "UNKNOWN", "primary_blocker": "NONE", "last_rank": 3},
                ],
            }
        },
    }
    master_shortlist = {
        "campaign_run_id": "campaign_promote_memory",
        "rows": [
            {
                "ticker": "AAA",
                "source_runs": [{"campaign_item": "item_c", "universe_run_id": "run_c", "batch_run_id": "run_c_depth_batch"}],
                "best_rank_seen": 3,
                "value_gate_status": "WATCH",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": "UNKNOWN",
                "mos_epv": "UNKNOWN",
                "mos_netnet": "UNKNOWN",
                "owner_earnings_yield_ev_3y": "UNKNOWN",
                "yield_metric_used": "UNKNOWN",
                "primary_blocker": "NONE",
                "latest_primary_blocker": "NONE",
                "composite_score_total": 20.0,
                "memo_path": "",
                "derived_from": [],
            }
        ],
    }

    payload = build_promotion_state("campaign_promote_memory", master_watchlist, master_shortlist)
    row = payload["rows"][0]

    assert row["priority_lane"] == LANE_2_RESEARCH_QUEUE
    assert row["memory_priority_total"] > 0
    assert "MEMORY_PRIORITY_SUPPORT" in row["promotion_reason_codes"]


def test_master_shortlist_ranking_changes_under_value_first_memory(monkeypatch, tmp_path):
    cfg, universe = _init_cfg(monkeypatch, tmp_path)

    def _fake_run_autopilot(**kwargs):
        universe_run_id = kwargs["universe_run_id"]
        _write_child_outputs(cfg, universe_run_id=universe_run_id)
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

    seed = {
        "AAA": _memory_entry(
            "AAA",
            appearances=3,
            gate="WATCH",
            lane=LANE_3_MONITOR,
            blocker="NONE",
        )
    }
    _seed_memory(seed)
    default_payload = run_campaign(_campaign_spec(universe, policy="value_first"), campaign_run_id="campaign_priority_default", resume=True)
    assert default_payload["status"] == CAMPAIGN_DONE
    default_shortlist = json.loads((cfg.campaigns_dir / "campaign_priority_default" / "master_shortlist.json").read_text(encoding="utf-8"))
    assert [row["ticker"] for row in default_shortlist["rows"][:2]] == ["BBB", "AAA"]

    _seed_memory(seed)
    memory_payload = run_campaign(_campaign_spec(universe, policy="value_first_memory"), campaign_run_id="campaign_priority_memory", resume=True)
    assert memory_payload["status"] == CAMPAIGN_DONE
    memory_shortlist = json.loads((cfg.campaigns_dir / "campaign_priority_memory" / "master_shortlist.json").read_text(encoding="utf-8"))
    memory_summary = json.loads((cfg.campaigns_dir / "campaign_priority_memory" / "campaign_summary.json").read_text(encoding="utf-8"))

    assert [row["ticker"] for row in memory_shortlist["rows"][:2]] == ["AAA", "BBB"]
    assert memory_shortlist["rows"][0]["memory_priority_total"] > memory_shortlist["rows"][1]["memory_priority_total"]
    assert memory_summary["top_memory_priority_candidates"][0]["ticker"] == "AAA"

    priority_cmd = runner.invoke(app, ["universe-research-memory-priority-open"])
    assert priority_cmd.exit_code == 0, priority_cmd.output
    priority_payload = json.loads(priority_cmd.output)
    assert priority_payload["status"] == "OK"
    assert "top_memory_priority_candidates" in priority_payload


def test_escalation_queue_order_respects_memory_priority_within_lane(monkeypatch, tmp_path):
    _cfg, _universe = _init_cfg(monkeypatch, tmp_path)
    _seed_memory(
        {
            "AAA": _memory_entry(
                "AAA",
                appearances=3,
                gate="WATCH",
                lane=LANE_3_MONITOR,
                blocker="NONE",
            ),
            "BBB": _memory_entry(
                "BBB",
                appearances=3,
                gate="FAIL",
                lane=LANE_3_MONITOR,
                blocker="FAIL_CONFIRMED",
            ),
        }
    )
    promotion_rows = [
        {
            "ticker": "AAA",
            "priority_lane": LANE_3_MONITOR,
            "appearances_count": 3,
            "latest_value_gate_status": "WATCH",
            "latest_implied_return_base": 0.12,
            "implied_return_base": 0.12,
            "latest_primary_blocker": "NONE",
            "primary_blocker": "NONE",
            "mos_epv": 0.11,
            "yield_metric_used": "UNKNOWN",
            "memo_path": "",
            "source_runs": [{"campaign_item": "item_a", "universe_run_id": "run_a", "batch_run_id": "run_a_depth_batch"}],
            "risk_flags": [],
            "history": [],
        },
        {
            "ticker": "BBB",
            "priority_lane": LANE_3_MONITOR,
            "appearances_count": 3,
            "latest_value_gate_status": "WATCH",
            "latest_implied_return_base": 0.12,
            "implied_return_base": 0.12,
            "latest_primary_blocker": "FAIL_CONFIRMED",
            "primary_blocker": "FAIL_CONFIRMED",
            "mos_epv": 0.11,
            "yield_metric_used": "UNKNOWN",
            "memo_path": "",
            "source_runs": [{"campaign_item": "item_b", "universe_run_id": "run_b", "batch_run_id": "run_b_depth_batch"}],
            "risk_flags": [],
            "history": [],
        },
    ]
    promotion_state = {
        "campaign_run_id": "campaign_priority_queue",
        "rows": promotion_rows,
        "lane_counts": {LANE_3_MONITOR: 2},
    }
    priority_lanes = {
        "campaign_run_id": "campaign_priority_queue",
        "lane_3_monitor": promotion_rows,
    }

    payload = build_escalation_plan(
        "campaign_priority_queue",
        promotion_state,
        priority_lanes,
        config={"as_of_date": "2026-02-14", "source_campaign_file": "data/universe/sample_campaign.json", "top_n": 5, "policy": "value_first"},
    )
    queue_rows = [row for row in payload["queue"] if row["action_type"] == ACTION_SCHEDULE_LIGHT_REFRESH]

    assert [row["ticker"] for row in queue_rows] == ["AAA", "BBB"]
    assert queue_rows[0]["memory_priority_total"] > queue_rows[1]["memory_priority_total"]
