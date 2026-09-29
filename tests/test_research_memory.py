from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.campaign import CAMPAIGN_DONE, run_campaign
from app.universe.promotion import LANE_1_HIGH_PRIORITY, LANE_2_RESEARCH_QUEUE, LANE_3_MONITOR
from app.universe.research_memory import (
    ESCALATION_IMPROVED,
    GATE_UNCHANGED,
    GATE_UPGRADE,
    LANE_UPGRADE,
    build_ticker_delta,
    diff_research_memory,
    load_research_memory,
    open_research_memory,
    update_research_memory_from_campaign,
)


runner = CliRunner()


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "sample_universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker\nAAA\nNVDA\n", encoding="utf-8")
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


def _campaign_summary(cfg, campaign_run_id: str) -> dict:
    root = cfg.campaigns_dir / campaign_run_id
    return {
        "campaign_run_id": campaign_run_id,
        "campaign_summary_path": str(root / "campaign_summary.json"),
        "master_shortlist_json_path": str(root / "master_shortlist.json"),
        "master_watchlist_state_path": str(root / "master_watchlist_state.json"),
        "promotion_state_path": str(root / "promotion_state.json"),
    }


def _artifact_bundle(
    cfg,
    *,
    campaign_run_id: str,
    ticker: str,
    gate: str,
    lane: str,
    implied_return_base,
    mos_epv,
    blocker: str,
    memo_path: str,
) -> tuple[dict, dict, dict, dict]:
    summary = _campaign_summary(cfg, campaign_run_id)
    master_shortlist = {
        "campaign_run_id": campaign_run_id,
        "rows": [
            {
                "ticker": ticker,
                "value_gate_status": gate,
                "latest_value_gate_status": gate,
                "implied_return_base": implied_return_base,
                "mos_epv": mos_epv,
                "yield_metric_used": "owner_earnings_yield_ev_3y" if implied_return_base != "UNKNOWN" else "UNKNOWN",
                "primary_blocker": blocker,
                "latest_primary_blocker": blocker,
                "memo_path": memo_path,
                "derived_from": [f"shortlist.{campaign_run_id}.{ticker}"],
            }
        ],
    }
    master_watchlist_state = {
        "campaign_run_id": campaign_run_id,
        "ticker_count": 1,
        "tickers": {
            ticker: {
                "latest_value_gate_status": gate,
                "latest_implied_return_base": implied_return_base,
                "latest_primary_blocker": blocker,
                "appearances_count": 1,
            }
        },
    }
    promotion_state = {
        "campaign_run_id": campaign_run_id,
        "rows": [
            {
                "ticker": ticker,
                "latest_value_gate_status": gate,
                "priority_lane": lane,
                "implied_return_base": implied_return_base,
                "mos_epv": mos_epv,
                "yield_metric_used": "owner_earnings_yield_ev_3y" if implied_return_base != "UNKNOWN" else "UNKNOWN",
                "latest_primary_blocker": blocker,
                "primary_blocker": blocker,
                "memo_path": memo_path,
                "derived_from": [f"promotion.{campaign_run_id}.{ticker}"],
            }
        ],
    }
    return summary, master_shortlist, master_watchlist_state, promotion_state


def _campaign_spec(path: Path) -> dict:
    return {
        "as_of_date": "2026-02-14",
        "items": [
            {
                "label": "memory_core",
                "universe_file": str(path),
                "max_runs": 1,
                "top_n": 5,
                "policy": "value_first",
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
            "value_gate_status": "PASS",
            "implied_return_base": 0.30,
            "mos_epv": 0.35,
            "owner_yield": 0.07,
            "primary_blocker": "NONE",
            "composite_score_total": 80.0,
        },
        {
            "ticker": "NVDA",
            "rank": 2,
            "value_gate_status": "WATCH",
            "implied_return_base": 0.18,
            "mos_epv": 0.19,
            "owner_yield": 0.04,
            "primary_blocker": "PRICE_UNKNOWN",
            "composite_score_total": 62.0,
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
                    "mos_netnet": 0.05,
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
                "owner_earnings_yield_ev_3y": candidate["owner_yield"],
                "yield_metric_used": "owner_earnings_yield_ev_3y",
                "primary_blocker": candidate["primary_blocker"],
                "composite_score_total": candidate["composite_score_total"],
                "derived_from": [f"shortlist.{ticker}"],
            }
        )

    memo_manifest_path = batch_dir / "memo_pack" / "memo_pack_manifest.json"
    _write_json(memo_manifest_path, {"universe_run_id": universe_run_id, "batch_run_id": batch_run_id, "memo_count": len(memo_entries), "memos": memo_entries})
    _write_json(batch_dir / "global_shortlist.json", {"universe_run_id": universe_run_id, "batch_run_id": batch_run_id, "rows": shortlist_rows})
    _write_json(batch_dir / "global_rollup.json", {"run_count_total": 1, "run_count_done": 1, "candidate_count_ranked": len(shortlist_rows)})
    (batch_dir / "global_shortlist.md").write_text("# shortlist\n", encoding="utf-8")
    _write_json(dossier_dir / "dossier_pack_manifest.json", {"candidate_count": len(shortlist_rows)})
    _write_json(autopilot_dir / "watchlist_state.json", {"universe_run_id": universe_run_id, "ticker_count": len(watch_tickers), "tickers": watch_tickers})
    _write_json(autopilot_dir / "autopilot_state.json", {"universe_run_id": universe_run_id, "status": "DONE"})
    _write_json(autopilot_dir / "autopilot_summary.json", {"status": "DONE", "memo_pack_manifest_path": str(memo_manifest_path)})


def test_research_memory_new_ticker_insertion(monkeypatch, tmp_path):
    cfg, _universe = _init_cfg(monkeypatch, tmp_path)
    summary, master_shortlist, master_watchlist_state, promotion_state = _artifact_bundle(
        cfg,
        campaign_run_id="campaign_mem_001",
        ticker="NVDA",
        gate="WATCH",
        lane=LANE_3_MONITOR,
        implied_return_base="UNKNOWN",
        mos_epv="UNKNOWN",
        blocker="PRICE_UNKNOWN",
        memo_path="memo/NVDA.md",
    )

    payload = update_research_memory_from_campaign(
        "campaign_mem_001",
        summary,
        master_shortlist,
        master_watchlist_state,
        promotion_state,
    )

    memory = load_research_memory()
    entry = memory["tickers"]["NVDA"]
    assert entry["first_seen_run_id"] == "campaign_mem_001"
    assert entry["appearances_count"] == 1
    assert entry["history"][0]["campaign_run_id"] == "campaign_mem_001"
    assert entry["delta_latest"]["gate_change"] == GATE_UNCHANGED
    assert Path(payload["research_memory_path"]).exists()
    assert Path(payload["research_memory_summary_path"]).exists()


def test_research_memory_repeated_ticker_appends_history(monkeypatch, tmp_path):
    cfg, _universe = _init_cfg(monkeypatch, tmp_path)
    first = _artifact_bundle(
        cfg,
        campaign_run_id="campaign_mem_001",
        ticker="AAA",
        gate="WATCH",
        lane=LANE_3_MONITOR,
        implied_return_base=0.10,
        mos_epv=0.12,
        blocker="PRICE_UNKNOWN",
        memo_path="memo/AAA_v1.md",
    )
    second = _artifact_bundle(
        cfg,
        campaign_run_id="campaign_mem_002",
        ticker="AAA",
        gate="PASS",
        lane=LANE_1_HIGH_PRIORITY,
        implied_return_base=0.22,
        mos_epv=0.30,
        blocker="NONE",
        memo_path="memo/AAA_v2.md",
    )

    update_research_memory_from_campaign("campaign_mem_001", *first)
    update_research_memory_from_campaign("campaign_mem_002", *second)

    diff_payload = diff_research_memory("AAA")
    memory = load_research_memory()
    entry = memory["tickers"]["AAA"]
    assert entry["appearances_count"] == 2
    assert len(entry["history"]) == 2
    assert entry["last_seen_run_id"] == "campaign_mem_002"
    assert diff_payload["prior_state"]["campaign_run_id"] == "campaign_mem_001"
    assert diff_payload["latest_state"]["campaign_run_id"] == "campaign_mem_002"


def test_research_memory_gate_and_lane_delta_logic_is_deterministic():
    delta = build_ticker_delta(
        {
            "campaign_run_id": "campaign_prev",
            "value_gate_status": "WATCH",
            "priority_lane": LANE_3_MONITOR,
        },
        {
            "campaign_run_id": "campaign_new",
            "value_gate_status": "PASS",
            "priority_lane": LANE_1_HIGH_PRIORITY,
        },
    )

    assert delta["gate_change"] == GATE_UPGRADE
    assert delta["lane_change"] == LANE_UPGRADE


def test_research_memory_unknown_to_known_numeric_transitions():
    delta = build_ticker_delta(
        {
            "campaign_run_id": "campaign_prev",
            "implied_return_base": "UNKNOWN",
            "mos_epv": "UNKNOWN",
        },
        {
            "campaign_run_id": "campaign_new",
            "implied_return_base": 0.18,
            "mos_epv": 0.27,
        },
    )

    assert delta["implied_return_change"]["direction"] == "UNKNOWN_TO_KNOWN"
    assert delta["mos_epv_change"]["direction"] == "UNKNOWN_TO_KNOWN"


def test_research_memory_blocker_change_classification():
    delta = build_ticker_delta(
        {
            "campaign_run_id": "campaign_prev",
            "primary_blocker": "FAIL_CONFIRMED",
        },
        {
            "campaign_run_id": "campaign_new",
            "primary_blocker": "MISSING_EV",
        },
    )

    assert delta["blocker_change"]["changed"] is True
    assert delta["blocker_change"]["classification"] == "IMPROVED"


def test_research_memory_escalation_effect_classification():
    delta = build_ticker_delta(
        {
            "campaign_run_id": "campaign_prev",
            "value_gate_status": "WATCH",
            "priority_lane": LANE_2_RESEARCH_QUEUE,
            "implied_return_base": 0.12,
        },
        {
            "campaign_run_id": "campaign_new",
            "value_gate_status": "PASS",
            "priority_lane": LANE_1_HIGH_PRIORITY,
            "implied_return_base": 0.25,
            "escalation_rows": [
                {
                    "ticker": "AAA",
                    "lane_before": LANE_2_RESEARCH_QUEUE,
                    "lane_after": LANE_1_HIGH_PRIORITY,
                    "status": "DONE",
                }
            ],
        },
    )

    assert delta["escalation_effect"] == ESCALATION_IMPROVED


def test_campaign_integration_writes_research_memory_artifacts(monkeypatch, tmp_path):
    cfg, universe = _init_cfg(monkeypatch, tmp_path)
    spec = _campaign_spec(universe)
    spec_path = cfg.data_dir / "universe" / "sample_campaign.json"
    _write_json(spec_path, spec)

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

    payload = run_campaign(spec, campaign_run_id="campaign_memory_integration", resume=True)
    assert payload["status"] == CAMPAIGN_DONE

    campaign_summary = json.loads((cfg.campaigns_dir / "campaign_memory_integration" / "campaign_summary.json").read_text(encoding="utf-8"))
    assert campaign_summary["research_memory_path"].endswith("research_memory.json")
    assert campaign_summary["research_memory_summary_path"].endswith("research_memory_summary.json")
    assert Path(campaign_summary["research_memory_path"]).exists()
    assert Path(campaign_summary["research_memory_summary_path"]).exists()

    summary_cmd = runner.invoke(app, ["universe-research-memory-summary"])
    assert summary_cmd.exit_code == 0, summary_cmd.output
    summary_payload = json.loads(summary_cmd.output)
    assert summary_payload["status"] == "OK"
    assert summary_payload["ticker_count"] >= 2

    open_cmd = runner.invoke(app, ["universe-research-memory-open", "--ticker", "NVDA"])
    assert open_cmd.exit_code == 0, open_cmd.output
    open_payload = json.loads(open_cmd.output)
    assert open_payload["status"] == "OK"
    assert open_payload["entry"]["latest"]["campaign_run_id"] == "campaign_memory_integration"

    diff_cmd = runner.invoke(app, ["universe-research-memory-diff", "--ticker", "NVDA"])
    assert diff_cmd.exit_code == 0, diff_cmd.output
    diff_payload = json.loads(diff_cmd.output)
    assert diff_payload["status"] == "OK"
    assert diff_payload["latest_state"]["campaign_run_id"] == "campaign_memory_integration"

    open_payload_fn = open_research_memory("NVDA")
    assert open_payload_fn["status"] == "OK"
