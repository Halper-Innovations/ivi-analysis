from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.escalation import (
    ACTION_ADD_TO_ACTIVE_WATCHLIST,
    ACTION_BUILD_REFRESHED_MEMO,
    ACTION_CLEAR_BLOCKERS,
    ACTION_REBUILD_DEPTH_RUN,
    ACTION_RECHECK_PROMOTION,
    ACTION_SCHEDULE_LIGHT_REFRESH,
)
from app.universe.escalation_runner import (
    RESULT_DONE,
    RESULT_SKIPPED,
    RUNNER_STATUS_CANCELLED,
    RUNNER_STATUS_DONE,
    RUNNER_STATUS_PARTIAL,
    STOP_CANCEL_REQUESTED,
    STOP_COMPLETED,
    STOP_DRY_RUN,
    STOP_MAX_ITEMS_REACHED,
    cancel_escalation_queue,
    open_escalation_status,
    resume_escalation_queue,
    run_escalation_queue,
)
from app.universe.promotion import LANE_1_HIGH_PRIORITY, LANE_2_RESEARCH_QUEUE, LANE_3_MONITOR


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
    return cfg


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _queue_row(
    *,
    queue_rank: int,
    ticker: str,
    lane: str,
    action_type: str,
    blocker: str = "NONE",
    implied_return_base: float | str = "UNKNOWN",
    source_run_ids: list[str] | None = None,
) -> dict:
    return {
        "queue_rank": queue_rank,
        "ticker": ticker,
        "priority_lane": lane,
        "action_type": action_type,
        "action_reason": f"reason_{action_type.lower()}",
        "blocking_reason_code": blocker,
        "recommended_command": f"python -m app.cli noop --ticker {ticker.lower()}",
        "source_campaign_run_id": "campaign_exec",
        "source_universe_run_ids": source_run_ids or [f"campaign_exec__{ticker.lower()}"],
        "latest_metrics": {
            "implied_return_base": implied_return_base,
            "mos_epv": "UNKNOWN",
            "yield_metric_used": "UNKNOWN",
            "value_gate_status": "WATCH",
        },
        "artifacts_to_read": {
            "memo_path": "",
            "watchlist_state_path": "",
            "source_rollup_paths": [],
        },
        "status": "PLANNED",
        "appearances_count": 1,
    }


def _prepare_campaign_root(cfg, campaign_run_id: str, queue_rows: list[dict]) -> Path:
    root = cfg.campaigns_dir / campaign_run_id
    _write_json(
        root / "campaign_state.json",
        {
            "campaign_run_id": campaign_run_id,
            "status": "DONE",
            "as_of_date": "2026-02-14",
            "source_campaign_file": "data/universe/sample_campaign.json",
            "items": [],
        },
    )
    _write_json(
        root / "escalation_queue.json",
        {
            "campaign_run_id": campaign_run_id,
            "queue_count": len(queue_rows),
            "rows": queue_rows,
        },
    )
    _write_json(
        root / "escalation_summary.json",
        {
            "campaign_run_id": campaign_run_id,
            "escalation_queue_count": len(queue_rows),
            "lane_to_action_counts": {},
            "top_10_escalation_actions": [],
        },
    )
    _write_json(root / "escalation_plan.json", {"campaign_run_id": campaign_run_id, "queue": queue_rows})
    return root


def test_escalation_runner_dry_run_writes_state_without_execution(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    campaign_run_id = "campaign_exec_dry"
    _prepare_campaign_root(
        cfg,
        campaign_run_id,
        [_queue_row(queue_rank=1, ticker="AAA", lane=LANE_3_MONITOR, action_type=ACTION_SCHEDULE_LIGHT_REFRESH)],
    )

    def _forbid(**_kwargs):
        raise AssertionError("dry-run should not execute queue items")

    monkeypatch.setattr("app.universe.escalation_runner._execute_queue_item", _forbid)

    payload = run_escalation_queue(campaign_run_id, dry_run=True)

    assert payload["status"] == RUNNER_STATUS_PARTIAL
    assert payload["stop_reason_code"] == STOP_DRY_RUN
    state = json.loads((cfg.campaigns_dir / campaign_run_id / "escalation_state.json").read_text(encoding="utf-8"))
    assert state["status"] == RUNNER_STATUS_PARTIAL
    assert state["stop_reason_code"] == STOP_DRY_RUN
    results = json.loads((cfg.campaigns_dir / campaign_run_id / "escalation_results.json").read_text(encoding="utf-8"))
    assert results["result_count"] == 0


def test_escalation_runner_resume_skips_completed_items(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    campaign_run_id = "campaign_exec_resume"
    _prepare_campaign_root(
        cfg,
        campaign_run_id,
        [
            _queue_row(
                queue_rank=1,
                ticker="AAA",
                lane=LANE_3_MONITOR,
                action_type=ACTION_SCHEDULE_LIGHT_REFRESH,
                implied_return_base=0.10,
            ),
            _queue_row(
                queue_rank=2,
                ticker="BBB",
                lane=LANE_2_RESEARCH_QUEUE,
                action_type=ACTION_CLEAR_BLOCKERS,
                blocker="PRICE_UNKNOWN",
                implied_return_base=0.20,
            ),
        ],
    )

    calls: list[str] = []

    def _fake_run_action_autopilot(**kwargs):
        calls.append(str(kwargs["execution_run_id"]))
        return {"status": "DONE"}

    monkeypatch.setattr("app.universe.escalation_runner._run_action_autopilot", _fake_run_action_autopilot)
    monkeypatch.setattr(
        "app.universe.escalation_runner._merge_targeted_result_into_campaign",
        lambda *args, **kwargs: {"latest_primary_blocker": "NONE"},
    )

    first = run_escalation_queue(campaign_run_id, max_items=1)
    second = resume_escalation_queue(campaign_run_id)

    assert first["status"] == RUNNER_STATUS_PARTIAL
    assert first["stop_reason_code"] == STOP_MAX_ITEMS_REACHED
    assert second["status"] == RUNNER_STATUS_DONE
    assert second["stop_reason_code"] == STOP_COMPLETED
    assert calls == [
        "campaign_exec_resume__refresh__aaa__001",
        "campaign_exec_resume__escalate__bbb__002",
    ]

    results = json.loads((cfg.campaigns_dir / campaign_run_id / "escalation_results.json").read_text(encoding="utf-8"))
    assert [row["queue_rank"] for row in results["rows"]] == [1, 2]
    summary = json.loads((cfg.campaigns_dir / campaign_run_id / "campaign_summary.json").read_text(encoding="utf-8"))
    assert summary["executed_queue_count"] == 2


def test_escalation_runner_cancel_prevents_future_execution(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    campaign_run_id = "campaign_exec_cancel"
    _prepare_campaign_root(
        cfg,
        campaign_run_id,
        [_queue_row(queue_rank=1, ticker="AAA", lane=LANE_2_RESEARCH_QUEUE, action_type=ACTION_CLEAR_BLOCKERS)],
    )

    run_escalation_queue(campaign_run_id, dry_run=True)
    cancel_payload = cancel_escalation_queue(campaign_run_id, "operator cancel")

    def _forbid(**_kwargs):
        raise AssertionError("cancelled queue should not execute new actions")

    monkeypatch.setattr("app.universe.escalation_runner._run_action_autopilot", _forbid)
    resumed = run_escalation_queue(campaign_run_id, resume=True)

    assert cancel_payload["run_status"] == RUNNER_STATUS_CANCELLED
    assert cancel_payload["stop_reason_code"] == STOP_CANCEL_REQUESTED
    assert resumed["status"] == RUNNER_STATUS_CANCELLED
    state = json.loads((cfg.campaigns_dir / campaign_run_id / "escalation_state.json").read_text(encoding="utf-8"))
    assert state["status"] == RUNNER_STATUS_CANCELLED


def test_escalation_runner_execution_run_ids_are_deterministic(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    campaign_run_id = "campaign_exec_ids"
    _prepare_campaign_root(
        cfg,
        campaign_run_id,
        [
            _queue_row(queue_rank=1, ticker="AAA", lane=LANE_3_MONITOR, action_type=ACTION_SCHEDULE_LIGHT_REFRESH),
            _queue_row(queue_rank=2, ticker="BBB", lane=LANE_2_RESEARCH_QUEUE, action_type=ACTION_CLEAR_BLOCKERS),
            _queue_row(queue_rank=3, ticker="CCC", lane=LANE_1_HIGH_PRIORITY, action_type=ACTION_REBUILD_DEPTH_RUN),
        ],
    )

    monkeypatch.setattr("app.universe.escalation_runner._run_action_autopilot", lambda **_kwargs: {"status": "DONE"})
    monkeypatch.setattr(
        "app.universe.escalation_runner._merge_targeted_result_into_campaign",
        lambda *args, **kwargs: {"latest_primary_blocker": "NONE"},
    )

    payload = run_escalation_queue(campaign_run_id)

    assert payload["status"] == RUNNER_STATUS_DONE
    results = json.loads((cfg.campaigns_dir / campaign_run_id / "escalation_results.json").read_text(encoding="utf-8"))
    assert [row["execution_run_id"] for row in results["rows"]] == [
        "campaign_exec_ids__refresh__aaa__001",
        "campaign_exec_ids__escalate__bbb__002",
        "campaign_exec_ids__rebuild__ccc__003",
    ]


def test_escalation_runner_adds_to_active_watchlist(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    campaign_run_id = "campaign_exec_watchlist"
    _prepare_campaign_root(
        cfg,
        campaign_run_id,
        [
            _queue_row(
                queue_rank=1,
                ticker="AAA",
                lane=LANE_1_HIGH_PRIORITY,
                action_type=ACTION_ADD_TO_ACTIVE_WATCHLIST,
                source_run_ids=["campaign_exec_watchlist__software_core"],
            )
        ],
    )

    payload = run_escalation_queue(campaign_run_id)

    assert payload["status"] == RUNNER_STATUS_DONE
    active_watchlist = json.loads((cfg.campaigns_dir / campaign_run_id / "active_watchlist.json").read_text(encoding="utf-8"))
    assert active_watchlist["entry_count"] == 1
    assert active_watchlist["entries"][0]["ticker"] == "AAA"
    assert active_watchlist["entries"][0]["lane"] == LANE_1_HIGH_PRIORITY
    assert active_watchlist["entries"][0]["priority_lane"] == LANE_1_HIGH_PRIORITY
    assert active_watchlist["entries"][0]["timestamp"]
    results = json.loads((cfg.campaigns_dir / campaign_run_id / "escalation_results.json").read_text(encoding="utf-8"))
    assert results["rows"][0]["status"] == RESULT_DONE
    summary = json.loads((cfg.campaigns_dir / campaign_run_id / "campaign_summary.json").read_text(encoding="utf-8"))
    assert summary["active_watchlist_count"] == 1


def test_escalation_runner_build_refreshed_memo_skips_missing_artifacts(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    campaign_run_id = "campaign_exec_memo_skip"
    _prepare_campaign_root(
        cfg,
        campaign_run_id,
        [
            _queue_row(
                queue_rank=1,
                ticker="AAA",
                lane=LANE_1_HIGH_PRIORITY,
                action_type=ACTION_BUILD_REFRESHED_MEMO,
                source_run_ids=["campaign_exec_memo_skip__missing"],
            )
        ],
    )

    payload = run_escalation_queue(campaign_run_id)

    assert payload["status"] == RUNNER_STATUS_DONE
    results = json.loads((cfg.campaigns_dir / campaign_run_id / "escalation_results.json").read_text(encoding="utf-8"))
    assert results["rows"][0]["status"] == RESULT_SKIPPED
    assert results["rows"][0]["outcome_summary"] == {"reason": "SKIPPED_MISSING_ARTIFACTS"}


def test_escalation_runner_cancel_before_first_run_initializes_state(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    campaign_run_id = "campaign_exec_cancel_prerun"
    _prepare_campaign_root(
        cfg,
        campaign_run_id,
        [_queue_row(queue_rank=1, ticker="AAA", lane=LANE_2_RESEARCH_QUEUE, action_type=ACTION_CLEAR_BLOCKERS)],
    )

    cancel_payload = cancel_escalation_queue(campaign_run_id, "operator cancel before run")

    def _forbid(**_kwargs):
        raise AssertionError("cancelled queue should not execute after pre-run cancel")

    monkeypatch.setattr("app.universe.escalation_runner._run_action_autopilot", _forbid)
    resumed = run_escalation_queue(campaign_run_id, resume=True)
    status_payload = open_escalation_status(campaign_run_id)

    assert cancel_payload["status"] == "OK"
    assert cancel_payload["run_status"] == RUNNER_STATUS_CANCELLED
    assert resumed["status"] == RUNNER_STATUS_CANCELLED
    assert status_payload["run_status"] == RUNNER_STATUS_CANCELLED
    assert status_payload["queue_count"] == 1
    assert status_payload["completed_count"] == 0


def test_escalation_runner_force_restart_resets_runtime_only_artifacts(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    campaign_run_id = "campaign_exec_force_restart"
    _prepare_campaign_root(
        cfg,
        campaign_run_id,
        [
            _queue_row(
                queue_rank=1,
                ticker="AAA",
                lane=LANE_1_HIGH_PRIORITY,
                action_type=ACTION_ADD_TO_ACTIVE_WATCHLIST,
            )
        ],
    )

    first = run_escalation_queue(campaign_run_id)
    assert first["status"] == RUNNER_STATUS_DONE

    _prepare_campaign_root(
        cfg,
        campaign_run_id,
        [_queue_row(queue_rank=1, ticker="BBB", lane=LANE_3_MONITOR, action_type=ACTION_SCHEDULE_LIGHT_REFRESH)],
    )
    monkeypatch.setattr("app.universe.escalation_runner._run_action_autopilot", lambda **_kwargs: {"status": "DONE"})
    monkeypatch.setattr(
        "app.universe.escalation_runner._merge_targeted_result_into_campaign",
        lambda *args, **kwargs: {"latest_primary_blocker": "NONE"},
    )

    restarted = run_escalation_queue(campaign_run_id, force_restart=True)

    assert restarted["status"] == RUNNER_STATUS_DONE
    active_watchlist = json.loads((cfg.campaigns_dir / campaign_run_id / "active_watchlist.json").read_text(encoding="utf-8"))
    assert active_watchlist["entry_count"] == 0
    status_payload = open_escalation_status(campaign_run_id)
    assert status_payload["active_watchlist_count"] == 0


def test_escalation_runner_recheck_promotion_updates_lane_after(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    campaign_run_id = "campaign_exec_recheck"
    _prepare_campaign_root(
        cfg,
        campaign_run_id,
        [
            _queue_row(
                queue_rank=1,
                ticker="BBB",
                lane=LANE_2_RESEARCH_QUEUE,
                action_type=ACTION_RECHECK_PROMOTION,
                blocker="PRICE_UNKNOWN",
            )
        ],
    )

    monkeypatch.setattr(
        "app.universe.escalation_runner._result_for_recheck",
        lambda *_args, **_kwargs: {
            "lane_after": LANE_1_HIGH_PRIORITY,
            "blocker_after": "NONE",
            "artifacts_written": ["promotion_state.json", "escalation_plan.json"],
        },
    )

    payload = run_escalation_queue(campaign_run_id)
    status_payload = open_escalation_status(campaign_run_id)

    assert payload["status"] == RUNNER_STATUS_DONE
    assert status_payload["lane_change_counts"] == {f"{LANE_2_RESEARCH_QUEUE}->{LANE_1_HIGH_PRIORITY}": 1}
    assert status_payload["top_lane_changes"][0]["ticker"] == "BBB"
    results = json.loads((cfg.campaigns_dir / campaign_run_id / "escalation_results.json").read_text(encoding="utf-8"))
    assert results["rows"][0]["lane_after"] == LANE_1_HIGH_PRIORITY
    assert results["rows"][0]["blocker_after"] == "NONE"


def test_escalation_runner_cli_status_reports_progress(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    campaign_run_id = "campaign_exec_cli"
    from app.config import get_config

    cfg = get_config()
    _prepare_campaign_root(
        cfg,
        campaign_run_id,
        [_queue_row(queue_rank=1, ticker="AAA", lane=LANE_3_MONITOR, action_type=ACTION_SCHEDULE_LIGHT_REFRESH)],
    )
    monkeypatch.setattr("app.universe.escalation_runner._run_action_autopilot", lambda **_kwargs: {"status": "DONE"})
    monkeypatch.setattr(
        "app.universe.escalation_runner._merge_targeted_result_into_campaign",
        lambda *args, **kwargs: {"latest_primary_blocker": "NONE"},
    )

    run_cmd = runner.invoke(
        app,
        ["universe-campaign-escalation-run", "--campaign-run-id", campaign_run_id, "--resume"],
    )
    assert run_cmd.exit_code == 0, run_cmd.output

    status_cmd = runner.invoke(
        app,
        ["universe-campaign-escalation-status", "--campaign-run-id", campaign_run_id],
    )
    assert status_cmd.exit_code == 0, status_cmd.output
    payload = json.loads(status_cmd.output)
    assert payload["status"] == "OK"
    assert payload["run_status"] == RUNNER_STATUS_DONE
    assert payload["completed_count"] == 1
