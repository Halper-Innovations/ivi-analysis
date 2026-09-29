from __future__ import annotations

import json
from pathlib import Path

from app.config import get_config
from app.db import init_db
from app.universe.batch_runner import (
    BATCH_STATUS_CANCELLED,
    BATCH_STATUS_DONE,
    BATCH_STATUS_PARTIAL,
    STOP_CANCEL_REQUESTED,
    create_batch_plan,
    load_depth_queue,
    run_depth_batch,
    cancel_depth_batch,
    resume_depth_batch,
)


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker,cik,name\nAAA,1,AAA\nBBB,2,BBB\nCCC,3,CCC\nDDD,4,DDD\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _write_depth_queue(
    cfg,
    *,
    run_id: str,
    entries: list[dict[str, object]],
    as_of_date: str = "2026-02-14",
) -> Path:
    queue_path = cfg.outputs_dir / "universe" / run_id / "depth_queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "as_of_date": as_of_date,
                "queue_count": len(entries),
                "entries": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return queue_path


def _fake_metrics(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": "DONE",
        "stop_reason_code": "COMPLETED",
        "known_implied_return_count": 2,
        "unknown_implied_return_count": 1,
        "pass_watch_fail_counts": {"PASS": 1, "WATCH": 1, "FAIL": 1},
        "dominant_blockers": {
            "valuation_reason_counts_top": [{"reason_code": "MISSING_PRICE", "count": 1}],
            "price_unknown_reason_counts_top": [{"reason_code": "OFFLINE_NO_CACHE", "count": 1}],
        },
        "artifacts_paths": {"run_dir": f"stub://{run_id}"},
    }


def test_universe_batch_plan_order_is_deterministic(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _write_depth_queue(
        cfg,
        run_id="queue_plan_det",
        entries=[
            {"rank": 2, "sector_suggested": "Software", "tickers": ["CCC"], "recommended_flags": {"iterations": 2, "top_k": 2}},
            {"rank": 1, "sector_suggested": "Beta", "tickers": ["BBB"], "recommended_flags": {"iterations": 1, "top_k": 1}},
            {"rank": 1, "sector_suggested": "Alpha", "tickers": ["AAA"], "recommended_flags": {"iterations": 1, "top_k": 1}},
        ],
    )
    queue = load_depth_queue("queue_plan_det")
    plan = create_batch_plan(queue, max_runs=3, selection_policy="QUEUE_ORDER", batch_run_id="depth_batch_det")
    assert [row["sector"] for row in plan] == ["Alpha", "Beta", "Software"]
    assert [row["run_id"] for row in plan] == [
        "depth_batch_det__Alpha__001",
        "depth_batch_det__Beta__002",
        "depth_batch_det__Software__003",
    ]
    assert [row["tickers"] for row in plan] == [["AAA"], ["BBB"], ["CCC"]]


def test_universe_batch_dry_run_writes_state_without_execution(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _write_depth_queue(
        cfg,
        run_id="queue_dry",
        entries=[
            {"rank": 1, "sector_suggested": "Software", "tickers": ["AAA", "BBB"], "recommended_flags": {"iterations": 1, "top_k": 2}},
            {"rank": 2, "sector_suggested": "Healthcare", "tickers": ["CCC"], "recommended_flags": {"iterations": 1, "top_k": 1}},
        ],
    )
    queue = load_depth_queue("queue_dry")
    plan = create_batch_plan(queue, max_runs=2, selection_policy="QUEUE_ORDER", batch_run_id="depth_batch_dry")

    def _should_not_execute(**_kwargs):
        raise AssertionError("dry-run should not execute depth runs")

    monkeypatch.setattr("app.universe.batch_runner._execute_plan_item", _should_not_execute)
    payload = run_depth_batch(
        plan,
        batch_run_id="depth_batch_dry",
        dry_run=True,
        max_runs=None,
        mode="depth",
        iterations=1,
        top_k=2,
        workers=1,
        with_prices=True,
        budgets={"llm_budget": None, "sec_budget": None},
    )
    assert payload["status"] == "OK"
    assert payload["run_status"] == BATCH_STATUS_DONE
    state_path = cfg.outputs_dir / "universe" / "queue_dry" / "depth_batches" / "depth_batch_dry" / "batch_state.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == BATCH_STATUS_DONE
    assert state["cursor_next_idx"] == 0
    assert state["completed_runs"] == []


def test_universe_batch_resume_continues_from_cursor(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    _write_depth_queue(
        get_config(),
        run_id="queue_resume",
        entries=[
            {"rank": 1, "sector_suggested": "Software", "tickers": ["AAA"], "recommended_flags": {"iterations": 1, "top_k": 1}},
            {"rank": 2, "sector_suggested": "Software", "tickers": ["BBB"], "recommended_flags": {"iterations": 1, "top_k": 1}},
            {"rank": 3, "sector_suggested": "Software", "tickers": ["CCC"], "recommended_flags": {"iterations": 1, "top_k": 1}},
        ],
    )
    queue = load_depth_queue("queue_resume")
    plan = create_batch_plan(queue, max_runs=3, selection_policy="QUEUE_ORDER", batch_run_id="depth_batch_resume")

    phase_one_calls: list[str] = []

    def _execute_phase_one(**kwargs):
        run_id = str((kwargs.get("plan_item") or {}).get("run_id") or "")
        phase_one_calls.append(run_id)
        if run_id.endswith("__002"):
            raise RuntimeError("simulated interruption")
        return {"status": "DONE"}

    monkeypatch.setattr("app.universe.batch_runner._execute_plan_item", _execute_phase_one)
    monkeypatch.setattr(
        "app.universe.batch_runner._collect_run_metrics",
        lambda **kwargs: _fake_metrics(str(kwargs.get("run_id") or "")),
    )
    first = run_depth_batch(
        plan,
        batch_run_id="depth_batch_resume",
        dry_run=False,
        max_runs=None,
        mode="depth",
        iterations=1,
        top_k=1,
        workers=1,
        with_prices=True,
        budgets={"llm_budget": None, "sec_budget": None},
    )
    assert first["status"] == "OK"
    assert first["run_status"] == BATCH_STATUS_PARTIAL
    assert first["cursor_next_idx"] == 2

    phase_two_calls: list[str] = []

    def _execute_phase_two(**kwargs):
        run_id = str((kwargs.get("plan_item") or {}).get("run_id") or "")
        phase_two_calls.append(run_id)
        return {"status": "DONE"}

    monkeypatch.setattr("app.universe.batch_runner._execute_plan_item", _execute_phase_two)
    resumed = resume_depth_batch("depth_batch_resume")
    assert resumed["status"] == "OK"
    assert resumed["run_status"] == BATCH_STATUS_DONE
    assert resumed["cursor_next_idx"] == 3
    assert phase_one_calls[:2] == [
        "depth_batch_resume__Software__001",
        "depth_batch_resume__Software__002",
    ]
    assert phase_two_calls == ["depth_batch_resume__Software__003"]


def test_universe_batch_cancel_stops_future_runs(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    _write_depth_queue(
        get_config(),
        run_id="queue_cancel",
        entries=[
            {"rank": 1, "sector_suggested": "Software", "tickers": ["AAA"], "recommended_flags": {"iterations": 1, "top_k": 1}},
            {"rank": 2, "sector_suggested": "Software", "tickers": ["BBB"], "recommended_flags": {"iterations": 1, "top_k": 1}},
            {"rank": 3, "sector_suggested": "Software", "tickers": ["CCC"], "recommended_flags": {"iterations": 1, "top_k": 1}},
        ],
    )
    queue = load_depth_queue("queue_cancel")
    plan = create_batch_plan(queue, max_runs=3, selection_policy="QUEUE_ORDER", batch_run_id="depth_batch_cancel")
    executed: list[str] = []

    def _execute_and_cancel(**kwargs):
        run_id = str((kwargs.get("plan_item") or {}).get("run_id") or "")
        executed.append(run_id)
        cancel_depth_batch("depth_batch_cancel", "cancel during checkpoint")
        return {"status": "DONE"}

    monkeypatch.setattr("app.universe.batch_runner._execute_plan_item", _execute_and_cancel)
    monkeypatch.setattr(
        "app.universe.batch_runner._collect_run_metrics",
        lambda **kwargs: _fake_metrics(str(kwargs.get("run_id") or "")),
    )
    payload = run_depth_batch(
        plan,
        batch_run_id="depth_batch_cancel",
        dry_run=False,
        max_runs=None,
        mode="depth",
        iterations=1,
        top_k=1,
        workers=1,
        with_prices=True,
        budgets={"llm_budget": None, "sec_budget": None},
    )
    assert payload["status"] == "OK"
    assert payload["run_status"] == BATCH_STATUS_CANCELLED
    assert executed == ["depth_batch_cancel__Software__001"]
    state = json.loads(
        (
            get_config().outputs_dir
            / "universe"
            / "queue_cancel"
            / "depth_batches"
            / "depth_batch_cancel"
            / "batch_state.json"
        ).read_text(encoding="utf-8")
    )
    assert state["status"] == BATCH_STATUS_CANCELLED
    assert state["stop_reason_code"] == STOP_CANCEL_REQUESTED
