from __future__ import annotations

from app.db import get_db, init_db
from app.rlm.state import (
    init_loop_state,
    insert_rlm_iteration_row,
    load_loop_state,
    persist_iteration_artifacts,
    save_loop_state,
    upsert_rlm_run_row,
)


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_rlm_state_save_and_resume(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)

    state = init_loop_state(
        run_id="rlm_persist_test",
        sector="Software",
        as_of_date="2026-02-13",
        max_iterations=3,
        llm_budget_usd=2.5,
        sec_budget_count=100,
    )
    state.peer_set = ["AAPL", "MSFT"]
    state.top_k_current = ["AAPL"]
    state.action_history.append({"iteration": 0, "planner": {"actions": []}})

    path = save_loop_state(state)
    assert path.exists()

    resumed = load_loop_state("rlm_persist_test")
    assert resumed is not None
    assert resumed.run_id == "rlm_persist_test"
    assert resumed.peer_set == ["AAPL", "MSFT"]


def test_rlm_iteration_artifacts_and_db_rows(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)

    state = init_loop_state(
        run_id="rlm_db_test",
        sector="Software",
        as_of_date="2026-02-13",
        max_iterations=2,
        llm_budget_usd=1.0,
        sec_budget_count=50,
    )
    save_loop_state(state)
    upsert_rlm_run_row(state)

    paths = persist_iteration_artifacts(
        run_id="rlm_db_test",
        iteration=0,
        planner_payload={"iteration": 0, "actions": []},
        actions_payload={"results": []},
        critic_payload={"iteration": 0, "confidence": "LOW"},
    )
    assert paths["planner_path"].endswith("iter_000_planner.json")

    insert_rlm_iteration_row(
        run_id="rlm_db_test",
        iteration=0,
        planner_json={"iteration": 0, "actions": []},
        critic_json={"iteration": 0, "confidence": "LOW"},
        progress_json={"ok": True},
    )

    with get_db() as conn:
        run_row = conn.execute("SELECT run_id, status FROM rlm_runs WHERE run_id = ?", ("rlm_db_test",)).fetchone()
        iter_row = conn.execute(
            "SELECT run_id, iteration FROM rlm_iterations WHERE run_id = ? AND iteration = ?",
            ("rlm_db_test", 0),
        ).fetchone()
    assert run_row is not None
    assert run_row["run_id"] == "rlm_db_test"
    assert iter_row is not None
    assert int(iter_row["iteration"]) == 0
