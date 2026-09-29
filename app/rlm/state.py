from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_config
from app.db import get_db, utc_now_iso


class LoopBudgets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sec_budget_remaining: int
    llm_budget_remaining: float
    max_iterations: int


class LoopState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rlm_version: str = "v0"
    mode: str = "auto"
    run_id: str
    sector: str
    as_of_date: str
    iteration: int = 0
    status: str = "RUNNING"
    with_prices: bool = True
    peer_set: list[str] = Field(default_factory=list)
    top_k_current: list[str] = Field(default_factory=list)
    artifacts: dict[str, str | None] = Field(default_factory=dict)
    gap_summary: dict[str, dict[str, Any]] = Field(default_factory=dict)
    evidence_delta_counters: dict[str, int] = Field(default_factory=dict)
    rubric_weights: dict[str, float] = Field(default_factory=dict)
    gate_threshold_overrides: dict[str, float] = Field(default_factory=dict)
    gate_thresholds_effective: dict[str, float] = Field(default_factory=dict)
    budgets_remaining: LoopBudgets
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    progress_history: list[dict[str, Any]] = Field(default_factory=list)
    no_progress_streak: int = 0
    ranking_stable_streak: int = 0
    llm_cost_used: float = 0.0
    sec_requests_used: int = 0
    stop_reason_code: str | None = None
    stop_summary: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


def rlm_run_dir(run_id: str) -> Path:
    cfg = get_config()
    path = cfg.sectors_dir / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def rlm_state_path(run_id: str) -> Path:
    return rlm_run_dir(run_id) / "rlm_state.json"


def rlm_iterations_dir(run_id: str) -> Path:
    path = rlm_run_dir(run_id) / "rlm_iterations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_loop_state(run_id: str) -> LoopState | None:
    path = rlm_state_path(run_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LoopState.model_validate(payload)


def save_loop_state(state: LoopState) -> Path:
    state.updated_at = utc_now_iso()
    path = rlm_state_path(state.run_id)
    path.write_text(json.dumps(state.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def persist_iteration_artifacts(
    *,
    run_id: str,
    iteration: int,
    planner_payload: dict[str, Any],
    actions_payload: dict[str, Any],
    critic_payload: dict[str, Any],
    delta_payload: dict[str, Any] | None = None,
) -> dict[str, str]:
    base = rlm_iterations_dir(run_id)
    stem = f"iter_{int(iteration):03d}"
    planner_path = base / f"{stem}_planner.json"
    actions_path = base / f"{stem}_actions.json"
    critic_path = base / f"{stem}_critic.json"
    planner_path.write_text(json.dumps(planner_payload, indent=2), encoding="utf-8")
    actions_path.write_text(json.dumps(actions_payload, indent=2), encoding="utf-8")
    critic_path.write_text(json.dumps(critic_payload, indent=2), encoding="utf-8")
    out = {
        "planner_path": str(planner_path),
        "actions_path": str(actions_path),
        "critic_path": str(critic_path),
    }
    if isinstance(delta_payload, dict):
        delta_path = base / f"{stem}_scoreboard_delta.json"
        delta_path.write_text(json.dumps(delta_payload, indent=2), encoding="utf-8")
        out["scoreboard_delta_path"] = str(delta_path)
    return out


def upsert_rlm_run_row(state: LoopState) -> None:
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO rlm_runs(
                run_id, sector, as_of_date, status, created_at, updated_at, iterations, stop_reason
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                sector = excluded.sector,
                as_of_date = excluded.as_of_date,
                status = excluded.status,
                updated_at = excluded.updated_at,
                iterations = excluded.iterations,
                stop_reason = excluded.stop_reason
            """,
            (
                state.run_id,
                state.sector,
                state.as_of_date,
                state.status,
                state.created_at,
                now,
                int(state.iteration),
                state.stop_reason_code,
            ),
        )


def insert_rlm_iteration_row(
    *,
    run_id: str,
    iteration: int,
    planner_json: dict[str, Any],
    critic_json: dict[str, Any],
    progress_json: dict[str, Any],
) -> None:
    created_at = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO rlm_iterations(
                run_id, iteration, planner_json, critic_json, progress_json, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, iteration) DO UPDATE SET
                planner_json = excluded.planner_json,
                critic_json = excluded.critic_json,
                progress_json = excluded.progress_json,
                created_at = rlm_iterations.created_at
            """,
            (
                run_id,
                int(iteration),
                json.dumps(planner_json),
                json.dumps(critic_json),
                json.dumps(progress_json),
                created_at,
            ),
        )


def get_max_iteration_for_run(run_id: str) -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT MAX(iteration) AS max_iteration FROM rlm_iterations WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    if not row:
        return -1
    value = row["max_iteration"]
    return int(value) if isinstance(value, int) else -1


def get_rlm_run_row(run_id: str) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT run_id, sector, as_of_date, status, created_at, updated_at, iterations, stop_reason
            FROM rlm_runs
            WHERE run_id = ?
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "run_id": str(row["run_id"]),
        "sector": str(row["sector"]),
        "as_of_date": str(row["as_of_date"]),
        "status": str(row["status"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "iterations": int(row["iterations"] or 0),
        "stop_reason": row["stop_reason"],
    }


def init_loop_state(
    *,
    run_id: str,
    sector: str,
    as_of_date: str,
    max_iterations: int,
    llm_budget_usd: float,
    sec_budget_count: int,
    gate_threshold_overrides: dict[str, float] | None = None,
    gate_thresholds_effective: dict[str, float] | None = None,
) -> LoopState:
    return LoopState(
        run_id=run_id,
        sector=sector,
        as_of_date=as_of_date,
        iteration=0,
        status="RUNNING",
        gate_threshold_overrides={
            str(key): float(value)
            for key, value in ((gate_threshold_overrides or {}).items())
            if isinstance(value, (int, float))
        },
        gate_thresholds_effective={
            str(key): float(value)
            for key, value in ((gate_thresholds_effective or {}).items())
            if isinstance(value, (int, float))
        },
        budgets_remaining=LoopBudgets(
            sec_budget_remaining=max(0, int(sec_budget_count)),
            llm_budget_remaining=max(0.0, float(llm_budget_usd)),
            max_iterations=max(1, int(max_iterations)),
        ),
    )
