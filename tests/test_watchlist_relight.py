"""Fail-closed contracts for the scoped watchlist relight runner."""

from __future__ import annotations

import json
import os
import re
import subprocess

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.config import get_config
from app.watchlist import relight
from app.watchlist.relight import (
    RelightAuthorizationError,
    RelightPlanError,
    RelightPreflightError,
    _positive_budget,
    _positive_cell_reservation_floor,
    estimate_relight_cost,
    execute_relight,
    parse_cell_order,
    parse_ticker_list,
    plan_relight,
    relight_preflight,
    state_path,
)


READY_BINDING = {
    "provider": "openai",
    "model": "gpt-5.4-mini",
    "credential_present": True,
    "ready": True,
    "relight_provider_policy": "exact_provider_model_strict_no_fallback",
}


@pytest.fixture
def relight_env(monkeypatch, tmp_path):
    """A planned relight with the provider ready and preflight green."""

    monkeypatch.setattr(relight, "_configured_provider_binding", lambda: dict(READY_BINDING))
    monkeypatch.setattr(
        relight,
        "relight_preflight",
        lambda manifest_path=None: {
            "manifest_path": "/tmp/manifest.json",
            "manifest_usable": True,
            "ready": True,
            "blockers": [],
        },
    )
    monkeypatch.setattr(
        relight,
        "resolve_cells",
        lambda tickers, db_path=None: (
            [
                {
                    "cell_id": "energy:mid_cap",
                    "sector": "energy",
                    "band": "mid_cap",
                    "tickers": ["GPOR"],
                    "status": "PENDING",
                }
            ],
            [],
        ),
    )
    monkeypatch.setattr(relight, "_artifact_cost_samples", lambda **_kwargs: [])
    monkeypatch.setattr(
        relight,
        "verify_relight_eligibility",
        lambda state, manifest_path=None: {
            "checked_at": "2026-07-28T00:00:00Z",
            "decision_eligible_run_ids": [],
            "ineligible_run_ids": [],
        },
    )
    return tmp_path


# ---------------------------------------------------------------- budget gate


def test_absent_budget_authorizes_nothing():
    with pytest.raises(RelightAuthorizationError, match="authorizes nothing"):
        _positive_budget(None)


@pytest.mark.parametrize("value", [0, -1.0, "", "abc", float("inf"), float("nan")])
def test_non_positive_or_non_finite_budget_is_refused(value):
    with pytest.raises(RelightAuthorizationError):
        _positive_budget(value)


@pytest.mark.parametrize("value", [None, 0, -1.0, "abc", float("inf"), float("nan")])
def test_non_positive_or_non_finite_cell_reservation_floor_is_refused(value):
    with pytest.raises(RelightAuthorizationError):
        _positive_cell_reservation_floor(value)


def test_execute_without_budget_spends_nothing(relight_env, monkeypatch):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    calls: list[list[str]] = []

    def runner(argv, **_kwargs):
        calls.append(list(argv))
        raise AssertionError("must not spawn a paid child without a budget")

    with pytest.raises(RelightAuthorizationError):
        execute_relight(
            relight_id=plan["relight_id"],
            budget_usd=None,
            output_root=relight_env,
            command_runner=runner,
        )
    assert calls == []


# ------------------------------------------------------------ preflight gate


def test_unusable_manifest_refuses_to_spend(relight_env, monkeypatch):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    monkeypatch.setattr(
        relight,
        "relight_preflight",
        lambda manifest_path=None: {
            "manifest_path": None,
            "manifest_usable": False,
            "ready": False,
            "blockers": ["MANIFEST_NOT_USABLE_FRESH_RUNS_WOULD_STAY_DARK"],
        },
    )

    def runner(argv, **_kwargs):
        raise AssertionError("must not spawn a paid child when preflight is blocked")

    with pytest.raises(RelightPreflightError, match="could not become decision-eligible"):
        execute_relight(
            relight_id=plan["relight_id"],
            budget_usd=100.0,
            output_root=relight_env,
            command_runner=runner,
        )


def test_preflight_reports_unusable_manifest(monkeypatch):
    monkeypatch.setattr(relight, "financial_integrity_manifest_is_usable", lambda _p=None: False)
    monkeypatch.setattr(relight, "active_financial_integrity_manifest_path", lambda _p=None: None)
    result = relight_preflight()
    assert result["ready"] is False
    assert "NO_ACTIVE_FINANCIAL_INTEGRITY_MANIFEST" in result["blockers"]
    assert "MANIFEST_NOT_USABLE_FRESH_RUNS_WOULD_STAY_DARK" in result["blockers"]


# ---------------------------------------------------------- provider bindings


class _Cfg:
    """Minimal config stand-in; attributes are set per test."""

    llm_provider = "disabled"
    anthropic_model = ""
    anthropic_api_key = None
    openai_model = ""
    openai_api_key = None


def test_deepseek_binding_is_supported(monkeypatch):
    cfg = _Cfg()
    cfg.llm_provider = "deepseek"
    cfg.deepseek_model = "deepseek-v4-pro"
    cfg.deepseek_api_key = "sk-test"
    monkeypatch.setattr(relight, "get_config", lambda: cfg)

    binding = relight._configured_provider_binding()
    assert binding["provider"] == "deepseek"
    assert binding["model"] == "deepseek-v4-pro"
    assert binding["ready"] is True


def test_deepseek_without_config_surface_is_unready_not_an_error(monkeypatch):
    """Planning before the DeepSeek config lands must report, not raise."""

    cfg = _Cfg()
    cfg.llm_provider = "deepseek"  # no deepseek_model / deepseek_api_key attributes
    monkeypatch.setattr(relight, "get_config", lambda: cfg)

    binding = relight._configured_provider_binding()
    assert binding["ready"] is False
    assert binding["model"] == "disabled"


def test_deepseek_without_credential_is_unready(monkeypatch):
    cfg = _Cfg()
    cfg.llm_provider = "deepseek"
    cfg.deepseek_model = "deepseek-v4-pro"
    cfg.deepseek_api_key = None
    monkeypatch.setattr(relight, "get_config", lambda: cfg)

    assert relight._configured_provider_binding()["ready"] is False


def test_unknown_provider_is_never_ready(monkeypatch):
    cfg = _Cfg()
    cfg.llm_provider = "some-new-provider"
    monkeypatch.setattr(relight, "get_config", lambda: cfg)

    assert relight._configured_provider_binding()["ready"] is False


# ------------------------------------------------------------- provider drift


def test_provider_drift_after_planning_refuses(relight_env, monkeypatch):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    drifted = dict(READY_BINDING, model="gpt-5.4")
    monkeypatch.setattr(relight, "_configured_provider_binding", lambda: drifted)

    with pytest.raises(RelightAuthorizationError, match="changed after planning"):
        execute_relight(
            relight_id=plan["relight_id"],
            budget_usd=100.0,
            output_root=relight_env,
            command_runner=lambda argv, **_k: pytest.fail("must not spawn"),
        )


def test_ceiling_below_estimate_refuses_without_opt_in(relight_env):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    estimate = plan["cost_estimate"]["estimated_cost_usd"]

    with pytest.raises(RelightAuthorizationError, match="below the plan estimate"):
        execute_relight(
            relight_id=plan["relight_id"],
            budget_usd=estimate / 2,
            output_root=relight_env,
            command_runner=lambda argv, **_k: pytest.fail("must not spawn"),
        )


# -------------------------------------------------------------------- planning


def test_planning_requires_a_non_empty_target(relight_env):
    with pytest.raises(RelightPlanError, match="explicit ticker list"):
        plan_relight(tickers=[], output_root=relight_env)


def test_plan_writes_a_resumable_checkpoint(relight_env):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    path = state_path(plan["relight_id"], relight_env)
    assert path.is_file()
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == relight.RELIGHT_SCHEMA_VERSION
    assert persisted["status"] == "PLANNED"
    assert persisted["cumulative_cost_usd"] == 0.0
    assert persisted["cells"][0]["status"] == "PENDING"


def test_parse_ticker_list_normalizes_and_dedupes():
    assert parse_ticker_list(" gpor, tdw  crto ,gpor ") == ["CRTO", "GPOR", "TDW"]
    assert parse_ticker_list(None) == []


def _ordered_plan_cells():
    return [
        {
            "cell_id": "automotive:mid_cap",
            "sector": "automotive",
            "band": "mid_cap",
            "tickers": ["LKQ", "PHIN"],
            "status": "PENDING",
        },
        {
            "cell_id": "energy:mid_cap",
            "sector": "energy",
            "band": "mid_cap",
            "tickers": ["GPOR"],
            "status": "PENDING",
        },
        {
            "cell_id": "insurance:mid_cap",
            "sector": "insurance",
            "band": "mid_cap",
            "tickers": ["AGO", "MCY", "WTM"],
            "status": "PENDING",
        },
    ]


def _use_ordered_plan_cells(monkeypatch):
    monkeypatch.setattr(
        relight,
        "resolve_cells",
        lambda tickers, db_path=None: (_ordered_plan_cells(), []),
    )


def test_parse_cell_order_preserves_explicit_tokens():
    assert parse_cell_order(
        " insurance:mid_cap, energy:mid_cap,automotive:mid_cap "
    ) == ["insurance:mid_cap", "energy:mid_cap", "automotive:mid_cap"]
    assert parse_cell_order(None) is None


def test_cli_cell_order_forwards_exact_tokens(monkeypatch):
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        cells = _ordered_plan_cells()
        return {
            "relight_id": "relight_20260731T160000Z_abcdef12",
            "routed_tickers": ["AGO", "GPOR", "LKQ", "MCY", "PHIN", "WTM"],
            "cells": cells,
            "provider_binding": dict(READY_BINDING),
            "cost_estimate": {
                "estimated_cost_usd": 0.38,
                "estimated_cost_per_review_usd": 0.05,
                "estimated_base_cost_per_cell_usd": 0.05,
                "safety_multiplier": 1.5,
                "method": "no_matching_history_conservative_fallback_with_50pct_reserve",
                "matching_historical_runs": 0,
                "per_cell_estimated_cost_usd": {
                    "automotive:mid_cap": 0.225,
                    "energy:mid_cap": 0.15,
                    "insurance:mid_cap": 0.3,
                },
            },
            "unroutable": [],
            "preflight": {"ready": True, "blockers": []},
        }

    monkeypatch.setattr(relight, "plan_relight", fake_plan)
    result = CliRunner().invoke(
        app,
        [
            "watchlist",
            "relight-plan",
            "--tickers",
            "AGO,GPOR,LKQ,MCY,PHIN,WTM",
            "--cell-order",
            "insurance:mid_cap,energy:mid_cap,automotive:mid_cap",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["cell_order"] == [
        "insurance:mid_cap",
        "energy:mid_cap",
        "automotive:mid_cap",
    ]


def test_plan_without_cell_order_preserves_legacy_payload(monkeypatch, relight_env):
    _use_ordered_plan_cells(monkeypatch)
    monkeypatch.setattr(relight, "_utc_now", lambda: "2026-07-31T16:00:00Z")
    plan = plan_relight(
        tickers=["WTM", "GPOR", "PHIN", "AGO", "MCY", "LKQ"],
        as_of="2026-07-29",
        relight_id="relight_20260731T160000Z_abcdef12",
        output_root=relight_env,
    )

    assert [cell["cell_id"] for cell in plan["cells"]] == [
        "automotive:mid_cap",
        "energy:mid_cap",
        "insurance:mid_cap",
    ]
    assert plan["cells"] == _ordered_plan_cells()
    assert "cell_order" not in plan
    persisted = json.loads(
        state_path(plan["relight_id"], relight_env).read_text(encoding="utf-8")
    )
    assert persisted == plan


def test_valid_cell_order_persists_exact_permutation(monkeypatch, relight_env):
    _use_ordered_plan_cells(monkeypatch)
    plan = plan_relight(
        tickers=["AGO", "GPOR", "LKQ", "MCY", "PHIN", "WTM"],
        cell_order=[
            "insurance:mid_cap",
            "energy:mid_cap",
            "automotive:mid_cap",
        ],
        relight_id="relight_20260731T160001Z_abcdef12",
        output_root=relight_env,
    )

    assert [cell["cell_id"] for cell in plan["cells"]] == [
        "insurance:mid_cap",
        "energy:mid_cap",
        "automotive:mid_cap",
    ]
    assert plan["cells"][0]["tickers"] == ["AGO", "MCY", "WTM"]
    persisted = json.loads(
        state_path(plan["relight_id"], relight_env).read_text(encoding="utf-8")
    )
    assert persisted["cells"] == plan["cells"]


@pytest.mark.parametrize(
    ("cell_order", "message", "relight_id"),
    [
        (
            ["energy:mid_cap", "energy:mid_cap", "automotive:mid_cap"],
            "cell order contains duplicate cell_id(s): energy:mid_cap",
            "relight_20260731T160002Z_abcdef12",
        ),
        (
            ["energy:mid_cap", "insurance:mid_cap", "materials:mid_cap"],
            "cell order contains unknown cell_id(s): materials:mid_cap",
            "relight_20260731T160003Z_abcdef12",
        ),
        (
            ["energy:mid_cap", "insurance:mid_cap"],
            "cell order omits planned cell_id(s): automotive:mid_cap",
            "relight_20260731T160004Z_abcdef12",
        ),
    ],
)
def test_invalid_cell_order_refuses_without_plan_file(
    monkeypatch,
    relight_env,
    cell_order,
    message,
    relight_id,
):
    _use_ordered_plan_cells(monkeypatch)
    with pytest.raises(RelightAuthorizationError, match=re.escape(message)):
        plan_relight(
            tickers=["AGO", "GPOR", "LKQ", "MCY", "PHIN", "WTM"],
            cell_order=cell_order,
            relight_id=relight_id,
            output_root=relight_env,
        )

    assert not state_path(relight_id, relight_env).exists()


def test_estimate_separates_fixed_and_variable_cost():
    cells = [
        {"cell_id": "a:mid_cap", "tickers": ["A"]},
        {"cell_id": "b:mid_cap", "tickers": ["B", "C"]},
    ]
    estimate = estimate_relight_cost(provider="openai", model="nope", cells=cells)
    # No history for this binding -> conservative fallback, still per-cell aware.
    assert estimate["method"].startswith("no_matching_history")
    assert estimate["cell_count"] == 2
    assert estimate["review_units"] == 3
    assert (
        estimate["per_cell_estimated_cost_usd"]["b:mid_cap"]
        > (estimate["per_cell_estimated_cost_usd"]["a:mid_cap"])
    )


# ------------------------------------------------------------------ execution


def _child_result(returncode: int = 0, cost: float = 0.10, run_id: str = "run_x"):
    payload = {
        "run_id": run_id,
        "provider_usage_attestation": {
            "valid": True,
            "cost_estimate_usd": cost,
            "physical_attempt_count": 1,
        },
    }
    return subprocess.CompletedProcess(
        args=["x"], returncode=returncode, stdout=json.dumps(payload), stderr=""
    )


def _rewrite_execution_state(
    plan,
    output_root,
    *,
    estimate: float,
    spent: float = 0.0,
    cell_status: str = "PENDING",
    prior_reservation: float | None = None,
) -> None:
    path = state_path(plan["relight_id"], output_root)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    persisted["cost_estimate"]["estimated_cost_usd"] = estimate
    persisted["cost_estimate"]["per_cell_estimated_cost_usd"]["energy:mid_cap"] = estimate
    persisted["cumulative_cost_usd"] = spent
    persisted["status"] = "INCOMPLETE" if cell_status == "FAILED" else "PLANNED"
    persisted["cells"][0]["status"] = cell_status
    if prior_reservation is not None:
        persisted["cells"][0]["reservation_usd"] = prior_reservation
        persisted["cells"][0]["cost_usd"] = spent
    path.write_text(
        json.dumps(persisted, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _argv_max_cost(argv: list[str]) -> str:
    return argv[argv.index("--max-cost-usd") + 1]


def test_cli_cell_reservation_floor_defaults_and_forwards_override(monkeypatch):
    captured: list[dict[str, object]] = []

    def fake_execute(**kwargs):
        captured.append(kwargs)
        return {
            "status": "COMPLETED",
            "relight_id": kwargs["relight_id"],
            "cumulative_cost_usd": 0.0,
            "authorized_max_cost_usd": kwargs["budget_usd"],
            "eligibility": {
                "decision_eligible_run_ids": [],
                "ineligible_run_ids": [],
            },
        }

    monkeypatch.setattr(relight, "execute_relight", fake_execute)
    runner = CliRunner()
    default = runner.invoke(
        app,
        [
            "watchlist",
            "relight-run",
            "--relight-id",
            "relight_20260729T013258Z_a023afdd",
            "--budget-usd",
            "5",
        ],
    )
    override = runner.invoke(
        app,
        [
            "watchlist",
            "relight-run",
            "--relight-id",
            "relight_20260729T013258Z_a023afdd",
            "--budget-usd",
            "5",
            "--cell-reservation-floor-usd",
            "0.75",
        ],
    )

    assert default.exit_code == 0
    assert override.exit_code == 0
    assert captured[0]["cell_reservation_floor_usd"] == 0.50
    assert captured[1]["cell_reservation_floor_usd"] == 0.75


def test_cell_reservation_uses_floor_above_estimate(relight_env):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    _rewrite_execution_state(plan, relight_env, estimate=0.125)
    captured: list[list[str]] = []

    state = execute_relight(
        relight_id=plan["relight_id"],
        budget_usd=2.0,
        cell_reservation_floor_usd=0.50,
        output_root=relight_env,
        command_runner=lambda argv, **_kwargs: (
            captured.append(list(argv)) or _child_result(cost=0.01)
        ),
    )

    assert _argv_max_cost(captured[0]) == "0.500000"
    assert state["cell_reservation_floor_usd"] == 0.50
    assert state["cells"][0]["reservation_usd"] == 0.50
    assert state["cumulative_cost_usd"] == 0.01


def test_cell_reservation_uses_estimate_above_floor(relight_env):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    _rewrite_execution_state(plan, relight_env, estimate=0.75)
    captured: list[list[str]] = []

    state = execute_relight(
        relight_id=plan["relight_id"],
        budget_usd=2.0,
        cell_reservation_floor_usd=0.50,
        output_root=relight_env,
        command_runner=lambda argv, **_kwargs: (
            captured.append(list(argv)) or _child_result(cost=0.01)
        ),
    )

    assert _argv_max_cost(captured[0]) == "0.750000"
    assert state["cell_reservation_floor_usd"] == 0.50
    assert state["cells"][0]["reservation_usd"] == 0.75
    assert state["cumulative_cost_usd"] == 0.01


def test_cell_reservation_is_clamped_by_remaining_campaign_budget(relight_env):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    _rewrite_execution_state(plan, relight_env, estimate=0.75, spent=1.40)
    captured: list[list[str]] = []

    state = execute_relight(
        relight_id=plan["relight_id"],
        budget_usd=2.0,
        cell_reservation_floor_usd=0.50,
        output_root=relight_env,
        command_runner=lambda argv, **_kwargs: (
            captured.append(list(argv)) or _child_result(cost=0.01)
        ),
    )

    assert _argv_max_cost(captured[0]) == "0.600000"
    assert state["cell_reservation_floor_usd"] == 0.50
    assert state["cells"][0]["reservation_usd"] == 0.60
    assert state["cumulative_cost_usd"] == 1.41


def test_cell_does_not_run_when_remaining_budget_is_below_floor(relight_env):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    _rewrite_execution_state(plan, relight_env, estimate=0.125, spent=1.51)
    calls: list[list[str]] = []

    state = execute_relight(
        relight_id=plan["relight_id"],
        budget_usd=2.0,
        cell_reservation_floor_usd=0.50,
        output_root=relight_env,
        command_runner=lambda argv, **_kwargs: calls.append(list(argv)),
    )

    assert calls == []
    assert state["status"] == "BUDGET_EXHAUSTED"
    assert state["cell_reservation_floor_usd"] == 0.50
    assert state["cells"][0]["status"] == "SKIPPED_BUDGET_EXHAUSTED"
    assert state["cumulative_cost_usd"] == 1.51


def test_resume_recomputes_failed_cell_reservation_with_default_floor(
    relight_env,
):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    _rewrite_execution_state(
        plan,
        relight_env,
        estimate=0.0762,
        spent=0.01,
        cell_status="FAILED",
        prior_reservation=0.0762,
    )
    captured: list[list[str]] = []

    state = execute_relight(
        relight_id=plan["relight_id"],
        budget_usd=5.0,
        output_root=relight_env,
        command_runner=lambda argv, **_kwargs: (
            captured.append(list(argv)) or _child_result(cost=0.01)
        ),
    )
    persisted = json.loads(state_path(plan["relight_id"], relight_env).read_text(encoding="utf-8"))

    assert _argv_max_cost(captured[0]) == "0.500000"
    assert state["status"] == "COMPLETED"
    assert state["cumulative_cost_usd"] == 0.02
    assert state["cell_reservation_floor_usd"] == 0.50
    assert state["cells"][0]["reservation_usd"] == 0.50
    assert persisted["cell_reservation_floor_usd"] == 0.50
    assert persisted["cells"][0]["reservation_usd"] == 0.50


def test_resume_retries_live_shaped_orphaned_running_cell_without_double_charge(
    relight_env,
):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    path = state_path(plan["relight_id"], relight_env)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    persisted["cost_estimate"]["estimated_cost_usd"] = 3.37
    persisted["cost_estimate"]["per_cell_estimated_cost_usd"]["energy:mid_cap"] = 0.0762
    persisted["status"] = "RUNNING"
    persisted["cumulative_cost_usd"] = 0.028664
    persisted["cells"][0].update(
        {
            "status": "RUNNING",
            "reservation_usd": 0.50,
            "started_at": "2026-07-29T23:40:38Z",
        }
    )
    for field in ("run_id", "cost_usd", "cost_attested", "returncode", "completed_at"):
        persisted["cells"][0].pop(field, None)
    path.write_text(
        json.dumps(persisted, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dispatch_snapshots: list[dict[str, object]] = []

    def runner(argv, **_kwargs):
        current = json.loads(path.read_text(encoding="utf-8"))
        dispatch_snapshots.append(
            {
                "max_cost": _argv_max_cost(list(argv)),
                "campaign_spend": current["cumulative_cost_usd"],
                "cell_status": current["cells"][0]["status"],
                "cell_reservation": current["cells"][0]["reservation_usd"],
                "cell_run_id": current["cells"][0].get("run_id"),
                "cell_cost": current["cells"][0].get("cost_usd"),
            }
        )
        return _child_result(cost=0.018650, run_id="run_after_orphan_retry")

    state = execute_relight(
        relight_id=plan["relight_id"],
        budget_usd=5.0,
        output_root=relight_env,
        command_runner=runner,
    )

    assert dispatch_snapshots == [
        {
            "max_cost": "0.500000",
            "campaign_spend": 0.028664,
            "cell_status": "RUNNING",
            "cell_reservation": 0.50,
            "cell_run_id": None,
            "cell_cost": None,
        }
    ]
    assert state["status"] == "COMPLETED"
    assert state["cumulative_cost_usd"] == 0.047314
    assert state["cells"][0]["cost_usd"] == 0.018650
    assert state["cells"][0]["cost_attested"] is True
    assert state["cells"][0]["run_id"] == "run_after_orphan_retry"


def test_execution_charges_recorded_cost_and_checkpoints(relight_env):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    state = execute_relight(
        relight_id=plan["relight_id"],
        budget_usd=100.0,
        output_root=relight_env,
        command_runner=lambda argv, **_k: _child_result(cost=0.25),
    )
    assert state["status"] == "COMPLETED"
    assert state["cumulative_cost_usd"] == pytest.approx(0.25)
    assert state["cells"][0]["status"] == "COMPLETED"
    assert state["cells"][0]["cost_attested"] is True
    assert state["cells"][0]["run_id"] == "run_x"


def test_split_lineage_pre_step_runs_before_any_cell(relight_env, monkeypatch):
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    get_config.cache_clear()
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    events: list[str] = []

    def prepare(**kwargs):
        events.append("prepare")
        assert os.environ["VOE_SAFE_MODE"] == "true"
        assert kwargs["tickers"] == ["GPOR"]
        assert kwargs["as_of_date"] == plan["effective_as_of"]
        return {
            "as_of_date": plan["effective_as_of"],
            "requested": 1,
            "ready": 1,
            "unknown": 0,
            "results": [{"ticker": "GPOR", "status": "READY"}],
        }

    def runner(argv, **_kwargs):
        events.append("cell")
        assert os.environ["VOE_SAFE_MODE"] == "true"
        assert "env" not in _kwargs
        return _child_result(cost=0.25)

    monkeypatch.setattr(relight, "prepare_relight_split_lineage_evidence", prepare)
    state = execute_relight(
        relight_id=plan["relight_id"],
        budget_usd=100.0,
        output_root=relight_env,
        command_runner=runner,
    )

    assert events == ["prepare", "cell"]
    assert state["split_lineage_evidence"]["ready"] == 1
    assert state["split_lineage_evidence"]["unknown"] == 0
    assert os.environ["VOE_SAFE_MODE"] == "true"


def test_unattested_child_cost_is_charged_at_full_reservation(relight_env):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    unreadable = subprocess.CompletedProcess(
        args=["x"], returncode=0, stdout="not json at all", stderr=""
    )
    state = execute_relight(
        relight_id=plan["relight_id"],
        budget_usd=100.0,
        output_root=relight_env,
        command_runner=lambda argv, **_k: unreadable,
    )
    assert state["cells"][0]["cost_attested"] is False
    assert state["cells"][0]["reservation_usd"] == 0.50
    assert state["cumulative_cost_usd"] == 0.50


def test_completed_cells_are_skipped_on_resume(relight_env):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    execute_relight(
        relight_id=plan["relight_id"],
        budget_usd=100.0,
        output_root=relight_env,
        command_runner=lambda argv, **_k: _child_result(cost=0.25),
    )
    calls: list[list[str]] = []

    def runner(argv, **_kwargs):
        calls.append(list(argv))
        return _child_result()

    resumed = execute_relight(
        relight_id=plan["relight_id"],
        budget_usd=100.0,
        output_root=relight_env,
        command_runner=runner,
    )
    assert calls == []
    assert resumed["cumulative_cost_usd"] == pytest.approx(0.25)


def test_resume_refuses_when_recorded_spend_exceeds_new_ceiling(relight_env):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    execute_relight(
        relight_id=plan["relight_id"],
        budget_usd=100.0,
        output_root=relight_env,
        command_runner=lambda argv, **_k: _child_result(cost=5.0),
    )
    with pytest.raises(RelightAuthorizationError, match="already exceeds"):
        execute_relight(
            relight_id=plan["relight_id"],
            budget_usd=1.0,
            output_root=relight_env,
            accept_estimate_shortfall=True,
            command_runner=lambda argv, **_k: pytest.fail("must not spawn"),
        )


def test_child_argv_passes_explicit_tickers_and_strict_cost_cap(relight_env):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    captured: list[list[str]] = []

    def runner(argv, **_kwargs):
        captured.append(list(argv))
        return _child_result()

    execute_relight(
        relight_id=plan["relight_id"],
        budget_usd=100.0,
        output_root=relight_env,
        command_runner=runner,
    )
    argv = captured[0]
    assert "autonomous-sector-run" in argv
    assert "--strict-cost-cap" in argv
    assert argv[argv.index("--tickers") + 1] == "GPOR"
    assert argv[argv.index("--sector") + 1] == "energy"
    assert argv[argv.index("--market-cap-focus") + 1] == "mid_cap"
    # The ceiling is always handed to the child; no unbounded child is possible.
    assert float(argv[argv.index("--max-cost-usd") + 1]) > 0


def test_failed_cell_leaves_campaign_incomplete(relight_env):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    state = execute_relight(
        relight_id=plan["relight_id"],
        budget_usd=100.0,
        output_root=relight_env,
        command_runner=lambda argv, **_k: _child_result(returncode=3, cost=0.05),
    )
    assert state["status"] == "INCOMPLETE"
    assert state["cells"][0]["status"] == "FAILED"


def test_failed_cell_records_stdout_diagnostic_despite_noisy_stderr(relight_env):
    plan = plan_relight(tickers=["GPOR"], output_root=relight_env)
    failed = subprocess.CompletedProcess(
        args=["x"],
        returncode=3,
        stdout="NEEDS_DATA: MARKET_CAP_MISSING, QUOTE_PRICE_MISSING",
        stderr="INFO provider usage ledger closed cleanly",
    )

    state = execute_relight(
        relight_id=plan["relight_id"],
        budget_usd=100.0,
        output_root=relight_env,
        command_runner=lambda argv, **_kwargs: failed,
    )

    assert state["cells"][0]["error"] == (
        "stdout tail:\nNEEDS_DATA: MARKET_CAP_MISSING, QUOTE_PRICE_MISSING\n"
        "stderr tail:\nINFO provider usage ledger closed cleanly"
    )


def test_relight_never_writes_watchlist_rows_directly():
    """The runner must delegate persistence to the authorized publish path."""

    source = (relight.__file__).replace(".pyc", ".py")
    text = open(source, encoding="utf-8").read()
    for forbidden in ("INSERT INTO watchlist", "UPDATE watchlist", "DELETE FROM watchlist"):
        assert forbidden not in text
