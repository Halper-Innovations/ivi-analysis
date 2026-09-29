from __future__ import annotations

import pytest

from app.rlm.schemas import Action, PlannerOutput


def test_planner_output_validates_and_normalizes_actions():
    payload = PlannerOutput.model_validate(
        {
            "rlm_version": "v0",
            "iteration": 1,
            "objective": "Improve ranking confidence",
            "actions": [
                {
                    "action_type": "RUN_SYNTHESIS",
                    "target": {"scope": "ticker", "value": "aapl"},
                },
                {
                    "action_type": "STOP",
                    "reason_code": "DONE",
                    "summary": "No more useful actions",
                },
            ],
        }
    )
    assert payload.actions[0].target == {"scope": "ticker", "value": "AAPL"}
    assert payload.actions[-1].action_type == "STOP"


def test_action_enum_is_enforced():
    with pytest.raises(Exception):
        Action.model_validate({"action_type": "FREEFORM"})


def test_run_synthesis_requires_target_scope_and_value():
    with pytest.raises(Exception):
        Action.model_validate({"action_type": "RUN_SYNTHESIS", "target": {"scope": "sector"}})

    with pytest.raises(Exception):
        Action.model_validate({"action_type": "RUN_SYNTHESIS", "target": {"value": "Software"}})


def test_planner_stop_must_be_last():
    with pytest.raises(Exception):
        PlannerOutput.model_validate(
            {
                "iteration": 0,
                "objective": "bad stop placement",
                "actions": [
                    {
                        "action_type": "STOP",
                        "reason_code": "x",
                        "summary": "stop",
                    },
                    {"action_type": "REBUILD_SCOREBOARD"},
                ],
            }
        )
