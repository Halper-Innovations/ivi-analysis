from __future__ import annotations

from typing import Any


NEGATIVE_NEW_FLAGS = {
    "investigation",
    "restructuring",
    "going_concern",
    "material_weakness",
    "sec_subpoena",
    "bankruptcy",
}


def compute_change_momentum_adjustments(change_context: dict[str, Any] | None) -> tuple[dict[str, float], float, list[str]]:
    if not isinstance(change_context, dict):
        return (
            {"change_momentum_bonus": 0.0, "change_momentum_penalty": 0.0, "change_momentum_net": 0.0},
            0.0,
            [],
        )

    bonus = 0.0
    penalty = 0.0
    reasons: list[str] = []

    new_filing_recent = int(change_context.get("new_filing_recent_count") or 0)
    research_stale = bool(change_context.get("research_stale", True))
    if new_filing_recent > 0 and not research_stale:
        bonus += 3.0
        reasons.append("Recent filing change with fresh research coverage.")

    new_negative_flags = change_context.get("new_negative_flags") or []
    if isinstance(new_negative_flags, list):
        hit = sorted(flag for flag in new_negative_flags if flag in NEGATIVE_NEW_FLAGS)
        if hit:
            penalty += min(8.0, float(len(hit)) * 2.0)
            reasons.append(f"New negative risk flags triggered: {', '.join(hit[:5])}.")

    net = round(bonus - penalty, 2)
    return (
        {
            "change_momentum_bonus": round(bonus, 2),
            "change_momentum_penalty": -round(penalty, 2),
            "change_momentum_net": net,
        },
        net,
        reasons,
    )
