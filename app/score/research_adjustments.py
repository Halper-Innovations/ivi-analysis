from __future__ import annotations

from typing import Any


NEGATIVE_RISK_FLAGS = {
    "guidance_lowered",
    "restructuring",
    "investigation",
    "sec_subpoena",
    "bankruptcy",
    "going_concern",
    "material_weakness",
}


def compute_research_adjustments(signals: dict[str, Any] | None) -> tuple[dict[str, float], float, list[str]]:
    if not isinstance(signals, dict):
        return (
            {"research_signal_bonus": 0.0, "research_signal_penalty": -4.0, "research_signal_net": -4.0},
            -4.0,
            ["Research signals missing; applied stale-coverage penalty."],
        )

    reasons: list[str] = []
    bonus = 0.0
    penalty = 0.0

    recency = signals.get("recency_days_min")
    if recency is None:
        penalty += 4.0
        reasons.append("No non-EDGAR recency signal available.")
    elif isinstance(recency, (int, float)) and recency > 180:
        penalty += 3.0
        reasons.append("External information flow is stale (>180 days).")

    item_count_30d = signals.get("item_count_30d")
    if isinstance(item_count_30d, (int, float)) and item_count_30d >= 3:
        bonus += 2.0
        reasons.append("Recent external information flow is active (>=3 items in 30 days).")

    sentiment_flags = signals.get("sentiment_flags") or []
    if isinstance(sentiment_flags, list):
        hit = sorted(flag for flag in sentiment_flags if flag in NEGATIVE_RISK_FLAGS)
        if hit:
            penalty += min(6.0, 1.5 * float(len(hit)))
            reasons.append(f"Negative research risk flags triggered: {', '.join(hit[:5])}.")

    net = round(bonus - penalty, 2)
    return (
        {
            "research_signal_bonus": round(bonus, 2),
            "research_signal_penalty": -round(penalty, 2),
            "research_signal_net": net,
        },
        net,
        reasons,
    )
