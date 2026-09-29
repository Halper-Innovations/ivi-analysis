"""Input guardrails for AI-planned deterministic tool calls."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.util.companyfacts_aliases import normalize_companyfacts_line_items


DEFAULT_COMPANYFACTS_LINE_ITEMS = [
    "revenue",
    "operating_income",
    "net_income",
    "cfo",
    "capex",
    "cash",
    "total_debt",
    "shares_outstanding",
]

DEFAULT_FILING_KEYWORDS = [
    "risk factors",
    "liquidity",
    "competition",
    "revenue",
    "cash flow",
]

DEFAULT_PEER_METRIC = "roic"
DEFAULT_COMPANYFACTS_YEARS = 5
SUPPORTED_PEER_METRICS = {"roic", "operating_margin", "revenue_growth_5y"}

@dataclass(frozen=True)
class ToolInputRepairResult:
    """Repaired tool input plus audit notes for the run artifact."""

    tool_input: dict[str, Any]
    repair_notes: list[str]
    degraded_states: list[str]


def _clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    return list(dict.fromkeys(cleaned))


def _clean_metric_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return _clean_string_list(value)
    if value is None:
        return []
    token = str(value).strip()
    return [token] if token else []


def repair_alpha_tool_input(tool_name: str, tool_input: dict[str, Any] | None) -> ToolInputRepairResult:
    """Repair common missing required inputs without taking tool choice away from the AI.

    The autonomous runtimes intentionally let the provider select tools. This function
    only protects deterministic tools from obvious schema omissions that would otherwise
    burn budget on non-informative errors.
    """

    name = str(tool_name or "").strip()
    repaired = dict(tool_input or {})
    notes: list[str] = []
    degraded_states: list[str] = []

    if name == "fetch_companyfacts_timeseries":
        raw_line_items = _clean_string_list(repaired.get("line_items"))
        line_items, translated = normalize_companyfacts_line_items(raw_line_items)
        if translated:
            notes.append(
                "translated companyfacts line_items aliases to normalized annual fields"
            )
            degraded_states.append("TOOL_INPUT_REPAIRED")
        if not line_items:
            raw_metrics = [
                *_clean_metric_values(repaired.get("metric")),
                *_clean_metric_values(repaired.get("metrics")),
                *_clean_metric_values(repaired.get("line_item")),
            ]
            line_items, translated = normalize_companyfacts_line_items(raw_metrics)
            if line_items:
                notes.append(
                    "translated companyfacts metric aliases to normalized annual fields"
                )
                degraded_states.append("TOOL_INPUT_REPAIRED")
            else:
                line_items = list(DEFAULT_COMPANYFACTS_LINE_ITEMS)
                notes.append("defaulted missing companyfacts line_items to core annual financial fields")
                degraded_states.append("TOOL_INPUT_DEFAULTED")
        repaired["line_items"] = line_items
        if repaired.get("years") is None:
            repaired["years"] = DEFAULT_COMPANYFACTS_YEARS

    elif name == "fetch_filing_section":
        keywords = _clean_string_list(repaired.get("keywords"))
        if not keywords:
            section_focus = str(repaired.get("section_focus") or "").strip()
            keywords = [section_focus] if section_focus else list(DEFAULT_FILING_KEYWORDS)
            notes.append("defaulted missing filing keywords to broad financial-risk terms")
            degraded_states.append("TOOL_INPUT_DEFAULTED")
        repaired["keywords"] = keywords

    elif name == "compare_peer_metric":
        metric = str(repaired.get("metric") or "").strip()
        if not metric:
            repaired["metric"] = DEFAULT_PEER_METRIC
            notes.append("defaulted missing peer metric to roic")
            degraded_states.append("TOOL_INPUT_DEFAULTED")
        elif metric not in SUPPORTED_PEER_METRICS:
            notes.append("left unsupported peer metric unchanged for tool-level rejection")
            degraded_states.append("TOOL_INPUT_UNSUPPORTED")

    return ToolInputRepairResult(
        tool_input=repaired,
        repair_notes=notes,
        degraded_states=list(dict.fromkeys(degraded_states)),
    )
