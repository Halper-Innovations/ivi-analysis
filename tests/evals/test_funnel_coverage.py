from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.eval_gate


DROP_RATIO_LIMIT = 0.9
MIN_LOADED_FOR_DROP_CHECK = 20
HIGH_UNUSABLE_RATIO = 0.5


def _count_items(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _warning_strings(data: dict[str, Any]) -> list[str]:
    candidate_selection = data.get("candidate_selection") if isinstance(data.get("candidate_selection"), dict) else {}
    warnings = list(candidate_selection.get("warnings") or [])
    warnings.extend(data.get("universe_warnings") or [])
    return [str(warning) for warning in warnings]


def _filtered_unusable_count(data: dict[str, Any]) -> int:
    count = 0
    for warning in _warning_strings(data):
        match = re.search(r"FILTERED_UNUSABLE_CANDIDATES:(\d+)", warning)
        if match:
            count = max(count, int(match.group(1)))
    return count


def _loaded_and_selected_counts(data: dict[str, Any]) -> tuple[int, int]:
    candidate_selection = data.get("candidate_selection") if isinstance(data.get("candidate_selection"), dict) else {}
    loaded = _count_items(candidate_selection.get("loaded_tickers")) or _count_items(
        candidate_selection.get("loaded_companies")
    )
    selected = _count_items(candidate_selection.get("selected_tickers")) or _count_items(
        candidate_selection.get("selected_candidates")
    )
    return loaded, selected


def _delta_audit_accounted(data: dict[str, Any], loaded: int, selected: int) -> bool:
    """Delta sweeps (--only-unswept) intentionally drop already-covered names.

    The audit reconstructs the pre-delta selection (swept + carried +
    to_review); the drop check then applies to the NORMAL funnel portion
    (loaded -> pre-delta selected), so a delta run is held to the same
    standard as the full sweep it filters.
    """
    candidate_selection = data.get("candidate_selection") if isinstance(data.get("candidate_selection"), dict) else {}
    audit = candidate_selection.get("delta_audit")
    if not isinstance(audit, dict) or not audit.get("only_unswept"):
        return False
    swept = _count_items(audit.get("swept_excluded"))
    carried = len(audit.get("carried_verdicts") or {})
    to_review = _count_items(audit.get("to_review"))
    if selected != to_review:
        return False
    pre_delta_selected = swept + carried + to_review
    funnel_drop = (loaded - pre_delta_selected) / loaded
    return funnel_drop <= DROP_RATIO_LIMIT


def funnel_coverage_failures(
    data: dict[str, Any],
    *,
    path: Path = Path("synthetic_artifact.json"),
) -> list[str]:
    loaded, selected = _loaded_and_selected_counts(data)
    if loaded < MIN_LOADED_FOR_DROP_CHECK or loaded <= 0:
        return []

    drop_ratio = (loaded - selected) / loaded
    if drop_ratio <= DROP_RATIO_LIMIT:
        return []

    if _delta_audit_accounted(data, loaded, selected):
        return []

    unusable_count = _filtered_unusable_count(data)
    if unusable_count / loaded >= HIGH_UNUSABLE_RATIO:
        return []

    return [
        f"{path}: EXCESSIVE_DROP loaded={loaded} selected={selected} "
        f"drop_ratio={drop_ratio:.2%} filtered_unusable={unusable_count}"
    ]


def test_funnel_coverage_accepts_low_drop_ratio() -> None:
    data = {"candidate_selection": {"loaded_tickers": [f"T{i}" for i in range(100)], "selected_tickers": [f"T{i}" for i in range(40)]}}

    assert funnel_coverage_failures(data) == []


def test_funnel_coverage_accepts_high_unusable_filtering() -> None:
    data = {
        "candidate_selection": {
            "loaded_tickers": [f"T{i}" for i in range(57)],
            "selected_tickers": ["ACT", "BHF", "HG", "NMIH"],
            "warnings": ["FILTERED_UNUSABLE_CANDIDATES:53", "SELECTABLE_CANDIDATES_BELOW_LIMIT:4/8"],
        }
    }

    assert funnel_coverage_failures(data) == []


def test_funnel_coverage_accepts_delta_audit_accounted_drop() -> None:
    data = {
        "candidate_selection": {
            "loaded_tickers": [f"T{i}" for i in range(107)],
            "selected_tickers": ["T0"],
            "delta_audit": {
                "only_unswept": True,
                "swept_excluded": [f"T{i}" for i in range(1, 107)],
                "carried_verdicts": {},
                "to_review": ["T0"],
            },
        }
    }

    assert funnel_coverage_failures(data) == []


def test_funnel_coverage_still_flags_unaccounted_delta_drop() -> None:
    # Pre-delta selection was 4 of 100 (96% normal-funnel drop) — the delta
    # audit must not launder a drop the full sweep would have been flagged for.
    data = {
        "candidate_selection": {
            "loaded_tickers": [f"T{i}" for i in range(100)],
            "selected_tickers": ["T0"],
            "delta_audit": {
                "only_unswept": True,
                "swept_excluded": ["T1", "T2", "T3"],
                "carried_verdicts": {},
                "to_review": ["T0"],
            },
        }
    }

    failures = funnel_coverage_failures(data)

    assert len(failures) == 1
    assert "EXCESSIVE_DROP" in failures[0]


def test_funnel_coverage_flags_excessive_drop_without_unusable_filtering() -> None:
    data = {
        "candidate_selection": {
            "loaded_tickers": [f"T{i}" for i in range(100)],
            "selected_tickers": [f"T{i}" for i in range(8)],
            "warnings": [],
        }
    }

    failures = funnel_coverage_failures(data)

    assert len(failures) == 1
    assert "EXCESSIVE_DROP" in failures[0]
    assert "loaded=100" in failures[0]
    assert "selected=8" in failures[0]
