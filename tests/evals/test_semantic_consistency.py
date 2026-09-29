from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# The recommendation-parsing logic is shared with the write-time consistency
# check in app/watchlist/store.py so the two cannot drift.
from app.autonomous.semantic_consistency import (
    ACTIONABLE_VERDICTS,
    normalize_verdict as _normalize_verdict,
    recommended_verdicts as _recommended_verdicts,
)
from tests.evals.eval_utils import (
    EvalCheckResult,
)


pytestmark = pytest.mark.eval_gate


def _candidate_theses(data: dict[str, Any]) -> dict[str, str]:
    memo_body = data.get("memo_body") if isinstance(data.get("memo_body"), dict) else {}
    candidates = memo_body.get("candidates") if isinstance(memo_body.get("candidates"), dict) else {}
    theses: dict[str, str] = {}
    for ticker, section in candidates.items():
        if not isinstance(section, dict):
            continue
        thesis = section.get("thesis") or section.get("thesis_text")
        if thesis:
            theses[str(ticker).upper()] = str(thesis)
    return theses


def _candidate_verdicts(data: dict[str, Any]) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    for row in data.get("relative_ranking") or []:
        if not isinstance(row, dict) or not row.get("ticker"):
            continue
        verdict = _normalize_verdict(
            row.get("conviction_grade")
            or row.get("company_autonomy_verdict")
            or row.get("verdict")
            or row.get("final_verdict")
        )
        if verdict:
            verdicts[str(row["ticker"]).upper()] = verdict

    selected_ticker = str(data.get("selected_ticker") or "").upper()
    final_verdict = _normalize_verdict(data.get("final_verdict"))
    if selected_ticker and final_verdict in ACTIONABLE_VERDICTS and selected_ticker not in verdicts:
        verdicts[selected_ticker] = final_verdict
    return verdicts


def semantic_consistency_results(
    data: dict[str, Any],
    *,
    path: Path = Path("synthetic_artifact.json"),
) -> list[EvalCheckResult]:
    theses = _candidate_theses(data)
    verdicts = _candidate_verdicts(data)
    failures: list[str] = []

    for ticker, thesis in sorted(theses.items()):
        recommendations = _recommended_verdicts(thesis)
        if not recommendations:
            continue
        structured_verdict = verdicts.get(ticker)
        if "WATCHLIST_ONLY" in recommendations and structured_verdict in ACTIONABLE_VERDICTS:
            failures.append(
                f"{ticker}: thesis recommends WATCHLIST_ONLY but structured verdict is {structured_verdict}"
            )
        if "AVOID" in recommendations and structured_verdict != "AVOID":
            failures.append(
                f"{ticker}: thesis recommends AVOID but structured verdict is {structured_verdict}"
            )

    return [
        EvalCheckResult(
            check="semantic_final_verdict_consistency",
            passed=not failures,
            message=f"{path}: semantic verdict mismatch(es): {failures}",
        )
    ]


def test_semantic_consistency_accepts_matching_watchlist_recommendation() -> None:
    data = {
        "memo_body": {
            "candidates": {
                "CXM": {
                    "thesis": "The correct verdict is WATCHLIST_ONLY until the valuation gap closes."
                }
            }
        },
        "relative_ranking": [{"ticker": "CXM", "company_autonomy_verdict": "WATCHLIST_ONLY"}],
    }

    assert semantic_consistency_results(data)[0].passed


def test_semantic_consistency_flags_watchlist_recommendation_promoted_to_actionable() -> None:
    data = {
        "memo_body": {
            "candidates": {
                "CXM": {
                    "thesis": "The correct company-level verdict at WATCHLIST_ONLY rather than actionable."
                }
            }
        },
        "relative_ranking": [{"ticker": "CXM", "company_autonomy_verdict": "ACTIONABLE"}],
    }

    result = semantic_consistency_results(data)[0]

    assert not result.passed
    assert "CXM" in result.message
    assert "WATCHLIST_ONLY" in result.message


def test_semantic_consistency_flags_avoid_recommendation_with_non_avoid_verdict() -> None:
    data = {
        "memo_body": {
            "candidates": {
                "BAD": {"thesis": "The final verdict is avoid because the capital loss risk is permanent."}
            }
        },
        "relative_ranking": [{"ticker": "BAD", "company_autonomy_verdict": "WATCHLIST_ONLY"}],
    }

    result = semantic_consistency_results(data)[0]

    assert not result.passed
    assert "AVOID" in result.message
