from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from tests.evals.eval_utils import (
    EvalCheckResult,
    artifact_soundness_results,
    discover_artifact_paths,
    discover_sector_artifact_paths,
    discover_sector_report_paths,
    discover_single_report_paths,
    format_failures,
    load_json,
    report_shape_results,
)
from tests.evals.test_funnel_coverage import funnel_coverage_failures
from tests.evals.test_plausibility import plausibility_failures_for_artifact
from tests.evals.test_semantic_consistency import semantic_consistency_results


pytestmark = [pytest.mark.eval_gate, pytest.mark.live_artifact]


def _result_failures(
    paths: list[Path],
    evaluator: Callable[[Path], list[EvalCheckResult]],
) -> list[tuple[Path, list[EvalCheckResult]]]:
    failures: list[tuple[Path, list[EvalCheckResult]]] = []
    for path in paths:
        path_failures = [result for result in evaluator(path) if not result.passed]
        if path_failures:
            failures.append((path, path_failures))
    return failures


def _format_result_failures(
    failures: list[tuple[Path, list[EvalCheckResult]]],
) -> str:
    return "\n\n".join(
        format_failures(path, path_failures)
        for path, path_failures in failures
    )


def _sector_artifact_paths_or_skip() -> list[Path]:
    """Return the saved sector artifacts, or skip if there are none to audit.

    ``RUNS_ROOT`` is repo-relative, so a checkout without ``data/outputs/runs``
    -- a fresh worktree, another machine -- discovers nothing. Without this
    guard the collection loops below iterate zero paths and their
    ``assert not failures`` passes vacuously, reporting a green audit of an
    empty corpus. The path-based audits above already skip explicitly; these
    now do the same.
    """

    paths = discover_sector_artifact_paths()
    if not paths:
        pytest.skip("no saved autonomous sector artifacts found")
    return paths


def test_live_autonomous_artifact_soundness() -> None:
    """Audit every current saved artifact; discovery happens only in the call."""
    paths = discover_artifact_paths()
    if not paths:
        pytest.skip("no saved autonomous artifacts found")

    failures = _result_failures(paths, artifact_soundness_results)

    assert not failures, _format_result_failures(failures)


def test_live_autonomous_sector_report_shape() -> None:
    """Audit every current saved sector report at test-call time."""
    paths = discover_sector_report_paths()
    if not paths:
        pytest.skip("no saved autonomous sector reports found")

    failures = _result_failures(paths, report_shape_results)

    assert not failures, _format_result_failures(failures)


def test_live_autonomous_single_candidate_report_shape() -> None:
    """Audit every current saved single-company report at test-call time."""
    paths = discover_single_report_paths()
    if not paths:
        pytest.skip("no saved autonomous single-company reports found")

    failures = _result_failures(paths, report_shape_results)

    assert not failures, _format_result_failures(failures)


def test_live_sector_artifact_funnel_coverage() -> None:
    failures: list[str] = []
    for path in _sector_artifact_paths_or_skip():
        failures.extend(funnel_coverage_failures(load_json(path), path=path))

    assert not failures, "\n".join(failures)


def test_live_sector_artifact_plausibility_magnitude() -> None:
    failures: list[str] = []
    for path in _sector_artifact_paths_or_skip():
        failures.extend(
            plausibility_failures_for_artifact(load_json(path), path=path)
        )

    assert not failures, "\n".join(failures)


def test_live_sector_artifact_semantic_consistency() -> None:
    failures: list[EvalCheckResult] = []
    for path in _sector_artifact_paths_or_skip():
        failures.extend(
            result
            for result in semantic_consistency_results(load_json(path), path=path)
            if not result.passed
        )

    assert not failures, "\n".join(failure.message for failure in failures)
