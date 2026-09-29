from __future__ import annotations

from pathlib import Path

import pytest

from tests.evals.eval_utils import artifact_soundness_results, format_failures


pytestmark = pytest.mark.eval_gate


def test_autonomous_artifact_soundness(artifact_path: Path):
    failures = [result for result in artifact_soundness_results(artifact_path) if not result.passed]

    assert not failures, format_failures(artifact_path, failures)
