from __future__ import annotations

from pathlib import Path

import pytest

from tests.evals.eval_utils import format_failures, report_shape_results


pytestmark = pytest.mark.eval_gate


def test_autonomous_sector_report_shape(sector_report_path: Path):
    failures = [result for result in report_shape_results(sector_report_path) if not result.passed]

    assert not failures, format_failures(sector_report_path, failures)


def test_autonomous_single_candidate_report_shape(single_report_path: Path):
    failures = [result for result in report_shape_results(single_report_path) if not result.passed]

    assert not failures, format_failures(single_report_path, failures)
