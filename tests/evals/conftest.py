from __future__ import annotations

from pathlib import Path

import pytest


FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "evals"


@pytest.fixture(
    params=(
        pytest.param(FIXTURES_ROOT / "single" / "autonomous_run.json", id="fixed-single"),
        pytest.param(
            FIXTURES_ROOT / "sector" / "autonomous_sector_run.json",
            id="fixed-sector",
        ),
    )
)
def artifact_path(request: pytest.FixtureRequest) -> Path:
    """Committed artifacts for the hermetic mechanical soundness gate."""
    return request.param


@pytest.fixture
def sector_report_path() -> Path:
    """Committed sector report for the hermetic mechanical shape gate."""
    return FIXTURES_ROOT / "sector" / "autonomous_sector_report.md"


@pytest.fixture
def single_report_path() -> Path:
    """Committed single-company report for the hermetic mechanical shape gate."""
    return FIXTURES_ROOT / "single" / "autonomous_research_report.md"
