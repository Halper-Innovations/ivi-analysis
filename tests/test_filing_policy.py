from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

from app.ingest.filings import _coverage_score, select_filing_stubs_for_policy
from app.ingest.sec_client import FilingStub


def _stub(form: str, filing_date: str, accession: str) -> FilingStub:
    cik = "320193"
    nodash = accession.replace("-", "")
    return FilingStub(
        cik=cik,
        accession=accession,
        accession_nodash=nodash,
        form_type=form,
        filing_date=date.fromisoformat(filing_date),
        period_end="2025-12-31",
        primary_document="doc.htm",
        primary_doc_url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/doc.htm",
        filing_index_url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/index.json",
    )


def test_filing_selection_policy_picks_most_recent_per_form():
    filings = [
        _stub("10-K", "2024-06-01", "0001-24-000001"),
        _stub("10-K", "2025-12-20", "0001-25-000010"),
        _stub("10-Q", "2025-11-01", "0001-25-000020"),
        _stub("10-Q", "2025-08-01", "0001-25-000015"),
        _stub("8-K", "2025-12-28", "0001-25-000030"),
        _stub("DEF 14A", "2024-01-01", "0001-24-000050"),  # too old for 540d as of 2026-02-13
    ]
    windows = {"10-K": 540, "10-Q": 210, "8-K": 90, "20-F": 540, "DEF 14A": 540}
    selected = select_filing_stubs_for_policy(
        filings,
        as_of_date=date.fromisoformat("2026-02-13"),
        windows_days=windows,
    )
    by_form = {f.form_type: f.accession for f in selected}
    assert by_form["10-K"] == "0001-25-000010"
    assert by_form["10-Q"] == "0001-25-000020"
    assert by_form["8-K"] == "0001-25-000030"
    assert "DEF 14A" not in by_form


def test_filing_selection_policy_includes_amended_forms():
    filings = [
        _stub("10-K/A", "2025-12-20", "0001-25-000010"),
        _stub("10-Q/A", "2025-11-01", "0001-25-000020"),
        _stub("8-K/A", "2025-12-28", "0001-25-000030"),
        _stub("20-F/A", "2025-09-01", "0001-25-000040"),
    ]
    windows = {
        "10-K": 540,
        "10-K/A": 540,
        "10-Q": 210,
        "10-Q/A": 210,
        "8-K": 90,
        "8-K/A": 90,
        "20-F": 540,
        "20-F/A": 540,
        "40-F": 540,
        "40-F/A": 540,
        "DEF 14A": 540,
    }
    selected = select_filing_stubs_for_policy(
        filings,
        as_of_date=date.fromisoformat("2026-02-13"),
        windows_days=windows,
    )
    by_form = {f.form_type: f.accession for f in selected}
    assert by_form["10-K/A"] == "0001-25-000010"
    assert by_form["10-Q/A"] == "0001-25-000020"
    assert by_form["8-K/A"] == "0001-25-000030"
    assert by_form["20-F/A"] == "0001-25-000040"


def test_coverage_score_counts_amended_forms_toward_required_buckets():
    score, missing_required = _coverage_score({"10-K/A", "10-Q/A", "8-K/A", "DEF 14A"})
    assert score == 100.0
    assert missing_required == []


@pytest.mark.subprocess
def test_ingest_module_imports_without_dossier_cycle():
    result = subprocess.run(
        [sys.executable, "-c", "import app.ingest.filings"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
