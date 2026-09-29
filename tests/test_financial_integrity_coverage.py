from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from app.autonomous import sweep_delta
from app.web.readmodel import coverage


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


@pytest.mark.parametrize(
    ("pipeline_version", "complete_disposition"),
    [
        ("v1", "LLM_CANDIDATE_REVIEW_COMPLETED"),
        ("v2", "UNDERWRITTEN"),
    ],
)
def test_invalid_latest_coverage_reopens_without_resurrecting_older_row(
    monkeypatch, pipeline_version, complete_disposition
):
    conn = _conn()
    old_run = f"{pipeline_version}_old_valid"
    invalid_latest_run = f"{pipeline_version}_new_invalid"
    sweep_delta.record_loaded_set(
        conn,
        run_id=old_run,
        sector="energy",
        market_cap_focus="mid_cap",
        source="sector_scan_db",
        tickers=["AAA"],
        loaded_at="2026-07-20T12:00:00Z",
        pipeline_version=pipeline_version,
        candidate_dispositions={"AAA": complete_disposition},
    )
    sweep_delta.record_loaded_set(
        conn,
        run_id=invalid_latest_run,
        sector="energy",
        market_cap_focus="mid_cap",
        source="sector_scan_db",
        tickers=["AAA"],
        loaded_at="2026-07-21T12:00:00Z",
        pipeline_version=pipeline_version,
        candidate_dispositions={"AAA": complete_disposition},
    )
    authorized_runs = {old_run}
    monkeypatch.setattr(
        sweep_delta,
        "_authorized_terminal_dispositions",
        lambda run_id, *, pipeline_version, conn=None: (
            {"AAA": complete_disposition} if run_id in authorized_runs else None
        ),
    )

    assert (
        sweep_delta.swept_tickers_for_band(
            conn,
            "mid_cap",
            "energy",
            pipeline_version=pipeline_version,
        )
        == set()
    )

    # A genuinely newer valid run can close coverage again.
    sweep_delta.record_loaded_set(
        conn,
        run_id=f"{pipeline_version}_newest_valid",
        sector="energy",
        market_cap_focus="mid_cap",
        source="sector_scan_db",
        tickers=["AAA"],
        loaded_at="2026-07-22T12:00:00Z",
        pipeline_version=pipeline_version,
        candidate_dispositions={"AAA": complete_disposition},
    )
    assert (
        sweep_delta.swept_tickers_for_band(
            conn,
            "mid_cap",
            "energy",
            pipeline_version=pipeline_version,
        )
        == set()
    )
    authorized_runs.add(f"{pipeline_version}_newest_valid")
    assert sweep_delta.swept_tickers_for_band(
        conn,
        "mid_cap",
        "energy",
        pipeline_version=pipeline_version,
    ) == {"AAA"}


def test_web_coverage_excludes_invalid_latest_run_and_preserves_ledger_history(monkeypatch):
    conn = _conn()
    old_run = "autonomous_sector_energy_20260720_old"
    invalid_latest_run = "autonomous_sector_energy_20260721_invalid"
    for run_id, loaded_at in (
        (old_run, "2026-07-20T12:00:00Z"),
        (invalid_latest_run, "2026-07-21T12:00:00Z"),
    ):
        sweep_delta.record_loaded_set(
            conn,
            run_id=run_id,
            sector="energy",
            market_cap_focus="mid_cap",
            source="sector_scan_db",
            tickers=["AAA"],
            loaded_at=loaded_at,
            pipeline_version="v1",
            candidate_dispositions={"AAA": "LLM_CANDIDATE_REVIEW_COMPLETED"},
        )
    monkeypatch.setattr(
        sweep_delta,
        "_authorized_terminal_dispositions",
        lambda run_id, *, pipeline_version, conn=None: (
            {"AAA": "LLM_CANDIDATE_REVIEW_COMPLETED"} if run_id == old_run else None
        ),
    )
    monkeypatch.setattr(
        coverage,
        "run_id_is_decision_eligible",
        lambda run_id: run_id == old_run,
    )

    cell = coverage.coverage_cell(conn, sector="energy", band="mid_cap")
    assert cell["tickers"] == []
    assert cell["runs"] == []
    atlas = coverage.coverage_atlas(conn)
    assert atlas["cells"] == []
    assert (
        conn.execute("SELECT COUNT(*) FROM sector_run_loaded_sets WHERE ticker = 'AAA'").fetchone()[
            0
        ]
        == 2
    )


def test_artifact_backfill_skips_manifest_invalid_run_without_ledger_write(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "autonomous_sector" / "autonomous_sector_energy_invalid"
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "autonomous_sector_run.json"
    artifact_path.write_text(
        json.dumps(
            {
                "run_id": "autonomous_sector_energy_invalid",
                "status": "COMPLETED",
                "sector": "energy",
                "market_cap_focus": "mid_cap",
                "created_at": "2026-07-22T12:00:00Z",
                "candidate_selection": {
                    "sector": "energy",
                    "market_cap_focus": "mid_cap",
                    "source": "sector_scan_db",
                    "loaded_tickers": ["AAA"],
                },
            }
        ),
        encoding="utf-8",
    )
    conn = _conn()

    @contextmanager
    def fake_get_db():
        yield conn

    monkeypatch.setattr("app.db.get_db", fake_get_db)
    monkeypatch.setattr(
        sweep_delta,
        "active_financial_integrity_manifest_path",
        lambda: Path("/configured/financial_integrity_manifest.json"),
    )
    monkeypatch.setattr(
        sweep_delta,
        "authorized_artifact_bytes",
        lambda path, manifest: ("INVALID", None),
    )

    result = sweep_delta.backfill_loaded_sets_from_artifacts(runs_root)
    assert result == {
        "artifacts": 1,
        "runs_recorded": 0,
        "rows_inserted": 0,
        "skipped": 1,
    }
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' "
            "AND name = 'sector_run_loaded_sets'"
        ).fetchone()[0]
        == 0
    )


def test_artifact_backfill_consumes_authorized_bytes_not_swapped_closing_bytes(
    tmp_path, monkeypatch
):
    runs_root = tmp_path / "runs"
    run_id = "autonomous_sector_energy_byte_swap"
    run_dir = runs_root / "autonomous_sector" / run_id
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "autonomous_sector_run.json"
    non_closing = {
        "run_id": run_id,
        "status": "COMPLETED",
        "pipeline_version": "v1",
        "created_at": "2026-07-22T12:00:00Z",
        "candidate_selection": {
            "sector": "energy",
            "market_cap_focus": "mid_cap",
            "source": "sector_scan_db",
            "loaded_tickers": ["AAA"],
        },
        "company_packets": [{"ticker": "AAA"}],
        "memo_body": {
            "candidates": {
                "AAA": {
                    "ticker": "AAA",
                    "source": "deterministic_fallback",
                    "status": "DEGRADED_STATE",
                    "thesis": "Fallback thesis.",
                    "key_risks": ["Fallback risk."],
                    "falsifiers": ["Fallback falsifier."],
                    "open_questions": ["Fallback question?"],
                }
            }
        },
        "provider_usage": [],
    }
    forged_closing = {
        **non_closing,
        "memo_body": {
            "candidates": {
                "AAA": {
                    "ticker": "AAA",
                    "source": "llm",
                    "status": "OK",
                    "thesis": "Forged provider thesis.",
                    "key_risks": ["Forged risk."],
                    "falsifiers": ["Forged falsifier."],
                    "open_questions": ["Forged question?"],
                }
            }
        },
        "provider_usage": [
            {
                "schema_name": "autonomous_sector_candidate_memo_aaa",
                "status": "OK",
            }
        ],
    }
    artifact_path.write_text(json.dumps(non_closing), encoding="utf-8")
    authorized_bytes = artifact_path.read_bytes()

    conn = _conn()

    @contextmanager
    def fake_get_db():
        yield conn

    monkeypatch.setattr("app.db.get_db", fake_get_db)
    monkeypatch.setattr(
        sweep_delta,
        "active_financial_integrity_manifest_path",
        lambda: Path("/configured/financial_integrity_manifest.json"),
    )
    monkeypatch.setattr(
        sweep_delta,
        "financial_integrity_manifest_is_usable",
        lambda _manifest=None: True,
    )

    def authorize_then_swap(path, manifest):
        assert Path(path) == artifact_path
        assert manifest == Path("/configured/financial_integrity_manifest.json")
        artifact_path.write_text(json.dumps(forged_closing), encoding="utf-8")
        return "PASS", authorized_bytes

    monkeypatch.setattr(
        sweep_delta,
        "authorized_artifact_bytes",
        authorize_then_swap,
    )
    result = sweep_delta.backfill_loaded_sets_from_artifacts(runs_root)
    assert result == {
        "artifacts": 1,
        "runs_recorded": 1,
        "rows_inserted": 1,
        "skipped": 0,
    }
    row = conn.execute(
        """
        SELECT candidate_disposition, coverage_complete
        FROM sector_run_loaded_sets
        WHERE ticker = 'AAA'
        """
    ).fetchone()
    assert tuple(row) == ("LLM_CANDIDATE_REVIEW_FAILED", 0)
    assert (
        sweep_delta.swept_tickers_for_band(
            conn,
            "mid_cap",
            "energy",
            pipeline_version="v1",
        )
        == set()
    )
