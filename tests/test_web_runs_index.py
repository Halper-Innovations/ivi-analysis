from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.web.readmodel.runs_index import (
    UI_SCHEMA_SQL,
    discover_artifacts,
    list_indexed_runs,
    refresh_index,
)

V1_PAYLOAD = {
    "run_id": "autonomous_sector_biotech_20260101_abc123",
    "contract_version": "autonomous_sector_financial_run_v1",
    "sector": "biotech",
    "market_cap_focus": "mid_cap",
    "scan_family": "normal",
    "as_of_date": "2026-01-01",
    "created_at": "2026-01-01T10:00:00Z",
    "completed_at": "2026-01-01T11:00:00Z",
    "status": "COMPLETED",
    "final_verdict": "NO_SELECTION",
    "selected_ticker": None,
    "no_selection_reason": "No finalist cleared underwriting.",
    "company_packets": [{"ticker": "AAA"}, {"ticker": "BBB"}, {"ticker": "CCC"}],
    "relative_ranking": [{"ticker": "AAA"}],
}

V2_PAYLOAD = {
    "run_id": "all_sector_v2_20260715_aerospace_defense",
    "contract_version": "autonomous_sector_financial_run_v2",
    "pipeline_version": "v2",
    "sector": "aerospace_defense",
    "market_cap_focus": "large_and_mega",
    "scan_family": "normal",
    "as_of_date": "2026-07-15",
    "created_at": "2026-07-17T09:31:34+00:00",
    "completed_at": "2026-07-17T09:32:31+00:00",
    "status": "COMPLETED",
    "execution_status": "COMPLETED",
    "decision_status": "INCOMPLETE",
    "final_verdict": "SELECTED",
    "selected_ticker": "BA",
    "candidate_dispositions": [
        {"primary_ticker": "BA", "last_completed_stage": "DETERMINISTIC_SCREEN"},
        {"primary_ticker": "LHX", "last_completed_stage": "DETERMINISTIC_SCREEN"},
        {"primary_ticker": "HEI", "last_completed_stage": "UNDERWRITING"},
    ],
    "lane_usage": {
        "artifact_type": "autonomous_sector_lane_usage_v2",
        "cost_unit": "microdollars",
        "lanes": {
            "parent_research": {"cost_microdollars": 1000000},
            "company_underwriting": {"cost_microdollars": 234567},
        },
        "aggregate": {"cost_microdollars": 1234567},
    },
}


def _ui_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(UI_SCHEMA_SQL)
    return conn


def _build_tree(tmp_path: Path) -> Path:
    runs_dir = tmp_path / "runs"
    v1_dir = runs_dir / "autonomous_sector" / "autonomous_sector_biotech_20260101_abc123"
    v1_dir.mkdir(parents=True)
    (v1_dir / "autonomous_sector_run.json").write_text(json.dumps(V1_PAYLOAD), encoding="utf-8")
    (v1_dir / "autonomous_sector_report.md").write_text("# Report", encoding="utf-8")
    v2_dir = runs_dir / "all_sector_v2_replay" / "aerospace_defense"
    v2_dir.mkdir(parents=True)
    (v2_dir / "autonomous_sector_run.json").write_text(json.dumps(V2_PAYLOAD), encoding="utf-8")
    broken_dir = runs_dir / "autonomous_sector" / "autonomous_sector_broken_20260102_zzz999"
    broken_dir.mkdir(parents=True)
    (broken_dir / "autonomous_sector_run.json").write_text("{not json", encoding="utf-8")
    archive_dir = runs_dir / "_archive" / "old_run"
    archive_dir.mkdir(parents=True)
    (archive_dir / "autonomous_sector_run.json").write_text(
        json.dumps(V1_PAYLOAD), encoding="utf-8"
    )
    return runs_dir


def test_discover_artifacts_excludes_archive(tmp_path):
    runs_dir = _build_tree(tmp_path)
    paths = discover_artifacts(runs_dir)
    assert len(paths) == 3
    assert not any("_archive" in p.parts for p in paths)


def test_refresh_index_extracts_v1_and_v2_summaries(tmp_path):
    runs_dir = _build_tree(tmp_path)
    conn = _ui_conn()
    summary = refresh_index(conn, runs_dir=runs_dir)
    assert summary == {
        "discovered": 3,
        "parsed": 2,
        "unchanged": 0,
        "quarantined": 1,
        "removed": 0,
    }

    rows = {row["run_id"]: row for row in list_indexed_runs(conn)}
    v1 = rows["autonomous_sector_biotech_20260101_abc123"]
    assert v1["slug"] == "autonomous_sector/autonomous_sector_biotech_20260101_abc123"
    assert v1["sector"] == "biotech"
    assert v1["market_cap_focus"] == "mid_cap"
    assert v1["pipeline_version"] == "v1"
    assert v1["final_verdict"] == "NO_SELECTION"
    assert v1["no_selection_reason"] == "No finalist cleared underwriting."
    assert v1["examined_count"] == 3
    assert v1["cost_microdollars"] is None
    assert v1["parse_error"] is None
    assert v1["report_path"].endswith("autonomous_sector_report.md")

    v2 = rows["all_sector_v2_20260715_aerospace_defense"]
    assert v2["slug"] == "all_sector_v2_replay/aerospace_defense"
    assert v2["pipeline_version"] == "v2"
    assert v2["selected_ticker"] == "BA"
    assert v2["examined_count"] == 3
    assert v2["cost_microdollars"] == 1234567
    assert v2["disposition_counts_json"] == '{"DETERMINISTIC_SCREEN": 2, "UNDERWRITING": 1}'
    assert v2["report_path"] is None

    broken = rows["autonomous_sector_broken_20260102_zzz999"]
    assert broken["parse_error"] is not None
    assert broken["parse_error"].startswith("JSONDecodeError:")


def test_refresh_index_skips_unchanged_files(tmp_path):
    runs_dir = _build_tree(tmp_path)
    conn = _ui_conn()
    refresh_index(conn, runs_dir=runs_dir)
    second = refresh_index(conn, runs_dir=runs_dir)
    assert second == {
        "discovered": 3,
        "parsed": 0,
        "unchanged": 3,
        "quarantined": 0,
        "removed": 0,
    }


def test_refresh_index_removes_vanished_artifacts(tmp_path):
    runs_dir = _build_tree(tmp_path)
    conn = _ui_conn()
    refresh_index(conn, runs_dir=runs_dir)
    target = runs_dir / "all_sector_v2_replay" / "aerospace_defense" / "autonomous_sector_run.json"
    target.unlink()
    summary = refresh_index(conn, runs_dir=runs_dir)
    assert summary["removed"] == 1
    run_ids = [row["run_id"] for row in list_indexed_runs(conn)]
    assert "all_sector_v2_20260715_aerospace_defense" not in run_ids
    assert len(run_ids) == 2


def test_list_indexed_runs_filters_and_orders(tmp_path):
    runs_dir = _build_tree(tmp_path)
    conn = _ui_conn()
    refresh_index(conn, runs_dir=runs_dir)
    biotech_only = list_indexed_runs(conn, sector="biotech")
    assert [row["run_id"] for row in biotech_only] == ["autonomous_sector_biotech_20260101_abc123"]
    v2_only = list_indexed_runs(conn, pipeline_version="v2")
    assert [row["run_id"] for row in v2_only] == ["all_sector_v2_20260715_aerospace_defense"]
    ordered = list_indexed_runs(conn)
    # created_at DESC with NULLs (the quarantined row) last.
    assert [row["run_id"] for row in ordered] == [
        "all_sector_v2_20260715_aerospace_defense",
        "autonomous_sector_biotech_20260101_abc123",
        "autonomous_sector_broken_20260102_zzz999",
    ]
    with pytest.raises(ValueError):
        list_indexed_runs(conn, limit=0)


@pytest.mark.financial_integrity_contract
def test_list_indexed_runs_ignores_fabricated_cached_decision_fields(
    monkeypatch,
    tmp_path,
):
    from app.autonomous.financial_integrity import (
        FinancialIntegrityScope,
        require_financial_integrity_scope,
    )
    from app.autonomous.output_store import persist_autonomous_sector_run
    from app.autonomous.sector_contract import SectorCompanyFinancialPacket
    from app.autonomous.sector_runtime import _financial_integrity_run_binding
    from app.config import get_config
    from tests.financial_integrity_helpers import canonicalize_financial_packet
    from tests.test_classic_postwrite_authorization import (
        _baseline_manifest,
        _valid_classic_artifact,
    )

    _baseline_manifest(monkeypatch, tmp_path)
    run_id = "autonomous_sector_energy_20260724_cache"
    artifact = _valid_classic_artifact(run_id)
    packet = canonicalize_financial_packet(
        SectorCompanyFinancialPacket(
            ticker="AAA",
            financial_status="READY",
            model_fit_status="SUPPORTED",
            data_quality_status="COMPLETE",
            current_price=50.0,
            current_price_source="fixture_quote",
            current_price_source_url="https://example.test/quotes/AAA",
            financial_integrity_status="PASS",
            financial_integrity_violations=[],
            metric_traces={},
        ),
        as_of_date=artifact.as_of_date,
        shares_mm=10.0,
    )
    artifact.company_packets = [packet]
    scope = FinancialIntegrityScope(
        context="autonomous_sector_pre_provider",
        run_as_of_date=artifact.as_of_date,
        packets=(packet,),
    )
    gate = require_financial_integrity_scope(scope)
    artifact.candidate_selection["financial_integrity"] = gate.to_dict()
    artifact.candidate_selection["financial_integrity_binding"] = _financial_integrity_run_binding(
        scope,
        scope_fingerprint=gate.scope_fingerprint,
    )
    monkeypatch.setattr(
        "app.autonomous.output_store.bind_authorized_valuation_rows",
        lambda **_kwargs: 0,
    )
    persist_autonomous_sector_run(artifact)
    runs_dir = get_config().runs_dir
    conn = _ui_conn()
    refresh_index(conn, runs_dir=runs_dir)
    conn.execute(
        """
        UPDATE run_index
        SET final_verdict = 'SELECTED',
            selected_ticker = 'FORGED',
            sector = 'forged_sector'
        WHERE run_id = ?
        """,
        (run_id,),
    )
    conn.commit()

    rows = {row["run_id"]: row for row in list_indexed_runs(conn)}
    current = rows[run_id]
    assert current["sector"] == "energy"
    assert current["final_verdict"] == "NO_SELECTION"
    assert current["selected_ticker"] is None
    assert list_indexed_runs(conn, sector="forged_sector") == []
