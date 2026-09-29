from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.analyst import output_store as analyst_output_store
from app.autonomous.artifact_financial_audit import (
    UNAUDITED,
    financial_integrity_manifest_is_usable,
)
from app.watchlist import digest, store
from app.web.readmodel import company, coverage, reader, today
from app.web.readmodel.db import OfflineError


def _valid_manifest_payload(tmp_path: Path) -> dict:
    del tmp_path
    from app.config import get_config

    cfg = get_config()
    audit_roots = {
        "autonomous_sector": Path(cfg.runs_dir) / "autonomous_sector",
        "analyst_output": Path(cfg.analyst_outputs_dir),
        "scan": Path(cfg.outputs_dir) / "scans",
        "research_output": Path(cfg.research_dir),
        "watchlist_report": Path(cfg.outputs_dir) / "digests",
    }
    for audit_root in audit_roots.values():
        audit_root.mkdir(parents=True, exist_ok=True)
    sentinel = audit_roots["autonomous_sector"] / "audited_fixture.json"
    sentinel.write_text("{}\n", encoding="utf-8")
    return {
        "schema_version": "financial_integrity_audit_v1",
        "audit_scope_id": "ivi_current_decision_artifacts_v1",
        "generated_at": "2026-07-22T12:00:00Z",
        "complete": True,
        "source_roots": [
            {
                "family": family,
                "root_id": {
                    "autonomous_sector": "autonomous_sector_runs",
                    "analyst_output": "analyst_outputs",
                    "scan": "scan_outputs",
                    "research_output": "research_outputs",
                    "watchlist_report": "watchlist_reports",
                }[family],
                "path": str(root.resolve()),
            }
            for family, root in audit_roots.items()
        ],
        "summary": {
            "artifacts_scanned": 1,
            "tickers_scanned": 0,
            "violations": 0,
            "violations_by_invariant": {},
            "affected_run_ids": 0,
            "affected_tickers": 0,
            "affected_run_id_values": [],
            "affected_ticker_values": [],
            "earliest_date": None,
            "latest_date": None,
            "llm_consumed_violation_count": 0,
            "source_artifacts_rewritten": 0,
        },
        "invalid_run_ids": [],
        "artifacts": [
            {
                "path": str(sentinel.resolve()),
                "family": "autonomous_sector",
                "sha256": hashlib.sha256(sentinel.read_bytes()).hexdigest(),
                "integrity_status": "PASS",
                "decision_eligible": True,
                "run_id": None,
            }
        ],
        "violations": [],
    }


@pytest.fixture(params=("missing", "malformed", "empty"))
def unusable_manifest(request, tmp_path, monkeypatch) -> Path:
    path = tmp_path / f"{request.param}_financial_integrity_manifest.json"
    if request.param == "malformed":
        path.write_text("{not valid json", encoding="utf-8")
    elif request.param == "empty":
        payload = _valid_manifest_payload(tmp_path)
        payload["summary"]["artifacts_scanned"] = 0
        payload["artifacts"] = []
        path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("VOE_FINANCIAL_INTEGRITY_MANIFEST", str(path))
    return path


def _unexpected_db_access(*args, **kwargs):
    del args, kwargs
    raise AssertionError("fail-closed guard must run before database access")


def test_missing_or_malformed_manifest_blocks_current_watchlist_without_db_access(
    unusable_manifest, monkeypatch
):
    assert financial_integrity_manifest_is_usable() is False
    monkeypatch.setattr(store, "_connect", _unexpected_db_access)

    assert store.list_active() == []
    assert store.get_latest("AAA") is None
    assert store.watchlist_queue() == []
    assert store.stats()["total_entries"] == 0


def test_missing_or_malformed_manifest_blocks_digest_and_trigger_writes(
    unusable_manifest, tmp_path, monkeypatch
):
    monkeypatch.setattr(digest, "_connect", _unexpected_db_access)
    monkeypatch.setattr(digest, "check_watchlist_triggers", _unexpected_db_access)

    markdown = digest.render_digest()
    assert "BLOCKED: the canonical financial-integrity audit manifest" in markdown
    result = digest.write_digest(
        output_path=tmp_path / "blocked_digest.md",
        check_prices_first=True,
    )
    assert result.trigger_checked is False
    assert "Current decision rows are suppressed" in result.markdown


def test_missing_or_malformed_manifest_blocks_analyst_and_web_current_surfaces(
    unusable_manifest, tmp_path, monkeypatch
):
    monkeypatch.setattr(analyst_output_store, "get_db", _unexpected_db_access)
    assert analyst_output_store.latest_analysis_output_path("AAA", "analysis_report") is None

    class NoQueryConnection:
        execute = _unexpected_db_access

    conn = NoQueryConnection()
    assert coverage.coverage_atlas(conn)["cells"] == []
    assert coverage.coverage_cell(conn, sector="energy", band="mid_cap")["tickers"] == []
    assert today.open_decisions(conn) == []
    assert company.watchlist_profile(conn, "AAA") is None
    with pytest.raises(OfflineError) as exc_info:
        company.company_snapshot(conn, "AAA", queue_row=None)
    assert exc_info.value.precondition == "financial_integrity_manifest_unusable"

    outputs = Path(reader.outputs_dir())
    digest_dir = outputs / "digests"
    digest_dir.mkdir(parents=True, exist_ok=True)
    digest_path = digest_dir / "digest_2026-07-22.md"
    digest_path.write_text("# Historical digest\n\nAAA decision", encoding="utf-8")

    assert reader.library(None)["families"] == []
    assert today.latest_digest() is None
    rendered = reader.render_artifact(str(digest_path))
    assert rendered["integrity_status"] == UNAUDITED
    assert rendered["decision_eligible"] is False
    assert "excluded from current decisions" in rendered["html"]


def test_newest_malformed_discovered_manifest_does_not_fall_back(tmp_path, monkeypatch):
    from app.config import get_config

    monkeypatch.delenv("VOE_FINANCIAL_INTEGRITY_MANIFEST", raising=False)
    analysis_dir = Path(get_config().outputs_dir) / "analysis"
    analysis_dir.mkdir(parents=True)
    older = analysis_dir / "financial_integrity_audit_20260722T120000000000Z.json"
    older.write_text(json.dumps(_valid_manifest_payload(tmp_path)), encoding="utf-8")
    newer = analysis_dir / "financial_integrity_audit_20260722T130000000000Z.json"
    newer.write_text("{broken", encoding="utf-8")

    assert financial_integrity_manifest_is_usable() is False


def test_manifest_cache_is_bound_to_current_canonical_roots(tmp_path, monkeypatch):
    from app.config import get_config

    first_data_dir = tmp_path / "first_data"
    monkeypatch.setenv("VOE_DATA_DIR", str(first_data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(first_data_dir / "engine.db"))
    get_config.cache_clear()
    try:
        manifest_path = tmp_path / "root_bound_manifest.json"
        manifest_path.write_text(
            json.dumps(_valid_manifest_payload(tmp_path)),
            encoding="utf-8",
        )
        monkeypatch.setenv(
            "VOE_FINANCIAL_INTEGRITY_MANIFEST",
            str(manifest_path),
        )
        assert financial_integrity_manifest_is_usable() is True

        second_data_dir = tmp_path / "second_data"
        monkeypatch.setenv("VOE_DATA_DIR", str(second_data_dir))
        monkeypatch.setenv("VOE_DB_PATH", str(second_data_dir / "engine.db"))
        get_config.cache_clear()

        assert financial_integrity_manifest_is_usable() is False
    finally:
        get_config.cache_clear()


@pytest.mark.parametrize(
    "defect",
    (
        "artifact_count_mismatch",
        "unknown_integrity_status",
        "contradictory_eligibility",
        "malformed_source_root",
        "violation_count_mismatch",
        "pass_with_violation",
        "invalid_without_violation",
        "wrong_family",
        "out_of_root",
        "missing_root",
        "arbitrary_existing_root",
        "partial_root",
        "arbitrary_family",
        "duplicate_normalized_root",
        "duplicate_artifact_path",
        "missing_scope_id",
    ),
)
def test_incoherent_manifest_census_is_not_usable(defect, tmp_path, monkeypatch):
    payload = _valid_manifest_payload(tmp_path)
    if defect == "artifact_count_mismatch":
        payload["summary"]["artifacts_scanned"] = 2
    elif defect == "unknown_integrity_status":
        payload["artifacts"][0]["integrity_status"] = "UNKNOWN"
    elif defect == "contradictory_eligibility":
        payload["artifacts"][0]["decision_eligible"] = False
    elif defect == "malformed_source_root":
        payload["source_roots"] = ["fixture"]
    elif defect == "violation_count_mismatch":
        payload["summary"]["violations"] = 1
    elif defect == "pass_with_violation":
        record = payload["artifacts"][0]
        payload["summary"]["violations"] = 1
        payload["summary"]["violations_by_invariant"] = {"FIXTURE_INVALID": 1}
        payload["violations"] = [
            {
                "artifact_path": record["path"],
                "artifact_sha256": record["sha256"],
                "invariant": "FIXTURE_INVALID",
                "run_id": None,
                "ticker": None,
                "llm_consumed": False,
            }
        ]
    elif defect == "invalid_without_violation":
        payload["artifacts"][0]["integrity_status"] = "INVALID"
        payload["artifacts"][0]["decision_eligible"] = False
    elif defect == "wrong_family":
        payload["artifacts"][0]["family"] = "other"
    elif defect == "out_of_root":
        outside = tmp_path / "outside.json"
        outside.write_text("{}\n", encoding="utf-8")
        payload["artifacts"][0]["path"] = str(outside.resolve())
        payload["artifacts"][0]["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    elif defect == "missing_root":
        payload["source_roots"][0]["path"] = str((tmp_path / "missing_root").resolve())
    elif defect == "arbitrary_existing_root":
        arbitrary_root = tmp_path / "existing_but_wrong_root"
        arbitrary_root.mkdir()
        payload["source_roots"][0]["path"] = str(arbitrary_root.resolve())
    elif defect == "partial_root":
        payload["source_roots"].pop()
    elif defect == "arbitrary_family":
        payload["source_roots"][0]["family"] = "arbitrary"
    elif defect == "duplicate_normalized_root":
        first = Path(payload["source_roots"][0]["path"])
        payload["source_roots"][1]["path"] = str(first / ".." / first.name)
    elif defect == "duplicate_artifact_path":
        payload["artifacts"].append(dict(payload["artifacts"][0]))
        payload["summary"]["artifacts_scanned"] = 2
    else:
        payload.pop("audit_scope_id")
    path = tmp_path / f"{defect}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("VOE_FINANCIAL_INTEGRITY_MANIFEST", str(path))

    assert financial_integrity_manifest_is_usable() is False


def test_company_profile_requires_positive_run_authorization(tmp_path, monkeypatch):
    payload = _valid_manifest_payload(tmp_path)
    path = tmp_path / "valid_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("VOE_FINANCIAL_INTEGRITY_MANIFEST", str(path))

    class OneRowConnection:
        def execute(self, *args, **kwargs):
            del args, kwargs
            return self

        def fetchone(self):
            return {
                "id": 1,
                "thesis_text": "Unaudited current thesis",
                "key_risks_json": "[]",
                "open_questions_json": "[]",
                "valuation_anchor_method": "dcf",
                "valuation_anchor_value": 100.0,
                "current_price_at_addition": 80.0,
                "source_run_id": "unlisted_run",
                "cap_asof": "2026-07-22",
                "added_at": "2026-07-22T12:00:00Z",
            }

    assert financial_integrity_manifest_is_usable() is True
    assert company.watchlist_profile(OneRowConnection(), "AAA") is None
