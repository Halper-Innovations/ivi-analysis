from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.autonomous.artifact_financial_audit import (
    INVALID,
    MISSING_PROVENANCE_UNREPAIRABLE,
    PASS,
    REQUIRES_LLM_REREVIEW,
    STALE_AUDIT,
    UNAUDITED,
    artifact_decision_eligibility,
    audit_artifact_tree,
    audit_payload,
    authorized_artifact_bytes,
    authorized_run_artifact_binding,
    financial_integrity_manifest_is_usable,
    invalid_run_ids_from_active_manifest,
    run_id_is_decision_eligible,
)
from app.autonomous.financial_integrity import stable_quote_hash, stable_quote_snapshot_id
from app.autonomous.sector_report import render_autonomous_sector_report
from app.watchlist.contract import WatchlistEntry
from app.web.readmodel import reader
from app.web.readmodel.run_detail import load_run_report
from app.web.readmodel.runs_index import UI_SCHEMA_SQL, list_indexed_runs, refresh_index


def _canonical_roots() -> dict[str, Path]:
    from app.config import get_config

    cfg = get_config()
    roots = {
        "runs": Path(cfg.runs_dir) / "autonomous_sector",
        "analyst_outputs": Path(cfg.analyst_outputs_dir),
        "scans": Path(cfg.outputs_dir) / "scans",
        "research": Path(cfg.research_dir),
        "digests": Path(cfg.outputs_dir) / "digests",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    return roots


def _canonical_family_roots() -> dict[str, Path]:
    roots = _canonical_roots()
    return {
        "autonomous_sector": roots["runs"],
        "analyst_output": roots["analyst_outputs"],
        "scan": roots["scans"],
        "research_output": roots["research"],
        "watchlist_report": roots["digests"],
    }


def _write_valid_run_pair(
    roots: dict[str, Path],
    *,
    run_id: str,
) -> tuple[Path, Path]:
    from tests.test_classic_postwrite_authorization import _valid_classic_artifact

    artifact = _valid_classic_artifact(run_id)
    run_dir = roots["runs"] / run_id
    run_dir.mkdir(parents=True)
    run_path = run_dir / "autonomous_sector_run.json"
    report_path = run_dir / "autonomous_sector_report.md"
    run_path.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(
        render_autonomous_sector_report(artifact),
        encoding="utf-8",
    )
    return run_path, report_path


def _valid_packet_payload() -> dict:
    payload = {
        "run_id": "autonomous_sector_synthetic_20260722_good",
        "sector": "synthetic",
        "as_of_date": "2026-07-22",
        "company_packets": [
            {
                "ticker": "AAA",
                "current_price": 100.0,
                "cap_stage_price": 100.0,
                "current_price_as_of_date": "2026-07-21",
                "cap_stage_price_as_of_date": "2026-07-21",
                "current_price_source": "fixture",
                "cap_stage_price_source": "fixture",
                "current_price_source_url": "https://example.test/AAA",
                "current_price_currency": "USD",
                "cap_stage_price_currency": "USD",
                "market_cap_unit": "USD_millions",
                "current_price_unit": "USD_per_share",
                "price_basis": "SPLIT_ADJUSTED",
                "shares_outstanding_mm": 2.0,
                "shares_unit": "shares_millions",
                "shares_basis": "SPLIT_ADJUSTED",
                "split_adjustment_factor": 1.0,
                "split_effective_date": None,
                "market_cap_mm": 200.0,
                "metric_traces": {
                    "fcf_yield": {
                        "metric": "fcf_yield",
                        "formula": "ttm_fcf_mm / market_cap_mm",
                        "inputs": {"ttm_fcf_mm": 20.0, "market_cap_mm": 200.0},
                        "output": 0.1,
                        "output_unit": "ratio",
                        "recomputed_output": 0.1,
                        "reconciles": True,
                    }
                },
                "valuation": {
                    "current_price": 100.0,
                    "ttm_fcf": 20.0,
                    "fcf_yield": 0.1,
                },
            }
        ],
        "candidate_selection": {"cap_classifications": {"AAA": {"shares_mm": 2.0}}},
    }
    packet = payload["company_packets"][0]
    packet["quote_snapshot_id"] = stable_quote_snapshot_id(packet)
    return payload


def _quote_snapshot(
    ticker: str,
    price: float,
    *,
    as_of_date: str = "2026-07-22",
    price_basis: str = "UNADJUSTED",
    raw_price: float | None = None,
    split_adjustment_factor: float = 1.0,
    split_effective_date: str | None = None,
) -> dict:
    snapshot = {
        "ticker": ticker,
        "price": price,
        "currency": "USD",
        "as_of_date": as_of_date,
        "source": "fixture",
        "source_url": f"https://example.test/{ticker}",
        "price_unit": "USD_per_share",
        "price_basis": price_basis,
        "raw_price": raw_price,
        "split_adjustment_factor": split_adjustment_factor,
        "split_effective_date": split_effective_date,
    }
    snapshot["quote_snapshot_id"] = stable_quote_hash(snapshot)
    return snapshot


def _bad_run_payload(analyst_path: Path) -> dict:
    payload = _valid_packet_payload()
    payload["run_id"] = "autonomous_sector_synthetic_20260722_bad"
    packet = payload["company_packets"][0]
    packet["cap_stage_price"] = 120.0
    packet["valuation"]["fcf_yield"] = 100_000.0
    payload["provider_usage"] = [{"provider": "fixture_llm", "status": "OK"}]
    analyst_context = {
        "ticker": "AAA",
        "as_of_date": "2026-07-22",
        "paths": {"analysis_report_json": str(analyst_path)},
        "valuation": {"price": 150.0},
    }
    payload["company_autonomy_runs"] = [
        {
            "ticker": "AAA",
            "artifact": {
                "provider_usage": [{"provider": "fixture_llm", "status": "OK"}],
                "evidence": [
                    {
                        "source_type": "analysis_report",
                        "source_label": "analyst_context",
                        "excerpt": json.dumps(analyst_context),
                    }
                ],
            },
        }
    ]
    return payload


def _write_tree(tmp_path: Path) -> dict[str, Path]:
    del tmp_path
    from app.config import get_config

    cfg = get_config()
    canonical = _canonical_roots()
    default_sentinel = canonical["runs"] / "audited_test_fixture.json"
    if default_sentinel.exists():
        default_sentinel.unlink()
    outputs = Path(cfg.outputs_dir)
    runs_root = Path(cfg.runs_dir)
    run_dir = canonical["runs"] / "autonomous_sector_synthetic_20260722_bad"
    analyst_dir = canonical["analyst_outputs"] / "AAA_2026-07-22"
    scans_root = canonical["scans"]
    digests_root = canonical["digests"]
    research_root = canonical["research"]
    for directory in (run_dir, analyst_dir, scans_root, digests_root, research_root):
        directory.mkdir(parents=True, exist_ok=True)

    analyst_json = analyst_dir / "analysis_report.json"
    analyst_json.write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "as_of_date": "2026-07-22",
                "valuation": {"price": 150.0},
            }
        ),
        encoding="utf-8",
    )
    (analyst_dir / "analysis_report.md").write_text("# AAA analyst report\n", encoding="utf-8")
    (analyst_dir / "analysis_evidence_bundle.json").write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "as_of_date": "2026-07-22",
                "valuation": {"current_price": 150.0},
            }
        ),
        encoding="utf-8",
    )

    run_json = run_dir / "autonomous_sector_run.json"
    run_json.write_text(json.dumps(_bad_run_payload(analyst_json)), encoding="utf-8")
    run_report = run_dir / "autonomous_sector_report.md"
    run_report.write_text("# Original historical report\n", encoding="utf-8")
    (scans_root / "synthetic_2026-07-22_artifacts.json").write_text(
        json.dumps({"sector": "synthetic", "triage": []}), encoding="utf-8"
    )
    (scans_root / "synthetic_2026-07-22.md").write_text("# Scan\n", encoding="utf-8")
    digest = digests_root / "digest_2026-07-22.md"
    digest.write_text("| Ticker | Status |\n|---|---|\n| AAA | ACTIVE |\n", encoding="utf-8")
    decision_state = {
        "ticker": "AAA",
        "source_run_id": "autonomous_sector_synthetic_20260722_bad",
    }
    (digests_root / "digest_2026-07-22.lineage.json").write_text(
        json.dumps(
            {
                "schema_version": "financial_integrity_digest_lineage_v2",
                "digest_path": str(digest.resolve()),
                "digest_sha256": hashlib.sha256(digest.read_bytes()).hexdigest(),
                "rows": [
                    {
                        "ticker": "AAA",
                        "source_run_id": "autonomous_sector_synthetic_20260722_bad",
                        "source_artifact_path": str(run_json.resolve()),
                        "source_artifact_sha256": hashlib.sha256(run_json.read_bytes()).hexdigest(),
                        "decision_state": decision_state,
                        "decision_state_sha256": hashlib.sha256(
                            json.dumps(
                                decision_state,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "outputs": outputs,
        "runs_root": runs_root,
        "run_json": run_json,
        "run_report": run_report,
        "analyst_root": outputs / "analyst_outputs",
        "analyst_json": analyst_json,
        "scans_root": scans_root,
        "research_root": research_root,
        "digests_root": digests_root,
        "digest": digest,
    }


def _audit_tree(tmp_path: Path):
    paths = _write_tree(tmp_path)
    source_hash = hashlib.sha256(paths["run_json"].read_bytes()).hexdigest()
    reports = audit_artifact_tree(
        runs_root=paths["runs_root"] / "autonomous_sector",
        analyst_outputs_root=paths["analyst_root"],
        scans_root=paths["scans_root"],
        research_outputs_root=paths["research_root"],
        watchlist_report_roots=[paths["digests_root"]],
        analysis_dir=paths["outputs"] / "analysis",
        generated_at=datetime(2026, 7, 22, 20, 0, tzinfo=timezone.utc),
    )
    assert hashlib.sha256(paths["run_json"].read_bytes()).hexdigest() == source_hash
    return paths, reports


def test_audit_detects_scale_quote_and_nested_analyst_violations_without_rewrites(tmp_path):
    paths, reports = _audit_tree(tmp_path)
    manifest = json.loads(reports.manifest_json.read_text(encoding="utf-8"))

    invariants = {item["invariant"] for item in manifest["violations"]}
    assert "FCF_YIELD_USD_VS_USD_MILLIONS" in invariants
    assert "QUOTE_SNAPSHOT_MISMATCH" in invariants
    assert "NESTED_ANALYST_QUOTE_MISMATCH" in invariants
    assert "ANALYST_QUOTE_PROVENANCE_MISSING" in invariants
    assert "DIGEST_SOURCE_LINEAGE_INVALID" in invariants
    assert "DIGEST_SOURCE_LINEAGE_MISSING" in invariants
    assert manifest["summary"]["artifacts_scanned"] == 9
    assert manifest["summary"]["tickers_scanned"] == 1
    assert manifest["summary"]["earliest_date"] == "2026-07-22"
    assert manifest["summary"]["latest_date"] == "2026-07-22"
    assert manifest["summary"]["source_artifacts_rewritten"] == 0

    million = next(
        item
        for item in manifest["violations"]
        if item["invariant"] == "FCF_YIELD_USD_VS_USD_MILLIONS"
    )
    assert million["source_values"]["expected_fcf_yield"] == 0.1
    assert million["source_values"]["observed_fcf_yield"] == 100_000.0
    assert million["source_values"]["observed_to_expected"] == 1_000_000.0
    assert million["llm_consumed"] is True
    assert million["repair_classification"] == REQUIRES_LLM_REREVIEW

    analyst = next(
        item
        for item in manifest["violations"]
        if item["invariant"] == "ANALYST_QUOTE_PROVENANCE_MISSING"
    )
    assert analyst["repair_classification"] == MISSING_PROVENANCE_UNREPAIRABLE
    records = {item["path"]: item for item in manifest["artifacts"]}
    assert records[str(paths["run_json"].resolve())]["integrity_status"] == INVALID
    assert records[str(paths["run_report"].resolve())]["integrity_status"] == INVALID
    assert records[str(paths["analyst_json"].resolve())]["integrity_status"] == INVALID
    assert records[str(paths["digest"].resolve())]["integrity_status"] == INVALID
    assert "Original historical report" in paths["run_report"].read_text(encoding="utf-8")
    assert reports.report_markdown.is_file()


def test_hash_bound_eligibility_and_direct_payload_gate(tmp_path):
    paths, reports = _audit_tree(tmp_path)
    assert financial_integrity_manifest_is_usable(reports.manifest_json) is True
    assert invalid_run_ids_from_active_manifest(reports.manifest_json) == {
        "autonomous_sector_synthetic_20260722_bad"
    }
    assert artifact_decision_eligibility(paths["run_json"], reports.manifest_json) == INVALID
    assert (
        run_id_is_decision_eligible(
            "autonomous_sector_synthetic_20260722_bad",
            reports.manifest_json,
        )
        is False
    )
    assert artifact_decision_eligibility(paths["run_report"], reports.manifest_json) == INVALID
    assert artifact_decision_eligibility(_valid_packet_payload()) == PASS
    assert artifact_decision_eligibility(_bad_run_payload(paths["analyst_json"])) == INVALID

    unknown = paths["outputs"] / "new_after_audit.json"
    unknown.write_text("{}", encoding="utf-8")
    assert artifact_decision_eligibility(unknown, reports.manifest_json) == UNAUDITED

    copied = paths["outputs"] / "copied_audited_bytes.json"
    copied.write_bytes(paths["run_report"].read_bytes())
    assert artifact_decision_eligibility(copied, reports.manifest_json) == UNAUDITED

    paths["run_json"].write_text("{}", encoding="utf-8")
    assert artifact_decision_eligibility(paths["run_json"], reports.manifest_json) == STALE_AUDIT


def test_same_size_replacement_with_restored_mtime_cannot_retain_pass(tmp_path):
    roots = _canonical_roots()
    sentinel = roots["scans"] / "sentinel.json"
    sentinel.write_bytes(b"{}\n")
    report = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst_outputs"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "analysis",
        generated_at=datetime(2026, 7, 22, 20, 0, tzinfo=timezone.utc),
    )

    assert artifact_decision_eligibility(sentinel, report.manifest_json) == PASS
    prior_stat = sentinel.stat()
    sentinel.write_bytes(b"[]\n")
    os.utime(
        sentinel,
        ns=(prior_stat.st_atime_ns, prior_stat.st_mtime_ns),
    )
    assert sentinel.stat().st_size == prior_stat.st_size
    assert artifact_decision_eligibility(sentinel, report.manifest_json) == STALE_AUDIT


def test_reader_suppresses_run_report_when_exact_sibling_source_changes(
    monkeypatch,
    tmp_path,
):
    roots = _canonical_roots()
    run_path, report_path = _write_valid_run_pair(
        roots,
        run_id="autonomous_sector_energy_20260724_reader_source",
    )
    report = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst_outputs"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "analysis",
        generated_at=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setenv(
        "VOE_FINANCIAL_INTEGRITY_MANIFEST",
        str(report.manifest_json),
    )
    from app.config import get_config

    get_config.cache_clear()

    initial = reader.render_artifact(str(report_path))
    assert initial["integrity_status"] == PASS
    assert initial["decision_eligible"] is True
    assert "energy" in initial["html"]

    prior_stat = run_path.stat()
    original = run_path.read_bytes()
    replacement = original.replace(b'"sector": "energy"', b'"sector": "metals"', 1)
    assert replacement != original
    assert len(replacement) == len(original)
    run_path.write_bytes(replacement)
    os.utime(
        run_path,
        ns=(prior_stat.st_atime_ns, prior_stat.st_mtime_ns),
    )

    assert authorized_artifact_bytes(run_path) == (STALE_AUDIT, None)
    assert authorized_artifact_bytes(report_path) == (STALE_AUDIT, None)
    rendered = reader.render_artifact(str(report_path))
    assert rendered["integrity_status"] == STALE_AUDIT
    assert rendered["decision_eligible"] is False
    assert "excluded from current decisions" in rendered["html"]
    assert "energy" not in rendered["html"]


def test_audited_run_report_requires_canonical_source_render(tmp_path):
    roots = _canonical_roots()
    _run_path, report_path = _write_valid_run_pair(
        roots,
        run_id="autonomous_sector_energy_20260724_noncanonical_report",
    )
    report_path.write_text("# Noncanonical decision report\n", encoding="utf-8")
    report = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst_outputs"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "analysis",
        generated_at=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert authorized_artifact_bytes(report_path, report.manifest_json) == (INVALID, None)


def test_digest_named_symlink_cannot_impersonate_audited_run_report(
    monkeypatch,
    tmp_path,
):
    roots = _canonical_roots()
    _run_path, report_path = _write_valid_run_pair(
        roots,
        run_id="autonomous_sector_energy_20260724_alias_target",
    )
    report = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst_outputs"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "analysis",
        generated_at=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setenv(
        "VOE_FINANCIAL_INTEGRITY_MANIFEST",
        str(report.manifest_json),
    )
    from app.config import get_config

    get_config.cache_clear()
    alias = roots["digests"] / "digest_2026-07-24.md"
    alias.symlink_to(report_path)

    assert authorized_artifact_bytes(alias) == (UNAUDITED, None)
    library = reader.library(None)
    assert all(
        item["path"] != str(alias) for family in library["families"] for item in family["items"]
    )
    with pytest.raises(reader.ArtifactRefused, match="symlink or alias"):
        reader.render_artifact(str(alias))


def test_audit_rejects_duplicate_raw_paths_resolving_to_one_artifact(tmp_path):
    roots = _canonical_roots()
    target = roots["scans"] / "canonical.json"
    target.write_text("{}\n", encoding="utf-8")
    alias = roots["digests"] / "digest_2026-07-24.md"
    alias.symlink_to(target)

    with pytest.raises(ValueError, match="duplicate artifact aliases"):
        audit_artifact_tree(
            runs_root=roots["runs"],
            analyst_outputs_root=roots["analyst_outputs"],
            scans_root=roots["scans"],
            research_outputs_root=roots["research"],
            watchlist_report_roots=[roots["digests"]],
            analysis_dir=tmp_path / "analysis",
            generated_at=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
        )


def test_audit_rejects_duplicate_hard_link_file_identities(tmp_path):
    roots = _canonical_roots()
    target = roots["scans"] / "canonical.json"
    target.write_text("{}\n", encoding="utf-8")
    alias = roots["digests"] / "digest_2026-07-24.md"
    os.link(target, alias)

    with pytest.raises(ValueError, match="duplicate artifact file identities"):
        audit_artifact_tree(
            runs_root=roots["runs"],
            analyst_outputs_root=roots["analyst_outputs"],
            scans_root=roots["scans"],
            research_outputs_root=roots["research"],
            watchlist_report_roots=[roots["digests"]],
            analysis_dir=tmp_path / "analysis",
            generated_at=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
        )


def test_plain_report_copy_cannot_impersonate_a_lineage_free_digest(tmp_path):
    roots = _canonical_roots()
    copied_report = roots["digests"] / "digest_2026-07-24.md"
    copied_report.write_text(
        "# Autonomous Sector Report\n\n"
        "The selected company appears undervalued on normalized earnings.\n",
        encoding="utf-8",
    )

    report = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst_outputs"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "analysis",
        generated_at=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )
    manifest = json.loads(report.manifest_json.read_text(encoding="utf-8"))
    record = next(
        item for item in manifest["artifacts"] if item["path"] == str(copied_report.resolve())
    )

    assert record["integrity_status"] == INVALID
    assert record["decision_eligible"] is False
    assert "DIGEST_SOURCE_LINEAGE_MISSING" in record["violation_invariants"]
    assert authorized_artifact_bytes(copied_report, report.manifest_json) == (INVALID, None)


def test_exact_canonical_blocked_digest_remains_nondecision_eligible(tmp_path):
    roots = _canonical_roots()
    blocked_digest = roots["digests"] / "digest_2026-07-24.md"
    blocked_digest.write_text(
        "# IVI Watchlist Daily Digest\n\n"
        "- Generated: 2026-07-24T12:00:00+00:00\n\n"
        "**BLOCKED: the canonical financial-integrity audit manifest is missing, "
        "malformed, or unreadable. Current decision rows are suppressed.**\n",
        encoding="utf-8",
    )

    report = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst_outputs"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "analysis",
        generated_at=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert authorized_artifact_bytes(blocked_digest, report.manifest_json) == (
        PASS,
        blocked_digest.read_bytes(),
    )


def test_existing_manifest_fails_closed_when_records_become_hard_links(tmp_path):
    roots = _canonical_roots()
    target = roots["scans"] / "canonical.json"
    target.write_text("{}\n", encoding="utf-8")
    alias = roots["digests"] / "digest_2026-07-24.md"
    alias.write_text("{}\n", encoding="utf-8")
    report = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst_outputs"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "analysis",
        generated_at=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )
    assert financial_integrity_manifest_is_usable(report.manifest_json) is True

    alias.unlink()
    os.link(target, alias)

    assert financial_integrity_manifest_is_usable(report.manifest_json) is False
    assert authorized_artifact_bytes(target, report.manifest_json) == (UNAUDITED, None)
    assert authorized_artifact_bytes(alias, report.manifest_json) == (UNAUDITED, None)


def test_same_size_manifest_corruption_with_restored_mtime_is_rehashed(tmp_path):
    roots = _canonical_roots()
    (roots["scans"] / "sentinel.json").write_bytes(b"{}\n")
    report = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst_outputs"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "analysis",
        generated_at=datetime(2026, 7, 22, 20, 0, tzinfo=timezone.utc),
    )
    assert financial_integrity_manifest_is_usable(report.manifest_json) is True

    prior_stat = report.manifest_json.stat()
    original = report.manifest_json.read_bytes()
    corrupted = original.replace(
        b"ivi_current_decision_artifacts_v1",
        b"ivi_current_decision_artifacts_v2",
        1,
    )
    assert len(corrupted) == len(original)
    report.manifest_json.write_bytes(corrupted)
    os.utime(
        report.manifest_json,
        ns=(prior_stat.st_atime_ns, prior_stat.st_mtime_ns),
    )
    assert financial_integrity_manifest_is_usable(report.manifest_json) is False


def test_audit_refuses_partial_or_duplicate_canonical_roots(tmp_path):
    roots = _canonical_roots()

    with pytest.raises(ValueError, match="exactly one root"):
        audit_artifact_tree(
            runs_root=roots["runs"],
            analyst_outputs_root=roots["analyst_outputs"],
            scans_root=roots["scans"],
            analysis_dir=tmp_path / "analysis",
        )
    with pytest.raises(ValueError, match="must be distinct"):
        audit_artifact_tree(
            runs_root=roots["runs"],
            analyst_outputs_root=roots["analyst_outputs"],
            scans_root=roots["scans"],
            research_outputs_root=roots["research"],
            watchlist_report_roots=[roots["scans"] / ".." / "scans"],
            analysis_dir=tmp_path / "analysis",
        )


def test_audit_refuses_existing_but_noncanonical_root(tmp_path):
    roots = _canonical_roots()
    arbitrary_runs = tmp_path / "existing_arbitrary_runs"
    arbitrary_runs.mkdir()

    with pytest.raises(ValueError, match="active runtime scope"):
        audit_artifact_tree(
            runs_root=arbitrary_runs,
            analyst_outputs_root=roots["analyst_outputs"],
            scans_root=roots["scans"],
            research_outputs_root=roots["research"],
            watchlist_report_roots=[roots["digests"]],
            analysis_dir=tmp_path / "analysis",
        )


def test_run_authorization_requires_audited_pass_and_current_bytes(tmp_path):
    payload = _valid_packet_payload()
    run_id = payload["run_id"]
    roots = _canonical_roots()
    run_dir = roots["runs"] / run_id
    run_dir.mkdir(parents=True)
    run_path = run_dir / "autonomous_sector_run.json"
    run_path.write_text(json.dumps(payload), encoding="utf-8")
    companion_path = run_dir / "financial_packet.json"
    companion_path.write_text(json.dumps(payload), encoding="utf-8")
    reports = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst_outputs"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "analysis",
        generated_at=datetime(2026, 7, 22, 20, 0, tzinfo=timezone.utc),
    )

    assert run_id_is_decision_eligible(run_id, reports.manifest_json) is True
    assert run_id_is_decision_eligible("unaudited_run", reports.manifest_json) is False

    companion_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert run_id_is_decision_eligible(run_id, reports.manifest_json) is False


def test_duplicate_primary_run_identity_is_invalid_and_binding_never_uses_path_order(
    monkeypatch,
    tmp_path,
):
    from app.autonomous import artifact_financial_audit as audit_module

    run_id = "autonomous_sector_energy_20260724_duplicate_primary"
    roots = _canonical_roots()
    canonical_path, canonical_report = _write_valid_run_pair(roots, run_id=run_id)
    duplicate_dir = roots["runs"] / "aaa_duplicate_copy"
    duplicate_dir.mkdir()
    duplicate_path = duplicate_dir / "autonomous_sector_run.json"
    duplicate_report = duplicate_dir / "autonomous_sector_report.md"
    duplicate_path.write_bytes(canonical_path.read_bytes())
    duplicate_report.write_bytes(canonical_report.read_bytes())

    reports = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst_outputs"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "analysis",
        generated_at=datetime(2026, 7, 24, 20, 0, tzinfo=timezone.utc),
    )
    manifest = json.loads(reports.manifest_json.read_text(encoding="utf-8"))
    primary_records = [
        record
        for record in manifest["artifacts"]
        if record.get("run_id") == run_id
        and Path(record["path"]).name == "autonomous_sector_run.json"
    ]
    assert len(primary_records) == 2
    assert {record["integrity_status"] for record in primary_records} == {INVALID}
    assert {
        violation["invariant"]
        for violation in manifest["violations"]
        if violation.get("run_id") == run_id
    } == {"AUTONOMOUS_RUN_PRIMARY_IDENTITY_INVALID", "SOURCE_ARTIFACT_INVALID"}
    assert financial_integrity_manifest_is_usable(reports.manifest_json) is True
    assert run_id_is_decision_eligible(run_id, reports.manifest_json) is False
    assert authorized_run_artifact_binding(run_id, reports.manifest_json) is None

    monkeypatch.setattr(audit_module, "run_id_is_decision_eligible", lambda *_args: True)
    assert authorized_run_artifact_binding(run_id, reports.manifest_json) is None


def test_mislocated_primary_run_identity_is_invalid(tmp_path):
    from tests.test_classic_postwrite_authorization import _valid_classic_artifact

    run_id = "autonomous_sector_energy_20260724_mislocated_primary"
    roots = _canonical_roots()
    artifact = _valid_classic_artifact(run_id)
    wrong_dir = roots["runs"] / "wrong_directory"
    wrong_dir.mkdir()
    run_path = wrong_dir / "autonomous_sector_run.json"
    report_path = wrong_dir / "autonomous_sector_report.md"
    run_path.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(
        render_autonomous_sector_report(artifact),
        encoding="utf-8",
    )

    reports = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst_outputs"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "analysis",
        generated_at=datetime(2026, 7, 24, 20, 1, tzinfo=timezone.utc),
    )
    manifest = json.loads(reports.manifest_json.read_text(encoding="utf-8"))
    run_record = next(
        record for record in manifest["artifacts"] if record["path"] == str(run_path.resolve())
    )
    assert run_record["integrity_status"] == INVALID
    assert "AUTONOMOUS_RUN_PRIMARY_IDENTITY_INVALID" in run_record["violation_invariants"]
    assert financial_integrity_manifest_is_usable(reports.manifest_json) is True
    assert run_id_is_decision_eligible(run_id, reports.manifest_json) is False
    assert authorized_run_artifact_binding(run_id, reports.manifest_json) is None


def test_run_authorization_rejects_declared_invalid_dependency(tmp_path):
    run_id = "autonomous_sector_synthetic_20260722_source"
    roots = _canonical_family_roots()
    source = roots["autonomous_sector"] / "source.json"
    dependent = roots["analyst_output"] / "dependent.json"
    source.write_text("{}\n", encoding="utf-8")
    dependent.write_text("{}\n", encoding="utf-8")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    dependent_sha = hashlib.sha256(dependent.read_bytes()).hexdigest()
    invariant = "DEPENDENT_FINANCIAL_CONTEXT_INVALID"
    manifest = {
        "schema_version": "financial_integrity_audit_v1",
        "audit_scope_id": "ivi_current_decision_artifacts_v1",
        "generated_at": "2026-07-22T20:00:00Z",
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
            for family, root in roots.items()
        ],
        "summary": {
            "artifacts_scanned": 2,
            "tickers_scanned": 0,
            "violations": 1,
            "violations_by_invariant": {invariant: 1},
            "affected_run_ids": 1,
            "affected_tickers": 0,
            "affected_run_id_values": [run_id],
            "affected_ticker_values": [],
            "earliest_date": None,
            "latest_date": None,
            "llm_consumed_violation_count": 0,
            "source_artifacts_rewritten": 0,
        },
        "invalid_run_ids": [run_id],
        "artifacts": [
            {
                "path": str(source.resolve()),
                "family": "autonomous_sector",
                "sha256": source_sha,
                "integrity_status": PASS,
                "decision_eligible": True,
                "run_id": run_id,
            },
            {
                "path": str(dependent.resolve()),
                "family": "analyst_output",
                "sha256": dependent_sha,
                "integrity_status": INVALID,
                "decision_eligible": False,
                "run_id": None,
            },
        ],
        "violations": [
            {
                "artifact_path": str(dependent.resolve()),
                "artifact_sha256": dependent_sha,
                "invariant": invariant,
                "run_id": run_id,
                "ticker": None,
                "llm_consumed": False,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert financial_integrity_manifest_is_usable(manifest_path) is True
    assert run_id_is_decision_eligible(run_id, manifest_path) is False


def test_packet_contract_gap_is_one_bounded_finding_and_algebra_dates_are_checked():
    legacy = _valid_packet_payload()
    packet = legacy["company_packets"][0]
    for field in (
        "market_cap_unit",
        "quote_snapshot_id",
        "current_price_unit",
        "price_basis",
        "shares_unit",
        "shares_basis",
        "split_adjustment_factor",
        "metric_traces",
    ):
        packet.pop(field)
    legacy_violations = audit_payload(legacy)
    missing = [
        item
        for item in legacy_violations
        if item["invariant"] == "FINANCIAL_CONTRACT_PROVENANCE_MISSING"
    ]
    assert len(missing) == 1
    assert missing[0]["source_values"]["missing_fields"] == [
        "market_cap_unit",
        "quote_snapshot_id",
        "current_price_unit",
        "price_basis",
        "shares_unit",
        "shares_basis",
        "split_adjustment_factor",
        "metric_traces",
    ]
    assert missing[0]["repair_classification"] == MISSING_PROVENANCE_UNREPAIRABLE
    assert missing[0]["llm_consumed"] is False

    legacy["provider_usage"] = [{"provider": "fixture_llm", "status": "OK"}]
    llm_consumed_missing = [
        item
        for item in audit_payload(legacy)
        if item["invariant"] == "FINANCIAL_CONTRACT_PROVENANCE_MISSING"
    ]
    assert len(llm_consumed_missing) == 1
    assert llm_consumed_missing[0]["llm_consumed"] is True
    assert llm_consumed_missing[0]["repair_classification"] == MISSING_PROVENANCE_UNREPAIRABLE

    contradictory = _valid_packet_payload()
    contradictory_packet = contradictory["company_packets"][0]
    contradictory_packet["market_cap_mm"] = 250.0
    contradictory_packet["current_price_as_of_date"] = "2026-07-23"
    contradictory_packet["valuation"]["unexpected_nonfinite"] = float("inf")
    invariants = {item["invariant"] for item in audit_payload(contradictory)}
    assert "MARKET_CAP_RECONCILIATION_MISMATCH" in invariants
    assert "FUTURE_DATED_FINANCIAL_INPUT" in invariants
    assert "NONFINITE_FINANCIAL_VALUE" in invariants


def test_standalone_analyst_and_scan_financial_contexts_require_provenance():
    analyst = {
        "ticker": "AAA",
        "as_of_date": "2026-07-22",
        "valuation": {"price": 100.0, "base_case_value": 125.0},
    }
    analyst_findings = audit_payload(analyst)
    assert [item["invariant"] for item in analyst_findings] == ["ANALYST_QUOTE_PROVENANCE_MISSING"]
    assert analyst_findings[0]["source_values"]["analyst_price"] == 100.0
    assert analyst_findings[0]["repair_classification"] == MISSING_PROVENANCE_UNREPAIRABLE
    assert analyst_findings[0]["llm_consumed"] is True

    analyst["valuation"].update(
        {
            "price_source": "fixture",
            "price_source_url": "https://example.test/AAA",
            "price_as_of_date": "2026-07-22",
            "price_currency": "USD",
            "price_unit": "USD_per_share",
            "price_basis": "UNADJUSTED",
            "split_adjustment_factor": 1.0,
            "split_effective_date": None,
        }
    )
    analyst["valuation"]["quote_snapshot_id"] = stable_quote_hash(
        ticker="AAA",
        price=100.0,
        as_of_date="2026-07-22",
        currency="USD",
        source="fixture",
        source_url="https://example.test/AAA",
        price_basis="UNADJUSTED",
        raw_price=None,
        split_adjustment_factor=1.0,
        split_effective_date=None,
    )
    assert audit_payload(analyst) == []

    scan = {
        "sector": "synthetic",
        "triage": {
            "full_prompt": "#1 AAA\nVALUATION:\n  current_price: 100.0000\n  dcf_base: 125.0000",
            "survivors": ["AAA"],
        },
        "deep_reviews": [{"ticker": "AAA", "memo": "candidate review"}],
    }
    scan_findings = audit_payload(scan)
    assert [item["invariant"] for item in scan_findings] == ["SCAN_FINANCIAL_PROVENANCE_MISSING"]
    assert scan_findings[0]["llm_consumed"] is True
    assert scan_findings[0]["repair_classification"] == MISSING_PROVENANCE_UNREPAIRABLE


def test_family_audit_rejects_nonfinite_values_despite_valid_quote_contracts():
    analyst = {
        "ticker": "AAA",
        "as_of_date": "2026-07-22",
        "valuation": {
            "price": 100.0,
            "base_case_value": float("inf"),
            "price_source": "fixture",
            "price_source_url": "https://example.test/AAA",
            "price_as_of_date": "2026-07-22",
            "price_currency": "USD",
            "price_unit": "USD_per_share",
            "price_basis": "UNADJUSTED",
            "split_adjustment_factor": 1.0,
            "split_effective_date": None,
        },
    }
    analyst["valuation"]["quote_snapshot_id"] = stable_quote_hash(
        ticker="AAA",
        price=100.0,
        as_of_date="2026-07-22",
        currency="USD",
        source="fixture",
        source_url="https://example.test/AAA",
        price_basis="UNADJUSTED",
        raw_price=None,
        split_adjustment_factor=1.0,
        split_effective_date=None,
    )
    assert [item["invariant"] for item in audit_payload(analyst)] == ["NONFINITE_FINANCIAL_VALUE"]

    scan = {
        "as_of_date": "2026-07-22",
        "ranked": [{"ticker": "AAA", "current_price": float("nan")}],
        "financial_integrity": {
            "status": PASS,
            "quote_snapshots": [_quote_snapshot("AAA", 100.0)],
        },
    }
    assert [item["invariant"] for item in audit_payload(scan)] == ["NONFINITE_FINANCIAL_VALUE"]


def test_scan_quote_contract_requires_exact_semantic_ticker_coverage():
    scan = {
        "as_of_date": "2026-07-22",
        "ranked": [
            {"ticker": "AAA", "current_price": 100.0},
            {"ticker": "BBB", "current_price": 50.0},
        ],
        "financial_integrity": {
            "status": PASS,
            "quote_snapshots": [
                _quote_snapshot("AAA", 100.0),
                _quote_snapshot("BBB", 50.0),
            ],
        },
    }
    assert audit_payload(scan) == []

    one_snapshot = json.loads(json.dumps(scan))
    one_snapshot["financial_integrity"]["quote_snapshots"].pop()
    assert [item["invariant"] for item in audit_payload(one_snapshot)] == [
        "SCAN_FINANCIAL_PROVENANCE_MISSING"
    ]

    wrong_hash = json.loads(json.dumps(scan))
    wrong_hash["financial_integrity"]["quote_snapshots"][0]["quote_snapshot_id"] = "a" * 64
    assert [item["invariant"] for item in audit_payload(wrong_hash)] == [
        "SCAN_FINANCIAL_PROVENANCE_MISSING"
    ]

    future_quote = json.loads(json.dumps(scan))
    future_snapshot = _quote_snapshot("AAA", 100.0, as_of_date="2026-07-23")
    future_quote["financial_integrity"]["quote_snapshots"][0] = future_snapshot
    assert [item["invariant"] for item in audit_payload(future_quote)] == [
        "SCAN_FINANCIAL_PROVENANCE_MISSING"
    ]

    bad_literals = json.loads(json.dumps(scan))
    bad_literals["financial_integrity"]["quote_snapshots"][0]["currency"] = "EUR"
    bad_literals["financial_integrity"]["quote_snapshots"][0]["price_unit"] = "dollars"
    bad_literals["financial_integrity"]["quote_snapshots"][0]["price_basis"] = "RAW"
    bad_literals["financial_integrity"]["quote_snapshots"][0]["quote_snapshot_id"] = (
        stable_quote_hash(bad_literals["financial_integrity"]["quote_snapshots"][0])
    )
    assert [item["invariant"] for item in audit_payload(bad_literals)] == [
        "SCAN_FINANCIAL_PROVENANCE_MISSING"
    ]

    adjusted_without_date = json.loads(json.dumps(scan))
    adjusted_without_date["financial_integrity"]["quote_snapshots"][0] = _quote_snapshot(
        "AAA",
        100.0,
        price_basis="SPLIT_ADJUSTED",
        raw_price=200.0,
        split_adjustment_factor=2.0,
        split_effective_date=None,
    )
    assert [item["invariant"] for item in audit_payload(adjusted_without_date)] == [
        "SCAN_FINANCIAL_PROVENANCE_MISSING"
    ]


def test_analyst_split_adjustment_requires_date_and_reconciliation():
    analyst = {
        "ticker": "AAA",
        "as_of_date": "2026-07-22",
        "valuation": {
            "price": 100.0,
            "base_case_value": 125.0,
            "price_source": "fixture",
            "price_source_url": "https://example.test/AAA",
            "price_as_of_date": "2026-07-22",
            "price_currency": "USD",
            "price_unit": "USD_per_share",
            "price_basis": "SPLIT_ADJUSTED",
            "raw_price": 200.0,
            "split_adjustment_factor": 2.0,
            "split_effective_date": None,
        },
    }
    analyst["valuation"]["quote_snapshot_id"] = stable_quote_hash(
        ticker="AAA",
        price=100.0,
        as_of_date="2026-07-22",
        currency="USD",
        source="fixture",
        source_url="https://example.test/AAA",
        price_basis="SPLIT_ADJUSTED",
        raw_price=200.0,
        split_adjustment_factor=2.0,
        split_effective_date=None,
    )
    finding = audit_payload(analyst)[0]
    assert finding["invariant"] == "ANALYST_QUOTE_PROVENANCE_MISSING"
    assert (
        "split_effective_date_required_for_adjusted_basis"
        in finding["source_values"]["missing_fields"]
    )


def test_invalid_scan_json_quarantines_only_its_matching_markdown(tmp_path):
    roots = _canonical_roots()
    scans = roots["scans"]
    source = scans / "synthetic_2026-07-22_artifacts.json"
    source.write_text(
        json.dumps(
            {
                "sector": "synthetic",
                "triage": {"full_prompt": "current_price: 100.0", "survivors": ["AAA"]},
            }
        ),
        encoding="utf-8",
    )
    matching = scans / "synthetic_2026-07-22.md"
    matching.write_text("# Current scan\n", encoding="utf-8")
    unrelated = scans / "other_2026-07-22.md"
    unrelated.write_text("# Unrelated scan\n", encoding="utf-8")
    reports = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst_outputs"],
        scans_root=scans,
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "analysis",
        generated_at=datetime(2026, 7, 22, 20, 0, tzinfo=timezone.utc),
    )
    manifest = json.loads(reports.manifest_json.read_text(encoding="utf-8"))
    records = {item["path"]: item for item in manifest["artifacts"]}

    assert records[str(source.resolve())]["integrity_status"] == INVALID
    assert records[str(matching.resolve())]["integrity_status"] == INVALID
    assert records[str(unrelated.resolve())]["integrity_status"] == PASS


def test_analyst_dependency_propagation_never_reverses_into_evidence(tmp_path):
    roots = _canonical_roots()
    analyst_dir = roots["analyst_outputs"] / "AAA_2026-07-22"
    analyst_dir.mkdir()
    evidence = analyst_dir / "analysis_evidence_bundle.json"
    evidence.write_text(
        json.dumps({"ticker": "AAA", "as_of_date": "2026-07-22", "filings": []}),
        encoding="utf-8",
    )
    report = analyst_dir / "analysis_report.json"
    report.write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "as_of_date": "2026-07-22",
                "valuation": {"price": 100.0, "base_case_value": 125.0},
            }
        ),
        encoding="utf-8",
    )
    rendering = analyst_dir / "analysis_report.md"
    rendering.write_text("# Analyst report\n", encoding="utf-8")

    reports = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst_outputs"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "analysis",
        generated_at=datetime(2026, 7, 22, 20, 0, tzinfo=timezone.utc),
    )
    manifest = json.loads(reports.manifest_json.read_text(encoding="utf-8"))
    records = {item["path"]: item for item in manifest["artifacts"]}

    assert records[str(evidence.resolve())]["integrity_status"] == PASS
    assert records[str(evidence.resolve())]["source_artifact_path"] is None
    assert records[str(report.resolve())]["integrity_status"] == INVALID
    assert records[str(rendering.resolve())]["integrity_status"] == INVALID
    assert records[str(rendering.resolve())]["source_artifact_path"] == str(report.resolve())


def test_invalid_analyst_evidence_invalidates_only_downstream_report(tmp_path):
    roots = _canonical_roots()
    analyst_dir = roots["analyst_outputs"] / "AAA_2026-07-22"
    analyst_dir.mkdir()
    evidence = analyst_dir / "analysis_evidence_bundle.json"
    evidence.write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "as_of_date": "2026-07-22",
                "valuation": {
                    "current_price": 100.0,
                    "dcf_base": float("inf"),
                },
            }
        ),
        encoding="utf-8",
    )
    report = analyst_dir / "analysis_report.json"
    valuation = {
        "price": 100.0,
        "base_case_value": 125.0,
        "price_source": "fixture",
        "price_source_url": "https://example.test/AAA",
        "price_as_of_date": "2026-07-22",
        "price_currency": "USD",
        "price_unit": "USD_per_share",
        "price_basis": "UNADJUSTED",
        "split_adjustment_factor": 1.0,
        "split_effective_date": None,
    }
    valuation["quote_snapshot_id"] = stable_quote_hash(
        ticker="AAA",
        price=100.0,
        as_of_date="2026-07-22",
        currency="USD",
        source="fixture",
        source_url="https://example.test/AAA",
        price_basis="UNADJUSTED",
        raw_price=None,
        split_adjustment_factor=1.0,
        split_effective_date=None,
    )
    report.write_text(
        json.dumps({"ticker": "AAA", "as_of_date": "2026-07-22", "valuation": valuation}),
        encoding="utf-8",
    )
    rendering = analyst_dir / "analysis_report.md"
    rendering.write_text("# Analyst report\n", encoding="utf-8")

    reports = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst_outputs"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "analysis",
        generated_at=datetime(2026, 7, 22, 20, 0, tzinfo=timezone.utc),
    )
    manifest = json.loads(reports.manifest_json.read_text(encoding="utf-8"))
    records = {item["path"]: item for item in manifest["artifacts"]}

    evidence_record = records[str(evidence.resolve())]
    assert evidence_record["integrity_status"] == INVALID
    assert evidence_record["repair_classification"] == MISSING_PROVENANCE_UNREPAIRABLE
    assert set(evidence_record["violation_invariants"]) == {
        "ANALYST_QUOTE_PROVENANCE_MISSING",
        "NONFINITE_FINANCIAL_VALUE",
    }
    assert records[str(report.resolve())]["source_artifact_path"] == str(evidence.resolve())
    assert records[str(rendering.resolve())]["source_artifact_path"] == str(report.resolve())


def test_research_outputs_are_audited_and_only_matching_rendering_inherits(tmp_path):
    roots = _canonical_roots()
    source = roots["research"] / "AAA_2026-07-22_20260722T200000000000Z.json"
    source.write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "as_of_date": "2026-07-22",
                "scorecard_price": 100.0,
                "scorecard_dcf": 125.0,
                "thesis": "Investment conclusion consumed the scorecard.",
            }
        ),
        encoding="utf-8",
    )
    matching = roots["research"] / f"{source.stem}_report.md"
    matching.write_text("# Matching research report\n", encoding="utf-8")
    unrelated = roots["research"] / "BBB_2026-07-22_report.md"
    unrelated.write_text("# Unrelated research report\n", encoding="utf-8")

    valid = roots["research"] / "CCC_2026-07-22.json"
    valid.write_text(
        json.dumps(
            {
                "ticker": "CCC",
                "as_of_date": "2026-07-22",
                "scorecard_price": 50.0,
                "scorecard_dcf": 60.0,
                "thesis": "Proven local decision context.",
                "financial_integrity": {
                    "status": PASS,
                    "quote_snapshots": [_quote_snapshot("CCC", 50.0)],
                },
            }
        ),
        encoding="utf-8",
    )

    reports = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst_outputs"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "analysis",
        generated_at=datetime(2026, 7, 22, 20, 0, tzinfo=timezone.utc),
    )
    manifest = json.loads(reports.manifest_json.read_text(encoding="utf-8"))
    records = {item["path"]: item for item in manifest["artifacts"]}

    assert financial_integrity_manifest_is_usable(reports.manifest_json) is True
    assert records[str(source.resolve())]["integrity_status"] == INVALID
    assert records[str(matching.resolve())]["integrity_status"] == INVALID
    assert records[str(unrelated.resolve())]["integrity_status"] == PASS
    assert records[str(valid.resolve())]["integrity_status"] == PASS
    assert any(
        item["invariant"] == "RESEARCH_FINANCIAL_PROVENANCE_MISSING"
        for item in manifest["violations"]
    )


def test_invalid_history_is_labeled_but_not_decision_eligible(tmp_path, monkeypatch):
    paths, reports = _audit_tree(tmp_path)
    monkeypatch.setenv("VOE_FINANCIAL_INTEGRITY_MANIFEST", str(reports.manifest_json))
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(UI_SCHEMA_SQL)

    refresh_index(conn, runs_dir=paths["runs_root"])
    history = list_indexed_runs(conn)
    assert len(history) == 1
    assert history[0]["integrity_status"] == INVALID
    assert history[0]["decision_eligible"] == 0
    assert list_indexed_runs(conn, decision_eligible=True) == []

    rendered = load_run_report(conn, history[0]["slug"])
    assert rendered is not None
    assert rendered["integrity_status"] == INVALID
    assert rendered["decision_eligible"] is False
    assert "excluded from current decisions" in rendered["html"]
    assert "original report body is suppressed" in rendered["html"]
    assert "Original historical report" not in rendered["html"]


def test_latest_analyst_output_excludes_manifest_invalid_artifact(tmp_path, monkeypatch):
    paths, reports = _audit_tree(tmp_path)
    monkeypatch.setenv("VOE_FINANCIAL_INTEGRITY_MANIFEST", str(reports.manifest_json))
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE analyst_outputs (
            ticker TEXT, as_of_date TEXT, output_type TEXT, output_path TEXT,
            output_hash TEXT, created_at TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO analyst_outputs VALUES (?, ?, ?, ?, ?, ?)",
        (
            "AAA",
            "2026-07-22",
            "analysis_report",
            str(paths["analyst_json"]),
            "ignored",
            "2026-07-22T20:00:00Z",
        ),
    )

    @contextmanager
    def fake_get_db():
        yield conn

    import app.analyst.output_store as output_store

    monkeypatch.setattr(output_store, "get_db", fake_get_db)
    assert output_store.latest_analysis_output_path("AAA", "analysis_report") is None


def test_watchlist_population_stops_before_any_database_write(monkeypatch):
    import app.watchlist.store as store

    monkeypatch.setattr(store, "artifact_decision_eligibility", lambda payload: INVALID)
    artifact = SimpleNamespace(
        pipeline_version="v1",
        execution_status="COMPLETED",
        status="COMPLETED",
        company_packets=[SimpleNamespace(ticker="AAA")],
        to_dict=lambda: {"company_packets": [{"ticker": "AAA"}]},
    )
    result = store.populate_from_sector_artifact(artifact)
    assert result.added_or_updated == 0
    assert result.skipped == 1
    assert result.skipped_reasons == {"AAA": "INVALID_FINANCIAL_INPUT:INVALID"}


def test_invalid_latest_watchlist_source_is_removed_after_selection_without_resurrection(
    tmp_path, monkeypatch
):
    import app.events.cheapness as cheapness
    import app.watchlist.store as store

    db_path = tmp_path / "watchlist.db"
    old_run = "autonomous_sector_synthetic_20260721_old"
    invalid_new_run = "autonomous_sector_synthetic_20260722_bad"

    def entry(run_id: str, added_at: str, target: float) -> WatchlistEntry:
        return WatchlistEntry(
            ticker="AAA",
            status="ACTIVE",
            source_run_id=run_id,
            added_at=added_at,
            conviction_grade="WATCHLIST_ONLY",
            confidence="LOW",
            conviction_source="company_autonomy",
            valuation_anchor_method="dcf",
            valuation_anchor_value=target * 1.5,
            buy_price_target=target,
            current_price_at_addition=100.0,
            source_sector="synthetic",
        )

    store.add_or_update(entry(old_run, "2026-07-21T12:00:00Z", 80.0), db_path=db_path)
    store.add_or_update(entry(invalid_new_run, "2026-07-22T12:00:00Z", 50.0), db_path=db_path)
    monkeypatch.setattr(
        store,
        "watchlist_row_is_decision_eligible",
        lambda row, *_args, **_kwargs: row["source_run_id"] == old_run and row["ticker"] == "AAA",
    )
    monkeypatch.setattr(cheapness, "latest_cheapness_by_ticker", lambda conn: {})

    assert store.get_latest("AAA", db_path=db_path) is None
    assert store.list_active(db_path=db_path) == []
    assert store.watchlist_queue(db_path=db_path) == []
    assert store.stats(db_path=db_path)["total_entries"] == 0

    conn = sqlite3.connect(db_path)
    try:
        assert (
            conn.execute("SELECT COUNT(*) FROM watchlist WHERE ticker = 'AAA'").fetchone()[0] == 2
        )
    finally:
        conn.close()
