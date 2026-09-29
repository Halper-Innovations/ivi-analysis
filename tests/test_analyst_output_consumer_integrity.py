from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

import app.analyst.output_store as analyst_output_store
from app.autonomous.artifact_financial_audit import (
    INVALID,
    PASS,
    authorized_artifact_bytes,
)
from app.ops.gating import _latest_analyst_output
from app.report.memo_builder import (
    _latest_output,
    _latest_output_path,
    _load_json_bytes,
)
from app.score.ranker import _latest_analyst_decision


def _artifact_record(path: Path, status: str) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "family": "analyst_output",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "run_id": None,
        "ticker": "AAA",
        "as_of_date": "2026-07-22",
        "integrity_status": status,
        "decision_eligible": status == PASS,
    }


def _write_integrity_manifest(path: Path, older_path: Path, latest_path: Path) -> None:
    older = _artifact_record(older_path, PASS)
    latest = _artifact_record(latest_path, INVALID)
    scope_parent = older_path.parent.parent.parent
    roots = {
        "autonomous_sector": scope_parent / "autonomous_sector_runs",
        "analyst_output": older_path.parent.parent,
        "scan": scope_parent / "scan_outputs",
        "research_output": scope_parent / "research_outputs",
        "watchlist_report": scope_parent / "watchlist_reports",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    violation = {
        "artifact_path": latest["path"],
        "artifact_sha256": latest["sha256"],
        "invariant": "TEST_FINANCIAL_INTEGRITY_INVALID",
        "run_id": None,
        "ticker": "AAA",
        "llm_consumed": False,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "financial_integrity_audit_v1",
                "audit_scope_id": "ivi_current_decision_artifacts_v1",
                "generated_at": "2026-07-23T00:00:00Z",
                "complete": True,
                "source_roots": [
                    {
                        "family": family,
                        "root_id": root.name,
                        "path": str(root.resolve()),
                    }
                    for family, root in roots.items()
                ],
                "summary": {
                    "artifacts_scanned": 2,
                    "tickers_scanned": 1,
                    "violations": 1,
                    "violations_by_invariant": {
                        "TEST_FINANCIAL_INTEGRITY_INVALID": 1,
                    },
                    "affected_run_ids": 0,
                    "affected_tickers": 1,
                    "affected_run_id_values": [],
                    "affected_ticker_values": ["AAA"],
                    "earliest_date": "2026-07-21",
                    "latest_date": "2026-07-22",
                    "llm_consumed_violation_count": 0,
                    "source_artifacts_rewritten": 0,
                },
                "invalid_run_ids": [],
                "artifacts": [older, latest],
                "violations": [violation],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _analyst_output_rows(
    tmp_path: Path,
    *,
    output_type: str,
    older_payload: dict[str, object],
    latest_payload: dict[str, object],
) -> tuple[sqlite3.Connection, Path, Path]:
    older_path = tmp_path / "analyst_outputs" / "AAA_2026-07-21" / f"{output_type}.json"
    latest_path = tmp_path / "analyst_outputs" / "AAA_2026-07-22" / f"{output_type}.json"
    older_path.parent.mkdir(parents=True)
    latest_path.parent.mkdir(parents=True)
    older_path.write_text(json.dumps(older_payload), encoding="utf-8")
    latest_path.write_text(json.dumps(latest_payload), encoding="utf-8")

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE analyst_outputs (
            ticker TEXT,
            as_of_date TEXT,
            output_type TEXT,
            output_path TEXT,
            output_hash TEXT,
            created_at TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO analyst_outputs VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                "AAA",
                "2026-07-21",
                output_type,
                str(older_path),
                "older-hash",
                "2026-07-21T12:00:00Z",
            ),
            (
                "AAA",
                "2026-07-22",
                output_type,
                str(latest_path),
                "latest-hash",
                "2026-07-22T12:00:00Z",
            ),
        ],
    )
    return conn, older_path, latest_path


def _activate_integrity_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    state: str,
    older_path: Path,
    latest_path: Path,
) -> None:
    monkeypatch.setattr(
        analyst_output_store,
        "authorized_artifact_bytes",
        authorized_artifact_bytes,
    )
    manifest_path = tmp_path / "financial_integrity_manifest.json"
    if state == "latest_invalid":
        _write_integrity_manifest(manifest_path, older_path, latest_path)
    monkeypatch.setenv("VOE_FINANCIAL_INTEGRITY_MANIFEST", str(manifest_path))


@pytest.mark.parametrize("integrity_state", ["latest_invalid", "missing_manifest"])
def test_ranker_does_not_consume_ineligible_latest_decision_or_resurrect_older(
    monkeypatch,
    tmp_path,
    integrity_state,
):
    conn, older_path, latest_path = _analyst_output_rows(
        tmp_path,
        output_type="decision",
        older_payload={"decision": {"classification": "BUY"}},
        latest_payload={"decision": {"classification": "AVOID"}},
    )
    _activate_integrity_state(
        monkeypatch,
        tmp_path,
        state=integrity_state,
        older_path=older_path,
        latest_path=latest_path,
    )

    try:
        assert _latest_analyst_decision(conn, "AAA") is None
    finally:
        conn.close()


@pytest.mark.parametrize("integrity_state", ["latest_invalid", "missing_manifest"])
def test_gating_does_not_consume_ineligible_latest_hypotheses_or_resurrect_older(
    monkeypatch,
    tmp_path,
    integrity_state,
):
    conn, older_path, latest_path = _analyst_output_rows(
        tmp_path,
        output_type="hypotheses",
        older_payload={"hypotheses": [{"claim": "older audited claim"}]},
        latest_payload={"hypotheses": [{"claim": "latest invalid claim"}]},
    )
    _activate_integrity_state(
        monkeypatch,
        tmp_path,
        state=integrity_state,
        older_path=older_path,
        latest_path=latest_path,
    )

    try:
        assert _latest_analyst_output(conn, "AAA", "2026-07-22", "hypotheses") is None
    finally:
        conn.close()


@pytest.mark.parametrize("integrity_state", ["latest_invalid", "missing_manifest"])
def test_memo_does_not_consume_ineligible_latest_output(
    monkeypatch,
    tmp_path,
    integrity_state,
):
    conn, older_path, latest_path = _analyst_output_rows(
        tmp_path,
        output_type="red_team",
        older_payload={"red_team": [{"claim": "older audited risk"}]},
        latest_payload={"red_team": [{"claim": "latest invalid risk"}]},
    )
    _activate_integrity_state(
        monkeypatch,
        tmp_path,
        state=integrity_state,
        older_path=older_path,
        latest_path=latest_path,
    )

    try:
        assert _latest_output_path(conn, "AAA", "red_team", "2026-07-22") is None
    finally:
        conn.close()


def _analysis_report_payload(*, verdict: str, summary: str) -> dict[str, object]:
    return {
        "analysis_id": f"A-{verdict}",
        "ticker": "AAA",
        "as_of_date": "2026-07-22",
        "generated_at": "2026-07-22T12:00:00Z",
        "verdict": verdict,
        "confidence_label": "MODERATE",
        "confidence_score": 55,
        "thesis_summary": summary,
        "valuation": {
            "price": 50.0,
            "base_case_value": 60.0,
            "bear_case_value": 35.0,
            "bull_case_value": 80.0,
            "margin_of_safety": 0.1667,
        },
    }


def test_latest_analysis_report_parses_authorized_bytes_not_swapped_path(monkeypatch, tmp_path):
    conn, _older_path, latest_path = _analyst_output_rows(
        tmp_path,
        output_type="analysis_report",
        older_payload=_analysis_report_payload(
            verdict="WATCH",
            summary="Older report.",
        ),
        latest_payload=_analysis_report_payload(
            verdict="WATCH",
            summary="Authorized latest report.",
        ),
    )
    authorized_bytes = latest_path.read_bytes()
    forged_payload = _analysis_report_payload(
        verdict="BUY",
        summary="Forged replacement report.",
    )

    @contextmanager
    def fake_get_db():
        yield conn

    def authorize_then_swap(path):
        assert Path(path) == latest_path
        latest_path.write_text(json.dumps(forged_payload), encoding="utf-8")
        return PASS, authorized_bytes

    monkeypatch.setattr(
        analyst_output_store,
        "financial_integrity_manifest_is_usable",
        lambda: True,
    )
    monkeypatch.setattr(analyst_output_store, "get_db", fake_get_db)
    monkeypatch.setattr(
        analyst_output_store,
        "authorized_artifact_bytes",
        authorize_then_swap,
    )

    report = analyst_output_store.latest_analysis_report("AAA")
    assert report is not None
    assert report.verdict == "WATCH"
    assert report.thesis_summary == "Authorized latest report."
    conn.close()


@pytest.mark.parametrize(
    ("output_type", "authorized_payload", "consume"),
    [
        (
            "decision",
            {"decision": {"classification": "WATCH"}},
            lambda conn: _latest_analyst_decision(conn, "AAA"),
        ),
        (
            "hypotheses",
            {"hypotheses": [{"claim": "Authorized hypothesis."}]},
            lambda conn: _latest_analyst_output(
                conn,
                "AAA",
                "2026-07-22",
                "hypotheses",
            ),
        ),
        (
            "red_team",
            {"red_team": [{"claim": "Authorized risk."}]},
            lambda conn: _load_json_bytes(
                (_latest_output(conn, "AAA", "red_team", "2026-07-22") or (None, None))[1]
            ),
        ),
    ],
)
def test_adjacent_analyst_consumers_use_authorized_bytes(
    monkeypatch,
    tmp_path,
    output_type,
    authorized_payload,
    consume,
):
    conn, _older_path, latest_path = _analyst_output_rows(
        tmp_path,
        output_type=output_type,
        older_payload=authorized_payload,
        latest_payload=authorized_payload,
    )
    authorized_bytes = latest_path.read_bytes()

    def authorize_then_remove(path):
        assert Path(path) == latest_path
        latest_path.unlink()
        return PASS, authorized_bytes

    monkeypatch.setattr(
        analyst_output_store,
        "financial_integrity_manifest_is_usable",
        lambda: True,
    )
    monkeypatch.setattr(
        analyst_output_store,
        "authorized_artifact_bytes",
        authorize_then_remove,
    )
    assert consume(conn) == (
        authorized_payload["decision"] if output_type == "decision" else authorized_payload
    )
    conn.close()
