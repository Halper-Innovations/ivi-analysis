from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.autonomous.artifact_financial_audit import (
    AUDIT_SCHEMA_VERSION,
    AUDIT_SCOPE_ID,
    CANONICAL_AUDIT_ROOT_IDS,
)
from app.calibration.calibration_report import (
    build_calibration_report,
    latest_grade_status_report,
)
from app.calibration.return_resolver import resolve_open_outcomes
from app.db import init_db, utc_now_iso
from app.outcomes.lineage import (
    authorized_emitted_decision_binding,
    bind_outcome_row,
    outcome_row_is_decision_eligible,
)
from app.outcomes.store import add_outcome, close_outcome
from app.valuation.lineage import (
    bind_authorized_valuation_rows,
    latest_decision_eligible_valuation_row,
    latest_decision_eligible_valuation_rows,
    valuation_integrity_fingerprint,
    valuation_row_is_decision_eligible,
    valuation_source_record,
)
from app.watchlist.schema import ensure_watchlist_schema
from app.web.main import app
from app.web.readmodel import company, company_depth, outcomes

client = TestClient(app)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _init(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    ensure_watchlist_schema(cfg.db_path)
    return cfg


def _canonical_roots(tmp_path: Path, *, autonomous_root: Path) -> dict[str, Path]:
    del tmp_path
    from app.config import get_config

    cfg = get_config()
    expected_autonomous_root = (Path(cfg.runs_dir) / "autonomous_sector").resolve()
    supplied_autonomous_root = autonomous_root.resolve()
    if supplied_autonomous_root not in {
        Path(cfg.runs_dir).resolve(),
        expected_autonomous_root,
    }:
        raise AssertionError("test manifest must use the configured runtime roots")
    roots = {
        "autonomous_sector": expected_autonomous_root,
        "analyst_output": Path(cfg.analyst_outputs_dir).resolve(),
        "scan": (Path(cfg.outputs_dir) / "scans").resolve(),
        "research_output": Path(cfg.research_dir).resolve(),
        "watchlist_report": (Path(cfg.outputs_dir) / "digests").resolve(),
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    return roots


def _install_manifest(
    monkeypatch,
    tmp_path: Path,
    *,
    roots: dict[str, Path],
    artifacts: list[tuple[str, Path, str]],
) -> Path:
    records = [
        {
            "path": str(path.resolve()),
            "family": family,
            "sha256": _sha256(path),
            "integrity_status": "PASS",
            "decision_eligible": True,
            "run_id": run_id,
        }
        for family, path, run_id in artifacts
    ]
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_scope_id": AUDIT_SCOPE_ID,
        "generated_at": "2026-07-23T20:00:00Z",
        "complete": True,
        "source_roots": [
            {
                "family": family,
                "root_id": CANONICAL_AUDIT_ROOT_IDS[family],
                "path": str(root.resolve()),
            }
            for family, root in roots.items()
        ],
        "summary": {
            "artifacts_scanned": len(records),
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
        "artifacts": records,
        "violations": [],
    }
    path = tmp_path / "financial_integrity_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    monkeypatch.setenv("VOE_FINANCIAL_INTEGRITY_MANIFEST", str(path))
    return path


def _source_artifact(
    root: Path,
    run_id: str,
    *tickers: str,
    emit_decisions: bool = True,
    issuer_ciks: dict[str, str] | None = None,
    valuation_source_records: list[dict] | None = None,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "autonomous_sector_run.json"
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "company_packets": [
                    {
                        "ticker": ticker,
                        "financial_integrity_status": "PASS",
                        **(
                            {"issuer_cik": issuer_ciks[ticker]}
                            if issuer_ciks and ticker in issuer_ciks
                            else {}
                        ),
                    }
                    for ticker in tickers
                ],
                "relative_ranking": (
                    [
                        {
                            "ticker": ticker,
                            "company_autonomy_verdict": "ACTIONABLE",
                        }
                        for ticker in tickers
                    ]
                    if emit_decisions
                    else []
                ),
                **(
                    {"valuation_source_records": valuation_source_records}
                    if valuation_source_records is not None
                    else {}
                ),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _insert_valuation(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    as_of_date: str,
    method: str,
    value: float,
    source_path: Path | None,
    source_run_id: str | None,
    outputs: dict | None = None,
) -> int:
    if outputs is None:
        outputs = (
            {
                "status": "OK",
                "base": value,
                "low": value - 10.0,
                "high": value + 10.0,
            }
            if method == "dcf"
            else {"status": "OK", "value_per_share": value}
        )
    conn.execute(
        """
        INSERT INTO valuations(
            ticker, as_of_date, method, inputs_json, outputs_json, warnings_json,
            created_at, valuation_writer_version, quality_gate_verdict,
            confidence_class, gate_reason_codes, valuation_headwinds,
            valuation_supports, source_run_id, source_artifact_path,
            source_artifact_sha256, financial_integrity_fingerprint
        ) VALUES (?, ?, ?, '{}', ?, '[]', ?, 'test', 'PROCEED', 'HIGH',
                  '[]', '[]', '[]', ?, ?, ?, NULL)
        """,
        (
            ticker,
            as_of_date,
            method,
            json.dumps(outputs, sort_keys=True),
            f"{as_of_date}T12:00:00+00:00",
            source_run_id,
            str(source_path.resolve()) if source_path is not None else None,
            _sha256(source_path) if source_path is not None else None,
        ),
    )
    row_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM valuations WHERE id = ?", (row_id,)).fetchone()
    if source_path is not None:
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(source_payload.get("company_packets"), list):
            source_record = valuation_source_record(row)
            assert source_record is not None
            source_records = list(source_payload.get("valuation_source_records") or [])
            if source_record not in source_records:
                source_records.append(source_record)
                source_records.sort(
                    key=lambda record: (
                        str(record["row"]["ticker"]),
                        str(record["row"]["as_of_date"]),
                        str(record["row"]["method"]),
                        str(record["row"]["created_at"]),
                    )
                )
                source_payload["valuation_source_records"] = source_records
                source_path.write_text(
                    json.dumps(source_payload, sort_keys=True),
                    encoding="utf-8",
                )
                current_sha256 = _sha256(source_path)
                conn.execute(
                    """
                    UPDATE valuations
                    SET source_artifact_sha256 = ?
                    WHERE source_artifact_path = ?
                    """,
                    (current_sha256, str(source_path.resolve())),
                )
                bound_rows = conn.execute(
                    "SELECT * FROM valuations WHERE source_artifact_path = ?",
                    (str(source_path.resolve()),),
                ).fetchall()
                for bound_row in bound_rows:
                    conn.execute(
                        """
                        UPDATE valuations
                        SET financial_integrity_fingerprint = ?
                        WHERE id = ?
                        """,
                        (valuation_integrity_fingerprint(bound_row), int(bound_row["id"])),
                    )
                manifest_text = os.getenv("VOE_FINANCIAL_INTEGRITY_MANIFEST")
                if manifest_text:
                    manifest_path = Path(manifest_text)
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    for artifact_record in manifest.get("artifacts", []):
                        if artifact_record.get("path") == str(source_path.resolve()):
                            artifact_record["sha256"] = current_sha256
                    manifest_path.write_text(
                        json.dumps(manifest, sort_keys=True),
                        encoding="utf-8",
                    )
                row = conn.execute(
                    "SELECT * FROM valuations WHERE id = ?",
                    (row_id,),
                ).fetchone()
    fingerprint = valuation_integrity_fingerprint(row)
    conn.execute(
        "UPDATE valuations SET financial_integrity_fingerprint = ? WHERE id = ?",
        (fingerprint, row_id),
    )
    return row_id


def _insert_outcome(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    run_id: str,
    status: str = "CLOSED",
    excess: float | None = 5.0,
    as_of_date: str = "2026-01-01",
    grade: str = "ACTIONABLE",
    decision: str = "BUY",
    discovery_run_id: str | None = None,
    bind: bool = True,
    updated_at: str | None = None,
) -> int:
    conn.row_factory = sqlite3.Row
    now = updated_at or utc_now_iso()
    conn.execute(
        """
        INSERT INTO ticker_outcomes(
            ticker, as_of_date, run_id, discovery_run_id, decision, conviction, horizon_days,
            thesis_tags_json, outcome_status, realized_return_pct,
            excess_return_pct, entry_price, entry_date, benchmark_symbol,
            grade, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, 4, 30, '[]', ?, 9.0, ?,
                  100.0, ?, 'SPY', ?,
                  'DEPLOY_READY', ?, ?)
        """,
        (
            ticker,
            as_of_date,
            run_id,
            discovery_run_id,
            decision,
            status,
            excess,
            as_of_date,
            grade,
            now,
            now,
        ),
    )
    outcome_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    if bind:
        bind_outcome_row(conn, outcome_id)
    return outcome_id


def test_company_selects_newest_before_exact_source_authorization(monkeypatch, tmp_path):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    valid_path = _source_artifact(roots["autonomous_sector"], "run_valid", "AAA")
    unlisted_path = _source_artifact(roots["autonomous_sector"], "run_unlisted", "AAA")
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", valid_path, "run_valid")],
    )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    _insert_valuation(
        conn,
        ticker="AAA",
        as_of_date="2026-06-01",
        method="dcf",
        value=80.0,
        source_path=valid_path,
        source_run_id="run_valid",
    )
    _insert_valuation(
        conn,
        ticker="AAA",
        as_of_date="2026-07-01",
        method="dcf",
        value=120.0,
        source_path=unlisted_path,
        source_run_id="run_unlisted",
    )
    conn.commit()

    assert company.latest_valuations(conn, "AAA") == []
    assert company.valuation_evolution(conn, "AAA") == []

    conn.execute("DELETE FROM valuations WHERE as_of_date = '2026-07-01'")
    conn.commit()
    cards = company.latest_valuations(conn, "AAA")
    assert cards[0]["fair_value"] == {
        "kind": "band",
        "low": 70.0,
        "base": 80.0,
        "high": 90.0,
    }
    original_stat = valid_path.stat()
    original = valid_path.read_text(encoding="utf-8")
    mutated = original.replace('"AAA"', '"BBB"')
    assert len(mutated) == len(original)
    valid_path.write_text(mutated, encoding="utf-8")
    os.utime(
        valid_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert company.latest_valuations(conn, "AAA") == []
    conn.close()


def test_postwrite_binder_authorizes_only_pending_rows_for_exact_run(monkeypatch, tmp_path):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    source_record = valuation_source_record(
        {
            "ticker": "AAA",
            "as_of_date": "2026-07-01",
            "method": "dcf",
            "inputs_json": "{}",
            "outputs_json": json.dumps(
                {"status": "OK", "base": 80.0, "low": 70.0, "high": 90.0},
                sort_keys=True,
            ),
            "warnings_json": "[]",
            "created_at": "2026-07-01T12:00:00+00:00",
            "valuation_writer_version": "test",
            "quality_gate_verdict": "PROCEED",
            "confidence_class": "HIGH",
            "gate_reason_codes": "[]",
            "valuation_headwinds": "[]",
            "valuation_supports": "[]",
            "source_run_id": "run_bound",
        }
    )
    assert source_record is not None
    source_path = _source_artifact(
        roots["autonomous_sector"],
        "run_bound",
        "AAA",
        valuation_source_records=[source_record],
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", source_path, "run_bound")],
    )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    _insert_valuation(
        conn,
        ticker="AAA",
        as_of_date="2026-07-01",
        method="dcf",
        value=80.0,
        source_path=None,
        source_run_id="run_bound",
    )
    _insert_valuation(
        conn,
        ticker="AAA",
        as_of_date="2026-07-01",
        method="epv",
        value=70.0,
        source_path=None,
        source_run_id="different_run",
    )
    conn.commit()
    conn.close()

    assert (
        bind_authorized_valuation_rows(
            run_id="run_bound",
            tickers={"AAA"},
            as_of_date="2026-07-01",
            cfg=cfg,
        )
        == 1
    )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    dcf = conn.execute(
        "SELECT * FROM valuations WHERE ticker = 'AAA' AND method = 'dcf'"
    ).fetchone()
    epv = conn.execute(
        "SELECT * FROM valuations WHERE ticker = 'AAA' AND method = 'epv'"
    ).fetchone()
    history = conn.execute(
        """
        SELECT *
        FROM valuations_history
        WHERE ticker = 'AAA' AND method = 'dcf'
        """
    ).fetchall()
    assert valuation_row_is_decision_eligible(dcf) is True
    assert epv["source_artifact_path"] is None
    assert valuation_row_is_decision_eligible(epv) is False
    assert len(history) == 1
    assert history[0]["source_run_id"] == "run_bound"
    assert history[0]["source_artifact_path"] is None
    conn.close()


def test_exact_audited_research_artifact_authorizes_its_own_db_payload(monkeypatch, tmp_path):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    research_payload = {
        "run_id": "research_AAA_exact",
        "ticker": "AAA",
        "as_of_date": "2026-07-01",
        "status": "OK",
        "thesis": {"decision": "WATCH"},
        "warnings": [],
    }
    research_path = roots["research_output"] / "research_AAA_exact.json"
    research_path.write_text(
        json.dumps(research_payload, sort_keys=True),
        encoding="utf-8",
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("research_output", research_path, "research_AAA_exact")],
    )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        INSERT INTO valuations(
            ticker, as_of_date, method, inputs_json, outputs_json, warnings_json,
            created_at, valuation_writer_version, source_run_id,
            source_artifact_path, source_artifact_sha256
        ) VALUES (
            'AAA', '2026-07-01', 'deep_research', '{}', ?, '[]',
            '2026-07-01T12:00:00Z', 'deep_research_v1', ?, ?, ?
        )
        """,
        (
            json.dumps(research_payload, sort_keys=True),
            research_payload["run_id"],
            str(research_path.resolve()),
            _sha256(research_path),
        ),
    )
    row_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    row = conn.execute("SELECT * FROM valuations WHERE id = ?", (row_id,)).fetchone()
    conn.execute(
        """
        UPDATE valuations
        SET financial_integrity_fingerprint = ?
        WHERE id = ?
        """,
        (valuation_integrity_fingerprint(row), row_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM valuations WHERE id = ?", (row_id,)).fetchone()
    assert valuation_row_is_decision_eligible(row) is True

    conn.execute(
        'UPDATE valuations SET outputs_json = \'{"status":"OK"}\' WHERE id = ?',
        (row_id,),
    )
    conn.commit()
    tampered = conn.execute("SELECT * FROM valuations WHERE id = ?", (row_id,)).fetchone()
    assert valuation_row_is_decision_eligible(tampered) is False
    conn.close()


def test_company_research_rejects_fabricated_newest_row_bound_to_unrelated_pass(
    monkeypatch, tmp_path
):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    report_path = roots["research_output"] / "AAA_report.md"
    report_path.write_text("# AAA research\n", encoding="utf-8")
    exact_payload = {
        "run_id": "research_AAA_exact",
        "ticker": "AAA",
        "as_of_date": "2026-07-01",
        "status": "OK",
        "report_path": str(report_path),
        "conviction_class": "MODERATE",
        "conviction_score": 55,
        "thesis": {"original_dcf": 80.0},
    }
    exact_path = roots["research_output"] / "research_AAA_exact.json"
    unrelated_path = _source_artifact(
        roots["autonomous_sector"],
        "autonomous_sector_unrelated",
        "AAA",
    )
    exact_path.write_text(json.dumps(exact_payload, sort_keys=True), encoding="utf-8")
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[
            ("research_output", exact_path, exact_payload["run_id"]),
            ("autonomous_sector", unrelated_path, "autonomous_sector_unrelated"),
        ],
    )
    monkeypatch.setattr(
        company_depth,
        "artifact_decision_eligibility",
        lambda _path: "PASS",
    )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row

    def insert_research_row(
        payload: dict[str, object],
        *,
        as_of_date: str,
        source_path: Path,
        source_run_id: str,
    ) -> int:
        conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at, valuation_writer_version,
                quality_gate_verdict, confidence_class, gate_reason_codes,
                valuation_headwinds, valuation_supports, source_run_id,
                source_artifact_path, source_artifact_sha256,
                financial_integrity_fingerprint
            ) VALUES (
                'AAA', ?, 'deep_research', '{}', ?, '[]', ?,
                'deep_research_v1', 'PROCEED', 'MODERATE', '[]', '[]', '[]',
                ?, ?, ?, NULL
            )
            """,
            (
                as_of_date,
                json.dumps(payload, sort_keys=True),
                f"{as_of_date}T12:00:00Z",
                source_run_id,
                str(source_path.resolve()),
                _sha256(source_path),
            ),
        )
        row_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        row = conn.execute(
            "SELECT * FROM valuations WHERE id = ?",
            (row_id,),
        ).fetchone()
        conn.execute(
            """
            UPDATE valuations
            SET financial_integrity_fingerprint = ?
            WHERE id = ?
            """,
            (valuation_integrity_fingerprint(row), row_id),
        )
        return row_id

    exact_row_id = insert_research_row(
        exact_payload,
        as_of_date="2026-07-01",
        source_path=exact_path,
        source_run_id=exact_payload["run_id"],
    )
    fabricated_payload = {
        "run_id": "autonomous_sector_unrelated",
        "ticker": "AAA",
        "as_of_date": "2026-07-02",
        "status": "OK",
        "report_path": str(report_path),
        "conviction_class": "HIGH",
        "conviction_score": 99,
        "thesis": {"original_dcf": 999.0},
    }
    fabricated_row_id = insert_research_row(
        fabricated_payload,
        as_of_date="2026-07-02",
        source_path=unrelated_path,
        source_run_id="autonomous_sector_unrelated",
    )
    conn.commit()

    blocked = company_depth.research(conn, "AAA")
    assert blocked == {
        "available": False,
        "reason": "FINANCIAL_INTEGRITY_REPORT_BLOCKED",
    }

    conn.execute("DELETE FROM valuations WHERE id = ?", (fabricated_row_id,))
    conn.commit()
    authorized = company_depth.research(conn, "AAA")
    assert authorized["available"] is True
    assert authorized["as_of_date"] == "2026-07-01"
    assert authorized["conviction_score"] == 55
    assert authorized["thesis"]["original_dcf"] == 80.0
    assert exact_row_id > 0
    conn.close()


def test_search_does_not_promote_unaudited_valuation(monkeypatch, tmp_path):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    valid_path = _source_artifact(roots["autonomous_sector"], "run_valid", "GOOD")
    unlisted_path = _source_artifact(roots["autonomous_sector"], "run_unlisted", "BAD")
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", valid_path, "run_valid")],
    )
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    for cik, ticker in (("0001", "GOOD"), ("0002", "BAD")):
        conn.execute(
            """
            INSERT INTO sec_registrants(
                cik, primary_ticker, all_tickers, name, exchange_scope,
                operating_status, first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, 'US_LISTED', 'OPERATING', 'now', 'now')
            """,
            (cik, ticker, ticker, f"{ticker} COMPANY"),
        )
    _insert_valuation(
        conn,
        ticker="GOOD",
        as_of_date="2026-07-01",
        method="dcf",
        value=80.0,
        source_path=valid_path,
        source_run_id="run_valid",
    )
    _insert_valuation(
        conn,
        ticker="BAD",
        as_of_date="2026-07-01",
        method="dcf",
        value=90.0,
        source_path=unlisted_path,
        source_run_id="run_unlisted",
    )
    conn.commit()
    conn.close()

    good = client.get("/api/search", params={"q": "GOOD"}).json()["results"][0]
    bad = client.get("/api/search", params={"q": "BAD"}).json()["results"][0]
    assert good["covered"] is True
    assert bad["covered"] is False


class _PriceProvider:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def get_price_asof(self, ticker: str, as_of_date: str):
        self.calls.append((ticker, as_of_date))
        return SimpleNamespace(price=110.0, as_of_date=as_of_date, source="fixture")


def test_invalid_outcomes_never_resolve_or_enter_aggregates(monkeypatch, tmp_path):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    valid_path = _source_artifact(roots["autonomous_sector"], "run_valid", "GOOD")
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", valid_path, "run_valid")],
    )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    _insert_outcome(conn, ticker="GOOD", run_id="run_valid", status="OPEN", excess=None)
    _insert_outcome(conn, ticker="BAD", run_id="run_unlisted", status="OPEN", excess=None)
    _insert_outcome(conn, ticker="CLOSED_GOOD", run_id="run_valid")
    _insert_outcome(conn, ticker="CLOSED_BAD", run_id="run_unlisted", excess=90.0)
    conn.commit()
    conn.close()

    provider = _PriceProvider()
    summary = resolve_open_outcomes(
        "2026-03-01",
        provider=provider,
        db_path=cfg.db_path,
    )
    assert summary.eligible == 1
    assert summary.closed == 1
    assert summary.skipped_unauthorized == 1
    assert [ticker for ticker, _ in provider.calls if ticker in {"GOOD", "BAD"}] == ["GOOD"]

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    result = outcomes.realized_returns(conn)
    assert result["overall"] == {
        "n": 1,
        "avg_excess": 10.0,
        "median_excess": 10.0,
        "hit_rate": 1.0,
        "avg_realized": 10.0,
    }
    invalid_status = conn.execute(
        "SELECT outcome_status FROM ticker_outcomes WHERE ticker = 'BAD'"
    ).fetchone()[0]
    assert invalid_status == "OPEN"
    conn.close()


def test_packet_membership_without_emitted_decision_cannot_authorize_outcome(
    monkeypatch,
    tmp_path,
):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    source_path = _source_artifact(
        roots["autonomous_sector"],
        "run_packet_only",
        "OPEN_ONLY",
        "CLOSED_ONLY",
        emit_decisions=False,
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", source_path, "run_packet_only")],
    )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    open_id = _insert_outcome(
        conn,
        ticker="OPEN_ONLY",
        run_id="run_packet_only",
        status="OPEN",
        excess=None,
    )
    closed_id = _insert_outcome(
        conn,
        ticker="CLOSED_ONLY",
        run_id="run_packet_only",
        status="CLOSED",
        excess=999.0,
    )
    conn.commit()
    for outcome_id in (open_id, closed_id):
        row = conn.execute(
            "SELECT * FROM ticker_outcomes WHERE id = ?",
            (outcome_id,),
        ).fetchone()
        assert row["source_decision_fingerprint"] is None
        assert row["financial_integrity_fingerprint"] is None
        assert outcome_row_is_decision_eligible(row) is False
    conn.close()

    provider = _PriceProvider()
    summary = resolve_open_outcomes(
        "2026-03-01",
        provider=provider,
        db_path=cfg.db_path,
    )
    assert summary.eligible == 0
    assert summary.closed == 0
    assert summary.skipped_unauthorized == 1
    assert provider.calls == []
    with pytest.raises(ValueError, match="exact authorized source and row lineage"):
        close_outcome(
            ticker="OPEN_ONLY",
            run_id="run_packet_only",
            realized_return_pct=999.0,
            close_date="2026-03-01",
        )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    assert (
        conn.execute(
            "SELECT outcome_status FROM ticker_outcomes WHERE id = ?",
            (open_id,),
        ).fetchone()[0]
        == "OPEN"
    )
    assert outcomes.realized_returns(conn)["overall"]["n"] == 0
    assert company_depth.decisions(conn, "CLOSED_ONLY")["outcomes"] == []
    conn.close()
    report = build_calibration_report("2026-07-23", db_path=cfg.db_path)
    assert report["source_run_ids"] == []
    assert report["source_outcomes"] == []
    assert report["overall"]["n"] == 0


def test_same_ticker_outcome_mutation_breaks_row_and_report_authorization(
    monkeypatch,
    tmp_path,
):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    source_path = _source_artifact(
        roots["autonomous_sector"],
        "run_exact_outcome",
        "GOOD",
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", source_path, "run_exact_outcome")],
    )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    outcome_id = _insert_outcome(
        conn,
        ticker="GOOD",
        run_id="run_exact_outcome",
        status="CLOSED",
        excess=5.0,
    )
    conn.commit()
    authorized = conn.execute(
        "SELECT * FROM ticker_outcomes WHERE id = ?",
        (outcome_id,),
    ).fetchone()
    assert outcome_row_is_decision_eligible(authorized) is True
    conn.execute(
        "UPDATE ticker_outcomes SET excess_return_pct = 999.0 WHERE id = ?",
        (outcome_id,),
    )
    conn.commit()
    tampered = conn.execute(
        "SELECT * FROM ticker_outcomes WHERE id = ?",
        (outcome_id,),
    ).fetchone()
    assert outcome_row_is_decision_eligible(tampered) is False
    assert outcomes.realized_returns(conn)["overall"]["n"] == 0
    assert company_depth.decisions(conn, "GOOD")["outcomes"] == []
    conn.close()

    report = build_calibration_report("2026-07-23", db_path=cfg.db_path)
    assert report["source_run_ids"] == []
    assert report["source_outcomes"] == []
    assert report["overall"]["n"] == 0


def test_finalized_outcome_cannot_be_upserted_into_an_authorized_hybrid(
    monkeypatch,
    tmp_path,
):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    source_path = _source_artifact(
        roots["autonomous_sector"],
        "run_immutable_outcome",
        "GOOD",
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", source_path, "run_immutable_outcome")],
    )

    opened = add_outcome(
        ticker="GOOD",
        as_of_date="2026-01-01",
        run_id="run_immutable_outcome",
        decision="BUY",
        conviction=4,
        horizon_days=30,
        entry_price=100.0,
        entry_price_source="watchlist_population",
        entry_date="2026-01-01",
        grade="ACTIONABLE",
        status="DEPLOY_READY",
        benchmark_symbol="SPY",
    )
    assert opened["financial_integrity_fingerprint"]
    closed = close_outcome(
        ticker="GOOD",
        run_id="run_immutable_outcome",
        realized_return_pct=10.0,
        close_date="2026-01-31",
    )
    assert closed["outcome_status"] == "CLOSED"
    with pytest.raises(ValueError, match="finalized outcome row"):
        close_outcome(
            ticker="GOOD",
            run_id="run_immutable_outcome",
            realized_return_pct=999.0,
            close_date="2026-02-01",
        )

    with pytest.raises(ValueError, match="finalized outcome row"):
        add_outcome(
            ticker="GOOD",
            as_of_date="2026-01-01",
            run_id="run_immutable_outcome",
            decision="BUY",
            conviction=1,
            horizon_days=365,
            entry_price=1.0,
            entry_price_source="watchlist_population",
            entry_date="2026-01-01",
            grade="ACTIONABLE",
            status="ACTIVE",
            benchmark_symbol="IWM",
        )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM ticker_outcomes WHERE id = ?",
        (int(opened["id"]),),
    ).fetchone()
    assert row["outcome_status"] == "CLOSED"
    assert row["conviction"] == 4
    assert row["horizon_days"] == 30
    assert row["entry_price"] == 100.0
    assert row["realized_return_pct"] == 10.0
    assert outcome_row_is_decision_eligible(row) is True
    conn.close()


def test_latest_invalid_outcome_suppresses_older_authorized_resolution_and_aggregates(
    monkeypatch,
    tmp_path,
):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    source_path = _source_artifact(
        roots["autonomous_sector"],
        "autonomous_sector_latest_outcome",
        "OPEN_NAME",
        "CLOSED_NAME",
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[
            (
                "autonomous_sector",
                source_path,
                "autonomous_sector_latest_outcome",
            )
        ],
    )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    older_open_id = _insert_outcome(
        conn,
        ticker="OPEN_NAME",
        run_id="autonomous_sector_latest_outcome",
        status="OPEN",
        excess=None,
        as_of_date="2026-01-01",
        updated_at="2026-01-01T00:00:00Z",
    )
    _insert_outcome(
        conn,
        ticker="OPEN_NAME",
        run_id="autonomous_sector_latest_outcome",
        status="OPEN",
        excess=None,
        as_of_date="2026-01-02",
        bind=False,
        updated_at="2026-01-02T00:00:00Z",
    )
    _insert_outcome(
        conn,
        ticker="CLOSED_NAME",
        run_id="autonomous_sector_latest_outcome",
        status="CLOSED",
        excess=10.0,
        as_of_date="2026-01-01",
        updated_at="2026-01-01T00:00:00Z",
    )
    _insert_outcome(
        conn,
        ticker="CLOSED_NAME",
        run_id="autonomous_sector_latest_outcome",
        status="CLOSED",
        excess=999.0,
        as_of_date="2026-01-02",
        bind=False,
        updated_at="2026-01-02T00:00:00Z",
    )
    conn.commit()

    provider = _PriceProvider()
    summary = resolve_open_outcomes(
        "2026-03-01",
        provider=provider,
        db_path=cfg.db_path,
    )
    assert summary.eligible == 0
    assert summary.closed == 0
    assert summary.skipped_unauthorized == 1
    assert provider.calls == []
    assert (
        conn.execute(
            "SELECT outcome_status FROM ticker_outcomes WHERE id = ?",
            (older_open_id,),
        ).fetchone()[0]
        == "OPEN"
    )
    assert outcomes.realized_returns(conn)["overall"]["n"] == 0
    conn.close()

    report = build_calibration_report("2026-07-24", db_path=cfg.db_path)
    assert report["overall"]["n"] == 0
    assert report["source_outcomes"] == []


def test_carried_verdict_requires_exact_outcome_row_and_newest_invalid_suppresses(
    monkeypatch,
    tmp_path,
):
    from app.autonomous.sweep_delta import carried_verdicts

    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    run_id = "autonomous_sector_exact_carry"
    source_path = _source_artifact(
        roots["autonomous_sector"],
        run_id,
        "CARRY",
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", source_path, run_id)],
    )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    _insert_outcome(
        conn,
        ticker="CARRY",
        run_id=run_id,
        as_of_date="2026-01-01",
        updated_at="2026-01-01T00:00:00Z",
    )
    _insert_outcome(
        conn,
        ticker="CARRY",
        run_id=run_id,
        as_of_date="2026-01-02",
        grade="AVOID",
        decision="PASS",
        bind=False,
        updated_at="2026-01-02T00:00:00Z",
    )
    _insert_outcome(
        conn,
        ticker="FORGED",
        run_id=run_id,
        as_of_date="2026-01-01",
        bind=False,
        updated_at="2026-01-01T00:00:00Z",
    )
    conn.commit()

    assert carried_verdicts(conn, ["CARRY", "FORGED"]) == {}
    conn.close()


def test_closing_holding_does_not_mutate_finalized_linked_outcome(
    monkeypatch,
    tmp_path,
):
    from app.holdings import add_holding, close_holding

    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    run_id = "autonomous_sector_holding_outcome"
    source_path = _source_artifact(
        roots["autonomous_sector"],
        run_id,
        "HELD",
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", source_path, run_id)],
    )
    opened = add_outcome(
        ticker="HELD",
        as_of_date="2026-01-01",
        run_id=run_id,
        decision="BUY",
        conviction=4,
        horizon_days=30,
        entry_price=100.0,
        entry_price_source="watchlist_population",
        entry_date="2026-01-01",
        grade="ACTIONABLE",
        status="DEPLOY_READY",
        benchmark_symbol="SPY",
    )
    close_outcome(
        ticker="HELD",
        run_id=run_id,
        realized_return_pct=10.0,
        close_date="2026-01-31",
    )
    holding = add_holding(
        ticker="HELD",
        entry_date="2026-01-01",
        entry_price=100.0,
        db_path=cfg.db_path,
    )
    assert holding["thesis_outcome_id"] == opened["id"]

    conn = sqlite3.connect(cfg.db_path)
    conn.execute(
        """
        INSERT INTO price_quotes(
            ticker, provider, as_of_date, price, currency, status,
            fetched_at, expires_at, raw_json, quote_hash
        ) VALUES ('HELD', 'test', '2026-01-15', 60.0, 'USD', 'OK',
                  '2026-01-15T21:00:00Z', '2026-01-15T21:00:00Z', '{}', 'held-quote')
        """
    )
    conn.commit()
    conn.close()

    close_holding(
        holding_id=int(holding["id"]),
        close_price=90.0,
        closed_at="2026-02-01T00:00:00Z",
        db_path=cfg.db_path,
    )
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    outcome = conn.execute(
        "SELECT * FROM ticker_outcomes WHERE id = ?",
        (int(opened["id"]),),
    ).fetchone()
    assert outcome["outcome_status"] == "CLOSED"
    assert outcome["max_drawdown_pct"] is None
    assert outcome_row_is_decision_eligible(outcome) is True
    conn.close()


def test_closing_holding_refreshes_open_authorized_outcome_drawdown(
    monkeypatch,
    tmp_path,
):
    from app.holdings import add_holding, close_holding

    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    run_id = "autonomous_sector_open_holding_outcome"
    source_path = _source_artifact(
        roots["autonomous_sector"],
        run_id,
        "OPEN_HELD",
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", source_path, run_id)],
    )
    opened = add_outcome(
        ticker="OPEN_HELD",
        as_of_date="2026-01-01",
        run_id=run_id,
        decision="BUY",
        conviction=4,
        horizon_days=30,
        entry_price=100.0,
        entry_price_source="watchlist_population",
        entry_date="2026-01-01",
        grade="ACTIONABLE",
        status="DEPLOY_READY",
        benchmark_symbol="SPY",
    )
    holding = add_holding(
        ticker="OPEN_HELD",
        entry_date="2026-01-01",
        entry_price=100.0,
        db_path=cfg.db_path,
    )
    assert holding["thesis_outcome_id"] == opened["id"]

    conn = sqlite3.connect(cfg.db_path)
    conn.execute(
        """
        INSERT INTO price_quotes(
            ticker, provider, as_of_date, price, currency, status,
            fetched_at, expires_at, raw_json, quote_hash
        ) VALUES ('OPEN_HELD', 'test', '2026-01-15', 60.0, 'USD', 'OK',
                  '2026-01-15T21:00:00Z', '2026-01-15T21:00:00Z', '{}',
                  'open-held-quote')
        """
    )
    conn.commit()
    conn.close()

    close_holding(
        holding_id=int(holding["id"]),
        close_price=90.0,
        closed_at="2026-02-01T00:00:00Z",
        db_path=cfg.db_path,
    )
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    outcome = conn.execute(
        "SELECT * FROM ticker_outcomes WHERE id = ?",
        (int(opened["id"]),),
    ).fetchone()
    history_count = conn.execute(
        "SELECT COUNT(*) FROM ticker_outcomes_history WHERE source_id = ?",
        (int(opened["id"]),),
    ).fetchone()[0]
    assert outcome["outcome_status"] == "OPEN"
    assert outcome["max_drawdown_pct"] == -40.0
    assert outcome_row_is_decision_eligible(outcome) is True
    assert history_count == 1
    conn.close()


def test_discovery_calibration_uses_latest_exact_outcomes_and_authorized_bytes(
    monkeypatch,
    tmp_path,
):
    from app.discovery.calibration import (
        latest_calibration_report,
        run_calibration,
    )
    from app.discovery.runner import _persist_candidate
    from app.discovery.schemas import DiscoveryCandidate

    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    run_id = "autonomous_sector_discovery_calibration"
    source_path = _source_artifact(
        roots["autonomous_sector"],
        run_id,
        "GOOD_CAL",
        "BAD_CAL",
        "FORGED_CAL",
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", source_path, run_id)],
    )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    now = "2026-07-24T12:00:00Z"
    conn.execute(
        """
        INSERT INTO discovery_runs(
            run_id, run_as_of_date, seed_hash, config_hash, seed_path, status,
            processed_count, phase, tickers_targeted_json,
            processed_effective_dates_json, stats_json, created_at, updated_at
        ) VALUES (
            'discovery_exact', '2026-01-01', 'seed', 'cfg', 'seed.csv',
            'COMPLETED', 3, 'full', '["GOOD_CAL","BAD_CAL","FORGED_CAL"]', '{}', '{}', ?, ?
        )
        """,
        (now, now),
    )
    for ticker, stage in (
        ("GOOD_CAL", "ADVANCE_TO_DEEP"),
        ("BAD_CAL", "WATCHLIST_ONLY"),
    ):
        _persist_candidate(
            conn,
            DiscoveryCandidate(
                ticker=ticker,
                cik=f"000000{1 if ticker == 'GOOD_CAL' else 2}",
                run_id="discovery_exact",
                run_as_of_date="2026-01-01",
                effective_as_of_date="2026-01-01",
                market_cap=1_000_000_000.0,
                discovery_score=75.0,
                stage=stage,
                whale_fit_score=19.0,
                evidence_strength_score=7.0,
                key_reasons=["fixture"],
                recommended_action=(
                    "ADD_TO_UNIVERSE" if stage == "ADVANCE_TO_DEEP" else "WATCHLIST_ONLY"
                ),
                suggested_next_pipeline=("FULL_RESEARCH" if stage == "ADVANCE_TO_DEEP" else "NONE"),
            ),
        )
    conn.execute(
        """
        INSERT INTO discovery_candidates(
            ticker, run_id, discovery_score, payload_json, created_at
        ) VALUES ('FORGED_CAL', 'discovery_exact', 999.0, ?, ?)
        """,
        (
            json.dumps(
                {
                    "ticker": "FORGED_CAL",
                    "run_id": "discovery_exact",
                    "stage": "ADVANCE_TO_DEEP",
                    "whale_fit_score": 99.0,
                    "evidence_strength_score": 99.0,
                    "key_reasons": ["fabricated-current-db-row"],
                    "gaps": [],
                },
                sort_keys=True,
            ),
            now,
        ),
    )
    _insert_outcome(
        conn,
        ticker="GOOD_CAL",
        run_id=run_id,
        status="CLOSED",
        excess=5.0,
        discovery_run_id="discovery_exact",
        as_of_date="2026-01-01",
        updated_at="2026-01-01T00:00:00Z",
    )
    _insert_outcome(
        conn,
        ticker="BAD_CAL",
        run_id=run_id,
        status="CLOSED",
        excess=-5.0,
        discovery_run_id="discovery_exact",
        as_of_date="2026-01-01",
        updated_at="2026-01-01T00:00:00Z",
    )
    _insert_outcome(
        conn,
        ticker="BAD_CAL",
        run_id=run_id,
        status="CLOSED",
        excess=999.0,
        discovery_run_id="discovery_exact",
        as_of_date="2026-01-02",
        bind=False,
        updated_at="2026-01-02T00:00:00Z",
    )
    _insert_outcome(
        conn,
        ticker="FORGED_CAL",
        run_id=run_id,
        status="CLOSED",
        excess=999.0,
        discovery_run_id="discovery_exact",
        as_of_date="2026-01-01",
        updated_at="2026-01-01T00:00:00Z",
    )
    conn.commit()
    conn.close()

    report = run_calibration(run_id="discovery_exact", last_n=1)
    assert report["outcomes_total"] == 1
    assert report["excluded_outcomes_total"] == 0
    assert report["closed_outcomes_total"] == 1
    assert len(report["source_outcomes"]) == 1
    assert report["source_outcomes"][0]["outcome_state"]["ticker"] == "GOOD_CAL"
    latest = latest_calibration_report()
    assert latest is not None
    assert latest["run_id"] == "discovery_exact"
    assert latest["closed_outcomes_total"] == 1
    assert len(latest["source_candidates"]) == 2
    assert {binding["candidate_state"]["ticker"] for binding in latest["source_candidates"]} == {
        "GOOD_CAL",
        "BAD_CAL",
    }
    assert "FORGED_CAL" not in {
        binding["candidate_state"]["ticker"] for binding in latest["source_candidates"]
    }
    assert latest["hit_rate_by_stage"] == {
        "ADVANCE_TO_DEEP": {
            "closed": 1,
            "positive": 1,
            "negative": 0,
            "hit_rate_pct": 100.0,
        }
    }

    report_path = Path(report["json_path"])
    report_bytes = report_path.read_bytes()
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    assert len(outcomes.calibration(conn)) == 1
    original_candidate_json = conn.execute(
        """
        SELECT payload_json
        FROM discovery_candidates
        WHERE run_id = 'discovery_exact' AND ticker = 'GOOD_CAL'
        """
    ).fetchone()[0]
    tampered_candidate = json.loads(original_candidate_json)
    tampered_candidate["stage"] = "WATCHLIST_ONLY"
    conn.execute(
        """
        UPDATE discovery_candidates
        SET payload_json = ?
        WHERE run_id = 'discovery_exact' AND ticker = 'GOOD_CAL'
        """,
        (json.dumps(tampered_candidate, sort_keys=True),),
    )
    conn.commit()
    assert latest_calibration_report() is None
    assert outcomes.calibration(conn) == []
    conn.execute(
        """
        UPDATE discovery_candidates
        SET payload_json = ?
        WHERE run_id = 'discovery_exact' AND ticker = 'GOOD_CAL'
        """,
        (original_candidate_json,),
    )
    conn.commit()
    assert latest_calibration_report() is not None
    assert len(outcomes.calibration(conn)) == 1
    conn.execute(
        """
        UPDATE discovery_candidates
        SET ticker = 'MISSING_CAL'
        WHERE run_id = 'discovery_exact' AND ticker = 'GOOD_CAL'
        """
    )
    conn.commit()
    assert latest_calibration_report() is None
    assert outcomes.calibration(conn) == []
    conn.execute(
        """
        UPDATE discovery_candidates
        SET ticker = 'GOOD_CAL'
        WHERE run_id = 'discovery_exact' AND ticker = 'MISSING_CAL'
        """
    )
    conn.commit()
    conn.close()
    assert latest_calibration_report() is not None

    from app.autonomous.artifact_financial_audit import (
        write_typed_financial_authorization,
    )

    duplicate_payload = json.loads(report_bytes)
    duplicate_payload["source_candidates"].append(duplicate_payload["source_candidates"][0])
    report_path.write_text(
        json.dumps(duplicate_payload, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="inconsistent"):
        write_typed_financial_authorization(
            report_path,
            Path(report["md_path"]),
            artifact_type="discovery_calibration",
        )
    report_path.write_bytes(report_bytes)
    assert latest_calibration_report() is not None

    report_path.write_text(
        report_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert latest_calibration_report() is None


def test_v1_conflicting_or_duplicate_decision_rows_fail_closed(monkeypatch, tmp_path):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))

    duplicate_path = _source_artifact(
        roots["autonomous_sector"],
        "run_duplicate_decision",
        "DUP",
    )
    duplicate_payload = json.loads(duplicate_path.read_text(encoding="utf-8"))
    duplicate_payload["relative_ranking"] = [
        {"ticker": "DUP", "company_autonomy_verdict": "ACTIONABLE"},
        {"ticker": "DUP", "company_autonomy_verdict": "AVOID"},
    ]
    duplicate_path.write_text(
        json.dumps(duplicate_payload, sort_keys=True),
        encoding="utf-8",
    )

    conflicting_path = _source_artifact(
        roots["autonomous_sector"],
        "run_conflicting_decision",
        "CONFLICT",
    )
    conflicting_payload = json.loads(conflicting_path.read_text(encoding="utf-8"))
    conflicting_payload["relative_ranking"] = [
        {
            "ticker": "CONFLICT",
            "company_autonomy_verdict": "ACTIONABLE",
            "verdict": "AVOID",
        }
    ]
    conflicting_path.write_text(
        json.dumps(conflicting_payload, sort_keys=True),
        encoding="utf-8",
    )

    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[
            ("autonomous_sector", duplicate_path, "run_duplicate_decision"),
            ("autonomous_sector", conflicting_path, "run_conflicting_decision"),
        ],
    )

    assert authorized_emitted_decision_binding("run_duplicate_decision", "DUP") is None
    assert authorized_emitted_decision_binding("run_conflicting_decision", "CONFLICT") is None


def test_manual_watchlist_add_rejects_before_price_fetch_and_preserves_exact_row(
    monkeypatch,
    tmp_path,
):
    from typer.testing import CliRunner

    from app.cli import app as cli_app
    from app.watchlist.contract import WatchlistEntry
    from app.watchlist.store import add_or_update

    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    source_path = _source_artifact(
        roots["autonomous_sector"],
        "run_authorized_prior",
        "GOOD",
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", source_path, "run_authorized_prior")],
    )
    add_or_update(
        WatchlistEntry(
            ticker="GOOD",
            status="ACTIVE",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
            conviction_source="company_autonomy",
            source_run_id="run_authorized_prior",
            source_sector="technology",
            added_at="2026-07-23T12:00:00Z",
            thesis_text="Exact authorized prior thesis.",
        ),
        db_path=cfg.db_path,
    )
    price_calls: list[str] = []
    monkeypatch.setattr(
        "app.cli._fetch_watchlist_manual_add_price",
        lambda ticker: price_calls.append(ticker),
    )

    result = CliRunner().invoke(
        cli_app,
        [
            "watchlist",
            "add",
            "GOOD",
            "--thesis",
            "Unaudited replacement.",
            "--buy-price",
            "1.00",
        ],
    )
    assert result.exit_code == 1
    assert "exact authorized run/ticker lineage" in result.output
    assert price_calls == []
    conn = sqlite3.connect(cfg.db_path)
    rows = conn.execute(
        "SELECT source_run_id, thesis_text FROM watchlist WHERE ticker = 'GOOD' ORDER BY id"
    ).fetchall()
    conn.close()
    assert rows == [("run_authorized_prior", "Exact authorized prior thesis.")]


def test_method_scoreboard_requires_exact_source_valuation(monkeypatch, tmp_path):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    valid_record = valuation_source_record(
        {
            "ticker": "GOOD",
            "as_of_date": "2026-01-01",
            "method": "deep_research",
            "inputs_json": "{}",
            "outputs_json": json.dumps(
                {"status": "OK", "value_per_share": 120.0},
                sort_keys=True,
            ),
            "warnings_json": "[]",
            "created_at": "2026-01-01T12:00:00+00:00",
            "valuation_writer_version": "test",
            "quality_gate_verdict": "PROCEED",
            "confidence_class": "HIGH",
            "gate_reason_codes": "[]",
            "valuation_headwinds": "[]",
            "valuation_supports": "[]",
            "source_run_id": "run_valid",
        }
    )
    assert valid_record is not None
    valid_path = _source_artifact(
        roots["autonomous_sector"],
        "run_valid",
        "GOOD",
        valuation_source_records=[valid_record],
    )
    unlisted_path = _source_artifact(roots["autonomous_sector"], "run_unlisted", "BAD")
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", valid_path, "run_valid")],
    )
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    valid_valuation = _insert_valuation(
        conn,
        ticker="GOOD",
        as_of_date="2026-01-01",
        method="deep_research",
        value=120.0,
        source_path=valid_path,
        source_run_id="run_valid",
    )
    invalid_valuation = _insert_valuation(
        conn,
        ticker="BAD",
        as_of_date="2026-02-01",
        method="deep_research",
        value=120.0,
        source_path=unlisted_path,
        source_run_id="run_unlisted",
    )
    for ticker, source_date, valuation_id, result in (
        ("GOOD", "2026-01-01", valid_valuation, "CORRECT"),
        ("BAD", "2026-02-01", invalid_valuation, "INCORRECT"),
    ):
        source = conn.execute("SELECT * FROM valuations WHERE id = ?", (valuation_id,)).fetchone()
        conn.execute(
            """
            INSERT INTO deep_research_outcomes(
                ticker, source_as_of_date, source_valuation_id, source_run_id,
                source_artifact_path, source_artifact_sha256,
                source_valuation_fingerprint, horizon_days,
                target_exit_date, scan_date, entry_price, exit_price,
                exit_as_of_date, exit_source, price_change_pct, thesis_verdict,
                verdict_outcome, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 30, '2026-03-01', '2026-03-02',
                      100.0, 110.0,
                      '2026-03-01', 'fixture', 10.0, 'UNDERVALUED', ?, 'now')
            """,
            (
                ticker,
                source_date,
                valuation_id,
                source["source_run_id"],
                source["source_artifact_path"],
                source["source_artifact_sha256"],
                source["financial_integrity_fingerprint"],
                result,
            ),
        )
        outcome_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.execute(
            """
            INSERT INTO deep_research_method_outcomes(
                outcome_id, method, predicted_value, entry_price, direction,
                outcome, unadjusted_source, created_at
            ) VALUES (?, 'dcf', 120.0, 100.0, 'UNDERVALUED', ?, 0, 'now')
            """,
            (outcome_id, result),
        )
    conn.commit()

    expected_scoreboard = {
        "methods": [
            {
                "method": "dcf",
                "n": 1,
                "correct": 1,
                "incorrect": 0,
                "inconclusive": 0,
                "unadjusted_source": False,
                "hit_rate": 1.0,
            }
        ],
        "monthly": [
            {
                "month": "2026-01",
                "method": "dcf",
                "n": 1,
                "correct": 1,
                "incorrect": 0,
            }
        ],
    }
    assert outcomes.method_scoreboard(conn) == expected_scoreboard

    # A later valuation write may reuse the live row's id.  The outcome must
    # remain bound to the exact historical valuation version it measured.
    from app.valuation.valuation_writer import _archive_valuation_row

    replacement_outputs = json.dumps({"intrinsic_value": 999.0}, sort_keys=True)
    _archive_valuation_row(
        conn,
        ticker="GOOD",
        as_of_date="2026-01-01",
        method="deep_research",
        new_outputs_json=replacement_outputs,
        new_source_run_id="replacement_run",
        new_source_artifact_path="/replacement/not_authorized.json",
        new_source_artifact_sha256="b" * 64,
        new_financial_integrity_fingerprint="replacement_fingerprint",
    )
    conn.execute(
        """
        UPDATE valuations
        SET outputs_json = ?,
            source_run_id = 'replacement_run',
            source_artifact_path = '/replacement/not_authorized.json',
            source_artifact_sha256 = ?,
            financial_integrity_fingerprint = 'replacement_fingerprint'
        WHERE id = ?
        """,
        (replacement_outputs, "b" * 64, valid_valuation),
    )
    conn.commit()
    assert (
        conn.execute(
            """
            SELECT COUNT(*) FROM valuations_history
            WHERE source_id = ?
              AND financial_integrity_fingerprint != 'replacement_fingerprint'
            """,
            (valid_valuation,),
        ).fetchone()[0]
        == 1
    )
    assert outcomes.method_scoreboard(conn) == expected_scoreboard
    conn.close()


def test_calibration_report_uses_only_authorized_runs(monkeypatch, tmp_path):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    valid_path = _source_artifact(roots["autonomous_sector"], "run_valid", "GOOD")
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", valid_path, "run_valid")],
    )
    conn = sqlite3.connect(cfg.db_path)
    _insert_outcome(conn, ticker="GOOD", run_id="run_valid", excess=5.0)
    _insert_outcome(conn, ticker="FABRICATED", run_id="run_valid", excess=999.0)
    _insert_outcome(conn, ticker="BAD", run_id="run_unlisted", excess=99.0)
    conn.commit()
    conn.close()

    report = build_calibration_report("2026-07-23", db_path=cfg.db_path)
    assert report["source_run_ids"] == ["run_valid"]
    assert report["overall"] == {
        "n": 1,
        "hit_rate": 1.0,
        "excess_hit_rate": 1.0,
        "target_hit_rate": None,
        "avg_return": 9.0,
        "median_return": 9.0,
        "avg_excess": 5.0,
    }


def test_calibration_build_is_exact_authorized_and_web_readable(monkeypatch, tmp_path):
    from app.autonomous.artifact_financial_audit import (
        PASS,
        TYPED_AUTHORIZATION_SUFFIX,
        authorized_artifact_bytes,
    )

    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    valid_path = _source_artifact(
        roots["autonomous_sector"],
        "run_valid",
        "GOOD",
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", valid_path, "run_valid")],
    )
    conn = sqlite3.connect(cfg.db_path)
    _insert_outcome(conn, ticker="GOOD", run_id="run_valid", excess=5.0)
    conn.commit()
    conn.close()

    report = build_calibration_report("2026-07-23", db_path=cfg.db_path)
    report_path = Path(cfg.calibration_dir) / "calibration_report_2026-07-23.json"
    authorization_path = report_path.with_name(f"{report_path.stem}{TYPED_AUTHORIZATION_SUFFIX}")
    assert authorization_path.is_file()
    status, exact_bytes = authorized_artifact_bytes(report_path)
    assert status == PASS
    assert json.loads(exact_bytes or b"{}") == report

    latest = latest_grade_status_report()
    assert latest is not None
    assert latest["run_id"] == report["run_id"]
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    web_reports = outcomes.calibration(conn)
    conn.close()
    assert [item["run_id"] for item in web_reports] == [report["run_id"]]
    response = client.get("/api/outcomes")
    assert response.status_code == 200
    assert [item["run_id"] for item in response.json()["calibration"]] == [report["run_id"]]

    original = report_path.read_text(encoding="utf-8")
    mutated = original.replace('"n": 1', '"n": 9', 1)
    assert mutated != original and len(mutated) == len(original)
    report_path.write_text(mutated, encoding="utf-8")
    assert authorized_artifact_bytes(report_path)[0] != PASS
    assert latest_grade_status_report() is None
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    assert outcomes.calibration(conn) == []
    conn.close()


def test_refresh_false_rederives_current_authorized_bytes_not_stale_cache(
    monkeypatch,
    tmp_path,
):
    cfg = _init(monkeypatch, tmp_path)
    run_path = _source_artifact(
        Path(cfg.runs_dir) / "autonomous_sector",
        "run_current",
        "AAA",
    )
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "sector": "technology",
            "created_at": "2026-07-23T10:00:00Z",
            "final_verdict": "SELECTED",
            "selected_ticker": "AAA",
        }
    )
    run_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", run_path, "run_current")],
    )
    from app.web.readmodel import runs_index

    first = client.get("/api/runs")
    assert [row["run_id"] for row in first.json()["runs"]] == ["run_current"]

    original_stat = run_path.stat()
    original = run_path.read_text(encoding="utf-8")
    mutated = original.replace('"AAA"', '"BBB"')
    assert len(mutated) == len(original)
    run_path.write_text(mutated, encoding="utf-8")
    os.utime(
        run_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", run_path, "run_current")],
    )

    with runs_index.open_ui_db(cfg) as ui_conn:
        stale_rows = runs_index.list_indexed_runs(ui_conn)
    assert len(stale_rows) == 1
    assert stale_rows[0]["integrity_status"] == "PASS"
    assert stale_rows[0]["decision_eligible"] == 1
    assert stale_rows[0]["selected_ticker"] == "BBB"
    assert stale_rows[0]["final_verdict"] == "SELECTED"

    refreshed = client.get("/api/runs", params={"refresh": "false"})
    assert refreshed.status_code == 200
    assert [row["selected_ticker"] for row in refreshed.json()["runs"]] == ["BBB"]


def test_central_valuation_selectors_never_resurrect_older_authorized_rows(
    monkeypatch,
    tmp_path,
):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    valid_path = _source_artifact(
        roots["autonomous_sector"],
        "run_valid_old",
        "AAA",
    )
    invalid_path = _source_artifact(
        roots["autonomous_sector"],
        "run_unlisted_new",
        "AAA",
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", valid_path, "run_valid_old")],
    )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    _insert_valuation(
        conn,
        ticker="AAA",
        as_of_date="2026-06-01",
        method="scorecard",
        value=80.0,
        source_path=valid_path,
        source_run_id="run_valid_old",
    )
    _insert_valuation(
        conn,
        ticker="AAA",
        as_of_date="2026-07-01",
        method="scorecard",
        value=120.0,
        source_path=invalid_path,
        source_run_id="run_unlisted_new",
    )
    conn.commit()

    assert (
        latest_decision_eligible_valuation_row(
            conn,
            ticker="AAA",
            method="scorecard",
            as_of_date="2026-07-23",
        )
        is None
    )
    assert (
        latest_decision_eligible_valuation_rows(
            conn,
            ticker="AAA",
            methods=["scorecard"],
            as_of_date="2026-07-23",
        )
        == []
    )

    conn.execute("DELETE FROM valuations WHERE ticker = 'AAA' AND as_of_date = '2026-07-01'")
    conn.commit()
    selected = latest_decision_eligible_valuation_row(
        conn,
        ticker="AAA",
        method="scorecard",
        as_of_date="2026-07-23",
    )
    assert selected is not None
    assert selected["as_of_date"] == "2026-06-01"
    conn.close()


def test_valuation_selector_requires_exact_artifact_issuer_binding(
    monkeypatch,
    tmp_path,
):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    scorecard_outputs = {"status": "OK", "value_per_share": 80.0}
    source_record = valuation_source_record(
        {
            "ticker": "AAA",
            "as_of_date": "2026-07-01",
            "method": "scorecard",
            "inputs_json": "{}",
            "outputs_json": json.dumps(scorecard_outputs, sort_keys=True),
            "warnings_json": "[]",
            "created_at": "2026-07-01T12:00:00+00:00",
            "valuation_writer_version": "test",
            "quality_gate_verdict": "PROCEED",
            "confidence_class": "HIGH",
            "gate_reason_codes": "[]",
            "valuation_headwinds": "[]",
            "valuation_supports": "[]",
            "source_run_id": "run_exact_issuer",
        }
    )
    assert source_record is not None
    source_path = _source_artifact(
        roots["autonomous_sector"],
        "run_exact_issuer",
        "AAA",
        issuer_ciks={"AAA": "0000000042"},
        valuation_source_records=[source_record],
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", source_path, "run_exact_issuer")],
    )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    _insert_valuation(
        conn,
        ticker="AAA",
        as_of_date="2026-07-01",
        method="scorecard",
        value=80.0,
        source_path=source_path,
        source_run_id="run_exact_issuer",
        outputs=scorecard_outputs,
    )
    conn.commit()

    selected = latest_decision_eligible_valuation_row(
        conn,
        ticker="AAA",
        method="scorecard",
        as_of_date="2026-07-23",
        expected_issuer_cik="42",
        expected_issuer_aliases=("AAA", "PRIMARY"),
        require_exact_issuer_binding=True,
    )
    assert selected is not None
    assert selected["as_of_date"] == "2026-07-01"
    assert (
        latest_decision_eligible_valuation_row(
            conn,
            ticker="AAA",
            method="scorecard",
            as_of_date="2026-07-23",
            expected_issuer_cik="43",
            expected_issuer_aliases=("AAA", "PRIMARY"),
            require_exact_issuer_binding=True,
        )
        is None
    )
    assert (
        latest_decision_eligible_valuation_row(
            conn,
            ticker="AAA",
            method="scorecard",
            as_of_date="2026-07-23",
            expected_issuer_aliases=("AAA",),
            require_exact_issuer_binding=True,
        )
        is None
    )
    conn.close()


def test_production_valuation_consumers_suppress_an_invalid_newest_row(
    monkeypatch,
    tmp_path,
):
    from app.alpha.signal_assembler import _load_scorecard_record
    from app.alpha.solvency_scanner import _load_market_cap
    from app.autonomous.sector_financial_packets import _expectations_gap_from_db
    from app.evidence.packet_builder import (
        _build_packet_payload,
        _latest_valuation_as_of,
    )
    from app.research.deep_research import _load_reverse_dcf, _load_scorecard
    from app.sector.synthesis import _load_valuation_snapshots
    from app.universe.sector_universe import _load_writer_scorecard_payload

    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    valid_path = _source_artifact(
        roots["autonomous_sector"],
        "run_valid_old",
        "AAA",
    )
    invalid_path = _source_artifact(
        roots["autonomous_sector"],
        "run_unlisted_new",
        "AAA",
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", valid_path, "run_valid_old")],
    )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    scorecard_outputs = {
        "status": "OK",
        "pricing_zone": "MARGIN_OF_SAFETY",
        "pricing_zone_detail": {
            "current_price": 10.0,
            "market_cap": 100.0,
        },
    }
    reverse_dcf_outputs = {
        "status": "OK",
        "outputs": {"implied_growth": 0.03},
        "expectations_gap": {
            "bucket": "EXPECTATIONS_GAP_LOW",
            "implied_growth": 0.03,
        },
    }
    for method, outputs in (
        ("scorecard", scorecard_outputs),
        ("reverse_dcf", reverse_dcf_outputs),
    ):
        _insert_valuation(
            conn,
            ticker="AAA",
            as_of_date="2026-06-01",
            method=method,
            value=80.0,
            source_path=valid_path,
            source_run_id="run_valid_old",
            outputs=outputs,
        )
        _insert_valuation(
            conn,
            ticker="AAA",
            as_of_date="2026-07-01",
            method=method,
            value=120.0,
            source_path=invalid_path,
            source_run_id="run_unlisted_new",
            outputs=outputs,
        )
    conn.execute(
        """
        INSERT INTO fundamentals(
            ticker, as_of_date, metrics_json, quality_flags_json, created_at
        ) VALUES ('AAA', '2026-07-01', '{}', '[]', '2026-07-01T12:00:00+00:00')
        """
    )
    conn.execute(
        """
        INSERT INTO fundamentals(
            ticker, as_of_date, metrics_json, quality_flags_json, created_at
        ) VALUES ('AAA', '2026-06-01', '{}', '[]', '2026-06-01T12:00:00+00:00')
        """
    )
    conn.commit()
    conn.close()

    assert _load_scorecard_record(
        "AAA",
        as_of_date="2026-07-23",
        db_path=cfg.db_path,
    ) == (None, {})
    assert (
        _load_market_cap(
            "AAA",
            as_of_date="2026-07-23",
            db_path=cfg.db_path,
        )
        is None
    )
    assert _load_scorecard("AAA", None) == (None, None)
    assert _load_reverse_dcf("AAA", None) is None
    assert (
        _load_valuation_snapshots(
            tickers=["AAA"],
            as_of_date="2026-07-01",
        )
        == {}
    )
    assert (
        _load_writer_scorecard_payload(
            ticker="AAA",
            as_of_date="2026-07-01",
        )
        == {}
    )
    assert _expectations_gap_from_db(
        "AAA",
        as_of_date="2026-07-01",
    ) == {"bucket": "EXPECTATIONS_GAP_UNRELIABLE"}

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    assert _latest_valuation_as_of(conn, "AAA") is None
    payload = _build_packet_payload(conn, "AAA", "2026-07-01")
    conn.close()
    assert payload is not None
    assert payload["valuations"] == {}

    conn = sqlite3.connect(cfg.db_path)
    conn.execute("DELETE FROM valuations WHERE ticker = 'AAA' AND as_of_date = '2026-07-01'")
    conn.commit()
    conn.close()

    assert _load_scorecard_record(
        "AAA",
        as_of_date="2026-07-23",
        db_path=cfg.db_path,
    ) == ("2026-06-01", scorecard_outputs)
    assert (
        _load_market_cap(
            "AAA",
            as_of_date="2026-07-23",
            db_path=cfg.db_path,
        )
        == 100.0
    )
    assert _load_scorecard("AAA", None) == (
        scorecard_outputs,
        "2026-06-01",
    )
    assert _load_reverse_dcf("AAA", None) == reverse_dcf_outputs
    assert set(
        _load_valuation_snapshots(
            tickers=["AAA"],
            as_of_date="2026-06-01",
        )["AAA"]
    ) == {"reverse_dcf", "scorecard"}
    assert (
        _load_writer_scorecard_payload(
            ticker="AAA",
            as_of_date="2026-06-01",
        )
        == scorecard_outputs
    )
    assert (
        _expectations_gap_from_db(
            "AAA",
            as_of_date="2026-06-01",
        )["bucket"]
        == "EXPECTATIONS_GAP_LOW"
    )

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    assert _latest_valuation_as_of(conn, "AAA") == "2026-06-01"
    payload = _build_packet_payload(conn, "AAA", "2026-06-01")
    conn.close()
    assert payload is not None
    assert set(payload["valuations"]) == {"reverse_dcf", "scorecard"}


def test_render_and_publication_readers_suppress_invalid_newest_valuations(
    monkeypatch,
    tmp_path,
):
    from app.alpha.report_writer import _load_insurance_packet_summary
    from app.insurance.packet import load_latest_insurance_packet
    from app.insurance.sources import latest_scorecard
    from app.llm.synthesis_agent import append_synthesis_section
    from app.synthesis.variant_builder import _load_valuation_payload
    from app.valuation.valuation_render import _load_method_payloads
    from app.valuation.valuation_writer import (
        _append_valuation_section_inner,
        _prior_run_delta,
    )

    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=Path(cfg.runs_dir))
    valid_path = _source_artifact(
        roots["autonomous_sector"],
        "run_valid_old",
        "AAA",
    )
    invalid_path = _source_artifact(
        roots["autonomous_sector"],
        "run_unlisted_new",
        "AAA",
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[("autonomous_sector", valid_path, "run_valid_old")],
    )

    valid_outputs = {
        "dcf": {
            "status": "OK",
            "base": 80.0,
            "low": 70.0,
            "high": 90.0,
        },
        "scorecard": {
            "status": "OK",
            "signal": "VALID_SIGNAL",
            "quality_context": {
                "gate_action": "PROCEED",
                "valuation_supports": ["VALID_SUPPORT"],
            },
        },
        "insurance_packet": {
            "status": "OK",
            "marker": "VALID_INSURANCE",
        },
    }
    invalid_outputs = {
        "dcf": {
            "status": "OK",
            "base": 120.0,
            "low": 110.0,
            "high": 130.0,
        },
        "scorecard": {
            "status": "OK",
            "signal": "INVALID_SIGNAL",
            "quality_context": {
                "gate_action": "PROCEED",
                "valuation_supports": ["INVALID_SUPPORT"],
            },
        },
        "insurance_packet": {
            "status": "OK",
            "marker": "INVALID_INSURANCE",
        },
    }
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    for method in ("dcf", "scorecard", "insurance_packet"):
        _insert_valuation(
            conn,
            ticker="AAA",
            as_of_date="2026-06-01",
            method=method,
            value=80.0,
            source_path=valid_path,
            source_run_id="run_valid_old",
            outputs=valid_outputs[method],
        )
        _insert_valuation(
            conn,
            ticker="AAA",
            as_of_date="2026-07-01",
            method=method,
            value=120.0,
            source_path=invalid_path,
            source_run_id="run_unlisted_new",
            outputs=invalid_outputs[method],
        )

    synthesis_path = tmp_path / "synthesis.json"
    synthesis_path.write_text(
        json.dumps(
            {
                "as_of_date": "2026-07-01",
                "business_quality_summary": "Test quality.",
                "valuation_interpretation": "Test valuation.",
                "decision_frame": {"stance": "watchlist"},
            }
        ),
        encoding="utf-8",
    )
    conn.execute(
        """
        INSERT INTO synthesis_packets(
            ticker, as_of_date, run_id, packet_path, packet_hash, packet_json,
            prompt_hash, input_hash, provider, model, usage_json,
            cost_estimate_usd, from_cache, created_at
        ) VALUES(
            'AAA', '2026-07-01', 'synthesis_run', ?, 'packet-hash', '{}',
            'prompt-hash', 'input-hash', 'disabled', 'fixture', '{}',
            0, 0, '2026-07-01T12:00:00+00:00'
        )
        """,
        (str(synthesis_path),),
    )
    conn.commit()

    assert _load_method_payloads("AAA", None) == {}
    assert _load_method_payloads("AAA", "2026-07-01") == {}
    assert latest_scorecard("AAA", as_of_date="2026-07-23") == (None, {})
    assert load_latest_insurance_packet("AAA") == {}
    assert _load_insurance_packet_summary("AAA") == {}
    assert (
        _load_valuation_payload(
            ticker="AAA",
            as_of_date="2026-07-01",
        )
        == {}
    )
    assert _prior_run_delta(
        conn,
        "AAA",
        "2026-08-01",
        "dcf",
        100.0,
    ) == {"status": "PRIOR_RUN_NOT_AVAILABLE"}

    valuation_dossier = tmp_path / "valuation_dossier.md"
    valuation_dossier.write_text("# Dossier\n", encoding="utf-8")
    _append_valuation_section_inner(
        "AAA",
        "2026-07-01",
        str(valuation_dossier),
    )
    assert valuation_dossier.read_text(encoding="utf-8") == "# Dossier\n"

    synthesis_dossier = tmp_path / "synthesis_dossier.md"
    synthesis_dossier.write_text("# Dossier\n", encoding="utf-8")
    append_synthesis_section(
        ticker="AAA",
        run_id="synthesis_run",
        dossier_md_path=str(synthesis_dossier),
    )
    blocked_synthesis = synthesis_dossier.read_text(encoding="utf-8")
    assert "INVALID_SUPPORT" not in blocked_synthesis
    assert "| Gate Verdict | UNKNOWN |" in blocked_synthesis

    conn.execute("DELETE FROM valuations WHERE ticker = 'AAA' AND as_of_date = '2026-07-01'")
    conn.commit()

    assert _load_method_payloads("AAA", None)["dcf"]["base"] == 80.0
    assert latest_scorecard("AAA", as_of_date="2026-07-23") == (
        "2026-06-01",
        valid_outputs["scorecard"],
    )
    assert load_latest_insurance_packet("AAA") == valid_outputs["insurance_packet"]
    assert _load_insurance_packet_summary("AAA") == valid_outputs["insurance_packet"]
    assert set(
        _load_valuation_payload(
            ticker="AAA",
            as_of_date="2026-06-01",
        )
    ) == {"dcf", "insurance_packet", "scorecard"}
    prior_delta = _prior_run_delta(
        conn,
        "AAA",
        "2026-08-01",
        "dcf",
        100.0,
    )
    assert prior_delta == {
        "status": "OK",
        "prior_as_of_date": "2026-06-01",
        "prior_value": 80.0,
        "current_value": 100.0,
        "value_change_pct": 0.25,
    }

    _append_valuation_section_inner(
        "AAA",
        "2026-06-01",
        str(valuation_dossier),
    )
    assert "VALID_SIGNAL" in valuation_dossier.read_text(encoding="utf-8")

    synthesis_path.write_text(
        json.dumps(
            {
                "as_of_date": "2026-06-01",
                "business_quality_summary": "Test quality.",
                "valuation_interpretation": "Test valuation.",
                "decision_frame": {"stance": "watchlist"},
            }
        ),
        encoding="utf-8",
    )
    authorized_synthesis_dossier = tmp_path / "authorized_synthesis_dossier.md"
    authorized_synthesis_dossier.write_text("# Dossier\n", encoding="utf-8")
    append_synthesis_section(
        ticker="AAA",
        run_id="synthesis_run",
        dossier_md_path=str(authorized_synthesis_dossier),
    )
    assert "VALID_SUPPORT" in authorized_synthesis_dossier.read_text(encoding="utf-8")
    conn.close()
