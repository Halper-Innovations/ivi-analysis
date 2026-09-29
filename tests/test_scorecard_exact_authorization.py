from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.autonomous.artifact_financial_audit import (
    AUDIT_SCHEMA_VERSION,
    AUDIT_SCOPE_ID,
    CANONICAL_AUDIT_ROOT_IDS,
)
from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.config import get_config
from app.db import init_db
from app.valuation.lineage import (
    valuation_integrity_fingerprint,
    valuation_source_record,
)
from app.valuation.valuation_writer import valuation_facts_fingerprint


RUN_AS_OF = "2026-07-23"
PRICE_SNAPSHOT = {
    "price": 10.0,
    "currency": "USD",
    "as_of_date": "2026-06-30",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_roots(cfg) -> dict[str, Path]:
    roots = {
        "autonomous_sector": (Path(cfg.runs_dir) / "autonomous_sector").resolve(),
        "analyst_output": Path(cfg.analyst_outputs_dir).resolve(),
        "scan": (Path(cfg.outputs_dir) / "scans").resolve(),
        "research_output": Path(cfg.research_dir).resolve(),
        "watchlist_report": (Path(cfg.outputs_dir) / "digests").resolve(),
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    return roots


def _source_artifact(root: Path, *, run_id: str) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "autonomous_sector_run.json"
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "company_packets": [
                    {
                        "ticker": "AAA",
                        "issuer_cik": "0000000042",
                        "financial_integrity_status": "PASS",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path.resolve()


def _install_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    roots: dict[str, Path],
    source_path: Path,
    source_run_id: str,
) -> None:
    record = {
        "path": str(source_path),
        "family": "autonomous_sector",
        "sha256": _sha256(source_path),
        "integrity_status": "PASS",
        "decision_eligible": True,
        "run_id": source_run_id,
    }
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_scope_id": AUDIT_SCOPE_ID,
        "generated_at": "2026-07-23T20:00:00Z",
        "complete": True,
        "source_roots": [
            {
                "family": family,
                "root_id": CANONICAL_AUDIT_ROOT_IDS[family],
                "path": str(root),
            }
            for family, root in roots.items()
        ],
        "summary": {
            "artifacts_scanned": 1,
            "tickers_scanned": 1,
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
        "artifacts": [record],
        "violations": [],
    }
    manifest_path = tmp_path / "financial_integrity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    monkeypatch.setenv(
        "VOE_FINANCIAL_INTEGRITY_MANIFEST",
        str(manifest_path),
    )


def _insert_scorecard(
    conn: sqlite3.Connection,
    *,
    as_of_date: str,
    anchor: float,
    source_path: Path,
    source_run_id: str,
    facts_fingerprint: str,
) -> int:
    inputs = {
        "pipeline_version": "v2",
        "require_filed_asof": True,
        "issuer_cik": "0000000042",
        "market_price": 10.0,
        "price_currency": "USD",
        "price_as_of_date": "2026-06-30",
        "facts_fingerprint": facts_fingerprint,
    }
    outputs = {"status": "OK", "anchor": anchor}
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
            'AAA', ?, 'scorecard', ?, ?, '[]', ?, 'test', 'PROCEED',
            'HIGH', '[]', '[]', '[]', ?, ?, ?, NULL
        )
        """,
        (
            as_of_date,
            json.dumps(inputs, sort_keys=True),
            json.dumps(outputs, sort_keys=True),
            f"{as_of_date}T12:00:00+00:00",
            source_run_id,
            str(source_path),
            _sha256(source_path),
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
    _seal_source_artifact(conn, source_path)
    return row_id


def _seal_source_artifact(conn: sqlite3.Connection, source_path: Path) -> None:
    """Mirror the production writer's exact row claims in the source bytes."""

    rows = conn.execute(
        "SELECT * FROM valuations WHERE source_artifact_path = ?",
        (str(source_path),),
    ).fetchall()
    records = [valuation_source_record(row) for row in rows]
    assert all(record is not None for record in records)
    canonical_records = sorted(
        (record for record in records if record is not None),
        key=lambda record: (
            record["row"]["ticker"],
            record["row"]["as_of_date"],
            record["row"]["method"],
            record["row"]["created_at"],
        ),
    )
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["valuation_source_records"] = canonical_records
    source_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    source_sha256 = _sha256(source_path)

    for row in rows:
        values = dict(row)
        values["source_artifact_sha256"] = source_sha256
        conn.execute(
            """
            UPDATE valuations
            SET source_artifact_sha256 = ?,
                financial_integrity_fingerprint = ?
            WHERE id = ?
            """,
            (
                source_sha256,
                valuation_integrity_fingerprint(values),
                int(row["id"]),
            ),
        )

    manifest_path = Path(os.environ["VOE_FINANCIAL_INTEGRITY_MANIFEST"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matched = False
    for record in manifest["artifacts"]:
        if record.get("path") == str(source_path):
            record["sha256"] = source_sha256
            matched = True
    if matched:
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _build_exact_source_book(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SimpleNamespace:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)

    roots = _canonical_roots(cfg)
    valid_path = _source_artifact(
        roots["autonomous_sector"],
        run_id="run_valid_old",
    )
    invalid_path = _source_artifact(
        roots["autonomous_sector"],
        run_id="run_unlisted_new",
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        source_path=valid_path,
        source_run_id="run_valid_old",
    )
    get_config.cache_clear()

    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        INSERT INTO companies(ticker, cik, name, created_at)
        VALUES ('AAA', '42', 'AAA', '2026-07-23T00:00:00+00:00')
        """
    )
    conn.execute(
        """
        INSERT INTO companyfacts_facts(
            ticker, fiscal_year, period_type, period_end, line_item, value,
            units, source_url, fetched_at, filed_date, form, accession
        ) VALUES (
            'AAA', 2025, 'FY', '2025-12-31', 'revenue', 100.0,
            'USD_millions',
            'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json',
            '2026-02-15T00:00:00+00:00', '2026-02-15', '10-K',
            '0000000042-26-000001'
        )
        """
    )
    facts_fingerprint = valuation_facts_fingerprint(
        "AAA",
        conn,
        as_of_date=RUN_AS_OF,
        issuer_cik="42",
        issuer_aliases=("AAA",),
    )
    valid_row_id = _insert_scorecard(
        conn,
        as_of_date="2026-07-01",
        anchor=80.0,
        source_path=valid_path,
        source_run_id="run_valid_old",
        facts_fingerprint=facts_fingerprint,
    )
    invalid_row_id = _insert_scorecard(
        conn,
        as_of_date="2026-07-02",
        anchor=999.0,
        source_path=invalid_path,
        source_run_id="run_unlisted_new",
        facts_fingerprint=facts_fingerprint,
    )
    conn.commit()
    conn.close()
    return SimpleNamespace(
        cfg=cfg,
        valid_row_id=valid_row_id,
        invalid_row_id=invalid_row_id,
    )


def test_v2_provenance_never_resurrects_older_authorized_scorecard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    book = _build_exact_source_book(monkeypatch, tmp_path)
    from app.valuation.provenance import validate_v2_scorecard_provenance

    conn = sqlite3.connect(str(book.cfg.db_path))
    conn.row_factory = sqlite3.Row
    blocked = validate_v2_scorecard_provenance(
        conn,
        "AAA",
        as_of_date=RUN_AS_OF,
        issuer_cik="42",
        issuer_aliases=("AAA",),
        price_snapshot=PRICE_SNAPSHOT,
    )
    assert blocked == {
        "row_id": book.invalid_row_id,
        "raw_asof": "2026-07-02",
        "validated_asof": None,
        "mismatch_reasons": ["VALUATION_EXACT_SOURCE_UNAUTHORIZED"],
        "inputs": {},
        "outputs": {},
    }

    conn.execute("DELETE FROM valuations WHERE id = ?", (book.invalid_row_id,))
    conn.commit()
    authorized = validate_v2_scorecard_provenance(
        conn,
        "AAA",
        as_of_date=RUN_AS_OF,
        issuer_cik="42",
        issuer_aliases=("AAA",),
        price_snapshot=PRICE_SNAPSHOT,
    )
    conn.close()

    assert authorized["row_id"] == book.valid_row_id
    assert authorized["raw_asof"] == "2026-07-01"
    assert authorized["validated_asof"] == "2026-07-01"
    assert authorized["mismatch_reasons"] == []
    assert authorized["outputs"] == {"status": "OK", "anchor": 80.0}


def test_v2_checkpoint_snapshot_omits_invalid_newest_scorecard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    book = _build_exact_source_book(monkeypatch, tmp_path)
    from app.autonomous.evidence_resolution import (
        _latest_scorecard_asof,
        _valuation_evidence_snapshot,
    )

    conn = sqlite3.connect(str(book.cfg.db_path))
    conn.row_factory = sqlite3.Row
    assert _latest_scorecard_asof(conn, "AAA", cutoff_date=RUN_AS_OF) == "2026-07-02"
    assert (
        _valuation_evidence_snapshot(
            conn,
            "AAA",
            as_of_date=RUN_AS_OF,
        )
        == {}
    )

    exact_older = _valuation_evidence_snapshot(
        conn,
        "AAA",
        as_of_date="2026-07-01",
    )
    conn.close()

    assert exact_older["id"] == book.valid_row_id
    assert exact_older["as_of_date"] == "2026-07-01"
    assert exact_older["source_run_id"] == "run_valid_old"
    assert exact_older["financial_integrity_fingerprint"]


def test_discover_exact_snapshot_requires_real_source_authorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    book = _build_exact_source_book(monkeypatch, tmp_path)
    import app.discover.sweep as sweep

    converted: list[int] = []

    def convert(row, *, engine_db_path):
        del engine_db_path
        converted.append(int(row["id"]))
        return (
            str(row["ticker"]),
            str(row["as_of_date"]),
            json.loads(row["outputs_json"]),
            {"row_id": int(row["id"])},
        )

    monkeypatch.setattr(sweep, "_authorized_scorecard_input", convert)

    with pytest.raises(InvalidFinancialInputError):
        sweep.load_authorized_discover_universe(book.cfg.db_path)
    assert converted == []

    with pytest.raises(InvalidFinancialInputError):
        sweep._load_scorecards_for_snapshot(
            book.cfg.db_path,
            [("AAA", "2026-07-02")],
        )
    assert converted == []

    universe, packets = sweep._load_scorecards_for_snapshot(
        book.cfg.db_path,
        [("AAA", "2026-07-01")],
    )
    assert universe == [("AAA", "2026-07-01", {"status": "OK", "anchor": 80.0})]
    assert packets == {"AAA": {"row_id": book.valid_row_id}}
    assert converted == [book.valid_row_id]
