from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from app.autonomous.artifact_financial_audit import (
    AUDIT_SCHEMA_VERSION,
    AUDIT_SCOPE_ID,
    CANONICAL_AUDIT_ROOT_IDS,
)
from app.db import init_db, utc_now_iso
from app.valuation.lineage import (
    valuation_integrity_fingerprint,
    valuation_source_record,
)


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


def _source_artifact(root: Path, run_id: str, tickers: tuple[str, ...]) -> Path:
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
                    }
                    for ticker in tickers
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path.resolve()


def _install_manifest(
    monkeypatch,
    tmp_path: Path,
    *,
    roots: dict[str, Path],
    source_path: Path,
) -> None:
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_scope_id": AUDIT_SCOPE_ID,
        "generated_at": utc_now_iso(),
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
                "path": str(source_path),
                "family": "autonomous_sector",
                "sha256": _sha256(source_path),
                "integrity_status": "PASS",
                "decision_eligible": True,
                "run_id": "run_valid",
            }
        ],
        "violations": [],
    }
    manifest_path = tmp_path / "financial_integrity_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "VOE_FINANCIAL_INTEGRITY_MANIFEST",
        str(manifest_path),
    )


def _init_authorized_book(
    monkeypatch,
    tmp_path: Path,
    *tickers: str,
):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    roots = _canonical_roots(cfg)
    normalized_tickers = tuple(str(ticker).strip().upper() for ticker in tickers)
    valid_path = _source_artifact(
        roots["autonomous_sector"],
        "run_valid",
        normalized_tickers,
    )
    invalid_path = _source_artifact(
        roots["autonomous_sector"],
        "run_invalid",
        normalized_tickers,
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        source_path=valid_path,
    )
    get_config.cache_clear()
    return get_config(), valid_path, invalid_path


def _insert_valuation(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    as_of_date: str,
    method: str,
    source_path: Path,
    source_run_id: str,
    inputs: dict | None = None,
    outputs: dict | None = None,
    created_at: str | None = None,
) -> int:
    conn.execute(
        """
        INSERT INTO valuations(
            ticker, as_of_date, method, inputs_json, outputs_json, warnings_json,
            created_at, valuation_writer_version, quality_gate_verdict,
            confidence_class, gate_reason_codes, valuation_headwinds,
            valuation_supports, source_run_id, source_artifact_path,
            source_artifact_sha256, financial_integrity_fingerprint
        ) VALUES (?, ?, ?, ?, ?, '[]', ?, 'test', 'PROCEED', 'HIGH',
                  '[]', '[]', '[]', ?, ?, ?, NULL)
        """,
        (
            ticker.upper(),
            as_of_date,
            method,
            json.dumps(inputs or {}, sort_keys=True),
            json.dumps(outputs or {}, sort_keys=True),
            created_at or f"{as_of_date}T12:00:00+00:00",
            source_run_id,
            str(source_path),
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


def _seed_membership(conn: sqlite3.Connection, ticker: str) -> None:
    conn.execute(
        """
        INSERT INTO sector_inference(
            ticker, as_of_date, inferred_sector, score, derived_from, created_at
        ) VALUES (?, '2026-07-01', 'biotech', 1.0, '[]',
                  '2026-07-01T00:00:00+00:00')
        """,
        (ticker,),
    )
    conn.execute(
        """
        INSERT INTO sec_registrants(
            cik, primary_ticker, all_tickers, name, exchange_scope,
            operating_status, in_scope, sector, intake_status,
            first_seen_at, last_seen_at
        ) VALUES (?, ?, ?, ?, 'US_PRIMARY', 'OPERATING', 1, 'biotech',
                  'PENDING', '2026-07-01T00:00:00+00:00',
                  '2026-07-01T00:00:00+00:00')
        """,
        (
            f"cik-{ticker.lower()}",
            ticker,
            json.dumps([ticker]),
            f"{ticker} Corp",
        ),
    )


def test_invalid_newest_scorecard_is_missing_for_coverage_consumers(
    monkeypatch,
    tmp_path,
):
    cfg, valid_path, invalid_path = _init_authorized_book(
        monkeypatch,
        tmp_path,
        "AAA",
        "BBB",
    )
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    _seed_membership(conn, "AAA")
    _seed_membership(conn, "BBB")
    _insert_valuation(
        conn,
        ticker="AAA",
        as_of_date="2026-06-01",
        method="scorecard",
        source_path=valid_path,
        source_run_id="run_valid",
        outputs={"pricing_zone_detail": {"current_price": 10.0}},
    )
    _insert_valuation(
        conn,
        ticker="AAA",
        as_of_date="2026-07-01",
        method="scorecard",
        source_path=invalid_path,
        source_run_id="run_invalid",
        outputs={"pricing_zone_detail": {"current_price": 99.0}},
    )
    _insert_valuation(
        conn,
        ticker="BBB",
        as_of_date="2026-07-01",
        method="scorecard",
        source_path=valid_path,
        source_run_id="run_valid",
        outputs={"pricing_zone_detail": {"current_price": 25.0}},
    )
    conn.commit()

    from app.sector.catalog import list_scannable_sectors
    from app.sector.classifier import load_scorecarded_tickers
    from app.universe.expand import _existing_scorecards
    from app.universe.registrant_intake import _has_scorecard, _ingest_scope

    assert load_scorecarded_tickers(conn) == ["BBB"]
    assert _existing_scorecards(cfg.db_path) == {"BBB"}
    assert _has_scorecard(conn, "AAA") is False
    assert _has_scorecard(conn, "BBB") is True
    assert [row["primary_ticker"] for row in _ingest_scope(conn)] == ["AAA"]
    assert list_scannable_sectors(db_path=cfg.db_path) == [{"sector": "biotech", "count": 1}]
    conn.close()

    observed_prices: dict[str, float | None] = {}

    def fake_cap_classification(
        ticker,
        *,
        as_of_date,
        asof_price,
        db_path,
        price_lookup,
    ):
        del as_of_date, db_path, price_lookup
        observed_prices[ticker] = asof_price
        return SimpleNamespace(
            cap_source="UNKNOWN_CAP",
            band_label="unknown",
            cap_band=None,
            market_cap_mm=None,
            detail="test",
        )

    monkeypatch.setattr(
        "app.autonomous.cap_census.classify_market_cap_for_band_filter",
        fake_cap_classification,
    )
    from app.autonomous.cap_census import census_cap_resolution

    census_cap_resolution(
        as_of_date="2026-07-01",
        db_path=cfg.db_path,
        price_lookup=lambda _ticker, _as_of: None,
    )
    assert observed_prices == {"AAA": None, "BBB": 25.0}

    monkeypatch.setattr(
        "app.autonomous.sector_candidates._security_filter_for_ticker",
        lambda _ticker: SimpleNamespace(is_common_equity=True),
    )
    from app.autonomous.sweep_delta import _current_v1_membership

    membership = _current_v1_membership()
    assert membership["missing_scorecard_tickers"] == ["AAA"]

    packet_calls: list[str] = []

    def fake_build_packet(ticker):
        packet_calls.append(ticker)
        return tmp_path / f"{ticker}.json"

    monkeypatch.setattr(
        "app.evidence.packet_builder.build_packet_for_ticker",
        fake_build_packet,
    )
    from app.evidence.packet_builder import build_all_packets

    assert build_all_packets() == 1
    assert packet_calls == ["BBB"]


def test_value_readers_stop_at_invalid_newest_row(monkeypatch, tmp_path):
    cfg, valid_path, invalid_path = _init_authorized_book(
        monkeypatch,
        tmp_path,
        "AAA",
        "PPC",
    )
    research_path = tmp_path / "AAA.json"
    research_path.write_text("{}\n", encoding="utf-8")
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    invalid_scorecard_id = _insert_valuation(
        conn,
        ticker="AAA",
        as_of_date="2026-07-01",
        method="scorecard",
        source_path=invalid_path,
        source_run_id="run_invalid",
        inputs={"market_price": 99.0},
        outputs={"pricing_zone_detail": {"current_price": 99.0}},
    )
    _insert_valuation(
        conn,
        ticker="AAA",
        as_of_date="2026-06-01",
        method="scorecard",
        source_path=valid_path,
        source_run_id="run_valid",
        inputs={"market_price": 80.0},
        outputs={"pricing_zone_detail": {"current_price": 80.0}},
    )
    invalid_research_id = _insert_valuation(
        conn,
        ticker="AAA",
        as_of_date="2026-07-01",
        method="deep_research",
        source_path=invalid_path,
        source_run_id="run_invalid",
        outputs={"artifact_path": str(research_path)},
        created_at="2026-07-01T13:00:00+00:00",
    )
    _insert_valuation(
        conn,
        ticker="AAA",
        as_of_date="2026-06-01",
        method="deep_research",
        source_path=valid_path,
        source_run_id="run_valid",
        outputs={"artifact_path": str(research_path)},
    )
    _insert_valuation(
        conn,
        ticker="PPC",
        as_of_date="2026-07-01",
        method="dcf",
        source_path=valid_path,
        source_run_id="run_valid",
        inputs={"market_price": 40.0},
        created_at="2026-07-01T10:00:00+00:00",
    )
    invalid_price_id = _insert_valuation(
        conn,
        ticker="PPC",
        as_of_date="2026-07-01",
        method="epv",
        source_path=invalid_path,
        source_run_id="run_invalid",
        inputs={"market_price": 400.0},
        created_at="2026-07-01T14:00:00+00:00",
    )
    conn.commit()

    from app.calibration.perception_tracker import (
        _load_registered_market_price,
    )
    from app.discover.stage4_context import _fetch_historical_scorecards
    from app.research import latest_research_artifact_path
    from app.valuation.peer_context import _latest_market_price

    assert (
        _latest_market_price(
            conn,
            "AAA",
            as_of_date="2026-07-01",
        )
        is None
    )
    assert _fetch_historical_scorecards("AAA", 2) == ("No scorecards found for AAA")
    assert latest_research_artifact_path("AAA") is None
    assert (
        _load_registered_market_price(
            ticker="PPC",
            as_of_date="2026-07-01",
        )
        == 40.0
    )

    conn.execute(
        "DELETE FROM valuations WHERE id IN (?, ?, ?)",
        (invalid_scorecard_id, invalid_research_id, invalid_price_id),
    )
    conn.commit()

    assert (
        _latest_market_price(
            conn,
            "AAA",
            as_of_date="2026-07-01",
        )
        == 80.0
    )
    assert _fetch_historical_scorecards("AAA", 2).startswith("2026-06-01:")
    assert latest_research_artifact_path("AAA") == research_path
    assert (
        _load_registered_market_price(
            ticker="PPC",
            as_of_date="2026-07-01",
        )
        == 40.0
    )
    conn.close()


def test_gating_and_promotion_require_exact_authorized_rows(
    monkeypatch,
    tmp_path,
):
    cfg, valid_path, invalid_path = _init_authorized_book(
        monkeypatch,
        tmp_path,
        "GATE",
        "TECH",
    )
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        INSERT INTO fundamentals(
            ticker, as_of_date, metrics_json, quality_flags_json, created_at
        ) VALUES ('GATE', '2026-07-01', '{}', '[]',
                  '2026-07-01T00:00:00+00:00')
        """
    )
    _insert_valuation(
        conn,
        ticker="GATE",
        as_of_date="2026-07-01",
        method="dcf",
        source_path=valid_path,
        source_run_id="run_valid",
    )
    invalid_reverse_id = _insert_valuation(
        conn,
        ticker="GATE",
        as_of_date="2026-07-01",
        method="reverse_dcf",
        source_path=invalid_path,
        source_run_id="run_invalid",
    )
    invalid_tech_id = _insert_valuation(
        conn,
        ticker="TECH",
        as_of_date="2026-07-01",
        method="tech_adjustment",
        source_path=invalid_path,
        source_run_id="run_invalid",
        outputs={"category": "TECH"},
    )
    conn.commit()

    from app.ops.gating import _ticker_report
    from app.universe.promotion import (
        _load_tech_adjustment_payload_for_ticker,
    )

    report = _ticker_report(
        conn,
        "GATE",
        as_of_date="2026-07-01",
        run_id="test-run",
        with_research=False,
        artifact_paths={},
    )
    assert report["valuation"] is False
    assert (
        _load_tech_adjustment_payload_for_ticker(
            "TECH",
            {"as_of_date": "2026-07-01", "tech_adjustment_cache": {}},
            {"valuation_cache": {}},
        )
        == {}
    )

    conn.execute(
        "DELETE FROM valuations WHERE id IN (?, ?)",
        (invalid_reverse_id, invalid_tech_id),
    )
    _insert_valuation(
        conn,
        ticker="GATE",
        as_of_date="2026-07-01",
        method="reverse_dcf",
        source_path=valid_path,
        source_run_id="run_valid",
    )
    _insert_valuation(
        conn,
        ticker="TECH",
        as_of_date="2026-07-01",
        method="tech_adjustment",
        source_path=valid_path,
        source_run_id="run_valid",
        outputs={"category": "TECH"},
    )
    conn.commit()

    report = _ticker_report(
        conn,
        "GATE",
        as_of_date="2026-07-01",
        run_id="test-run",
        with_research=False,
        artifact_paths={},
    )
    assert report["valuation"] is True
    assert _load_tech_adjustment_payload_for_ticker(
        "TECH",
        {"as_of_date": "2026-07-01", "tech_adjustment_cache": {}},
        {"valuation_cache": {}},
    ) == {"category": "TECH"}
    conn.close()
