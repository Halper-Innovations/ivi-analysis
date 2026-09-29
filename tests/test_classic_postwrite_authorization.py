from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.autonomous import artifact_financial_audit
from app.autonomous.artifact_financial_audit import (
    PASS,
    STALE_AUDIT,
    UNAUDITED,
    audit_artifact_tree,
    authorized_artifact_bytes,
    run_id_is_decision_eligible,
)
from app.autonomous.financial_integrity import stable_quote_snapshot_id
from app.autonomous.output_store import persist_autonomous_sector_run
from app.autonomous.run_contract import ToolCallRecord
from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    SectorCompanyFinancialPacket,
)
from app.autonomous.sweep_delta import (
    V1_ATOMIC_BANDS,
    record_loaded_set,
    record_unknown_cap_cross_band_coverage,
    record_zero_cost_coverage,
    swept_tickers_for_band,
)
from app.config import get_config
from app.db import get_db, init_db
from app.valuation.lineage import (
    valuation_integrity_fingerprint,
    valuation_row_is_decision_eligible,
)
from app.valuation.valuation_writer import ensure_valuation
from app.watchlist.contract import WatchlistEntry
from app.watchlist.digest import write_digest
from app.watchlist.dispositions import close_disposition, sync_at_target_dispositions
from app.watchlist.lineage import watchlist_row_is_decision_eligible
from app.watchlist.store import add_or_update
from app.web.readmodel import company, reader, today


@pytest.fixture(autouse=True)
def _classic_fixture_bypasses_run_binding_gate(monkeypatch):
    """These legacy postwrite fixtures isolate authorization after persistence."""

    monkeypatch.setattr(
        "app.autonomous.output_store._require_publication_financial_integrity_binding",
        lambda *args, **_kwargs: None,
    )


def _baseline_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    cfg = get_config()
    roots = {
        "runs": cfg.runs_dir / "autonomous_sector",
        "analyst": cfg.outputs_dir / "analyst_outputs",
        "scans": cfg.outputs_dir / "scans",
        "research": cfg.research_dir,
        "digests": cfg.outputs_dir / "digests",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    (roots["scans"] / "empty_scan_fixture.json").write_text("{}\n", encoding="utf-8")
    reports = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "audit",
        generated_at=datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setenv(
        "VOE_FINANCIAL_INTEGRITY_MANIFEST",
        str(reports.manifest_json),
    )
    return reports


def _lineage_packet() -> SectorCompanyFinancialPacket:
    packet = SectorCompanyFinancialPacket(
        ticker="AAA",
        financial_status="READY",
        model_fit_status="SUPPORTED",
        data_quality_status="COMPLETE",
        market_cap_unit="USD_millions",
        current_price_unit="USD_per_share",
        price_basis="UNADJUSTED",
        shares_unit="shares_millions",
        shares_basis="UNADJUSTED",
        split_adjustment_factor=1.0,
        financial_integrity_status="PASS",
    )
    packet.quote_snapshot_id = stable_quote_snapshot_id(packet.to_dict())
    return packet


def _valid_classic_artifact(
    run_id: str,
    *,
    company_packets: list[SectorCompanyFinancialPacket] | None = None,
) -> AutonomousSectorFinancialRunArtifact:
    return AutonomousSectorFinancialRunArtifact(
        run_id=run_id,
        sector="energy",
        market_cap_focus="mid_cap",
        objective="Complete a deterministic no-selection review.",
        as_of_date="2026-07-23",
        created_at="2026-07-23T12:01:00Z",
        completed_at="2026-07-23T12:02:00Z",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence="LOW",
        candidate_selection={
            "source": "sector_scan_db",
            "loaded_tickers": ["AAA"],
            "selected_tickers": [],
        },
        company_packets=list(company_packets or []),
        tool_calls=[
            ToolCallRecord(
                call_id="TC1",
                tool_name="fetch_financial_history",
                tool_input={"ticker": "AAA"},
                rationale="Confirm the deterministic review inputs.",
                status="OK",
                evidence_ref_ids=[],
            )
        ],
    )


def test_postwrite_authorization_binds_pending_valuation_rows(
    monkeypatch,
    tmp_path,
):
    _baseline_manifest(monkeypatch, tmp_path)
    init_db()
    run_id = "autonomous_sector_energy_20260723_valuation_lineage"
    from tests.test_valuation_writer import _fake_quote, _seed_companyfacts

    with get_db() as conn:
        _seed_companyfacts(conn, ticker="AAA")
    price_provider = MagicMock()
    price_provider.get_quote.return_value = _fake_quote(
        150.0,
        ticker="AAA",
        as_of_date="2026-07-23",
    )
    written_records = ensure_valuation(
        "AAA",
        "2026-07-23",
        provider=price_provider,
        run_id=run_id,
        force_refresh=True,
        raise_on_error=True,
    )
    controlled_source_record = next(
        record for record in written_records if record["row"]["method"] == "dcf"
    )

    fabricated_controlled_record = json.loads(json.dumps(controlled_source_record))
    fabricated_controlled_record["row"]["method"] = "fabricated_method"
    fabricated_controlled_record["row"]["outputs_json"] = '{"value_per_share":999999.0}'
    with pytest.raises(RuntimeError, match="uncontrolled or malformed"):
        persist_autonomous_sector_run(
            _valid_classic_artifact(run_id, company_packets=[_lineage_packet()]),
            valuation_source_records=[fabricated_controlled_record],
        )

    allowlisted_forgery = json.loads(json.dumps(controlled_source_record))
    allowlisted_forgery["row"]["outputs_json"] = json.dumps(
        {"status": "OK", "low": 999999.0, "base": 999999.0, "high": 999999.0},
        separators=(",", ":"),
    )
    with pytest.raises(RuntimeError, match="artifact-embedded writer results"):
        persist_autonomous_sector_run(
            _valid_classic_artifact(run_id, company_packets=[_lineage_packet()]),
            valuation_source_records=[allowlisted_forgery],
        )

    with get_db() as conn:
        conn.execute(
            """
            UPDATE valuations
            SET outputs_json = ?
            WHERE ticker = 'AAA' AND as_of_date = '2026-07-23' AND method = 'dcf'
            """,
            (allowlisted_forgery["row"]["outputs_json"],),
        )
        forged_row = conn.execute(
            """
            SELECT *
            FROM valuations
            WHERE ticker = 'AAA' AND as_of_date = '2026-07-23' AND method = 'dcf'
            """
        ).fetchone()
        conn.execute(
            """
            UPDATE valuations
            SET financial_integrity_fingerprint = ?
            WHERE id = ?
            """,
            (valuation_integrity_fingerprint(forged_row), forged_row["id"]),
        )
    embedded_forgery = _valid_classic_artifact(
        run_id,
        company_packets=[_lineage_packet()],
    )
    embedded_forgery.candidate_selection["data_gap_repair"] = {
        "candidate_states": [
            {
                "ticker": "AAA",
                "valuation_source_records": [allowlisted_forgery],
            }
        ]
    }
    with pytest.raises(RuntimeError, match="deterministic-writer receipts"):
        persist_autonomous_sector_run(embedded_forgery)

    with get_db() as conn:
        conn.execute(
            """
            UPDATE valuations
            SET outputs_json = ?
            WHERE ticker = 'AAA' AND as_of_date = '2026-07-23' AND method = 'dcf'
            """,
            (controlled_source_record["row"]["outputs_json"],),
        )
        restored_row = conn.execute(
            """
            SELECT *
            FROM valuations
            WHERE ticker = 'AAA' AND as_of_date = '2026-07-23' AND method = 'dcf'
            """
        ).fetchone()
        conn.execute(
            """
            UPDATE valuations
            SET financial_integrity_fingerprint = ?
            WHERE id = ?
            """,
            (valuation_integrity_fingerprint(restored_row), restored_row["id"]),
        )

    production_artifact = _valid_classic_artifact(
        run_id,
        company_packets=[_lineage_packet()],
    )
    production_artifact.candidate_selection["data_gap_repair"] = {
        "candidate_states": [
            {
                "ticker": "AAA",
                "valuation_source_records": [controlled_source_record],
            }
        ]
    }
    paths = persist_autonomous_sector_run(production_artifact)

    with get_db() as conn:
        current = conn.execute(
            "SELECT * FROM valuations WHERE ticker = 'AAA' AND method = 'dcf'"
        ).fetchone()
        history = conn.execute(
            """
            SELECT *
            FROM valuations_history
            WHERE ticker = 'AAA' AND method = 'dcf'
            """
        ).fetchall()
    assert current["source_artifact_path"] == str(paths.artifact_json.resolve())
    assert (
        current["source_artifact_sha256"]
        == hashlib.sha256(paths.artifact_json.read_bytes()).hexdigest()
    )
    assert valuation_row_is_decision_eligible(current) is True
    assert len(history) == 1
    assert history[0]["source_artifact_path"] is None


def test_valid_classic_write_is_exactly_authorized_before_same_invocation_coverage(
    monkeypatch,
    tmp_path,
):
    _baseline_manifest(monkeypatch, tmp_path)
    init_db()
    from app.autonomous.structural_gate import evaluate_structural_gate
    from app.sector.scan import classify_tickers_for_market_cap

    run_id = "autonomous_sector_energy_20260723_exact"
    with get_db() as db_conn:
        db_conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form,
                accession
            )
            VALUES(
                'AAA', 2025, 'FY', '2025-12-31', ?,
                ?, 'USD_millions',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                '2026-02-15T00:00:00Z', '2026-02-15', '10-K',
                '0000000001-26-000001'
            )
            """,
            (("net_income", 12.3), ("cfo", -4.5)),
        )
    cfg = get_config()
    classifications = {
        ticker: classification.to_dict()
        for ticker, classification in classify_tickers_for_market_cap(
            tickers=["AAA", "BARE", "FORGED"],
            as_of_date="2026-07-23",
            db_path=cfg.db_path,
            pipeline_version="v1",
            allow_live_market_data=False,
            cfg=cfg,
        ).items()
    }
    full_gate = evaluate_structural_gate(
        "AAA",
        as_of_date="2026-07-23",
        price=classifications["AAA"]["price_used"],
        market_cap_mm=classifications["AAA"]["market_cap_mm"],
        db_path=cfg.db_path,
    ).to_dict()
    assert full_gate["triggered_codes"] == ["EARNINGS_QUALITY_DIVERGENCE"]
    forged_gate = json.loads(json.dumps(full_gate))
    forged_gate["ticker"] = "FORGED"

    artifact = _valid_classic_artifact(run_id)
    artifact.candidate_selection.update(
        {
            "loaded_tickers": ["AAA", "BARE", "FORGED"],
            "cap_classifications": classifications,
            "structural_gate_results": {
                "AAA": full_gate,
                "BARE": {"quarantined": True, "excluded_error": False},
                "FORGED": forged_gate,
            },
        }
    )
    paths = persist_autonomous_sector_run(artifact)

    assert paths.authorization_json is not None
    authorization = json.loads(paths.authorization_json.read_text(encoding="utf-8"))
    records = {row["kind"]: row for row in authorization["artifacts"]}
    assert records["artifact_json"]["path"] == str(paths.artifact_json.resolve())
    assert records["report_markdown"]["path"] == str(paths.report_md.resolve())
    assert (
        records["artifact_json"]["sha256"]
        == hashlib.sha256(paths.artifact_json.read_bytes()).hexdigest()
    )
    assert (
        records["report_markdown"]["sha256"]
        == hashlib.sha256(paths.report_md.read_bytes()).hexdigest()
    )
    assert run_id_is_decision_eligible(run_id) is True

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    for ticker, loaded_at in (
        ("AAA", "2026-07-23T12:03:00Z"),
        ("BARE", "2026-07-23T12:03:01Z"),
        ("FORGED", "2026-07-23T12:03:02Z"),
    ):
        record_loaded_set(
            conn,
            run_id=run_id,
            sector="energy",
            market_cap_focus="mid_cap",
            source="sector_scan_db",
            tickers=[ticker],
            loaded_at=loaded_at,
            pipeline_version="v1",
            candidate_dispositions={ticker: "STRUCTURAL_SCREENED"},
        )
    assert swept_tickers_for_band(
        conn,
        "mid_cap",
        "energy",
        pipeline_version="v1",
    ) == {"AAA"}
    conn.execute(
        """
        UPDATE sector_run_loaded_sets
        SET market_cap_focus = 'micro_cap'
        WHERE run_id = ? AND ticker = 'AAA'
        """,
        (run_id,),
    )
    assert swept_tickers_for_band(conn, "micro_cap", "energy") == set()
    conn.execute(
        """
        UPDATE sector_run_loaded_sets
        SET market_cap_focus = 'mid_cap', sector = 'utilities'
        WHERE run_id = ? AND ticker = 'AAA'
        """,
        (run_id,),
    )
    assert swept_tickers_for_band(conn, "mid_cap", "utilities") == set()


def test_v1_artifact_filters_require_independent_replay_and_bind_ledger_source(
    monkeypatch,
    tmp_path,
):
    from app.autonomous import sweep_delta

    _baseline_manifest(monkeypatch, tmp_path)
    init_db()
    with get_db() as conn:
        conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form,
                accession
            ) VALUES(
                ?, ?, 'FY', ?, 'revenue', 10.0, 'USD_millions',
                ?, '2026-02-15T00:00:00Z', ?, '10-K', ?
            )
            """,
            [
                (
                    "SPARSE",
                    2025,
                    "2025-12-31",
                    "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
                    "2026-02-15",
                    "0000000001-26-000001",
                ),
                *[
                    (
                        "RICH",
                        fiscal_year,
                        f"{fiscal_year}-12-31",
                        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000002.json",
                        f"{fiscal_year + 1}-02-15",
                        f"0000000002-{str(fiscal_year + 1)[-2:]}-000001",
                    )
                    for fiscal_year in (2023, 2024, 2025)
                ],
            ],
        )

    run_id = "autonomous_sector_energy_20260724_filter_replay"
    artifact = _valid_classic_artifact(run_id)
    artifact.candidate_selection.update(
        {
            "source": "sector_scan_db",
            "loaded_tickers": ["CARRY", "SPARSE", "RICH", "FRAME"],
            "delta_audit": {
                "carried_verdicts": {
                    "CARRY": {
                        "outcome_id": 999999,
                        "verdict": "ACTIONABLE",
                        "run_id": "autonomous_missing",
                    }
                }
            },
            "financial_history_filter": {
                "status": "FILTERED_SPARSE_FINANCIAL_HISTORY",
                "minimum_rows": 3,
                "excluded_tickers": ["SPARSE", "RICH"],
                "year_counts": {"SPARSE": 1, "RICH": 0},
            },
            "framework_evidence_filter": {
                "status": "FILTERED_ZERO_PACKET_SUPPORT",
                "excluded_tickers": ["FRAME"],
            },
        }
    )
    paths = persist_autonomous_sector_run(artifact)
    assert paths.authorization_json is not None

    explicit_run_id = "autonomous_sector_energy_20260724_explicit_source"
    explicit_artifact = _valid_classic_artifact(explicit_run_id)
    explicit_artifact.candidate_selection.update(
        {
            "source": "explicit_tickers",
            "loaded_tickers": ["EXPLICIT"],
            "financial_history_filter": {
                "status": "FILTERED_SPARSE_FINANCIAL_HISTORY",
                "minimum_rows": 3,
                "excluded_tickers": ["EXPLICIT"],
                "year_counts": {"EXPLICIT": 0},
            },
        }
    )
    explicit_paths = persist_autonomous_sector_run(explicit_artifact)
    assert explicit_paths.authorization_json is not None

    with get_db() as conn:
        record_loaded_set(
            conn,
            run_id=run_id,
            sector="energy",
            market_cap_focus="mid_cap",
            source="sector_scan_db",
            tickers=["CARRY", "SPARSE", "RICH", "FRAME"],
            pipeline_version="v1",
            candidate_dispositions={
                "CARRY": "CARRIED_VERDICT",
                "SPARSE": "NEEDS_DATA_SPARSE_HISTORY",
                "RICH": "NEEDS_DATA_SPARSE_HISTORY",
                "FRAME": "NEEDS_DATA_FRAMEWORK_EVIDENCE",
            },
        )
        assert sweep_delta._authorized_terminal_dispositions(
            run_id,
            pipeline_version="v1",
            conn=conn,
        ) == {"SPARSE": "NEEDS_DATA_SPARSE_HISTORY"}
        assert swept_tickers_for_band(conn, "mid_cap", "energy") == {"SPARSE"}
        conn.execute(
            """
            UPDATE sector_run_loaded_sets
            SET source = 'mutated_sweep_source'
            WHERE run_id = ?
            """,
            (run_id,),
        )
        assert (
            sweep_delta._authorized_terminal_dispositions(
                run_id,
                pipeline_version="v1",
                conn=conn,
            )
            is None
        )
        assert swept_tickers_for_band(conn, "mid_cap", "energy") == set()

        record_loaded_set(
            conn,
            run_id=explicit_run_id,
            sector="energy",
            market_cap_focus="mid_cap",
            source="explicit_tickers",
            tickers=["EXPLICIT"],
            pipeline_version="v1",
            candidate_dispositions={"EXPLICIT": "NEEDS_DATA_SPARSE_HISTORY"},
        )
        assert (
            sweep_delta._authorized_terminal_dispositions(
                explicit_run_id,
                pipeline_version="v1",
                conn=conn,
            )
            is None
        )
        assert "EXPLICIT" not in swept_tickers_for_band(conn, "mid_cap", "energy")

        conn.execute(
            """
            UPDATE sector_run_loaded_sets
            SET source = 'sector_scan_db'
            WHERE run_id = ?
            """,
            (explicit_run_id,),
        )
        assert (
            sweep_delta._authorized_terminal_dispositions(
                explicit_run_id,
                pipeline_version="v1",
                conn=conn,
            )
            is None
        )
        assert "EXPLICIT" not in swept_tickers_for_band(conn, "mid_cap", "energy")


def test_zero_cost_coverage_requires_and_reuses_exact_canonical_receipt(
    monkeypatch,
    tmp_path,
):
    _baseline_manifest(monkeypatch, tmp_path)
    init_db()
    from app.autonomous.structural_gate import evaluate_structural_gate
    from app.sector.scan import classify_tickers_for_market_cap

    cfg = get_config()
    with get_db() as conn:
        conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form,
                accession
            )
            VALUES(
                'DIVG', 2025, 'FY', '2025-12-31', ?,
                ?, 'USD_millions',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                '2026-02-15T00:00:00Z', '2026-02-15', '10-K',
                '0000000001-26-000001'
            )
            """,
            (("net_income", 12.3), ("cfo", -4.5)),
        )
    classification = classify_tickers_for_market_cap(
        tickers=["DIVG"],
        as_of_date="2026-07-23",
        db_path=cfg.db_path,
        pipeline_version="v1",
        allow_live_market_data=False,
        cfg=cfg,
    )["DIVG"].to_dict()
    gate = evaluate_structural_gate(
        "DIVG",
        as_of_date="2026-07-23",
        price=classification["price_used"],
        market_cap_mm=classification["market_cap_mm"],
        db_path=cfg.db_path,
    ).to_dict()
    assert gate["triggered_codes"] == ["EARNINGS_QUALITY_DIVERGENCE"]
    evidence = {
        "DIVG": {
            "disposition": "STRUCTURAL_SCREENED",
            "effective_as_of": "2026-07-23",
            "sector": "energy",
            "market_cap_focus": "micro_cap",
            "cap_classification": classification,
            "structural_gate_result": gate,
        }
    }
    with get_db() as conn:
        first = record_zero_cost_coverage(
            conn,
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["DIVG"],
            candidate_dispositions={"DIVG": "STRUCTURAL_SCREENED"},
            coverage_evidence_by_ticker=evidence,
        )
        assert first["rows_inserted"] == 1
        assert swept_tickers_for_band(conn, "micro_cap", "energy") == {"DIVG"}
        row = conn.execute(
            "SELECT coverage_authority_kind, coverage_authority_path, "
            "coverage_authority_sha256 FROM sector_run_loaded_sets WHERE ticker = 'DIVG'"
        ).fetchone()
        assert row["coverage_authority_kind"] == "ZERO_COST"
        assert len(row["coverage_authority_sha256"]) == 64

        second = record_zero_cost_coverage(
            conn,
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["DIVG"],
            candidate_dispositions={"DIVG": "STRUCTURAL_SCREENED"},
            coverage_evidence_by_ticker=evidence,
        )
        assert second["run_id"] == first["run_id"]
        assert second["rows_inserted"] == 0
        assert swept_tickers_for_band(conn, "micro_cap", "energy") == {"DIVG"}

        receipt_path = Path(row["coverage_authority_path"])
        alias_path = receipt_path.with_name("coverage_receipt_alias.json")
        alias_path.hardlink_to(receipt_path)
        assert swept_tickers_for_band(conn, "micro_cap", "energy") == set()
        alias_path.unlink()
        assert swept_tickers_for_band(conn, "micro_cap", "energy") == {"DIVG"}

        fabricated_gate = evaluate_structural_gate(
            "FAB",
            as_of_date="2026-07-23",
            price=0.50,
            market_cap_mm=1000.0,
            db_path=cfg.db_path,
        ).to_dict()
        record_zero_cost_coverage(
            conn,
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["FAB"],
            candidate_dispositions={"FAB": "STRUCTURAL_SCREENED"},
            coverage_evidence_by_ticker={
                "FAB": {
                    "disposition": "STRUCTURAL_SCREENED",
                    "effective_as_of": "2026-07-23",
                    "sector": "energy",
                    "market_cap_focus": "micro_cap",
                    "cap_classification": {
                        "cap_source": "classified",
                        "price_used": 100.0,
                        "market_cap_mm": 1000.0,
                    },
                    "structural_gate_result": fabricated_gate,
                }
            },
        )
        assert swept_tickers_for_band(conn, "micro_cap", "energy") == {"DIVG"}

        receipt_bytes = receipt_path.read_bytes()
        receipt_path.write_bytes(receipt_bytes.replace(b'"DIVG"', b'"FAKE"', 1))
        assert len(receipt_path.read_bytes()) == len(receipt_bytes)
        assert swept_tickers_for_band(conn, "micro_cap", "energy") == set()

    receipt_path.write_bytes(receipt_bytes)
    same_id_artifact = _valid_classic_artifact(first["run_id"])
    same_id_artifact.market_cap_focus = "micro_cap"
    same_id_artifact.candidate_selection.update(
        {
            "loaded_tickers": ["DIVG"],
            "cap_classifications": {"DIVG": classification},
            "structural_gate_results": {"DIVG": gate},
        }
    )
    persist_autonomous_sector_run(same_id_artifact)
    assert run_id_is_decision_eligible(first["run_id"]) is True

    with get_db() as conn:
        assert swept_tickers_for_band(conn, "micro_cap", "energy") == {"DIVG"}
        conn.execute(
            """
            UPDATE sector_run_loaded_sets
            SET coverage_authority_kind = NULL,
                coverage_authority_path = NULL,
                coverage_authority_sha256 = NULL,
                coverage_source_run_id = NULL
            WHERE run_id = ?
            """,
            (first["run_id"],),
        )
        assert swept_tickers_for_band(conn, "micro_cap", "energy") == set()


def test_unknown_cap_projection_chains_to_exact_authorized_primary_bytes(
    monkeypatch,
    tmp_path,
):
    _baseline_manifest(monkeypatch, tmp_path)
    init_db()
    from app.autonomous.structural_gate import evaluate_structural_gate
    from app.sector.scan import classify_tickers_for_market_cap

    run_id = "autonomous_sector_energy_20260723_unknown_cap"
    with get_db() as db_conn:
        db_conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form,
                accession
            )
            VALUES(
                'UNKNOWN', 2025, 'FY', '2025-12-31', ?,
                ?, 'USD_millions',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                '2026-02-15T00:00:00Z', '2026-02-15', '10-K',
                '0000000001-26-000001'
            )
            """,
            (("net_income", 12.3), ("cfo", -4.5)),
        )
    cfg = get_config()
    classification = classify_tickers_for_market_cap(
        tickers=["UNKNOWN"],
        as_of_date="2026-07-23",
        db_path=cfg.db_path,
        pipeline_version="v1",
        allow_live_market_data=False,
        cfg=cfg,
    )["UNKNOWN"].to_dict()
    gate = evaluate_structural_gate(
        "UNKNOWN",
        as_of_date="2026-07-23",
        price=classification["price_used"],
        market_cap_mm=classification["market_cap_mm"],
        db_path=cfg.db_path,
    ).to_dict()
    assert classification["cap_source"] == "unknown"
    assert gate["triggered_codes"] == ["EARNINGS_QUALITY_DIVERGENCE"]
    artifact = _valid_classic_artifact(run_id)
    artifact.market_cap_focus = "micro_cap"
    artifact.candidate_selection.update(
        {
            "sector": "energy",
            "market_cap_focus": "micro_cap",
            "coverage_campaign_id": "fresh-campaign",
            "loaded_tickers": ["UNKNOWN"],
            "structural_gate_results": {
                "UNKNOWN": gate,
            },
            "cap_classifications": {
                "UNKNOWN": classification,
            },
        }
    )
    paths = persist_autonomous_sector_run(artifact)
    with get_db() as conn:
        record_loaded_set(
            conn,
            run_id=run_id,
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["UNKNOWN"],
            pipeline_version="v1",
            candidate_dispositions={"UNKNOWN": "STRUCTURAL_SCREENED"},
            coverage_campaign_id="fresh-campaign",
        )
        projected = record_unknown_cap_cross_band_coverage(
            conn,
            run_id=run_id,
            sector="energy",
            source="sector_scan_db",
            primary_band="micro_cap",
            unknown_cap_tickers=["UNKNOWN"],
            candidate_dispositions={"UNKNOWN": "STRUCTURAL_SCREENED"},
            coverage_campaign_id="fresh-campaign",
        )
        assert projected["rows_inserted_or_upgraded"] == 4
        for band in V1_ATOMIC_BANDS:
            assert swept_tickers_for_band(
                conn,
                band,
                "energy",
                coverage_campaign_id="fresh-campaign",
            ) == {"UNKNOWN"}

        projection_run_id = f"{run_id}__unknown_cap__small_cap"
        same_id_artifact = _valid_classic_artifact(projection_run_id)
        same_id_artifact.market_cap_focus = "small_cap"
        same_id_artifact.candidate_selection.update(
            {
                "loaded_tickers": ["UNKNOWN"],
                "cap_classifications": {"UNKNOWN": classification},
                "structural_gate_results": {"UNKNOWN": gate},
            }
        )
        persist_autonomous_sector_run(same_id_artifact)
        assert run_id_is_decision_eligible(projection_run_id) is True
        conn.execute(
            """
            UPDATE sector_run_loaded_sets
            SET coverage_authority_kind = NULL,
                coverage_authority_path = NULL,
                coverage_authority_sha256 = NULL,
                coverage_source_run_id = NULL
            WHERE run_id = ?
            """,
            (projection_run_id,),
        )
        assert (
            swept_tickers_for_band(
                conn,
                "small_cap",
                "energy",
                coverage_campaign_id="fresh-campaign",
            )
            == set()
        )

        paths.artifact_json.write_bytes(
            paths.artifact_json.read_bytes().replace(b'"UNKNOWN"', b'"FORGED!"', 1)
        )
        for band in V1_ATOMIC_BANDS:
            assert (
                swept_tickers_for_band(
                    conn,
                    band,
                    "energy",
                    coverage_campaign_id="fresh-campaign",
                )
                == set()
            )


def test_classic_authorization_rejects_arbitrary_markdown_even_with_matching_hash(
    monkeypatch,
    tmp_path,
):
    _baseline_manifest(monkeypatch, tmp_path)
    run_id = "autonomous_sector_energy_20260723_report_binding"
    paths = persist_autonomous_sector_run(_valid_classic_artifact(run_id))
    paths.report_md.write_text("# FORGED BUY CLAIM\n", encoding="utf-8")
    authorization = json.loads(paths.authorization_json.read_text(encoding="utf-8"))
    for record in authorization["artifacts"]:
        if record["kind"] == "report_markdown":
            record["sha256"] = hashlib.sha256(paths.report_md.read_bytes()).hexdigest()
    paths.authorization_json.write_text(
        json.dumps(authorization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert run_id_is_decision_eligible(run_id) is False


def test_postwrite_authorization_rejects_midread_manifest_replacement(
    monkeypatch,
    tmp_path,
):
    reports = _baseline_manifest(monkeypatch, tmp_path)
    run_id = "autonomous_sector_energy_20260723_manifest_race"
    paths = persist_autonomous_sector_run(_valid_classic_artifact(run_id))
    original_current_sha256 = artifact_financial_audit._current_sha256
    manifest_path = reports.manifest_json.resolve()
    manifest_hash_calls = 0

    def replace_on_second_manifest_hash(path):
        nonlocal manifest_hash_calls
        if path.resolve() == manifest_path:
            manifest_hash_calls += 1
            if manifest_hash_calls == 2:
                reports.manifest_json.write_text("{invalid json", encoding="utf-8")
        return original_current_sha256(path)

    monkeypatch.setattr(
        artifact_financial_audit,
        "_current_sha256",
        replace_on_second_manifest_hash,
    )

    status, artifact_bytes = authorized_artifact_bytes(paths.artifact_json)

    assert manifest_hash_calls == 2
    assert status == UNAUDITED
    assert artifact_bytes is None


def test_two_step_decision_binding_rejects_bytes_that_do_not_match_stated_sha(
    monkeypatch,
):
    from app.outcomes import lineage as outcome_lineage

    monkeypatch.setattr(
        outcome_lineage,
        "authorized_run_artifact_binding",
        lambda *_args, **_kwargs: {
            "source_run_id": "run_sha_mismatch",
            "source_artifact_path": "/tmp/run_sha_mismatch.json",
            "source_artifact_sha256": "0" * 64,
        },
    )
    monkeypatch.setattr(
        outcome_lineage,
        "authorized_artifact_bytes",
        lambda *_args, **_kwargs: ("PASS", b'{"run_id":"run_sha_mismatch"}'),
    )

    assert (
        outcome_lineage.authorized_emitted_decision_binding(
            "run_sha_mismatch",
            "AAA",
        )
        is None
    )


def test_unlisted_newest_classic_row_suppresses_older_authorized_coverage(
    monkeypatch,
    tmp_path,
):
    _baseline_manifest(monkeypatch, tmp_path)
    authorized_run = "autonomous_sector_energy_20260723_authorized"
    persist_autonomous_sector_run(_valid_classic_artifact(authorized_run))

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    for run_id, loaded_at in (
        (authorized_run, "2026-07-23T12:03:00Z"),
        ("autonomous_sector_energy_20260723_unlisted", "2026-07-23T12:04:00Z"),
    ):
        record_loaded_set(
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

    assert (
        swept_tickers_for_band(
            conn,
            "mid_cap",
            "energy",
            pipeline_version="v1",
        )
        == set()
    )


def test_cross_date_digest_uses_exact_source_lineage_and_reader_suppresses_stale_claims(
    monkeypatch,
    tmp_path,
):
    _baseline_manifest(monkeypatch, tmp_path)
    run_id = "autonomous_sector_energy_20260723_digest_source"
    source_artifact = _valid_classic_artifact(
        run_id,
        company_packets=[_lineage_packet()],
    )
    source_artifact.relative_ranking = [
        {
            "ticker": "AAA",
            "company_autonomy_verdict": "WATCHLIST_ONLY",
            "company_autonomy_confidence": "MODERATE",
            "positioning_summary": "Exact-source digest fixture.",
        }
    ]
    init_db()
    paths = persist_autonomous_sector_run(source_artifact)
    cfg = get_config()
    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            confidence="MODERATE",
            conviction_source="company_autonomy",
            source_run_id=run_id,
            source_sector="energy",
            added_at="2026-07-23T12:05:00Z",
            thesis_text="Exact-source digest fixture.",
        ),
        db_path=cfg.db_path,
    )
    blocked_id = add_or_update(
        WatchlistEntry(
            ticker="BAD",
            status="DEPLOY_READY",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
            conviction_source="company_autonomy",
            source_run_id="autonomous_sector_energy_20260723_unlisted",
            source_sector="energy",
            added_at="2026-07-23T12:06:00Z",
            thesis_text="This unlisted source must never enter the digest.",
        ),
        db_path=cfg.db_path,
    )
    add_or_update(
        WatchlistEntry(
            ticker="FORGED",
            status="DEPLOY_READY",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
            conviction_source="company_autonomy",
            source_run_id=run_id,
            source_sector="energy",
            added_at="2026-07-23T12:06:30Z",
            thesis_text=(
                "A valid run ID must not authorize a ticker absent from the exact source bytes."
            ),
        ),
        db_path=cfg.db_path,
    )
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO dispositions(
                ticker, watchlist_id, kind, status, opened_at, opened_by,
                trigger_snapshot_json
            ) VALUES (
                'BAD', ?, 'AT_TARGET', 'OPEN',
                '2026-07-23T12:07:00Z', 'test', '{}'
            )
            """,
            (blocked_id,),
        )
    assert sync_at_target_dispositions(cfg.db_path)["opened"] == 0
    with get_db() as conn:
        assert company.watchlist_profile(conn, "FORGED") is None
    with pytest.raises(ValueError, match="not financial-integrity decision-eligible"):
        close_disposition(
            ticker="FORGED",
            status="ACTED",
            operator="test",
            reason_code="ENTERED_POSITION",
            rationale="must remain blocked",
            db_path=cfg.db_path,
            write_journal=False,
        )
    from app.watchlist import digest as digest_module

    original_watchlist_rows = digest_module._watchlist_rows
    original_compute_data_health = digest_module.compute_data_health
    watchlist_read_count = 0
    data_health_read_count = 0

    def counted_watchlist_rows(conn, *, manifest_path):
        nonlocal watchlist_read_count
        watchlist_read_count += 1
        return original_watchlist_rows(conn, manifest_path=manifest_path)

    def counted_compute_data_health(db_path=None, *, now=None, conn=None):
        nonlocal data_health_read_count
        data_health_read_count += 1
        assert conn is not None
        assert conn.in_transaction
        return original_compute_data_health(db_path, now=now, conn=conn)

    monkeypatch.setattr(digest_module, "_watchlist_rows", counted_watchlist_rows)
    monkeypatch.setattr(
        digest_module,
        "compute_data_health",
        counted_compute_data_health,
    )
    digest_path = cfg.outputs_dir / "digests" / "digest_2026-07-24.md"
    result = write_digest(
        output_path=digest_path,
        db_path=cfg.db_path,
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )
    assert watchlist_read_count == 1
    assert data_health_read_count == 1
    assert "This unlisted source must never enter the digest." not in result.markdown
    assert "| BAD |" not in result.markdown
    assert "| FORGED |" not in result.markdown
    assert result.lineage_path is not None
    lineage = json.loads(result.lineage_path.read_text(encoding="utf-8"))
    assert lineage["schema_version"] == "financial_integrity_digest_lineage_v4"
    assert set(lineage["rendered_state"]) == {
        "watchlist_rows",
        "latest_prices",
        "history_rows",
        "review_rows",
        "cheapness",
        "open_dispositions",
        "held_exit_rows",
        "data_health",
    }
    assert lineage["rendered_state"]["watchlist_rows"][0]["ticker"] == "AAA"
    assert lineage["rendered_state"]["data_health"]["blocking"] is False
    assert lineage["rendered_state"]["data_health"]["checks"][0]["name"] == "db_readable"
    assert lineage["rendered_state_sha256"]
    assert lineage["rendered_markdown_sha256"]
    assert lineage["manifest_path"]
    assert lineage["manifest_sha256"]
    assert len(lineage["rows"]) == 1
    lineage_row = lineage["rows"][0]
    assert lineage_row["source_artifact_path"] == str(paths.artifact_json.resolve())
    assert (
        lineage_row["source_artifact_sha256"]
        == hashlib.sha256(paths.artifact_json.read_bytes()).hexdigest()
    )
    assert lineage_row["source_run_id"] == run_id
    assert len(lineage_row["source_decision_fingerprint"]) == 64
    assert lineage_row["ticker"] == "AAA"
    assert lineage_row["decision_state"]["ticker"] == "AAA"
    assert lineage_row["decision_state"]["source_run_id"] == run_id
    assert lineage_row["decision_state"]["thesis_text"] == "Exact-source digest fixture."
    assert (
        lineage_row["decision_state_sha256"]
        == hashlib.sha256(
            json.dumps(
                lineage_row["decision_state"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )

    roots = {
        "runs": cfg.runs_dir / "autonomous_sector",
        "analyst": cfg.outputs_dir / "analyst_outputs",
        "scans": cfg.outputs_dir / "scans",
        "research": cfg.research_dir,
        "digests": cfg.outputs_dir / "digests",
    }
    current = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "current_audit",
        generated_at=datetime(2026, 7, 24, 12, 1, tzinfo=timezone.utc),
    )
    current_manifest = json.loads(current.manifest_json.read_text(encoding="utf-8"))
    current_records = {row["path"]: row for row in current_manifest["artifacts"]}
    assert current_records[str(digest_path.resolve())]["integrity_status"] == "PASS"
    monkeypatch.setenv("VOE_FINANCIAL_INTEGRITY_MANIFEST", str(current.manifest_json))
    assert authorized_artifact_bytes(digest_path) == (PASS, digest_path.read_bytes())
    assert today.latest_digest() is not None

    original_source_bytes = paths.artifact_json.read_bytes()
    paths.artifact_json.write_bytes(original_source_bytes + b"\n")
    assert authorized_artifact_bytes(digest_path) == (STALE_AUDIT, None)
    assert today.latest_digest() is None
    stale_render = reader.render_artifact(str(digest_path))
    assert stale_render["decision_eligible"] is False
    assert "Exact-source digest fixture" not in stale_render["html"]

    paths.artifact_json.unlink()
    assert authorized_artifact_bytes(digest_path) == (STALE_AUDIT, None)
    assert today.latest_digest() is None
    paths.artifact_json.write_bytes(original_source_bytes)
    assert authorized_artifact_bytes(digest_path) == (PASS, digest_path.read_bytes())
    assert today.latest_digest() is not None

    assert result.lineage_path is not None
    original_lineage_bytes = result.lineage_path.read_bytes()
    tampered_lineage = json.loads(original_lineage_bytes)
    tampered_lineage["rows"][0]["decision_state"]["status"] = "DEPLOY_READY"
    # Recompute the sidecar's own state hash.  The audit must still reject it
    # because the digest embeds an independently hash-bound rendered-state
    # record sourced from the same SQLite snapshot.
    tampered_lineage["rows"][0]["decision_state_sha256"] = hashlib.sha256(
        json.dumps(
            tampered_lineage["rows"][0]["decision_state"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    result.lineage_path.write_text(
        json.dumps(tampered_lineage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "tampered_lineage_audit",
        generated_at=datetime(2026, 7, 24, 12, 1, 30, tzinfo=timezone.utc),
    )
    tampered_manifest = json.loads(tampered.manifest_json.read_text(encoding="utf-8"))
    tampered_records = {row["path"]: row for row in tampered_manifest["artifacts"]}
    assert tampered_records[str(digest_path.resolve())]["integrity_status"] == "INVALID"
    assert any(
        finding["artifact_path"] == str(digest_path.resolve())
        and finding["invariant"] == "DIGEST_SOURCE_LINEAGE_INVALID"
        for finding in tampered_manifest["violations"]
    )
    result.lineage_path.write_bytes(original_lineage_bytes)

    original_digest_bytes = digest_path.read_bytes()
    tampered_markdown = original_digest_bytes.decode("utf-8").replace(
        "| AAA | ACTIVE |",
        "| AAA | DEPLOY_READY |",
        1,
    )
    assert tampered_markdown.encode("utf-8") != original_digest_bytes
    digest_path.write_text(tampered_markdown, encoding="utf-8")
    tampered_digest_sidecar = json.loads(original_lineage_bytes)
    # Rebinding only the outer digest hash is insufficient: the separately
    # pinned visible-Markdown hash must reject a changed financial status.
    tampered_digest_sidecar["digest_sha256"] = hashlib.sha256(
        tampered_markdown.encode("utf-8")
    ).hexdigest()
    result.lineage_path.write_text(
        json.dumps(tampered_digest_sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    visible_tamper = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "visible_digest_tamper_audit",
        generated_at=datetime(2026, 7, 24, 12, 1, 45, tzinfo=timezone.utc),
    )
    visible_tamper_manifest = json.loads(visible_tamper.manifest_json.read_text(encoding="utf-8"))
    visible_tamper_records = {row["path"]: row for row in visible_tamper_manifest["artifacts"]}
    assert visible_tamper_records[str(digest_path.resolve())]["integrity_status"] == "INVALID"
    assert any(
        finding["artifact_path"] == str(digest_path.resolve())
        and finding["invariant"] == "DIGEST_SOURCE_LINEAGE_INVALID"
        for finding in visible_tamper_manifest["violations"]
    )
    digest_path.write_bytes(original_digest_bytes)
    result.lineage_path.write_bytes(original_lineage_bytes)

    paths.artifact_json.write_text(
        paths.artifact_json.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    stale = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "stale_audit",
        generated_at=datetime(2026, 7, 24, 12, 2, tzinfo=timezone.utc),
    )
    stale_manifest = json.loads(stale.manifest_json.read_text(encoding="utf-8"))
    stale_records = {row["path"]: row for row in stale_manifest["artifacts"]}
    assert stale_records[str(digest_path.resolve())]["integrity_status"] == "INVALID"
    assert any(
        finding["artifact_path"] == str(digest_path.resolve())
        and finding["invariant"] == "INVALID_SOURCE_RUN_IN_CURRENT_REPORT"
        for finding in stale_manifest["violations"]
    )
    assert run_id not in stale_manifest["invalid_run_ids"]

    monkeypatch.setenv("VOE_FINANCIAL_INTEGRITY_MANIFEST", str(stale.manifest_json))
    rendered = reader.render_artifact(str(digest_path))
    assert rendered["integrity_status"] == "INVALID"
    assert rendered["decision_eligible"] is False
    assert "original artifact body and claims are suppressed" in rendered["html"]
    assert "Exact-source digest fixture" not in rendered["html"]


def test_watchlist_authorization_rejects_same_grade_mutable_claim_laundering(
    monkeypatch,
    tmp_path,
):
    _baseline_manifest(monkeypatch, tmp_path)
    run_id = "autonomous_sector_energy_20260723_watchlist_exact_state"
    source_artifact = _valid_classic_artifact(
        run_id,
        company_packets=[_lineage_packet()],
    )
    source_artifact.relative_ranking = [
        {
            "ticker": "AAA",
            "company_autonomy_verdict": "WATCHLIST_ONLY",
            "company_autonomy_confidence": "MODERATE",
            "positioning_summary": "Exact source thesis.",
        }
    ]
    init_db()
    persist_autonomous_sector_run(source_artifact)
    cfg = get_config()
    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            confidence="MODERATE",
            conviction_source="company_autonomy",
            source_run_id=run_id,
            source_sector="energy",
            added_at="2026-07-23T12:05:00Z",
            thesis_text="Exact source thesis.",
        ),
        db_path=cfg.db_path,
    )

    def current_row():
        with get_db() as conn:
            return conn.execute(
                "SELECT * FROM watchlist WHERE ticker = 'AAA' ORDER BY id DESC LIMIT 1"
            ).fetchone()

    assert watchlist_row_is_decision_eligible(current_row()) is True
    with get_db() as conn:
        conn.execute(
            "UPDATE watchlist SET thesis_text = 'Fabricated same-grade thesis' WHERE ticker = 'AAA'"
        )
    assert watchlist_row_is_decision_eligible(current_row()) is False

    with get_db() as conn:
        conn.execute(
            """
            UPDATE watchlist
            SET thesis_text = 'Exact source thesis.',
                buy_price_target = 999999.0
            WHERE ticker = 'AAA'
            """
        )
    assert watchlist_row_is_decision_eligible(current_row()) is False

    with get_db() as conn:
        conn.execute(
            """
            UPDATE watchlist
            SET buy_price_target = NULL,
                confidence = 'HIGH'
            WHERE ticker = 'AAA'
            """
        )
    assert watchlist_row_is_decision_eligible(current_row()) is False

    with get_db() as conn:
        conn.execute(
            """
            UPDATE watchlist
            SET confidence = 'MODERATE',
                thesis_text = 'Exact source thesis.'
            WHERE ticker = 'AAA'
            """
        )
    valid = current_row()
    assert watchlist_row_is_decision_eligible(valid) is True

    from app.web.readmodel.run_detail import _watchlist_links

    engine = sqlite3.connect(":memory:")
    engine.row_factory = sqlite3.Row
    with get_db() as conn:
        columns = [
            (str(row["name"]), str(row["type"] or "TEXT"))
            for row in conn.execute("PRAGMA table_info(watchlist)").fetchall()
        ]
    engine.execute(
        "CREATE TABLE watchlist ("
        + ", ".join(f'"{name}" {sql_type}' for name, sql_type in columns)
        + ")"
    )
    column_names = [name for name, _sql_type in columns]
    placeholders = ", ".join("?" for _ in column_names)
    column_sql = ", ".join(f'"{name}"' for name in column_names)
    engine.execute(
        f"INSERT INTO watchlist({column_sql}) VALUES({placeholders})",
        tuple(valid[name] for name in column_names),
    )
    forged = {name: valid[name] for name in column_names}
    forged["id"] = int(valid["id"]) + 1
    forged["conviction_grade"] = "ACTIONABLE"
    forged["confidence"] = "HIGH"
    forged["added_at"] = "2026-07-24T12:05:00Z"
    engine.execute(
        f"INSERT INTO watchlist({column_sql}) VALUES({placeholders})",
        tuple(forged[name] for name in column_names),
    )
    assert _watchlist_links(engine, run_id) == []


def test_typed_calibration_authorization_recomputes_claims_and_exact_markdown():
    from app.autonomous.artifact_financial_audit import (
        _discovery_calibration_payload_is_authorizable,
        _grade_status_calibration_payload_is_authorizable,
        _typed_report_bytes_are_authorizable,
    )
    from app.calibration.calibration_report import (
        _to_markdown as grade_markdown,
        recompute_grade_status_calibration_claims,
    )
    from app.discovery.calibration import (
        _to_markdown as discovery_markdown,
        recompute_discovery_calibration_claims,
    )

    grade_claims = recompute_grade_status_calibration_claims([])
    assert grade_claims is not None
    grade_payload = {
        "run_id": "calibration_report_2026-07-23",
        "report_family": "grade_status_calibration",
        "as_of_date": "2026-07-23",
        "generated_at": "2026-07-23T12:00:00Z",
        "source_run_ids": [],
        "source_outcomes": [],
        "headline_hit_metric": "excess_return_pct>0",
        "hit_metrics": [
            "excess_return_pct>0",
            "realized_return_pct>0",
            "reached_buy_target",
        ],
        "avoid_sign_inverted": True,
        **grade_claims,
    }
    assert _grade_status_calibration_payload_is_authorizable(
        grade_payload,
        expected_run_id=grade_payload["run_id"],
    )
    fabricated_grade = json.loads(json.dumps(grade_payload))
    fabricated_grade["overall"]["n"] = 999
    assert not _grade_status_calibration_payload_is_authorizable(
        fabricated_grade,
        expected_run_id=fabricated_grade["run_id"],
    )
    extra_grade = {**grade_payload, "fabricated_headline": {"n": 999}}
    assert not _grade_status_calibration_payload_is_authorizable(
        extra_grade,
        expected_run_id=extra_grade["run_id"],
    )
    assert _typed_report_bytes_are_authorizable(
        grade_payload,
        grade_markdown(grade_payload).encode("utf-8"),
        artifact_type="grade_status_calibration",
    )
    assert not _typed_report_bytes_are_authorizable(
        grade_payload,
        b"# unrelated but nonempty\n",
        artifact_type="grade_status_calibration",
    )

    discovery_claims = recompute_discovery_calibration_claims([], [])
    assert discovery_claims is not None
    discovery_payload = {
        "run_id": "calibration_20260723T120000Z",
        "report_family": "discovery_calibration",
        "target_run_id": None,
        "generated_at": "2026-07-23T12:00:00Z",
        "discovery_run_ids": [],
        "source_run_ids": [],
        "source_outcomes": [],
        "source_candidates": [],
        **discovery_claims,
    }
    assert _discovery_calibration_payload_is_authorizable(
        discovery_payload,
        expected_run_id=discovery_payload["run_id"],
    )
    fabricated_discovery = json.loads(json.dumps(discovery_payload))
    fabricated_discovery["threshold_suggestions"] = ["fabricated threshold change"]
    assert not _discovery_calibration_payload_is_authorizable(
        fabricated_discovery,
        expected_run_id=fabricated_discovery["run_id"],
    )
    extra_discovery = {**discovery_payload, "overall": {"n": 999, "hit_rate": 1.0}}
    assert not _discovery_calibration_payload_is_authorizable(
        extra_discovery,
        expected_run_id=extra_discovery["run_id"],
    )
    assert _typed_report_bytes_are_authorizable(
        discovery_payload,
        discovery_markdown(discovery_payload).encode("utf-8"),
        artifact_type="discovery_calibration",
    )
    assert not _typed_report_bytes_are_authorizable(
        discovery_payload,
        b"# unrelated but nonempty\n",
        artifact_type="discovery_calibration",
    )


def test_digest_pins_one_unusable_manifest_decision_for_the_whole_write(
    monkeypatch,
    tmp_path,
):
    _baseline_manifest(monkeypatch, tmp_path)
    init_db()
    from app.watchlist import digest as digest_module

    calls = 0

    def becomes_usable(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return calls > 1

    monkeypatch.setattr(
        digest_module,
        "financial_integrity_manifest_is_usable",
        becomes_usable,
    )
    output_path = tmp_path / "blocked_digest.md"
    result = write_digest(
        output_path=output_path,
        now=datetime(2026, 7, 24, 13, 0, tzinfo=timezone.utc),
    )

    assert calls == 1
    assert "BLOCKED: the canonical financial-integrity audit manifest" in result.markdown
    assert result.lineage_path is None


def test_digest_refuses_publish_when_pinned_manifest_changes_mid_render(
    monkeypatch,
    tmp_path,
):
    baseline = _baseline_manifest(monkeypatch, tmp_path)
    init_db()
    from app.watchlist import digest as digest_module

    original_materialize = digest_module._materialize_digest_snapshot

    def mutate_after_snapshot(*args, **kwargs):
        snapshot = original_materialize(*args, **kwargs)
        baseline.manifest_json.write_text(
            baseline.manifest_json.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        return snapshot

    monkeypatch.setattr(
        digest_module,
        "_materialize_digest_snapshot",
        mutate_after_snapshot,
    )
    output_path = tmp_path / "must_not_publish.md"
    with pytest.raises(RuntimeError, match="pinned financial-integrity manifest changed"):
        write_digest(
            output_path=output_path,
            now=datetime(2026, 7, 24, 13, 5, tzinfo=timezone.utc),
        )
    assert not output_path.exists()


def test_decision_digest_without_lineage_is_unrepairable(tmp_path):
    cfg = get_config()
    roots = {
        "runs": cfg.runs_dir / "autonomous_sector",
        "analyst": cfg.analyst_outputs_dir,
        "scans": cfg.outputs_dir / "scans",
        "research": cfg.research_dir,
        "digests": cfg.outputs_dir / "digests",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    digest_path = roots["digests"] / "digest_2026-07-24.md"
    digest_path.write_text(
        "| Ticker | Status |\n| --- | --- |\n| AAA | ACTIVE |\n",
        encoding="utf-8",
    )

    report = audit_artifact_tree(
        runs_root=roots["runs"],
        analyst_outputs_root=roots["analyst"],
        scans_root=roots["scans"],
        research_outputs_root=roots["research"],
        watchlist_report_roots=[roots["digests"]],
        analysis_dir=tmp_path / "audit",
        generated_at=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )
    manifest = json.loads(report.manifest_json.read_text(encoding="utf-8"))
    finding = next(
        row for row in manifest["violations"] if row["artifact_path"] == str(digest_path.resolve())
    )
    assert finding["invariant"] == "DIGEST_SOURCE_LINEAGE_MISSING"
    assert finding["repair_classification"] == "MISSING_PROVENANCE_UNREPAIRABLE"
