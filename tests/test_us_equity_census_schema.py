"""Schema contract for the registry-first U.S. equity census."""

from __future__ import annotations

import sqlite3

import pytest

from app.db import init_db


_NOW = "2026-07-17T19:00:00+00:00"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn=conn)
    conn.commit()
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(row["name"] for row in conn.execute(f"PRAGMA table_info({table})"))


def test_census_schema_exposes_exact_normalized_columns() -> None:
    conn = _connect()
    try:
        assert _columns(conn, "us_equity_census_runs") == (
            "run_id",
            "as_of_date",
            "target_band",
            "status",
            "current_stage",
            "acceptance_status",
            "registry_source_name",
            "registry_source_url",
            "registry_snapshot_path",
            "registry_snapshot_sha256",
            "registry_retrieved_at",
            "source_snapshot_count",
            "source_manifest_json",
            "crosscheck_sources_json",
            "input_fingerprint",
            "resume_fingerprint",
            "checkpoint_path",
            "discovered_security_count",
            "deduplicated_issuer_count",
            "us_domiciled_issuer_count",
            "foreign_issuer_count",
            "unresolved_identity_count",
            "unresolved_cap_count",
            "unresolved_sector_count",
            "status_detail_json",
            "created_at",
            "started_at",
            "updated_at",
            "completed_at",
            "interrupted_at",
        )
        assert _columns(conn, "us_equity_census_securities") == (
            "id",
            "run_id",
            "security_key",
            "source_registry",
            "source_security_id",
            "ticker",
            "listed_name",
            "exchange_name",
            "exchange_mic",
            "listing_status",
            "listing_status_as_of_date",
            "listing_source_url",
            "listing_retrieved_at",
            "issuer_key",
            "issuer_relationship_type",
            "related_security_key",
            "share_class",
            "is_primary_security",
            "is_secondary_class",
            "is_duplicate_listing",
            "security_type",
            "security_type_status",
            "is_common_equity",
            "is_adr",
            "adr_ratio",
            "adr_ratio_as_of_date",
            "adr_ratio_source_url",
            "share_class_ratio",
            "share_class_ratio_as_of_date",
            "share_class_ratio_source_url",
            "price_usd",
            "price_as_of_date",
            "price_source_provider",
            "price_source_url",
            "price_retrieved_at",
            "price_confidence",
            "identity_status",
            "identity_source",
            "identity_source_url",
            "identity_as_of_date",
            "identity_retrieved_at",
            "identity_confidence",
            "last_completed_stage",
            "next_stage",
            "processing_status",
            "terminal_disposition",
            "terminal_reason_code",
            "terminal_detail_json",
            "terminal_at",
            "source_snapshot_ref",
            "raw_source_json",
            "provenance_json",
            "created_at",
            "updated_at",
        )
        assert _columns(conn, "us_equity_census_issuers") == (
            "id",
            "run_id",
            "issuer_key",
            "cik",
            "legal_name",
            "display_name",
            "identity_status",
            "identity_source_provider",
            "identity_source_url",
            "identity_as_of_date",
            "identity_retrieved_at",
            "identity_confidence",
            "issuer_type",
            "operating_structure",
            "operating_status",
            "is_operating_company",
            "membership_status",
            "scope_status",
            "scope_reason_code",
            "domicile_country_code",
            "domicile_jurisdiction",
            "is_us_domiciled",
            "is_us_listed_foreign_issuer",
            "filer_status",
            "domicile_source_provider",
            "domicile_source_url",
            "domicile_as_of_date",
            "domicile_retrieved_at",
            "domicile_confidence",
            "primary_security_key",
            "primary_ticker",
            "primary_exchange_name",
            "primary_exchange_mic",
            "primary_selection_status",
            "primary_selection_method",
            "primary_selection_source_url",
            "primary_selection_as_of_date",
            "primary_selection_retrieved_at",
            "primary_selection_confidence",
            "listed_security_count",
            "market_cap_status",
            "market_cap_usd",
            "market_cap_currency",
            "market_cap_as_of_date",
            "market_cap_method",
            "market_cap_source_provider",
            "market_cap_source_url",
            "market_cap_retrieved_at",
            "market_cap_confidence",
            "cap_security_key",
            "cap_price_usd",
            "cap_price_as_of_date",
            "cap_price_source_provider",
            "cap_price_source_url",
            "shares_outstanding",
            "shares_as_of_date",
            "shares_source_provider",
            "shares_source_url",
            "shares_retrieved_at",
            "shares_confidence",
            "cap_ratio_adjustment",
            "cap_ratio_source_url",
            "cap_derivation_json",
            "cap_band",
            "cap_band_status",
            "canonical_sector",
            "source_sector_label",
            "source_sector_system",
            "sector_status",
            "sector_mapping_method",
            "sector_mapping_version",
            "sector_source_url",
            "sector_as_of_date",
            "sector_confidence",
            "sector_provenance_json",
            "facts_status",
            "facts_as_of_date",
            "facts_source_provider",
            "facts_source_url",
            "facts_detail_json",
            "filings_status",
            "latest_annual_filing_date",
            "latest_annual_filing_accession",
            "filings_source_provider",
            "filings_source_url",
            "filings_detail_json",
            "packet_status",
            "packet_as_of_date",
            "packet_path",
            "packet_sha256",
            "packet_detail_json",
            "last_completed_stage",
            "next_stage",
            "processing_status",
            "terminal_disposition",
            "terminal_reason_code",
            "terminal_detail_json",
            "terminal_at",
            "provenance_json",
            "created_at",
            "updated_at",
        )
        assert _columns(conn, "us_equity_census_attempts") == (
            "id",
            "run_id",
            "attempt_id",
            "attempt_number",
            "stage",
            "entity_kind",
            "security_key",
            "issuer_key",
            "status",
            "resume_from_attempt_id",
            "input_fingerprint",
            "checkpoint_path",
            "worker_id",
            "source_provider",
            "source_url",
            "retrieved_at",
            "input_json",
            "output_json",
            "error_code",
            "error_message",
            "detail_json",
            "started_at",
            "heartbeat_at",
            "finished_at",
            "created_at",
            "updated_at",
        )
    finally:
        conn.close()


def test_census_schema_has_exact_indexes_and_foreign_key_targets() -> None:
    conn = _connect()
    try:
        indexes = tuple(
            row["name"]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'index' AND name LIKE 'idx_us_equity_census_%'
                ORDER BY name
                """
            )
        )
        assert indexes == (
            "idx_us_equity_census_attempts_issuer",
            "idx_us_equity_census_attempts_security",
            "idx_us_equity_census_attempts_status",
            "idx_us_equity_census_issuers_cap",
            "idx_us_equity_census_issuers_cik",
            "idx_us_equity_census_issuers_scope",
            "idx_us_equity_census_issuers_sector",
            "idx_us_equity_census_issuers_terminal",
            "idx_us_equity_census_runs_acceptance",
            "idx_us_equity_census_runs_asof",
            "idx_us_equity_census_securities_issuer",
            "idx_us_equity_census_securities_listing",
            "idx_us_equity_census_securities_terminal",
            "idx_us_equity_census_securities_ticker",
        )

        expected_targets = {
            "us_equity_census_runs": set(),
            "us_equity_census_securities": {
                "us_equity_census_runs",
                "us_equity_census_issuers",
                "us_equity_census_securities",
            },
            "us_equity_census_issuers": {
                "us_equity_census_runs",
                "us_equity_census_securities",
            },
            "us_equity_census_attempts": {
                "us_equity_census_runs",
                "us_equity_census_securities",
                "us_equity_census_issuers",
                "us_equity_census_attempts",
            },
        }
        for table, targets in expected_targets.items():
            actual = {row["table"] for row in conn.execute(f"PRAGMA foreign_key_list({table})")}
            assert actual == targets
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_census_schema_is_idempotent_and_allows_partial_resumable_rows() -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO us_equity_census_runs(
                run_id, as_of_date, target_band, status, current_stage,
                registry_source_name, registry_snapshot_sha256,
                created_at, started_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "census_20260717",
                "2026-07-17",
                "large_and_mega",
                "RUNNING",
                "MEMBERSHIP",
                "sec_exchange_registry",
                "abc123",
                _NOW,
                _NOW,
                _NOW,
            ),
        )
        conn.execute(
            """
            INSERT INTO us_equity_census_securities(
                run_id, security_key, source_registry, source_security_id,
                ticker, exchange_mic, listing_status, source_snapshot_ref,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "census_20260717",
                "SEC:0000000001:XNAS:AAA",
                "sec_exchange_registry",
                "0000000001:XNAS:AAA",
                "AAA",
                "XNAS",
                "ACTIVE",
                "snapshot:abc123",
                _NOW,
                _NOW,
            ),
        )
        conn.execute(
            """
            INSERT INTO us_equity_census_issuers(
                run_id, issuer_key, cik, legal_name, identity_status,
                membership_status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "census_20260717",
                "CIK:0000000001",
                "0000000001",
                "Example Operating Company",
                "RESOLVED",
                "DISCOVERED",
                _NOW,
                _NOW,
            ),
        )
        conn.execute(
            """
            UPDATE us_equity_census_securities
            SET issuer_key = ?, issuer_relationship_type = ?, updated_at = ?
            WHERE run_id = ? AND security_key = ?
            """,
            (
                "CIK:0000000001",
                "ISSUED_BY",
                _NOW,
                "census_20260717",
                "SEC:0000000001:XNAS:AAA",
            ),
        )
        conn.execute(
            """
            UPDATE us_equity_census_issuers
            SET primary_security_key = ?, primary_ticker = ?, primary_exchange_mic = ?,
                primary_selection_status = ?, updated_at = ?
            WHERE run_id = ? AND issuer_key = ?
            """,
            (
                "SEC:0000000001:XNAS:AAA",
                "AAA",
                "XNAS",
                "SELECTED",
                _NOW,
                "census_20260717",
                "CIK:0000000001",
            ),
        )
        conn.executemany(
            """
            INSERT INTO us_equity_census_attempts(
                run_id, attempt_id, attempt_number, stage, entity_kind,
                security_key, issuer_key, status, checkpoint_path,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    "census_20260717",
                    "attempt-membership-1",
                    1,
                    "MEMBERSHIP",
                    "RUN",
                    None,
                    None,
                    "COMPLETED",
                    "/tmp/census_20260717.json",
                    _NOW,
                    _NOW,
                ),
                (
                    "census_20260717",
                    "attempt-identity-1",
                    2,
                    "IDENTITY",
                    "SECURITY",
                    "SEC:0000000001:XNAS:AAA",
                    None,
                    "COMPLETED",
                    "/tmp/census_20260717.json",
                    _NOW,
                    _NOW,
                ),
                (
                    "census_20260717",
                    "attempt-cap-1",
                    3,
                    "MARKET_CAP_RESOLUTION",
                    "ISSUER",
                    None,
                    "CIK:0000000001",
                    "NEEDS_DATA",
                    "/tmp/census_20260717.json",
                    _NOW,
                    _NOW,
                ),
            ),
        )
        conn.commit()

        init_db(conn=conn)
        conn.commit()

        security = conn.execute(
            """
            SELECT issuer_key, processing_status, terminal_disposition,
                   raw_source_json, provenance_json
            FROM us_equity_census_securities
            WHERE run_id = 'census_20260717'
            """
        ).fetchone()
        assert dict(security) == {
            "issuer_key": "CIK:0000000001",
            "processing_status": "PENDING",
            "terminal_disposition": None,
            "raw_source_json": "{}",
            "provenance_json": "{}",
        }
        issuer = conn.execute(
            """
            SELECT primary_security_key, market_cap_usd, canonical_sector,
                   processing_status, terminal_disposition
            FROM us_equity_census_issuers
            WHERE run_id = 'census_20260717'
            """
        ).fetchone()
        assert dict(issuer) == {
            "primary_security_key": "SEC:0000000001:XNAS:AAA",
            "market_cap_usd": None,
            "canonical_sector": None,
            "processing_status": "PENDING",
            "terminal_disposition": None,
        }
        assert conn.execute("SELECT COUNT(*) FROM us_equity_census_runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM us_equity_census_securities").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM us_equity_census_issuers").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM us_equity_census_attempts").fetchone()[0] == 3
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []

        conn.execute("DELETE FROM us_equity_census_runs WHERE run_id = 'census_20260717'")
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM us_equity_census_securities").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM us_equity_census_issuers").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM us_equity_census_attempts").fetchone()[0] == 0
    finally:
        conn.close()


def test_census_schema_enforces_run_scope_and_literal_checks() -> None:
    conn = _connect()
    try:
        for run_id in ("run-a", "run-b"):
            conn.execute(
                """
                INSERT INTO us_equity_census_runs(run_id, as_of_date, created_at, updated_at)
                VALUES (?, '2026-07-17', ?, ?)
                """,
                (run_id, _NOW, _NOW),
            )
            conn.execute(
                """
                INSERT INTO us_equity_census_securities(
                    run_id, security_key, ticker, created_at, updated_at
                ) VALUES (?, 'SEC:XNAS:AAA', 'AAA', ?, ?)
                """,
                (run_id, _NOW, _NOW),
            )
            conn.execute(
                """
                INSERT INTO us_equity_census_issuers(
                    run_id, issuer_key, cik, created_at, updated_at
                ) VALUES (?, 'CIK:0000000001', '0000000001', ?, ?)
                """,
                (run_id, _NOW, _NOW),
            )
        conn.commit()

        assert conn.execute("SELECT COUNT(*) FROM us_equity_census_securities").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM us_equity_census_issuers").fetchone()[0] == 2

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO us_equity_census_securities(
                    run_id, security_key, ticker, created_at, updated_at
                ) VALUES ('run-a', 'SEC:XNAS:AAA', 'AAA2', ?, ?)
                """,
                (_NOW, _NOW),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO us_equity_census_issuers(
                    run_id, issuer_key, cik, created_at, updated_at
                ) VALUES ('run-a', 'CIK:DUPLICATE', '0000000001', ?, ?)
                """,
                (_NOW, _NOW),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO us_equity_census_issuers(
                    run_id, issuer_key, market_cap_usd, created_at, updated_at
                ) VALUES ('run-a', 'CIK:NEGATIVE-CAP', -1, ?, ?)
                """,
                (_NOW, _NOW),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO us_equity_census_attempts(
                    run_id, attempt_id, attempt_number, stage, entity_kind,
                    security_key, status, created_at, updated_at
                ) VALUES (
                    'run-a', 'bad-run-target', 1, 'MEMBERSHIP', 'RUN',
                    'SEC:XNAS:AAA', 'PENDING', ?, ?
                )
                """,
                (_NOW, _NOW),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO us_equity_census_runs(
                    run_id, as_of_date, target_band, created_at, updated_at
                ) VALUES ('bad-band', '2026-07-17', 'nano_cap', ?, ?)
                """,
                (_NOW, _NOW),
            )
    finally:
        conn.close()
