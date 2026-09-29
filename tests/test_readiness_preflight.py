from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.autonomous import readiness_preflight as rp


LINEAGE = {
    "run_id": "accepted-run-1",
    "as_of_date": "2026-07-17",
    "target_band": "large_and_mega",
    "input_fingerprint": "input-fingerprint",
    "resume_fingerprint": "input-fingerprint",
    "semantic_output_fingerprint": "semantic-fingerprint",
    "source_manifest_sha256": "source-manifest",
    "policy_manifest_sha256": "policy-manifest",
    "taxonomy_version": "taxonomy-v1",
    "taxonomy_hash": "taxonomy-hash",
    "external_industry_taxonomy_version": "external-v1",
    "external_industry_taxonomy_hash": "external-hash",
    "checkpoint_path": "checkpoint.json",
    "cohort_fingerprint": "cohort-fingerprint",
    "member_count": 2,
}


def _cap(ticker: str, issuer_key: str, security_key: str, cik: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "as_of_date": "2026-07-17",
        "market_cap_mm": 25_000.0,
        "cap_source": "accepted_census",
        "cap_band": "large_cap",
        "issuer_cik": cik,
        "issuer_primary_ticker": ticker,
        "issuer_listed_tickers": [ticker],
        "security_role": "PRIMARY",
        "is_secondary_class": False,
        "is_adr": False,
        "issuer_key": issuer_key,
        "security_key": security_key,
        "census_run_id": "accepted-run-1",
    }


@dataclass(frozen=True)
class _Lineage:
    def to_dict(self) -> dict[str, Any]:
        return dict(LINEAGE)


@dataclass(frozen=True)
class _Cap:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class _Member:
    ticker: str
    cik: str
    issuer_key: str
    security_key: str
    source_sector_label: str = "energy"
    canonical_sector: str = "energy"

    def to_cap_classification(self, *, lineage: Any) -> _Cap:
        assert lineage.to_dict() == LINEAGE
        return _Cap(_cap(self.ticker, self.issuer_key, self.security_key, self.cik))


class _Cohort:
    lineage = _Lineage()
    members = (
        _Member("AAA", "0000000001", "issuer-1", "security-1"),
        _Member("BBB", "0000000002", "issuer-2", "security-2"),
    )

    def members_for_sector(self, sector: str) -> tuple[_Member, ...]:
        return self.members if sector == "energy" else ()


class _Authority:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def load(self, *, db_path: str | Path, as_of_date: str) -> _Cohort:
        self.calls.append((Path(db_path), as_of_date))
        return _Cohort()


def _payload() -> dict[str, dict[str, Any]]:
    return {
        "energy": {
            "sector": "energy",
            "market_cap_focus": "large_and_mega",
            "source": "accepted_census",
            "selected_tickers": ["AAA", "BBB"],
            "loaded_tickers": ["AAA", "BBB"],
            "excluded_tickers": [],
            "membership_tickers": ["AAA", "BBB"],
            "execution_tickers": ["AAA"],
            "deferred_by_bound_tickers": ["BBB"],
            "execution_bound": 1,
            "membership_fingerprint": "518d1d0ec11d9c4a85cbc86de03771a3cc2b7c35eb40e6b70c08b0c736552490",
            "execution_fingerprint": "48e6f6a04a3a679b4e2bdb382af448541fff6a9ae6b13a5ea16146559adaa3f5",
            "execution_bound_frozen": True,
            "census_lineage": dict(LINEAGE),
            "cap_classifications": {
                "AAA": _cap("AAA", "issuer-1", "security-1", "0000000001"),
                "BBB": _cap("BBB", "issuer-2", "security-2", "0000000002"),
            },
            "execution_as_of_date": "2026-07-17",
            "request_fingerprint": "request-fingerprint",
            "execution_set_fingerprint": "1553680d350cb3f680c661c3f83528d9791f1346e9fd3f8594f92d3c16066178",
        }
    }


def _empty_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.close()


def test_readiness_preflight_preserves_exact_frozen_execution_and_zero_usage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "engine.db"
    _empty_db(db_path)
    authority = _Authority()
    observed: list[tuple[str, str]] = []

    def fake_candidate_readiness(_conn: Any, **kwargs: Any) -> dict[str, Any]:
        observed.append((kwargs["sector"], kwargs["ticker"]))
        return {
            "ticker": "AAA",
            "sector": "energy",
            "issuer_cik": "1",
            "issuer_key": "issuer-1",
            "security_key": "security-1",
            "readiness": "NEEDS_DATA",
            "missing_inputs": ["PRICE", "VALUATION"],
            "reason_codes": ["PRICE_QUOTE_NOT_FOUND_AS_OF_DATE", "V2_SCORECARD_MISSING"],
            "packet_materialized": False,
        }

    monkeypatch.setattr(rp, "_candidate_readiness", fake_candidate_readiness)
    result = rp.run_v2_readiness_preflight(
        sectors=["energy"],
        candidate_payloads=_payload(),
        as_of_date="2026-07-17",
        market_cap_focus="large_and_mega",
        execution_set_fingerprint="1553680d350cb3f680c661c3f83528d9791f1346e9fd3f8594f92d3c16066178",
        request_fingerprint="request-fingerprint",
        accepted_census_authority=authority,
        cfg=SimpleNamespace(db_path=db_path),
    )

    assert authority.calls == [(db_path, "2026-07-17")]
    assert observed == [("energy", "AAA")]
    assert result["status"] == "COMPLETED"
    assert result["readiness_status"] == "NEEDS_DATA"
    assert result["authority_validation_status"] == "PASSED"
    assert result["counts"] == {
        "sector_count": 1,
        "membership_candidates": 2,
        "execution_candidates": 1,
        "deferred_by_bound": 1,
        "excluded_candidates": 0,
        "ready": 0,
        "needs_data": 1,
        "incomplete": 0,
    }
    assert result["sector_results"]["energy"]["execution_tickers"] == ["AAA"]
    assert result["sector_results"]["energy"]["deferred_by_bound_tickers"] == [
        "BBB"
    ]
    assert result["actual_usage"] == {
        "model_calls": 0,
        "search_calls": 0,
        "network_calls": 0,
        "cost_microdollars": 0,
        "cost_usd": "0.000000",
    }
    assert result["database_access"] == {
        "mode": "ro",
        "query_only": True,
        "writes": 0,
    }
    assert result["packet_materialized_count"] == 0
    assert result["production_sector_scan_exercised"] is False


def test_readiness_preflight_rejects_execution_widening_before_queries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "engine.db"
    _empty_db(db_path)
    payload = _payload()
    payload["energy"]["execution_tickers"] = ["AAA", "BBB"]
    monkeypatch.setattr(
        rp,
        "_candidate_readiness",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("readiness query must not run after authority drift")
        ),
    )

    result = rp.run_v2_readiness_preflight(
        sectors=["energy"],
        candidate_payloads=payload,
        as_of_date="2026-07-17",
        market_cap_focus="large_and_mega",
        execution_set_fingerprint="1553680d350cb3f680c661c3f83528d9791f1346e9fd3f8594f92d3c16066178",
        request_fingerprint="request-fingerprint",
        accepted_census_authority=_Authority(),
        cfg=SimpleNamespace(db_path=db_path),
    )

    assert result["status"] == "INCOMPLETE"
    assert result["authority_validation_status"] == "FAILED"
    assert result["sector_results"] == {}
    assert result["validation_errors"] == [
        "ValueError:energy: execution/deferred ledgers do not partition membership"
    ]
    assert result["actual_usage"]["network_calls"] == 0


def test_readiness_connection_is_sqlite_query_only(tmp_path: Path) -> None:
    db_path = tmp_path / "engine.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sentinel(value TEXT)")
    conn.commit()
    conn.close()

    with rp._read_only_connection(db_path) as read_conn:
        assert read_conn.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            read_conn.execute("INSERT INTO sentinel(value) VALUES ('changed')")


def test_candidate_readiness_returns_terminal_needs_data_from_empty_local_cache(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "engine.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE companyfacts_facts (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            period_type TEXT NOT NULL,
            period_end TEXT NOT NULL,
            line_item TEXT NOT NULL,
            value REAL,
            units TEXT,
            source_url TEXT,
            fetched_at TEXT NOT NULL,
            filed_date TEXT,
            form TEXT,
            accession TEXT
        );
        CREATE TABLE filings (
            id INTEGER PRIMARY KEY,
            cik TEXT NOT NULL,
            ticker TEXT,
            accession TEXT NOT NULL,
            form_type TEXT NOT NULL,
            filing_date TEXT,
            local_path TEXT,
            status TEXT NOT NULL
        );
        CREATE TABLE parsed_filings (accession TEXT PRIMARY KEY, parsed_at TEXT NOT NULL);
        CREATE TABLE price_quotes (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            provider TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            price REAL,
            currency TEXT,
            source_url TEXT,
            status TEXT NOT NULL
        );
        CREATE TABLE valuations (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            method TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            outputs_json TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

    with rp._read_only_connection(db_path) as read_conn:
        result = rp._candidate_readiness(
            read_conn,
            ticker="AAA",
            identity={
                "issuer_cik": "0000000001",
                "issuer_key": "issuer-1",
                "security_key": "security-1",
            },
            sector="energy",
            as_of_date="2026-07-17",
        )

    assert result["readiness"] == "NEEDS_DATA"
    assert result["missing_inputs"] == [
        "FACTS",
        "FILING",
        "PARSING",
        "PRICE",
        "VALUATION",
    ]
    assert result["reason_codes"] == [
        "NORMALIZED_FACTS_REQUIRED_LINE_ITEMS_MISSING",
        "ANNUAL_FILING_CONTENT_NOT_LOCAL",
        "PRICE_QUOTE_NOT_FOUND_AS_OF_DATE",
        "V2_SCORECARD_MISSING",
    ]
    assert result["packet_materialized"] is False
