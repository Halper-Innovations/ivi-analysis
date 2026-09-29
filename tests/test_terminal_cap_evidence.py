from __future__ import annotations

import json
import sqlite3

import pytest

from app.autonomous.terminal_cap_evidence import (
    load_terminal_cap_evidence,
    terminal_cap_lookup_from_path,
)
from app.autonomous.cap_resolver import SecurityIdentity
from app.autonomous.cap_resolver import (
    CAP_SOURCE_TERMINAL_EXCHANGE,
    classify_market_cap_for_band_filter,
)


def test_postmortem_csv_shape_loads_direct_exchange_cap(tmp_path) -> None:
    path = tmp_path / "terminal_caps.csv"
    path.write_text(
        "ticker,cik,record_provenance,resolution_status,resolved_ticker,resolved_name,"
        "security_type,listing_status,resolved_market_cap_usd,is_current_common_equity,"
        "resolution_confidence,evidence_source,evidence_url,evidence_as_of,notes\n"
        "ZXAA,1234567,PERSISTED_RUN_ARTIFACT,CURRENT_LARGE_CAP_COMMON_EQUITY,ZXAA,"
        "Example Class A,COMMON_EQUITY,ACTIVE,12500000000,true,HIGH,"
        "Nasdaq downloadable stock screener,https://api.nasdaq.com/api/screener/stocks,"
        "2026-07-15,dated direct issuer cap\n"
        "ZXPF,7654321,PERSISTED_RUN_ARTIFACT,NON_COMMON_SECURITY,ZXPF,Example Preferred,"
        "PREFERRED_EQUITY,ACTIVE,9000000000,false,HIGH,"
        "Nasdaq downloadable stock screener,https://api.nasdaq.com/api/screener/stocks,"
        "2026-07-15,non-common security\n",
        encoding="utf-8",
    )

    ledger = load_terminal_cap_evidence(path)

    assert sorted(ledger) == ["ZXAA"]
    evidence = ledger["ZXAA"][0]
    assert evidence.market_cap_mm == 12_500.0
    assert evidence.source_kind == "EXCHANGE"
    assert evidence.source_url == "https://api.nasdaq.com/api/screener/stocks"
    assert evidence.as_of_date == "2026-07-15"
    assert evidence.confidence == "HIGH"
    assert evidence.issuer_cik == "0001234567"


def test_json_ledger_lookup_is_ticker_exact_and_preserves_search_binding(tmp_path) -> None:
    path = tmp_path / "terminal_caps.json"
    path.write_text(
        """{
          "rows": [
            {
              "ticker": "ZXBB",
              "issuer_cik": "2468101",
              "market_cap_mm": 18000,
              "market_cap_basis": "DIRECT_ISSUER_MARKET_CAP",
              "market_cap_units": "USD_millions",
              "source_kind": "SEARCH",
              "source_name": "search-backed exchange result",
              "source_url": "https://issuer.example/investors/market-data",
              "as_of_date": "2026-07-15",
              "confidence": "MEDIUM"
            }
          ]
        }""",
        encoding="utf-8",
    )
    lookup = terminal_cap_lookup_from_path(path)

    found = list(
        lookup(
            "ZXBB",
            "2026-07-16",
            SecurityIdentity(ticker="ZXBB", issuer_cik="0002468101"),
        )
    )
    missing = list(
        lookup(
            "ZXBC",
            "2026-07-16",
            SecurityIdentity(ticker="ZXBC", issuer_cik="0002468102"),
        )
    )

    assert len(found) == 1
    assert found[0].source_kind == "SEARCH"
    assert found[0].issuer_cik == "0002468101"
    assert missing == []


def test_ledger_rejects_cap_number_without_direct_issuer_basis(tmp_path) -> None:
    path = tmp_path / "unproven.json"
    path.write_text(
        """{
          "ticker": "ZXNO",
          "issuer_cik": "1357911",
          "market_cap_mm": 22000,
          "source_kind": "PROVIDER",
          "source_name": "derived quote cache",
          "source_url": "https://provider.example/ZXNO",
          "as_of_date": "2026-07-15",
          "confidence": "HIGH"
        }""",
        encoding="utf-8",
    )

    assert load_terminal_cap_evidence(path) == {}


@pytest.mark.parametrize(
    ("cap_fields", "expected_market_cap_mm"),
    [
        (
            {"market_cap_mm": 12_500.0, "market_cap_units": "USD_MILLIONS"},
            12_500.0,
        ),
        (
            {"market_cap_usd": 12_500_000_000.0, "market_cap_units": "USD"},
            12_500.0,
        ),
        (
            {"market_cap": 12_500_000_000.0, "market_cap_units": "DOLLARS"},
            12_500.0,
        ),
    ],
)
def test_direct_cap_field_unit_matrix_normalizes_unambiguous_values(
    tmp_path,
    cap_fields,
    expected_market_cap_mm,
) -> None:
    path = tmp_path / "valid-units.json"
    path.write_text(
        json.dumps(
            {
                "ticker": "UNIT",
                "issuer_cik": "1234567",
                "market_cap_basis": "DIRECT_ISSUER_MARKET_CAP",
                "source_kind": "SEARCH",
                "source_name": "dated issuer market data",
                "source_url": "https://issuer.example/UNIT",
                "as_of_date": "2026-07-15",
                "confidence": "HIGH",
                **cap_fields,
            }
        ),
        encoding="utf-8",
    )

    evidence = load_terminal_cap_evidence(path)["UNIT"][0]

    assert evidence.market_cap_mm == expected_market_cap_mm


@pytest.mark.parametrize(
    "cap_fields",
    [
        {"market_cap_mm": 12_500_000_000.0, "market_cap_units": "USD"},
        {"market_cap_usd": 12_500.0, "market_cap_units": "USD_MILLIONS"},
        {
            "market_cap_mm": 12_500.0,
            "market_cap_usd": 12_500_000_000.0,
            "market_cap_units": "USD_MILLIONS",
        },
    ],
)
def test_direct_cap_field_unit_matrix_rejects_mismatch_or_ambiguity(
    tmp_path,
    cap_fields,
) -> None:
    path = tmp_path / "invalid-units.json"
    path.write_text(
        json.dumps(
            {
                "ticker": "UNIT",
                "issuer_cik": "1234567",
                "market_cap_basis": "DIRECT_ISSUER_MARKET_CAP",
                "source_kind": "SEARCH",
                "source_name": "dated issuer market data",
                "source_url": "https://issuer.example/UNIT",
                "as_of_date": "2026-07-15",
                "confidence": "HIGH",
                **cap_fields,
            }
        ),
        encoding="utf-8",
    )

    assert load_terminal_cap_evidence(path) == {}


def test_local_authority_requires_controlled_schema_and_provenance(tmp_path) -> None:
    path = tmp_path / "untrusted-local.json"
    row = {
        "ticker": "LOCL",
        "issuer_cik": "1234567",
        "market_cap_mm": 12_500.0,
        "market_cap_basis": "DIRECT_ISSUER_MARKET_CAP",
        "market_cap_units": "USD_MILLIONS",
        "source_kind": "LOCAL_AUTHORITATIVE",
        "source_name": "claimed local cache",
        "source_url": "https://random-blog.example/LOCL",
        "as_of_date": "2026-07-15",
        "confidence": "HIGH",
    }
    path.write_text(json.dumps(row), encoding="utf-8")

    assert load_terminal_cap_evidence(path) == {}

    row["local_authority_schema"] = "VOE_TERMINAL_CAP_EVIDENCE_V1"
    row["record_provenance"] = "PERSISTED_RUN_ARTIFACT"
    path.write_text(json.dumps(row), encoding="utf-8")
    evidence = load_terminal_cap_evidence(path)["LOCL"][0]

    assert evidence.source_kind == "LOCAL_AUTHORITATIVE"
    assert evidence.local_authority_schema == "VOE_TERMINAL_CAP_EVIDENCE_V1"
    assert evidence.record_provenance == "PERSISTED_RUN_ARTIFACT"


def test_terminal_ledger_requires_cik_and_lookup_binds_resolved_issuer(tmp_path) -> None:
    path = tmp_path / "issuer-binding.json"
    row = {
        "ticker": "BIND",
        "market_cap_mm": 12_500.0,
        "market_cap_basis": "DIRECT_ISSUER_MARKET_CAP",
        "market_cap_units": "USD_MILLIONS",
        "source_kind": "SEARCH",
        "source_name": "dated issuer market data",
        "source_url": "https://issuer.example/BIND",
        "as_of_date": "2026-07-15",
        "confidence": "HIGH",
    }
    path.write_text(json.dumps(row), encoding="utf-8")

    assert load_terminal_cap_evidence(path) == {}

    row["issuer_cik"] = "1234567"
    path.write_text(json.dumps(row), encoding="utf-8")
    lookup = terminal_cap_lookup_from_path(path)

    assert list(lookup("BIND", "2026-07-16", SecurityIdentity(ticker="BIND"))) == []
    assert (
        list(
            lookup(
                "BIND",
                "2026-07-16",
                SecurityIdentity(ticker="BIND", issuer_cik="0007654321"),
            )
        )
        == []
    )
    assert (
        len(
            list(
                lookup(
                    "BIND",
                    "2026-07-16",
                    SecurityIdentity(ticker="BIND", issuer_cik="0001234567"),
                )
            )
        )
        == 1
    )


def test_postmortem_like_row_requires_cap_resolved_status_and_full_evidence(
    tmp_path,
) -> None:
    path = tmp_path / "incomplete.csv"
    path.write_text(
        "record_provenance,resolution_status,resolved_ticker,resolved_market_cap_usd,"
        "security_type,listing_status,is_current_common_equity,resolution_confidence,"
        "evidence_source,evidence_url,evidence_as_of\n"
        "PERSISTED_RUN_ARTIFACT,UNKNOWN_CAP_EXCLUDED,ZXNO,22000000000,COMMON_EQUITY,"
        "ACTIVE,true,HIGH,Some source,https://provider.example/ZXNO,2026-07-15\n",
        encoding="utf-8",
    )

    assert load_terminal_cap_evidence(path) == {}


def test_prevalidated_ledger_exchange_cap_precedes_stale_shares_derivation(
    monkeypatch, tmp_path
) -> None:
    ledger_path = tmp_path / "direct.json"
    ledger_path.write_text(
        """{
          "ticker": "ZXLD",
          "issuer_cik": "1234567",
          "market_cap_mm": 21000,
          "market_cap_basis": "DIRECT_ISSUER_MARKET_CAP",
          "market_cap_units": "USD_millions",
          "source_kind": "EXCHANGE",
          "source_name": "dated exchange market cap",
          "source_url": "https://www.nasdaq.com/market-activity/stocks/ZXLD",
          "as_of_date": "2026-07-15",
          "confidence": "HIGH"
        }""",
        encoding="utf-8",
    )
    db_path = tmp_path / "engine.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE companies (ticker TEXT, cik TEXT)")
        conn.execute("INSERT INTO companies VALUES ('ZXLD', '1234567')")
        conn.execute(
            """
            CREATE TABLE companyfacts_facts (
                ticker TEXT, fiscal_year INTEGER, period_type TEXT,
                period_end TEXT, line_item TEXT, value REAL, source_url TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO companyfacts_facts VALUES "
            "('ZXLD', 2025, 'FY', '2025-12-31', 'shares_outstanding', 100.0, NULL)"
        )
    monkeypatch.setattr(
        "app.valuation.shares.resolve_market_cap_from_price_asof",
        lambda **kwargs: (None, {}),
    )

    result = classify_market_cap_for_band_filter(
        "ZXLD",
        as_of_date="2026-07-15",
        current_price=10.0,
        db_path=db_path,
        terminal_cap_lookup=terminal_cap_lookup_from_path(ledger_path),
        pipeline_version="v2",
    )

    assert result.market_cap_mm == 21_000.0
    assert result.cap_source == CAP_SOURCE_TERMINAL_EXCHANGE
