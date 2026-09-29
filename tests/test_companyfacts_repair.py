from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.autonomous import companyfacts_repair as repair


@pytest.fixture()
def facts_db(tmp_path: Path) -> Path:
    path = tmp_path / "facts.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE companyfacts_facts (
            ticker TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            period_type TEXT NOT NULL,
            period_end TEXT,
            line_item TEXT NOT NULL,
            value REAL,
            units TEXT,
            source_url TEXT,
            fetched_at TEXT,
            filed_date TEXT,
            form TEXT,
            accession TEXT,
            UNIQUE(ticker, fiscal_year, period_type, line_item)
        );
        CREATE TABLE companyfacts_vintages (
            ticker TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            period_type TEXT NOT NULL,
            period_end TEXT,
            line_item TEXT NOT NULL,
            value REAL,
            units TEXT,
            filed_date TEXT NOT NULL,
            form TEXT,
            accession TEXT,
            recorded_at TEXT,
            issuer_cik TEXT,
            source_url TEXT,
            UNIQUE(ticker, fiscal_year, period_type, line_item, filed_date, value)
        );
        """
    )
    conn.commit()
    conn.close()
    return path


def _ok_fetch(raw: dict) -> dict:
    return {
        "status": "OK",
        "reason_code": "FETCH_OK",
        "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json",
        "source_resolution": "companyfacts_fetch",
        "network_attempted": True,
        "attempts_made": 1,
        "companyfacts": raw,
    }


def test_repair_writes_only_facts_filed_by_fixed_asof(
    facts_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        repair,
        "normalize_annual_facts_from_raw",
        lambda *args, **kwargs: [
            {
                "fiscal_year": 2025,
                "period_type": "FY",
                "period_end": "2025-12-31",
                "line_item": "revenue",
                "value": 125.0,
                "units": "USD_millions",
                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json",
                "filed_date": "2026-02-15",
                "form": "10-K",
                "accession": "0001",
            },
            {
                "fiscal_year": 2026,
                "period_type": "FY",
                "period_end": "2026-12-31",
                "line_item": "revenue",
                "value": 150.0,
                "units": "USD_millions",
                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json",
                "filed_date": "2027-02-15",
                "form": "10-K",
                "accession": "0002",
            },
        ],
    )

    result = repair.repair_issuer_annual_companyfacts(
        "AL2",
        issuer_cik="42",
        as_of_date="2026-06-11",
        db_path=facts_db,
        storage_ticker="PRI2",
        fetcher=lambda *args, **kwargs: _ok_fetch({"facts": {"us-gaap": {}}}),
    )

    assert result["outcome"] == "FETCHED"
    assert result["terminal"] is False
    assert result["visible_rows"] == 1
    assert result["future_rows_rejected"] == 1
    assert result["rows_written"] == 1
    assert result["vintages_written"] == 1
    conn = sqlite3.connect(facts_db)
    rows = conn.execute(
        "SELECT ticker, fiscal_year, line_item, value, filed_date "
        "FROM companyfacts_facts"
    ).fetchall()
    vintages = conn.execute(
        "SELECT ticker, fiscal_year, line_item, value, filed_date, issuer_cik, source_url "
        "FROM companyfacts_vintages"
    ).fetchall()
    conn.close()
    assert rows == [("PRI2", 2025, "revenue", 125.0, "2026-02-15")]
    assert vintages == [
        (
            "PRI2",
            2025,
            "revenue",
            125.0,
            "2026-02-15",
            "0000000042",
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json",
        )
    ]


def test_repair_preserves_ambiguous_live_row_and_writes_historical_vintage(
    facts_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = sqlite3.connect(facts_db)
    conn.execute(
        """
        INSERT INTO companyfacts_facts(
            ticker, fiscal_year, period_type, period_end, line_item, value,
            units, source_url, fetched_at, filed_date, form, accession
        ) VALUES(
            'PRIMARY', 2025, 'FY', '2025-12-31', 'revenue', 200.0,
            'USD_millions',
            'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json',
            '2026-08-01T00:00:00+00:00', NULL, '10-K/A', 'ambiguous-live'
        )
        """
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        repair,
        "normalize_annual_facts_from_raw",
        lambda *args, **kwargs: [
            {
                "fiscal_year": 2025,
                "period_type": "FY",
                "period_end": "2025-12-31",
                "line_item": "revenue",
                "value": 100.0,
                "units": "USD_millions",
                "source_url": (
                    "https://data.sec.gov/api/xbrl/companyfacts/"
                    "CIK0000000042.json"
                ),
                "filed_date": "2026-03-01",
                "form": "10-K",
                "accession": "historical",
            }
        ],
    )

    result = repair.repair_issuer_annual_companyfacts(
        "PRIMARY",
        issuer_cik="42",
        as_of_date="2026-06-11",
        db_path=facts_db,
        fetcher=lambda *args, **kwargs: _ok_fetch(
            {"facts": {"us-gaap": {"Revenue": {"units": {"USD": []}}}}}
        ),
    )

    conn = sqlite3.connect(facts_db)
    current = conn.execute(
        "SELECT value, filed_date, accession FROM companyfacts_facts"
    ).fetchone()
    vintage = conn.execute(
        "SELECT value, filed_date, accession FROM companyfacts_vintages"
    ).fetchone()
    conn.close()
    assert result["rows_written"] == 0
    assert result["vintages_written"] == 1
    assert current == (200.0, None, "ambiguous-live")
    assert vintage == (100.0, "2026-03-01", "historical")


@pytest.mark.parametrize(
    ("raw", "expected_reason"),
    [
        (
            {"facts": {"ifrs-full": {"Revenue": {"units": {"USD": []}}}}},
            "IFRS_FACTS_UNSUPPORTED",
        ),
        (
            {
                "facts": {
                    "ifrs-full": {"Revenue": {"units": {"CAD": []}}},
                    "us-gaap": {},
                }
            },
            "IFRS_FACTS_UNSUPPORTED",
        ),
        (
            {"facts": {"us-gaap": {"Revenue": {"units": {"EUR": []}}}}},
            "NON_USD_FACTS_UNNORMALIZED",
        ),
        (
            {
                "facts": {
                    "us-gaap": {},
                    "dei-custom": {"Revenue": {"units": {"CAD": []}}},
                }
            },
            "NON_USD_FACTS_UNNORMALIZED",
        ),
    ],
)
def test_repair_explicitly_descopes_foreign_normalization(
    facts_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: dict,
    expected_reason: str,
) -> None:
    monkeypatch.setattr(
        repair,
        "normalize_annual_facts_from_raw",
        lambda *args, **kwargs: pytest.fail("unsupported foreign facts must not normalize"),
    )

    result = repair.repair_issuer_annual_companyfacts(
        "FRGN",
        issuer_cik="42",
        as_of_date="2026-06-11",
        db_path=facts_db,
        fetcher=lambda *args, **kwargs: _ok_fetch(raw),
    )

    assert result["outcome"] == "NEEDS_DATA"
    assert result["reason_code"] == expected_reason
    assert result["terminal"] is True


@pytest.mark.parametrize(
    ("reason", "terminal"),
    [
        ("FETCH_5XX", False),
        ("BUDGET_EXHAUSTED", False),
        ("FETCH_4XX", True),
    ],
)
def test_repair_distinguishes_transient_from_exhausted_sources(
    facts_db: Path, reason: str, terminal: bool
) -> None:
    result = repair.repair_issuer_annual_companyfacts(
        "MISS",
        issuer_cik="42",
        as_of_date="2026-06-11",
        db_path=facts_db,
        fetcher=lambda *args, **kwargs: {
            "status": "MISSING",
            "reason_code": reason,
            "reason_detail": "fixture",
            "companyfacts": None,
        },
    )

    assert result["outcome"] == "NEEDS_DATA"
    assert result["reason_code"] == f"COMPANYFACTS_{reason}"
    assert result["terminal"] is terminal


def test_repair_refuses_missing_issuer_identity(facts_db: Path) -> None:
    result = repair.repair_issuer_annual_companyfacts(
        "MISS",
        issuer_cik=None,
        as_of_date="2026-06-11",
        db_path=facts_db,
        fetcher=lambda *args, **kwargs: pytest.fail("no CIK must not fetch"),
    )

    assert result == {
        "outcome": "NEEDS_DATA",
        "reason_code": "ISSUER_CIK_UNRESOLVED",
        "terminal": False,
        "actions": [],
    }


def test_repair_rejects_companyfacts_payload_for_different_issuer(
    facts_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        repair,
        "normalize_annual_facts_from_raw",
        lambda *args, **kwargs: pytest.fail("mismatched issuer must not normalize"),
    )

    result = repair.repair_issuer_annual_companyfacts(
        "PRIMARY",
        issuer_cik="42",
        as_of_date="2026-06-11",
        db_path=facts_db,
        fetcher=lambda *args, **kwargs: _ok_fetch(
            {"cik": 999, "facts": {"us-gaap": {}}}
        ),
    )

    assert result["outcome"] == "NEEDS_DATA"
    assert result["reason_code"] == "COMPANYFACTS_ISSUER_MISMATCH"
    assert result["terminal"] is True
    assert result["requested_issuer_cik"] == "42"
    assert result["observed_issuer_ciks"] == ["999"]
