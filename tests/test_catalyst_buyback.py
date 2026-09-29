from __future__ import annotations

import sqlite3


def _make_conn() -> sqlite3.Connection:
    """An in-memory sqlite mirroring the companyfacts_facts columns used here."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE companyfacts_facts (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            period_type TEXT NOT NULL DEFAULT 'FY',
            period_end TEXT NOT NULL,
            line_item TEXT NOT NULL,
            value REAL,
            units TEXT,
            source_url TEXT,
            filed_date TEXT,
            accession TEXT,
            fetched_at TEXT NOT NULL
        )
        """
    )
    return conn


def _insert(
    conn,
    ticker,
    fiscal_year,
    line_item,
    value,
    period_type="FY",
    *,
    units="USD_millions",
    source_url="https://example.test/companyfacts",
    filed_date="AUTO",
    accession="AUTO",
):
    if filed_date == "AUTO":
        filed_date = f"{fiscal_year + 1}-02-15"
    if accession == "AUTO":
        accession = f"0000000000-{str(fiscal_year)[-2:]}-000001"
    conn.execute(
        """
        INSERT INTO companyfacts_facts
            (ticker, fiscal_year, period_type, period_end, line_item, value,
             units, source_url, filed_date, accession, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticker,
            fiscal_year,
            period_type,
            f"{fiscal_year}-12-31",
            line_item,
            value,
            units,
            source_url,
            filed_date,
            accession,
            "2024-01-01T00:00:00Z",
        ),
    )
    conn.commit()


def test_buyback_accelerating():
    from app.catalyst.buyback import detect_buyback_signal

    conn = _make_conn()
    _insert(conn, "ACEL", 2023, "share_repurchases_amount", 200.0)
    _insert(conn, "ACEL", 2022, "share_repurchases_amount", 100.0)

    signal = detect_buyback_signal(conn, "ACEL", "2024-06-01")

    assert signal["label"] == "ACCELERATING"
    assert signal["latest_fy"] == 200.0
    assert signal["prior_fy"] == 100.0
    assert signal["fiscal_year"] == 2023
    assert signal["unit"] == "USD_millions"
    assert signal["status"] == "OK"
    assert signal["reason_codes"] == []
    assert signal["source_lineage"][0] == {
        "line_item": "share_repurchases_amount",
        "fiscal_year": 2023,
        "reported_value": 200.0,
        "normalized_value": 200.0,
        "unit": "USD_millions",
        "period_end": "2023-12-31",
        "filed_date": "2024-02-15",
        "accession": "0000000000-23-000001",
        "source_url": "https://example.test/companyfacts",
        "source_reference": (
            "https://example.test/companyfacts"
            "#line_item=share_repurchases_amount&accession=0000000000-23-000001"
        ),
        "role": "latest_fy",
    }


def test_buyback_steady():
    from app.catalyst.buyback import detect_buyback_signal

    conn = _make_conn()
    _insert(conn, "ACEL", 2023, "share_repurchases_amount", 105.0)
    _insert(conn, "ACEL", 2022, "share_repurchases_amount", 100.0)

    signal = detect_buyback_signal(conn, "ACEL", "2024-06-01")

    assert signal["label"] == "STEADY"
    assert signal["latest_fy"] == 105.0
    assert signal["prior_fy"] == 100.0


def test_buyback_none_when_latest_zero():
    from app.catalyst.buyback import detect_buyback_signal

    conn = _make_conn()
    _insert(conn, "ACEL", 2023, "share_repurchases_amount", 0)
    _insert(conn, "ACEL", 2022, "share_repurchases_amount", 100.0)

    signal = detect_buyback_signal(conn, "ACEL", "2024-06-01")

    assert signal["label"] == "NONE"


def test_buyback_negative_sign_issuer_accelerating():
    from app.catalyst.buyback import detect_buyback_signal

    conn = _make_conn()
    _insert(conn, "ACEL", 2023, "share_repurchases_amount", -200.0)
    _insert(conn, "ACEL", 2022, "share_repurchases_amount", -100.0)

    signal = detect_buyback_signal(conn, "ACEL", "2024-06-01")

    assert signal["label"] == "ACCELERATING"
    assert signal["latest_fy"] == 200.0
    assert signal["prior_fy"] == 100.0


def test_buyback_none_when_no_rows():
    from app.catalyst.buyback import detect_buyback_signal

    conn = _make_conn()

    signal = detect_buyback_signal(conn, "ACEL", "2024-06-01")

    assert signal["label"] == "NONE"
    assert signal["latest_fy"] is None
    assert signal["status"] == "NO_DATA"
    assert signal["reason_codes"] == ["NO_VISIBLE_FILED_ASOF_FACTS"]


def test_detect_buyback_excludes_fiscal_years_after_as_of():
    # A FY whose period_end is AFTER the as-of date must not be visible.
    from app.catalyst.buyback import detect_buyback_signal

    conn = _make_conn()
    # Insert FY2022 and FY2023 normally; FY2024 is "future" vs the as-of date.
    _insert(
        conn,
        "BBK",
        2022,
        "share_repurchases_amount",
        100.0,
        filed_date="2023-02-15",
    )
    _insert(
        conn,
        "BBK",
        2023,
        "share_repurchases_amount",
        110.0,
        filed_date="2023-12-31",
    )
    _insert(
        conn,
        "BBK",
        2024,
        "share_repurchases_amount",
        100_000.0,
        filed_date="2025-02-15",
    )

    signal = detect_buyback_signal(conn, "BBK", "2023-12-31")

    assert signal["fiscal_year"] == 2023
    assert signal["latest_fy"] == 110.0
    assert signal["prior_fy"] == 100.0
    assert signal["label"] == "STEADY"


def test_detect_buyback_excludes_post_asof_filing_even_for_ended_period():
    from app.catalyst.buyback import detect_buyback_signal

    conn = _make_conn()
    _insert(
        conn,
        "PIT",
        2023,
        "share_repurchases_amount",
        100.0,
        filed_date="2024-02-15",
        accession="0000000000-23-000010",
    )
    _insert(
        conn,
        "PIT",
        2022,
        "share_repurchases_amount",
        10.0,
        filed_date="2023-02-15",
        accession="0000000000-22-000010",
    )

    signal = detect_buyback_signal(conn, "PIT", "2024-01-31")

    assert signal["label"] == "STEADY"
    assert signal["latest_fy"] == 10.0
    assert signal["prior_fy"] is None
    assert signal["fiscal_year"] == 2022
    assert signal["status"] == "OK"
    assert signal["source_lineage"][0]["filed_date"] == "2023-02-15"
    assert signal["source_lineage"][0]["accession"] == "0000000000-22-000010"


def test_detect_buyback_missing_filed_date_or_source_fails_closed():
    from app.catalyst.buyback import detect_buyback_signal

    conn = _make_conn()
    _insert(
        conn,
        "MISS",
        2023,
        "share_repurchases_amount",
        100.0,
        filed_date=None,
    )
    _insert(
        conn,
        "MISS",
        2022,
        "share_repurchases_amount",
        100.0,
        source_url=None,
    )

    signal = detect_buyback_signal(conn, "MISS", "2024-06-01")

    assert signal["label"] == "NONE"
    assert signal["latest_fy"] is None
    assert signal["prior_fy"] is None
    assert signal["fiscal_year"] is None
    assert signal["status"] == "NEEDS_DATA"
    assert signal["reason_codes"] == [
        "MISSING_OR_INVALID_FILED_DATE",
        "MISSING_SOURCE_URL",
    ]
    assert signal["source_lineage"] == []


def test_detect_buyback_blank_unit_or_accession_is_needs_data():
    from app.catalyst.buyback import detect_buyback_signal

    conn = _make_conn()
    _insert(
        conn,
        "PROV",
        2023,
        "share_repurchases_amount",
        200.0,
        units="",
    )
    _insert(
        conn,
        "PROV",
        2022,
        "share_repurchases_amount",
        100.0,
        accession="",
    )

    signal = detect_buyback_signal(conn, "PROV", "2024-06-01")

    assert signal["label"] == "NONE"
    assert signal["status"] == "NEEDS_DATA"
    assert signal["reason_codes"] == [
        "INVALID_OR_MISSING_UNIT",
        "MISSING_ACCESSION",
    ]
    assert signal["latest_fy"] is None
    assert signal["source_lineage"] == []
