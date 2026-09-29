from __future__ import annotations

import json
from datetime import datetime, timezone

from app.db import get_db, init_db
from app.util.financial_data_access import (
    ANNUAL_CACHED_FILING_FORM_TYPES,
    ANNUAL_COMPANYFACTS_PERIOD_TYPES,
    companyfacts_is_fresh,
    companyfacts_map_for_latest_period,
    companyfacts_rows,
    foreign_normalized_facts_gap_reason,
    issuer_companyfacts_rows,
    issuer_filing_rows,
    latest_filing_ids_by_ticker,
    latest_filing_row,
)


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config as _gc

    _gc.cache_clear()
    cfg = _gc()
    init_db(cfg)
    return cfg


def _insert_filing(
    conn,
    *,
    cik: str,
    ticker: str,
    accession: str,
    form_type: str,
    filing_date: str,
    period_end: str,
    status: str,
    local_path: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO filings(
            cik, ticker, accession, form_type, filing_date, period_end,
            primary_doc_url, local_path, status, created_at, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cik,
            ticker,
            accession,
            form_type,
            filing_date,
            period_end,
            "https://example.com/filing",
            local_path,
            status,
            "2026-03-20T00:00:00+00:00",
            "2026-03-20T00:00:00+00:00",
        ),
    )


def test_latest_filing_row_ignores_unsupported_forms_and_uses_ok_status(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    with get_db() as conn:
        _insert_filing(
            conn,
            cik="0000000001",
            ticker="ABC",
            accession="abc-10k",
            form_type="10-K",
            filing_date="2025-02-01",
            period_end="2024-12-31",
            status="OK",
            local_path="/tmp/abc-10k.html",
        )
        _insert_filing(
            conn,
            cik="0000000001",
            ticker="ABC",
            accession="abc-8k",
            form_type="8-K",
            filing_date="2025-03-01",
            period_end="2025-03-01",
            status="parsed",
            local_path="/tmp/abc-8k.html",
        )
        row = latest_filing_row(
            conn,
            "ABC",
            columns=("accession", "form_type", "status"),
            form_types=ANNUAL_CACHED_FILING_FORM_TYPES,
            require_local_path=True,
        )

    assert row["accession"] == "abc-10k"
    assert row["form_type"] == "10-K"
    assert row["status"] == "OK"


def test_latest_filing_ids_by_ticker_selects_latest_supported_cached_filing(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    with get_db() as conn:
        _insert_filing(
            conn,
            cik="0000000001",
            ticker="AAA",
            accession="aaa-10q",
            form_type="10-Q",
            filing_date="2025-05-01",
            period_end="2025-03-31",
            status="parsed",
        )
        _insert_filing(
            conn,
            cik="0000000001",
            ticker="AAA",
            accession="aaa-8k",
            form_type="8-K",
            filing_date="2025-05-05",
            period_end="2025-05-05",
            status="parsed",
        )
        _insert_filing(
            conn,
            cik="0000000002",
            ticker="BBB",
            accession="bbb-20f",
            form_type="20-F",
            filing_date="2025-04-01",
            period_end="2024-12-31",
            status="OK",
        )
        ids = latest_filing_ids_by_ticker(conn)
        rows = conn.execute(
            "SELECT accession, ticker FROM filings WHERE id IN (?, ?)", (ids["AAA"], ids["BBB"])
        ).fetchall()

    by_ticker = {row["ticker"]: row["accession"] for row in rows}
    assert by_ticker["AAA"] == "aaa-10q"
    assert by_ticker["BBB"] == "bbb-20f"


def test_companyfacts_rows_filters_annual_period_type(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at
            )
            VALUES('AAA', 2024, 'FY', '2024-12-31', 'revenue', 1000.0, 'USD_millions', 'test', '2026-03-20T00:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at
            )
            VALUES('AAA', 2024, 'Q1', '2024-03-31', 'revenue', 100.0, 'USD_millions', 'test', '2026-03-20T00:00:00+00:00')
            """
        )
        annual_rows = companyfacts_rows(
            conn,
            "AAA",
            columns=("period_type", "value"),
            period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
            line_items=("revenue",),
        )
        quarterly_rows = companyfacts_rows(
            conn,
            "AAA",
            columns=("period_type", "value"),
            exclude_period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
            line_items=("revenue",),
        )

    assert [(row["period_type"], row["value"]) for row in annual_rows] == [("FY", 1000.0)]
    assert [(row["period_type"], row["value"]) for row in quarterly_rows] == [("Q1", 100.0)]


def test_companyfacts_rows_explicit_filed_asof_excludes_future_and_undated_rows(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)

    with get_db() as conn:
        for ticker, period_end, filed_date, source_url, accession in (
            ("VISIBLE", "2025-12-31", "2026-02-01", "test", "visible-2025"),
            ("FUTURE", "2025-12-31", "2026-07-01", "test", "future-2025"),
            ("NULL_OLD", "2025-12-31", None, "test", "null-old-2025"),
            ("NULL_RECENT", "2026-04-30", None, "test", "null-recent-2025"),
            ("FILED_BEFORE_PERIOD", "2026-03-31", "2026-02-01", "test", "bad-order"),
            ("NO_SOURCE", "2025-12-31", "2026-02-01", "", "no-source"),
            ("NO_ACCESSION", "2025-12-31", "2026-02-01", "test", ""),
        ):
            conn.execute(
                "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, "
                "period_end, line_item, value, units, source_url, fetched_at, "
                "filed_date, accession) VALUES (?, 2025, 'FY', ?, 'revenue', 100, "
                "'USD_millions', ?, 'x', ?, ?)",
                (ticker, period_end, source_url, filed_date, accession),
            )
        visible = []
        for ticker in (
            "VISIBLE",
            "FUTURE",
            "NULL_OLD",
            "NULL_RECENT",
            "FILED_BEFORE_PERIOD",
            "NO_SOURCE",
            "NO_ACCESSION",
        ):
            rows = companyfacts_rows(
                conn,
                ticker,
                columns=("ticker",),
                period_types=("FY",),
                as_of_date="2026-06-11",
                require_filed_asof=True,
            )
            visible.extend(str(row["ticker"]) for row in rows)

    assert visible == ["VISIBLE"]


def test_companyfacts_map_for_latest_period_honors_as_of_date(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    with get_db() as conn:
        rows = [
            (2024, "FY", "2024-12-31", "2025-02-15", "revenue", 1000.0),
            (2024, "FY", "2024-12-31", "2025-02-15", "gross_profit", 400.0),
            (2025, "FY", "2025-12-31", "2026-02-15", "revenue", 1200.0),
            (2025, "FY", "2025-12-31", "2026-02-15", "gross_profit", 480.0),
        ]
        for fiscal_year, period_type, period_end, filed_date, line_item, value in rows:
            conn.execute(
                """
                    INSERT INTO companyfacts_facts(
                        ticker, fiscal_year, period_type, period_end, filed_date,
                        line_item, value, units, source_url, fetched_at, accession
                    )
                    VALUES(
                        'AAA', ?, ?, ?, ?, ?, ?, 'USD_millions', 'test',
                        '2026-03-20T00:00:00+00:00', ?
                    )
                """,
                (
                    fiscal_year,
                    period_type,
                    period_end,
                    filed_date,
                    line_item,
                    value,
                    f"AAA-{fiscal_year}",
                ),
            )
        payload, period_end = companyfacts_map_for_latest_period(
            conn,
            "AAA",
            as_of_date="2025-06-01",
            period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
        )

    assert period_end == "2024-12-31"
    assert payload["revenue"] == 1000.0
    assert payload["gross_profit"] == 400.0


def test_companyfacts_is_fresh_respects_period_filters(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at
            )
            VALUES('AAA', 2024, 'FY', '2024-12-31', 'revenue', 1000.0, 'USD_millions', 'test', '2000-01-01T00:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at
            )
            VALUES('AAA', 2025, ?, '2024-03-31', 'revenue', 100.0, 'USD_millions', 'test', '2026-03-20T00:00:00+00:00')
            """,
            ("Q1",),
        )
        conn.execute(
            """
            UPDATE companyfacts_facts
            SET fetched_at = ?
            WHERE ticker = 'AAA' AND fiscal_year = 2025 AND period_type = 'Q1' AND line_item = 'revenue'
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        annual_fresh = companyfacts_is_fresh(
            conn,
            "AAA",
            ttl_seconds=86400,
            period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
        )
        quarterly_fresh = companyfacts_is_fresh(
            conn,
            "AAA",
            ttl_seconds=86400,
            exclude_period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
        )

    assert annual_fresh is False
    assert quarterly_fresh is True


def test_issuer_filing_rows_recovers_all_annual_forms_from_same_cik_alias(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    forms = ("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO sec_registrants(
                cik, primary_ticker, all_tickers, exchange_scope,
                operating_status, first_seen_at, last_seen_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "0000000042",
                "PRIMARY",
                '["PRIMARY", "ALIAS"]',
                "US_EXCHANGE",
                "OPERATING",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        for index, form_type in enumerate(forms, start=1):
            _insert_filing(
                conn,
                cik="42",
                ticker="PRIMARY",
                accession=f"0000000042-26-0000{index:02d}",
                form_type=form_type,
                filing_date=f"2026-0{index}-01",
                period_end="2025-12-31",
                status="parsed",
            )
        scope, rows = issuer_filing_rows(
            conn,
            "ALIAS",
            columns=("accession", "form_type"),
            form_types=ANNUAL_CACHED_FILING_FORM_TYPES,
        )

    assert scope.issuer_cik == "42"
    assert scope.aliases == ("ALIAS", "PRIMARY")
    assert [row["form_type"] for row in rows] == [
        "40-F/A",
        "40-F",
        "20-F/A",
        "20-F",
        "10-K/A",
        "10-K",
    ]


def test_issuer_filing_rows_known_cik_rejects_conflicting_alias_rows(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _insert_filing(
            conn,
            cik="42",
            ticker="PRIMARY",
            accession="correct-cik",
            form_type="20-F",
            filing_date="2025-03-01",
            period_end="2024-12-31",
            status="parsed",
        )
        _insert_filing(
            conn,
            cik="999",
            ticker="ALIAS",
            accession="wrong-cik-newer",
            form_type="20-F",
            filing_date="2026-03-01",
            period_end="2025-12-31",
            status="parsed",
        )
        scope, rows = issuer_filing_rows(
            conn,
            "ALIAS",
            columns=("cik", "accession"),
            issuer_cik="42",
            aliases=("ALIAS", "PRIMARY"),
            form_types=ANNUAL_CACHED_FILING_FORM_TYPES,
        )

    assert scope.issuer_cik == "42"
    assert [(row["cik"], row["accession"]) for row in rows] == [("42", "correct-cik")]


def test_issuer_companyfacts_rows_known_cik_rejects_conflicting_alias_rows(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        for ticker, value, source_url in (
            (
                "PRIMARY",
                100.0,
                "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json",
            ),
            (
                "ALIAS",
                999.0,
                "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000999.json",
            ),
        ):
            conn.execute(
                """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at
                ) VALUES(?, 2025, 'FY', '2025-12-31', 'revenue', ?,
                         'USD_millions', ?, '2026-03-01T00:00:00+00:00')
                """,
                (ticker, value, source_url),
            )
        scope, rows = issuer_companyfacts_rows(
            conn,
            "ALIAS",
            columns=("ticker", "value"),
            issuer_cik="42",
            aliases=("ALIAS", "PRIMARY"),
            period_types=("FY",),
            line_items=("revenue",),
        )

    assert scope.issuer_cik == "42"
    assert [(row["ticker"], row["value"]) for row in rows] == [("PRIMARY", 100.0)]


def test_issuer_companyfacts_rows_discovers_alias_from_authoritative_source_cik(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at
            ) VALUES(
                'PRIMARY', 2025, 'FY', '2025-12-31', 'revenue', 100.0,
                'USD_millions',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json',
                '2026-03-01T00:00:00+00:00'
            )
            """
        )
        scope, rows = issuer_companyfacts_rows(
            conn,
            "ADR_ALIAS",
            columns=("ticker", "value"),
            issuer_cik="42",
            aliases=("ADR_ALIAS",),
            period_types=("FY",),
            line_items=("revenue",),
        )
        indexes = {str(row[1]) for row in conn.execute("PRAGMA index_list(companyfacts_facts)")}

    assert scope.aliases == ("ADR_ALIAS", "PRIMARY")
    assert scope.sources == ("caller_cik", "companyfacts_source_identity")
    assert [(row["ticker"], row["value"]) for row in rows] == [("PRIMARY", 100.0)]
    assert "idx_companyfacts_source_url" in indexes


def test_issuer_companyfacts_rows_uses_visible_vintage_before_future_live_restatement(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form, accession
            ) VALUES(
                'PRIMARY', 2025, 'FY', '2025-12-31', 'revenue', 140.0,
                'USD_millions',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json',
                '2026-08-01T00:00:00+00:00', '2026-08-01', '10-K/A', 'future'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO companyfacts_vintages(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, filed_date, form, accession, recorded_at,
                issuer_cik, source_url
            ) VALUES(
                'PRIMARY', 2025, 'FY', '2025-12-31', 'revenue', 100.0,
                'USD_millions', '2026-03-01', '10-K', 'original',
                '2026-03-01T00:00:00+00:00', '0000000042',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO companyfacts_vintages(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, filed_date, form, accession, recorded_at,
                issuer_cik, source_url
            ) VALUES(
                'ALIAS', 2025, 'FY', '2025-12-31', 'revenue', 999.0,
                'USD_millions', '2026-02-15', '10-K', 'wrong-issuer',
                '2026-02-15T00:00:00+00:00', '0000000999',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000999.json'
            )
            """
        )
        scope, rows = issuer_companyfacts_rows(
            conn,
            "ALIAS",
            columns=("ticker", "value", "filed_date"),
            issuer_cik="42",
            aliases=("ALIAS", "PRIMARY"),
            period_types=("FY",),
            line_items=("revenue",),
            as_of_date="2026-06-11",
            require_filed_asof=True,
        )

    assert scope.issuer_cik == "42"
    assert [(row["ticker"], row["value"], row["filed_date"]) for row in rows] == [
        ("PRIMARY", 100.0, "2026-03-01")
    ]


def test_issuer_companyfacts_rows_preverifies_identity_absent_vintage_aliases_once(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    traced_sql: list[str] = []
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form, accession
            ) VALUES(
                'primary', 2025, 'FY', '2025-12-31', 'revenue', 140.0,
                'USD_millions',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json',
                '2026-08-01T00:00:00+00:00', '2026-08-01', '10-K/A', 'future'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('company_alias', '0000000042', 'Issuer', '2026-01-01')
            """
        )
        for ticker, line_item, value, accession in (
            ("PRIMARY", "revenue", 100.0, "verified-facts-vintage"),
            ("COMPANY_ALIAS", "net_income", 20.0, "verified-company-vintage"),
            ("UNVERIFIED_ALIAS", "equity", 999.0, "unverified-legacy-vintage"),
        ):
            conn.execute(
                """
                INSERT INTO companyfacts_vintages(
                    ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, filed_date, form, accession, recorded_at,
                    issuer_cik, source_url
                ) VALUES(
                    ?, 2025, 'FY', '2025-12-31', ?, ?,
                    'USD_millions', '2026-03-01', '10-K', ?,
                    '2026-03-01T00:00:00+00:00', NULL, NULL
                )
                """,
                (ticker, line_item, value, accession),
            )
        conn.set_trace_callback(traced_sql.append)
        scope, rows = issuer_companyfacts_rows(
            conn,
            "UNVERIFIED_ALIAS",
            columns=("ticker", "line_item", "value", "accession"),
            issuer_cik="42",
            aliases=("UNVERIFIED_ALIAS", "PRIMARY", "COMPANY_ALIAS"),
            period_types=("FY",),
            line_items=("revenue", "net_income", "equity"),
            as_of_date="2026-06-11",
            require_filed_asof=True,
        )
        conn.set_trace_callback(None)

    pit_queries = [sql for sql in traced_sql if "candidates AS" in sql]
    assert scope.issuer_cik == "42"
    assert [(row["ticker"], row["line_item"], row["value"], row["accession"]) for row in rows] == [
        ("COMPANY_ALIAS", "net_income", 20.0, "verified-company-vintage"),
        ("PRIMARY", "revenue", 100.0, "verified-facts-vintage"),
    ]
    assert len(pit_queries) == 1
    assert "CORRELATED" not in pit_queries[0].upper()
    assert "EXISTS (SELECT 1 FROM companyfacts_facts current_fact" not in pit_queries[0]


def test_issuer_companyfacts_rows_keeps_explicit_vintage_without_live_alias_proof(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_vintages(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, filed_date, form, accession, recorded_at,
                issuer_cik, source_url
            ) VALUES(
                'LEGACY', 2024, 'FY', '2024-12-31', 'revenue', 88.0,
                'USD_millions', '2025-03-01', '10-K', 'explicit-only',
                '2025-03-01T00:00:00+00:00', '0000000042',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json'
            )
            """
        )
        _, rows = issuer_companyfacts_rows(
            conn,
            "LEGACY",
            columns=("ticker", "value", "accession"),
            issuer_cik="42",
            aliases=("LEGACY",),
            period_types=("FY",),
            as_of_date="2025-06-01",
            require_filed_asof=True,
        )

    assert [(row["ticker"], row["value"], row["accession"]) for row in rows] == [
        ("LEGACY", 88.0, "explicit-only")
    ]


def test_foreign_normalized_facts_gap_reports_ifrs_unsupported(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    raw_path = tmp_path / "ifrs.json"
    raw_path.write_text(
        json.dumps({"facts": {"ifrs-full": {"Revenue": {"units": {"USD": [{"val": 100.0}]}}}}}),
        encoding="utf-8",
    )
    with get_db() as conn:
        reason = foreign_normalized_facts_gap_reason(
            conn,
            "ADR",
            issuer_cik="42",
            form_type="20-F/A",
            as_of_date="2026-04-01",
            raw_companyfacts_path=raw_path,
        )

    assert reason == "IFRS_FACTS_UNSUPPORTED"


def test_foreign_normalized_facts_gap_reports_non_usd_unnormalized(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    raw_path = tmp_path / "cad.json"
    raw_path.write_text(
        json.dumps(
            {
                "facts": {
                    "dei": {},
                    "custom": {"Revenue": {"units": {"CAD": [{"val": 125.0}]}}},
                }
            }
        ),
        encoding="utf-8",
    )
    with get_db() as conn:
        reason = foreign_normalized_facts_gap_reason(
            conn,
            "CAN",
            issuer_cik="43",
            form_type="40-F",
            as_of_date="2026-04-01",
            raw_companyfacts_path=raw_path,
        )

    assert reason == "NON_USD_FACTS_UNNORMALIZED"


def test_foreign_gap_does_not_blame_incidental_non_usd_disclosure(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    raw_path = tmp_path / "mixed.json"
    raw_path.write_text(
        json.dumps(
            {
                "facts": {
                    "us-gaap": {
                        "Revenue": {"units": {"USD": [{"val": 100.0}]}},
                        "ForeignCurrencyTransactionGainLoss": {"units": {"EUR": [{"val": 2.0}]}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with get_db() as conn:
        reason = foreign_normalized_facts_gap_reason(
            conn,
            "MIXD",
            issuer_cik="44",
            form_type="20-F",
            as_of_date="2026-04-01",
            raw_companyfacts_path=raw_path,
            required_line_items=("revenue", "net_income"),
        )

    assert reason == "FOREIGN_NORMALIZED_FACTS_UNAVAILABLE"


def test_foreign_normalized_facts_gap_clears_when_fy_rows_exist(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ADR",
                2025,
                "FY",
                "2025-12-31",
                "revenue",
                100.0,
                "USD_millions",
                "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json",
                "2026-03-01T00:00:00+00:00",
            ),
        )
        reason = foreign_normalized_facts_gap_reason(
            conn,
            "ADR",
            issuer_cik="42",
            form_type="20-F",
            as_of_date="2026-04-01",
            raw_companyfacts_path=tmp_path / "missing.json",
        )

    assert reason is None


def test_foreign_gap_known_cik_ignores_normalized_rows_from_conflicting_alias(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    raw_path = tmp_path / "ifrs.json"
    raw_path.write_text(
        json.dumps({"facts": {"ifrs-full": {"Revenue": {"units": {"USD": [{"val": 100.0}]}}}}}),
        encoding="utf-8",
    )
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at
            ) VALUES('ADR', 2025, 'FY', '2025-12-31', 'revenue', 999.0,
                     'USD_millions',
                     'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000999.json',
                     '2026-03-01T00:00:00+00:00')
            """
        )
        reason = foreign_normalized_facts_gap_reason(
            conn,
            "ADR",
            issuer_cik="42",
            aliases=("ADR",),
            form_type="20-F",
            as_of_date="2026-04-01",
            raw_companyfacts_path=raw_path,
        )

    assert reason == "IFRS_FACTS_UNSUPPORTED"
