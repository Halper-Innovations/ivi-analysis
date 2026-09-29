from __future__ import annotations

from app.db import get_db, init_db, utc_now_iso
from app.discovery.metrics import _shares_series_for_accessions, compute_discovery_metrics


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _bind_companyfacts_to_filing(
    conn,
    *,
    ticker: str,
    accession: str,
    filed_date: str,
) -> None:
    conn.execute(
        """
        UPDATE companyfacts_facts
        SET filed_date = ?, form = '10-K', accession = ?
        WHERE ticker = ?
        """,
        (filed_date, accession, ticker),
    )


def test_compute_discovery_metrics_uses_companyfacts_annual_fallback(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        now = utc_now_iso()
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end, primary_doc_url,
                local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, 'parsed', ?, ?)
            """,
            (
                "0000000001",
                "AAA",
                "0000000001-26-000001",
                "10-K",
                "2026-02-13",
                "2025-12-31",
                "https://www.sec.gov/example",
                "2026-02-13",
                now,
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_end, line_item, value, units, source_url, fetched_at
            ) VALUES
                ('AAA', 2025, '2025-12-31', 'revenue', 100.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('AAA', 2025, '2025-12-31', 'gross_profit', 40.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('AAA', 2025, '2025-12-31', 'operating_income', 20.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('AAA', 2025, '2025-12-31', 'cfo', 18.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('AAA', 2025, '2025-12-31', 'capex', 3.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('AAA', 2025, '2025-12-31', 'net_income', 15.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('AAA', 2025, '2025-12-31', 'cash', 10.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('AAA', 2025, '2025-12-31', 'total_debt', 22.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('AAA', 2025, '2025-12-31', 'shares_outstanding', 50.0, 'shares_millions', 'https://www.sec.gov/companyfacts', ?)
            """,
            (now, now, now, now, now, now, now, now, now),
        )
        _bind_companyfacts_to_filing(
            conn,
            ticker="AAA",
            accession="0000000001-26-000001",
            filed_date="2026-02-13",
        )

        result = compute_discovery_metrics(
            conn, ticker="AAA", selected_accessions=["0000000001-26-000001"]
        )

    assert result is not None
    assert result.metrics["ttm_revenue"] == 100.0
    assert result.metrics["gross_margin"] == 0.4
    assert result.metrics["fcf"] == 15.0
    assert result.metrics["shares_outstanding"] == 50.0
    claim = next(claim for claim in result.claims if claim["label"] == "ttm_revenue")
    assert "companyfacts_facts.revenue" in claim["derived_from"]


def test_companyfacts_annual_fallback_ignores_quarterly_rows(monkeypatch, tmp_path):
    """A quarter inserted after the FY row (higher id) must not shadow the full-year value."""
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        now = utc_now_iso()
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end, primary_doc_url,
                local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, 'parsed', ?, ?)
            """,
            (
                "0000004281",
                "AA",
                "0000004281-25-000001",
                "10-K",
                "2025-02-13",
                "2024-12-31",
                "https://www.sec.gov/example",
                "2025-02-13",
                now,
                now,
            ),
        )
        # FY row inserted FIRST (lower id), quarter inserted AFTER (higher id).
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at
            ) VALUES('AA', 2024, 'FY', '2024-12-31', 'revenue', 11895.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?)
            """,
            (now,),
        )
        _bind_companyfacts_to_filing(
            conn,
            ticker="AA",
            accession="0000004281-25-000001",
            filed_date="2025-02-13",
        )
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at
            ) VALUES('AA', 2024, 'Q3', '2024-09-30', 'revenue', 2602.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?)
            """,
            (now,),
        )
        _bind_companyfacts_to_filing(
            conn,
            ticker="AA",
            accession="0000004281-25-000001",
            filed_date="2025-02-13",
        )

        result = compute_discovery_metrics(
            conn, ticker="AA", selected_accessions=["0000004281-25-000001"]
        )

    assert result is not None
    assert result.metrics["ttm_revenue"] == 11895.0


def test_shares_fallback_prefers_annual_over_quarterly(monkeypatch, tmp_path):
    """The companyfacts shares fallback must use the FY count, not a later-fiscal-year quarter."""
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        now = utc_now_iso()
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end, primary_doc_url,
                local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, 'parsed', ?, ?)
            """,
            (
                "0000004281",
                "AA",
                "0000004281-25-000001",
                "10-K",
                "2025-02-13",
                "2024-12-31",
                "https://www.sec.gov/example",
                "2025-02-13",
                now,
                now,
            ),
        )
        # The latest fiscal_year is a quarter (2025/Q1). The FY row is fiscal_year 2024.
        # Without a period_type filter, ORDER BY fiscal_year DESC would surface the quarter first.
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at
            ) VALUES('AA', 2025, 'Q1', '2025-03-31', 'shares_outstanding', 999.0, 'shares_millions', 'https://www.sec.gov/companyfacts', ?)
            """,
            (now,),
        )
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at
            ) VALUES('AA', 2024, 'FY', '2024-12-31', 'shares_outstanding', 178.0, 'shares_millions', 'https://www.sec.gov/companyfacts', ?)
            """,
            (now,),
        )
        _bind_companyfacts_to_filing(
            conn,
            ticker="AA",
            accession="0000004281-25-000001",
            filed_date="2025-02-13",
        )

        shares = compute_discovery_metrics(
            conn, ticker="AA", selected_accessions=["0000004281-25-000001"]
        )

    assert shares is not None
    assert shares.metrics["shares_outstanding"] == 178.0


def test_shares_series_fallback_excludes_quarterly_counts(monkeypatch, tmp_path):
    """The companyfacts shares-series fallback must only count FY rows, not quarters."""
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        now = utc_now_iso()
        # No extracted_facts shares and no matching accession rows, so the
        # companyfacts fallback branch runs. A higher-fiscal-year quarter would
        # otherwise lead the fiscal_year DESC ordering.
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at
            ) VALUES
                ('AA', 2025, 'Q1', '2025-03-31', 'shares_outstanding', 999.0, 'shares_millions', 'https://www.sec.gov/companyfacts', ?),
                ('AA', 2024, 'FY', '2024-12-31', 'shares_outstanding', 178.0, 'shares_millions', 'https://www.sec.gov/companyfacts', ?),
                ('AA', 2023, 'FY', '2023-12-31', 'shares_outstanding', 180.0, 'shares_millions', 'https://www.sec.gov/companyfacts', ?)
            """,
            (now, now, now),
        )
        _bind_companyfacts_to_filing(
            conn,
            ticker="AA",
            accession="0000004281-25-000001",
            filed_date="2025-02-13",
        )

        series = _shares_series_for_accessions(conn, "AA", ["0000004281-25-000001"])

    assert series == [178.0, 180.0]


def test_compute_discovery_metrics_marks_bank_like_issuer_without_fcf_gap(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        now = utc_now_iso()
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end, primary_doc_url,
                local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, 'parsed', ?, ?)
            """,
            (
                "0000019617",
                "JPM",
                "0000019617-26-000001",
                "10-K",
                "2026-02-13",
                "2025-12-31",
                "https://www.sec.gov/example",
                "2026-02-13",
                now,
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_end, line_item, value, units, source_url, fetched_at
            ) VALUES
                ('JPM', 2025, '2025-12-31', 'revenue', 100.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('JPM', 2025, '2025-12-31', 'gross_profit', 60.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('JPM', 2025, '2025-12-31', 'operating_income', 20.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('JPM', 2025, '2025-12-31', 'cfo', -40.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('JPM', 2025, '2025-12-31', 'cash', 200.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('JPM', 2025, '2025-12-31', 'total_debt', 150.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('JPM', 2025, '2025-12-31', 'deposits', 2500.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('JPM', 2025, '2025-12-31', 'loans', 1400.0, 'USD_millions', 'https://www.sec.gov/companyfacts', ?),
                ('JPM', 2025, '2025-12-31', 'shares_outstanding', 50.0, 'shares_millions', 'https://www.sec.gov/companyfacts', ?)
            """,
            (now, now, now, now, now, now, now, now, now),
        )
        _bind_companyfacts_to_filing(
            conn,
            ticker="JPM",
            accession="0000019617-26-000001",
            filed_date="2026-02-13",
        )

        result = compute_discovery_metrics(
            conn, ticker="JPM", selected_accessions=["0000019617-26-000001"]
        )

    assert result is not None
    assert result.metrics["issuer_classification"] == "financial"
    assert result.metrics["fcf_applicability"] == "sector_limited"
    assert result.metrics["fcf"] == "UNKNOWN"
    assert "FINANCIAL_ISSUER_CASHFLOW_FRAME" in result.flags
    assert "FINANCIALS_INCOMPLETE" not in result.flags


def test_companyfacts_later_amendment_cannot_replace_selected_filing(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        now = utc_now_iso()
        selected_accession = "0000000001-26-000001"
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, hash, ingested_as_of, status,
                created_at, updated_at
            ) VALUES(
                '0000000001', 'PITX', ?, '10-K', '2026-02-13',
                '2025-12-31', 'https://www.sec.gov/selected', NULL, NULL,
                '2026-02-13', 'parsed', ?, ?
            )
            """,
            (selected_accession, now, now),
        )
        filing_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.execute(
            """
            INSERT INTO financials(
                filing_id, statement_type, line_item, value, units, period,
                source_url, snippet, created_at
            ) VALUES(
                ?, 'income_statement', 'revenue', 100.0, 'USD_millions',
                '2025-12-31', 'https://www.sec.gov/selected',
                'selected filing revenue', ?
            )
            """,
            (filing_id, now),
        )
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value,
                units, source_url, fetched_at, filed_date, form, accession
            ) VALUES(
                'PITX', 2025, 'FY', '2025-12-31', 'revenue', 999.0,
                'USD_millions', 'https://www.sec.gov/later-amendment', ?,
                '2026-03-01', '10-K/A', '0000000001-26-000099'
            )
            """,
            (now,),
        )

        result = compute_discovery_metrics(
            conn,
            ticker="PITX",
            selected_accessions=[selected_accession],
        )

    assert result is not None
    assert result.metrics["ttm_revenue"] == 100.0
    revenue_claim = next(claim for claim in result.claims if claim["label"] == "ttm_revenue")
    assert revenue_claim["citations"][0]["source_url"] == ("https://www.sec.gov/selected")
