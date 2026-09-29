from __future__ import annotations


from app.db import get_db, init_db, utc_now_iso
from app.research.adapters.base import AdapterContext
from app.research.adapters.sec_exhibits import SecExhibitsAdapter


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    monkeypatch.setenv("VOE_RESEARCH_EXHIBITS_ENABLED", "true")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_sec_exhibits_adapter_extracts_8k_evidence(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    filing_path = cfg.raw_filings_dir / "test_8k.htm"
    filing_path.parent.mkdir(parents=True, exist_ok=True)
    filing_path.write_text(
        """
        <html><head><title>Form 8-K Results</title></head><body>
        Exhibit 99.1 Press Release announcing quarterly results of operations.
        Exhibit 99.2 Investor presentation slide deck.
        Guidance raised for full year.
        </body></html>
        """,
        encoding="utf-8",
    )

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('AAPL', '320193', 'Apple', ?)
            """,
            (utc_now_iso(),),
        )
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES(
                '320193', 'AAPL', '0000320193-26-000999', '8-K', '2026-02-12', '2025-12-31',
                'https://www.sec.gov/Archives/edgar/data/320193/test8k.htm', ?, 'h', '2026-02-13', 'parsed', ?, ?
            )
            """,
            (str(filing_path), utc_now_iso(), utc_now_iso()),
        )

    adapter = SecExhibitsAdapter(cfg)
    result = adapter.collect(
        AdapterContext(
            ticker="AAPL",
            as_of_date="2026-02-13",
            company_name="Apple",
            packet={},
            run_id="run_test",
        )
    )

    assert result.evidence_items
    assert any(item.source_type == "sec_exhibit" for item in result.evidence_items)
    assert any("earnings_release" in (item.source_title or "").lower() for item in result.evidence_items)


def test_sec_exhibits_adapter_detects_item_202_financial_results_without_exact_old_phrase(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    filing_path = cfg.raw_filings_dir / "test_item202_8k.htm"
    filing_path.parent.mkdir(parents=True, exist_ok=True)
    filing_path.write_text(
        """
        <html><head><title>Current Report</title></head><body>
        Item 2.02 Results of Financial Condition and Operations.
        EX-99.1 Earnings release for first quarter financial results.
        Management raised full-year outlook and discussed conference call details.
        </body></html>
        """,
        encoding="utf-8",
    )

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('MSFT', '789019', 'Microsoft', ?)
            """,
            (utc_now_iso(),),
        )
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES(
                '789019', 'MSFT', '0000789019-26-000999', '8-K', '2026-02-12', '2025-12-31',
                'https://www.sec.gov/Archives/edgar/data/789019/testitem202.htm', ?, 'h', '2026-02-13', 'parsed', ?, ?
            )
            """,
            (str(filing_path), utc_now_iso(), utc_now_iso()),
        )

    adapter = SecExhibitsAdapter(cfg)
    result = adapter.collect(
        AdapterContext(
            ticker="MSFT",
            as_of_date="2026-02-13",
            company_name="Microsoft",
            packet={},
            run_id="run_test",
        )
    )

    assert result.evidence_items
    item = result.evidence_items[0]
    assert "earnings_release" in str(item.source_title or "").lower()
    assert "item_202" in str(item.source_title or "").lower()
    assert item.citations[0].section_label == "8k_202_earnings_release"


def test_sec_exhibits_adapter_surfaces_focus_labels_for_liquidity_and_non_gaap(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    filing_path = cfg.raw_filings_dir / "test_item701_8k.htm"
    filing_path.parent.mkdir(parents=True, exist_ok=True)
    filing_path.write_text(
        """
        <html><head><title>Investor Presentation</title></head><body>
        Item 7.01 Regulation FD Disclosure.
        EX-99.2 Investor presentation.
        Non-GAAP reconciliation of adjusted results.
        Liquidity, covenant, and maturity profile discussion.
        Credit reserve trends, charge-off outlook, and provision commentary.
        Share repurchase framework and dividend outlook.
        </body></html>
        """,
        encoding="utf-8",
    )

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('JPM', '19617', 'JPMorgan Chase', ?)
            """,
            (utc_now_iso(),),
        )
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES(
                '19617', 'JPM', '0000019617-26-000999', '8-K', '2026-02-12', '2025-12-31',
                'https://www.sec.gov/Archives/edgar/data/19617/testitem701.htm', ?, 'h', '2026-02-13', 'parsed', ?, ?
            )
            """,
            (str(filing_path), utc_now_iso(), utc_now_iso()),
        )

    adapter = SecExhibitsAdapter(cfg)
    result = adapter.collect(
        AdapterContext(
            ticker="JPM",
            as_of_date="2026-02-13",
            company_name="JPMorgan Chase",
            packet={},
            run_id="run_test",
        )
    )

    assert result.evidence_items
    item = result.evidence_items[0]
    assert "investor_presentation" in str(item.source_title or "").lower()
    assert (
        "liquidity_and_debt" in str(item.source_title or "").lower()
        or "non_gaap" in str(item.source_title or "").lower()
        or "credit_reserves" in str(item.source_title or "").lower()
    )
    assert item.citations[0].section_label.startswith("8k_701_")


def test_sec_exhibits_adapter_emits_credit_reserve_item_when_bank_commentary_present(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    filing_path = cfg.raw_filings_dir / "test_credit_8k.htm"
    filing_path.parent.mkdir(parents=True, exist_ok=True)
    filing_path.write_text(
        """
        <html><head><title>Investor Presentation</title></head><body>
        Item 7.01 Regulation FD Disclosure.
        Exhibit 99.2 Investor presentation.
        Reserve build commentary, charge-off trends, credit normalization, and provision outlook were discussed.
        Liquidity and capital return were also reviewed.
        </body></html>
        """,
        encoding="utf-8",
    )
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('JPM', '19617', 'JPMorgan Chase', ?)
            """,
            (utc_now_iso(),),
        )
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES(
                '19617', 'JPM', '0000019617-26-001222', '8-K', '2026-02-12', '2025-12-31',
                'https://www.sec.gov/Archives/edgar/data/19617/testcredit.htm', ?, 'h', '2026-02-13', 'parsed', ?, ?
            )
            """,
            (str(filing_path), utc_now_iso(), utc_now_iso()),
        )

    adapter = SecExhibitsAdapter(cfg)
    result = adapter.collect(
        AdapterContext(
            ticker="JPM",
            as_of_date="2026-02-13",
            company_name="JPMorgan Chase",
            packet={},
            run_id="run_test",
        )
    )

    assert result.evidence_items
    assert any("credit_reserves" in str(item.source_title or "").lower() for item in result.evidence_items)


def test_sec_exhibits_adapter_gap_is_specific_when_8k_exists_but_has_no_relevant_exhibit_markers(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    filing_path = cfg.raw_filings_dir / "plain_event_8k.htm"
    filing_path.parent.mkdir(parents=True, exist_ok=True)
    filing_path.write_text(
        """
        <html><head><title>Current Report on Form 8-K</title></head><body>
        The company announced executive leadership changes and board committee updates.
        </body></html>
        """,
        encoding="utf-8",
    )

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('JPM', '19617', 'JPMorgan Chase', ?)
            """,
            (utc_now_iso(),),
        )
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES(
                '19617', 'JPM', '0000019617-26-001111', '8-K', '2026-02-05', '2025-12-31',
                'https://www.sec.gov/Archives/edgar/data/19617/plain8k.htm', ?, 'h', '2026-02-13', 'parsed', ?, ?
            )
            """,
            (str(filing_path), utc_now_iso(), utc_now_iso()),
        )

    adapter = SecExhibitsAdapter(cfg)
    result = adapter.collect(
        AdapterContext(
            ticker="JPM",
            as_of_date="2026-02-13",
            company_name="JPMorgan Chase",
            packet={},
            run_id="run_test",
        )
    )

    assert result.evidence_items == []
    assert result.evidence_gaps
    assert "scanned available 8-k filings" in result.evidence_gaps[0].summary.lower()
    assert "confirm whether the issuer published relevant 8-k exhibits" in result.evidence_gaps[0].recommended_action.lower()
