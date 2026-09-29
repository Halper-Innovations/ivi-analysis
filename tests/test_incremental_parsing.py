from __future__ import annotations

from app.db import get_db, init_db, utc_now_iso
from app.parse.filing_parser import parse_pending_filings


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


def test_incremental_parsing_skips_accession_already_parsed(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    doc = cfg.raw_filings_dir / "320193" / "000032019326000001" / "doc.htm"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("<html><body>Revenue 100</body></html>", encoding="utf-8")

    accession = "0000320193-26-000001"
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES(
                '320193','AAPL', ?, '10-Q', '2026-02-10', '2025-12-31',
                'https://www.sec.gov/Archives/edgar/data/320193/doc.htm', ?, 'h1', '2026-02-13', 'downloaded', ?, ?
            )
            """,
            (accession, str(doc), utc_now_iso(), utc_now_iso()),
        )
        conn.execute(
            "INSERT INTO parsed_filings(accession, parsed_at, content_hash) VALUES(?, ?, ?)",
            (accession, utc_now_iso(), "h1"),
        )

    parsed_count = parse_pending_filings(limit=10, tickers=["AAPL"])
    assert parsed_count == 0

    with get_db() as conn:
        status = conn.execute("SELECT status FROM filings WHERE accession = ?", (accession,)).fetchone()["status"]
        assert status == "parsed"


def test_parser_scopes_annual_forms_and_fixed_asof_before_limit(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    doc = tmp_path / "annual.htm"
    doc.write_text("<html><body>eligible annual filing</body></html>", encoding="utf-8")
    with get_db() as conn:
        rows = [
            ("eligible", "10-K", "2026-02-15"),
            ("future", "10-K", "2026-07-15"),
            *[(f"event-{index}", "8-K", f"2026-05-{index + 1:02d}") for index in range(11)],
        ]
        for accession, form_type, filing_date in rows:
            conn.execute(
                """
                INSERT INTO filings(
                    cik, ticker, accession, form_type, filing_date, period_end,
                    primary_doc_url, local_path, hash, ingested_as_of,
                    status, created_at, updated_at
                ) VALUES(
                    '42', 'ALIAS', ?, ?, ?, '2025-12-31',
                    'https://www.sec.gov/filing.htm', ?, ?, '2026-06-11',
                    'downloaded', ?, ?
                )
                """,
                (
                    accession,
                    form_type,
                    filing_date,
                    str(doc),
                    f"hash-{accession}",
                    utc_now_iso(),
                    utc_now_iso(),
                ),
            )

    parsed_count = parse_pending_filings(
        limit=1,
        issuer_cik="42",
        form_types=("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"),
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
    )

    with get_db() as conn:
        statuses = dict(conn.execute("SELECT accession, status FROM filings"))
    assert parsed_count == 1
    assert statuses["eligible"] == "parsed"
    assert statuses["future"] == "downloaded"
    assert all(statuses[f"event-{index}"] == "downloaded" for index in range(11))
