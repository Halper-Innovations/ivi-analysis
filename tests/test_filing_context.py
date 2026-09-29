from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.config import get_config
from app.db import get_db, init_db
from app.research.filing_context import load_research_filing_context


def _init_temp_db(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _write_local_filing(tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return str(path)


def _insert_filing(
    *,
    ticker: str,
    cik: str,
    accession: str,
    form_type: str,
    filing_date: str,
    local_path: str | None,
    primary_doc_url: str,
    period_end: str | None = None,
    status: str = "parsed",
    content_hash: str | None = None,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO filings(
                cik,
                ticker,
                accession,
                form_type,
                filing_date,
                period_end,
                primary_doc_url,
                local_path,
                hash,
                status,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (
                cik,
                ticker,
                accession,
                form_type,
                filing_date,
                period_end,
                primary_doc_url,
                local_path,
                content_hash,
                status,
            ),
        )


def test_downloaded_filing_is_immediately_visible_to_research_context(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    annual_path = _write_local_filing(
        tmp_path,
        "downloaded-annual.html",
        "<h1>Risk Factors</h1><p>Freshly downloaded annual filing.</p>",
    )
    _insert_filing(
        ticker="FRESH",
        cik="0000000042",
        accession="0000000042-26-000001",
        form_type="10-K",
        filing_date="2026-02-15",
        period_end="2025-12-31",
        local_path=annual_path,
        primary_doc_url="https://www.sec.gov/Archives/fresh-annual.htm",
        status="downloaded",
    )

    context = load_research_filing_context(
        "FRESH",
        as_of_date="2026-04-18",
        issuer_cik="42",
        issuer_aware=True,
        include_material_events=False,
        allow_annual_download=False,
    )
    legacy = load_research_filing_context(
        "FRESH",
        as_of_date="2026-04-18",
        include_material_events=False,
        allow_annual_download=False,
    )

    assert [document.accession for document in context.documents] == ["0000000042-26-000001"]
    assert context.documents[0].html.endswith("<p>Freshly downloaded annual filing.</p>")
    assert legacy.documents == []
    assert "annual_filing_missing" in legacy.warnings


def test_load_research_filing_context_includes_recent_material_events(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    annual_path = _write_local_filing(
        tmp_path, "annual.html", "<h1>Item 7</h1><p>Annual filing.</p>"
    )
    quarter_path = _write_local_filing(
        tmp_path, "quarter.html", "<h1>Item 2</h1><p>Quarterly filing.</p>"
    )
    event_one_path = _write_local_filing(
        tmp_path, "event-one.html", "<h1>Item 5.02</h1><p>Leadership change.</p>"
    )
    event_two_path = _write_local_filing(
        tmp_path, "event-two.html", "<h1>Item 4.02</h1><p>Restatement update.</p>"
    )
    stale_event_path = _write_local_filing(
        tmp_path, "event-stale.html", "<h1>Item 8.01</h1><p>Old event.</p>"
    )

    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-25-000001",
        form_type="10-K",
        filing_date="2025-02-15",
        period_end="2024-12-31",
        local_path=annual_path,
        primary_doc_url="https://www.sec.gov/Archives/annual.htm",
    )
    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-26-000001",
        form_type="10-Q",
        filing_date="2026-02-01",
        period_end="2025-12-31",
        local_path=quarter_path,
        primary_doc_url="https://www.sec.gov/Archives/quarter.htm",
    )
    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-26-000101",
        form_type="8-K",
        filing_date="2026-04-10",
        local_path=event_one_path,
        primary_doc_url="https://www.sec.gov/Archives/event-one.htm",
    )
    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-25-000201",
        form_type="8-K/A",
        filing_date="2025-08-01",
        local_path=event_two_path,
        primary_doc_url="https://www.sec.gov/Archives/event-two.htm",
    )
    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-25-000301",
        form_type="8-K",
        filing_date="2025-03-01",
        local_path=stale_event_path,
        primary_doc_url="https://www.sec.gov/Archives/event-stale.htm",
    )

    context = load_research_filing_context("EXAMPLE", as_of_date="2026-04-18", quarters=1)

    material_event_accessions = [
        doc.accession for doc in context.documents if doc.role == "material_event"
    ]
    assert [doc.role for doc in context.documents] == [
        "material_event",
        "quarterly",
        "material_event",
        "annual",
    ]
    assert material_event_accessions == ["0000000001-26-000101", "0000000001-25-000201"]
    assert context.latest_document is not None
    assert context.latest_document.form_type == "8-K"
    assert context.latest_document.accession == "0000000001-26-000101"
    assert context.warnings == []


def test_load_research_filing_context_truncates_material_events(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    annual_path = _write_local_filing(
        tmp_path, "annual.html", "<h1>Item 7</h1><p>Annual filing.</p>"
    )
    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-25-000001",
        form_type="10-K",
        filing_date="2025-02-15",
        period_end="2024-12-31",
        local_path=annual_path,
        primary_doc_url="https://www.sec.gov/Archives/annual.htm",
    )

    for offset in range(26):
        event_path = _write_local_filing(
            tmp_path,
            f"event-{offset}.html",
            f"<h1>Item 8.01</h1><p>Event {offset}</p>",
        )
        day = 18 - offset
        filing_date = f"2026-04-{day:02d}" if day > 0 else f"2026-03-{31 + day:02d}"
        _insert_filing(
            ticker="EXAMPLE",
            cik="0000000001",
            accession=f"0000000001-26-000{offset:03d}",
            form_type="8-K",
            filing_date=filing_date,
            local_path=event_path,
            primary_doc_url=f"https://www.sec.gov/Archives/event-{offset}.htm",
        )

    context = load_research_filing_context("EXAMPLE", as_of_date="2026-04-18", quarters=0)

    assert len([doc for doc in context.documents if doc.role == "material_event"]) == 24
    assert "material_event_filings_truncated:24/26" in context.warnings


def test_load_research_filing_context_warns_on_unreadable_material_event(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    annual_path = _write_local_filing(
        tmp_path, "annual.html", "<h1>Item 7</h1><p>Annual filing.</p>"
    )
    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-25-000001",
        form_type="10-K",
        filing_date="2025-02-15",
        period_end="2024-12-31",
        local_path=annual_path,
        primary_doc_url="https://www.sec.gov/Archives/annual.htm",
    )
    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-26-000401",
        form_type="8-K",
        filing_date="2026-04-10",
        local_path="",
        primary_doc_url="",
    )

    context = load_research_filing_context("EXAMPLE", as_of_date="2026-04-18", quarters=0)

    assert context.latest_document is not None
    assert context.latest_document.form_type == "10-K"
    assert "material_event_filing_unreadable:0000000001-26-000401" in context.warnings


def test_offline_context_rejects_out_of_root_path_and_reads_cache_without_copy(
    monkeypatch,
    tmp_path,
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    cik = "0000000042"
    accession = "0000000042-26-000001"
    primary_url = "https://www.sec.gov/Archives/edgar/data/42/filing.htm"
    outside_path = tmp_path / "untrusted-owner-copy.htm"
    outside_path.write_text("<p>stale out-of-root bytes</p>", encoding="utf-8")
    cached_path = cfg.cache_dir / "filings" / cik / accession / "primary_document.html"
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    trusted_bytes = b"<h1>Risk Factors</h1><p>trusted cached filing</p>"
    cached_path.write_bytes(trusted_bytes)
    _insert_filing(
        ticker="ROOTS",
        cik=cik,
        accession=accession,
        form_type="10-K",
        filing_date="2026-02-15",
        period_end="2025-12-31",
        local_path=str(outside_path),
        primary_doc_url=primary_url,
        content_hash=hashlib.sha256(trusted_bytes).hexdigest(),
    )
    _insert_filing(
        ticker="ROOTS",
        cik=cik,
        accession="0000000042-26-000002",
        form_type="10-Q",
        filing_date="2026-04-01",
        period_end="2026-03-31",
        local_path=None,
        primary_doc_url=("https://www.sec.gov/Archives/edgar/data/42/quarter.htm"),
    )
    raw_target = cfg.raw_filings_dir / cik / accession.replace("-", "") / "filing.htm"

    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("offline filing lookup attempted a write or download")

    monkeypatch.setattr(
        "app.research.filing_context.warm_cached_filing_to_raw",
        unexpected_write,
    )
    monkeypatch.setattr(
        "app.research.filing_context._download_primary_document_to_raw",
        unexpected_write,
    )

    context = load_research_filing_context(
        "ROOTS",
        as_of_date="2026-04-18",
        quarters=1,
        include_material_events=False,
        allow_annual_download=False,
        allow_network_materialization=False,
        cfg=cfg,
        allowed_filing_roots=(cfg.raw_filings_dir, cfg.cache_dir),
    )

    assert len(context.documents) == 1
    document = context.documents[0]
    assert document.local_path == str(cached_path)
    assert document.materialized_from == "filing_cache"
    assert document.html == trusted_bytes.decode()
    assert document.content_revision == hashlib.sha256(trusted_bytes).hexdigest()
    assert context.warnings == [
        "quarterly_filing_unreadable:0000000042-26-000002",
        "quarterly_filings_partial:0/1",
    ]
    assert not raw_target.exists()


def test_offline_context_rejects_cached_filing_hash_mismatch(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    cik = "0000000043"
    accession = "0000000043-26-000001"
    cached_path = cfg.cache_dir / "filings" / cik / accession / "primary_document.html"
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_bytes(b"<p>mutated filing bytes</p>")
    _insert_filing(
        ticker="HASHED",
        cik=cik,
        accession=accession,
        form_type="10-K",
        filing_date="2026-02-15",
        period_end="2025-12-31",
        local_path=str(cached_path),
        primary_doc_url=("https://www.sec.gov/Archives/edgar/data/43/hashed.htm"),
        content_hash=hashlib.sha256(b"<p>expected filing bytes</p>").hexdigest(),
    )

    context = load_research_filing_context(
        "HASHED",
        as_of_date="2026-04-18",
        include_material_events=False,
        allow_annual_download=False,
        cfg=cfg,
        allowed_filing_roots=(cfg.raw_filings_dir, cfg.cache_dir),
    )

    assert context.documents == []
    assert f"annual_filing_hash_mismatch:{accession}" in context.warnings
    assert not (cfg.raw_filings_dir / cik / accession.replace("-", "")).exists()


def test_filing_hash_revision_and_html_use_one_byte_snapshot(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    trusted_bytes = b"<h1>Risk Factors</h1><p>trusted snapshot</p>"
    replacement_bytes = b"<h1>Risk Factors</h1><p>replacement snapshot</p>"
    filing_path = cfg.raw_filings_dir / "snapshot.htm"
    filing_path.parent.mkdir(parents=True, exist_ok=True)
    filing_path.write_bytes(trusted_bytes)
    _insert_filing(
        ticker="SNAPSHOT",
        cik="0000000044",
        accession="0000000044-26-000001",
        form_type="10-K",
        filing_date="2026-02-15",
        period_end="2025-12-31",
        local_path=str(filing_path),
        primary_doc_url=("https://www.sec.gov/Archives/edgar/data/44/snapshot.htm"),
        content_hash=hashlib.sha256(trusted_bytes).hexdigest(),
    )
    original_read_bytes = Path.read_bytes
    snapshot_reads = 0

    def changing_read_bytes(path: Path) -> bytes:
        nonlocal snapshot_reads
        if path == filing_path:
            snapshot_reads += 1
            return trusted_bytes if snapshot_reads == 1 else replacement_bytes
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", changing_read_bytes)

    context = load_research_filing_context(
        "SNAPSHOT",
        as_of_date="2026-04-18",
        include_material_events=False,
        allow_annual_download=False,
        cfg=cfg,
        allowed_filing_roots=(cfg.raw_filings_dir, cfg.cache_dir),
    )

    assert snapshot_reads == 1
    assert len(context.documents) == 1
    assert context.documents[0].html == trusted_bytes.decode()
    assert context.documents[0].content_revision == hashlib.sha256(trusted_bytes).hexdigest()


def test_load_research_filing_context_falls_back_to_older_readable_annual(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    older_annual_path = _write_local_filing(
        tmp_path, "older-annual.html", "<h1>Item 7</h1><p>Older annual filing.</p>"
    )
    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-24-000001",
        form_type="10-K",
        filing_date="2024-02-15",
        period_end="2023-12-31",
        local_path=older_annual_path,
        primary_doc_url="https://www.sec.gov/Archives/older-annual.htm",
    )
    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-25-000001",
        form_type="10-K",
        filing_date="2025-02-15",
        period_end="2024-12-31",
        local_path="",
        primary_doc_url="",
    )

    context = load_research_filing_context("EXAMPLE", as_of_date="2025-04-18", quarters=0)

    assert [doc.accession for doc in context.documents] == ["0000000001-24-000001"]
    assert "annual_filing_unreadable:0000000001-25-000001" in context.warnings
    assert "annual_filing_missing" not in context.warnings


def test_load_research_filing_context_all_unreadable_annuals_do_not_emit_missing(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)

    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-24-000001",
        form_type="10-K",
        filing_date="2024-02-15",
        period_end="2023-12-31",
        local_path="",
        primary_doc_url="",
    )
    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-25-000001",
        form_type="10-K",
        filing_date="2025-02-15",
        period_end="2024-12-31",
        local_path="",
        primary_doc_url="",
    )

    context = load_research_filing_context("EXAMPLE", as_of_date="2025-04-18", quarters=0)

    assert context.documents == []
    assert "annual_filing_unreadable:0000000001-25-000001" in context.warnings
    assert "annual_filing_unreadable:0000000001-24-000001" in context.warnings
    assert "annual_filing_missing" not in context.warnings


def test_load_research_filing_context_quarterly_partial_counts_readable_docs(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    annual_path = _write_local_filing(
        tmp_path, "annual.html", "<h1>Item 7</h1><p>Annual filing.</p>"
    )
    readable_quarter_path = _write_local_filing(
        tmp_path, "readable-quarter.html", "<h1>Item 2</h1><p>Readable quarter.</p>"
    )
    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-25-000001",
        form_type="10-K",
        filing_date="2025-02-15",
        period_end="2024-12-31",
        local_path=annual_path,
        primary_doc_url="https://www.sec.gov/Archives/annual.htm",
    )
    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-26-000001",
        form_type="10-Q",
        filing_date="2026-02-01",
        period_end="2025-12-31",
        local_path=readable_quarter_path,
        primary_doc_url="https://www.sec.gov/Archives/readable-quarter.htm",
    )
    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-25-000901",
        form_type="10-Q",
        filing_date="2025-11-01",
        period_end="2025-09-30",
        local_path="",
        primary_doc_url="",
    )

    context = load_research_filing_context("EXAMPLE", as_of_date="2026-04-18", quarters=2)

    quarter_accessions = [doc.accession for doc in context.documents if doc.role == "quarterly"]
    assert quarter_accessions == ["0000000001-26-000001"]
    assert "quarterly_filing_unreadable:0000000001-25-000901" in context.warnings
    assert "quarterly_filings_partial:1/2" in context.warnings


def test_load_research_filing_context_recovers_missing_local_path_from_sec_primary_document(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.research.filing_context.SecClient.download_bytes",
        lambda self, url, use_cache=True: (
            b"<html><body><p>Recovered quarterly filing text.</p></body></html>"
        ),
    )

    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-26-000001",
        form_type="10-Q",
        filing_date="2026-02-01",
        period_end="2025-12-31",
        local_path=str(tmp_path / "missing-quarter.html"),
        primary_doc_url="https://www.sec.gov/Archives/edgar/data/1/000000000126000001/example-10q.htm",
        status="download_error",
    )

    context = load_research_filing_context("EXAMPLE", as_of_date="2026-04-18", quarters=1)

    assert [doc.accession for doc in context.documents] == ["0000000001-26-000001"]
    assert context.documents[0].materialized_from == "sec_primary_document_fetch"
    assert "Recovered quarterly filing text" in context.documents[0].html
    assert context.recent_filing_status == "RECENT_FILING_CONTEXT_AVAILABLE"
    assert context.recovered_document_count == 1
    assert context.warnings == ["annual_filing_missing"]


def test_load_research_filing_context_distinguishes_unreadable_recent_filings(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)

    def _raise_fetch(self, url, use_cache=True):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("app.research.filing_context.SecClient.download_bytes", _raise_fetch)
    _insert_filing(
        ticker="EXAMPLE",
        cik="0000000001",
        accession="0000000001-26-000001",
        form_type="10-Q",
        filing_date="2026-02-01",
        period_end="2025-12-31",
        local_path=str(tmp_path / "missing-quarter.html"),
        primary_doc_url="https://www.sec.gov/Archives/edgar/data/1/000000000126000001/example-10q.htm",
        status="download_error",
    )

    context = load_research_filing_context("EXAMPLE", as_of_date="2026-04-18", quarters=1)

    assert context.documents == []
    assert context.recent_filing_status == "RECENT_FILINGS_UNREADABLE"
    assert context.recovered_document_count == 0
    assert "quarterly_filing_unreadable:0000000001-26-000001" in context.warnings


@pytest.mark.parametrize("form_type", ["20-F", "20-F/A", "40-F", "40-F/A"])
def test_load_research_filing_context_recovers_cached_foreign_annual_by_cik_alias(
    monkeypatch,
    tmp_path,
    form_type,
):
    _init_temp_db(monkeypatch, tmp_path)
    annual_path = _write_local_filing(
        tmp_path,
        f"{form_type.replace('/', '-').lower()}.html",
        "<h1>Risk Factors</h1><p>Cached foreign annual filing.</p>",
    )
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
                '["PRIMARY", "ADR"]',
                "US_EXCHANGE",
                "OPERATING",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
    _insert_filing(
        ticker="PRIMARY",
        cik="42",
        accession="0000000042-26-000001",
        form_type=form_type,
        filing_date="2026-03-01",
        period_end="2025-12-31",
        local_path=annual_path,
        primary_doc_url="https://www.sec.gov/Archives/foreign-annual.htm",
    )

    context = load_research_filing_context(
        "ADR",
        as_of_date="2026-04-01",
        issuer_aware=True,
        quarters=0,
    )

    assert context.issuer_cik == "42"
    assert context.resolved_aliases == ("ADR", "PRIMARY")
    assert len(context.documents) == 1
    assert context.documents[0].form_type == form_type
    assert context.documents[0].ticker == "PRIMARY"
    assert context.documents[0].accession == "0000000042-26-000001"


def test_default_v1_context_does_not_expand_to_issuer_alias(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    annual_path = _write_local_filing(
        tmp_path,
        "primary-only-20f.html",
        "<h1>Risk Factors</h1><p>Primary issuer filing.</p>",
    )
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO sec_registrants(
                cik, primary_ticker, all_tickers, exchange_scope,
                operating_status, first_seen_at, last_seen_at
            ) VALUES('42', 'PRIMARY', '["PRIMARY", "ADR"]',
                     'US_EXCHANGE', 'OPERATING', 'x', 'x')
            """
        )
    _insert_filing(
        ticker="PRIMARY",
        cik="42",
        accession="primary-only",
        form_type="20-F",
        filing_date="2026-03-01",
        period_end="2025-12-31",
        local_path=annual_path,
        primary_doc_url="https://www.sec.gov/Archives/primary-only.htm",
    )

    legacy = load_research_filing_context(
        "ADR",
        as_of_date="2026-04-01",
        quarters=0,
        include_material_events=False,
    )
    v2 = load_research_filing_context(
        "ADR",
        as_of_date="2026-04-01",
        issuer_aware=True,
        quarters=0,
        include_material_events=False,
        allow_annual_download=False,
    )

    assert legacy.documents == []
    assert legacy.issuer_cik is None
    assert v2.issuer_cik == "42"
    assert [document.accession for document in v2.documents] == ["primary-only"]


def test_annual_prefers_older_readable_alias_before_newer_missing_path(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    older_path = _write_local_filing(
        tmp_path,
        "older-readable-20f.html",
        "<h1>Risk Factors</h1><p>Readable cached alias filing.</p>",
    )
    _insert_filing(
        ticker="ADR",
        cik="42",
        accession="0000000042-26-000002",
        form_type="20-F",
        filing_date="2026-03-15",
        period_end="2025-12-31",
        local_path="",
        primary_doc_url=(
            "https://www.sec.gov/Archives/edgar/data/42/000000004226000002/newer-20f.htm"
        ),
        status="download_error",
    )
    _insert_filing(
        ticker="PRIMARY",
        cik="42",
        accession="0000000042-25-000001",
        form_type="20-F",
        filing_date="2025-03-01",
        period_end="2024-12-31",
        local_path=older_path,
        primary_doc_url="https://www.sec.gov/Archives/older-20f.htm",
    )
    download_calls: list[str] = []

    def _download_must_not_run(self, url, use_cache=True):
        download_calls.append(url)
        raise AssertionError("cached annual should satisfy the default limit")

    monkeypatch.setattr(
        "app.research.filing_context.SecClient.download_bytes",
        _download_must_not_run,
    )

    context = load_research_filing_context(
        "ADR",
        as_of_date="2026-04-01",
        issuer_aware=True,
        annual_filing_limit=12,
        quarters=0,
        include_material_events=False,
        allow_annual_download=False,
    )

    assert download_calls == []
    assert [document.accession for document in context.documents] == ["0000000042-25-000001"]
    assert context.documents[0].ticker == "PRIMARY"
    assert context.documents[0].form_type == "20-F"
    assert "annual_filing_unreadable:0000000042-26-000002" in context.warnings


def test_failed_annual_materialization_keeps_next_cached_alias_candidate(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    cached_path = _write_local_filing(
        tmp_path,
        "cached-prior-full.html",
        "<h1>Risk Factors</h1><p>Cached prior full annual.</p>",
    )
    _insert_filing(
        ticker="ADR",
        cik="84",
        accession="0000000084-26-000002",
        form_type="20-F/A",
        filing_date="2026-03-20",
        period_end="2025-12-31",
        local_path="",
        primary_doc_url=(
            "https://www.sec.gov/Archives/edgar/data/84/000000008426000002/amended-20f.htm"
        ),
        status="download_error",
    )
    _insert_filing(
        ticker="PRIMARY",
        cik="84",
        accession="0000000084-26-000001",
        form_type="20-F",
        filing_date="2026-03-01",
        period_end="2025-12-31",
        local_path=cached_path,
        primary_doc_url="https://www.sec.gov/Archives/full-20f.htm",
    )
    download_calls: list[str] = []

    def _fail_download(self, url, use_cache=True):
        download_calls.append(url)
        raise RuntimeError("offline replay")

    monkeypatch.setattr(
        "app.research.filing_context.SecClient.download_bytes",
        _fail_download,
    )

    context = load_research_filing_context(
        "ADR",
        as_of_date="2026-04-01",
        issuer_aware=True,
        annual_filing_limit=2,
        quarters=0,
        include_material_events=False,
    )

    assert len(download_calls) == 1
    assert [document.accession for document in context.documents] == ["0000000084-26-000001"]
    assert context.documents[0].form_type == "20-F"
    assert "annual_filing_unreadable:0000000084-26-000002" in context.warnings
