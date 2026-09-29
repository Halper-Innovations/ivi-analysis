from __future__ import annotations

import json
from datetime import datetime, timezone

from app.db import get_db, init_db, utc_now_iso
from app.market.shares_provider import FilingsSharesProvider, SharesCache, SharesSnapshot


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _insert_shares_fact(
    *,
    ticker: str,
    filing_date: str,
    shares_value: int | float | str,
    section_label: str = "cover_page",
    accession: str = "0000000000-26-000001",
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end, primary_doc_url,
                local_path, hash, ingested_as_of, status, created_at, updated_at
            )
            VALUES(?, ?, ?, '10-K', ?, ?, ?, '', '', ?, 'parsed', ?, ?)
            """,
            (
                "0000320193",
                ticker,
                accession,
                filing_date,
                filing_date,
                f"https://www.sec.gov/Archives/{accession}",
                filing_date,
                now,
                now,
            ),
        )
        filing_id = int(conn.execute("SELECT id FROM filings WHERE accession = ?", (accession,)).fetchone()["id"])
        conn.execute(
            """
            INSERT INTO extracted_facts(
                filing_id, fact_type, value_json, source_url, snippet, section_label, created_at
            ) VALUES(?, 'shares_outstanding', ?, ?, ?, ?, ?)
            """,
            (
                filing_id,
                json.dumps({"value": shares_value}),
                f"https://www.sec.gov/Archives/{accession}",
                "shares outstanding snippet",
                section_label,
                now,
            ),
        )


def test_shares_provider_cache_precedence(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    cache = SharesCache(cfg)
    cache.store(
        "AAA",
        requested_as_of_date="2026-02-14",
        snapshot=SharesSnapshot(
            ticker="AAA",
            as_of_date="2026-02-13",
            shares_outstanding=50.0,
            source="disk_cache_seed",
            retrieved_at=utc_now_iso(),
            confidence="HIGH",
            resolved_via="CACHED",
        ),
    )
    _insert_shares_fact(
        ticker="AAA",
        filing_date="2026-02-12",
        shares_value=70,
        section_label="cover_page",
    )
    run_id = "shares_cache_precedence"
    run_dir = cfg.outputs_dir / "shares" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "AAA.json").write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "requested_as_of_date": "2026-02-14",
                "status": "OK",
                "snapshot": {
                    "ticker": "AAA",
                    "as_of_date": "2026-02-14",
                    "shares_outstanding": 60.0,
                    "unit": "shares",
                    "source": "run_scoped_seed",
                    "retrieved_at": utc_now_iso(),
                    "confidence": "HIGH",
                    "resolved_via": "CACHED",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    provider = FilingsSharesProvider(cfg)
    scoped = provider.get_shares_asof("AAA", "2026-02-14", run_id=run_id)
    assert scoped is not None
    assert scoped.shares_outstanding == 60.0
    scoped_diag = provider.get_last_diagnostic("AAA", "2026-02-14", run_id=run_id)
    assert scoped_diag is not None
    assert scoped_diag["result"]["reason_code"] == "CACHE_HIT"
    assert scoped_diag["cache"]["hit"] is True

    cached = provider.get_shares_asof("AAA", "2026-02-14")
    assert cached is not None
    assert cached.shares_outstanding == 50.0
    cached_diag = provider.get_last_diagnostic("AAA", "2026-02-14")
    assert cached_diag is not None
    assert cached_diag["result"]["reason_code"] == "CACHE_HIT"
    assert cached_diag["cache"]["path"].endswith("data/cache/shares/AAA.json")


def test_shares_provider_reason_codes(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    provider = FilingsSharesProvider(cfg)

    missing = provider.get_shares_asof("ZZZ", "2026-02-14")
    assert missing is None
    missing_diag = provider.get_last_diagnostic("ZZZ", "2026-02-14")
    assert missing_diag is not None
    assert missing_diag["result"]["reason_code"] == "NO_FILINGS"

    _insert_shares_fact(
        ticker="YYY",
        filing_date="2026-02-10",
        shares_value="N/A",
        section_label="cover_page",
        accession="0000000000-26-000002",
    )
    _insert_shares_fact(
        ticker="YYY",
        filing_date="2026-02-10",
        shares_value="N/A",
        section_label="xbrl",
        accession="0000000000-26-000003",
    )
    bad = provider.get_shares_asof("YYY", "2026-02-14")
    assert bad is None
    bad_diag = provider.get_last_diagnostic("YYY", "2026-02-14")
    assert bad_diag is not None
    assert bad_diag["result"]["reason_code"] in {"COVER_PAGE_PARSE_MISS", "XBRL_MISS"}
    statuses = [row.get("status") for row in (bad_diag.get("provider_attempts") or [])]
    assert "CACHE_MISS" in statuses
