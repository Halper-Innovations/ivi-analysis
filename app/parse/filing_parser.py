from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import get_config
from app.db import get_db, utc_now_iso
from app.logging import get_logger
from app.parse.extractors.advanced_signals import (
    extract_cash_flow_quality_signals,
    extract_customer_concentration_signal,
    extract_debt_maturity_and_covenants,
    extract_non_gaap_reconciliation_signals,
    extract_sbc_dilution_signals,
    extract_segment_signals,
)
from app.parse.extractors.cover_page import extract_cover_page_facts
from app.parse.extractors.debt_and_liquidity import extract_debt_liquidity_signals
from app.parse.extractors.footnotes_signals import extract_footnote_signals
from app.util.financial_data_access import normalize_cik


logger = get_logger(__name__)


def _load_text(local_path: str) -> str | None:
    path = Path(local_path)
    if not path.exists() or not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


@contextmanager
def _parser_db(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    if db_path is None:
        with get_db() as conn:
            yield conn
        return
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def parse_pending_filings(
    limit: int = 50,
    tickers: list[str] | None = None,
    *,
    db_path: str | Path | None = None,
    issuer_cik: str | None = None,
    form_types: list[str] | tuple[str, ...] | None = None,
    as_of_date: str | None = None,
) -> int:
    parsed = 0
    tickers = [t.upper() for t in (tickers or []) if t.strip()]
    normalized_cik = normalize_cik(issuer_cik)
    normalized_forms = [
        str(form).strip().upper() for form in (form_types or ()) if str(form).strip()
    ]
    with _parser_db(db_path) as conn:
        clauses = ["status IN ('downloaded', 'new')"]
        params: list[object] = []
        if normalized_cik is not None:
            clauses.append("CAST(cik AS INTEGER) = ?")
            params.append(int(normalized_cik))
        elif tickers:
            placeholders = ",".join("?" for _ in tickers)
            clauses.append(f"ticker IN ({placeholders})")
            params.extend(tickers)
        if normalized_forms:
            placeholders = ",".join("?" for _ in normalized_forms)
            clauses.append(f"UPPER(form_type) IN ({placeholders})")
            params.extend(normalized_forms)
        if as_of_date:
            clauses.append("filing_date <= ?")
            params.append(str(as_of_date)[:10])
        rows = conn.execute(
            f"""
            SELECT id, local_path, primary_doc_url
            FROM filings
            WHERE {' AND '.join(clauses)}
            ORDER BY filing_date DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        filing_ids = [row["id"] for row in rows]

    for filing_id in filing_ids:
        if parse_filing_by_id(filing_id, db_path=db_path):
            parsed += 1

    logger.info("parse_completed", extra={"stage_name": "parse", "stage_count": parsed})
    return parsed


def parse_filing_by_id(
    filing_id: int,
    *,
    max_retries: int | None = None,
    retry_backoff_seconds: float | None = None,
    db_path: str | Path | None = None,
) -> bool:
    cfg = get_config()
    retries = int(max_retries if max_retries is not None else 3)
    retries = max(0, retries)
    backoff = float(
        retry_backoff_seconds
        if retry_backoff_seconds is not None
        else max(0.1, cfg.sec_backoff_seconds)
    )
    for attempt in range(retries + 1):
        try:
            return _parse_filing_by_id_once(filing_id, db_path=db_path)
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if "database is locked" not in message or attempt >= retries:
                raise
            sleep_s = backoff * (2**attempt)
            logger.warning(
                "parse_retry_sqlite_lock",
                extra={
                    "stage_name": "parse",
                    "stage_filing_id": filing_id,
                    "stage_attempt": attempt + 1,
                    "stage_sleep_seconds": round(sleep_s, 4),
                },
            )
            time.sleep(sleep_s)
    return False


def _parse_filing_by_id_once(
    filing_id: int,
    *,
    db_path: str | Path | None = None,
) -> bool:
    with _parser_db(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, accession, hash, local_path, primary_doc_url
            FROM filings
            WHERE id = ?
            """,
            (filing_id,),
        ).fetchone()
        if not row:
            return False

        parsed = conn.execute(
            "SELECT parsed_at FROM parsed_filings WHERE accession = ? LIMIT 1",
            (row["accession"],),
        ).fetchone()
        if parsed:
            conn.execute(
                "UPDATE filings SET status='parsed', updated_at=? WHERE id=?",
                (utc_now_iso(), filing_id),
            )
            return False

        local_path = row["local_path"]
        if not local_path:
            conn.execute(
                "UPDATE filings SET status='parse_skipped', updated_at=? WHERE id=?",
                (utc_now_iso(), filing_id),
            )
            return False

        text = _load_text(local_path)
        if not text:
            conn.execute(
                "UPDATE filings SET status='parse_skipped', updated_at=? WHERE id=?",
                (utc_now_iso(), filing_id),
            )
            return False

        source_url = row["primary_doc_url"]
        facts = []
        facts.extend(extract_cover_page_facts(text, source_url))
        facts.extend(extract_footnote_signals(text, source_url))
        facts.extend(extract_debt_liquidity_signals(text, source_url))
        facts.extend(extract_segment_signals(text, source_url))
        facts.extend(extract_sbc_dilution_signals(text, source_url))
        facts.extend(extract_debt_maturity_and_covenants(text, source_url))
        facts.extend(extract_customer_concentration_signal(text, source_url))
        facts.extend(extract_cash_flow_quality_signals(text, source_url))
        facts.extend(extract_non_gaap_reconciliation_signals(text, source_url))

        conn.execute("DELETE FROM extracted_facts WHERE filing_id = ?", (filing_id,))
        conn.execute("DELETE FROM financials WHERE filing_id = ?", (filing_id,))

        for fact in facts:
            conn.execute(
                """
                INSERT INTO extracted_facts(
                    filing_id, fact_type, value_json, source_url, snippet,
                    section_label, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    filing_id,
                    fact["fact_type"],
                    json.dumps(fact["value_json"]),
                    fact["source_url"],
                    fact.get("snippet"),
                    fact.get("section_label"),
                    utc_now_iso(),
                ),
            )

        conn.execute(
            "UPDATE filings SET status='parsed', updated_at=? WHERE id=?",
            (utc_now_iso(), filing_id),
        )
        conn.execute(
            """
            INSERT INTO parsed_filings(accession, parsed_at, content_hash)
            VALUES(?, ?, ?)
            ON CONFLICT(accession) DO UPDATE SET
                parsed_at=excluded.parsed_at,
                content_hash=excluded.content_hash
            """,
            (row["accession"], utc_now_iso(), row["hash"]),
        )
        return True
