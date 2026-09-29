from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterator

from app.config import get_config
from app.db import get_db, utc_now_iso
from app.dossier.filing_cache import warm_cached_filing_to_raw
from app.ingest.sec_client import FilingStub, SecClient
from app.logging import get_logger
from app.parse.document_store import file_hash, filing_local_path
from app.util.financial_data_access import (
    ANNUAL_CACHED_FILING_FORM_TYPES,
    MATERIAL_EVENT_CACHED_FILING_FORM_TYPES,
    QUARTERLY_CACHED_FILING_FORM_TYPES,
)


logger = get_logger(__name__)


@contextmanager
def _filing_db(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
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


DEFAULT_FILING_POLICY = {
    "10-K": "filing_lookback_days_10k",
    "10-K/A": "filing_lookback_days_10k",
    "10-Q": "filing_lookback_days_10q",
    "10-Q/A": "filing_lookback_days_10q",
    "8-K": "filing_lookback_days_8k",
    "8-K/A": "filing_lookback_days_8k",
    "20-F": "filing_lookback_days_20f",
    "20-F/A": "filing_lookback_days_20f",
    "40-F": "filing_lookback_days_20f",
    "40-F/A": "filing_lookback_days_20f",
    "DEF 14A": "filing_lookback_days_def14a",
}


def default_filing_windows_days() -> dict[str, int]:
    cfg = get_config()
    return {form: int(getattr(cfg, attr)) for form, attr in DEFAULT_FILING_POLICY.items()}


def list_form4_stubs(cik: str, since: date) -> list[FilingStub]:
    """Read Form-4 filing stubs from the cached SEC submissions JSON only.

    This is a lightweight, network-free reader for the catalyst overlay. It does
    NOT enroll Form 4 into DEFAULT_FILING_POLICY / _download_filing, so the heavy
    ingest path is untouched. Returns FilingStub rows (form_type == '4') filed on
    or after ``since``, sorted filing_date desc then accession desc.
    """
    client = SecClient()
    return client.list_cached_filings_window(
        cik,
        start_date=since,
        end_date=date.max,
        forms=["4"],
    )


def _companies_for_active_universe(
    conn, tickers: list[str] | None = None, limit: int | None = None
) -> list:
    tickers_set = {t.upper() for t in (tickers or []) if t.strip()}
    if tickers_set:
        # An explicit ticker scope names the work outright: resolve those
        # companies directly. Routing the request through the active-universe
        # snapshot silently no-opped for any name outside it — census-discovered
        # registrants and evidence-resolution repairs were dropped this way.
        rows = conn.execute("SELECT ticker, cik FROM companies ORDER BY ticker").fetchall()
        rows = [row for row in rows if str(row["ticker"]).upper() in tickers_set]
        if limit is not None and limit > 0:
            rows = rows[:limit]
        return rows
    state_row = conn.execute(
        "SELECT value_json FROM state WHERE key = 'active_universe'"
    ).fetchone()
    if not state_row:
        rows = conn.execute("SELECT ticker, cik FROM companies ORDER BY ticker").fetchall()
        if tickers_set:
            rows = [row for row in rows if str(row["ticker"]).upper() in tickers_set]
        if limit is not None and limit > 0:
            rows = rows[:limit]
        return rows
    try:
        payload = json.loads(state_row["value_json"])
        universe_id = payload.get("universe_id")
    except Exception:
        universe_id = None
    if not universe_id:
        rows = conn.execute("SELECT ticker, cik FROM companies ORDER BY ticker").fetchall()
        if tickers_set:
            rows = [row for row in rows if str(row["ticker"]).upper() in tickers_set]
        if limit is not None and limit > 0:
            rows = rows[:limit]
        return rows
    rows = conn.execute(
        "SELECT ticker, cik FROM companies WHERE universe_id = ?", (universe_id,)
    ).fetchall()
    if rows:
        selected = rows
    else:
        selected = conn.execute("SELECT ticker, cik FROM companies ORDER BY ticker").fetchall()
    if tickers_set:
        selected = [row for row in selected if str(row["ticker"]).upper() in tickers_set]
    if limit is not None and limit > 0:
        selected = selected[:limit]
    return selected


def select_filing_stubs_for_policy(
    filings: list[FilingStub],
    *,
    as_of_date: date,
    windows_days: dict[str, int],
) -> list[FilingStub]:
    selected_by_form: dict[str, FilingStub] = {}
    for filing in filings:
        form = filing.form_type.upper().strip()
        if form not in windows_days:
            continue
        if filing.filing_date > as_of_date:
            continue
        lookback = max(1, int(windows_days[form]))
        age_days = (as_of_date - filing.filing_date).days
        if age_days > lookback:
            continue
        existing = selected_by_form.get(form)
        if existing is None or filing.filing_date > existing.filing_date:
            selected_by_form[form] = filing

    selected = list(selected_by_form.values())
    selected.sort(key=lambda f: f.filing_date, reverse=True)
    return selected


def _coverage_score(forms_included: set[str]) -> tuple[float, list[str]]:
    score = 100.0
    missing_required: list[str] = []
    has_annual = bool(forms_included & set(ANNUAL_CACHED_FILING_FORM_TYPES))
    has_quarter = bool(forms_included & set(QUARTERLY_CACHED_FILING_FORM_TYPES))
    has_material_event = bool(forms_included & set(MATERIAL_EVENT_CACHED_FILING_FORM_TYPES))
    if not has_annual:
        score -= 40.0
        missing_required.append("10-K_OR_20-F")
    if not has_quarter:
        score -= 30.0
        missing_required.append("10-Q")
    if not has_material_event:
        score -= 15.0
    if "DEF 14A" not in forms_included:
        score -= 5.0
    return max(0.0, score), missing_required


def _persist_filing_coverage(
    conn,
    *,
    ticker: str,
    run_id: str,
    as_of_date: str,
    selected: list[FilingStub],
) -> None:
    forms = sorted({f.form_type.upper().strip() for f in selected})
    accessions = sorted({f.accession for f in selected})
    coverage_score, missing_required = _coverage_score(set(forms))
    conn.execute(
        """
        INSERT INTO filing_coverage(
            ticker, run_id, as_of_date, forms_included_json, accession_numbers_json,
            coverage_score, missing_required_json, created_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, run_id, as_of_date) DO UPDATE SET
            forms_included_json=excluded.forms_included_json,
            accession_numbers_json=excluded.accession_numbers_json,
            coverage_score=excluded.coverage_score,
            missing_required_json=excluded.missing_required_json,
            created_at=excluded.created_at
        """,
        (
            ticker,
            run_id,
            as_of_date,
            json.dumps(forms),
            json.dumps(accessions),
            coverage_score,
            json.dumps(missing_required),
            utc_now_iso(),
        ),
    )


def _upsert_filing(conn, ticker: str, filing: FilingStub, ingested_as_of: str) -> int:
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO filings(
            cik, ticker, accession, form_type, filing_date, period_end,
            primary_doc_url, ingested_as_of, status, created_at, updated_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, 'new', ?, ?)
        ON CONFLICT(cik, accession) DO UPDATE SET
            form_type=excluded.form_type,
            filing_date=excluded.filing_date,
            period_end=excluded.period_end,
            primary_doc_url=excluded.primary_doc_url,
            ingested_as_of=excluded.ingested_as_of,
            updated_at=excluded.updated_at
        """,
        (
            filing.cik,
            ticker,
            filing.accession,
            filing.form_type,
            filing.filing_date.isoformat(),
            filing.period_end,
            filing.primary_doc_url,
            ingested_as_of,
            now,
            now,
        ),
    )
    row = conn.execute(
        "SELECT id FROM filings WHERE cik = ? AND accession = ?",
        (filing.cik, filing.accession),
    ).fetchone()
    return int(row["id"])


def _download_filing(conn, client: SecClient, filing_id: int, filing: FilingStub) -> str | None:
    cfg = get_config()
    parsed_row = conn.execute(
        "SELECT parsed_at FROM parsed_filings WHERE accession = ? LIMIT 1",
        (filing.accession,),
    ).fetchone()
    if parsed_row:
        existing = conn.execute(
            "SELECT local_path FROM filings WHERE id = ? LIMIT 1",
            (filing_id,),
        ).fetchone()
        if existing and existing["local_path"] and Path(existing["local_path"]).exists():
            conn.execute(
                "UPDATE filings SET status='parsed', updated_at=? WHERE id=?",
                (utc_now_iso(), filing_id),
            )
            return str(existing["local_path"])

    doc_name = filing.primary_document or SecClient.filename_from_url(filing.primary_doc_url)
    local_path = filing_local_path(filing.cik, filing.accession, doc_name)
    index_path = local_path.parent / "index.json"

    existing = conn.execute(
        "SELECT status, local_path FROM filings WHERE id = ? LIMIT 1",
        (filing_id,),
    ).fetchone()
    if (
        existing
        and existing["status"] == "parsed"
        and existing["local_path"]
        and Path(existing["local_path"]).exists()
    ):
        return str(existing["local_path"])
    if warm_cached_filing_to_raw(filing=filing, target_path=local_path):
        digest = file_hash(local_path)
        conn.execute(
            "UPDATE filings SET local_path=?, hash=?, status='downloaded', updated_at=? WHERE id=?",
            (str(local_path), digest, utc_now_iso(), filing_id),
        )
        return str(local_path)
    net_disabled = str(getattr(cfg, "net_provider", "enabled")).strip().lower() == "disabled"
    if net_disabled:
        conn.execute(
            "UPDATE filings SET status='download_error', updated_at=?, hash=NULL WHERE id=?",
            (utc_now_iso(), filing_id),
        )
        return None

    try:
        # Pull index.json for document inventory and reproducibility.
        index_raw = client.download_bytes(filing.filing_index_url, use_cache=True)
        index_path.write_bytes(index_raw)
    except Exception as exc:
        logger.info(
            "filing_index_download_failed",
            extra={"stage_name": "ingest", "stage_error": str(exc), "stage_filing_id": filing_id},
        )

    try:
        payload = client.download_bytes(filing.primary_doc_url, use_cache=True)
    except Exception as exc:
        conn.execute(
            "UPDATE filings SET status='download_error', updated_at=?, hash=NULL WHERE id=?",
            (utc_now_iso(), filing_id),
        )
        logger.error(
            "filing_download_failed",
            extra={"stage_name": "ingest", "stage_error": str(exc), "stage_filing_id": filing_id},
        )
        return None

    local_path.write_bytes(payload)
    digest = file_hash(local_path)
    conn.execute(
        "UPDATE filings SET local_path=?, hash=?, status='downloaded', updated_at=? WHERE id=?",
        (str(local_path), digest, utc_now_iso(), filing_id),
    )
    return str(local_path)


def ingest_since(
    since: date,
    forms: list[str],
    as_of_date: str | None = None,
    tickers: list[str] | None = None,
    limit: int | None = None,
) -> int:
    as_of_date = as_of_date or date.today().isoformat()
    client = SecClient()
    inserted = 0
    with get_db() as conn:
        rows = _companies_for_active_universe(conn, tickers=tickers, limit=limit)
        for company in rows:
            ticker = company["ticker"]
            cik = company["cik"]
            filings = client.list_recent_filings(cik, since, forms)
            for filing in filings:
                filing_id = _upsert_filing(conn, ticker, filing, as_of_date)
                _download_filing(conn, client, filing_id, filing)
                inserted += 1
    logger.info("ingest_completed", extra={"stage_name": "ingest", "stage_count": inserted})
    return inserted


def ingest_with_policy(
    *,
    as_of_date: str,
    run_id: str,
    tickers: list[str] | None = None,
    limit: int | None = None,
    windows_days: dict[str, int] | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    as_of = date.fromisoformat(as_of_date)
    windows = windows_days or default_filing_windows_days()
    since = as_of - date.resolution
    if windows:
        max_window = max(int(v) for v in windows.values())
        since = date.fromordinal(as_of.toordinal() - max_window)

    client = SecClient()
    inserted = 0
    selected_total = 0
    downloaded_total = 0
    download_error_total = 0
    with _filing_db(db_path) as conn:
        rows = _companies_for_active_universe(conn, tickers=tickers, limit=limit)
        for company in rows:
            ticker = company["ticker"]
            cik = company["cik"]
            raw = client.list_recent_filings(cik, since, list(windows.keys()))
            selected = select_filing_stubs_for_policy(raw, as_of_date=as_of, windows_days=windows)
            _persist_filing_coverage(
                conn, ticker=ticker, run_id=run_id, as_of_date=as_of_date, selected=selected
            )
            selected_total += len(selected)
            for filing in selected:
                filing_id = _upsert_filing(conn, ticker, filing, as_of_date)
                local_path = _download_filing(conn, client, filing_id, filing)
                if local_path is None:
                    download_error_total += 1
                else:
                    downloaded_total += 1
                inserted += 1

    logger.info("ingest_policy_completed", extra={"stage_name": "ingest", "stage_count": inserted})
    return {
        "tickers": len(tickers or []),
        "filings_considered": selected_total,
        "filings_upserted": inserted,
        "filings_downloaded": downloaded_total,
        "filing_download_errors": download_error_total,
    }


def ingest_filings_between(
    since: date,
    until: date,
    forms: list[str],
    as_of_date: str | None = None,
    tickers: list[str] | None = None,
    limit: int | None = None,
) -> int:
    as_of_date = as_of_date or date.today().isoformat()
    client = SecClient()
    count = 0
    with get_db() as conn:
        rows = _companies_for_active_universe(conn, tickers=tickers, limit=limit)
        for company in rows:
            filings = client.list_recent_filings(company["cik"], since, forms)
            for filing in filings:
                if filing.filing_date > until:
                    continue
                filing_id = _upsert_filing(conn, company["ticker"], filing, as_of_date)
                _download_filing(conn, client, filing_id, filing)
                count += 1
    return count


def download_filing_by_id(filing_id: int) -> bool:
    client = SecClient()
    with get_db() as conn:
        row = conn.execute(
            "SELECT cik, accession, form_type, filing_date, period_end, primary_doc_url FROM filings WHERE id = ?",
            (filing_id,),
        ).fetchone()
        if not row:
            return False

        filing_date = date.fromisoformat(row["filing_date"]) if row["filing_date"] else date.today()
        doc_name = SecClient.filename_from_url(row["primary_doc_url"])
        filing = FilingStub(
            cik=row["cik"],
            accession=row["accession"],
            accession_nodash=str(row["accession"]).replace("-", ""),
            form_type=row["form_type"],
            filing_date=filing_date,
            period_end=row["period_end"],
            primary_document=doc_name,
            primary_doc_url=row["primary_doc_url"],
            filing_index_url=f"https://www.sec.gov/Archives/edgar/data/{int(row['cik'])}/{str(row['accession']).replace('-', '')}/index.json",
        )
        local = _download_filing(conn, client, filing_id, filing)
        return local is not None
