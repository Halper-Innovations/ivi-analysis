from __future__ import annotations

import json

from app.db import get_db, utc_now_iso
from app.ingest.cik_registry import resolve
from app.ingest.companyfacts import (
    TAG_MAP,
    _normalization_cutoff_year,
    fetch_annual_facts,
    fetch_quarterly_facts,
    normalize_annual_facts_from_raw,
    normalize_quarterly_facts_from_raw,
)
from app.market.company_facts_provider import companyfacts_cache_path
from app.util.financial_data_access import (
    ANNUAL_COMPANYFACTS_PERIOD_TYPES,
    companyfacts_is_fresh,
)

_TTL_SECONDS = 24 * 3600
# The companyfacts_facts.written_by marker for rows this writer upserts.
WRITTEN_BY = "facts_writer"


def _is_fresh(ticker: str) -> bool:
    """Return True if annual companyfacts_facts data for this ticker is fresh."""
    with get_db() as conn:
        return companyfacts_is_fresh(
            conn,
            ticker.upper(),
            ttl_seconds=_TTL_SECONDS,
            period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
        )


def _record_vintage(conn, ticker: str, fact: dict, now: str) -> None:
    """Append the as-reported value to the vintage table (idempotent on
    the (key, filed_date, value) tuple; restatements append, re-ingests
    dedupe). Best-effort — vintage archival never blocks the fact write."""
    try:
        conn.execute(
            """INSERT OR IGNORE INTO companyfacts_vintages
               (ticker, fiscal_year, period_type, period_end, line_item, value,
                units, filed_date, form, accession, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ticker,
                fact["fiscal_year"],
                fact.get("period_type", "FY"),
                fact.get("period_end"),
                fact["line_item"],
                fact.get("value"),
                fact.get("units"),
                str(fact.get("filed_date") or ""),
                fact.get("form"),
                fact.get("accession"),
                now,
            ),
        )
    except Exception:  # noqa: BLE001
        pass


_QUARTERLY_PERIOD_TYPES = ("Q1", "Q2", "Q3", "Q4")


def _purge_stale_rows(
    conn, ticker: str, *, period_types: tuple[str, ...], years_back: int
) -> None:
    """Delete this ticker's rows the normalizer covers, for the refreshed periods.

    The upsert only touches rows the normalizer emits now, so a row it has since
    refused (a guard-refused share count, a debt total now UNKNOWN) or relabelled
    (a quarterly period) would linger in an existing store. Scope: this ticker,
    the period types of this refresh, fiscal years inside the refresh window, and
    the line items the normalizer maps, and only rows this writer owns: a row the
    point-in-time repair writer (app/autonomous/companyfacts_repair.py) wrote is
    marked ``written_by = 'companyfacts_repair'`` and kept (it used to be deleted
    with the rest). A row from before the marker existed (NULL) is treated as this
    writer's, as before. It runs in the caller's transaction, ahead of the inserts,
    so a failed write rolls the purge back with it.
    """
    cutoff_year = _normalization_cutoff_year(years_back=years_back, filed_as_of=None)
    types = ",".join("?" for _ in period_types)
    items = ",".join("?" for _ in TAG_MAP)
    conn.execute(
        f"""DELETE FROM companyfacts_facts
            WHERE ticker = ? AND period_type IN ({types}) AND fiscal_year >= ?
              AND line_item IN ({items})
              AND COALESCE(written_by, ?) = ?""",
        (ticker, *period_types, cutoff_year, *TAG_MAP, WRITTEN_BY, WRITTEN_BY),
    )


def ensure_facts(ticker: str, years_back: int = 10) -> None:
    """
    Guarantee companyfacts_facts is populated and fresh for ticker.
    No-ops if data is within TTL. Fetches, normalizes, and upserts otherwise.
    """
    upper = ticker.upper().strip()
    if _is_fresh(upper):
        return
    cik = resolve(upper)
    facts = _facts_from_local_cache(cik, years_back=years_back)
    if not facts:
        facts = fetch_annual_facts(cik, years_back=years_back)
    now = utc_now_iso()
    with get_db() as conn:
        # An empty result is indistinguishable from a failed fetch: never wipe on it.
        if facts:
            _purge_stale_rows(
                conn, upper, period_types=tuple(ANNUAL_COMPANYFACTS_PERIOD_TYPES), years_back=years_back
            )
        for fact in facts:
            period_type = fact.get("period_type", "FY")
            conn.execute(
                """INSERT INTO companyfacts_facts
                   (ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at,
                    filed_date, form, accession, source_tags, written_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, fiscal_year, period_type, line_item)
                   DO UPDATE SET value=excluded.value, period_end=excluded.period_end,
                                 fetched_at=excluded.fetched_at,
                                 filed_date=COALESCE(NULLIF(excluded.filed_date, ''), companyfacts_facts.filed_date),
                                 form=COALESCE(NULLIF(excluded.form, ''), companyfacts_facts.form),
                                 accession=COALESCE(NULLIF(excluded.accession, ''), companyfacts_facts.accession),
                                 source_tags=excluded.source_tags,
                                 written_by=excluded.written_by""",
                (upper, fact["fiscal_year"], period_type, fact["period_end"], fact["line_item"],
                 fact["value"], fact["units"], fact["source_url"], now,
                 fact.get("filed_date"), fact.get("form"), fact.get("accession"),
                 fact.get("source_tags"), WRITTEN_BY),
            )
            _record_vintage(conn, upper, fact, now)


def _facts_from_local_cache(cik: str, *, years_back: int) -> list[dict[str, object]]:
    path = companyfacts_cache_path(cik)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw = payload.get("companyfacts") if isinstance(payload.get("companyfacts"), dict) else payload
    if not isinstance(raw, dict):
        return []
    try:
        return normalize_annual_facts_from_raw(raw, cik=str(cik).zfill(10), years_back=years_back)
    except Exception:
        return []


def _is_quarterly_fresh(ticker: str) -> bool:
    """Return True if quarterly data for this ticker was fetched within TTL."""
    with get_db() as conn:
        return companyfacts_is_fresh(
            conn,
            ticker.upper(),
            ttl_seconds=_TTL_SECONDS,
            exclude_period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
        )


def _quarterly_facts_from_local_cache(cik: str, *, years_back: int) -> list[dict[str, object]]:
    path = companyfacts_cache_path(cik)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw = payload.get("companyfacts") if isinstance(payload.get("companyfacts"), dict) else payload
    if not isinstance(raw, dict):
        return []
    try:
        return normalize_quarterly_facts_from_raw(raw, cik=str(cik).zfill(10), years_back=years_back)
    except Exception:
        return []


def ensure_quarterly_facts(ticker: str, years_back: int = 10) -> None:
    """Guarantee quarterly companyfacts_facts rows are populated and fresh for ticker."""
    upper = ticker.upper().strip()
    if _is_quarterly_fresh(upper):
        return
    cik = resolve(upper)
    facts = _quarterly_facts_from_local_cache(cik, years_back=years_back)
    if not facts:
        facts = fetch_quarterly_facts(cik, years_back=years_back)
    now = utc_now_iso()
    with get_db() as conn:
        if facts:
            _purge_stale_rows(
                conn, upper, period_types=_QUARTERLY_PERIOD_TYPES, years_back=years_back
            )
        for fact in facts:
            conn.execute(
                """INSERT INTO companyfacts_facts
                   (ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at,
                    filed_date, form, accession, source_tags, written_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, fiscal_year, period_type, line_item)
                   DO UPDATE SET value=excluded.value, period_end=excluded.period_end,
                                 fetched_at=excluded.fetched_at,
                                 filed_date=COALESCE(NULLIF(excluded.filed_date, ''), companyfacts_facts.filed_date),
                                 form=COALESCE(NULLIF(excluded.form, ''), companyfacts_facts.form),
                                 accession=COALESCE(NULLIF(excluded.accession, ''), companyfacts_facts.accession),
                                 source_tags=excluded.source_tags,
                                 written_by=excluded.written_by""",
                (upper, fact["fiscal_year"], fact["period_type"],
                 fact["period_end"], fact["line_item"],
                 fact["value"], fact["units"], fact["source_url"], now,
                 fact.get("filed_date"), fact.get("form"), fact.get("accession"),
                 fact.get("source_tags"), WRITTEN_BY),
            )
            _record_vintage(conn, upper, fact, now)


def ensure_all_facts(ticker: str, years_back: int = 10) -> None:
    """Ensure both annual and quarterly facts are populated for ticker."""
    ensure_facts(ticker, years_back=years_back)
    ensure_quarterly_facts(ticker, years_back=years_back)
