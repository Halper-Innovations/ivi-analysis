"""Daily poll / backfill orchestration for the corporate-events feed.

poll_day refuses scan_date >= today (US/Eastern) BEFORE any fetch: the
current day's daily index exists intraday and grows until ~10 PM ET, and
SecClient.download_bytes caches forever — a cached partial index plus the
idempotent OK scan row would silently drop the rest of that day's filings
permanently.

Backfill parses the quarterly full indexes once each, groups rows by filing
date, and runs the identical per-day path in date order, so a backfill is a
faithful day-by-day simulation of daily polling. Resumability comes from
corporate_event_scans: days with an OK scan row are skipped unless force.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests

from app.db import get_db, init_db
from app.events import store
from app.events.busted_ipo_qualifier import qualify_busted_ipos
from app.events.detectors import (
    apply_spinoff_auto_effect,
    detect_atarget_8k_freshness,
    detect_busted_ipos,
    detect_ch11,
    detect_queue_protection,
    detect_spinoffs,
)
from app.events.index_feed import (
    IndexRow,
    candidate_rows,
    daily_index_url,
    full_index_url,
    parse_master_idx,
)
from app.events.resolve import resolve_unknown_tickers
from app.events.submissions_adapter import SecSubmissionsAdapter


@dataclass
class ScanSummary:
    scan_date: str
    mode: str
    status: str  # 'OK' | 'NO_INDEX' | 'ERROR'
    index_rows: int = 0
    candidate_rows: int = 0
    events_created: int = 0
    events_updated: int = 0
    skips_recorded: int = 0


def _today_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def _ensure_watchlist_schema() -> None:
    """The flag sync writes watchlist.event_pending — make sure it exists."""
    try:
        from app.watchlist.schema import ensure_watchlist_schema

        ensure_watchlist_schema()
    except Exception:
        # Tmp DBs without the watchlist tables are fine; sync reports zeros.
        pass


def watchlist_cik_map(conn: sqlite3.Connection) -> dict[str, str]:
    """cik10 -> ticker for every active (non-REMOVED) watchlist name.

    Primary source is the SEC company_tickers map; sec_registrants fills in
    names the 7-day-TTL cache has not caught up with. Unresolvable tickers
    are returned under the '' key count by the caller via set difference.

    OPEN holdings are merged into the scope — queue protection follows
    capital at risk, not just watchlist presentation state.
    """
    from app.universe.ticker_cik_map import load_ticker_cik_map

    try:
        tickers = {
            str(row["ticker"]).upper()
            for row in conn.execute(
                "SELECT DISTINCT ticker FROM watchlist WHERE status != 'REMOVED'"
            ).fetchall()
        }
    except sqlite3.OperationalError:
        return {}
    try:
        from app.holdings import held_cik_map

        held = held_cik_map(conn)
        tickers |= set(held.values())
    except Exception:  # noqa: BLE001 - holdings table may not exist yet
        held = {}
    out: dict[str, str] = {}
    try:
        mapping = load_ticker_cik_map()
    except Exception:
        mapping = {}
    for ticker in tickers:
        cik = mapping.get(ticker)
        if cik:
            out[str(cik).zfill(10)] = ticker
    missing = tickers - set(out.values())
    if missing:
        try:
            placeholders = ",".join("?" for _ in missing)
            for row in conn.execute(
                f"SELECT cik, primary_ticker FROM sec_registrants WHERE primary_ticker IN ({placeholders})",
                sorted(missing),
            ).fetchall():
                out[str(row["cik"]).zfill(10)] = str(row["primary_ticker"]).upper()
        except sqlite3.OperationalError:
            pass
    for cik, ticker in held.items():
        out.setdefault(cik, ticker)
    return out


def atarget_cik_map(
    conn: sqlite3.Connection, watchlist_ciks: dict[str, str]
) -> dict[str, str]:
    """watchlist_ciks restricted to at-target names (DEPLOY_READY/BUY_CONFIRMED, not AVOID)."""
    try:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM watchlist "
            "WHERE status IN ('DEPLOY_READY', 'BUY_CONFIRMED') "
            "AND COALESCE(conviction_grade, '') != 'AVOID'"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    tickers = {str(row["ticker"]).upper() for row in rows}
    return {cik: ticker for cik, ticker in watchlist_ciks.items() if ticker in tickers}


def apply_expiry(conn: sqlite3.Connection, *, scan_date: str) -> int:
    """Deterministic expiry, checked against scan_date (point-in-time)."""
    expired = 0
    policies = (
        ("spinoff", "detection_date", 270, "NO_EFFECTIVENESS_270D"),
        ("ch11_emergence", "detection_date", 540, "NO_RELISTING_540D"),
    )
    for event_type, column, days, reason in policies:
        rows = conn.execute(
            f"SELECT id, {column} AS basis FROM corporate_events "
            "WHERE event_type = ? AND status = 'DETECTED'",
            (event_type,),
        ).fetchall()
        for row in rows:
            deadline = (date.fromisoformat(row["basis"]) + timedelta(days=days)).isoformat()
            if deadline < scan_date:
                if store.mark_expired(conn, event_id=int(row["id"]), expiry_reason=reason):
                    expired += 1
    rows = conn.execute(
        """
        SELECT id, json_extract(detail_json, '$.priced_date') AS priced
        FROM corporate_events
        WHERE event_type = 'busted_ipo' AND status = 'DETECTED' AND priced IS NOT NULL
        """
    ).fetchall()
    for row in rows:
        deadline = (date.fromisoformat(row["priced"]) + timedelta(days=730)).isoformat()
        if deadline < scan_date:
            if store.mark_expired(
                conn, event_id=int(row["id"]), expiry_reason="QUALIFIER_WINDOW_ELAPSED"
            ):
                expired += 1
    return expired


def _process_day(
    conn: sqlite3.Connection,
    rows: list[IndexRow],
    malformed: int,
    *,
    scan_date: str,
    mode: str,
    submissions,
    history,
    registrations,
    mapping,
    watchlist_ciks: dict[str, str],
) -> ScanSummary:
    candidates = candidate_rows(rows)
    # Commit after each detector: every detector interleaves SEC fetches with
    # writes, so one day-wide transaction holds the engine.db write lock for
    # minutes and starves concurrent heartbeat writers (2026-07-16). A crash
    # between commits is safe — the OK scan row lands last, so scan-gap
    # self-healing re-runs the day and the detectors dedupe by fingerprint.
    counters = []
    counters.append(
        detect_spinoffs(
            conn, candidates, scan_date=scan_date, source_mode=mode,
            registrations=registrations,
        )
    )
    conn.commit()
    counters.append(
        detect_ch11(
            conn, candidates, scan_date=scan_date, source_mode=mode,
            submissions=submissions,
        )
    )
    conn.commit()
    counters.append(
        detect_busted_ipos(
            conn, candidates, scan_date=scan_date, source_mode=mode,
            history=history,
        )
    )
    conn.commit()
    # Queue-protection forms (425, SC 13D, 424B*, ...) are outside
    # CANDIDATE_FORMS — pass the full row set, scoped by watchlist CIK.
    counters.append(
        detect_queue_protection(
            conn, rows, scan_date=scan_date, source_mode=mode,
            watchlist_ciks=watchlist_ciks, submissions=submissions,
        )
    )
    conn.commit()
    # 8-K freshness runs only on daily polls (backfill would anachronistically
    # gate historical days on the current at-target queue) and needs a reader
    # with the filings_window enumeration (SecSubmissionsAdapter has it).
    from app.config import get_config

    if (
        mode == "daily"
        and get_config().events_8k_freshness_enabled
        and hasattr(submissions, "filings_window")
    ):
        counters.append(
            detect_atarget_8k_freshness(
                conn, atarget_cik_map(conn, watchlist_ciks),
                scan_date=scan_date,
                lookback_days=get_config().events_8k_freshness_lookback_days,
                source_mode=mode, filings=submissions,
            )
        )
        conn.commit()
    apply_spinoff_auto_effect(conn, scan_date=scan_date)
    qualify_busted_ipos(conn, as_of=scan_date)
    apply_expiry(conn, scan_date=scan_date)
    conn.commit()
    resolve_unknown_tickers(conn, mapping=mapping)
    from app.events.flags import sync_event_pending_flags

    sync_event_pending_flags(conn)

    detail: dict[str, int] = {"n_malformed": malformed}
    for counter in counters:
        for key, value in counter.counts.items():
            detail[key] = detail.get(key, 0) + value
    summary = ScanSummary(
        scan_date=scan_date,
        mode=mode,
        status="OK",
        index_rows=len(rows),
        candidate_rows=len(candidates),
        events_created=sum(c.events_created for c in counters),
        events_updated=sum(c.events_updated for c in counters),
        skips_recorded=sum(c.skips for c in counters),
    )
    store.record_scan(
        conn, scan_date=scan_date, mode=mode, status="OK",
        counters={
            "index_rows": summary.index_rows,
            "candidate_rows": summary.candidate_rows,
            "events_created": summary.events_created,
            "events_updated": summary.events_updated,
            "skips_recorded": summary.skips_recorded,
            **detail,
        },
    )
    return summary


def _classify_fetch_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404:
            return "NO_INDEX"
    return "ERROR"


def poll_day(
    scan_date: str,
    *,
    client=None,
    submissions=None,
    history=None,
    registrations=None,
    mapping=None,
    watchlist_ciks: dict[str, str] | None = None,
    today_et: str | None = None,
) -> ScanSummary:
    today = today_et or _today_et()
    if scan_date >= today:
        raise ValueError(
            f"poll_day refuses scan_date {scan_date} >= today {today} (US/Eastern): "
            "the current day's index is partial intraday and would be cached forever."
        )
    if client is None:
        from app.ingest.sec_client import SecClient

        client = SecClient()
    adapter = SecSubmissionsAdapter(client)
    submissions = submissions or adapter
    history = history or adapter
    registrations = registrations or adapter

    init_db()
    _ensure_watchlist_schema()
    with get_db() as conn:
        wl_ciks = watchlist_ciks if watchlist_ciks is not None else watchlist_cik_map(conn)
        try:
            raw = client.download_bytes(daily_index_url(scan_date))
        except Exception as exc:
            status = _classify_fetch_error(exc)
            store.record_scan(conn, scan_date=scan_date, mode="daily", status=status, counters={})
            return ScanSummary(scan_date=scan_date, mode="daily", status=status)
        rows, malformed = parse_master_idx(raw)
        return _process_day(
            conn, rows, malformed, scan_date=scan_date, mode="daily",
            submissions=submissions, history=history, registrations=registrations,
            mapping=mapping, watchlist_ciks=wl_ciks,
        )


def _quarters_in_window(start: date, end: date) -> list[tuple[int, int]]:
    quarters = []
    year, quarter = start.year, (start.month - 1) // 3 + 1
    while (year, quarter) <= (end.year, (end.month - 1) // 3 + 1):
        quarters.append((year, quarter))
        quarter += 1
        if quarter == 5:
            year, quarter = year + 1, 1
    return quarters


def backfill(
    start_date: str,
    end_date: str,
    *,
    client=None,
    force: bool = False,
    sec_budget: int | None = 50_000,
    watchlist_ciks: dict[str, str] | None = None,
    mapping: dict[str, str] | None = None,
    today_et: str | None = None,
) -> list[ScanSummary]:
    from app.util.http import temporary_sec_domain_budget

    today = today_et or _today_et()
    if client is None:
        from app.ingest.sec_client import SecClient

        client = SecClient()
    adapter = SecSubmissionsAdapter(client)

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    summaries: list[ScanSummary] = []
    init_db()
    _ensure_watchlist_schema()
    today_date = date.fromisoformat(today)
    open_quarter = (today_date.year, (today_date.month - 1) // 3 + 1)
    with temporary_sec_domain_budget(sec_budget=sec_budget):
        by_day: dict[str, list[IndexRow]] = {}
        malformed_total = 0
        for year, quarter in _quarters_in_window(start, end):
            url = full_index_url(year, quarter)
            if (year, quarter) >= open_quarter and hasattr(client, "http"):
                # The open quarter's full index is cumulative and still
                # growing — a forever-TTL fetch would pin a partial quarter
                # (the same hazard the current-day guard exists for). Cache
                # it for hours, not forever.
                raw = client.http.get_bytes(url, use_cache=True, cache_ttl_seconds=6 * 3600)
            else:
                raw = client.download_bytes(url)
            rows, malformed = parse_master_idx(raw)
            malformed_total += malformed
            for row in rows:
                if start_date <= row.date_filed <= end_date and row.date_filed < today:
                    by_day.setdefault(row.date_filed, []).append(row)
        with get_db() as conn:
            wl_ciks = watchlist_ciks if watchlist_ciks is not None else watchlist_cik_map(conn)
            for scan_date in sorted(by_day):
                if not force:
                    existing = conn.execute(
                        "SELECT status FROM corporate_event_scans WHERE scan_date = ?",
                        (scan_date,),
                    ).fetchone()
                    if existing is not None and existing["status"] == "OK":
                        continue
                summaries.append(
                    _process_day(
                        conn, by_day[scan_date], 0, scan_date=scan_date, mode="backfill",
                        submissions=adapter, history=adapter, registrations=adapter,
                        mapping=mapping, watchlist_ciks=wl_ciks,
                    )
                )
                # Commit per day: scan rows persist as they land (true
                # mid-run resumability) and other writers get a turn instead
                # of a window-length exclusive transaction.
                conn.commit()
    return summaries

# --------------------------------------------------------------------------- #
# Scan-gap self-healing — the heartbeat targets exactly the previous
# business day, so a missed morning (asleep laptop, crash) used to become a
# PERMANENT hole in the event feed. find_scan_gaps() lists trailing business
# days without an OK scan row and run_scan_gap_backfill() feeds them through
# the existing idempotent backfill(); a weekday genuinely absent from the
# EDGAR index after a clean backfill (market holiday) is closed with a
# zero-count OK row so it stops re-alerting.
# --------------------------------------------------------------------------- #


def find_scan_gaps(
    conn,
    *,
    window_days: int = 30,
    today_et_str: str | None = None,
) -> list[str]:
    """Trailing-window business days (Mon-Fri) with no OK scan row."""
    today = date.fromisoformat(today_et_str or _today_et())
    start = today - timedelta(days=max(1, int(window_days)))
    ok_dates = {
        str(row["scan_date"])
        for row in conn.execute(
            "SELECT scan_date FROM corporate_event_scans WHERE status = 'OK' AND scan_date >= ?",
            (start.isoformat(),),
        ).fetchall()
    }
    gaps: list[str] = []
    day = start
    while day < today:
        if day.weekday() < 5 and day.isoformat() not in ok_dates:
            gaps.append(day.isoformat())
        day += timedelta(days=1)
    return gaps


def run_scan_gap_backfill(
    *,
    window_days: int = 30,
    client=None,
    sec_budget: int | None = 50_000,
    today_et: str | None = None,
) -> dict:
    """Detect trailing scan gaps and recover them via backfill().

    Returns {"gaps", "recovered", "closed_no_index", "unrecovered"}; the CLI
    exits nonzero when unrecovered days remain so the heartbeat alerts.
    """
    from app.db import get_db, init_db

    init_db()
    with get_db() as conn:
        gaps = find_scan_gaps(conn, window_days=window_days, today_et_str=today_et)
    if not gaps:
        return {"gaps": [], "recovered": [], "closed_no_index": [], "unrecovered": []}

    backfill(
        gaps[0],
        gaps[-1],
        client=client,
        sec_budget=sec_budget,
        today_et=today_et,
    )

    with get_db() as conn:
        remaining = set(find_scan_gaps(conn, window_days=window_days, today_et_str=today_et))
        closed_no_index: list[str] = []
        for day in sorted(remaining & set(gaps)):
            # The backfill completed without error and the day still has no
            # scan row: the EDGAR index had no filings dated that weekday —
            # a market holiday. Close it explicitly (append-preserving:
            # record_scan never downgrades an OK row).
            store.record_scan(
                conn,
                scan_date=day,
                mode="gap_backfill_no_index",
                status="OK",
                counters={
                    "index_rows": 0,
                    "candidate_rows": 0,
                    "events_created": 0,
                    "events_updated": 0,
                    "skips_recorded": 0,
                    "note": "no index rows for this weekday after clean backfill (market holiday)",
                },
            )
            closed_no_index.append(day)
        conn.commit()
        unrecovered = sorted(
            set(find_scan_gaps(conn, window_days=window_days, today_et_str=today_et))
        )
    recovered = sorted(set(gaps) - set(unrecovered) - set(closed_no_index))
    return {
        "gaps": gaps,
        "recovered": recovered,
        "closed_no_index": closed_no_index,
        "unrecovered": unrecovered,
    }
