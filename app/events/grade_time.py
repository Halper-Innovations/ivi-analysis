"""Grade-time adverse-events check for scan/sweep landings.

The daily heartbeat scans (dockets, adverse news, 8-K freshness) run on a
24h cadence, which leaves a window where a sweep can present a freshly-graded
ACTIONABLE name that is cheap BECAUSE of a two-week-old lawsuit or scandal.
This hook closes that window: called with the tickers a run just landed, it
runs the docket + news scans restricted to those names before the brief
renders, so an adverse hit flags EVENT_PENDING on the very first surface the
owner sees.

Best-effort by design: a scan failure (missing token, network, rate limit)
must never fail the sweep that calls it — the daily heartbeat re-covers the
same names within a day. Disabled sources report status so briefs can say
"not checked" honestly instead of implying a clean bill.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

from app.config import get_config


def grade_time_adverse_check(
    conn: sqlite3.Connection,
    tickers: list[str],
    *,
    scan_date: str | None = None,
    searcher=None,
    fetcher=None,
) -> dict:
    """Run docket + adverse-news scans for just-landed tickers; sync flags.

    Returns {"checked": [...], "sources": {"dockets": <status>, "news": <status>},
    "events_created": N, "flagged": {ticker: [flags]}} — JSON-safe, renderable
    in a brief. Never raises.
    """
    from app.events.flags import open_flags_by_ticker, sync_event_pending_flags

    cfg = get_config()
    wanted = sorted({str(t).upper() for t in tickers if str(t).strip()})
    effective_date = scan_date or date.today().isoformat()
    summary: dict = {
        "checked": wanted,
        "scan_date": effective_date,
        "sources": {"dockets": "DISABLED", "news": "DISABLED"},
        "events_created": 0,
        "flagged": {},
    }
    if not wanted:
        return summary

    if cfg.events_dockets_enabled and cfg.courtlistener_api_token:
        try:
            from app.events.courtlistener import (
                CourtListenerSearcher,
                litigation_targets,
            )
            from app.events.detectors import detect_litigation_dockets

            targets = [
                t for t in litigation_targets(conn) if t["ticker"] in wanted
            ]
            counters = detect_litigation_dockets(
                conn, targets, scan_date=effective_date,
                lookback_days=cfg.events_docket_lookback_days,
                nos_codes=set(cfg.events_docket_nos_codes),
                source_mode="grade_time",
                searcher=searcher or CourtListenerSearcher(cfg),
            )
            summary["events_created"] += counters.events_created
            summary["sources"]["dockets"] = "OK"
        except Exception as exc:  # noqa: BLE001 — never fail the calling sweep
            summary["sources"]["dockets"] = f"ERROR: {exc}"

    if cfg.events_adverse_news_enabled and cfg.research_alpha_vantage_api_key:
        try:
            from app.events.adverse_news import (
                AlphaVantageNewsFetcher,
                scan_adverse_news,
            )

            targets = [{"cik": cik, "ticker": ticker} for cik, ticker in sorted(
                _cik_by_ticker(conn, wanted).items(), key=lambda kv: kv[1]
            )]
            counters = scan_adverse_news(
                conn, targets, scan_date=effective_date, source_mode="grade_time",
                fetcher=fetcher or AlphaVantageNewsFetcher(cfg),
                max_tickers=cfg.events_adverse_news_max_tickers,
            )
            summary["events_created"] += counters.events_created
            summary["sources"]["news"] = "OK"
        except Exception as exc:  # noqa: BLE001
            summary["sources"]["news"] = f"ERROR: {exc}"

    try:
        sync_event_pending_flags(conn)
        flags = open_flags_by_ticker(conn)
        summary["flagged"] = {t: flags[t] for t in wanted if t in flags}
    except Exception as exc:  # noqa: BLE001
        summary["sources"]["flag_sync"] = f"ERROR: {exc}"
    return summary


def _cik_by_ticker(conn: sqlite3.Connection, tickers: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for ticker in tickers:
        try:
            row = conn.execute(
                "SELECT cik FROM sec_registrants WHERE primary_ticker = ?",
                (ticker,),
            ).fetchone()
        except sqlite3.OperationalError:
            return out
        if row is not None:
            out[str(row["cik"]).zfill(10)] = ticker
    return out

def retroactive_queue_protection(
    conn: sqlite3.Connection,
    tickers: list[str],
    *,
    lookback_days: int = 270,
    scan_date: str | None = None,
) -> dict:
    """Newly added names inherit protection for events already on file.

    The daily poll only protects names that were on the watchlist when the
    filing landed — a name added AFTER its merger paper or Form 25 filed
    would arrive unprotected. This scans the name's trailing filings from
    the cached EDGAR submissions payload (no network) through the same
    form -> queue-protection map the live poll uses, opens events
    idempotently, and re-syncs EVENT_PENDING flags. Never raises.
    """
    from app.events import store
    from app.events.detectors import queue_protection_type_for_form
    from app.events.flags import sync_event_pending_flags

    wanted = sorted({str(t).upper() for t in tickers if str(t).strip()})
    effective_date = scan_date or date.today().isoformat()
    cutoff = (date.fromisoformat(effective_date) - timedelta(days=lookback_days)).isoformat()
    summary: dict = {
        "checked": wanted,
        "scan_date": effective_date,
        "events_created": 0,
        "events_updated": 0,
        "no_cik": [],
        "no_submissions": [],
    }
    if not wanted:
        return summary

    try:
        from app.universe.ticker_cik_map import load_ticker_cik_map

        mapping = load_ticker_cik_map(refresh_if_missing=False) or {}
    except Exception:  # noqa: BLE001 - no registry map means nothing to scan
        summary["no_cik"] = wanted
        return summary

    for ticker in wanted:
        cik = mapping.get(ticker)
        if not cik:
            summary["no_cik"].append(ticker)
            continue
        try:
            from app.universe.sector_universe import load_company_submissions

            payload = load_company_submissions(cik, refresh_if_missing=False)
        except Exception:  # noqa: BLE001 - cache-only read; missing is reported
            payload = None
        recent = (
            (payload.get("filings") or {}).get("recent")
            if isinstance(payload, dict)
            else None
        )
        if not isinstance(recent, dict):
            summary["no_submissions"].append(ticker)
            continue
        forms = recent.get("form") or []
        dates = recent.get("filingDate") or []
        accessions = recent.get("accessionNumber") or []
        company_name = str(payload.get("name") or ticker)
        cik_norm = str(cik).strip().lstrip("0")
        for form, filed, accession in zip(forms, dates, accessions):
            filed_str = str(filed or "")
            if not filed_str or filed_str < cutoff or filed_str > effective_date:
                continue
            event_type = queue_protection_type_for_form(str(form or ""))
            if event_type is None:
                continue
            try:
                existing = store.find_active_event(conn, cik=cik_norm, event_type=event_type)
                if existing is None:
                    event_id = store.upsert_event(
                        conn,
                        cik=cik_norm,
                        event_type=event_type,
                        anchor_accession=str(accession),
                        company_name=company_name,
                        detection_date=filed_str,
                        source_mode="retroactive_intake",
                    )
                    store.set_ticker(conn, event_id=event_id, ticker=ticker)
                    summary["events_created"] += 1
                else:
                    attached = store.attach_filing(
                        conn,
                        event_id=int(existing["id"]),
                        cik=cik_norm,
                        accession=str(accession),
                        form_type=str(form),
                        filing_date=filed_str,
                        role="RETROACTIVE_INTAKE",
                    )
                    if attached:
                        summary["events_updated"] += 1
            except sqlite3.Error:
                continue

    try:
        sync_event_pending_flags(conn)
        conn.commit()
    except sqlite3.Error:
        pass
    return summary

