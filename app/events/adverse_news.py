"""Alpha Vantage adverse-news scan for the adverse-events gate.

A lawsuit/scandal that never touches EDGAR or a federal docket usually still
hits the financial press within days. The scan pulls NEWS_SENTIMENT headlines
for at-target names (the AV free tier is 25 requests/day — widen
events_adverse_news_max_tickers only on a paid key), keyword-classifies them,
and opens adverse_news queue-protection events (anchor "NEWS-<url hash>" —
synthetic, no EDGAR accession) that block DEPLOY_READY presentation until
`ivi events dispose`. Zero LLM spend: the classifier is a word-boundary
keyword match over title + summary.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date

from app.config import get_config
from app.events import store
from app.events.detectors import DetectorCounters

ADVERSE_KEYWORDS = (
    "lawsuit",
    "sues",
    "sued",
    "class action",
    "fraud",
    "investigation",
    "probe",
    "subpoena",
    "sec charges",
    "restatement",
    "going concern",
    "delisting",
    "bankruptcy",
    "scandal",
    "indicted",
    "indictment",
    "whistleblower",
    "short seller",
)

_KEYWORD_RES = {
    keyword: re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    for keyword in ADVERSE_KEYWORDS
}


@dataclass
class NewsItem:
    url: str
    title: str
    summary: str
    published_at: str | None
    sentiment: str


def classify_adverse(title: str, summary: str) -> list[str]:
    """Matched adverse keywords (word-boundary) across title + summary."""
    text = f"{title}\n{summary}"
    return [keyword for keyword, pattern in _KEYWORD_RES.items() if pattern.search(text)]


def news_anchor(url: str) -> str:
    return "NEWS-" + hashlib.sha256(url.strip().lower().encode("utf-8")).hexdigest()[:16]


class AlphaVantageNewsFetcher:
    def __init__(self, cfg=None, http=None) -> None:
        from app.util.http import HttpClient

        self.cfg = cfg or get_config()
        self.http = http or HttpClient(self.cfg)

    def fetch(self, ticker: str) -> list[NewsItem] | None:
        from app.research.adapters.external_news import (
            _parse_alpha_vantage_time,
            _request_url,
        )

        url = _request_url(
            ticker,
            str(self.cfg.research_alpha_vantage_api_key or ""),
            self.cfg.research_external_news_max_items,
        )
        try:
            raw = self.http.get_bytes(url, use_cache=True, cache_ttl_seconds=6 * 3600)
            payload = json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("Information") or payload.get("Note") or payload.get("Error Message"):
            # Rate-limit / entitlement message, not a news payload.
            return None
        feed = payload.get("feed") if isinstance(payload.get("feed"), list) else []
        items: list[NewsItem] = []
        for row in feed:
            if not isinstance(row, dict):
                continue
            item_url = str(row.get("url") or "").strip()
            title = str(row.get("title") or "").strip()
            if not item_url or not title:
                continue
            items.append(
                NewsItem(
                    url=item_url,
                    title=title,
                    summary=str(row.get("summary") or "").strip(),
                    published_at=_parse_alpha_vantage_time(row.get("time_published")),
                    sentiment=str(row.get("overall_sentiment_label") or "").strip(),
                )
            )
        return items


def adverse_news_targets(conn: sqlite3.Connection) -> list[dict[str, str]]:
    """[{cik, ticker}] for at-target names (DEPLOY_READY/BUY_CONFIRMED, not
    AVOID). Tighter scope than the docket scan: the AV free tier allows only
    25 requests/day."""
    try:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM watchlist "
            "WHERE status IN ('DEPLOY_READY', 'BUY_CONFIRMED') "
            "AND COALESCE(conviction_grade, '') != 'AVOID'"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    targets: list[dict[str, str]] = []
    for ticker in sorted({str(row["ticker"]).upper() for row in rows}):
        try:
            reg = conn.execute(
                "SELECT cik FROM sec_registrants WHERE primary_ticker = ?",
                (ticker,),
            ).fetchone()
        except sqlite3.OperationalError:
            return []
        if reg is None:
            continue
        targets.append({"cik": str(reg["cik"]).zfill(10), "ticker": ticker})
    return targets


def scan_adverse_news(
    conn: sqlite3.Connection,
    targets: list[dict[str, str]],
    *,
    scan_date: str,
    source_mode: str,
    fetcher,
    max_tickers: int,
) -> DetectorCounters:
    counters = DetectorCounters()
    if len(targets) > max_tickers:
        counters.bump("n_news_targets_over_cap", len(targets) - max_tickers)
        targets = targets[:max_tickers]
    for target in targets:
        items = fetcher.fetch(target["ticker"])
        if items is None:
            counters.bump("n_news_fetch_failed")
            continue
        for item in items:
            matched = classify_adverse(item.title, item.summary)
            if not matched:
                continue
            anchor = news_anchor(item.url)
            known = conn.execute(
                "SELECT 1 FROM corporate_events WHERE anchor_accession = ? LIMIT 1",
                (anchor,),
            ).fetchone()
            if known is not None:
                continue
            event_id = store.upsert_event(
                conn, cik=target["cik"], event_type="adverse_news",
                anchor_accession=anchor, company_name=target["ticker"],
                detection_date=(item.published_at or scan_date)[:10],
                source_mode=source_mode,
                detail={
                    "watchlist_ticker": target["ticker"],
                    "url": item.url,
                    "title": item.title,
                    "published_at": item.published_at,
                    "matched_keywords": matched,
                    "av_sentiment": item.sentiment,
                },
            )
            store.set_ticker(conn, event_id=event_id, ticker=target["ticker"])
            counters.events_created += 1
    return counters


def run_news_scan(
    scan_date: str | None = None,
    *,
    tickers: list[str] | None = None,
    dry_run: bool = False,
    fetcher=None,
) -> dict:
    """Entry point behind `ivi events news-scan`; returns a JSON-safe summary."""
    from app.db import get_db, init_db
    from app.events.flags import sync_event_pending_flags

    cfg = get_config()
    if not cfg.events_adverse_news_enabled:
        return {"status": "DISABLED", "detail": "set VOE_EVENTS_ADVERSE_NEWS_ENABLED=true"}
    if fetcher is None:
        if not cfg.research_alpha_vantage_api_key:
            return {"status": "NO_KEY", "detail": "set ALPHA_VANTAGE_API_KEY"}
        fetcher = AlphaVantageNewsFetcher(cfg)
    effective_date = scan_date or date.today().isoformat()
    init_db()
    with get_db() as conn:
        targets = adverse_news_targets(conn)
        if tickers:
            wanted = {t.upper() for t in tickers}
            targets = [t for t in targets if t["ticker"] in wanted]
        if dry_run:
            hits = []
            for target in targets[: cfg.events_adverse_news_max_tickers]:
                for item in fetcher.fetch(target["ticker"]) or []:
                    matched = classify_adverse(item.title, item.summary)
                    if matched:
                        hits.append({
                            "ticker": target["ticker"],
                            "title": item.title,
                            "url": item.url,
                            "matched_keywords": matched,
                            "published_at": item.published_at,
                        })
            return {
                "status": "DRY_RUN",
                "scan_date": effective_date,
                "targets": len(targets),
                "hits": hits,
            }
        counters = scan_adverse_news(
            conn, targets, scan_date=effective_date, source_mode="daily",
            fetcher=fetcher, max_tickers=cfg.events_adverse_news_max_tickers,
        )
        sync = sync_event_pending_flags(conn)
        return {
            "status": "OK",
            "scan_date": effective_date,
            "targets": len(targets),
            "events_created": counters.events_created,
            "counts": counters.counts,
            "flag_sync": sync,
        }
