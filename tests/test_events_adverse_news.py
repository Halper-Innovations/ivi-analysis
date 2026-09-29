"""Adverse-news gate: keyword-matched Alpha Vantage headlines open adverse_news
queue-protection events with synthetic NEWS-<url hash> anchors."""

from __future__ import annotations

from app.config import get_config
from app.db import get_db, init_db
from app.events.adverse_news import (
    NewsItem,
    adverse_news_targets,
    classify_adverse,
    news_anchor,
    scan_adverse_news,
)
from app.events.flags import sync_event_pending_flags
from app.watchlist.schema import ensure_watchlist_schema


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()
    ensure_watchlist_schema()


class FakeFetcher:
    def __init__(self, by_ticker=None, fail_tickers=()):
        self.by_ticker = by_ticker or {}
        self.fail_tickers = set(fail_tickers)
        self.calls = []

    def fetch(self, ticker):
        self.calls.append(ticker)
        if ticker in self.fail_tickers:
            return None
        return self.by_ticker.get(ticker, [])


def _item(url="https://news.example.com/ncsm-suit",
          title="NCS Multistage faces securities class action",
          summary="Shareholders filed a class action alleging misleading disclosures.",
          sentiment="Bearish"):
    return NewsItem(
        url=url, title=title, summary=summary,
        published_at="2026-06-28T13:00:00+00:00", sentiment=sentiment,
    )


TARGETS = [{"cik": "0001692427", "ticker": "NCSM"}]


def test_adverse_headline_opens_event_and_blocks(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    fetcher = FakeFetcher(by_ticker={"NCSM": [_item()]})
    with get_db() as conn:
        conn.execute(
            "INSERT INTO watchlist(ticker, status, source_run_id, added_at) "
            "VALUES('NCSM', 'DEPLOY_READY', 'run_NCSM', '2026-06-01T00:00:00Z')"
        )
        counters = scan_adverse_news(
            conn, TARGETS, scan_date="2026-07-01", source_mode="daily",
            fetcher=fetcher, max_tickers=25,
        )
        assert counters.events_created == 1
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["event_type"] == "adverse_news"
        assert event["anchor_accession"] == news_anchor("https://news.example.com/ncsm-suit")
        assert event["ticker"] == "NCSM"
        assert event["detection_date"] == "2026-06-28"
        sync_event_pending_flags(conn)
        row = conn.execute("SELECT event_pending FROM watchlist WHERE ticker='NCSM'").fetchone()
        assert row["event_pending"] == "EVENT_PENDING:ADVERSE_NEWS"


def test_neutral_headline_ignored(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    neutral = _item(
        url="https://news.example.com/ncsm-earnings",
        title="NCS Multistage reports third-quarter results",
        summary="Revenue rose 4% on international demand.",
        sentiment="Neutral",
    )
    fetcher = FakeFetcher(by_ticker={"NCSM": [neutral]})
    with get_db() as conn:
        counters = scan_adverse_news(
            conn, TARGETS, scan_date="2026-07-01", source_mode="daily",
            fetcher=fetcher, max_tickers=25,
        )
        assert counters.events_created == 0
        assert conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"] == 0


def test_rerun_same_url_idempotent(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    fetcher = FakeFetcher(by_ticker={"NCSM": [_item()]})
    with get_db() as conn:
        for _ in range(2):
            counters = scan_adverse_news(
                conn, TARGETS, scan_date="2026-07-01", source_mode="daily",
                fetcher=fetcher, max_tickers=25,
            )
        assert counters.events_created == 0
        assert conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"] == 1


def test_ticker_cap_respected(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    targets = [
        {"cik": "0000000001", "ticker": "AAAA"},
        {"cik": "0000000002", "ticker": "BBBB"},
        {"cik": "0000000003", "ticker": "CCCC"},
    ]
    fetcher = FakeFetcher()
    with get_db() as conn:
        counters = scan_adverse_news(
            conn, targets, scan_date="2026-07-01", source_mode="daily",
            fetcher=fetcher, max_tickers=2,
        )
        assert fetcher.calls == ["AAAA", "BBBB"]
        assert counters.counts == {"n_news_targets_over_cap": 1}


def test_fetch_failure_counted_not_fatal(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    fetcher = FakeFetcher(fail_tickers={"NCSM"})
    with get_db() as conn:
        counters = scan_adverse_news(
            conn, TARGETS, scan_date="2026-07-01", source_mode="daily",
            fetcher=fetcher, max_tickers=25,
        )
        assert counters.events_created == 0
        assert counters.counts == {"n_news_fetch_failed": 1}


def test_classify_adverse_word_boundaries():
    assert classify_adverse("Company faces securities class action", "") == ["class action"]
    assert classify_adverse("SEC charges former CFO with fraud", "") == ["fraud", "sec charges"]
    assert classify_adverse("Auditor flags going concern doubt", "") == ["going concern"]
    # 'probe' must not match inside other words.
    assert classify_adverse("New space probes launched", "") == []
    assert classify_adverse("Quarterly results beat estimates", "") == []


def test_adverse_news_targets_scope(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        for ticker, status, grade in (
            ("NCSM", "DEPLOY_READY", "ACTIONABLE"),
            ("BOOM", "ACTIVE", "ACTIONABLE"),
            ("AVDT", "DEPLOY_READY", "AVOID"),
        ):
            conn.execute(
                "INSERT INTO watchlist(ticker, status, conviction_grade, source_run_id, added_at) "
                "VALUES(?, ?, ?, ?, '2026-06-01T00:00:00Z')",
                (ticker, status, grade, f"run_{ticker}"),
            )
        conn.execute(
            "INSERT INTO sec_registrants(cik, primary_ticker, name, exchange_scope, operating_status, first_seen_at, last_seen_at) "
            "VALUES('0001692427', 'NCSM', 'NCS Multistage Holdings, Inc.', 'ALL', 'OPERATING', "
            "'2026-06-01T00:00:00Z', '2026-06-01T00:00:00Z')"
        )
        assert adverse_news_targets(conn) == [{"cik": "0001692427", "ticker": "NCSM"}]
