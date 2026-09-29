"""Grade-time adverse check: newly-ACTIONABLE names are docket/news-checked at
landing time, not just at the next daily heartbeat."""

from __future__ import annotations

from app.config import get_config
from app.db import get_db, init_db
from app.events.adverse_news import NewsItem
from app.events.courtlistener import DocketHit, nos_code_from_text
from app.events.grade_time import grade_time_adverse_check
from app.watchlist.schema import ensure_watchlist_schema


def _init(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_config.cache_clear()
    init_db()
    ensure_watchlist_schema()


def _seed(conn, ticker="NCSM", cik="0001692427", name="NCS Multistage Holdings, Inc."):
    conn.execute(
        "INSERT INTO watchlist(ticker, status, conviction_grade, source_run_id, added_at) "
        "VALUES(?, 'ACTIVE', 'ACTIONABLE', ?, '2026-07-01T00:00:00Z')",
        (ticker, f"run_{ticker}"),
    )
    conn.execute(
        "INSERT INTO sec_registrants(cik, primary_ticker, name, exchange_scope, operating_status, first_seen_at, last_seen_at) "
        "VALUES(?, ?, ?, 'ALL', 'OPERATING', '2026-06-01T00:00:00Z', '2026-06-01T00:00:00Z')",
        (cik, ticker, name),
    )


class FakeSearcher:
    def __init__(self, hits):
        self.hits = hits

    def search_recent_dockets(self, company_name, *, filed_after):
        return self.hits


class FakeFetcher:
    def __init__(self, items):
        self.items = items

    def fetch(self, ticker):
        return self.items


def test_both_sources_disabled_reports_honestly(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed(conn)
        summary = grade_time_adverse_check(conn, ["NCSM"])
        assert summary["sources"] == {"dockets": "DISABLED", "news": "DISABLED"}
        assert summary["events_created"] == 0
        assert summary["flagged"] == {}


def test_docket_hit_flags_freshly_actionable_name(monkeypatch, tmp_path):
    _init(
        monkeypatch, tmp_path,
        VOE_EVENTS_DOCKETS_ENABLED="true",
        VOE_COURTLISTENER_API_TOKEN="test-token",
    )
    nature = "850 Securities/Commodities/Exchange"
    hit = DocketHit(
        docket_id="777", case_name="Doe v. NCS Multistage Holdings, Inc.",
        court="txsd", date_filed="2026-06-25", nature_of_suit=nature,
        nos_code=nos_code_from_text(nature),
        url="https://www.courtlistener.com/docket/777/",
    )
    with get_db() as conn:
        _seed(conn)
        summary = grade_time_adverse_check(
            conn, ["NCSM"], scan_date="2026-07-03", searcher=FakeSearcher([hit]),
        )
        assert summary["sources"]["dockets"] == "OK"
        assert summary["events_created"] == 1
        assert summary["flagged"] == {"NCSM": ["EVENT_PENDING:LITIGATION_DOCKET"]}
        row = conn.execute("SELECT event_pending FROM watchlist WHERE ticker='NCSM'").fetchone()
        assert row["event_pending"] == "EVENT_PENDING:LITIGATION_DOCKET"


def test_news_hit_flags_freshly_actionable_name(monkeypatch, tmp_path):
    _init(
        monkeypatch, tmp_path,
        VOE_EVENTS_ADVERSE_NEWS_ENABLED="true",
        ALPHA_VANTAGE_API_KEY="test-key",
    )
    item = NewsItem(
        url="https://news.example.com/ncsm",
        title="NCS Multistage hit with securities class action",
        summary="Plaintiffs allege misleading statements.",
        published_at="2026-06-28T13:00:00+00:00", sentiment="Bearish",
    )
    with get_db() as conn:
        _seed(conn)
        summary = grade_time_adverse_check(
            conn, ["NCSM"], scan_date="2026-07-03", fetcher=FakeFetcher([item]),
        )
        assert summary["sources"]["news"] == "OK"
        assert summary["events_created"] == 1
        assert summary["flagged"] == {"NCSM": ["EVENT_PENDING:ADVERSE_NEWS"]}


def test_scan_failure_never_raises(monkeypatch, tmp_path):
    _init(
        monkeypatch, tmp_path,
        VOE_EVENTS_DOCKETS_ENABLED="true",
        VOE_COURTLISTENER_API_TOKEN="test-token",
    )

    class BoomSearcher:
        def search_recent_dockets(self, company_name, *, filed_after):
            raise RuntimeError("boom")

    with get_db() as conn:
        _seed(conn)
        summary = grade_time_adverse_check(
            conn, ["NCSM"], scan_date="2026-07-03", searcher=BoomSearcher(),
        )
        # The detector counts per-name failures; a raising searcher inside the
        # detector loop escapes to the hook's containment.
        assert summary["sources"]["dockets"].startswith("ERROR:")
        assert summary["events_created"] == 0


def test_empty_ticker_list_no_ops(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        summary = grade_time_adverse_check(conn, [])
        assert summary["checked"] == []
        assert summary["events_created"] == 0
