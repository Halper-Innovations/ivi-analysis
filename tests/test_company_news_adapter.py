from __future__ import annotations

from pathlib import Path

from app.research.adapters.base import AdapterContext
from app.research.adapters.company_news import CompanyNewsAdapter


def _cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "false")
    monkeypatch.setenv("VOE_RESEARCH_COMPANY_NEWS_ENABLED", "true")
    monkeypatch.setenv("VOE_RESEARCH_ALLOWLIST_DOMAINS", "example.com,investor.example.com")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    return _get_config()


def test_company_news_discovers_feed_from_homepage(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path)
    adapter = CompanyNewsAdapter(cfg)

    fixtures = Path(__file__).parent / "fixtures"
    homepage = (fixtures / "company_news_homepage.html").read_text(encoding="utf-8")
    news_page = (fixtures / "company_news_page.html").read_text(encoding="utf-8")
    feed_xml = (fixtures / "company_news_feed.xml").read_text(encoding="utf-8")

    def _fake_get(url, **kwargs):
        if url == "https://example.com":
            return homepage.encode("utf-8")
        if url == "https://example.com/news":
            return news_page.encode("utf-8")
        if url == "https://example.com/news/feed.xml":
            return feed_xml.encode("utf-8")
        return b""

    monkeypatch.setattr(adapter.http, "get_bytes", _fake_get)

    result = adapter.collect(
        AdapterContext(
            ticker="AAPL",
            as_of_date="2026-02-13",
            company_name="Apple Inc.",
            packet={},
            run_id="run_test",
            homepage_url="https://example.com",
        )
    )
    assert result.evidence_items
    assert result.evidence_items[0].source_type == "company_news"


def test_company_news_fallback_paths_when_homepage_missing(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path)
    adapter = CompanyNewsAdapter(cfg)

    fixtures = Path(__file__).parent / "fixtures"
    listing = (fixtures / "company_news_listing.html").read_text(encoding="utf-8")

    def _fake_get(url, **kwargs):
        if url.startswith("https://investor.example.com/"):
            return listing.encode("utf-8")
        return b""

    monkeypatch.setattr(adapter.http, "get_bytes", _fake_get)

    result = adapter.collect(
        AdapterContext(
            ticker="MSFT",
            as_of_date="2026-02-13",
            company_name="Microsoft Corporation",
            packet={},
            run_id="run_test",
            homepage_url=None,
        )
    )
    assert any(g.gap_id == "GAP_HOMEPAGE_URL_MISSING" for g in result.evidence_gaps)
    assert result.evidence_items


def test_company_news_allows_per_ticker_allowlist_domains(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path)
    adapter = CompanyNewsAdapter(cfg)

    homepage = "<html><body><a href='/news'>News</a></body></html>"
    news_page = "<html><head><link href='/news/feed.xml' rel='alternate'></head></html>"
    feed_xml = """<?xml version='1.0' encoding='UTF-8'?>
    <rss version='2.0'><channel>
      <item><title>Company update</title><link>https://ticker.example.net/news/update</link>
      <pubDate>Fri, 13 Feb 2026 10:00:00 GMT</pubDate><description>Useful company update.</description></item>
    </channel></rss>"""

    def _fake_get(url, **kwargs):
        if url == "https://ticker.example.net":
            return homepage.encode("utf-8")
        if url == "https://ticker.example.net/news":
            return news_page.encode("utf-8")
        if url == "https://ticker.example.net/news/feed.xml":
            return feed_xml.encode("utf-8")
        return b""

    monkeypatch.setattr(adapter.http, "get_bytes", _fake_get)

    result = adapter.collect(
        AdapterContext(
            ticker="TCKR",
            as_of_date="2026-02-13",
            company_name="Ticker Example",
            packet={},
            run_id="run_test",
            homepage_url="https://ticker.example.net",
            allowlist_domains=("ticker.example.net",),
        )
    )

    assert [item.source_url for item in result.evidence_items] == [
        "https://ticker.example.net/news",
        "https://ticker.example.net/news/update",
    ]
