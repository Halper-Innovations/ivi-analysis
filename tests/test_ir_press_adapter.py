from __future__ import annotations

from app.research.adapters.base import AdapterContext
from app.research.adapters.ir_press import IRPressAdapter


def _cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "false")
    monkeypatch.setenv("VOE_RESEARCH_IR_PRESS_ENABLED", "true")
    monkeypatch.setenv("VOE_RESEARCH_ALLOWLIST_DOMAINS", "news.microsoft.com")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    return _get_config()


def test_ir_press_missing_url_creates_gap(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path)
    adapter = IRPressAdapter(cfg)
    result = adapter.collect(
        AdapterContext(
            ticker="MSFT",
            as_of_date="2026-02-13",
            company_name="Microsoft",
            packet={},
            ir_rss_url=None,
            homepage_url="https://www.microsoft.com",
        )
    )
    assert result.evidence_items == []
    assert result.evidence_gaps
    assert result.evidence_gaps[0].gap_id == "GAP_IR_RSS_MISSING"


def test_ir_press_disallowed_domain_creates_gap(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path)
    adapter = IRPressAdapter(cfg)
    result = adapter.collect(
        AdapterContext(
            ticker="MSFT",
            as_of_date="2026-02-13",
            company_name="Microsoft",
            packet={},
            ir_rss_url="https://example.com/feed.xml",
            homepage_url="https://www.microsoft.com",
        )
    )
    assert result.evidence_items == []
    assert result.evidence_gaps
    assert result.evidence_gaps[0].gap_id == "GAP_IR_RSS_DOMAIN_NOT_ALLOWLISTED"


def test_ir_press_parses_feed_items(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path)
    adapter = IRPressAdapter(cfg)

    xml = """<?xml version='1.0' encoding='UTF-8'?>
    <rss version='2.0'><channel>
      <item><title>Press 1</title><link>https://news.microsoft.com/p1</link>
      <pubDate>Fri, 13 Feb 2026 10:00:00 GMT</pubDate><description>Hello world</description></item>
      <item><title>Press 2</title><link>https://news.microsoft.com/p2</link>
      <pubDate>Thu, 12 Feb 2026 10:00:00 GMT</pubDate><description>Another item</description></item>
    </channel></rss>"""

    monkeypatch.setattr(adapter.http, "get_bytes", lambda *args, **kwargs: xml.encode("utf-8"))

    result = adapter.collect(
        AdapterContext(
            ticker="MSFT",
            as_of_date="2026-02-13",
            company_name="Microsoft",
            packet={},
            ir_rss_url="https://news.microsoft.com/feed/",
            homepage_url="https://www.microsoft.com",
        )
    )
    assert len(result.evidence_items) == 2
    assert all(item.source_type == "ir_press" for item in result.evidence_items)


def test_ir_press_allows_per_ticker_allowlist_domain(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path)
    adapter = IRPressAdapter(cfg)

    xml = """<?xml version='1.0' encoding='UTF-8'?>
    <rss version='2.0'><channel>
      <item><title>Press 1</title><link>https://investors.example.net/p1</link>
      <pubDate>Fri, 13 Feb 2026 10:00:00 GMT</pubDate><description>Hello world</description></item>
    </channel></rss>"""

    monkeypatch.setattr(adapter.http, "get_bytes", lambda *args, **kwargs: xml.encode("utf-8"))

    result = adapter.collect(
        AdapterContext(
            ticker="TCKR",
            as_of_date="2026-02-13",
            company_name="Ticker Example",
            packet={},
            ir_rss_url="https://investors.example.net/feed/",
            homepage_url="https://ticker.example.net",
            allowlist_domains=("investors.example.net",),
        )
    )

    assert len(result.evidence_items) == 1
    assert result.evidence_items[0].source_url == "https://investors.example.net/p1"
