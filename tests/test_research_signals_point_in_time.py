"""Research signals see only what existed by the as-of day.

An item published after the as-of day must not raise a sentiment or risk
flag, add a topic or count as an item. An undated item counts only when it was
retrieved by the as-of day (a live run). Timestamps without an offset are UTC,
not the machine's local time.
"""

from __future__ import annotations

import json

from app.research.adapters.base import AdapterContext
from app.research.adapters.external_news import ExternalNewsAdapter, _request_url
from app.research.schemas import CitationRef, EvidenceItem
from app.research.signals import _parse_ts, compute_research_signals


def _item(title: str, pub: str | None, excerpt: str, *, retrieved: str = "2026-09-29T00:00:00+00:00") -> EvidenceItem:
    url = f"https://news.example.org/{title.replace(' ', '-')}"
    return EvidenceItem(
        id=f"ev_{title}",
        ticker="AAA",
        as_of_date="2026-02-13",
        source_type="company_news",  # type: ignore[arg-type]
        source_url=url,
        source_title=title,
        source_published_at=pub,
        retrieved_at=retrieved,
        excerpt_text=excerpt,
        citations=[CitationRef(source_url=url, snippet=excerpt[:80], section_label="test")],
        hash="h",
        content_hash="c",
        dedupe_key=f"d_{title}",
        adapter_run_id="run_test",
    )


def test_post_as_of_items_do_not_drive_signals():
    items = [
        _item("Quarterly update", "2026-02-10T00:00:00+00:00", "Steady quarter."),
        _item("Chapter 11 filing", "2026-02-20T00:00:00+00:00", "The company filed for chapter 11 bankruptcy."),
        _item("Undated going concern", None, "Substantial doubt about its ability to continue as a going concern."),
    ]
    signals = compute_research_signals(
        ticker="AAA", as_of_date="2026-02-13", run_id="r", evidence_items=items
    )
    assert signals.sentiment_flags == []
    assert signals.evidence_item_ids == ["ev_Quarterly update"]
    assert "chapter" not in signals.key_topics


def test_item_published_on_the_as_of_day_counts():
    items = [_item("Chapter 11 filing", "2026-02-13T18:00:00+00:00", "Filed for chapter 11.")]
    signals = compute_research_signals(
        ticker="AAA", as_of_date="2026-02-13", run_id="r", evidence_items=items
    )
    assert signals.sentiment_flags == ["bankruptcy"]
    assert signals.recency_days_min == 0


def test_undated_item_retrieved_by_the_as_of_day_counts():
    items = [
        _item("Restructuring", None, "Announced a restructuring.", retrieved="2026-02-13T09:00:00+00:00")
    ]
    signals = compute_research_signals(
        ticker="AAA", as_of_date="2026-02-13", run_id="r", evidence_items=items
    )
    assert signals.sentiment_flags == ["restructuring"]


def test_naive_timestamp_is_utc():
    assert _parse_ts("2026-02-13T23:30:00").isoformat() == "2026-02-13T23:30:00+00:00"
    assert _parse_ts("2026-02-13T23:30:00-05:00").isoformat() == "2026-02-14T04:30:00+00:00"


def test_external_news_request_is_bounded_by_the_as_of_day():
    url = _request_url("AAA", "k", 5, "2026-04-18")
    assert url == (
        "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAA&sort=LATEST"
        "&limit=5&time_to=20260418T2359&apikey=k"
    )


def test_external_news_drops_stories_after_the_as_of_day(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_SAFE_MODE", "false")
    monkeypatch.setenv("VOE_RESEARCH_EXTERNAL_NEWS_ENABLED", "true")
    monkeypatch.setenv("VOE_RESEARCH_EXTERNAL_NEWS_PROVIDER", "alpha_vantage")
    monkeypatch.setenv("VOE_ALPHA_VANTAGE_API_KEY", "test-key")
    from app.config import get_config

    get_config.cache_clear()
    adapter = ExternalNewsAdapter(get_config())

    def row(title: str, when: str | None) -> dict:
        out = {"title": title, "url": f"https://n.example.org/{title}", "summary": f"{title} summary"}
        if when:
            out["time_published"] = when
        return out

    feed = {"feed": [row("before", "20260418T2300"), row("after", "20260419T0100"), row("undated", None)]}
    monkeypatch.setattr(adapter.http, "get_bytes", lambda url, **kw: json.dumps(feed).encode())
    ctx = AdapterContext(ticker="AAA", as_of_date="2026-04-18", company_name="A", packet={}, run_id="r")
    result = adapter.collect(ctx)
    get_config.cache_clear()
    assert [item.source_title for item in result.evidence_items] == ["before"]
