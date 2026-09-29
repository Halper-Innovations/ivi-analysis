from __future__ import annotations

import json

from app.research.adapters.base import AdapterContext
from app.research.adapters.external_news import ExternalNewsAdapter


def _cfg(monkeypatch, tmp_path, *, enabled: bool = True, safe_mode: bool = False, reputation_path=None):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true" if safe_mode else "false")
    monkeypatch.setenv("VOE_RESEARCH_EXTERNAL_NEWS_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv("VOE_RESEARCH_EXTERNAL_NEWS_PROVIDER", "alpha_vantage" if enabled else "disabled")
    monkeypatch.setenv("VOE_RESEARCH_EXTERNAL_NEWS_MAX_ITEMS", "2")
    if reputation_path is not None:
        monkeypatch.setenv("VOE_RESEARCH_SOURCE_REPUTATION_PATH", str(reputation_path))
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key" if enabled else "")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    return _get_config()


def _ctx() -> AdapterContext:
    return AdapterContext(
        ticker="AAA",
        as_of_date="2026-04-18",
        company_name="Example Co",
        packet={},
        run_id="run_test",
    )


def test_external_news_adapter_parses_alpha_vantage_feed(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path)
    adapter = ExternalNewsAdapter(cfg)
    requested_urls: list[str] = []

    def fake_get(url, **kwargs):
        requested_urls.append(url)
        return json.dumps(
            {
                "feed": [
                    {
                        "title": "Industry channel reports demand improvement",
                        "url": "https://reputable.example.com/story",
                        "time_published": "20260417T1300",
                        "summary": "Independent channel checks reported better retention and pipeline conversion.",
                        "source": "Reputable News",
                        "source_domain": "reputable.example.com",
                        "overall_sentiment_score": "0.35",
                        "overall_sentiment_label": "Somewhat-Bullish",
                    },
                    {
                        "title": "Duplicate",
                        "url": "https://reputable.example.com/story",
                        "time_published": "20260417T1400",
                        "summary": "Duplicate URL should be ignored.",
                    },
                ]
            }
        ).encode("utf-8")

    monkeypatch.setattr(adapter.http, "get_bytes", fake_get)

    result = adapter.collect(_ctx())

    assert len(result.evidence_items) == 1
    item = result.evidence_items[0]
    assert "function=NEWS_SENTIMENT" in requested_urls[0]
    assert "tickers=AAA" in requested_urls[0]
    assert "apikey=test-key" in requested_urls[0]
    assert item.source_type == "external_news"
    assert item.source_url == "https://reputable.example.com/story"
    assert item.source_title == "Industry channel reports demand improvement"
    assert item.source_published_at == "2026-04-17T13:00:00+00:00"
    assert "Source: Reputable News" in item.excerpt_text
    assert "Overall sentiment: Somewhat-Bullish (0.35)." in item.excerpt_text
    assert item.citations[0].section_label == "external_news"
    assert item.source_quality == {
        "source_family": "external_news",
        "source_origin": "secondary",
        "source_independence": "independent_or_third_party",
        "source_domain": "reputable.example.com",
        "freshness_days": 1,
        "freshness_bucket": "recent_7d",
        "source_quality_score": 0.71,
        "calibration_status": "heuristic_unvalidated",
        "reason_codes": [
            "SOURCE_EXTERNAL_SECONDARY",
            "SECONDARY_SOURCE_CALIBRATION_PENDING",
            "FRESHNESS_RECENT_7D",
        ],
    }


def test_external_news_adapter_applies_source_reputation_history(monkeypatch, tmp_path):
    reputation_path = tmp_path / "source_reputation_history.csv"
    reputation_path.write_text(
        "domain,as_of_date,reputation_score,sample_size\n"
        "reputable.example.com,2026-04-01,0.92,12\n",
        encoding="utf-8",
    )
    cfg = _cfg(monkeypatch, tmp_path, reputation_path=reputation_path)
    adapter = ExternalNewsAdapter(cfg)

    monkeypatch.setattr(
        adapter.http,
        "get_bytes",
        lambda *args, **kwargs: json.dumps(
            {
                "feed": [
                    {
                        "title": "Industry channel reports demand improvement",
                        "url": "https://reputable.example.com/story",
                        "time_published": "20260417T1300",
                        "summary": "Independent channel checks reported better retention and pipeline conversion.",
                        "source": "Reputable News",
                        "source_domain": "reputable.example.com",
                    }
                ]
            }
        ).encode("utf-8"),
    )

    result = adapter.collect(_ctx())

    assert len(result.evidence_items) == 1
    assert result.evidence_items[0].source_quality == {
        "source_family": "external_news",
        "source_origin": "secondary",
        "source_independence": "independent_or_third_party",
        "source_domain": "reputable.example.com",
        "freshness_days": 1,
        "freshness_bucket": "recent_7d",
        "source_quality_score": 0.82,
        "calibration_status": "domain_reputation_calibrated",
        "reason_codes": [
            "SOURCE_EXTERNAL_SECONDARY",
            "FRESHNESS_RECENT_7D",
            "SOURCE_REPUTATION_HISTORY",
            "SOURCE_REPUTATION_HIGH",
        ],
        "source_reputation_status": "known",
        "source_reputation_score": 0.92,
        "source_reputation_as_of_date": "2026-04-01",
        "source_reputation_sample_size": 12,
    }


def test_external_news_adapter_disabled_creates_gap(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path, enabled=False)
    adapter = ExternalNewsAdapter(cfg)

    result = adapter.collect(_ctx())

    assert result.evidence_items == []
    assert result.evidence_gaps[0].gap_id == "GAP_EXTERNAL_NEWS_DISABLED"


def test_external_news_adapter_provider_message_creates_gap(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path)
    adapter = ExternalNewsAdapter(cfg)
    monkeypatch.setattr(
        adapter.http,
        "get_bytes",
        lambda *args, **kwargs: json.dumps({"Note": "rate limit"}).encode("utf-8"),
    )

    result = adapter.collect(_ctx())

    assert result.evidence_items == []
    assert [gap.gap_id for gap in result.evidence_gaps] == ["GAP_EXTERNAL_NEWS_PROVIDER_MESSAGE"]
