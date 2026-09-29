from __future__ import annotations

import json

from app.research.adapters.base import AdapterContext
from app.research.adapters.transcripts import TranscriptAdapter


def _cfg(monkeypatch, tmp_path, *, enabled: bool = True, safe_mode: bool = False):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true" if safe_mode else "false")
    monkeypatch.setenv("VOE_RESEARCH_ENABLE_TRANSCRIPTS", "true" if enabled else "false")
    monkeypatch.setenv("VOE_RESEARCH_TRANSCRIPT_PROVIDER", "alpha_vantage" if enabled else "disabled")
    monkeypatch.setenv("VOE_RESEARCH_TRANSCRIPT_MAX_QUARTERS", "2")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key" if enabled else "")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    return _get_config()


def _ctx(as_of_date: str = "2026-04-18") -> AdapterContext:
    return AdapterContext(
        ticker="AAA",
        as_of_date=as_of_date,
        company_name="Example Co",
        packet={},
        run_id="run_test",
    )


def test_transcript_adapter_parses_alpha_vantage_segments(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path)
    adapter = TranscriptAdapter(cfg)
    requested_urls: list[str] = []

    def fake_get(url, **kwargs):
        requested_urls.append(url)
        if "quarter=2026Q1" in url:
            return json.dumps(
                {
                    "symbol": "AAA",
                    "quarter": "2026Q1",
                    "transcript": [
                        {
                            "speaker": "Jane CEO",
                            "title": "Chief Executive Officer",
                            "content": "We raised guidance because retention and pipeline improved.",
                            "sentiment": "positive",
                        },
                        {
                            "speaker": "Analyst",
                            "content": "Can you discuss gross margin durability?",
                        },
                    ],
                }
            ).encode("utf-8")
        return json.dumps({"transcript": []}).encode("utf-8")

    monkeypatch.setattr(adapter.http, "get_bytes", fake_get)

    # Migrated 2026-09-29: the provider gives no call date, so a 2026Q1
    # transcript is dated 2026-05-15 (quarter end + 45 days), not 2026-03-31;
    # the as-of date moves past that so the parse is still exercised.
    result = adapter.collect(_ctx("2026-06-02"))

    assert len(result.evidence_items) == 1
    item = result.evidence_items[0]
    assert "apikey=test-key" in requested_urls[0]
    assert item.source_type == "TRANSCRIPT"
    assert item.source_url == "https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol=AAA&quarter=2026Q1"
    assert item.source_title == "AAA earnings call transcript 2026Q1"
    assert item.source_published_at == "2026-05-15"
    assert item.source_quality == {
        "source_family": "management_transcript",
        "source_origin": "primary_company_controlled",
        "source_independence": "company_controlled",
        "source_domain": "www.alphavantage.co",
        "freshness_days": 18,
        "freshness_bucket": "current_30d",
        "source_quality_score": 0.88,
        "calibration_status": "deterministic_heuristic",
        "reason_codes": [
            "SOURCE_MANAGEMENT_TRANSCRIPT",
            "SOURCE_ISSUER_BIAS_POSSIBLE",
            "FRESHNESS_CURRENT_30D",
        ],
    }
    assert "Jane CEO / Chief Executive Officer: We raised guidance" in item.excerpt_text
    assert "sentiment=positive" in item.excerpt_text
    assert item.citations[0].section_label == "TRANSCRIPT"
    assert "apikey" not in item.citations[0].source_url


def test_transcript_adapter_disabled_creates_gap(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path, enabled=False)
    adapter = TranscriptAdapter(cfg)

    result = adapter.collect(_ctx())

    assert result.evidence_items == []
    assert result.evidence_gaps[0].gap_id == "GAP_TRANSCRIPTS_DISABLED"


def test_transcript_adapter_provider_message_creates_gap(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path)
    adapter = TranscriptAdapter(cfg)
    monkeypatch.setattr(
        adapter.http,
        "get_bytes",
        lambda *args, **kwargs: json.dumps({"Note": "rate limit"}).encode("utf-8"),
    )

    result = adapter.collect(_ctx())

    assert result.evidence_items == []
    assert [gap.gap_id for gap in result.evidence_gaps] == [
        "GAP_TRANSCRIPT_PROVIDER_MESSAGE",
        "GAP_TRANSCRIPT_PROVIDER_MESSAGE",
    ]


def _one_transcript(payload_extra: dict):
    def fake_get(url, **kwargs):
        if "quarter=2026Q1" in url:
            return json.dumps(
                {"symbol": "AAA", "quarter": "2026Q1", "transcript": "We raised guidance.", **payload_extra}
            ).encode("utf-8")
        return json.dumps({"transcript": []}).encode("utf-8")

    return fake_get


def test_transcript_without_call_date_is_not_visible_before_quarter_end_plus_45(monkeypatch, tmp_path):
    """No call date from the provider: the call is dated quarter end + 45 days
    (2026-05-15 for 2026Q1), so on 2026-04-18 it had not happened yet."""
    adapter = TranscriptAdapter(_cfg(monkeypatch, tmp_path))
    monkeypatch.setattr(adapter.http, "get_bytes", _one_transcript({}))

    assert adapter.collect(_ctx("2026-04-18")).evidence_items == []

    item = adapter.collect(_ctx("2026-05-15")).evidence_items[0]
    assert item.source_published_at == "2026-05-15"
    assert item.excerpt_text.startswith(
        "[Dated 2026-05-15: the provider gave no call date, so the transcript is dated "
        "45 days after its quarter ends.]"
    )


def test_transcript_with_call_date_is_dated_by_the_call(monkeypatch, tmp_path):
    adapter = TranscriptAdapter(_cfg(monkeypatch, tmp_path))
    monkeypatch.setattr(adapter.http, "get_bytes", _one_transcript({"date": "2026-04-16"}))

    item = adapter.collect(_ctx("2026-04-18")).evidence_items[0]
    assert item.source_published_at == "2026-04-16"
    assert item.excerpt_text == "We raised guidance."

    assert adapter.collect(_ctx("2026-04-15")).evidence_items == []
