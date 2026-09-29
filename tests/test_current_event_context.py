from __future__ import annotations

from datetime import datetime, timezone

from app.db import get_db, init_db
from app.research.adapters.base import AdapterResult
from app.research.current_event_context import load_current_event_context
from app.research.schemas import CitationRef, EvidenceItem


def _cfg(
    monkeypatch,
    tmp_path,
    *,
    safe_mode: bool = True,
    ir_enabled: bool = True,
    news_enabled: bool = True,
    transcripts_enabled: bool = False,
    external_news_enabled: bool = False,
):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true" if safe_mode else "false")
    monkeypatch.setenv("VOE_RESEARCH_IR_PRESS_ENABLED", "true" if ir_enabled else "false")
    monkeypatch.setenv("VOE_RESEARCH_COMPANY_NEWS_ENABLED", "true" if news_enabled else "false")
    monkeypatch.setenv("VOE_RESEARCH_ENABLE_TRANSCRIPTS", "true" if transcripts_enabled else "false")
    monkeypatch.setenv("VOE_RESEARCH_TRANSCRIPT_PROVIDER", "alpha_vantage" if transcripts_enabled else "disabled")
    monkeypatch.setenv("VOE_RESEARCH_EXTERNAL_NEWS_ENABLED", "true" if external_news_enabled else "false")
    monkeypatch.setenv("VOE_RESEARCH_EXTERNAL_NEWS_PROVIDER", "alpha_vantage" if external_news_enabled else "disabled")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key" if transcripts_enabled or external_news_enabled else "")
    monkeypatch.setenv("VOE_RESEARCH_ALLOWLIST_DOMAINS", "example.com,investor.example.com")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    return _get_config()


def _item(
    *,
    source_type: str,
    source_url: str,
    title: str,
    published_at: str | None,
    excerpt_text: str,
    source_quality: dict | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        id=f"{source_type}:{title}",
        ticker="TEST",
        as_of_date="2026-04-18",
        source_type=source_type,
        source_url=source_url,
        source_title=title,
        source_published_at=published_at,
        retrieved_at=datetime(2026, 4, 18, tzinfo=timezone.utc).isoformat(),
        excerpt_text=excerpt_text,
        citations=[CitationRef(source_url=source_url, snippet=excerpt_text[:40], section_label=source_type)],
        hash=f"hash:{title}",
        content_hash=f"content:{title}",
        dedupe_key=f"dedupe:{title}",
        source_quality=source_quality,
    )


def test_load_current_event_context_allows_safe_mode_for_canonical(monkeypatch, tmp_path):
    _cfg(monkeypatch, tmp_path, safe_mode=True)
    monkeypatch.setattr(
        "app.research.current_event_context._load_company_metadata",
        lambda ticker: {"name": "Example Co", "homepage_url": "https://example.com", "ir_rss_url": "https://example.com/feed.xml"},
    )

    observed_safe_modes: list[bool] = []

    def fake_collect(self, ctx):
        observed_safe_modes.append(self.cfg.safe_mode)
        return AdapterResult(
            evidence_items=[
                _item(
                    source_type=self.source_type,
                    source_url=f"https://example.com/{self.source_type}/one",
                    title=f"{self.source_type} headline",
                    published_at="2026-04-17T10:00:00+00:00",
                    excerpt_text="Recent company-controlled update.",
                )
            ]
        )

    monkeypatch.setattr("app.research.current_event_context.IRPressAdapter.collect", fake_collect)
    monkeypatch.setattr("app.research.current_event_context.CompanyNewsAdapter.collect", fake_collect)

    context = load_current_event_context("TEST", as_of_date="2026-04-18")

    assert [doc.source_type for doc in context.documents] == ["ir_press", "company_news"]
    assert observed_safe_modes == [False, False]


def test_load_current_event_context_includes_enabled_transcripts(monkeypatch, tmp_path):
    _cfg(
        monkeypatch,
        tmp_path,
        safe_mode=False,
        ir_enabled=False,
        news_enabled=False,
        transcripts_enabled=True,
    )
    monkeypatch.setattr(
        "app.research.current_event_context._load_company_metadata",
        lambda ticker: {"name": "Example Co", "homepage_url": "", "ir_rss_url": ""},
    )

    def fake_transcript_collect(self, ctx):
        return AdapterResult(
            evidence_items=[
                _item(
                    source_type="TRANSCRIPT",
                    source_url="https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol=TEST&quarter=2026Q1",
                    title="TEST earnings call transcript 2026Q1",
                    published_at="2026-03-31",
                    excerpt_text="Management raised guidance and discussed durable margins.",
                )
            ]
        )

    monkeypatch.setattr("app.research.current_event_context.TranscriptAdapter.collect", fake_transcript_collect)
    monkeypatch.setattr("app.research.current_event_context.IRPressAdapter.collect", lambda self, ctx: AdapterResult())
    monkeypatch.setattr("app.research.current_event_context.CompanyNewsAdapter.collect", lambda self, ctx: AdapterResult())

    context = load_current_event_context("TEST", as_of_date="2026-04-18")

    assert [doc.source_type for doc in context.documents] == ["TRANSCRIPT"]
    assert context.documents[0].title == "TEST earnings call transcript 2026Q1"
    assert context.warnings == [
        "current_event_source_disabled:ir_press",
        "current_event_source_disabled:company_news",
    ]


def test_load_current_event_context_includes_enabled_external_news(monkeypatch, tmp_path):
    _cfg(
        monkeypatch,
        tmp_path,
        safe_mode=False,
        ir_enabled=False,
        news_enabled=False,
        external_news_enabled=True,
    )
    monkeypatch.setattr(
        "app.research.current_event_context._load_company_metadata",
        lambda ticker: {"name": "Example Co", "homepage_url": "", "ir_rss_url": ""},
    )

    def fake_external_news_collect(self, ctx):
        return AdapterResult(
            evidence_items=[
                _item(
                    source_type="external_news",
                    source_url="https://reputable.example.com/story",
                    title="External channel checks improve",
                    published_at="2026-04-17T13:00:00+00:00",
                    excerpt_text="Independent source reported improved retention.",
                    source_quality={
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
                    },
                )
            ]
        )

    monkeypatch.setattr("app.research.current_event_context.ExternalNewsAdapter.collect", fake_external_news_collect)
    monkeypatch.setattr("app.research.current_event_context.IRPressAdapter.collect", lambda self, ctx: AdapterResult())
    monkeypatch.setattr("app.research.current_event_context.CompanyNewsAdapter.collect", lambda self, ctx: AdapterResult())

    context = load_current_event_context("TEST", as_of_date="2026-04-18")

    assert [doc.source_type for doc in context.documents] == ["external_news"]
    assert context.documents[0].source_url == "https://reputable.example.com/story"
    assert context.documents[0].source_quality == {
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
    assert context.warnings == [
        "current_event_source_disabled:ir_press",
        "current_event_source_disabled:company_news",
    ]


def test_load_current_event_context_dedupes_across_sources_and_caps(monkeypatch, tmp_path):
    _cfg(monkeypatch, tmp_path, safe_mode=True)
    monkeypatch.setattr(
        "app.research.current_event_context._load_company_metadata",
        lambda ticker: {"name": "Example Co", "homepage_url": "https://example.com", "ir_rss_url": "https://example.com/feed.xml"},
    )

    def fake_ir(self, ctx):
        return AdapterResult(
            evidence_items=[
                _item(
                    source_type="ir_press",
                    source_url="https://example.com/shared",
                    title="Shared title",
                    published_at="2026-04-17T10:00:00+00:00",
                    excerpt_text="Shared item",
                )
            ]
            + [
                _item(
                    source_type="ir_press",
                    source_url=f"https://example.com/ir/{idx}",
                    title=f"IR {idx}",
                    published_at=f"2026-03-{idx:02d}T10:00:00+00:00",
                    excerpt_text=f"IR item {idx}",
                )
                for idx in range(1, 23)
            ]
        )

    def fake_news(self, ctx):
        return AdapterResult(
            evidence_items=[
                _item(
                    source_type="company_news",
                    source_url="https://example.com/shared",
                    title="Shared title",
                    published_at="2026-04-17T10:00:00+00:00",
                    excerpt_text="Shared item from news",
                )
            ]
        )

    monkeypatch.setattr("app.research.current_event_context.IRPressAdapter.collect", fake_ir)
    monkeypatch.setattr("app.research.current_event_context.CompanyNewsAdapter.collect", fake_news)

    context = load_current_event_context("TEST", as_of_date="2026-04-18")

    assert len(context.documents) == 20
    assert context.documents[0].source_type == "ir_press"
    assert context.documents[0].source_url == "https://example.com/shared"
    assert "current_event_items_truncated:20/23" in context.warnings


def test_load_current_event_context_filters_old_items_without_warning(monkeypatch, tmp_path):
    _cfg(monkeypatch, tmp_path, safe_mode=True)
    monkeypatch.setattr(
        "app.research.current_event_context._load_company_metadata",
        lambda ticker: {"name": "Example Co", "homepage_url": "https://example.com", "ir_rss_url": "https://example.com/feed.xml"},
    )

    def fake_ir(self, ctx):
        return AdapterResult(
            evidence_items=[
                _item(
                    source_type="ir_press",
                    source_url="https://example.com/old",
                    title="Old item",
                    published_at="2025-12-01T10:00:00+00:00",
                    excerpt_text="Too old",
                )
            ]
        )

    def fake_news(self, ctx):
        return AdapterResult(
            evidence_gaps=[
                type("Gap", (), {"gap_id": "GAP_COMPANY_NEWS_NOT_FOUND"})(),
            ]
        )

    monkeypatch.setattr("app.research.current_event_context.IRPressAdapter.collect", fake_ir)
    monkeypatch.setattr("app.research.current_event_context.CompanyNewsAdapter.collect", fake_news)

    context = load_current_event_context("TEST", as_of_date="2026-04-18")

    assert context.documents == []
    assert context.warnings == []


def test_load_current_event_context_warns_when_sources_disabled(monkeypatch, tmp_path):
    _cfg(monkeypatch, tmp_path, safe_mode=True, ir_enabled=False, news_enabled=False)
    monkeypatch.setattr(
        "app.research.current_event_context._load_company_metadata",
        lambda ticker: {"name": "Example Co", "homepage_url": "https://example.com", "ir_rss_url": "https://example.com/feed.xml"},
    )

    context = load_current_event_context("TEST", as_of_date="2026-04-18")

    assert context.documents == []
    assert context.warnings == [
        "current_event_source_disabled:ir_press",
        "current_event_source_disabled:company_news",
    ]


def test_load_current_event_context_falls_back_to_universe_member_metadata(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path, safe_mode=True)
    init_db(cfg)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO universe_members(
                universe_id, ticker, cik, name, ir_rss_url, homepage_url, allowlist_domains, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                "TEST",
                "0000000001",
                "Universe Example",
                "https://ir.universe-example.com/feed.xml",
                "https://universe-example.com",
                "universe-example.com,ir.universe-example.com",
                "2026-04-18T00:00:00+00:00",
            ),
        )

    observed = []

    def fake_collect(self, ctx):
        observed.append(
            (
                self.source_type,
                ctx.company_name,
                ctx.homepage_url,
                ctx.ir_rss_url,
                ctx.allowlist_domains,
            )
        )
        return AdapterResult()

    monkeypatch.setattr("app.research.current_event_context.IRPressAdapter.collect", fake_collect)
    monkeypatch.setattr("app.research.current_event_context.CompanyNewsAdapter.collect", fake_collect)

    context = load_current_event_context("TEST", as_of_date="2026-04-18")

    assert context.metadata_source == "universe_members"
    assert context.homepage_url_present is True
    assert context.ir_rss_url_present is True
    assert context.allowlist_domains == ["universe-example.com", "ir.universe-example.com"]
    assert observed == [
        (
            "ir_press",
            "Universe Example",
            "https://universe-example.com",
            "https://ir.universe-example.com/feed.xml",
            ("universe-example.com", "ir.universe-example.com"),
        ),
        (
            "company_news",
            "Universe Example",
            "https://universe-example.com",
            "https://ir.universe-example.com/feed.xml",
            ("universe-example.com", "ir.universe-example.com"),
        ),
    ]


def test_load_current_event_context_falls_back_to_local_metadata_files(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    universe_dir = data_dir / "universe"
    universe_dir.mkdir(parents=True)
    (universe_dir / "metadata_overrides.csv").write_text(
        "ticker,name,homepage_url,ir_rss_url,allowlist_domains,notes\n"
        "TEST,Override Example,https://override.example.com,https://ir.override.example.com/feed.xml,"
        "override.example.com;ir.override.example.com,local metadata\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe_dir / "universe.csv"))
    cfg = _cfg(monkeypatch, tmp_path, safe_mode=True)
    init_db(cfg)

    observed = []

    def fake_collect(self, ctx):
        observed.append(
            (
                self.source_type,
                ctx.company_name,
                ctx.homepage_url,
                ctx.ir_rss_url,
                ctx.allowlist_domains,
            )
        )
        return AdapterResult()

    monkeypatch.setattr("app.research.current_event_context.IRPressAdapter.collect", fake_collect)
    monkeypatch.setattr("app.research.current_event_context.CompanyNewsAdapter.collect", fake_collect)

    context = load_current_event_context("TEST", as_of_date="2026-04-18")

    assert context.metadata_source == "metadata_overrides_csv"
    assert context.homepage_url_present is True
    assert context.ir_rss_url_present is True
    assert context.allowlist_domains == ["override.example.com", "ir.override.example.com"]
    assert context.allowlist_source == "metadata_overrides_csv"
    assert observed == [
        (
            "ir_press",
            "Override Example",
            "https://override.example.com",
            "https://ir.override.example.com/feed.xml",
            ("override.example.com", "ir.override.example.com"),
        ),
        (
            "company_news",
            "Override Example",
            "https://override.example.com",
            "https://ir.override.example.com/feed.xml",
            ("override.example.com", "ir.override.example.com"),
        ),
    ]
