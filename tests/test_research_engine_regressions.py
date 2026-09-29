"""Regressions for the research layer (app/research/).

Each test asserts the correct answer for a defect that has been fixed; a
future defect of this kind belongs here as a strict ``xfail`` until fixed. All
fixtures are hermetic: temp DB, adapters stubbed, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.db import get_db, init_db, utc_now_iso
from app.research import engine
from app.research.adapters.base import AdapterContext, AdapterResult, ResearchAdapter
from app.research.schemas import CitationRef

TICKER = "XMPL"
AS_OF = "2024-06-30"


def _init_temp_db(monkeypatch, tmp_path, **env: str):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


class _StubAdapter(ResearchAdapter):
    source_type = "ir_press"

    def collect(self, ctx: AdapterContext) -> AdapterResult:
        out = AdapterResult()
        out.evidence_items.append(
            self._make_item(
                ctx=ctx,
                source_type="ir_press",
                source_url="https://example.com/pr/1",
                source_title="Results",
                source_published_at="2024-05-01T00:00:00+00:00",
                excerpt_text="quarterly results and cash flow",
                citations=[
                    CitationRef(
                        source_url="https://example.com/pr/1", snippet="s", section_label="ir"
                    )
                ],
            )
        )
        return out


def _evidence_packet(dcf_base: float, filing_date: str) -> dict:
    return {
        "ticker": TICKER,
        "fundamentals": {
            "revenue": 100.0,
            "operating_margin": 0.1,
            "fcf": 5.0,
            "net_debt": 1.0,
            "issuer_classification": "operating",
        },
        "valuations": {"dcf": {"outputs": {"base": dcf_base}}},
        "extracted_facts": [],
        "financials": [],
        "filings_used": [{"form_type": "10-K", "filing_date": filing_date}],
    }


def _seed_packets(tmp_path: Path) -> None:
    """One packet visible on the as-of date (2024-03-31) and one from the future."""
    packets_dir = tmp_path / "packets"
    packets_dir.mkdir(exist_ok=True)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO companies(ticker, cik, name, created_at) VALUES(?, '1', ?, ?)",
            (TICKER, "Example Co", utc_now_iso()),
        )
        for as_of, base in (("2024-03-31", 10.0), ("2025-12-31", 99.0)):
            path = packets_dir / f"{TICKER}_{as_of}.json"
            path.write_text(json.dumps(_evidence_packet(base, as_of)), encoding="utf-8")
            conn.execute(
                "INSERT INTO evidence_packets(ticker, as_of_date, packet_path, packet_hash, "
                "created_at) VALUES(?, ?, ?, 'h', ?)",
                (TICKER, as_of, str(path), utc_now_iso()),
            )


# ── engine._latest_packet_row: fallback has no as-of bound ──────────────────


def test_packet_fallback_returns_future_packet(monkeypatch, tmp_path):
    """When no packet matches the requested date exactly, the newest packet on
    or before that date must be used — never one dated after it.

    Observed: as-of 2024-06-30 with packets dated 2024-03-31 and 2025-12-31
    returns the 2025-12-31 packet. Correct: 2024-03-31.
    """
    _init_temp_db(monkeypatch, tmp_path)
    _seed_packets(tmp_path)
    with get_db() as conn:
        row = engine._latest_packet_row(conn, TICKER, as_of_date=AS_OF)
    assert row["as_of_date"] == "2024-03-31"


def test_historical_research_run_consumes_future_valuation(monkeypatch, tmp_path):
    """End to end: the finding must cite the as-of-visible DCF (10.0), and the
    research packet must be stored under the requested as-of date.

    Observed: finding cites "DCF base=99.0", the catalyst names a 10-K filed
    2025-12-31, research_packets holds ('2025-12-31', run) and the lookup for
    2024-06-30 returns None.
    """
    _init_temp_db(monkeypatch, tmp_path)
    _seed_packets(tmp_path)
    monkeypatch.setattr(engine, "_select_adapters", lambda cfg, source_filters: [_StubAdapter(cfg)])
    out_path = engine.run_research_agent_for_ticker(TICKER, as_of_date=AS_OF, run_id="run_hist")
    payload = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert "DCF base=10.0" in payload["findings"][0]["summary"]
    with get_db() as conn:
        assert engine.latest_research_packet_path(conn, TICKER, as_of_date=AS_OF) is not None


# ── engine._top_ranked_tickers: ranks stale rows, returns duplicates ────────


def _seed_scores() -> None:
    rows = [
        ("AAA", "2024-01-31", 90.0),  # stale: five months old
        ("AAA", AS_OF, 40.0),  # AAA's CURRENT score on the as-of date
        ("BBB", AS_OF, 70.0),
        ("CCC", AS_OF, 60.0),
    ]
    with get_db() as conn:
        for ticker, as_of, score in rows:
            conn.execute(
                "INSERT INTO scores(ticker, as_of_date, run_id, subscores_json, total_score, "
                "decision, reasons_json, created_at) VALUES(?, ?, 'r', '{}', ?, 'WATCH', '[]', ?)",
                (ticker, as_of, score, utc_now_iso()),
            )


def test_top_ranked_uses_stale_score(monkeypatch, tmp_path):
    """Observed: top 2 on 2024-06-30 = ['AAA', 'BBB'] (AAA's January 90 beats
    its current 40). Correct: ['BBB', 'CCC']."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_scores()
    with get_db() as conn:
        assert engine._top_ranked_tickers(conn, top_n=2, as_of_date=AS_OF) == ["BBB", "CCC"]


def test_top_ranked_returns_duplicate_tickers(monkeypatch, tmp_path):
    """Observed: top 4 = ['AAA', 'BBB', 'CCC', 'AAA']. Correct: ['BBB', 'CCC', 'AAA']."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_scores()
    with get_db() as conn:
        assert engine._top_ranked_tickers(conn, top_n=4, as_of_date=AS_OF) == ["BBB", "CCC", "AAA"]


# ── signals.compute_research_signals: future / undated evidence is "fresh" ──


def _item(adapter, ctx, url: str, published: str | None):
    return adapter._make_item(
        ctx=ctx,
        source_type="ir_press",
        source_url=url,
        source_title="Press release",
        source_published_at=published,
        excerpt_text="results",
        citations=[CitationRef(source_url=url, snippet="s", section_label="ir")],
    )


def _signals_for(monkeypatch, tmp_path, published_second: str | None):
    from app.research.signals import compute_research_signals

    cfg = _init_temp_db(monkeypatch, tmp_path)
    adapter = _StubAdapter(cfg)
    ctx = AdapterContext(
        ticker=TICKER, as_of_date=AS_OF, company_name="Example", packet={}, run_id="r"
    )
    old = _item(adapter, ctx, "https://example.com/pr/old", "2023-12-01T00:00:00+00:00")
    other = _item(adapter, ctx, "https://example.com/pr/other", published_second)
    return compute_research_signals(
        ticker=TICKER, as_of_date=AS_OF, run_id="r", evidence_items=[old, other]
    )


def test_future_dated_evidence_scores_as_recent(monkeypatch, tmp_path):
    """The same item is labelled FRESHNESS_FUTURE_DATED by source_quality and
    counted as a 0-day-old, 30-day-flow item by compute_research_signals.

    Observed (items published 2023-12-01 and 2025-01-15, as-of 2024-06-30):
    recency_days_min 0, item_count_30d 1, RECENT / SPARSE. Correct: 212, 0,
    VERY_STALE / QUIET (the future item carries no recency).
    """
    sig = _signals_for(monkeypatch, tmp_path, "2025-01-15T00:00:00+00:00")
    assert sig.recency_days_min == 212
    assert sig.item_count_30d == 0
    assert sig.summary["freshness_bucket"] == "VERY_STALE"


def test_undated_evidence_scores_as_recent_in_historical_run(monkeypatch, tmp_path):
    """Observed: recency_days_min 0, item_count_30d 1, RECENT. Correct: 212, 0,
    VERY_STALE (an undated item carries no recency)."""
    sig = _signals_for(monkeypatch, tmp_path, None)
    assert sig.recency_days_min == 212
    assert sig.item_count_30d == 0
    assert sig.summary["freshness_bucket"] == "VERY_STALE"


# ── research_quality._freshness_score: measured from the wall clock ─────────


def test_research_quality_freshness_depends_on_run_day(monkeypatch, tmp_path):
    """An item published ON the as-of date is 0 days old for that run.

    Observed (run on 2026-09-01): freshness_score 15.0, overall 78.75.
    Correct: 100.0 / 100.0 regardless of the day the score is computed.
    """
    from app.research.research_quality import score_research_packet
    from app.research.schemas import EvidenceLinkedEntry, KeyQuestion, ResearchPacket

    cfg = _init_temp_db(monkeypatch, tmp_path)
    adapter = _StubAdapter(cfg)
    ctx = AdapterContext(
        ticker=TICKER, as_of_date=AS_OF, company_name="Example", packet={}, run_id="r"
    )
    same_day = _item(adapter, ctx, "https://example.com/pr/sameday", f"{AS_OF}T00:00:00+00:00")
    entry = EvidenceLinkedEntry(
        entry_id="F1",
        summary="x",
        evidence_item_ids=[same_day.id],
        citations=[],
        derived_from=["fundamentals.revenue"],
    )
    packet = ResearchPacket(
        run_id="r",
        ticker=TICKER,
        as_of_date=AS_OF,
        generated_at=f"{AS_OF}T00:00:00+00:00",
        key_questions=[KeyQuestion(question_id=f"Q{i}", question="q") for i in range(1, 6)],
        evidence_items=[same_day],
        findings=[entry],
        risks=[entry],
        catalysts=[entry],
        disconfirming_evidence=[entry],
        next_actions=[],
        evidence_gaps=[],
        claims=[],
        signals={},
    )
    quality = score_research_packet(packet, cfg)
    assert quality.freshness_score == 100.0
    assert quality.overall_research_score == 100.0


# ── adapters: date filter has a lower bound only ────────────────────────────

_FEED = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><title>Q1 2024 results</title><link>https://example.com/pr/2024-q1</link>
        <pubDate>Wed, 01 May 2024 12:00:00 GMT</pubDate><description>before as-of</description></item>
  <item><title>Q4 2024 results</title><link>https://example.com/pr/2024-q4</link>
        <pubDate>Wed, 15 Jan 2025 12:00:00 GMT</pubDate><description>AFTER as-of</description></item>
  <item><title>Ancient release</title><link>https://example.com/pr/2019</link>
        <pubDate>Tue, 01 Jan 2019 12:00:00 GMT</pubDate><description>older than max_days_back</description></item>
</channel></rss>"""

_HOME = (
    b'<html><head><link rel="alternate" type="application/rss+xml" '
    b'href="https://example.com/rss"></head><body></body></html>'
)


def _adapter_ctx() -> AdapterContext:
    return AdapterContext(
        ticker=TICKER,
        as_of_date=AS_OF,
        company_name="Example",
        packet={},
        run_id="r",
        ir_rss_url="https://example.com/rss",
        homepage_url="https://example.com",
    )


def test_ir_press_adapter_collects_post_as_of_items(monkeypatch, tmp_path):
    """Observed: 'Q1 2024 results' (2024-05-01) AND 'Q4 2024 results' (2025-01-15)
    collected for an as-of of 2024-06-30. Correct: only the Q1 item."""
    from app.research.adapters.ir_press import IRPressAdapter

    cfg = _init_temp_db(
        monkeypatch,
        tmp_path,
        VOE_SAFE_MODE="false",
        VOE_RESEARCH_IR_PRESS_ENABLED="true",
        VOE_RESEARCH_ALLOWLIST_DOMAINS="example.com",
    )
    adapter = IRPressAdapter(cfg)
    monkeypatch.setattr(adapter, "http_get_bytes", lambda *a, **k: _FEED)
    titles = [item.source_title for item in adapter.collect(_adapter_ctx()).evidence_items]
    assert titles == ["Q1 2024 results"]


def test_company_news_adapter_collects_post_as_of_items(monkeypatch, tmp_path):
    """Observed: both the 2024-05-01 and the 2025-01-15 items collected for an
    as-of of 2024-06-30. Correct: only the Q1 item."""
    from app.research.adapters.company_news import CompanyNewsAdapter

    cfg = _init_temp_db(
        monkeypatch,
        tmp_path,
        VOE_SAFE_MODE="false",
        VOE_RESEARCH_COMPANY_NEWS_ENABLED="true",
        VOE_RESEARCH_ALLOWLIST_DOMAINS="example.com",
    )
    adapter = CompanyNewsAdapter(cfg)
    monkeypatch.setattr(
        adapter,
        "http_get_bytes",
        lambda url, *a, **k: _FEED if str(url).endswith("/rss") else _HOME,
    )
    titles = [item.source_title for item in adapter.collect(_adapter_ctx()).evidence_items]
    assert titles == ["Q1 2024 results"]


# ── engine._persist_evidence_items: upsert leaves rows belonging to no run ──


def test_evidence_upsert_orphans_the_row_from_every_run(monkeypatch, tmp_path):
    """After the second run persists the same press release, the gating count
    for (ticker, as_of_date, run_id) of that run must be 1.

    Observed: stored row as_of_date=2024-06-30 run_id=run_2025; the count for
    (2025-06-30, run_2025) is 0 (and so is (2024-06-30, run_2024)).
    """
    cfg = _init_temp_db(monkeypatch, tmp_path)
    adapter = _StubAdapter(cfg)

    def _item_for(as_of: str, run_id: str):
        ctx = AdapterContext(
            ticker=TICKER, as_of_date=as_of, company_name="Example", packet={}, run_id=run_id
        )
        return _item(adapter, ctx, "https://example.com/pr/1", "2024-05-01T00:00:00+00:00")

    first = _item_for("2024-06-30", "run_2024")
    second = _item_for("2025-06-30", "run_2025")
    assert first.id == second.id
    with get_db() as conn:
        engine._persist_evidence_items(conn, "run_2024", [first])
        engine._persist_evidence_items(conn, "run_2025", [second])
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM evidence_items WHERE ticker = ? AND as_of_date = ? "
            "AND run_id = ?",
            (TICKER, "2025-06-30", "run_2025"),
        ).fetchone()["n"]
    assert count == 1


def test_recollecting_evidence_keeps_the_earlier_runs_count(monkeypatch, tmp_path):
    """Run 2024 collected the press release, then run 2025 collected it again.

    The single evidence_items row now names run 2025, which is right for run 2025,
    but run 2024 still collected that item: its gating count must stay 1, not
    drop to 0 (a re-collection must not lose prior run links).
    """
    from app.ops.gating import evidence_count_for_run

    cfg = _init_temp_db(monkeypatch, tmp_path)
    adapter = _StubAdapter(cfg)

    def _item_for(as_of: str, run_id: str):
        ctx = AdapterContext(
            ticker=TICKER, as_of_date=as_of, company_name="Example", packet={}, run_id=run_id
        )
        return _item(adapter, ctx, "https://example.com/pr/1", "2024-05-01T00:00:00+00:00")

    with get_db() as conn:
        engine._persist_evidence_items(conn, "run_2024", [_item_for("2024-06-30", "run_2024")])
        engine._persist_evidence_items(conn, "run_2025", [_item_for("2025-06-30", "run_2025")])
        assert evidence_count_for_run(conn, TICKER, "2024-06-30", "run_2024") == 1
        assert evidence_count_for_run(conn, TICKER, "2025-06-30", "run_2025") == 1
        assert evidence_count_for_run(conn, TICKER, "2025-06-30", "run_2024") == 0
