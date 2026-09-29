from __future__ import annotations

from app.db import get_db, init_db, utc_now_iso
from app.research.schemas import CitationRef, EvidenceItem
from app.research.signals import compute_research_signals, persist_research_signals, row_to_signals_dict
from app.score.rubric import score_packet


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _sample_packet():
    return {
        "fundamentals": {
            "revenue": 1000,
            "operating_margin": 0.12,
            "fcf": 100,
            "fcf_margin": 0.10,
            "net_debt": 200,
            "liquidity_stress_score": 3,
        },
        "valuations": {
            "dcf_lite": {"outputs": {"confidence": "MEDIUM"}},
            "reverse_dcf": {"inputs": {"market_price": "UNKNOWN"}},
        },
        "deltas_vs_prior_period": {"revenue": 50, "fcf": 10},
        "extracted_facts": [],
    }


def _item(source_type: str, title: str, pub: str, excerpt: str, url: str) -> EvidenceItem:
    return EvidenceItem(
        id=f"ev_{source_type}_{title}",
        ticker="AAPL",
        as_of_date="2026-02-13",
        source_type=source_type,  # type: ignore[arg-type]
        source_url=url,
        source_title=title,
        source_published_at=pub,
        retrieved_at=utc_now_iso(),
        excerpt_text=excerpt,
        citations=[CitationRef(source_url=url, snippet=excerpt[:80], section_label="test")],
        hash="h",
        content_hash="c",
        dedupe_key=f"d_{title}",
        adapter_run_id="run_test",
    )


def test_research_signals_persist(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    items = [
        _item(
            "company_news",
            "Guidance Raised",
            "2026-02-10T00:00:00+00:00",
            "Company raised guidance for fiscal year.",
            "https://example.com/news/1",
        ),
        _item(
            "sec_exhibit",
            "Earnings Release [earnings_release]",
            "2026-02-11T00:00:00+00:00",
            "Press release and quarterly results of operations.",
            "https://www.sec.gov/Archives/edgar/data/1/8k.htm",
        ),
    ]
    signals = compute_research_signals(ticker="AAPL", as_of_date="2026-02-13", run_id="run_test", evidence_items=items)
    with get_db() as conn:
        persist_research_signals(conn, signals)
        row = conn.execute(
            "SELECT * FROM research_signals WHERE ticker='AAPL' AND as_of_date='2026-02-13' AND run_id='run_test'"
        ).fetchone()
    payload = row_to_signals_dict(row)
    assert payload is not None
    assert payload["item_count_30d"] >= 2
    assert payload["has_earnings_release"] is True


def test_score_changes_when_research_signals_change():
    packet = _sample_packet()
    stale_signals = {
        "recency_days_min": 365,
        "item_count_30d": 0,
        "sentiment_flags": ["material_weakness"],
        "has_earnings_release": False,
        "has_investor_presentation": False,
    }
    fresh_signals = {
        "recency_days_min": 5,
        "item_count_30d": 5,
        "sentiment_flags": [],
        "has_earnings_release": True,
        "has_investor_presentation": True,
    }

    stale = score_packet(packet, {"classification": "WATCHLIST"}, research_signals=stale_signals)
    fresh = score_packet(packet, {"classification": "WATCHLIST"}, research_signals=fresh_signals)

    stale_subscores, stale_total, _, _ = stale
    fresh_subscores, fresh_total, _, _ = fresh
    assert "research_signal_net" in stale_subscores
    assert "research_signal_net" in fresh_subscores
    assert fresh_total > stale_total
