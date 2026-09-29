from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from app.db import get_db, init_db
from app.discovery.metrics import DiscoveryMetricsResult
from app.discovery.policy import DiscoveryFilingSelection
from app.discovery.runner import run_discovery
from app.ingest.sec_client import FilingStub


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    monkeypatch.setenv("VOE_PRICE_PROVIDER", "disabled")
    monkeypatch.setenv("VOE_DISCOVERY_WORKERS", "4")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _selection_for(cik: str, filing_date: str = "2026-01-30") -> DiscoveryFilingSelection:
    accession = f"000{int(cik):07d}-26-000001"
    accession_nodash = accession.replace("-", "")
    filing = FilingStub(
        cik=cik,
        accession=accession,
        accession_nodash=accession_nodash,
        form_type="10-Q",
        filing_date=date.fromisoformat(filing_date),
        period_end="2025-12-31",
        primary_document="doc.htm",
        primary_doc_url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/doc.htm",
        filing_index_url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/index.json",
    )
    return DiscoveryFilingSelection(annual=None, quarters=[filing], event=None)


def _patch_perf_discovery(monkeypatch, tickers: list[str], *, eligible_count: int | None = None) -> None:
    ticker_to_cik = {ticker: str(1000 + i) for i, ticker in enumerate(tickers)}
    cik_to_ticker = {cik: ticker for ticker, cik in ticker_to_cik.items()}
    if eligible_count is None:
        eligible_count = len(tickers)
    eligible_tickers = set(tickers[: max(0, eligible_count)])

    monkeypatch.setattr(
        "app.universe.ticker_cik_map.load_ticker_cik_map",
        lambda refresh_if_missing=True: ticker_to_cik,
    )

    def _select(*args, **kwargs):
        cik = str(kwargs.get("cik") or "")
        ticker = cik_to_ticker.get(cik, "")
        if ticker and ticker in eligible_tickers:
            return _selection_for(cik)
        return DiscoveryFilingSelection(annual=None, quarters=[], event=None)

    monkeypatch.setattr("app.discovery.runner.select_discovery_filings_from_submissions", _select)

    def _fake_fetch_parse(conn, **kwargs):
        _ = conn
        selection = kwargs["selection"]
        accessions = [f.accession for f in selection.all_filings]
        filing_ids = list(range(1, len(accessions) + 1))
        return accessions, filing_ids

    monkeypatch.setattr("app.discovery.runner._fetch_and_parse_minimal_filings", _fake_fetch_parse)
    monkeypatch.setattr("app.discovery.runner.parse_filing_by_id", lambda filing_id: bool(filing_id))
    monkeypatch.setattr("app.discovery.runner._load_filing_text", lambda conn, ticker, accessions: "software platform")

    def _fake_metrics(conn, *, ticker: str, selected_accessions: list[str]):
        idx = tickers.index(ticker) + 1
        revenue = float(1_000_000_000 + idx * 10_000_000)
        return DiscoveryMetricsResult(
            effective_as_of_date="2026-01-30",
            metrics={
                "ttm_revenue": revenue,
                "gross_margin": 0.40,
                "operating_margin": 0.12,
                "cfo": 100_000_000.0,
                "capex": 20_000_000.0,
                "fcf": 80_000_000.0,
                "shares_outstanding": 200_000_000.0,
                "revenue_growth_recent": 0.10,
                "revenue_acceleration": 0.02,
                "gross_margin_change_qoq": 0.01,
                "operating_margin_change_qoq": 0.01,
                "fcf_change_qoq": 5_000_000.0,
                "liquidity_stress_score": 2,
                "net_debt": 100_000_000.0,
            },
            claims=[
                {
                    "claim_id": f"rev_{ticker}",
                    "label": "ttm_revenue",
                    "value": revenue,
                    "citations": [{"source_url": "https://www.sec.gov/doc", "snippet": "revenue", "section_label": None}],
                    "derived_from": ["discovery.metrics.ttm_revenue"],
                }
            ],
            flags=[],
            filing_accessions_used=selected_accessions,
        )

    monkeypatch.setattr("app.discovery.runner.compute_discovery_metrics", _fake_metrics)

    class _FakeSecClient:
        def __init__(self):
            self.http = type("Http", (), {"metrics": staticmethod(lambda: {"throttled_count": 0})})()

        def submissions(self, cik: str):
            ticker = cik_to_ticker.get(cik, "UNKNOWN")
            return {"name": f"{ticker} Corp"}

    monkeypatch.setattr("app.discovery.runner.SecClient", _FakeSecClient)


def _seed_file(tmp_path: Path, tickers: list[str]) -> Path:
    seed = tmp_path / "seed.csv"
    lines = ["ticker", *tickers]
    seed.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return seed


def test_discovery_interrupt_partial_finalization(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    tickers = [f"T{i:02d}" for i in range(20)]
    _patch_perf_discovery(monkeypatch, tickers)
    seed = _seed_file(tmp_path, tickers)

    summary = run_discovery(
        as_of_date="2026-02-13",
        tickers=tickers,
        seed_path=seed,
        limit=20,
        top_k=5,
        workers=4,
        _cancel_after=6,
    )

    assert summary["status"] == "PARTIAL"
    assert Path(summary["discovery_candidates_path"]).exists()
    assert Path(summary["discovery_shortlist_path"]).exists()
    assert Path(summary["discovery_report_path"]).exists()
    assert Path(summary["discovery_stats_path"]).exists()

    stats = json.loads(Path(summary["discovery_stats_path"]).read_text(encoding="utf-8"))
    assert stats["status"] == "PARTIAL"
    with get_db() as conn:
        row = conn.execute("SELECT status FROM discovery_runs WHERE run_id = ?", (summary["run_id"],)).fetchone()
    assert row and row["status"] == "PARTIAL"


def test_discovery_resume_completes(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    tickers = [f"T{i:02d}" for i in range(20)]
    _patch_perf_discovery(monkeypatch, tickers)
    seed = _seed_file(tmp_path, tickers)

    partial = run_discovery(
        as_of_date="2026-02-13",
        tickers=tickers,
        seed_path=seed,
        limit=20,
        top_k=5,
        workers=4,
        _cancel_after=5,
    )
    resumed = run_discovery(
        as_of_date="2026-02-13",
        run_id=partial["run_id"],
        resume=True,
        workers=4,
    )

    assert resumed["status"] == "COMPLETED"
    assert resumed["processed_count"] == resumed["target_count"] == 20
    with get_db() as conn:
        row = conn.execute(
            "SELECT status, processed_count FROM discovery_runs WHERE run_id = ?",
            (partial["run_id"],),
        ).fetchone()
    assert row and row["status"] == "COMPLETED"
    assert int(row["processed_count"]) == 20


def test_discovery_two_phase_prefilter_reduces_phase2_work(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    tickers = [f"T{i:02d}" for i in range(20)]
    _patch_perf_discovery(monkeypatch, tickers)
    seed = _seed_file(tmp_path, tickers)

    summary = run_discovery(
        as_of_date="2026-02-13",
        tickers=tickers,
        seed_path=seed,
        limit=20,
        top_k=5,
        prefilter_cap=6,
        prefilter_keep_ratio=0.3,
        workers=4,
    )

    assert summary["status"] == "COMPLETED"
    assert summary["phase1_count"] == 20
    assert summary["phase2_count"] <= 6
    assert summary["phase2_count"] < 20


def test_discovery_cache_skip(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    tickers = [f"T{i:02d}" for i in range(10)]
    _patch_perf_discovery(monkeypatch, tickers)
    seed = _seed_file(tmp_path, tickers)

    first = run_discovery(
        as_of_date="2026-02-13",
        tickers=tickers,
        seed_path=seed,
        limit=10,
        top_k=5,
        workers=4,
    )
    second = run_discovery(
        as_of_date="2026-02-13",
        tickers=tickers,
        seed_path=seed,
        limit=10,
        top_k=5,
        workers=4,
    )

    assert first["status"] == "COMPLETED"
    assert second["status"] == "COMPLETED"
    assert int(second["tickers_skipped_cached"]) > 0
    assert int(second["tickers_downloaded"]) == 0
