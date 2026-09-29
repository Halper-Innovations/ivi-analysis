from __future__ import annotations

import threading
import time
from datetime import date
from pathlib import Path

from app.db import init_db
from app.dossier.collector import DossierStage1Filing
from app.dossier.runner import run_dossier_for_peer_set
from app.ingest.sec_client import FilingStub
from app.util.db_lock import DB_WRITE_LOCK


def _init_temp_db(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.refresh_ticker_cik_cache",
        lambda http=None: {},
    )
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_dossier_workers_2_uses_serial_parse_lock(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    raw = cfg.raw_filings_dir / "1" / "0001" / "doc.html"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(
        "Item 1. Business\nItem 1A. Risk Factors\nItem 7. Management's Discussion and Analysis\n"
        "Item 8. Financial Statements\nNotes to Consolidated Financial Statements\n",
        encoding="utf-8",
    )

    def _stage1_for_ticker(*, ticker: str, as_of_date: str, years_back: int):
        filing = FilingStub(
            cik="1",
            accession=f"0000001-25-{ticker[-2:]}0001",
            accession_nodash=f"000000125{ticker[-2:]}0001",
            form_type="10-K",
            filing_date=date(2025, 12, 31),
            period_end="2025-12-31",
            primary_document="doc.html",
            primary_doc_url="https://www.sec.gov/Archives/edgar/data/1/doc.html",
            filing_index_url="https://www.sec.gov/Archives/edgar/data/1/index.json",
        )
        return [
            DossierStage1Filing(
                ticker=ticker,
                cik="1",
                filing=filing,
                local_path=str(raw),
            )
        ]

    state = {"active": 0, "max_active": 0}
    state_lock = threading.Lock()

    def _mock_parse_filing_by_id(*args, **kwargs):
        assert DB_WRITE_LOCK.locked()
        with state_lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.02)
        with state_lock:
            state["active"] -= 1
        return False

    monkeypatch.setattr("app.dossier.runner._collect_stage1_for_ticker", _stage1_for_ticker)
    monkeypatch.setattr("app.dossier.collector.parse_filing_by_id", _mock_parse_filing_by_id)
    monkeypatch.setattr(
        "app.dossier.runner.run_synthesis_for_ticker",
        lambda *_args, **_kwargs: None,
    )

    summary = run_dossier_for_peer_set(
        tickers=["AAA", "BBB"],
        as_of_date="2026-02-13",
        years_back=2,
        run_id="dossier_locking_test",
        workers=2,
    )
    assert summary["status"] == "DONE"
    assert state["max_active"] == 1
    assert set(summary["tickers_built"]) == {"AAA", "BBB"}
    assert Path(summary["summary_path"]).exists()
