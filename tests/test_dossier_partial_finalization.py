from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.db import init_db
from app.dossier.runner import compare_dossier_metric, open_dossier_run, run_dossier_for_peer_set


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


def test_dossier_partial_finalization_and_open(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "dossier_partial_test"

    def _fake_collect(*, ticker: str, as_of_date: str, years_back: int):
        return [{"ticker": ticker, "ok": True}]

    def _fake_build(*, ticker: str, as_of_date: str, run_id: str, stage1, years_back: int = 10):
        if ticker == "BBB":
            raise RuntimeError("forced dossier error for test")
        ticker_dir = cfg.dossiers_dir / run_id / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        md_path = ticker_dir / "dossier.md"
        json_path = ticker_dir / "dossier.json"
        md_path.write_text("# dossier", encoding="utf-8")
        json_path.write_text(json.dumps({"ticker": ticker}), encoding="utf-8")
        return {
            "ticker": ticker,
            "time_series": {
                "derived_signals": [
                    {"signal": "revenue_cagr_proxy", "value": 0.10},
                    {"signal": "gross_margin_delta", "value": 0.02},
                    {"signal": "operating_margin_delta", "value": 0.01},
                    {"signal": "fcf_margin_delta", "value": 0.01},
                    {"signal": "dilution_rate_proxy", "value": 0.00},
                ]
            },
            "claims": [
                {
                    "citations": [
                        {
                            "source_url": "https://www.sec.gov/test",
                            "snippet": "snippet",
                            "section_label": "financial_statements",
                        }
                    ]
                }
            ],
            "items": [],
            "artifacts": {
                "dossier_json_path": str(json_path),
                "dossier_md_path": str(md_path),
            },
        }

    monkeypatch.setattr("app.dossier.runner._collect_stage1_for_ticker", _fake_collect)
    monkeypatch.setattr("app.dossier.runner._build_ticker_payload_from_stage1", _fake_build)

    summary = run_dossier_for_peer_set(
        tickers=["AAA", "BBB"],
        as_of_date="2026-02-13",
        years_back=5,
        run_id=run_id,
        workers=2,
    )

    assert summary["status"] == "PARTIAL"
    assert "AAA" in summary["tickers_built"]
    assert "BBB" in summary["tickers_failed"]
    assert Path(summary["summary_path"]).exists()

    opened = open_dossier_run(run_id)
    assert opened["status"] == "PARTIAL"
    assert opened["summary_path"] is not None
    assert "AAA" in opened["ticker_results"]
    assert opened["ticker_results"]["BBB"]["status"] == "FAILED"


def test_compare_metric_missing_ticker_error(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "dossier_metric_missing_test"
    run_dir = cfg.dossiers_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "peer_rankings.json").write_text(
        json.dumps(
            {
                "rankings": [
                    {
                        "ticker": "AAA",
                        "metric_values": {"revenue_cagr_proxy": 0.1},
                        "metric_ranks": {"revenue_cagr_proxy": 1},
                    },
                    {
                        "ticker": "BBB",
                        "metric_values": {"gross_margin_delta": 0.02},
                        "metric_ranks": {"gross_margin_delta": 2},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="metric missing for ticker\\(s\\): BBB"):
        compare_dossier_metric(run_id=run_id, metric="revenue_cagr_proxy")

