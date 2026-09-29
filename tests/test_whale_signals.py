from __future__ import annotations

import json
from pathlib import Path

from app.db import init_db
from app.dossier.peer_report import build_peer_report
from app.dossier.time_series import build_time_series
from app.dossier.whale_signals import build_whale_signals


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


def _dossier_fixture(ticker: str, *, missing_metrics: bool = False) -> dict:
    items: list[dict] = []
    for i, year in enumerate(range(2016, 2026), start=1):
        revenue = float(100 + (12 * i))
        gross_profit = revenue * 0.54
        operating_income = revenue * 0.20
        net_income = revenue * 0.15
        cfo = revenue * 0.18
        capex = revenue * 0.05
        fcf = cfo - capex
        shares = float(1000 + i * 3)
        net_debt = float(220 - i * 4)
        if missing_metrics:
            net_debt = "UNKNOWN"
            capex = "UNKNOWN" if i % 2 == 0 else capex
        for metric, value in [
            ("revenue", revenue),
            ("gross_profit", gross_profit),
            ("operating_income", operating_income),
            ("net_income", net_income),
            ("cfo", cfo),
            ("capex", capex),
            ("fcf", fcf),
            ("shares_outstanding", shares),
            ("net_debt", net_debt),
        ]:
            items.append(
                {
                    "year": year,
                    "metric": metric,
                    "value": value,
                    "citations": [{"source_url": "https://www.sec.gov/x", "snippet": "x"}],
                    "derived_from": [f"filing.{year}.{metric}"],
                }
            )
    return {
        "ticker": ticker,
        "run_id": "whale_test",
        "as_of_date": "2026-02-13",
        "time_series": build_time_series(items),
        "claims": [{"citations": [{"source_url": "https://www.sec.gov/x", "snippet": "x"}]}],
        "items": [],
    }


def test_whale_signals_are_deterministic():
    dossier = _dossier_fixture("AAA")
    first = build_whale_signals(dossier)
    second = build_whale_signals(dossier)
    assert first == second
    assert first["whale_signals_version"] == "v1.1"
    assert isinstance(first["whale_signature_score"], float)
    assert len(first["signals"]) == 6
    assert all("signal_confidence" in row for row in first["signals"])
    assert all("why_it_matters" in row for row in first["signals"])
    assert all("thresholds" in row for row in first["signals"])


def test_whale_signals_missing_metrics_add_gaps_and_penalty():
    dossier = _dossier_fixture("AAA", missing_metrics=True)
    payload = build_whale_signals(dossier)
    assert payload["total_penalty"] > 0
    assert payload["gaps"]
    assert any(gap["signal"] == "balance_sheet_resilience" for gap in payload["gaps"])
    assert payload["critical_unknowns"]
    assert "penalty_rule" in payload


def test_peer_report_includes_whale_signature_rank(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    dossiers = [_dossier_fixture("AAA"), _dossier_fixture("BBB", missing_metrics=True)]
    summary = build_peer_report(run_id="whale_peer_report_test", as_of_date="2026-02-13", dossiers=dossiers)
    rankings_path = Path(summary["peer_rankings_path"])
    payload = json.loads(rankings_path.read_text(encoding="utf-8"))
    assert "whale_signature_rank" in payload
    assert payload["rankings"][0]["metric_ranks"]["whale_signature_rank"] >= 1
    assert (cfg.dossiers_dir / "whale_peer_report_test" / "peer_report.md").exists()
