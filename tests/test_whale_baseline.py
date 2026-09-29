from __future__ import annotations

import json
from pathlib import Path

from app.db import init_db
from app.dossier.baseline import run_whale_baseline
from app.dossier.time_series import build_time_series


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


def _dossier_payload(*, ticker: str, run_id: str, missing: bool = False) -> dict:
    items: list[dict] = []
    for i, year in enumerate(range(2016, 2026), start=1):
        revenue = float(100 + 10 * i)
        cfo = revenue * 0.18
        capex = revenue * 0.05
        net_debt: float | str = float(220 - 3 * i)
        if missing:
            capex = "UNKNOWN" if i % 2 == 0 else capex
            net_debt = "UNKNOWN"
        for metric, value in [
            ("revenue", revenue),
            ("gross_profit", revenue * 0.55),
            ("operating_income", revenue * 0.20),
            ("net_income", revenue * 0.14),
            ("cfo", cfo),
            ("capex", capex),
            ("fcf", (float(cfo) - float(capex)) if isinstance(capex, (int, float)) else "UNKNOWN"),
            ("shares_outstanding", float(1000 + i * 3)),
            ("net_debt", net_debt),
        ]:
            items.append(
                {
                    "year": year,
                    "metric": metric,
                    "value": value,
                    "citations": [{"source_url": "https://www.sec.gov/x", "snippet": "x"}],
                    "derived_from": [f"{ticker}.{year}.{metric}"],
                }
            )
    return {
        "ticker": ticker,
        "run_id": run_id,
        "as_of_date": "2026-02-13",
        "items": items,
        "time_series": build_time_series(items),
    }


def _seed_dossiers(cfg, *, run_id: str, tickers: list[str], missing_controls: bool = False) -> None:
    for ticker in tickers:
        ticker_dir = cfg.dossiers_dir / run_id / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        payload = _dossier_payload(
            ticker=ticker,
            run_id=run_id,
            missing=missing_controls and ticker.startswith("C"),
        )
        (ticker_dir / "dossier.json").write_text(json.dumps(payload), encoding="utf-8")


def test_whale_baseline_early_window_truncation(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "baseline_window_test"

    def _fake_dossier_run(**kwargs):
        _seed_dossiers(cfg, run_id=run_id, tickers=kwargs["tickers"])
        return {"status": "DONE"}

    monkeypatch.setattr("app.dossier.baseline.run_dossier_for_peer_set", _fake_dossier_run)
    payload = run_whale_baseline(
        tickers=["AAPL"],
        controls=["MSFT"],
        as_of_date="2026-02-13",
        years_back=10,
        early_window_years=5,
        run_id=run_id,
    )
    rows_payload = json.loads(Path(payload["baseline_rows_path"]).read_text(encoding="utf-8"))
    aapl_row = next(row for row in rows_payload["rows"] if row["ticker"] == "AAPL")
    assert aapl_row["years_used_count"] == 5
    assert aapl_row["years_used"] == [2016, 2017, 2018, 2019, 2020]


def test_whale_baseline_controls_generation_is_deterministic(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)

    def _fake_dossier_run(**kwargs):
        _seed_dossiers(cfg, run_id=kwargs["run_id"], tickers=kwargs["tickers"])
        return {"status": "DONE"}

    monkeypatch.setattr("app.dossier.baseline.run_dossier_for_peer_set", _fake_dossier_run)
    monkeypatch.setattr(
        "app.dossier.baseline.select_sector_peers",
        lambda **kwargs: {
            "selected_tickers": ["AAPL", "CRM", "NOW", "ADBE"],
            "peer_selection_summary": {"mode": kwargs.get("mode", "hybrid")},
        },
    )
    first = run_whale_baseline(
        tickers=["AAPL"],
        controls=None,
        as_of_date="2026-02-13",
        years_back=10,
        early_window_years=5,
        peer_mode="hybrid",
        sector="Software",
        run_id="baseline_controls_test_1",
    )
    second = run_whale_baseline(
        tickers=["AAPL"],
        controls=None,
        as_of_date="2026-02-13",
        years_back=10,
        early_window_years=5,
        peer_mode="hybrid",
        sector="Software",
        run_id="baseline_controls_test_2",
    )
    assert first["controls"] == ["CRM", "NOW", "ADBE"]
    assert second["controls"] == ["CRM", "NOW", "ADBE"]


def test_whale_baseline_controls_count_excludes_winners(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)

    def _fake_dossier_run(**kwargs):
        _seed_dossiers(cfg, run_id=kwargs["run_id"], tickers=kwargs["tickers"])
        return {"status": "DONE"}

    monkeypatch.setattr("app.dossier.baseline.run_dossier_for_peer_set", _fake_dossier_run)
    monkeypatch.setattr(
        "app.dossier.baseline.select_sector_peers",
        lambda **kwargs: {
            "selected_tickers": ["AAPL", "MSFT", "CRM", "NOW", "ADBE", "ORCL"],
            "all_rows": [],
            "counts": {"selected": 6},
            "peer_selection_summary": {"mode": kwargs.get("mode", "hybrid"), "final_selected_count": 6},
        },
    )
    payload = run_whale_baseline(
        tickers=["AAPL", "MSFT"],
        controls=None,
        controls_count=3,
        as_of_date="2026-02-13",
        years_back=10,
        early_window_years=5,
        peer_mode="hybrid",
        sector="Software",
        run_id="baseline_controls_count_test",
    )
    assert payload["controls"] == ["CRM", "NOW", "ADBE"]
    assert "AAPL" not in payload["controls"]
    assert "MSFT" not in payload["controls"]
    summary = json.loads(Path(payload["baseline_summary_path"]).read_text(encoding="utf-8"))
    assert summary["controls_count_requested"] == 3
    assert summary["controls_generation"]["requested_count"] == 3


def test_whale_baseline_report_includes_deltas_and_unknown_rates(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "baseline_report_test"

    def _fake_dossier_run(**kwargs):
        _seed_dossiers(cfg, run_id=run_id, tickers=kwargs["tickers"], missing_controls=True)
        return {"status": "DONE"}

    monkeypatch.setattr("app.dossier.baseline.run_dossier_for_peer_set", _fake_dossier_run)
    payload = run_whale_baseline(
        tickers=["AAPL"],
        controls=["CRM"],
        as_of_date="2026-02-13",
        years_back=10,
        early_window_years=5,
        run_id=run_id,
    )
    report_text = Path(payload["baseline_report_path"]).read_text(encoding="utf-8")
    assert "Signal hit rates per cohort (winners vs controls)" in report_text
    assert "Most discriminative signals" in report_text
    assert "Unknown rates and UNKNOWN causes" in report_text
    summary = json.loads(Path(payload["baseline_summary_path"]).read_text(encoding="utf-8"))
    assert summary["signal_hit_rates"]
    assert "delta" in summary["signal_hit_rates"][0]
    assert "unknown_rate" in summary["signal_hit_rates"][0]


def test_whale_baseline_controls_empty_section(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "baseline_controls_empty_test"

    def _fake_dossier_run(**kwargs):
        _seed_dossiers(cfg, run_id=run_id, tickers=kwargs["tickers"])
        return {"status": "DONE"}

    monkeypatch.setattr("app.dossier.baseline.run_dossier_for_peer_set", _fake_dossier_run)
    monkeypatch.setattr(
        "app.dossier.baseline.select_sector_peers",
        lambda **kwargs: {
            "selected_tickers": [],
            "all_rows": [],
            "counts": {"selected": 0},
            "peer_selection_summary": {"mode": kwargs.get("mode", "hybrid"), "final_selected_count": 0},
        },
    )
    payload = run_whale_baseline(
        tickers=["AAPL"],
        controls=None,
        controls_count=5,
        as_of_date="2026-02-13",
        years_back=10,
        early_window_years=5,
        peer_mode="hybrid",
        sector="Software",
        run_id=run_id,
    )
    report = Path(payload["baseline_report_path"]).read_text(encoding="utf-8")
    assert "Controls Empty" in report
    assert "Requested controls: `5`" in report


def test_whale_baseline_report_includes_suppressed_tickers_table(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "baseline_suppression_table_test"

    def _fake_dossier_run(**kwargs):
        _seed_dossiers(cfg, run_id=run_id, tickers=kwargs["tickers"])
        return {"status": "DONE"}

    monkeypatch.setattr("app.dossier.baseline.run_dossier_for_peer_set", _fake_dossier_run)
    monkeypatch.setattr(
        "app.dossier.baseline.select_sector_peers",
        lambda **kwargs: {
            "selected_tickers": ["AAPL", "CRM"],
            "all_rows": [
                {"ticker": "BANK", "suppression_reasons": ["SUPPRESSION_FINANCIALS"]},
            ],
            "peer_selection_summary": {"mode": kwargs.get("mode", "hybrid")},
        },
    )
    payload = run_whale_baseline(
        tickers=["AAPL"],
        controls=None,
        as_of_date="2026-02-13",
        years_back=10,
        early_window_years=5,
        peer_mode="hybrid",
        sector="Software",
        run_id=run_id,
    )
    report_text = Path(payload["baseline_report_path"]).read_text(encoding="utf-8")
    assert "Suppressed tickers during control selection" in report_text
    assert "SUPPRESSION_FINANCIALS" in report_text
    summary = json.loads(Path(payload["baseline_summary_path"]).read_text(encoding="utf-8"))
    assert summary["controls_generation"]["suppressed_tickers"][0]["ticker"] == "BANK"
