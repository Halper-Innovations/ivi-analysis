from __future__ import annotations

import json

from app.db import init_db
from app.valuation.fcf import resolve_fcf_asof, write_fcf_coverage_for_run


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _write_dossier(path, *, ticker: str, as_of_date: str, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": ticker,
        "run_id": path.parent.parent.name,
        "as_of_date": as_of_date,
        "time_series": {
            "standardized_rows": [row],
            "standardized_row_traces": {
                str(row["year"]): {
                    "fcf": {"derived_from": [f"dossiers.{path.parent.parent.name}.{ticker}.fcf"], "citations": []},
                    "cfo": {"derived_from": [f"dossiers.{path.parent.parent.name}.{ticker}.cfo"], "citations": []},
                    "capex": {"derived_from": [f"dossiers.{path.parent.parent.name}.{ticker}.capex"], "citations": []},
                }
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_fcf_resolver_derives_from_cfo_minus_capex(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "fcf_derive_current"
    _write_dossier(
        cfg.dossiers_dir / run_id / "AAA" / "dossier.json",
        ticker="AAA",
        as_of_date="2026-02-14",
        row={"year": 2025, "fcf": "UNKNOWN", "cfo": 20.0, "capex": 7.0},
    )

    value, coverage = resolve_fcf_asof(ticker="AAA", as_of_date="2026-02-14", run_id=run_id)
    assert value == 13.0
    assert coverage["fcf_status"] == "OK"
    assert coverage["fcf_reason_code"] == "OK"
    assert any("derived:fcf=cfo-capex" in str(ref) for ref in coverage["derived_from"])


def test_fcf_resolver_historical_fallback(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "fcf_historical_fallback"
    (cfg.sectors_dir / run_id).mkdir(parents=True, exist_ok=True)
    _write_dossier(
        cfg.dossiers_dir / "hist_fcf" / "AAA" / "dossier.json",
        ticker="AAA",
        as_of_date="2026-02-13",
        row={"year": 2025, "fcf": 9.5, "cfo": 15.0, "capex": 5.5},
    )

    value, coverage = resolve_fcf_asof(ticker="AAA", as_of_date="2026-02-14", run_id=run_id)
    assert value == 9.5
    assert coverage["fcf_status"] == "OK"
    assert coverage["fcf_reason_code"] == "HISTORICAL_DOSSIER_HIT"
    assert coverage["fcf_source"] == "historical_dossier:hist_fcf"

    summary = write_fcf_coverage_for_run(
        run_id=run_id,
        as_of_date="2026-02-14",
        tickers=["AAA", "BBB"],
        output_dir=cfg.sectors_dir / run_id,
        cfg=cfg,
    )
    assert summary["ticker_count"] == 2
    assert summary["reason_counts"]["HISTORICAL_DOSSIER_HIT"] == 1
    assert summary["reason_counts"]["NO_HISTORICAL_DOSSIER"] == 1
