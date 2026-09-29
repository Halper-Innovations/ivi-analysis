from __future__ import annotations

import json

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db


runner = CliRunner()


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


def test_dossier_cli_commands(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)

    monkeypatch.setattr(
        "app.dossier.runner.run_dossier_for_peer_set",
        lambda **kwargs: {
            "run_id": "dossier_test_cli",
            "as_of_date": kwargs["as_of_date"],
            "tickers_requested": kwargs["tickers"],
            "tickers_built": kwargs["tickers"],
            "peer_report_path": str(cfg.dossiers_dir / "dossier_test_cli" / "peer_report.md"),
            "peer_rankings_path": str(cfg.dossiers_dir / "dossier_test_cli" / "peer_rankings.json"),
        },
    )

    run = runner.invoke(
        app,
        [
            "dossier-run",
            "--tickers",
            "AAPL,MSFT",
            "--as-of",
            "2026-02-13",
            "--years-back",
            "5",
            "--run-id",
            "dossier_test_cli",
            "--workers",
            "1",
        ],
    )
    assert run.exit_code == 0, run.output
    payload = json.loads(run.output)
    assert payload["run_id"] == "dossier_test_cli"
    assert payload["tickers_built"] == ["AAPL", "MSFT"]

    run_dir = cfg.dossiers_dir / "dossier_test_cli"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dossier_summary.json").write_text(json.dumps({"run_id": "dossier_test_cli"}), encoding="utf-8")
    (run_dir / "peer_report.md").write_text("# peer report", encoding="utf-8")
    (run_dir / "peer_rankings.json").write_text(
        json.dumps(
            {
                "rankings": [
                    {
                        "ticker": "AAPL",
                        "metric_values": {"revenue_cagr_proxy": 0.1},
                        "metric_ranks": {"revenue_cagr_proxy": 1},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "AAPL").mkdir(parents=True, exist_ok=True)

    opened = runner.invoke(app, ["dossier-open", "--run-id", "dossier_test_cli"])
    assert opened.exit_code == 0, opened.output
    opened_payload = json.loads(opened.output)
    assert opened_payload["run_id"] == "dossier_test_cli"
    assert "AAPL" in opened_payload["tickers"]

    compared = runner.invoke(
        app,
        ["dossier-compare", "--run-id", "dossier_test_cli", "--metric", "revenue_cagr_proxy"],
    )
    assert compared.exit_code == 0, compared.output
    compared_payload = json.loads(compared.output)
    assert compared_payload["rows"][0]["ticker"] == "AAPL"

    monkeypatch.setattr(
        "app.dossier.runner.resume_dossier_run",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "status": "DONE",
            "tickers_built": ["AAPL"],
        },
    )
    resumed = runner.invoke(
        app,
        ["dossier-resume", "--run-id", "dossier_test_cli", "--workers", "1"],
    )
    assert resumed.exit_code == 0, resumed.output
    resumed_payload = json.loads(resumed.output)
    assert resumed_payload["run_id"] == "dossier_test_cli"
    assert resumed_payload["status"] == "DONE"

    monkeypatch.setattr(
        "app.dossier.whale_signals.run_whale_signals_for_run",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "whale_signature_rank": ["AAPL"],
            "rows": [{"ticker": "AAPL", "whale_signature_score": 72.0}],
            "summary_path": str(run_dir / "whale_signals_summary.json"),
        },
    )
    whale = runner.invoke(app, ["dossier-whale-signals", "--run-id", "dossier_test_cli"])
    assert whale.exit_code == 0, whale.output
    whale_payload = json.loads(whale.output)
    assert whale_payload["run_id"] == "dossier_test_cli"
    assert whale_payload["whale_signature_rank"] == ["AAPL"]

    baseline_kwargs: dict = {}

    def _fake_baseline(**kwargs):
        baseline_kwargs.update(kwargs)
        return {
            "run_id": kwargs["run_id"],
            "baseline_report_path": "data/outputs/whales/whale_baseline_2015/baseline_report.md",
        }

    monkeypatch.setattr("app.dossier.baseline.run_whale_baseline", _fake_baseline)
    baseline = runner.invoke(
        app,
        [
            "whale-baseline",
            "--tickers",
            "AAPL,MSFT",
            "--as-of",
            "2015-02-13",
            "--years-back",
            "10",
            "--sec-budget",
            "1500",
            "--run-id",
            "whale_baseline_2015",
        ],
    )
    assert baseline.exit_code == 0, baseline.output
    baseline_payload = json.loads(baseline.output)
    assert baseline_payload["run_id"] == "whale_baseline_2015"
    assert baseline_kwargs["sec_budget"] == 1500
