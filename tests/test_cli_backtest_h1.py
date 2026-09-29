from __future__ import annotations
from typer.testing import CliRunner
from app.config import get_config
from app.db import init_db


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def test_run_h1_cli_invokes_orchestrator(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    import app.cli_backtest as cli
    from app.backtest.runner import BacktestRunSummary

    captured = {}

    def _fake_run_h1(**kwargs):
        captured.update(kwargs)
        return BacktestRunSummary(universe_size=4, reconstructed=4, skipped=0, rows_written=8)

    monkeypatch.setattr(cli, "run_h1", _fake_run_h1, raising=False)

    result = CliRunner().invoke(
        cli.backtest_app,
        ["run-h1", "--per-band", "5", "--seed", "7", "--dates", "2022-01-03", "--horizons", "365"],
    )
    assert result.exit_code == 0, result.output
    assert captured["per_band"] == 5
    assert captured["seed"] == 7
    assert captured["dates"] == ["2022-01-03"]
    assert captured["horizons"] == [365]
    assert '"rows_written": 8' in result.output
