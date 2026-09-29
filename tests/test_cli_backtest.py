from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

from typer.testing import CliRunner

from app.cli import app
from app.config import get_config
from app.db import init_db

runner = CliRunner()


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def test_backtest_help_lists_run_and_report(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    result = runner.invoke(app, ["backtest", "--help"])
    assert result.exit_code == 0, result.output
    assert "run" in result.output
    assert "report" in result.output


def test_backtest_report_empty_is_clean(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    result = runner.invoke(app, ["backtest", "report", "--run-id-prefix", "bt", "--horizon", "365"])
    assert result.exit_code == 0, result.output
    assert "Edge Backtest Report" in result.output


def test_backtest_run_with_explicit_tickers_no_network(monkeypatch, tmp_path):
    """Run with an explicit ticker; stub run_backtest to avoid any network/DB work."""
    _init(monkeypatch, tmp_path)

    @dataclass
    class _Summary:
        universe_size: int = 1
        reconstructed: int = 0
        skipped: int = 1
        rows_written: int = 0
        skipped_reasons: dict = field(default_factory=lambda: {"ZZZ": "NO_FACTS"})

    stub = MagicMock(return_value=_Summary())

    monkeypatch.setattr("app.backtest.runner.run_backtest", stub)

    result = runner.invoke(
        app,
        ["backtest", "run", "--universe", "ZZZ", "--dates", "2024-01-02", "--horizons", "90"],
    )
    assert result.exit_code == 0, result.output
    assert '"universe_size"' in result.output
    assert '"skipped"' in result.output


def test_backtest_run_rejects_bad_horizon(monkeypatch, tmp_path):
    """Non-integer horizon token should raise BadParameter (non-zero exit)."""
    _init(monkeypatch, tmp_path)
    result = runner.invoke(
        app,
        ["backtest", "run", "--universe", "ZZZ", "--dates", "2024-01-02", "--horizons", "90,abc"],
    )
    assert result.exit_code != 0
    assert "abc" in result.output or "Invalid horizon" in result.output


def test_backtest_run_rejects_bad_date(monkeypatch, tmp_path):
    """Malformed date token should raise BadParameter (non-zero exit)."""
    _init(monkeypatch, tmp_path)
    result = runner.invoke(
        app,
        ["backtest", "run", "--universe", "ZZZ", "--dates", "not-a-date", "--horizons", "90"],
    )
    assert result.exit_code != 0
    assert "not-a-date" in result.output or "Invalid --dates" in result.output
