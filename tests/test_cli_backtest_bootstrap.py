from __future__ import annotations

import sqlite3

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


def _make_outcomes_db(tmp_path, rows):
    db_path = tmp_path / "bt.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE ticker_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            run_id TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            outcome_status TEXT NOT NULL,
            realized_return_pct REAL,
            excess_return_pct REAL,
            grade TEXT,
            status TEXT,
            benchmark_symbol TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO ticker_outcomes(ticker, as_of_date, run_id, horizon_days, outcome_status, "
        "realized_return_pct, excess_return_pct, grade, status, benchmark_symbol) "
        "VALUES (?, ?, 'test_run', 365, 'CLOSED', ?, ?, ?, 'CHEAP_VS_EXPECTATIONS', 'SPY')",
        rows,
    )
    conn.commit()
    conn.close()
    return db_path


_ROWS = [
    # (ticker, as_of_date, realized, excess, grade) — same hand fixture as
    # tests/test_backtest_inference.py: mean contrast 5.0
    ("AAA", "2024-01-02", 11.0, 10.0, "DEPLOY_READY"),
    ("BBB", "2024-01-02", 7.0, 6.0, "DEPLOY_READY"),
    ("CCC", "2024-07-01", 9.0, 8.0, "DEPLOY_READY"),
    ("DDD", "2025-01-02", 3.0, 2.0, "DEPLOY_READY"),
    ("EEE", "2025-01-02", 5.0, 4.0, "DEPLOY_READY"),
    ("FFF", "2024-01-02", 3.0, 2.0, "NOT_CHEAP"),
    ("GGG", "2024-07-01", -1.0, -2.0, "NOT_CHEAP"),
    ("HHH", "2024-07-01", 1.0, 0.0, "NOT_CHEAP"),
    ("III", "2025-01-02", 5.0, 4.0, "NOT_CHEAP"),
]


def test_bootstrap_cli_reports_contrast(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    db_path = _make_outcomes_db(tmp_path, _ROWS)
    result = runner.invoke(
        app,
        [
            "backtest", "bootstrap",
            "--run-id", "test_run",
            "--n-boot", "200",
            "--seed", "7",
            "--db-path", str(db_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "# Cluster Bootstrap Contrast (test_run)" in result.output
    assert "observed mean contrast: 5.0" in result.output
    assert "n_treat=5 n_control=4 n_clusters=3" in result.output
    assert "unreliable with so few clusters" in result.output
    assert "Leave-one-out" in result.output


def test_bootstrap_cli_no_rows_exits_nonzero(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    db_path = _make_outcomes_db(tmp_path, [])
    result = runner.invoke(
        app,
        ["backtest", "bootstrap", "--run-id", "missing_run", "--db-path", str(db_path)],
    )
    assert result.exit_code != 0
    assert "No CLOSED rows" in result.output


def test_backtest_help_lists_bootstrap(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    result = runner.invoke(app, ["backtest", "--help"])
    assert result.exit_code == 0, result.output
    assert "bootstrap" in result.output
