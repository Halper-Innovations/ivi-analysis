from __future__ import annotations

from app.config import get_config
from app.db import get_db, init_db


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def _closed(
    conn,
    *,
    ticker,
    grade,
    excess,
    run_id="bt_h365",
    horizon=365,
    status="CHEAP_VS_EXPECTATIONS",
):
    conn.execute(
        "INSERT INTO ticker_outcomes(ticker, as_of_date, run_id, outcome_status, decision, "
        "conviction, horizon_days, entry_price, entry_price_source, entry_date, grade, status, "
        "benchmark_symbol, excess_return_pct, realized_return_pct, created_at, updated_at) "
        "VALUES(?, '2024-01-02', ?, 'CLOSED', 'BUY', 2, ?, 100.0, 'historical_backtest', "
        "'2024-01-02', ?, ?, 'IWM', ?, ?, '2026-01-01', '2026-01-01')",
        (ticker, run_id, horizon, grade, status, excess, excess + 5.0),
    )


def test_backtest_report_computes_edge_estimate(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _closed(conn, ticker="A", grade="DEPLOY_READY", excess=12.0)
        _closed(conn, ticker="B", grade="DEPLOY_READY", excess=8.0)
        _closed(conn, ticker="C", grade="NOT_CHEAP", excess=-2.0)
        _closed(conn, ticker="D", grade="NOT_CHEAP", excess=-4.0)
        conn.commit()

    from app.backtest.report import backtest_report

    rep = backtest_report(run_id_prefix="bt", horizon=365)
    assert rep["n"] == 4
    assert rep["by_signal"]["DEPLOY_READY"]["n"] == 2
    assert rep["by_signal"]["DEPLOY_READY"]["n_with_excess"] == 2
    assert rep["by_signal"]["DEPLOY_READY"]["avg_excess"] == 10.0
    assert rep["by_signal"]["NOT_CHEAP"]["avg_excess"] == -3.0
    assert rep["edge_estimate"] == 13.0


def test_render_backtest_report_markdown(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _closed(conn, ticker="X", grade="DEPLOY_READY", excess=6.0)
        _closed(conn, ticker="Y", grade="NOT_CHEAP", excess=-2.0)
        conn.commit()

    from app.backtest.report import backtest_report, render_backtest_report

    rep = backtest_report(run_id_prefix="bt", horizon=365)
    md = render_backtest_report(rep)

    assert "# Edge Backtest Report" in md
    assert "Edge estimate" in md
    # edge = 6.0 - (-2.0) = 8.0
    assert "Edge estimate (cheap minus not-cheap avg excess): 8.0" in md
