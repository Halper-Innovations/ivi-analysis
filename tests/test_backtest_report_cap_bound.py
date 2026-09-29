from __future__ import annotations
from app.config import get_config
from app.db import get_db, init_db


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def _closed(conn, ticker, grade, cap, realized, excess, status="FAIRLY_PRICED"):
    conn.execute(
        "INSERT INTO ticker_outcomes(ticker, as_of_date, run_id, decision, conviction, "
        "horizon_days, outcome_status, entry_price_source, grade, status, cap_category, "
        "realized_return_pct, excess_return_pct, created_at, updated_at) "
        "VALUES(?, '2024-01-02', 'h1_sample_h365', 'BUY', 2, 365, 'CLOSED', "
        "'historical_backtest', ?, ?, ?, ?, ?, '2025', '2025')",
        (ticker, grade, status, cap, realized, excess),
    )


def _unresolved(conn, ticker, grade, cap):
    conn.execute(
        "INSERT INTO ticker_outcomes(ticker, as_of_date, run_id, decision, conviction, "
        "horizon_days, outcome_status, entry_price_source, grade, status, cap_category, "
        "created_at, updated_at) "
        "VALUES(?, '2024-01-02', 'h1_sample_h365', 'BUY', 2, 365, 'UNRESOLVED', "
        "'historical_backtest', ?, 'CHEAP_VS_EXPECTATIONS', ?, '2025', '2025')",
        (ticker, grade, cap),
    )


def test_report_by_cap_category(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    from app.backtest.report import backtest_report

    with get_db() as conn:
        _closed(conn, "BIG", "DEPLOY_READY", "large_cap", 20.0, 8.0)
        _closed(conn, "SML", "DEPLOY_READY", "small", 10.0, 4.0)
        _closed(conn, "MIC", "NOT_CHEAP", "micro", -5.0, -3.0)

    report = backtest_report(run_id_prefix="h1_sample", horizon=365)
    assert set(report["by_cap_category"].keys()) == {"large_cap", "small", "micro"}
    assert report["by_cap_category"]["large_cap"]["n"] == 1
    assert report["by_cap_category"]["large_cap"]["avg_realized"] == 20.0


def test_survivorship_bound(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    from app.backtest.report import compute_survivorship_bound

    with get_db() as conn:
        # DEPLOY_READY cohort: two resolved (+20, +10) -> avg 15.0; plus one UNRESOLVED
        _closed(conn, "D1", "DEPLOY_READY", "small", 20.0, 8.0)
        _closed(conn, "D2", "DEPLOY_READY", "small", 10.0, 4.0)
        _unresolved(conn, "D3", "DEPLOY_READY", "small")
        # NOT_CHEAP cohort: one resolved (0.0)
        _closed(conn, "N1", "NOT_CHEAP", "micro", 0.0, 0.0)

    bound = compute_survivorship_bound(run_id_prefix="h1_sample", horizon=365)
    dr = bound["by_signal"]["DEPLOY_READY"]
    assert dr["n_closed"] == 2
    assert dr["n_unresolved"] == 1
    assert dr["avg_realized_excluded"] == 15.0                 # (20+10)/2
    assert dr["avg_realized_minus50"] == -6.6667               # (20+10-50)/3
    assert dr["avg_realized_minus100"] == -23.3333             # (20+10-100)/3
    assert bound["edge_excluded"] == 15.0                      # 15.0 - 0.0
    assert bound["edge_minus50"] == -6.6667                    # -6.6667 - 0.0
    assert bound["edge_minus100"] == -23.3333                  # -23.3333 - 0.0
