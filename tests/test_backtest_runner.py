from __future__ import annotations
from app.config import get_config
from app.db import get_db, init_db


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def test_run_backtest_snapshots_one_row_per_valid_signal(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    from app.backtest import runner
    from app.backtest.reconstruct import ReconstructedSignal, ReconstructResult

    def _fake_reconstruct(ticker, as_of_date, **kw):
        if ticker == "SKIP":
            return ReconstructResult(ticker, as_of_date, None, "NO_PRICE_AT_AS_OF")
        return ReconstructResult(
            ticker,
            as_of_date,
            ReconstructedSignal(
                ticker,
                as_of_date,
                anchor=100.0,
                price=60.0,
                buy_price_target=75.0,
                mos=0.4,
                deploy_ready=True,
                expectations_gap_bucket="CHEAP_VS_EXPECTATIONS",
                cap_category=None,
            ),
            None,
        )

    monkeypatch.setattr(runner, "reconstruct_signal_asof", _fake_reconstruct)

    summary = runner.run_backtest(
        universe=["AAA", "SKIP"],
        dates=["2024-01-02"],
        horizons=[90, 365],
        run_id="backtest_pilot",
    )
    assert summary.reconstructed == 1
    assert summary.skipped == 1
    assert summary.skipped_reasons["SKIP"] == "NO_PRICE_AT_AS_OF"
    assert summary.rows_written == 2
    assert summary.universe_size == 2

    with get_db() as conn:
        rows = conn.execute(
            "SELECT ticker, horizon_days, decision, grade, entry_price, entry_price_source, "
            "benchmark_symbol, status, buy_price_target FROM ticker_outcomes "
            "WHERE run_id IN ('backtest_pilot_h90','backtest_pilot_h365') ORDER BY horizon_days"
        ).fetchall()
    assert [r["horizon_days"] for r in rows] == [90, 365]
    assert rows[0]["ticker"] == "AAA"
    assert rows[0]["entry_price"] == 60.0
    assert rows[0]["entry_price_source"] == "historical_backtest"
    assert rows[0]["decision"] == "BUY"
    assert rows[0]["grade"] == "DEPLOY_READY"
    assert rows[0]["benchmark_symbol"] == "IWM"   # cap unknown -> small/mid default
    assert rows[0]["status"] == "CHEAP_VS_EXPECTATIONS"
    assert rows[0]["buy_price_target"] == 75.0
