from __future__ import annotations

from dataclasses import dataclass

from app.config import get_config
from app.db import get_db, init_db


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


@dataclass
class _Snap:
    ticker: str
    as_of_date: str
    price: float


class _FakeProvider:
    def get_price_asof(self, ticker, as_of_date):
        return _Snap(ticker, as_of_date, 50.0)


def test_run_h1_builds_samples_and_snapshots(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    from app.backtest import h1
    from app.backtest.universe import EligibleName
    from app.backtest.reconstruct import ReconstructedSignal, ReconstructResult

    # Stub eligibility -> a fixed 2-name sample per date, both 'small'.
    eligibility_calls = []

    def _eligible(d, **kwargs):
        eligibility_calls.append((d, kwargs))
        return [
            EligibleName("AAA", 1500.0, "small", 50.0),
            EligibleName("BBB", 1600.0, "small", 50.0),
        ]

    monkeypatch.setattr(
        h1,
        "eligible_universe_asof",
        _eligible,
    )

    # Stub reconstruct -> always a valid DEPLOY_READY small-cap signal.
    def _fake_recon(ticker, as_of_date, **kw):
        return ReconstructResult(
            ticker,
            as_of_date,
            ReconstructedSignal(
                ticker, as_of_date, 100.0, 50.0, 75.0, 0.5, True, "CHEAP_VS_EXPECTATIONS", "small"
            ),
            None,
        )

    monkeypatch.setattr("app.backtest.runner.reconstruct_signal_asof", _fake_recon)

    summary = h1.run_h1(
        per_band=10,
        seed=42,
        dates=["2022-01-03", "2022-04-01"],
        horizons=[365],
        run_id="h1_sample",
        provider=_FakeProvider(),
        db_path=tmp_path / "historical.db",
    )
    # 2 names x 2 dates x 1 horizon = 4 rows
    assert summary.rows_written == 4
    with get_db() as conn:
        n = conn.execute(
            "SELECT COUNT(*) c FROM ticker_outcomes WHERE run_id='h1_sample_h365' "
            "AND benchmark_symbol='IWM' AND cap_category='small'"
        ).fetchone()["c"]
    assert n == 4
    assert [call[1]["db_path"] for call in eligibility_calls] == [
        tmp_path / "historical.db",
        tmp_path / "historical.db",
    ]


def test_h1_date_grid_is_14_quarters():
    from app.backtest.h1 import H1_DATE_GRID

    assert len(H1_DATE_GRID) == 14
    assert H1_DATE_GRID[0] == "2022-01-03"
    assert H1_DATE_GRID[-1] == "2025-04-01"
