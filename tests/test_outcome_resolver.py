from __future__ import annotations

from app.calibration.outcome_resolver import resolve_perception
from app.calibration.schemas import PerceptionTrackingRecord
from app.db import init_db


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker,cik,name\nAAA,1,AAA\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _tracking(direction: str = "UNDERVALUED") -> PerceptionTrackingRecord:
    return PerceptionTrackingRecord(
        perception_id="p1",
        ticker="AAA",
        as_of_date="2026-03-22",
        thesis="Test",
        direction=direction,
        confidence="MEDIUM",
        testable_prediction="Test",
        falsification_trigger="Test",
        time_horizon="SHORT",
        expected_resolution_date="2027-03-22",
        supporting_signal_sources=["VALUATION"],
        pattern_ids_involved=[],
        diff_signal_types_involved=[],
        status="PENDING",
        registered_at="2026-03-22T00:00:00+00:00",
        resolved_at=None,
        resolution=None,
        derived_from=[],
        registered_market_price=100.0,
    )


def test_resolve_undervalued_confirmed(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.calibration.outcome_resolver.utc_now_iso", lambda: "2027-03-22T00:00:00+00:00")
    record = resolve_perception(_tracking(), outcome_data={"price_change_pct": 15.0}, cfg=cfg)
    assert record.status == "CONFIRMED"
    assert record.resolution is not None
    assert record.resolution.resolution_method == "price_directional_v1"


def test_resolve_undervalued_disconfirmed(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.calibration.outcome_resolver.utc_now_iso", lambda: "2027-03-22T00:00:00+00:00")
    record = resolve_perception(_tracking(), outcome_data={"price_change_pct": -20.0}, cfg=cfg)
    assert record.status == "DISCONFIRMED"


def test_resolve_undervalued_inconclusive(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.calibration.outcome_resolver.utc_now_iso", lambda: "2027-03-22T00:00:00+00:00")
    record = resolve_perception(_tracking(), outcome_data={"price_change_pct": 5.0}, cfg=cfg)
    assert record.status == "INCONCLUSIVE"


def test_resolve_with_no_price_data(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.calibration.outcome_resolver.utc_now_iso", lambda: "2027-03-22T00:00:00+00:00")
    record = resolve_perception(_tracking(), outcome_data={}, cfg=cfg)
    assert record.status == "INSUFFICIENT_DATA"


def test_resolve_overvalued_confirmed(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.calibration.outcome_resolver.utc_now_iso", lambda: "2027-03-22T00:00:00+00:00")
    record = resolve_perception(_tracking(direction="OVERVALUED"), outcome_data={"price_change_pct": -15.0}, cfg=cfg)
    assert record.status == "CONFIRMED"


# FIX 2: direct-payload price_change_pct is a PERCENT value and must not be
# rescaled by a <=1.0 magnitude heuristic. A real +0.8% move must stay +0.8%
# (INCONCLUSIVE), not become +80% (CONFIRMED).
def test_resolve_small_positive_percent_stays_inconclusive(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.calibration.outcome_resolver.utc_now_iso", lambda: "2027-03-22T00:00:00+00:00")
    record = resolve_perception(_tracking(), outcome_data={"price_change_pct": 0.8}, cfg=cfg)
    assert record.status == "INCONCLUSIVE"
    assert record.resolution is not None
    assert record.resolution.price_change_pct == 0.8


def test_resolve_fractional_percent_not_rescaled(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.calibration.outcome_resolver.utc_now_iso", lambda: "2027-03-22T00:00:00+00:00")
    record = resolve_perception(_tracking(), outcome_data={"price_change_pct": 0.5}, cfg=cfg)
    assert record.status == "INCONCLUSIVE"
    assert record.resolution is not None
    assert record.resolution.price_change_pct == 0.5


# FIX 1: when no direct payload / current_valuation is provided, the point-in-time
# price must be fetched via the date-aware market provider's get_price_asof, NOT
# the date-ignoring valuation get_quote. A snapshot of 130 vs registered 100 = +30%.
def test_infer_uses_market_provider_get_price_asof(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.calibration.outcome_resolver.utc_now_iso", lambda: "2027-03-22T00:00:00+00:00")

    from app.market.price_provider import PriceSnapshot

    calls: list[tuple[str, str]] = []

    class _StubProvider:
        def get_price_asof(self, ticker: str, as_of_date: str):
            calls.append((ticker, as_of_date))
            return PriceSnapshot(
                ticker=ticker.upper(),
                as_of_date=as_of_date,
                price=130.0,
                currency="USD",
                source="stub",
                retrieved_at="2027-03-22T00:00:00+00:00",
                confidence="HIGH",
            )

    monkeypatch.setattr(
        "app.market.price_provider.get_default_provider",
        lambda *args, **kwargs: _StubProvider(),
    )

    def _fail_quote(*args, **kwargs):
        raise AssertionError("valuation get_quote must not be used (look-ahead bias)")

    monkeypatch.setattr("app.valuation.price_provider.get_default_provider", _fail_quote)

    record = resolve_perception(
        _tracking(),
        outcome_data={"as_of_date": "2027-03-22"},
        cfg=cfg,
    )
    assert record.status == "CONFIRMED"
    assert record.resolution is not None
    assert record.resolution.price_change_pct == 30.0
    assert calls == [("AAA", "2027-03-22")]


def test_infer_returns_insufficient_when_no_point_in_time_price(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.calibration.outcome_resolver.utc_now_iso", lambda: "2027-03-22T00:00:00+00:00")

    class _NoneProvider:
        def get_price_asof(self, ticker: str, as_of_date: str):
            return None

    monkeypatch.setattr(
        "app.market.price_provider.get_default_provider",
        lambda *args, **kwargs: _NoneProvider(),
    )

    record = resolve_perception(
        _tracking(),
        outcome_data={"as_of_date": "2027-03-22"},
        cfg=cfg,
    )
    assert record.status == "INSUFFICIENT_DATA"
    assert record.resolution is not None
    assert record.resolution.price_change_pct is None
