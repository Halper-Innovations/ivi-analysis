"""Tests for app.market.commodity_context — injecting commodity prices into sector scans."""

from __future__ import annotations

import json
import time
from datetime import datetime

import pytest

from app.market.commodity_context import (
    COMMODITY_CATALOG,
    SECTOR_COMMODITIES,
    _cache_path,
    _classify_position,
    compute_snapshot_from_prices,
    fetch_commodity_snapshot,
    format_commodity_block,
    get_sector_commodity_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear_prices(
    n: int, start: float, end: float, start_date: str = "2026-01-01"
) -> list[tuple[str, float]]:
    """Build n (date, price) pairs linearly ramping from start to end."""
    base = datetime.fromisoformat(start_date)
    out: list[tuple[str, float]] = []
    if n == 1:
        return [(base.strftime("%Y-%m-%d"), start)]
    step = (end - start) / (n - 1)
    for i in range(n):
        d = base.replace(day=min(base.day, 1) + (i % 28))  # safe day arithmetic
        d = datetime(2026, 1, 1).toordinal() + i
        date_str = datetime.fromordinal(d).strftime("%Y-%m-%d")
        out.append((date_str, start + i * step))
    return out


class _FakeYfHistory:
    """Minimal stand-in for yf.Ticker.history() return value."""

    def __init__(self, prices: list[tuple[str, float]]):
        self._prices = prices

    def __len__(self) -> int:
        return len(self._prices)

    def iterrows(self):
        for ds, px in self._prices:
            ts = datetime.fromisoformat(ds)
            yield ts, {"Close": px}


class _FakeYfTicker:
    def __init__(self, symbol: str, prices: list[tuple[str, float]]):
        self._symbol = symbol
        self._prices = prices

    def history(self, start=None, end=None):
        return _FakeYfHistory(self._prices)


def _ticker_factory(prices_by_symbol: dict[str, list[tuple[str, float]]]):
    def _factory(symbol: str):
        if symbol not in prices_by_symbol:
            raise ValueError(f"unexpected symbol {symbol}")
        return _FakeYfTicker(symbol, prices_by_symbol[symbol])

    return _factory


# ---------------------------------------------------------------------------
# compute_snapshot_from_prices — pure math
# ---------------------------------------------------------------------------


def test_compute_snapshot_returns_none_for_empty():
    assert compute_snapshot_from_prices(symbol="CL=F", prices=[]) is None


def test_compute_snapshot_single_point():
    s = compute_snapshot_from_prices(
        symbol="CL=F",
        prices=[("2026-04-14", 90.0)],
    )
    assert s is not None
    assert s.symbol == "CL=F"
    assert s.current_price == 90.0
    assert s.display_name == "WTI Crude"
    assert s.unit == "$/bbl"
    # Single point → range collapsed → position undefined
    assert s.position_in_180d_range is None
    assert s.interpretation == "neutral"


def test_compute_snapshot_rising_prices_elevated():
    """Linear ramp from $60 to $95 → current should be ELEVATED."""
    prices = _make_linear_prices(180, 60.0, 95.0)
    s = compute_snapshot_from_prices(symbol="CL=F", prices=prices)
    assert s is not None
    assert s.current_price == pytest.approx(95.0)
    assert s.interpretation == "elevated"
    assert s.position_in_180d_range is not None
    assert s.position_in_180d_range > 0.9  # very near the top


def test_compute_snapshot_falling_prices_depressed():
    """Linear ramp from $95 to $60 → current should be DEPRESSED."""
    prices = _make_linear_prices(180, 95.0, 60.0)
    s = compute_snapshot_from_prices(symbol="CL=F", prices=prices)
    assert s is not None
    assert s.current_price == pytest.approx(60.0)
    assert s.interpretation == "depressed"
    assert s.position_in_180d_range is not None
    assert s.position_in_180d_range < 0.1


def test_compute_snapshot_flat_prices_neutral():
    """Perfectly flat prices → range collapsed → interpretation neutral."""
    prices = [(f"2026-0{(i // 30) + 1}-{(i % 30) + 1:02d}", 70.0) for i in range(90)]
    prices = _make_linear_prices(180, 70.0, 70.0)
    s = compute_snapshot_from_prices(symbol="CL=F", prices=prices)
    assert s is not None
    # Flat range means we can't compute a meaningful position
    assert s.position_in_180d_range is None
    assert s.interpretation == "neutral"


def test_compute_snapshot_pct_change_90d():
    """Ramp from $60 to $95 over 180 days: 90 days ago ≈ $77.50 midpoint, current $95."""
    prices = _make_linear_prices(180, 60.0, 95.0)
    s = compute_snapshot_from_prices(symbol="CL=F", prices=prices)
    assert s is not None
    # Current $95, 90-day window starts around midpoint; pct change should be positive and meaningful
    assert s.pct_change_90d is not None
    assert s.pct_change_90d > 10.0  # clearly risen


def test_classify_position_boundaries():
    assert _classify_position(None) == "neutral"
    assert _classify_position(0.0) == "depressed"
    assert _classify_position(0.25) == "depressed"
    assert _classify_position(0.26) == "neutral"
    assert _classify_position(0.5) == "neutral"
    assert _classify_position(0.74) == "neutral"
    assert _classify_position(0.75) == "elevated"
    assert _classify_position(1.0) == "elevated"


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------


def test_cache_roundtrip(tmp_path):
    """A snapshot written to cache can be read back identically."""
    prices = _make_linear_prices(180, 60.0, 95.0)
    factory = _ticker_factory({"CL=F": prices})

    cache_dir = tmp_path / "commodities"

    # First call: writes to cache
    s1 = fetch_commodity_snapshot(
        "CL=F", cache_dir=cache_dir, yf_ticker_factory=factory
    )
    assert s1 is not None
    assert (cache_dir / "CL_F.json").exists()

    # Second call: reads from cache (factory should NOT be called)
    def _fail_factory(sym):
        raise AssertionError("should have hit cache")

    s2 = fetch_commodity_snapshot(
        "CL=F", cache_dir=cache_dir, yf_ticker_factory=_fail_factory
    )
    assert s2 is not None
    assert s2.current_price == s1.current_price
    assert s2.interpretation == s1.interpretation


def test_cache_ttl_expiry(tmp_path):
    """If cache is older than TTL, we re-fetch."""
    prices = _make_linear_prices(30, 60.0, 70.0)
    factory = _ticker_factory({"CL=F": prices})

    cache_dir = tmp_path / "commodities"
    s1 = fetch_commodity_snapshot(
        "CL=F", cache_dir=cache_dir, cache_ttl_seconds=86400, yf_ticker_factory=factory
    )
    assert s1 is not None

    # Age the cache file past TTL
    cache_file = cache_dir / "CL_F.json"
    old_time = time.time() - 100000  # > 24h
    import os as _os

    _os.utime(cache_file, (old_time, old_time))

    calls = {"n": 0}

    def _counting_factory(sym):
        calls["n"] += 1
        return _FakeYfTicker(sym, prices)

    s2 = fetch_commodity_snapshot(
        "CL=F",
        cache_dir=cache_dir,
        cache_ttl_seconds=86400,
        yf_ticker_factory=_counting_factory,
    )
    assert s2 is not None
    assert calls["n"] == 1  # re-fetched


def test_cache_path_sanitizes_symbol(tmp_path):
    """Symbols with '=' and '/' become filesystem-safe."""
    p = _cache_path("CL=F", tmp_path)
    assert "=" not in p.name
    assert p.name == "CL_F.json"


# ---------------------------------------------------------------------------
# fetch_commodity_snapshot failure modes
# ---------------------------------------------------------------------------


def test_fetch_returns_none_on_yfinance_exception(tmp_path):
    def _broken_factory(sym):
        raise RuntimeError("network down")

    s = fetch_commodity_snapshot(
        "CL=F", cache_dir=tmp_path, yf_ticker_factory=_broken_factory
    )
    assert s is None


def test_fetch_returns_none_on_empty_history(tmp_path):
    factory = _ticker_factory({"CL=F": []})
    s = fetch_commodity_snapshot(
        "CL=F", cache_dir=tmp_path, yf_ticker_factory=factory
    )
    assert s is None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def test_format_commodity_block_empty():
    assert format_commodity_block("energy", []) == ""


def test_format_commodity_block_contains_numbers_and_sector_notes():
    prices = _make_linear_prices(180, 60.0, 95.0)
    s = compute_snapshot_from_prices(symbol="CL=F", prices=prices)
    assert s is not None

    block = format_commodity_block("energy", [s])
    # Header
    assert "COMMODITY CONTEXT" in block
    assert "energy" in block
    # Current price appears
    assert "95.00" in block or "95.0" in block
    # Interpretation appears
    assert "ELEVATED" in block
    # Sector-specific interpretive note
    assert "Upstream" in block  # energy sector note mentions upstream E&P


def test_format_commodity_block_skips_sector_note_for_unmapped():
    prices = _make_linear_prices(30, 60.0, 65.0)
    s = compute_snapshot_from_prices(symbol="CL=F", prices=prices)
    assert s is not None
    block = format_commodity_block("unknown_sector", [s])
    # Still renders the commodity table
    assert "COMMODITY CONTEXT" in block
    # But has no sector-specific interpretive note
    assert "INTERPRETIVE NOTES" not in block


# ---------------------------------------------------------------------------
# get_sector_commodity_context — integration
# ---------------------------------------------------------------------------


def test_get_sector_context_empty_for_non_commodity_sector(tmp_path):
    """Enterprise software has no commodity mapping → empty block."""
    block, snapshots = get_sector_commodity_context(
        "enterprise_software", cache_dir=tmp_path
    )
    assert block == ""
    assert snapshots == []


def test_get_sector_context_energy_returns_all_three(tmp_path):
    """Energy maps to WTI, Brent, NatGas — all should be fetched."""
    prices_map = {
        "CL=F": _make_linear_prices(180, 60.0, 95.0),
        "BZ=F": _make_linear_prices(180, 62.0, 98.0),
        "NG=F": _make_linear_prices(180, 3.5, 2.6),
    }
    factory = _ticker_factory(prices_map)

    block, snapshots = get_sector_commodity_context(
        "energy", cache_dir=tmp_path, yf_ticker_factory=factory
    )
    assert len(snapshots) == 3
    symbols = {s.symbol for s in snapshots}
    assert symbols == {"CL=F", "BZ=F", "NG=F"}
    # WTI rose → elevated
    wti = next(s for s in snapshots if s.symbol == "CL=F")
    assert wti.interpretation == "elevated"
    # NatGas fell → depressed
    ng = next(s for s in snapshots if s.symbol == "NG=F")
    assert ng.interpretation == "depressed"
    # Block contains all three display names
    assert "WTI Crude" in block
    assert "Brent Crude" in block
    assert "Natural Gas" in block


def test_get_sector_context_returns_empty_when_all_fetches_fail(tmp_path):
    """If yfinance fails for all commodities, fall back to empty block."""
    def _broken(sym):
        raise RuntimeError("offline")

    block, snapshots = get_sector_commodity_context(
        "energy", cache_dir=tmp_path, yf_ticker_factory=_broken
    )
    assert block == ""
    assert snapshots == []


def test_get_sector_context_partial_success(tmp_path):
    """If one commodity fetch fails, the other succeeds → block still rendered."""
    prices_map = {
        "CL=F": _make_linear_prices(180, 60.0, 95.0),
    }

    def _selective(sym):
        if sym in prices_map:
            return _FakeYfTicker(sym, prices_map[sym])
        raise RuntimeError(f"no data for {sym}")

    block, snapshots = get_sector_commodity_context(
        "energy", cache_dir=tmp_path, yf_ticker_factory=_selective
    )
    assert len(snapshots) == 1
    assert snapshots[0].symbol == "CL=F"
    assert "WTI Crude" in block


# ---------------------------------------------------------------------------
# Snapshot serialization
# ---------------------------------------------------------------------------


def test_snapshot_as_dict_is_json_serializable():
    prices = _make_linear_prices(30, 60.0, 70.0)
    s = compute_snapshot_from_prices(symbol="CL=F", prices=prices)
    assert s is not None

    d = s.as_dict()
    # Round-trip through JSON
    recovered = json.loads(json.dumps(d))
    # Ranges came back as lists (JSON has no tuples)
    assert isinstance(recovered["range_30d"], list)
    assert len(recovered["range_30d"]) == 2


# ---------------------------------------------------------------------------
# Sanity: catalog and mapping stay in sync
# ---------------------------------------------------------------------------


def test_every_mapped_commodity_is_in_catalog():
    """Every symbol in SECTOR_COMMODITIES must exist in COMMODITY_CATALOG."""
    for sector, syms in SECTOR_COMMODITIES.items():
        for sym in syms:
            assert sym in COMMODITY_CATALOG, (
                f"sector {sector} maps to {sym} but it's not in COMMODITY_CATALOG"
            )


def test_energy_sector_maps_to_oil_and_gas():
    """Regression: energy must always include at least crude oil."""
    syms = SECTOR_COMMODITIES.get("energy", [])
    assert "CL=F" in syms  # WTI must be there
    assert "NG=F" in syms  # Natural gas must be there
