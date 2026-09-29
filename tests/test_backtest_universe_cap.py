from __future__ import annotations
from app.backtest.universe import cap_category_from_market_cap, sampling_band


def test_cap_category_boundaries():
    # millions USD; boundaries are lower-inclusive, upper-exclusive.
    # 2026-06-11 band redefinition: micro < 500, small 500-2,500,
    # mid 2,500-10,000 (canonical: sector_candidates.MARKET_CAP_FOCUS_TIERS).
    assert cap_category_from_market_cap(150.0) == "micro"
    assert cap_category_from_market_cap(499.0) == "micro"
    assert cap_category_from_market_cap(500.0) == "small"
    assert cap_category_from_market_cap(2_499.0) == "small"
    assert cap_category_from_market_cap(2_500.0) == "mid"
    assert cap_category_from_market_cap(9_999.0) == "mid"
    assert cap_category_from_market_cap(10_000.0) == "large_cap"
    assert cap_category_from_market_cap(199_999.0) == "large_cap"
    assert cap_category_from_market_cap(200_000.0) == "mega_cap"
    assert cap_category_from_market_cap(3_000_000.0) == "mega_cap"


def test_cap_category_invalid_is_none():
    assert cap_category_from_market_cap(None) is None
    assert cap_category_from_market_cap(0.0) is None
    assert cap_category_from_market_cap(-5.0) is None


def test_sampling_band_folds_mega_into_large():
    assert sampling_band("micro") == "micro"
    assert sampling_band("small") == "small"
    assert sampling_band("mid") == "mid"
    assert sampling_band("large_cap") == "large"
    assert sampling_band("mega_cap") == "large"
    assert sampling_band("UNKNOWN_CAP") is None
    assert sampling_band(None) is None
