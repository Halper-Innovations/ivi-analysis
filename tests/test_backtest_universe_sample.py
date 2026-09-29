from __future__ import annotations
from app.backtest.universe import EligibleName, stratified_sample


def _mk(ticker, category):
    return EligibleName(ticker=ticker, market_cap_asof=1.0, cap_category_asof=category, price_asof=1.0)


def test_stratified_sample_is_deterministic_and_per_band():
    eligible = (
        [_mk(f"MI{i}", "micro") for i in range(10)]
        + [_mk(f"SM{i}", "small") for i in range(10)]
        + [_mk(f"MD{i}", "mid") for i in range(1)]        # under-full band
        + [_mk(f"LG{i}", "large_cap") for i in range(3)]
        + [_mk(f"MG{i}", "mega_cap") for i in range(3)]   # folded into 'large'
        + [_mk(f"UN{i}", "UNKNOWN_CAP") for i in range(5)]  # never sampled
    )
    s1 = stratified_sample(eligible, per_band=2, seed=42)
    s2 = stratified_sample(eligible, per_band=2, seed=42)
    assert s1 == s2                                  # deterministic
    assert len([t for t in s1 if t.startswith("MI")]) == 2
    assert len([t for t in s1 if t.startswith("SM")]) == 2
    assert len([t for t in s1 if t.startswith("MD")]) == 1   # under-full -> take all
    # 'large' band = large_cap + mega_cap, per_band=2 -> 2 names from the 6
    assert len([t for t in s1 if t.startswith(("LG", "MG"))]) == 2
    assert not [t for t in s1 if t.startswith("UN")]         # UNKNOWN_CAP excluded
    assert s1 == sorted(s1)                                  # returned sorted


def test_stratified_sample_different_seed_differs():
    eligible = [_mk(f"MI{i}", "micro") for i in range(50)]
    assert stratified_sample(eligible, per_band=5, seed=1) != stratified_sample(
        eligible, per_band=5, seed=2
    )
