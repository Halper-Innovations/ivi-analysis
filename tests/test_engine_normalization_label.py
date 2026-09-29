"""The engine's assumption card must not advertise a normalisation window it never uses.

build_ticker_valuation stamps
``"normalization_window_years": min(5, len(rows))`` (engine.py:1965), but the only
free-cash-flow observation it ever capitalises is the latest one:
``fcf_values = [float(fcf_value)]`` (engine.py:1553), so the "median" in
``input_snapshot["normalized_fcf_median"]`` is that single value. Observed on five
rows of FCF 10, 20, 30, 40, 50: the card said 5 years, ``positive_fcf_observations``
said 1, and the "median" was 50 (the median of five would be 30). The label must say
what was used.
"""

from __future__ import annotations

from app.valuation.engine import build_ticker_valuation


def _payload() -> dict:
    rows = []
    for year, fcf in [(2021, 10.0), (2022, 20.0), (2023, 30.0), (2024, 40.0), (2025, 50.0)]:
        rows.append(
            {
                "year": year,
                "fcf": fcf,
                "cfo": fcf + 5.0,
                "capex": 5.0,
                "shares_outstanding": 10.0,
                "net_debt": 0.0,
                "revenue": 1000.0,
                "op_margin": 0.10,
                "fcf_margin": fcf / 1000.0,
            }
        )
    return {"ticker": "TST", "rows": rows, "derived_signals": {}}


def test_the_normalization_window_label_equals_the_observations_actually_used():
    out = build_ticker_valuation(_payload(), run_id=None, as_of_date=None, with_prices=False)
    assert out["valuation_inputs"]["positive_fcf_observations"] == 1
    assert out["input_snapshot"]["normalized_fcf_median"] == 50.0
    assert out["assumption_card"]["normalization_window_years"] == 1
