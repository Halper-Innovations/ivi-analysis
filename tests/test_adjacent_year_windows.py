"""A "recent window" must be calendar-adjacent fiscal years ending at the latest year.

Taking the newest N ROWS spliced old years across a gap (2016 beside 2021-2024) and
labelled the result a recent window. With a gap, only the adjacent run ending at
the latest year counts; a run shorter than the module's minimum is UNKNOWN /
insufficient, with a reason. One test per module, plus the scout equity-bridge fix.
"""

from __future__ import annotations

import json

from app.valuation.adjacent_years import trailing_adjacent_run

UNKNOWN = "UNKNOWN"


def test_trailing_adjacent_run_keeps_only_the_run_ending_at_the_latest_year():
    assert trailing_adjacent_run([(2016, 1), (2021, 2), (2022, 3), (2023, 4)]) == [
        (2021, 2),
        (2022, 3),
        (2023, 4),
    ]
    assert trailing_adjacent_run([]) == []


# -- cyclical_normalization ---------------------------------------------------


def test_cyclical_normalization_does_not_reach_across_a_year_gap():
    """Years 2015-2016 then 2021-2023: only 2021-2023 (three adjacent years) is the
    window. Before the fix the two old, very different years entered the median
    and the cyclicality read (CLEARLY_CYCLICAL from 5 positive points)."""
    from app.valuation.cyclical_normalization import compute_cyclical_normalization

    def rows(pairs):
        return [{"year": y, "value": v, "derived_from": []} for y, v in pairs]

    oe = rows([(2015, 1000.0), (2016, 5.0), (2021, 100.0), (2022, 100.0), (2023, 100.0)])
    out = compute_cyclical_normalization(
        ticker="T", as_of_date="2024-06-30", owner_earnings_series=oe, fcf_series=[], cfo_series=[]
    )
    assert out["series_points_available"] == 3
    assert out["positive_series_points"] == 3
    assert out["cyclical_profile_class"] == "LOW_CYCLICALITY"
    assert "NON_ADJACENT_YEARS_DROPPED" in out["cyclical_normalization_reason_codes"]

    # Two adjacent years after the gap is below the three-point minimum: UNKNOWN.
    short = compute_cyclical_normalization(
        ticker="T",
        as_of_date="2024-06-30",
        owner_earnings_series=rows([(2015, 9.0), (2016, 8.0), (2017, 7.0), (2022, 100.0), (2023, 100.0)]),
        fcf_series=[],
        cfo_series=[],
    )
    assert short["cyclical_profile_class"] == "CYCLICALITY_UNKNOWN"
    assert short["conservative_cyclical_denominator"] == UNKNOWN


# -- intangible_economics ------------------------------------------------------


def test_intangible_gross_margin_window_ignores_years_before_a_gap():
    """Margins 0.90, 0.10 in 2015-2016 then 0.50 for 2021-2023. Only 2021-2023 is
    recent: average 0.5, floor 0.5. Before the fix the newest five rows were
    2016, 2021, 2022, 2023 and 2015 (floor 0.1, a large volatility)."""
    from app.valuation.intangible_economics import _gross_margin_durability

    rows = [
        {"year": y, "value": v, "derived_from": []}
        for y, v in [(2015, 0.9), (2016, 0.1), (2021, 0.5), (2022, 0.5), (2023, 0.5)]
    ]
    out = _gross_margin_durability({"gross_margin": rows})
    assert out["gross_margin_avg_5y"]["value"] == 0.5
    assert out["gross_margin_floor_5y"]["value"] == 0.5

    # After the gap only two adjacent years remain: fewer than three -> UNKNOWN score.
    two = _gross_margin_durability(
        {"gross_margin": [{"year": y, "value": 0.5, "derived_from": []} for y in (2015, 2016, 2017, 2022, 2023)]}
    )
    assert two["gross_margin_durability_score"]["value"] == UNKNOWN
    assert "INSUFFICIENT_GROSS_MARGIN_HISTORY" in two["reason_codes"]


# -- depreciation_audit --------------------------------------------------------


def test_depreciation_audit_needs_three_adjacent_years_for_a_trend():
    """Rates 12.5% (2017), 10% (2020), 8% (2021): the last three ROWS decline, but
    2017 to 2020 is a gap, so there is no three-year trend and no DECLINING flag
    (nor a capex-below-depreciation flag); before the fix all three rows counted."""
    from app.valuation.depreciation_audit import compute_depreciation_audit

    years = [2017, 2020, 2021]
    da = [(y, 10.0) for y in years]
    ppe_by_year = {2017: 80.0, 2020: 100.0, 2021: 125.0}  # 12.5% -> 10% -> 8%
    facts = {
        "depreciation_amortization": da,
        "gross_ppe": [(y, ppe_by_year[y]) for y in years],
        "capex": [(y, 1.0) for y in years],
    }
    out = compute_depreciation_audit(facts)
    assert "DEPRECIATION_RATE_DECLINING" not in out["depreciation_flags"]
    assert "CAPEX_BELOW_DEPRECIATION" not in out["depreciation_flags"]
    assert "NON_ADJACENT_YEARS" in out["reason_codes"]

    adjacent = {
        "depreciation_amortization": [(y, 10.0) for y in (2019, 2020, 2021)],
        "gross_ppe": [(2019, 80.0), (2020, 100.0), (2021, 125.0)],
        "capex": [(y, 1.0) for y in (2019, 2020, 2021)],
    }
    out2 = compute_depreciation_audit(adjacent)
    assert "DEPRECIATION_RATE_DECLINING" in out2["depreciation_flags"]
    assert "CAPEX_BELOW_DEPRECIATION" in out2["depreciation_flags"]


# -- lenses --------------------------------------------------------------------


def test_lenses_use_only_the_adjacent_run_and_refuse_when_it_is_too_short():
    """Operating income for 2016, 2017, 2018 (a five-year window would be padded
    with them) then only 2022-2023: two adjacent years is below the three-year
    minimum, so the anchor is METHOD_INSUFFICIENT_DATA with a reason flag."""
    from app.valuation.lenses import ev_ebit_anchor, fcf_yield_anchor

    facts = {
        "operating_income": [(2016, 100.0), (2017, 100.0), (2018, 100.0), (2022, 100.0), (2023, 100.0)],
        "cfo": [(2016, 100.0), (2017, 100.0), (2018, 100.0), (2022, 100.0), (2023, 100.0)],
        "capex": [(2016, 10.0), (2017, 10.0), (2018, 10.0), (2022, 10.0), (2023, 10.0)],
    }
    ev = ev_ebit_anchor(facts, shares=10.0, bridge_deduction=0.0, category="TRADITIONAL_OPERATING")
    assert ev["status"] == "METHOD_INSUFFICIENT_DATA"
    assert "NON_ADJACENT_YEARS" in ev["flags"]
    fy = fcf_yield_anchor(facts, shares=10.0, category="TRADITIONAL_OPERATING")
    assert fy["status"] == "METHOD_INSUFFICIENT_DATA"
    assert "NON_ADJACENT_YEARS" in fy["flags"]

    # With three adjacent years after the gap the median uses only those.
    facts["operating_income"] = [(2016, 1000.0), (2021, 100.0), (2022, 100.0), (2023, 100.0)]
    ok = ev_ebit_anchor(facts, shares=10.0, bridge_deduction=0.0, category="TRADITIONAL_OPERATING")
    assert ok["basis"]["ebit_normalized"] == 100.0


# -- owner_earnings ------------------------------------------------------------


def test_owner_earnings_normalization_does_not_bridge_a_missing_year(monkeypatch, tmp_path):
    """Owner earnings for 2016 (1000) and 2021-2022 (100 each): 2016 is not one of
    'the last three years'. Only two adjacent years remain, so the method is
    AVG_2Y = 100, not MEDIAN_3Y = 100 with 2016 counted as a third year."""
    from app.valuation.owner_earnings import compute_owner_earnings_series

    def fact(year, value):
        return {"end": f"{year}-12-31", "filed": f"{year + 1}-02-01", "val": value}

    cfo = [fact(2016, 1000.0), fact(2021, 100.0), fact(2022, 100.0)]
    capex = [fact(2016, 0.0), fact(2021, 0.0), fact(2022, 0.0)]
    payload = {
        "companyfacts": {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": cfo}},
                    "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": capex}},
                }
            }
        }
    }
    path = tmp_path / "cf.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "app.valuation.owner_earnings.resolve_financial_facts_asof",
        lambda **_kw: {"ticker": "GGG", "cache_path": str(path), "derived_from": []},
    )
    out = compute_owner_earnings_series("GGG", "2023-06-30", years_back=5)
    assert [row["year"] for row in out["series"]] == [2021, 2022]
    assert out["summary"]["owner_earnings_normalized_method"] == "AVG_2Y"
    assert out["summary"]["owner_earnings_points"] == 2
    assert "NON_ADJACENT_YEARS_DROPPED" in out["reason_codes"]


# -- universe scout ------------------------------------------------------------


def test_scout_intrinsic_proxy_does_not_subtract_net_debt_from_equity_cash_flow(tmp_path):
    """The scout's own proxy is 12x FCF (CFO less capex). CFO is after interest, so
    that is already equity value: FCF 5 x 12 = 60 over 10 shares = 6.0 per share.
    Before the fix net debt of 50 was subtracted again: (60 - 50) / 10 = 1.0."""
    from app.universe import scout

    cache = tmp_path / "companyfacts.json"
    cache.write_text(json.dumps({"companyfacts": {"facts": {"us-gaap": {}}}}), encoding="utf-8")
    facts_row = {
        "status": "OK",
        "shares_status": "OK",
        "cfo_status": "OK",
        "capex_status": "OK",
        "fcf_status": "OK",
        "shares_value": 10.0,
        "cfo_value": 20.0,
        "capex_value": 15.0,
        "fcf_value": 5.0,
        "cache_path": str(cache),
        "derived_from": [],
    }
    record, _, _ = scout._build_scout_record(
        ticker="TEST",
        as_of_date="2025-06-30",
        price_row={"status": "OK", "price": 10.0},
        facts_row=facts_row,
        net_debt_resolved={"net_debt_proxy": 50.0, "reason_code": "OK", "derived_from": []},
        thresholds=scout.normalize_scout_thresholds(),
        require_ev_yield=False,
    )
    assert record["metric_values"]["intrinsic_per_share_proxy"] == 6.0
