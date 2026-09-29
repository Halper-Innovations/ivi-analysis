"""Regressions for app/valuation/valuation_writer.py.

Each test asserts the CORRECT answer for a defect that has been fixed; a future
defect of this kind belongs here as ``xfail(strict=True)`` until fixed, at which
point the test flips to XPASS and the marker must be removed.

Each docstring states the mechanism, the observed wrong answer, and the
correct answer.
"""

from __future__ import annotations


from app.valuation.valuation_writer import (
    _compute_downside_scenario,
    _compute_owner_earnings,
    _compute_quality_wacc,
    _discounted_owner_earnings,
    _margin_of_safety_scorecard,
    _normalize_capex,
)

# A 50%/yr revenue grower: 100 -> 506.25 over four years (CAGR exactly 0.50).
FAST_GROWER_REVENUE = [
    (2020, 100.0),
    (2021, 150.0),
    (2022, 225.0),
    (2023, 337.5),
    (2024, 506.25),
]
FLAT_OI_100 = [(year, 100.0) for year, _ in FAST_GROWER_REVENUE]
FLAT_REVENUE = [(2022, 100.0), (2023, 100.0), (2024, 100.0)]


# ── _compute_downside_scenario ──────────────────────────────────────────────


def test_bear_case_dcf_exceeds_base_case_for_fast_grower():
    """The bear-case DCF must never exceed the base-case DCF.

    Mechanism: ``_compute_downside_scenario`` uses ``bear_growth = max(cagr*0.5,
    -0.10)`` (floor only) while the base DCF in ``_discounted_owner_earnings``
    caps its base scenario at +8% and its high scenario at +15%. For a 50% CAGR
    grower the "bear" case projects +25%/yr for five years.

    Observed (OE 100, shares 1, net debt 0, WACC 10%): base 1636.61, high
    2164.80, bear_case_dcf 2640.65 — the bear case is 61% ABOVE the base case
    and 22% above the bull case, and the downside class reads LIMITED.
    Correct: bear_case_dcf <= base (a downside scenario cannot be worth more
    than the central one).
    """
    base = _discounted_owner_earnings(100.0, 1.0, 0.0, FAST_GROWER_REVENUE, wacc=0.10)
    bear = _compute_downside_scenario(
        owner_earnings=100.0,
        shares=1.0,
        net_debt=0.0,
        revenue_series=FAST_GROWER_REVENUE,
        operating_income_series=FLAT_OI_100,
        wacc=0.10,
        base_case_dcf=base["base"],
        current_price=2000.0,
    )
    assert round(base["base"], 2) == 1636.61
    assert bear["bear_case_dcf"] <= round(base["base"], 2)


def test_bear_case_epv_credits_tax_shield_on_loss_year():
    """A negative 5-year-low operating income must not be reduced by tax.

    Mechanism: ``nopat = low_oi * (1 - _TAX_RATE)`` is applied unconditionally,
    so a -100 operating loss becomes -79 "after tax" — the bear case is made
    21% LESS bad by a tax benefit the company may never realise.

    Observed: OI [50, -100, 60], net debt 0, shares 100, WACC 10% ->
    bear_case_epv -7.9. Correct: -1000/100 = -10.0 (no shield on a loss).
    """
    bear = _compute_downside_scenario(
        owner_earnings=10.0,
        shares=100.0,
        net_debt=0.0,
        revenue_series=FLAT_REVENUE,
        operating_income_series=[(2024, 50.0), (2023, -100.0), (2022, 60.0)],
        wacc=0.10,
        base_case_dcf=1.0,
    )
    assert bear["bear_case_epv"] == -10.0


# ── _compute_owner_earnings / _normalize_capex ──────────────────────────────

NEGATED_CAPEX_FACTS = {
    "cfo": [(year, 100.0) for year in range(2020, 2025)],
    # A filer that reports PaymentsToAcquirePropertyPlantAndEquipment negated
    # (ingest PLAUSIBILITY admits capex down to -500,000; 58 FY rows across 36
    # issuers are negative in the live store as of 2026-09-01).
    "capex": [(year, -20.0) for year in range(2020, 2025)],
    "sbc": [(year, 0.0) for year in range(2020, 2025)],
    "revenue": [(year, 100.0) for year in range(2020, 2025)],
}


def test_negated_capex_is_added_to_owner_earnings():
    """Owner earnings = CFO - |capex| regardless of the filer's sign convention.

    Mechanism: ``owner_earnings = latest_cfo - norm_capex - sbc + interest`` with
    ``norm_capex`` taken raw from the series. The same file's
    ``_local_v2_facts_row`` already uses ``abs(capex)`` for its FCF — the two
    computations disagree on the same input.

    Observed: CFO 100, capex -20 every year -> owner_earnings_latest 120.0,
    normalized_capex -20.0, no flag. Correct: 80.0 (capex 20.0).
    """
    result = _compute_owner_earnings(NEGATED_CAPEX_FACTS)
    assert result["status"] == "OK"
    assert result["normalized_capex"] == 20.0
    assert result["owner_earnings_latest"] == 80.0


def test_capex_spike_cap_inverts_on_negated_series():
    """The 2.0x spike cap must act on capex MAGNITUDE.

    Mechanism: ``threshold = 2.0 * mean`` and ``v > threshold`` — on a negative
    series the threshold is more negative than every non-spike year, so the
    non-spike years (-20 > -72) are "capped" DOWN to the mean (-36) while the
    spike (-100) is untouched.

    Observed: [-100, -20, -20, -20, -20] -> -48.8. The positive mirror
    [100, 20, 20, 20, 20] -> 23.2. Correct: 23.2 for both.
    """
    negated = [(2024, -100.0), (2023, -20.0), (2022, -20.0), (2021, -20.0), (2020, -20.0)]
    positive = [(year, -value) for year, value in negated]
    assert _normalize_capex(positive) == 23.2
    assert _normalize_capex(negated) == 23.2


# ── _margin_of_safety_scorecard ─────────────────────────────────────────────


def test_weak_moat_margin_of_safety_zone_never_gets_deep_discount_context():
    """A weak-moat name in the MARGIN_OF_SAFETY zone must carry the caution.

    Mechanism: ``_signal_for_ivs`` returns only INSUFFICIENT_DATA / DEEP_VALUE /
    UNDERVALUED / OVERVALUED / FAIRLY_VALUED, but the third moat-coupling branch
    tests ``legacy_signal == "MARGIN_OF_SAFETY"`` — a pricing-ZONE name — so
    ``MOS_REQUIRES_DEEP_DISCOUNT`` can never be emitted.

    Observed: EPV 130, DCF 110, price 100 (zone MARGIN_OF_SAFETY, legacy
    FAIRLY_VALUED), moat WEAK_MOAT -> signal_context "FAIRLY_VALUED".
    Correct: "MOS_REQUIRES_DEEP_DISCOUNT".
    """
    methods = {
        "dcf": {"status": "OK", "base": 110.0},
        "epv": {"status": "OK", "value_per_share": 130.0, "avg_operating_income": 50.0},
        "graham": {"status": "METHOD_INSUFFICIENT_DATA"},
        "ncav": {"status": "METHOD_INSUFFICIENT_DATA"},
    }
    quality_ctx = {
        "earnings_quality": "LOW",
        "revenue_trend_class": "FLAT",
        "epv_quality": "STABLE",
        "allocation_grade": "C",
    }
    card = _margin_of_safety_scorecard(
        methods,
        100.0,
        shares=10.0,
        net_debt=0.0,
        revenue_latest=500.0,
        quality_ctx=quality_ctx,
    )
    assert card["pricing_zone"] == "MARGIN_OF_SAFETY"
    assert card["legacy_signal"] == "FAIRLY_VALUED"
    assert card["moat_strength"]["moat_class"] == "WEAK_MOAT"
    assert card["signal_context"] == "MOS_REQUIRES_DEEP_DISCOUNT"


# ── _compute_quality_wacc ───────────────────────────────────────────────────

WACC_YEARS = [2022, 2023, 2024]


def _levered_facts(operating_income: float) -> dict[str, list[tuple[int, float]]]:
    """Net debt 1000 (debt 1100, cash 100) against the given operating income."""
    return {
        "revenue": [(year, 1000.0) for year in WACC_YEARS],
        "gross_profit": [(year, 300.0) for year in WACC_YEARS],
        "operating_income": [(year, operating_income) for year in WACC_YEARS],
        "net_income": [(year, 100.0) for year in WACC_YEARS],
        "cfo": [(year, 100.0) for year in WACC_YEARS],
        "total_debt": [(year, 1100.0) for year in WACC_YEARS],
        "cash": [(year, 100.0) for year in WACC_YEARS],
    }


def _fired(result: dict) -> dict[str, float | None]:
    return {r["code"]: r["metric_value"] for r in result["rule_evaluations"] if r["fired"]}


def test_levered_profitable_issuer_pays_the_leverage_premium():
    """Control: net debt 20x EBITDA fires LEVERAGE_RISK (+1pp) on today's code."""
    result = _compute_quality_wacc(_levered_facts(50.0))
    assert _fired(result).get("LEVERAGE_RISK") == 20.0
    assert result["adjusted_wacc"] == 0.115


def test_leverage_rule_fails_open_when_ebitda_is_not_positive():
    """Same net debt with worse earnings must never earn a lower discount rate.

    Mechanism: ``_compute_quality_wacc`` evaluates LEVERAGE_RISK on
    ``_latest_net_debt_to_ebitda_proxy``, which returns ``None`` for
    ``ebitda <= 0``; the rule treats a missing metric as "did not fire". The
    interest-coverage rules cannot catch it either: they require an
    ``interest_expense`` series, which the fact set (like most EDGAR pulls in
    the store) does not carry.

    Observed (net debt 1000): operating income +50 -> LEVERAGE_RISK fired at
    20.0x, adjusted WACC 0.115; operating income -50 -> LEVERAGE_RISK metric
    None, not fired, adjusted WACC 0.105; operating income 0 -> same 0.105.
    Correct: the loss-making twin's WACC is at least the profitable twin's
    (net debt against non-positive EBITDA is the most levered case, not the
    least).
    """
    profitable = _compute_quality_wacc(_levered_facts(50.0))
    loss_making = _compute_quality_wacc(_levered_facts(-50.0))
    breakeven = _compute_quality_wacc(_levered_facts(0.0))
    assert profitable["adjusted_wacc"] == 0.115
    assert loss_making["adjusted_wacc"] >= profitable["adjusted_wacc"]
    assert breakeven["adjusted_wacc"] >= profitable["adjusted_wacc"]
