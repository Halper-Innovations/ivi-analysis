"""Characterization tests and known-defect pins for the existing EPV implementation.

The engine already ships an earnings power value method: `_epv` in
`app/valuation/valuation_writer.py` (line ~1193), reached from the writer at three
call sites (`epv`, `epv_adjusted`, `epv_cash_adjusted`) and persisted under
`method IN ('dcf', 'epv', 'graham')` in `app/db.py`. No second implementation was
written; this file tests the one that exists.

Every test below PASSES against the current code. Tests named `test_defect_*` lock
in behaviour that is still wrong on Greenwald's terms; each docstring states the
correct answer next to the actual one so the gap is auditable. Every other test
asserts the correct answer.

2026-09-02: the headline entry is CLOSED. `_epv` now applies a normalized MARGIN
to CURRENT revenue, so the grower and the shrinker below are no longer the same
company, and the tests that pinned the levels average now assert Greenwald's
answers.

2026-09-28: six more entries are CLOSED and now assert the right answer: the tax
rate is supplied per issuer (21% statutory fallback, flagged; a rate outside
[0, 0.50] or not a finite number is refused), a zero, negative or non-finite
WACC is refused, NaN/infinite/boolean net debt is refused with its own reason,
any gate directive outside NONE/USE_NORMALIZED/BLOCK fails closed, a
USE_NORMALIZED directive with no normalized figure is refused instead of dropped,
every refusal has one result shape, the margin is applied to the NEWEST revenue
year even when operating income for it is not on file yet, and the result echoes
its tax rate, WACC and years.

2026-09-29: four more are CLOSED. EBIT is adjusted by (D&A - maintenance capex)
when the caller supplies both; the SBC treatment (EXPENSED) is stated on every
result; no earnings power, or an equity value net debt takes to zero or below,
returns EPV_NEGATIVE with no per-share value; and a margin window with missing
years is flagged. The entries that remain `test_defect_*` are the absent
non-recurring-items strip and the absent reproduction value (a missing
feature, not a wrong number: franchise value needs a reproduction-cost
estimate this engine does not build).

Reference method (Greenwald, *Value Investing: From Graham to Buffett and Beyond*):

    normalized EBIT   = mean operating MARGIN over a full cycle x CURRENT revenue
    adjusted EBIT     = normalized EBIT + (D&A - maintenance capex) - non-recurring
    NOPAT             = adjusted EBIT x (1 - normalized tax rate), rate in [0, 1]
    EPV(operations)   = NOPAT / WACC                      (no growth term, by design)
    EPV(equity)       = EPV(operations) + excess assets - debt - other claims
    franchise value   = EPV(equity) - reproduction value  (the useful comparison)
"""

from __future__ import annotations

import inspect
import math

import pytest

from app.valuation.valuation_writer import (
    _MIN_YEARS,
    _TAX_RATE,
    _WACC,
    _epv,
    _normalized_tax_rate,
)

# ── Fixture series ────────────────────────────────────────────────────────────
# `_n_years(facts, "operating_income", n=5)` hands `_epv` a DESCENDING-by-year
# list of (fiscal_year, operating_income) pairs. All series below match that shape.

# Hand-computed known-answer case. Flat-ish operating income, 5 clean years.
KNOWN_ANSWER_OI = [
    (2024, 120.0),
    (2023, 110.0),
    (2022, 100.0),
    (2021, 90.0),
    (2020, 80.0),
]

# Same 10% operating margin every year; revenue 100 -> 500 (grower).
GROWER_OI = [(2024, 50.0), (2023, 40.0), (2022, 30.0), (2021, 20.0), (2020, 10.0)]
GROWER_REVENUE = [(2024, 500.0), (2023, 400.0), (2022, 300.0), (2021, 200.0), (2020, 100.0)]
GROWER_CURRENT_REVENUE = 500.0

# Same 10% operating margin every year; revenue 500 -> 100 (shrinker).
SHRINKER_OI = [(2024, 10.0), (2023, 20.0), (2022, 30.0), (2021, 40.0), (2020, 50.0)]
SHRINKER_REVENUE = [(2024, 100.0), (2023, 200.0), (2022, 300.0), (2021, 400.0), (2020, 500.0)]
SHRINKER_CURRENT_REVENUE = 100.0

FLAT_MARGIN = 0.10

# Revenue held flat at 1,000 against the known-answer income series, so the mean
# MARGIN (0.12, 0.11, 0.10, 0.09, 0.08 -> 0.10) applied to CURRENT revenue is
# exactly the 100.0 the levels average used to give. Every hand-computed literal
# in this file therefore still holds under the margin-normalized method, and any
# number that moves is the method's own doing rather than a changed fixture.
KNOWN_ANSWER_REVENUE = [(year, 1000.0) for year, _ in KNOWN_ANSWER_OI]


def _flat_revenue(series: list[tuple[int, float]], revenue: float = 1000.0):
    """A flat revenue line for a series whose levels answer must be preserved."""
    return [(year, revenue) for year, _ in series]


def _greenwald_value_per_share(
    current_revenue: float,
    mean_margin: float,
    *,
    tax_rate: float,
    wacc: float,
    net_debt: float,
    shares: float,
) -> float:
    """Textbook EPV per share: mean margin on CURRENT revenue, capitalized at WACC."""
    normalized_ebit = current_revenue * mean_margin
    nopat = normalized_ebit * (1.0 - tax_rate)
    return (nopat / wacc - net_debt) / shares


# ── 1. Known-answer case ──────────────────────────────────────────────────────


def test_known_answer_case_exact():
    """Hand-computed, exact literals.

    mean OI      = (120 + 110 + 100 + 90 + 80) / 5      = 100.0
    NOPAT        = 100.0 x (1 - 0.21)                    =  79.0
    EPV(ops)     = 79.0 / 0.10                           = 790.0
    EPV(equity)  = 790.0 - 190.0 net debt                = 600.0
    per share    = 600.0 / 50 shares                     =  12.0
    """
    result = _epv(
        KNOWN_ANSWER_OI,
        190.0,
        50.0,
        revenue_series=KNOWN_ANSWER_REVENUE,
        wacc=0.10,
        tax_rate=0.21,
    )

    assert result["status"] == "OK"
    assert result["avg_operating_income"] == 100.0
    assert result["value_per_share"] == 12.0
    assert result["flags"] == []
    assert result["tax_rate_source"] == "ISSUER"


def test_known_answer_net_cash_raises_equity_value():
    """Negative net debt is net cash and must ADD to equity value.

    EPV(ops) 790.0 - (-10.0) = 800.0; / 50 shares = 16.0.
    """
    result = _epv(KNOWN_ANSWER_OI, -10.0, 50.0, revenue_series=KNOWN_ANSWER_REVENUE, wacc=0.10)

    assert result["value_per_share"] == 16.0


def test_no_growth_term_epv_is_a_pure_perpetuity():
    """EPV must not contain a growth term — it is the zero-growth floor.

    Doubling WACC exactly halves the operations value; a Gordon-style (wacc - g)
    denominator would not. 79.0 / 0.20 = 395.0 -> (395.0 - 190.0) / 50 = 4.1.
    """
    result = _epv(KNOWN_ANSWER_OI, 190.0, 50.0, revenue_series=KNOWN_ANSWER_REVENUE, wacc=0.20)

    assert result["value_per_share"] == 4.1


# ── 2. DEFECT: normalizes EBIT LEVELS, not MARGIN on current revenue ──────────


def test_margin_on_current_revenue_separates_the_grower_from_the_shrinker():
    """THE headline entry, now closed. Fixed 2026-09-02.

    Both companies earn a flat 10% operating margin every year. One grew revenue
    100 -> 500, the other shrank 500 -> 100. Averaging the operating-income
    LEVELS gave both 30.0 and both $2.37/share — the same company, twice.

    Greenwald (mean margin x CURRENT revenue), which is what `_epv` now does:
        grower   0.10 x 500 = 50.0 EBIT -> 50.0 x 0.79 / 0.10 / 100 sh = $3.95
        shrinker 0.10 x 100 = 10.0 EBIT -> 10.0 x 0.79 / 0.10 / 100 sh = $0.79

    The old answer overstated the shrinker exactly 3.0x — $158 of fictitious
    enterprise value on a $79 business, landing precisely on the declining
    businesses an earnings floor exists to protect against.
    """
    grower = _epv(GROWER_OI, 0.0, 100.0, revenue_series=GROWER_REVENUE, wacc=0.10)
    shrinker = _epv(SHRINKER_OI, 0.0, 100.0, revenue_series=SHRINKER_REVENUE, wacc=0.10)

    correct_grower = _greenwald_value_per_share(
        GROWER_CURRENT_REVENUE, FLAT_MARGIN, tax_rate=0.21, wacc=0.10, net_debt=0.0, shares=100.0
    )
    correct_shrinker = _greenwald_value_per_share(
        SHRINKER_CURRENT_REVENUE, FLAT_MARGIN, tax_rate=0.21, wacc=0.10, net_debt=0.0, shares=100.0
    )
    assert correct_grower == 3.95
    assert correct_shrinker == 0.79

    assert grower["value_per_share"] == correct_grower
    assert shrinker["value_per_share"] == correct_shrinker
    assert grower["value_per_share"] != shrinker["value_per_share"]

    # The base capitalized is the normalized margin on current revenue; the mean
    # of the levels is kept beside it, and is the same 30.0 for both companies.
    assert grower["normalized_margin"] == pytest.approx(FLAT_MARGIN, abs=1e-12)
    assert shrinker["normalized_margin"] == pytest.approx(FLAT_MARGIN, abs=1e-12)
    assert grower["current_revenue"] == GROWER_CURRENT_REVENUE
    assert shrinker["current_revenue"] == SHRINKER_CURRENT_REVENUE
    assert grower["avg_operating_income"] == 50.0
    assert shrinker["avg_operating_income"] == 10.0
    assert grower["avg_operating_income_levels"] == 30.0
    assert shrinker["avg_operating_income_levels"] == 30.0


def test_epv_signature_takes_the_revenue_series_the_normalization_needs():
    """The correct normalization is reachable: revenue is an input. Fixed 2026-09-02."""
    params = set(inspect.signature(_epv).parameters)

    assert params == {
        "operating_income_series",
        "net_debt",
        "shares",
        "revenue_series",
        "wacc",
        "epv_adjustment",
        "normalized_earnings",
        "tax_rate",
        "depreciation_amortization",
        "maintenance_capex",
    }
    assert {p for p in params if "revenue" in p}


def test_margin_is_applied_to_the_newest_revenue_year_not_the_last_paired_one():
    """Fixed 2026-09-28.

    Operating income is on file for 2020-2022 at a flat 10% margin on revenue of
    100; the newest filing reports 2023 revenue of 500 and no operating income
    yet. The margin is the historical part; the revenue is today's:
        0.10 x 500 = 50.0 EBIT -> 50.0 x 0.79 / 0.10 / 100 sh = $3.95
    Applying the margin to 2022's revenue of 100 gave $0.79, a fifth of it.
    """
    oi = [(2020, 10.0), (2021, 10.0), (2022, 10.0)]
    revenue = [(2020, 100.0), (2021, 100.0), (2022, 100.0), (2023, 500.0)]

    result = _epv(oi, 0.0, 100.0, revenue_series=revenue, wacc=0.10, tax_rate=0.21)

    assert result["status"] == "OK"
    assert result["margin_years"] == [2020, 2021, 2022]
    assert result["current_revenue_year"] == 2023
    assert result["current_revenue"] == 500.0
    assert result["avg_operating_income"] == 50.0
    assert result["value_per_share"] == 3.95


def test_non_positive_current_revenue_is_refused():
    """A newest revenue year of zero or less has no earnings power to capitalize."""
    oi = [(2020, 10.0), (2021, 10.0), (2022, 10.0)]
    revenue = [(2020, 100.0), (2021, 100.0), (2022, 100.0), (2023, 0.0)]

    result = _epv(oi, 0.0, 100.0, revenue_series=revenue, wacc=0.10, tax_rate=0.21)

    assert result["status"] == "METHOD_INSUFFICIENT_DATA"
    assert result["flags"] == ["CURRENT_REVENUE_NOT_POSITIVE"]
    assert result["value_per_share"] is None


def test_without_a_paired_revenue_history_the_method_refuses():
    """No margin to normalize, so no earnings power — not a fallback to levels.

    The levels average is what the fix removed; falling back to it whenever
    revenue is thin would keep the wrong answer alive on exactly the issuers
    whose filings are hardest to read.
    """
    no_revenue = _epv(KNOWN_ANSWER_OI, 190.0, 50.0, wacc=0.10)
    one_year = _epv(
        KNOWN_ANSWER_OI, 190.0, 50.0, revenue_series=[(2024, 1000.0)], wacc=0.10
    )
    zero_revenue = _epv(
        KNOWN_ANSWER_OI,
        190.0,
        50.0,
        revenue_series=[(year, 0.0) for year, _ in KNOWN_ANSWER_OI],
        wacc=0.10,
    )

    for result in (no_revenue, one_year, zero_revenue):
        assert result["status"] == "METHOD_INSUFFICIENT_DATA"
        assert result["flags"] == ["INSUFFICIENT_MARGIN_HISTORY"]
        assert result["value_per_share"] is None


# ── 3. D&A vs maintenance capex, and the stated SBC treatment ────────────────


def test_depreciation_over_maintenance_capex_is_added_to_ebit():
    """Fixed 2026-09-29. Greenwald adds back (D&A - maintenance capex).

    GAAP operating income is already net of full book D&A. A company with D&A
    of 20.0 against maintenance capex of 12.0 has normalized EBIT understated
    by 8.0:
        adjusted EBIT = 100.0 + (20.0 - 12.0)      = 108.0
        NOPAT         = 108.0 x 0.79                =  85.32
        EPV(ops)      = 85.32 / 0.10                = 853.2   (was 790.0)
        per share     = (853.2 - 190.0) / 50        =  13.264 (was 12.0)
    The $63.2 of enterprise value the old method left out is now there.
    """
    result = _epv(
        KNOWN_ANSWER_OI,
        190.0,
        50.0,
        revenue_series=KNOWN_ANSWER_REVENUE,
        wacc=0.10,
        tax_rate=0.21,
        depreciation_amortization=20.0,
        maintenance_capex=12.0,
    )

    assert result["status"] == "OK"
    assert result["normalized_ebit"] == 100.0
    assert result["da_maintenance_capex_adjustment"] == 8.0
    assert result["da_maintenance_capex_basis"] == "ADJUSTED"
    assert result["avg_operating_income"] == 108.0
    assert result["value_per_share"] == pytest.approx(13.264, abs=1e-9)
    assert result["epv_operations"] - 790.0 == pytest.approx(63.2, abs=1e-9)
    assert result["flags"] == ["EPV_DA_MAINTENANCE_CAPEX_ADJUSTED"]


def test_maintenance_capex_above_depreciation_lowers_ebit():
    """The adjustment cuts both ways: D&A 10.0 against maintenance capex 30.0
    takes 20.0 off: 80.0 x 0.79 / 0.10 = 632.0 -> (632.0 - 190.0) / 50 = 8.84."""
    result = _epv(
        KNOWN_ANSWER_OI,
        190.0,
        50.0,
        revenue_series=KNOWN_ANSWER_REVENUE,
        wacc=0.10,
        tax_rate=0.21,
        depreciation_amortization=10.0,
        maintenance_capex=30.0,
    )

    assert result["da_maintenance_capex_adjustment"] == -20.0
    assert result["value_per_share"] == pytest.approx(8.84, abs=1e-9)


def test_without_both_figures_no_adjustment_is_made_and_the_result_says_so():
    """Either figure missing: no guess, the basis reads NOT_ADJUSTED."""
    for kwargs in ({}, {"depreciation_amortization": 20.0}, {"maintenance_capex": 12.0}):
        result = _epv(
            KNOWN_ANSWER_OI,
            190.0,
            50.0,
            revenue_series=KNOWN_ANSWER_REVENUE,
            wacc=0.10,
            tax_rate=0.21,
            **kwargs,
        )
        assert result["value_per_share"] == 12.0, kwargs
        assert result["da_maintenance_capex_basis"] == "NOT_ADJUSTED", kwargs
        assert result["da_maintenance_capex_adjustment"] is None, kwargs
        assert result["flags"] == [], kwargs


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf, True])
def test_invalid_depreciation_or_maintenance_capex_is_refused(bad):
    for kwargs in (
        {"depreciation_amortization": bad, "maintenance_capex": 12.0},
        {"depreciation_amortization": 20.0, "maintenance_capex": bad},
    ):
        result = _epv(
            KNOWN_ANSWER_OI,
            190.0,
            50.0,
            revenue_series=KNOWN_ANSWER_REVENUE,
            wacc=0.10,
            tax_rate=0.21,
            **kwargs,
        )
        assert result["status"] == "METHOD_INSUFFICIENT_DATA"
        assert result["flags"] == ["INVALID_DA_MAINTENANCE_CAPEX"]
        assert result["value_per_share"] is None


def test_defect_no_non_recurring_items_strip():
    """Still open: Greenwald also strips non-recurring items from EBIT.

    `nonrecurring_filter` and `nonrecurring_revenue` modules exist in
    `app/valuation/` and are not consulted by `_epv`; the margin window's
    ratio of sums dilutes a one-off but does not remove it.
    """
    params = set(inspect.signature(_epv).parameters)
    assert not {p for p in params if "nonrecurring" in p or "non_recurring" in p}


def test_stock_based_compensation_treatment_is_stated():
    """Fixed 2026-09-29. SBC is EXPENSED, and the result says so.

    GAAP operating income already deducts SBC and the method adds none of it
    back: SBC pays for the labour that produces the earnings, so it is a real
    cost. Before, that was true only by accident of the input and nothing in
    the output recorded it. The writer relabels the R&D-capitalized variant,
    which un-expenses the SBC embedded in R&D.
    """
    result = _epv(
        KNOWN_ANSWER_OI,
        190.0,
        50.0,
        revenue_series=KNOWN_ANSWER_REVENUE,
        wacc=0.10,
        tax_rate=0.21,
    )

    assert result["sbc_treatment"] == "EXPENSED"
    assert result["value_per_share"] == 12.0


# ── 4. Tax rate: supplied per issuer, documented fallback, invalid refused ──


def test_tax_rate_is_supplied_per_issuer():
    """Fixed 2026-09-28. The rate is an input, not a constant.

    Before, `_TAX_RATE = 0.21` was applied to every company. A full-NOL issuer
    paying 0% and a foreign-heavy issuer at 32% now get their own rates:
        0%   ->  100.0 / 0.10          = 1000.0 -> (1000.0 - 190.0) / 50 = 16.2
        32%  ->  100.0 x 0.68 / 0.10   =  680.0 -> ( 680.0 - 190.0) / 50 =  9.8
    """
    zero = _epv(
        KNOWN_ANSWER_OI, 190.0, 50.0, revenue_series=KNOWN_ANSWER_REVENUE, wacc=0.10, tax_rate=0.0
    )
    high = _epv(
        KNOWN_ANSWER_OI, 190.0, 50.0, revenue_series=KNOWN_ANSWER_REVENUE, wacc=0.10, tax_rate=0.32
    )

    assert zero["value_per_share"] == 16.2
    assert zero["tax_rate"] == 0.0
    assert zero["flags"] == []
    assert high["value_per_share"] == pytest.approx(9.8, abs=1e-12)
    assert high["tax_rate"] == 0.32
    assert high["tax_rate_source"] == "ISSUER"


def test_missing_tax_rate_falls_back_to_statutory_and_says_so():
    """The documented fallback is the 21% US federal statutory rate, flagged."""
    result = _epv(KNOWN_ANSWER_OI, 190.0, 50.0, revenue_series=KNOWN_ANSWER_REVENUE, wacc=0.10)

    assert _TAX_RATE == 0.21
    assert result["value_per_share"] == 12.0
    assert result["tax_rate"] == 0.21
    assert result["tax_rate_source"] == "STATUTORY_DEFAULT"
    assert result["flags"] == ["EPV_TAX_RATE_STATUTORY_DEFAULT"]


@pytest.mark.parametrize("bad_rate", [-0.30, -0.01, 0.51, 1.40, math.nan, math.inf, True, "0.21"])
def test_invalid_tax_rates_are_refused_not_clamped(bad_rate):
    """A rate outside [0, 0.50], non-finite, boolean or not a number is refused.

    Clamping would publish a value built on an input known to be wrong; a
    negative rate would flip the sign of NOPAT. The bounds match dcf_lite's.
    """
    result = _epv(
        KNOWN_ANSWER_OI,
        190.0,
        50.0,
        revenue_series=KNOWN_ANSWER_REVENUE,
        wacc=0.10,
        tax_rate=bad_rate,
    )

    assert result["status"] == "METHOD_INSUFFICIENT_DATA"
    assert result["flags"] == ["INVALID_TAX_RATE"]
    assert result["value_per_share"] is None


def test_tax_rate_bounds_are_inclusive_and_match_dcf_lite():
    import app.valuation.valuation_writer as writer
    from app.valuation import dcf_lite

    assert writer._TAX_RATE_FLOOR == 0.0
    assert writer._TAX_RATE_CEILING == 0.50
    if hasattr(dcf_lite, "TAX_RATE_FLOOR"):
        assert dcf_lite.TAX_RATE_FLOOR == writer._TAX_RATE_FLOOR
        assert dcf_lite.TAX_RATE_CEILING == writer._TAX_RATE_CEILING

    ceiling = _epv(
        KNOWN_ANSWER_OI, 190.0, 50.0, revenue_series=KNOWN_ANSWER_REVENUE, wacc=0.10, tax_rate=0.50
    )
    assert ceiling["status"] == "OK"
    assert ceiling["value_per_share"] == 6.2  # (100 x 0.5 / 0.1 - 190) / 50


def test_issuer_tax_rate_is_a_ratio_of_sums_over_paired_years():
    """Total tax / total pre-tax income over the newest five paired years.

    tax 10 + 30 + 20 + 25 + 15 = 100 on pre-tax 100 x 5 = 500 -> 0.20; the 2018
    year is outside the five-year window and does not count.
    """
    facts = {
        "income_tax_expense": [
            (2024, 10.0), (2023, 30.0), (2022, 20.0), (2021, 25.0), (2020, 15.0), (2018, 90.0)
        ],
        "pretax_income": [
            (2024, 100.0), (2023, 100.0), (2022, 100.0), (2021, 100.0), (2020, 100.0),
            (2018, 100.0),
        ],
    }

    out = _normalized_tax_rate(facts)

    assert out["tax_rate"] == 0.2
    assert out["reason_code"] == "OK"
    assert out["years"] == [2020, 2021, 2022, 2023, 2024]


def test_issuer_tax_rate_unavailable_reasons():
    short = _normalized_tax_rate(
        {"income_tax_expense": [(2024, 10.0), (2023, 10.0)], "pretax_income": [(2024, 50.0), (2023, 50.0)]}
    )
    losses = _normalized_tax_rate(
        {
            "income_tax_expense": [(2024, 1.0), (2023, 1.0), (2022, 1.0)],
            "pretax_income": [(2024, -50.0), (2023, 20.0), (2022, 10.0)],
        }
    )
    benefit = _normalized_tax_rate(
        {
            "income_tax_expense": [(2024, -30.0), (2023, 1.0), (2022, 1.0)],
            "pretax_income": [(2024, 50.0), (2023, 50.0), (2022, 50.0)],
        }
    )

    assert short["tax_rate"] is None and short["reason_code"] == "TAX_HISTORY_SHORT"
    assert losses["tax_rate"] is None and losses["reason_code"] == "PRETAX_INCOME_NOT_POSITIVE"
    assert benefit["tax_rate"] is None
    assert benefit["reason_code"] == "EFFECTIVE_TAX_RATE_OUT_OF_RANGE"
    assert benefit["rejected_rate"] == pytest.approx(-28.0 / 150.0, abs=1e-12)


# ── 5. WACC guard: zero, negative and non-finite discount rates are refused ──


@pytest.mark.parametrize("bad_wacc", [0.0, -0.10, math.nan, math.inf, True])
def test_impossible_discount_rate_is_refused_with_a_status(bad_wacc):
    """Fixed 2026-09-28. Before, 0.0 raised ZeroDivisionError and -0.10 returned
    a confident -$19.60 reading EPV_NEGATIVE. Every sibling method returns
    METHOD_INSUFFICIENT_DATA / INVALID_DISCOUNT_ASSUMPTIONS; so does EPV now.
    """
    result = _epv(
        KNOWN_ANSWER_OI,
        190.0,
        50.0,
        revenue_series=KNOWN_ANSWER_REVENUE,
        wacc=bad_wacc,
        tax_rate=0.21,
    )

    assert result["status"] == "METHOD_INSUFFICIENT_DATA"
    assert result["flags"] == ["INVALID_DISCOUNT_ASSUMPTIONS"]
    assert result["value_per_share"] is None


def test_default_wacc_constant_is_ten_percent():
    """Documented default, used when no caller WACC is supplied."""
    assert _WACC == 0.10
    assert _epv(KNOWN_ANSWER_OI, 190.0, 50.0, revenue_series=KNOWN_ANSWER_REVENUE)["value_per_share"] == 12.0


# ── 6. Negative normalized EBIT ──────────────────────────────────────────────


def test_negative_normalized_ebit_is_not_tax_shielded():
    """Losses are not shrunk 21% by a fictitious tax shield.

    mean OI -10.0 -> NOPAT -10.0 (not -7.9) -> -10.0 / 0.10 = -100.0 of
    operations "value" -> -$1.00 a share, kept only as the negative reading.
    """
    losses = [(2024, -10.0), (2023, -10.0), (2022, -10.0)]
    result = _epv(
        losses, 0.0, 100.0, revenue_series=_flat_revenue(losses), wacc=0.10, tax_rate=0.21
    )

    assert result["avg_operating_income"] == -10.0
    assert result["epv_operations"] == -100.0
    assert result["negative_value_per_share"] == -1.0
    assert result["status"] == "EPV_NEGATIVE"
    assert result["flags"] == ["EPV_NO_TAX_SHIELD_ON_LOSSES", "EPV_NO_EARNINGS_POWER"]


def test_no_earnings_power_yields_no_per_share_value():
    """Fixed 2026-09-29. A business with no earnings power gets no EPV.

    Greenwald: with no normalized earnings power, EPV(operations) is not
    defined and the floor falls to asset/reproduction value. Before, the loss
    was capitalized into perpetuity (-10.0 / 0.10 = -$100.0 of "enterprise
    value") and published as value_per_share, which `method_tension.py` and
    `hypothesis_generator.py` read as a number. Now the status is EPV_NEGATIVE
    and value_per_share is None.
    """
    losses = [(2024, -10.0), (2023, -10.0), (2022, -10.0)]
    result = _epv(losses, 0.0, 100.0, revenue_series=_flat_revenue(losses), wacc=0.10)

    assert result["status"] == "EPV_NEGATIVE"
    assert result["value_per_share"] is None
    assert "EPV_NO_EARNINGS_POWER" in result["flags"]


def test_net_cash_beside_a_loss_does_not_make_a_positive_value():
    """Fixed 2026-09-29. Status came from the per-share sign only, so net cash
    masked the loss: -10.0 / 0.10 = -100.0 EV, net cash 200.0 lifted equity
    to +100.0 -> $1.00/share, status OK, for a business with no earnings
    power at all. The cash is not earnings power; the result is EPV_NEGATIVE
    with no value. Its negative reading is the operations value per share,
    -100.0 / 100 = -1.00, not the cash-lifted +1.00.
    """
    losses = [(2024, -10.0), (2023, -10.0), (2022, -10.0)]
    result = _epv(
        losses, -200.0, 100.0, revenue_series=_flat_revenue(losses), wacc=0.10, tax_rate=0.21
    )

    assert result["status"] == "EPV_NEGATIVE"
    assert result["value_per_share"] is None
    assert result["epv_equity"] == 100.0
    assert result["negative_value_per_share"] == -1.0
    assert result["flags"] == ["EPV_NO_TAX_SHIELD_ON_LOSSES", "EPV_NO_EARNINGS_POWER"]


def test_net_debt_above_earnings_power_yields_no_per_share_value():
    """Positive earnings power that net debt more than consumes: 790.0 EV
    against 1,000.0 of net debt is -210.0 of equity. EPV_NEGATIVE, no value;
    the negative reading is -210.0 / 50 = -4.20."""
    result = _epv(
        KNOWN_ANSWER_OI, 1000.0, 50.0, revenue_series=KNOWN_ANSWER_REVENUE, wacc=0.10, tax_rate=0.21
    )

    assert result["status"] == "EPV_NEGATIVE"
    assert result["value_per_share"] is None
    assert result["negative_value_per_share"] == pytest.approx(-4.2, abs=1e-12)
    assert result["flags"] == ["EPV_NET_DEBT_EXCEEDS_EARNINGS_POWER"]


# ── 7. Non-computable results ────────────────────────────────────────────────


def test_insufficient_history_is_a_first_class_result():
    assert _MIN_YEARS == 3
    result = _epv([(2024, 50.0), (2023, 40.0)], 0.0, 100.0, revenue_series=[(2024, 500.0), (2023, 400.0)], wacc=0.10)

    assert result["status"] == "METHOD_INSUFFICIENT_DATA"
    assert result["flags"] == ["INSUFFICIENT_OI_HISTORY"]
    assert result["value_per_share"] is None


def test_empty_series_is_insufficient_history():
    result = _epv([], 0.0, 100.0, revenue_series=[], wacc=0.10)

    assert result["status"] == "METHOD_INSUFFICIENT_DATA"
    assert result["flags"] == ["INSUFFICIENT_OI_HISTORY"]


def test_zero_shares_is_a_first_class_result():
    result = _epv(KNOWN_ANSWER_OI, 190.0, 0.0, revenue_series=KNOWN_ANSWER_REVENUE, wacc=0.10)

    assert result["status"] == "METHOD_INSUFFICIENT_DATA"
    assert result["flags"] == ["SHARES_ZERO"]
    assert result["value_per_share"] is None


def test_unknown_net_debt_is_a_first_class_result():
    result = _epv(KNOWN_ANSWER_OI, None, 50.0, revenue_series=KNOWN_ANSWER_REVENUE, wacc=0.10)

    assert result["status"] == "METHOD_INSUFFICIENT_DATA"
    assert result["flags"] == ["NET_DEBT_UNKNOWN"]
    assert result["value_per_share"] is None


def test_secular_decline_block_short_circuits_before_any_arithmetic():
    result = _epv(KNOWN_ANSWER_OI, 190.0, 50.0, revenue_series=KNOWN_ANSWER_REVENUE, wacc=0.10, epv_adjustment="BLOCK")

    assert result["status"] == "EPV_BLOCKED_SECULAR_DECLINE"
    assert result["flags"] == ["EPV_BLOCKED_SECULAR_DECLINE"]
    assert result["value_per_share"] is None
    assert result["avg_operating_income"] is None


@pytest.mark.parametrize(
    "token", ["Block", "block", "BLOCK_SECULAR_DECLINE", "USE-NORMALIZED", "wat", "", None]
)
def test_unrecognized_adjustment_token_fails_closed(token):
    """Fixed 2026-09-28.

    The gate's vocabulary is NONE / USE_NORMALIZED / BLOCK. Before, any other
    token (a case variant, a renamed directive, a typo) valued the company at
    $12.00 with no flag. A directive the method cannot read is now refused, so
    a typo fails toward no number rather than toward a number.
    """
    result = _epv(
        KNOWN_ANSWER_OI,
        190.0,
        50.0,
        revenue_series=KNOWN_ANSWER_REVENUE,
        wacc=0.10,
        epv_adjustment=token,
        tax_rate=0.21,
    )

    assert result["status"] == "METHOD_INSUFFICIENT_DATA"
    assert result["flags"] == ["UNKNOWN_EPV_ADJUSTMENT"]
    assert result["value_per_share"] is None
    assert result["epv_adjustment"] == str(token)


def test_recognized_adjustment_tokens_still_compute():
    for token in ("NONE", "USE_NORMALIZED"):
        result = _epv(
            KNOWN_ANSWER_OI,
            190.0,
            50.0,
            revenue_series=KNOWN_ANSWER_REVENUE,
            wacc=0.10,
            epv_adjustment=token,
            normalized_earnings=500.0,
            tax_rate=0.21,
        )
        assert result["status"] == "OK", token
        assert result["value_per_share"] == 12.0, token


def test_result_shape_is_uniform_across_refusal_branches():
    """Fixed 2026-09-28. Every refusal carries status, flags, value_per_share
    None and avg_operating_income None, so "not computed" is never confused
    with "not present"."""
    refusals = [
        _epv(KNOWN_ANSWER_OI, 190.0, 50.0, revenue_series=KNOWN_ANSWER_REVENUE, wacc=0.10, epv_adjustment="BLOCK"),
        _epv([(2024, 50.0), (2023, 40.0)], 0.0, 100.0, revenue_series=[(2024, 500.0), (2023, 400.0)], wacc=0.10),
        _epv(KNOWN_ANSWER_OI, 190.0, 0.0, revenue_series=KNOWN_ANSWER_REVENUE, wacc=0.10),
        _epv(KNOWN_ANSWER_OI, None, 50.0, revenue_series=KNOWN_ANSWER_REVENUE, wacc=0.10),
        _epv(KNOWN_ANSWER_OI, 190.0, 50.0, wacc=0.10),
    ]

    for result in refusals:
        assert {"status", "flags", "value_per_share", "avg_operating_income"} <= set(result)
        assert result["value_per_share"] is None
        assert result["avg_operating_income"] is None
        assert len(result["flags"]) == 1


# ── 8. Non-finite and boolean net debt are refused, each with its reason ────


@pytest.mark.parametrize("bad_net_debt", [math.nan, math.inf, -math.inf, True, False])
def test_non_finite_or_boolean_net_debt_is_refused(bad_net_debt):
    """Fixed 2026-09-28.

    Before, NaN net debt came back status OK with a NaN per-share value (the
    status test `value_per_share < 0` is False for NaN), infinity came back as
    -inf, and `True` was valued as one dollar of net debt ($15.78). `dcf_lite`
    already guarded the same input; EPV now refuses it with NET_DEBT_INVALID,
    distinct from the NET_DEBT_UNKNOWN an absent figure gets.
    """
    result = _epv(
        KNOWN_ANSWER_OI,
        bad_net_debt,
        50.0,
        revenue_series=KNOWN_ANSWER_REVENUE,
        wacc=0.10,
        tax_rate=0.21,
    )

    assert result["status"] == "METHOD_INSUFFICIENT_DATA"
    assert result["flags"] == ["NET_DEBT_INVALID"]
    assert result["value_per_share"] is None


@pytest.mark.parametrize("bad_shares", [math.nan, math.inf, True])
def test_non_finite_or_boolean_share_count_is_refused(bad_shares):
    result = _epv(
        KNOWN_ANSWER_OI,
        190.0,
        bad_shares,
        revenue_series=KNOWN_ANSWER_REVENUE,
        wacc=0.10,
        tax_rate=0.21,
    )

    assert result["status"] == "METHOD_INSUFFICIENT_DATA"
    assert result["flags"] == ["SHARES_INVALID"]
    assert result["value_per_share"] is None


# ── 9. Cyclical normalization, including a window with missing years ─────────


def test_normalization_binds_only_downward():
    """USE_NORMALIZED may only LOWER the anchor, never raise it.

    The grower normalizes to 0.10 x 500 = 50.0; a normalized 20.0 binds ->
    20.0 x 0.79 / 0.10 / 100 = $1.58/share, flagged EPV_CYCLICALLY_NORMALIZED.
    """
    result = _epv(
        GROWER_OI,
        0.0,
        100.0,
        revenue_series=GROWER_REVENUE,
        wacc=0.10,
        epv_adjustment="USE_NORMALIZED",
        normalized_earnings=20.0,
        tax_rate=0.21,
    )

    assert result["avg_operating_income"] == 20.0
    assert result["value_per_share"] == 1.58
    assert result["flags"] == ["EPV_CYCLICALLY_NORMALIZED"]


def test_normalization_above_the_normalized_base_does_not_bind():
    result = _epv(
        GROWER_OI,
        0.0,
        100.0,
        revenue_series=GROWER_REVENUE,
        wacc=0.10,
        epv_adjustment="USE_NORMALIZED",
        normalized_earnings=60.0,
        tax_rate=0.21,
    )

    assert result["avg_operating_income"] == 50.0
    assert result["value_per_share"] == 3.95
    assert result["flags"] == ["EPV_NORMALIZATION_NOT_BINDING"]


@pytest.mark.parametrize("missing", [None, math.nan, True])
def test_use_normalized_without_a_normalized_figure_is_refused(missing):
    """Fixed 2026-09-28. The gate asked for normalization and none arrived.

    Before, the directive was dropped with no trace and the result was
    byte-identical to epv_adjustment="NONE". The normalization can only LOWER
    the base, so the un-normalized value is an upper bound on what the gate
    asked for; the method now refuses rather than publish it.
    """
    result = _epv(
        GROWER_OI,
        0.0,
        100.0,
        revenue_series=GROWER_REVENUE,
        wacc=0.10,
        epv_adjustment="USE_NORMALIZED",
        normalized_earnings=missing,
        tax_rate=0.21,
    )

    assert result["status"] == "METHOD_INSUFFICIENT_DATA"
    assert result["flags"] == ["EPV_NORMALIZATION_MISSING"]
    assert result["value_per_share"] is None


def test_window_with_missing_years_is_flagged():
    """Fixed 2026-09-29. A 3-of-5 window was averaged as if contiguous and
    the gap was invisible in the status and flags.

    FY2024 = 50.0, FY2023 and FY2022 not filed, FY2021 = 10.0, FY2020 = 10.0.
    The value is still computed from the years on file (23.333... on flat
    revenue, the same as three consecutive years), but the result now carries
    EPV_MARGIN_WINDOW_GAP and names the missing years, so a reader does not
    mistake a holed window for a continuous one. Conservative call: flag, not
    refuse — the ratio of sums over the filed years is still the best
    through-cycle margin on file, and refusing would drop issuers whose one
    late filing is merely not ingested.
    """
    gappy = [(2024, 50.0), (2021, 10.0), (2020, 10.0)]
    contiguous = [(2024, 50.0), (2023, 10.0), (2022, 10.0)]

    gappy_result = _epv(
        gappy, 0.0, 100.0, revenue_series=_flat_revenue(gappy), wacc=0.10, tax_rate=0.21
    )
    contiguous_result = _epv(
        contiguous, 0.0, 100.0, revenue_series=_flat_revenue(contiguous), wacc=0.10, tax_rate=0.21
    )

    assert gappy_result["avg_operating_income"] == pytest.approx(23.333333333333332, abs=1e-9)
    assert gappy_result["value_per_share"] == pytest.approx(1.843333333333333, abs=1e-9)
    assert gappy_result["status"] == "OK"
    assert gappy_result["flags"] == ["EPV_MARGIN_WINDOW_GAP"]
    assert gappy_result["margin_years"] == [2020, 2021, 2024]
    assert gappy_result["margin_window_missing_years"] == [2022, 2023]
    assert contiguous_result["margin_years"] == [2022, 2023, 2024]
    assert contiguous_result["flags"] == []
    assert contiguous_result["margin_window_missing_years"] == []
    assert gappy_result["value_per_share"] == contiguous_result["value_per_share"]


def test_defect_recency_is_unweighted_so_year_order_is_irrelevant():
    """Reversing the series changes nothing: the mean discards all time structure.

    Combined with the levels-not-margins defect, this is why a collapsing business
    and a compounding one are indistinguishable to `_epv`.
    """
    forward = _epv(GROWER_OI, 0.0, 100.0, revenue_series=GROWER_REVENUE, wacc=0.10)
    reversed_series = _epv(list(reversed(GROWER_OI)), 0.0, 100.0, revenue_series=GROWER_REVENUE, wacc=0.10)

    assert forward == reversed_series


# ── 10. DEFECT: no reproduction value, so franchise value is never reported ──


def test_defect_no_reproduction_value_anywhere_so_franchise_value_is_unavailable():
    """The analytically useful output — EPV vs reproduction value — does not exist.

    Left open on purpose (2026-09-29): this is a missing feature, not a wrong
    number. Nothing the method publishes is mis-stated by its absence.

    Greenwald's point is the COMPARISON: EPV above reproduction value is franchise
    value (a moat earning above the cost of reproducing the assets); EPV below it
    means capital is being destroyed. `_epv` returns a level and nothing else — no
    reproduction value, no book value, no per-share book comparison, and the module
    exposes no such symbol. `app/valuation/method_tension.py` compares EPV to DCF
    instead, which measures the market's growth premium, not franchise value.
    """
    import app.valuation.valuation_writer as writer

    result = _epv(KNOWN_ANSWER_OI, 190.0, 50.0, revenue_series=KNOWN_ANSWER_REVENUE, wacc=0.10)

    assert not [k for k in result if "reproduction" in k or "book" in k or "franchise" in k]
    assert not [n for n in dir(writer) if "reproduction" in n.lower()]
    assert not [n for n in dir(writer) if "franchise" in n.lower()]


def test_output_carries_its_assumptions_and_provenance():
    """Fixed 2026-09-28. The result echoes the tax rate and its source, the WACC,
    the fiscal years paired, the margin taken and the revenue (and its year) it
    was applied to, so the base is reconstructable from the output alone."""
    result = _epv(
        KNOWN_ANSWER_OI,
        190.0,
        50.0,
        revenue_series=KNOWN_ANSWER_REVENUE,
        wacc=0.123,
        tax_rate=0.25,
    )

    assert result["wacc"] == 0.123
    assert result["tax_rate"] == 0.25
    assert result["tax_rate_source"] == "ISSUER"
    assert result["margin_years"] == [2020, 2021, 2022, 2023, 2024]
    assert result["normalized_margin"] == pytest.approx(0.10, abs=1e-12)
    assert result["current_revenue"] == 1000.0
    assert result["current_revenue_year"] == 2024
    assert result["sbc_treatment"] == "EXPENSED"
    assert result["da_maintenance_capex_basis"] == "NOT_ADJUSTED"
    assert result["margin_window_missing_years"] == []
    assert set(result) == {
        "status",
        "value_per_share",
        "negative_value_per_share",
        "avg_operating_income",
        "avg_operating_income_levels",
        "normalized_ebit",
        "da_maintenance_capex_adjustment",
        "da_maintenance_capex_basis",
        "depreciation_amortization",
        "maintenance_capex",
        "sbc_treatment",
        "epv_operations",
        "epv_equity",
        "normalized_margin",
        "current_revenue",
        "current_revenue_year",
        "margin_years",
        "margin_window_missing_years",
        "tax_rate",
        "tax_rate_source",
        "wacc",
        "flags",
    }


# ── 11. Bridge to equity ─────────────────────────────────────────────────────


def test_bridge_deducts_only_the_caller_supplied_net_debt_scalar():
    """`_epv` sees one scalar; the bridge composition lives entirely in the caller.

    The writer builds it as total_debt - total CASH + preferred equity +
    noncontrolling interest (`valuation_writer.py` ~line 2802), flagging
    SENIOR_CLAIMS_DEDUCTED / SENIOR_CLAIMS_UNKNOWN. Two gaps follow from that
    composition: it nets ALL cash rather than EXCESS cash (operating cash the
    business needs to run is credited to shareholders), and it adds no excess
    non-operating assets. Operating lease liabilities are deliberately excluded
    (LEASE_EXCLUDED_POSTLEASE_FLOWS) because the flow base is post-lease.

    `_epv` itself cannot audit any of that: 790.0 - 190.0 is all it knows.
    """
    result = _epv(
        KNOWN_ANSWER_OI,
        190.0,
        50.0,
        revenue_series=KNOWN_ANSWER_REVENUE,
        wacc=0.10,
        tax_rate=0.21,
    )

    assert result["value_per_share"] == 12.0
    assert (790.0 - 190.0) / 50.0 == 12.0
    assert not [f for f in result["flags"] if "CLAIM" in f or "CASH" in f]
