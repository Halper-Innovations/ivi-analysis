"""Known defects in app/valuation/pre_valuation_gate.py, pinned as strict expected failures.

Every test asserts the correct answer and currently fails; strict ``xfail`` keeps the
suite green until the defect is fixed, then flags the marker for removal.
"""

from __future__ import annotations

from app.valuation.accounting_quality import compute_accounting_quality
from app.valuation.pre_valuation_gate import compute_quality_context

YEARS = list(range(2020, 2025))

# Egregious accruals: net income 100 every year backed by CFO 20 and FCF 5.
# Fed straight to the accounting-quality module this is LOW_ACCOUNTING_QUALITY
# (cfo/ni 0.20 < 0.75, fcf/ni 0.05 < 0.50 -> headwind strength 7, support 0).
ACCRUAL_HEAVY_FACTS: dict[str, list[tuple[int, float]]] = {
    "revenue": [(year, 1000.0) for year in YEARS],
    "operating_income": [(year, 120.0) for year in YEARS],
    "net_income": [(year, 100.0) for year in YEARS],
    "cfo": [(year, 20.0) for year in YEARS],
    "capex": [(year, 15.0) for year in YEARS],
    "sbc": [(year, 2.0) for year in YEARS],
    "total_assets": [(year, 800.0) for year in YEARS],
    "accounts_receivable": [(year, 100.0 * 1.6**i) for i, year in enumerate(YEARS)],
    "inventory": [(year, 50.0) for year in YEARS],
    "total_debt": [(year, 100.0) for year in YEARS],
    "cash": [(year, 50.0) for year in YEARS],
    "shares_outstanding": [(year, 10.0) for year in YEARS],
}


def _module_verdict_for_the_same_facts() -> str:
    rows = [
        {
            "year": year,
            "net_income": 100.0,
            "cfo": 20.0,
            "fcf": 5.0,
            "revenue": 1000.0,
            "capex": 15.0,
        }
        for year in YEARS
    ]
    payload = compute_accounting_quality(
        "TEST",
        "2025-06-30",
        fundamentals={"ticker": "TEST", "rows": rows},
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    return str(payload["accounting_quality_class"])


def test_module_alone_calls_the_fixture_low_quality():
    """Control: the accounting-quality module classifies this company LOW."""
    assert _module_verdict_for_the_same_facts() == "LOW_ACCOUNTING_QUALITY"


def test_gate_discards_the_facts_it_was_handed_for_accounting_quality():
    """The gate holds every series the module needs and passes none of them.

    Mechanism: ``compute_quality_context`` receives ``facts`` (net_income, cfo,
    capex, revenue, total_assets, receivables, inventory, sbc ...) and calls
    ``compute_accounting_quality(ticker, as_of_date, cfg=cfg)`` — no
    ``fundamentals`` — so the module returns ACCOUNTING_QUALITY_UNKNOWN, which
    ``_AQ_MAP`` turns into "MODERATE". Every branch keyed on
    ``earnings_quality == "LOW"`` (ACCOUNTING_QUALITY_HEADWIND, the
    LOW_QUALITY_HIGH_LEVERAGE block, the LOW_EARNINGS_QUALITY adjust,
    CONFIDENCE_LOW) is unreachable on the valuation path, and the moat score
    always receives MODERATE's +1.

    Observed: earnings_quality "MODERATE", headwinds without
    ACCOUNTING_QUALITY_HEADWIND. Correct: "LOW" (what the module says when it
    is given the facts).
    """
    ctx = compute_quality_context(
        "TEST",
        "2025-06-30",
        facts=ACCRUAL_HEAVY_FACTS,
        facts_row=None,
    )
    assert ctx["earnings_quality"] == "LOW"
    assert "ACCOUNTING_QUALITY_HEADWIND" in ctx["valuation_headwinds"]


# Net debt 1,000 against CFO 100 (10x) with a 9% cash cushion. Fed straight
# to balance_sheet_stress this is HIGH_BALANCE_SHEET_STRESS (HIGH_NET_DEBT_TO_CFO,
# WEAK_CASH_CUSHION, REFINANCING_DEPENDENCE, CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE).
OVER_LEVERED_FACTS: dict[str, list[tuple[int, float]]] = {
    **ACCRUAL_HEAVY_FACTS,
    "cfo": [(year, 100.0) for year in YEARS],
    "total_debt": [(year, 1100.0) for year in YEARS],
    "cash": [(year, 100.0) for year in YEARS],
}


# The same balance sheet against the accrual-heavy cash flow: net debt 1,000
# against CFO 20 (50x) with net income 100. This is the pair the gate's
# LOW_QUALITY_HIGH_LEVERAGE block was written for.
ACCRUAL_HEAVY_AND_LEVERED_FACTS: dict[str, list[tuple[int, float]]] = {
    **ACCRUAL_HEAVY_FACTS,
    "total_debt": [(year, 1100.0) for year in YEARS],
    "cash": [(year, 100.0) for year in YEARS],
}


def test_module_alone_calls_the_levered_fixture_high_stress():
    """Control: the balance-sheet-stress module classifies this company HIGH."""
    from app.valuation.balance_sheet_stress import compute_balance_sheet_stress

    rows = [
        {
            "year": year,
            "net_debt": 1000.0,
            "cfo": 100.0,
            "fcf": 80.0,
            "total_debt": 1100.0,
            "cash": 100.0,
        }
        for year in YEARS
    ]
    payload = compute_balance_sheet_stress(
        "TEST",
        "2025-06-30",
        fundamentals={"ticker": "TEST", "rows": rows},
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    assert payload["balance_sheet_stress_class"] == "HIGH_BALANCE_SHEET_STRESS"


def test_gate_discards_the_facts_it_was_handed_for_leverage_stress():
    """Sibling of the accounting-quality call: ``compute_balance_sheet_stress(
    ticker, as_of_date, cfg=cfg)`` receives no debt, cash or CFO series, returns
    BALANCE_SHEET_STRESS_UNKNOWN, and ``_BS_MAP`` renders that "MODERATE".
    (``compute_capital_allocation_discipline`` is called the same way and pins
    ``allocation_grade`` to "C".)

    Observed: leverage_stress "MODERATE" for a 10x-levered issuer. Correct:
    "HIGH", with LEVERAGE_STRESS_HEADWIND in the headwinds.
    """
    ctx = compute_quality_context(
        "TEST",
        "2025-06-30",
        facts=OVER_LEVERED_FACTS,
        facts_row=None,
    )
    assert ctx["leverage_stress"] == "HIGH"
    assert "LEVERAGE_STRESS_HEADWIND" in ctx["valuation_headwinds"]


def test_low_quality_high_leverage_block_is_unreachable():
    """An accrual-heavy, heavily levered issuer is the gate's own textbook BLOCK.

    Observed: gate_action "PROCEED", gate_reason_codes []. Correct: "BLOCK"
    with LOW_QUALITY_HIGH_LEVERAGE.

    Fixture corrected 2026-09-02: this case needs BOTH inputs, and
    OVER_LEVERED_FACTS raises cash flow to 100 against net income 100 to make
    the leverage point, which is a 1.0x cash conversion — healthy earnings
    quality, not accrual-heavy. Fed the facts, that fixture reads HIGH quality
    and HIGH leverage and correctly does not trip a block that asks for LOW
    quality. The company this test describes keeps the accrual-heavy cash flow
    AND the debt; the assertion is unchanged.
    """
    ctx = compute_quality_context(
        "TEST",
        "2025-06-30",
        facts=ACCRUAL_HEAVY_AND_LEVERED_FACTS,
        facts_row=None,
    )
    assert ctx["gate_action"] == "BLOCK"
    assert "LOW_QUALITY_HIGH_LEVERAGE" in ctx["gate_reason_codes"]


# ── ZERO_OWNER_EARNINGS block: sign-blind on a filer-negated capex series ─────


def _capex_starved_facts(capex_sign: float) -> dict[str, list[tuple[int, float]]]:
    """CFO 50 against capex 200 every year: owner earnings deeply negative."""
    return {
        "revenue": [(year, 1000.0) for year in YEARS],
        "operating_income": [(year, 10.0) for year in YEARS],
        "net_income": [(year, 5.0) for year in YEARS],
        "cfo": [(year, 50.0) for year in YEARS],
        "capex": [(year, capex_sign * 200.0) for year in YEARS],
        "sbc": [(year, 10.0) for year in YEARS],
        "total_assets": [(year, 800.0) for year in YEARS],
        "total_debt": [(year, 100.0) for year in YEARS],
        "cash": [(year, 50.0) for year in YEARS],
        "shares_outstanding": [(year, 10.0) for year in YEARS],
    }


def test_positive_capex_starved_issuer_is_blocked_for_zero_owner_earnings():
    """Control: with capex reported positive the block fires."""
    ctx = compute_quality_context(
        "TEST", "2025-06-30", facts=_capex_starved_facts(1.0), facts_row=None
    )
    assert ctx["gate_action"] == "BLOCK"
    assert "ZERO_OWNER_EARNINGS" in ctx["gate_reason_codes"]


def test_zero_owner_earnings_block_never_fires_for_negated_capex():
    """The same issuer with capex stored as -200 must be blocked the same way.

    Mechanism: ``cfo_val < (maintenance_capex_pct * capex_val + sbc_val)`` —
    with capex -200 the right-hand side is -190, so CFO 50 passes for every
    year and ``negative_oe_years`` stays 0. The companion of the writer's
    negated-capex defects in the valuation writer: the same
    population (negative-capex FY rows) gets inflated owner earnings from the writer AND sails through
    the gate's owner-earnings block.

    Observed: gate_action "PROCEED", gate_reason_codes [], epv_adjustment
    "NONE". Correct: "BLOCK" with ZERO_OWNER_EARNINGS (as for +200).
    """
    ctx = compute_quality_context(
        "TEST", "2025-06-30", facts=_capex_starved_facts(-1.0), facts_row=None
    )
    assert ctx["gate_action"] == "BLOCK"
    assert "ZERO_OWNER_EARNINGS" in ctx["gate_reason_codes"]


# ── Corollary of the starved calls: STRONG_MOAT / PREMIUM_JUSTIFIED unreachable


_GROWTH = 1.06
PRISTINE_YEARS = list(range(2018, 2025))
# Debt-free, 6%/yr grower converting 120% of net income to cash: the
# accounting-quality module's own HIGH (support strength 6, headwinds 0).
PRISTINE_FACTS: dict[str, list[tuple[int, float]]] = {
    "revenue": [(y, 1000.0 * _GROWTH**i) for i, y in enumerate(PRISTINE_YEARS)],
    "operating_income": [(y, 150.0 * _GROWTH**i) for i, y in enumerate(PRISTINE_YEARS)],
    "net_income": [(y, 100.0 * _GROWTH**i) for i, y in enumerate(PRISTINE_YEARS)],
    "cfo": [(y, 120.0 * _GROWTH**i) for i, y in enumerate(PRISTINE_YEARS)],
    "capex": [(y, 20.0 * _GROWTH**i) for i, y in enumerate(PRISTINE_YEARS)],
    "sbc": [(y, 10.0 * _GROWTH**i) for i, y in enumerate(PRISTINE_YEARS)],
    "total_assets": [(y, 800.0 * _GROWTH**i) for i, y in enumerate(PRISTINE_YEARS)],
    "accounts_receivable": [(y, 100.0 * _GROWTH**i) for i, y in enumerate(PRISTINE_YEARS)],
    "inventory": [(y, 50.0 * _GROWTH**i) for i, y in enumerate(PRISTINE_YEARS)],
    "total_debt": [(y, 0.0) for y in PRISTINE_YEARS],
    "cash": [(y, 300.0) for y in PRISTINE_YEARS],
    "shares_outstanding": [(y, 10.0) for y in PRISTINE_YEARS],
}


def test_module_alone_calls_the_pristine_fixture_high_quality():
    """Control: handed the same facts, the accounting-quality module says HIGH."""
    rows = [
        {
            "year": y,
            "revenue": 1000.0 * _GROWTH**i,
            "net_income": 100.0 * _GROWTH**i,
            "cfo": 120.0 * _GROWTH**i,
            "fcf": 100.0 * _GROWTH**i,
            "capex": 20.0 * _GROWTH**i,
            "sbc_total": 10.0 * _GROWTH**i,
            "accounts_receivable": 100.0 * _GROWTH**i,
            "total_assets": 800.0 * _GROWTH**i,
        }
        for i, y in enumerate(PRISTINE_YEARS)
    ]
    payload = compute_accounting_quality(
        "TEST",
        "2025-06-30",
        fundamentals={"ticker": "TEST", "rows": rows},
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    assert payload["accounting_quality_class"] == "HIGH_ACCOUNTING_QUALITY"


def test_strong_moat_and_premium_justified_are_unreachable():
    """A debt-free, growing, cash-converting issuer priced above its EPV and
    DCF must read STRONG_MOAT / PREMIUM_JUSTIFIED, not a plain OVERVALUED.

    Corollary of the starved accounting-quality call (same root cause as
    ``test_defect_gate_discards_the_facts_it_was_handed_for_accounting_quality``)
    with its own observable: ``_classify_moat_strength`` on the valuation path
    receives earnings_quality / revenue_trend_class / epv_quality /
    allocation_grade only; MODERATE (+1) + GROWING (+2) + STABLE (+2) + C (0)
    = 5 < 6, so no issuer can be STRONG_MOAT and ``signal_context`` can never be
    PREMIUM_JUSTIFIED.

    Observed: earnings_quality "MODERATE", moat_score 5, MODERATE_MOAT,
    signal_context "OVERVALUED". Correct: HIGH -> score 6 -> STRONG_MOAT ->
    "PREMIUM_JUSTIFIED".
    """
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    ctx = compute_quality_context("TEST", "2025-06-30", facts=PRISTINE_FACTS, facts_row=None)
    assert ctx["revenue_trend_class"] == "GROWING"
    assert ctx["epv_quality"] == "STABLE"
    methods = {
        "dcf": {"status": "OK", "base": 60.0},
        "epv": {"status": "OK", "value_per_share": 50.0, "avg_operating_income": 150.0},
        "graham": {"status": "METHOD_INSUFFICIENT_DATA"},
        "ncav": {"status": "METHOD_INSUFFICIENT_DATA"},
    }
    card = _margin_of_safety_scorecard(
        methods, 100.0, shares=10.0, net_debt=-300.0, revenue_latest=1500.0, quality_ctx=ctx
    )
    assert card["legacy_signal"] == "OVERVALUED"
    assert card["moat_strength"]["moat_class"] == "STRONG_MOAT"
    assert card["signal_context"] == "PREMIUM_JUSTIFIED"
