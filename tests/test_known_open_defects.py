"""Known open valuation defects, pinned as strict expected failures.

Each test below asserts the right answer and is a strict expected failure: it fails
on the current code for the reason in its docstring and turns red-for-the-right-reason
the moment a fix lands (remove the marker then). None of these is a wrong unit, a
wrong sign, an inverted label, a dropped term, or an unreachable guard with one
obvious repair; each is a methodology choice that was deliberately not changed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.valuation.balance_sheet_stress import compute_balance_sheet_stress
from app.valuation.conviction import compute_conviction
from app.valuation.cyclical_normalization import compute_cyclical_normalization
from app.valuation.method_tension import analyze_method_tensions
from app.valuation.nonrecurring_filter import detect_nonrecurring_items
from app.valuation.returns_persistence import compute_returns_persistence
from app.valuation.sbc_trajectory import compute_sbc_trajectory
from app.valuation.share_count_stability import select_stable_shares
from app.valuation.valuation_writer import _compute_quality_wacc

XFAIL = pytest.mark.xfail(strict=True, reason="known open defect; see the docstring")


def test_defect_cash_conversion_averages_ratios_so_one_breakeven_year_dominates():
    """valuation_writer.py:943 averages per-year CFO/net-income ratios: net income
    100, 100, 1 against CFO 100 each year gives 34.0x and fires
    STRONG_CASH_CONVERSION (−0.5pp of WACC). The ratio of sums is 1.49x, below the
    1.5x rule, so the rule must not fire (reproduced)."""
    facts = {
        "cfo": [(2025, 100.0), (2024, 100.0), (2023, 100.0)],
        "net_income": [(2025, 1.0), (2024, 100.0), (2023, 100.0)],
        "revenue": [(2025, 1000.0), (2024, 1000.0), (2023, 1000.0)],
    }
    out = _compute_quality_wacc(facts)
    assert "STRONG_CASH_CONVERSION" not in [a["code"] for a in out["adjustments"]]
    assert out["adjusted_wacc"] == 0.105


def test_defect_two_methods_with_no_price_read_as_full_agreement():
    """method_tension.py:153-155 and 171-174: with no price every discount is None,
    the consensus is "FAIR", ``methods_agree`` is True and the strength is 0; the
    conviction score then awards 25 of 25 for agreement that was never measured."""
    out = analyze_method_tensions(
        dcf_value=100.0,
        epv_value=50.0,
        graham_value=None,
        ncav_value=None,
        current_price=None,
        revenue_cagr_5y=0.05,
        wacc=0.10,
        terminal_growth=0.02,
    )
    assert out["methods_agree"] is False


def test_defect_a_blocked_gate_still_yields_high_conviction():
    """conviction.py:71-72 scores a BLOCKED gate 5 points (an unknown gate scores 0).
    A report whose gate said BLOCK, with agreeing methods, full coverage and every
    hypothesis resolved, scores 80 and is labelled HIGH. A blocked name must not read
    as high conviction."""
    thesis = SimpleNamespace(
        average_coverage=1.0,
        hypotheses_confirmed=4,
        hypotheses_contradicted=0,
        hypotheses_partially_confirmed=0,
        hypotheses_inconclusive=0,
        hypotheses_unclassified=0,
    )
    report = SimpleNamespace(
        status="OK",
        method_count=3,
        methods_agree=True,
        consensus_strength=3,
        investigation_ran=True,
        thesis=thesis,
        gate_action="BLOCK",
        tension_type="NONE",
    )
    out = compute_conviction(report)
    assert out.conviction_class != "HIGH"


def test_defect_returns_persistence_is_high_with_no_return_on_capital_at_all():
    """returns_persistence.py:390-402: the evidence test is satisfied by intangible
    sub-scores alone, so a name with no ROIC, ROE or ROA and facts marked missing
    reads HIGH_RETURNS_PERSISTENCE."""
    out = compute_returns_persistence(
        "TST",
        "2026-01-01",
        fundamentals={"rows": [{"year": 2024, "revenue": 100.0}, {"year": 2025, "revenue": 105.0}]},
        owner_quality_payload={},
        intangible_payload={
            "gross_margin_durability_score": 4.0,
            "cycle_resilience_score": 4.0,
            "balance_sheet_optionality_score": 4.0,
            "owner_value_capture_score": 3.0,
        },
        reinvestment_efficiency_payload={"reinvestment_efficiency_class": "HIGH_REINVESTMENT_EFFICIENCY"},
        capital_allocation_discipline_payload={"capital_allocation_class": "OWNER_FRIENDLY_DISCIPLINED"},
        revenue_dependence_payload={},
        roic_proxy="UNKNOWN",
        roe_proxy="UNKNOWN",
        roa_proxy="UNKNOWN",
        return_on_retained_earnings="UNKNOWN",
        revenue_cagr_proxy=0.05,
        invested_capital_cagr_proxy=0.05,
        price_status="OK",
        facts_status="MISSING",
        shares_status="OK",
        fcf_status="OK",
    )
    assert out["returns_persistence_class"] != "HIGH_RETURNS_PERSISTENCE"


def test_defect_cyclicality_ignores_the_loss_years():
    """cyclical_normalization.py:102-108 and 178: the coefficient of variation is
    taken over POSITIVE values only, so 200, −100, 200, −100, 200 reads
    LOW_CYCLICALITY with a "conservative" denominator of 200 (the five-year mean is
    80). The loss years are the swings the measure exists to see."""
    series = [{"year": y, "value": v} for y, v in [(2021, 200.0), (2022, -100.0), (2023, 200.0), (2024, -100.0), (2025, 200.0)]]
    out = compute_cyclical_normalization("TST", "2026-01-01", owner_earnings_series=series)
    assert out["cyclical_profile_class"] != "LOW_CYCLICALITY"


def test_defect_sbc_trend_is_latest_minus_the_oldest_year_on_file():
    """sbc_trajectory.py:49-57: the trend is the latest ratio minus the OLDEST ratio in
    the whole history. Three flat recent years (2.3%, 2.4%, 2.5%) read STABLE alone
    but INCREASING with SBC_ACCELERATING once a 2008 point at 1.0% is on file."""
    facts = {
        "sbc": [(2025, 25.0), (2024, 24.0), (2023, 23.0), (2008, 10.0)],
        "revenue": [(2025, 1000.0), (2024, 1000.0), (2023, 1000.0), (2008, 1000.0)],
        "shares_outstanding": [(2025, 100.0), (2024, 100.0), (2023, 100.0), (2008, 100.0)],
    }
    out = compute_sbc_trajectory(facts)
    assert out["sbc_revenue_trend"] == "STABLE"
    assert out["sbc_flags"] == []


def test_defect_a_pre_tax_charge_is_added_back_to_after_tax_net_income_untaxed():
    """nonrecurring_filter.py:73-77 adds the restructuring charge (pre-tax) to net
    income (after tax): net income 100 plus a 50 charge reads 150. The add-back to an
    after-tax line cannot exceed the charge net of tax."""
    out = detect_nonrecurring_items(
        {
            "net_income": [(2025, 100.0)],
            "operating_income": [(2025, 120.0)],
            "revenue": [(2025, 1000.0)],
            "restructuring_charges": [(2025, 50.0)],
        }
    )
    assert out["adjusted_operating_income"][2025] == 170.0
    assert out["adjusted_net_income"][2025] < 150.0


def test_defect_a_restructuring_charge_from_2009_flags_the_company_today():
    """nonrecurring_filter.py:34-39 has no window: any charge anywhere in the filed
    history sets ``has_nonrecurring`` and the writer raises
    NONRECURRING_ITEMS_HEADWIND on today's valuation (valuation_writer.py:3461-3469)."""
    out = detect_nonrecurring_items(
        {
            "net_income": [(2025, 100.0), (2009, 80.0)],
            "operating_income": [(2025, 120.0), (2009, 90.0)],
            "revenue": [(2025, 1000.0), (2009, 800.0)],
            "restructuring_charges": [(2009, 5.0)],
        }
    )
    assert out["has_nonrecurring"] is False


def test_defect_a_levered_cash_burner_reads_moderate_stress():
    """balance_sheet_stress.py:147-148: a non-positive cash flow makes the leverage
    ratio UNKNOWN, so net debt 500 against cash from operations −50 raises no
    leverage headwind at all and the class is MODERATE. The most dangerous balance
    sheet the module can be shown is the one it cannot grade."""
    out = compute_balance_sheet_stress(
        "TST",
        "2026-01-01",
        fundamentals={
            "rows": [
                {
                    "year": 2025,
                    "net_debt": 500.0,
                    "cfo": -50.0,
                    "owner_earnings": -60.0,
                    "total_debt": 800.0,
                    "cash": 300.0,
                }
            ]
        },
        facts_status="OK",
    )
    assert out["balance_sheet_stress_class"] == "HIGH_BALANCE_SHEET_STRESS"


def test_defect_a_stable_fiscal_year_series_is_kept_over_a_newer_count_that_contradicts_it():
    """share_count_stability.py:63-74: when the fiscal-year series is internally
    stable the corroborating count is never consulted, so a two-for-one split (or
    a large issuance) after the last fiscal-year row leaves the stale count in the
    divisor with no flag. CECO Environmental's stored rows divide by 35.7M against a
    cover-page count of 58.6M dated the same day; Coherent's by 155.8M against
    195.6M; IES and Mueller carry pre-split counts against post-split quotes. Which
    count should divide is a doctrine choice; that the contradiction is silent is
    not. The right answer at minimum names it."""
    shares, flag = select_stable_shares(
        [(2025, 35.666), (2024, 35.0), (2023, 34.5), (2022, 34.0)],
        corroborating_count=58.57,
    )
    assert flag is not None
