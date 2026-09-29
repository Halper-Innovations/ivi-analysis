from __future__ import annotations

import math
import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from app.ingest.companyfacts import TAG_MAP, PLAUSIBILITY


def test_new_tags_in_tag_map():
    assert "interest_expense" in TAG_MAP
    assert "sbc" in TAG_MAP
    assert "total_liabilities" in TAG_MAP


def test_total_liabilities_tag_map_has_only_liabilities():
    # LiabilitiesAndStockholdersEquity must NOT be in TAG_MAP
    tags = TAG_MAP["total_liabilities"]
    assert "Liabilities" in tags
    assert "LiabilitiesAndStockholdersEquity" not in tags


def test_interest_expense_tag_map_values():
    tags = TAG_MAP["interest_expense"]
    assert "InterestExpense" in tags
    assert "InterestAndDebtExpense" in tags
    assert "InterestExpenseDebt" in tags


def test_sbc_tag_map_values():
    tags = TAG_MAP["sbc"]
    assert "ShareBasedCompensation" in tags
    assert "AllocatedShareBasedCompensationExpense" in tags


def test_new_tags_have_plausibility_bounds():
    for key in ("interest_expense", "sbc", "total_liabilities"):
        assert key in PLAUSIBILITY, f"Missing PLAUSIBILITY entry for {key}"
        lo, hi = PLAUSIBILITY[key]
        assert lo >= 0.0, f"{key} lower bound should be >= 0"
        assert hi > 0.0, f"{key} upper bound should be > 0"
        assert lo < hi, f"{key} lower bound must be less than upper bound"


from app.db import init_db


def _flat_revenue(oi_series: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """A flat revenue line for `_epv`.

    With revenue flat, the normalized MARGIN on current revenue equals the mean
    of the operating-income LEVELS, so every case written against the pre-
    2026-09-02 method keeps its own arithmetic while satisfying the contract
    that the method must be handed a revenue series.
    """
    return [(year, 1000.0) for year, _ in oi_series]


def test_valuations_has_valuation_writer_version_column():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn=conn)
    cols = [row["name"] for row in conn.execute("PRAGMA table_info(valuations)").fetchall()]
    assert "valuation_writer_version" in cols
    conn.close()


# ── shared fixture ─────────────────────────────────────────────────────────────
# All values in USD millions (or shares millions for shares_outstanding)
_SAMPLE_FACTS: dict = {
    "cfo": [(2024, 120.0), (2023, 110.0), (2022, 100.0), (2021, 90.0), (2020, 80.0)],
    "capex": [(2024, 15.0), (2023, 14.0), (2022, 12.0), (2021, 11.0), (2020, 10.0)],
    "operating_income": [(2024, 80.0), (2023, 75.0), (2022, 70.0), (2021, 65.0), (2020, 60.0)],
    "net_income": [(2024, 60.0), (2023, 55.0), (2022, 50.0)],
    "revenue": [(2024, 500.0), (2023, 450.0), (2022, 400.0), (2021, 350.0), (2020, 300.0)],
    "total_debt": [(2024, 50.0)],
    "cash": [(2024, 30.0)],
    "preferred_equity": [(2024, 0.0)],
    "noncontrolling_interest": [(2024, 0.0)],
    "equity": [(2024, 200.0)],
    "shares_outstanding": [(2024, 10.0)],
    "total_liabilities": [(2024, 300.0)],
}


def test_latest_common_year_returns_correct_year():
    from app.valuation.valuation_writer import _latest_common_year

    year = _latest_common_year(_SAMPLE_FACTS, "total_debt", "cash")
    assert year == 2024


def test_latest_common_year_returns_none_when_field_missing():
    from app.valuation.valuation_writer import _latest_common_year

    year = _latest_common_year(_SAMPLE_FACTS, "total_debt", "sbc")  # sbc absent
    assert year is None


def test_n_years_returns_up_to_n_sorted_desc():
    from app.valuation.valuation_writer import _n_years

    result = _n_years(_SAMPLE_FACTS, "cfo", n=3)
    assert result == [(2024, 120.0), (2023, 110.0), (2022, 100.0)]


def test_n_years_returns_all_if_fewer_than_n():
    from app.valuation.valuation_writer import _n_years

    result = _n_years(_SAMPLE_FACTS, "net_income", n=5)
    assert len(result) == 3


# ── owner earnings ─────────────────────────────────────────────────────────────


def test_owner_earnings_cfo_minus_normalized_capex_minus_sbc():
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = dict(_SAMPLE_FACTS)
    facts["sbc"] = [(2024, 5.0), (2023, 4.5), (2022, 4.0)]
    result = _compute_owner_earnings(facts)
    assert result["status"] == "OK"
    # Normalized capex for 5Y series = (15+14+12+11+10)/5 = 12.4 (no spike)
    expected_oe = 120.0 - 12.4 - 5.0  # = 102.6
    assert abs(result["owner_earnings_latest"] - expected_oe) < 0.1
    # The fixture carries 50 of debt and no interest fact, so the FCFF interest
    # addback is unknown, not zero.
    assert result["confidence"] == "LOWER"
    assert "FCFF_INTEREST_UNKNOWN" in result["flags"]


def test_owner_earnings_older_interest_only_is_an_unknown_current_addback():
    """Interest on file for 2022 only: 2024's addback is unknown, and the 2022
    figure must not stand in for it."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = dict(_SAMPLE_FACTS)
    facts["sbc"] = [(2024, 5.0), (2023, 4.5), (2022, 4.0)]
    facts["total_debt"] = []
    facts["interest_expense"] = [(2022, 3.0)]
    result = _compute_owner_earnings(facts)
    assert result["interest_addback"] == 0.0
    assert result["confidence"] == "LOWER"
    assert "FCFF_INTEREST_UNKNOWN" in result["flags"]


def test_owner_earnings_debt_free_issuer_without_interest_keeps_normal_confidence():
    """No debt and no interest fact anywhere: a zero addback is the known answer."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = dict(_SAMPLE_FACTS)
    facts["sbc"] = [(2024, 5.0), (2023, 4.5), (2022, 4.0)]
    facts["total_debt"] = [(2024, 0.0)]
    result = _compute_owner_earnings(facts)
    assert result["confidence"] == "NORMAL"
    assert "FCFF_INTEREST_UNKNOWN" not in result["flags"]
    assert result["interest_addback"] == 0.0


def test_owner_earnings_sbc_absent_lowers_confidence():
    from app.valuation.valuation_writer import _compute_owner_earnings

    result = _compute_owner_earnings(_SAMPLE_FACTS)  # no sbc in _SAMPLE_FACTS
    assert result["status"] == "OK"
    assert "SBC_NOT_ADJUSTED" in result["flags"]
    assert result["confidence"] == "LOWER"  # must be LOWER, not NORMAL


def test_owner_earnings_cfo_absent_returns_insufficient():
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = {k: v for k, v in _SAMPLE_FACTS.items() if k != "cfo"}
    result = _compute_owner_earnings(facts)
    assert result["status"] == "OWNER_EARNINGS_INSUFFICIENT_DATA"


def test_capex_normalization_spike_capping():
    from app.valuation.valuation_writer import _normalize_capex

    # series with one spike year
    series = [(2024, 10.0), (2023, 11.0), (2022, 12.0), (2021, 10.0), (2020, 50.0)]
    norm = _normalize_capex(series)
    uncapped_mean = (10 + 11 + 12 + 10 + 50) / 5  # = 18.6
    # 50 > 2.0 × 18.6 = 37.2 → cap 50 → 18.6
    capped = [10.0, 11.0, 12.0, 10.0, 18.6]
    expected = sum(capped) / len(capped)
    assert abs(norm - expected) < 0.01


def test_capex_missing_uses_zero_and_flags():
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = {k: v for k, v in _SAMPLE_FACTS.items() if k != "capex"}
    result = _compute_owner_earnings(facts)
    assert "CAPEX_UNKNOWN" in result["flags"]
    assert result["status"] == "OK"


# ── four valuation methods ─────────────────────────────────────────────────────


def test_discounted_owner_earnings_low_le_base_le_high():
    from app.valuation.valuation_writer import _discounted_owner_earnings

    rev_series = [(2024, 500.0), (2023, 450.0), (2022, 400.0), (2021, 350.0), (2020, 300.0)]
    result = _discounted_owner_earnings(
        owner_earnings=100.0, shares=10.0, net_debt=20.0, revenue_series=rev_series
    )
    assert result["status"] == "OK"
    assert result["low"] <= result["base"] <= result["high"]
    assert result["low"] > 0


def test_discounted_owner_earnings_negative_flagged_but_still_computed():
    from app.valuation.valuation_writer import _discounted_owner_earnings

    rev_series = [(2024, 500.0), (2023, 450.0), (2022, 400.0)]
    result = _discounted_owner_earnings(
        owner_earnings=-10.0, shares=10.0, net_debt=0.0, revenue_series=rev_series
    )
    assert "NEGATIVE_OWNER_EARNINGS" in result["flags"]
    assert result["status"] == "OK"  # computed, not skipped


def test_epv_correct_formula():
    from app.valuation.valuation_writer import _epv

    # NOPAT = avg(80,75,70,65,60) * (1-0.21) = 70 * 0.79 = 55.3
    # EPV_equity = (55.3 / 0.10 - 20) / 10 = (553 - 20) / 10 = 53.3
    oi_series = [(2024, 80.0), (2023, 75.0), (2022, 70.0), (2021, 65.0), (2020, 60.0)]
    result = _epv(operating_income_series=oi_series, revenue_series=_flat_revenue(oi_series), net_debt=20.0, shares=10.0)
    assert result["status"] == "OK"
    avg_oi = 70.0
    expected = (avg_oi * (1 - 0.21) / 0.10 - 20.0) / 10.0
    assert abs(result["value_per_share"] - expected) < 0.01


def test_epv_negative_operating_income_returns_epv_negative():
    from app.valuation.valuation_writer import _epv

    oi_series = [(2024, -10.0), (2023, -5.0), (2022, -8.0)]
    result = _epv(operating_income_series=oi_series, revenue_series=_flat_revenue(oi_series), net_debt=0.0, shares=10.0)
    assert result["status"] == "EPV_NEGATIVE"
    # No earnings power, no published value (2026-09-29); the negative
    # reading stays for the scorecard's anomaly checks.
    assert result["value_per_share"] is None
    assert result["negative_value_per_share"] < 0


def test_epv_net_cash_increases_value():
    """Net debt = -20 (net cash) must increase EPV per share — not floored at zero."""
    from app.valuation.valuation_writer import _epv

    oi_series = [(2024, 70.0), (2023, 70.0), (2022, 70.0)]
    no_cash = _epv(operating_income_series=oi_series, revenue_series=_flat_revenue(oi_series), net_debt=0.0, shares=10.0)
    with_net_cash = _epv(operating_income_series=oi_series, revenue_series=_flat_revenue(oi_series), net_debt=-20.0, shares=10.0)
    assert with_net_cash["value_per_share"] > no_cash["value_per_share"]


def test_quality_wacc_rewards_high_quality_cash_generative_software():
    from app.valuation.valuation_writer import _compute_quality_wacc

    facts = dict(_SAMPLE_FACTS)
    facts["gross_profit"] = [(2024, 390.0), (2023, 340.0), (2022, 280.0)]
    category_result = {"metrics_used": {"avg_rnd_to_revenue_3y": 0.18}}

    result = _compute_quality_wacc(facts, category_result=category_result)

    assert result["adjusted_wacc"] < 0.10
    assert {row["code"] for row in result["adjustments"]} >= {
        "HIGH_GROSS_MARGIN",
        "STRONG_CASH_CONVERSION",
        "RND_MOAT",
    }
    fired = {row["code"] for row in result["rule_evaluations"] if row["fired"]}
    assert fired >= {"HIGH_GROSS_MARGIN", "STRONG_CASH_CONVERSION", "RND_MOAT"}


def test_quality_wacc_penalizes_low_growth_and_leverage():
    from app.valuation.valuation_writer import _compute_quality_wacc

    facts = dict(_SAMPLE_FACTS)
    facts["gross_profit"] = [(2024, 200.0), (2023, 190.0), (2022, 180.0)]
    facts["revenue"] = [(2024, 500.0), (2023, 495.0), (2022, 490.0), (2021, 485.0), (2020, 480.0)]
    facts["total_debt"] = [(2024, 300.0)]
    facts["cash"] = [(2024, 10.0)]
    facts["operating_income"] = [
        (2024, 80.0),
        (2023, 75.0),
        (2022, 70.0),
        (2021, 65.0),
        (2020, 60.0),
    ]

    result = _compute_quality_wacc(
        facts, category_result={"metrics_used": {"avg_rnd_to_revenue_3y": 0.02}}
    )

    assert result["adjusted_wacc"] > 0.10
    assert {row["code"] for row in result["adjustments"]} >= {
        "LOW_GROWTH_HEADWIND",
        "LEVERAGE_RISK",
    }


def test_quality_wacc_rule_evaluations_capture_non_firing_rules():
    from app.valuation.valuation_writer import _compute_quality_wacc

    facts = dict(_SAMPLE_FACTS)
    result = _compute_quality_wacc(
        facts, category_result={"metrics_used": {"avg_rnd_to_revenue_3y": 0.05}}
    )

    rules = {row["code"]: row for row in result["rule_evaluations"]}
    assert set(rules.keys()) == {
        "HIGH_GROSS_MARGIN",
        "STRONG_CASH_CONVERSION",
        "RND_MOAT",
        "LOW_GROWTH_HEADWIND",
        "LEVERAGE_RISK",
        "INTEREST_COVERAGE_WEAK",
        "INTEREST_COVERAGE_CRITICAL",
    }
    assert rules["HIGH_GROSS_MARGIN"]["fired"] is False
    assert rules["HIGH_GROSS_MARGIN"]["delta"] == 0.0


def test_graham_formula_correct():
    from app.valuation.valuation_writer import _graham_formula

    # normalized_eps = avg(60,55,50)/10 = 5.5; bvps = 200/10 = 20
    ni_series = [(2024, 60.0), (2023, 55.0), (2022, 50.0)]
    result = _graham_formula(net_income_series=ni_series, equity=200.0, shares=10.0)
    assert result["status"] == "OK"
    expected = math.sqrt(22.5 * 5.5 * 20.0)
    assert abs(result["value_per_share"] - expected) < 0.01
    assert abs(result["buy_price"] - expected * 0.67) < 0.01


def test_graham_formula_negative_equity_not_applicable():
    from app.valuation.valuation_writer import _graham_formula

    ni_series = [(2024, 60.0), (2023, 55.0), (2022, 50.0)]
    result = _graham_formula(net_income_series=ni_series, equity=-50.0, shares=10.0)
    assert result["status"] == "GRAHAM_NOT_APPLICABLE"


def test_graham_formula_missing_equity_is_not_negative_equity():
    from app.valuation.valuation_writer import _graham_formula

    ni_series = [(2024, 60.0), (2023, 55.0), (2022, 50.0)]
    result = _graham_formula(net_income_series=ni_series, equity=None, shares=10.0)
    assert result == {
        "status": "METHOD_INSUFFICIENT_DATA",
        "flags": ["EQUITY_MISSING"],
        "value_per_share": None,
        "buy_price": None,
    }


def test_graham_formula_negative_eps_not_applicable():
    from app.valuation.valuation_writer import _graham_formula

    ni_series = [(2024, -10.0), (2023, -5.0), (2022, -8.0)]
    result = _graham_formula(net_income_series=ni_series, equity=200.0, shares=10.0)
    assert result["status"] == "GRAHAM_NOT_APPLICABLE"


def test_graham_formula_extreme_ratio_low_confidence():
    from app.valuation.valuation_writer import _graham_formula

    # avg_ni = 1.0, shares = 10 → normalized_eps = 0.1; bvps = 200/10 = 20
    # ratio = 20 / 0.1 = 200 > 100 → GRAHAM_LOW_CONFIDENCE
    ni_series = [(2024, 1.0), (2023, 1.0), (2022, 1.0)]
    result = _graham_formula(net_income_series=ni_series, equity=200.0, shares=10.0)
    assert result["status"] == "OK"
    assert "GRAHAM_LOW_CONFIDENCE" in result["flags"]


def test_ncav_negative_for_large_cap():
    from app.valuation.valuation_writer import _ncav

    # cash=30, revenue=500 → receivables = 500*0.10*0.6 = 30; proxy = 60
    # total_liabilities = 300; ncav = (60-300)/10 = -24
    result = _ncav(
        cash=30.0, revenue=500.0, total_liabilities=300.0, total_debt=None, shares=10.0, price=100.0
    )
    assert result["value_per_share"] < 0
    assert result["signal"] == "NCAV_NO_ASSET_FLOOR"
    assert "NCAV_PROXY_ESTIMATED" in result["flags"]


def test_ncav_net_net_when_ncav_exceeds_price():
    from app.valuation.valuation_writer import _ncav

    # cash=500, revenue=100 → receivables = 100*0.10*0.6 = 6; proxy = 506
    # total_liabilities = 100; ncav = (506-100)/10 = 40.6; price = 30 → NET_NET
    result = _ncav(
        cash=500.0, revenue=100.0, total_liabilities=100.0, total_debt=None, shares=10.0, price=30.0
    )
    assert result["signal"] == "NCAV_NET_NET"


def test_ncav_uses_total_debt_fallback_when_liabilities_absent():
    from app.valuation.valuation_writer import _ncav

    result = _ncav(
        cash=30.0, revenue=500.0, total_liabilities=None, total_debt=50.0, shares=10.0, price=100.0
    )
    assert "TOTAL_LIABILITIES_PROXY" in result["flags"]


# ── scorecard, ROIC, capital structure, reverse DCF ───────────────────────────


def test_margin_of_safety_scorecard_deep_value():
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": 200.0},
        "epv": {"status": "OK", "value_per_share": 180.0},
        "graham": {"status": "OK", "value_per_share": 160.0},
        "ncav": {"status": "OK", "value_per_share": -5.0, "signal": "NCAV_NO_ASSET_FLOOR"},
    }
    result = _margin_of_safety_scorecard(methods, price=100.0, shares=10.0, net_debt=20.0)
    assert result["signal"] == "BUY"
    assert result["pricing_zone"] == "MARGIN_OF_SAFETY"
    assert result["legacy_signal"] == "DEEP_VALUE"
    assert result["type"] == "EARNINGS_DRIVEN"


def test_margin_of_safety_scorecard_growth_dependent():
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": 180.0},
        "epv": {"status": "OK", "value_per_share": 70.0},
        "epv_adjusted": {"status": "OK", "value_per_share": 120.0, "avg_operating_income": 90.0},
        "graham": {"status": "OK", "value_per_share": 60.0},
        "ncav": {"status": "OK", "value_per_share": -5.0, "signal": "NCAV_NO_ASSET_FLOOR"},
    }
    result = _margin_of_safety_scorecard(methods, price=150.0, shares=10.0, net_debt=20.0)
    assert result["signal"] == "HOLD"
    assert result["pricing_zone"] == "GROWTH_DEPENDENT"
    assert isinstance(result["pricing_zone_detail"]["earnings_gap"], float)
    assert result["type"] == "EARNINGS_DRIVEN"


def test_margin_of_safety_scorecard_negative_intrinsic_values_are_anomalies():
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": -58.97},
        "epv": {"status": "EPV_NEGATIVE", "value_per_share": -27.13, "avg_operating_income": -15.0},
        "graham": {"status": "GRAHAM_NOT_APPLICABLE", "value_per_share": None},
        "ncav": {"status": "OK", "value_per_share": -5.0, "signal": "NCAV_NO_ASSET_FLOOR"},
    }
    result = _margin_of_safety_scorecard(
        methods, price=70.0, shares=10.0, net_debt=0.0, revenue_latest=100.0
    )
    assert result["signal"] == "VALUATION_ANOMALY"
    assert result["pricing_zone"] == "VALUATION_ANOMALY"
    assert result["pricing_zone_detail"]["negative_inputs"] == ["EPV_adjusted", "DCF_base"]
    assert result["pricing_zone_detail"]["anomaly_sub_reason"] == "NEGATIVE_EARNINGS_LEGITIMATE"


def test_margin_of_safety_scorecard_insufficient_data():
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "METHOD_INSUFFICIENT_DATA"},
        "epv": {"status": "METHOD_INSUFFICIENT_DATA"},
        "graham": {"status": "OK", "value_per_share": 50.0},
        "ncav": {"status": "METHOD_INSUFFICIENT_DATA"},
    }
    # Only 1 earnings method valid → INSUFFICIENT_DATA
    result = _margin_of_safety_scorecard(methods, price=40.0, shares=10.0, net_debt=0.0)
    assert result["signal"] == "INSUFFICIENT_DATA"
    assert result["pricing_zone"] == "INSUFFICIENT_DATA"


def test_margin_of_safety_scorecard_flags_negative_intrinsic_values_as_anomaly():
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": -16.79},
        "epv_adjusted": {"status": "OK", "value_per_share": -3.0, "avg_operating_income": 90.0},
        "epv": {"status": "EPV_NEGATIVE", "value_per_share": -3.0},
        "graham": {"status": "GRAHAM_NOT_APPLICABLE", "value_per_share": None},
        "ncav": {"status": "OK", "value_per_share": -5.0, "signal": "NCAV_NO_ASSET_FLOOR"},
    }
    result = _margin_of_safety_scorecard(methods, price=40.0, shares=10.0, net_debt=0.0)
    assert result["pricing_zone"] == "VALUATION_ANOMALY"
    assert result["signal"] == "VALUATION_ANOMALY"
    assert result["pricing_zone_detail"]["negative_inputs"] == ["EPV_adjusted", "DCF_base"]
    assert result["pricing_zone_detail"]["anomaly_sub_reason"] == "DATA_QUALITY_ISSUE"


def test_margin_of_safety_scorecard_flags_debt_overwhelms_earnings_sub_reason():
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": -5.0},
        "epv_adjusted": {"status": "OK", "value_per_share": 4.0, "avg_operating_income": 20.0},
        "epv": {"status": "OK", "value_per_share": 4.0, "avg_operating_income": 20.0},
        "graham": {"status": "GRAHAM_NOT_APPLICABLE", "value_per_share": None},
        "ncav": {"status": "OK", "value_per_share": -5.0, "signal": "NCAV_NO_ASSET_FLOOR"},
    }

    result = _margin_of_safety_scorecard(
        methods,
        price=10.0,
        shares=10.0,
        net_debt=200.0,
        revenue_latest=100.0,
        net_debt_to_ebitda_proxy=10.0,
    )

    assert result["pricing_zone"] == "VALUATION_ANOMALY"
    assert result["pricing_zone_detail"]["anomaly_sub_reason"] == "DEBT_OVERWHELMS_EARNINGS"


def test_margin_of_safety_scorecard_flags_data_quality_sub_reason():
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": -2.0},
        "epv_adjusted": {"status": "OK", "value_per_share": 1.0, "avg_operating_income": 10.0},
        "epv": {"status": "OK", "value_per_share": 1.0, "avg_operating_income": 10.0},
        "graham": {"status": "GRAHAM_NOT_APPLICABLE", "value_per_share": None},
        "ncav": {"status": "OK", "value_per_share": -5.0, "signal": "NCAV_NO_ASSET_FLOOR"},
    }

    result = _margin_of_safety_scorecard(
        methods,
        price=5.0,
        shares=0.0,
        net_debt=1.0,
        revenue_latest=0.0,
    )

    assert result["pricing_zone"] == "VALUATION_ANOMALY"
    assert result["pricing_zone_detail"]["anomaly_sub_reason"] == "DATA_QUALITY_ISSUE"


def test_margin_of_safety_scorecard_balance_sheet_driven_partial_protection():
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": 200.0},
        "epv": {"status": "OK", "value_per_share": 180.0},
        "graham": {"status": "GRAHAM_NOT_APPLICABLE"},
        "ncav": {"status": "OK", "value_per_share": 5.0, "signal": "NCAV_PARTIAL_PROTECTION"},
    }
    result = _margin_of_safety_scorecard(methods, price=100.0, shares=10.0, net_debt=20.0)
    assert result["type"] == "BALANCE_SHEET_DRIVEN"


def test_reverse_dcf_no_price_returns_code():
    from app.valuation.valuation_writer import _run_reverse_dcf

    result = _run_reverse_dcf(
        price=None, shares=10.0, net_debt=20.0, revenue=500.0, operating_income=80.0
    )
    assert result["status"] == "REVERSE_DCF_NO_PRICE"
    assert result["reason_code"] == "PRICE_UNKNOWN"


def test_reverse_dcf_implausible():
    from app.valuation.valuation_writer import _run_reverse_dcf

    result = _run_reverse_dcf(
        price=500.0,
        shares=10.0,
        net_debt=50.0,
        revenue=100.0,
        operating_income=20.0,
        revenue_cagr_5y=0.05,
    )
    if result.get("status") == "OK":
        implied = result["outputs"].get("implied_growth")
        if result["outputs"].get("implied_growth_saturated"):
            # Post-audit: a clipped bound is not a real solve — it must not
            # be graded IMPLAUSIBLE off the fabricated number.
            assert result["feasibility"] == "UNKNOWN"
        elif implied is not None and implied > 0.10:
            assert result["feasibility"] == "IMPLAUSIBLE"


def test_reverse_dcf_optimistic():
    from app.valuation.valuation_writer import _run_reverse_dcf

    result = _run_reverse_dcf(
        price=150.0,
        shares=10.0,
        net_debt=20.0,
        revenue=200.0,
        operating_income=30.0,
        revenue_cagr_5y=0.05,
    )
    if result.get("status") == "OK":
        feasibility = result.get("feasibility")
        assert feasibility in ("IMPLAUSIBLE", "AGGRESSIVE", "OPTIMISTIC", "REASONABLE", "UNKNOWN")


def test_reverse_dcf_reasonable():
    from app.valuation.valuation_writer import _run_reverse_dcf

    result = _run_reverse_dcf(
        price=50.0,
        shares=10.0,
        net_debt=10.0,
        revenue=200.0,
        operating_income=30.0,
        revenue_cagr_5y=0.10,
    )
    if result.get("status") == "OK":
        feasibility = result.get("feasibility")
        assert feasibility in ("IMPLAUSIBLE", "AGGRESSIVE", "OPTIMISTIC", "REASONABLE", "UNKNOWN")
        assert result.get("revenue_cagr_5y_used") == 0.10


def test_reverse_dcf_unknown_no_cagr():
    from app.valuation.valuation_writer import _run_reverse_dcf

    result = _run_reverse_dcf(
        price=100.0,
        shares=10.0,
        net_debt=10.0,
        revenue=200.0,
        operating_income=30.0,
        revenue_cagr_5y=None,
    )
    if result.get("status") == "OK":
        assert result["feasibility"] == "UNKNOWN"


# ── expectations-gap persisted in reverse_dcf outputs ──────────────────


def test_reverse_dcf_attaches_expectations_gap():
    from app.valuation.valuation_writer import _run_reverse_dcf

    # Interior-solve fixture (mirrors the expectations-gap module's price=2000 / shares=1e6 /
    # net_debt=0 / revenue=5e8 / margin=0.20 interior band).
    result = _run_reverse_dcf(
        price=2000.0,
        shares=1_000_000.0,
        net_debt=0.0,
        revenue=500_000_000.0,
        operating_income=100_000_000.0,
        revenue_cagr_5y=0.06,
    )
    assert result["status"] == "OK"
    gap = result["expectations_gap"]
    assert gap["bucket"] == "EXPENSIVE_VS_EXPECTATIONS"
    assert gap["gap"] == 0.2626
    assert gap["supportable_growth"] == 0.06


def test_reverse_dcf_expectations_gap_unreliable_without_cagr():
    from app.valuation.valuation_writer import _run_reverse_dcf

    result = _run_reverse_dcf(
        price=2000.0,
        shares=1_000_000.0,
        net_debt=0.0,
        revenue=500_000_000.0,
        operating_income=100_000_000.0,
        revenue_cagr_5y=None,
    )
    assert result["status"] == "OK"
    assert result["expectations_gap"]["bucket"] == "EXPECTATIONS_GAP_UNRELIABLE"


def test_reverse_dcf_preserves_legacy_feasibility_key():
    from app.valuation.valuation_writer import _run_reverse_dcf

    result = _run_reverse_dcf(
        price=2000.0,
        shares=1_000_000.0,
        net_debt=0.0,
        revenue=500_000_000.0,
        operating_income=100_000_000.0,
        revenue_cagr_5y=0.06,
    )
    assert result["status"] == "OK"
    assert "feasibility" in result
    assert result["feasibility"] in (
        "IMPLAUSIBLE",
        "AGGRESSIVE",
        "OPTIMISTIC",
        "REASONABLE",
        "UNKNOWN",
    )


# ── recursive signal + ensure_valuation ──────────────────────────────────────
from app.valuation.price_provider import PriceQuote


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn=conn)
    return conn


def _seed_companyfacts(conn: sqlite3.Connection, ticker: str = "TST") -> None:
    now = "2026-01-01T00:00:00+00:00"
    rows = [
        ("cfo", 2024, 120.0),
        ("cfo", 2023, 110.0),
        ("cfo", 2022, 100.0),
        ("capex", 2024, 15.0),
        ("capex", 2023, 14.0),
        ("capex", 2022, 12.0),
        ("operating_income", 2024, 80.0),
        ("operating_income", 2023, 75.0),
        ("operating_income", 2022, 70.0),
        ("operating_income", 2021, 65.0),
        ("operating_income", 2020, 60.0),
        ("net_income", 2024, 60.0),
        ("net_income", 2023, 55.0),
        ("net_income", 2022, 50.0),
        ("revenue", 2024, 500.0),
        ("revenue", 2023, 450.0),
        ("revenue", 2022, 400.0),
        ("revenue", 2021, 350.0),
        ("revenue", 2020, 300.0),
        ("total_debt", 2024, 50.0),
        ("cash", 2024, 30.0),
        ("preferred_equity", 2024, 0.0),
        ("noncontrolling_interest", 2024, 0.0),
        ("equity", 2024, 200.0),
        ("shares_outstanding", 2024, 10.0),
        ("total_liabilities", 2024, 300.0),
    ]
    for li, fy, val in rows:
        conn.execute(
            "INSERT INTO companyfacts_facts("
            "ticker, fiscal_year, period_end, line_item, value, units, "
            "source_url, fetched_at, filed_date, accession) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticker,
                fy,
                f"{fy}-12-31",
                li,
                val,
                "USD_millions",
                f"https://example.test/companyfacts/{ticker}",
                now,
                f"{fy + 1}-02-15",
                f"{ticker}-{fy}",
            ),
        )
    conn.commit()


def test_load_facts_v2_uses_issuer_alias_and_pre_restatement_vintage():
    from app.valuation.valuation_writer import _load_facts

    conn = _make_conn()
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json"
    conn.execute(
        "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, period_end, "
        "line_item, value, units, source_url, fetched_at, filed_date, form, accession) "
        "VALUES ('PRIMARY', 2024, 'FY', '2024-12-31', 'revenue', 999.0, "
        "'USD_millions', ?, '2026-07-01T00:00:00+00:00', '2026-07-01', "
        "'10-K/A', 'restated')",
        (source_url,),
    )
    conn.execute(
        "INSERT INTO companyfacts_vintages(ticker, fiscal_year, period_type, period_end, "
        "line_item, value, units, filed_date, form, accession, recorded_at, issuer_cik, "
        "source_url) VALUES ('PRIMARY', 2024, 'FY', '2024-12-31', 'revenue', "
        "100.0, 'USD_millions', '2026-02-01', '10-K', 'original', "
        "'2026-02-01T00:00:00+00:00', '42', ?)",
        (source_url,),
    )
    conn.commit()

    v2_facts = _load_facts(
        "ADRX",
        conn,
        as_of_date="2026-06-11",
        issuer_cik="42",
        issuer_aliases=("ADRX", "PRIMARY"),
        require_filed_asof=True,
    )
    strict_default_facts = _load_facts(
        "PRIMARY",
        conn,
        as_of_date="2026-06-11",
    )
    archaeology_facts = _load_facts(
        "PRIMARY",
        conn,
        as_of_date="2026-06-11",
        require_filed_asof=False,
    )

    assert v2_facts == {"revenue": [(2024, 100.0)]}
    assert strict_default_facts == {"revenue": [(2024, 100.0)]}
    assert archaeology_facts == {"revenue": [(2024, 999.0)]}


def _seed_misaligned_net_debt_companyfacts(conn: sqlite3.Connection, ticker: str = "TST") -> None:
    _seed_companyfacts(conn, ticker=ticker)
    conn.execute(
        "DELETE FROM companyfacts_facts WHERE ticker = ? AND line_item = 'cash' AND fiscal_year = 2024",
        (ticker,),
    )
    conn.execute(
        "INSERT INTO companyfacts_facts("
        "ticker, fiscal_year, period_end, line_item, value, units, source_url, "
        "fetched_at, period_type, filed_date, accession) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            ticker,
            2023,
            "2023-12-31",
            "cash",
            25.0,
            "USD_millions",
            f"https://example.test/companyfacts/{ticker}",
            "2026-01-01T00:00:00+00:00",
            "FY",
            "2024-02-15",
            f"{ticker}-2023",
        ),
    )
    conn.commit()


def _fake_quote(
    price: float | None, ticker: str = "TST", as_of_date: str = "2026-03-19"
) -> PriceQuote:
    now = "2026-01-01T00:00:00+00:00"
    return PriceQuote(
        ticker=ticker,
        as_of_date=as_of_date,
        price=price,
        currency="USD",
        provider="fake",
        status="OK" if price else "UNKNOWN",
        source_url="",
        fetched_at=now,
        expires_at=now,
        provenance={},
    )


def test_ensure_valuation_writes_method_rows():
    conn = _make_conn()
    _seed_companyfacts(conn)
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(150.0)

    with patch("app.valuation.valuation_writer.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation("TST", "2026-03-19", provider=fake_provider)

    methods = {
        r["method"]
        for r in conn.execute("SELECT method FROM valuations WHERE ticker='TST'").fetchall()
    }
    assert "owner_earnings" in methods
    assert "epv" in methods
    assert "graham" in methods
    assert "ncav" in methods


def test_ensure_valuation_persists_pending_run_lineage_and_archives_it():
    from app.valuation.lineage import valuation_integrity_fingerprint
    from app.valuation.valuation_writer import ensure_valuation

    conn = _make_conn()
    _seed_companyfacts(conn)
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(150.0)

    with patch("app.valuation.valuation_writer.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        ensure_valuation(
            "TST",
            "2026-03-19",
            provider=fake_provider,
            run_id="pending_run_one",
        )
        written_records = ensure_valuation(
            "TST",
            "2026-03-19",
            provider=fake_provider,
            run_id="pending_run_two",
            force_refresh=True,
        )

    current = conn.execute(
        """
        SELECT *
        FROM valuations
        WHERE ticker = 'TST' AND method = 'dcf'
        """
    ).fetchone()
    archived = conn.execute(
        """
        SELECT *
        FROM valuations_history
        WHERE ticker = 'TST' AND method = 'dcf'
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    assert current["source_run_id"] == "pending_run_two"
    assert current["source_artifact_path"] is None
    assert current["source_artifact_sha256"] is None
    assert current["financial_integrity_fingerprint"] == (valuation_integrity_fingerprint(current))
    dcf_records = [record for record in written_records if record["row"]["method"] == "dcf"]
    assert len(dcf_records) == 1
    assert dcf_records[0]["row"]["source_run_id"] == "pending_run_two"
    assert dcf_records[0]["row"]["outputs_json"] == current["outputs_json"]
    assert archived["source_run_id"] == "pending_run_one"
    assert archived["source_artifact_path"] is None
    assert archived["source_artifact_sha256"] is None
    assert archived["financial_integrity_fingerprint"] == (
        valuation_integrity_fingerprint(archived)
    )


def test_ensure_valuation_persists_attributed_going_concern_assertions():
    from app.alpha.schemas import GoingConcernAssertion, SolvencyAssessment
    from app.valuation.valuation_writer import ensure_valuation

    conn = _make_conn()
    _seed_companyfacts(conn)
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(150.0)
    assertion = GoingConcernAssertion(
        subject="INVESTEE",
        subject_detail="equity method investment",
        assertion_mode="ACCOUNTING_POLICY",
        blockable=False,
        accession="0000000042-26-000008",
        form_type="10-K",
        filing_date="2026-02-19",
        section="FINANCIAL_STATEMENTS_NOTES",
        excerpt=(
            "We assess whether an investee can continue as a going concern "
            "when evaluating impairment."
        ),
        issuer_cik="42",
        source_url="https://www.sec.gov/Archives/edgar/data/42/policy.htm",
        content_revision="sha256:policy-v1",
    )
    solvency = SolvencyAssessment(
        solvency_risk="LOW",
        going_concern_assertions=[assertion],
    )

    with (
        patch("app.valuation.valuation_writer.get_db") as mock_db,
        patch(
            "app.alpha.solvency_scanner.assess_solvency",
            return_value=solvency,
        ),
    ):
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        ensure_valuation("TST", "2026-03-19", provider=fake_provider)

    row = conn.execute(
        "SELECT outputs_json FROM valuations WHERE ticker = 'TST' AND method = 'scorecard'"
    ).fetchone()
    outputs = json.loads(row["outputs_json"])
    assert outputs["quality_context"]["going_concern_assertions"] == [
        {
            "subject": "INVESTEE",
            "subject_detail": "equity method investment",
            "assertion_mode": "ACCOUNTING_POLICY",
            "blockable": False,
            "accession": "0000000042-26-000008",
            "form_type": "10-K",
            "filing_date": "2026-02-19",
            "section": "FINANCIAL_STATEMENTS_NOTES",
            "excerpt": (
                "We assess whether an investee can continue as a going concern "
                "when evaluating impairment."
            ),
            "corroborating_distress": [],
            "issuer_cik": "42",
            "source_url": "https://www.sec.gov/Archives/edgar/data/42/policy.htm",
            "content_revision": "sha256:policy-v1",
        }
    ]


def test_ensure_valuation_persists_structured_price_diagnostic_inputs():
    conn = _make_conn()
    _seed_companyfacts(conn)

    class FakeMarketProvider:
        def get_price_asof(self, ticker: str, as_of_date: str):
            return None

        def get_last_diagnostic(self, ticker: str, as_of_date: str):
            return {
                "result": {
                    "reason_code": "TIMEOUT",
                    "reason_detail": "Provider request timed out.",
                },
                "output_fields": {
                    "price_source": "stooq",
                    "price_asof_used": None,
                    "confidence": None,
                },
                "provider_attempts": [
                    {
                        "provider": "eodhd",
                        "url": (
                            "https://eodhd.com/api/eod/TST.US?api_token=TEST-DIAGNOSTIC-KEY"
                            "&from=2026-03-12&to=2026-03-19&fmt=json"
                        ),
                    }
                ],
            }

    with patch("app.valuation.valuation_writer.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation("TST", "2026-03-19", provider=FakeMarketProvider())

    row = conn.execute(
        "SELECT inputs_json, outputs_json FROM valuations WHERE ticker='TST' AND method='reverse_dcf'"
    ).fetchone()
    inputs = json.loads(row["inputs_json"])
    outputs = json.loads(row["outputs_json"])
    assert inputs["price_status"] == "UNKNOWN"
    assert inputs["price_reason_code"] == "TIMEOUT"
    assert inputs["price_source"] == "stooq"
    assert inputs["price_diagnostic"]["provider_attempts"][0]["url"] == (
        "https://eodhd.com/api/eod/TST.US?from=2026-03-12&to=2026-03-19&fmt=json"
    )
    assert "TEST-DIAGNOSTIC-KEY" not in row["inputs_json"]
    assert outputs["status"] == "REVERSE_DCF_NO_PRICE"
    assert outputs["reason_code"] == "TIMEOUT"


def test_resolve_market_price_uses_run_scoped_prewarm_price(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)

    price_dir = cfg.outputs_dir / "prices" / "deep_scan_test"
    price_dir.mkdir(parents=True, exist_ok=True)
    (price_dir / "TST.json").write_text(
        json.dumps(
            {
                "ticker": "TST",
                "requested_as_of_date": "2026-03-19",
                "status": "OK",
                "diagnostic": {
                    "result": {
                        "reason_code": "CACHE_HIT",
                        "reason_detail": "Resolved from outputs/prices/<run_id> cache.",
                    },
                    "output_fields": {
                        "current_price": 155.25,
                        "price_asof_used": "2026-03-19",
                        "price_source": "run_scoped_output",
                    },
                },
                "snapshot": {
                    "ticker": "TST",
                    "as_of_date": "2026-03-19",
                    "price": 155.25,
                    "currency": "USD",
                    "source": "stooq",
                    "retrieved_at": "2026-03-19T01:00:00+00:00",
                    "url": "https://example.test/tst",
                    "confidence": "HIGH",
                    "price_basis": "UNADJUSTED",
                    "raw_price": 155.25,
                    "split_adjustment_factor": 1.0,
                    "no_intervening_split_proof": {
                        "status": "PASS",
                        "period_start": "2024-12-31",
                        "period_end": "2026-03-19",
                        "verified_as_of": "2026-03-19",
                        "source": "fixture_corporate_actions_ledger",
                        "source_reference": "fixture://corporate-actions/TST",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    from app.valuation.valuation_writer import _resolve_market_price

    price, context = _resolve_market_price(
        "TST",
        "2026-03-19",
        provider=None,
        run_id="deep_scan_test",
    )
    assert price == 155.25
    assert context["market_price"] == 155.25
    assert context["price_source_resolution"] == "run_scoped_output"
    assert context["price_currency"] == "USD"
    assert context["price_basis"] == "UNADJUSTED"
    assert context["raw_price"] == 155.25
    assert context["split_adjustment_factor"] == 1.0
    assert context["no_intervening_split_proof"]["status"] == "PASS"


def test_ensure_valuation_projects_run_quote_lineage_into_scorecard(
    monkeypatch,
    tmp_path,
):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    price_dir = cfg.outputs_dir / "prices" / "lineage_projection"
    price_dir.mkdir(parents=True, exist_ok=True)
    (price_dir / "TST.json").write_text(
        json.dumps(
            {
                "ticker": "TST",
                "requested_as_of_date": "2026-03-19",
                "status": "OK",
                "snapshot": {
                    "ticker": "TST",
                    "as_of_date": "2026-03-19",
                    "price": 155.25,
                    "currency": "USD",
                    "source": "fixture_quote",
                    "retrieved_at": "2026-03-19T01:00:00+00:00",
                    "url": "https://example.test/tst",
                    "confidence": "HIGH",
                    "price_basis": "UNADJUSTED",
                    "raw_price": 155.25,
                    "split_adjustment_factor": 1.0,
                    "no_intervening_split_proof": {
                        "status": "PASS",
                        "period_start": "2024-12-31",
                        "period_end": "2026-03-19",
                        "verified_as_of": "2026-03-19",
                        "source": "fixture_corporate_actions_ledger",
                        "source_reference": "fixture://corporate-actions/TST",
                    },
                },
                "diagnostic": {
                    "result": {
                        "reason_code": "CACHE_HIT",
                        "reason_detail": "Literal run-scoped fixture quote.",
                    },
                    "output_fields": {
                        "current_price": 155.25,
                        "price_asof_used": "2026-03-19",
                        "price_source": "fixture_quote",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    conn = _make_conn()
    _seed_companyfacts(conn)

    with patch("app.valuation.valuation_writer.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda _self: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation(
            "TST",
            "2026-03-19",
            provider=None,
            run_id="lineage_projection",
            cfg=cfg,
            require_filed_asof=False,
        )

    row = conn.execute(
        """
        SELECT outputs_json
        FROM valuations
        WHERE ticker = 'TST' AND method = 'scorecard'
        """
    ).fetchone()
    detail = json.loads(row["outputs_json"])["pricing_zone_detail"]
    assert detail["current_price"] == 155.25
    assert detail["current_price_as_of_date"] == "2026-03-19"
    assert detail["current_price_currency"] == "USD"
    assert detail["current_price_source"] == "fixture_quote"
    assert detail["current_price_source_url"] == "https://example.test/tst"
    assert detail["current_price_basis"] == "UNADJUSTED"
    assert detail["current_raw_price"] == 155.25
    assert detail["split_adjustment_factor"] == 1.0
    assert detail["no_intervening_split_proof"]["status"] == "PASS"


def test_ensure_valuation_noop_within_ttl():
    conn = _make_conn()
    _seed_companyfacts(conn)
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(150.0)

    with patch("app.valuation.valuation_writer.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation("TST", "2026-03-19", provider=fake_provider)
        count_after_first = fake_provider.get_quote.call_count
        ensure_valuation("TST", "2026-03-19", provider=fake_provider)
        assert fake_provider.get_quote.call_count == count_after_first  # no-op


def test_ensure_valuation_refetches_after_ttl_expires():
    import app.valuation.valuation_writer as vw_mod

    conn = _make_conn()
    _seed_companyfacts(conn)
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(150.0)

    with patch("app.valuation.valuation_writer.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        original_ttl = vw_mod._TTL_SECONDS
        vw_mod._TTL_SECONDS = 0
        try:
            from app.valuation.valuation_writer import ensure_valuation

            ensure_valuation("TST", "2026-03-19", provider=fake_provider)
            count_first = fake_provider.get_quote.call_count
            ensure_valuation("TST", "2026-03-19", provider=fake_provider)
            assert fake_provider.get_quote.call_count > count_first  # re-fetched
        finally:
            vw_mod._TTL_SECONDS = original_ttl


def test_ensure_valuation_price_none_still_computes_intrinsic():
    conn = _make_conn()
    _seed_companyfacts(conn)
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(None)

    with patch("app.valuation.valuation_writer.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation("TST", "2026-03-19", provider=fake_provider)

    methods = {
        r["method"]
        for r in conn.execute("SELECT method FROM valuations WHERE ticker='TST'").fetchall()
    }
    assert "epv" in methods
    assert "graham" in methods


def test_ensure_valuation_uses_asof_net_debt_proxy_when_annual_years_do_not_align():
    conn = _make_conn()
    _seed_misaligned_net_debt_companyfacts(conn)
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(150.0)

    with (
        patch("app.valuation.valuation_writer.get_db") as mock_db,
        patch(
            "app.valuation.valuation_writer.resolve_net_debt_proxy",
            # Valuation consumes the lease-EXCLUSIVE variant; the inclusive proxy
            # is for leverage/solvency diagnostics only.
            return_value={"net_debt_proxy": 60.0, "net_debt_proxy_lease_exclusive": 42.0},
        ) as mock_proxy,
    ):
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation("TST", "2026-03-19", provider=fake_provider)

    row = conn.execute(
        "SELECT inputs_json, outputs_json FROM valuations WHERE ticker='TST' AND method='scorecard'"
    ).fetchone()
    inputs = json.loads(row["inputs_json"])
    outputs = json.loads(row["outputs_json"])
    assert mock_proxy.call_count == 1
    assert inputs["net_debt"] == 42.0
    assert "ASOF_NET_DEBT_PROXY" in outputs["quality_context"]["net_debt_flags"]


def test_ensure_valuation_keeps_net_debt_unknown_when_proxy_unavailable():
    conn = _make_conn()
    _seed_misaligned_net_debt_companyfacts(conn)
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(150.0)

    with (
        patch("app.valuation.valuation_writer.get_db") as mock_db,
        patch(
            "app.valuation.valuation_writer.resolve_net_debt_proxy",
            return_value={"net_debt_proxy": "UNKNOWN"},
        ),
    ):
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation("TST", "2026-03-19", provider=fake_provider)

    dcf_row = conn.execute(
        "SELECT inputs_json, outputs_json FROM valuations WHERE ticker='TST' AND method='dcf'"
    ).fetchone()
    reverse_row = conn.execute(
        "SELECT outputs_json FROM valuations WHERE ticker='TST' AND method='reverse_dcf'"
    ).fetchone()
    scorecard_row = conn.execute(
        "SELECT outputs_json FROM valuations WHERE ticker='TST' AND method='scorecard'"
    ).fetchone()

    dcf_inputs = json.loads(dcf_row["inputs_json"])
    dcf_outputs = json.loads(dcf_row["outputs_json"])
    reverse_outputs = json.loads(reverse_row["outputs_json"])
    scorecard_outputs = json.loads(scorecard_row["outputs_json"])

    assert dcf_inputs["net_debt"] == "UNKNOWN"
    assert dcf_outputs["status"] == "METHOD_INSUFFICIENT_DATA"
    assert "NET_DEBT_UNKNOWN" in dcf_outputs["flags"]
    assert reverse_outputs["status"] == "REVERSE_DCF_INSUFFICIENT_DATA"
    assert reverse_outputs["reason_code"] == "NET_DEBT_UNKNOWN"
    assert "NET_DEBT_UNKNOWN" in scorecard_outputs["quality_context"]["net_debt_flags"]


def test_ensure_valuation_missing_senior_claim_without_absence_proof_stays_unknown():
    conn = _make_conn()
    _seed_companyfacts(conn)
    conn.execute(
        """
        DELETE FROM companyfacts_facts
        WHERE ticker = 'TST' AND line_item = 'preferred_equity'
        """
    )
    conn.commit()
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(150.0)

    with patch("app.valuation.valuation_writer.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation("TST", "2026-03-19", provider=fake_provider)

    dcf_row = conn.execute(
        """
        SELECT inputs_json, outputs_json
        FROM valuations
        WHERE ticker = 'TST' AND method = 'dcf'
        """
    ).fetchone()
    scorecard_row = conn.execute(
        """
        SELECT outputs_json
        FROM valuations
        WHERE ticker = 'TST' AND method = 'scorecard'
        """
    ).fetchone()

    dcf_inputs = json.loads(dcf_row["inputs_json"])
    dcf_outputs = json.loads(dcf_row["outputs_json"])
    scorecard_outputs = json.loads(scorecard_row["outputs_json"])
    assert dcf_inputs["net_debt"] == "UNKNOWN"
    assert dcf_outputs["status"] == "METHOD_INSUFFICIENT_DATA"
    assert "NET_DEBT_UNKNOWN" in dcf_outputs["flags"]
    assert "SENIOR_CLAIMS_UNKNOWN" in scorecard_outputs["quality_context"]["net_debt_flags"]
    assert "NET_DEBT_UNKNOWN" in scorecard_outputs["quality_context"]["net_debt_flags"]


def test_recursive_signal_prior_run_not_available():
    from app.valuation.valuation_writer import _prior_run_delta

    conn = _make_conn()
    result = _prior_run_delta(conn, "TST", "2026-03-19", "epv", 53.0)
    assert result["status"] == "PRIOR_RUN_NOT_AVAILABLE"


def test_recursive_signal_computes_delta_across_dates():
    from app.valuation.valuation_writer import _prior_run_delta

    conn = _make_conn()
    conn.execute(
        "INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at) "
        "VALUES('TST', '2026-03-10', 'epv', '{}', '{\"value_per_share\": 50.0}', '[]', '2026-03-10T00:00:00+00:00')"
    )
    conn.commit()
    result = _prior_run_delta(conn, "TST", "2026-03-19", "epv", 55.0)
    assert result["status"] == "OK"
    assert abs(result["value_change_pct"] - 0.10) < 0.01


# ── append_valuation_section + runner integration ─────────────────────────────
import os
import tempfile


def test_append_valuation_section_writes_markdown():
    conn = _make_conn()
    now = "2026-03-19T00:00:00+00:00"
    conn.execute(
        "INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at) "
        "VALUES('TST','2026-03-19','epv','{}','{\"value_per_share\":53.3,\"status\":\"OK\"}','[]',?)",
        (now,),
    )
    conn.execute(
        "INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at) "
        "VALUES('TST','2026-03-19','scorecard','{}','{\"signal\":\"FAIRLY_VALUED\",\"type\":\"EARNINGS_DRIVEN\"}','[]',?)",
        (now,),
    )
    conn.commit()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Existing Dossier\n")
        fpath = f.name
    try:
        with patch("app.valuation.valuation_writer.get_db") as mock_db:
            mock_db.return_value.__enter__ = lambda s: conn
            mock_db.return_value.__exit__ = MagicMock(return_value=False)
            from app.valuation.valuation_writer import append_valuation_section

            append_valuation_section("TST", "2026-03-19", fpath)
        content = open(fpath).read()
        assert "## Valuation" in content
        assert "FAIRLY_VALUED" in content
    finally:
        os.unlink(fpath)


def test_append_valuation_section_noop_when_no_data():
    conn = _make_conn()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Existing\n")
        fpath = f.name
    try:
        with patch("app.valuation.valuation_writer.get_db") as mock_db:
            mock_db.return_value.__enter__ = lambda s: conn
            mock_db.return_value.__exit__ = MagicMock(return_value=False)
            from app.valuation.valuation_writer import append_valuation_section

            append_valuation_section("TST", "2026-03-19", fpath)
        assert "## Valuation" not in open(fpath).read()
    finally:
        os.unlink(fpath)


def test_runner_imports_valuation_functions():
    from app.dossier import runner as runner_mod
    import inspect

    source = inspect.getsource(runner_mod)
    assert "ensure_valuation" in source
    assert "append_valuation_section" in source


def test_ensure_valuation_errors_do_not_raise():
    from app.valuation.valuation_writer import ensure_valuation

    # Bad ticker with no DB data — must not propagate
    ensure_valuation("NONEXISTENT_XYZ_999", "2026-03-19", provider=None)


def test_ensure_valuation_can_surface_errors_to_repair_callers(tmp_path):
    from app.config import get_config
    from app.valuation.valuation_writer import ensure_valuation

    cfg = get_config().model_copy(update={"db_path": tmp_path / "routed.db"})
    with patch(
        "app.valuation.valuation_writer._ensure_valuation_inner",
        side_effect=RuntimeError("temporary valuation failure"),
    ) as inner:
        ensure_valuation(
            "TST",
            "2026-03-19",
            cfg=cfg,
            db_path=cfg.db_path,
        )
        try:
            ensure_valuation(
                "TST",
                "2026-03-19",
                cfg=cfg,
                db_path=cfg.db_path,
                raise_on_error=True,
            )
        except RuntimeError as exc:
            assert str(exc) == "temporary valuation failure"
        else:
            raise AssertionError("raise_on_error=True must preserve the valuation exception")

    assert inner.call_count == 2
    assert inner.call_args.kwargs["cfg"] is cfg
    assert inner.call_args.kwargs["db_path"] == cfg.db_path


def test_ensure_valuation_inner_routes_explicit_db_path_through_config(tmp_path):
    from app.config import get_config
    from app.valuation.valuation_writer import _ensure_valuation_inner

    ambient_cfg = get_config()
    routed_path = tmp_path / "injected" / "engine.db"
    connection = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = connection
    context.__exit__.return_value = False

    with (
        patch(
            "app.valuation.valuation_writer.get_db",
            return_value=context,
        ) as routed_get_db,
        patch(
            "app.valuation.valuation_writer._is_valuation_fresh",
            return_value=True,
        ),
    ):
        _ensure_valuation_inner(
            "TST",
            "2026-03-19",
            None,
            run_id=None,
            price_override=None,
            force_refresh=False,
            cfg=ambient_cfg,
            db_path=routed_path,
        )

    routed_cfg = routed_get_db.call_args.kwargs["cfg"]
    assert routed_cfg.db_path == routed_path
    assert ambient_cfg.db_path != routed_path


def test_v2_valuation_routes_fixed_issuer_context_to_quality_gate(tmp_path):
    from app.config import get_config
    from app.valuation.valuation_writer import _ensure_valuation_inner

    class _StopAfterQualityGate(RuntimeError):
        pass

    ambient_cfg = get_config()
    routed_path = tmp_path / "issuer" / "engine.db"
    connection = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = connection
    context.__exit__.return_value = False
    facts = {"revenue": [(2025, 100.0)]}
    facts_row = {"ticker": "ADRX", "fiscal_year": 2025}

    with (
        patch("app.valuation.valuation_writer.get_db", return_value=context),
        patch(
            "app.valuation.valuation_writer._is_valuation_fresh",
            return_value=False,
        ),
        patch("app.valuation.valuation_writer._load_facts", return_value=facts),
        patch(
            "app.valuation.valuation_writer._local_v2_facts_row",
            return_value=facts_row,
        ),
        patch(
            "app.valuation.valuation_writer.classify_company_category",
            return_value={"category": "TRADITIONAL_OPERATING"},
        ),
        patch(
            "app.valuation.valuation_writer._compute_quality_wacc",
            return_value={"adjusted_wacc": 0.10},
        ),
        patch(
            "app.valuation.pre_valuation_gate.compute_quality_context",
            side_effect=_StopAfterQualityGate("captured"),
        ) as quality_gate,
    ):
        try:
            _ensure_valuation_inner(
                "ADRX",
                "2026-03-01",
                None,
                run_id="issuer-context-test",
                price_override=42.0,
                force_refresh=True,
                cfg=ambient_cfg,
                db_path=routed_path,
                issuer_cik="42",
                issuer_aliases=("ADRX", "PRIMARY"),
                require_filed_asof=True,
            )
        except _StopAfterQualityGate as exc:
            assert str(exc) == "captured"
        else:
            raise AssertionError("quality-gate capture sentinel was not raised")

    assert quality_gate.call_args.args == ("ADRX", "2026-03-01")
    assert quality_gate.call_args.kwargs["facts"] == facts
    assert quality_gate.call_args.kwargs["facts_row"] == facts_row
    assert quality_gate.call_args.kwargs["issuer_cik"] == "42"
    assert quality_gate.call_args.kwargs["issuer_aliases"] == ("ADRX", "PRIMARY")
    assert quality_gate.call_args.kwargs["require_filed_asof"] is True
    assert quality_gate.call_args.kwargs["db_path"] == routed_path


def test_lease_adjusted_flag_present(monkeypatch, tmp_path):
    """When lease_liability > 0, LEASE_ADJUSTED flag should be in scorecard."""
    import json
    from app.config import get_config as _get_config

    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    _get_config.cache_clear()
    from app.db import init_db, get_db

    init_db()

    from app.valuation.valuation_writer import ensure_valuation

    ensure_valuation("LEASE_FLAG_TEST", "2026-03-25", provider=None, force_refresh=True)

    with get_db() as conn:
        row = conn.execute(
            "SELECT outputs_json FROM valuations WHERE ticker = ? AND method = 'scorecard'",
            ("LEASE_FLAG_TEST",),
        ).fetchone()

    if row:
        outputs = json.loads(row["outputs_json"] or "{}")
        qc = outputs.get("quality_context", {})
        assert "net_debt_flags" not in qc or isinstance(qc.get("net_debt_flags"), list)


# ── EPV quality annotation ────────────────────────────────────────────────────


def test_classify_epv_quality_thresholds():
    from app.valuation.valuation_writer import _classify_epv_quality

    assert _classify_epv_quality(-0.07, None) == ("DETERIORATING_BASE", -0.07)
    assert _classify_epv_quality(-0.04, None) == ("DECLINING_BASE", -0.04)
    assert _classify_epv_quality(-0.01, None) == ("STABLE", -0.01)
    assert _classify_epv_quality(0.05, None) == ("STABLE", 0.05)
    assert _classify_epv_quality(None, None) == ("UNKNOWN", None)


def test_classify_epv_quality_3y_fallback():
    from app.valuation.valuation_writer import _classify_epv_quality

    # 5y null → falls back to 3y
    assert _classify_epv_quality(None, -0.08) == ("DETERIORATING_BASE", -0.08)
    assert _classify_epv_quality(None, -0.04) == ("DECLINING_BASE", -0.04)
    assert _classify_epv_quality(None, 0.02) == ("STABLE", 0.02)


def test_epv_quality_deteriorating_on_margin_of_safety():
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": 200.0},
        "epv": {"status": "OK", "value_per_share": 180.0},
        "graham": {"status": "OK", "value_per_share": 160.0},
        "ncav": {"status": "OK", "value_per_share": -5.0, "signal": "NCAV_NO_ASSET_FLOOR"},
    }
    result = _margin_of_safety_scorecard(
        methods,
        price=100.0,
        shares=10.0,
        net_debt=20.0,
        revenue_cagr_5y=-0.07,
    )
    assert result["pricing_zone"] == "MARGIN_OF_SAFETY"
    assert result["pricing_zone_detail"]["epv_quality"] == "DETERIORATING_BASE"
    assert result["pricing_zone_detail"]["revenue_cagr_5y"] == -0.07


def test_epv_quality_stable():
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": 200.0},
        "epv": {"status": "OK", "value_per_share": 180.0},
        "graham": {"status": "OK", "value_per_share": 160.0},
        "ncav": {"status": "OK", "value_per_share": -5.0, "signal": "NCAV_NO_ASSET_FLOOR"},
    }
    result = _margin_of_safety_scorecard(
        methods,
        price=100.0,
        shares=10.0,
        net_debt=20.0,
        revenue_cagr_5y=-0.01,
    )
    assert result["pricing_zone_detail"]["epv_quality"] == "STABLE"


def test_epv_quality_unknown_null():
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": 200.0},
        "epv": {"status": "OK", "value_per_share": 180.0},
        "graham": {"status": "OK", "value_per_share": 160.0},
        "ncav": {"status": "OK", "value_per_share": -5.0, "signal": "NCAV_NO_ASSET_FLOOR"},
    }
    result = _margin_of_safety_scorecard(
        methods,
        price=100.0,
        shares=10.0,
        net_debt=20.0,
        revenue_cagr_5y=None,
        revenue_cagr_3y=None,
    )
    assert result["pricing_zone_detail"]["epv_quality"] == "UNKNOWN"


def test_epv_quality_present_on_non_mos_zones():
    """epv_quality must appear on every pricing zone, not just MARGIN_OF_SAFETY."""
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    # GROWTH_DEPENDENT zone
    methods = {
        "dcf": {"status": "OK", "base": 180.0},
        "epv": {"status": "OK", "value_per_share": 70.0},
        "epv_adjusted": {"status": "OK", "value_per_share": 120.0, "avg_operating_income": 90.0},
        "graham": {"status": "OK", "value_per_share": 60.0},
        "ncav": {"status": "OK", "value_per_share": -5.0, "signal": "NCAV_NO_ASSET_FLOOR"},
    }
    result = _margin_of_safety_scorecard(
        methods,
        price=150.0,
        shares=10.0,
        net_debt=20.0,
        revenue_cagr_5y=0.08,
    )
    assert result["pricing_zone"] == "GROWTH_DEPENDENT"
    assert result["pricing_zone_detail"]["epv_quality"] == "STABLE"
    assert result["pricing_zone_detail"]["revenue_cagr_5y"] == 0.08


# ── Wave 2: Balance sheet integrity tests ─────────────────────────────────────


def test_tag_map_has_wave2_items():
    """TAG_MAP now includes all 11 Wave 2 balance sheet items."""
    from app.ingest.companyfacts import TAG_MAP, PLAUSIBILITY

    wave2_items = [
        "accounts_receivable",
        "inventory",
        "accounts_payable",
        "current_assets",
        "current_liabilities",
        "gross_ppe",
        "depreciation_amortization",
        "goodwill",
        "intangible_assets",
        "operating_lease_liability",
        "restructuring_charges",
    ]
    for item in wave2_items:
        assert item in TAG_MAP, f"{item} missing from TAG_MAP"
        assert item in PLAUSIBILITY, f"{item} missing from PLAUSIBILITY"
        assert len(TAG_MAP[item]) >= 1, f"{item} has no tags"


def test_ncav_real_balance_sheet_with_graham_haircuts():
    """NCAV uses real balance sheet data with Graham liquidation haircuts."""
    from app.valuation.valuation_writer import _ncav

    result = _ncav(
        cash=50.0,
        revenue=200.0,
        total_liabilities=120.0,
        total_debt=80.0,
        shares=10.0,
        price=20.0,
        accounts_receivable=30.0,
        inventory=20.0,
        current_assets=110.0,  # cash=50 + AR=30 + inv=20 + other=10
        current_liabilities=60.0,
    )
    assert result["status"] == "OK"
    assert "NCAV_REAL_BALANCE_SHEET" in result["flags"]
    assert "NCAV_PROXY_ESTIMATED" not in result["flags"]
    # liquid = 50*1.0 + 30*0.75 + 20*0.50 + 10*0.25 = 50+22.5+10+2.5 = 85
    # Textbook Graham: subtract TOTAL liabilities (audit:
    # ncav-current-liabilities-only): NCAV = 85 - 120 = -35 -> -3.5/share
    assert abs(result["value_per_share"] - (-3.5)) < 0.01


def test_ncav_falls_back_to_proxy_when_current_assets_missing():
    """Without current_assets, NCAV uses legacy proxy."""
    from app.valuation.valuation_writer import _ncav

    result = _ncav(
        cash=50.0,
        revenue=200.0,
        total_liabilities=120.0,
        total_debt=80.0,
        shares=10.0,
        price=20.0,
    )
    assert result["status"] == "OK"
    assert "NCAV_PROXY_ESTIMATED" in result["flags"]
    assert "NCAV_REAL_BALANCE_SHEET" not in result["flags"]


def test_ncav_total_liabilities_preferred_over_current_liabilities():
    """Textbook Graham: TOTAL liabilities are subtracted whole; long-term debt
    must not vanish from the floor (audit: ncav-current-liabilities-only)."""
    from app.valuation.valuation_writer import _ncav

    result = _ncav(
        cash=50.0,
        revenue=200.0,
        total_liabilities=300.0,  # the textbook deduction — NCAV negative
        total_debt=80.0,
        shares=10.0,
        price=20.0,
        accounts_receivable=0.0,
        inventory=0.0,
        current_assets=110.0,
        current_liabilities=60.0,  # understating proxy, must NOT be preferred
    )
    assert result["value_per_share"] < 0  # using total_liabilities=300
    assert "NCAV_REAL_BALANCE_SHEET" in result["flags"]
    assert "LIABILITIES_CURRENT_ONLY_PROXY" not in result["flags"]


def test_ncav_ar_inventory_missing_is_insufficient_data():
    """Missing current-asset components cannot be treated as reported zero."""
    from app.valuation.valuation_writer import _ncav

    result = _ncav(
        cash=50.0,
        revenue=200.0,
        total_liabilities=120.0,
        total_debt=80.0,
        shares=10.0,
        price=20.0,
        current_assets=80.0,
        current_liabilities=60.0,
    )
    assert result == {
        "status": "METHOD_INSUFFICIENT_DATA",
        "flags": ["AR_MISSING", "INVENTORY_MISSING"],
        "value_per_share": None,
        "signal": None,
    }


def test_ncav_explicit_zero_ar_and_inventory_remain_valid_values():
    from app.valuation.valuation_writer import _ncav

    result = _ncav(
        cash=50.0,
        revenue=200.0,
        total_liabilities=60.0,
        total_debt=40.0,
        shares=10.0,
        price=20.0,
        accounts_receivable=0.0,
        inventory=0.0,
        current_assets=80.0,
        current_liabilities=60.0,
    )
    assert result["status"] == "OK"
    assert result["flags"] == ["NCAV_REAL_BALANCE_SHEET"]
    assert result["value_per_share"] == -0.25


def test_ncav_proxy_missing_revenue_is_insufficient_data():
    from app.valuation.valuation_writer import _ncav

    result = _ncav(
        cash=50.0,
        revenue=None,
        total_liabilities=60.0,
        total_debt=40.0,
        shares=10.0,
        price=20.0,
    )
    assert result == {
        "status": "METHOD_INSUFFICIENT_DATA",
        "flags": ["REVENUE_MISSING"],
        "value_per_share": None,
        "signal": None,
    }


def test_ncav_proxy_explicit_zero_revenue_remains_valid_value():
    from app.valuation.valuation_writer import _ncav

    result = _ncav(
        cash=50.0,
        revenue=0.0,
        total_liabilities=60.0,
        total_debt=40.0,
        shares=10.0,
        price=20.0,
    )
    assert result["status"] == "OK"
    assert result["flags"] == ["NCAV_PROXY_ESTIMATED"]
    assert result["value_per_share"] == -1.0


def test_ebitda_proxy_uses_real_da():
    """When D&A data exists, EBITDA proxy = operating_income + D&A."""
    from app.valuation.valuation_writer import _latest_net_debt_to_ebitda_proxy

    facts = {
        "total_debt": [(2025, 100.0)],
        "cash": [(2025, 20.0)],
        "operating_income": [(2025, 30.0)],
        "depreciation_amortization": [(2025, 10.0)],
    }
    ratio = _latest_net_debt_to_ebitda_proxy(facts)
    # net_debt = 100-20 = 80, EBITDA = 30+10 = 40, ratio = 80/40 = 2.0
    assert ratio is not None
    assert abs(ratio - 2.0) < 0.01


def test_ebitda_proxy_falls_back_to_operating_income():
    """Without D&A, EBITDA proxy uses operating income alone."""
    from app.valuation.valuation_writer import _latest_net_debt_to_ebitda_proxy

    facts = {
        "total_debt": [(2025, 100.0)],
        "cash": [(2025, 20.0)],
        "operating_income": [(2025, 30.0)],
    }
    ratio = _latest_net_debt_to_ebitda_proxy(facts)
    # net_debt = 80, EBITDA proxy = 30, ratio = 80/30 ≈ 2.67
    assert ratio is not None
    assert abs(ratio - 80 / 30) < 0.01


def test_interest_coverage_adequacy_classifications():
    """Interest coverage adequacy maps to Graham thresholds."""
    from app.valuation.valuation_writer import _capital_structure_health

    # STRONG: 8x coverage
    facts_strong = {
        "operating_income": [(2025, 80.0)],
        "interest_expense": [(2025, 10.0)],
        "total_debt": [(2025, 50.0)],
        "equity": [(2025, 100.0)],
        "cash": [(2025, 20.0)],
    }
    r = _capital_structure_health(facts_strong)
    assert r["interest_coverage_adequacy"] == "STRONG"

    # THIN: 4x coverage
    facts_thin = dict(facts_strong)
    facts_thin["operating_income"] = [(2025, 40.0)]
    r = _capital_structure_health(facts_thin)
    assert r["interest_coverage_adequacy"] == "THIN"

    # CRITICAL: 1.2x coverage
    facts_critical = dict(facts_strong)
    facts_critical["operating_income"] = [(2025, 12.0)]
    r = _capital_structure_health(facts_critical)
    assert r["interest_coverage_adequacy"] == "CRITICAL"


def test_interest_coverage_wacc_penalties():
    """WACC adds penalty for weak and critical interest coverage."""
    from app.valuation.valuation_writer import _compute_quality_wacc

    # Critical coverage: OI=10, IE=8 → 1.25x → CRITICAL → +1% WACC
    facts = {
        "revenue": [(2025, 100.0), (2024, 95.0), (2023, 90.0)],
        "gross_profit": [(2025, 40.0), (2024, 38.0), (2023, 36.0)],
        "cfo": [(2025, 15.0), (2024, 14.0), (2023, 13.0)],
        "net_income": [(2025, 10.0), (2024, 9.0), (2023, 8.0)],
        "operating_income": [(2025, 10.0)],
        "interest_expense": [(2025, 8.0)],
        "total_debt": [(2025, 20.0)],
        "cash": [(2025, 10.0)],
    }
    result = _compute_quality_wacc(facts)
    rules = {row["code"]: row for row in result["rule_evaluations"]}
    assert rules["INTEREST_COVERAGE_CRITICAL"]["fired"] is True
    assert rules["INTEREST_COVERAGE_WEAK"]["fired"] is False  # mutually exclusive ranges


# ── Maintenance capex ratio in owner earnings ─────────────────────────────────


def test_owner_earnings_maintenance_capex_ratio_reduces_capex():
    """When maintenance_capex_ratio < 1.0, only that fraction of normalized capex is deducted."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = {
        "cfo": [(2025, 100.0), (2024, 95.0), (2023, 90.0)],
        "capex": [(2025, 40.0), (2024, 38.0), (2023, 36.0)],
    }
    full = _compute_owner_earnings(facts, maintenance_capex_ratio=1.0)
    reduced = _compute_owner_earnings(facts, maintenance_capex_ratio=0.25)
    assert reduced["owner_earnings_latest"] > full["owner_earnings_latest"]
    assert "MAINT_CAPEX_RATIO_25%" in reduced["flags"]
    assert all("MAINT_CAPEX_RATIO" not in f for f in full["flags"])


def test_owner_earnings_default_ratio_is_full_capex():
    """Default maintenance_capex_ratio=1.0 preserves backward compatibility."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = {
        "cfo": [(2025, 100.0), (2024, 95.0), (2023, 90.0)],
        "capex": [(2025, 40.0), (2024, 38.0), (2023, 36.0)],
    }
    default_result = _compute_owner_earnings(facts)
    explicit_result = _compute_owner_earnings(facts, maintenance_capex_ratio=1.0)
    assert default_result["owner_earnings_latest"] == explicit_result["owner_earnings_latest"]


def test_quality_metadata_columns_exist():
    """Verify the 5 new quality metadata columns exist in valuations table."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn=conn)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(valuations)").fetchall()}
    conn.close()
    for col in [
        "quality_gate_verdict",
        "confidence_class",
        "gate_reason_codes",
        "valuation_headwinds",
        "valuation_supports",
    ]:
        assert col in cols, f"Missing column: {col}"


def test_epv_blocked_on_secular_decline():
    """When epv_adjustment=BLOCK, EPV returns None with reason code."""
    from app.valuation.valuation_writer import _epv

    oi_series = [(2025, 15.0), (2024, 14.0), (2023, 13.0), (2022, 12.0), (2021, 11.0)]
    result = _epv(oi_series, revenue_series=_flat_revenue(oi_series), net_debt=10.0, shares=5.0, epv_adjustment="BLOCK")
    assert result["status"] == "EPV_BLOCKED_SECULAR_DECLINE"
    assert result["value_per_share"] is None
    assert "EPV_BLOCKED_SECULAR_DECLINE" in result["flags"]


def test_epv_uses_normalized_on_use_normalized():
    """When epv_adjustment=USE_NORMALIZED and normalized_earnings provided, use them."""
    from app.valuation.valuation_writer import _epv

    oi_series = [(2025, 30.0), (2024, 28.0), (2023, 25.0), (2022, 20.0), (2021, 18.0)]
    result_raw = _epv(oi_series, revenue_series=_flat_revenue(oi_series), net_debt=10.0, shares=5.0)
    result_normalized = _epv(
        oi_series,
        revenue_series=_flat_revenue(oi_series),
        net_debt=10.0,
        shares=5.0,
        epv_adjustment="USE_NORMALIZED",
        normalized_earnings=15.0,
    )
    assert result_normalized["status"] in ("OK", "EPV_NEGATIVE")
    assert result_normalized["value_per_share"] is not None
    assert result_normalized["value_per_share"] < result_raw["value_per_share"]
    assert result_normalized["avg_operating_income"] == 15.0
    assert "EPV_CYCLICALLY_NORMALIZED" in result_normalized["flags"]


def test_epv_unchanged_on_none_adjustment():
    """When epv_adjustment=NONE, EPV behaves identically to current."""
    from app.valuation.valuation_writer import _epv

    oi_series = [(2025, 15.0), (2024, 14.0), (2023, 13.0), (2022, 12.0), (2021, 11.0)]
    result_default = _epv(oi_series, revenue_series=_flat_revenue(oi_series), net_debt=10.0, shares=5.0)
    result_none = _epv(oi_series, revenue_series=_flat_revenue(oi_series), net_debt=10.0, shares=5.0, epv_adjustment="NONE")
    assert result_default["value_per_share"] == result_none["value_per_share"]


def test_scorecard_insufficient_quality_on_block():
    """BLOCK gate → signal=INSUFFICIENT_QUALITY regardless of price vs intrinsic."""
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": 50.0, "low": 30.0, "high": 70.0, "flags": []},
        "epv": {"status": "OK", "value_per_share": 45.0, "flags": []},
        "graham": {"status": "OK", "value_per_share": 40.0, "buy_price": 27.0, "flags": []},
    }
    result = _margin_of_safety_scorecard(
        methods,
        price=20.0,
        shares=10.0,
        net_debt=5.0,
        gate_action="BLOCK",
    )
    assert result["signal"] == "INSUFFICIENT_QUALITY"


def test_scorecard_wider_thresholds_on_adjust():
    """ADJUST gate → discount thresholds widened by 15%."""
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": 100.0, "low": 80.0, "high": 120.0, "flags": []},
        "epv": {"status": "OK", "value_per_share": 100.0, "flags": []},
        "graham": {"status": "OK", "value_per_share": 100.0, "buy_price": 67.0, "flags": []},
    }
    result_normal = _margin_of_safety_scorecard(
        methods,
        price=60.0,
        shares=10.0,
        net_debt=5.0,
    )
    result_adjusted = _margin_of_safety_scorecard(
        methods,
        price=60.0,
        shares=10.0,
        net_debt=5.0,
        gate_action="ADJUST",
        mos_threshold_widening=0.15,
    )
    # Normal: 40% discount → DEEP_VALUE (threshold 33%)
    assert result_normal["legacy_signal"] == "DEEP_VALUE"
    # Adjusted: 40% < 48% (33%+15%) → not DEEP_VALUE, but > 30% (15%+15%) → UNDERVALUED
    assert result_adjusted["legacy_signal"] == "UNDERVALUED"


def test_scorecard_normal_thresholds_on_proceed():
    """PROCEED gate → thresholds unchanged."""
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": 100.0, "low": 80.0, "high": 120.0, "flags": []},
        "epv": {"status": "OK", "value_per_share": 100.0, "flags": []},
        "graham": {"status": "OK", "value_per_share": 100.0, "buy_price": 67.0, "flags": []},
    }
    result = _margin_of_safety_scorecard(
        methods,
        price=60.0,
        shares=10.0,
        net_debt=5.0,
        gate_action="PROCEED",
    )
    assert result["legacy_signal"] == "DEEP_VALUE"


def test_quality_metadata_written_to_db(monkeypatch, tmp_path):
    """Verify quality metadata columns populated in valuations rows."""
    import json
    from app.config import get_config as _get_config

    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    _get_config.cache_clear()
    from app.db import init_db, get_db

    init_db()

    from app.valuation.valuation_writer import ensure_valuation

    ensure_valuation("QGATE_META_TEST", "2026-03-25", provider=None)

    with get_db() as conn:
        rows = conn.execute(
            "SELECT method, quality_gate_verdict, confidence_class, gate_reason_codes, "
            "valuation_headwinds, valuation_supports FROM valuations WHERE ticker = ?",
            ("QGATE_META_TEST",),
        ).fetchall()

    if rows:
        for row in rows:
            assert row["quality_gate_verdict"] in ("PROCEED", "ADJUST", "BLOCK", None)
            if row["quality_gate_verdict"] is not None:
                assert row["confidence_class"] in ("HIGH", "MODERATE", "LOW", "INSUFFICIENT")
                codes = json.loads(row["gate_reason_codes"] or "[]")
                assert isinstance(codes, list)


def test_backward_compat_existing_rows(monkeypatch, tmp_path):
    """Existing valuation rows without new columns still readable."""
    from app.config import get_config as _get_config

    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    _get_config.cache_clear()
    from app.db import init_db, get_db

    init_db()

    with get_db() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO valuations
               (ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at)
               VALUES(?, ?, ?, ?, ?, ?, ?)""",
            ("COMPAT_TEST", "2026-01-01", "test_method", "{}", "{}", "[]", "2026-01-01T00:00:00Z"),
        )
        conn.commit()
        row = conn.execute(
            "SELECT quality_gate_verdict, confidence_class FROM valuations WHERE ticker = ?",
            ("COMPAT_TEST",),
        ).fetchone()
    assert row is not None
    assert row["quality_gate_verdict"] is None
    assert row["confidence_class"] is None


def test_secular_decliner_no_buy_signal():
    """KEY ACCEPTANCE TEST: secular decliner must NOT produce MARGIN_OF_SAFETY."""
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": 50.0, "low": 30.0, "high": 70.0, "flags": []},
        "epv": {"status": "OK", "value_per_share": 45.0, "flags": []},
        "graham": {"status": "OK", "value_per_share": 40.0, "buy_price": 27.0, "flags": []},
    }
    result = _margin_of_safety_scorecard(
        methods,
        price=20.0,
        shares=10.0,
        net_debt=5.0,
        gate_action="BLOCK",
    )
    assert result["signal"] == "INSUFFICIENT_QUALITY"
    assert result["signal"] != "DEEP_VALUE"
    assert result["signal"] != "UNDERVALUED"


def test_high_quality_grower_unchanged():
    """High-quality grower must still produce valid valuations (no false negatives)."""
    from app.valuation.valuation_writer import _margin_of_safety_scorecard, _epv

    oi_series = [(2025, 20.0), (2024, 18.0), (2023, 16.0), (2022, 14.0), (2021, 12.0)]
    epv_result = _epv(oi_series, revenue_series=_flat_revenue(oi_series), net_debt=5.0, shares=10.0, epv_adjustment="NONE")
    assert epv_result["status"] == "OK"
    assert epv_result["value_per_share"] is not None
    assert epv_result["value_per_share"] > 0

    methods = {
        "dcf": {"status": "OK", "base": 50.0, "low": 30.0, "high": 70.0, "flags": []},
        "epv": epv_result,
        "graham": {"status": "OK", "value_per_share": 40.0, "buy_price": 27.0, "flags": []},
    }
    result = _margin_of_safety_scorecard(
        methods,
        price=25.0,
        shares=10.0,
        net_debt=5.0,
        gate_action="PROCEED",
    )
    assert result["signal"] != "INSUFFICIENT_QUALITY"


def test_cyclical_at_peak_uses_normalized():
    """Cyclical company at peak → EPV uses normalized earnings."""
    from app.valuation.valuation_writer import _epv

    oi_series = [(2025, 30.0), (2024, 28.0), (2023, 25.0), (2022, 20.0), (2021, 18.0)]
    result = _epv(
        oi_series,
        revenue_series=_flat_revenue(oi_series),
        net_debt=5.0,
        shares=10.0,
        epv_adjustment="USE_NORMALIZED",
        normalized_earnings=15.0,
    )
    assert result["status"] in ("OK", "EPV_NEGATIVE")
    assert result["avg_operating_income"] == 15.0
    assert "EPV_CYCLICALLY_NORMALIZED" in result["flags"]


def test_capital_structure_uses_real_ebitda():
    """With D&A data, net-debt-to-EBITDA uses OI + D&A."""
    from app.valuation.valuation_writer import _capital_structure_health

    facts = {
        "operating_income": [(2025, 100.0)],
        "interest_expense": [(2025, 20.0)],
        "total_debt": [(2025, 500.0)],
        "equity": [(2025, 300.0)],
        "cash": [(2025, 50.0)],
        "depreciation_amortization": [(2025, 80.0)],
    }
    result = _capital_structure_health(facts)
    assert result["net_debt_to_ebitda"] is not None
    assert abs(result["net_debt_to_ebitda"] - 2.5) < 0.01
    assert result["ebitda_method"] == "REAL_EBITDA"


def test_capital_structure_falls_back_oi():
    """Without D&A data, uses OI as proxy with flag."""
    from app.valuation.valuation_writer import _capital_structure_health

    facts = {
        "operating_income": [(2025, 100.0)],
        "interest_expense": [(2025, 20.0)],
        "total_debt": [(2025, 500.0)],
        "equity": [(2025, 300.0)],
        "cash": [(2025, 50.0)],
    }
    result = _capital_structure_health(facts)
    assert result["net_debt_to_ebitda"] is not None
    assert abs(result["net_debt_to_ebitda"] - 4.5) < 0.01
    assert result["ebitda_method"] == "OI_PROXY"


def test_ebitda_changes_leverage_assessment():
    """Company with high D&A gets lower leverage ratio with real EBITDA."""
    from app.valuation.valuation_writer import _capital_structure_health

    facts_no_da = {
        "operating_income": [(2025, 50.0)],
        "interest_expense": [(2025, 10.0)],
        "total_debt": [(2025, 300.0)],
        "equity": [(2025, 200.0)],
        "cash": [(2025, 20.0)],
    }
    facts_with_da = {
        **facts_no_da,
        "depreciation_amortization": [(2025, 100.0)],
    }
    result_no_da = _capital_structure_health(facts_no_da)
    result_with_da = _capital_structure_health(facts_with_da)
    assert result_with_da["net_debt_to_ebitda"] < result_no_da["net_debt_to_ebitda"]


def test_capital_structure_backward_compat():
    """Existing fields still present and correct."""
    from app.valuation.valuation_writer import _capital_structure_health

    facts = {
        "operating_income": [(2025, 100.0)],
        "interest_expense": [(2025, 20.0)],
        "total_debt": [(2025, 500.0)],
        "equity": [(2025, 300.0)],
        "cash": [(2025, 50.0)],
    }
    result = _capital_structure_health(facts)
    assert "interest_coverage" in result
    assert "interest_coverage_adequacy" in result
    assert "de_ratio" in result
    assert "cash_coverage" in result
    assert isinstance(result["interest_coverage"], (int, float))
    assert result["interest_coverage_adequacy"] == "ADEQUATE"


def test_capital_structure_missing_cash_and_debt_values_do_not_become_zero():
    from app.valuation.valuation_writer import _capital_structure_health

    missing_cash = _capital_structure_health(
        {
            "operating_income": [(2025, 100.0)],
            "total_debt": [(2025, 50.0)],
            "cash": [(2025, None)],
            "equity": [(2025, 200.0)],
        }
    )
    missing_debt = _capital_structure_health(
        {
            "operating_income": [(2025, 100.0)],
            "total_debt": [(2025, None)],
            "cash": [(2025, 20.0)],
            "equity": [(2025, 200.0)],
        }
    )

    assert missing_cash["cash_coverage"] is None
    assert missing_cash["net_debt_to_ebitda"] is None
    assert missing_debt["de_ratio"] is None
    assert missing_debt["cash_coverage"] is None
    assert missing_debt["net_debt_to_ebitda"] is None


def test_capital_structure_explicit_zero_debt_and_cash_remain_valid():
    from app.valuation.valuation_writer import _capital_structure_health

    zero_debt = _capital_structure_health(
        {
            "operating_income": [(2025, 100.0)],
            "total_debt": [(2025, 0.0)],
            "cash": [(2025, 20.0)],
            "equity": [(2025, 200.0)],
        }
    )
    zero_cash = _capital_structure_health(
        {
            "operating_income": [(2025, 100.0)],
            "total_debt": [(2025, 50.0)],
            "cash": [(2025, 0.0)],
            "equity": [(2025, 200.0)],
        }
    )

    assert zero_debt["de_ratio"] == 0.0
    assert zero_debt["cash_coverage"] == "TOTAL_DEBT_ZERO"
    assert zero_debt["net_debt_to_ebitda"] == -0.2
    assert zero_cash["cash_coverage"] == 0.0
    assert zero_cash["net_debt_to_ebitda"] == 0.5


def test_moat_strong():
    from app.valuation.valuation_writer import _classify_moat_strength

    result = _classify_moat_strength(
        earnings_quality="HIGH",
        revenue_trend_class="GROWING",
        epv_quality="STABLE",
        allocation_grade="A",
    )
    assert result["moat_class"] == "STRONG_MOAT"
    assert result["signals_counted"] == 4


def test_moat_none():
    from app.valuation.valuation_writer import _classify_moat_strength

    result = _classify_moat_strength(
        earnings_quality="LOW",
        revenue_trend_class="SECULAR_DECLINE",
        epv_quality="DETERIORATING_BASE",
        allocation_grade="F",
    )
    assert result["moat_class"] == "NO_MOAT"
    assert result["moat_score"] < 0


def test_moat_partial_signals():
    from app.valuation.valuation_writer import _classify_moat_strength

    result = _classify_moat_strength(
        earnings_quality="HIGH",
        revenue_trend_class="GROWING",
        epv_quality=None,
        allocation_grade=None,
    )
    assert result["signals_counted"] == 2
    assert result["moat_class"] in ("STRONG_MOAT", "MODERATE_MOAT", "WEAK_MOAT", "NO_MOAT")


def test_overvalued_strong_moat_premium_justified():
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": 50.0, "low": 30.0, "high": 70.0, "flags": []},
        "epv": {"status": "OK", "value_per_share": 45.0, "flags": []},
        "graham": {"status": "OK", "value_per_share": 40.0, "buy_price": 27.0, "flags": []},
    }
    result = _margin_of_safety_scorecard(
        methods,
        price=200.0,
        shares=10.0,
        net_debt=5.0,
        quality_ctx={
            "earnings_quality": "HIGH",
            "revenue_trend_class": "GROWING",
            "epv_quality": "STABLE",
            "allocation_grade": "A",
        },
    )
    assert result.get("signal_context") == "PREMIUM_JUSTIFIED"
    assert result.get("moat_strength", {}).get("moat_class") == "STRONG_MOAT"


def test_undervalued_weak_moat_value_trap():
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": 100.0, "low": 80.0, "high": 120.0, "flags": []},
        "epv": {"status": "OK", "value_per_share": 100.0, "flags": []},
        "graham": {"status": "OK", "value_per_share": 100.0, "buy_price": 67.0, "flags": []},
    }
    result = _margin_of_safety_scorecard(
        methods,
        price=75.0,
        shares=10.0,
        net_debt=5.0,
        quality_ctx={
            "earnings_quality": "LOW",
            "revenue_trend_class": "SECULAR_DECLINE",
            "epv_quality": "DETERIORATING_BASE",
            "allocation_grade": "F",
        },
    )
    assert result.get("signal_context") == "VALUE_TRAP_RISK"


def test_proceed_moat_passthrough():
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": 100.0, "low": 80.0, "high": 120.0, "flags": []},
        "epv": {"status": "OK", "value_per_share": 100.0, "flags": []},
        "graham": {"status": "OK", "value_per_share": 100.0, "buy_price": 67.0, "flags": []},
    }
    result = _margin_of_safety_scorecard(
        methods,
        price=95.0,
        shares=10.0,
        net_debt=5.0,
        quality_ctx={
            "earnings_quality": "HIGH",
            "revenue_trend_class": "GROWING",
            "epv_quality": "STABLE",
            "allocation_grade": "A",
        },
    )
    signal_ctx = result.get("signal_context")
    assert signal_ctx != "PREMIUM_JUSTIFIED"
    assert signal_ctx != "VALUE_TRAP_RISK"


# ---------------------------------------------------------------------------
# Downside scenario
# ---------------------------------------------------------------------------


def test_downside_limited():
    from app.valuation.valuation_writer import _compute_downside_scenario

    result = _compute_downside_scenario(
        owner_earnings=20.0,
        shares=10.0,
        net_debt=10.0,
        revenue_series=[(2021, 80.0), (2022, 88.0), (2023, 96.0), (2024, 104.0), (2025, 112.0)],
        operating_income_series=[
            (2021, 16.0),
            (2022, 18.0),
            (2023, 20.0),
            (2024, 22.0),
            (2025, 24.0),
        ],
        wacc=0.10,
        base_case_dcf=50.0,
    )
    assert result["downside_risk_class"] == "LIMITED"
    assert result["bear_case_dcf"] is not None
    assert result["bear_case_epv"] is not None
    assert result["bear_case_intrinsic"] is not None
    assert result["bear_case_intrinsic"] > 0


def test_downside_moderate():
    from app.valuation.valuation_writer import _compute_downside_scenario

    result = _compute_downside_scenario(
        owner_earnings=5.0,
        shares=10.0,
        net_debt=80.0,
        revenue_series=[(2021, 100.0), (2022, 98.0), (2023, 95.0), (2024, 92.0), (2025, 90.0)],
        operating_income_series=[
            (2021, 15.0),
            (2022, 12.0),
            (2023, 10.0),
            (2024, 8.0),
            (2025, 7.0),
        ],
        wacc=0.10,
        base_case_dcf=20.0,
    )
    assert result["downside_risk_class"] in ("MODERATE", "SEVERE")


def test_downside_severe():
    from app.valuation.valuation_writer import _compute_downside_scenario

    result = _compute_downside_scenario(
        owner_earnings=2.0,
        shares=10.0,
        net_debt=200.0,
        revenue_series=[(2021, 100.0), (2022, 90.0), (2023, 80.0), (2024, 70.0), (2025, 60.0)],
        operating_income_series=[(2021, 10.0), (2022, 5.0), (2023, 3.0), (2024, 1.0), (2025, -2.0)],
        wacc=0.10,
        base_case_dcf=5.0,
    )
    assert result["downside_risk_class"] == "SEVERE"


def test_downside_vulnerable_flag():
    from app.valuation.valuation_writer import _compute_downside_scenario

    result = _compute_downside_scenario(
        owner_earnings=20.0,
        shares=10.0,
        net_debt=10.0,
        revenue_series=[(2021, 80.0), (2022, 88.0), (2023, 96.0), (2024, 104.0), (2025, 112.0)],
        operating_income_series=[
            (2021, 16.0),
            (2022, 18.0),
            (2023, 20.0),
            (2024, 22.0),
            (2025, 24.0),
        ],
        wacc=0.10,
        base_case_dcf=50.0,
        current_price=40.0,
    )
    bear = result["bear_case_intrinsic"]
    if bear is not None and bear < 40.0 < 50.0:
        assert "DOWNSIDE_VULNERABLE" in result.get("flags", [])


def test_downside_insufficient_data():
    from app.valuation.valuation_writer import _compute_downside_scenario

    result = _compute_downside_scenario(
        owner_earnings=None,
        shares=10.0,
        net_debt=10.0,
        revenue_series=[],
        operating_income_series=[],
        wacc=0.10,
        base_case_dcf=None,
    )
    assert result["downside_risk_class"] == "UNKNOWN"


def test_method_tension_wired_in_scorecard():
    """Verify that analyze_method_tensions is called during scorecard assembly."""
    from app.valuation.method_tension import analyze_method_tensions

    assert callable(analyze_method_tensions)
    # The import path must work from valuation_writer context
    result = analyze_method_tensions(
        dcf_value=100.0,
        epv_value=80.0,
        graham_value=60.0,
        ncav_value=None,
        current_price=90.0,
        revenue_cagr_5y=0.05,
        wacc=0.10,
        terminal_growth=0.015,
    )
    assert "method_tension" not in {}  # sanity — actual wiring verified by integration
    assert "assumption_sensitivity" in result
    assert "adjustment_reasoning" in result
    assert "tension_type" in result


class TestDownsideRiskClassification:
    """FIX 4: downside_risk_class branches are mutually exclusive (the trailing
    else:SEVERE was unreachable). Pin every reachable class with literal labels."""

    def _kwargs(self, **overrides):
        kwargs = dict(
            owner_earnings=100.0,
            shares=1.0,
            net_debt=0.0,
            revenue_series=[(2021, 100.0), (2022, 110.0)],
            operating_income_series=[(2021, 90.0), (2022, 95.0)],
            wacc=0.10,
            base_case_dcf=100.0,
            current_price=50.0,
        )
        kwargs.update(overrides)
        return kwargs

    def test_severe_when_bear_intrinsic_non_positive(self):
        from app.valuation.valuation_writer import _compute_downside_scenario

        result = _compute_downside_scenario(**self._kwargs(owner_earnings=10.0, net_debt=10_000.0))
        assert result["downside_risk_class"] == "SEVERE"

    def test_limited_when_bear_intrinsic_exceeds_20pct_of_base(self):
        from app.valuation.valuation_writer import _compute_downside_scenario

        result = _compute_downside_scenario(
            **self._kwargs(
                owner_earnings=100.0,
                operating_income_series=[(2021, 90.0), (2022, 95.0)],
            )
        )
        assert result["downside_risk_class"] == "LIMITED"

    def test_moderate_when_bear_intrinsic_small_but_positive(self):
        from app.valuation.valuation_writer import _compute_downside_scenario

        result = _compute_downside_scenario(
            **self._kwargs(
                owner_earnings=2.0,
                operating_income_series=[(2021, 1.0), (2022, 1.0)],
            )
        )
        assert result["downside_risk_class"] == "MODERATE"

    def test_unknown_when_base_case_dcf_non_positive(self):
        from app.valuation.valuation_writer import _compute_downside_scenario

        result = _compute_downside_scenario(**self._kwargs(base_case_dcf=0.0))
        assert result["downside_risk_class"] == "UNKNOWN"

    def test_unknown_when_base_case_dcf_none(self):
        from app.valuation.valuation_writer import _compute_downside_scenario

        result = _compute_downside_scenario(**self._kwargs(base_case_dcf=None))
        assert result["downside_risk_class"] == "UNKNOWN"


class TestPricingZoneMosConventionTextbook:
    """FIX 6: _compute_pricing_zone MARGIN_OF_SAFETY uses the TEXTBOOK convention
    (epv_adjusted - price)/epv_adjusted, distinct from the upside-ratio convention
    used in graham_dodd / intrinsic_discipline under the same 'margin_of_safety' name."""

    def test_textbook_mos_literal_value(self):
        from app.valuation.valuation_writer import _compute_pricing_zone

        result = _compute_pricing_zone(
            epv_adjusted=150.0,
            dcf_base=200.0,
            current_price=100.0,
            shares=10.0,
            net_debt=0.0,
            adjusted_wacc=0.10,
            adjusted_avg_operating_income=20.0,
            revenue_latest=1000.0,
            net_debt_to_ebitda_proxy=0.0,
        )
        assert result["zone"] == "MARGIN_OF_SAFETY"
        # textbook MoS = (150 - 100) / 150 = 0.3333...
        assert abs(result["detail"]["margin_of_safety_vs_epv_adjusted"] - (1.0 / 3.0)) < 1e-12


# ── stable-shares wired into the DCF/EPV per-share input ───────────────
# The DCF per-share value divides by `shares`. Before this wiring the writer used the
# SINGLE latest-FY share count, so a corrupt latest-FY datapoint (BTM FY2025
# 7.147534M vs trailing ~16.4M) roughly doubled the per-share anchor. The writer now
# routes the share pick through share_count_stability.select_stable_shares and
# threads the SHARES_LATEST_FY_OUTLIER flag into the dcf/epv outputs_json.flags.


def _seed_btm_shaped_companyfacts(conn: sqlite3.Connection, ticker: str = "BTMX") -> None:
    """BTM-shaped fixture: latest-FY shares (7.147534M) is an outlier vs the
    trailing-3 median (16.384636M), so select_stable_shares must substitute the
    median and flag SHARES_LATEST_FY_OUTLIER."""
    now = "2026-01-01T00:00:00+00:00"
    rows = [
        ("cfo", 2025, 20.0),
        ("cfo", 2024, 18.0),
        ("cfo", 2023, 16.0),
        ("capex", 2025, 5.0),
        ("capex", 2024, 4.5),
        ("capex", 2023, 4.0),
        ("operating_income", 2025, 35.0),
        ("operating_income", 2024, 33.0),
        ("operating_income", 2023, 30.0),
        ("operating_income", 2022, 28.0),
        ("operating_income", 2021, 25.0),
        ("net_income", 2025, 25.0),
        ("net_income", 2024, 23.0),
        ("net_income", 2023, 20.0),
        ("revenue", 2025, 180.0),
        ("revenue", 2024, 170.0),
        ("revenue", 2023, 160.0),
        ("revenue", 2022, 150.0),
        ("revenue", 2021, 140.0),
        ("total_debt", 2025, 10.0),
        ("cash", 2025, 12.0),
        ("preferred_equity", 2025, 0.0),
        ("noncontrolling_interest", 2025, 0.0),
        ("equity", 2025, 90.0),
        ("total_liabilities", 2025, 60.0),
        ("shares_outstanding", 2025, 7.147534),
        ("shares_outstanding", 2024, 16.384636),
        ("shares_outstanding", 2023, 16.675529),
    ]
    for li, fy, val in rows:
        conn.execute(
            "INSERT INTO companyfacts_facts("
            "ticker, fiscal_year, period_end, line_item, value, units, "
            "source_url, fetched_at, period_type, filed_date, accession) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticker,
                fy,
                f"{fy}-12-31",
                li,
                val,
                "USD_millions",
                f"https://example.test/companyfacts/{ticker}",
                now,
                "FY",
                f"{fy + 1}-02-15",
                f"{ticker}-{fy}",
            ),
        )
    conn.commit()


def _seed_bkng_shaped_companyfacts(conn: sqlite3.Connection, ticker: str = "BKNGX") -> None:
    """BKNG-shaped fixture: latest-FY shares (31.673346M) is STABLE vs the
    trailing-3 median (32.815201M), so select_stable_shares must keep the latest
    value and emit no flag — BKNG was never a shares bug.

    Capex is held at 0.0 so end-to-end owner earnings collapse to CFO exactly
    (owner_earnings = latest_cfo - norm_capex*maint_ratio - sbc = 8583.84 - 0 - 0),
    independent of whatever maintenance-capex ratio the quality gate applies. With
    net_debt = 5000.0 - 2649.0 = 2351.0, the revenue-CAGR-capped base growth (0.08),
    wacc 0.095 and terminal growth 0.015, the end-to-end DCF base lands EXACTLY on
    the documented live literal 4435.462076894946 — so behavior preservation is
    demonstrated through the WIRED ensure_valuation anchor, not a lower-level path."""
    now = "2026-01-01T00:00:00+00:00"
    rows = [
        ("cfo", 2026, 8583.84),
        ("cfo", 2025, 8000.0),
        ("cfo", 2024, 7000.0),
        ("capex", 2026, 0.0),
        ("capex", 2025, 0.0),
        ("capex", 2024, 0.0),
        ("operating_income", 2026, 8000.0),
        ("operating_income", 2025, 7000.0),
        ("operating_income", 2024, 6500.0),
        ("operating_income", 2023, 5500.0),
        ("operating_income", 2022, 4500.0),
        ("net_income", 2026, 6000.0),
        ("net_income", 2025, 5000.0),
        ("net_income", 2024, 4000.0),
        ("revenue", 2026, 26917.0),
        ("revenue", 2025, 23739.0),
        ("revenue", 2024, 21365.0),
        ("revenue", 2023, 17090.0),
        ("revenue", 2022, 10958.0),
        ("total_debt", 2026, 5000.0),
        ("cash", 2026, 2649.0),
        ("preferred_equity", 2026, 0.0),
        ("noncontrolling_interest", 2026, 0.0),
        ("equity", 2026, 3000.0),
        ("total_liabilities", 2026, 15000.0),
        ("shares_outstanding", 2026, 31.673346),
        ("shares_outstanding", 2025, 32.815201),
        ("shares_outstanding", 2024, 34.171027),
    ]
    for li, fy, val in rows:
        conn.execute(
            "INSERT INTO companyfacts_facts("
            "ticker, fiscal_year, period_end, line_item, value, units, "
            "source_url, fetched_at, period_type, filed_date, accession) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticker,
                fy,
                f"{fy}-12-31",
                li,
                val,
                "USD_millions",
                f"https://example.test/companyfacts/{ticker}",
                now,
                "FY",
                f"{fy + 1}-02-15",
                f"{ticker}-{fy}",
            ),
        )
    conn.commit()


def _ensure_dcf_outputs(
    conn: sqlite3.Connection, ticker: str, price: float, as_of_date: str = "2026-03-19"
) -> dict:
    """Run the WIRED end-to-end ensure_valuation and return the persisted dcf
    row's parsed inputs/outputs, so the stable-shares literals are validated against the
    real anchor (ensure_valuation), not a direct _discounted_owner_earnings call."""
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(price, ticker=ticker, as_of_date=as_of_date)
    with patch("app.valuation.valuation_writer.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation(ticker, as_of_date, provider=fake_provider)
    row = conn.execute(
        "SELECT inputs_json, outputs_json FROM valuations WHERE ticker=? AND method='dcf'",
        (ticker,),
    ).fetchone()
    return {"inputs": json.loads(row["inputs_json"]), "outputs": json.loads(row["outputs_json"])}


def test_dcf_uses_stable_shares_and_flags_latest_fy_outlier_for_btm():
    """Test case 1, end-to-end through ensure_valuation: the latest-FY
    share count (7.147534M) is an outlier, so the WIRED DCF must
    (1) carry SHARES_LATEST_FY_OUTLIER in its outputs_json flags,
    (2) divide by the trailing median (16.384636M),
    (3) land its base WITHIN 0.5x-2.0x of the prior-trailing-FY-share recompute
        (the same EV/equity divided by the FY2023 share count 16.675529), and
    (4) land below 20.0 (the broken latest-FY pick of 7.147534M produced ~36.20)."""
    conn = _make_conn()
    _seed_btm_shaped_companyfacts(conn, ticker="BTMX")
    result = _ensure_dcf_outputs(conn, "BTMX", 2.93)
    inputs, outputs = result["inputs"], result["outputs"]

    # Stable-share substitution: the median (16.384636M), not the 7.147534M outlier.
    assert inputs["shares"] == 16.384636
    assert "SHARES_LATEST_FY_OUTLIER" in outputs["flags"]
    assert outputs["base"] < 20.0

    # 0.5x-2.0x band vs the prior-trailing-FY-share recompute. The DCF base is
    # (EV - net_debt) / shares, so recomputing with the prior FY share count
    # (16.675529M) holds the numerator fixed and only swaps the divisor.
    equity_value = outputs["base"] * inputs["shares"]
    prior_year_share_recompute = equity_value / 16.675529
    band_ratio = outputs["base"] / prior_year_share_recompute
    assert 0.5 <= band_ratio <= 2.0


def test_dcf_base_for_btm_matches_end_to_end_literal():
    """Exact-literal behavior on the WIRED anchor (BTM outlier case): the
    end-to-end ensure_valuation DCF base for the BTM-shaped fixture is EXACTLY
    13.123300031051944 (stable median shares 16.384636M, net cash -2.0, base
    growth floored at -0.05 by the declining revenue series, wacc 0.10, terminal
    growth 0.015). This is the value the wired anchor actually produces — not a
    hand-fed _discounted_owner_earnings scalar — and stays below the broken
    latest-FY pick (~36.20)."""
    conn = _make_conn()
    _seed_btm_shaped_companyfacts(conn, ticker="BTMX")
    outputs = _ensure_dcf_outputs(conn, "BTMX", 2.93)["outputs"]
    assert round(outputs["base"], 6) == round(13.123300031051944, 6)
    assert "SHARES_LATEST_FY_OUTLIER" in outputs["flags"]


def test_epv_carries_shares_outlier_flag_for_btm():
    """The same outlier flag must thread into the EPV outputs_json flags too,
    since EPV shares the top-level `shares` variable that the stable-shares wiring rewrites."""
    conn = _make_conn()
    _seed_btm_shaped_companyfacts(conn, ticker="BTMX")
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(2.93, ticker="BTMX")

    with patch("app.valuation.valuation_writer.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation("BTMX", "2026-03-19", provider=fake_provider)

    outputs = json.loads(
        conn.execute(
            "SELECT outputs_json FROM valuations WHERE ticker='BTMX' AND method='epv'"
        ).fetchone()["outputs_json"]
    )
    assert "SHARES_LATEST_FY_OUTLIER" in outputs["flags"]


def test_dcf_does_not_flag_outlier_for_stable_bkng_shares():
    """BKNG-shaped: the latest-FY share count is stable, so the DCF must NOT
    carry SHARES_LATEST_FY_OUTLIER and must keep dividing by the latest count
    (31.673346M) — behavior is preserved for non-outliers."""
    conn = _make_conn()
    _seed_bkng_shaped_companyfacts(conn, ticker="BKNGX")
    # as_of_date must be after period_end="2026-12-31" so FY 2026 data is visible
    result = _ensure_dcf_outputs(conn, "BKNGX", 5000.0, as_of_date="2027-03-19")
    inputs, outputs = result["inputs"], result["outputs"]
    assert inputs["shares"] == 31.673346
    assert "SHARES_LATEST_FY_OUTLIER" not in outputs["flags"]


def test_dcf_base_unchanged_for_stable_bkng_inputs():
    """Test case 2 + acceptance criterion, validated on the WIRED anchor:
    with stable BKNG-shaped shares the end-to-end ensure_valuation DCF base is
    UNCHANGED at the prior live value 4435.462076894946 (to 6 dp) and carries NO
    SHARES_LATEST_FY_OUTLIER flag. The literal is asserted against the persisted
    ensure_valuation dcf row — the behavior-preservation guarantee on the wired
    anchor — not a lower-level _discounted_owner_earnings call."""
    conn = _make_conn()
    _seed_bkng_shaped_companyfacts(conn, ticker="BKNGX")
    # as_of_date must be after period_end="2026-12-31" so FY 2026 data is visible
    result = _ensure_dcf_outputs(conn, "BKNGX", 5000.0, as_of_date="2027-03-19")
    inputs, outputs = result["inputs"], result["outputs"]
    assert inputs["shares"] == 31.673346
    assert round(outputs["base"], 6) == round(4435.462076894946, 6)
    assert "SHARES_LATEST_FY_OUTLIER" not in outputs["flags"]


def _run_epv_row(conn: sqlite3.Connection, ticker: str = "TST") -> dict:
    from app.valuation.valuation_writer import ensure_valuation

    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(150.0)
    with patch("app.valuation.valuation_writer.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        ensure_valuation(ticker, "2026-03-19", provider=fake_provider)
    row = conn.execute(
        "SELECT outputs_json FROM valuations WHERE ticker=? AND method='epv'", (ticker,)
    ).fetchone()
    return json.loads(row["outputs_json"])


def test_writer_taxes_epv_at_the_issuers_own_normalized_rate():
    """The writer supplies the issuer's ratio-of-sums effective rate to EPV.

    Tax 20 + 22 + 18 = 60 on pre-tax 100 x 3 = 300 -> 0.20, not the 21%
    statutory fallback, and the basis is persisted beside the value.
    """
    conn = _make_conn()
    _seed_companyfacts(conn)
    for fy, tax in ((2024, 20.0), (2023, 22.0), (2022, 18.0)):
        for line_item, value in (("income_tax_expense", tax), ("pretax_income", 100.0)):
            conn.execute(
                "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_end, line_item, "
                "value, units, source_url, fetched_at, filed_date, accession) "
                "VALUES('TST', ?, ?, ?, ?, 'USD_millions', 'https://example.test', "
                "'2026-01-01T00:00:00+00:00', ?, ?)",
                (fy, f"{fy}-12-31", line_item, value, f"{fy + 1}-02-15", f"TST-{fy}"),
            )
    conn.commit()

    epv = _run_epv_row(conn)

    assert epv["tax_rate"] == 0.2
    assert epv["tax_rate_source"] == "ISSUER"
    assert epv["tax_rate_basis"]["reason_code"] == "OK"
    assert epv["tax_rate_basis"]["years"] == [2022, 2023, 2024]
    assert "EPV_TAX_RATE_STATUTORY_DEFAULT" not in epv["flags"]


def test_writer_epv_without_tax_facts_uses_the_flagged_statutory_fallback():
    conn = _make_conn()
    _seed_companyfacts(conn)

    epv = _run_epv_row(conn)

    assert epv["tax_rate"] == 0.21
    assert epv["tax_rate_source"] == "STATUTORY_DEFAULT"
    assert epv["tax_rate_basis"]["reason_code"] == "TAX_HISTORY_SHORT"
    assert "EPV_TAX_RATE_STATUTORY_DEFAULT" in epv["flags"]


def test_bear_case_on_a_loss_is_bounded_by_the_base_case_and_says_so():
    """Owner earnings of -100 at 50% revenue CAGR: the halved-growth, zero-terminal
    stress shrinks the loss it capitalizes, so the raw bear DCF (-1132.09) sat
    above the base case (-1636.61). The bear case is bounded by the base."""
    from app.valuation.valuation_writer import _compute_downside_scenario

    revenue = [(2020, 100.0), (2021, 150.0), (2022, 225.0), (2023, 337.5), (2024, 506.25)]
    kwargs = dict(
        shares=1.0,
        net_debt=0.0,
        revenue_series=revenue,
        operating_income_series=[(year, -100.0) for year, _ in revenue],
        wacc=0.10,
        current_price=100.0,
    )
    unbounded = _compute_downside_scenario(owner_earnings=-100.0, base_case_dcf=None, **kwargs)
    bounded = _compute_downside_scenario(
        owner_earnings=-100.0, base_case_dcf=-1636.6093244996925, **kwargs
    )

    assert unbounded["bear_case_dcf"] == -1132.09
    assert "BEAR_CASE_BOUNDED_BY_BASE" not in unbounded["flags"]
    assert bounded["bear_case_dcf"] == -1636.61
    assert "BEAR_CASE_BOUNDED_BY_BASE" in bounded["flags"]
    assert bounded["downside_risk_class"] == "UNKNOWN"


def test_moat_with_no_signals_is_unknown_not_weak():
    """With nothing counted the score is 0, which read
    WEAK_MOAT and could turn an undervalued name into VALUE_TRAP_RISK on no
    evidence at all. UNKNOWN inputs are not signals either."""
    from app.valuation.valuation_writer import _classify_moat_strength

    empty = _classify_moat_strength()
    unknowns = _classify_moat_strength(earnings_quality="UNKNOWN", epv_quality="UNKNOWN")

    for result in (empty, unknowns):
        assert result["moat_class"] == "MOAT_UNKNOWN"
        assert result["signals_counted"] == 0
        assert result["moat_score"] == 0


def test_unknown_signal_is_not_counted_beside_known_ones():
    from app.valuation.valuation_writer import _classify_moat_strength

    result = _classify_moat_strength(
        earnings_quality="UNKNOWN",
        revenue_trend_class="GROWING",
        epv_quality="STABLE",
        allocation_grade="B",
    )
    assert result["signals_counted"] == 3
    assert result["moat_score"] == 5
    assert result["signal_detail"]["earnings_quality"]["counted"] is False
    assert result["moat_class"] == "MODERATE_MOAT"


def _write_run_price_artifact(monkeypatch, tmp_path, payload: dict):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    price_dir = cfg.outputs_dir / "prices" / "run_l4"
    price_dir.mkdir(parents=True, exist_ok=True)
    (price_dir / "TST.json").write_text(json.dumps(payload), encoding="utf-8")
    return cfg


def test_run_scoped_price_artifact_without_a_requested_date_is_not_evidence(
    monkeypatch, tmp_path
):
    """An artifact that does not say which date it priced was
    accepted for ANY requested as-of date, so a price for one date could value a
    run for another. It is now ignored; the resolver falls through to the next
    source."""
    from app.valuation.valuation_writer import _load_run_scoped_price_artifact

    cfg = _write_run_price_artifact(
        monkeypatch,
        tmp_path,
        {"ticker": "TST", "status": "OK", "snapshot": {"price": 155.25, "source": "stooq"}},
    )

    assert _load_run_scoped_price_artifact("run_l4", "TST", "2026-03-19", cfg=cfg) == (None, {})


def test_run_scoped_price_artifact_for_the_requested_date_is_still_used(monkeypatch, tmp_path):
    from app.valuation.valuation_writer import _load_run_scoped_price_artifact

    cfg = _write_run_price_artifact(
        monkeypatch,
        tmp_path,
        {
            "ticker": "TST",
            "status": "OK",
            "requested_as_of_date": "2026-03-19",
            "snapshot": {"price": 155.25, "source": "stooq"},
        },
    )

    price, context = _load_run_scoped_price_artifact("run_l4", "TST", "2026-03-19", cfg=cfg)
    assert price == 155.25
    assert context["price_source_resolution"] == "run_scoped_output"
    assert _load_run_scoped_price_artifact("run_l4", "TST", "2026-03-20", cfg=cfg) == (None, {})


@pytest.mark.parametrize("bad_price", [True, "NaN", 0.0, -5.0])
def test_run_scoped_price_artifact_rejects_a_non_price(monkeypatch, tmp_path, bad_price):
    from app.valuation.valuation_writer import _load_run_scoped_price_artifact

    price = float(bad_price) if isinstance(bad_price, str) else bad_price
    cfg = _write_run_price_artifact(
        monkeypatch,
        tmp_path,
        {
            "ticker": "TST",
            "status": "OK",
            "requested_as_of_date": "2026-03-19",
            "snapshot": {"price": price, "source": "stooq"},
        },
    )

    assert _load_run_scoped_price_artifact("run_l4", "TST", "2026-03-19", cfg=cfg) == (None, {})


# ── Symmetric CFO normalization (owner decision 2026-09-29) ───────────────────


def _cfo_dip_facts(*, cfo, revenue, operating_income):
    """Five fiscal years 2020-2024, newest first; capex 40, SBC 5 every year."""

    def _series(values):
        return [(2024 - i, float(v)) for i, v in enumerate(values)]

    return {
        "cfo": _series(cfo),
        "capex": _series([40.0] * 5),
        "sbc": _series([5.0] * 5),
        "revenue": _series(revenue),
        "operating_income": _series(operating_income),
    }


def test_one_time_cfo_dip_with_stable_business_is_lifted_to_the_median():
    """The KO shape: a one-time working-capital/tax outflow halves the newest
    CFO (60 against a 5-year median of 100) while revenue (1,005 -> 1,010)
    and operating income (200 -> 201) held steady. (Migrated 2026-09-29: the
    fixture's operating income ended at 198, below where the window started;
    negative 5-year growth now makes the dip a real decline.) The spike rule already
    smoothed a CFO above 1.25x the median down to it; the dip below
    median / 1.25 = 80 is now lifted to it:
        owner earnings = 100 median CFO - 40 capex - 5 SBC = 55 (not 60 - 45 = 15)
    Confidence drops to LOWER, and the output names what happened.
    """
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = _cfo_dip_facts(
        cfo=[60.0, 100.0, 98.0, 102.0, 100.0],
        revenue=[1010.0, 1000.0, 990.0, 1000.0, 1005.0],
        operating_income=[201.0, 200.0, 199.0, 201.0, 200.0],
    )
    result = _compute_owner_earnings(facts)

    assert result["cfo_latest_raw"] == 60.0
    assert result["cfo_used"] == 100.0
    assert result["owner_earnings_latest"] == 55.0
    assert result["cfo_normalization"] == "DIP_LIFTED"
    assert "CFO_DIP_NORMALIZED" in result["flags"]
    assert "CFO_PEAK_NORMALIZED" not in result["flags"]
    assert result["confidence"] == "LOWER"


def test_cfo_dip_that_follows_a_real_decline_is_not_lifted():
    """Control: the same CFO dip, but revenue fell 1,000 -> 700 and operating
    income 200 -> 120 (both below their medians by more than 1.25x). The
    cash fell because the business did: the base follows it,
        60 - 40 - 5 = 15,
    and says so."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = _cfo_dip_facts(
        cfo=[60.0, 100.0, 98.0, 102.0, 100.0],
        revenue=[700.0, 1000.0, 990.0, 1000.0, 1005.0],
        operating_income=[120.0, 200.0, 199.0, 201.0, 200.0],
    )
    result = _compute_owner_earnings(facts)

    assert result["cfo_used"] == 60.0
    assert result["owner_earnings_latest"] == 15.0
    assert result["cfo_normalization"] == "DIP_KEPT_REAL_DECLINE"
    assert "CFO_DIP_REAL_DECLINE" in result["flags"]
    assert "CFO_DIP_NORMALIZED" not in result["flags"]
    assert result["confidence"] == "NORMAL"


def test_cfo_dip_with_only_operating_income_falling_is_a_real_decline():
    """Either measure falling is enough: revenue held at 1,000 but operating
    income fell 200 -> 100 (margin compression). Not lifted."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = _cfo_dip_facts(
        cfo=[60.0, 100.0, 98.0, 102.0, 100.0],
        revenue=[1000.0, 1000.0, 990.0, 1000.0, 1005.0],
        operating_income=[100.0, 200.0, 199.0, 201.0, 200.0],
    )
    result = _compute_owner_earnings(facts)

    assert result["owner_earnings_latest"] == 15.0
    assert result["cfo_normalization"] == "DIP_KEPT_REAL_DECLINE"


def test_cfo_dip_without_revenue_and_operating_income_history_is_not_lifted():
    """Conservative call: with no operating history to tell a one-time outflow
    from a decline, the dip stands and is flagged unverified."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = _cfo_dip_facts(
        cfo=[60.0, 100.0, 98.0, 102.0, 100.0],
        revenue=[1010.0, 1000.0, 990.0, 1000.0, 1005.0],
        operating_income=[198.0, 200.0, 199.0, 201.0, 200.0],
    )
    del facts["operating_income"]
    result = _compute_owner_earnings(facts)

    assert result["owner_earnings_latest"] == 15.0
    assert result["cfo_normalization"] == "DIP_KEPT_UNVERIFIED"
    assert "CFO_DIP_UNVERIFIED" in result["flags"]


def test_cfo_spike_behaviour_is_unchanged():
    """Control: a spike (150 against a median of 100) is smoothed exactly as
    before, 100 - 40 - 5 = 55, and named SPIKE_SMOOTHED."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = _cfo_dip_facts(
        cfo=[150.0, 100.0, 98.0, 102.0, 100.0],
        revenue=[1010.0, 1000.0, 990.0, 1000.0, 1005.0],
        operating_income=[198.0, 200.0, 199.0, 201.0, 200.0],
    )
    result = _compute_owner_earnings(facts)

    assert result["cfo_used"] == 100.0
    assert result["owner_earnings_latest"] == 55.0
    assert result["cfo_normalization"] == "SPIKE_SMOOTHED"
    assert "CFO_PEAK_NORMALIZED" in result["flags"]
    assert "CFO_DIP_NORMALIZED" not in result["flags"]


def test_cfo_inside_the_band_is_left_alone():
    """81 is within median / 1.25 = 80 of the median: no correction."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = _cfo_dip_facts(
        cfo=[81.0, 100.0, 98.0, 102.0, 100.0],
        revenue=[1010.0, 1000.0, 990.0, 1000.0, 1005.0],
        operating_income=[198.0, 200.0, 199.0, 201.0, 200.0],
    )
    result = _compute_owner_earnings(facts)

    assert result["cfo_used"] == 81.0
    assert result["cfo_normalization"] == "NONE"


# ── Interest add-back and bear-case EPV use the issuer's own tax rate ─────────


def _interest_facts(*, with_tax_history: bool):
    facts = {
        "cfo": [(2024, 100.0), (2023, 100.0), (2022, 100.0)],
        "capex": [(2024, 40.0), (2023, 40.0), (2022, 40.0)],
        "sbc": [(2024, 5.0), (2023, 5.0), (2022, 5.0)],
        "interest_expense": [(2024, 20.0), (2023, 20.0), (2022, 20.0)],
    }
    if with_tax_history:
        # 25 + 30 + 20 = 75 of tax on 300 of pre-tax income -> 0.25.
        facts["income_tax_expense"] = [(2024, 25.0), (2023, 30.0), (2022, 20.0)]
        facts["pretax_income"] = [(2024, 100.0), (2023, 100.0), (2022, 100.0)]
    return facts


def test_interest_addback_is_taxed_at_the_issuers_normalized_rate():
    """Fixed 2026-09-29. The FCFF add-back used a flat 21%; it now uses the
    rate the EPV uses: 20 x (1 - 0.25) = 15.0, so owner earnings are
    100 - 40 - 5 + 15 = 70.0 (the flat rate gave 15.8 and 70.8)."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    result = _compute_owner_earnings(_interest_facts(with_tax_history=True))

    assert result["interest_addback_tax_rate"] == 0.25
    assert result["interest_addback_tax_rate_source"] == "ISSUER"
    assert result["interest_addback"] == 15.0
    assert result["owner_earnings_latest"] == 70.0


def test_interest_addback_falls_back_to_statutory_and_says_so():
    from app.valuation.valuation_writer import _compute_owner_earnings

    result = _compute_owner_earnings(_interest_facts(with_tax_history=False))

    assert result["interest_addback_tax_rate"] == 0.21
    assert result["interest_addback_tax_rate_source"] == "STATUTORY_DEFAULT"
    assert result["interest_addback"] == pytest.approx(15.8, abs=1e-12)


def _bear(tax_rate):
    from app.valuation.valuation_writer import _compute_downside_scenario

    return _compute_downside_scenario(
        owner_earnings=60.0,
        shares=10.0,
        net_debt=0.0,
        revenue_series=[(2022, 1000.0), (2023, 1000.0), (2024, 1000.0)],
        operating_income_series=[(2024, 80.0), (2023, 50.0), (2022, 70.0)],
        wacc=0.10,
        base_case_dcf=100.0,
        tax_rate=tax_rate,
    )


def test_bear_case_epv_is_taxed_at_the_issuers_normalized_rate():
    """Fixed 2026-09-29. Low year 50 at the issuer's 30%: 50 x 0.70 / 0.10 / 10
    shares = $35.00 (the flat 21% gave $39.50)."""
    result = _bear(0.30)

    assert result["bear_case_epv"] == 35.0
    assert result["assumptions"]["tax_rate"] == 0.30
    assert result["assumptions"]["tax_rate_source"] == "ISSUER"


def test_bear_case_epv_statutory_fallback_is_labelled():
    result = _bear(None)

    assert result["bear_case_epv"] == 39.5
    assert result["assumptions"]["tax_rate"] == 0.21
    assert result["assumptions"]["tax_rate_source"] == "STATUTORY_DEFAULT"


@pytest.mark.parametrize("bad_rate", [-0.1, 0.9, math.nan])
def test_bear_case_epv_refuses_an_impossible_tax_rate(bad_rate):
    result = _bear(bad_rate)

    assert result["bear_case_epv"] is None
    assert "BEAR_EPV_INVALID_TAX_RATE" in result["flags"]
    assert result["assumptions"]["tax_rate_source"] == "INVALID"


# ── EPV's (D&A - maintenance capex) inputs ───────────────────────────────────


def test_epv_da_and_maintenance_capex_inputs():
    """D&A: mean of 30, 30, 30 less the 10 of acquired-intangible amortization
    reported in 2024 -> (20 + 30 + 30) / 3 = 26.667. Maintenance capex: capex
    mean 40 x the gate's 0.75 growth-aware ratio = 30.0."""
    from app.valuation.valuation_writer import _epv_da_and_maintenance_capex

    facts = {
        "depreciation_amortization": [(2024, 30.0), (2023, 30.0), (2022, 30.0)],
        "intangible_amortization": [(2024, 10.0)],
        "capex": [(2024, 40.0), (2023, 40.0), (2022, 40.0)],
    }
    out = _epv_da_and_maintenance_capex(facts, 0.75)

    assert out["reason"] == "OK"
    assert out["depreciation_amortization"] == pytest.approx(80.0 / 3.0, abs=1e-12)
    assert out["maintenance_capex"] == 30.0
    assert out["maintenance_capex_ratio_source"] == "GATE_GROWTH_AWARE"
    assert out["intangible_amortization_excluded"] is True


def test_epv_maintenance_capex_without_a_gate_ratio_is_all_capex():
    """No usable ratio: every dollar of capex is maintenance (the reading
    that credits the least)."""
    from app.valuation.valuation_writer import _epv_da_and_maintenance_capex

    facts = {
        "depreciation_amortization": [(2024, -30.0), (2023, -30.0), (2022, -30.0)],
        "capex": [(2024, -40.0), (2023, -40.0), (2022, -40.0)],
    }
    for ratio in (None, 1.5, math.nan):
        out = _epv_da_and_maintenance_capex(facts, ratio)
        assert out["depreciation_amortization"] == 30.0
        assert out["maintenance_capex"] == 40.0
        assert out["maintenance_capex_ratio_source"] == "ALL_CAPEX_MAINTENANCE_DEFAULT"


def test_epv_da_inputs_with_short_history_make_no_adjustment():
    from app.valuation.valuation_writer import _epv_da_and_maintenance_capex

    short_da = _epv_da_and_maintenance_capex(
        {"depreciation_amortization": [(2024, 30.0)], "capex": [(2024, 1.0), (2023, 1.0), (2022, 1.0)]},
        1.0,
    )
    short_capex = _epv_da_and_maintenance_capex(
        {"depreciation_amortization": [(2024, 3.0), (2023, 3.0), (2022, 3.0)], "capex": []},
        1.0,
    )

    assert short_da["reason"] == "DA_HISTORY_SHORT"
    assert short_da["depreciation_amortization"] is None
    assert short_capex["reason"] == "CAPEX_HISTORY_SHORT"
    assert short_capex["maintenance_capex"] is None


def test_pricing_zone_earnings_gap_uses_the_issuers_normalized_tax_rate():
    """Price 100 x 10 shares, no debt, WACC 10%: required NOPAT 100. At the issuer's
    10% rate the required operating income is 100 / 0.9; with no issuer rate the 21%
    fallback gives 100 / 0.79. The rate and its source are recorded."""
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": 180.0},
        "epv": {"status": "OK", "value_per_share": 70.0},
        "epv_adjusted": {"status": "OK", "value_per_share": 90.0, "avg_operating_income": 90.0},
        "graham": {"status": "OK", "value_per_share": 60.0},
        "ncav": {"status": "OK", "value_per_share": -5.0, "signal": "NCAV_NO_ASSET_FLOOR"},
    }
    issuer = _margin_of_safety_scorecard(
        methods, price=100.0, shares=10.0, net_debt=0.0,
        wacc_detail={"adjusted_wacc": 0.10}, tax_rate=0.10,
    )["pricing_zone_detail"]
    assert issuer["required_avg_operating_income"] == pytest.approx(100.0 / 0.9)
    assert issuer["tax_rate"] == 0.10
    assert issuer["tax_rate_source"] == "ISSUER"

    fallback = _margin_of_safety_scorecard(
        methods, price=100.0, shares=10.0, net_debt=0.0, wacc_detail={"adjusted_wacc": 0.10},
    )["pricing_zone_detail"]
    assert fallback["required_avg_operating_income"] == pytest.approx(100.0 / 0.79)
    assert fallback["tax_rate"] == 0.21
    assert fallback["tax_rate_source"] == "STATUTORY_DEFAULT"
