from __future__ import annotations

import json

import pytest

from app.alpha.schemas import TickerSignalPacket
from app.autonomous.sector_financial_packets import (
    _annual_fact_rows,
    _altman_z_interpretation,
    _altman_z_score_from_rows,
    _beneish_interpretation,
    _beneish_m_score_from_rows,
    _expectations_gap_from_db,
    _fcf_yield_metrics,
    _fcf_yield_from_inputs,
    _cash_conversion_cycle_metrics,
    _gross_margin_metrics,
    _historical_fiscal_year_prices,
    _historical_multiple_bands,
    _historical_multiple_bands_from_inputs,
    _normalized_operating_margin_metrics,
    _packet_annual_fact_provenance_rows,
    _packet_annual_fact_rows,
    _piotroski_f_score_from_rows,
    _piotroski_interpretation,
    _percentile_within_history,
    _returns_on_capital_metrics,
    _share_count_cagr_metrics,
    _valuation,
    build_sector_company_financial_packet,
    build_sector_company_financial_packets,
    build_sector_company_financial_packets_from_signal_packets,
)


def _base_packet() -> TickerSignalPacket:
    return TickerSignalPacket(
        ticker="AAA",
        dcf_value=100.0,
        epv_value=70.0,
        current_price=50.0,
        margin_of_safety_verdict="UNDERVALUED",
        gate_verdict="PROCEED",
        confidence_class="HIGH",
        moat_score=5,
        moat_classification="WIDE_MOAT",
        downside_risk_class="LIMITED",
        valuation_supports=["GROWING_REVENUE_SUPPORT"],
        valuation_headwinds=["CUSTOMER_CONCENTRATION_WATCH"],
        peer_position="LEADER",
        roic_vs_median=1.8,
        op_margin_vs_median=1.4,
        revenue_growth_vs_median=1.6,
        filing_risk_status="OK",
        filing_risk_signals={"customer_concentration": "LOW", "summary": "No major filing risk."},
        research_status="OK",
        solvency_risk="LOW",
        method_tension_type="NONE",
        growth_dependency_ratio=0.25,
        methods_agree=True,
        consensus_direction="UNDERVALUED",
        intrinsic_range_low=70.0,
        intrinsic_range_high=115.0,
        quarterly_revenue_trend="STABLE",
        raw_quality_ctx={
            "revenue_cagr_5y": 0.12,
            "revenue_cagr_3y": 0.15,
            "earnings_quality": "HIGH",
            "cash_conversion_ratio": 0.91,
            "fcf_margin": 0.13,
            "dilution_rate_shares_cagr": -0.01,
            "capital_allocation_score": 4.0,
        },
        raw_valuation={"pricing_zone_detail": {"current_price": 50.0, "gate_action": "PROCEED"}},
        research_report={
            "solvency": {
                "risk": "LOW",
                "signals": [],
                "details": "Low leverage and ample interest coverage.",
                "negative_equity": False,
                "current_ratio": 2.1,
                "cash_runway_quarters": None,
                "going_concern_language": False,
                "going_concern_assertions": [
                    {
                        "subject": "INVESTEE",
                        "assertion_mode": "ACCOUNTING_POLICY",
                        "blockable": False,
                        "accession": "0000000001-26-000001",
                        "section": "FINANCIAL_STATEMENTS_NOTES",
                        "excerpt": "We assess investee going-concern indicators.",
                    }
                ],
                "no_assurance_financing": False,
                "debt_due_within_12mo": False,
            }
        },
    )


def test_build_sector_company_financial_packet_maps_core_financial_pillars():
    packet = build_sector_company_financial_packet(_base_packet())

    assert packet.ticker == "AAA"
    assert packet.financial_status == "Financially Viable"
    assert packet.model_fit_status == "VALID_GENERIC"
    assert packet.data_quality_status == "OK"
    assert packet.current_price == 50.0
    assert packet.business_quality["moat_score"] == 5
    assert packet.business_quality["revenue_cagr_5y"] == 0.12
    assert packet.reinvestment["growth_dependency_status"] == "MODERATE_OR_LOW"
    assert packet.returns_on_capital["roic_vs_median"] == 1.8
    assert packet.cash_conversion["cash_conversion_ratio"] == 0.91
    assert packet.balance_sheet["current_ratio"] == 2.1
    assert packet.balance_sheet["no_assurance_financing"] is False
    assert packet.balance_sheet["going_concern_assertions"] == [
        {
            "subject": "INVESTEE",
            "assertion_mode": "ACCOUNTING_POLICY",
            "blockable": False,
            "accession": "0000000001-26-000001",
            "section": "FINANCIAL_STATEMENTS_NOTES",
            "excerpt": "We assess investee going-concern indicators.",
        }
    ]
    assert packet.capital_allocation["dilution_rate_shares_cagr"] == -0.01
    assert packet.accounting_quality["filing_risk_status"] == "OK"
    assert packet.valuation["anchor_method"] == "dcf"
    assert packet.valuation["valuation_anchor"] == 100.0
    assert packet.valuation["discount_to_anchor"] == 0.5
    assert packet.expected_return["status"] == "SCENARIO_REQUIRED"
    assert packet.expected_return["base_anchor_discount"] == 0.5
    assert packet.score_components["discount_to_anchor"] == 0.5
    assert packet.blockers == []
    assert packet.confidence_caps == []


def test_returns_on_capital_computes_roic_spread_and_incremental_roic(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._annual_fact_rows",
        lambda ticker, **_kwargs: {
            2022: {
                "operating_income": 80.0,
                "income_tax_expense": 20.0,
                "pretax_income": 100.0,
                "total_debt": 100.0,
                "equity": 300.0,
                "cash": 50.0,
            },
            2023: {
                "operating_income": 90.0,
                "income_tax_expense": 21.0,
                "pretax_income": 100.0,
                "total_debt": 150.0,
                "equity": 350.0,
                "cash": 50.0,
            },
            2024: {
                "operating_income": 95.0,
                "income_tax_expense": 22.0,
                "pretax_income": 100.0,
                "total_debt": 180.0,
                "equity": 425.0,
                "cash": 55.0,
            },
            2025: {
                "operating_income": 100.0,
                "income_tax_expense": 25.0,
                "pretax_income": 100.0,
                "total_debt": 200.0,
                "equity": 500.0,
                "cash": 50.0,
            },
        },
    )

    metrics = _returns_on_capital_metrics("AAA", as_of_date="2026-04-26")

    assert metrics["roic"] == 0.11538461538461539
    assert metrics["roic_wacc_spread"] == 0.015384615384615385
    assert metrics["incremental_roic_3y"] == 0.03666666666666667
    assert metrics["roic_trajectory_5y"][-1]["effective_tax_rate"] == 0.25
    assert metrics["roic_not_computable_reasons"] == []


def test_returns_on_capital_surfaces_not_computable_reason(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._annual_fact_rows",
        lambda ticker, **_kwargs: {
            2025: {
                "operating_income": 100.0,
                "total_debt": 200.0,
                "equity": 500.0,
                "cash": 50.0,
            },
        },
    )

    metrics = _returns_on_capital_metrics("AAA", as_of_date="2026-04-26")

    assert metrics["roic"] is None
    assert metrics["roic_wacc_spread"] is None
    assert metrics["roic_not_computable_reasons"] == [
        "ROIC_NOT_COMPUTABLE",
        "TRAILING_EFFECTIVE_TAX_RATE_MISSING",
        "INCREMENTAL_ROIC_HISTORY_INSUFFICIENT",
    ]


def test_fcf_yield_from_inputs_uses_literal_hand_computed_case():
    fcf, fcf_yield = _fcf_yield_from_inputs(120.0, 20.0, 2000.0)

    assert fcf == 100.0
    assert fcf_yield == 0.05


@pytest.mark.parametrize(
    ("cfo", "capex", "market_cap_mm"),
    [
        (float("nan"), 20.0, 2000.0),
        (float("inf"), 20.0, 2000.0),
        (120.0, float("nan"), 2000.0),
        (120.0, float("-inf"), 2000.0),
        (120.0, 20.0, float("nan")),
        (120.0, 20.0, float("inf")),
    ],
    ids=("cfo-nan", "cfo-inf", "capex-nan", "capex-neg-inf", "cap-nan", "cap-inf"),
)
def test_fcf_yield_from_inputs_rejects_non_finite_values(cfo, capex, market_cap_mm):
    assert _fcf_yield_from_inputs(cfo, capex, market_cap_mm) == (None, None)


def test_fcf_yield_from_inputs_rejects_non_finite_derived_result():
    assert _fcf_yield_from_inputs(-1e308, 1e308, 1.0) == (None, None)


@pytest.mark.parametrize(
    ("market_cap_mm", "expected_yield"),
    [
        (3_460_000.0, 0.03050693641618497),
        (346_000.0, 0.3050693641618497),
    ],
)
def test_v2_fcf_yield_preserves_market_cap_millions_contract(
    monkeypatch, market_cap_mm, expected_yield
):
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._packet_annual_fact_rows",
        lambda *args, **kwargs: {2025: {"cfo": 118_254.0, "capex": 12_700.0}},
    )

    metrics = _fcf_yield_metrics(
        "MEGA",
        as_of_date="2026-07-20",
        v2_data_plane=True,
        market_cap_override_mm=market_cap_mm,
    )

    assert metrics["ttm_fcf"] == 105_554.0
    assert metrics["market_cap"] == market_cap_mm
    assert metrics["fcf_yield"] == expected_yield
    assert metrics["fcf_yield_reasons"] == []


@pytest.mark.parametrize(
    ("cfo", "capex", "market_cap_mm", "expected_reasons"),
    [
        (float("nan"), 12_700.0, 3_460_000.0, ["CFO_NON_FINITE"]),
        (float("inf"), 12_700.0, 3_460_000.0, ["CFO_NON_FINITE"]),
        (118_254.0, float("nan"), 3_460_000.0, ["CAPEX_NON_FINITE"]),
        (118_254.0, float("-inf"), 3_460_000.0, ["CAPEX_NON_FINITE"]),
        (118_254.0, 12_700.0, float("nan"), ["MARKET_CAP_NON_FINITE"]),
        (118_254.0, 12_700.0, float("inf"), ["MARKET_CAP_NON_FINITE"]),
    ],
    ids=("cfo-nan", "cfo-inf", "capex-nan", "capex-neg-inf", "cap-nan", "cap-inf"),
)
def test_v2_fcf_yield_rejects_non_finite_inputs_with_exact_reason(
    monkeypatch, cfo, capex, market_cap_mm, expected_reasons
):
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._packet_annual_fact_rows",
        lambda *args, **kwargs: {2025: {"cfo": cfo, "capex": capex}},
    )

    metrics = _fcf_yield_metrics(
        "MEGA",
        as_of_date="2026-07-20",
        v2_data_plane=True,
        market_cap_override_mm=market_cap_mm,
    )

    assert metrics["ttm_fcf"] is None
    assert metrics["fcf_yield"] is None
    assert metrics["fcf_yield_reasons"] == expected_reasons


def test_v2_fcf_yield_rejects_non_finite_derived_result(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._packet_annual_fact_rows",
        lambda *args, **kwargs: {2025: {"cfo": -1e308, "capex": 1e308}},
    )

    metrics = _fcf_yield_metrics(
        "MEGA",
        as_of_date="2026-07-20",
        v2_data_plane=True,
        market_cap_override_mm=1.0,
    )

    assert metrics["ttm_fcf"] is None
    assert metrics["fcf_yield"] is None
    assert metrics["fcf_yield_reasons"] == ["FCF_YIELD_NON_FINITE"]


def test_gross_margin_computes_from_revenue_and_cogs(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._annual_fact_rows",
        lambda ticker, **_kwargs: {2025: {"revenue": 1000.0, "cost_of_revenue": 620.0}},
    )

    metrics = _gross_margin_metrics("AAA", as_of_date="2026-05-07")

    assert metrics["gross_margin"] == 0.38
    assert metrics["gross_margin_trajectory_5y"] == [
        {
            "fiscal_year": 2025,
            "gross_margin": 0.38,
            "gross_profit": 380.0,
            "cost_of_revenue": 620.0,
            "cost_of_revenue_source": "reported_cost_of_revenue",
            "not_computable_reasons": [],
        }
    ]
    assert metrics["gross_margin_not_computable_reasons"] == []


def test_gross_margin_computes_from_gross_profit(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._annual_fact_rows",
        lambda ticker, **_kwargs: {2025: {"revenue": 800.0, "gross_profit": 320.0}},
    )

    metrics = _gross_margin_metrics("AAA", as_of_date="2026-05-07")

    assert metrics["gross_margin"] == 0.4
    assert metrics["gross_margin_trajectory_5y"][0]["gross_profit"] == 320.0
    assert metrics["gross_margin_trajectory_5y"][0]["cost_of_revenue"] == 480.0
    assert metrics["gross_margin_trajectory_5y"][0]["cost_of_revenue_source"] == "gross_profit"


def test_gross_margin_missing_inputs_returns_reason(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._annual_fact_rows",
        lambda ticker, **_kwargs: {2025: {"revenue": 500.0}},
    )

    metrics = _gross_margin_metrics("AAA", as_of_date="2026-05-07")

    assert metrics["gross_margin"] is None
    assert metrics["gross_margin_trajectory_5y"][0]["not_computable_reasons"] == [
        "GROSS_MARGIN_NOT_COMPUTABLE",
        "GROSS_PROFIT_OR_COGS_MISSING",
    ]
    assert metrics["gross_margin_not_computable_reasons"] == [
        "GROSS_MARGIN_NOT_COMPUTABLE",
        "GROSS_PROFIT_OR_COGS_MISSING",
    ]


def test_normalized_operating_margin_uses_hand_computed_5y_average(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._annual_fact_rows",
        lambda ticker, **_kwargs: {
            2021: {"revenue": 100.0, "operating_income": 10.0},
            2022: {"revenue": 100.0, "operating_income": 20.0},
            2023: {"revenue": 100.0, "operating_income": 15.0},
            2024: {"revenue": 200.0, "operating_income": 40.0},
            2025: {"revenue": 100.0, "operating_income": 8.0},
        },
    )

    metrics = _normalized_operating_margin_metrics("AAA", as_of_date="2026-05-10")

    assert metrics["normalized_operating_margin"] == pytest.approx(0.146, abs=1e-6)
    assert metrics["latest_operating_margin"] == 0.08
    assert metrics["normalized_operating_margin_status"] == "OK"
    assert metrics["normalized_operating_margin_not_computable_reasons"] == []
    assert metrics["operating_margin_trajectory_5y"] == [
        {"fiscal_year": 2021, "operating_margin": 0.1, "not_computable_reasons": []},
        {"fiscal_year": 2022, "operating_margin": 0.2, "not_computable_reasons": []},
        {"fiscal_year": 2023, "operating_margin": 0.15, "not_computable_reasons": []},
        {"fiscal_year": 2024, "operating_margin": 0.2, "not_computable_reasons": []},
        {"fiscal_year": 2025, "operating_margin": 0.08, "not_computable_reasons": []},
    ]


def test_normalized_operating_margin_requires_three_years(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._annual_fact_rows",
        lambda ticker, **_kwargs: {
            2022: {"revenue": 100.0, "operating_income": 10.0},
            2023: {"revenue": 100.0},
            2024: {"revenue": 0.0, "operating_income": 15.0},
            2025: {"revenue": 100.0, "operating_income": 20.0},
        },
    )

    metrics = _normalized_operating_margin_metrics("AAA", as_of_date="2026-05-10")

    assert metrics["normalized_operating_margin"] is None
    assert metrics["latest_operating_margin"] == 0.2
    assert metrics["normalized_operating_margin_status"] == "NORMALIZED_MARGIN_INSUFFICIENT_HISTORY"
    assert metrics["normalized_operating_margin_not_computable_reasons"] == [
        "NORMALIZED_MARGIN_INSUFFICIENT_HISTORY",
        "OPERATING_MARGIN_NOT_COMPUTABLE",
        "OPERATING_INCOME_MISSING",
        "REVENUE_NONPOSITIVE",
    ]


def test_cash_conversion_cycle_computes_components(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._annual_fact_rows",
        lambda ticker, **_kwargs: {
            2025: {
                "revenue": 1000.0,
                "gross_profit": 400.0,
                "accounts_receivable": 100.0,
                "inventory": 90.0,
                "accounts_payable": 60.0,
            }
        },
    )

    metrics = _cash_conversion_cycle_metrics("AAA", as_of_date="2026-05-07")

    assert metrics["cash_conversion_cycle"] == 54.75
    assert metrics["days_sales_outstanding"] == 36.5
    assert metrics["days_inventory_outstanding"] == 54.75
    assert metrics["days_payable_outstanding"] == 36.5
    assert metrics["cash_conversion_cycle_not_computable_reasons"] == []


def test_cash_conversion_cycle_missing_component_lists_specific_input(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._annual_fact_rows",
        lambda ticker, **_kwargs: {
            2025: {
                "revenue": 1000.0,
                "gross_profit": 400.0,
                "accounts_receivable": 100.0,
                "inventory": 90.0,
            }
        },
    )

    metrics = _cash_conversion_cycle_metrics("AAA", as_of_date="2026-05-07")

    assert metrics["cash_conversion_cycle"] is None
    assert metrics["days_sales_outstanding"] == 36.5
    assert metrics["days_inventory_outstanding"] == 54.75
    assert metrics["days_payable_outstanding"] is None
    assert metrics["cash_conversion_cycle_not_computable_reasons"] == [
        "CCC_NOT_COMPUTABLE",
        "ACCOUNTS_PAYABLE_MISSING",
    ]


def test_share_count_cagr_buyback_case(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._packet_annual_fact_rows",
        lambda ticker, as_of_date, v2_data_plane: {
            year: {
                "shares_outstanding": value,
                "shares_outstanding_split_lineage": {
                    "raw_value": value,
                    "normalized_value": value,
                    "shares_basis": "UNADJUSTED",
                    "split_adjustment_factor": 1.0,
                    "split_effective_date": None,
                },
            }
            for year, value in {
                2021: 100.0,
                2022: 90.0,
                2023: 81.0,
                2024: 72.9,
                2025: 65.61,
            }.items()
        },
    )

    metrics = _share_count_cagr_metrics("AAA", as_of_date="2026-05-07")

    assert metrics["share_count_cagr"] == -0.09999999999999998
    assert metrics["share_count_cagr_direction"] == "BUYBACKS"
    assert metrics["share_count_oldest"] == 100.0
    assert metrics["share_count_latest"] == 65.61
    assert metrics["share_count_cagr_not_computable_reasons"] == []


def test_share_count_cagr_dilution_case(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._packet_annual_fact_rows",
        lambda ticker, as_of_date, v2_data_plane: {
            year: {
                "shares_outstanding": value,
                "shares_outstanding_split_lineage": {
                    "raw_value": value,
                    "normalized_value": value,
                    "shares_basis": "UNADJUSTED",
                    "split_adjustment_factor": 1.0,
                    "split_effective_date": None,
                },
            }
            for year, value in {
                2021: 100.0,
                2022: 110.0,
                2023: 121.0,
                2024: 133.1,
                2025: 146.41,
            }.items()
        },
    )

    metrics = _share_count_cagr_metrics("AAA", as_of_date="2026-05-07")

    assert metrics["share_count_cagr"] == 0.10000000000000009
    assert metrics["share_count_cagr_direction"] == "DILUTION"
    assert metrics["share_count_oldest"] == 100.0
    assert metrics["share_count_latest"] == 146.41


def test_piotroski_f_score_uses_hand_computed_example():
    metrics = _piotroski_f_score_from_rows(
        {
            2024: {
                "net_income": 10.0,
                "cfo": 9.0,
                "total_assets": 80.0,
                "revenue": 80.0,
                "total_debt": 40.0,
                "current_assets": 40.0,
                "current_liabilities": 24.0,
                "shares_outstanding": 11.0,
                "gross_profit": 24.0,
            },
            2025: {
                "net_income": 20.0,
                "cfo": 25.0,
                "total_assets": 100.0,
                "revenue": 120.0,
                "total_debt": 30.0,
                "current_assets": 50.0,
                "current_liabilities": 25.0,
                "shares_outstanding": 10.0,
                "gross_profit": 48.0,
            },
        }
    )

    assert metrics["value"] == 9
    assert metrics["interpretation"] == "high quality"
    assert metrics["missing_inputs"] == []
    assert metrics["status"] == "OK"


def test_piotroski_f_score_reports_missing_inputs():
    metrics = _piotroski_f_score_from_rows({2024: {"net_income": 10.0}, 2025: {"net_income": 12.0}})

    assert metrics["value"] is None
    assert metrics["interpretation"] == "insufficient data"
    assert metrics["missing_inputs"] == [
        "CFO_CURRENT_MISSING",
        "TOTAL_ASSETS_CURRENT_MISSING",
        "TOTAL_ASSETS_PRIOR_MISSING",
        "TOTAL_DEBT_CURRENT_MISSING",
        "TOTAL_DEBT_PRIOR_MISSING",
        "CURRENT_ASSETS_CURRENT_MISSING",
        "CURRENT_ASSETS_PRIOR_MISSING",
        "CURRENT_LIABILITIES_CURRENT_MISSING",
        "CURRENT_LIABILITIES_PRIOR_MISSING",
        "SHARES_OUTSTANDING_CURRENT_MISSING",
        "SHARES_OUTSTANDING_PRIOR_MISSING",
        "REVENUE_CURRENT_MISSING",
        "REVENUE_PRIOR_MISSING",
        "GROSS_MARGIN_CURRENT_MISSING",
        "GROSS_MARGIN_PRIOR_MISSING",
    ]
    assert metrics["status"] == "INSUFFICIENT_DATA"


def test_beneish_m_score_uses_hand_computed_example():
    metrics = _beneish_m_score_from_rows(
        {
            2024: {
                "revenue": 100.0,
                "accounts_receivable": 10.0,
                "gross_profit": 40.0,
                "current_assets": 35.0,
                "gross_ppe": 45.0,
                "total_assets": 100.0,
                "depreciation": 8.0,
                "sga": 12.0,
                "total_debt": 25.0,
                "current_liabilities": 15.0,
                "cfo": 18.0,
                "income_continuing": 12.0,
            },
            2025: {
                "revenue": 120.0,
                "accounts_receivable": 18.0,
                "gross_profit": 42.0,
                "current_assets": 40.0,
                "gross_ppe": 50.0,
                "total_assets": 120.0,
                "depreciation": 10.0,
                "sga": 18.0,
                "total_debt": 30.0,
                "current_liabilities": 20.0,
                "cfo": 20.0,
                "income_continuing": 14.0,
            },
        }
    )

    assert metrics["value"] == -1.966595
    assert metrics["interpretation"] == "no manipulation flag"
    assert metrics["missing_inputs"] == []
    assert metrics["status"] == "OK"


def test_beneish_m_score_uses_net_income_when_continuing_income_absent():
    metrics = _beneish_m_score_from_rows(
        {
            2024: {
                "revenue": 100.0,
                "accounts_receivable": 10.0,
                "gross_profit": 40.0,
                "current_assets": 35.0,
                "gross_ppe": 45.0,
                "total_assets": 100.0,
                "depreciation": 8.0,
                "sga": 12.0,
                "total_debt": 25.0,
                "current_liabilities": 15.0,
                "cfo": 18.0,
                "net_income": 12.0,
            },
            2025: {
                "revenue": 120.0,
                "accounts_receivable": 18.0,
                "gross_profit": 42.0,
                "current_assets": 40.0,
                "gross_ppe": 50.0,
                "total_assets": 120.0,
                "depreciation": 10.0,
                "sga": 18.0,
                "total_debt": 30.0,
                "current_liabilities": 20.0,
                "cfo": 20.0,
                "net_income": 14.0,
            },
        }
    )

    assert metrics["value"] == -1.966595
    assert metrics["interpretation"] == "no manipulation flag"
    assert metrics["missing_inputs"] == []
    assert metrics["status"] == "OK"


def test_beneish_m_score_reports_missing_inputs():
    metrics = _beneish_m_score_from_rows(
        {
            2024: {
                "revenue": 100.0,
                "accounts_receivable": 10.0,
                "gross_profit": 40.0,
                "current_assets": 35.0,
                "gross_ppe": 45.0,
                "total_assets": 100.0,
                "depreciation": 8.0,
                "total_debt": 25.0,
                "current_liabilities": 15.0,
                "cfo": 18.0,
                "income_continuing": 12.0,
            },
            2025: {
                "revenue": 120.0,
                "accounts_receivable": 18.0,
                "gross_profit": 42.0,
                "current_assets": 40.0,
                "gross_ppe": 50.0,
                "total_assets": 120.0,
                "depreciation": 10.0,
                "total_debt": 30.0,
                "current_liabilities": 20.0,
                "cfo": 20.0,
                "income_continuing": 14.0,
            },
        }
    )

    assert metrics["value"] is None
    assert metrics["interpretation"] == "insufficient data"
    assert metrics["missing_inputs"] == ["SGA_CURRENT_MISSING", "SGA_PRIOR_MISSING"]
    assert metrics["status"] == "INSUFFICIENT_DATA"


def test_altman_z_double_prime_uses_hand_computed_example():
    metrics = _altman_z_score_from_rows(
        {
            2025: {
                "current_assets": 50.0,
                "current_liabilities": 20.0,
                "total_assets": 100.0,
                "retained_earnings": 20.0,
                "operating_income": 15.0,
                "shares_outstanding": 8.0,
                "total_liabilities": 40.0,
            }
        },
        current_price=10.0,
    )

    assert metrics["value"] == 5.728
    assert metrics["interpretation"] == "safe"
    assert metrics["missing_inputs"] == []
    assert metrics["status"] == "OK"


def test_altman_z_double_prime_reports_missing_inputs():
    metrics = _altman_z_score_from_rows(
        {
            2025: {
                "current_assets": 50.0,
                "current_liabilities": 20.0,
                "total_assets": 100.0,
                "operating_income": 15.0,
                "shares_outstanding": 8.0,
                "total_liabilities": 40.0,
            }
        },
        current_price=10.0,
    )

    assert metrics["value"] is None
    assert metrics["interpretation"] == "insufficient data"
    assert metrics["missing_inputs"] == ["RETAINED_EARNINGS_MISSING"]
    assert metrics["status"] == "INSUFFICIENT_DATA"


def test_forensic_score_threshold_interpretations():
    assert _piotroski_interpretation(8) == "high quality"
    assert _piotroski_interpretation(6) == "moderate"
    assert _piotroski_interpretation(2) == "low quality"
    assert _beneish_interpretation(-1.77) == "potential manipulation"
    assert _beneish_interpretation(-1.78) == "no manipulation flag"
    assert _altman_z_interpretation(2.61) == "safe"
    assert _altman_z_interpretation(2.60) == "gray zone"
    assert _altman_z_interpretation(1.10) == "gray zone"
    assert _altman_z_interpretation(1.09) == "distress"


def _historical_multiple_rows(years: list[int]) -> dict[int, dict[str, float]]:
    return {
        year: {
            "shares_outstanding": 10.0,
            "net_income": 10.0,
            "operating_income": 20.0,
            "depreciation_amortization": 5.0,
            "total_debt": 10.0,
            "cash": 5.0,
            "equity": 50.0,
            "cfo": 12.0,
            "capex": 2.0,
        }
        for year in years
    }


def test_historical_multiple_bands_compute_all_four_with_full_history():
    years = list(range(2016, 2026))
    bands = _historical_multiple_bands_from_inputs(
        by_year=_historical_multiple_rows(years),
        period_ends={year: f"{year}-12-31" for year in years},
        prices_by_year={
            2016: 10.0,
            2017: 20.0,
            2018: 30.0,
            2019: 40.0,
            2020: 50.0,
            2021: 60.0,
            2022: 70.0,
            2023: 80.0,
            2024: 90.0,
            2025: 100.0,
        },
        current_price=50.0,
    )

    assert bands["pe"]["status"] == "OK"
    assert bands["pe"]["current_value"] == 50.0
    assert bands["pe"]["range_min"] == 10.0
    assert bands["pe"]["range_q1"] == 32.5
    assert bands["pe"]["range_median"] == 55.0
    assert bands["pe"]["range_q3"] == 77.5
    assert bands["pe"]["range_max"] == 100.0
    assert bands["pe"]["current_percentile"] == 44.44444444444444
    assert bands["ev_to_ebitda"]["current_value"] == 20.2
    assert bands["ev_to_ebitda"]["range_median"] == 22.2
    assert bands["price_to_book"]["current_value"] == 10.0
    assert bands["price_to_book"]["range_median"] == 11.0
    assert bands["fcf_yield"]["current_value"] == 0.02
    assert bands["fcf_yield"]["range_min"] == 0.01
    assert bands["fcf_yield"]["range_max"] == 0.1
    assert bands["fcf_yield"]["current_percentile"] == 55.55555555555556


def test_historical_multiple_bands_mark_partial_metric_insufficient():
    years = list(range(2020, 2026))
    rows = _historical_multiple_rows(years)
    rows[2020].pop("net_income")
    rows[2021].pop("net_income")

    bands = _historical_multiple_bands_from_inputs(
        by_year=rows,
        period_ends={year: f"{year}-12-31" for year in years},
        prices_by_year={
            2020: 10.0,
            2021: 20.0,
            2022: 30.0,
            2023: 40.0,
            2024: 50.0,
            2025: 60.0,
        },
        current_price=60.0,
    )

    assert bands["pe"]["status"] == "INSUFFICIENT_HISTORY"
    assert bands["pe"]["years_of_history"] == 4
    assert bands["pe"]["not_computable_reasons"] == ["HISTORICAL_MULTIPLE_HISTORY_INSUFFICIENT"]
    assert bands["ev_to_ebitda"]["status"] == "OK"
    assert bands["ev_to_ebitda"]["years_of_history"] == 6
    assert bands["price_to_book"]["status"] == "OK"
    assert bands["fcf_yield"]["status"] == "OK"


def test_historical_multiple_bands_mark_no_historical_prices_data_missing():
    years = [2023, 2024, 2025]

    bands = _historical_multiple_bands_from_inputs(
        by_year=_historical_multiple_rows(years),
        period_ends={year: f"{year}-12-31" for year in years},
        prices_by_year={},
        current_price=40.0,
    )

    assert bands["pe"]["status"] == "DATA_MISSING"
    assert bands["pe"]["years_of_history"] == 0
    assert bands["pe"]["not_computable_reasons"] == ["HISTORICAL_PRICE_OR_FUNDAMENTAL_DATA_MISSING"]
    assert bands["ev_to_ebitda"]["status"] == "DATA_MISSING"
    assert bands["price_to_book"]["status"] == "DATA_MISSING"
    assert bands["fcf_yield"]["status"] == "DATA_MISSING"


class _MissingHistoricalPriceProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_price_asof(self, ticker: str, as_of_date: str):
        self.calls.append((ticker, as_of_date))
        return None


class _TimeoutHistoricalPriceProvider(_MissingHistoricalPriceProvider):
    def get_price_asof(self, ticker: str, as_of_date: str):
        self.calls.append((ticker, as_of_date))
        raise TimeoutError("historical price timeout")


class _LatestOnlyHistoricalPriceProvider(_MissingHistoricalPriceProvider):
    def get_price_asof(self, ticker: str, as_of_date: str):
        from app.market.price_provider import PriceSnapshot

        self.calls.append((ticker, as_of_date))
        if as_of_date == "2025-12-31":
            return PriceSnapshot(ticker=ticker, as_of_date=as_of_date, price=50.0)
        return None


def test_historical_price_fetch_stops_after_first_provider_miss(monkeypatch):
    provider = _MissingHistoricalPriceProvider()
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._historical_price_provider", lambda: provider
    )

    prices = _historical_fiscal_year_prices(
        "AAA",
        {2023: "2023-12-31", 2024: "2024-12-31", 2025: "2025-12-31"},
        as_of_date="2026-05-15",
    )

    assert prices == {}
    assert provider.calls == [("AAA", "2025-12-31")]


def test_historical_price_fetch_preserves_latest_price_then_stops_on_older_miss(monkeypatch):
    provider = _LatestOnlyHistoricalPriceProvider()
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._historical_price_provider", lambda: provider
    )

    prices = _historical_fiscal_year_prices(
        "AAA",
        {2023: "2023-12-31", 2024: "2024-12-31", 2025: "2025-12-31"},
        as_of_date="2026-05-15",
    )

    assert prices == {2025: 50.0}
    assert provider.calls == [("AAA", "2025-12-31"), ("AAA", "2024-12-31")]


def test_historical_multiple_bands_mark_price_unavailable_after_timeout(monkeypatch):
    provider = _TimeoutHistoricalPriceProvider()
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._historical_price_provider", lambda: provider
    )
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._annual_fact_rows",
        lambda ticker, **_kwargs: _historical_multiple_rows([2023, 2024, 2025]),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._annual_period_ends",
        lambda ticker, **_kwargs: {
            2023: "2023-12-31",
            2024: "2024-12-31",
            2025: "2025-12-31",
        },
    )

    bands = _historical_multiple_bands("AAA", as_of_date="2026-05-15", current_price=50.0)

    assert provider.calls == [("AAA", "2025-12-31")]
    assert bands["pe"]["status"] == "DATA_MISSING"
    assert bands["pe"]["not_computable_reasons"] == [
        "HISTORICAL_PRICE_NOT_AVAILABLE",
        "HISTORICAL_PRICE_OR_FUNDAMENTAL_DATA_MISSING",
    ]


def test_percentile_within_history_interpolates_and_clamps():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]

    assert _percentile_within_history(35.0, values) == 62.5
    assert _percentile_within_history(30.0, values) == 50.0
    assert _percentile_within_history(5.0, values) == 0.0
    assert _percentile_within_history(55.0, values) == 100.0


def test_build_sector_company_financial_packet_carries_impairment_context():
    source = _base_packet()
    source.raw_valuation["impairment_classification_detail"] = {
        "impairment_class_primary": "TEMPORARY_WEAKNESS",
        "primary_underwriting_caution": "POSSIBLE_CYCLE_DISTORTION",
        "impairment_class_reason_codes": ["CYCLICAL_TROUGH", "CYCLICAL_SUPPORT"],
        "impairment_support_signals": ["TROUGH_EARNINGS_RISK"],
        "impairment_rebuttal_signals": ["CYCLE_RESILIENCE_PRESENT"],
    }

    packet = build_sector_company_financial_packet(source)

    assert packet.business_quality["impairment_class_primary"] == "TEMPORARY_WEAKNESS"
    assert packet.business_quality["primary_underwriting_caution"] == "POSSIBLE_CYCLE_DISTORTION"
    assert packet.business_quality["impairment_classification"] == {
        "impairment_class_primary": "TEMPORARY_WEAKNESS",
        "primary_underwriting_caution": "POSSIBLE_CYCLE_DISTORTION",
        "impairment_class_reason_codes": ["CYCLICAL_TROUGH", "CYCLICAL_SUPPORT"],
        "impairment_support_signals": ["TROUGH_EARNINGS_RISK"],
        "impairment_rebuttal_signals": ["CYCLE_RESILIENCE_PRESENT"],
    }


def test_sector_specific_insurance_anchor_takes_priority_and_suppresses_generic_status():
    source = _base_packet()
    source.ticker = "INS"
    source.dcf_value = None
    source.epv_value = None
    source.insurance_value = 80.0
    source.insurance_method = "insurance_common"
    source.insurance_packet = {
        "generic_valuation_valid": False,
        "model_status": "OK",
        "routing": {"security_type": "common"},
    }
    source.model_status = "OK"

    packet = build_sector_company_financial_packet(source)

    assert packet.ticker == "INS"
    assert packet.model_fit_status == "VALID_SECTOR_SPECIFIC"
    assert packet.valuation["anchor_method"] == "insurance_common"
    assert packet.valuation["valuation_anchor"] == 80.0
    assert packet.valuation["discount_to_anchor"] == 0.375
    assert packet.valuation["generic_valuation_valid"] is False
    assert packet.valuation["available_methods"] == ["insurance_common"]


def test_technology_adjusted_anchor_takes_priority_over_generic_dcf():
    source = _base_packet()
    source.ticker = "TECH"
    source.raw_valuation["tech_valuation_divergence_diagnostics"] = {
        "gaap_anchor": 100.0,
        "adjusted_anchor": 120.0,
        "raw_divergence": 0.20,
        "capped_divergence": 0.20,
        "status": "OK",
    }

    packet = build_sector_company_financial_packet(source, sector="semiconductors")

    assert packet.ticker == "TECH"
    assert packet.model_fit_status == "VALID_SECTOR_SPECIFIC"
    assert packet.valuation["anchor_method"] == "technology_adjusted_dcf"
    assert packet.valuation["valuation_anchor"] == 120.0
    assert packet.valuation["discount_to_anchor"] == 0.5833333333333334
    assert packet.valuation["generic_anchor_method"] == "dcf"
    assert packet.valuation["generic_anchor_value"] == 100.0
    assert packet.valuation["sector_specific_anchor_method"] == "technology_adjusted_dcf"
    assert packet.valuation["sector_specific_anchor_value"] == 120.0
    assert packet.valuation["available_methods"] == ["technology_adjusted_dcf", "dcf", "epv"]
    assert packet.expected_return["base_anchor_method"] == "technology_adjusted_dcf"
    assert packet.expected_return["base_anchor_discount"] == 0.5833333333333334


def test_technology_adjusted_anchor_fires_on_persisted_diagnostics_in_any_sweep_sector():
    """Review ANCHOR-1: the backtest applies the adjusted-anchor leg whenever
    the persisted divergence diagnostics are OK and positive — it has no
    sweep-sector input. The live sweep-token gate made the measured rule and
    the deployed rule diverge on ~11% of rows. Decision: both surfaces key on
    the persisted writer diagnostics (the fundamentals-based category already
    decided whether diagnostics exist), so a non-tech-token sweep sector must
    fire the leg identically."""
    source = _base_packet()
    source.ticker = "IND"
    source.raw_valuation["tech_valuation_divergence_diagnostics"] = {
        "gaap_anchor": 100.0,
        "adjusted_anchor": 120.0,
        "raw_divergence": 0.20,
        "capped_divergence": 0.20,
        "status": "OK",
    }

    packet = build_sector_company_financial_packet(source, sector="industrial")

    assert packet.ticker == "IND"
    assert packet.model_fit_status == "VALID_SECTOR_SPECIFIC"
    assert packet.valuation["anchor_method"] == "technology_adjusted_dcf"
    assert packet.valuation["valuation_anchor"] == 120.0
    assert packet.valuation["sector_specific_anchor_method"] == "technology_adjusted_dcf"
    assert packet.valuation["sector_specific_anchor_value"] == 120.0
    assert packet.valuation["generic_anchor_method"] == "dcf"
    assert packet.valuation["generic_anchor_value"] == 100.0
    assert packet.valuation["available_methods"] == ["technology_adjusted_dcf", "dcf", "epv"]
    assert packet.expected_return["base_anchor_method"] == "technology_adjusted_dcf"


def test_technology_adjusted_anchor_suppressed_for_anomaly_zone():
    """Review ANCHOR-3: signal_assembler nulls the generic methods for
    VALUATION_ANOMALY names, but the tech leg read the divergence diagnostics
    straight out of raw_valuation — re-anchoring (and potentially deploying)
    exactly the names the backtest excludes. The leg must short-circuit on
    the anomaly zone."""
    source = _base_packet()
    source.ticker = "ANML"
    source.pricing_zone = "VALUATION_ANOMALY"
    # Mirror signal_assembler's anomaly nulling of generic methods:
    source.dcf_value = None
    source.epv_value = None
    source.raw_valuation["tech_valuation_divergence_diagnostics"] = {
        "gaap_anchor": 100.0,
        "adjusted_anchor": 120.0,
        "raw_divergence": 0.20,
        "capped_divergence": 0.20,
        "status": "OK",
    }

    packet = build_sector_company_financial_packet(source, sector="semiconductors")

    assert packet.valuation["valuation_anchor"] is None
    assert packet.valuation["anchor_method"] is None
    assert packet.valuation["sector_specific_anchor_method"] is None
    assert packet.model_fit_status != "VALID_SECTOR_SPECIFIC"


def test_negative_insurance_value_not_labeled_sector_specific():
    """Review ANCHOR-8: a negative insurance value is skipped by
    _sector_specific_valuation_anchor so it must not
    label the packet VALID_SECTOR_SPECIFIC nor appear in available_methods
    — select_anchor can never choose it."""
    source = _base_packet()
    source.ticker = "NEGI"
    source.insurance_value = -5.0
    source.insurance_method = "insurance_residual_income"

    packet = build_sector_company_financial_packet(source, sector="insurance")

    assert packet.model_fit_status == "VALID_GENERIC"
    assert "insurance_residual_income" not in packet.valuation["available_methods"]
    assert packet.valuation["anchor_method"] == "dcf"


def test_missing_price_or_anchor_becomes_blocking_data_insufficiency():
    source = TickerSignalPacket(
        ticker="MISS",
        current_price=None,
        filing_risk_status="NO_FILING",
        quarterly_revenue_trend="UNKNOWN",
    )

    packet = build_sector_company_financial_packet(source)

    assert packet.financial_status == "Data Insufficient"
    assert packet.model_fit_status == "UNKNOWN"
    assert packet.data_quality_status == "MISSING_PRICE"
    assert packet.current_price is None
    assert packet.valuation["valuation_anchor"] is None
    assert packet.valuation["discount_to_anchor"] is None
    assert packet.expected_return["status"] == "INSUFFICIENT_INPUTS"
    assert packet.blockers == ["MISSING_PRICE", "MISSING_VALUATION"]
    assert "FILING_RISK_NO_FILING" in packet.confidence_caps
    assert "QUARTERLY_REVENUE_TREND_UNKNOWN" in packet.confidence_caps


def test_risk_section_extraction_failure_is_distinct_from_missing_filing_cap():
    source = _base_packet()
    source.ticker = "SECT"
    source.filing_risk_status = "NO_FILING"
    source.filing_risk_metadata = {
        "evidence_status": "RISK_SECTION_NOT_FOUND",
        "source_accession": "0000000000-26-000001",
        "source_form_type": "10-K",
        "source_filing_date": "2026-02-26",
        "source_filing_age_days": 59,
        "risk_text_chars": 0,
        "warnings": ["risk_section_start_not_found"],
    }

    packet = build_sector_company_financial_packet(source)

    assert packet.financial_status == "Financially Viable With Evidence Caps"
    assert packet.data_quality_status == "RISK_SECTION_NOT_FOUND"
    assert "FILING_RISK_SECTION_NOT_FOUND" in packet.confidence_caps
    assert "FILING_RISK_NO_FILING" not in packet.confidence_caps
    assert packet.accounting_quality["filing_risk_evidence_status"] == "RISK_SECTION_NOT_FOUND"
    assert packet.accounting_quality["filing_risk_source_accession"] == "0000000000-26-000001"
    assert packet.accounting_quality["filing_risk_source_filing_age_days"] == 59
    assert packet.accounting_quality["filing_risk_text_chars"] == 0
    assert packet.accounting_quality["filing_risk_warnings"] == ["risk_section_start_not_found"]


def test_stale_readable_filing_becomes_evidence_cap_not_missing_filing():
    source = _base_packet()
    source.ticker = "OLD"
    source.filing_risk_status = "OK"
    source.filing_risk_metadata = {
        "evidence_status": "STALE_READABLE_RISK_SECTION",
        "source_accession": "0000000000-21-000001",
        "source_form_type": "10-K",
        "source_filing_date": "2021-02-18",
        "source_filing_age_days": 1893,
        "risk_text_chars": 30000,
        "warnings": ["stale_annual_filing:1893d"],
    }

    packet = build_sector_company_financial_packet(source)

    assert packet.financial_status == "Financially Viable With Evidence Caps"
    assert packet.data_quality_status == "STALE_FILING_RISK"
    assert "FILING_RISK_STALE_ANNUAL_FILING" in packet.confidence_caps
    assert "FILING_RISK_NO_FILING" not in packet.confidence_caps
    assert packet.accounting_quality["filing_risk_evidence_status"] == "STALE_READABLE_RISK_SECTION"
    assert packet.accounting_quality["filing_risk_source_filing_age_days"] == 1893
    assert packet.accounting_quality["filing_risk_warnings"] == ["stale_annual_filing:1893d"]


def test_model_blockers_and_critical_solvency_propagate_to_packet():
    source = _base_packet()
    source.ticker = "BAD"
    source.model_status = "MODEL_BLOCKED"
    source.model_blockers = ["SECURITY_IDENTITY_UNVERIFIED"]
    source.model_fit_warnings = ["GENERIC_DCF_EPV_SUPPRESSED"]
    source.solvency_risk = "CRITICAL"

    packet = build_sector_company_financial_packet(source)

    assert packet.financial_status == "Balance-Sheet Constrained"
    assert packet.model_fit_status == "BLOCKED"
    assert packet.data_quality_status == "MODEL_BLOCKED"
    assert packet.blockers == ["SOLVENCY_CRITICAL", "SECURITY_IDENTITY_UNVERIFIED"]
    assert "MODEL_WARNING_GENERIC_DCF_EPV_SUPPRESSED" in packet.confidence_caps


def test_high_growth_dependency_and_method_tension_cap_confidence():
    source = _base_packet()
    source.ticker = "GRO"
    source.method_tension_type = "GROWTH_VS_EARNINGS_POWER"
    source.growth_dependency_ratio = 0.82

    packet = build_sector_company_financial_packet(source)

    assert packet.financial_status == "Financially Viable"
    assert packet.reinvestment["growth_dependency_status"] == "HIGH"
    assert "METHOD_TENSION_GROWTH_VS_EARNINGS_POWER" in packet.confidence_caps
    assert "HIGH_GROWTH_DEPENDENCY" in packet.confidence_caps


def test_build_packets_from_signal_packets_returns_ticker_sorted_packets():
    bbb = _base_packet()
    bbb.ticker = "BBB"
    aaa = _base_packet()
    aaa.ticker = "AAA"

    packets = build_sector_company_financial_packets_from_signal_packets({"BBB": bbb, "AAA": aaa})

    assert [packet.ticker for packet in packets] == ["AAA", "BBB"]


def test_v1_packet_carries_canonical_signal_cap_and_security_identity():
    source = _base_packet()
    source.current_price = 100.0
    source.current_price_as_of_date = "2026-04-26"
    source.current_price_currency = "USD"
    source.current_price_source = "fixture_quote"
    source.current_price_source_url = "https://prices.example/aaa"
    source.price_basis = "UNADJUSTED"
    source.raw_price = 100.0
    source.split_adjustment_factor = 1.0
    source.market_cap_mm = 2_000.0
    source.market_cap_unit = "USD_millions"
    source.market_cap_source = "price_times_shares"
    source.market_cap_method = "price_times_shares_divided_by_issuer_quote_ratio"
    source.market_cap_effective_as_of_date = "2026-04-26"
    source.cap_stage_price = 100.0
    source.cap_stage_price_as_of_date = "2026-04-26"
    source.cap_stage_price_currency = "USD"
    source.cap_stage_price_source = "fixture_quote"
    source.cap_stage_price_source_url = "https://prices.example/aaa"
    source.cap_stage_quote_snapshot_id = "a" * 64
    source.shares_outstanding_mm = 20.0
    source.raw_shares_outstanding_mm = 20.0
    source.shares_unit = "shares_millions"
    source.shares_basis = "UNADJUSTED"
    source.shares_as_of_date = "2026-03-31"
    source.shares_filed_date = "2026-04-20"
    source.shares_source = "fixture_companyfacts"
    source.shares_source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"
    source.issuer_quote_ratio = 1.0
    source.issuer_cik = "0000000001"
    source.issuer_primary_ticker = "AAA"
    source.issuer_listed_tickers = ["AAA"]
    source.security_role = "PRIMARY"
    source.is_secondary_class = False
    source.is_adr = False
    source.identity_source = "sec_submissions_exchange_binding"
    source.identity_source_url = "https://data.sec.gov/submissions/CIK0000000001.json"
    source.identity_as_of_date = "2026-04-26"
    source.identity_confidence = "HIGH"
    source.ratio_source_url = source.identity_source_url
    source.ratio_source_accession = "0000000001-26-000001"
    source.ratio_security_symbol = "AAA"

    packet = build_sector_company_financial_packet(source, as_of_date="2026-04-26")

    assert packet.market_cap_mm == 2_000.0
    assert packet.current_price == 100.0
    assert packet.shares_outstanding_mm == 20.0
    assert packet.issuer_quote_ratio == 1.0
    assert packet.security_role == "PRIMARY"
    assert packet.identity_source_url == source.identity_source_url
    assert packet.ratio_source_accession == "0000000001-26-000001"
    assert packet.ratio_security_symbol == "AAA"
    assert packet.metric_traces["market_cap_mm"]["reconciles"] is True
    restored = type(packet).from_dict(packet.to_dict())
    assert restored.ratio_source_accession == packet.ratio_source_accession
    assert restored.ratio_security_symbol == packet.ratio_security_symbol


def test_build_packets_from_signal_packets_passes_sector_to_technology_anchor():
    source = _base_packet()
    source.ticker = "TECH"
    source.raw_valuation["tech_valuation_divergence_diagnostics"] = {
        "gaap_anchor": 100.0,
        "adjusted_anchor": 120.0,
        "raw_divergence": 0.20,
        "capped_divergence": 0.20,
        "status": "OK",
    }

    packets = build_sector_company_financial_packets_from_signal_packets(
        {"TECH": source}, sector="semiconductors"
    )

    assert len(packets) == 1
    assert packets[0].valuation["anchor_method"] == "technology_adjusted_dcf"
    assert packets[0].valuation["valuation_anchor"] == 120.0


def test_v2_packet_replaces_scorecard_price_with_cap_stage_price():
    source = _base_packet()

    packet = build_sector_company_financial_packet(
        source,
        pipeline_version="v2",
        cap_classification={
            "price_used": 40.0,
            "price_currency": "USD",
            "price_as_of_date": "2026-06-11",
            "price_source": "fixed_asof_provider",
        },
    )

    assert packet.current_price == 40.0
    assert packet.cap_stage_price == 40.0
    assert packet.valuation["discount_to_anchor"] == 0.6


def test_v2_packet_uses_repaired_valuation_price_without_rewriting_cap_price():
    source = _base_packet()

    packet = build_sector_company_financial_packet(
        source,
        pipeline_version="v2",
        cap_classification={
            "price_used": 40.0,
            "price_currency": "USD",
            "price_as_of_date": "2026-06-09",
            "price_source": "cap_provider",
            "price_source_url": "https://cap.example/aaa",
            "price_confidence": "HIGH",
            "current_price": 42.0,
            "current_price_currency": "USD",
            "current_price_as_of_date": "2026-06-11",
            "current_price_source": "price_repair_provider",
            "current_price_source_url": "https://prices.example/aaa",
            "current_price_confidence": "MEDIUM",
        },
    )

    assert packet.current_price == 42.0
    assert packet.current_price_currency == "USD"
    assert packet.current_price_as_of_date == "2026-06-11"
    assert packet.current_price_source == "price_repair_provider"
    assert packet.current_price_source_url == "https://prices.example/aaa"
    assert packet.current_price_confidence == "MEDIUM"
    assert packet.cap_stage_price == 40.0
    assert packet.cap_stage_price_currency == "USD"
    assert packet.cap_stage_price_as_of_date == "2026-06-09"
    assert packet.cap_stage_price_source == "cap_provider"
    assert packet.cap_stage_price_source_url == "https://cap.example/aaa"
    assert packet.cap_stage_price_confidence == "HIGH"
    assert packet.valuation["discount_to_anchor"] == 0.58
    restored = type(packet).from_dict(packet.to_dict())
    assert restored.current_price_currency == "USD"
    assert restored.cap_stage_price_currency == "USD"


def test_v2_packet_does_not_fall_back_to_scorecard_price_when_cap_price_missing():
    source = _base_packet()

    packet = build_sector_company_financial_packet(
        source,
        pipeline_version="v2",
        cap_classification={},
    )

    assert packet.current_price is None
    assert packet.cap_stage_price is None
    assert "MISSING_PRICE" in packet.blockers


def test_v2_packet_rejects_non_usd_repaired_and_cap_prices():
    packet = build_sector_company_financial_packet(
        _base_packet(),
        pipeline_version="v2",
        cap_classification={
            "price_used": 40.0,
            "price_currency": "CAD",
            "price_as_of_date": "2026-06-09",
            "current_price": 42.0,
            "current_price_currency": "CAD",
            "current_price_as_of_date": "2026-06-11",
        },
    )

    assert packet.cap_stage_price == 40.0
    assert packet.cap_stage_price_currency == "CAD"
    assert packet.current_price is None
    assert packet.current_price_currency is None
    assert "MISSING_PRICE" in packet.blockers


def test_v2_packet_ignores_unproven_repaired_quote_and_uses_valid_usd_cap_price():
    packet = build_sector_company_financial_packet(
        _base_packet(),
        pipeline_version="v2",
        cap_classification={
            "price_used": 40.0,
            "price_currency": "USD",
            "price_as_of_date": "2026-06-09",
            "current_price": 999.0,
        },
    )

    assert packet.current_price == 40.0
    assert packet.current_price_currency == "USD"
    assert packet.current_price_as_of_date == "2026-06-09"
    assert packet.cap_stage_price == 40.0


def test_v2_packet_fact_lenses_receive_cap_stage_issuer_identity(monkeypatch):
    import app.autonomous.sector_financial_packets as packets_mod

    calls: list[dict] = []

    def fake_annual_fact_rows(ticker, **kwargs):
        calls.append({"ticker": ticker, **kwargs})
        return {}

    monkeypatch.setattr(packets_mod, "_annual_fact_rows", fake_annual_fact_rows)
    monkeypatch.setattr(packets_mod, "_valuation", lambda *args, **kwargs: {})

    packet = build_sector_company_financial_packet(
        _base_packet(),
        as_of_date="2026-06-11",
        pipeline_version="v2",
        cap_classification={
            "issuer_cik": "42",
            "issuer_primary_ticker": "PRIMARY",
            "issuer_listed_tickers": ["PRIMARY", "AAA"],
            "issuer_aliases": ["AAA", "PRIMARY"],
            "price_used": 40.0,
        },
    )

    assert packet.ticker == "AAA"
    assert calls
    assert all(call["issuer_aware"] is True for call in calls)
    assert all(call["issuer_cik"] == "42" for call in calls)
    assert all(call["aliases"] == ("AAA", "PRIMARY") for call in calls)


def test_v1_build_packets_forward_historical_cutoff_to_signal_assembler(monkeypatch):
    source = _base_packet()
    captured: dict[str, object] = {}

    def fake_assemble(tickers, **kwargs):
        captured["tickers"] = tickers
        captured.update(kwargs)
        return {"AAA": source}

    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets.assemble_sector_packets", fake_assemble
    )

    packets = build_sector_company_financial_packets(
        ["AAA"],
        sector="specialty_manufacturing",
        as_of_date="2026-04-26",
        market_cap_focus="small_cap",
    )

    assert captured["tickers"] == ["AAA"]
    assert captured["as_of_date"] == "2026-04-26"
    assert len(packets) == 1
    assert packets[0].ticker == "AAA"
    assert packets[0].valuation["anchor_method"] == "dcf"


# ── expectations gap carried into the autonomous packet ──────────────────


def _init_temp_db(monkeypatch, tmp_path):
    from app.config import get_config
    from app.db import init_db

    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    get_config.cache_clear()
    init_db()
    return db_path


def _seed_reverse_dcf_row(ticker: str, as_of_date: str, outputs_json: dict) -> None:
    from app.db import get_db

    with get_db() as conn:
        conn.execute(
            """INSERT INTO valuations
               (ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at)
               VALUES (?, ?, 'reverse_dcf', '{}', ?, '[]', ?)""",
            (ticker.upper(), as_of_date, json.dumps(outputs_json), f"{as_of_date}T00:00:00+00:00"),
        )


def test_expectations_gap_from_db_reads_persisted_sub_dict(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_reverse_dcf_row(
        "AAA",
        "2026-05-16",
        {
            "status": "OK",
            "outputs": {"implied_growth": 0.04},
            "expectations_gap": {
                "gap": -0.06,
                "bucket": "CHEAP_VS_EXPECTATIONS",
                "supportable_growth": 0.10,
                "implied_growth_saturated": False,
                "line": "Price implies 4% growth for 5y vs 10% supportable -> CHEAP_VS_EXPECTATIONS",
            },
        },
    )

    result = _expectations_gap_from_db("AAA", as_of_date="2026-05-16")

    assert result["bucket"] == "CHEAP_VS_EXPECTATIONS"
    assert result["gap"] == -0.06
    assert result["supportable_growth"] == 0.10


def test_expectations_gap_from_db_defaults_unreliable_when_no_row(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    result = _expectations_gap_from_db("ZZZ", as_of_date="2026-05-16")

    assert result == {"bucket": "EXPECTATIONS_GAP_UNRELIABLE"}


def test_valuation_carries_expectations_gap_keys(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_reverse_dcf_row(
        "AAA",
        "2026-05-16",
        {
            "status": "OK",
            "outputs": {"implied_growth": 0.04},
            "expectations_gap": {
                "gap": -0.06,
                "bucket": "CHEAP_VS_EXPECTATIONS",
                "supportable_growth": 0.10,
                "implied_growth_saturated": False,
                "line": "Price implies 4% growth for 5y vs 10% supportable -> CHEAP_VS_EXPECTATIONS",
            },
        },
    )
    source = _base_packet()
    source.ticker = "AAA"

    valuation = _valuation(source, "dcf", 100.0, sector=None, as_of_date="2026-05-16")

    assert "implied_growth" in valuation
    assert "supportable_growth" in valuation
    assert "expectations_gap" in valuation
    assert "expectations_gap_bucket" in valuation
    assert valuation["expectations_gap_bucket"] == "CHEAP_VS_EXPECTATIONS"
    assert valuation["implied_growth"] == 0.04
    assert valuation["supportable_growth"] == 0.10
    assert valuation["expectations_gap"] == -0.06


def test_valuation_expectations_gap_unreliable_when_no_reverse_dcf_row(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    source = _base_packet()
    source.ticker = "ZZZ"

    valuation = _valuation(source, "dcf", 100.0, sector=None, as_of_date="2026-05-16")

    assert valuation["expectations_gap_bucket"] == "EXPECTATIONS_GAP_UNRELIABLE"
    assert valuation["implied_growth"] is None
    assert valuation["supportable_growth"] is None
    assert valuation["expectations_gap"] is None


def test_expectations_gap_from_db_suppresses_saturated_implied_growth(monkeypatch, tmp_path):
    # A saturated / UNRELIABLE reverse-DCF row must NOT surface its numeric
    # implied_growth (the saturated bound) to the cross-sectional gap factor;
    # ranking a distressed name on a known-unreliable bound is exactly the bug.
    _init_temp_db(monkeypatch, tmp_path)
    _seed_reverse_dcf_row(
        "AAA",
        "2026-05-16",
        {
            "status": "OK",
            "outputs": {"implied_growth": -0.25},
            "expectations_gap": {
                "gap": None,
                "bucket": "EXPECTATIONS_GAP_UNRELIABLE",
                "supportable_growth": 0.10,
                "implied_growth_saturated": True,
            },
        },
    )

    result = _expectations_gap_from_db("AAA", as_of_date="2026-05-16")

    assert result["bucket"] == "EXPECTATIONS_GAP_UNRELIABLE"
    assert result["implied_growth"] is None


def _seed_companyfacts_fact(
    ticker: str, fiscal_year: int, line_item: str, value: float, *, period_end: str = "2025-12-31"
) -> None:
    from app.db import get_db

    with get_db() as conn:
        conn.execute(
            "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, "
            "period_end, line_item, value, units, source_url, fetched_at, "
            "filed_date, form, accession) "
            "VALUES(?, ?, 'FY', ?, ?, ?, 'USD_millions', "
            "'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json', "
            "'2026-02-13T00:00:00Z', '2026-02-13', '10-K', "
            "'0000000001-26-000001')",
            (ticker.upper(), fiscal_year, period_end, line_item, float(value)),
        )
        conn.commit()


def test_v2_annual_packet_facts_recover_issuer_alias_and_exclude_future_filed_rows(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)
    from app.db import get_db

    with get_db() as conn:
        conn.execute(
            "INSERT INTO companies(ticker, cik, name, created_at) "
            "VALUES ('ALIAS', '88', 'Alias Security', 'x')"
        )
        conn.execute(
            "INSERT INTO filings(cik, ticker, accession, form_type, filing_date, "
            "period_end, primary_doc_url, local_path, status, created_at, updated_at) "
            "VALUES ('88', 'PRIMARY', 'a88', '10-K', '2026-02-01', "
            "'2025-12-31', 'https://www.sec.gov/Archives/a88.htm', NULL, "
            "'OK', 'x', 'x')"
        )
        for year in (2023, 2024, 2025):
            filed_date = f"{year + 1}-02-01"
            for line_item, value in (
                ("revenue", 100.0 + year),
                ("gross_profit", 50.0 + year),
            ):
                conn.execute(
                    "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, "
                    "period_end, line_item, value, units, source_url, fetched_at, "
                    "filed_date, form, accession) VALUES "
                    "('PRIMARY', ?, 'FY', ?, ?, ?, 'USD_millions', ?, 'x', ?, "
                    "'10-K', '0000000088-26-000001')",
                    (
                        year,
                        f"{year}-12-31",
                        line_item,
                        value,
                        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000088.json",
                        filed_date,
                    ),
                )
        conn.execute(
            "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, "
            "period_end, line_item, value, units, source_url, fetched_at, "
            "filed_date, form, accession) "
            "VALUES ('PRIMARY', 2025, 'FY', '2025-12-31', 'operating_income', "
            "999, 'USD_millions', ?, 'x', '2026-07-01', '10-K', "
            "'0000000088-26-000002')",
            ("https://data.sec.gov/api/xbrl/companyfacts/CIK0000000088.json",),
        )

    legacy = _annual_fact_rows("ALIAS", as_of_date="2026-06-11")
    v2 = _annual_fact_rows(
        "ALIAS",
        as_of_date="2026-06-11",
        issuer_aware=True,
        require_filed_asof=True,
    )
    metrics = _gross_margin_metrics(
        "ALIAS",
        as_of_date="2026-06-11",
        v2_data_plane=True,
    )

    assert legacy == {}
    assert sorted(v2) == [2023, 2024, 2025]
    assert v2[2025]["revenue"] == 2_125.0
    assert "operating_income" not in v2[2025]
    assert metrics["gross_margin"] == pytest.approx(0.9764705882352941)


def test_v1_packet_facts_reject_post_asof_filing_and_preserve_provenance(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    from app.db import get_db

    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000099.json"
    with get_db() as conn:
        for line_item, value, filed_date, accession in (
            ("cfo", 40.0, "2026-02-15", "visible-cfo"),
            ("capex", 10.0, "2026-02-15", "visible-capex"),
            ("cash", 25.0, "2026-02-15", "visible-cash"),
            ("total_debt", 15.0, "2026-07-01", "future-debt"),
        ):
            conn.execute(
                "INSERT INTO companyfacts_facts("
                "ticker, fiscal_year, period_type, period_end, line_item, "
                "value, units, source_url, fetched_at, filed_date, form, accession"
                ") VALUES ('PITV1', 2025, 'FY', '2025-12-31', ?, ?, "
                "'USD_millions', ?, 'x', ?, '10-K', ?)",
                (line_item, value, source_url, filed_date, accession),
            )

    values = _packet_annual_fact_rows(
        "PITV1",
        as_of_date="2026-06-30",
        v2_data_plane=False,
    )
    provenance = _packet_annual_fact_provenance_rows(
        "PITV1",
        as_of_date="2026-06-30",
        v2_data_plane=False,
    )

    assert values[2025] == {"capex": 10.0, "cash": 25.0, "cfo": 40.0}
    assert "total_debt" not in values[2025]
    assert provenance[2025]["cfo"] == {
        "value": 40.0,
        "unit": "USD_millions",
        "source": "SEC_COMPANYFACTS",
        "period_end": "2025-12-31",
        "filed_date": "2026-02-15",
        "source_reference": source_url,
        "source_url": source_url,
        "accession": "visible-cfo",
    }


def test_returns_on_capital_financial_issuer_uses_equity_basis_via_real_fetch(
    monkeypatch, tmp_path
):
    # A bank (deposits+loans present, no total_debt) must classify FINANCIAL
    # through the REAL _annual_fact_rows fetch and use the equity-only invested-
    # capital basis. Before the fix _annual_fact_rows never fetched deposits/loans,
    # so classification fell back to OPERATING and the basis stayed with_goodwill.
    _init_temp_db(monkeypatch, tmp_path)
    for line_item, value in (
        ("deposits", 8000.0),
        ("loans", 7000.0),
        ("operating_income", 300.0),
        ("equity", 1000.0),
        ("pretax_income", 250.0),
        ("income_tax_expense", 50.0),
    ):
        _seed_companyfacts_fact("BANKCO", 2024, line_item, value)

    result = _returns_on_capital_metrics("BANKCO", as_of_date=None)

    assert result["invested_capital_basis"] == "equity_only_financial"
    assert result["roic"] is not None


def test_valuation_suppresses_saturated_implied_growth_end_to_end(monkeypatch, tmp_path):
    # The load-bearing UNRELIABLE/saturated path through _valuation. A
    # persisted reverse_dcf row with bucket UNRELIABLE + a numeric
    # outputs.implied_growth must surface implied_growth=None so the gap factor
    # drops it rather than ranking on a saturated bound.
    _init_temp_db(monkeypatch, tmp_path)
    _seed_reverse_dcf_row(
        "AAA",
        "2026-05-16",
        {
            "status": "OK",
            "outputs": {"implied_growth": -0.25},
            "expectations_gap": {
                "gap": None,
                "bucket": "EXPECTATIONS_GAP_UNRELIABLE",
                "supportable_growth": 0.10,
                "implied_growth_saturated": True,
            },
        },
    )
    source = _base_packet()
    source.ticker = "AAA"

    valuation = _valuation(source, "dcf", 100.0, sector=None, as_of_date="2026-05-16")

    assert valuation["expectations_gap_bucket"] == "EXPECTATIONS_GAP_UNRELIABLE"
    assert valuation["implied_growth"] is None


def test_decline_class_name_anchors_on_no_growth_basis():
    """Part-0c decline-cap policy: a DECLINING-trend name must not take the
    growth-bearing DCF (100) as its deployed anchor; capped at EPV (70). The
    conviction-widened MoS stacks on top downstream."""
    source = _base_packet()
    source.ticker = "DCLN"
    source.raw_quality_ctx["revenue_trend_class"] = "DECLINING"

    packet = build_sector_company_financial_packet(source)

    assert packet.valuation["anchor_method"] == "epv"
    assert packet.valuation["valuation_anchor"] == 70.0


def test_volatile_cyclical_name_keeps_max_positive_anchor():
    """Cyclical-trough control: VOLATILE is not a decline class — the EPV
    OI-median normalization handles the trough, not a cap."""
    source = _base_packet()
    source.ticker = "CYCL"
    source.raw_quality_ctx["revenue_trend_class"] = "VOLATILE"

    packet = build_sector_company_financial_packet(source)

    assert packet.valuation["anchor_method"] == "dcf"
    assert packet.valuation["valuation_anchor"] == 100.0


def test_balance_sheet_says_whether_going_concern_language_is_backed_by_a_filed_assertion():
    """The packet carries going_concern_asserted next to the raw flag so the runtime
    blocks only on a stored, blockable, filed excerpt, never on the bare flag."""
    from app.autonomous.sector_financial_packets import _balance_sheet

    def _packet(assertions):
        return TickerSignalPacket(
            ticker="AAA",
            research_report={
                "solvency": {
                    "going_concern_language": True,
                    "going_concern_assertions": assertions,
                }
            },
        )

    backed = _balance_sheet(
        _packet(
            [{"blockable": True, "excerpt": "There is substantial doubt about our ability."}]
        )
    )
    bare = _balance_sheet(_packet([]))
    assert (backed["going_concern_language"], backed["going_concern_asserted"]) == (True, True)
    assert (bare["going_concern_language"], bare["going_concern_asserted"]) == (True, False)
