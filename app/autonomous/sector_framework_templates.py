from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Any


_GENERIC_ECONOMIC_MODELS = {
    "financially underwritten 5-10 year per-share return analysis.",
    "sector-specific financial return underwriting.",
    "small-cap financial underwriting.",
}


def _policy(
    approach: str,
    *,
    rules: list[str],
    thresholds: list[tuple[str, str, str]],
) -> dict[str, Any]:
    return {
        "approach": approach,
        "rules": rules,
        "numeric_thresholds": [
            {"name": name, "value": value, "units": units}
            for name, value, units in thresholds
        ],
    }


def _base_policy() -> dict[str, Any]:
    return _policy(
        "Use deterministic tools to underwrite per-share returns before any sector selection.",
        rules=[
            "Do not force a winner when evidence quality or model fit is insufficient.",
            "Separate relative sector rank from absolute actionability.",
        ],
        thresholds=[("base_return_hurdle", "12", "percent_annualized")],
    )


def _template(
    *,
    economic_model: str,
    selected_value_drivers: list[str],
    selected_metrics: list[str],
    valid_valuation_methods: list[str],
    invalid_valuation_methods: list[str],
    required_evidence: list[str],
    sector_specific_risks: list[str],
    normalization_rules: list[str],
    weighting_rules: list[str],
    hurdle_thresholds: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    base = _base_policy()
    base_thresholds = [("base_return_hurdle", "12", "percent_annualized")]
    return {
        "horizon_years": [5, 10],
        "economic_model": economic_model,
        "selected_value_drivers": selected_value_drivers,
        "selected_metrics": selected_metrics,
        "valid_valuation_methods": valid_valuation_methods,
        "invalid_valuation_methods": invalid_valuation_methods,
        "required_evidence": required_evidence,
        "normalization_policy": _policy(
            base["approach"],
            rules=[*base["rules"], *normalization_rules],
            thresholds=base_thresholds,
        ),
        "hurdle_rate_policy": _policy(
            "Raise or lower the base return hurdle only when sector cyclicality, leverage, or durability evidence justifies it.",
            rules=[
                "Require a larger return cushion for cyclical, levered, or structurally disrupted sectors.",
                "Use WATCHLIST or NO_SELECTION when sector-specific evidence does not support the hurdle.",
            ],
            thresholds=hurdle_thresholds or [("base_return_hurdle", "12", "percent_annualized")],
        ),
        "weighting_policy": _policy(
            "Weight expected return only after model fit and sector-specific evidence clear minimum standards.",
            rules=[*base["rules"], *weighting_rules],
            thresholds=[("expected_return_minimum_weight", "25", "percent")],
        ),
        "sector_specific_risks": sector_specific_risks,
    }


_TEMPLATES: dict[str, dict[str, Any]] = {
    "financial_services": _template(
        economic_model="Financial balance-sheet compounding driven by underwriting, credit, spread, capital adequacy, and tangible book value growth.",
        selected_value_drivers=[
            "underwriting_margin",
            "reserve_or_credit_quality",
            "net_investment_or_interest_spread",
            "tangible_book_value_per_share_growth",
            "regulatory_or_rating_agency_capital",
            "capital_return_capacity",
        ],
        selected_metrics=[
            "combined_ratio_or_efficiency_ratio",
            "loss_ratio_or_credit_loss_rate",
            "reserve_development",
            "net_investment_yield_or_net_interest_margin",
            "tangible_book_value_per_share_cagr",
            "normalized_roe",
            "capital_ratio_or_rbc",
        ],
        valid_valuation_methods=[
            "tangible_book_value_compounding",
            "normalized_roe_to_book_value",
            "excess_capital_return",
            "sector_specific_insurance_anchor",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "generic_dcf",
            "generic_epv",
            "revenue_multiple",
            "ebitda_multiple",
            "unadjusted_enterprise_value_multiple",
        ],
        required_evidence=[
            "statutory_or_regulatory_capital_evidence",
            "reserve_or_credit_quality_evidence",
            "underwriting_or_spread_trend_evidence",
            "book_value_per_share_history",
            "security_type_and_common_equity_routing",
        ],
        sector_specific_risks=[
            "reserve_restatement",
            "credit_cycle_losses",
            "duration_mismatch",
            "capital_adequacy_shortfall",
            "non_common_security_misrouting",
        ],
        normalization_rules=[
            "Do not use generic enterprise-value anchors for regulated financial balance sheets.",
            "Normalize for credit/reserve cycles and investment gains before comparing returns.",
        ],
        weighting_rules=[
            "Give model-fit and capital adequacy veto power before expected-return ranking.",
            "Weight per-share book value compounding above company-level revenue growth.",
        ],
        hurdle_thresholds=[("base_return_hurdle", "11", "percent_annualized")],
    ),
    "payments_fintech": _template(
        economic_model="Payments and fintech network model driven by payment volume, transaction growth, take rate, merchant and consumer retention, fraud and credit losses, funding or float economics, regulatory compliance, operating leverage, and capital intensity.",
        selected_value_drivers=[
            "payment_volume_and_transaction_growth",
            "take_rate_and_pricing_mix",
            "merchant_or_consumer_retention",
            "fraud_credit_and_chargeback_losses",
            "funding_cost_or_float_income",
            "network_acceptance_and_partner_concentration",
            "regulatory_and_compliance_resilience",
            "operating_leverage_and_capital_intensity",
        ],
        selected_metrics=[
            "gross_payment_volume_growth",
            "transaction_growth",
            "net_revenue_take_rate",
            "merchant_retention_or_churn",
            "fraud_loss_rate",
            "credit_loss_rate",
            "funding_cost_or_net_interest_margin",
            "free_cash_flow_margin",
            "tangible_common_equity_or_capital_ratio",
        ],
        valid_valuation_methods=[
            "payment_volume_take_rate_underwriting",
            "normalized_free_cash_flow",
            "unit_economics_dcf",
            "network_quality_expected_return",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "unadjusted_revenue_multiple",
            "gross_payment_volume_multiple_without_take_rate",
            "ebitda_multiple_without_credit_or_fraud_losses",
            "generic_book_value_anchor_without_regulatory_capital_context",
            "growth_multiple_without_retention_or_compliance_evidence",
        ],
        required_evidence=[
            "payment_volume_transaction_and_take_rate_evidence",
            "merchant_consumer_retention_evidence",
            "fraud_credit_loss_and_chargeback_evidence",
            "funding_float_or_regulatory_capital_evidence",
            "partner_concentration_and_compliance_evidence",
        ],
        sector_specific_risks=[
            "take_rate_compression",
            "merchant_or_consumer_churn",
            "fraud_or_credit_loss_spike",
            "funding_cost_pressure",
            "regulatory_or_compliance_action",
            "partner_or_network_disintermediation",
        ],
        normalization_rules=[
            "Separate durable transaction-volume growth from take-rate compression, promotional incentives, and partner mix shifts.",
            "Normalize profitability for credit, fraud, chargeback, funding, and compliance costs before comparing expected returns.",
            "Require capital and liquidity evidence when fintech economics include lending, stored value, or balance-sheet exposure.",
        ],
        weighting_rules=[
            "Weight payment volume quality, take-rate durability, retention, fraud/credit losses, and regulatory capital evidence before generic growth.",
            "Penalize expected returns that rely on gross volume growth without take-rate, retention, compliance, and loss evidence.",
        ],
        hurdle_thresholds=[("base_return_hurdle", "14", "percent_annualized")],
    ),
    "real_estate": _template(
        economic_model="Asset-heavy real-estate cash-flow model driven by same-property NOI, occupancy, lease duration, cap rates, leverage, and NAV/FFO compounding.",
        selected_value_drivers=[
            "same_property_noi_growth",
            "occupancy_and_rent_spread",
            "lease_duration_and_rollover_risk",
            "affo_per_share_growth",
            "net_asset_value_discount",
            "leverage_and_refinancing_capacity",
        ],
        selected_metrics=[
            "same_store_noi_growth",
            "occupancy_rate",
            "ffo_per_share",
            "affo_payout_ratio",
            "net_debt_to_ebitda",
            "weighted_average_lease_term",
            "implied_cap_rate",
        ],
        valid_valuation_methods=[
            "nav_discount",
            "ffo_multiple",
            "affo_yield",
            "cap_rate_sensitivity",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "generic_epv",
            "generic_book_value_anchor",
            "unadjusted_gaap_net_income_multiple",
            "revenue_multiple",
        ],
        required_evidence=[
            "property_or_segment_noi_evidence",
            "occupancy_and_lease_maturity_evidence",
            "debt_maturity_and_rate_evidence",
            "maintenance_capex_or_affo_reconciliation",
            "nav_or_cap_rate_assumption_support",
        ],
        sector_specific_risks=[
            "refinancing_risk",
            "tenant_concentration",
            "cap_rate_expansion",
            "maintenance_capex_understatement",
            "external_financing_dependence",
        ],
        normalization_rules=[
            "Use FFO/AFFO or NOI-based metrics instead of GAAP net income where depreciation distorts comparability.",
            "Normalize leverage for lease/property-level obligations and debt maturities.",
        ],
        weighting_rules=[
            "Weight balance-sheet/refinancing evidence before apparent NAV discounts.",
            "Require per-share FFO/AFFO evidence before selecting on yield.",
        ],
    ),
    "healthcare": _template(
        economic_model="Healthcare value creation driven by reimbursement durability, utilization, clinical/regulatory risk, payer mix, margin discipline, and cash conversion.",
        selected_value_drivers=[
            "reimbursement_durability",
            "volume_and_utilization_growth",
            "payer_mix",
            "regulatory_or_clinical_milestones",
            "gross_margin_durability",
            "free_cash_flow_conversion",
        ],
        selected_metrics=[
            "revenue_cagr_5y",
            "gross_margin",
            "operating_margin",
            "payer_or_customer_concentration",
            "r_and_d_or_clinical_spend_intensity",
            "free_cash_flow_margin",
            "regulatory_milestone_status",
        ],
        valid_valuation_methods=[
            "cash_flow_dcf",
            "owner_earnings_multiple",
            "pipeline_or_asset_risk_adjusted_sotp",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "unvalidated_revenue_multiple",
            "generic_book_value_anchor",
            "asset_liquidation_anchor",
            "preclinical_pipeline_value_without_milestone_evidence",
        ],
        required_evidence=[
            "payer_mix_or_reimbursement_evidence",
            "clinical_or_regulatory_status_evidence",
            "margin_and_cash_conversion_evidence",
            "customer_or_product_concentration_evidence",
            "latest_current_event_or_filing_update",
        ],
        sector_specific_risks=[
            "reimbursement_cut",
            "clinical_failure",
            "regulatory_delay",
            "customer_concentration",
            "cash_burn_or_financing_need",
        ],
        normalization_rules=[
            "Separate recurring care/service economics from one-time milestone, license, or collaboration revenue.",
            "Treat pre-commercial or cash-burning companies as model-fit constrained until financing runway and milestone evidence are explicit.",
        ],
        weighting_rules=[
            "Give regulatory/reimbursement evidence more weight than generic revenue growth.",
            "Require cash runway and dilution evidence before selecting early-stage healthcare names.",
        ],
        hurdle_thresholds=[("base_return_hurdle", "15", "percent_annualized")],
    ),
    "medical_devices": _template(
        economic_model="Medical-device compounding model driven by procedure volumes, installed base, utilization, reimbursement durability, FDA and quality-system status, gross-margin durability, channel concentration, and innovation cadence.",
        selected_value_drivers=[
            "procedure_volume_and_utilization",
            "installed_base_and_consumables_pull_through",
            "reimbursement_and_site_of_care_durability",
            "fda_quality_system_and_recall_status",
            "gross_margin_and_manufacturing_scale",
            "hospital_capex_or_purchasing_cycle",
            "sales_channel_and_customer_concentration",
            "product_pipeline_and_replacement_cycle",
        ],
        selected_metrics=[
            "procedure_volume_growth",
            "installed_base_growth",
            "consumables_or_recurring_revenue_mix",
            "gross_margin",
            "r_and_d_intensity",
            "sales_and_marketing_intensity",
            "free_cash_flow_margin",
            "customer_or_channel_concentration",
            "regulatory_or_recall_event_status",
        ],
        valid_valuation_methods=[
            "procedure_volume_installed_base_underwriting",
            "normalized_free_cash_flow",
            "unit_economics_dcf",
            "pipeline_or_product_cycle_risk_adjusted_sotp",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "unvalidated_revenue_multiple",
            "generic_book_value_anchor",
            "asset_liquidation_anchor",
            "pipeline_value_without_regulatory_or_adoption_evidence",
            "gross_margin_multiple_without_quality_or_recall_evidence",
        ],
        required_evidence=[
            "procedure_volume_utilization_or_installed_base_evidence",
            "reimbursement_site_of_care_or_payer_evidence",
            "fda_clearance_quality_system_or_recall_evidence",
            "gross_margin_manufacturing_and_cash_conversion_evidence",
            "channel_customer_concentration_or_hospital_capex_evidence",
        ],
        sector_specific_risks=[
            "procedure_volume_slowdown",
            "reimbursement_or_site_of_care_pressure",
            "fda_warning_letter_or_recall",
            "hospital_capex_cycle_delay",
            "surgeon_or_channel_adoption_stall",
            "manufacturing_quality_or_margin_pressure",
        ],
        normalization_rules=[
            "Separate durable installed-base or consumables growth from one-time stocking, distributor load-in, or procedure catch-up.",
            "Normalize margins for launch costs, manufacturing scale-up, quality-system remediation, and recall expenses.",
            "Require current FDA, recall, reimbursement, and hospital purchasing evidence before treating growth as durable.",
        ],
        weighting_rules=[
            "Weight procedure utilization, installed-base pull-through, reimbursement durability, and quality-system evidence before raw revenue growth.",
            "Penalize expected returns that rely on product-pipeline optionality without regulatory clearance, adoption, and cash-conversion evidence.",
        ],
        hurdle_thresholds=[("base_return_hurdle", "15", "percent_annualized")],
    ),
    "consumer": _template(
        economic_model="Consumer and retail unit-economics model driven by comparable sales, traffic and ticket, gross margin, inventory turns, lease obligations, brand durability, and store or channel expansion returns.",
        selected_value_drivers=[
            "same_store_sales_or_comparable_growth",
            "traffic_and_ticket_mix",
            "gross_margin_and_merchandise_margin",
            "inventory_turns_and_markdown_risk",
            "store_or_channel_unit_economics",
            "lease_adjusted_cash_flow",
            "brand_loyalty_and_customer_retention",
        ],
        selected_metrics=[
            "comparable_sales_growth",
            "traffic_growth",
            "average_ticket_growth",
            "gross_margin",
            "inventory_turnover",
            "lease_adjusted_net_debt_to_ebitdar",
            "free_cash_flow_margin",
            "store_count_or_channel_growth",
        ],
        valid_valuation_methods=[
            "lease_adjusted_owner_earnings",
            "normalized_free_cash_flow",
            "unit_economics_dcf",
            "roic_reinvestment_underwriting",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "unadjusted_revenue_multiple",
            "generic_book_value_anchor",
            "ebitda_multiple_without_lease_adjustment",
            "peak_margin_multiple_without_markdown_cycle",
            "store_growth_multiple_without_unit_economics",
        ],
        required_evidence=[
            "same_store_sales_or_traffic_evidence",
            "gross_margin_and_input_cost_evidence",
            "inventory_turnover_and_markdown_evidence",
            "lease_obligation_and_store_base_evidence",
            "unit_economics_or_channel_profitability_evidence",
        ],
        sector_specific_risks=[
            "traffic_decline",
            "markdown_and_inventory_obsolescence",
            "lease_fixed_cost_deleverage",
            "brand_or_customer_relevance_decay",
            "supplier_or_input_cost_pressure",
            "channel_shift_margin_dilution",
        ],
        normalization_rules=[
            "Normalize margins through inventory and markdown cycles before comparing expected returns.",
            "Adjust leverage and cash flow for leases and store closure obligations.",
            "Separate profitable unit growth from sales growth driven by uneconomic promotions.",
        ],
        weighting_rules=[
            "Weight comparable sales, inventory discipline, and lease-adjusted cash conversion before raw revenue growth.",
            "Penalize expected returns that rely on store or channel expansion without unit-economics evidence.",
        ],
        hurdle_thresholds=[("base_return_hurdle", "13", "percent_annualized")],
    ),
    "automotive": _template(
        economic_model="Automotive and mobility cycle model driven by unit volumes, pricing and incentive discipline, platform and powertrain transitions, supplier or dealer channel health, warranty quality, capital intensity, inventory, and residual-value risk.",
        selected_value_drivers=[
            "unit_volume_and_mix",
            "pricing_and_incentive_discipline",
            "platform_or_powertrain_transition",
            "supplier_or_dealer_channel_health",
            "warranty_quality_and_recall_risk",
            "inventory_and_working_capital_cycle",
            "capital_intensity_and_tooling_reinvestment",
            "finance_or_residual_value_exposure",
        ],
        selected_metrics=[
            "unit_sales_growth",
            "average_selling_price",
            "incentive_spend_ratio",
            "gross_margin",
            "warranty_expense_ratio",
            "inventory_days",
            "capital_expenditures_to_sales",
            "free_cash_flow_through_cycle",
            "net_debt_to_ebitda",
        ],
        valid_valuation_methods=[
            "cycle_normalized_owner_earnings",
            "mid_cycle_earnings_power",
            "sum_of_parts_for_finance_or_parts_segments",
            "roic_reinvestment_underwriting",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "unadjusted_revenue_multiple",
            "unadjusted_peak_earnings_multiple",
            "ebitda_multiple_without_tooling_and_warranty_costs",
            "unit_growth_multiple_without_pricing_or_inventory_evidence",
            "book_value_anchor_without_finance_residual_risk",
        ],
        required_evidence=[
            "unit_volume_mix_and_pricing_evidence",
            "incentive_inventory_and_channel_evidence",
            "platform_powertrain_or_capex_plan_evidence",
            "warranty_recall_or_quality_evidence",
            "finance_residual_value_or_leverage_evidence",
        ],
        sector_specific_risks=[
            "auto_cycle_downturn",
            "pricing_incentive_pressure",
            "platform_transition_execution",
            "supplier_or_dealer_channel_stress",
            "warranty_or_recall_liability",
            "residual_value_or_finance_credit_loss",
        ],
        normalization_rules=[
            "Normalize volumes, margins, and cash flow across auto demand and inventory cycles before comparing expected returns.",
            "Separate recurring tooling, warranty, and platform-transition spending from discretionary growth capital.",
            "Require downside sensitivity around incentives, residual values, credit losses, and supplier or dealer stress.",
        ],
        weighting_rules=[
            "Weight pricing discipline, inventory/channel evidence, warranty quality, and through-cycle cash flow before headline unit growth.",
            "Penalize expected returns that rely on peak-cycle margins, unsupported platform transitions, or ignored finance/residual-value exposure.",
        ],
        hurdle_thresholds=[("base_return_hurdle", "15", "percent_annualized")],
    ),
    "communications_media": _template(
        economic_model="Communications and media model driven by subscriber or audience growth, ARPU and pricing, churn or engagement retention, network or content investment, advertising cyclicality, leverage, and free-cash-flow conversion.",
        selected_value_drivers=[
            "subscriber_or_audience_growth",
            "arpu_and_pricing",
            "churn_or_engagement_retention",
            "network_capex_or_content_investment",
            "advertising_and_affiliate_revenue_mix",
            "spectrum_or_distribution_rights",
            "leverage_and_refinancing_capacity",
            "free_cash_flow_conversion",
        ],
        selected_metrics=[
            "subscriber_growth",
            "arpu_growth",
            "churn_rate",
            "broadband_or_wireless_net_adds",
            "advertising_revenue_growth",
            "content_or_programming_cost_ratio",
            "capital_intensity",
            "net_debt_to_ebitda",
            "free_cash_flow_margin",
        ],
        valid_valuation_methods=[
            "subscriber_ltv_to_cac",
            "normalized_free_cash_flow",
            "dcf_with_capex_or_content_cycle",
            "network_or_content_asset_sum_of_parts",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "unadjusted_revenue_multiple",
            "ebitda_multiple_without_capex_or_content_costs",
            "subscriber_multiple_without_churn_or_arpu",
            "advertising_peak_multiple_without_cycle_normalization",
            "book_value_anchor_without_spectrum_or_content_context",
        ],
        required_evidence=[
            "subscriber_arpu_and_churn_evidence",
            "network_capex_or_content_spend_evidence",
            "advertising_or_affiliate_mix_evidence",
            "spectrum_distribution_or_rights_evidence",
            "leverage_and_refinancing_evidence",
        ],
        sector_specific_risks=[
            "subscriber_churn_or_price_compression",
            "content_cost_inflation",
            "advertising_cycle_drawdown",
            "network_capex_underinvestment",
            "regulatory_or_spectrum_risk",
            "leverage_refinancing_risk",
        ],
        normalization_rules=[
            "Normalize advertising, affiliate, and subscriber acquisition economics across demand cycles before comparing expected returns.",
            "Adjust cash flow for recurring network capex, content obligations, spectrum costs, and distribution rights.",
            "Separate durable subscriber or audience growth from promotional additions with weak retention evidence.",
        ],
        weighting_rules=[
            "Weight subscriber quality, ARPU durability, churn, capex/content obligations, and leverage before headline revenue growth.",
            "Penalize expected returns that rely on peak advertising conditions, unsupported subscriber multiples, or deferred network investment.",
        ],
    ),
    "industrial": _template(
        economic_model="Industrial compounding model driven by backlog conversion, pricing versus input costs, operating leverage, maintenance capex, working-capital turns, and cycle-normalized margins.",
        selected_value_drivers=[
            "organic_order_and_backlog_conversion",
            "pricing_power_vs_input_costs",
            "gross_margin_through_cycle",
            "operating_leverage",
            "maintenance_capex_intensity",
            "working_capital_turns",
            "per_share_capital_allocation",
        ],
        selected_metrics=[
            "revenue_cagr_5y",
            "book_to_bill_or_backlog_growth",
            "gross_margin",
            "normalized_operating_margin",
            "maintenance_capex_to_sales",
            "cash_conversion_cycle",
            "share_count_cagr",
        ],
        valid_valuation_methods=[
            "cycle_normalized_owner_earnings",
            "dcf",
            "epv_with_cycle_normalization",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "unadjusted_trough_earnings_multiple",
            "unadjusted_peak_earnings_multiple",
            "generic_book_value_anchor",
            "revenue_multiple_without_margin_evidence",
        ],
        required_evidence=[
            "backlog_or_order_trend_evidence",
            "pricing_and_input_cost_evidence",
            "maintenance_capex_evidence",
            "working_capital_evidence",
            "cycle_normalization_evidence",
        ],
        sector_specific_risks=[
            "cyclical_order_decline",
            "input_cost_inflation",
            "customer_concentration",
            "maintenance_capex_underestimate",
            "working_capital_reversal",
        ],
        normalization_rules=[
            "Normalize margins and earnings across the cycle before comparing expected returns.",
            "Separate maintenance capex from growth capex when estimating owner earnings.",
        ],
        weighting_rules=[
            "Weight backlog/order quality and cash conversion above generic growth scores.",
            "Penalize expected returns that depend on peak-cycle margin continuation.",
        ],
    ),
    "energy": _template(
        economic_model="Energy asset and cash-flow model driven by reserve quality, production decline, realized commodity pricing, hedge coverage, reinvestment intensity, and cycle-normalized free cash flow.",
        selected_value_drivers=[
            "reserve_quality_and_life",
            "production_decline_and_replacement",
            "realized_price_vs_benchmark",
            "hedge_coverage_and_rolloff",
            "lifting_cost_and_margin",
            "maintenance_capex_and_reinvestment",
            "balance_sheet_and_decommissioning_liabilities",
        ],
        selected_metrics=[
            "production_growth_or_decline",
            "reserve_life_index",
            "finding_and_development_cost",
            "lifting_cost_per_unit",
            "realized_price_vs_benchmark",
            "hedge_coverage",
            "free_cash_flow_after_maintenance_capex",
            "net_debt_to_ebitda",
        ],
        valid_valuation_methods=[
            "proved_reserve_nav",
            "cycle_normalized_free_cash_flow",
            "commodity_sensitivity_nav",
            "recycle_ratio_underwriting",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "unadjusted_revenue_multiple",
            "spot_price_extrapolation_without_sensitivity",
            "generic_book_value_anchor",
            "ebitda_multiple_without_maintenance_capex",
            "dcf_without_commodity_sensitivity",
        ],
        required_evidence=[
            "reserve_report_or_production_evidence",
            "commodity_price_and_hedge_evidence",
            "lifting_cost_and_margin_evidence",
            "maintenance_capex_and_decline_rate_evidence",
            "decommissioning_or_environmental_liability_evidence",
        ],
        sector_specific_risks=[
            "commodity_price_drawdown",
            "reserve_replacement_failure",
            "decline_rate_underinvestment",
            "hedge_rolloff",
            "environmental_or_decommissioning_liability",
            "leverage_refinancing_risk",
        ],
        normalization_rules=[
            "Normalize cash flow across commodity cycles instead of extrapolating spot prices.",
            "Separate maintenance capital needed to hold production flat from growth capital before underwriting owner earnings.",
            "Require commodity sensitivity around price, hedge rolloff, and decline-rate assumptions.",
        ],
        weighting_rules=[
            "Weight reserve quality, hedge coverage, and maintenance-capex evidence before near-term earnings momentum.",
            "Penalize expected returns that require sustained peak commodity prices without downside sensitivity.",
        ],
        hurdle_thresholds=[("base_return_hurdle", "16", "percent_annualized")],
    ),
    "materials": _template(
        economic_model="Materials cycle model driven by commodity and feedstock spreads, volume mix, cost-curve position, capacity utilization, sustaining capital, environmental liabilities, and balance-sheet resilience.",
        selected_value_drivers=[
            "commodity_or_feedstock_spread",
            "volume_and_mix",
            "cost_curve_position",
            "capacity_utilization",
            "sustaining_capex_intensity",
            "working_capital_cycle",
            "environmental_and_reclamation_liabilities",
            "balance_sheet_resilience",
        ],
        selected_metrics=[
            "realized_price_vs_benchmark",
            "feedstock_cost_spread",
            "production_volume_growth",
            "capacity_utilization",
            "cash_cost_per_unit",
            "sustaining_capex_to_sales",
            "free_cash_flow_through_cycle",
            "net_debt_to_ebitda",
        ],
        valid_valuation_methods=[
            "cycle_normalized_free_cash_flow",
            "mid_cycle_earnings_power",
            "commodity_sensitivity_nav",
            "replacement_cost_with_cycle_check",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "unadjusted_revenue_multiple",
            "spot_price_extrapolation_without_sensitivity",
            "unadjusted_peak_earnings_multiple",
            "ebitda_multiple_without_sustaining_capex",
            "book_value_anchor_without_impairment_check",
        ],
        required_evidence=[
            "commodity_price_or_feedstock_spread_evidence",
            "volume_and_capacity_utilization_evidence",
            "cost_curve_or_cash_cost_evidence",
            "sustaining_capex_and_working_capital_evidence",
            "environmental_reclamation_or_regulatory_liability_evidence",
        ],
        sector_specific_risks=[
            "commodity_spread_compression",
            "feedstock_cost_inflation",
            "overcapacity_cycle",
            "volume_curtailment",
            "environmental_or_reclamation_liability",
            "leverage_refinancing_risk",
        ],
        normalization_rules=[
            "Normalize margins and cash flow across commodity and feedstock cycles before comparing expected returns.",
            "Separate sustaining capital and reclamation obligations from discretionary growth capital.",
            "Require downside sensitivity around realized prices, input costs, utilization, and working-capital swings.",
        ],
        weighting_rules=[
            "Weight cost-curve position, spread evidence, sustaining-capex coverage, and balance-sheet resilience before headline revenue growth.",
            "Penalize expected returns that rely on peak-cycle pricing, tight supply conditions, or unsupported utilization recovery.",
        ],
        hurdle_thresholds=[("base_return_hurdle", "15", "percent_annualized")],
    ),
    "utilities": _template(
        economic_model="Regulated utility compounding model driven by rate-base growth, allowed ROE, regulatory construct quality, capital-plan execution, customer affordability, leverage, and dividend sustainability.",
        selected_value_drivers=[
            "rate_base_growth",
            "allowed_roe_and_equity_ratio",
            "regulatory_recovery_mechanisms",
            "capital_plan_execution",
            "customer_affordability_and_load_growth",
            "debt_funding_capacity",
            "dividend_coverage",
        ],
        selected_metrics=[
            "rate_base_cagr",
            "allowed_roe",
            "equity_ratio",
            "regulated_capex_plan",
            "funds_from_operations_to_debt",
            "debt_to_capital",
            "dividend_payout_ratio",
            "customer_bill_growth",
        ],
        valid_valuation_methods=[
            "rate_base_compounding",
            "allowed_roe_to_book_value",
            "dividend_discount_model",
            "regulated_utility_expected_return",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "unadjusted_revenue_multiple",
            "generic_dcf_without_rate_case_support",
            "ebitda_multiple_without_capex_and_debt_funding",
            "asset_liquidation_anchor",
            "peak_earnings_multiple_without_regulatory_normalization",
        ],
        required_evidence=[
            "rate_case_or_regulatory_order_evidence",
            "rate_base_and_capex_plan_evidence",
            "allowed_roe_and_equity_ratio_evidence",
            "debt_funding_and_credit_metric_evidence",
            "dividend_coverage_and_affordability_evidence",
        ],
        sector_specific_risks=[
            "adverse_rate_case",
            "capex_disallowance",
            "customer_affordability_pressure",
            "credit_rating_downgrade",
            "equity_issuance_dilution",
            "storm_or_environmental_liability",
        ],
        normalization_rules=[
            "Normalize earnings around approved rate cases and recovery mechanisms rather than trailing reported EPS alone.",
            "Treat utility capex as a funding and regulatory recovery question, not a simple growth expense.",
            "Require leverage and dividend coverage evidence before underwriting yield as return support.",
        ],
        weighting_rules=[
            "Weight regulatory construct, credit metrics, and rate-base evidence above raw dividend yield.",
            "Penalize expected returns that depend on unsupported rate-base growth or unapproved recovery assumptions.",
        ],
        hurdle_thresholds=[("base_return_hurdle", "10", "percent_annualized")],
    ),
    "capital_markets": _template(
        economic_model="Capital-markets fee-earnings model driven by assets under management, net flows, fee rates, performance fees, compensation discipline, regulatory capital, and capital intensity.",
        selected_value_drivers=[
            "assets_under_management",
            "organic_net_flows",
            "management_fee_rate",
            "performance_fee_normalization",
            "compensation_ratio",
            "regulatory_capital_and_capital_intensity",
        ],
        selected_metrics=[
            "aum_growth",
            "organic_net_flow_rate",
            "effective_fee_rate",
            "normalized_fee_related_earnings",
            "compensation_to_revenue",
            "excess_regulatory_capital",
        ],
        valid_valuation_methods=[
            "normalized_fee_earnings",
            "aum_fee_rate_model",
            "excess_capital_return",
            "segment_sum_of_the_parts",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "unadjusted_revenue_multiple",
            "peak_performance_fee_multiple",
            "generic_enterprise_value_multiple_without_regulatory_capital",
        ],
        required_evidence=[
            "aum_and_net_flow_evidence",
            "fee_rate_and_performance_fee_evidence",
            "compensation_and_fee_earnings_evidence",
            "regulatory_and_excess_capital_evidence",
        ],
        sector_specific_risks=[
            "market_beta_to_aum",
            "persistent_net_outflows",
            "fee_compression",
            "performance_fee_reversal",
            "regulatory_capital_shortfall",
        ],
        normalization_rules=[
            "Separate market appreciation from organic net flows in AUM growth.",
            "Normalize performance fees and principal-investment gains across a full cycle.",
        ],
        weighting_rules=[
            "Weight organic flows, normalized fee earnings, and excess capital above headline AUM growth.",
        ],
        hurdle_thresholds=[("base_return_hurdle", "13", "percent_annualized")],
    ),
    "education_services": _template(
        economic_model="Education-services cohort model driven by enrollment, retention, revenue per learner, verified outcomes, acquisition payback, accreditation, and Title IV or equivalent regulatory exposure.",
        selected_value_drivers=[
            "enrollment_growth",
            "student_retention_and_completion",
            "revenue_per_learner",
            "verified_student_outcomes",
            "learner_acquisition_cost_and_payback",
            "accreditation_and_title_iv_exposure",
        ],
        selected_metrics=[
            "new_and_total_enrollment",
            "retention_and_completion_rate",
            "revenue_per_learner",
            "graduate_employment_or_outcome_rate",
            "learner_acquisition_payback",
            "title_iv_revenue_share",
        ],
        valid_valuation_methods=[
            "cohort_unit_economics",
            "normalized_owner_earnings",
            "lease_adjusted_free_cash_flow",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "enrollment_multiple_without_retention",
            "revenue_multiple_without_outcomes",
            "ebitda_multiple_without_lease_adjustment",
        ],
        required_evidence=[
            "enrollment_and_retention_evidence",
            "student_outcome_evidence",
            "accreditation_and_regulatory_evidence",
            "marketing_spend_and_acquisition_evidence",
        ],
        sector_specific_risks=[
            "enrollment_contraction",
            "poor_student_outcomes",
            "accreditation_loss",
            "title_iv_or_regulatory_sanction",
            "marketing_payback_deterioration",
        ],
        normalization_rules=[
            "Normalize enrollment cohorts for retention and completion rather than treating starts as durable revenue.",
            "Lease-adjust owner earnings and separate learner acquisition spend from unsupported growth claims.",
        ],
        weighting_rules=[
            "Weight verified outcomes, retention, and regulatory standing above enrollment growth.",
        ],
        hurdle_thresholds=[("base_return_hurdle", "15", "percent_annualized")],
    ),
    "hospitality_gaming": _template(
        economic_model="Hospitality and gaming property model driven by occupancy, ADR and RevPAR or same-property gaming revenue, normalized hold, lease-adjusted leverage, maintenance capital, and licensing resilience.",
        selected_value_drivers=[
            "occupancy_adr_and_revpar",
            "same_property_gaming_revenue",
            "gaming_hold_normalization",
            "lease_adjusted_leverage",
            "maintenance_capex",
            "licensing_and_regulatory_resilience",
        ],
        selected_metrics=[
            "occupancy_rate",
            "average_daily_rate",
            "revpar",
            "same_property_gaming_revenue_growth",
            "normalized_hold_percentage",
            "lease_adjusted_net_leverage",
            "maintenance_capex_to_revenue",
        ],
        valid_valuation_methods=[
            "normalized_free_cash_flow",
            "lease_adjusted_owner_earnings",
            "property_or_segment_sum_of_the_parts",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "unadjusted_ebitda_multiple_without_leases",
            "peak_hold_earnings_multiple",
            "revenue_multiple_without_property_economics",
        ],
        required_evidence=[
            "occupancy_adr_revpar_or_gaming_kpi_evidence",
            "lease_and_debt_obligation_evidence",
            "maintenance_and_growth_capex_evidence",
            "gaming_license_and_regulatory_evidence",
        ],
        sector_specific_risks=[
            "travel_or_discretionary_demand_shock",
            "gaming_hold_volatility",
            "lease_adjusted_leverage",
            "deferred_property_maintenance",
            "license_or_regulatory_loss",
        ],
        normalization_rules=[
            "Normalize gaming hold and property demand across a cycle before ranking expected returns.",
            "Treat leases as financing obligations and separate maintenance from expansion capital.",
        ],
        weighting_rules=[
            "Weight lease-adjusted cash flow, maintenance coverage, and regulatory standing above reported EBITDA.",
        ],
        hurdle_thresholds=[("base_return_hurdle", "15", "percent_annualized")],
    ),
    "technology": _template(
        economic_model="Technology compounding model driven by retention, durable growth, gross margin, product reinvestment efficiency, free-cash-flow conversion, and dilution discipline.",
        selected_value_drivers=[
            "net_revenue_retention_or_churn",
            "durable_organic_revenue_growth",
            "gross_margin_stability",
            "r_and_d_efficiency",
            "free_cash_flow_conversion",
            "stock_based_compensation_and_share_dilution",
        ],
        selected_metrics=[
            "revenue_cagr_5y",
            "gross_margin",
            "operating_margin",
            "free_cash_flow_margin",
            "sbc_percent_revenue",
            "share_count_cagr",
            "net_revenue_retention",
        ],
        valid_valuation_methods=[
            "technology_adjusted_dcf",
            "cash_flow_dcf",
            "owner_earnings_multiple",
            "rule_of_40_contextual_check",
            "expected_return_scenarios",
        ],
        invalid_valuation_methods=[
            "generic_book_value_anchor",
            "asset_liquidation_anchor",
            "revenue_multiple_without_profitability_path",
            "ebitda_multiple_excluding_sbc_without_dilution_cost",
        ],
        required_evidence=[
            "retention_or_churn_evidence",
            "gross_margin_and_fcf_evidence",
            "sbc_and_share_count_evidence",
            "product_or_customer_concentration_evidence",
            "latest_current_event_or_filing_update",
        ],
        sector_specific_risks=[
            "growth_deceleration",
            "platform_or_customer_concentration",
            "sbc_dilution",
            "technology_obsolescence",
            "unprofitable_growth",
        ],
        normalization_rules=[
            "Treat stock-based compensation as a real dilution cost when assessing per-share value creation.",
            "Do not capitalize growth narratives without retention, gross-margin, and FCF evidence.",
        ],
        weighting_rules=[
            "Weight retention and free-cash-flow conversion above raw revenue growth.",
            "Cap confidence when expected return depends on unvalidated revenue multiples.",
        ],
        hurdle_thresholds=[("base_return_hurdle", "14", "percent_annualized")],
    ),
}


# Every canonical all-sector scan token resolves by exact match before the
# looser narrative aliases below. This is the product contract surface: adding
# a sector requires an explicit mapping, even when it intentionally shares an
# established financial framework.
CANONICAL_SECTOR_FRAMEWORK_CONTRACTS: dict[str, str] = {
    "aerospace_defense": "industrial",
    "automotive": "automotive",
    "biotech": "healthcare",
    "building_products": "industrial",
    "business_services": "industrial",
    "capital_markets": "capital_markets",
    "chemicals": "materials",
    "construction_machinery": "industrial",
    "construction_services": "industrial",
    "consumer_discretionary": "consumer",
    "consumer_services": "consumer",
    "consumer_staples": "consumer",
    "diversified_industrials": "industrial",
    "education_services": "education_services",
    "energy": "energy",
    "enterprise_software": "technology",
    "healthcare_pharma": "healthcare",
    "healthcare_services": "healthcare",
    "hospitality_gaming": "hospitality_gaming",
    "industrial_tech": "industrial",
    "insurance": "financial_services",
    "internet_services": "technology",
    "large_cap_financials": "financial_services",
    "media_entertainment": "communications_media",
    "medical_devices": "medical_devices",
    "metals_mining": "materials",
    "payments_fintech": "payments_fintech",
    "reits": "real_estate",
    "restaurants_food_service": "consumer",
    "retail": "consumer",
    "semiconductors": "technology",
    "telecom": "communications_media",
    "transportation_logistics": "industrial",
    "utilities": "utilities",
}


# V2 evaluates only the deterministic structural rules declared by the
# selected sector contract.  These IDs intentionally mirror the stable rule
# IDs emitted by ``app.autonomous.structural_gate`` without importing the
# evaluator into this pure template module.
_COMMON_DETERMINISTIC_SCREEN_RULE_IDS: tuple[str, ...] = (
    "NON_PRIMARY_LISTING",
    "DELISTING_NOTICE",
    "PENNY_FLOOR",
    "NANO_FLOOR",
    "EARNINGS_QUALITY_DIVERGENCE",
    "GOING_CONCERN",
)
_BALANCE_SHEET_DETERMINISTIC_SCREEN_RULE_IDS: tuple[str, ...] = (
    "NON_PRIMARY_LISTING",
    "DELISTING_NOTICE",
    "PENNY_FLOOR",
    "NANO_FLOOR",
    "GOING_CONCERN",
)

# Generic net-income-versus-CFO divergence is not a valid deterministic
# rejection rule for balance-sheet financials, capital-markets firms, or
# REIT-style businesses.  Their cash-flow statement economics require the
# sector-specific underwriting lens instead.  Every concrete framework,
# including the generic v2 fallback, is named here so applicability cannot be
# inherited accidentally when a framework is added.
V2_FRAMEWORK_SCREEN_RULE_IDS: dict[str, tuple[str, ...]] = {
    "automotive": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "capital_markets": _BALANCE_SHEET_DETERMINISTIC_SCREEN_RULE_IDS,
    "communications_media": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "consumer": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "education_services": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "energy": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "financial_services": _BALANCE_SHEET_DETERMINISTIC_SCREEN_RULE_IDS,
    "generic": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "healthcare": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "hospitality_gaming": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "industrial": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "materials": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "medical_devices": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "payments_fintech": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "real_estate": _BALANCE_SHEET_DETERMINISTIC_SCREEN_RULE_IDS,
    "technology": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "utilities": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
}

# The canonical-sector registry is deliberately explicit rather than derived
# at runtime from CANONICAL_SECTOR_FRAMEWORK_CONTRACTS.  A new sweep sector
# therefore cannot enter v2 without choosing its deterministic screen rules.
V2_CANONICAL_SECTOR_SCREEN_RULE_IDS: dict[str, tuple[str, ...]] = {
    "aerospace_defense": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "automotive": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "biotech": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "building_products": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "business_services": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "capital_markets": _BALANCE_SHEET_DETERMINISTIC_SCREEN_RULE_IDS,
    "chemicals": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "construction_machinery": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "construction_services": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "consumer_discretionary": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "consumer_services": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "consumer_staples": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "diversified_industrials": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "education_services": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "energy": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "enterprise_software": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "healthcare_pharma": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "healthcare_services": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "hospitality_gaming": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "industrial_tech": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "insurance": _BALANCE_SHEET_DETERMINISTIC_SCREEN_RULE_IDS,
    "internet_services": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "large_cap_financials": _BALANCE_SHEET_DETERMINISTIC_SCREEN_RULE_IDS,
    "media_entertainment": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "medical_devices": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "metals_mining": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "payments_fintech": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "reits": _BALANCE_SHEET_DETERMINISTIC_SCREEN_RULE_IDS,
    "restaurants_food_service": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "retail": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "semiconductors": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "telecom": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "transportation_logistics": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
    "utilities": _COMMON_DETERMINISTIC_SCREEN_RULE_IDS,
}


_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("payments_fintech", ("payments", "payment", "fintech", "financial technology", "merchant acquiring", "card network", "processor", "wallet")),
    ("financial_services", ("insurance", "insurer", "bank", "financial", "lender", "credit", "asset manager")),
    ("real_estate", ("real estate", "reit", "property", "housing", "rental")),
    ("medical_devices", ("medical devices", "medical device", "medtech", "diagnostics", "surgical device", "implant", "hospital equipment")),
    ("healthcare", ("health", "biotech", "pharma", "medical", "hospital", "care", "device")),
    ("consumer", ("consumer", "retail", "restaurant", "food service", "staples", "discretionary", "apparel", "grocery", "brand")),
    ("automotive", ("automotive", "auto", "vehicle", "vehicles", "mobility", "automaker", "auto parts", "dealer network")),
    ("communications_media", ("telecom", "communications", "communication services", "media", "entertainment", "broadband", "wireless", "cable", "streaming", "content")),
    ("technology", ("technology", "software", "semiconductor", "internet", "cloud", "cyber", "data")),
    ("industrial", ("industrial", "manufacturing", "manufacturer", "machinery", "aerospace", "distribution", "logistics")),
    ("materials", ("materials", "basic materials", "metals", "mining", "chemical", "chemicals", "fertilizer", "steel", "copper", "gold", "lithium", "paper", "packaging")),
    ("energy", ("energy", "oil", "gas", "exploration", "production", "midstream", "pipeline", "commodity")),
    ("utilities", ("utility", "utilities", "regulated power", "electric", "water utility", "gas utility")),
)


_ACTIVE_PIPELINE_VERSION: ContextVar[str] = ContextVar(
    "autonomous_sector_framework_pipeline_version", default="v1"
)


@contextmanager
def sector_framework_pipeline_version(pipeline_version: str):
    """Scope framework routing to the explicitly selected pipeline version."""

    normalized = str(pipeline_version or "v1").strip().lower()
    if normalized not in {"v1", "v2"}:
        raise ValueError("pipeline version must be 'v1' or 'v2'")
    token = _ACTIVE_PIPELINE_VERSION.set(normalized)
    try:
        yield
    finally:
        _ACTIVE_PIPELINE_VERSION.reset(token)


def _template_key(sector: str, *, pipeline_version: str | None = None) -> str | None:
    canonical = str(sector or "").strip().lower().replace("-", "_").replace(" ", "_")
    active_pipeline = str(
        pipeline_version or _ACTIVE_PIPELINE_VERSION.get()
    ).strip().lower()
    if active_pipeline == "v2" and canonical in CANONICAL_SECTOR_FRAMEWORK_CONTRACTS:
        return CANONICAL_SECTOR_FRAMEWORK_CONTRACTS[canonical]
    normalized = canonical.replace("_", " ")
    for key, aliases in _ALIASES:
        if any(alias in normalized for alias in aliases):
            return key
    return None


def sector_framework_contract_id(
    sector: str, *, pipeline_version: str | None = None
) -> str:
    """Return the explicit framework contract key or the generic fallback."""

    return _template_key(sector, pipeline_version=pipeline_version) or "generic"


def sector_framework_screen_rule_ids(
    sector: str, *, pipeline_version: str | None = None
) -> list[str]:
    """Return the v2 contract's applicable deterministic screen-rule IDs.

    V1 has no sector-rule registry and therefore returns an empty list.  The
    returned list is a copy so callers cannot mutate the immutable registry.
    """

    active_pipeline = str(
        pipeline_version or _ACTIVE_PIPELINE_VERSION.get()
    ).strip().lower()
    if active_pipeline != "v2":
        return []
    canonical = str(sector or "").strip().lower().replace("-", "_").replace(" ", "_")
    if canonical in V2_CANONICAL_SECTOR_SCREEN_RULE_IDS:
        return list(V2_CANONICAL_SECTOR_SCREEN_RULE_IDS[canonical])
    contract_id = _template_key(sector, pipeline_version="v2") or "generic"
    return list(V2_FRAMEWORK_SCREEN_RULE_IDS[contract_id])


def _dedupe(items: list[Any]) -> list[str]:
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _threshold_key(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("name") or "").strip().lower()


def _merge_policy(raw: Any, template: dict[str, Any]) -> dict[str, Any]:
    raw_policy = raw if isinstance(raw, dict) else {}
    approach = str(raw_policy.get("approach") or template.get("approach") or "")
    raw_rules = raw_policy.get("rules") if isinstance(raw_policy.get("rules"), list) else []
    template_rules = template.get("rules") if isinstance(template.get("rules"), list) else []
    thresholds: list[dict[str, str]] = []
    seen_thresholds: set[str] = set()
    for item in [
        *(raw_policy.get("numeric_thresholds") if isinstance(raw_policy.get("numeric_thresholds"), list) else []),
        *(template.get("numeric_thresholds") if isinstance(template.get("numeric_thresholds"), list) else []),
    ]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        key = _threshold_key(item)
        if key in seen_thresholds:
            continue
        seen_thresholds.add(key)
        thresholds.append(
            {
                "name": name,
                "value": str(item.get("value") or ""),
                "units": str(item.get("units") or ""),
            }
        )
    return {
        "approach": approach,
        "rules": _dedupe([*raw_rules, *template_rules]),
        "numeric_thresholds": thresholds,
    }


def sector_framework_template_payload(
    sector: str,
    market_cap_focus: str,
    *,
    pipeline_version: str | None = None,
) -> dict[str, Any]:
    key = _template_key(sector, pipeline_version=pipeline_version)
    template = deepcopy(_TEMPLATES.get(key) or _template(
        economic_model="Financially underwritten 5-10 year per-share return analysis.",
        selected_value_drivers=["per_share_return", "business_quality", "cash_conversion", "capital_allocation"],
        selected_metrics=["base_case_annualized_return", "valuation_anchor", "revenue_quality", "balance_sheet_risk"],
        valid_valuation_methods=["deterministic_valuation_anchor", "expected_return_scenarios"],
        invalid_valuation_methods=["unsupported_narrative_multiple"],
        required_evidence=["expected_return_evidence", "company_specific_evidence", "filing_or_current_evidence"],
        sector_specific_risks=["evidence_quality", "cyclicality", "capital_intensity", "valuation_anchor_error"],
        normalization_rules=["Use explicit model-fit blockers when a generic valuation method is inappropriate."],
        weighting_rules=["Do not let generic quality metrics override binding model-fit blockers."],
    ))
    template["framework_contract_id"] = key or "generic"
    template["sector"] = sector
    template["market_cap_focus"] = market_cap_focus
    active_pipeline = str(
        pipeline_version or _ACTIVE_PIPELINE_VERSION.get()
    ).strip().lower()
    if active_pipeline == "v2":
        template["applicable_screen_rule_ids"] = sector_framework_screen_rule_ids(
            sector,
            pipeline_version="v2",
        )
    return template


def augment_sector_framework_payload(
    payload: dict[str, Any] | None,
    *,
    sector: str,
    market_cap_focus: str,
    pipeline_version: str | None = None,
) -> dict[str, Any]:
    template = sector_framework_template_payload(
        sector,
        market_cap_focus,
        pipeline_version=pipeline_version,
    )
    raw = dict(payload or {})
    economic_model = str(raw.get("economic_model") or "").strip()
    if not economic_model or economic_model.lower() in _GENERIC_ECONOMIC_MODELS:
        economic_model = str(template["economic_model"])
    result = {
        "sector": str(raw.get("sector") or sector),
        "market_cap_focus": str(raw.get("market_cap_focus") or market_cap_focus),
        "horizon_years": [int(item) for item in (raw.get("horizon_years") or template["horizon_years"] or [5, 10])],
        "economic_model": economic_model,
        "framework_contract_id": str(template["framework_contract_id"]),
        "selected_value_drivers": _dedupe(
            [*(raw.get("selected_value_drivers") if isinstance(raw.get("selected_value_drivers"), list) else []), *template["selected_value_drivers"]]
        ),
        "selected_metrics": _dedupe(
            [*(raw.get("selected_metrics") if isinstance(raw.get("selected_metrics"), list) else []), *template["selected_metrics"]]
        ),
        "valid_valuation_methods": _dedupe(
            [*(raw.get("valid_valuation_methods") if isinstance(raw.get("valid_valuation_methods"), list) else []), *template["valid_valuation_methods"]]
        ),
        "invalid_valuation_methods": _dedupe(
            [*(raw.get("invalid_valuation_methods") if isinstance(raw.get("invalid_valuation_methods"), list) else []), *template["invalid_valuation_methods"]]
        ),
        "required_evidence": _dedupe(
            [*(raw.get("required_evidence") if isinstance(raw.get("required_evidence"), list) else []), *template["required_evidence"]]
        ),
        "normalization_policy": _merge_policy(raw.get("normalization_policy"), template["normalization_policy"]),
        "hurdle_rate_policy": _merge_policy(raw.get("hurdle_rate_policy"), template["hurdle_rate_policy"]),
        "weighting_policy": _merge_policy(raw.get("weighting_policy"), template["weighting_policy"]),
        "sector_specific_risks": _dedupe(
            [*(raw.get("sector_specific_risks") if isinstance(raw.get("sector_specific_risks"), list) else []), *template["sector_specific_risks"]]
        ),
    }
    if "applicable_screen_rule_ids" in template:
        result["applicable_screen_rule_ids"] = list(
            template["applicable_screen_rule_ids"]
        )
    return result


__all__ = [
    "CANONICAL_SECTOR_FRAMEWORK_CONTRACTS",
    "V2_CANONICAL_SECTOR_SCREEN_RULE_IDS",
    "V2_FRAMEWORK_SCREEN_RULE_IDS",
    "augment_sector_framework_payload",
    "sector_framework_contract_id",
    "sector_framework_pipeline_version",
    "sector_framework_screen_rule_ids",
    "sector_framework_template_payload",
]
