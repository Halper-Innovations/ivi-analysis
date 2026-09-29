from app.autonomous.sector_framework_templates import (
    CANONICAL_SECTOR_FRAMEWORK_CONTRACTS,
    V2_CANONICAL_SECTOR_SCREEN_RULE_IDS,
    V2_FRAMEWORK_SCREEN_RULE_IDS,
    augment_sector_framework_payload,
    sector_framework_contract_id,
    sector_framework_screen_rule_ids,
    sector_framework_template_payload,
)
from app.autonomous.sweep_delta import CANONICAL_SWEEP_SECTORS
from app.sector.canonical_taxonomy import ACTIVE_SECTOR_LABELS


_COMMON_SCREEN_RULE_IDS = (
    "NON_PRIMARY_LISTING",
    "DELISTING_NOTICE",
    "PENNY_FLOOR",
    "NANO_FLOOR",
    "EARNINGS_QUALITY_DIVERGENCE",
    "GOING_CONCERN",
)
_BALANCE_SHEET_SCREEN_RULE_IDS = (
    "NON_PRIMARY_LISTING",
    "DELISTING_NOTICE",
    "PENNY_FLOOR",
    "NANO_FLOOR",
    "GOING_CONCERN",
)


def test_all_34_canonical_sectors_have_explicit_framework_contracts():
    assert CANONICAL_SECTOR_FRAMEWORK_CONTRACTS == {
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
    assert tuple(CANONICAL_SECTOR_FRAMEWORK_CONTRACTS) == ACTIVE_SECTOR_LABELS
    assert CANONICAL_SWEEP_SECTORS == ACTIVE_SECTOR_LABELS


def test_every_v2_framework_has_an_explicit_nonempty_screen_rule_registry():
    assert V2_FRAMEWORK_SCREEN_RULE_IDS == {
        "automotive": _COMMON_SCREEN_RULE_IDS,
        "capital_markets": _BALANCE_SHEET_SCREEN_RULE_IDS,
        "communications_media": _COMMON_SCREEN_RULE_IDS,
        "consumer": _COMMON_SCREEN_RULE_IDS,
        "education_services": _COMMON_SCREEN_RULE_IDS,
        "energy": _COMMON_SCREEN_RULE_IDS,
        "financial_services": _BALANCE_SHEET_SCREEN_RULE_IDS,
        "generic": _COMMON_SCREEN_RULE_IDS,
        "healthcare": _COMMON_SCREEN_RULE_IDS,
        "hospitality_gaming": _COMMON_SCREEN_RULE_IDS,
        "industrial": _COMMON_SCREEN_RULE_IDS,
        "materials": _COMMON_SCREEN_RULE_IDS,
        "medical_devices": _COMMON_SCREEN_RULE_IDS,
        "payments_fintech": _COMMON_SCREEN_RULE_IDS,
        "real_estate": _BALANCE_SHEET_SCREEN_RULE_IDS,
        "technology": _COMMON_SCREEN_RULE_IDS,
        "utilities": _COMMON_SCREEN_RULE_IDS,
    }
    assert all(V2_FRAMEWORK_SCREEN_RULE_IDS.values())


def test_all_34_v2_sectors_have_explicit_nonempty_screen_rule_mappings():
    assert V2_CANONICAL_SECTOR_SCREEN_RULE_IDS == {
        "aerospace_defense": _COMMON_SCREEN_RULE_IDS,
        "automotive": _COMMON_SCREEN_RULE_IDS,
        "biotech": _COMMON_SCREEN_RULE_IDS,
        "building_products": _COMMON_SCREEN_RULE_IDS,
        "business_services": _COMMON_SCREEN_RULE_IDS,
        "capital_markets": _BALANCE_SHEET_SCREEN_RULE_IDS,
        "chemicals": _COMMON_SCREEN_RULE_IDS,
        "construction_machinery": _COMMON_SCREEN_RULE_IDS,
        "construction_services": _COMMON_SCREEN_RULE_IDS,
        "consumer_discretionary": _COMMON_SCREEN_RULE_IDS,
        "consumer_services": _COMMON_SCREEN_RULE_IDS,
        "consumer_staples": _COMMON_SCREEN_RULE_IDS,
        "diversified_industrials": _COMMON_SCREEN_RULE_IDS,
        "education_services": _COMMON_SCREEN_RULE_IDS,
        "energy": _COMMON_SCREEN_RULE_IDS,
        "enterprise_software": _COMMON_SCREEN_RULE_IDS,
        "healthcare_pharma": _COMMON_SCREEN_RULE_IDS,
        "healthcare_services": _COMMON_SCREEN_RULE_IDS,
        "hospitality_gaming": _COMMON_SCREEN_RULE_IDS,
        "industrial_tech": _COMMON_SCREEN_RULE_IDS,
        "insurance": _BALANCE_SHEET_SCREEN_RULE_IDS,
        "internet_services": _COMMON_SCREEN_RULE_IDS,
        "large_cap_financials": _BALANCE_SHEET_SCREEN_RULE_IDS,
        "media_entertainment": _COMMON_SCREEN_RULE_IDS,
        "medical_devices": _COMMON_SCREEN_RULE_IDS,
        "metals_mining": _COMMON_SCREEN_RULE_IDS,
        "payments_fintech": _COMMON_SCREEN_RULE_IDS,
        "reits": _BALANCE_SHEET_SCREEN_RULE_IDS,
        "restaurants_food_service": _COMMON_SCREEN_RULE_IDS,
        "retail": _COMMON_SCREEN_RULE_IDS,
        "semiconductors": _COMMON_SCREEN_RULE_IDS,
        "telecom": _COMMON_SCREEN_RULE_IDS,
        "transportation_logistics": _COMMON_SCREEN_RULE_IDS,
        "utilities": _COMMON_SCREEN_RULE_IDS,
    }
    assert tuple(V2_CANONICAL_SECTOR_SCREEN_RULE_IDS) == ACTIVE_SECTOR_LABELS
    assert CANONICAL_SWEEP_SECTORS == ACTIVE_SECTOR_LABELS
    assert all(V2_CANONICAL_SECTOR_SCREEN_RULE_IDS.values())


def test_12_newly_validated_labels_have_explicit_v2_routes_without_fallback():
    expected_frameworks = {
        "automotive": "automotive",
        "building_products": "industrial",
        "business_services": "industrial",
        "chemicals": "materials",
        "construction_machinery": "industrial",
        "construction_services": "industrial",
        "consumer_discretionary": "consumer",
        "consumer_services": "consumer",
        "healthcare_services": "healthcare",
        "large_cap_financials": "financial_services",
        "restaurants_food_service": "consumer",
        "retail": "consumer",
    }
    expected_screen_rules = {
        "automotive": _COMMON_SCREEN_RULE_IDS,
        "building_products": _COMMON_SCREEN_RULE_IDS,
        "business_services": _COMMON_SCREEN_RULE_IDS,
        "chemicals": _COMMON_SCREEN_RULE_IDS,
        "construction_machinery": _COMMON_SCREEN_RULE_IDS,
        "construction_services": _COMMON_SCREEN_RULE_IDS,
        "consumer_discretionary": _COMMON_SCREEN_RULE_IDS,
        "consumer_services": _COMMON_SCREEN_RULE_IDS,
        "healthcare_services": _COMMON_SCREEN_RULE_IDS,
        "large_cap_financials": _BALANCE_SHEET_SCREEN_RULE_IDS,
        "restaurants_food_service": _COMMON_SCREEN_RULE_IDS,
        "retail": _COMMON_SCREEN_RULE_IDS,
    }

    assert {
        label: CANONICAL_SECTOR_FRAMEWORK_CONTRACTS[label]
        for label in expected_frameworks
    } == expected_frameworks
    assert {
        label: V2_CANONICAL_SECTOR_SCREEN_RULE_IDS[label]
        for label in expected_screen_rules
    } == expected_screen_rules
    assert {
        label: sector_framework_contract_id(label, pipeline_version="v2")
        for label in expected_frameworks
    } == expected_frameworks
    assert all(contract != "generic" for contract in expected_frameworks.values())


def test_every_active_v2_label_resolves_through_the_explicit_registry():
    assert CANONICAL_SWEEP_SECTORS == ACTIVE_SECTOR_LABELS
    assert tuple(CANONICAL_SECTOR_FRAMEWORK_CONTRACTS) == ACTIVE_SECTOR_LABELS
    assert tuple(V2_CANONICAL_SECTOR_SCREEN_RULE_IDS) == ACTIVE_SECTOR_LABELS
    assert {
        label: sector_framework_contract_id(label, pipeline_version="v2")
        for label in ACTIVE_SECTOR_LABELS
    } == CANONICAL_SECTOR_FRAMEWORK_CONTRACTS
    assert "generic" not in CANONICAL_SECTOR_FRAMEWORK_CONTRACTS.values()


def test_v2_screen_rules_are_sector_aware_and_exposed_in_framework_payload():
    insurance = sector_framework_screen_rule_ids("insurance", pipeline_version="v2")
    capital_markets = sector_framework_screen_rule_ids(
        "capital_markets", pipeline_version="v2"
    )
    reits = sector_framework_screen_rule_ids("reits", pipeline_version="v2")
    payments = sector_framework_screen_rule_ids(
        "payments_fintech", pipeline_version="v2"
    )

    assert insurance == list(_BALANCE_SHEET_SCREEN_RULE_IDS)
    assert capital_markets == list(_BALANCE_SHEET_SCREEN_RULE_IDS)
    assert reits == list(_BALANCE_SHEET_SCREEN_RULE_IDS)
    assert payments == list(_COMMON_SCREEN_RULE_IDS)
    assert "EARNINGS_QUALITY_DIVERGENCE" not in insurance
    assert "EARNINGS_QUALITY_DIVERGENCE" not in capital_markets
    assert "EARNINGS_QUALITY_DIVERGENCE" not in reits
    assert "EARNINGS_QUALITY_DIVERGENCE" in payments

    payload = sector_framework_template_payload(
        "reits", "large_and_mega", pipeline_version="v2"
    )
    assert payload["framework_contract_id"] == "real_estate"
    assert payload["applicable_screen_rule_ids"] == list(
        _BALANCE_SHEET_SCREEN_RULE_IDS
    )
    augmented = augment_sector_framework_payload(
        {"applicable_screen_rule_ids": ["PROVIDER_INVENTED_RULE"]},
        sector="enterprise_software",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
    )
    assert augmented["applicable_screen_rule_ids"] == list(_COMMON_SCREEN_RULE_IDS)


def test_v1_framework_payload_does_not_inherit_v2_screen_rule_contract():
    assert sector_framework_screen_rule_ids("insurance") == []
    assert "applicable_screen_rule_ids" not in sector_framework_template_payload(
        "insurance", "small_cap"
    )
    assert "applicable_screen_rule_ids" not in augment_sector_framework_payload(
        {},
        sector="insurance",
        market_cap_focus="small_cap",
    )


def test_new_canonical_contracts_do_not_fall_through_or_misroute():
    hospitality = sector_framework_template_payload(
        "hospitality_gaming", "large_and_mega", pipeline_version="v2"
    )
    capital_markets = sector_framework_template_payload(
        "capital_markets", "large_and_mega", pipeline_version="v2"
    )
    education = sector_framework_template_payload(
        "education_services", "large_and_mega", pipeline_version="v2"
    )
    healthcare = sector_framework_template_payload(
        "healthcare_pharma", "large_and_mega", pipeline_version="v2"
    )

    assert hospitality["framework_contract_id"] == "hospitality_gaming"
    assert hospitality["economic_model"] != healthcare["economic_model"]
    assert "lease_adjusted_owner_earnings" in hospitality["valid_valuation_methods"]
    assert "gaming_license_and_regulatory_evidence" in hospitality["required_evidence"]
    assert capital_markets["framework_contract_id"] == "capital_markets"
    assert "normalized_fee_earnings" in capital_markets["valid_valuation_methods"]
    assert "aum_and_net_flow_evidence" in capital_markets["required_evidence"]
    assert education["framework_contract_id"] == "education_services"
    assert "cohort_unit_economics" in education["valid_valuation_methods"]
    assert "student_outcome_evidence" in education["required_evidence"]
    assert sector_framework_contract_id(
        "unmapped specialist services", pipeline_version="v2"
    ) == "generic"


def test_v1_framework_routing_remains_unchanged_until_its_band_is_enabled():
    assert sector_framework_contract_id("hospitality_gaming") == "healthcare"
    assert sector_framework_contract_id("capital_markets") == "generic"
    assert sector_framework_contract_id("education_services") == "generic"


def test_financial_services_template_blocks_generic_enterprise_value_methods():
    payload = sector_framework_template_payload("insurance", "small_cap")

    assert payload["economic_model"] == (
        "Financial balance-sheet compounding driven by underwriting, credit, spread, capital adequacy, "
        "and tangible book value growth."
    )
    assert payload["selected_value_drivers"] == [
        "underwriting_margin",
        "reserve_or_credit_quality",
        "net_investment_or_interest_spread",
        "tangible_book_value_per_share_growth",
        "regulatory_or_rating_agency_capital",
        "capital_return_capacity",
    ]
    assert payload["selected_metrics"] == [
        "combined_ratio_or_efficiency_ratio",
        "loss_ratio_or_credit_loss_rate",
        "reserve_development",
        "net_investment_yield_or_net_interest_margin",
        "tangible_book_value_per_share_cagr",
        "normalized_roe",
        "capital_ratio_or_rbc",
    ]
    assert payload["valid_valuation_methods"] == [
        "tangible_book_value_compounding",
        "normalized_roe_to_book_value",
        "excess_capital_return",
        "sector_specific_insurance_anchor",
        "expected_return_scenarios",
    ]
    assert payload["invalid_valuation_methods"] == [
        "generic_dcf",
        "generic_epv",
        "revenue_multiple",
        "ebitda_multiple",
        "unadjusted_enterprise_value_multiple",
    ]
    assert payload["required_evidence"] == [
        "statutory_or_regulatory_capital_evidence",
        "reserve_or_credit_quality_evidence",
        "underwriting_or_spread_trend_evidence",
        "book_value_per_share_history",
        "security_type_and_common_equity_routing",
    ]


def test_payments_fintech_template_requires_volume_take_rate_losses_and_capital_evidence():
    payload = sector_framework_template_payload("payments_fintech", "small_cap")

    assert payload["economic_model"] == (
        "Payments and fintech network model driven by payment volume, transaction growth, take rate, "
        "merchant and consumer retention, fraud and credit losses, funding or float economics, regulatory "
        "compliance, operating leverage, and capital intensity."
    )
    assert payload["selected_value_drivers"] == [
        "payment_volume_and_transaction_growth",
        "take_rate_and_pricing_mix",
        "merchant_or_consumer_retention",
        "fraud_credit_and_chargeback_losses",
        "funding_cost_or_float_income",
        "network_acceptance_and_partner_concentration",
        "regulatory_and_compliance_resilience",
        "operating_leverage_and_capital_intensity",
    ]
    assert payload["selected_metrics"] == [
        "gross_payment_volume_growth",
        "transaction_growth",
        "net_revenue_take_rate",
        "merchant_retention_or_churn",
        "fraud_loss_rate",
        "credit_loss_rate",
        "funding_cost_or_net_interest_margin",
        "free_cash_flow_margin",
        "tangible_common_equity_or_capital_ratio",
    ]
    assert payload["valid_valuation_methods"] == [
        "payment_volume_take_rate_underwriting",
        "normalized_free_cash_flow",
        "unit_economics_dcf",
        "network_quality_expected_return",
        "expected_return_scenarios",
    ]
    assert payload["invalid_valuation_methods"] == [
        "unadjusted_revenue_multiple",
        "gross_payment_volume_multiple_without_take_rate",
        "ebitda_multiple_without_credit_or_fraud_losses",
        "generic_book_value_anchor_without_regulatory_capital_context",
        "growth_multiple_without_retention_or_compliance_evidence",
    ]
    assert payload["required_evidence"] == [
        "payment_volume_transaction_and_take_rate_evidence",
        "merchant_consumer_retention_evidence",
        "fraud_credit_loss_and_chargeback_evidence",
        "funding_float_or_regulatory_capital_evidence",
        "partner_concentration_and_compliance_evidence",
    ]
    assert sector_framework_template_payload("financial technology", "small_cap")["economic_model"] == payload["economic_model"]
    assert payload["hurdle_rate_policy"]["numeric_thresholds"] == [
        {"name": "base_return_hurdle", "value": "14", "units": "percent_annualized"},
    ]


def test_real_estate_template_uses_nav_and_affo_lens():
    payload = sector_framework_template_payload("real_estate_reit", "small_cap")

    assert payload["economic_model"] == (
        "Asset-heavy real-estate cash-flow model driven by same-property NOI, occupancy, lease duration, "
        "cap rates, leverage, and NAV/FFO compounding."
    )
    assert payload["selected_metrics"] == [
        "same_store_noi_growth",
        "occupancy_rate",
        "ffo_per_share",
        "affo_payout_ratio",
        "net_debt_to_ebitda",
        "weighted_average_lease_term",
        "implied_cap_rate",
    ]
    assert payload["valid_valuation_methods"] == [
        "nav_discount",
        "ffo_multiple",
        "affo_yield",
        "cap_rate_sensitivity",
        "expected_return_scenarios",
    ]
    assert payload["invalid_valuation_methods"] == [
        "generic_epv",
        "generic_book_value_anchor",
        "unadjusted_gaap_net_income_multiple",
        "revenue_multiple",
    ]


def test_healthcare_template_requires_reimbursement_and_regulatory_evidence():
    payload = sector_framework_template_payload("healthcare services", "small_cap")

    assert payload["economic_model"] == (
        "Healthcare value creation driven by reimbursement durability, utilization, clinical/regulatory risk, "
        "payer mix, margin discipline, and cash conversion."
    )
    assert payload["selected_value_drivers"] == [
        "reimbursement_durability",
        "volume_and_utilization_growth",
        "payer_mix",
        "regulatory_or_clinical_milestones",
        "gross_margin_durability",
        "free_cash_flow_conversion",
    ]
    assert payload["selected_metrics"] == [
        "revenue_cagr_5y",
        "gross_margin",
        "operating_margin",
        "payer_or_customer_concentration",
        "r_and_d_or_clinical_spend_intensity",
        "free_cash_flow_margin",
        "regulatory_milestone_status",
    ]
    assert payload["required_evidence"] == [
        "payer_mix_or_reimbursement_evidence",
        "clinical_or_regulatory_status_evidence",
        "margin_and_cash_conversion_evidence",
        "customer_or_product_concentration_evidence",
        "latest_current_event_or_filing_update",
    ]
    assert payload["invalid_valuation_methods"] == [
        "unvalidated_revenue_multiple",
        "generic_book_value_anchor",
        "asset_liquidation_anchor",
        "preclinical_pipeline_value_without_milestone_evidence",
    ]
    assert "technology_adjusted_dcf" not in payload["valid_valuation_methods"]


def test_technology_template_accepts_adjusted_dcf_anchor():
    payload = sector_framework_template_payload("semiconductors", "large_cap")

    assert payload["economic_model"] == (
        "Technology compounding model driven by retention, durable growth, gross margin, product "
        "reinvestment efficiency, free-cash-flow conversion, and dilution discipline."
    )
    assert payload["valid_valuation_methods"] == [
        "technology_adjusted_dcf",
        "cash_flow_dcf",
        "owner_earnings_multiple",
        "rule_of_40_contextual_check",
        "expected_return_scenarios",
    ]


def test_medical_devices_template_requires_procedure_installed_base_and_fda_evidence():
    payload = sector_framework_template_payload("medical_devices", "small_cap")

    assert payload["economic_model"] == (
        "Medical-device compounding model driven by procedure volumes, installed base, utilization, "
        "reimbursement durability, FDA and quality-system status, gross-margin durability, channel "
        "concentration, and innovation cadence."
    )
    assert payload["selected_value_drivers"] == [
        "procedure_volume_and_utilization",
        "installed_base_and_consumables_pull_through",
        "reimbursement_and_site_of_care_durability",
        "fda_quality_system_and_recall_status",
        "gross_margin_and_manufacturing_scale",
        "hospital_capex_or_purchasing_cycle",
        "sales_channel_and_customer_concentration",
        "product_pipeline_and_replacement_cycle",
    ]
    assert payload["selected_metrics"] == [
        "procedure_volume_growth",
        "installed_base_growth",
        "consumables_or_recurring_revenue_mix",
        "gross_margin",
        "r_and_d_intensity",
        "sales_and_marketing_intensity",
        "free_cash_flow_margin",
        "customer_or_channel_concentration",
        "regulatory_or_recall_event_status",
    ]
    assert payload["valid_valuation_methods"] == [
        "procedure_volume_installed_base_underwriting",
        "normalized_free_cash_flow",
        "unit_economics_dcf",
        "pipeline_or_product_cycle_risk_adjusted_sotp",
        "expected_return_scenarios",
    ]
    assert payload["invalid_valuation_methods"] == [
        "unvalidated_revenue_multiple",
        "generic_book_value_anchor",
        "asset_liquidation_anchor",
        "pipeline_value_without_regulatory_or_adoption_evidence",
        "gross_margin_multiple_without_quality_or_recall_evidence",
    ]
    assert payload["required_evidence"] == [
        "procedure_volume_utilization_or_installed_base_evidence",
        "reimbursement_site_of_care_or_payer_evidence",
        "fda_clearance_quality_system_or_recall_evidence",
        "gross_margin_manufacturing_and_cash_conversion_evidence",
        "channel_customer_concentration_or_hospital_capex_evidence",
    ]
    assert sector_framework_template_payload("medtech", "small_cap")["economic_model"] == payload["economic_model"]
    assert payload["hurdle_rate_policy"]["numeric_thresholds"] == [
        {"name": "base_return_hurdle", "value": "15", "units": "percent_annualized"},
    ]


def test_consumer_template_requires_comps_inventory_leases_and_unit_economics():
    payload = sector_framework_template_payload("restaurants_food_service", "small_cap")

    assert payload["economic_model"] == (
        "Consumer and retail unit-economics model driven by comparable sales, traffic and ticket, gross margin, "
        "inventory turns, lease obligations, brand durability, and store or channel expansion returns."
    )
    assert payload["selected_value_drivers"] == [
        "same_store_sales_or_comparable_growth",
        "traffic_and_ticket_mix",
        "gross_margin_and_merchandise_margin",
        "inventory_turns_and_markdown_risk",
        "store_or_channel_unit_economics",
        "lease_adjusted_cash_flow",
        "brand_loyalty_and_customer_retention",
    ]
    assert payload["selected_metrics"] == [
        "comparable_sales_growth",
        "traffic_growth",
        "average_ticket_growth",
        "gross_margin",
        "inventory_turnover",
        "lease_adjusted_net_debt_to_ebitdar",
        "free_cash_flow_margin",
        "store_count_or_channel_growth",
    ]
    assert payload["valid_valuation_methods"] == [
        "lease_adjusted_owner_earnings",
        "normalized_free_cash_flow",
        "unit_economics_dcf",
        "roic_reinvestment_underwriting",
        "expected_return_scenarios",
    ]
    assert payload["invalid_valuation_methods"] == [
        "unadjusted_revenue_multiple",
        "generic_book_value_anchor",
        "ebitda_multiple_without_lease_adjustment",
        "peak_margin_multiple_without_markdown_cycle",
        "store_growth_multiple_without_unit_economics",
    ]
    assert payload["required_evidence"] == [
        "same_store_sales_or_traffic_evidence",
        "gross_margin_and_input_cost_evidence",
        "inventory_turnover_and_markdown_evidence",
        "lease_obligation_and_store_base_evidence",
        "unit_economics_or_channel_profitability_evidence",
    ]
    assert payload["hurdle_rate_policy"]["numeric_thresholds"] == [
        {"name": "base_return_hurdle", "value": "13", "units": "percent_annualized"},
    ]


def test_automotive_template_requires_unit_pricing_warranty_and_residual_evidence():
    payload = sector_framework_template_payload("automotive", "small_cap")

    assert payload["economic_model"] == (
        "Automotive and mobility cycle model driven by unit volumes, pricing and incentive discipline, "
        "platform and powertrain transitions, supplier or dealer channel health, warranty quality, capital "
        "intensity, inventory, and residual-value risk."
    )
    assert payload["selected_value_drivers"] == [
        "unit_volume_and_mix",
        "pricing_and_incentive_discipline",
        "platform_or_powertrain_transition",
        "supplier_or_dealer_channel_health",
        "warranty_quality_and_recall_risk",
        "inventory_and_working_capital_cycle",
        "capital_intensity_and_tooling_reinvestment",
        "finance_or_residual_value_exposure",
    ]
    assert payload["selected_metrics"] == [
        "unit_sales_growth",
        "average_selling_price",
        "incentive_spend_ratio",
        "gross_margin",
        "warranty_expense_ratio",
        "inventory_days",
        "capital_expenditures_to_sales",
        "free_cash_flow_through_cycle",
        "net_debt_to_ebitda",
    ]
    assert payload["valid_valuation_methods"] == [
        "cycle_normalized_owner_earnings",
        "mid_cycle_earnings_power",
        "sum_of_parts_for_finance_or_parts_segments",
        "roic_reinvestment_underwriting",
        "expected_return_scenarios",
    ]
    assert payload["invalid_valuation_methods"] == [
        "unadjusted_revenue_multiple",
        "unadjusted_peak_earnings_multiple",
        "ebitda_multiple_without_tooling_and_warranty_costs",
        "unit_growth_multiple_without_pricing_or_inventory_evidence",
        "book_value_anchor_without_finance_residual_risk",
    ]
    assert payload["required_evidence"] == [
        "unit_volume_mix_and_pricing_evidence",
        "incentive_inventory_and_channel_evidence",
        "platform_powertrain_or_capex_plan_evidence",
        "warranty_recall_or_quality_evidence",
        "finance_residual_value_or_leverage_evidence",
    ]
    assert sector_framework_template_payload("auto parts", "small_cap")["economic_model"] == payload["economic_model"]
    assert payload["hurdle_rate_policy"]["numeric_thresholds"] == [
        {"name": "base_return_hurdle", "value": "15", "units": "percent_annualized"},
    ]


def test_communications_media_template_requires_subscriber_capex_and_leverage_evidence():
    payload = sector_framework_template_payload("telecom", "small_cap")

    assert payload["economic_model"] == (
        "Communications and media model driven by subscriber or audience growth, ARPU and pricing, "
        "churn or engagement retention, network or content investment, advertising cyclicality, leverage, "
        "and free-cash-flow conversion."
    )
    assert payload["selected_value_drivers"] == [
        "subscriber_or_audience_growth",
        "arpu_and_pricing",
        "churn_or_engagement_retention",
        "network_capex_or_content_investment",
        "advertising_and_affiliate_revenue_mix",
        "spectrum_or_distribution_rights",
        "leverage_and_refinancing_capacity",
        "free_cash_flow_conversion",
    ]
    assert payload["selected_metrics"] == [
        "subscriber_growth",
        "arpu_growth",
        "churn_rate",
        "broadband_or_wireless_net_adds",
        "advertising_revenue_growth",
        "content_or_programming_cost_ratio",
        "capital_intensity",
        "net_debt_to_ebitda",
        "free_cash_flow_margin",
    ]
    assert payload["valid_valuation_methods"] == [
        "subscriber_ltv_to_cac",
        "normalized_free_cash_flow",
        "dcf_with_capex_or_content_cycle",
        "network_or_content_asset_sum_of_parts",
        "expected_return_scenarios",
    ]
    assert payload["invalid_valuation_methods"] == [
        "unadjusted_revenue_multiple",
        "ebitda_multiple_without_capex_or_content_costs",
        "subscriber_multiple_without_churn_or_arpu",
        "advertising_peak_multiple_without_cycle_normalization",
        "book_value_anchor_without_spectrum_or_content_context",
    ]
    assert payload["required_evidence"] == [
        "subscriber_arpu_and_churn_evidence",
        "network_capex_or_content_spend_evidence",
        "advertising_or_affiliate_mix_evidence",
        "spectrum_distribution_or_rights_evidence",
        "leverage_and_refinancing_evidence",
    ]
    assert sector_framework_template_payload("media_entertainment", "small_cap")["economic_model"] == payload["economic_model"]
    assert "Technology compounding model" in sector_framework_template_payload("internet_services", "small_cap")["economic_model"]


def test_energy_template_requires_reserves_hedges_and_decline_evidence():
    payload = sector_framework_template_payload("energy", "small_cap")

    assert payload["economic_model"] == (
        "Energy asset and cash-flow model driven by reserve quality, production decline, realized commodity "
        "pricing, hedge coverage, reinvestment intensity, and cycle-normalized free cash flow."
    )
    assert payload["selected_value_drivers"] == [
        "reserve_quality_and_life",
        "production_decline_and_replacement",
        "realized_price_vs_benchmark",
        "hedge_coverage_and_rolloff",
        "lifting_cost_and_margin",
        "maintenance_capex_and_reinvestment",
        "balance_sheet_and_decommissioning_liabilities",
    ]
    assert payload["selected_metrics"] == [
        "production_growth_or_decline",
        "reserve_life_index",
        "finding_and_development_cost",
        "lifting_cost_per_unit",
        "realized_price_vs_benchmark",
        "hedge_coverage",
        "free_cash_flow_after_maintenance_capex",
        "net_debt_to_ebitda",
    ]
    assert payload["valid_valuation_methods"] == [
        "proved_reserve_nav",
        "cycle_normalized_free_cash_flow",
        "commodity_sensitivity_nav",
        "recycle_ratio_underwriting",
        "expected_return_scenarios",
    ]
    assert payload["invalid_valuation_methods"] == [
        "unadjusted_revenue_multiple",
        "spot_price_extrapolation_without_sensitivity",
        "generic_book_value_anchor",
        "ebitda_multiple_without_maintenance_capex",
        "dcf_without_commodity_sensitivity",
    ]
    assert payload["required_evidence"] == [
        "reserve_report_or_production_evidence",
        "commodity_price_and_hedge_evidence",
        "lifting_cost_and_margin_evidence",
        "maintenance_capex_and_decline_rate_evidence",
        "decommissioning_or_environmental_liability_evidence",
    ]
    assert payload["hurdle_rate_policy"]["numeric_thresholds"] == [
        {"name": "base_return_hurdle", "value": "16", "units": "percent_annualized"},
    ]


def test_materials_template_requires_spreads_cost_curve_and_reclamation_evidence():
    payload = sector_framework_template_payload("metals_mining", "small_cap")

    assert payload["economic_model"] == (
        "Materials cycle model driven by commodity and feedstock spreads, volume mix, cost-curve position, "
        "capacity utilization, sustaining capital, environmental liabilities, and balance-sheet resilience."
    )
    assert payload["selected_value_drivers"] == [
        "commodity_or_feedstock_spread",
        "volume_and_mix",
        "cost_curve_position",
        "capacity_utilization",
        "sustaining_capex_intensity",
        "working_capital_cycle",
        "environmental_and_reclamation_liabilities",
        "balance_sheet_resilience",
    ]
    assert payload["selected_metrics"] == [
        "realized_price_vs_benchmark",
        "feedstock_cost_spread",
        "production_volume_growth",
        "capacity_utilization",
        "cash_cost_per_unit",
        "sustaining_capex_to_sales",
        "free_cash_flow_through_cycle",
        "net_debt_to_ebitda",
    ]
    assert payload["valid_valuation_methods"] == [
        "cycle_normalized_free_cash_flow",
        "mid_cycle_earnings_power",
        "commodity_sensitivity_nav",
        "replacement_cost_with_cycle_check",
        "expected_return_scenarios",
    ]
    assert payload["invalid_valuation_methods"] == [
        "unadjusted_revenue_multiple",
        "spot_price_extrapolation_without_sensitivity",
        "unadjusted_peak_earnings_multiple",
        "ebitda_multiple_without_sustaining_capex",
        "book_value_anchor_without_impairment_check",
    ]
    assert payload["required_evidence"] == [
        "commodity_price_or_feedstock_spread_evidence",
        "volume_and_capacity_utilization_evidence",
        "cost_curve_or_cash_cost_evidence",
        "sustaining_capex_and_working_capital_evidence",
        "environmental_reclamation_or_regulatory_liability_evidence",
    ]
    assert sector_framework_template_payload("chemicals", "small_cap")["economic_model"] == payload["economic_model"]
    assert payload["hurdle_rate_policy"]["numeric_thresholds"] == [
        {"name": "base_return_hurdle", "value": "15", "units": "percent_annualized"},
    ]


def test_utilities_template_requires_rate_case_rate_base_and_credit_evidence():
    payload = sector_framework_template_payload("utilities", "small_cap")

    assert payload["economic_model"] == (
        "Regulated utility compounding model driven by rate-base growth, allowed ROE, regulatory construct "
        "quality, capital-plan execution, customer affordability, leverage, and dividend sustainability."
    )
    assert payload["selected_value_drivers"] == [
        "rate_base_growth",
        "allowed_roe_and_equity_ratio",
        "regulatory_recovery_mechanisms",
        "capital_plan_execution",
        "customer_affordability_and_load_growth",
        "debt_funding_capacity",
        "dividend_coverage",
    ]
    assert payload["selected_metrics"] == [
        "rate_base_cagr",
        "allowed_roe",
        "equity_ratio",
        "regulated_capex_plan",
        "funds_from_operations_to_debt",
        "debt_to_capital",
        "dividend_payout_ratio",
        "customer_bill_growth",
    ]
    assert payload["valid_valuation_methods"] == [
        "rate_base_compounding",
        "allowed_roe_to_book_value",
        "dividend_discount_model",
        "regulated_utility_expected_return",
        "expected_return_scenarios",
    ]
    assert payload["invalid_valuation_methods"] == [
        "unadjusted_revenue_multiple",
        "generic_dcf_without_rate_case_support",
        "ebitda_multiple_without_capex_and_debt_funding",
        "asset_liquidation_anchor",
        "peak_earnings_multiple_without_regulatory_normalization",
    ]
    assert payload["required_evidence"] == [
        "rate_case_or_regulatory_order_evidence",
        "rate_base_and_capex_plan_evidence",
        "allowed_roe_and_equity_ratio_evidence",
        "debt_funding_and_credit_metric_evidence",
        "dividend_coverage_and_affordability_evidence",
    ]
    assert payload["hurdle_rate_policy"]["numeric_thresholds"] == [
        {"name": "base_return_hurdle", "value": "10", "units": "percent_annualized"},
    ]


def test_documented_sector_taxonomy_avoids_generic_framework_fallbacks():
    generic_model = "Financially underwritten 5-10 year per-share return analysis."
    categories = [
        "aerospace_defense",
        "automotive",
        "biotech",
        "chemicals",
        "construction_machinery",
        "consumer_staples",
        "diversified_industrials",
        "energy",
        "enterprise_software",
        "healthcare_pharma",
        "industrial_tech",
        "insurance",
        "internet_services",
        "large_cap_financials",
        "media_entertainment",
        "medical_devices",
        "metals_mining",
        "payments_fintech",
        "reits",
        "restaurants_food_service",
        "retail",
        "semiconductors",
        "telecom",
        "transportation_logistics",
        "utilities",
    ]

    generic_categories = [
        category
        for category in categories
        if sector_framework_template_payload(category, "small_cap")["economic_model"] == generic_model
    ]

    assert generic_categories == []


def test_augment_framework_replaces_generic_model_and_adds_sector_metrics():
    payload = augment_sector_framework_payload(
        {
            "sector": "software",
            "market_cap_focus": "small_cap",
            "horizon_years": [5, 10],
            "economic_model": "Financially underwritten 5-10 year per-share return analysis.",
            "selected_value_drivers": ["per_share_return"],
            "selected_metrics": ["base_case_annualized_return"],
            "valid_valuation_methods": ["expected_return_scenarios"],
            "invalid_valuation_methods": ["unsupported_narrative_multiple"],
            "required_evidence": ["expected_return_evidence"],
            "normalization_policy": {
                "approach": "Provider policy.",
                "rules": ["Preserve provider rule."],
                "numeric_thresholds": [{"name": "provider_threshold", "value": "7", "units": "percent"}],
            },
            "hurdle_rate_policy": {
                "approach": "Provider hurdle.",
                "rules": [],
                "numeric_thresholds": [],
            },
            "weighting_policy": {
                "approach": "Provider weights.",
                "rules": [],
                "numeric_thresholds": [],
            },
            "sector_specific_risks": ["provider_risk"],
        },
        sector="software",
        market_cap_focus="small_cap",
    )

    assert payload["economic_model"] == (
        "Technology compounding model driven by retention, durable growth, gross margin, product "
        "reinvestment efficiency, free-cash-flow conversion, and dilution discipline."
    )
    assert payload["selected_value_drivers"][:3] == [
        "per_share_return",
        "net_revenue_retention_or_churn",
        "durable_organic_revenue_growth",
    ]
    assert payload["selected_metrics"] == [
        "base_case_annualized_return",
        "revenue_cagr_5y",
        "gross_margin",
        "operating_margin",
        "free_cash_flow_margin",
        "sbc_percent_revenue",
        "share_count_cagr",
        "net_revenue_retention",
    ]
    assert payload["invalid_valuation_methods"] == [
        "unsupported_narrative_multiple",
        "generic_book_value_anchor",
        "asset_liquidation_anchor",
        "revenue_multiple_without_profitability_path",
        "ebitda_multiple_excluding_sbc_without_dilution_cost",
    ]
    assert payload["normalization_policy"]["approach"] == "Provider policy."
    assert payload["normalization_policy"]["rules"][:2] == [
        "Preserve provider rule.",
        "Do not force a winner when evidence quality or model fit is insufficient.",
    ]
    assert payload["normalization_policy"]["numeric_thresholds"] == [
        {"name": "provider_threshold", "value": "7", "units": "percent"},
        {"name": "base_return_hurdle", "value": "12", "units": "percent_annualized"},
    ]
    assert payload["sector_specific_risks"][:2] == ["provider_risk", "growth_deceleration"]


def test_augment_framework_preserves_specific_provider_model():
    payload = augment_sector_framework_payload(
        {
            "economic_model": "Provider selected aerospace aftermarket cycle model.",
            "selected_value_drivers": [],
            "selected_metrics": [],
            "valid_valuation_methods": [],
            "invalid_valuation_methods": [],
            "required_evidence": [],
            "normalization_policy": {},
            "hurdle_rate_policy": {},
            "weighting_policy": {},
            "sector_specific_risks": [],
        },
        sector="aerospace manufacturing",
        market_cap_focus="small_cap",
    )

    assert payload["economic_model"] == "Provider selected aerospace aftermarket cycle model."
    assert payload["selected_value_drivers"] == [
        "organic_order_and_backlog_conversion",
        "pricing_power_vs_input_costs",
        "gross_margin_through_cycle",
        "operating_leverage",
        "maintenance_capex_intensity",
        "working_capital_turns",
        "per_share_capital_allocation",
    ]
    assert "cycle_normalized_owner_earnings" in payload["valid_valuation_methods"]
