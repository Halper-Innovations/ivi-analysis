from app.sector.canonical_taxonomy import (
    ACTIVE_SECTOR_LABELS,
    CANONICAL_SECTOR_CONTRACTS,
    CANONICAL_SECTOR_TAXONOMY_HASH,
    CANONICAL_SECTOR_TAXONOMY_VERSION,
    SCREEN_PROFILE_RULE_IDS,
    canonical_sector_taxonomy_manifest,
    normalize_sector_label,
    resolve_canonical_sector,
    validate_canonical_sector_taxonomy,
)


def test_active_taxonomy_is_the_versioned_34_label_registry():
    assert CANONICAL_SECTOR_TAXONOMY_VERSION == "us_equity_sector_taxonomy.v1"
    assert CANONICAL_SECTOR_TAXONOMY_HASH == (
        "04f8cc1a1daafc8f468cb37753dfee98d11ea8aecccf5144eacffa7d6b1e1241"
    )
    assert ACTIVE_SECTOR_LABELS == (
        "aerospace_defense",
        "automotive",
        "biotech",
        "building_products",
        "business_services",
        "capital_markets",
        "chemicals",
        "construction_machinery",
        "construction_services",
        "consumer_discretionary",
        "consumer_services",
        "consumer_staples",
        "diversified_industrials",
        "education_services",
        "energy",
        "enterprise_software",
        "healthcare_pharma",
        "healthcare_services",
        "hospitality_gaming",
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
    )


def test_every_active_label_has_an_explicit_framework_and_screen_profile():
    assert {
        label: (
            contract.canonical_sector_id,
            contract.framework_contract_id,
            contract.screen_profile_id,
        )
        for label, contract in CANONICAL_SECTOR_CONTRACTS.items()
    } == {
        "aerospace_defense": (
            "aerospace_defense",
            "industrial",
            "deterministic_standard_v1",
        ),
        "automotive": ("automotive", "automotive", "deterministic_standard_v1"),
        "biotech": ("biotech", "healthcare", "deterministic_standard_v1"),
        "building_products": (
            "building_products",
            "industrial",
            "deterministic_standard_v1",
        ),
        "business_services": (
            "business_services",
            "industrial",
            "deterministic_standard_v1",
        ),
        "capital_markets": (
            "capital_markets",
            "capital_markets",
            "deterministic_balance_sheet_v1",
        ),
        "chemicals": ("chemicals", "materials", "deterministic_standard_v1"),
        "construction_machinery": (
            "construction_machinery",
            "industrial",
            "deterministic_standard_v1",
        ),
        "construction_services": (
            "construction_services",
            "industrial",
            "deterministic_standard_v1",
        ),
        "consumer_discretionary": (
            "consumer_discretionary",
            "consumer",
            "deterministic_standard_v1",
        ),
        "consumer_services": (
            "consumer_services",
            "consumer",
            "deterministic_standard_v1",
        ),
        "consumer_staples": (
            "consumer_staples",
            "consumer",
            "deterministic_standard_v1",
        ),
        "diversified_industrials": (
            "diversified_industrials",
            "industrial",
            "deterministic_standard_v1",
        ),
        "education_services": (
            "education_services",
            "education_services",
            "deterministic_standard_v1",
        ),
        "energy": ("energy", "energy", "deterministic_standard_v1"),
        "enterprise_software": (
            "enterprise_software",
            "technology",
            "deterministic_standard_v1",
        ),
        "healthcare_pharma": (
            "healthcare_pharma",
            "healthcare",
            "deterministic_standard_v1",
        ),
        "healthcare_services": (
            "healthcare_services",
            "healthcare",
            "deterministic_standard_v1",
        ),
        "hospitality_gaming": (
            "hospitality_gaming",
            "hospitality_gaming",
            "deterministic_standard_v1",
        ),
        "industrial_tech": (
            "industrial_tech",
            "industrial",
            "deterministic_standard_v1",
        ),
        "insurance": (
            "insurance",
            "financial_services",
            "deterministic_balance_sheet_v1",
        ),
        "internet_services": (
            "internet_services",
            "technology",
            "deterministic_standard_v1",
        ),
        "large_cap_financials": (
            "banking",
            "financial_services",
            "deterministic_balance_sheet_v1",
        ),
        "media_entertainment": (
            "media_entertainment",
            "communications_media",
            "deterministic_standard_v1",
        ),
        "medical_devices": (
            "medical_devices",
            "medical_devices",
            "deterministic_standard_v1",
        ),
        "metals_mining": (
            "metals_mining",
            "materials",
            "deterministic_standard_v1",
        ),
        "payments_fintech": (
            "payments_fintech",
            "payments_fintech",
            "deterministic_standard_v1",
        ),
        "reits": ("reits", "real_estate", "deterministic_balance_sheet_v1"),
        "restaurants_food_service": (
            "restaurants_food_service",
            "consumer",
            "deterministic_standard_v1",
        ),
        "retail": ("retail", "consumer", "deterministic_standard_v1"),
        "semiconductors": (
            "semiconductors",
            "technology",
            "deterministic_standard_v1",
        ),
        "telecom": ("telecom", "communications_media", "deterministic_standard_v1"),
        "transportation_logistics": (
            "transportation_logistics",
            "industrial",
            "deterministic_standard_v1",
        ),
        "utilities": ("utilities", "utilities", "deterministic_standard_v1"),
    }


def test_screen_profiles_have_explicit_exact_rule_contracts():
    assert dict(SCREEN_PROFILE_RULE_IDS) == {
        "deterministic_standard_v1": (
            "NON_PRIMARY_LISTING",
            "DELISTING_NOTICE",
            "PENNY_FLOOR",
            "NANO_FLOOR",
            "EARNINGS_QUALITY_DIVERGENCE",
            "GOING_CONCERN",
        ),
        "deterministic_balance_sheet_v1": (
            "NON_PRIMARY_LISTING",
            "DELISTING_NOTICE",
            "PENNY_FLOOR",
            "NANO_FLOOR",
            "GOING_CONCERN",
        ),
    }


def test_lookup_is_exact_after_formatting_normalization_only():
    assert normalize_sector_label("  Large-Cap   Financials ") == "large_cap_financials"
    assert normalize_sector_label("CONSUMER__STAPLES") == "consumer_staples"

    legacy = resolve_canonical_sector("  Large-Cap   Financials ")
    assert legacy.disposition == "RESOLVED"
    assert legacy.reason_code == "EXACT_ACTIVE_TAXONOMY_MATCH"
    assert legacy.contract is not None
    assert legacy.contract.source_label == "large_cap_financials"
    assert legacy.contract.canonical_sector_id == "banking"
    assert legacy.contract.framework_contract_id == "financial_services"
    assert legacy.contract.screen_profile_id == "deterministic_balance_sheet_v1"

    assert resolve_canonical_sector("banking").disposition == "NEEDS_DATA"
    assert resolve_canonical_sector("fintech platform").disposition == "NEEDS_DATA"
    assert resolve_canonical_sector("retail services").disposition == "NEEDS_DATA"


def test_unknown_label_has_needs_data_disposition_without_generic_fallback():
    resolution = resolve_canonical_sector("Unmapped Growth Sector")

    assert resolution.to_dict() == {
        "input_label": "Unmapped Growth Sector",
        "normalized_label": "unmapped_growth_sector",
        "disposition": "NEEDS_DATA",
        "reason_code": "SECTOR_CLASSIFICATION_UNRESOLVED",
        "taxonomy_version": "us_equity_sector_taxonomy.v1",
        "taxonomy_hash": "04f8cc1a1daafc8f468cb37753dfee98d11ea8aecccf5144eacffa7d6b1e1241",
        "source_label": None,
        "canonical_sector_id": None,
        "framework_contract_id": None,
        "screen_profile_id": None,
    }


def test_manifest_and_registry_invariants_are_deterministic():
    validate_canonical_sector_taxonomy()
    manifest = canonical_sector_taxonomy_manifest()

    assert manifest["taxonomy_version"] == "us_equity_sector_taxonomy.v1"
    assert manifest["taxonomy_hash"] == (
        "04f8cc1a1daafc8f468cb37753dfee98d11ea8aecccf5144eacffa7d6b1e1241"
    )
    assert manifest["active_label_count"] == 34
    assert len(manifest["contracts"]) == 34
    assert manifest["contracts"][22] == {
        "source_label": "large_cap_financials",
        "canonical_sector_id": "banking",
        "framework_contract_id": "financial_services",
        "screen_profile_id": "deterministic_balance_sheet_v1",
    }
