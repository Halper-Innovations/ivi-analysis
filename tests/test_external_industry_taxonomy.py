from collections import Counter

from app.sector.canonical_taxonomy import ACTIVE_SECTOR_LABELS
from app.sector.external_industry_taxonomy import (
    EXACT_EXTERNAL_INDUSTRY_MATCH,
    EXTERNAL_INDUSTRY_UNRESOLVED,
    STOCKANALYSIS_BOUNDARY_CAP_FLOOR_USD,
    STOCKANALYSIS_BOUNDARY_INDUSTRY_INVENTORY,
    STOCKANALYSIS_BOUNDARY_ROW_COUNT,
    STOCKANALYSIS_INDUSTRY_INVENTORY,
    STOCKANALYSIS_INDUSTRY_TAXONOMY_HASH,
    STOCKANALYSIS_INDUSTRY_TAXONOMY_VERSION,
    STOCKANALYSIS_INDUSTRY_TO_SECTOR,
    STOCKANALYSIS_SNAPSHOT_AS_OF,
    STOCKANALYSIS_SNAPSHOT_NON_NULL_INDUSTRY_COUNT,
    STOCKANALYSIS_SNAPSHOT_ROW_COUNT,
    STOCKANALYSIS_SNAPSHOT_SOURCE_URL,
    STOCKANALYSIS_SNAPSHOT_UNIQUE_INDUSTRY_COUNT,
    STOCKANALYSIS_UNRESOLVED_INDUSTRIES,
    external_industry_taxonomy_manifest,
    resolve_stockanalysis_industry,
    validate_external_industry_taxonomy,
)


def test_stockanalysis_inventory_has_one_explicit_disposition_per_raw_label():
    assert STOCKANALYSIS_INDUSTRY_TAXONOMY_VERSION == ("stockanalysis_industry_to_ivi_sector.v1")
    assert STOCKANALYSIS_INDUSTRY_TAXONOMY_HASH == (
        "cccffea84695cc998b5649e2f203487da509b421129795b0e88e73201d17fbaa"
    )
    assert STOCKANALYSIS_SNAPSHOT_AS_OF == "2026-07-17"
    assert STOCKANALYSIS_SNAPSHOT_SOURCE_URL == ("https://stockanalysis.com/stocks/screener/")
    assert STOCKANALYSIS_SNAPSHOT_ROW_COUNT == 5_598
    assert STOCKANALYSIS_SNAPSHOT_NON_NULL_INDUSTRY_COUNT == 5_596
    assert STOCKANALYSIS_SNAPSHOT_UNIQUE_INDUSTRY_COUNT == 152
    assert len(STOCKANALYSIS_INDUSTRY_INVENTORY) == 152
    assert len(STOCKANALYSIS_INDUSTRY_TO_SECTOR) == 148
    assert STOCKANALYSIS_UNRESOLVED_INDUSTRIES == {
        "Asset Management - Cryptocurrency": (
            "PROVIDER_LABEL_IS_NON_OPERATING_CRYPTO_INVESTMENT_VEHICLE_IN_SNAPSHOT"
        ),
        "Asset Management - Income": (
            "PROVIDER_LABEL_IS_NON_OPERATING_REGISTERED_FUND_IN_SNAPSHOT"
        ),
        "Other": "PROVIDER_OTHER_LABEL_HAS_NO_AUDITABLE_SECTOR_MEANING",
        "Shell Companies": ("NON_OPERATING_SHELL_REQUIRES_SECURITY_DISPOSITION_BEFORE_SECTOR"),
    }
    assert set(STOCKANALYSIS_INDUSTRY_TO_SECTOR).isdisjoint(STOCKANALYSIS_UNRESOLVED_INDUSTRIES)
    assert set(STOCKANALYSIS_INDUSTRY_TO_SECTOR) | set(STOCKANALYSIS_UNRESOLVED_INDUSTRIES) == set(
        STOCKANALYSIS_INDUSTRY_INVENTORY
    )


def test_every_route_targets_the_active_34_label_registry_without_fallback():
    assert set(STOCKANALYSIS_INDUSTRY_TO_SECTOR.values()) == set(ACTIVE_SECTOR_LABELS)
    assert Counter(STOCKANALYSIS_INDUSTRY_TO_SECTOR.values()) == {
        "aerospace_defense": 1,
        "automotive": 4,
        "biotech": 1,
        "building_products": 3,
        "business_services": 7,
        "capital_markets": 6,
        "chemicals": 3,
        "construction_machinery": 1,
        "construction_services": 2,
        "consumer_discretionary": 6,
        "consumer_services": 1,
        "consumer_staples": 9,
        "diversified_industrials": 5,
        "education_services": 1,
        "energy": 9,
        "enterprise_software": 4,
        "healthcare_pharma": 3,
        "healthcare_services": 2,
        "hospitality_gaming": 4,
        "industrial_tech": 9,
        "insurance": 7,
        "internet_services": 1,
        "large_cap_financials": 2,
        "media_entertainment": 6,
        "medical_devices": 5,
        "metals_mining": 9,
        "payments_fintech": 1,
        "reits": 12,
        "restaurants_food_service": 2,
        "retail": 7,
        "semiconductors": 2,
        "telecom": 1,
        "transportation_logistics": 6,
        "utilities": 6,
    }
    assert "generic" not in STOCKANALYSIS_INDUSTRY_TO_SECTOR.values()
    assert "other" not in STOCKANALYSIS_INDUSTRY_TO_SECTOR.values()
    assert "unknown" not in STOCKANALYSIS_INDUSTRY_TO_SECTOR.values()
    validate_external_industry_taxonomy()


def test_every_snapshot_industry_at_large_cap_boundary_resolves():
    assert STOCKANALYSIS_BOUNDARY_CAP_FLOOR_USD == 9_500_000_000
    assert STOCKANALYSIS_BOUNDARY_ROW_COUNT == 981
    assert len(STOCKANALYSIS_BOUNDARY_INDUSTRY_INVENTORY) == 128
    assert set(STOCKANALYSIS_BOUNDARY_INDUSTRY_INVENTORY) <= set(STOCKANALYSIS_INDUSTRY_TO_SECTOR)
    assert set(STOCKANALYSIS_BOUNDARY_INDUSTRY_INVENTORY).isdisjoint(
        STOCKANALYSIS_UNRESOLVED_INDUSTRIES
    )


def test_lookup_is_raw_exact_and_preserves_external_industry():
    resolved = resolve_stockanalysis_industry("Banks - Regional")
    assert resolved.raw_industry == "Banks - Regional"
    assert resolved.disposition == "RESOLVED"
    assert resolved.reason_code == EXACT_EXTERNAL_INDUSTRY_MATCH
    assert resolved.reason_detail == "exact raw StockAnalysis industry registry match"
    assert resolved.source_sector_label == "large_cap_financials"
    assert resolved.taxonomy_version == STOCKANALYSIS_INDUSTRY_TAXONOMY_VERSION
    assert resolved.taxonomy_hash == STOCKANALYSIS_INDUSTRY_TAXONOMY_HASH

    # Formatting variants are deliberately not aliases.
    variant = resolve_stockanalysis_industry(" banks - regional ")
    assert variant.raw_industry == " banks - regional "
    assert variant.disposition == "NEEDS_DATA"
    assert variant.reason_code == EXTERNAL_INDUSTRY_UNRESOLVED
    assert variant.reason_detail == "INDUSTRY_NOT_IN_VERSIONED_EXTERNAL_INVENTORY"
    assert variant.source_sector_label is None


def test_provider_ambiguous_unknown_and_missing_labels_never_fall_back():
    ambiguous = resolve_stockanalysis_industry("Other")
    assert ambiguous.raw_industry == "Other"
    assert ambiguous.disposition == "NEEDS_DATA"
    assert ambiguous.reason_code == EXTERNAL_INDUSTRY_UNRESOLVED
    assert ambiguous.reason_detail == "PROVIDER_OTHER_LABEL_HAS_NO_AUDITABLE_SECTOR_MEANING"
    assert ambiguous.source_sector_label is None

    unknown = resolve_stockanalysis_industry("Quantum Widgets")
    assert unknown.raw_industry == "Quantum Widgets"
    assert unknown.disposition == "NEEDS_DATA"
    assert unknown.reason_code == EXTERNAL_INDUSTRY_UNRESOLVED
    assert unknown.reason_detail == "INDUSTRY_NOT_IN_VERSIONED_EXTERNAL_INVENTORY"
    assert unknown.source_sector_label is None

    missing = resolve_stockanalysis_industry(None)
    assert missing.raw_industry is None
    assert missing.disposition == "NEEDS_DATA"
    assert missing.reason_code == EXTERNAL_INDUSTRY_UNRESOLVED
    assert missing.reason_detail == "MISSING_EXTERNAL_INDUSTRY"
    assert missing.source_sector_label is None


def test_manifest_carries_stable_snapshot_and_mapping_provenance():
    manifest = external_industry_taxonomy_manifest()
    assert manifest["taxonomy_version"] == "stockanalysis_industry_to_ivi_sector.v1"
    assert manifest["taxonomy_hash"] == (
        "cccffea84695cc998b5649e2f203487da509b421129795b0e88e73201d17fbaa"
    )
    assert manifest["resolved_industry_count"] == 148
    assert manifest["unresolved_industry_count"] == 4
    assert manifest["snapshot"] == {
        "as_of": "2026-07-17",
        "source_url": "https://stockanalysis.com/stocks/screener/",
        "row_count": 5_598,
        "non_null_industry_count": 5_596,
        "unique_industry_count": 152,
        "boundary_cap_floor_usd": 9_500_000_000,
        "boundary_row_count": 981,
    }
    assert manifest["resolved"]["Semiconductors"] == "semiconductors"
    assert manifest["resolved"]["Software - Infrastructure"] == "enterprise_software"
    assert manifest["unresolved"]["Shell Companies"] == (
        "NON_OPERATING_SHELL_REQUIRES_SECURITY_DISPOSITION_BEFORE_SECTOR"
    )
