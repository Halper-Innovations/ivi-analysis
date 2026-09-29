"""Exact StockAnalysis industry routing for the U.S. equity census.

StockAnalysis industry names are external evidence, not IVI sector aliases.  A
label is routed only when its raw string is present in this versioned registry.
Unknown labels remain visible as ``NEEDS_DATA``; this module intentionally has
no substring, regex, or generic fallback path.

The tracked inventories below were extracted from the 5,598-row StockAnalysis
stocks-screener snapshot captured on 2026-07-17.  The boundary inventory is the
set of industries represented by rows with provider market cap of at least
$9.5 billion, so taxonomy drift near the large-cap gate fails validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

from app.sector.canonical_taxonomy import ACTIVE_SECTOR_LABELS


STOCKANALYSIS_INDUSTRY_TAXONOMY_VERSION: Final = "stockanalysis_industry_to_ivi_sector.v1"
STOCKANALYSIS_SNAPSHOT_AS_OF: Final = "2026-07-17"
STOCKANALYSIS_SNAPSHOT_SOURCE_URL: Final = "https://stockanalysis.com/stocks/screener/"
STOCKANALYSIS_SNAPSHOT_ROW_COUNT: Final = 5_598
STOCKANALYSIS_SNAPSHOT_NON_NULL_INDUSTRY_COUNT: Final = 5_596
STOCKANALYSIS_SNAPSHOT_UNIQUE_INDUSTRY_COUNT: Final = 152
STOCKANALYSIS_BOUNDARY_CAP_FLOOR_USD: Final = 9_500_000_000
STOCKANALYSIS_BOUNDARY_ROW_COUNT: Final = 981

EXTERNAL_INDUSTRY_UNRESOLVED: Final = "EXTERNAL_INDUSTRY_UNRESOLVED"
EXACT_EXTERNAL_INDUSTRY_MATCH: Final = "EXACT_STOCKANALYSIS_INDUSTRY_MATCH"


@dataclass(frozen=True, slots=True)
class ExternalIndustryResolution:
    """Exact routing result that retains the provider's raw industry value."""

    raw_industry: str | None
    disposition: str
    reason_code: str
    reason_detail: str
    source_sector_label: str | None
    taxonomy_version: str
    taxonomy_hash: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "raw_industry": self.raw_industry,
            "disposition": self.disposition,
            "reason_code": self.reason_code,
            "reason_detail": self.reason_detail,
            "source_sector_label": self.source_sector_label,
            "taxonomy_version": self.taxonomy_version,
            "taxonomy_hash": self.taxonomy_hash,
        }


# Exact provider strings.  Grouping comments are for review only; lookup uses
# the literal dictionary keys and never derives a route from those comments.
_INDUSTRY_TO_SECTOR: Final = {
    # Communications and media.
    "Advertising Agencies": "media_entertainment",
    "Broadcasting": "media_entertainment",
    "Electronic Gaming & Multimedia": "media_entertainment",
    "Entertainment": "media_entertainment",
    "Internet Content & Information": "internet_services",
    "Media & Entertainment": "media_entertainment",
    "Publishing": "media_entertainment",
    "Telecom Services": "telecom",
    # Aerospace, transportation, and industrial operations.
    "Aerospace & Defense": "aerospace_defense",
    "Airlines": "transportation_logistics",
    "Airports & Air Services": "transportation_logistics",
    "Business Equipment & Supplies": "business_services",
    "Conglomerates": "diversified_industrials",
    "Consulting Services": "business_services",
    "Electrical Equipment & Parts": "industrial_tech",
    "Electronics & Computer Distribution": "industrial_tech",
    "Industrial Distribution": "diversified_industrials",
    "Infrastructure Operations": "diversified_industrials",
    "Integrated Freight & Logistics": "transportation_logistics",
    "Marine Shipping": "transportation_logistics",
    "Packaging & Containers": "diversified_industrials",
    "Pollution & Treatment Controls": "industrial_tech",
    "Railroads": "transportation_logistics",
    "Rental & Leasing Services": "business_services",
    "Scientific & Technical Instruments": "industrial_tech",
    "Security & Protection Services": "business_services",
    "Specialty Business Services": "business_services",
    "Specialty Industrial Machinery": "industrial_tech",
    "Staffing & Employment Services": "business_services",
    "Tools & Accessories": "diversified_industrials",
    "Trucking": "transportation_logistics",
    "Waste Management": "business_services",
    # Automotive, construction, and building products.
    "Auto & Truck Dealerships": "automotive",
    "Auto Manufacturers": "automotive",
    "Auto Parts": "automotive",
    "Building Materials": "building_products",
    "Building Products & Equipment": "building_products",
    "Engineering & Construction": "construction_services",
    "Farm & Heavy Construction Machinery": "construction_machinery",
    "Lumber & Wood Production": "building_products",
    "Recreational Vehicles": "automotive",
    "Residential Construction": "construction_services",
    # Technology hardware, software, and semiconductors.
    "Communication Equipment": "industrial_tech",
    "Computer Hardware": "industrial_tech",
    "Consumer Electronics": "industrial_tech",
    "Electronic Components": "industrial_tech",
    "Information Technology Services": "enterprise_software",
    "Semiconductor Equipment & Materials": "semiconductors",
    "Semiconductors": "semiconductors",
    "Software - Application": "enterprise_software",
    "Software - Infrastructure": "enterprise_software",
    "Software - Services": "enterprise_software",
    # Financial services.
    "Asset Management": "capital_markets",
    "Banks - Diversified": "large_cap_financials",
    "Banks - Regional": "large_cap_financials",
    "Capital Markets": "capital_markets",
    "Credit Services": "payments_fintech",
    "Financial - Capital Markets": "capital_markets",
    "Financial Conglomerates": "capital_markets",
    "Financial Data & Stock Exchanges": "capital_markets",
    "Insurance - Diversified": "insurance",
    "Insurance - Life": "insurance",
    "Insurance - Property & Casualty": "insurance",
    "Insurance - Reinsurance": "insurance",
    "Insurance - Specialty": "insurance",
    "Insurance Brokers": "insurance",
    "Mortgage Finance": "capital_markets",
    # Healthcare.
    "Biotechnology": "biotech",
    "Diagnostics & Research": "medical_devices",
    "Drug Manufacturers - General": "healthcare_pharma",
    "Drug Manufacturers - Specialty & Generic": "healthcare_pharma",
    "Health Information Services": "healthcare_services",
    "Healthcare Plans": "insurance",
    "Medical - Specialties": "medical_devices",
    "Medical Care Facilities": "healthcare_services",
    "Medical Devices": "medical_devices",
    "Medical Distribution": "medical_devices",
    "Medical Instruments & Supplies": "medical_devices",
    "Pharmaceutical Retailers": "healthcare_pharma",
    # Energy, utilities, chemicals, and mined materials.
    "Agricultural Inputs": "chemicals",
    "Aluminum": "metals_mining",
    "Chemicals": "chemicals",
    "Coking Coal": "energy",
    "Copper": "metals_mining",
    "Gold": "metals_mining",
    "Metal Fabrication": "metals_mining",
    "Oil & Gas Drilling": "energy",
    "Oil & Gas Equipment & Services": "energy",
    "Oil & Gas Exploration & Production": "energy",
    "Oil & Gas Integrated": "energy",
    "Oil & Gas Midstream": "energy",
    "Oil & Gas Refining & Marketing": "energy",
    "Other Industrial Metals & Mining": "metals_mining",
    "Other Precious Metals & Mining": "metals_mining",
    "Silver": "metals_mining",
    "Solar": "energy",
    "Specialty Chemicals": "chemicals",
    "Steel": "metals_mining",
    "Thermal Coal": "energy",
    "Uranium": "metals_mining",
    "Utilities - Diversified": "utilities",
    "Utilities - Independent Power Producers": "utilities",
    "Utilities - Regulated Electric": "utilities",
    "Utilities - Regulated Gas": "utilities",
    "Utilities - Regulated Water": "utilities",
    "Utilities - Renewable": "utilities",
    # Consumer products, retail, services, hospitality, and food.
    "Apparel Manufacturing": "consumer_discretionary",
    "Apparel Retail": "retail",
    "Beverages - Brewers": "consumer_staples",
    "Beverages - Non-Alcoholic": "consumer_staples",
    "Beverages - Wineries & Distilleries": "consumer_staples",
    "Confectioners": "consumer_staples",
    "Department Stores": "retail",
    "Discount Stores": "retail",
    "Education & Training Services": "education_services",
    "Farm Products": "consumer_staples",
    "Food Distribution": "restaurants_food_service",
    "Footwear & Accessories": "consumer_discretionary",
    "Furnishings, Fixtures & Appliances": "consumer_discretionary",
    "Gambling": "hospitality_gaming",
    "Grocery Stores": "retail",
    "Home Improvement Retail": "retail",
    "Household & Personal Products": "consumer_staples",
    "Internet Retail": "retail",
    "Leisure": "consumer_discretionary",
    "Lodging": "hospitality_gaming",
    "Luxury Goods": "consumer_discretionary",
    "Packaged Foods": "consumer_staples",
    "Paper & Paper Products": "consumer_staples",
    "Personal Services": "consumer_services",
    "Resorts & Casinos": "hospitality_gaming",
    "Restaurants": "restaurants_food_service",
    "Specialty Retail": "retail",
    "Textile Manufacturing": "consumer_discretionary",
    "Tobacco": "consumer_staples",
    "Travel Services": "hospitality_gaming",
    # Real estate.  The active IVI source label is historically named ``reits``
    # and is the explicit contract for both REIT and operating real-estate rows.
    "REIT - Diversified": "reits",
    "REIT - Healthcare Facilities": "reits",
    "REIT - Hotel & Motel": "reits",
    "REIT - Industrial": "reits",
    "REIT - Mortgage": "reits",
    "REIT - Office": "reits",
    "REIT - Residential": "reits",
    "REIT - Retail": "reits",
    "REIT - Specialty": "reits",
    "Real Estate - Development": "reits",
    "Real Estate - Diversified": "reits",
    "Real Estate Services": "reits",
}


_UNRESOLVED_INDUSTRIES: Final = {
    "Asset Management - Cryptocurrency": (
        "PROVIDER_LABEL_IS_NON_OPERATING_CRYPTO_INVESTMENT_VEHICLE_IN_SNAPSHOT"
    ),
    "Asset Management - Income": ("PROVIDER_LABEL_IS_NON_OPERATING_REGISTERED_FUND_IN_SNAPSHOT"),
    "Other": "PROVIDER_OTHER_LABEL_HAS_NO_AUDITABLE_SECTOR_MEANING",
    "Shell Companies": "NON_OPERATING_SHELL_REQUIRES_SECURITY_DISPOSITION_BEFORE_SECTOR",
}


# Independent, sorted inventory contract from the 2026-07-17 hydration payload.
STOCKANALYSIS_INDUSTRY_INVENTORY: Final = (
    "Advertising Agencies",
    "Aerospace & Defense",
    "Agricultural Inputs",
    "Airlines",
    "Airports & Air Services",
    "Aluminum",
    "Apparel Manufacturing",
    "Apparel Retail",
    "Asset Management",
    "Asset Management - Cryptocurrency",
    "Asset Management - Income",
    "Auto & Truck Dealerships",
    "Auto Manufacturers",
    "Auto Parts",
    "Banks - Diversified",
    "Banks - Regional",
    "Beverages - Brewers",
    "Beverages - Non-Alcoholic",
    "Beverages - Wineries & Distilleries",
    "Biotechnology",
    "Broadcasting",
    "Building Materials",
    "Building Products & Equipment",
    "Business Equipment & Supplies",
    "Capital Markets",
    "Chemicals",
    "Coking Coal",
    "Communication Equipment",
    "Computer Hardware",
    "Confectioners",
    "Conglomerates",
    "Consulting Services",
    "Consumer Electronics",
    "Copper",
    "Credit Services",
    "Department Stores",
    "Diagnostics & Research",
    "Discount Stores",
    "Drug Manufacturers - General",
    "Drug Manufacturers - Specialty & Generic",
    "Education & Training Services",
    "Electrical Equipment & Parts",
    "Electronic Components",
    "Electronic Gaming & Multimedia",
    "Electronics & Computer Distribution",
    "Engineering & Construction",
    "Entertainment",
    "Farm & Heavy Construction Machinery",
    "Farm Products",
    "Financial - Capital Markets",
    "Financial Conglomerates",
    "Financial Data & Stock Exchanges",
    "Food Distribution",
    "Footwear & Accessories",
    "Furnishings, Fixtures & Appliances",
    "Gambling",
    "Gold",
    "Grocery Stores",
    "Health Information Services",
    "Healthcare Plans",
    "Home Improvement Retail",
    "Household & Personal Products",
    "Industrial Distribution",
    "Information Technology Services",
    "Infrastructure Operations",
    "Insurance - Diversified",
    "Insurance - Life",
    "Insurance - Property & Casualty",
    "Insurance - Reinsurance",
    "Insurance - Specialty",
    "Insurance Brokers",
    "Integrated Freight & Logistics",
    "Internet Content & Information",
    "Internet Retail",
    "Leisure",
    "Lodging",
    "Lumber & Wood Production",
    "Luxury Goods",
    "Marine Shipping",
    "Media & Entertainment",
    "Medical - Specialties",
    "Medical Care Facilities",
    "Medical Devices",
    "Medical Distribution",
    "Medical Instruments & Supplies",
    "Metal Fabrication",
    "Mortgage Finance",
    "Oil & Gas Drilling",
    "Oil & Gas Equipment & Services",
    "Oil & Gas Exploration & Production",
    "Oil & Gas Integrated",
    "Oil & Gas Midstream",
    "Oil & Gas Refining & Marketing",
    "Other",
    "Other Industrial Metals & Mining",
    "Other Precious Metals & Mining",
    "Packaged Foods",
    "Packaging & Containers",
    "Paper & Paper Products",
    "Personal Services",
    "Pharmaceutical Retailers",
    "Pollution & Treatment Controls",
    "Publishing",
    "REIT - Diversified",
    "REIT - Healthcare Facilities",
    "REIT - Hotel & Motel",
    "REIT - Industrial",
    "REIT - Mortgage",
    "REIT - Office",
    "REIT - Residential",
    "REIT - Retail",
    "REIT - Specialty",
    "Railroads",
    "Real Estate - Development",
    "Real Estate - Diversified",
    "Real Estate Services",
    "Recreational Vehicles",
    "Rental & Leasing Services",
    "Residential Construction",
    "Resorts & Casinos",
    "Restaurants",
    "Scientific & Technical Instruments",
    "Security & Protection Services",
    "Semiconductor Equipment & Materials",
    "Semiconductors",
    "Shell Companies",
    "Silver",
    "Software - Application",
    "Software - Infrastructure",
    "Software - Services",
    "Solar",
    "Specialty Business Services",
    "Specialty Chemicals",
    "Specialty Industrial Machinery",
    "Specialty Retail",
    "Staffing & Employment Services",
    "Steel",
    "Telecom Services",
    "Textile Manufacturing",
    "Thermal Coal",
    "Tobacco",
    "Tools & Accessories",
    "Travel Services",
    "Trucking",
    "Uranium",
    "Utilities - Diversified",
    "Utilities - Independent Power Producers",
    "Utilities - Regulated Electric",
    "Utilities - Regulated Gas",
    "Utilities - Regulated Water",
    "Utilities - Renewable",
    "Waste Management",
)


# All non-null industries represented at or above $9.5B in the same snapshot.
STOCKANALYSIS_BOUNDARY_INDUSTRY_INVENTORY: Final = (
    "Advertising Agencies",
    "Aerospace & Defense",
    "Agricultural Inputs",
    "Airlines",
    "Airports & Air Services",
    "Aluminum",
    "Apparel Manufacturing",
    "Apparel Retail",
    "Asset Management",
    "Auto & Truck Dealerships",
    "Auto Manufacturers",
    "Auto Parts",
    "Banks - Diversified",
    "Banks - Regional",
    "Beverages - Brewers",
    "Beverages - Non-Alcoholic",
    "Beverages - Wineries & Distilleries",
    "Biotechnology",
    "Building Materials",
    "Building Products & Equipment",
    "Capital Markets",
    "Chemicals",
    "Communication Equipment",
    "Computer Hardware",
    "Confectioners",
    "Conglomerates",
    "Consulting Services",
    "Consumer Electronics",
    "Copper",
    "Credit Services",
    "Diagnostics & Research",
    "Discount Stores",
    "Drug Manufacturers - General",
    "Drug Manufacturers - Specialty & Generic",
    "Electrical Equipment & Parts",
    "Electronic Components",
    "Electronic Gaming & Multimedia",
    "Electronics & Computer Distribution",
    "Engineering & Construction",
    "Entertainment",
    "Farm & Heavy Construction Machinery",
    "Farm Products",
    "Financial Conglomerates",
    "Financial Data & Stock Exchanges",
    "Food Distribution",
    "Footwear & Accessories",
    "Furnishings, Fixtures & Appliances",
    "Gambling",
    "Gold",
    "Grocery Stores",
    "Health Information Services",
    "Healthcare Plans",
    "Home Improvement Retail",
    "Household & Personal Products",
    "Industrial Distribution",
    "Information Technology Services",
    "Infrastructure Operations",
    "Insurance - Diversified",
    "Insurance - Life",
    "Insurance - Property & Casualty",
    "Insurance - Reinsurance",
    "Insurance - Specialty",
    "Insurance Brokers",
    "Integrated Freight & Logistics",
    "Internet Content & Information",
    "Internet Retail",
    "Leisure",
    "Lodging",
    "Luxury Goods",
    "Medical Care Facilities",
    "Medical Devices",
    "Medical Distribution",
    "Medical Instruments & Supplies",
    "Metal Fabrication",
    "Mortgage Finance",
    "Oil & Gas Equipment & Services",
    "Oil & Gas Exploration & Production",
    "Oil & Gas Integrated",
    "Oil & Gas Midstream",
    "Oil & Gas Refining & Marketing",
    "Other Industrial Metals & Mining",
    "Other Precious Metals & Mining",
    "Packaged Foods",
    "Packaging & Containers",
    "Paper & Paper Products",
    "Personal Services",
    "Pollution & Treatment Controls",
    "Publishing",
    "REIT - Diversified",
    "REIT - Healthcare Facilities",
    "REIT - Hotel & Motel",
    "REIT - Industrial",
    "REIT - Mortgage",
    "REIT - Office",
    "REIT - Residential",
    "REIT - Retail",
    "REIT - Specialty",
    "Railroads",
    "Real Estate Services",
    "Rental & Leasing Services",
    "Residential Construction",
    "Resorts & Casinos",
    "Restaurants",
    "Scientific & Technical Instruments",
    "Security & Protection Services",
    "Semiconductor Equipment & Materials",
    "Semiconductors",
    "Software - Application",
    "Software - Infrastructure",
    "Solar",
    "Specialty Business Services",
    "Specialty Chemicals",
    "Specialty Industrial Machinery",
    "Specialty Retail",
    "Steel",
    "Telecom Services",
    "Tobacco",
    "Tools & Accessories",
    "Travel Services",
    "Trucking",
    "Uranium",
    "Utilities - Diversified",
    "Utilities - Independent Power Producers",
    "Utilities - Regulated Electric",
    "Utilities - Regulated Gas",
    "Utilities - Regulated Water",
    "Utilities - Renewable",
    "Waste Management",
)


STOCKANALYSIS_INDUSTRY_TO_SECTOR: Mapping[str, str] = MappingProxyType(_INDUSTRY_TO_SECTOR)
STOCKANALYSIS_UNRESOLVED_INDUSTRIES: Mapping[str, str] = MappingProxyType(_UNRESOLVED_INDUSTRIES)


def _hash_payload() -> dict[str, object]:
    return {
        "taxonomy_version": STOCKANALYSIS_INDUSTRY_TAXONOMY_VERSION,
        "snapshot": {
            "as_of": STOCKANALYSIS_SNAPSHOT_AS_OF,
            "source_url": STOCKANALYSIS_SNAPSHOT_SOURCE_URL,
            "row_count": STOCKANALYSIS_SNAPSHOT_ROW_COUNT,
            "non_null_industry_count": STOCKANALYSIS_SNAPSHOT_NON_NULL_INDUSTRY_COUNT,
            "unique_industry_count": STOCKANALYSIS_SNAPSHOT_UNIQUE_INDUSTRY_COUNT,
            "boundary_cap_floor_usd": STOCKANALYSIS_BOUNDARY_CAP_FLOOR_USD,
            "boundary_row_count": STOCKANALYSIS_BOUNDARY_ROW_COUNT,
        },
        "industry_inventory": list(STOCKANALYSIS_INDUSTRY_INVENTORY),
        "boundary_industry_inventory": list(STOCKANALYSIS_BOUNDARY_INDUSTRY_INVENTORY),
        "resolved": {
            industry: STOCKANALYSIS_INDUSTRY_TO_SECTOR[industry]
            for industry in sorted(STOCKANALYSIS_INDUSTRY_TO_SECTOR)
        },
        "unresolved": {
            industry: STOCKANALYSIS_UNRESOLVED_INDUSTRIES[industry]
            for industry in sorted(STOCKANALYSIS_UNRESOLVED_INDUSTRIES)
        },
    }


def _compute_taxonomy_hash() -> str:
    encoded = json.dumps(
        _hash_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


STOCKANALYSIS_INDUSTRY_TAXONOMY_HASH: Final = _compute_taxonomy_hash()


def validate_external_industry_taxonomy() -> None:
    """Raise when exact inventory, routing, or boundary invariants drift."""

    inventory = set(STOCKANALYSIS_INDUSTRY_INVENTORY)
    resolved = set(STOCKANALYSIS_INDUSTRY_TO_SECTOR)
    unresolved = set(STOCKANALYSIS_UNRESOLVED_INDUSTRIES)
    active = set(ACTIVE_SECTOR_LABELS)

    if len(STOCKANALYSIS_INDUSTRY_INVENTORY) != 152 or len(inventory) != 152:
        raise ValueError("StockAnalysis inventory must contain 152 unique industry labels")
    if tuple(sorted(inventory)) != STOCKANALYSIS_INDUSTRY_INVENTORY:
        raise ValueError("StockAnalysis industry inventory must remain sorted")
    if resolved & unresolved:
        raise ValueError("an external industry cannot be both resolved and unresolved")
    if resolved | unresolved != inventory:
        raise ValueError("every tracked external industry requires one explicit disposition")
    if not unresolved:
        raise ValueError("provider-ambiguous industries must remain explicitly visible")
    if any(not reason.strip() for reason in STOCKANALYSIS_UNRESOLVED_INDUSTRIES.values()):
        raise ValueError("every unresolved external industry requires a reason")
    invalid_targets = set(STOCKANALYSIS_INDUSTRY_TO_SECTOR.values()) - active
    if invalid_targets:
        raise ValueError(f"external industry routes to inactive labels: {sorted(invalid_targets)}")
    if any(target in {"generic", "other", "unknown"} for target in _INDUSTRY_TO_SECTOR.values()):
        raise ValueError("external industry routing cannot use a generic fallback target")

    boundary = set(STOCKANALYSIS_BOUNDARY_INDUSTRY_INVENTORY)
    if len(STOCKANALYSIS_BOUNDARY_INDUSTRY_INVENTORY) != 128 or len(boundary) != 128:
        raise ValueError("StockAnalysis $9.5B boundary inventory must contain 128 labels")
    if tuple(sorted(boundary)) != STOCKANALYSIS_BOUNDARY_INDUSTRY_INVENTORY:
        raise ValueError("StockAnalysis boundary inventory must remain sorted")
    if not boundary <= resolved:
        raise ValueError("every industry at the $9.5B boundary must resolve explicitly")
    if _compute_taxonomy_hash() != STOCKANALYSIS_INDUSTRY_TAXONOMY_HASH:
        raise ValueError("external industry taxonomy hash does not match its registry")


def resolve_stockanalysis_industry(industry: object) -> ExternalIndustryResolution:
    """Resolve one raw StockAnalysis industry only by exact registry lookup."""

    if industry is None:
        raw_industry = None
    elif isinstance(industry, str):
        raw_industry = industry
    else:
        raw_industry = str(industry)

    sector = STOCKANALYSIS_INDUSTRY_TO_SECTOR.get(raw_industry or "")
    if sector is not None:
        return ExternalIndustryResolution(
            raw_industry=raw_industry,
            disposition="RESOLVED",
            reason_code=EXACT_EXTERNAL_INDUSTRY_MATCH,
            reason_detail="exact raw StockAnalysis industry registry match",
            source_sector_label=sector,
            taxonomy_version=STOCKANALYSIS_INDUSTRY_TAXONOMY_VERSION,
            taxonomy_hash=STOCKANALYSIS_INDUSTRY_TAXONOMY_HASH,
        )

    detail = STOCKANALYSIS_UNRESOLVED_INDUSTRIES.get(raw_industry or "")
    if detail is None:
        detail = (
            "MISSING_EXTERNAL_INDUSTRY"
            if raw_industry is None or raw_industry == ""
            else "INDUSTRY_NOT_IN_VERSIONED_EXTERNAL_INVENTORY"
        )
    return ExternalIndustryResolution(
        raw_industry=raw_industry,
        disposition="NEEDS_DATA",
        reason_code=EXTERNAL_INDUSTRY_UNRESOLVED,
        reason_detail=detail,
        source_sector_label=None,
        taxonomy_version=STOCKANALYSIS_INDUSTRY_TAXONOMY_VERSION,
        taxonomy_hash=STOCKANALYSIS_INDUSTRY_TAXONOMY_HASH,
    )


def external_industry_taxonomy_manifest() -> dict[str, object]:
    """Return a stable JSON-ready provenance manifest for census artifacts."""

    return {
        **_hash_payload(),
        "taxonomy_hash": STOCKANALYSIS_INDUSTRY_TAXONOMY_HASH,
        "resolved_industry_count": len(STOCKANALYSIS_INDUSTRY_TO_SECTOR),
        "unresolved_industry_count": len(STOCKANALYSIS_UNRESOLVED_INDUSTRIES),
    }


validate_external_industry_taxonomy()


__all__ = [
    "EXACT_EXTERNAL_INDUSTRY_MATCH",
    "EXTERNAL_INDUSTRY_UNRESOLVED",
    "ExternalIndustryResolution",
    "STOCKANALYSIS_BOUNDARY_CAP_FLOOR_USD",
    "STOCKANALYSIS_BOUNDARY_INDUSTRY_INVENTORY",
    "STOCKANALYSIS_BOUNDARY_ROW_COUNT",
    "STOCKANALYSIS_INDUSTRY_INVENTORY",
    "STOCKANALYSIS_INDUSTRY_TAXONOMY_HASH",
    "STOCKANALYSIS_INDUSTRY_TAXONOMY_VERSION",
    "STOCKANALYSIS_INDUSTRY_TO_SECTOR",
    "STOCKANALYSIS_SNAPSHOT_AS_OF",
    "STOCKANALYSIS_SNAPSHOT_NON_NULL_INDUSTRY_COUNT",
    "STOCKANALYSIS_SNAPSHOT_ROW_COUNT",
    "STOCKANALYSIS_SNAPSHOT_SOURCE_URL",
    "STOCKANALYSIS_SNAPSHOT_UNIQUE_INDUSTRY_COUNT",
    "STOCKANALYSIS_UNRESOLVED_INDUSTRIES",
    "external_industry_taxonomy_manifest",
    "resolve_stockanalysis_industry",
    "validate_external_industry_taxonomy",
]
