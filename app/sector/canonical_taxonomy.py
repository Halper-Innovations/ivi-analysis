"""Versioned canonical sector contracts for the U.S. equity census.

This registry is intentionally narrower than narrative sector inference.  It
accepts an active source label only by exact match after superficial formatting
normalization; it never routes by substring and never falls back to a generic
framework.  An unregistered label therefore remains visible as ``NEEDS_DATA``.

``large_cap_financials`` is retained as the historical source label because it
already exists in persisted sector inference.  Its canonical sector ID is
``banking``.  ``banking`` is not an input alias: callers must migrate persisted
source labels explicitly rather than silently changing their meaning here.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping


CANONICAL_SECTOR_TAXONOMY_VERSION: Final = "us_equity_sector_taxonomy.v1"
SECTOR_CLASSIFICATION_UNRESOLVED: Final = "SECTOR_CLASSIFICATION_UNRESOLVED"

STANDARD_SCREEN_PROFILE_ID: Final = "deterministic_standard_v1"
BALANCE_SHEET_SCREEN_PROFILE_ID: Final = "deterministic_balance_sheet_v1"

_STANDARD_SCREEN_RULE_IDS: Final = (
    "NON_PRIMARY_LISTING",
    "DELISTING_NOTICE",
    "PENNY_FLOOR",
    "NANO_FLOOR",
    "EARNINGS_QUALITY_DIVERGENCE",
    "GOING_CONCERN",
)
_BALANCE_SHEET_SCREEN_RULE_IDS: Final = (
    "NON_PRIMARY_LISTING",
    "DELISTING_NOTICE",
    "PENNY_FLOOR",
    "NANO_FLOOR",
    "GOING_CONCERN",
)

SCREEN_PROFILE_RULE_IDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        STANDARD_SCREEN_PROFILE_ID: _STANDARD_SCREEN_RULE_IDS,
        BALANCE_SHEET_SCREEN_PROFILE_ID: _BALANCE_SHEET_SCREEN_RULE_IDS,
    }
)


@dataclass(frozen=True, slots=True)
class CanonicalSectorContract:
    """Exact, persisted routing contract for one active source label."""

    source_label: str
    canonical_sector_id: str
    framework_contract_id: str
    screen_profile_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source_label": self.source_label,
            "canonical_sector_id": self.canonical_sector_id,
            "framework_contract_id": self.framework_contract_id,
            "screen_profile_id": self.screen_profile_id,
        }


@dataclass(frozen=True, slots=True)
class SectorResolution:
    """Result of exact active-taxonomy lookup."""

    input_label: str
    normalized_label: str
    disposition: str
    reason_code: str
    taxonomy_version: str
    taxonomy_hash: str
    contract: CanonicalSectorContract | None

    def to_dict(self) -> dict[str, str | None]:
        contract = self.contract
        return {
            "input_label": self.input_label,
            "normalized_label": self.normalized_label,
            "disposition": self.disposition,
            "reason_code": self.reason_code,
            "taxonomy_version": self.taxonomy_version,
            "taxonomy_hash": self.taxonomy_hash,
            "source_label": contract.source_label if contract else None,
            "canonical_sector_id": contract.canonical_sector_id if contract else None,
            "framework_contract_id": contract.framework_contract_id if contract else None,
            "screen_profile_id": contract.screen_profile_id if contract else None,
        }


ACTIVE_SECTOR_LABELS: Final = (
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


_CONTRACTS: Final = (
    CanonicalSectorContract(
        "aerospace_defense", "aerospace_defense", "industrial", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract("automotive", "automotive", "automotive", STANDARD_SCREEN_PROFILE_ID),
    CanonicalSectorContract("biotech", "biotech", "healthcare", STANDARD_SCREEN_PROFILE_ID),
    CanonicalSectorContract(
        "building_products", "building_products", "industrial", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "business_services", "business_services", "industrial", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "capital_markets", "capital_markets", "capital_markets", BALANCE_SHEET_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract("chemicals", "chemicals", "materials", STANDARD_SCREEN_PROFILE_ID),
    CanonicalSectorContract(
        "construction_machinery",
        "construction_machinery",
        "industrial",
        STANDARD_SCREEN_PROFILE_ID,
    ),
    CanonicalSectorContract(
        "construction_services", "construction_services", "industrial", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "consumer_discretionary", "consumer_discretionary", "consumer", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "consumer_services", "consumer_services", "consumer", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "consumer_staples", "consumer_staples", "consumer", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "diversified_industrials",
        "diversified_industrials",
        "industrial",
        STANDARD_SCREEN_PROFILE_ID,
    ),
    CanonicalSectorContract(
        "education_services", "education_services", "education_services", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract("energy", "energy", "energy", STANDARD_SCREEN_PROFILE_ID),
    CanonicalSectorContract(
        "enterprise_software", "enterprise_software", "technology", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "healthcare_pharma", "healthcare_pharma", "healthcare", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "healthcare_services", "healthcare_services", "healthcare", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "hospitality_gaming",
        "hospitality_gaming",
        "hospitality_gaming",
        STANDARD_SCREEN_PROFILE_ID,
    ),
    CanonicalSectorContract(
        "industrial_tech", "industrial_tech", "industrial", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "insurance", "insurance", "financial_services", BALANCE_SHEET_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "internet_services", "internet_services", "technology", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "large_cap_financials", "banking", "financial_services", BALANCE_SHEET_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "media_entertainment",
        "media_entertainment",
        "communications_media",
        STANDARD_SCREEN_PROFILE_ID,
    ),
    CanonicalSectorContract(
        "medical_devices", "medical_devices", "medical_devices", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "metals_mining", "metals_mining", "materials", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "payments_fintech", "payments_fintech", "payments_fintech", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract("reits", "reits", "real_estate", BALANCE_SHEET_SCREEN_PROFILE_ID),
    CanonicalSectorContract(
        "restaurants_food_service",
        "restaurants_food_service",
        "consumer",
        STANDARD_SCREEN_PROFILE_ID,
    ),
    CanonicalSectorContract("retail", "retail", "consumer", STANDARD_SCREEN_PROFILE_ID),
    CanonicalSectorContract(
        "semiconductors", "semiconductors", "technology", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "telecom", "telecom", "communications_media", STANDARD_SCREEN_PROFILE_ID
    ),
    CanonicalSectorContract(
        "transportation_logistics",
        "transportation_logistics",
        "industrial",
        STANDARD_SCREEN_PROFILE_ID,
    ),
    CanonicalSectorContract("utilities", "utilities", "utilities", STANDARD_SCREEN_PROFILE_ID),
)


_FORMATTING_SEPARATOR_RE: Final = re.compile(r"[-\s]+")
_REPEATED_UNDERSCORE_RE: Final = re.compile(r"_+")


def normalize_sector_label(label: object) -> str:
    """Normalize casing and separators without performing semantic aliasing."""

    text = unicodedata.normalize("NFKC", str(label or "")).strip().casefold()
    text = _FORMATTING_SEPARATOR_RE.sub("_", text)
    return _REPEATED_UNDERSCORE_RE.sub("_", text).strip("_")


def _validate_contracts(contracts: tuple[CanonicalSectorContract, ...]) -> None:
    labels = tuple(contract.source_label for contract in contracts)
    if labels != ACTIVE_SECTOR_LABELS:
        raise ValueError("canonical sector contracts must exactly match the ordered active labels")
    if len(labels) != 34 or len(set(labels)) != 34:
        raise ValueError("canonical sector taxonomy must contain exactly 34 unique labels")
    for contract in contracts:
        if normalize_sector_label(contract.source_label) != contract.source_label:
            raise ValueError(f"source label is not normalized: {contract.source_label}")
        if not contract.canonical_sector_id:
            raise ValueError(f"canonical sector ID is missing: {contract.source_label}")
        if not contract.framework_contract_id or contract.framework_contract_id == "generic":
            raise ValueError(f"explicit framework contract is required: {contract.source_label}")
        if contract.screen_profile_id not in SCREEN_PROFILE_RULE_IDS:
            raise ValueError(f"unknown screen profile: {contract.source_label}")
        if not SCREEN_PROFILE_RULE_IDS[contract.screen_profile_id]:
            raise ValueError(f"screen profile has no rules: {contract.source_label}")
    legacy = next(
        contract for contract in contracts if contract.source_label == "large_cap_financials"
    )
    if legacy.canonical_sector_id != "banking":
        raise ValueError("large_cap_financials must retain the documented banking canonical ID")


_validate_contracts(_CONTRACTS)

CANONICAL_SECTOR_CONTRACTS: Mapping[str, CanonicalSectorContract] = MappingProxyType(
    {contract.source_label: contract for contract in _CONTRACTS}
)


def _hash_payload() -> dict[str, object]:
    return {
        "taxonomy_version": CANONICAL_SECTOR_TAXONOMY_VERSION,
        "contracts": [contract.to_dict() for contract in _CONTRACTS],
        "screen_profiles": {
            profile_id: list(SCREEN_PROFILE_RULE_IDS[profile_id])
            for profile_id in sorted(SCREEN_PROFILE_RULE_IDS)
        },
    }


def _compute_taxonomy_hash() -> str:
    encoded = json.dumps(
        _hash_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


CANONICAL_SECTOR_TAXONOMY_HASH: Final = _compute_taxonomy_hash()


def validate_canonical_sector_taxonomy() -> None:
    """Raise when active taxonomy invariants or the published hash drift."""

    _validate_contracts(_CONTRACTS)
    if tuple(CANONICAL_SECTOR_CONTRACTS) != ACTIVE_SECTOR_LABELS:
        raise ValueError("canonical sector contract index is not ordered with active labels")
    if _compute_taxonomy_hash() != CANONICAL_SECTOR_TAXONOMY_HASH:
        raise ValueError("canonical sector taxonomy hash does not match its registry")


def resolve_canonical_sector(label: object) -> SectorResolution:
    """Resolve one exact active label, or return an explicit ``NEEDS_DATA`` result."""

    input_label = str(label or "")
    normalized = normalize_sector_label(input_label)
    contract = CANONICAL_SECTOR_CONTRACTS.get(normalized)
    if contract is None:
        return SectorResolution(
            input_label=input_label,
            normalized_label=normalized,
            disposition="NEEDS_DATA",
            reason_code=SECTOR_CLASSIFICATION_UNRESOLVED,
            taxonomy_version=CANONICAL_SECTOR_TAXONOMY_VERSION,
            taxonomy_hash=CANONICAL_SECTOR_TAXONOMY_HASH,
            contract=None,
        )
    return SectorResolution(
        input_label=input_label,
        normalized_label=normalized,
        disposition="RESOLVED",
        reason_code="EXACT_ACTIVE_TAXONOMY_MATCH",
        taxonomy_version=CANONICAL_SECTOR_TAXONOMY_VERSION,
        taxonomy_hash=CANONICAL_SECTOR_TAXONOMY_HASH,
        contract=contract,
    )


def canonical_sector_taxonomy_manifest() -> dict[str, object]:
    """Return a stable JSON-ready manifest for census provenance."""

    return {
        **_hash_payload(),
        "taxonomy_hash": CANONICAL_SECTOR_TAXONOMY_HASH,
        "active_label_count": 34,
    }


__all__ = [
    "ACTIVE_SECTOR_LABELS",
    "BALANCE_SHEET_SCREEN_PROFILE_ID",
    "CANONICAL_SECTOR_CONTRACTS",
    "CANONICAL_SECTOR_TAXONOMY_HASH",
    "CANONICAL_SECTOR_TAXONOMY_VERSION",
    "CanonicalSectorContract",
    "SCREEN_PROFILE_RULE_IDS",
    "SECTOR_CLASSIFICATION_UNRESOLVED",
    "STANDARD_SCREEN_PROFILE_ID",
    "SectorResolution",
    "canonical_sector_taxonomy_manifest",
    "normalize_sector_label",
    "resolve_canonical_sector",
    "validate_canonical_sector_taxonomy",
]
