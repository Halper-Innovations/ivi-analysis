"""Registry-first, resumable U.S. public-company security and issuer census.

The census is deliberately separate from sector scans.  Membership starts with
dated security registries; sector, cap, facts, filings, and packet coverage are
attributes resolved only after security identity and issuer deduplication.

Every finalized security and issuer row has exactly one terminal disposition.
Unknown evidence is represented by a ``NEEDS_DATA_*`` disposition and is never
silently dropped or treated as a negative eligibility decision.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.autonomous.cap_resolver import band_for_market_cap
from app.db import init_db, utc_now_iso
from app.sector.canonical_taxonomy import (
    CANONICAL_SECTOR_TAXONOMY_HASH,
    CANONICAL_SECTOR_TAXONOMY_VERSION,
    canonical_sector_taxonomy_manifest,
    resolve_canonical_sector,
)
from app.sector.external_industry_taxonomy import (
    STOCKANALYSIS_INDUSTRY_TAXONOMY_HASH,
    STOCKANALYSIS_INDUSTRY_TAXONOMY_VERSION,
    external_industry_taxonomy_manifest,
    resolve_stockanalysis_industry,
)
from app.universe.us_equity_census_sources import (
    CompaniesMarketCapSecurity,
    NasdaqTraderSecurity,
    ParseIssue,
    SecExchangeSecurity,
    StockAnalysisSecurity,
    parse_companiesmarketcap_usa_html,
    parse_nasdaqlisted,
    parse_otherlisted,
    parse_sec_company_tickers_exchange,
    parse_stockanalysis_screener_html,
)


TARGET_EXCHANGES = frozenset({"NASDAQ", "NYSE", "NYSE AMERICAN"})
TARGET_BAND = "large_and_mega"
LARGE_CAP_FLOOR_USD = 10_000_000_000.0
BOUNDARY_INVESTIGATION_FLOOR_USD = 9_500_000_000.0
CENSUS_POLICY_VERSION = "us_equity_census.v1"
MAX_PAID_MODEL_COST_USD = 100.0

OFFICIAL_SOURCE_SNAPSHOT_CONTRACTS = {
    "2026-07-17": {
        "sec_exchange_security_rows": 10_426,
        "nasdaq_listed_security_rows": 5_565,
        "other_listed_security_rows": 7_492,
        "companiesmarketcap_rows": 800,
        "companiesmarketcap_large_rows": 711,
        "stockanalysis_rows": 5_598,
        "stockanalysis_cap_rows": 5_597,
    }
}

STAGES = (
    "MEMBERSHIP",
    "IDENTITY",
    "DOMICILE_LISTING_STATUS",
    "SECURITY_TYPE",
    "PRIMARY_SECURITY_SELECTION",
    "ISSUER_DEDUPLICATION",
    "MARKET_CAP_RESOLUTION",
    "CAP_BAND_ASSIGNMENT",
    "SECTOR_CLASSIFICATION",
    "FACTS_FILINGS_AVAILABILITY",
    "COMPANY_PACKET",
    "TERMINAL_DISPOSITION",
)

SECURITY_COMMON = "COMMON_EQUITY"
SECURITY_COMMON_EQUIVALENT = "COMMON_EQUITY_EQUIVALENT"
SECURITY_ADR = "ADR_ADS"
SECURITY_ETF = "ETF"
SECURITY_ETN = "ETN"
SECURITY_CLOSED_END_FUND = "CLOSED_END_FUND"
SECURITY_PREFERRED = "PREFERRED_EQUITY"
SECURITY_WARRANT = "WARRANT"
SECURITY_RIGHT = "RIGHT"
SECURITY_UNIT = "COMPOSITE_UNIT"
SECURITY_DEBT = "DEBT_SECURITY"
SECURITY_UNKNOWN = "UNKNOWN_SECURITY_TYPE"

COMMON_SECURITY_TYPES = frozenset({SECURITY_COMMON, SECURITY_COMMON_EQUIVALENT, SECURITY_ADR})
ADMITTED_PRIMARY_SECURITY_TYPES = frozenset({SECURITY_COMMON, SECURITY_COMMON_EQUIVALENT})

CENSUS_SECTOR_PROMOTION_PRODUCER = "us_equity_census_issuer_deduplication"

US_JURISDICTION_CODES = frozenset(
    {
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "DC",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
        "PR",
        "VI",
        "GU",
        "AS",
        "MP",
    }
)

_WARRANT_RE = re.compile(r"\b(?:warrant|warrants)\b", re.IGNORECASE)
_RIGHT_RE = re.compile(r"\b(?:right|rights)\b", re.IGNORECASE)
_PREFERRED_RE = re.compile(
    r"\b(?:preferred|preference|depositary shares?.*preferred)\b",
    re.IGNORECASE,
)
_DEBT_RE = re.compile(
    r"\b(?:notes?|debentures?|bonds?|baby bonds?|debt securities?|obligations?)\b",
    re.IGNORECASE,
)
_ADR_RE = re.compile(
    r"\b(?:american depositary (?:share|shares|receipt|receipts)|american depository shares?|ADS|ADR)\b",
    re.IGNORECASE,
)
_COMMON_RE = re.compile(
    r"\b(?:common stock|common shares?|ordinary shares?|shares? of beneficial interest|capital stock)\b",
    re.IGNORECASE,
)
_PTP_COMMON_UNIT_RE = re.compile(
    r"\b(?:common units?|limited partnership units?|limited partner interests?|limited partnership interests?)\b",
    re.IGNORECASE,
)
_COMPOSITE_UNIT_RE = re.compile(
    r"\bunits?,? each (?:consisting|composed|comprised)\b", re.IGNORECASE
)
_CLOSED_END_RE = re.compile(
    r"\b(?:closed[- ]end|municipal (?:income|opportunity|value) fund|investment fund|equity fund|income fund|opportunities fund)\b",
    re.IGNORECASE,
)
_ETF_RE = re.compile(r"\b(?:ETF|exchange[- ]traded fund)\b", re.IGNORECASE)
_ETN_RE = re.compile(r"\b(?:ETN|exchange[- ]traded notes?)\b", re.IGNORECASE)
_UNIT_RE = re.compile(r"\bunits?\b", re.IGNORECASE)

TERMINAL_CAP_EVIDENCE_SCHEMA_VERSION = "US_EQUITY_TERMINAL_CAP_EVIDENCE_V1"
TERMINAL_BOUNDARY_CHECK = "TERMINAL_BOUNDARY_CHECK"
CAP_EVIDENCE_TIERS = (
    "LOCAL_AUTHORITATIVE",
    "SEC_EXCHANGE",
    "APPROVED_PROVIDER",
    "SEARCH_DIRECT_ISSUER",
)


@dataclass(frozen=True, slots=True)
class SecurityTypeResolution:
    security_type: str
    status: str
    reason_code: str
    is_common_equity: bool
    is_adr: bool


@dataclass(frozen=True, slots=True)
class IssuerProfile:
    cik: str
    legal_name: str
    sic: int | None
    sic_description: str | None
    entity_type: str | None
    domicile_status: str
    domicile_country_code: str | None
    domicile_jurisdiction: str | None
    operating_status: str
    filer_status: str | None
    latest_annual_filing_date: str | None
    latest_annual_filing_accession: str | None
    source_url: str
    forms: tuple[str, ...]
    domicile_method: str = "UNRESOLVED"
    domicile_confidence: str = "LOW"


@dataclass(frozen=True, slots=True)
class PrimarySelection:
    status: str
    method: str
    security_key: str | None
    ticker: str | None
    confidence: str
    detail: str


@dataclass(frozen=True, slots=True)
class CapEvidenceRow:
    source_provider: str
    source_url: str
    source_ticker: str
    source_name: str
    market_cap_usd: float
    as_of_date: str
    retrieved_at: str | None = None
    source_rank: int | None = None
    identity_match_method: str = "EXACT_TICKER"
    evidence_tier: str = "APPROVED_PROVIDER"
    resolution_role: str = "PRIMARY_CAP_SOURCE"
    evidence_detail: str | None = None


@dataclass(frozen=True, slots=True)
class CapResolution:
    status: str
    market_cap_usd: float | None
    cap_band: str | None
    method: str
    source_provider: str | None
    source_url: str | None
    as_of_date: str | None
    confidence: str
    discrepancy_code: str | None
    evidence: tuple[CapEvidenceRow, ...]


@dataclass(frozen=True, slots=True)
class CensusInputPaths:
    sec_registry: Path
    nasdaq_listed: Path
    other_listed: Path
    companiesmarketcap_pages: tuple[Path, ...] = ()
    stockanalysis_html: Path | None = None
    terminal_cap_evidence: Path | None = None
    submissions_dir: Path | None = None
    fixed_cohort_dir: Path | None = None


@dataclass(slots=True)
class CensusRunResult:
    run_id: str
    as_of_date: str
    status: str
    acceptance_status: str
    output_dir: Path
    counts: dict[str, int]
    acceptance_checks: dict[str, bool]
    artifacts: dict[str, Path]
    actual_llm_calls: int = 0
    actual_llm_cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["artifacts"] = {key: str(path) for key, path in self.artifacts.items()}
        return payload


@dataclass(slots=True)
class CensusCostLedger:
    """Explicit no-call ledger and hard guard for any future paid-model extension."""

    paid_model_authorized: bool = False
    actual_llm_calls: int = 0
    actual_llm_cost_usd: float = 0.0

    def record_paid_model_call(self, *, cost_usd: float) -> None:
        projected = self.actual_llm_cost_usd + float(cost_usd)
        if not self.paid_model_authorized:
            raise RuntimeError("paid model call requires explicit preflight authorization")
        if projected > MAX_PAID_MODEL_COST_USD:
            raise RuntimeError(
                f"paid model cost stop triggered above ${MAX_PAID_MODEL_COST_USD:.2f}"
            )
        self.actual_llm_calls += 1
        self.actual_llm_cost_usd = projected


@dataclass(slots=True)
class SubmissionFetchResult:
    as_of_date: str
    output_dir: Path
    candidate_cik_count: int
    existing_snapshot_count: int
    fetched_snapshot_count: int
    failed_snapshot_count: int
    deferred_snapshot_count: int
    unresolved_candidate_count: int
    attempts: list[dict[str, Any]]
    unresolved_candidates: list[dict[str, Any]]
    actual_llm_calls: int = 0
    actual_llm_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def normalize_ticker(value: object) -> str:
    """Normalize cross-source class separators while preserving raw symbols elsewhere."""

    return str(value or "").strip().upper().replace(".", "-").replace("/", "-")


def normalize_exchange(value: object) -> str:
    token = re.sub(r"\s+", " ", str(value or "").strip().upper())
    aliases = {
        "NASDAQ GLOBAL SELECT MARKET": "NASDAQ",
        "NASDAQ GLOBAL MARKET": "NASDAQ",
        "NASDAQ CAPITAL MARKET": "NASDAQ",
        "NASDAQ": "NASDAQ",
        "NYSE": "NYSE",
        "NYSE AMERICAN": "NYSE AMERICAN",
        "NYSEAMERICAN": "NYSE AMERICAN",
        "NYSE MKT": "NYSE AMERICAN",
        "AMEX": "NYSE AMERICAN",
    }
    return aliases.get(token, token)


def stable_security_key(*, cik: object, ticker: object, exchange: object) -> str:
    return ":".join(
        (
            str(cik or "").strip().lstrip("0") or "0",
            normalize_ticker(ticker),
            normalize_exchange(exchange).replace(" ", "_"),
        )
    )


def classify_security_type(
    *,
    listed_name: object,
    ticker: object,
    etf_flag: object = None,
    issue_type: object = None,
) -> SecurityTypeResolution:
    """Classify one listed security without issuer-name substring routing.

    The structured ETF flag wins.  Text rules describe the security instrument,
    not the issuer's business sector.  PTP common units and REIT beneficial
    interests are admitted as common-equity equivalents.
    """

    name = " ".join(
        token for token in (str(listed_name or "").strip(), str(issue_type or "").strip()) if token
    )
    symbol = normalize_ticker(ticker)
    if str(etf_flag or "").strip().upper() == "Y":
        return SecurityTypeResolution(SECURITY_ETF, "RESOLVED", "STRUCTURED_ETF_FLAG", False, False)
    if _ETN_RE.search(name):
        return SecurityTypeResolution(SECURITY_ETN, "RESOLVED", "LISTING_NAME_ETN", False, False)
    if _WARRANT_RE.search(name) or re.search(r"-W[ST]?$", symbol):
        return SecurityTypeResolution(
            SECURITY_WARRANT, "RESOLVED", "LISTING_NAME_OR_SYMBOL_WARRANT", False, False
        )
    if _RIGHT_RE.search(name) or re.search(r"-(?:R|RT)$", symbol):
        return SecurityTypeResolution(
            SECURITY_RIGHT, "RESOLVED", "LISTING_NAME_OR_SYMBOL_RIGHT", False, False
        )
    if _PREFERRED_RE.search(name) or re.search(r"-(?:P|PR)[A-Z0-9]*$", symbol):
        return SecurityTypeResolution(
            SECURITY_PREFERRED, "RESOLVED", "LISTING_NAME_OR_SYMBOL_PREFERRED", False, False
        )
    if _DEBT_RE.search(name):
        return SecurityTypeResolution(SECURITY_DEBT, "RESOLVED", "LISTING_NAME_DEBT", False, False)
    if _ETF_RE.search(name):
        return SecurityTypeResolution(SECURITY_ETF, "RESOLVED", "LISTING_NAME_ETF", False, False)
    if _CLOSED_END_RE.search(name):
        return SecurityTypeResolution(
            SECURITY_CLOSED_END_FUND,
            "RESOLVED",
            "LISTING_NAME_CLOSED_END_FUND",
            False,
            False,
        )
    if _COMPOSITE_UNIT_RE.search(name) or re.search(r"-U$", symbol):
        return SecurityTypeResolution(
            SECURITY_UNIT, "RESOLVED", "LISTING_NAME_OR_SYMBOL_COMPOSITE_UNIT", False, False
        )
    if _PTP_COMMON_UNIT_RE.search(name):
        return SecurityTypeResolution(
            SECURITY_COMMON_EQUIVALENT,
            "RESOLVED",
            "PTP_COMMON_UNIT",
            True,
            False,
        )
    if _UNIT_RE.search(name):
        return SecurityTypeResolution(
            SECURITY_UNIT,
            "RESOLVED",
            "LISTING_NAME_GENERIC_COMPOSITE_UNIT",
            False,
            False,
        )
    if _ADR_RE.search(name):
        return SecurityTypeResolution(SECURITY_ADR, "RESOLVED", "LISTING_NAME_ADR_ADS", True, True)
    if _COMMON_RE.search(name):
        return SecurityTypeResolution(
            SECURITY_COMMON, "RESOLVED", "LISTING_NAME_COMMON_EQUITY", True, False
        )
    if str(etf_flag or "").strip().upper() == "N" and name:
        return SecurityTypeResolution(
            SECURITY_COMMON,
            "RESOLVED",
            "STRUCTURED_NON_ETF_PLAIN_EQUITY_LISTING",
            True,
            False,
        )
    return SecurityTypeResolution(
        SECURITY_UNKNOWN,
        "NEEDS_DATA",
        "SECURITY_TYPE_UNRESOLVED",
        False,
        False,
    )


def _recent_filings(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    filings = payload.get("filings")
    recent = filings.get("recent") if isinstance(filings, Mapping) else None
    if not isinstance(recent, Mapping):
        return []
    forms = list(recent.get("form") or [])
    dates = list(recent.get("filingDate") or [])
    accessions = list(recent.get("accessionNumber") or [])
    count = max(len(forms), len(dates), len(accessions))
    return [
        {
            "form": str(forms[idx] if idx < len(forms) else "").strip().upper(),
            "filing_date": str(dates[idx] if idx < len(dates) else "").strip(),
            "accession": str(accessions[idx] if idx < len(accessions) else "").strip(),
        }
        for idx in range(count)
    ]


def parse_submissions_profile(
    payload: Mapping[str, Any], *, source_url: str, fallback_cik: str = ""
) -> IssuerProfile:
    """Extract domicile, filer, operating, and filing evidence from SEC submissions."""

    recent = _recent_filings(payload)
    forms = tuple(row["form"] for row in recent if row["form"])
    has_domestic_forms = any(
        form.startswith(prefix)
        for form in forms
        for prefix in ("10-K", "10-Q", "8-K", "S-1", "S-3", "S-4")
    )
    has_foreign_forms = any(
        form.startswith(prefix)
        for form in forms
        for prefix in ("20-F", "40-F", "6-K", "F-1", "F-3", "F-4")
    )
    has_fund_forms = any(
        form.startswith(prefix)
        for form in forms
        for prefix in ("N-CSR", "N-CSRS", "N-CEN", "N-PORT", "N-Q")
    )
    annual_status_rows = [
        row for row in recent if row["form"].startswith(("10-K", "20-F", "40-F", "N-CSR", "N-CSRS"))
    ]
    latest_annual_status = max(annual_status_rows, key=lambda row: row["filing_date"], default=None)
    latest_annual_form = latest_annual_status["form"] if latest_annual_status else ""
    domestic_filer_evidence = (
        latest_annual_form.startswith("10-K")
        if latest_annual_form
        else has_domestic_forms and not has_foreign_forms
    )
    foreign_filer_evidence = (
        latest_annual_form.startswith(("20-F", "40-F"))
        if latest_annual_form
        else has_foreign_forms and not has_domestic_forms
    )
    fund_filer_evidence = (
        latest_annual_form.startswith(("N-CSR", "N-CSRS"))
        if latest_annual_form
        else has_fund_forms and not has_domestic_forms and not has_foreign_forms
    )
    state = str(payload.get("stateOfIncorporation") or "").strip().upper()
    state_description = str(payload.get("stateOfIncorporationDescription") or "").strip()
    addresses = payload.get("addresses")
    business = addresses.get("business") if isinstance(addresses, Mapping) else None
    business = business if isinstance(business, Mapping) else {}
    mailing = addresses.get("mailing") if isinstance(addresses, Mapping) else None
    mailing = mailing if isinstance(mailing, Mapping) else {}
    address_fields = (
        "stateOrCountry",
        "stateOrCountryDescription",
        "country",
        "countryCode",
        "foreignStateTerritory",
        "city",
    )
    if not any(business.get(field) for field in address_fields) and any(
        mailing.get(field) for field in address_fields
    ):
        business = mailing
        address_method = "SEC_MAILING_ADDRESS"
    else:
        address_method = "SEC_PRINCIPAL_BUSINESS_ADDRESS"
    business_state = str(business.get("stateOrCountry") or "").strip().upper()
    business_description = str(
        business.get("stateOrCountryDescription") or business.get("country") or ""
    ).strip()
    business_country_code = str(business.get("countryCode") or "").strip().upper()
    raw_foreign_location = business.get("isForeignLocation")
    is_foreign_location = raw_foreign_location is True or str(raw_foreign_location).lower() in {
        "1",
        "true",
    }
    business_is_us = business_state in US_JURISDICTION_CODES or business_country_code in {
        "US",
        "USA",
    }
    if foreign_filer_evidence:
        # Current foreign annual-report evidence is dispositive. SEC submissions
        # can carry values such as DC/DE in stateOfIncorporation for foreign
        # private issuers, so that field must not override a current 20-F/40-F.
        domicile_status = "FOREIGN_DOMICILED"
        country = None if business_is_us else business_country_code or business_state or None
        state = business_state
        state_description = business_description or "Foreign private issuer filing forms"
        domicile_method = "SEC_FOREIGN_PRIVATE_ISSUER_FORMS"
        domicile_confidence = "HIGH"
    elif state in US_JURISDICTION_CODES:
        domicile_status = "US_DOMICILED"
        country = "US"
        domicile_method = "SEC_STATE_OF_INCORPORATION"
        domicile_confidence = "HIGH"
    elif state:
        domicile_status = "FOREIGN_DOMICILED"
        country = state
        domicile_method = "SEC_STATE_OF_INCORPORATION"
        domicile_confidence = "HIGH"
    elif business_is_us and not is_foreign_location and domestic_filer_evidence:
        domicile_status = "US_DOMICILED"
        country = "US"
        state = business_state
        state_description = business_description or business_state
        domicile_method = address_method
        domicile_confidence = "MEDIUM"
    elif (
        is_foreign_location
        or business_country_code not in {"", "US", "USA"}
        or (business_state and business_state not in US_JURISDICTION_CODES)
    ):
        domicile_status = "FOREIGN_DOMICILED"
        country = business_country_code or business_state or None
        state = business_state
        state_description = business_description or business_state
        domicile_method = address_method
        domicile_confidence = "MEDIUM"
    else:
        domicile_status = "NEEDS_DATA_DOMICILE"
        country = None
        domicile_method = "UNRESOLVED"
        domicile_confidence = "LOW"

    sic_raw = str(payload.get("sic") or payload.get("sicCode") or "").strip()
    sic = int(sic_raw) if sic_raw.isdigit() else None
    entity_type = str(payload.get("entityType") or "").strip().lower()
    if sic == 6770:
        operating_status = "SHELL_OR_BLANK_CHECK"
    elif domestic_filer_evidence:
        # This intentionally preserves BDCs that use investment-company SICs
        # but file operating-company 10-K/10-Q reports.
        operating_status = "OPERATING"
    elif foreign_filer_evidence:
        operating_status = "FOREIGN_FILER"
    elif fund_filer_evidence:
        operating_status = "INVESTMENT_COMPANY"
    elif entity_type == "operating":
        operating_status = "OPERATING"
    else:
        operating_status = "NEEDS_DATA_OPERATING_STATUS"

    annual_rows = [
        row for row in recent if row["form"].startswith(("10-K", "20-F", "40-F", "N-CSR"))
    ]
    latest_annual = max(annual_rows, key=lambda row: row["filing_date"], default=None)
    return IssuerProfile(
        cik=str(payload.get("cik") or fallback_cik).strip().lstrip("0") or "0",
        legal_name=str(payload.get("name") or "").strip(),
        sic=sic,
        sic_description=str(payload.get("sicDescription") or "").strip() or None,
        entity_type=str(payload.get("entityType") or "").strip() or None,
        domicile_status=domicile_status,
        domicile_country_code=country,
        domicile_jurisdiction=state_description or state or None,
        operating_status=operating_status,
        filer_status=str(payload.get("category") or "").strip() or None,
        latest_annual_filing_date=latest_annual["filing_date"] if latest_annual else None,
        latest_annual_filing_accession=latest_annual["accession"] if latest_annual else None,
        source_url=source_url,
        forms=forms,
        domicile_method=domicile_method,
        domicile_confidence=domicile_confidence,
    )


def _apply_operating_bdc_security_override(
    profile: IssuerProfile | None, securities: Sequence[dict[str, Any]]
) -> int:
    """Preserve exchange-labeled closed-end BDC common stock when SEC forms prove operation."""

    if profile is None or profile.operating_status != "OPERATING":
        return 0
    has_bdc_registration = any(form.startswith("N-2") for form in profile.forms)
    has_operating_annual = any(form.startswith("10-K") for form in profile.forms)
    if not (has_bdc_registration and has_operating_annual):
        return 0
    changed = 0
    for security in securities:
        if (
            security.get("security_type") != SECURITY_CLOSED_END_FUND
            or security.get("listing_status") != "ACTIVE"
            or normalize_exchange(security.get("exchange_name")) not in TARGET_EXCHANGES
        ):
            continue
        security["security_type"] = SECURITY_COMMON_EQUIVALENT
        security["security_type_status"] = "RESOLVED"
        security["is_common_equity"] = 1
        provenance = json.loads(str(security.get("provenance_json") or "{}"))
        provenance["security_type_reason_code"] = "SEC_OPERATING_BDC_COMMON_EQUITY"
        provenance["bdc_evidence_forms"] = ["10-K", "N-2"]
        security["provenance_json"] = _json_dumps(provenance)
        changed += 1
    return changed


def select_primary_security(
    securities: Sequence[Mapping[str, Any]],
    *,
    preferred_source_tickers: Sequence[str] = (),
    legacy_primary_ticker: str | None = None,
) -> PrimarySelection:
    """Select one primary common equity only when evidence makes it unambiguous."""

    candidates = [
        row
        for row in securities
        if normalize_exchange(row.get("exchange_name")) in TARGET_EXCHANGES
        and str(row.get("security_type") or "") in COMMON_SECURITY_TYPES
        and str(row.get("listing_status") or "ACTIVE").upper() == "ACTIVE"
    ]
    if not candidates:
        return PrimarySelection(
            "NEEDS_DATA",
            "NO_ACTIVE_IN_SCOPE_COMMON_EQUITY",
            None,
            None,
            "LOW",
            "No active common-equity security was resolved on a target exchange.",
        )
    if len(candidates) == 1:
        row = candidates[0]
        return PrimarySelection(
            "RESOLVED",
            "SOLE_ACTIVE_IN_SCOPE_COMMON_EQUITY",
            str(row["security_key"]),
            str(row["ticker"]),
            "HIGH",
            "Only one active in-scope common-equity candidate exists.",
        )

    by_ticker = {normalize_ticker(row.get("ticker")): row for row in candidates}
    source_matches = []
    for ticker in preferred_source_tickers:
        normalized = normalize_ticker(ticker)
        if normalized in by_ticker and normalized not in source_matches:
            source_matches.append(normalized)
    if len(source_matches) == 1:
        row = by_ticker[source_matches[0]]
        return PrimarySelection(
            "RESOLVED",
            "DIRECT_CAP_SOURCE_SECURITY_MATCH",
            str(row["security_key"]),
            str(row["ticker"]),
            "MEDIUM",
            "Exactly one common-equity candidate matched a direct cap source ticker.",
        )
    if legacy_primary_ticker and normalize_ticker(legacy_primary_ticker) in by_ticker:
        row = by_ticker[normalize_ticker(legacy_primary_ticker)]
        return PrimarySelection(
            "RESOLVED",
            "LEGACY_SEC_REGISTRANT_PRIMARY_CORROBORATION",
            str(row["security_key"]),
            str(row["ticker"]),
            "MEDIUM",
            "Legacy registry primary was retained with all share classes preserved.",
        )
    return PrimarySelection(
        "NEEDS_DATA",
        "MULTIPLE_COMMON_CLASSES_PRIMARY_UNRESOLVED",
        None,
        None,
        "LOW",
        f"{len(candidates)} active common-equity candidates remain without unique primary evidence.",
    )


def resolve_direct_cap_evidence(
    evidence: Sequence[CapEvidenceRow], *, primary_ticker: str | None
) -> CapResolution:
    """Resolve issuer cap by the explicit source hierarchy without summing classes.

    A lower-tier terminal search may resolve a band conflict between approved
    providers only when at least two independently named terminal sources agree
    on the same side of the boundary.  All conflicting evidence remains in the
    returned derivation ledger.
    """

    usable = [row for row in evidence if row.market_cap_usd > 0]
    if not usable:
        return CapResolution(
            "NEEDS_DATA",
            None,
            None,
            "EXPLICIT_UNRESOLVED",
            None,
            None,
            None,
            "LOW",
            "MARKET_CAP_UNRESOLVED",
            (),
        )

    primary_norm = normalize_ticker(primary_ticker)
    invalid_tiers = sorted({row.evidence_tier for row in usable} - set(CAP_EVIDENCE_TIERS))
    if invalid_tiers:
        raise ValueError(f"unknown market-cap evidence tiers: {invalid_tiers}")

    by_provider: dict[str, list[CapEvidenceRow]] = defaultdict(list)
    for row in usable:
        by_provider[row.source_provider].append(row)
    selected: list[CapEvidenceRow] = []
    for provider in sorted(by_provider):
        rows = by_provider[provider]
        primary_rows = [row for row in rows if normalize_ticker(row.source_ticker) == primary_norm]
        chosen = (
            primary_rows[0]
            if len(primary_rows) == 1
            else sorted(
                rows,
                key=lambda row: (
                    row.source_rank if row.source_rank is not None else 10**9,
                    normalize_ticker(row.source_ticker),
                ),
            )[0]
        )
        selected.append(chosen)

    selected.sort(
        key=lambda row: (
            CAP_EVIDENCE_TIERS.index(row.evidence_tier),
            row.as_of_date,
            row.source_provider,
        )
    )

    def band_set(rows: Sequence[CapEvidenceRow]) -> set[str]:
        return {band_for_market_cap(row.market_cap_usd / 1_000_000.0) for row in rows}

    def resolved(
        rows: Sequence[CapEvidenceRow],
        *,
        method: str,
        discrepancy: str | None = None,
        forced_confidence: str | None = None,
    ) -> CapResolution:
        ordered = sorted(rows, key=lambda row: (row.as_of_date, row.source_provider), reverse=True)
        chosen = ordered[0]
        values = [row.market_cap_usd for row in ordered]
        high = max(values)
        low = min(values)
        spread = (high - low) / high if high > 0 else 0.0
        confidence = forced_confidence
        if confidence is None:
            if chosen.evidence_tier in {"LOCAL_AUTHORITATIVE", "SEC_EXCHANGE"}:
                confidence = "HIGH"
            else:
                confidence = "HIGH" if len(ordered) >= 2 and spread <= 0.05 else "MEDIUM"
        if discrepancy is None and len(ordered) >= 2 and spread > 0.15:
            confidence = "LOW"
            discrepancy = "MARKET_CAP_CROSS_SOURCE_SPREAD_GT_15PCT"
        return CapResolution(
            "RESOLVED",
            chosen.market_cap_usd,
            band_for_market_cap(chosen.market_cap_usd / 1_000_000.0),
            method,
            chosen.source_provider,
            chosen.source_url,
            chosen.as_of_date,
            confidence,
            discrepancy,
            tuple(selected),
        )

    method_prefix = {
        "LOCAL_AUTHORITATIVE": "LOCAL_AUTHORITATIVE_MARKET_CAP",
        "SEC_EXCHANGE": "SEC_EXCHANGE_MARKET_CAP",
        "APPROVED_PROVIDER": "APPROVED_PROVIDER_DIRECT_ISSUER_CAP",
        "SEARCH_DIRECT_ISSUER": "TERMINAL_SEARCH_DIRECT_ISSUER_CAP",
    }
    for tier in CAP_EVIDENCE_TIERS:
        tier_rows = [row for row in selected if row.evidence_tier == tier]
        if not tier_rows:
            continue
        bands = band_set(tier_rows)
        if len(bands) == 1:
            suffix = "_CONSENSUS" if len(tier_rows) >= 2 else ""
            lower_rows = [
                row
                for row in selected
                if CAP_EVIDENCE_TIERS.index(row.evidence_tier) > CAP_EVIDENCE_TIERS.index(tier)
            ]
            discrepancy = (
                "LOWER_TIER_MARKET_CAP_BAND_DISAGREEMENT"
                if lower_rows and not band_set(lower_rows) <= bands
                else None
            )
            return resolved(
                tier_rows,
                method=f"{method_prefix[tier]}{suffix}",
                discrepancy=discrepancy,
            )

        if tier != "APPROVED_PROVIDER":
            return CapResolution(
                "NEEDS_DATA",
                None,
                None,
                "CROSS_SOURCE_BAND_CONFLICT",
                None,
                None,
                None,
                "LOW",
                "MARKET_CAP_BAND_CONFLICT",
                tuple(selected),
            )

        terminal_rows = [
            row
            for row in selected
            if row.evidence_tier == "SEARCH_DIRECT_ISSUER"
            and row.resolution_role == TERMINAL_BOUNDARY_CHECK
        ]
        if len(terminal_rows) >= 2 and len(band_set(terminal_rows)) == 1:
            return resolved(
                terminal_rows,
                method="TERMINAL_SEARCH_DIRECT_ISSUER_CAP_CONSENSUS",
                discrepancy="MARKET_CAP_BAND_CONFLICT_RESOLVED_BY_TERMINAL_SEARCH",
                forced_confidence="MEDIUM",
            )
        return CapResolution(
            "NEEDS_DATA",
            None,
            None,
            "CROSS_SOURCE_BAND_CONFLICT",
            None,
            None,
            None,
            "LOW",
            "MARKET_CAP_BAND_CONFLICT",
            tuple(selected),
        )

    raise AssertionError("usable cap evidence did not enter a configured evidence tier")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_acquired_at(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _file_acquisition_date(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().date().isoformat()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        is not None
    )


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_terminal_cap_evidence_snapshot(
    path: Path, *, census_as_of_date: str
) -> tuple[tuple[CapEvidenceRow, ...], tuple[ParseIssue, ...]]:
    """Parse a dated human-reviewed boundary ledger with fail-visible issues."""

    payload = _safe_json(path)
    issues: list[ParseIssue] = []
    if payload.get("schema_version") != TERMINAL_CAP_EVIDENCE_SCHEMA_VERSION:
        return (), (
            ParseIssue(
                source=str(path),
                code="INVALID_TERMINAL_CAP_SCHEMA_VERSION",
                message=(
                    f"expected {TERMINAL_CAP_EVIDENCE_SCHEMA_VERSION}, got "
                    f"{payload.get('schema_version')!r}"
                ),
            ),
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return (), (
            ParseIssue(
                source=str(path),
                code="INVALID_TERMINAL_CAP_ROWS",
                message="terminal market-cap evidence rows must be a JSON array",
            ),
        )
    census_date = date.fromisoformat(census_as_of_date)
    parsed: list[CapEvidenceRow] = []
    for row_number, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            issues.append(
                ParseIssue(
                    source=str(path),
                    code="INVALID_TERMINAL_CAP_ROW",
                    message="terminal market-cap row must be an object",
                    row_number=row_number,
                    raw_record=raw,
                )
            )
            continue
        try:
            provider = str(raw["source_provider"]).strip()
            source_url = str(raw["source_url"]).strip()
            ticker = normalize_ticker(raw["source_ticker"])
            name = str(raw["source_name"]).strip()
            cap = float(raw["market_cap_usd"])
            evidence_date = date.fromisoformat(str(raw["evidence_as_of_date"]))
            role = str(raw.get("resolution_role") or "").strip()
            if not provider or not source_url.startswith(("https://", "http://")):
                raise ValueError("source_provider and an HTTP(S) source_url are required")
            if not ticker or not name or cap <= 0:
                raise ValueError(
                    "source_ticker, source_name, and positive market_cap_usd are required"
                )
            if role != TERMINAL_BOUNDARY_CHECK:
                raise ValueError(f"resolution_role must be {TERMINAL_BOUNDARY_CHECK}")
            age_days = (census_date - evidence_date).days
            if age_days < 0 or age_days > 7:
                raise ValueError(
                    "terminal evidence must be on/before and within 7 days of census as-of"
                )
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(
                ParseIssue(
                    source=str(path),
                    code="INVALID_TERMINAL_CAP_ROW",
                    message=str(exc),
                    row_number=row_number,
                    raw_record=dict(raw),
                )
            )
            continue
        parsed.append(
            CapEvidenceRow(
                source_provider=provider,
                source_url=source_url,
                source_ticker=ticker,
                source_name=name,
                market_cap_usd=cap,
                as_of_date=evidence_date.isoformat(),
                retrieved_at=str(raw.get("retrieved_at") or "").strip() or None,
                identity_match_method="EXACT_TICKER",
                evidence_tier="SEARCH_DIRECT_ISSUER",
                resolution_role=role,
                evidence_detail=str(raw.get("evidence_detail") or "").strip() or None,
            )
        )
    return tuple(parsed), tuple(issues)


def _submission_snapshot_files(inputs: CensusInputPaths) -> list[Path]:
    if inputs.submissions_dir is None or not inputs.submissions_dir.exists():
        return []
    return sorted(
        path
        for path in inputs.submissions_dir.glob("*.json")
        if re.fullmatch(r"[0-9]{10}\.json", path.name)
    )


def _source_role_files(inputs: CensusInputPaths) -> list[tuple[str, Path]]:
    paths = [
        ("company_tickers_exchange.json", inputs.sec_registry),
        ("nasdaqlisted.txt", inputs.nasdaq_listed),
        ("otherlisted.txt", inputs.other_listed),
    ]
    paths.extend(
        (f"companiesmarketcap/page_{page_number:03d}.html", path)
        for page_number, path in enumerate(inputs.companiesmarketcap_pages, start=1)
    )
    if inputs.stockanalysis_html is not None:
        paths.append(("stockanalysis/screener.html", inputs.stockanalysis_html))
    if inputs.terminal_cap_evidence is not None:
        paths.append(("terminal_cap_evidence.json", inputs.terminal_cap_evidence))
    paths.extend(
        (f"sec_submissions/{path.name}", path) for path in _submission_snapshot_files(inputs)
    )
    if inputs.fixed_cohort_dir is not None and inputs.fixed_cohort_dir.exists():
        paths.extend(
            (
                f"fixed_cohort/{path.relative_to(inputs.fixed_cohort_dir).as_posix()}",
                path,
            )
            for path in sorted(inputs.fixed_cohort_dir.glob("*/autonomous_sector_run.json"))
        )
    return paths


def _census_policy_manifest() -> list[dict[str, str]]:
    app_root = Path(__file__).parents[1]
    policy_paths = (
        Path(__file__),
        Path(__file__).with_name("us_equity_census_sources.py"),
        app_root / "sector" / "canonical_taxonomy.py",
        app_root / "sector" / "external_industry_taxonomy.py",
        app_root / "autonomous" / "cap_resolver.py",
        app_root / "db.py",
    )
    return [
        {
            "name": path.relative_to(app_root.parent).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in policy_paths
    ]


def census_input_fingerprint(
    inputs: CensusInputPaths,
    *,
    as_of_date: str,
    database_inputs: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "as_of_date": as_of_date,
        "census_policy_version": CENSUS_POLICY_VERSION,
        "policy_files": _census_policy_manifest(),
        "sources": [
            {
                "role": role,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "acquired_at": _file_acquired_at(path),
            }
            for role, path in _source_role_files(inputs)
        ],
        "taxonomy_version": CANONICAL_SECTOR_TAXONOMY_VERSION,
        "taxonomy_hash": CANONICAL_SECTOR_TAXONOMY_HASH,
        "external_industry_taxonomy_version": STOCKANALYSIS_INDUSTRY_TAXONOMY_VERSION,
        "external_industry_taxonomy_hash": STOCKANALYSIS_INDUSTRY_TAXONOMY_HASH,
        "database_input_sha256": (
            hashlib.sha256(_json_dumps(database_inputs).encode("utf-8")).hexdigest()
            if database_inputs is not None
            else None
        ),
    }
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _source_manifest(inputs: CensusInputPaths, *, as_of_date: str) -> list[dict[str, Any]]:
    source_rows = [
        {
            "artifact_kind": "SOURCE_SNAPSHOT",
            "file_name": role,
            "source_path": str(path),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "acquired_at": _file_acquired_at(path),
            "source_acquisition_date": _file_acquisition_date(path),
            "census_as_of_date": as_of_date,
        }
        for role, path in _source_role_files(inputs)
    ]
    policy_rows = [
        {
            "artifact_kind": "POLICY_CODE",
            "file_name": row["name"],
            "source_path": str(Path(__file__).parents[2] / row["name"]),
            "sha256": row["sha256"],
            "as_of_date": as_of_date,
        }
        for row in _census_policy_manifest()
    ]
    return [*source_rows, *policy_rows]


def _begin_attempt(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    stage: str,
    input_fingerprint: str,
) -> str:
    row = conn.execute(
        "SELECT COALESCE(MAX(attempt_number), 0) AS n FROM us_equity_census_attempts "
        "WHERE run_id = ? AND stage = ?",
        (run_id, stage),
    ).fetchone()
    number = int(row["n"] if row else 0) + 1
    attempt_id = f"{run_id}:{stage}:{number}"
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO us_equity_census_attempts(
            run_id, attempt_id, attempt_number, stage, entity_kind, status,
            input_fingerprint, input_json, output_json, detail_json,
            started_at, heartbeat_at, created_at, updated_at
        ) VALUES(?, ?, ?, ?, 'RUN', 'RUNNING', ?, '{}', '{}', '{}', ?, ?, ?, ?)
        """,
        (run_id, attempt_id, number, stage, input_fingerprint, now, now, now, now),
    )
    conn.execute(
        "UPDATE us_equity_census_runs SET current_stage = ?, status = 'RUNNING', "
        "updated_at = ? WHERE run_id = ?",
        (stage, now, run_id),
    )
    conn.commit()
    return attempt_id


def _finish_attempt(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    attempt_id: str,
    stage: str,
    output: Mapping[str, Any],
) -> None:
    now = utc_now_iso()
    conn.execute(
        "UPDATE us_equity_census_attempts SET status = 'COMPLETED', output_json = ?, "
        "heartbeat_at = ?, finished_at = ?, updated_at = ? "
        "WHERE run_id = ? AND attempt_id = ?",
        (_json_dumps(output), now, now, now, run_id, attempt_id),
    )
    conn.execute(
        "UPDATE us_equity_census_runs SET current_stage = ?, updated_at = ? WHERE run_id = ?",
        (stage, now, run_id),
    )
    conn.commit()


def _fail_attempt(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    attempt_id: str,
    error: Exception,
) -> None:
    now = utc_now_iso()
    conn.execute(
        "UPDATE us_equity_census_attempts SET status = 'FAILED', error_code = ?, "
        "error_message = ?, heartbeat_at = ?, finished_at = ?, updated_at = ? "
        "WHERE run_id = ? AND attempt_id = ?",
        (type(error).__name__, str(error), now, now, now, run_id, attempt_id),
    )
    conn.execute(
        "UPDATE us_equity_census_runs SET status = 'INTERRUPTED', interrupted_at = ?, "
        "updated_at = ?, status_detail_json = ? WHERE run_id = ?",
        (now, now, _json_dumps({"error": str(error)}), run_id),
    )
    conn.commit()


def _load_latest_sector_labels(
    conn: sqlite3.Connection, *, as_of_date: str
) -> dict[str, dict[str, Any]]:
    if not _table_exists(conn, "sector_inference"):
        return {}
    rows = conn.execute(
        """
        SELECT ticker, inferred_sector, as_of_date, derived_from, score
        FROM (
            SELECT ticker, inferred_sector, as_of_date, derived_from, score,
                   ROW_NUMBER() OVER (
                       PARTITION BY UPPER(ticker)
                       ORDER BY as_of_date DESC, id DESC
                   ) AS rn
            FROM sector_inference
            WHERE inferred_sector IS NOT NULL AND TRIM(inferred_sector) != ''
              AND as_of_date <= ?
        )
        WHERE rn = 1
        """,
        (as_of_date,),
    ).fetchall()
    return {
        normalize_ticker(row["ticker"]): {
            "label": str(row["inferred_sector"]),
            "as_of_date": str(row["as_of_date"]),
            "derived_from": str(row["derived_from"] or ""),
            "score": row["score"],
        }
        for row in rows
    }


def _legacy_registrants(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    if not _table_exists(conn, "sec_registrants"):
        return {}
    rows = conn.execute("SELECT * FROM sec_registrants").fetchall()
    return {str(row["cik"] or "").strip().lstrip("0") or "0": dict(row) for row in rows}


def _coverage_for_ticker(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    normalized = normalize_ticker(ticker)
    facts = {"count": 0, "latest_period_end": None}
    if _table_exists(conn, "companyfacts_facts"):
        row = conn.execute(
            """
            SELECT COUNT(*) AS n, MAX(period_end) AS latest
            FROM companyfacts_facts
            WHERE UPPER(ticker) = ? AND period_type = 'FY'
            """,
            (normalized,),
        ).fetchone()
        facts = {"count": int(row["n"] or 0), "latest_period_end": row["latest"]}
    filings = {"count": 0, "latest_filing_date": None, "latest_accession": None}
    if _table_exists(conn, "filings"):
        row = conn.execute(
            """
            SELECT COUNT(*) AS n, MAX(filing_date) AS latest
            FROM filings
            WHERE UPPER(ticker) = ? AND status IN ('OK', 'parsed')
            """,
            (normalized,),
        ).fetchone()
        latest = row["latest"]
        accession = None
        if latest:
            accession_row = conn.execute(
                "SELECT accession FROM filings WHERE UPPER(ticker) = ? AND filing_date = ? "
                "AND status IN ('OK', 'parsed') ORDER BY accession DESC LIMIT 1",
                (normalized, latest),
            ).fetchone()
            accession = accession_row["accession"] if accession_row else None
        filings = {
            "count": int(row["n"] or 0),
            "latest_filing_date": latest,
            "latest_accession": accession,
        }
    packet = None
    if _table_exists(conn, "evidence_packets"):
        packet_row = conn.execute(
            "SELECT packet_path, packet_hash, as_of_date FROM evidence_packets "
            "WHERE UPPER(ticker) = ? ORDER BY as_of_date DESC LIMIT 1",
            (normalized,),
        ).fetchone()
        packet = dict(packet_row) if packet_row else None
    return {"facts": facts, "filings": filings, "packet": packet}


def _load_coverage_by_ticker(
    conn: sqlite3.Connection, *, as_of_date: str
) -> dict[str, dict[str, Any]]:
    """Load facts, filings, and packet availability with one scan per table."""

    coverage: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "facts": {"count": 0, "latest_period_end": None},
            "filings": {
                "count": 0,
                "latest_filing_date": None,
                "latest_accession": None,
            },
            "packet": None,
        }
    )
    if _table_exists(conn, "companyfacts_facts"):
        for row in conn.execute(
            """
            SELECT UPPER(ticker) AS ticker, COUNT(*) AS n, MAX(period_end) AS latest
            FROM companyfacts_facts
            WHERE period_type = 'FY' AND period_end <= ?
            GROUP BY UPPER(ticker)
            """,
            (as_of_date,),
        ):
            coverage[normalize_ticker(row["ticker"])]["facts"] = {
                "count": int(row["n"] or 0),
                "latest_period_end": row["latest"],
            }
    if _table_exists(conn, "filings"):
        for row in conn.execute(
            """
            WITH ranked AS (
                SELECT UPPER(ticker) AS ticker, accession, filing_date,
                       COUNT(*) OVER (PARTITION BY UPPER(ticker)) AS n,
                       ROW_NUMBER() OVER (
                           PARTITION BY UPPER(ticker)
                           ORDER BY filing_date DESC, accession DESC
                       ) AS rn
                FROM filings
                WHERE status IN ('OK', 'parsed') AND ticker IS NOT NULL
                  AND (filing_date IS NULL OR filing_date <= ?)
            )
            SELECT ticker, n, filing_date, accession
            FROM ranked
            WHERE rn = 1
            """,
            (as_of_date,),
        ):
            coverage[normalize_ticker(row["ticker"])]["filings"] = {
                "count": int(row["n"] or 0),
                "latest_filing_date": row["filing_date"],
                "latest_accession": row["accession"],
            }
    if _table_exists(conn, "evidence_packets"):
        for row in conn.execute(
            """
            WITH ranked AS (
                SELECT UPPER(ticker) AS ticker, packet_path, packet_hash, as_of_date,
                       ROW_NUMBER() OVER (
                           PARTITION BY UPPER(ticker)
                           ORDER BY as_of_date DESC, id DESC
                       ) AS rn
                FROM evidence_packets
                WHERE as_of_date <= ?
            )
            SELECT ticker, packet_path, packet_hash, as_of_date
            FROM ranked
            WHERE rn = 1
            """,
            (as_of_date,),
        ):
            coverage[normalize_ticker(row["ticker"])]["packet"] = {
                "packet_path": row["packet_path"],
                "packet_hash": row["packet_hash"],
                "as_of_date": row["as_of_date"],
            }
    return dict(coverage)


def _database_input_snapshot(conn: sqlite3.Connection, *, as_of_date: str) -> dict[str, Any]:
    """Freeze every live database projection consumed by one census run."""

    legacy = _legacy_registrants(conn)
    registrants = {
        cik: {
            "primary_ticker": row.get("primary_ticker"),
            "operating_status": row.get("operating_status"),
            "latest_operating_form_date": row.get("latest_operating_form_date"),
        }
        for cik, row in sorted(legacy.items())
    }
    sectors = _load_latest_sector_labels(conn, as_of_date=as_of_date)
    coverage = _load_coverage_by_ticker(conn, as_of_date=as_of_date)
    return {
        "schema_version": "US_EQUITY_CENSUS_DATABASE_INPUTS_V1",
        "as_of_date": as_of_date,
        "sec_registrants": registrants,
        "latest_sector_labels": {key: sectors[key] for key in sorted(sectors)},
        "coverage_by_ticker": {key: coverage[key] for key in sorted(coverage)},
        "query_contracts": {
            "companyfacts_facts": "period_type=FY AND period_end<=as_of",
            "filings": "status IN (OK,parsed) AND filing_date<=as_of",
            "evidence_packets": "latest as_of_date<=census as_of",
            "sector_inference": "latest label with as_of_date<=census as_of",
        },
    }


def _write_database_input_snapshot(
    database_inputs: Mapping[str, Any], *, output_dir: Path
) -> dict[str, Any]:
    path = output_dir / "source_snapshots" / "database_inputs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(database_inputs, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "source_path": "sqlite:data/engine.db:as_of_bounded_projections",
        "snapshot_path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _load_submission_profile(inputs: CensusInputPaths, cik: str) -> IssuerProfile | None:
    if inputs.submissions_dir is None:
        return None
    normalized = str(cik).zfill(10)
    candidates = (
        inputs.submissions_dir / f"{normalized}.json",
        inputs.submissions_dir / f"CIK{normalized}.json",
    )
    for path in candidates:
        payload = _safe_json(path)
        if payload:
            return parse_submissions_profile(
                payload,
                fallback_cik=cik,
                source_url=f"https://data.sec.gov/submissions/CIK{normalized}.json",
            )
    return None


def _parse_fixed_cohort(inputs: CensusInputPaths) -> set[str]:
    root = inputs.fixed_cohort_dir
    if root is None or not root.exists():
        return set()
    tickers: set[str] = set()
    for path in sorted(root.glob("*/autonomous_sector_run.json")):
        payload = _safe_json(path)
        selection = payload.get("candidate_selection")
        if isinstance(selection, Mapping):
            for ticker in selection.get("loaded_tickers") or []:
                normalized = normalize_ticker(ticker)
                if normalized:
                    tickers.add(normalized)
    return tickers


def _copy_source_snapshots(inputs: CensusInputPaths, *, output_dir: Path) -> list[dict[str, Any]]:
    snapshot_dir = output_dir / "source_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for role, path in _source_role_files(inputs):
        target = snapshot_dir / role
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or _sha256_file(target) != _sha256_file(path):
            shutil.copy2(path, target)
        copied.append(
            {
                "artifact_kind": "SOURCE_SNAPSHOT",
                "logical_role": role,
                "source_path": str(path),
                "snapshot_path": str(target),
                "sha256": _sha256_file(target),
                "bytes": target.stat().st_size,
                "acquired_at": _file_acquired_at(path),
                "source_acquisition_date": _file_acquisition_date(path),
            }
        )
    copied.extend(
        {
            "artifact_kind": "POLICY_CODE",
            "logical_role": row["name"],
            "source_path": str(Path(__file__).parents[2] / row["name"]),
            "sha256": row["sha256"],
        }
        for row in _census_policy_manifest()
    )
    return copied


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _read_table(conn: sqlite3.Connection, table: str, run_id: str) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(f"SELECT * FROM {table} WHERE run_id = ?", (run_id,))]


def validate_terminal_dispositions(
    security_rows: Sequence[Mapping[str, Any]], issuer_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    issuer_allowed = {
        "OUT_OF_SCOPE_NOT_IN_CURRENT_OFFICIAL_REGISTRIES",
        "OUT_OF_SCOPE_NOT_IN_CURRENT_NASDAQ_TRADER_DIRECTORY",
        "OUT_OF_SCOPE_HISTORICAL_COMPARISON",
        "OUT_OF_SCOPE_EXCHANGE",
        "OUT_OF_SCOPE_FUND",
        "OUT_OF_SCOPE_SHELL",
        "OUT_OF_SCOPE_FOREIGN_ISSUER",
        "NEEDS_DATA_IDENTITY",
        "NEEDS_DATA_OPERATING_STATUS",
        "NEEDS_DATA_DOMICILE",
        "NEEDS_DATA_PRIMARY_SECURITY",
        "NEEDS_DATA_MARKET_CAP",
        "NEEDS_DATA_SECTOR",
        "ELIGIBLE_FUTURE_CAP_BAND",
        "ADMITTED_LARGE_AND_MEGA",
    }
    security_allowed = issuer_allowed | {
        "OUT_OF_SCOPE_ETF",
        "OUT_OF_SCOPE_ETN",
        "OUT_OF_SCOPE_CLOSED_END_FUND",
        "OUT_OF_SCOPE_PREFERRED",
        "OUT_OF_SCOPE_WARRANT",
        "OUT_OF_SCOPE_RIGHT",
        "OUT_OF_SCOPE_COMPOSITE_UNIT",
        "OUT_OF_SCOPE_DEBT",
        "OUT_OF_SCOPE_INACTIVE_OR_UNCORROBORATED_LISTING",
        "NEEDS_DATA_SECURITY_TYPE",
        "NEEDS_DATA_ISSUER_DISPOSITION",
        "SECONDARY_OR_DUPLICATE_SECURITY",
        "ADMITTED_PRIMARY_COMMON_EQUITY",
    }
    security_keys = [str(row.get("security_key") or "") for row in security_rows]
    issuer_keys = [str(row.get("issuer_key") or "") for row in issuer_rows]
    missing_security = [
        key
        for key, row in zip(security_keys, security_rows, strict=True)
        if not str(row.get("terminal_disposition") or "").strip()
    ]
    missing_issuer = [
        key
        for key, row in zip(issuer_keys, issuer_rows, strict=True)
        if not str(row.get("terminal_disposition") or "").strip()
    ]
    issuer_key_set = set(issuer_keys)
    securities_by_issuer: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in security_rows:
        securities_by_issuer[str(row.get("issuer_key") or "")].append(row)
    invalid_admitted_primary: list[str] = []
    multiple_primary_issuers: list[str] = []
    unresolved_primary_security_mislabels: list[str] = []
    for issuer in issuer_rows:
        issuer_key = str(issuer.get("issuer_key") or "")
        linked = securities_by_issuer.get(issuer_key, [])
        primaries = [row for row in linked if row.get("is_primary_security") == 1]
        if len(primaries) > 1:
            multiple_primary_issuers.append(issuer_key)
        if issuer.get("terminal_disposition") == "ADMITTED_LARGE_AND_MEGA":
            valid = [
                row
                for row in primaries
                if normalize_exchange(row.get("exchange_name")) in TARGET_EXCHANGES
                and row.get("security_type") in ADMITTED_PRIMARY_SECURITY_TYPES
                and row.get("is_adr") != 1
                and row.get("listing_status") == "ACTIVE"
                and row.get("terminal_disposition") == "ADMITTED_PRIMARY_COMMON_EQUITY"
            ]
            if len(valid) != 1 or len(primaries) != 1:
                invalid_admitted_primary.append(issuer_key)
        if issuer.get("terminal_disposition") == "NEEDS_DATA_PRIMARY_SECURITY":
            common_candidates = [
                row for row in linked if row.get("security_type") in COMMON_SECURITY_TYPES
            ]
            if primaries or any(
                row.get("terminal_disposition") != "NEEDS_DATA_PRIMARY_SECURITY"
                for row in common_candidates
            ):
                unresolved_primary_security_mislabels.append(issuer_key)
    return {
        "unique_security_keys": len(security_keys) == len(set(security_keys)),
        "unique_issuer_keys": len(issuer_keys) == len(set(issuer_keys)),
        "security_terminal_complete": not missing_security,
        "issuer_terminal_complete": not missing_issuer,
        "security_terminal_labels_allowed": all(
            row.get("terminal_disposition") in security_allowed for row in security_rows
        ),
        "issuer_terminal_labels_allowed": all(
            row.get("terminal_disposition") in issuer_allowed for row in issuer_rows
        ),
        "security_terminal_metadata_complete": all(
            row.get("terminal_reason_code") and row.get("terminal_at") for row in security_rows
        ),
        "issuer_terminal_metadata_complete": all(
            row.get("terminal_reason_code") and row.get("terminal_at") for row in issuer_rows
        ),
        "every_security_links_to_one_issuer": all(
            str(row.get("issuer_key") or "") in issuer_key_set for row in security_rows
        ),
        "admitted_issuer_primary_contract_complete": not invalid_admitted_primary,
        "at_most_one_primary_security_per_issuer": not multiple_primary_issuers,
        "unresolved_primary_candidates_remain_needs_data": not unresolved_primary_security_mislabels,
        "missing_security_terminal_keys": missing_security,
        "missing_issuer_terminal_keys": missing_issuer,
        "invalid_admitted_primary_issuer_keys": invalid_admitted_primary,
        "multiple_primary_issuer_keys": multiple_primary_issuers,
        "unresolved_primary_security_mislabel_issuer_keys": unresolved_primary_security_mislabels,
    }


def _semantic_replay_fingerprint(
    security_rows: Sequence[Mapping[str, Any]],
    issuer_rows: Sequence[Mapping[str, Any]],
    unmatched_caps: Sequence[Mapping[str, Any]],
) -> str:
    security_fields = (
        "security_key",
        "ticker",
        "listed_name",
        "exchange_name",
        "listing_status",
        "issuer_key",
        "issuer_relationship_type",
        "related_security_key",
        "share_class",
        "is_primary_security",
        "is_secondary_class",
        "is_duplicate_listing",
        "security_type",
        "security_type_status",
        "is_common_equity",
        "is_adr",
        "identity_status",
        "identity_source",
        "identity_confidence",
        "terminal_disposition",
        "terminal_reason_code",
    )
    issuer_fields = (
        "issuer_key",
        "cik",
        "legal_name",
        "identity_status",
        "issuer_type",
        "operating_status",
        "is_operating_company",
        "scope_status",
        "scope_reason_code",
        "domicile_country_code",
        "domicile_jurisdiction",
        "is_us_domiciled",
        "primary_security_key",
        "primary_ticker",
        "primary_exchange_name",
        "primary_selection_status",
        "primary_selection_method",
        "market_cap_status",
        "market_cap_usd",
        "market_cap_as_of_date",
        "market_cap_method",
        "market_cap_source_provider",
        "market_cap_source_url",
        "market_cap_confidence",
        "cap_derivation_json",
        "cap_band",
        "canonical_sector",
        "source_sector_label",
        "source_sector_system",
        "sector_status",
        "sector_mapping_method",
        "sector_mapping_version",
        "sector_source_url",
        "sector_provenance_json",
        "facts_status",
        "facts_as_of_date",
        "filings_status",
        "latest_annual_filing_date",
        "latest_annual_filing_accession",
        "packet_status",
        "packet_as_of_date",
        "packet_path",
        "packet_sha256",
        "provenance_json",
        "terminal_disposition",
        "terminal_reason_code",
    )

    def semantic_value(row: Mapping[str, Any], field: str) -> Any:
        value = row.get(field)
        if field not in {"cap_derivation_json", "sector_provenance_json", "provenance_json"}:
            return value
        try:
            decoded = json.loads(str(value or "{}"))
        except json.JSONDecodeError:
            return value

        def strip_volatile(item: Any) -> Any:
            if isinstance(item, Mapping):
                return {
                    str(key): strip_volatile(child)
                    for key, child in item.items()
                    if str(key) not in {"retrieved_at", "created_at", "updated_at", "terminal_at"}
                }
            if isinstance(item, list):
                return [strip_volatile(child) for child in item]
            return item

        return strip_volatile(decoded)

    payload = {
        "securities": [
            {field: row.get(field) for field in security_fields}
            for row in sorted(security_rows, key=lambda item: str(item.get("security_key")))
        ],
        "issuers": [
            {field: semantic_value(row, field) for field in issuer_fields}
            for row in sorted(issuer_rows, key=lambda item: str(item.get("issuer_key")))
        ],
        "unmatched_caps": sorted(
            (dict(row) for row in unmatched_caps), key=lambda row: _json_dumps(row)
        ),
    }
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _initialize_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    as_of_date: str,
    inputs: CensusInputPaths,
    output_dir: Path,
    fingerprint: str,
) -> None:
    manifest = _source_manifest(inputs, as_of_date=as_of_date)
    crosscheck_sources = ["Nasdaq Trader Symbol Directory"]
    if inputs.companiesmarketcap_pages:
        crosscheck_sources.append("CompaniesMarketCap USA")
    if inputs.stockanalysis_html is not None:
        crosscheck_sources.append("StockAnalysis Stocks Screener")
    if inputs.terminal_cap_evidence is not None:
        crosscheck_sources.append("Human-reviewed terminal cap evidence ledger")
    existing = conn.execute(
        "SELECT input_fingerprint FROM us_equity_census_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if existing and str(existing["input_fingerprint"]) != fingerprint:
        raise ValueError(
            f"run_id {run_id} already exists with a different immutable input fingerprint"
        )
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO us_equity_census_runs(
            run_id, as_of_date, target_band, status, current_stage,
            acceptance_status, registry_source_name, registry_source_url,
            registry_snapshot_path, registry_snapshot_sha256,
            registry_retrieved_at, source_snapshot_count, source_manifest_json,
            crosscheck_sources_json, input_fingerprint, resume_fingerprint,
            checkpoint_path, status_detail_json, created_at, started_at, updated_at
        ) VALUES(
            ?, ?, ?, 'RUNNING', 'MEMBERSHIP', 'NOT_EVALUATED',
            'SEC company_tickers_exchange',
            'https://www.sec.gov/files/company_tickers_exchange.json',
            ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?, ?
        )
        ON CONFLICT(run_id) DO UPDATE SET
            status = 'RUNNING', resume_fingerprint = excluded.resume_fingerprint,
            updated_at = excluded.updated_at, interrupted_at = NULL
        """,
        (
            run_id,
            as_of_date,
            TARGET_BAND,
            str(inputs.sec_registry),
            _sha256_file(inputs.sec_registry),
            datetime.fromtimestamp(
                inputs.sec_registry.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            len(manifest),
            _json_dumps(manifest),
            _json_dumps(crosscheck_sources),
            fingerprint,
            fingerprint,
            str(output_dir / "checkpoint.json"),
            now,
            now,
            now,
        ),
    )
    conn.execute(
        """
        UPDATE us_equity_census_attempts
        SET status = 'INTERRUPTED',
            error_code = 'PROCESS_INTERRUPTED_BEFORE_RESUME',
            error_message = 'Prior process ended without closing its running attempt.',
            heartbeat_at = ?, finished_at = ?, updated_at = ?
        WHERE run_id = ? AND status = 'RUNNING'
        """,
        (now, now, now, run_id),
    )
    conn.commit()


def _terminal_for_issuer(
    row: Mapping[str, Any], primary_security: Mapping[str, Any] | None = None
) -> tuple[str, str]:
    if str(row.get("scope_reason_code")) == "CAP_SOURCE_NOT_IN_CURRENT_REGISTRIES":
        return (
            "OUT_OF_SCOPE_NOT_IN_CURRENT_OFFICIAL_REGISTRIES",
            "ABSENT_FROM_CURRENT_SEC_AND_NASDAQ_REGISTRY_SNAPSHOTS",
        )
    if str(row.get("scope_reason_code")) == "HISTORICAL_COMPARISON_NOT_CURRENT_REGISTRY":
        return (
            "OUT_OF_SCOPE_HISTORICAL_COMPARISON",
            "HISTORICAL_COMPARISON_NOT_CURRENT_REGISTRY",
        )
    if str(row.get("scope_reason_code")) == "ABSENT_FROM_CURRENT_NASDAQ_TRADER_DIRECTORY":
        return (
            "OUT_OF_SCOPE_NOT_IN_CURRENT_NASDAQ_TRADER_DIRECTORY",
            "SEC_EXCHANGE_ROW_NOT_CORROBORATED_BY_CURRENT_SYMBOL_DIRECTORY",
        )
    if str(row.get("identity_status")) != "RESOLVED":
        return "NEEDS_DATA_IDENTITY", "ISSUER_IDENTITY_UNRESOLVED"
    if str(row.get("scope_reason_code")) == "NO_TARGET_EXCHANGE_LISTING":
        return "OUT_OF_SCOPE_EXCHANGE", "NO_TARGET_EXCHANGE_LISTING"
    if str(row.get("issuer_type")) == "FUND_SECURITY_SET":
        return "OUT_OF_SCOPE_FUND", "FUND_SECURITY_SET"
    if str(row.get("operating_status")) == "SHELL_OR_BLANK_CHECK":
        return "OUT_OF_SCOPE_SHELL", "SHELL_OR_BLANK_CHECK"
    if str(row.get("operating_status")) == "INVESTMENT_COMPANY":
        return "OUT_OF_SCOPE_FUND", "INVESTMENT_COMPANY"
    if str(row.get("operating_status")) == "FOREIGN_FILER":
        return "OUT_OF_SCOPE_FOREIGN_ISSUER", "FOREIGN_FILER"
    if row.get("is_us_listed_foreign_issuer") == 1:
        return "OUT_OF_SCOPE_FOREIGN_ISSUER", "FOREIGN_DOMICILE"
    if str(row.get("operating_status")) != "OPERATING":
        return "NEEDS_DATA_OPERATING_STATUS", "OPERATING_STATUS_UNRESOLVED"
    if row.get("is_operating_company") != 1:
        return "NEEDS_DATA_OPERATING_STATUS", "OPERATING_COMPANY_STATUS_UNRESOLVED"
    if row.get("is_us_domiciled") == 0:
        return "OUT_OF_SCOPE_FOREIGN_ISSUER", "FOREIGN_DOMICILE"
    if row.get("is_us_domiciled") is None:
        return "NEEDS_DATA_DOMICILE", "DOMICILE_UNRESOLVED"
    if str(row.get("primary_selection_status")) != "RESOLVED":
        return "NEEDS_DATA_PRIMARY_SECURITY", str(
            row.get("scope_reason_code") or "PRIMARY_SECURITY_UNRESOLVED"
        )
    primary_security_key = str(row.get("primary_security_key") or "")
    if (
        primary_security is None
        or str(primary_security.get("security_key") or "") != primary_security_key
    ):
        return "NEEDS_DATA_PRIMARY_SECURITY", "PRIMARY_SECURITY_ROW_MISSING"
    primary_security_type = str(primary_security.get("security_type") or SECURITY_UNKNOWN)
    if primary_security_type == SECURITY_ADR or primary_security.get("is_adr") == 1:
        return "OUT_OF_SCOPE_FOREIGN_ISSUER", "ADR_ONLY_PRIMARY_SECURITY"
    if primary_security_type not in ADMITTED_PRIMARY_SECURITY_TYPES:
        return "NEEDS_DATA_PRIMARY_SECURITY", "PRIMARY_SECURITY_NOT_ADMISSIBLE_COMMON_EQUITY"
    if str(row.get("market_cap_status")) != "RESOLVED":
        return "NEEDS_DATA_MARKET_CAP", str(row.get("scope_reason_code") or "MARKET_CAP_UNRESOLVED")
    if str(row.get("cap_band")) != "large_and_mega":
        return "ELIGIBLE_FUTURE_CAP_BAND", "BELOW_LARGE_CAP_FLOOR"
    if str(row.get("sector_status")) != "RESOLVED":
        return "NEEDS_DATA_SECTOR", "SECTOR_CLASSIFICATION_UNRESOLVED"
    return "ADMITTED_LARGE_AND_MEGA", "ALL_LARGE_AND_MEGA_CONTRACTS_RESOLVED"


def _security_terminal(row: Mapping[str, Any], issuer_terminal: str | None) -> tuple[str, str]:
    exchange = normalize_exchange(row.get("exchange_name"))
    security_type = str(row.get("security_type") or SECURITY_UNKNOWN)
    if exchange not in TARGET_EXCHANGES:
        return "OUT_OF_SCOPE_EXCHANGE", f"EXCHANGE_{exchange or 'UNKNOWN'}"
    if str(row.get("listing_status") or "") != "ACTIVE":
        return "OUT_OF_SCOPE_INACTIVE_OR_UNCORROBORATED_LISTING", str(
            row.get("listing_status") or "LISTING_STATUS_UNRESOLVED"
        )
    type_dispositions = {
        SECURITY_ETF: "OUT_OF_SCOPE_ETF",
        SECURITY_ETN: "OUT_OF_SCOPE_ETN",
        SECURITY_CLOSED_END_FUND: "OUT_OF_SCOPE_CLOSED_END_FUND",
        SECURITY_PREFERRED: "OUT_OF_SCOPE_PREFERRED",
        SECURITY_WARRANT: "OUT_OF_SCOPE_WARRANT",
        SECURITY_RIGHT: "OUT_OF_SCOPE_RIGHT",
        SECURITY_UNIT: "OUT_OF_SCOPE_COMPOSITE_UNIT",
        SECURITY_DEBT: "OUT_OF_SCOPE_DEBT",
    }
    if security_type in type_dispositions:
        return type_dispositions[security_type], f"SECURITY_TYPE_{security_type}"
    if security_type == SECURITY_UNKNOWN:
        return "NEEDS_DATA_SECURITY_TYPE", "SECURITY_TYPE_UNRESOLVED"
    if issuer_terminal == "NEEDS_DATA_PRIMARY_SECURITY" and security_type in COMMON_SECURITY_TYPES:
        return "NEEDS_DATA_PRIMARY_SECURITY", "ISSUER_PRIMARY_SECURITY_UNRESOLVED"
    if int(row.get("is_primary_security") or 0) != 1:
        return "SECONDARY_OR_DUPLICATE_SECURITY", "NOT_SELECTED_PRIMARY_SECURITY"
    if (
        issuer_terminal == "ADMITTED_LARGE_AND_MEGA"
        and security_type in ADMITTED_PRIMARY_SECURITY_TYPES
        and row.get("is_adr") != 1
    ):
        return "ADMITTED_PRIMARY_COMMON_EQUITY", "ISSUER_ADMITTED_LARGE_AND_MEGA"
    return issuer_terminal or "NEEDS_DATA_ISSUER_DISPOSITION", "MIRRORS_ISSUER_DISPOSITION"


def _cap_source_hierarchy_state(parsed: _ParsedSources) -> dict[str, Any]:
    approved_count = len(parsed.companiesmarketcap) + sum(
        row.market_cap_usd is not None for row in parsed.stockanalysis
    )
    counts = {
        "LOCAL_AUTHORITATIVE": 0,
        "SEC_EXCHANGE": 0,
        "APPROVED_PROVIDER": approved_count,
        "SEARCH_DIRECT_ISSUER": len(parsed.terminal_cap_evidence),
    }
    return {
        "precedence": list(CAP_EVIDENCE_TIERS),
        "tiers": {
            tier: {
                "record_count": counts[tier],
                "availability_status": "AVAILABLE"
                if counts[tier]
                else "NO_QUALIFYING_AS_OF_SNAPSHOT",
            }
            for tier in CAP_EVIDENCE_TIERS
        },
    }


def _source_acceptance_checks(
    parsed: _ParsedSources, *, inputs: CensusInputPaths, as_of_date: str
) -> dict[str, bool]:
    contract = OFFICIAL_SOURCE_SNAPSHOT_CONTRACTS.get(as_of_date, {})
    cmc_ranks = sorted(row.rank for row in parsed.companiesmarketcap)
    cmc_contiguous = bool(cmc_ranks) and cmc_ranks == list(range(1, max(cmc_ranks) + 1))
    nasdaq_dates = {
        str(row.file_creation_timestamp or "")[:10]
        for row in (*parsed.nasdaq, *parsed.other)
        if row.file_creation_timestamp
    }
    stock_caps = [row for row in parsed.stockanalysis if row.market_cap_usd is not None]
    cmc_dates = {row.as_of_date for row in parsed.companiesmarketcap}
    stock_dates = {row.as_of_date for row in parsed.stockanalysis}
    issue_counts = Counter(issue.code for issue in parsed.issues)
    cap_hierarchy = _cap_source_hierarchy_state(parsed)
    return {
        "official_security_registry_snapshots_nonempty": bool(
            parsed.sec and parsed.nasdaq and parsed.other
        ),
        "official_source_snapshot_row_contract": bool(
            contract
            and len(parsed.sec) == contract.get("sec_exchange_security_rows")
            and len(parsed.nasdaq) == contract.get("nasdaq_listed_security_rows")
            and len(parsed.other) == contract.get("other_listed_security_rows")
            and len(parsed.companiesmarketcap) == contract.get("companiesmarketcap_rows")
            and len(parsed.stockanalysis) == contract.get("stockanalysis_rows")
        ),
        "official_source_security_keys_unique": bool(
            len(parsed.sec) == len({(row.cik, row.ticker, row.exchange) for row in parsed.sec})
            and len(parsed.nasdaq) == len({row.symbol for row in parsed.nasdaq})
            and len(parsed.other) == len({row.symbol for row in parsed.other})
            and len(parsed.companiesmarketcap)
            == len({row.ticker for row in parsed.companiesmarketcap})
            and len(parsed.stockanalysis) == len({row.symbol for row in parsed.stockanalysis})
        ),
        "official_registry_snapshot_acquired_as_of": all(
            _file_acquisition_date(path) == as_of_date
            for path in (inputs.sec_registry, inputs.nasdaq_listed, inputs.other_listed)
        ),
        "nasdaq_symbol_directories_current_as_of": nasdaq_dates == {as_of_date},
        "two_independent_cap_source_snapshots_present": bool(
            parsed.companiesmarketcap and parsed.stockanalysis
        ),
        "cap_source_hierarchy_state_explicit": bool(
            cap_hierarchy["precedence"] == list(CAP_EVIDENCE_TIERS)
            and set(cap_hierarchy["tiers"]) == set(CAP_EVIDENCE_TIERS)
            and all(
                row["availability_status"] in {"AVAILABLE", "NO_QUALIFYING_AS_OF_SNAPSHOT"}
                for row in cap_hierarchy["tiers"].values()
            )
        ),
        "cap_source_snapshots_acquired_as_of": bool(
            cmc_dates == {as_of_date}
            and stock_dates == {as_of_date}
            and all(
                _file_acquisition_date(path) == as_of_date
                for path in inputs.companiesmarketcap_pages
            )
            and inputs.stockanalysis_html is not None
            and _file_acquisition_date(inputs.stockanalysis_html) == as_of_date
        ),
        "companiesmarketcap_ranking_contiguous_through_below_boundary": bool(
            cmc_contiguous
            and len(parsed.companiesmarketcap) == contract.get("companiesmarketcap_rows")
            and sum(row.market_cap_usd >= LARGE_CAP_FLOOR_USD for row in parsed.companiesmarketcap)
            == contract.get("companiesmarketcap_large_rows")
            and any(row.market_cap_usd < LARGE_CAP_FLOOR_USD for row in parsed.companiesmarketcap)
            and any(row.market_cap_usd >= LARGE_CAP_FLOOR_USD for row in parsed.companiesmarketcap)
        ),
        "stockanalysis_full_snapshot_contract": bool(
            len(parsed.stockanalysis) == contract.get("stockanalysis_rows")
            and len(stock_caps) == contract.get("stockanalysis_cap_rows")
            and any(float(row.market_cap_usd) < LARGE_CAP_FLOOR_USD for row in stock_caps)
            and any(float(row.market_cap_usd) >= LARGE_CAP_FLOOR_USD for row in stock_caps)
        ),
        "source_parse_issue_contract": issue_counts == {"MISSING_MARKET_CAP": 1},
    }


def _issuer_acceptance_checks(
    issuer_rows: Sequence[Mapping[str, Any]],
    *,
    cap_source_unmatched: int,
    fixed_cohort_count: int,
    fixed_cohort_issuer_count: int,
    source_checks: Mapping[str, bool],
    provider_free_replay_verified: bool,
    sector_population_promoted: bool,
    cost_ledger: CensusCostLedger,
) -> dict[str, bool]:
    def cap_evidence_values(row: Mapping[str, Any]) -> tuple[float, ...]:
        try:
            payload = json.loads(str(row.get("cap_derivation_json") or "{}"))
        except json.JSONDecodeError:
            return ()
        evidence = payload.get("evidence") if isinstance(payload, Mapping) else None
        if not isinstance(evidence, list):
            return ()
        values: list[float] = []
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            value = item.get("market_cap_usd")
            if isinstance(value, (int, float)) and float(value) > 0:
                values.append(float(value))
        return tuple(values)

    def is_potential_large(row: Mapping[str, Any]) -> bool:
        cap = row.get("market_cap_usd")
        return (
            isinstance(cap, (int, float)) and float(cap) >= BOUNDARY_INVESTIGATION_FLOOR_USD
        ) or any(value >= BOUNDARY_INVESTIGATION_FLOOR_USD for value in cap_evidence_values(row))

    def is_current_registry_candidate(row: Mapping[str, Any]) -> bool:
        return row.get("terminal_disposition") not in {
            "OUT_OF_SCOPE_NOT_IN_CURRENT_OFFICIAL_REGISTRIES",
            "OUT_OF_SCOPE_HISTORICAL_COMPARISON",
        }

    admitted = [
        row for row in issuer_rows if row.get("terminal_disposition") == "ADMITTED_LARGE_AND_MEGA"
    ]
    potential_large = [row for row in issuer_rows if is_potential_large(row)]
    current_potential_large = [row for row in potential_large if is_current_registry_candidate(row)]
    cap_source_only_large = [
        row
        for row in potential_large
        if row.get("terminal_disposition") == "OUT_OF_SCOPE_NOT_IN_CURRENT_OFFICIAL_REGISTRIES"
    ]
    otherwise_eligible_before_primary = [
        row
        for row in current_potential_large
        if row.get("identity_status") == "RESOLVED"
        and row.get("operating_status") == "OPERATING"
        and row.get("is_us_domiciled") == 1
        and row.get("scope_status") != "OUT_OF_SCOPE"
    ]
    otherwise_eligible_before_sector = [
        row
        for row in otherwise_eligible_before_primary
        if row.get("primary_selection_status") == "RESOLVED"
        and row.get("market_cap_status") == "RESOLVED"
        and row.get("cap_band") == "large_and_mega"
    ]
    current_us_operating_primary = [
        row
        for row in issuer_rows
        if is_current_registry_candidate(row)
        and row.get("identity_status") == "RESOLVED"
        and row.get("operating_status") == "OPERATING"
        and row.get("is_us_domiciled") == 1
        and row.get("primary_selection_status") == "RESOLVED"
        and row.get("scope_status") != "OUT_OF_SCOPE"
    ]
    return {
        "every_issuer_has_one_terminal_disposition": all(
            str(row.get("terminal_disposition") or "").strip() for row in issuer_rows
        ),
        "zero_unreconciled_large_cap_source_rows": cap_source_unmatched == 0,
        "cap_source_only_large_rows_have_explicit_registry_absence_disposition": all(
            row.get("scope_reason_code") == "CAP_SOURCE_NOT_IN_CURRENT_REGISTRIES"
            and row.get("identity_status") == "NEEDS_DATA"
            for row in cap_source_only_large
        ),
        "zero_unexplained_large_cap_identity_gaps": all(
            row.get("identity_status") == "RESOLVED"
            or row.get("terminal_disposition") == "OUT_OF_SCOPE_NOT_IN_CURRENT_OFFICIAL_REGISTRIES"
            for row in potential_large
        ),
        "zero_silent_unknown_cap": all(
            row.get("market_cap_status") != "NEEDS_DATA"
            or str(row.get("terminal_disposition") or "").startswith("NEEDS_DATA_")
            or str(row.get("terminal_disposition") or "").startswith("OUT_OF_SCOPE_")
            for row in issuer_rows
        ),
        "all_current_us_operating_primary_issuers_have_cap_band_decision": all(
            row.get("market_cap_status") == "RESOLVED" and row.get("cap_band") != "UNRESOLVED"
            for row in current_us_operating_primary
        ),
        "zero_unresolved_current_large_candidate_dispositions": all(
            not str(row.get("terminal_disposition") or "").startswith("NEEDS_DATA_")
            for row in current_potential_large
        ),
        "zero_unresolved_eligible_large_primary_securities": all(
            row.get("primary_selection_status") == "RESOLVED"
            for row in otherwise_eligible_before_primary
        ),
        "zero_unresolved_eligible_large_market_cap_boundaries": all(
            row.get("market_cap_status") == "RESOLVED"
            for row in otherwise_eligible_before_primary
            if row.get("primary_selection_status") == "RESOLVED"
        ),
        "zero_unresolved_eligible_large_sector_contracts": all(
            row.get("sector_status") == "RESOLVED"
            and row.get("canonical_sector")
            and row.get("sector_mapping_version") == CANONICAL_SECTOR_TAXONOMY_VERSION
            for row in otherwise_eligible_before_sector
        ),
        "all_admitted_have_resolved_sector_contract": all(
            row.get("sector_status") == "RESOLVED"
            and row.get("canonical_sector")
            and row.get("sector_mapping_version") == CANONICAL_SECTOR_TAXONOMY_VERSION
            for row in admitted
        ),
        "all_admitted_are_us_domiciled_operating": all(
            row.get("operating_status") == "OPERATING"
            and row.get("is_operating_company") == 1
            and row.get("is_us_domiciled") == 1
            and row.get("is_us_listed_foreign_issuer") == 0
            for row in admitted
        ),
        "large_cap_acceptance_precedes_mid_cap": all(
            row.get("cap_band") == "large_and_mega" for row in admitted
        ),
        "fixed_642_security_610_issuer_cohort_reconciled_as_comparison_not_boundary": (
            fixed_cohort_count == 642 and fixed_cohort_issuer_count == 610
        ),
        "provider_free_replay_verified": provider_free_replay_verified,
        "new_large_cap_scan_population_promoted": sector_population_promoted,
        "actual_llm_calls_zero": cost_ledger.actual_llm_calls == 0,
        "actual_llm_cost_zero": cost_ledger.actual_llm_cost_usd == 0.0,
        "production_sector_scan_exercised": False,
        **source_checks,
    }


# The staged orchestration and exporter follow below.  Keeping the pure
# security/domicile/primary/cap functions above independently testable makes
# census policy auditable without a live database or network.


@dataclass(frozen=True, slots=True)
class _ParsedSources:
    sec: tuple[SecExchangeSecurity, ...]
    nasdaq: tuple[NasdaqTraderSecurity, ...]
    other: tuple[NasdaqTraderSecurity, ...]
    companiesmarketcap: tuple[CompaniesMarketCapSecurity, ...]
    stockanalysis: tuple[StockAnalysisSecurity, ...]
    terminal_cap_evidence: tuple[CapEvidenceRow, ...]
    issues: tuple[ParseIssue, ...]


def _parse_source_snapshots(inputs: CensusInputPaths, *, as_of_date: str) -> _ParsedSources:
    sec = parse_sec_company_tickers_exchange(inputs.sec_registry.read_bytes())
    nasdaq = parse_nasdaqlisted(inputs.nasdaq_listed.read_text(encoding="utf-8-sig"))
    other = parse_otherlisted(inputs.other_listed.read_text(encoding="utf-8-sig"))
    cmc_records: list[CompaniesMarketCapSecurity] = []
    issues: list[ParseIssue] = [*sec.issues, *nasdaq.issues, *other.issues]
    for page_number, path in enumerate(inputs.companiesmarketcap_pages, start=1):
        source_url = (
            "https://companiesmarketcap.com/usa/largest-companies-in-the-usa-by-market-cap/"
        )
        if page_number > 1:
            source_url += f"?page={page_number}"
        result = parse_companiesmarketcap_usa_html(
            path.read_text(encoding="utf-8", errors="replace"),
            source_url=source_url,
            as_of_date=_file_acquisition_date(path),
        )
        cmc_records.extend(result.records)
        issues.extend(result.issues)
    stock_records: tuple[StockAnalysisSecurity, ...] = ()
    if inputs.stockanalysis_html is not None:
        stock = parse_stockanalysis_screener_html(
            inputs.stockanalysis_html.read_text(encoding="utf-8", errors="replace"),
            source_url="https://stockanalysis.com/stocks/screener/",
            as_of_date=_file_acquisition_date(inputs.stockanalysis_html),
        )
        stock_records = stock.records
        issues.extend(stock.issues)
    terminal_records: tuple[CapEvidenceRow, ...] = ()
    if inputs.terminal_cap_evidence is not None:
        terminal_records, terminal_issues = parse_terminal_cap_evidence_snapshot(
            inputs.terminal_cap_evidence, census_as_of_date=as_of_date
        )
        issues.extend(terminal_issues)
    return _ParsedSources(
        sec=sec.records,
        nasdaq=nasdaq.records,
        other=other.records,
        companiesmarketcap=tuple(cmc_records),
        stockanalysis=stock_records,
        terminal_cap_evidence=terminal_records,
        issues=tuple(issues),
    )


def _issuer_name_key(value: object) -> str:
    text = re.sub(r"[^A-Z0-9]+", " ", str(value or "").upper()).strip()
    suffixes = {
        "INC",
        "INCORPORATED",
        "CORP",
        "CORPORATION",
        "CO",
        "COMPANY",
        "LTD",
        "LIMITED",
        "PLC",
        "LP",
        "LLC",
    }
    tokens = text.split()
    while tokens and tokens[-1] in suffixes:
        tokens.pop()
    return " ".join(tokens)


def _issuer_name_tokens(value: object) -> frozenset[str]:
    stop = {"THE", "OF", "AND", "HOLDINGS", "GROUP", "CORP", "CORPORATION", "INC"}
    return frozenset(
        token for token in _issuer_name_key(value).split() if token and token not in stop
    )


def _best_unique_name_match(
    source_name: object, candidates: Mapping[str, Sequence[str]]
) -> str | None:
    source_tokens = _issuer_name_tokens(source_name)
    if not source_tokens:
        return None
    scores: list[tuple[float, str]] = []
    for candidate_key, names in candidates.items():
        best = 0.0
        for name in names:
            tokens = _issuer_name_tokens(name)
            if not tokens:
                continue
            best = max(best, len(source_tokens & tokens) / len(source_tokens | tokens))
        if best:
            scores.append((best, candidate_key))
    scores.sort(reverse=True)
    if not scores or scores[0][0] < 0.70:
        return None
    if len(scores) > 1 and scores[0][0] - scores[1][0] < 0.15:
        return None
    return scores[0][1]


def large_cap_submission_candidates(
    inputs: CensusInputPaths,
    *,
    as_of_date: str,
    boundary_floor_usd: float = BOUNDARY_INVESTIGATION_FLOOR_USD,
) -> tuple[dict[str, tuple[str, ...]], list[dict[str, Any]]]:
    """Resolve free SEC-submissions fetch targets from the cap-source union."""

    parsed = _parse_source_snapshots(inputs, as_of_date=as_of_date)
    sec_by_ticker: dict[str, list[SecExchangeSecurity]] = defaultdict(list)
    sec_by_compact_ticker: dict[str, list[SecExchangeSecurity]] = defaultdict(list)
    sec_by_name: dict[str, list[SecExchangeSecurity]] = defaultdict(list)
    sec_names_by_cik: dict[str, list[str]] = defaultdict(list)
    for row in parsed.sec:
        normalized = normalize_ticker(row.ticker)
        sec_by_ticker[normalized].append(row)
        sec_by_compact_ticker[normalized.replace("-", "")].append(row)
        sec_by_name[_issuer_name_key(row.name)].append(row)
        sec_names_by_cik[str(row.cik).strip().lstrip("0") or "0"].append(row.name)

    source_candidates: list[dict[str, Any]] = []
    source_candidates.extend(
        {
            "source_provider": "COMPANIESMARKETCAP_USA",
            "ticker": row.ticker,
            "name": row.name,
            "market_cap_usd": row.market_cap_usd,
        }
        for row in parsed.companiesmarketcap
        if row.market_cap_usd >= boundary_floor_usd
    )
    source_candidates.extend(
        {
            "source_provider": "STOCKANALYSIS",
            "ticker": row.symbol,
            "name": row.name,
            "market_cap_usd": row.market_cap_usd,
        }
        for row in parsed.stockanalysis
        if row.market_cap_usd is not None and row.market_cap_usd >= boundary_floor_usd
    )
    source_candidates.extend(
        {
            "source_provider": "FIXED_642_SECURITY_REPLAY_COMPARISON",
            "ticker": ticker,
            "name": "",
            "market_cap_usd": None,
        }
        for ticker in _parse_fixed_cohort(inputs)
    )

    tickers_by_cik: dict[str, set[str]] = defaultdict(set)
    unresolved: list[dict[str, Any]] = []
    seen_candidates: set[tuple[str, str]] = set()
    for candidate in source_candidates:
        ticker = normalize_ticker(candidate["ticker"])
        dedupe_key = (str(candidate["source_provider"]), ticker)
        if dedupe_key in seen_candidates:
            continue
        seen_candidates.add(dedupe_key)
        matches = list(sec_by_ticker.get(ticker) or [])
        if not matches:
            matches = list(sec_by_compact_ticker.get(ticker.replace("-", "")) or [])
        if not matches and candidate["name"]:
            matches = list(sec_by_name.get(_issuer_name_key(candidate["name"])) or [])
        if not matches and candidate["name"]:
            fuzzy_cik = _best_unique_name_match(candidate["name"], sec_names_by_cik)
            if fuzzy_cik:
                matches = [
                    row
                    for rows in sec_by_ticker.values()
                    for row in rows
                    if (str(row.cik).strip().lstrip("0") or "0") == fuzzy_cik
                ][:1]
        ciks = sorted({str(row.cik).strip().lstrip("0") or "0" for row in matches})
        if len(ciks) != 1:
            unresolved.append(
                {
                    **candidate,
                    "normalized_ticker": ticker,
                    "match_status": "UNMATCHED" if not ciks else "AMBIGUOUS",
                    "candidate_ciks": ciks,
                }
            )
            continue
        tickers_by_cik[ciks[0]].add(ticker)
    return (
        {cik: tuple(sorted(tickers)) for cik, tickers in sorted(tickers_by_cik.items())},
        unresolved,
    )


def fetch_large_cap_sec_submissions(
    inputs: CensusInputPaths,
    *,
    output_dir: Path,
    as_of_date: str,
    max_fetches: int | None = None,
    http: Any = None,
) -> SubmissionFetchResult:
    """Fetch free official SEC submissions for the large-cap and boundary union.

    Existing valid snapshots are reused.  Every attempted, failed, deferred, or
    unresolved candidate is written to the fetch summary; no paid provider or
    language-model surface is called.
    """

    from app.util.http import HttpClient

    canonical_as_of = date.fromisoformat(as_of_date).isoformat()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates, unresolved = large_cap_submission_candidates(inputs, as_of_date=canonical_as_of)
    http = http or HttpClient()
    attempts: list[dict[str, Any]] = []
    existing = 0
    fetched = 0
    failed = 0
    deferred = 0
    network_attempts = 0
    for cik, tickers in candidates.items():
        normalized = str(cik).zfill(10)
        path = output_dir / f"{normalized}.json"
        cached = _safe_json(path)
        if cached:
            existing += 1
            attempts.append(
                {
                    "cik": cik,
                    "tickers": list(tickers),
                    "status": "EXISTING_VALID_SNAPSHOT",
                    "path": str(path),
                    "sha256": _sha256_file(path),
                }
            )
            continue
        if max_fetches is not None and network_attempts >= max_fetches:
            deferred += 1
            attempts.append(
                {
                    "cik": cik,
                    "tickers": list(tickers),
                    "status": "DEFERRED_BY_FETCH_LIMIT",
                }
            )
            continue
        network_attempts += 1
        url = f"https://data.sec.gov/submissions/CIK{normalized}.json"
        try:
            payload = http.get_json(url, use_cache=True, cache_ttl_seconds=None)
            if not isinstance(payload, dict) or not payload:
                raise ValueError("SEC submissions response was not a non-empty object")
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            temporary.replace(path)
            fetched += 1
            attempts.append(
                {
                    "cik": cik,
                    "tickers": list(tickers),
                    "status": "FETCHED",
                    "source_url": url,
                    "path": str(path),
                    "sha256": _sha256_file(path),
                }
            )
        except Exception as exc:
            failed += 1
            attempts.append(
                {
                    "cik": cik,
                    "tickers": list(tickers),
                    "status": "FAILED",
                    "source_url": url,
                    "error_code": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    result = SubmissionFetchResult(
        as_of_date=canonical_as_of,
        output_dir=output_dir,
        candidate_cik_count=len(candidates),
        existing_snapshot_count=existing,
        fetched_snapshot_count=fetched,
        failed_snapshot_count=failed,
        deferred_snapshot_count=deferred,
        unresolved_candidate_count=len(unresolved),
        attempts=attempts,
        unresolved_candidates=unresolved,
    )
    (output_dir / "fetch_summary.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _listing_index(parsed: _ParsedSources) -> dict[str, list[NasdaqTraderSecurity]]:
    index: dict[str, list[NasdaqTraderSecurity]] = defaultdict(list)
    for row in (*parsed.nasdaq, *parsed.other):
        index[normalize_ticker(row.symbol)].append(row)
        if row.cqs_symbol:
            index[normalize_ticker(row.cqs_symbol)].append(row)
        if row.nasdaq_symbol:
            index[normalize_ticker(row.nasdaq_symbol)].append(row)
    return index


def _listing_for_sec_row(
    row: SecExchangeSecurity,
    index: Mapping[str, Sequence[NasdaqTraderSecurity]],
) -> NasdaqTraderSecurity | None:
    candidates = list(index.get(normalize_ticker(row.ticker)) or [])
    if not candidates:
        return None
    sec_exchange = normalize_exchange(row.exchange)
    if sec_exchange == "NASDAQ":
        exact = [
            candidate
            for candidate in candidates
            if normalize_exchange(candidate.exchange) == "NASDAQ"
        ]
        if exact:
            return exact[0]
    if sec_exchange == "NYSE":
        exact = [
            candidate
            for candidate in candidates
            if normalize_exchange(candidate.exchange) in {"NYSE", "NYSE AMERICAN"}
        ]
        if exact:
            return exact[0]
    return candidates[0]


def _share_class_from_name(name: str) -> str | None:
    match = re.search(r"\bClass\s+([A-Z0-9]+)\b", name, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _security_row_from_listing(
    *,
    sec: SecExchangeSecurity | None,
    listing: NasdaqTraderSecurity | None,
    as_of_date: str,
    retrieved_at: str,
) -> dict[str, Any]:
    ticker = sec.ticker if sec is not None else str(listing.symbol if listing else "")
    cik = str(sec.cik if sec is not None else "").strip().lstrip("0")
    exchange = normalize_exchange(
        listing.exchange
        if listing is not None and listing.exchange
        else sec.exchange
        if sec
        else ""
    )
    issuer_key = (
        f"CIK:{cik}" if cik else f"UNRESOLVED:{normalize_ticker(ticker)}:{exchange or 'UNKNOWN'}"
    )
    security_key = (
        stable_security_key(cik=cik, ticker=ticker, exchange=exchange)
        if cik
        else f"UNRESOLVED:{normalize_ticker(ticker)}:{exchange.replace(' ', '_') or 'UNKNOWN'}"
    )
    listed_name = (
        str(listing.security_name).strip()
        if listing is not None
        else str(sec.name if sec is not None else "").strip()
    )
    type_result = classify_security_type(
        listed_name=listed_name,
        ticker=ticker,
        etf_flag="Y" if listing is not None and listing.etf else "N" if listing else None,
    )
    listing_status = (
        "TEST_ISSUE"
        if listing is not None and listing.test_issue
        else "ACTIVE"
        if listing is not None
        else "SEC_REGISTRY_ONLY_UNCORROBORATED"
    )
    raw_payload = {
        "sec": asdict(sec) if sec is not None else None,
        "nasdaq_trader": asdict(listing) if listing is not None else None,
    }
    source_registry = (
        "SEC_EXCHANGE_REGISTRY+NASDAQ_TRADER"
        if sec is not None and listing is not None
        else "SEC_EXCHANGE_REGISTRY"
        if sec is not None
        else "NASDAQ_TRADER"
    )
    listing_source_url = (
        "https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs"
        if listing is not None
        else sec.source_url
        if sec is not None
        else None
    )
    return {
        "security_key": security_key,
        "source_registry": source_registry,
        "source_security_id": (
            f"SEC:{sec.source_row_number}"
            if sec is not None
            else f"NASDAQ:{listing.source_file}:{listing.source_line_number}"
        ),
        "ticker": normalize_ticker(ticker),
        "listed_name": listed_name,
        "exchange_name": exchange,
        "exchange_mic": None,
        "listing_status": listing_status,
        "listing_status_as_of_date": as_of_date,
        "listing_source_url": listing_source_url,
        "listing_retrieved_at": retrieved_at,
        "issuer_key": issuer_key,
        "issuer_relationship_type": "PENDING_PRIMARY_SELECTION",
        "related_security_key": None,
        "share_class": _share_class_from_name(listed_name),
        "is_primary_security": None,
        "is_secondary_class": None,
        "is_duplicate_listing": 0,
        "security_type": type_result.security_type,
        "security_type_status": type_result.status,
        "is_common_equity": int(type_result.is_common_equity),
        "is_adr": int(type_result.is_adr),
        "adr_ratio": None,
        "identity_status": "RESOLVED" if cik else "NEEDS_DATA",
        "identity_source": "SEC_CIK" if cik else "NASDAQ_LISTING_ONLY",
        "identity_source_url": sec.source_url if sec is not None else listing_source_url,
        "identity_as_of_date": as_of_date,
        "identity_retrieved_at": retrieved_at,
        "identity_confidence": "HIGH" if cik else "LOW",
        "last_completed_stage": "SECURITY_TYPE",
        "next_stage": "PRIMARY_SECURITY_SELECTION",
        "processing_status": "IN_PROGRESS",
        "terminal_disposition": None,
        "terminal_reason_code": None,
        "terminal_detail_json": "{}",
        "terminal_at": None,
        "source_snapshot_ref": source_registry,
        "raw_source_json": _json_dumps(raw_payload),
        "provenance_json": _json_dumps(
            {
                "security_type_reason_code": type_result.reason_code,
                "financial_status": listing.financial_status if listing else None,
                "nasdaq_file_timestamp": listing.file_creation_timestamp if listing else None,
            }
        ),
        "cik": cik or None,
        "issuer_legal_name": str(sec.name).strip() if sec is not None else listed_name,
    }


def _build_membership_rows(
    parsed: _ParsedSources,
    *,
    as_of_date: str,
    retrieved_at: str,
    fixed_cohort_tickers: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    listing_index = _listing_index(parsed)
    rows_by_key: dict[str, dict[str, Any]] = {}
    matched_listing_ids: set[tuple[str, int]] = set()
    for sec in parsed.sec:
        listing = _listing_for_sec_row(sec, listing_index)
        if listing is not None:
            matched_listing_ids.add((listing.source_file, listing.source_line_number))
        row = _security_row_from_listing(
            sec=sec, listing=listing, as_of_date=as_of_date, retrieved_at=retrieved_at
        )
        existing = rows_by_key.get(row["security_key"])
        if existing is None:
            rows_by_key[row["security_key"]] = row
        else:
            existing["is_duplicate_listing"] = 1
            provenance = json.loads(existing["provenance_json"])
            provenance.setdefault("duplicate_source_rows", []).append(row["source_security_id"])
            existing["provenance_json"] = _json_dumps(provenance)

    for listing in (*parsed.nasdaq, *parsed.other):
        listing_id = (listing.source_file, listing.source_line_number)
        if listing_id in matched_listing_ids:
            continue
        row = _security_row_from_listing(
            sec=None, listing=listing, as_of_date=as_of_date, retrieved_at=retrieved_at
        )
        rows_by_key.setdefault(row["security_key"], row)

    # A near-boundary cap-source row is itself a discovered candidate.  If
    # neither registry contains it, preserve it as a cap-only security with an
    # explicit unresolved identity rather than losing it before reconciliation.
    cap_candidates: list[tuple[str, str, str]] = []
    cap_candidates.extend(
        (row.ticker, row.name, "COMPANIESMARKETCAP_USA")
        for row in parsed.companiesmarketcap
        if row.market_cap_usd >= BOUNDARY_INVESTIGATION_FLOOR_USD
    )
    cap_candidates.extend(
        (row.symbol, row.name, "STOCKANALYSIS")
        for row in parsed.stockanalysis
        if row.market_cap_usd is not None and row.market_cap_usd >= BOUNDARY_INVESTIGATION_FLOOR_USD
    )
    known_tickers = {normalize_ticker(row["ticker"]) for row in rows_by_key.values()}
    current_names_by_issuer: dict[str, list[str]] = defaultdict(list)
    for row in rows_by_key.values():
        if row.get("cik") and row.get("issuer_legal_name"):
            current_names_by_issuer[str(row["issuer_key"])].append(str(row["issuer_legal_name"]))
    for ticker, name, provider in cap_candidates:
        normalized = normalize_ticker(ticker)
        if normalized in known_tickers:
            continue
        if _best_unique_name_match(name, current_names_by_issuer) is not None:
            # The cap source is using a stale/alternate symbol for a uniquely
            # name-bound current issuer.  Cap reconciliation records that
            # mapping; it is not a separate security membership row.
            continue
        key = f"CAP_ONLY:{normalized}"
        type_result = classify_security_type(listed_name=name, ticker=normalized)
        rows_by_key[key] = {
            "security_key": key,
            "source_registry": provider,
            "source_security_id": f"{provider}:{normalized}",
            "ticker": normalized,
            "listed_name": name,
            "exchange_name": "",
            "exchange_mic": None,
            "listing_status": "NEEDS_DATA",
            "listing_status_as_of_date": as_of_date,
            "listing_source_url": None,
            "listing_retrieved_at": retrieved_at,
            "issuer_key": f"UNRESOLVED:{normalized}:CAP_SOURCE",
            "issuer_relationship_type": "UNRESOLVED",
            "related_security_key": None,
            "share_class": _share_class_from_name(name),
            "is_primary_security": None,
            "is_secondary_class": None,
            "is_duplicate_listing": 0,
            "security_type": type_result.security_type,
            "security_type_status": type_result.status,
            "is_common_equity": int(type_result.is_common_equity),
            "is_adr": int(type_result.is_adr),
            "adr_ratio": None,
            "identity_status": "NEEDS_DATA",
            "identity_source": provider,
            "identity_source_url": None,
            "identity_as_of_date": as_of_date,
            "identity_retrieved_at": retrieved_at,
            "identity_confidence": "LOW",
            "last_completed_stage": "SECURITY_TYPE",
            "next_stage": "IDENTITY",
            "processing_status": "NEEDS_DATA",
            "terminal_disposition": None,
            "terminal_reason_code": None,
            "terminal_detail_json": "{}",
            "terminal_at": None,
            "source_snapshot_ref": provider,
            "raw_source_json": _json_dumps({"ticker": ticker, "name": name}),
            "provenance_json": _json_dumps({"security_type_reason_code": type_result.reason_code}),
            "cik": None,
            "issuer_legal_name": name,
        }
        known_tickers.add(normalized)

    for ticker in sorted(fixed_cohort_tickers or set()):
        normalized = normalize_ticker(ticker)
        if not normalized or normalized in known_tickers:
            continue
        key = f"FIXED_COHORT_ONLY:{normalized}"
        rows_by_key[key] = {
            "security_key": key,
            "source_registry": "FIXED_642_SECURITY_REPLAY_COMPARISON",
            "source_security_id": f"FIXED_REPLAY:{normalized}",
            "ticker": normalized,
            "listed_name": "",
            "exchange_name": "",
            "exchange_mic": None,
            "listing_status": "HISTORICAL_COMPARISON_ONLY",
            "listing_status_as_of_date": "2026-07-15",
            "listing_source_url": None,
            "listing_retrieved_at": retrieved_at,
            "issuer_key": f"UNRESOLVED:{normalized}:FIXED_REPLAY",
            "issuer_relationship_type": "UNRESOLVED",
            "related_security_key": None,
            "share_class": None,
            "is_primary_security": None,
            "is_secondary_class": None,
            "is_duplicate_listing": 0,
            "security_type": SECURITY_UNKNOWN,
            "security_type_status": "NEEDS_DATA",
            "is_common_equity": 0,
            "is_adr": 0,
            "adr_ratio": None,
            "identity_status": "NEEDS_DATA",
            "identity_source": "FIXED_REPLAY_COMPARISON",
            "identity_source_url": None,
            "identity_as_of_date": "2026-07-15",
            "identity_retrieved_at": retrieved_at,
            "identity_confidence": "LOW",
            "last_completed_stage": "MEMBERSHIP",
            "next_stage": "IDENTITY",
            "processing_status": "NEEDS_DATA",
            "terminal_disposition": None,
            "terminal_reason_code": None,
            "terminal_detail_json": "{}",
            "terminal_at": None,
            "source_snapshot_ref": "FIXED_642_SECURITY_REPLAY_COMPARISON",
            "raw_source_json": _json_dumps({"ticker": normalized}),
            "provenance_json": _json_dumps(
                {"reason_code": "FIXED_COHORT_TICKER_NOT_IN_CURRENT_REGISTRIES"}
            ),
            "cik": None,
            "issuer_legal_name": "",
        }
        known_tickers.add(normalized)

    rows = sorted(rows_by_key.values(), key=lambda row: row["security_key"])
    counts = {
        "sec_security_rows": len(parsed.sec),
        "nasdaq_trader_security_rows": len(parsed.nasdaq) + len(parsed.other),
        "union_security_rows": len(rows),
        "parse_issue_count": len(parsed.issues),
        "identity_unresolved_security_rows": sum(
            row["identity_status"] != "RESOLVED" for row in rows
        ),
    }
    return rows, counts


_SECURITY_DB_COLUMNS = (
    "security_key",
    "source_registry",
    "source_security_id",
    "ticker",
    "listed_name",
    "exchange_name",
    "exchange_mic",
    "listing_status",
    "listing_status_as_of_date",
    "listing_source_url",
    "listing_retrieved_at",
    "issuer_key",
    "issuer_relationship_type",
    "related_security_key",
    "share_class",
    "is_primary_security",
    "is_secondary_class",
    "is_duplicate_listing",
    "security_type",
    "security_type_status",
    "is_common_equity",
    "is_adr",
    "adr_ratio",
    "identity_status",
    "identity_source",
    "identity_source_url",
    "identity_as_of_date",
    "identity_retrieved_at",
    "identity_confidence",
    "last_completed_stage",
    "next_stage",
    "processing_status",
    "terminal_disposition",
    "terminal_reason_code",
    "terminal_detail_json",
    "terminal_at",
    "source_snapshot_ref",
    "raw_source_json",
    "provenance_json",
)


def _persist_security_rows(
    conn: sqlite3.Connection, *, run_id: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    now = utc_now_iso()
    columns = ("run_id", *_SECURITY_DB_COLUMNS, "created_at", "updated_at")
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(
        f"{column}=excluded.{column}"
        for column in (*_SECURITY_DB_COLUMNS, "updated_at")
        if column != "security_key"
    )
    sql = (
        f"INSERT INTO us_equity_census_securities({', '.join(columns)}) "
        f"VALUES({placeholders}) ON CONFLICT(run_id, security_key) DO UPDATE SET {updates}"
    )
    conn.executemany(
        sql,
        [
            (
                run_id,
                *(row.get(column) for column in _SECURITY_DB_COLUMNS),
                now,
                now,
            )
            for row in rows
        ],
    )
    conn.commit()


def _persist_membership_issuer_placeholders(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["issuer_key"])].append(row)
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO us_equity_census_issuers(
            run_id, issuer_key, cik, legal_name, display_name,
            identity_status, membership_status, listed_security_count,
            last_completed_stage, next_stage, processing_status,
            terminal_detail_json, provenance_json, created_at, updated_at
        ) VALUES(?, ?, ?, ?, ?, ?, 'DISCOVERED', ?, 'MEMBERSHIP', 'IDENTITY',
                 'IN_PROGRESS', '{}', '{}', ?, ?)
        ON CONFLICT(run_id, issuer_key) DO UPDATE SET
            cik = excluded.cik,
            legal_name = excluded.legal_name,
            display_name = excluded.display_name,
            identity_status = excluded.identity_status,
            listed_security_count = excluded.listed_security_count,
            last_completed_stage = excluded.last_completed_stage,
            next_stage = excluded.next_stage,
            processing_status = excluded.processing_status,
            updated_at = excluded.updated_at
        """,
        [
            (
                run_id,
                issuer_key,
                next((row.get("cik") for row in group if row.get("cik")), None),
                next(
                    (
                        str(row.get("issuer_legal_name") or "")
                        for row in group
                        if row.get("issuer_legal_name")
                    ),
                    "",
                ),
                next(
                    (
                        str(row.get("issuer_legal_name") or "")
                        for row in group
                        if row.get("issuer_legal_name")
                    ),
                    "",
                ),
                "RESOLVED" if any(row.get("cik") for row in group) else "NEEDS_DATA",
                len(group),
                now,
                now,
            )
            for issuer_key, group in sorted(grouped.items())
        ],
    )
    conn.commit()


def _cap_evidence_by_issuer(
    parsed: _ParsedSources,
    security_rows: Sequence[Mapping[str, Any]],
    *,
    retrieved_at: str,
) -> tuple[dict[str, list[CapEvidenceRow]], list[dict[str, Any]]]:
    issuer_by_ticker: dict[str, set[str]] = defaultdict(set)
    names_by_issuer: dict[str, list[str]] = defaultdict(list)
    for row in security_rows:
        issuer_by_ticker[normalize_ticker(row.get("ticker"))].add(str(row["issuer_key"]))
        if row.get("issuer_legal_name"):
            names_by_issuer[str(row["issuer_key"])].append(str(row["issuer_legal_name"]))
    result: dict[str, list[CapEvidenceRow]] = defaultdict(list)
    unmatched: list[dict[str, Any]] = []

    def add(
        *,
        provider: str,
        ticker: str,
        name: str,
        cap: int,
        source_url: str,
        as_of: str,
        rank: int | None,
        evidence_tier: str = "APPROVED_PROVIDER",
        resolution_role: str = "PRIMARY_CAP_SOURCE",
        evidence_detail: str | None = None,
        evidence_retrieved_at: str | None = None,
    ) -> None:
        issuer_keys = issuer_by_ticker.get(normalize_ticker(ticker), set())
        identity_match_method = "EXACT_TICKER"
        if len(issuer_keys) != 1:
            fuzzy_issuer = _best_unique_name_match(name, names_by_issuer)
            if fuzzy_issuer is not None:
                issuer_keys = {fuzzy_issuer}
                identity_match_method = "UNIQUE_ISSUER_NAME_TOKEN_MATCH"
        if len(issuer_keys) != 1:
            if cap >= BOUNDARY_INVESTIGATION_FLOOR_USD:
                unmatched.append(
                    {
                        "source_provider": provider,
                        "source_ticker": ticker,
                        "source_name": name,
                        "market_cap_usd": cap,
                        "issuer_match_count": len(issuer_keys),
                        "reason_code": "CAP_SOURCE_ISSUER_MATCH_UNRESOLVED",
                    }
                )
            return
        issuer_key = next(iter(issuer_keys))
        result[issuer_key].append(
            CapEvidenceRow(
                source_provider=provider,
                source_url=source_url,
                source_ticker=ticker,
                source_name=name,
                market_cap_usd=float(cap),
                as_of_date=as_of,
                retrieved_at=evidence_retrieved_at or retrieved_at,
                source_rank=rank,
                identity_match_method=identity_match_method,
                evidence_tier=evidence_tier,
                resolution_role=resolution_role,
                evidence_detail=evidence_detail,
            )
        )

    for row in parsed.companiesmarketcap:
        add(
            provider="COMPANIESMARKETCAP_USA",
            ticker=row.ticker,
            name=row.name,
            cap=row.market_cap_usd,
            source_url=row.source_url,
            as_of=row.as_of_date,
            rank=row.rank,
        )
    for row in parsed.stockanalysis:
        if row.market_cap_usd is None:
            continue
        add(
            provider="STOCKANALYSIS",
            ticker=row.symbol,
            name=row.name,
            cap=row.market_cap_usd,
            source_url=row.source_url,
            as_of=row.as_of_date,
            rank=row.source_row_number,
        )
    for row in parsed.terminal_cap_evidence:
        add(
            provider=row.source_provider,
            ticker=row.source_ticker,
            name=row.source_name,
            cap=int(row.market_cap_usd),
            source_url=row.source_url,
            as_of=row.as_of_date,
            rank=row.source_rank,
            evidence_tier=row.evidence_tier,
            resolution_role=row.resolution_role,
            evidence_detail=row.evidence_detail,
            evidence_retrieved_at=row.retrieved_at,
        )
    return result, unmatched


def _issuer_rows(
    conn: sqlite3.Connection,
    *,
    inputs: CensusInputPaths,
    parsed: _ParsedSources,
    security_rows: list[dict[str, Any]],
    as_of_date: str,
    retrieved_at: str,
    database_inputs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in security_rows:
        grouped[str(row["issuer_key"])].append(row)
    legacy = dict(database_inputs.get("sec_registrants") or {})
    sectors = dict(database_inputs.get("latest_sector_labels") or {})
    coverage_by_ticker = dict(database_inputs.get("coverage_by_ticker") or {})
    stockanalysis_by_ticker: dict[str, list[StockAnalysisSecurity]] = defaultdict(list)
    for row in parsed.stockanalysis:
        stockanalysis_by_ticker[normalize_ticker(row.symbol)].append(row)
    cap_by_issuer, unmatched_caps = _cap_evidence_by_issuer(
        parsed, security_rows, retrieved_at=retrieved_at
    )
    issuer_rows: list[dict[str, Any]] = []
    for issuer_key, securities in sorted(grouped.items()):
        cik = next((str(row.get("cik")) for row in securities if row.get("cik")), None)
        legacy_row = legacy.get(str(cik)) if cik else None
        profile = _load_submission_profile(inputs, str(cik)) if cik else None
        bdc_security_overrides = _apply_operating_bdc_security_override(profile, securities)
        cap_evidence = cap_by_issuer.get(issuer_key, [])
        cmc_tickers = [
            row.source_ticker
            for row in cap_evidence
            if row.source_provider == "COMPANIESMARKETCAP_USA"
        ]
        stock_tickers = [
            row.source_ticker for row in cap_evidence if row.source_provider == "STOCKANALYSIS"
        ]
        primary = select_primary_security(
            securities,
            preferred_source_tickers=(*cmc_tickers, *stock_tickers),
            legacy_primary_ticker=str(legacy_row["primary_ticker"]) if legacy_row else None,
        )
        primary_row = next(
            (row for row in securities if row["security_key"] == primary.security_key), None
        )
        for security in securities:
            is_primary = int(security["security_key"] == primary.security_key)
            security["is_primary_security"] = is_primary
            security["is_secondary_class"] = int(
                bool(primary.security_key)
                and not is_primary
                and security["security_type"] in COMMON_SECURITY_TYPES
            )
            security["issuer_relationship_type"] = (
                "PRIMARY"
                if is_primary
                else "SECONDARY_CLASS"
                if security["is_secondary_class"]
                else "SECONDARY_SECURITY"
            )
            security["related_security_key"] = primary.security_key

        cap = resolve_direct_cap_evidence(cap_evidence, primary_ticker=primary.ticker)
        raw_cap_band = cap.cap_band
        if raw_cap_band in {"large_cap", "mega_cap"}:
            db_cap_band = "large_and_mega"
        elif raw_cap_band == "mid":
            db_cap_band = "mid_cap"
        elif raw_cap_band == "small":
            db_cap_band = "small_cap"
        elif raw_cap_band == "micro":
            db_cap_band = "micro_cap"
        else:
            db_cap_band = "UNRESOLVED"

        sector_candidates: list[tuple[str, dict[str, Any]]] = []
        ticker_order = [primary.ticker] if primary.ticker else []
        ticker_order.extend(str(row["ticker"]) for row in securities)
        for ticker in ticker_order:
            if ticker and normalize_ticker(ticker) in sectors:
                sector_candidates.append(
                    (normalize_ticker(ticker), sectors[normalize_ticker(ticker)])
                )
        selected_sector = sector_candidates[0] if sector_candidates else None
        local_source_sector = selected_sector[1]["label"] if selected_sector else None
        local_sector_resolution = resolve_canonical_sector(local_source_sector)
        external_industry_row: StockAnalysisSecurity | None = None
        external_industry_resolution = None
        for ticker in ticker_order:
            candidates = stockanalysis_by_ticker.get(normalize_ticker(ticker), [])
            if not candidates:
                continue
            external_industry_row = candidates[0]
            external_industry_resolution = resolve_stockanalysis_industry(
                external_industry_row.industry
            )
            break
        if local_sector_resolution.contract is not None:
            source_sector = local_source_sector
            sector_resolution = local_sector_resolution
            source_sector_system = "sector_inference"
            sector_mapping_method = sector_resolution.reason_code
            sector_source_url = None
            sector_as_of_date = selected_sector[1]["as_of_date"] if selected_sector else None
        elif (
            external_industry_resolution is not None
            and external_industry_resolution.source_sector_label is not None
        ):
            source_sector = external_industry_resolution.source_sector_label
            sector_resolution = resolve_canonical_sector(source_sector)
            source_sector_system = "STOCKANALYSIS_INDUSTRY_EXACT"
            sector_mapping_method = external_industry_resolution.reason_code
            sector_source_url = external_industry_row.source_url if external_industry_row else None
            sector_as_of_date = external_industry_row.as_of_date if external_industry_row else None
        else:
            source_sector = local_source_sector
            sector_resolution = local_sector_resolution
            source_sector_system = "sector_inference" if local_source_sector else None
            sector_mapping_method = (
                external_industry_resolution.reason_code
                if external_industry_resolution is not None
                else sector_resolution.reason_code
            )
            sector_source_url = external_industry_row.source_url if external_industry_row else None
            sector_as_of_date = (
                selected_sector[1]["as_of_date"]
                if selected_sector
                else external_industry_row.as_of_date
                if external_industry_row
                else None
            )
        coverage = coverage_by_ticker.get(
            normalize_ticker(primary.ticker or securities[0]["ticker"]),
            {
                "facts": {"count": 0, "latest_period_end": None},
                "filings": {
                    "count": 0,
                    "latest_filing_date": None,
                    "latest_accession": None,
                },
                "packet": None,
            },
        )

        target_listing_count = sum(
            normalize_exchange(row["exchange_name"]) in TARGET_EXCHANGES
            and row.get("listing_status") == "ACTIVE"
            for row in securities
        )
        uncorroborated_target_listing_count = sum(
            normalize_exchange(row["exchange_name"]) in TARGET_EXCHANGES
            and row.get("listing_status") == "SEC_REGISTRY_ONLY_UNCORROBORATED"
            for row in securities
        )
        all_security_types = {str(row["security_type"]) for row in securities}
        source_registries = {str(row.get("source_registry") or "") for row in securities}
        if source_registries == {"FIXED_642_SECURITY_REPLAY_COMPARISON"}:
            scope_status = "OUT_OF_SCOPE"
            scope_reason = "HISTORICAL_COMPARISON_NOT_CURRENT_REGISTRY"
        elif target_listing_count == 0 and any(
            provider in source_registries
            for provider in ("COMPANIESMARKETCAP_USA", "STOCKANALYSIS")
        ):
            scope_status = "OUT_OF_SCOPE"
            scope_reason = "CAP_SOURCE_NOT_IN_CURRENT_REGISTRIES"
        elif target_listing_count == 0 and uncorroborated_target_listing_count:
            scope_status = "OUT_OF_SCOPE"
            scope_reason = "ABSENT_FROM_CURRENT_NASDAQ_TRADER_DIRECTORY"
        elif target_listing_count == 0:
            scope_status = "OUT_OF_SCOPE"
            scope_reason = "NO_TARGET_EXCHANGE_LISTING"
        else:
            scope_status = "PENDING_EVIDENCE"
            scope_reason = cap.discrepancy_code or primary.method

        if profile is not None:
            operating_status = profile.operating_status
            is_operating = int(profile.operating_status in {"OPERATING", "FOREIGN_FILER"})
            is_us = (
                1
                if profile.domicile_status == "US_DOMICILED"
                else 0
                if profile.domicile_status == "FOREIGN_DOMICILED"
                else None
            )
            is_us_listed_foreign = int(
                profile.operating_status == "FOREIGN_FILER"
                or profile.domicile_status == "FOREIGN_DOMICILED"
            )
            domicile_country = profile.domicile_country_code
            domicile_jurisdiction = profile.domicile_jurisdiction
            filer_status = profile.filer_status
            latest_annual_date = profile.latest_annual_filing_date
            latest_annual_accession = profile.latest_annual_filing_accession
            domicile_url = profile.source_url
            legal_name = profile.legal_name
            domicile_confidence = profile.domicile_confidence
        else:
            legacy_status = str(legacy_row["operating_status"]) if legacy_row else ""
            if legacy_status == "OPERATING":
                operating_status = "OPERATING"
                is_operating = 1
                is_us_listed_foreign = None
            elif legacy_status == "FOREIGN_FILER_ONLY":
                operating_status = "FOREIGN_FILER"
                is_operating = 1
                is_us_listed_foreign = 1
            elif legacy_status in {"BLANK_CHECK", "BLANK_CHECK_SIC"}:
                operating_status = "SHELL_OR_BLANK_CHECK"
                is_operating = 0
                is_us_listed_foreign = None
            elif legacy_status in {"FUND_OR_TRUST_SIC"}:
                operating_status = "INVESTMENT_COMPANY"
                is_operating = 0
                is_us_listed_foreign = None
            else:
                operating_status = "NEEDS_DATA_OPERATING_STATUS"
                is_operating = None
                is_us_listed_foreign = None
            is_us = None
            domicile_country = None
            domicile_jurisdiction = None
            filer_status = None
            latest_annual_date = legacy_row["latest_operating_form_date"] if legacy_row else None
            latest_annual_accession = None
            domicile_url = None
            legal_name = next(
                (
                    str(row.get("issuer_legal_name") or "")
                    for row in securities
                    if row.get("issuer_legal_name")
                ),
                "",
            )
            domicile_confidence = "LOW"

        issuer_type = (
            "FUND_SECURITY_SET"
            if all_security_types
            and all_security_types <= {SECURITY_ETF, SECURITY_ETN, SECURITY_CLOSED_END_FUND}
            else "OPERATING_ISSUER"
            if is_operating
            else "UNRESOLVED_ISSUER_TYPE"
        )
        facts = coverage["facts"]
        filings = coverage["filings"]
        packet = coverage["packet"]
        cap_payload = {
            "raw_cap_band": raw_cap_band,
            "discrepancy_code": cap.discrepancy_code,
            "evidence": [asdict(row) for row in cap.evidence],
        }
        sector_payload = {
            "candidate_ticker_labels": [
                {"ticker": ticker, **metadata} for ticker, metadata in sector_candidates
            ],
            "taxonomy_hash": CANONICAL_SECTOR_TAXONOMY_HASH,
            "external_industry_resolution": (
                external_industry_resolution.to_dict()
                if external_industry_resolution is not None
                else None
            ),
            "external_industry_taxonomy": {
                "taxonomy_version": STOCKANALYSIS_INDUSTRY_TAXONOMY_VERSION,
                "taxonomy_hash": STOCKANALYSIS_INDUSTRY_TAXONOMY_HASH,
            },
            "cross_source_sector_disagreement": bool(
                local_sector_resolution.contract is not None
                and external_industry_resolution is not None
                and external_industry_resolution.source_sector_label is not None
                and local_sector_resolution.contract.canonical_sector_id
                != external_industry_resolution.source_sector_label
            ),
        }
        issuer_rows.append(
            {
                "issuer_key": issuer_key,
                "cik": cik,
                "legal_name": legal_name,
                "display_name": legal_name,
                "identity_status": "RESOLVED" if cik else "NEEDS_DATA",
                "identity_source_provider": "SEC_EXCHANGE_REGISTRY"
                if cik
                else "CAP_OR_LISTING_SOURCE",
                "identity_source_url": "https://www.sec.gov/files/company_tickers_exchange.json"
                if cik
                else None,
                "identity_as_of_date": as_of_date,
                "identity_retrieved_at": retrieved_at,
                "identity_confidence": "HIGH" if cik else "LOW",
                "issuer_type": issuer_type,
                "operating_structure": "ADR"
                if any(row.get("is_adr") for row in securities)
                else "DOMESTIC_OR_DIRECT_LISTING",
                "operating_status": operating_status,
                "is_operating_company": is_operating,
                "membership_status": "DISCOVERED",
                "scope_status": scope_status,
                "scope_reason_code": scope_reason,
                "domicile_country_code": domicile_country,
                "domicile_jurisdiction": domicile_jurisdiction,
                "is_us_domiciled": is_us,
                "is_us_listed_foreign_issuer": is_us_listed_foreign,
                "filer_status": filer_status,
                "domicile_source_provider": "SEC_SUBMISSIONS"
                if profile
                else "LEGACY_REGISTRY_ONLY",
                "domicile_source_url": domicile_url,
                "domicile_as_of_date": as_of_date if profile else None,
                "domicile_retrieved_at": retrieved_at if profile else None,
                "domicile_confidence": domicile_confidence,
                "primary_security_key": primary.security_key,
                "primary_ticker": primary.ticker,
                "primary_exchange_name": primary_row["exchange_name"] if primary_row else None,
                "primary_exchange_mic": primary_row["exchange_mic"] if primary_row else None,
                "primary_selection_status": primary.status,
                "primary_selection_method": primary.method,
                "primary_selection_source_url": cap.source_url,
                "primary_selection_as_of_date": as_of_date,
                "primary_selection_retrieved_at": retrieved_at,
                "primary_selection_confidence": primary.confidence,
                "listed_security_count": len(securities),
                "market_cap_status": cap.status,
                "market_cap_usd": cap.market_cap_usd,
                "market_cap_currency": "USD" if cap.market_cap_usd is not None else None,
                "market_cap_as_of_date": cap.as_of_date,
                "market_cap_method": cap.method,
                "market_cap_source_provider": cap.source_provider,
                "market_cap_source_url": cap.source_url,
                "market_cap_retrieved_at": retrieved_at if cap.market_cap_usd is not None else None,
                "market_cap_confidence": cap.confidence,
                "cap_security_key": primary.security_key,
                "cap_derivation_json": _json_dumps(cap_payload),
                "cap_band": db_cap_band,
                "cap_band_status": cap.status,
                "canonical_sector": sector_resolution.contract.canonical_sector_id
                if sector_resolution.contract
                else None,
                "source_sector_label": source_sector,
                "source_sector_system": source_sector_system,
                "sector_status": sector_resolution.disposition,
                "sector_mapping_method": sector_mapping_method,
                "sector_mapping_version": sector_resolution.taxonomy_version,
                "sector_source_url": sector_source_url,
                "sector_as_of_date": sector_as_of_date,
                "sector_confidence": "HIGH" if sector_resolution.contract else "LOW",
                "sector_provenance_json": _json_dumps(sector_payload),
                "facts_status": "AVAILABLE" if facts["count"] else "NEEDS_DATA",
                "facts_as_of_date": facts["latest_period_end"],
                "facts_source_provider": "SEC_COMPANYFACTS_LOCAL",
                "facts_source_url": None,
                "facts_detail_json": _json_dumps(facts),
                "filings_status": "AVAILABLE"
                if filings["count"] or latest_annual_date
                else "NEEDS_DATA",
                "latest_annual_filing_date": latest_annual_date or filings["latest_filing_date"],
                "latest_annual_filing_accession": latest_annual_accession
                or filings["latest_accession"],
                "filings_source_provider": "SEC_SUBMISSIONS_AND_LOCAL_FILINGS"
                if profile
                else "LOCAL_FILINGS",
                "filings_source_url": profile.source_url if profile else None,
                "filings_detail_json": _json_dumps(filings),
                "packet_status": "AVAILABLE" if packet else "NEEDS_DATA",
                "packet_as_of_date": packet["as_of_date"] if packet else None,
                "packet_path": packet["packet_path"] if packet else None,
                "packet_sha256": packet["packet_hash"] if packet else None,
                "packet_detail_json": _json_dumps(packet or {}),
                "last_completed_stage": "COMPANY_PACKET",
                "next_stage": "TERMINAL_DISPOSITION",
                "processing_status": "IN_PROGRESS",
                "terminal_disposition": None,
                "terminal_reason_code": None,
                "terminal_detail_json": "{}",
                "terminal_at": None,
                "provenance_json": _json_dumps(
                    {
                        "primary_selection_detail": primary.detail,
                        "operating_bdc_security_overrides": bdc_security_overrides,
                        "domicile_method": profile.domicile_method if profile else "UNRESOLVED",
                        "taxonomy": canonical_sector_taxonomy_manifest(),
                    }
                ),
            }
        )
    return issuer_rows, unmatched_caps


_ISSUER_DB_COLUMNS = (
    "issuer_key",
    "cik",
    "legal_name",
    "display_name",
    "identity_status",
    "identity_source_provider",
    "identity_source_url",
    "identity_as_of_date",
    "identity_retrieved_at",
    "identity_confidence",
    "issuer_type",
    "operating_structure",
    "operating_status",
    "is_operating_company",
    "membership_status",
    "scope_status",
    "scope_reason_code",
    "domicile_country_code",
    "domicile_jurisdiction",
    "is_us_domiciled",
    "is_us_listed_foreign_issuer",
    "filer_status",
    "domicile_source_provider",
    "domicile_source_url",
    "domicile_as_of_date",
    "domicile_retrieved_at",
    "domicile_confidence",
    "primary_security_key",
    "primary_ticker",
    "primary_exchange_name",
    "primary_exchange_mic",
    "primary_selection_status",
    "primary_selection_method",
    "primary_selection_source_url",
    "primary_selection_as_of_date",
    "primary_selection_retrieved_at",
    "primary_selection_confidence",
    "listed_security_count",
    "market_cap_status",
    "market_cap_usd",
    "market_cap_currency",
    "market_cap_as_of_date",
    "market_cap_method",
    "market_cap_source_provider",
    "market_cap_source_url",
    "market_cap_retrieved_at",
    "market_cap_confidence",
    "cap_security_key",
    "cap_derivation_json",
    "cap_band",
    "cap_band_status",
    "canonical_sector",
    "source_sector_label",
    "source_sector_system",
    "sector_status",
    "sector_mapping_method",
    "sector_mapping_version",
    "sector_source_url",
    "sector_as_of_date",
    "sector_confidence",
    "sector_provenance_json",
    "facts_status",
    "facts_as_of_date",
    "facts_source_provider",
    "facts_source_url",
    "facts_detail_json",
    "filings_status",
    "latest_annual_filing_date",
    "latest_annual_filing_accession",
    "filings_source_provider",
    "filings_source_url",
    "filings_detail_json",
    "packet_status",
    "packet_as_of_date",
    "packet_path",
    "packet_sha256",
    "packet_detail_json",
    "last_completed_stage",
    "next_stage",
    "processing_status",
    "terminal_disposition",
    "terminal_reason_code",
    "terminal_detail_json",
    "terminal_at",
    "provenance_json",
)


def _persist_issuer_rows(
    conn: sqlite3.Connection, *, run_id: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    now = utc_now_iso()
    columns = ("run_id", *_ISSUER_DB_COLUMNS, "created_at", "updated_at")
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(
        f"{column}=excluded.{column}"
        for column in (*_ISSUER_DB_COLUMNS, "updated_at")
        if column != "issuer_key"
    )
    sql = (
        f"INSERT INTO us_equity_census_issuers({', '.join(columns)}) VALUES({placeholders}) "
        f"ON CONFLICT(run_id, issuer_key) DO UPDATE SET {updates}"
    )
    conn.executemany(
        sql,
        [
            (
                run_id,
                *(row.get(column) for column in _ISSUER_DB_COLUMNS),
                now,
                now,
            )
            for row in rows
        ],
    )
    conn.commit()


def _stage_postcondition_summary(
    stage: str,
    *,
    issuer_rows: Sequence[Mapping[str, Any]],
    security_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and summarize the materialized contract for one census stage."""

    if stage not in STAGES[:-1]:
        raise ValueError(f"stage postconditions are not defined for {stage!r}")

    issuer_keys = [str(row.get("issuer_key") or "") for row in issuer_rows]
    security_keys = [str(row.get("security_key") or "") for row in security_rows]
    issuer_key_set = set(issuer_keys)
    linked_by_issuer: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in security_rows:
        linked_by_issuer[str(row.get("issuer_key") or "")].append(row)

    if stage == "MEMBERSHIP":
        checks = {
            "unique_nonempty_security_keys": bool(security_keys)
            and len(security_keys) == len(set(security_keys))
            and all(security_keys),
            "source_registry_explicit": all(
                bool(row.get("source_registry")) for row in security_rows
            ),
            "issuer_link_explicit": all(bool(row.get("issuer_key")) for row in security_rows),
        }
    elif stage == "IDENTITY":
        checks = {
            "unique_nonempty_issuer_keys": bool(issuer_keys)
            and len(issuer_keys) == len(set(issuer_keys))
            and all(issuer_keys),
            "unique_nonempty_security_keys": bool(security_keys)
            and len(security_keys) == len(set(security_keys))
            and all(security_keys),
            "identity_status_explicit": all(
                row.get("identity_status") in {"RESOLVED", "NEEDS_DATA"}
                for row in (*issuer_rows, *security_rows)
            ),
            "every_security_links_to_one_issuer": all(
                str(row.get("issuer_key") or "") in issuer_key_set for row in security_rows
            ),
        }
    elif stage == "DOMICILE_LISTING_STATUS":
        checks = {
            "listing_status_explicit": all(
                bool(row.get("listing_status")) for row in security_rows
            ),
            "exchange_membership_resolved_or_explicitly_absent": all(
                bool(row.get("exchange_name"))
                or row.get("listing_status")
                in {
                    "HISTORICAL_COMPARISON_ONLY",
                    "NEEDS_DATA",
                    "SEC_REGISTRY_ONLY_UNCORROBORATED",
                }
                for row in security_rows
            ),
            "issuer_scope_status_explicit": all(
                row.get("scope_status") in {"PENDING_EVIDENCE", "OUT_OF_SCOPE"}
                and bool(row.get("scope_reason_code"))
                for row in issuer_rows
            ),
            "operating_status_explicit": all(
                bool(row.get("operating_status")) for row in issuer_rows
            ),
            "domicile_is_tristate": all(
                row.get("is_us_domiciled") in {0, 1, None} for row in issuer_rows
            ),
        }
    elif stage == "SECURITY_TYPE":
        checks = {
            "security_type_allowed": all(
                row.get("security_type")
                in {
                    SECURITY_COMMON,
                    SECURITY_COMMON_EQUIVALENT,
                    SECURITY_ADR,
                    SECURITY_ETF,
                    SECURITY_ETN,
                    SECURITY_CLOSED_END_FUND,
                    SECURITY_PREFERRED,
                    SECURITY_WARRANT,
                    SECURITY_RIGHT,
                    SECURITY_UNIT,
                    SECURITY_DEBT,
                    SECURITY_UNKNOWN,
                }
                for row in security_rows
            ),
            "security_type_status_explicit": all(
                row.get("security_type_status") in {"RESOLVED", "NEEDS_DATA"}
                for row in security_rows
            ),
            "common_equity_flag_explicit": all(
                row.get("is_common_equity") in {0, 1} for row in security_rows
            ),
        }
    elif stage == "PRIMARY_SECURITY_SELECTION":
        primary_contracts: list[bool] = []
        for issuer in issuer_rows:
            linked = linked_by_issuer.get(str(issuer.get("issuer_key") or ""), [])
            primaries = [row for row in linked if row.get("is_primary_security") == 1]
            status = issuer.get("primary_selection_status")
            primary_contracts.append(
                status in {"RESOLVED", "NEEDS_DATA"}
                and len(primaries) <= 1
                and (
                    status != "RESOLVED"
                    or (
                        len(primaries) == 1
                        and issuer.get("primary_security_key") == primaries[0].get("security_key")
                    )
                )
                and (status != "NEEDS_DATA" or not primaries)
            )
        checks = {"one_or_explicitly_unresolved_primary_per_issuer": all(primary_contracts)}
    elif stage == "ISSUER_DEDUPLICATION":
        checks = {
            "one_row_per_issuer_key": len(issuer_keys) == len(set(issuer_keys)),
            "all_security_relationships_materialized": all(
                row.get("issuer_relationship_type")
                in {"PRIMARY", "SECONDARY_CLASS", "SECONDARY_SECURITY"}
                for row in security_rows
            ),
            "listed_security_counts_reconcile": all(
                int(row.get("listed_security_count") or 0)
                == len(linked_by_issuer.get(str(row.get("issuer_key") or ""), []))
                for row in issuer_rows
            ),
        }
    elif stage == "MARKET_CAP_RESOLUTION":
        checks = {
            "market_cap_status_explicit": all(
                row.get("market_cap_status") in {"RESOLVED", "NEEDS_DATA"} for row in issuer_rows
            ),
            "resolved_caps_are_positive": all(
                row.get("market_cap_status") != "RESOLVED"
                or (
                    float(row.get("market_cap_usd") or 0) > 0
                    and bool(row.get("market_cap_method"))
                    and bool(row.get("market_cap_source_provider"))
                    and bool(row.get("market_cap_as_of_date"))
                )
                for row in issuer_rows
            ),
            "unresolved_caps_have_no_value": all(
                row.get("market_cap_status") != "NEEDS_DATA" or row.get("market_cap_usd") is None
                for row in issuer_rows
            ),
        }
    elif stage == "CAP_BAND_ASSIGNMENT":
        checks = {
            "cap_band_matches_resolution_status": all(
                (
                    row.get("market_cap_status") == "RESOLVED"
                    and row.get("cap_band")
                    in {"mega_cap", "large_and_mega", "mid_cap", "small_cap", "micro_cap"}
                )
                or (
                    row.get("market_cap_status") == "NEEDS_DATA"
                    and row.get("cap_band") == "UNRESOLVED"
                )
                for row in issuer_rows
            )
        }
    elif stage == "SECTOR_CLASSIFICATION":
        checks = {
            "sector_status_explicit": all(
                row.get("sector_status") in {"RESOLVED", "NEEDS_DATA"} for row in issuer_rows
            ),
            "resolved_sector_contract_complete": all(
                row.get("sector_status") != "RESOLVED"
                or (
                    bool(row.get("canonical_sector"))
                    and row.get("sector_mapping_version") == CANONICAL_SECTOR_TAXONOMY_VERSION
                )
                for row in issuer_rows
            ),
            "unresolved_sector_has_no_canonical_label": all(
                row.get("sector_status") != "NEEDS_DATA" or row.get("canonical_sector") is None
                for row in issuer_rows
            ),
        }
    elif stage == "FACTS_FILINGS_AVAILABILITY":
        checks = {
            "facts_status_explicit": all(
                row.get("facts_status") in {"AVAILABLE", "NEEDS_DATA"} for row in issuer_rows
            ),
            "filings_status_explicit": all(
                row.get("filings_status") in {"AVAILABLE", "NEEDS_DATA"} for row in issuer_rows
            ),
        }
    else:
        checks = {
            "packet_status_explicit": all(
                row.get("packet_status") in {"AVAILABLE", "NEEDS_DATA"} for row in issuer_rows
            ),
            "available_packet_has_path": all(
                row.get("packet_status") != "AVAILABLE"
                or (
                    bool(row.get("packet_as_of_date"))
                    and bool(row.get("packet_path"))
                    and bool(row.get("packet_sha256"))
                )
                for row in issuer_rows
            ),
            "unavailable_packet_has_no_artifact": all(
                row.get("packet_status") != "NEEDS_DATA"
                or (not row.get("packet_path") and not row.get("packet_sha256"))
                for row in issuer_rows
            ),
        }

    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise ValueError(f"{stage} stage postcondition failed: {', '.join(failed)}")
    return {
        "issuer_count": len(issuer_rows)
        if issuer_rows
        else len({str(row.get("issuer_key") or "") for row in security_rows}),
        "security_count": len(security_rows),
        "checkpoint_stage": stage,
        "next_stage": STAGES[STAGES.index(stage) + 1],
        "postconditions": checks,
    }


def _checkpoint_stage_rows(
    stage: str,
    *,
    issuer_rows: list[dict[str, Any]],
    security_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = _stage_postcondition_summary(
        stage, issuer_rows=issuer_rows, security_rows=security_rows
    )
    next_stage = STAGES[STAGES.index(stage) + 1]
    for row in (*issuer_rows, *security_rows):
        row["last_completed_stage"] = stage
        row["next_stage"] = next_stage
        row["processing_status"] = "IN_PROGRESS"
    return summary


def _finalize_rows(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    issuer_rows: list[dict[str, Any]],
    security_rows: list[dict[str, Any]],
) -> None:
    now = utc_now_iso()
    securities_by_key = {
        str(row.get("security_key") or ""): row
        for row in security_rows
        if str(row.get("security_key") or "")
    }
    terminal_by_issuer: dict[str, str] = {}
    for row in issuer_rows:
        primary_security = securities_by_key.get(str(row.get("primary_security_key") or ""))
        disposition, reason = _terminal_for_issuer(row, primary_security)
        row["terminal_disposition"] = disposition
        row["terminal_reason_code"] = reason
        row["terminal_detail_json"] = _json_dumps(
            {
                "scope_status": row.get("scope_status"),
                "primary_selection_status": row.get("primary_selection_status"),
                "primary_security_type": (
                    primary_security.get("security_type") if primary_security is not None else None
                ),
                "market_cap_status": row.get("market_cap_status"),
                "cap_band": row.get("cap_band"),
                "sector_status": row.get("sector_status"),
            }
        )
        row["terminal_at"] = now
        row["last_completed_stage"] = "TERMINAL_DISPOSITION"
        row["next_stage"] = None
        row["processing_status"] = (
            "COMPLETED"
            if disposition.startswith(("ADMITTED_", "ELIGIBLE_"))
            else "OUT_OF_SCOPE"
            if disposition.startswith("OUT_OF_SCOPE_")
            else "NEEDS_DATA"
        )
        terminal_by_issuer[str(row["issuer_key"])] = disposition

    for row in security_rows:
        disposition, reason = _security_terminal(
            row, terminal_by_issuer.get(str(row["issuer_key"]))
        )
        row["terminal_disposition"] = disposition
        row["terminal_reason_code"] = reason
        row["terminal_detail_json"] = _json_dumps(
            {
                "issuer_terminal_disposition": terminal_by_issuer.get(str(row["issuer_key"])),
                "security_type": row.get("security_type"),
                "exchange": row.get("exchange_name"),
            }
        )
        row["terminal_at"] = now
        row["last_completed_stage"] = "TERMINAL_DISPOSITION"
        row["next_stage"] = None
        row["processing_status"] = (
            "COMPLETED"
            if disposition.startswith(("ADMITTED_", "SECONDARY_"))
            else "OUT_OF_SCOPE"
            if disposition.startswith("OUT_OF_SCOPE_")
            else "NEEDS_DATA"
        )

    _persist_security_rows(conn, run_id=run_id, rows=security_rows)
    _persist_issuer_rows(conn, run_id=run_id, rows=issuer_rows)


def _fixed_cohort_reconciliation(
    fixed_tickers: set[str], security_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_ticker: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in security_rows:
        by_ticker[normalize_ticker(row.get("ticker"))].append(row)
    output: list[dict[str, Any]] = []
    for ticker in sorted(fixed_tickers):
        matches = by_ticker.get(ticker, [])
        issuer_keys = sorted({str(row.get("issuer_key")) for row in matches})
        output.append(
            {
                "fixed_cohort_ticker": ticker,
                "match_status": "MATCHED"
                if len(issuer_keys) == 1
                else "UNMATCHED"
                if not issuer_keys
                else "AMBIGUOUS",
                "matched_security_count": len(matches),
                "matched_issuer_count": len(issuer_keys),
                "issuer_keys": _json_dumps(issuer_keys),
                "terminal_dispositions": _json_dumps(
                    sorted({str(row.get("terminal_disposition")) for row in matches})
                ),
            }
        )
    return output


def _fixed_cohort_issuer_count(
    fixed_tickers: set[str], security_rows: Sequence[Mapping[str, Any]]
) -> int:
    return len(
        {
            str(row.get("issuer_key"))
            for row in security_rows
            if normalize_ticker(row.get("ticker")) in fixed_tickers
        }
    )


def _write_outputs(
    *,
    conn: sqlite3.Connection,
    run_id: str,
    as_of_date: str,
    output_dir: Path,
    parsed: _ParsedSources,
    inputs: CensusInputPaths,
    unmatched_caps: Sequence[Mapping[str, Any]],
    source_snapshots: Sequence[Mapping[str, Any]],
    provider_free_replay_verified: bool,
    sector_population_promoted: bool,
    semantic_output_fingerprint: str,
    cost_ledger: CensusCostLedger,
) -> tuple[dict[str, Path], dict[str, int], dict[str, bool], str]:
    security_rows = _read_table(conn, "us_equity_census_securities", run_id)
    issuer_rows = _read_table(conn, "us_equity_census_issuers", run_id)
    fixed_tickers = _parse_fixed_cohort(inputs)
    fixed_rows = _fixed_cohort_reconciliation(fixed_tickers, security_rows)
    fixed_issuer_count = _fixed_cohort_issuer_count(fixed_tickers, security_rows)
    fixed_issuer_keys = {
        str(row.get("issuer_key"))
        for row in security_rows
        if normalize_ticker(row.get("ticker")) in fixed_tickers
    }

    security_fields = (
        "security_key",
        "ticker",
        "listed_name",
        "exchange_name",
        "listing_status",
        "issuer_key",
        "issuer_relationship_type",
        "share_class",
        "is_primary_security",
        "is_secondary_class",
        "is_duplicate_listing",
        "security_type",
        "security_type_status",
        "is_common_equity",
        "is_adr",
        "adr_ratio",
        "identity_status",
        "identity_confidence",
        "terminal_disposition",
        "terminal_reason_code",
        "source_registry",
        "listing_source_url",
        "provenance_json",
    )
    issuer_fields = (
        "issuer_key",
        "cik",
        "legal_name",
        "identity_status",
        "issuer_type",
        "operating_status",
        "is_operating_company",
        "domicile_country_code",
        "domicile_jurisdiction",
        "is_us_domiciled",
        "is_us_listed_foreign_issuer",
        "filer_status",
        "primary_ticker",
        "primary_exchange_name",
        "primary_selection_status",
        "primary_selection_method",
        "listed_security_count",
        "market_cap_status",
        "market_cap_usd",
        "market_cap_as_of_date",
        "market_cap_method",
        "market_cap_source_provider",
        "market_cap_source_url",
        "market_cap_confidence",
        "cap_band",
        "canonical_sector",
        "source_sector_label",
        "sector_status",
        "sector_mapping_method",
        "sector_mapping_version",
        "facts_status",
        "filings_status",
        "packet_status",
        "terminal_disposition",
        "terminal_reason_code",
        "provenance_json",
    )
    cap_fields = (
        "issuer_key",
        "cik",
        "primary_ticker",
        "market_cap_status",
        "market_cap_usd",
        "market_cap_as_of_date",
        "market_cap_method",
        "market_cap_source_provider",
        "market_cap_source_url",
        "market_cap_confidence",
        "cap_band",
        "cap_derivation_json",
        "terminal_disposition",
    )
    sector_fields = (
        "issuer_key",
        "cik",
        "primary_ticker",
        "source_sector_label",
        "canonical_sector",
        "sector_status",
        "sector_mapping_method",
        "sector_mapping_version",
        "sector_confidence",
        "sector_provenance_json",
        "terminal_disposition",
    )
    large_rows = [
        {
            **row,
            "in_fixed_642_security_cohort": int(str(row.get("issuer_key")) in fixed_issuer_keys),
            "new_vs_fixed_cohort": int(
                row.get("terminal_disposition") == "ADMITTED_LARGE_AND_MEGA"
                and str(row.get("issuer_key")) not in fixed_issuer_keys
            ),
        }
        for row in issuer_rows
        if row.get("cap_band") == "large_and_mega"
        or row.get("terminal_disposition")
        in {"NEEDS_DATA_MARKET_CAP", "NEEDS_DATA_IDENTITY", "NEEDS_DATA_SECTOR"}
    ]
    gap_fields = (
        "issuer_key",
        "cik",
        "legal_name",
        "primary_ticker",
        "primary_exchange_name",
        "identity_status",
        "operating_status",
        "is_us_domiciled",
        "market_cap_status",
        "market_cap_usd",
        "market_cap_source_provider",
        "cap_band",
        "source_sector_label",
        "canonical_sector",
        "sector_status",
        "terminal_disposition",
        "terminal_reason_code",
        "in_fixed_642_security_cohort",
        "new_vs_fixed_cohort",
    )

    artifacts = {
        "security_census": output_dir / "security_census.csv",
        "issuer_census": output_dir / "issuer_census.csv",
        "cap_resolution": output_dir / "cap_resolution.csv",
        "sector_reconciliation": output_dir / "sector_reconciliation.csv",
        "large_and_mega_gap_report": output_dir / "large_and_mega_gap_report.csv",
        "fixed_cohort_reconciliation": output_dir / "fixed_642_security_reconciliation.csv",
        "sector_scan_population_additions": output_dir / "sector_scan_population_additions.csv",
        "coverage_summary": output_dir / "coverage_summary.json",
        "coverage_report": output_dir / "coverage_report.md",
        "acceptance_report": output_dir / "acceptance_report.json",
        "parse_issues": output_dir / "parse_issues.json",
        "source_discrepancies": output_dir / "source_discrepancies.json",
        "source_manifest": output_dir / "source_manifest.json",
        "checkpoint": output_dir / "checkpoint.json",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(artifacts["security_census"], security_rows, security_fields)
    _write_csv(artifacts["issuer_census"], issuer_rows, issuer_fields)
    _write_csv(artifacts["cap_resolution"], issuer_rows, cap_fields)
    _write_csv(artifacts["sector_reconciliation"], issuer_rows, sector_fields)
    _write_csv(artifacts["large_and_mega_gap_report"], large_rows, gap_fields)
    _write_csv(
        artifacts["fixed_cohort_reconciliation"],
        fixed_rows,
        (
            "fixed_cohort_ticker",
            "match_status",
            "matched_security_count",
            "matched_issuer_count",
            "issuer_keys",
            "terminal_dispositions",
        ),
    )
    additions = [
        row
        for row in large_rows
        if row.get("terminal_disposition") == "ADMITTED_LARGE_AND_MEGA"
        and row.get("new_vs_fixed_cohort") == 1
    ]
    _write_csv(
        artifacts["sector_scan_population_additions"],
        additions,
        (
            "primary_ticker",
            "cik",
            "legal_name",
            "source_sector_label",
            "canonical_sector",
            "market_cap_usd",
            "market_cap_source_provider",
        ),
    )

    terminal_validation = validate_terminal_dispositions(security_rows, issuer_rows)
    disposition_counts = Counter(str(row["terminal_disposition"]) for row in issuer_rows)
    security_disposition_counts = Counter(str(row["terminal_disposition"]) for row in security_rows)
    admitted = [
        row for row in issuer_rows if row["terminal_disposition"] == "ADMITTED_LARGE_AND_MEGA"
    ]
    fixed_matched = sum(row["match_status"] == "MATCHED" for row in fixed_rows)
    counts = {
        "discovered_securities": len(security_rows),
        "deduplicated_issuers": len(issuer_rows),
        "us_domiciled_issuers": sum(row.get("is_us_domiciled") == 1 for row in issuer_rows),
        "in_scope_common_equity_securities": sum(
            normalize_exchange(row.get("exchange_name")) in TARGET_EXCHANGES
            and row.get("listing_status") == "ACTIVE"
            and row.get("security_type") in COMMON_SECURITY_TYPES
            for row in security_rows
        ),
        "primary_common_securities": sum(
            row.get("terminal_disposition") == "ADMITTED_PRIMARY_COMMON_EQUITY"
            for row in security_rows
        ),
        "secondary_or_duplicate_securities": sum(
            row.get("terminal_disposition") == "SECONDARY_OR_DUPLICATE_SECURITY"
            for row in security_rows
        ),
        "admitted_large_and_mega_issuers": len(admitted),
        "admitted_large_cap_issuers": sum(
            10_000_000_000.0 <= float(row["market_cap_usd"]) < 200_000_000_000.0 for row in admitted
        ),
        "admitted_mega_cap_issuers": sum(
            float(row["market_cap_usd"]) >= 200_000_000_000.0 for row in admitted
        ),
        "foreign_issuer_dispositions": disposition_counts["OUT_OF_SCOPE_FOREIGN_ISSUER"],
        "unresolved_identity_issuers": disposition_counts["NEEDS_DATA_IDENTITY"],
        "unresolved_cap_issuers": disposition_counts["NEEDS_DATA_MARKET_CAP"],
        "unresolved_sector_issuers": disposition_counts["NEEDS_DATA_SECTOR"],
        "fixed_cohort_security_tickers": len(fixed_tickers),
        "fixed_cohort_matched_tickers": fixed_matched,
        "fixed_cohort_reconciled_issuers": fixed_issuer_count,
        "new_large_and_mega_scan_population": len(additions),
        "cap_source_unmatched_large_rows": len(unmatched_caps),
        "parse_issues": len(parsed.issues),
        "needs_data_issuers": sum(
            str(row.get("terminal_disposition") or "").startswith("NEEDS_DATA_")
            for row in issuer_rows
        ),
        "packet_ready_admitted_issuers": sum(
            row.get("terminal_disposition") == "ADMITTED_LARGE_AND_MEGA"
            and row.get("packet_status") == "AVAILABLE"
            for row in issuer_rows
        ),
        "screened_issuers": 0,
        "underwritten_issuers": 0,
        "validated_actionable_issuers": 0,
    }
    acceptance = _issuer_acceptance_checks(
        issuer_rows,
        cap_source_unmatched=len(unmatched_caps),
        fixed_cohort_count=len(fixed_tickers),
        fixed_cohort_issuer_count=fixed_issuer_count,
        source_checks=_source_acceptance_checks(parsed, inputs=inputs, as_of_date=as_of_date),
        provider_free_replay_verified=provider_free_replay_verified,
        sector_population_promoted=sector_population_promoted,
        cost_ledger=cost_ledger,
    )
    acceptance["every_security_has_one_terminal_disposition"] = bool(
        terminal_validation["security_terminal_complete"]
    )
    acceptance["security_keys_unique"] = bool(terminal_validation["unique_security_keys"])
    acceptance["issuer_keys_unique"] = bool(terminal_validation["unique_issuer_keys"])
    for key in (
        "security_terminal_labels_allowed",
        "issuer_terminal_labels_allowed",
        "security_terminal_metadata_complete",
        "issuer_terminal_metadata_complete",
        "every_security_links_to_one_issuer",
        "admitted_issuer_primary_contract_complete",
        "at_most_one_primary_security_per_issuer",
        "unresolved_primary_candidates_remain_needs_data",
    ):
        acceptance[key] = bool(terminal_validation[key])
    acceptance["fixed_642_security_rows_accounted"] = (
        len(fixed_tickers) == 642 and fixed_matched == 642 and fixed_issuer_count == 610
    )
    required_checks = tuple(key for key in acceptance if key != "production_sector_scan_exercised")
    acceptance_status = "PASSED" if all(acceptance[key] for key in required_checks) else "FAILED"
    coverage = {
        "schema_version": "US_EQUITY_CENSUS_COVERAGE_V1",
        "run_id": run_id,
        "as_of_date": as_of_date,
        "target_band": TARGET_BAND,
        "acceptance_status": acceptance_status,
        "counts": counts,
        "issuer_disposition_counts": dict(sorted(disposition_counts.items())),
        "security_disposition_counts": dict(sorted(security_disposition_counts.items())),
        "cap_band_counts": dict(
            sorted(Counter(str(row.get("cap_band") or "UNRESOLVED") for row in issuer_rows).items())
        ),
        "admitted_sector_counts": dict(
            sorted(
                Counter(
                    str(row.get("canonical_sector"))
                    for row in admitted
                    if row.get("canonical_sector")
                ).items()
            )
        ),
        "cap_source_hierarchy_state": _cap_source_hierarchy_state(parsed),
        "terminal_validation": terminal_validation,
        "actual_llm_calls": cost_ledger.actual_llm_calls,
        "actual_llm_cost_usd": cost_ledger.actual_llm_cost_usd,
        "paid_model_authorized": cost_ledger.paid_model_authorized,
        "production_sector_scan_exercised": False,
        "provider_free_replay_verified": provider_free_replay_verified,
        "semantic_output_fingerprint": semantic_output_fingerprint,
        "taxonomy": canonical_sector_taxonomy_manifest(),
        "external_industry_taxonomy": external_industry_taxonomy_manifest(),
    }
    artifacts["coverage_summary"].write_text(
        json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8"
    )
    artifacts["acceptance_report"].write_text(
        json.dumps(
            {
                "schema_version": "US_EQUITY_CENSUS_ACCEPTANCE_V1",
                "run_id": run_id,
                "as_of_date": as_of_date,
                "status": acceptance_status,
                "checks": acceptance,
                "required_checks": list(required_checks),
                "actual_llm_calls": cost_ledger.actual_llm_calls,
                "actual_llm_cost_usd": cost_ledger.actual_llm_cost_usd,
                "paid_model_authorized": cost_ledger.paid_model_authorized,
                "production_sector_scan_exercised": False,
                "provider_free_replay_verified": provider_free_replay_verified,
                "semantic_output_fingerprint": semantic_output_fingerprint,
                "note": "This acceptance covers the registry/cap/sector data plane; it does not claim a production sector scan was run.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifacts["parse_issues"].write_text(
        json.dumps(
            [asdict(issue) for issue in parsed.issues], indent=2, sort_keys=True, default=str
        ),
        encoding="utf-8",
    )
    cap_discrepancies = [
        {
            "issuer_key": row["issuer_key"],
            "primary_ticker": row["primary_ticker"],
            "market_cap_status": row["market_cap_status"],
            "cap_derivation": json.loads(row["cap_derivation_json"] or "{}"),
        }
        for row in issuer_rows
        if "discrepancy_code" in (row["cap_derivation_json"] or "")
        and json.loads(row["cap_derivation_json"] or "{}").get("discrepancy_code")
    ]
    sector_discrepancies = [
        {
            "issuer_key": row["issuer_key"],
            "primary_ticker": row["primary_ticker"],
            "selected_source_sector_label": row["source_sector_label"],
            "sector_provenance": json.loads(row["sector_provenance_json"] or "{}"),
        }
        for row in issuer_rows
        if json.loads(row["sector_provenance_json"] or "{}").get("cross_source_sector_disagreement")
    ]
    artifacts["source_discrepancies"].write_text(
        json.dumps(
            {
                "cap_source_hierarchy_state": _cap_source_hierarchy_state(parsed),
                "unmatched_large_cap_source_rows": list(unmatched_caps),
                "cap_discrepancies": cap_discrepancies,
                "sector_discrepancies": sector_discrepancies,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifacts["source_manifest"].write_text(
        json.dumps(list(source_snapshots), indent=2, sort_keys=True), encoding="utf-8"
    )
    checkpoint = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "stage": "TERMINAL_DISPOSITION",
        "status": "COMPLETED",
        "acceptance_status": acceptance_status,
        "counts": counts,
    }
    artifacts["checkpoint"].write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8"
    )
    report_lines = [
        f"# U.S. Equity Universe Reconciliation — {as_of_date}",
        "",
        f"**Registry/cap/sector data-plane acceptance: {acceptance_status}.**",
        "",
        "This run is registry-first. Provider-free replay verification "
        f"{'passed' if provider_free_replay_verified else 'has not passed yet'}. "
        f"It made {cost_ledger.actual_llm_calls} LLM calls and spent "
        f"${cost_ledger.actual_llm_cost_usd:.2f}. It did not exercise a production sector scan, "
        "which is reported separately rather than implied.",
        "",
        "## Coverage",
        "",
        f"- Discovered securities: {counts['discovered_securities']:,}",
        f"- Deduplicated issuers: {counts['deduplicated_issuers']:,}",
        f"- Admitted large/mega issuers: {counts['admitted_large_and_mega_issuers']:,}",
        f"- Large-cap issuers: {counts['admitted_large_cap_issuers']:,}",
        f"- Mega-cap issuers: {counts['admitted_mega_cap_issuers']:,}",
        f"- Foreign issuer dispositions: {counts['foreign_issuer_dispositions']:,}",
        f"- New admitted names versus the fixed 642-security replay: {counts['new_large_and_mega_scan_population']:,}",
        "",
        "## Acceptance checks",
        "",
    ]
    report_lines.extend(
        f"- {'PASS' if value else 'FAIL'} — `{key}`" for key, value in acceptance.items()
    )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The fixed 642-security / 610-issuer replay is reconciled as a historical comparison cohort, "
            "not used as the membership boundary. Every security and issuer row has one terminal disposition; "
            "unresolved evidence remains `NEEDS_DATA`.",
            "",
        ]
    )
    artifacts["coverage_report"].write_text("\n".join(report_lines), encoding="utf-8")
    return artifacts, counts, acceptance, acceptance_status


def _promote_admitted_sector_population(
    conn: sqlite3.Connection,
    *,
    issuer_rows: Sequence[Mapping[str, Any]],
    as_of_date: str,
    run_id: str,
    input_fingerprint: str,
) -> int:
    if not _table_exists(conn, "sector_inference"):
        return 0

    expected: dict[str, tuple[str, str]] = {}
    for row in issuer_rows:
        if row.get("terminal_disposition") != "ADMITTED_LARGE_AND_MEGA":
            continue
        ticker = normalize_ticker(row.get("primary_ticker"))
        source_label = str(row.get("source_sector_label") or "")
        issuer_key = str(row.get("issuer_key") or "")
        if not ticker or not source_label or not issuer_key:
            raise ValueError(
                "admitted census promotion row is missing ticker, sector, or issuer key"
            )
        if ticker in expected:
            raise ValueError(f"duplicate admitted census promotion ticker: {ticker}")
        expected[ticker] = (source_label, issuer_key)

    existing_census: dict[str, sqlite3.Row] = {}
    existing_other: dict[str, sqlite3.Row] = {}
    for existing in conn.execute(
        "SELECT id, ticker, inferred_sector, derived_from FROM sector_inference "
        "WHERE as_of_date = ?",
        (as_of_date,),
    ):
        ticker = normalize_ticker(existing["ticker"])
        try:
            derived_from = json.loads(str(existing["derived_from"] or "{}"))
        except json.JSONDecodeError:
            derived_from = {}
        producer = (
            str(derived_from.get("producer") or "") if isinstance(derived_from, Mapping) else ""
        )
        destination = (
            existing_census if producer == CENSUS_SECTOR_PROMOTION_PRODUCER else existing_other
        )
        if ticker in existing_census or ticker in existing_other:
            raise ValueError(f"duplicate same-date sector inference ticker: {ticker}")
        destination[ticker] = existing

    for ticker, (source_label, _issuer_key) in expected.items():
        existing = existing_other.get(ticker)
        if existing is None:
            continue
        if str(existing["inferred_sector"] or "") != source_label:
            raise ValueError(
                f"sector inference conflict for {ticker} on {as_of_date}: "
                f"{existing['inferred_sector']} != {source_label}"
            )
        raise ValueError(
            f"sector inference ownership conflict for {ticker} on {as_of_date}: "
            "existing row was not produced by the U.S. equity census"
        )

    now = utc_now_iso()
    for ticker, existing in existing_census.items():
        if ticker not in expected:
            conn.execute("DELETE FROM sector_inference WHERE id = ?", (existing["id"],))

    for ticker, (source_label, issuer_key) in expected.items():
        lineage = _json_dumps(
            {
                "producer": CENSUS_SECTOR_PROMOTION_PRODUCER,
                "run_id": run_id,
                "input_fingerprint": input_fingerprint,
                "issuer_key": issuer_key,
                "taxonomy_version": CANONICAL_SECTOR_TAXONOMY_VERSION,
            }
        )
        existing = existing_census.get(ticker)
        if existing is not None:
            conn.execute(
                "UPDATE sector_inference SET ticker = ?, inferred_sector = ?, score = 1.0, "
                "derived_from = ?, created_at = ? WHERE id = ?",
                (ticker, source_label, lineage, now, existing["id"]),
            )
            continue
        conn.execute(
            "INSERT INTO sector_inference("
            "ticker, as_of_date, inferred_sector, score, derived_from, created_at"
            ") VALUES(?, ?, ?, 1.0, ?, ?)",
            (ticker, as_of_date, source_label, lineage, now),
        )
    return len(expected)


def _admitted_sector_population_complete(
    conn: sqlite3.Connection,
    *,
    issuer_rows: Sequence[Mapping[str, Any]],
    as_of_date: str,
    run_id: str,
    input_fingerprint: str,
) -> bool:
    expected = {
        normalize_ticker(row.get("primary_ticker")): (
            str(row.get("source_sector_label") or ""),
            str(row.get("issuer_key") or ""),
        )
        for row in issuer_rows
        if row.get("terminal_disposition") == "ADMITTED_LARGE_AND_MEGA"
    }
    if not expected or any(
        not ticker or not sector or not issuer_key
        for ticker, (sector, issuer_key) in expected.items()
    ):
        return False
    actual: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for row in conn.execute(
        "SELECT ticker, inferred_sector, derived_from FROM sector_inference WHERE as_of_date = ?",
        (as_of_date,),
    ):
        try:
            derived_from = json.loads(str(row["derived_from"] or "{}"))
        except json.JSONDecodeError:
            continue
        if not isinstance(derived_from, Mapping) or (
            derived_from.get("producer") != CENSUS_SECTOR_PROMOTION_PRODUCER
        ):
            continue
        ticker = normalize_ticker(row["ticker"])
        if ticker in actual:
            return False
        actual[ticker] = (str(row["inferred_sector"] or ""), derived_from)

    if set(actual) != set(expected):
        return False
    return all(
        actual[ticker][0] == sector
        and actual[ticker][1].get("run_id") == run_id
        and actual[ticker][1].get("input_fingerprint") == input_fingerprint
        and actual[ticker][1].get("issuer_key") == issuer_key
        and actual[ticker][1].get("taxonomy_version") == CANONICAL_SECTOR_TAXONOMY_VERSION
        for ticker, (sector, issuer_key) in expected.items()
    )


def run_us_equity_census(
    *,
    conn: sqlite3.Connection,
    inputs: CensusInputPaths,
    output_dir: Path,
    as_of_date: str | None = None,
    run_id: str | None = None,
    promote_sector_population: bool = True,
) -> CensusRunResult:
    """Run or resume the provider-free census from immutable local snapshots."""

    cost_ledger = CensusCostLedger(paid_model_authorized=False)
    canonical_as_of = date.fromisoformat(as_of_date or date.today().isoformat()).isoformat()
    run_id = run_id or f"us_equity_census_{canonical_as_of.replace('-', '')}"
    output_dir = Path(output_dir)
    conn.row_factory = sqlite3.Row
    init_db(conn=conn)
    prior_run = conn.execute(
        "SELECT status, input_fingerprint, checkpoint_path, status_detail_json "
        "FROM us_equity_census_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    persisted_database_inputs: dict[str, Any] = {}
    if prior_run is not None and prior_run["checkpoint_path"]:
        prior_snapshot_path = (
            Path(str(prior_run["checkpoint_path"])).parent
            / "source_snapshots"
            / "database_inputs.json"
        )
        persisted_database_inputs = _safe_json(prior_snapshot_path)
    database_inputs = (
        persisted_database_inputs
        if persisted_database_inputs.get("schema_version") == "US_EQUITY_CENSUS_DATABASE_INPUTS_V1"
        and persisted_database_inputs.get("as_of_date") == canonical_as_of
        else _database_input_snapshot(conn, as_of_date=canonical_as_of)
    )
    fingerprint = census_input_fingerprint(
        inputs, as_of_date=canonical_as_of, database_inputs=database_inputs
    )
    prior_detail: dict[str, Any] = {}
    if prior_run is not None:
        try:
            loaded_detail = json.loads(str(prior_run["status_detail_json"] or "{}"))
        except json.JSONDecodeError:
            loaded_detail = {}
        if isinstance(loaded_detail, dict):
            prior_detail = loaded_detail
    _initialize_run(
        conn,
        run_id=run_id,
        as_of_date=canonical_as_of,
        inputs=inputs,
        output_dir=output_dir,
        fingerprint=fingerprint,
    )
    source_snapshots = _copy_source_snapshots(inputs, output_dir=output_dir)
    source_snapshots.append(_write_database_input_snapshot(database_inputs, output_dir=output_dir))
    conn.execute(
        "UPDATE us_equity_census_runs SET source_snapshot_count = ?, "
        "source_manifest_json = ?, updated_at = ? WHERE run_id = ?",
        (len(source_snapshots), _json_dumps(source_snapshots), utc_now_iso(), run_id),
    )
    conn.commit()
    active_attempt: tuple[str, str] | None = None
    try:
        stage = "MEMBERSHIP"
        attempt = _begin_attempt(conn, run_id=run_id, stage=stage, input_fingerprint=fingerprint)
        active_attempt = (attempt, stage)
        parsed = _parse_source_snapshots(inputs, as_of_date=canonical_as_of)
        retrieved_at = utc_now_iso()
        security_rows, membership_counts = _build_membership_rows(
            parsed,
            as_of_date=canonical_as_of,
            retrieved_at=retrieved_at,
            fixed_cohort_tickers=_parse_fixed_cohort(inputs),
        )
        membership_summary = _checkpoint_stage_rows(
            "MEMBERSHIP", issuer_rows=[], security_rows=security_rows
        )
        _persist_membership_issuer_placeholders(conn, run_id=run_id, rows=security_rows)
        _persist_security_rows(conn, run_id=run_id, rows=security_rows)
        _finish_attempt(
            conn,
            run_id=run_id,
            attempt_id=attempt,
            stage=stage,
            output={**membership_counts, **membership_summary},
        )
        active_attempt = None

        stage = "IDENTITY"
        attempt = _begin_attempt(conn, run_id=run_id, stage=stage, input_fingerprint=fingerprint)
        active_attempt = (attempt, stage)
        issuer_rows, unmatched_caps = _issuer_rows(
            conn,
            inputs=inputs,
            parsed=parsed,
            security_rows=security_rows,
            as_of_date=canonical_as_of,
            retrieved_at=retrieved_at,
            database_inputs=database_inputs,
        )
        identity_summary = _checkpoint_stage_rows(
            "IDENTITY", issuer_rows=issuer_rows, security_rows=security_rows
        )
        _persist_security_rows(conn, run_id=run_id, rows=security_rows)
        _persist_issuer_rows(conn, run_id=run_id, rows=issuer_rows)
        _finish_attempt(
            conn,
            run_id=run_id,
            attempt_id=attempt,
            stage=stage,
            output={
                **identity_summary,
                "unmatched_large_cap_source_rows": len(unmatched_caps),
            },
        )
        active_attempt = None

        for stage in STAGES[2:-1]:
            attempt = _begin_attempt(
                conn, run_id=run_id, stage=stage, input_fingerprint=fingerprint
            )
            active_attempt = (attempt, stage)
            stage_summary = _checkpoint_stage_rows(
                stage, issuer_rows=issuer_rows, security_rows=security_rows
            )
            _persist_security_rows(conn, run_id=run_id, rows=security_rows)
            _persist_issuer_rows(conn, run_id=run_id, rows=issuer_rows)
            _finish_attempt(
                conn,
                run_id=run_id,
                attempt_id=attempt,
                stage=stage,
                output=stage_summary,
            )
            active_attempt = None

        stage = "TERMINAL_DISPOSITION"
        attempt = _begin_attempt(conn, run_id=run_id, stage=stage, input_fingerprint=fingerprint)
        active_attempt = (attempt, stage)
        _finalize_rows(
            conn,
            run_id=run_id,
            issuer_rows=issuer_rows,
            security_rows=security_rows,
        )
        semantic_output_fingerprint = _semantic_replay_fingerprint(
            security_rows, issuer_rows, unmatched_caps
        )
        provider_free_replay_verified = bool(
            prior_run is not None
            and prior_run["status"] == "COMPLETED"
            and prior_run["input_fingerprint"] == fingerprint
            and prior_detail.get("semantic_output_fingerprint") == semantic_output_fingerprint
        )
        _finish_attempt(
            conn,
            run_id=run_id,
            attempt_id=attempt,
            stage=stage,
            output={
                "issuer_count": len(issuer_rows),
                "security_count": len(security_rows),
                "semantic_output_fingerprint": semantic_output_fingerprint,
                "provider_free_replay_verified": provider_free_replay_verified,
            },
        )
        active_attempt = None

        artifacts, counts, acceptance, acceptance_status = _write_outputs(
            conn=conn,
            run_id=run_id,
            as_of_date=canonical_as_of,
            output_dir=output_dir,
            parsed=parsed,
            inputs=inputs,
            unmatched_caps=unmatched_caps,
            source_snapshots=source_snapshots,
            provider_free_replay_verified=provider_free_replay_verified,
            sector_population_promoted=False,
            semantic_output_fingerprint=semantic_output_fingerprint,
            cost_ledger=cost_ledger,
        )
        promotion_prerequisites = all(
            value
            for key, value in acceptance.items()
            if key
            not in {
                "new_large_cap_scan_population_promoted",
                "production_sector_scan_exercised",
            }
        )
        promoted_count = 0
        if promote_sector_population and promotion_prerequisites:
            promoted_count = _promote_admitted_sector_population(
                conn,
                issuer_rows=issuer_rows,
                as_of_date=canonical_as_of,
                run_id=run_id,
                input_fingerprint=fingerprint,
            )
            sector_population_promoted = _admitted_sector_population_complete(
                conn,
                issuer_rows=issuer_rows,
                as_of_date=canonical_as_of,
                run_id=run_id,
                input_fingerprint=fingerprint,
            )
            artifacts, counts, acceptance, acceptance_status = _write_outputs(
                conn=conn,
                run_id=run_id,
                as_of_date=canonical_as_of,
                output_dir=output_dir,
                parsed=parsed,
                inputs=inputs,
                unmatched_caps=unmatched_caps,
                source_snapshots=source_snapshots,
                provider_free_replay_verified=provider_free_replay_verified,
                sector_population_promoted=sector_population_promoted,
                semantic_output_fingerprint=semantic_output_fingerprint,
                cost_ledger=cost_ledger,
            )
        now = utc_now_iso()
        conn.execute(
            """
            UPDATE us_equity_census_runs
            SET status = 'COMPLETED', current_stage = 'TERMINAL_DISPOSITION',
                acceptance_status = ?, discovered_security_count = ?,
                deduplicated_issuer_count = ?, us_domiciled_issuer_count = ?,
                foreign_issuer_count = ?, unresolved_identity_count = ?,
                unresolved_cap_count = ?, unresolved_sector_count = ?,
                status_detail_json = ?, completed_at = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (
                acceptance_status,
                counts["discovered_securities"],
                counts["deduplicated_issuers"],
                sum(row.get("is_us_domiciled") == 1 for row in issuer_rows),
                counts["foreign_issuer_dispositions"],
                counts["unresolved_identity_issuers"],
                counts["unresolved_cap_issuers"],
                counts["unresolved_sector_issuers"],
                _json_dumps(
                    {
                        "acceptance_checks": acceptance,
                        "semantic_output_fingerprint": semantic_output_fingerprint,
                        "database_input_sha256": hashlib.sha256(
                            _json_dumps(database_inputs).encode("utf-8")
                        ).hexdigest(),
                        "provider_free_replay_verified": provider_free_replay_verified,
                        "promoted_sector_population_count": promoted_count,
                    }
                ),
                now,
                now,
                run_id,
            ),
        )
        conn.commit()
        return CensusRunResult(
            run_id=run_id,
            as_of_date=canonical_as_of,
            status="COMPLETED",
            acceptance_status=acceptance_status,
            output_dir=output_dir,
            counts=counts,
            acceptance_checks=acceptance,
            artifacts=artifacts,
            actual_llm_calls=cost_ledger.actual_llm_calls,
            actual_llm_cost_usd=cost_ledger.actual_llm_cost_usd,
            notes=[
                (
                    "Provider-free replay verified from invariant-equivalent immutable inputs."
                    if provider_free_replay_verified
                    else "A second identical invocation is required to verify provider-free replay."
                ),
                "Production sector scan was not exercised.",
            ],
        )
    except Exception as exc:
        conn.rollback()
        if active_attempt is not None:
            _fail_attempt(
                conn,
                run_id=run_id,
                attempt_id=active_attempt[0],
                error=exc,
            )
        else:
            now = utc_now_iso()
            conn.execute(
                "UPDATE us_equity_census_runs SET status = 'INTERRUPTED', interrupted_at = ?, "
                "updated_at = ?, status_detail_json = ? WHERE run_id = ?",
                (now, now, _json_dumps({"error": str(exc)}), run_id),
            )
            conn.commit()
        raise


def load_census_status(conn: sqlite3.Connection, *, run_id: str) -> dict[str, Any] | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM us_equity_census_runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    attempts = [
        dict(item)
        for item in conn.execute(
            "SELECT stage, attempt_number, status, started_at, finished_at, error_code, error_message "
            "FROM us_equity_census_attempts WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
    ]
    return {**dict(row), "attempts": attempts}


__all__ = [
    "BOUNDARY_INVESTIGATION_FLOOR_USD",
    "COMMON_SECURITY_TYPES",
    "CapEvidenceRow",
    "CapResolution",
    "CENSUS_POLICY_VERSION",
    "CensusCostLedger",
    "CensusInputPaths",
    "CensusRunResult",
    "IssuerProfile",
    "LARGE_CAP_FLOOR_USD",
    "MAX_PAID_MODEL_COST_USD",
    "PrimarySelection",
    "SecurityTypeResolution",
    "STAGES",
    "TARGET_BAND",
    "TARGET_EXCHANGES",
    "TERMINAL_CAP_EVIDENCE_SCHEMA_VERSION",
    "census_input_fingerprint",
    "classify_security_type",
    "fetch_large_cap_sec_submissions",
    "large_cap_submission_candidates",
    "load_census_status",
    "normalize_exchange",
    "normalize_ticker",
    "parse_submissions_profile",
    "parse_terminal_cap_evidence_snapshot",
    "resolve_direct_cap_evidence",
    "run_us_equity_census",
    "select_primary_security",
    "stable_security_key",
    "validate_terminal_dispositions",
]
