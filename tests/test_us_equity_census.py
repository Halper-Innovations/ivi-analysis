from __future__ import annotations

import json
import sqlite3

import pytest

from app.db import init_db
from app.universe.us_equity_census import (
    STAGES,
    TERMINAL_CAP_EVIDENCE_SCHEMA_VERSION,
    CapEvidenceRow,
    CensusCostLedger,
    CensusInputPaths,
    _admitted_sector_population_complete,
    _apply_operating_bdc_security_override,
    _issuer_acceptance_checks,
    _promote_admitted_sector_population,
    _terminal_for_issuer,
    census_input_fingerprint,
    classify_security_type,
    parse_submissions_profile,
    parse_terminal_cap_evidence_snapshot,
    resolve_direct_cap_evidence,
    run_us_equity_census,
    select_primary_security,
    validate_terminal_dispositions,
)


def test_paid_model_cost_ledger_requires_authorization_and_hard_stops_above_100():
    ledger = CensusCostLedger()
    with pytest.raises(RuntimeError, match="explicit preflight authorization"):
        ledger.record_paid_model_call(cost_usd=1.0)

    authorized = CensusCostLedger(paid_model_authorized=True)
    authorized.record_paid_model_call(cost_usd=99.0)
    assert authorized.actual_llm_calls == 1
    assert authorized.actual_llm_cost_usd == 99.0
    with pytest.raises(RuntimeError, match=r"above \$100.00"):
        authorized.record_paid_model_call(cost_usd=1.01)


def test_input_fingerprint_binds_cap_pages_to_explicit_ordered_roles(tmp_path):
    sec = tmp_path / "sec.json"
    nasdaq = tmp_path / "nasdaq.txt"
    other = tmp_path / "other.txt"
    page_a = tmp_path / "a.html"
    page_b = tmp_path / "b.html"
    for path, value in (
        (sec, "sec"),
        (nasdaq, "nasdaq"),
        (other, "other"),
        (page_a, "page-a"),
        (page_b, "page-b"),
    ):
        path.write_text(value, encoding="utf-8")

    forward = CensusInputPaths(
        sec_registry=sec,
        nasdaq_listed=nasdaq,
        other_listed=other,
        companiesmarketcap_pages=(page_a, page_b),
    )
    reversed_pages = CensusInputPaths(
        sec_registry=sec,
        nasdaq_listed=nasdaq,
        other_listed=other,
        companiesmarketcap_pages=(page_b, page_a),
    )
    assert census_input_fingerprint(forward, as_of_date="2026-07-17") != census_input_fingerprint(
        reversed_pages, as_of_date="2026-07-17"
    )


def test_sector_population_promotion_is_transactional_and_run_traceable():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn=conn)
    prior_issuer_rows = [
        {
            "issuer_key": "CIK:1001",
            "primary_ticker": "ALP",
            "source_sector_label": "enterprise_software",
            "terminal_disposition": "ADMITTED_LARGE_AND_MEGA",
        },
        {
            "issuer_key": "CIK:1002",
            "primary_ticker": "OLD",
            "source_sector_label": "retail",
            "terminal_disposition": "ADMITTED_LARGE_AND_MEGA",
        },
    ]
    added = _promote_admitted_sector_population(
        conn,
        issuer_rows=prior_issuer_rows,
        as_of_date="2026-07-17",
        run_id="promotion_v1",
        input_fingerprint="fingerprint-v1",
    )
    assert added == 2
    assert (
        _admitted_sector_population_complete(
            conn,
            issuer_rows=prior_issuer_rows,
            as_of_date="2026-07-17",
            run_id="promotion_v1",
            input_fingerprint="fingerprint-v1",
        )
        is True
    )
    conn.commit()
    conn.execute(
        "INSERT INTO sector_inference("
        "ticker, as_of_date, inferred_sector, score, derived_from, created_at"
        ") VALUES('MAN', '2026-07-17', 'utilities', 0.5, ?, '2026-07-17T00:00:00Z')",
        (json.dumps({"producer": "manual_review"}),),
    )
    conn.commit()

    corrected_issuer_rows = [
        {
            "issuer_key": "CIK:1001",
            "primary_ticker": "ALP",
            "source_sector_label": "business_services",
            "terminal_disposition": "ADMITTED_LARGE_AND_MEGA",
        },
        {
            "issuer_key": "CIK:1003",
            "primary_ticker": "NEW",
            "source_sector_label": "retail",
            "terminal_disposition": "ADMITTED_LARGE_AND_MEGA",
        },
    ]
    replaced = _promote_admitted_sector_population(
        conn,
        issuer_rows=corrected_issuer_rows,
        as_of_date="2026-07-17",
        run_id="promotion_v2",
        input_fingerprint="fingerprint-v2",
    )
    assert replaced == 2
    assert (
        _admitted_sector_population_complete(
            conn,
            issuer_rows=corrected_issuer_rows,
            as_of_date="2026-07-17",
            run_id="promotion_v2",
            input_fingerprint="fingerprint-v2",
        )
        is True
    )
    assert [
        tuple(row)
        for row in conn.execute(
            "SELECT ticker, inferred_sector FROM sector_inference "
            "WHERE as_of_date = '2026-07-17' ORDER BY ticker"
        )
    ] == [
        ("ALP", "business_services"),
        ("MAN", "utilities"),
        ("NEW", "retail"),
    ]
    conn.rollback()
    assert [
        tuple(row)
        for row in conn.execute(
            "SELECT ticker, inferred_sector FROM sector_inference "
            "WHERE as_of_date = '2026-07-17' ORDER BY ticker"
        )
    ] == [
        ("ALP", "enterprise_software"),
        ("MAN", "utilities"),
        ("OLD", "retail"),
    ]

    assert (
        _promote_admitted_sector_population(
            conn,
            issuer_rows=corrected_issuer_rows,
            as_of_date="2026-07-17",
            run_id="promotion_v2",
            input_fingerprint="fingerprint-v2",
        )
        == 2
    )
    conn.commit()
    promoted = conn.execute(
        "SELECT ticker, inferred_sector, score, derived_from FROM sector_inference "
        "WHERE as_of_date = '2026-07-17' ORDER BY ticker"
    ).fetchall()
    assert [(row["ticker"], row["inferred_sector"], row["score"]) for row in promoted] == [
        ("ALP", "business_services", 1.0),
        ("MAN", "utilities", 0.5),
        ("NEW", "retail", 1.0),
    ]
    assert json.loads(promoted[0]["derived_from"]) == {
        "input_fingerprint": "fingerprint-v2",
        "issuer_key": "CIK:1001",
        "producer": "us_equity_census_issuer_deduplication",
        "run_id": "promotion_v2",
        "taxonomy_version": "us_equity_sector_taxonomy.v1",
    }
    assert json.loads(promoted[2]["derived_from"]) == {
        "input_fingerprint": "fingerprint-v2",
        "issuer_key": "CIK:1003",
        "producer": "us_equity_census_issuer_deduplication",
        "run_id": "promotion_v2",
        "taxonomy_version": "us_equity_sector_taxonomy.v1",
    }
    conn.close()


def test_security_type_policy_preserves_included_structures_and_explicit_exclusions():
    ptp = classify_security_type(
        listed_name="Example LP Common Units Representing Limited Partner Interests",
        ticker="EXLP",
    )
    assert (
        ptp.security_type,
        ptp.status,
        ptp.reason_code,
        ptp.is_common_equity,
        ptp.is_adr,
    ) == (
        "COMMON_EQUITY_EQUIVALENT",
        "RESOLVED",
        "PTP_COMMON_UNIT",
        True,
        False,
    )

    reit = classify_security_type(
        listed_name="Example REIT Common Shares of Beneficial Interest",
        ticker="REIT",
    )
    assert reit.security_type == "COMMON_EQUITY"
    assert reit.is_common_equity is True

    adr = classify_security_type(
        listed_name="Foreign Issuer American Depositary Shares",
        ticker="FADS",
    )
    assert adr.security_type == "ADR_ADS"
    assert adr.is_adr is True

    preferred = classify_security_type(
        listed_name="Example 6.25% Series A Preferred Stock",
        ticker="EX-PRA",
    )
    assert preferred.security_type == "PREFERRED_EQUITY"
    assert preferred.is_common_equity is False

    etf = classify_security_type(
        listed_name="Uninformative Trust Name",
        ticker="FUND",
        etf_flag="Y",
    )
    assert etf.security_type == "ETF"
    assert etf.reason_code == "STRUCTURED_ETF_FLAG"

    partnership_units = classify_security_type(
        listed_name="Example Renewable Partners L.P. Limited Partnership Units",
        ticker="ERPU",
        etf_flag="N",
    )
    assert partnership_units.security_type == "COMMON_EQUITY_EQUIVALENT"
    assert partnership_units.reason_code == "PTP_COMMON_UNIT"

    acquisition_units = classify_security_type(
        listed_name="Example Acquisition Corp. Units",
        ticker="EXACU",
        etf_flag="N",
    )
    assert acquisition_units.security_type == "COMPOSITE_UNIT"
    assert acquisition_units.reason_code == "LISTING_NAME_GENERIC_COMPOSITE_UNIT"

    plain_w_suffix = classify_security_type(
        listed_name="Example Networks, Inc.", ticker="PANW", etf_flag="N"
    )
    assert plain_w_suffix.security_type == "COMMON_EQUITY"
    assert plain_w_suffix.reason_code == "STRUCTURED_NON_ETF_PLAIN_EQUITY_LISTING"

    plain_listing = classify_security_type(
        listed_name="Example Payments Inc.", ticker="V", etf_flag="N"
    )
    assert plain_listing.security_type == "COMMON_EQUITY"

    debt = classify_security_type(listed_name="Example Global 6.50% Notes Due 2029", ticker="EXN")
    assert debt.security_type == "DEBT_SECURITY"


def test_submissions_profile_uses_domicile_and_forms_without_blanket_bdc_exclusion():
    profile = parse_submissions_profile(
        {
            "cik": "0000001001",
            "name": "Operating BDC Inc.",
            "sic": "6726",
            "sicDescription": "Investment Offices, NEC",
            "stateOfIncorporation": "DE",
            "stateOfIncorporationDescription": "DE",
            "entityType": "operating",
            "category": "Accelerated filer",
            "filings": {
                "recent": {
                    "form": ["10-Q", "10-K", "8-K"],
                    "filingDate": ["2026-05-01", "2026-02-20", "2026-07-01"],
                    "accessionNumber": ["a", "0000001001-26-000001", "c"],
                }
            },
        },
        source_url="https://data.sec.gov/submissions/CIK0000001001.json",
    )

    assert profile.cik == "1001"
    assert profile.domicile_status == "US_DOMICILED"
    assert profile.domicile_country_code == "US"
    assert profile.operating_status == "OPERATING"
    assert profile.sic == 6726
    assert profile.latest_annual_filing_date == "2026-02-20"
    assert profile.latest_annual_filing_accession == "0000001001-26-000001"


def test_submissions_profile_falls_back_to_explicit_foreign_mailing_address():
    profile = parse_submissions_profile(
        {
            "cik": "0000001002",
            "name": "Foreign Legal Entity N.V.",
            "entityType": "operating",
            "stateOfIncorporation": "",
            "addresses": {
                "business": {
                    "stateOrCountry": None,
                    "stateOrCountryDescription": None,
                    "countryCode": None,
                },
                "mailing": {
                    "stateOrCountry": "P7",
                    "stateOrCountryDescription": "Netherlands",
                    "city": "Eindhoven",
                },
            },
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "filingDate": ["2026-02-20"],
                    "accessionNumber": ["0000001002-26-000001"],
                }
            },
        },
        source_url="https://data.sec.gov/submissions/CIK0000001002.json",
    )
    assert profile.domicile_status == "FOREIGN_DOMICILED"
    assert profile.domicile_country_code == "P7"
    assert profile.domicile_jurisdiction == "Netherlands"
    assert profile.domicile_method == "SEC_MAILING_ADDRESS"


@pytest.mark.parametrize(
    ("state_code", "cik"),
    (("DC", "1306965"), ("DE", "858446")),
)
def test_current_foreign_filer_evidence_overrides_ambiguous_us_state_code(state_code, cik):
    profile = parse_submissions_profile(
        {
            "cik": cik,
            "name": "Foreign Public Company plc",
            "stateOfIncorporation": state_code,
            "stateOfIncorporationDescription": state_code,
            "entityType": "operating",
            "addresses": {
                "business": {
                    "city": "LONDON",
                    "country": "United Kingdom",
                    "countryCode": "X0",
                    "isForeignLocation": 1,
                }
            },
            "filings": {
                "recent": {
                    "form": ["6-K", "20-F", "20-F"],
                    "filingDate": ["2026-07-01", "2026-03-12", "2025-03-25"],
                    "accessionNumber": ["current", "annual", "prior-annual"],
                }
            },
        },
        source_url=f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json",
    )

    assert profile.domicile_status == "FOREIGN_DOMICILED"
    assert profile.domicile_country_code == "X0"
    assert profile.domicile_jurisdiction == "United Kingdom"
    assert profile.domicile_method == "SEC_FOREIGN_PRIVATE_ISSUER_FORMS"
    assert profile.domicile_confidence == "HIGH"
    assert profile.operating_status == "FOREIGN_FILER"
    assert profile.latest_annual_filing_date == "2026-03-12"


def test_operating_bdc_forms_override_exchange_closed_end_fund_label():
    profile = parse_submissions_profile(
        {
            "cik": "0000001003",
            "name": "Operating BDC Inc.",
            "stateOfIncorporation": "MD",
            "entityType": "operating",
            "filings": {
                "recent": {
                    "form": ["10-K", "N-2ASR"],
                    "filingDate": ["2026-03-01", "2026-02-01"],
                    "accessionNumber": ["annual", "registration"],
                }
            },
        },
        source_url="https://data.sec.gov/submissions/CIK0000001003.json",
    )
    securities = [
        {
            "security_type": "CLOSED_END_FUND",
            "security_type_status": "RESOLVED",
            "is_common_equity": 0,
            "listing_status": "ACTIVE",
            "exchange_name": "NASDAQ",
            "provenance_json": "{}",
        }
    ]
    assert _apply_operating_bdc_security_override(profile, securities) == 1
    assert securities[0]["security_type"] == "COMMON_EQUITY_EQUIVALENT"
    assert securities[0]["is_common_equity"] == 1
    assert json.loads(securities[0]["provenance_json"])["security_type_reason_code"] == (
        "SEC_OPERATING_BDC_COMMON_EQUITY"
    )


def test_submissions_profile_uses_current_filing_family_for_blank_state_domicile():
    domestic = parse_submissions_profile(
        {
            "cik": "1002",
            "name": "Domestic Operating Co.",
            "entityType": "operating",
            "addresses": {
                "business": {
                    "stateOrCountry": "CA",
                    "countryCode": "US",
                    "isForeignLocation": False,
                }
            },
            "filings": {
                "recent": {
                    "form": ["10-K", "8-K"],
                    "filingDate": ["2026-02-15", "2026-07-01"],
                    "accessionNumber": ["annual", "current"],
                }
            },
        },
        source_url="https://data.sec.gov/submissions/CIK0000001002.json",
    )
    assert domestic.domicile_status == "US_DOMICILED"
    assert domestic.domicile_method == "SEC_PRINCIPAL_BUSINESS_ADDRESS"
    assert domestic.domicile_confidence == "MEDIUM"
    assert domestic.operating_status == "OPERATING"

    foreign_with_us_office = parse_submissions_profile(
        {
            "cik": "1003",
            "name": "Foreign Bank AG",
            "entityType": "operating",
            "addresses": {
                "business": {
                    "stateOrCountry": "NY",
                    "countryCode": "US",
                    "isForeignLocation": False,
                }
            },
            "filings": {
                "recent": {
                    "form": ["20-F", "6-K"],
                    "filingDate": ["2026-03-01", "2026-07-01"],
                    "accessionNumber": ["annual", "current"],
                }
            },
        },
        source_url="https://data.sec.gov/submissions/CIK0000001003.json",
    )
    assert foreign_with_us_office.domicile_status == "FOREIGN_DOMICILED"
    assert foreign_with_us_office.domicile_method == "SEC_FOREIGN_PRIVATE_ISSUER_FORMS"
    assert foreign_with_us_office.domicile_country_code is None
    assert foreign_with_us_office.operating_status == "FOREIGN_FILER"


def test_newer_domestic_annual_filing_preserves_us_domicile_after_historical_foreign_filing():
    profile = parse_submissions_profile(
        {
            "cik": "1004",
            "name": "Reorganized Domestic Co.",
            "stateOfIncorporation": "DE",
            "stateOfIncorporationDescription": "Delaware",
            "entityType": "operating",
            "addresses": {
                "business": {
                    "stateOrCountry": "DE",
                    "country": "United States",
                    "countryCode": "US",
                    "isForeignLocation": False,
                }
            },
            "filings": {
                "recent": {
                    "form": ["10-K", "8-K", "20-F", "6-K"],
                    "filingDate": ["2026-03-01", "2026-07-01", "2024-03-01", "2024-04-01"],
                    "accessionNumber": [
                        "domestic-annual",
                        "domestic-current",
                        "historical-foreign-annual",
                        "historical-foreign-current",
                    ],
                }
            },
        },
        source_url="https://data.sec.gov/submissions/CIK0000001004.json",
    )

    assert profile.domicile_status == "US_DOMICILED"
    assert profile.domicile_country_code == "US"
    assert profile.domicile_jurisdiction == "Delaware"
    assert profile.domicile_method == "SEC_STATE_OF_INCORPORATION"
    assert profile.operating_status == "OPERATING"
    assert profile.latest_annual_filing_date == "2026-03-01"
    assert profile.latest_annual_filing_accession == "domestic-annual"


def test_primary_selection_refuses_heuristic_tie_and_accepts_unique_direct_source_match():
    securities = [
        {
            "security_key": "1001:AAA-A:NYSE",
            "ticker": "AAA-A",
            "exchange_name": "NYSE",
            "security_type": "COMMON_EQUITY",
            "listing_status": "ACTIVE",
        },
        {
            "security_key": "1001:AAA-B:NYSE",
            "ticker": "AAA-B",
            "exchange_name": "NYSE",
            "security_type": "COMMON_EQUITY",
            "listing_status": "ACTIVE",
        },
    ]

    unresolved = select_primary_security(securities)
    assert unresolved.status == "NEEDS_DATA"
    assert unresolved.method == "MULTIPLE_COMMON_CLASSES_PRIMARY_UNRESOLVED"
    assert unresolved.security_key is None

    resolved = select_primary_security(securities, preferred_source_tickers=["AAA.B"])
    assert resolved.status == "RESOLVED"
    assert resolved.method == "DIRECT_CAP_SOURCE_SECURITY_MATCH"
    assert resolved.security_key == "1001:AAA-B:NYSE"
    assert resolved.ticker == "AAA-B"


def test_foreign_or_adr_primary_cannot_be_admitted_or_pass_acceptance():
    issuer = {
        "issuer_key": "CIK:1005",
        "identity_status": "RESOLVED",
        "issuer_type": "OPERATING_ISSUER",
        "operating_status": "OPERATING",
        "is_operating_company": 1,
        "is_us_domiciled": 1,
        "is_us_listed_foreign_issuer": 0,
        "primary_selection_status": "RESOLVED",
        "primary_security_key": "CIK:1005:FADS:NASDAQ",
        "market_cap_status": "RESOLVED",
        "market_cap_usd": 12_000_000_000.0,
        "cap_band": "large_and_mega",
        "sector_status": "RESOLVED",
        "canonical_sector": "energy",
        "sector_mapping_version": "us_equity_sector_taxonomy.v1",
        "scope_status": "PENDING_EVIDENCE",
        "cap_derivation_json": json.dumps({"evidence": [{"market_cap_usd": 12_000_000_000.0}]}),
    }
    adr_primary = {
        "security_key": "CIK:1005:FADS:NASDAQ",
        "issuer_key": "CIK:1005",
        "ticker": "FADS",
        "exchange_name": "NASDAQ",
        "listing_status": "ACTIVE",
        "security_type": "ADR_ADS",
        "is_adr": 1,
        "is_primary_security": 1,
    }

    assert _terminal_for_issuer(issuer, adr_primary) == (
        "OUT_OF_SCOPE_FOREIGN_ISSUER",
        "ADR_ONLY_PRIMARY_SECURITY",
    )
    assert _terminal_for_issuer(
        {**issuer, "operating_status": "FOREIGN_FILER", "is_us_listed_foreign_issuer": 1},
        adr_primary,
    ) == ("OUT_OF_SCOPE_FOREIGN_ISSUER", "FOREIGN_FILER")
    assert _terminal_for_issuer(
        issuer,
        {
            **adr_primary,
            "security_type": "COMMON_EQUITY",
            "is_adr": 0,
        },
    ) == ("ADMITTED_LARGE_AND_MEGA", "ALL_LARGE_AND_MEGA_CONTRACTS_RESOLVED")

    acceptance = _issuer_acceptance_checks(
        [
            {
                **issuer,
                "operating_status": "FOREIGN_FILER",
                "is_us_listed_foreign_issuer": 1,
                "terminal_disposition": "ADMITTED_LARGE_AND_MEGA",
            }
        ],
        cap_source_unmatched=0,
        fixed_cohort_count=642,
        fixed_cohort_issuer_count=610,
        source_checks={},
        provider_free_replay_verified=True,
        sector_population_promoted=True,
        cost_ledger=CensusCostLedger(),
    )
    assert acceptance["all_admitted_are_us_domiciled_operating"] is False

    terminal_validation = validate_terminal_dispositions(
        [
            {
                **adr_primary,
                "terminal_disposition": "ADMITTED_PRIMARY_COMMON_EQUITY",
                "terminal_reason_code": "ISSUER_ADMITTED_LARGE_AND_MEGA",
                "terminal_at": "2026-07-17T00:00:00Z",
            }
        ],
        [
            {
                **issuer,
                "terminal_disposition": "ADMITTED_LARGE_AND_MEGA",
                "terminal_reason_code": "ALL_LARGE_AND_MEGA_CONTRACTS_RESOLVED",
                "terminal_at": "2026-07-17T00:00:00Z",
            }
        ],
    )
    assert terminal_validation["admitted_issuer_primary_contract_complete"] is False
    assert terminal_validation["invalid_admitted_primary_issuer_keys"] == ["CIK:1005"]


def test_direct_cap_resolution_never_sums_share_classes_and_blocks_band_conflict():
    consensus = resolve_direct_cap_evidence(
        [
            CapEvidenceRow(
                "COMPANIESMARKETCAP_USA",
                "https://example.test/cmc",
                "AAA.A",
                "Alpha",
                12_000_000_000.0,
                "2026-07-17",
                source_rank=1,
            ),
            CapEvidenceRow(
                "STOCKANALYSIS",
                "https://example.test/sa",
                "AAA-A",
                "Alpha",
                12_300_000_000.0,
                "2026-07-17",
                source_rank=1,
            ),
        ],
        primary_ticker="AAA-A",
    )
    assert consensus.status == "RESOLVED"
    assert consensus.market_cap_usd == 12_300_000_000.0
    assert consensus.cap_band == "large_cap"
    assert consensus.method == "APPROVED_PROVIDER_DIRECT_ISSUER_CAP_CONSENSUS"
    assert consensus.confidence == "HIGH"

    conflict = resolve_direct_cap_evidence(
        [
            CapEvidenceRow(
                "COMPANIESMARKETCAP_USA",
                "https://example.test/cmc",
                "AAA",
                "Alpha",
                10_100_000_000.0,
                "2026-07-17",
            ),
            CapEvidenceRow(
                "STOCKANALYSIS",
                "https://example.test/sa",
                "AAA",
                "Alpha",
                9_900_000_000.0,
                "2026-07-17",
            ),
        ],
        primary_ticker="AAA",
    )
    assert conflict.status == "NEEDS_DATA"
    assert conflict.market_cap_usd is None
    assert conflict.discrepancy_code == "MARKET_CAP_BAND_CONFLICT"


def test_terminal_search_consensus_resolves_approved_provider_boundary_conflict():
    resolved = resolve_direct_cap_evidence(
        [
            CapEvidenceRow(
                "PROVIDER_A",
                "https://example.test/a",
                "AAA",
                "Alpha",
                10_100_000_000.0,
                "2026-07-17",
            ),
            CapEvidenceRow(
                "PROVIDER_B",
                "https://example.test/b",
                "AAA",
                "Alpha",
                9_900_000_000.0,
                "2026-07-17",
            ),
            CapEvidenceRow(
                "TERMINAL_DIRECT_A",
                "https://example.test/direct-a",
                "AAA",
                "Alpha",
                10_200_000_000.0,
                "2026-07-17",
                evidence_tier="SEARCH_DIRECT_ISSUER",
                resolution_role="TERMINAL_BOUNDARY_CHECK",
            ),
            CapEvidenceRow(
                "TERMINAL_DIRECT_B",
                "https://example.test/direct-b",
                "AAA",
                "Alpha",
                10_050_000_000.0,
                "2026-07-16",
                evidence_tier="SEARCH_DIRECT_ISSUER",
                resolution_role="TERMINAL_BOUNDARY_CHECK",
            ),
        ],
        primary_ticker="AAA",
    )
    assert resolved.status == "RESOLVED"
    assert resolved.market_cap_usd == 10_200_000_000.0
    assert resolved.cap_band == "large_cap"
    assert resolved.method == "TERMINAL_SEARCH_DIRECT_ISSUER_CAP_CONSENSUS"
    assert resolved.confidence == "MEDIUM"
    assert resolved.discrepancy_code == "MARKET_CAP_BAND_CONFLICT_RESOLVED_BY_TERMINAL_SEARCH"
    assert len(resolved.evidence) == 4


def test_terminal_cap_snapshot_validation_and_large_gap_acceptance(tmp_path):
    path = tmp_path / "terminal_cap_evidence.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": TERMINAL_CAP_EVIDENCE_SCHEMA_VERSION,
                "rows": [
                    {
                        "source_provider": "DIRECT_SOURCE",
                        "source_url": "https://example.test/issuer-cap",
                        "source_ticker": "aaa",
                        "source_name": "Alpha",
                        "market_cap_usd": 10_250_000_000,
                        "evidence_as_of_date": "2026-07-16",
                        "resolution_role": "TERMINAL_BOUNDARY_CHECK",
                        "evidence_detail": "direct issuer page",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    records, issues = parse_terminal_cap_evidence_snapshot(path, census_as_of_date="2026-07-17")
    assert issues == ()
    assert len(records) == 1
    assert records[0].source_ticker == "AAA"
    assert records[0].evidence_tier == "SEARCH_DIRECT_ISSUER"
    assert records[0].resolution_role == "TERMINAL_BOUNDARY_CHECK"

    unresolved_sector = {
        "terminal_disposition": "NEEDS_DATA_SECTOR",
        "identity_status": "RESOLVED",
        "operating_status": "OPERATING",
        "is_us_domiciled": 1,
        "scope_reason_code": "SOLE_ACTIVE_IN_SCOPE_COMMON_EQUITY",
        "primary_selection_status": "RESOLVED",
        "market_cap_status": "RESOLVED",
        "market_cap_usd": 12_000_000_000.0,
        "cap_band": "large_and_mega",
        "sector_status": "NEEDS_DATA",
        "canonical_sector": None,
        "sector_mapping_version": "us_equity_sector_taxonomy.v1",
        "cap_derivation_json": json.dumps({"evidence": [{"market_cap_usd": 12_000_000_000.0}]}),
    }
    checks = _issuer_acceptance_checks(
        [unresolved_sector],
        cap_source_unmatched=0,
        fixed_cohort_count=642,
        fixed_cohort_issuer_count=610,
        source_checks={},
        provider_free_replay_verified=True,
        sector_population_promoted=True,
        cost_ledger=CensusCostLedger(),
    )
    assert checks["zero_unresolved_current_large_candidate_dispositions"] is False
    assert checks["zero_unresolved_eligible_large_sector_contracts"] is False


def test_provider_free_census_smoke_persists_terminal_rows_and_exports(tmp_path):
    sec_path = tmp_path / "company_tickers_exchange.json"
    sec_path.write_text(
        json.dumps(
            {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [
                    [1001, "Alpha Inc.", "ALP", "Nasdaq"],
                    [1005, "Foreign Public Company plc", "FADS", "Nasdaq"],
                ],
            }
        ),
        encoding="utf-8",
    )
    nasdaq_path = tmp_path / "nasdaqlisted.txt"
    nasdaq_path.write_text(
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        "ALP|Alpha Inc. Common Stock|Q|N|N|100|N|N\n"
        "FADS|Foreign Public Company plc American Depositary Shares|Q|N|N|100|N|N\n"
        "File Creation Time: 0717202621:31|||||||\n",
        encoding="utf-8",
    )
    other_path = tmp_path / "otherlisted.txt"
    other_path.write_text(
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "File Creation Time: 0717202621:31||||||\n",
        encoding="utf-8",
    )
    cmc_path = tmp_path / "cmc.html"
    cmc_path.write_text(
        '<table><tr><td class="rank-td td-right">1</td>'
        '<td class="name-td"><div class="company-name">Alpha Inc.</div>'
        '<div class="company-code">ALP</div></td>'
        '<td class="td-right" data-sort="12000000000">$12 B</td></tr>'
        '<tr><td class="rank-td td-right">2</td>'
        '<td class="name-td"><div class="company-name">Foreign Public Company plc</div>'
        '<div class="company-code">FADS</div></td>'
        '<td class="td-right" data-sort="15000000000">$15 B</td></tr></table>',
        encoding="utf-8",
    )
    stock_path = tmp_path / "stockanalysis.html"
    stock_path.write_text(
        '<script>count:2,data:[{s:"ALP",n:"Alpha Inc.",marketCap:12000000000,'
        'industry:"Software - Infrastructure"},'
        '{s:"FADS",n:"Foreign Public Company plc",marketCap:15000000000,'
        'industry:"Oil & Gas Integrated"}]</script>',
        encoding="utf-8",
    )
    submissions_dir = tmp_path / "submissions"
    submissions_dir.mkdir()
    (submissions_dir / "0000001001.json").write_text(
        json.dumps(
            {
                "cik": "0000001001",
                "name": "Alpha Inc.",
                "sic": "3571",
                "stateOfIncorporation": "DE",
                "stateOfIncorporationDescription": "Delaware",
                "entityType": "operating",
                "category": "Large accelerated filer",
                "filings": {
                    "recent": {
                        "form": ["10-Q", "10-K"],
                        "filingDate": ["2026-05-01", "2026-02-20"],
                        "accessionNumber": ["q", "0000001001-26-000001"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (submissions_dir / "0000001005.json").write_text(
        json.dumps(
            {
                "cik": "0000001005",
                "name": "Foreign Public Company plc",
                "sic": "1311",
                "stateOfIncorporation": "DE",
                "stateOfIncorporationDescription": "DE",
                "entityType": "operating",
                "addresses": {
                    "business": {
                        "city": "LONDON",
                        "country": "United Kingdom",
                        "countryCode": "X0",
                        "isForeignLocation": 1,
                    }
                },
                "filings": {
                    "recent": {
                        "form": ["6-K", "20-F"],
                        "filingDate": ["2026-07-01", "2026-03-12"],
                        "accessionNumber": ["current", "annual"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn=conn)
    output_dir = tmp_path / "output"
    result = run_us_equity_census(
        conn=conn,
        inputs=CensusInputPaths(
            sec_registry=sec_path,
            nasdaq_listed=nasdaq_path,
            other_listed=other_path,
            companiesmarketcap_pages=(cmc_path,),
            stockanalysis_html=stock_path,
            submissions_dir=submissions_dir,
        ),
        output_dir=output_dir,
        as_of_date="2026-07-17",
        run_id="smoke_census",
        promote_sector_population=False,
    )

    assert result.status == "COMPLETED"
    assert result.counts["discovered_securities"] == 2
    assert result.counts["deduplicated_issuers"] == 2
    assert result.counts["admitted_large_and_mega_issuers"] == 1
    assert result.counts["admitted_large_cap_issuers"] == 1
    assert result.counts["admitted_mega_cap_issuers"] == 0
    assert result.actual_llm_calls == 0
    assert result.actual_llm_cost_usd == 0.0

    issuer = conn.execute(
        "SELECT primary_ticker, market_cap_usd, cap_band, canonical_sector, terminal_disposition "
        "FROM us_equity_census_issuers WHERE run_id = 'smoke_census'"
    ).fetchone()
    assert tuple(issuer) == (
        "ALP",
        12_000_000_000.0,
        "large_and_mega",
        "enterprise_software",
        "ADMITTED_LARGE_AND_MEGA",
    )
    security = conn.execute(
        "SELECT security_type, is_primary_security, terminal_disposition "
        "FROM us_equity_census_securities "
        "WHERE run_id = 'smoke_census' AND ticker = 'ALP'"
    ).fetchone()
    assert tuple(security) == (
        "COMMON_EQUITY",
        1,
        "ADMITTED_PRIMARY_COMMON_EQUITY",
    )
    foreign_issuer = conn.execute(
        "SELECT operating_status, is_us_domiciled, is_us_listed_foreign_issuer, "
        "terminal_disposition, terminal_reason_code "
        "FROM us_equity_census_issuers "
        "WHERE run_id = 'smoke_census' AND primary_ticker = 'FADS'"
    ).fetchone()
    assert tuple(foreign_issuer) == (
        "FOREIGN_FILER",
        0,
        1,
        "OUT_OF_SCOPE_FOREIGN_ISSUER",
        "FOREIGN_FILER",
    )
    foreign_security = conn.execute(
        "SELECT security_type, is_adr, is_primary_security, terminal_disposition "
        "FROM us_equity_census_securities "
        "WHERE run_id = 'smoke_census' AND ticker = 'FADS'"
    ).fetchone()
    assert tuple(foreign_security) == (
        "ADR_ADS",
        1,
        1,
        "OUT_OF_SCOPE_FOREIGN_ISSUER",
    )
    attempts = conn.execute(
        "SELECT stage, status, output_json FROM us_equity_census_attempts "
        "WHERE run_id = 'smoke_census' ORDER BY id"
    ).fetchall()
    assert [row["stage"] for row in attempts] == list(STAGES)
    assert {row["status"] for row in attempts} == {"COMPLETED"}
    for row in attempts[:-1]:
        output = json.loads(row["output_json"])
        assert output["checkpoint_stage"] == row["stage"]
        assert all(output["postconditions"].values())
    primary_output = json.loads(attempts[4]["output_json"])
    assert primary_output["next_stage"] == "ISSUER_DEDUPLICATION"
    assert (output_dir / "security_census.csv").exists()
    assert (output_dir / "issuer_census.csv").exists()
    assert (output_dir / "coverage_summary.json").exists()
    assert (output_dir / "coverage_report.md").exists()
    assert (output_dir / "source_snapshots" / "company_tickers_exchange.json").exists()
    assert json.loads((output_dir / "coverage_summary.json").read_text())["actual_llm_calls"] == 0

    replay = run_us_equity_census(
        conn=conn,
        inputs=CensusInputPaths(
            sec_registry=sec_path,
            nasdaq_listed=nasdaq_path,
            other_listed=other_path,
            companiesmarketcap_pages=(cmc_path,),
            stockanalysis_html=stock_path,
            submissions_dir=submissions_dir,
        ),
        output_dir=output_dir,
        as_of_date="2026-07-17",
        run_id="smoke_census",
        promote_sector_population=False,
    )
    assert replay.acceptance_checks["provider_free_replay_verified"] is True
    assert (
        "Provider-free replay verification passed"
        in (output_dir / "coverage_report.md").read_text()
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM us_equity_census_attempts WHERE run_id = 'smoke_census'"
        ).fetchone()[0]
        == 24
    )
    conn.close()


def test_uncorroborated_sec_listing_and_unresolved_common_classes_are_fail_visible(tmp_path):
    sec_path = tmp_path / "company_tickers_exchange.json"
    sec_path.write_text(
        json.dumps(
            {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [
                    [1002, "Beta Inc.", "BET", "Nasdaq"],
                    [1003, "Gamma Inc.", "GAMA", "Nasdaq"],
                    [1003, "Gamma Inc.", "GAMB", "Nasdaq"],
                ],
            }
        ),
        encoding="utf-8",
    )
    nasdaq_path = tmp_path / "nasdaqlisted.txt"
    nasdaq_path.write_text(
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        "GAMA|Gamma Inc. Class A Common Stock|Q|N|N|100|N|N\n"
        "GAMB|Gamma Inc. Class B Common Stock|Q|N|N|100|N|N\n"
        "File Creation Time: 0717202621:31|||||||\n",
        encoding="utf-8",
    )
    other_path = tmp_path / "otherlisted.txt"
    other_path.write_text(
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "File Creation Time: 0717202621:31||||||\n",
        encoding="utf-8",
    )
    cmc_path = tmp_path / "cmc.html"
    cmc_path.write_text(
        '<table><tr><td class="rank-td td-right">1</td>'
        '<td class="name-td"><div class="company-name">Beta Inc.</div>'
        '<div class="company-code">BET</div></td>'
        '<td class="td-right" data-sort="12000000000">$12 B</td></tr></table>',
        encoding="utf-8",
    )
    submissions_dir = tmp_path / "submissions"
    submissions_dir.mkdir()
    for cik, name in ((1002, "Beta Inc."), (1003, "Gamma Inc.")):
        (submissions_dir / f"{cik:010d}.json").write_text(
            json.dumps(
                {
                    "cik": f"{cik:010d}",
                    "name": name,
                    "sic": "3571",
                    "stateOfIncorporation": "DE",
                    "stateOfIncorporationDescription": "Delaware",
                    "entityType": "operating",
                    "filings": {
                        "recent": {
                            "form": ["10-K"],
                            "filingDate": ["2026-02-20"],
                            "accessionNumber": [f"{cik:010d}-26-000001"],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn=conn)
    run_us_equity_census(
        conn=conn,
        inputs=CensusInputPaths(
            sec_registry=sec_path,
            nasdaq_listed=nasdaq_path,
            other_listed=other_path,
            companiesmarketcap_pages=(cmc_path,),
            submissions_dir=submissions_dir,
        ),
        output_dir=tmp_path / "output",
        as_of_date="2026-07-17",
        run_id="disposition_regression",
        promote_sector_population=False,
    )

    beta_issuer = conn.execute(
        "SELECT scope_reason_code, terminal_disposition FROM us_equity_census_issuers "
        "WHERE run_id = 'disposition_regression' AND cik = '1002'"
    ).fetchone()
    assert tuple(beta_issuer) == (
        "ABSENT_FROM_CURRENT_NASDAQ_TRADER_DIRECTORY",
        "OUT_OF_SCOPE_NOT_IN_CURRENT_NASDAQ_TRADER_DIRECTORY",
    )
    beta_security = conn.execute(
        "SELECT listing_status, terminal_disposition FROM us_equity_census_securities "
        "WHERE run_id = 'disposition_regression' AND ticker = 'BET'"
    ).fetchone()
    assert tuple(beta_security) == (
        "SEC_REGISTRY_ONLY_UNCORROBORATED",
        "OUT_OF_SCOPE_INACTIVE_OR_UNCORROBORATED_LISTING",
    )

    gamma_issuer = conn.execute(
        "SELECT primary_selection_status, primary_security_key, terminal_disposition "
        "FROM us_equity_census_issuers "
        "WHERE run_id = 'disposition_regression' AND cik = '1003'"
    ).fetchone()
    assert tuple(gamma_issuer) == (
        "NEEDS_DATA",
        None,
        "NEEDS_DATA_PRIMARY_SECURITY",
    )
    gamma_securities = conn.execute(
        "SELECT ticker, is_primary_security, terminal_disposition "
        "FROM us_equity_census_securities "
        "WHERE run_id = 'disposition_regression' AND issuer_key = 'CIK:1003' "
        "ORDER BY ticker"
    ).fetchall()
    assert [tuple(row) for row in gamma_securities] == [
        ("GAMA", 0, "NEEDS_DATA_PRIMARY_SECURITY"),
        ("GAMB", 0, "NEEDS_DATA_PRIMARY_SECURITY"),
    ]
    conn.close()
