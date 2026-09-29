from __future__ import annotations

from app.universe.facts_blockers import (
    FACTS_ACTION_DEFER_UNTIL_EVIDENCE_REFRESH,
    FACTS_ACTION_RECHECK_FACTS_CACHE,
    FACTS_ACTION_RETRY_COMPANYFACTS_HYDRATION,
    FACTS_NO_CACHE_OFFLINE,
    FACTS_PARSE_FAILURE,
    FACTS_PARTIAL_COVERAGE,
    FACTS_PROVIDER_NO_DATA,
    FACTS_RETRYABLE_TIMEOUT,
    FACTS_TERMINAL_MISSING_KEY_INPUTS,
    FAIL_DOMAIN_ECONOMICS,
    FAIL_DOMAIN_EVIDENCE,
    FAIL_DOMAIN_MIXED,
    classify_fail_domain,
    classify_facts_blocker,
)


def test_retryable_companyfacts_timeout_maps_to_retryable_blocker_class():
    row = {
        "ticker": "AAA",
        "facts_status": "UNKNOWN",
        "fetch_reason_code": "SCOUT_FACTS_TIMEOUT",
        "shares_status": "UNKNOWN",
        "cfo_status": "UNKNOWN",
        "capex_status": "UNKNOWN",
        "fcf_status": "UNKNOWN",
    }

    payload = classify_facts_blocker(row)
    assert payload["facts_blocker_class"] == FACTS_RETRYABLE_TIMEOUT
    assert payload["facts_blocker_retryable"] is True
    assert payload["facts_recommended_action"] == FACTS_ACTION_RETRY_COMPANYFACTS_HYDRATION


def test_provider_no_data_parse_failure_and_offline_no_cache_map_deterministically():
    offline = classify_facts_blocker(
        {
            "ticker": "AAA",
            "facts_status": "UNKNOWN",
            "fetch_reason_code": "OFFLINE_NO_CACHE",
            "shares_status": "UNKNOWN",
            "cfo_status": "UNKNOWN",
            "capex_status": "UNKNOWN",
            "fcf_status": "UNKNOWN",
        }
    )
    assert offline["facts_blocker_class"] == FACTS_NO_CACHE_OFFLINE
    assert offline["facts_blocker_retryable"] is True
    assert offline["facts_recommended_action"] == FACTS_ACTION_RECHECK_FACTS_CACHE

    provider = classify_facts_blocker(
        {
            "ticker": "BBB",
            "facts_status": "UNKNOWN",
            "fetch_reason_code": "FETCH_4XX",
            "http_status": 404,
            "shares_status": "UNKNOWN",
            "cfo_status": "UNKNOWN",
            "capex_status": "UNKNOWN",
            "fcf_status": "UNKNOWN",
        }
    )
    assert provider["facts_blocker_class"] == FACTS_PROVIDER_NO_DATA
    assert provider["facts_blocker_terminal"] is True

    parse = classify_facts_blocker(
        {
            "ticker": "CCC",
            "facts_status": "UNKNOWN",
            "fetch_reason_code": "EXCEPTION",
            "fetch_last_exception_type": "JSONDecodeError",
            "fetch_last_exception_message": "Expecting value",
            "shares_status": "UNKNOWN",
            "cfo_status": "UNKNOWN",
            "capex_status": "UNKNOWN",
            "fcf_status": "UNKNOWN",
        }
    )
    assert parse["facts_blocker_class"] == FACTS_PARSE_FAILURE
    assert parse["facts_blocker_terminal"] is True


def test_partial_usable_facts_are_distinguished_from_terminal_missing_inputs():
    partial = classify_facts_blocker(
        {
            "ticker": "AAA",
            "facts_status": "PARTIAL",
            "fetch_reason_code": "CACHE_HIT",
            "shares_status": "OK",
            "cfo_status": "OK",
            "capex_status": "UNKNOWN",
            "fcf_status": "UNKNOWN",
            "capex_reason_code": "TAG_MISS",
            "fcf_reason_code": "TAG_MISS",
        }
    )
    assert partial["facts_blocker_class"] == FACTS_PARTIAL_COVERAGE
    assert partial["facts_blocker_partial_usable"] is True
    assert partial["facts_recommended_action"] == FACTS_ACTION_DEFER_UNTIL_EVIDENCE_REFRESH
    assert partial["facts_missing_key_inputs"] == ["CAPEX", "FCF"]

    terminal = classify_facts_blocker(
        {
            "ticker": "BBB",
            "facts_status": "UNKNOWN",
            "fetch_reason_code": "FETCH_OK",
            "shares_status": "UNKNOWN",
            "cfo_status": "UNKNOWN",
            "capex_status": "UNKNOWN",
            "fcf_status": "UNKNOWN",
            "shares_reason_code": "TAG_MISS",
            "cfo_reason_code": "TAG_MISS",
            "capex_reason_code": "TAG_MISS",
            "fcf_reason_code": "TAG_MISS",
        }
    )
    assert terminal["facts_blocker_class"] == FACTS_TERMINAL_MISSING_KEY_INPUTS
    assert terminal["facts_blocker_terminal"] is True


def test_fail_domain_separates_evidence_from_economic_weakness():
    evidence = classify_fail_domain(
        {
            "scout_status": "FAIL",
            "primary_blocker_category": "MISSING_FACTS",
            "blocker_categories": ["MISSING_FACTS"],
            "facts_blocker_class": FACTS_RETRYABLE_TIMEOUT,
        }
    )
    assert evidence["primary_fail_domain"] == FAIL_DOMAIN_EVIDENCE
    assert evidence["fail_due_to_missing_evidence"] is True
    assert evidence["fail_due_to_economic_weakness"] is False

    economics = classify_fail_domain(
        {
            "scout_status": "FAIL",
            "primary_blocker_category": "NEGATIVE_CFO",
            "blocker_categories": ["NEGATIVE_CFO"],
            "facts_blocker_class": "FACTS_OK",
        }
    )
    assert economics["primary_fail_domain"] == FAIL_DOMAIN_ECONOMICS
    assert economics["fail_due_to_missing_evidence"] is False
    assert economics["fail_due_to_economic_weakness"] is True

    mixed = classify_fail_domain(
        {
            "scout_status": "FAIL",
            "primary_blocker_category": "MISSING_FACTS",
            "blocker_categories": ["MISSING_FACTS", "NEGATIVE_CFO"],
            "facts_blocker_class": FACTS_RETRYABLE_TIMEOUT,
        }
    )
    assert mixed["primary_fail_domain"] == FAIL_DOMAIN_MIXED
    assert mixed["fail_due_to_missing_evidence"] is True
    assert mixed["fail_due_to_economic_weakness"] is True
