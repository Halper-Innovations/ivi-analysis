from __future__ import annotations

from typing import Any, Iterable, Mapping


UNKNOWN = "UNKNOWN"
NONE = "NONE"

FACTS_OK = "FACTS_OK"
FACTS_RETRYABLE_TIMEOUT = "FACTS_RETRYABLE_TIMEOUT"
FACTS_RETRYABLE_NETWORK_FAILURE = "FACTS_RETRYABLE_NETWORK_FAILURE"
FACTS_RETRYABLE_RATE_LIMIT = "FACTS_RETRYABLE_RATE_LIMIT"
FACTS_NO_CACHE_OFFLINE = "FACTS_NO_CACHE_OFFLINE"
FACTS_PROVIDER_NO_DATA = "FACTS_PROVIDER_NO_DATA"
FACTS_PARSE_FAILURE = "FACTS_PARSE_FAILURE"
FACTS_PARTIAL_COVERAGE = "FACTS_PARTIAL_COVERAGE"
FACTS_TERMINAL_MISSING_KEY_INPUTS = "FACTS_TERMINAL_MISSING_KEY_INPUTS"

FACTS_ACTION_NONE = "NONE"
FACTS_ACTION_RETRY_COMPANYFACTS_HYDRATION = "RETRY_COMPANYFACTS_HYDRATION"
FACTS_ACTION_RECHECK_FACTS_CACHE = "RECHECK_FACTS_CACHE"
FACTS_ACTION_DEFER_UNTIL_EVIDENCE_REFRESH = "DEFER_UNTIL_EVIDENCE_REFRESH"

FAIL_DOMAIN_NONE = "NONE"
FAIL_DOMAIN_EVIDENCE = "EVIDENCE"
FAIL_DOMAIN_ECONOMICS = "ECONOMICS"
FAIL_DOMAIN_MIXED = "MIXED"
FAIL_DOMAIN_OTHER = "OTHER"

_RAW_FACT_KEYS = {
    "facts_status",
    "fetch_reason_code",
    "shares_status",
    "shares_reason",
    "shares_reason_code",
    "cfo_status",
    "cfo_reason",
    "cfo_reason_code",
    "capex_status",
    "capex_reason",
    "capex_reason_code",
    "fcf_status",
    "fcf_reason",
    "fcf_reason_code",
}
_MISSING_INPUT_FIELDS = (
    ("shares_status", "SHARES"),
    ("cfo_status", "CFO"),
    ("capex_status", "CAPEX"),
    ("fcf_status", "FCF"),
)
_NETWORK_EXCEPTION_TYPES = {
    "CONNECTIONERROR",
    "CONNECTTIMEOUT",
    "READTIMEOUT",
    "TIMEOUT",
    "PROXYERROR",
    "SSLERROR",
    "REQUESTEXCEPTION",
}
_PARSE_EXCEPTION_TYPES = {
    "JSONDECODEERROR",
    "VALUEERROR",
    "TYPEERROR",
}
_EVIDENCE_BLOCKERS = {
    "MISSING_PRICE",
    "MISSING_FACTS",
    "MISSING_FCF",
    "MISSING_CFO",
    "MISSING_CAPEX",
    "MISSING_SHARES",
    "MISSING_EV",
    "MISSING_GD_INPUTS",
}
_ECONOMIC_BLOCKERS = {
    "NEGATIVE_CFO",
    "NEGATIVE_FCF",
    "LOW_YIELD_OWNER_EARNINGS",
    "LOW_YIELD_FCF",
    "LOW_YIELD_OWNER_EARNINGS_EV",
    "LOW_YIELD_FCF_EV",
    "INSUFFICIENT_MOS",
    "INSUFFICIENT_MOS_EPV",
    "INSUFFICIENT_MOS_NETNET",
    "EXCESS_NET_DEBT",
    "EXCESS_DILUTION",
}


def _token(value: Any) -> str:
    return str(value or "").strip().upper()


def _bool(value: Any) -> bool:
    return bool(value)


def _dedupe_tokens(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = _token(value)
        if not token or token == NONE or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _raw_facts_present(row: Mapping[str, Any]) -> bool:
    return any(key in row for key in _RAW_FACT_KEYS)


def _missing_key_inputs(row: Mapping[str, Any]) -> list[str]:
    missing = [
        label
        for key, label in _MISSING_INPUT_FIELDS
        if _token(row.get(key) or UNKNOWN) != "OK"
    ]
    if _token(row.get("fetch_reason_code")) == "CIK_MISSING":
        missing.insert(0, "CIK_MAPPING")
    return _dedupe_tokens(missing)


def _reason_codes(row: Mapping[str, Any]) -> list[str]:
    return _dedupe_tokens(
        [
            row.get("fetch_reason_code"),
            row.get("shares_reason"),
            row.get("shares_reason_code"),
            row.get("cfo_reason"),
            row.get("cfo_reason_code"),
            row.get("capex_reason"),
            row.get("capex_reason_code"),
            row.get("fcf_reason"),
            row.get("fcf_reason_code"),
        ]
    )


def _exception_type(row: Mapping[str, Any]) -> str:
    return _token(row.get("fetch_last_exception_type"))


def _exception_message(row: Mapping[str, Any]) -> str:
    return str(row.get("fetch_last_exception_message") or row.get("fetch_reason_detail") or "").strip()


def _http_status(row: Mapping[str, Any]) -> int | None:
    value = row.get("http_status")
    return int(value) if isinstance(value, int) else (int(value) if isinstance(value, str) and value.isdigit() else None)


def _ok_count(row: Mapping[str, Any]) -> int:
    return len([label for key, label in _MISSING_INPUT_FIELDS if _token(row.get(key) or UNKNOWN) == "OK"])


def _network_failure(row: Mapping[str, Any]) -> bool:
    exc_type = _exception_type(row)
    message = _exception_message(row).lower()
    fetch_reason = _token(row.get("fetch_reason_code"))
    http_status = _http_status(row)
    if fetch_reason == "FETCH_5XX":
        return True
    if exc_type in _NETWORK_EXCEPTION_TYPES:
        return True
    if http_status is not None and http_status >= 500:
        return True
    return any(token in message for token in ["connection", "connect", "timeout", "timed out", "dns", "name resolution", "ssl"])


def _parse_failure(row: Mapping[str, Any]) -> bool:
    exc_type = _exception_type(row)
    message = _exception_message(row).lower()
    return exc_type in _PARSE_EXCEPTION_TYPES or "json" in message or "not a json object" in message


def _recommended_action_for_class(blocker_class: str) -> str:
    token = _token(blocker_class)
    if token in {
        FACTS_RETRYABLE_TIMEOUT,
        FACTS_RETRYABLE_NETWORK_FAILURE,
        FACTS_RETRYABLE_RATE_LIMIT,
    }:
        return FACTS_ACTION_RETRY_COMPANYFACTS_HYDRATION
    if token == FACTS_NO_CACHE_OFFLINE:
        return FACTS_ACTION_RECHECK_FACTS_CACHE
    if token == FACTS_PARTIAL_COVERAGE:
        return FACTS_ACTION_DEFER_UNTIL_EVIDENCE_REFRESH
    return FACTS_ACTION_NONE


def classify_facts_blocker(row: Mapping[str, Any]) -> dict[str, Any]:
    existing_class = _token(row.get("facts_blocker_class"))
    if existing_class and not _raw_facts_present(row):
        return {
            "facts_blocker_class": existing_class,
            "facts_blocker_retryable": _bool(row.get("facts_blocker_retryable")),
            "facts_blocker_terminal": _bool(row.get("facts_blocker_terminal")),
            "facts_blocker_partial_usable": _bool(row.get("facts_blocker_partial_usable")),
            "facts_missing_key_inputs": [str(value) for value in (row.get("facts_missing_key_inputs") or []) if str(value).strip()],
            "facts_retry_recommended": _bool(row.get("facts_retry_recommended")),
            "facts_blocker_reason_codes": [
                str(value)
                for value in (row.get("facts_blocker_reason_codes") or [])
                if str(value).strip()
            ],
            "facts_recommended_action": str(row.get("facts_recommended_action") or _recommended_action_for_class(existing_class)),
        }

    if not _raw_facts_present(row):
        return {
            "facts_blocker_class": FACTS_OK,
            "facts_blocker_retryable": False,
            "facts_blocker_terminal": False,
            "facts_blocker_partial_usable": False,
            "facts_missing_key_inputs": [],
            "facts_retry_recommended": False,
            "facts_blocker_reason_codes": [],
            "facts_recommended_action": FACTS_ACTION_NONE,
        }

    facts_status = _token(row.get("facts_status") or row.get("status"))
    fetch_reason = _token(row.get("fetch_reason_code"))
    missing_key_inputs = _missing_key_inputs(row)
    ok_count = _ok_count(row)
    partial_usable = facts_status == "PARTIAL" or (0 < ok_count < len(_MISSING_INPUT_FIELDS))

    blocker_class = FACTS_OK
    retryable = False
    terminal = False

    if facts_status == "OK" and not missing_key_inputs:
        blocker_class = FACTS_OK
    elif fetch_reason == "SCOUT_FACTS_TIMEOUT":
        blocker_class = FACTS_RETRYABLE_TIMEOUT
        retryable = True
    elif fetch_reason == "OFFLINE_NO_CACHE":
        blocker_class = FACTS_NO_CACHE_OFFLINE
        retryable = True
    elif fetch_reason == "BUDGET_EXHAUSTED":
        blocker_class = FACTS_RETRYABLE_RATE_LIMIT
        retryable = True
    elif fetch_reason == "FETCH_4XX" and _http_status(row) in {403, 429}:
        blocker_class = FACTS_RETRYABLE_RATE_LIMIT
        retryable = True
    elif fetch_reason == "EXCEPTION" and "budget" in _exception_message(row).lower():
        blocker_class = FACTS_RETRYABLE_RATE_LIMIT
        retryable = True
    elif fetch_reason in {"FETCH_5XX", "EXCEPTION"} and _network_failure(row):
        blocker_class = FACTS_RETRYABLE_NETWORK_FAILURE
        retryable = True
    elif fetch_reason == "FETCH_4XX":
        blocker_class = FACTS_PROVIDER_NO_DATA
        terminal = True
    elif fetch_reason == "COMPANYFACTS_MISS":
        blocker_class = FACTS_PROVIDER_NO_DATA
        terminal = True
    elif fetch_reason == "CIK_MISSING":
        blocker_class = FACTS_TERMINAL_MISSING_KEY_INPUTS
        terminal = True
    elif fetch_reason == "EXCEPTION" and _parse_failure(row):
        blocker_class = FACTS_PARSE_FAILURE
        terminal = True
    elif partial_usable:
        blocker_class = FACTS_PARTIAL_COVERAGE
    elif missing_key_inputs:
        blocker_class = FACTS_TERMINAL_MISSING_KEY_INPUTS
        terminal = True
    elif facts_status in {"UNKNOWN", "PARTIAL"}:
        blocker_class = FACTS_TERMINAL_MISSING_KEY_INPUTS
        terminal = True

    facts_recommended_action = _recommended_action_for_class(blocker_class)
    return {
        "facts_blocker_class": blocker_class,
        "facts_blocker_retryable": retryable,
        "facts_blocker_terminal": terminal,
        "facts_blocker_partial_usable": blocker_class == FACTS_PARTIAL_COVERAGE,
        "facts_missing_key_inputs": missing_key_inputs,
        "facts_retry_recommended": facts_recommended_action != FACTS_ACTION_NONE,
        "facts_blocker_reason_codes": _reason_codes(row),
        "facts_recommended_action": facts_recommended_action,
    }


def classify_fail_domain(row: Mapping[str, Any]) -> dict[str, Any]:
    scout_status = _token(row.get("scout_status") or row.get("value_gate_status"))
    if scout_status != "FAIL":
        return {
            "fail_due_to_missing_evidence": False,
            "fail_due_to_economic_weakness": False,
            "primary_fail_domain": FAIL_DOMAIN_NONE,
        }

    categories = {
        _token(row.get("primary_blocker_category")),
        _token(row.get("primary_blocker")),
        *[_token(value) for value in (row.get("blocker_categories") or [])],
    }
    categories.discard("")
    categories.discard(NONE)
    categories.discard(UNKNOWN)

    facts_blocker_class = _token(row.get("facts_blocker_class"))
    evidence_fail = any(code in _EVIDENCE_BLOCKERS for code in categories) or facts_blocker_class not in {FACTS_OK, ""}
    economics_fail = any(code in _ECONOMIC_BLOCKERS for code in categories)

    if evidence_fail and economics_fail:
        domain = FAIL_DOMAIN_MIXED
    elif evidence_fail:
        domain = FAIL_DOMAIN_EVIDENCE
    elif economics_fail:
        domain = FAIL_DOMAIN_ECONOMICS
    else:
        domain = FAIL_DOMAIN_OTHER

    return {
        "fail_due_to_missing_evidence": domain in {FAIL_DOMAIN_EVIDENCE, FAIL_DOMAIN_MIXED},
        "fail_due_to_economic_weakness": domain in {FAIL_DOMAIN_ECONOMICS, FAIL_DOMAIN_MIXED},
        "primary_fail_domain": domain,
    }


def enrich_facts_blocker_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    enriched.update(classify_facts_blocker(enriched))
    enriched.update(classify_fail_domain(enriched))
    return enriched


def summarize_facts_blockers(rows: Iterable[Mapping[str, Any]], *, top_n: int = 10) -> dict[str, Any]:
    enriched_rows = [enrich_facts_blocker_fields(row) for row in rows]
    facts_blocker_histogram: dict[str, int] = {}
    recommended_action_counts: dict[str, int] = {}
    fail_domain_counts: dict[str, int] = {
        FAIL_DOMAIN_EVIDENCE: 0,
        FAIL_DOMAIN_ECONOMICS: 0,
        FAIL_DOMAIN_MIXED: 0,
        FAIL_DOMAIN_OTHER: 0,
    }
    retryable_rows: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []

    for row in enriched_rows:
        blocker_class = str(row.get("facts_blocker_class") or FACTS_OK)
        facts_blocker_histogram[blocker_class] = facts_blocker_histogram.get(blocker_class, 0) + 1
        action = str(row.get("facts_recommended_action") or FACTS_ACTION_NONE)
        recommended_action_counts[action] = recommended_action_counts.get(action, 0) + 1

        domain = str(row.get("primary_fail_domain") or FAIL_DOMAIN_NONE)
        if domain in fail_domain_counts:
            fail_domain_counts[domain] += 1

        summary_row = {
            "ticker": str(row.get("ticker") or ""),
            "scout_status": str(row.get("scout_status") or row.get("value_gate_status") or UNKNOWN),
            "primary_blocker_category": str(row.get("primary_blocker_category") or row.get("primary_blocker") or UNKNOWN),
            "facts_blocker_class": blocker_class,
            "facts_missing_key_inputs": [str(value) for value in (row.get("facts_missing_key_inputs") or []) if str(value).strip()],
            "facts_recommended_action": action,
            "fetch_reason_code": str(row.get("fetch_reason_code") or UNKNOWN),
            "primary_fail_domain": domain,
        }
        if _bool(row.get("facts_blocker_retryable")):
            retryable_rows.append(summary_row)
        if _bool(row.get("facts_blocker_terminal")):
            terminal_rows.append(summary_row)
        if _bool(row.get("facts_blocker_partial_usable")):
            partial_rows.append(summary_row)

    def _sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            str(row.get("facts_blocker_class") or ""),
            str(row.get("primary_blocker_category") or ""),
            str(row.get("ticker") or ""),
        )

    retryable_rows = sorted(retryable_rows, key=_sort_key)
    terminal_rows = sorted(terminal_rows, key=_sort_key)
    partial_rows = sorted(partial_rows, key=_sort_key)

    return {
        "facts_blocker_histogram": dict(sorted(facts_blocker_histogram.items(), key=lambda item: (-int(item[1]), item[0]))),
        "retryable_facts_blocker_count": len(retryable_rows),
        "terminal_facts_blocker_count": len(terminal_rows),
        "partial_usable_facts_count": len(partial_rows),
        "top_retryable_facts_blockers": retryable_rows[: max(1, int(top_n))],
        "top_terminal_facts_blockers": terminal_rows[: max(1, int(top_n))],
        "top_partial_usable_facts": partial_rows[: max(1, int(top_n))],
        "recommended_next_action_counts": dict(sorted(recommended_action_counts.items(), key=lambda item: item[0])),
        "economic_fail_count_vs_evidence_fail_count": {
            "evidence_fail_count": int(fail_domain_counts[FAIL_DOMAIN_EVIDENCE]),
            "economic_fail_count": int(fail_domain_counts[FAIL_DOMAIN_ECONOMICS]),
            "mixed_fail_count": int(fail_domain_counts[FAIL_DOMAIN_MIXED]),
            "other_fail_count": int(fail_domain_counts[FAIL_DOMAIN_OTHER]),
        },
    }
