from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.autonomous.cap_resolver import SecurityIdentity
from app.autonomous.terminal_cap_search import (
    WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE,
    authorization_from_whole_run_preflight,
    bind_v2_cost_preflight_for_terminal_cap_search,
    build_authorized_terminal_cap_search,
    estimate_terminal_cap_search_worst_case_cost_usd,
    search_terminal_cap_with_openai,
    terminal_cap_search_ledger_fingerprint,
)
from app.autonomous.all_sector_cost_preflight import build_production_v2_cost_preflight
from app.llm.providers.disabled_provider import LLMResult


SOURCE_URL = "https://example.com/company/market-cap"


def _payload(**overrides):
    payload = {
        "status": "RESOLVED",
        "ticker": "AAA",
        "issuer_name": "Alpha Incorporated",
        "issuer_cik": "0000000123",
        "market_cap_basis": "DIRECT_ISSUER_MARKET_CAP",
        "market_cap_mm": 12_500.0,
        "market_cap_currency": "USD",
        "market_cap_units": "USD_MILLIONS",
        "as_of_date": "2026-06-10",
        "source_name": "Example Market Data",
        "source_url": SOURCE_URL,
        "confidence": "MEDIUM",
        "detail": "The cited page directly reported total issuer market cap.",
    }
    payload.update(overrides)
    return payload


def _raw(*, source_urls=(SOURCE_URL,), call_count=1):
    output = []
    for index in range(call_count):
        output.append(
            {
                "type": "web_search_call",
                "id": f"ws_{index + 1}",
                "status": "completed",
                "action": {"sources": [{"url": url} for url in source_urls]},
            }
        )
    output.append(
        {
            "type": "message",
            "content": [
                {
                    "type": "output_text",
                    "text": "Structured result follows.",
                    "annotations": [{"type": "url_citation", "url": url} for url in source_urls],
                }
            ],
        }
    )
    return {"id": "resp_terminal_cap", "output": output}


class _FakeProvider:
    provider_name = "openai"

    def __init__(self, payload, *, raw=None):
        self.payload = payload
        self.raw = raw if raw is not None else _raw()
        self.calls = []

    def synthesize_json(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResult(
            json_text=json.dumps(self.payload),
            model="gpt-5.5",
            usage_input_tokens=2000,
            usage_output_tokens=1000,
            usage_cached_input_tokens=1000,
            raw=self.raw,
        )


def _identity(*, cik="123"):
    return SecurityIdentity(
        ticker="AAA",
        issuer_cik=cik,
        issuer_primary_ticker="AAA",
        issuer_listed_tickers=("AAA",),
        security_role="PRIMARY",
        is_adr=False,
        is_secondary_class=False,
    )


def _whole_run_preflight(
    *,
    max_attempts=2,
    max_tool_calls=2,
    ledger_path="/tmp/ivi-terminal-cap-test-ledger.json",
    **overrides,
):
    reserve = estimate_terminal_cap_search_worst_case_cost_usd(
        max_attempts=max_attempts,
        max_tool_calls_per_attempt=max_tool_calls,
    )
    other_lanes = 80.0 - reserve
    payload = {
        "artifact_type": WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE,
        "status": "AUTHORIZED",
        "run_id": "all-sector-preflight-1",
        "authorized_at": "2026-06-11T12:00:00+00:00",
        "model": "gpt-5.5",
        "request_fingerprint": "a" * 64,
        "max_cost_usd": 100.0,
        "worst_case_cost_usd": 80.0,
        "terminal_cap_search_reserved_cost_usd": reserve,
        "lane_worst_case_costs_usd": {
            "provider_preflight": 0.05,
            "parent_research": 10.0,
            "company_underwriting": other_lanes - 15.05,
            "selected_company_validation": 5.0,
            "repair_fallback": reserve,
        },
        "terminal_cap_search_max_attempts": max_attempts,
        "terminal_cap_search_max_tool_calls_per_attempt": max_tool_calls,
        "terminal_cap_search_ledger_fingerprint": (
            terminal_cap_search_ledger_fingerprint(ledger_path)
        ),
    }
    payload.update(overrides)
    return payload


def _authorization(*, max_tool_calls=2):
    return authorization_from_whole_run_preflight(
        _whole_run_preflight(max_tool_calls=max_tool_calls)
    )


def test_six_lane_v2_preflight_binds_to_terminal_ledger_without_losing_lane() -> None:
    estimate = build_production_v2_cost_preflight(
        sector_candidate_counts={"energy": 1},
        terminal_cap_search_attempts=1,
        parent_max_turns=1,
        prior_realized_cost_usd="1.000000",
    ).to_dict()
    assert estimate["status"] == "AUTHORIZED"
    preflight = {
        "artifact_type": "all_sector_execution_authorization_v1",
        "status": "AUTHORIZED",
        "spend_authorized": True,
        "reason_codes": [],
        "estimate": estimate,
    }
    ledger_path = "/tmp/v2-six-lane-terminal-cap-ledger.json"

    bound = bind_v2_cost_preflight_for_terminal_cap_search(
        preflight,
        run_id="v2-six-lane-run",
        authorized_at="2026-07-16T12:00:00+00:00",
        request_fingerprint="b" * 64,
        ledger_path=ledger_path,
        max_tool_calls_per_attempt=4,
    )
    authorization = authorization_from_whole_run_preflight(bound)

    assert set(bound["estimate"]["lane_costs"]) == {
        "provider_preflight",
        "parent_research",
        "company_underwriting",
        "selected_company_validation",
        "repair_fallback",
        "terminal_cap_search",
    }
    assert authorization.run_id == "v2-six-lane-run"
    assert authorization.max_attempts == 1
    assert authorization.max_tool_calls_per_attempt == 4
    assert authorization.terminal_cap_search_reserved_cost_usd == float(
        estimate["lane_costs"]["terminal_cap_search"]["cost_usd"]
    )
    assert authorization.ledger_fingerprint == terminal_cap_search_ledger_fingerprint(ledger_path)


def test_terminal_cap_search_returns_only_cited_direct_issuer_cap():
    provider = _FakeProvider(_payload())

    result = search_terminal_cap_with_openai(
        ticker="AAA",
        as_of_date="2026-06-11",
        identity=_identity(),
        provider=provider,
        max_tool_calls=2,
        authorization=_authorization(max_tool_calls=2),
    )

    assert result.status == "RESOLVED"
    assert result.reason_code == "DIRECT_ISSUER_CAP_RESOLVED"
    assert result.evidence is not None
    assert result.evidence.market_cap_mm == 12_500.0
    assert result.evidence.issuer_cik == "0000000123"
    assert result.evidence.source_url == SOURCE_URL
    assert result.evidence.price_used is None
    assert result.web_search_call_count == 1
    assert [record["call_type"] for record in result.usage_records] == [
        "responses_model",
        "web_search_call",
    ]
    assert result.usage_records[0]["cost_estimate_usd"] == 0.0355
    assert result.usage_records[1]["cost_estimate_usd"] == 0.01
    assert result.total_cost_usd == 0.0455

    request = provider.calls[0]
    assert request["tools"] == [{"type": "web_search", "search_context_size": "low"}]
    assert request["include"] == ["web_search_call.action.sources"]
    assert request["max_tool_calls"] == 2
    assert request["model"] == "gpt-5.5"
    assert request["service_tier"] == "default"
    assert request["allow_output_token_retry"] is False
    assert request["max_retries_per_request"] == 0
    assert "Never calculate market cap from shares" in request["prompt"]


def test_authorized_callback_rejects_ticker_outside_frozen_execution_set(
    tmp_path,
) -> None:
    ledger_path = tmp_path / "terminal-cap-bound-ledger.json"
    provider = _FakeProvider(_payload())
    preflight = _whole_run_preflight(
        max_attempts=1,
        ledger_path=str(ledger_path),
        allowed_tickers=["AAA"],
        execution_fingerprint="c" * 64,
    )
    callback = build_authorized_terminal_cap_search(
        preflight=preflight,
        provider=provider,
        ledger_path=ledger_path,
    )

    result = callback("BBB", "2026-06-11", _identity())

    assert result.status == "REJECTED"
    assert result.reason_code == "TICKER_OUTSIDE_AUTHORIZED_EXECUTION_SET"
    assert result.web_search_call_count == 0
    assert result.total_cost_usd == 0.0
    assert callback.attempt_count == 0
    assert provider.calls == []


def test_terminal_cap_search_rejects_uncited_source_url_but_counts_spend():
    provider = _FakeProvider(
        _payload(source_url="https://uncited.example/market-cap"),
        raw=_raw(source_urls=(SOURCE_URL,)),
    )

    result = search_terminal_cap_with_openai(
        ticker="AAA",
        as_of_date="2026-06-11",
        identity=_identity(),
        provider=provider,
        authorization=_authorization(),
    )

    assert result.status == "REJECTED"
    assert result.reason_code == "SOURCE_URL_NOT_IN_RESPONSE_SOURCES"
    assert result.evidence is None
    assert result.total_cost_usd == 0.0455


def test_terminal_cap_search_rejects_shares_times_quote_result():
    provider = _FakeProvider(_payload(market_cap_basis="SHARES_TIMES_QUOTE"))

    result = search_terminal_cap_with_openai(
        ticker="AAA",
        as_of_date="2026-06-11",
        identity=_identity(),
        provider=provider,
        authorization=_authorization(),
    )

    assert result.status == "REJECTED"
    assert result.reason_code == "NON_DIRECT_MARKET_CAP"
    assert result.evidence is None


def test_terminal_cap_search_rejects_future_evidence():
    provider = _FakeProvider(_payload(as_of_date="2026-06-12"))

    result = search_terminal_cap_with_openai(
        ticker="AAA",
        as_of_date="2026-06-11",
        identity=_identity(),
        provider=provider,
        authorization=_authorization(),
    )

    assert result.status == "REJECTED"
    assert result.reason_code == "EVIDENCE_DATE_OUT_OF_RANGE"


def test_terminal_cap_search_defensively_rejects_excess_tool_calls():
    provider = _FakeProvider(_payload(), raw=_raw(call_count=3))

    result = search_terminal_cap_with_openai(
        ticker="AAA",
        as_of_date="2026-06-11",
        identity=_identity(),
        provider=provider,
        max_tool_calls=2,
        authorization=_authorization(max_tool_calls=2),
    )

    assert result.status == "REJECTED"
    assert result.reason_code == "TOOL_CALL_LIMIT_EXCEEDED"
    assert result.web_search_call_count == 3
    assert len(result.usage_records) == 4
    assert result.total_cost_usd == 0.0655


def test_terminal_cap_search_requires_resolved_cik_before_spend():
    provider = _FakeProvider(_payload())

    result = search_terminal_cap_with_openai(
        ticker="AAA",
        as_of_date="2026-06-11",
        identity=_identity(cik=None),
        provider=provider,
        authorization=_authorization(),
    )

    assert result.status == "REJECTED"
    assert result.reason_code == "IDENTITY_CIK_REQUIRED"
    assert result.total_cost_usd == 0.0
    assert provider.calls == []


def test_terminal_cap_search_tool_bound_is_hard_capped():
    provider = _FakeProvider(_payload())
    with pytest.raises(ValueError, match="between 1 and 4"):
        search_terminal_cap_with_openai(
            ticker="AAA",
            as_of_date="2026-06-11",
            identity=_identity(),
            provider=provider,
            max_tool_calls=5,
        )
    assert provider.calls == []


def test_terminal_cap_worst_case_uses_full_long_context_price_bound() -> None:
    one_attempt = estimate_terminal_cap_search_worst_case_cost_usd(
        max_attempts=1,
        max_tool_calls_per_attempt=2,
    )
    ten_attempts = estimate_terminal_cap_search_worst_case_cost_usd(
        max_attempts=10,
        max_tool_calls_per_attempt=2,
    )

    # 1,048,800 long-context input tokens ($10/M), 1,200 output tokens
    # ($45/M), and two $0.01 web-search calls.
    assert one_attempt == 10.562
    assert ten_attempts == 105.62


def test_terminal_cap_authorization_requires_whole_run_preflight_under_100() -> None:
    with pytest.raises(ValueError, match="whole-run cost preflight artifact"):
        authorization_from_whole_run_preflight({"status": "AUTHORIZED"})

    with pytest.raises(ValueError, match="within \\$100.00"):
        authorization_from_whole_run_preflight(_whole_run_preflight(max_cost_usd=100.01))

    with pytest.raises(ValueError, match="terminal-cap reserve"):
        authorization_from_whole_run_preflight(
            _whole_run_preflight(terminal_cap_search_reserved_cost_usd=0.01)
        )

    with pytest.raises(ValueError, match="must be finite"):
        authorization_from_whole_run_preflight(_whole_run_preflight(max_cost_usd=float("nan")))

    with pytest.raises(ValueError, match="every canonical cost lane"):
        authorization_from_whole_run_preflight(_whole_run_preflight(lane_worst_case_costs_usd={}))

    with pytest.raises(ValueError, match="lane sum"):
        authorization_from_whole_run_preflight(
            _whole_run_preflight(
                lane_worst_case_costs_usd={
                    "provider_preflight": 0.05,
                    "parent_research": 10.0,
                    "company_underwriting": 10.0,
                    "selected_company_validation": 5.0,
                    "repair_fallback": 25.0,
                }
            )
        )


def test_terminal_cap_authorization_cannot_be_replayed_to_another_ledger(tmp_path) -> None:
    authorized_ledger = tmp_path / "authorized-ledger.json"
    different_ledger = tmp_path / "fresh-ledger.json"

    with pytest.raises(ValueError, match="bound to a different ledger"):
        build_authorized_terminal_cap_search(
            preflight=_whole_run_preflight(ledger_path=authorized_ledger),
            provider=_FakeProvider(_payload()),
            ledger_path=different_ledger,
        )

    assert not different_ledger.exists()


def test_authorized_callback_persists_attempt_evidence_and_exact_usage(tmp_path) -> None:
    provider = _FakeProvider(_payload())
    ledger_path = tmp_path / "terminal-cap-search.json"
    callback = build_authorized_terminal_cap_search(
        preflight=_whole_run_preflight(ledger_path=ledger_path),
        provider=provider,
        ledger_path=ledger_path,
    )

    first = callback("AAA", "2026-06-11", _identity())
    second = callback("AAA", "2026-06-11", _identity())

    assert first.status == "RESOLVED"
    assert second.status == "RESOLVED"
    assert second.reason_code == "DIRECT_ISSUER_CAP_RESOLVED_FROM_LEDGER"
    assert second.total_cost_usd == 0.0
    assert len(provider.calls) == 1
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["artifact_type"] == "terminal_cap_search_attempt_ledger_v1"
    assert ledger["attempt_count"] == 1
    assert ledger["resolved_count"] == 1
    assert ledger["attempts"][0]["status"] == "RESOLVED"
    assert ledger["evidence"][0] == {
        "as_of_date": "2026-06-10",
        "confidence": "MEDIUM",
        "detail": "The cited page directly reported total issuer market cap.",
        "issuer_cik": "0000000123",
        "issuer_name": "Alpha Incorporated",
        "market_cap_basis": "DIRECT_ISSUER_MARKET_CAP",
        "market_cap_currency": "USD",
        "market_cap_mm": 12_500.0,
        "market_cap_units": "USD_MILLIONS",
        "security_name": "AAA",
        "source_kind": "SEARCH",
        "source_name": "Example Market Data",
        "source_url": SOURCE_URL,
        "ticker": "AAA",
    }
    assert ledger["usage"] == {
        "cached_input_tokens": 1000,
        "cost_estimate_usd": 0.0455,
        "input_tokens": 2000,
        "output_tokens": 1000,
        "response_calls": 1,
        "reserved_web_search_calls": 0,
        "web_search_calls": 1,
        "worst_case_reserved_attempts": 0,
    }

    # The generic terminal evidence loader can replay the exact persisted row.
    from app.autonomous.terminal_cap_evidence import load_terminal_cap_evidence

    loaded = load_terminal_cap_evidence(ledger_path)
    assert loaded["AAA"][0].source_url == SOURCE_URL
    assert loaded["AAA"][0].issuer_cik == "0000000123"


def test_authorized_callback_refuses_attempt_without_full_remaining_reserve(tmp_path) -> None:
    provider = _FakeProvider(_payload())
    preflight = _whole_run_preflight(
        max_attempts=2,
        max_tool_calls=2,
        ledger_path=tmp_path / "terminal-cap-search.json",
    )
    callback = build_authorized_terminal_cap_search(
        preflight=preflight,
        provider=provider,
        ledger_path=tmp_path / "terminal-cap-search.json",
    )
    one_attempt_reserve = estimate_terminal_cap_search_worst_case_cost_usd(
        max_attempts=1,
        max_tool_calls_per_attempt=2,
    )
    callback._attempts.append(  # noqa: SLF001 - simulate resumed billed usage
        {
            "usage_records": [
                {
                    "call_type": "responses_model",
                    "cost_estimate_usd": one_attempt_reserve + 0.01,
                }
            ]
        }
    )
    callback._write_ledger()  # noqa: SLF001 - persist resumed billed usage

    rejected = callback("BBB", "2026-06-11", _identity())

    assert rejected.status == "REJECTED"
    assert rejected.reason_code == "AUTHORIZED_COST_RESERVE_EXHAUSTED"
    assert provider.calls == []


def test_authorized_callback_reserves_before_provider_crash(monkeypatch, tmp_path) -> None:
    ledger_path = tmp_path / "crash-safe-ledger.json"
    provider = _FakeProvider(_payload())
    callback = build_authorized_terminal_cap_search(
        preflight=_whole_run_preflight(
            max_attempts=1,
            max_tool_calls=2,
            ledger_path=ledger_path,
        ),
        provider=provider,
        ledger_path=ledger_path,
    )

    def crash_after_boundary(**kwargs):
        raise KeyboardInterrupt("simulated process interruption")

    monkeypatch.setattr(
        "app.autonomous.terminal_cap_search.search_terminal_cap_with_openai",
        crash_after_boundary,
    )

    with pytest.raises(KeyboardInterrupt, match="process interruption"):
        callback("AAA", "2026-06-11", _identity())

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["attempt_count"] == 1
    assert ledger["attempts"][0]["status"] == "PENDING"
    assert ledger["usage"]["worst_case_reserved_attempts"] == 1
    assert ledger["usage"]["reserved_web_search_calls"] == 2
    assert ledger["usage"]["cost_estimate_usd"] == 10.562

    resumed = build_authorized_terminal_cap_search(
        preflight=_whole_run_preflight(
            max_attempts=1,
            max_tool_calls=2,
            ledger_path=ledger_path,
        ),
        provider=provider,
        ledger_path=ledger_path,
    )
    rejected = resumed("AAA", "2026-06-11", _identity())
    assert rejected.reason_code == "AUTHORIZED_ATTEMPT_LIMIT_EXHAUSTED"
    assert provider.calls == []


def test_authorized_callbacks_share_one_cross_process_ledger_budget(tmp_path) -> None:
    ledger_path = tmp_path / "shared-ledger.json"
    provider = _FakeProvider(_payload())
    preflight = _whole_run_preflight(
        max_attempts=1,
        max_tool_calls=2,
        ledger_path=ledger_path,
    )
    first = build_authorized_terminal_cap_search(
        preflight=preflight,
        provider=provider,
        ledger_path=ledger_path,
    )
    second = build_authorized_terminal_cap_search(
        preflight=preflight,
        provider=provider,
        ledger_path=ledger_path,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda callback: callback("AAA", "2026-06-11", _identity()),
                (first, second),
            )
        )

    assert len(provider.calls) == 1
    assert {result.reason_code for result in results} == {
        "DIRECT_ISSUER_CAP_RESOLVED",
        "DIRECT_ISSUER_CAP_RESOLVED_FROM_LEDGER",
    }


def test_authorized_callback_keeps_full_reserve_when_usage_is_missing(tmp_path) -> None:
    class MissingUsageProvider(_FakeProvider):
        def synthesize_json(self, **kwargs):
            self.calls.append(kwargs)
            return LLMResult(
                json_text=json.dumps(self.payload),
                model="gpt-5.5",
                usage_input_tokens=None,
                usage_output_tokens=None,
                usage_cached_input_tokens=None,
                raw=self.raw,
            )

    ledger_path = tmp_path / "missing-usage-ledger.json"
    provider = MissingUsageProvider(_payload())
    callback = build_authorized_terminal_cap_search(
        preflight=_whole_run_preflight(
            max_attempts=1,
            max_tool_calls=2,
            ledger_path=ledger_path,
        ),
        provider=provider,
        ledger_path=ledger_path,
    )

    result = callback("AAA", "2026-06-11", _identity())

    assert result.status == "RESOLVED"
    assert result.total_cost_usd == 10.562
    assert result.usage_records[0]["billing_status"] == "WORST_CASE_RESERVED"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["attempts"][0]["usage_authoritative"] is False
    assert ledger["usage"]["cost_estimate_usd"] == 10.562
