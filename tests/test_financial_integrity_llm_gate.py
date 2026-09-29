from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    InvalidFinancialInputError,
    stable_quote_snapshot_id,
)
from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    SectorCompanyFinancialPacket,
)
from app.autonomous import runtime, sector_runtime
from tests.financial_integrity_helpers import canonicalize_financial_packet


class _Provider:
    provider_name = "test"
    model = "test-model"
    _handles_retry_guard = True

    def __init__(self, events: list[str]):
        self.events = events
        self.calls: list[dict] = []

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, **kwargs):
        self.events.append("provider")
        self.calls.append(dict(kwargs))
        return SimpleNamespace(json_text=json.dumps({}))


def _packet(*, current_price: float = 50.0) -> dict:
    packet = {
        "ticker": "AAA",
        "market_cap_mm": 500.0,
        "market_cap_unit": "USD_millions",
        "market_cap_source": "fixture_market_cap",
        "market_cap_method": "current_price_times_issuer_reported_shares",
        "market_cap_effective_as_of_date": "2026-07-22",
        "current_price": current_price,
        "current_price_as_of_date": "2026-07-22",
        "current_price_currency": "USD",
        "current_price_source": "fixture",
        "current_price_source_url": "https://example.test/quote",
        "current_price_unit": "USD_per_share",
        "price_basis": "UNADJUSTED",
        "raw_price": current_price,
        "shares_outstanding_mm": 10.0,
        "shares_unit": "shares_millions",
        "shares_basis": "UNADJUSTED",
        "shares_as_of_date": "2026-07-22",
        "shares_source": "fixture_shares",
        "split_adjustment_factor": 1.0,
        "split_effective_date": None,
        "split_lineage_proof": {
            "status": "PASS",
            "period_start": "2026-07-22",
            "period_end": "2026-07-22",
            "verified_as_of": "2026-07-22",
            "source": "fixture_corporate_actions",
            "source_reference": "https://example.test/actions",
        },
        "metric_traces": {},
    }
    packet["quote_snapshot_id"] = stable_quote_snapshot_id(packet)
    packet.update(
        {
            "cap_stage_price": packet["current_price"],
            "cap_stage_price_as_of_date": packet["current_price_as_of_date"],
            "cap_stage_price_currency": packet["current_price_currency"],
            "cap_stage_price_source": packet["current_price_source"],
            "cap_stage_price_source_url": packet["current_price_source_url"],
            "cap_stage_quote_snapshot_id": packet["quote_snapshot_id"],
        }
    )
    return canonicalize_financial_packet(
        packet,
        as_of_date="2026-07-22",
        shares_mm=10.0,
    )


def _scope(*, current_price: float = 50.0) -> FinancialIntegrityScope:
    return FinancialIntegrityScope(
        context="test_paid_call",
        run_as_of_date="2026-07-22",
        packets=(_packet(current_price=current_price),),
    )


def _sector_packet() -> SectorCompanyFinancialPacket:
    packet = SectorCompanyFinancialPacket(
        ticker="AAA",
        financial_status="COMPLETE",
        model_fit_status="FIT",
        data_quality_status="OK",
        market_cap_mm=500.0,
        market_cap_unit="USD_millions",
        market_cap_source="fixture_market_cap",
        market_cap_method="current_price_times_issuer_reported_shares",
        market_cap_effective_as_of_date="2026-07-22",
        current_price=50.0,
        current_price_as_of_date="2026-07-22",
        current_price_currency="USD",
        current_price_source="fixture",
        current_price_source_url="https://example.test/quote",
        current_price_unit="USD_per_share",
        price_basis="UNADJUSTED",
        raw_price=50.0,
        shares_outstanding_mm=10.0,
        shares_unit="shares_millions",
        shares_basis="UNADJUSTED",
        shares_as_of_date="2026-07-22",
        shares_source="fixture_shares",
        split_adjustment_factor=1.0,
        split_lineage_proof={
            "status": "PASS",
            "period_start": "2026-07-22",
            "period_end": "2026-07-22",
            "verified_as_of": "2026-07-22",
            "source": "fixture_corporate_actions",
            "source_reference": "https://example.test/actions",
        },
    )
    packet.quote_snapshot_id = stable_quote_snapshot_id(packet.to_dict())
    packet.cap_stage_price = packet.current_price
    packet.cap_stage_price_as_of_date = packet.current_price_as_of_date
    packet.cap_stage_price_currency = packet.current_price_currency
    packet.cap_stage_price_source = packet.current_price_source
    packet.cap_stage_price_source_url = packet.current_price_source_url
    packet.cap_stage_quote_snapshot_id = packet.quote_snapshot_id
    return canonicalize_financial_packet(
        packet,
        as_of_date="2026-07-22",
        shares_mm=10.0,
    )


@pytest.mark.parametrize(
    "runtime_module",
    [runtime, sector_runtime],
    ids=["single_company", "sector"],
)
def test_applied_integrity_scope_returns_post_application_fingerprint(
    runtime_module,
):
    packet = _sector_packet()
    scope = FinancialIntegrityScope(
        context="post_application_fingerprint",
        run_as_of_date="2026-07-22",
        packets=(packet,),
    )

    pre_application = runtime_module.require_financial_integrity_scope(scope)
    applied = runtime_module._require_applied_financial_integrity_scope(scope)
    unchanged = runtime_module.require_unchanged_financial_integrity_scope(
        scope,
        expected_scope_fingerprint=applied.scope_fingerprint,
    )

    assert packet.financial_integrity_status == "PASS"
    assert pre_application.scope_fingerprint != applied.scope_fingerprint
    assert unchanged.scope_fingerprint == applied.scope_fingerprint


def test_sector_publication_binding_uses_exact_post_application_packet():
    packet = _sector_packet()
    scope = FinancialIntegrityScope(
        context="sector_post_application_binding",
        run_as_of_date="2026-07-22",
        packets=(packet,),
    )
    applied = sector_runtime._require_applied_financial_integrity_scope(scope)
    binding = sector_runtime._financial_integrity_run_binding(
        scope,
        scope_fingerprint=applied.scope_fingerprint,
    )

    sector_runtime._require_candidate_financial_integrity_binding(
        {"financial_integrity_binding": binding},
        packets=(packet,),
        scenarios=(),
    )

    assert binding["scope_fingerprint"] == applied.scope_fingerprint
    assert binding["packets"][0]["financial_integrity_status"] == "PASS"


def test_sector_gate_runs_before_cost_reservation_and_strips_internal_scope(monkeypatch):
    events: list[str] = []
    provider = _Provider(events)
    require_scope = sector_runtime.require_financial_integrity_scope

    def gated(scope):
        events.append("gate")
        return require_scope(scope)

    monkeypatch.setattr(sector_runtime, "require_financial_integrity_scope", gated)
    monkeypatch.setattr(
        sector_runtime,
        "_reserve_llm_cost_budget",
        lambda provider, kwargs: events.append("reserve"),
    )

    sector_runtime._call_provider_result_with_meta(
        provider,
        {
            "integrity_scope": _scope(),
            "prompt": "test",
            "schema": {"type": "object"},
            "schema_name": "test_paid_call",
        },
    )

    assert events[:4] == ["gate", "gate", "reserve", "provider"]
    assert "integrity_scope" not in provider.calls[0]


def test_sector_gate_rejection_makes_zero_reservations_and_zero_provider_calls(
    monkeypatch,
):
    events: list[str] = []
    provider = _Provider(events)
    monkeypatch.setattr(
        sector_runtime,
        "_reserve_llm_cost_budget",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid financial input must not reserve cost")
        ),
    )

    with pytest.raises(InvalidFinancialInputError):
        sector_runtime._call_provider_result_with_meta(
            provider,
            {
                "integrity_scope": _scope(current_price=float("nan")),
                "prompt": "test",
                "schema": {"type": "object"},
                "schema_name": "test_paid_call",
            },
        )

    assert events == []
    assert provider.calls == []


def test_candidate_gate_runs_before_cost_reservation_and_strips_internal_scope(
    monkeypatch,
):
    events: list[str] = []
    provider = _Provider(events)
    require_scope = runtime.require_financial_integrity_scope

    def gated(scope):
        events.append("gate")
        return require_scope(scope)

    monkeypatch.setattr(runtime, "require_financial_integrity_scope", gated)
    monkeypatch.setattr(
        runtime,
        "_reserve_provider_call_cost",
        lambda provider, kwargs: events.append("reserve"),
    )

    runtime._call_provider_json(
        provider,
        {
            "integrity_scope": _scope(),
            "prompt": "test",
            "schema": {"type": "object"},
            "schema_name": "test_paid_call",
        },
    )

    assert events[:4] == ["gate", "gate", "reserve", "provider"]
    assert "integrity_scope" not in provider.calls[0]


def test_candidate_gate_rejection_makes_zero_reservations_and_zero_provider_calls(
    monkeypatch,
):
    events: list[str] = []
    provider = _Provider(events)
    monkeypatch.setattr(
        runtime,
        "_reserve_provider_call_cost",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid financial input must not reserve cost")
        ),
    )

    with pytest.raises(InvalidFinancialInputError):
        runtime._call_provider_json(
            provider,
            {
                "integrity_scope": _scope(current_price=float("nan")),
                "prompt": "test",
                "schema": {"type": "object"},
                "schema_name": "test_paid_call",
            },
        )

    assert events == []
    assert provider.calls == []


@pytest.mark.parametrize(
    "call",
    [
        lambda provider, kwargs: runtime._call_provider_json(provider, kwargs),
        lambda provider, kwargs: sector_runtime._call_provider_result_with_meta(
            provider,
            kwargs,
        ),
    ],
)
def test_legacy_provider_signature_is_adapted_before_exactly_one_call(call):
    class LegacyProvider:
        provider_name = "test"
        model = "test-model"
        _handles_retry_guard = True

        def __init__(self):
            self.calls = 0

        def synthesize_json(self, *, prompt, schema, schema_name=None):
            self.calls += 1
            assert prompt == "test"
            assert schema == {"type": "object"}
            assert schema_name == "legacy_signature"
            return SimpleNamespace(json_text="{}")

    provider = LegacyProvider()
    call(
        provider,
        {
            "integrity_scope": _scope(),
            "prompt": "test",
            "schema": {"type": "object"},
            "schema_name": "legacy_signature",
            "max_output_tokens": 123,
        },
    )

    assert provider.calls == 1


@pytest.mark.parametrize(
    "call",
    [
        lambda provider, kwargs: runtime._call_provider_json(provider, kwargs),
        lambda provider, kwargs: sector_runtime._call_provider_result_with_meta(
            provider,
            kwargs,
        ),
    ],
)
def test_post_transport_typeerror_is_never_retried(call):
    class PostTransportTypeErrorProvider:
        provider_name = "test"
        model = "test-model"
        _handles_retry_guard = True

        def __init__(self):
            self.calls = 0

        def synthesize_json(self, **_kwargs):
            self.calls += 1
            raise TypeError("post-transport response normalization failed")

    provider = PostTransportTypeErrorProvider()
    with pytest.raises(TypeError, match="post-transport"):
        call(
            provider,
            {
                "integrity_scope": _scope(),
                "prompt": "test",
                "schema": {"type": "object"},
                "schema_name": "post_transport_typeerror",
                "max_output_tokens": 123,
            },
        )

    assert provider.calls == 1


@pytest.mark.parametrize(
    ("module", "call_name", "reserve_name"),
    [
        (runtime, "_call_provider_json", "_reserve_provider_call_cost"),
        (
            sector_runtime,
            "_call_provider_result_with_meta",
            "_reserve_llm_cost_budget",
        ),
    ],
)
def test_missing_integrity_scope_fails_closed_before_reservation_or_provider(
    monkeypatch,
    module,
    call_name,
    reserve_name,
):
    events: list[str] = []
    provider = _Provider(events)
    monkeypatch.setattr(
        module,
        reserve_name,
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("a scope-less paid call must not reserve cost")
        ),
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        getattr(module, call_name)(
            provider,
            {
                "prompt": "test",
                "schema": {"type": "object"},
                "schema_name": "scope_required",
            },
        )

    assert exc_info.value.status == "NEEDS_DATA"
    assert events == []
    assert provider.calls == []


def test_invalid_memo_input_raises_instead_of_becoming_substantive_fallback():
    events: list[str] = []
    provider = _Provider(events)
    packet = _sector_packet()
    packet.market_cap_unit = None
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="test-run",
        sector="test-sector",
        market_cap_focus="large_and_mega",
        objective="Test invalid memo suppression.",
        as_of_date="2026-07-22",
        created_at="2026-07-22T00:00:00+00:00",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        company_packets=[packet],
    )

    with pytest.raises(InvalidFinancialInputError):
        sector_runtime._enrich_sector_artifact_memo_body_impl(
            artifact,
            provider=provider,
        )

    assert provider.calls == []
    assert artifact.memo_body == {}


def test_invalid_memo_public_entry_returns_explicit_non_substantive_terminal_state():
    events: list[str] = []
    provider = _Provider(events)
    packet = _sector_packet()
    packet.market_cap_unit = None
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="test-run",
        sector="test-sector",
        market_cap_focus="large_and_mega",
        objective="Test invalid memo suppression.",
        as_of_date="2026-07-22",
        created_at="2026-07-22T00:00:00+00:00",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        company_packets=[packet],
    )

    result = sector_runtime.enrich_sector_artifact_memo_body(
        artifact,
        provider=provider,
    )

    assert result.status == "FAILED"
    assert result.degraded_states == ["NEEDS_DATA"]
    assert result.memo_body["status"] == "NEEDS_DATA"
    assert result.memo_body["candidates"] == {}
    assert result.memo_body["usage"]["calls"] == []
    assert provider.calls == []


def _invalid_financial_error() -> InvalidFinancialInputError:
    with pytest.raises(InvalidFinancialInputError) as exc_info:
        runtime.require_financial_integrity_scope(_scope(current_price=float("nan")))
    return exc_info.value


def test_candidate_quota_fallback_preserves_financial_integrity_error(monkeypatch):
    primary = SimpleNamespace(provider_name="openai", enabled=lambda: True)
    fallback = SimpleNamespace(provider_name="anthropic", enabled=lambda: True)
    integrity_error = _invalid_financial_error()

    def _call(provider, kwargs):
        if provider is primary:
            raise RuntimeError("status=429 insufficient_quota")
        raise integrity_error

    monkeypatch.setattr(runtime, "_call_provider_json", _call)
    monkeypatch.setattr(runtime, "get_anthropic_provider", lambda: fallback)

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        runtime._synthesize_provider_json(
            primary,
            prompt="test",
            schema={"type": "object"},
            schema_name="candidate_quota_fallback_integrity",
        )

    assert exc_info.value is integrity_error


def test_sector_quota_fallback_preserves_financial_integrity_error(monkeypatch):
    primary = SimpleNamespace(provider_name="openai", enabled=lambda: True)
    fallback = SimpleNamespace(provider_name="anthropic", enabled=lambda: True)
    integrity_error = _invalid_financial_error()

    def _call(provider, kwargs):
        if provider is primary:
            raise RuntimeError("status=429 insufficient_quota")
        raise integrity_error

    monkeypatch.setattr(sector_runtime, "_call_provider_json_with_meta", _call)
    monkeypatch.setattr(sector_runtime, "get_anthropic_provider", lambda: fallback)

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        sector_runtime._synthesize_provider_json_with_meta(
            primary,
            prompt="test",
            schema={"type": "object"},
            schema_name="sector_quota_fallback_integrity",
        )

    assert exc_info.value is integrity_error


@pytest.mark.parametrize(
    ("call", "settle_target"),
    [
        (
            lambda provider, kwargs: runtime._call_provider_json(provider, kwargs),
            "app.autonomous.runtime._settle_provider_call_cost",
        ),
        (
            lambda provider, kwargs: sector_runtime._call_provider_result_with_meta(
                provider, kwargs
            ),
            "app.autonomous.sector_runtime._complete_llm_cost_budget",
        ),
    ],
)
def test_successful_paid_attempt_drift_settles_before_integrity_error(
    monkeypatch,
    call,
    settle_target,
):
    events: list[str] = []
    scope = _scope()

    class MutatingSuccessProvider:
        provider_name = "test"
        model = "test-model"
        _handles_retry_guard = True

        def synthesize_json(self, **_kwargs):
            events.append("provider")
            scope.packets[0]["audit_marker"] = "changed-after-authorization"
            return SimpleNamespace(json_text="{}")

    monkeypatch.setattr(
        settle_target,
        lambda *args, **kwargs: events.append("settled"),
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        call(
            MutatingSuccessProvider(),
            {
                "integrity_scope": scope,
                "prompt": "test",
                "schema": {"type": "object"},
                "schema_name": "success_drift",
            },
        )

    assert exc_info.value.violations[0].code == "BOUND_FINANCIAL_INPUT_MUTATED"
    assert events == ["provider", "settled"]


@pytest.mark.parametrize(
    ("call", "settle_target"),
    [
        (
            lambda provider, kwargs: runtime._call_provider_json(provider, kwargs),
            "app.autonomous.runtime._settle_provider_call_cost",
        ),
        (
            lambda provider, kwargs: sector_runtime._call_provider_result_with_meta(
                provider, kwargs
            ),
            "app.autonomous.sector_runtime._settle_failed_llm_cost_budget",
        ),
    ],
)
def test_terminal_nonretry_error_drift_settles_before_integrity_error(
    monkeypatch,
    call,
    settle_target,
):
    events: list[str] = []
    scope = _scope()

    class MutatingErrorProvider:
        provider_name = "test"
        model = "test-model"
        _handles_retry_guard = True

        def synthesize_json(self, **_kwargs):
            events.append("provider")
            scope.packets[0]["audit_marker"] = "changed-on-error"
            raise ValueError("terminal nonretry provider failure")

    monkeypatch.setattr(
        settle_target,
        lambda *args, **kwargs: events.append("settled"),
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        call(
            MutatingErrorProvider(),
            {
                "integrity_scope": scope,
                "prompt": "test",
                "schema": {"type": "object"},
                "schema_name": "terminal_error_drift",
            },
        )

    assert exc_info.value.violations[0].code == "BOUND_FINANCIAL_INPUT_MUTATED"
    assert events == ["provider", "settled"]


def test_quota_error_drift_never_calls_cross_provider_fallback(monkeypatch):
    scope = _scope()
    fallback_calls: list[dict] = []

    class QuotaMutatingProvider:
        provider_name = "openai"
        model = "test-model"
        _handles_retry_guard = True

        def synthesize_json(self, **_kwargs):
            scope.packets[0]["audit_marker"] = "changed-on-quota"
            raise RuntimeError("status=429 insufficient_quota")

    class FallbackProvider:
        provider_name = "anthropic"

        def enabled(self):
            return True

        def synthesize_json(self, **kwargs):
            fallback_calls.append(kwargs)
            return SimpleNamespace(json_text="{}")

    monkeypatch.setattr(runtime, "get_anthropic_provider", FallbackProvider)

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        runtime._synthesize_provider_json(
            QuotaMutatingProvider(),
            integrity_scope=scope,
            prompt="test",
            schema={"type": "object"},
            schema_name="quota_drift_no_fallback",
        )

    assert exc_info.value.violations[0].code == "BOUND_FINANCIAL_INPUT_MUTATED"
    assert fallback_calls == []


def test_pre_persist_binding_drift_writes_no_product_files(monkeypatch, tmp_path):
    from app.autonomous.output_store import persist_autonomous_run
    from app.autonomous.run_contract import (
        AutonomousRunArtifact,
        AutonomousRunBudget,
        AutonomousRunRequest,
        ToolCallRecord,
    )
    from app.config import get_config

    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    get_config.cache_clear()

    scope = _scope()
    gate = runtime.require_financial_integrity_scope(scope)
    binding = runtime._financial_integrity_run_binding(
        scope,
        scope_fingerprint=gate.scope_fingerprint,
    )
    request = AutonomousRunRequest(
        run_id="autonomous_AAA_20260722_binding_drift",
        objective="Test immutable publication binding.",
        as_of_date="2026-07-22",
        created_at="2026-07-22T00:00:00Z",
        candidate_scope={
            "mode": "single_candidate",
            "tickers": ["AAA"],
            "financial_integrity": gate.to_dict(),
            "financial_integrity_binding": binding,
        },
        allowed_tools=["fetch_kpi_trends"],
        budget=AutonomousRunBudget(
            max_tool_calls=1,
            max_turns=1,
            max_cost_usd=1.0,
            timebox_seconds=None,
            max_candidates=1,
        ),
    )
    artifact = AutonomousRunArtifact(
        request=request,
        status="COMPLETED",
        started_at="2026-07-22T00:00:00Z",
        completed_at="2026-07-22T00:01:00Z",
        final_verdict="WATCHLIST_ONLY",
        selected_ticker=None,
        confidence="LOW",
        tool_calls=[
            ToolCallRecord(
                call_id="TC1",
                tool_name="fetch_kpi_trends",
                tool_input={},
                rationale="Test successful evidence.",
                status="OK",
            )
        ],
    )
    binding["packets"][0]["audit_marker"] = "changed-before-persist"

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        persist_autonomous_run(artifact)

    assert exc_info.value.violations[0].code == "BOUND_FINANCIAL_INPUT_MUTATED"
    assert not (data_dir / "outputs" / "runs" / "autonomous").exists()
    get_config.cache_clear()
