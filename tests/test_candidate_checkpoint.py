from __future__ import annotations

import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.autonomous import candidate_checkpoint, sector_runtime
from app.autonomous.candidate_checkpoint import (
    CANDIDATE_MEMO_FINANCIAL_INTEGRITY_CONTRACT_VERSION,
    CANDIDATE_MEMO_PROMPT_VERSION,
    CandidateMemoCheckpointCorruptError,
    CandidateMemoCheckpointError,
    candidate_checkpoint_key_sha256,
    canonical_json_sha256,
)
from app.autonomous.candidate_review import (
    provider_usage_attestation,
    provider_usage_tail_not_in_snapshot,
)
from app.autonomous.run_contract import EvidenceReference
from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
)
from app.autonomous.sweep_delta import v1_terminal_coverage_from_artifact
from tests.financial_integrity_helpers import canonicalize_financial_packet


_TICKERS = ["AAA", "BBB", "CCC"]


class _DeterministicMemoProvider:
    provider_name = "openai"
    model = "gpt-5-mini"
    reasoning_effort = "low"
    service_tier = None
    _handles_retry_guard = True

    def __init__(
        self,
        *,
        delays: dict[str, float] | None = None,
        failures: dict[str, Exception] | None = None,
    ) -> None:
        self.delays = dict(delays or {})
        self.failures = dict(failures or {})
        self.calls: list[dict] = []
        self.network_calls = 0
        self.paid_calls = 0
        self._lock = threading.Lock()

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, **kwargs):
        schema_name = str(kwargs["schema_name"])
        with self._lock:
            self.calls.append(dict(kwargs))
        if schema_name in self.failures:
            raise self.failures[schema_name]
        ticker = (
            schema_name.removeprefix("autonomous_sector_candidate_memo_").upper()
            if schema_name.startswith("autonomous_sector_candidate_memo_")
            else ""
        )
        if ticker:
            time.sleep(float(self.delays.get(ticker, 0.0)))
            payload = {
                "ticker": ticker,
                "thesis": (
                    f"{ticker} is a fully underwritten deterministic test candidate. "
                    "The central tension is valuation versus execution durability. "
                    "The packet binds an exact current price and issuer share basis. "
                    "The verdict remains evidence-bound rather than promotional."
                ),
                "key_risks": [
                    f"{ticker} execution can miss the packet case.",
                    f"{ticker} valuation can compress from the recorded basis.",
                ],
                "falsifiers": [f"{ticker} no longer clears its deterministic evidence test."],
                "open_questions": [f"What new filing evidence changes the {ticker} case?"],
            }
        elif schema_name == "autonomous_sector_cohort_comparison":
            payload = {
                "paragraphs": [
                    "The deterministic cohort has three fully sourced candidates "
                    "with explicit quote and share bases."
                ]
            }
        elif schema_name == "autonomous_sector_triage_surprises":
            payload = {
                "items": [
                    "No prompt-contract divergence appeared in the deterministic "
                    "cohort. The exact packet ordering remains binding."
                ]
            }
        else:
            raise AssertionError(f"unexpected schema: {schema_name}")
        return SimpleNamespace(
            json_text=json.dumps(payload, sort_keys=True),
            model=self.model,
            usage_input_tokens=100,
            usage_cached_input_tokens=0,
            usage_output_tokens=20,
        )

    @property
    def candidate_calls(self) -> list[str]:
        return [
            str(call["schema_name"])
            for call in self.calls
            if str(call["schema_name"]).startswith("autonomous_sector_candidate_memo_")
        ]


def _valid_packet(
    ticker: str,
    *,
    price: float,
    shares_outstanding_mm: float,
) -> SectorCompanyFinancialPacket:
    packet = SectorCompanyFinancialPacket(
        ticker=ticker,
        financial_status="COMPLETE",
        model_fit_status="FIT",
        data_quality_status="OK",
        market_cap_mm=price * shares_outstanding_mm,
        market_cap_unit="USD_millions",
        market_cap_source="stored_fixture_market_cap",
        market_cap_method="current_price_times_issuer_reported_shares",
        market_cap_effective_as_of_date="2026-07-22",
        issuer_cik=f"0000000{len(ticker)}",
        issuer_primary_ticker=ticker,
        issuer_listed_tickers=[ticker],
        security_role="PRIMARY_COMMON_EQUITY",
        is_secondary_class=False,
        is_adr=False,
        identity_source="stored_fixture_identity",
        identity_source_url="https://example.test/identity",
        identity_as_of_date="2026-07-22",
        current_price=price,
        current_price_as_of_date="2026-07-22",
        current_price_currency="USD",
        current_price_source="stored_fixture_quote",
        current_price_source_url="https://example.test/quote",
        current_price_unit="USD_per_share",
        price_basis="UNADJUSTED",
        raw_price=price,
        shares_outstanding_mm=shares_outstanding_mm,
        shares_unit="shares_millions",
        shares_as_of_date="2026-07-22",
        shares_source="stored_fixture_shares",
        issuer_quote_ratio=1.0,
        split_adjustment_factor=1.0,
    )
    # Shares/split provenance is fail-closed at the memo preflight (explicit
    # split basis, exact raw source value and unit, source reference, filed
    # date, and a no-intervening-split proof). Derive all of it from the shared
    # canonical helper rather than restating the contract here.
    return canonicalize_financial_packet(
        packet,
        as_of_date="2026-07-22",
        shares_mm=shares_outstanding_mm,
    )


def _artifact(*, run_id: str) -> AutonomousSectorFinancialRunArtifact:
    packets = [
        _valid_packet("AAA", price=10.0, shares_outstanding_mm=50.0),
        _valid_packet("BBB", price=20.0, shares_outstanding_mm=30.0),
        _valid_packet("CCC", price=30.0, shares_outstanding_mm=40.0),
    ]
    return AutonomousSectorFinancialRunArtifact(
        run_id=run_id,
        sector="stored_semiconductor_equipment",
        market_cap_focus="small_cap",
        objective="Select or reject using stored deterministic packet evidence.",
        as_of_date="2026-07-22",
        created_at="2026-07-22T00:00:00+00:00",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        candidate_selection={
            "source": "stored_fixture",
            "selected_tickers": list(_TICKERS),
            "loaded_tickers": list(_TICKERS),
            "coverage_campaign_id": "candidate-resume-campaign",
            "execution_fingerprint": "a" * 64,
        },
        company_packets=packets,
        expected_return_scenarios=[
            SectorExpectedReturnScenario(
                scenario_id=f"{packet.ticker}-base",
                ticker=packet.ticker,
                scenario_name="base",
                horizon_years=5,
                current_price=packet.current_price,
                estimated_future_value_per_share=None,
                annualized_return=None,
                current_price_unit="USD_per_share",
                quote_snapshot_id=packet.quote_snapshot_id,
                price_basis="UNADJUSTED",
                assumptions={"source": "stored_fixture"},
                evidence_ref_ids=[f"E-{packet.ticker}"],
            )
            for packet in packets
        ],
        evidence=[
            EvidenceReference(
                evidence_id=f"E-{ticker}",
                source_type="stored_packet",
                source_label="stored_fixture_evidence",
                summary=f"{ticker} has a dated, issuer-bound packet.",
                ticker=ticker,
                source_date="2026-07-22",
                source_url="https://example.test/evidence",
                confidence="HIGH",
            )
            for ticker in _TICKERS
        ],
    )


def _semantic_memo(artifact: AutonomousSectorFinancialRunArtifact) -> dict:
    return {
        "status": artifact.memo_body["status"],
        "degraded_states": artifact.memo_body["degraded_states"],
        "cohort_comparison": artifact.memo_body["cohort_comparison"],
        "triage_surprises": artifact.memo_body["triage_surprises"],
        "candidates": artifact.memo_body["candidates"],
    }


def _candidate_checkpoint_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if "_memo_context" not in path.parts)


def _context_checkpoint_file(root: Path) -> Path:
    return next(path for path in root.rglob("*.json") if "_memo_context" in path.parts)


def _append_candidate(
    artifact: AutonomousSectorFinancialRunArtifact,
    ticker: str,
    *,
    price: float,
    shares_outstanding_mm: float,
) -> None:
    packet = _valid_packet(
        ticker,
        price=price,
        shares_outstanding_mm=shares_outstanding_mm,
    )
    artifact.company_packets.append(packet)
    artifact.expected_return_scenarios.append(
        SectorExpectedReturnScenario(
            scenario_id=f"{ticker}-base",
            ticker=ticker,
            scenario_name="base",
            horizon_years=5,
            current_price=packet.current_price,
            estimated_future_value_per_share=None,
            annualized_return=None,
            current_price_unit="USD_per_share",
            quote_snapshot_id=packet.quote_snapshot_id,
            price_basis="UNADJUSTED",
            assumptions={"source": "stored_fixture"},
            evidence_ref_ids=[f"E-{ticker}"],
        )
    )
    artifact.evidence.append(
        EvidenceReference(
            evidence_id=f"E-{ticker}",
            source_type="stored_packet",
            source_label="stored_fixture_evidence",
            summary=f"{ticker} has a dated, issuer-bound packet.",
            ticker=ticker,
            source_date="2026-07-22",
            source_url="https://example.test/evidence",
            confidence="HIGH",
        )
    )
    artifact.candidate_selection["selected_tickers"].append(ticker)
    artifact.candidate_selection["loaded_tickers"].append(ticker)


def test_interrupted_resume_calls_only_total_minus_completed_and_is_idempotent(
    tmp_path,
    monkeypatch,
):
    checkpoint_root = tmp_path / "checkpoints"
    first_provider = _DeterministicMemoProvider()
    written: list[str] = []

    monkeypatch.setattr(sector_runtime, "CANDIDATE_MEMO_MAX_WORKERS", 1)

    def interrupt_after_two(**event):
        written.append(str(event["ticker"]))
        if len(written) == 2:
            raise KeyboardInterrupt("injected after two durable candidates")

    monkeypatch.setattr(
        sector_runtime,
        "_notify_candidate_checkpoint_written",
        interrupt_after_two,
    )
    with pytest.raises(
        KeyboardInterrupt,
        match="injected after two durable candidates",
    ):
        sector_runtime.enrich_sector_artifact_memo_body(
            _artifact(run_id="attempt-001"),
            provider=first_provider,
            checkpoint_root=checkpoint_root,
        )

    assert written == ["AAA", "BBB"]
    assert first_provider.candidate_calls == [
        "autonomous_sector_candidate_memo_aaa",
        "autonomous_sector_candidate_memo_bbb",
    ]
    assert len(_candidate_checkpoint_files(checkpoint_root)) == 2

    monkeypatch.setattr(
        sector_runtime,
        "_notify_candidate_checkpoint_written",
        lambda **_event: None,
    )
    resume_provider = _DeterministicMemoProvider()
    resumed = sector_runtime.enrich_sector_artifact_memo_body(
        _artifact(run_id="attempt-002"),
        provider=resume_provider,
        checkpoint_root=checkpoint_root,
    )

    assert resume_provider.candidate_calls == ["autonomous_sector_candidate_memo_ccc"]
    assert len(resume_provider.calls) == 1
    assert len(first_provider.calls) + len(resume_provider.calls) == 5
    assert list(resumed.memo_body["candidates"]) == _TICKERS
    assert resumed.memo_body["candidate_checkpoints"]["exact_hit_tickers"] == [
        "AAA",
        "BBB",
    ]
    assert resumed.memo_body["candidate_checkpoints"]["written_tickers"] == ["CCC"]
    projection = v1_terminal_coverage_from_artifact(resumed)
    assert projection["llm_candidate_review_completed"] == _TICKERS
    assert projection["llm_candidate_review_failed"] == []

    total_attestation = provider_usage_attestation(resumed.to_dict())
    incremental_attestation = provider_usage_attestation(
        resumed.to_dict(),
        include_reused=False,
    )
    assert total_attestation["physical_attempt_count"] == 5
    assert incremental_attestation["physical_attempt_count"] == 1
    assert total_attestation["cost_estimate_usd"] > (incremental_attestation["cost_estimate_usd"])
    assert resumed.memo_body["usage"]["reused_call_count"] == 4
    assert len(_candidate_checkpoint_files(checkpoint_root)) == 3

    baseline_provider = _DeterministicMemoProvider()
    baseline = sector_runtime.enrich_sector_artifact_memo_body(
        _artifact(run_id="baseline-001"),
        provider=baseline_provider,
        checkpoint_root=tmp_path / "baseline-checkpoints",
    )
    assert len(baseline_provider.candidate_calls) == 3
    baseline_attestation = provider_usage_attestation(baseline.to_dict())
    assert baseline_attestation["physical_attempt_count"] == 5
    assert total_attestation["cost_estimate_usd"] == (baseline_attestation["cost_estimate_usd"])
    assert total_attestation["cost_estimate_usd"] - (
        incremental_attestation["cost_estimate_usd"]
    ) == pytest.approx(4 * incremental_attestation["cost_estimate_usd"])
    assert _semantic_memo(resumed) == _semantic_memo(baseline)
    assert (
        v1_terminal_coverage_from_artifact(baseline)["terminal_tickers"]
        == projection["terminal_tickers"]
    )

    file_count = len(list(checkpoint_root.rglob("*.json")))
    third_provider = _DeterministicMemoProvider()
    third = sector_runtime.enrich_sector_artifact_memo_body(
        _artifact(run_id="attempt-003"),
        provider=third_provider,
        checkpoint_root=checkpoint_root,
    )
    assert third_provider.calls == []
    assert _semantic_memo(third) == _semantic_memo(resumed)
    assert len(list(checkpoint_root.rglob("*.json"))) == file_count
    assert (
        provider_usage_attestation(
            third.to_dict(),
            include_reused=False,
        )["physical_attempt_count"]
        == 0
    )
    assert (
        provider_usage_attestation(third.to_dict())["cost_estimate_usd"]
        == total_attestation["cost_estimate_usd"]
    )


def test_out_of_order_completion_persists_immediately_and_assembles_in_order(
    tmp_path,
    monkeypatch,
):
    completion_order: list[str] = []
    provider = _DeterministicMemoProvider(delays={"AAA": 0.04, "BBB": 0.02, "CCC": 0.0})
    monkeypatch.setattr(
        sector_runtime,
        "_notify_candidate_checkpoint_written",
        lambda **event: completion_order.append(str(event["ticker"])),
    )

    artifact = sector_runtime.enrich_sector_artifact_memo_body(
        _artifact(run_id="out-of-order"),
        provider=provider,
        checkpoint_root=tmp_path / "checkpoints",
    )

    assert completion_order == ["CCC", "BBB", "AAA"]
    assert list(artifact.memo_body["candidates"]) == _TICKERS
    assert [
        row["schema_name"]
        for row in artifact.provider_usage
        if str(row["schema_name"]).startswith("autonomous_sector_candidate_memo_")
    ] == [
        "autonomous_sector_candidate_memo_aaa",
        "autonomous_sector_candidate_memo_bbb",
        "autonomous_sector_candidate_memo_ccc",
    ]


def test_failed_candidate_keeps_successful_sibling_checkpoints(tmp_path):
    provider = _DeterministicMemoProvider(
        failures={
            "autonomous_sector_candidate_memo_bbb": TimeoutError("injected candidate timeout")
        }
    )
    checkpoint_root = tmp_path / "checkpoints"

    artifact = sector_runtime.enrich_sector_artifact_memo_body(
        _artifact(run_id="sibling-failure"),
        provider=provider,
        checkpoint_root=checkpoint_root,
    )

    assert artifact.memo_body["candidates"]["AAA"]["source"] == "llm"
    assert artifact.memo_body["candidates"]["BBB"]["source"] == ("deterministic_fallback")
    assert artifact.memo_body["candidates"]["CCC"]["source"] == "llm"
    assert sorted(path.parent.name for path in _candidate_checkpoint_files(checkpoint_root)) == [
        "AAA",
        "CCC",
    ]
    projection = v1_terminal_coverage_from_artifact(artifact)
    assert projection["llm_candidate_review_completed"] == ["AAA", "CCC"]
    assert projection["llm_candidate_review_failed"] == ["BBB"]


@pytest.mark.parametrize(
    "corruption",
    ["invalid_json", "partial_tmp", "unique_partial_tmp"],
)
def test_corrupt_or_partial_exact_candidate_checkpoint_fails_closed(
    tmp_path,
    corruption,
):
    checkpoint_root = tmp_path / "checkpoints"
    sector_runtime.enrich_sector_artifact_memo_body(
        _artifact(run_id="corrupt-source"),
        provider=_DeterministicMemoProvider(),
        checkpoint_root=checkpoint_root,
    )
    target = next(
        path for path in _candidate_checkpoint_files(checkpoint_root) if path.parent.name == "AAA"
    )
    if corruption == "invalid_json":
        target.write_text("{", encoding="utf-8")
    elif corruption == "partial_tmp":
        target.unlink()
        target.with_suffix(".json.tmp").write_text("{", encoding="utf-8")
    else:
        target.unlink()
        (target.parent / f".{target.name}.orphan.tmp").write_text(
            "{",
            encoding="utf-8",
        )
    provider = _DeterministicMemoProvider()

    with pytest.raises(CandidateMemoCheckpointCorruptError):
        sector_runtime.enrich_sector_artifact_memo_body(
            _artifact(run_id="corrupt-resume"),
            provider=provider,
            checkpoint_root=checkpoint_root,
        )

    assert provider.calls == []


def test_invalid_financial_input_has_zero_calls_checkpoints_and_terminal_coverage(
    tmp_path,
):
    artifact = _artifact(run_id="invalid-financial")
    artifact.company_packets[0].market_cap_unit = None
    provider = _DeterministicMemoProvider()
    checkpoint_root = tmp_path / "checkpoints"

    result = sector_runtime.enrich_sector_artifact_memo_body(
        artifact,
        provider=provider,
        checkpoint_root=checkpoint_root,
    )

    assert provider.calls == []
    assert result.status == "FAILED"
    assert result.memo_body["candidates"] == {}
    assert list(checkpoint_root.rglob("*.json")) == []
    projection = v1_terminal_coverage_from_artifact(result)
    assert projection["llm_candidate_review_completed"] == []
    assert projection["terminal_tickers"] == []


def test_shared_context_checkpoint_binds_both_exact_paid_integrity_scopes(
    tmp_path,
):
    checkpoint_root = tmp_path / "checkpoints"
    artifact = sector_runtime.enrich_sector_artifact_memo_body(
        _artifact(run_id="scope-binding"),
        provider=_DeterministicMemoProvider(),
        checkpoint_root=checkpoint_root,
    )
    envelope = json.loads(_context_checkpoint_file(checkpoint_root).read_text(encoding="utf-8"))
    expected = {}
    for scope_name, context in (
        ("cohort", "autonomous_sector_cohort_comparison"),
        ("triage", "autonomous_sector_triage_surprises"),
    ):
        scope = sector_runtime._financial_integrity_scope(
            context=context,
            as_of_date=artifact.as_of_date,
            packets=artifact.company_packets,
            scenarios=artifact.expected_return_scenarios,
        )
        expected[scope_name] = sector_runtime.require_financial_integrity_scope(scope).to_dict()

    key = envelope["checkpoint_key"]
    assert envelope["financial_integrity_scopes"] == expected
    assert key["financial_integrity_scope_fingerprints"] == {
        "cohort": expected["cohort"]["scope_fingerprint"],
        "triage": expected["triage"]["scope_fingerprint"],
    }
    assert key["quote_snapshot_ids_by_scope"] == {
        "cohort": expected["cohort"]["ticker_snapshot_ids"],
        "triage": expected["triage"]["ticker_snapshot_ids"],
    }
    assert key["financial_integrity_scopes_sha256"] == (canonical_json_sha256(expected))
    assert expected["cohort"]["context"] == ("autonomous_sector_cohort_comparison")
    assert expected["triage"]["context"] == ("autonomous_sector_triage_surprises")


@pytest.mark.parametrize("drift", ["full_evidence_provenance", "peer_scenario"])
def test_full_provenance_or_peer_scenario_drift_invalidates_candidate_reuse(
    tmp_path,
    drift,
):
    checkpoint_root = tmp_path / "checkpoints"
    sector_runtime.enrich_sector_artifact_memo_body(
        _artifact(run_id="provenance-source"),
        provider=_DeterministicMemoProvider(),
        checkpoint_root=checkpoint_root,
    )
    changed = _artifact(run_id="provenance-resume")
    if drift == "full_evidence_provenance":
        changed.evidence[0].source_url = "https://example.test/evidence-revised"
    else:
        changed.expected_return_scenarios[-1].assumptions["non_prompt_provenance"] = "revised"
    provider = _DeterministicMemoProvider()

    sector_runtime.enrich_sector_artifact_memo_body(
        changed,
        provider=provider,
        checkpoint_root=checkpoint_root,
    )

    assert len(provider.calls) == 5
    assert sorted(provider.candidate_calls) == [
        "autonomous_sector_candidate_memo_aaa",
        "autonomous_sector_candidate_memo_bbb",
        "autonomous_sector_candidate_memo_ccc",
    ]


def test_non_campaign_checkpoints_are_isolated_by_run_id(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    first = _artifact(run_id="isolated-run-001")
    first.candidate_selection.pop("coverage_campaign_id")
    first.candidate_selection.pop("execution_fingerprint")
    sector_runtime.enrich_sector_artifact_memo_body(
        first,
        provider=_DeterministicMemoProvider(),
        checkpoint_root=checkpoint_root,
    )
    second = _artifact(run_id="isolated-run-002")
    second.candidate_selection.pop("coverage_campaign_id")
    second.candidate_selection.pop("execution_fingerprint")
    provider = _DeterministicMemoProvider()

    sector_runtime.enrich_sector_artifact_memo_body(
        second,
        provider=provider,
        checkpoint_root=checkpoint_root,
    )

    assert len(provider.calls) == 5
    assert len(provider.candidate_calls) == 3


def test_provider_without_output_cap_is_never_checkpointed(tmp_path):
    class _ProviderWithoutOutputCap(_DeterministicMemoProvider):
        def synthesize_json(self, *, prompt, schema, schema_name):
            return super().synthesize_json(
                prompt=prompt,
                schema=schema,
                schema_name=schema_name,
            )

    provider = _ProviderWithoutOutputCap()
    checkpoint_root = tmp_path / "checkpoints"

    artifact = sector_runtime.enrich_sector_artifact_memo_body(
        _artifact(run_id="no-output-cap"),
        provider=provider,
        checkpoint_root=checkpoint_root,
    )

    assert artifact.memo_body["status"] == "LLM_GENERATED"
    assert len(provider.calls) == 5
    assert list(checkpoint_root.rglob("*.json")) == []
    assert artifact.memo_body["candidate_checkpoints"]["written_tickers"] == []


def test_duplicate_same_snapshot_packet_fails_before_provider(tmp_path):
    artifact = _artifact(run_id="duplicate-packet")
    artifact.company_packets.append(copy.deepcopy(artifact.company_packets[0]))
    provider = _DeterministicMemoProvider()

    with pytest.raises(
        CandidateMemoCheckpointError,
        match="exactly one packet per ticker",
    ):
        sector_runtime.enrich_sector_artifact_memo_body(
            artifact,
            provider=provider,
            checkpoint_root=tmp_path / "checkpoints",
        )

    assert provider.calls == []
    assert list((tmp_path / "checkpoints").rglob("*.json")) == []


def test_fatal_candidate_stops_new_submissions_and_preserves_running_sibling(
    tmp_path,
    monkeypatch,
):
    artifact = _artifact(run_id="fatal-stop")
    _append_candidate(
        artifact,
        "DDD",
        price=40.0,
        shares_outstanding_mm=20.0,
    )
    _append_candidate(
        artifact,
        "EEE",
        price=50.0,
        shares_outstanding_mm=10.0,
    )
    provider = _DeterministicMemoProvider(delays={"BBB": 0.05})
    sibling_started = threading.Event()
    original_synthesize = provider.synthesize_json

    def synchronized_synthesize(**kwargs):
        schema_name = str(kwargs["schema_name"])
        if schema_name == "autonomous_sector_candidate_memo_bbb":
            sibling_started.set()
        elif schema_name == "autonomous_sector_candidate_memo_aaa":
            assert sibling_started.wait(timeout=1.0)
        return original_synthesize(**kwargs)

    provider.synthesize_json = synchronized_synthesize
    checkpoint_root = tmp_path / "checkpoints"
    original_persist = sector_runtime.persist_candidate_memo_checkpoint
    monkeypatch.setattr(sector_runtime, "CANDIDATE_MEMO_MAX_WORKERS", 2)

    def fail_first_candidate(path, **kwargs):
        if kwargs["checkpoint_key"]["ticker"] == "AAA":
            raise CandidateMemoCheckpointError("injected fatal checkpoint write")
        return original_persist(path, **kwargs)

    monkeypatch.setattr(
        sector_runtime,
        "persist_candidate_memo_checkpoint",
        fail_first_candidate,
    )

    with pytest.raises(
        CandidateMemoCheckpointError,
        match="injected fatal checkpoint write",
    ):
        sector_runtime.enrich_sector_artifact_memo_body(
            artifact,
            provider=provider,
            checkpoint_root=checkpoint_root,
        )

    assert set(provider.candidate_calls) == {
        "autonomous_sector_candidate_memo_aaa",
        "autonomous_sector_candidate_memo_bbb",
    }
    assert len(provider.candidate_calls) == 2
    assert [path.parent.name for path in _candidate_checkpoint_files(checkpoint_root)] == ["BBB"]


def test_same_key_writes_are_serialized_and_immutable(tmp_path):
    source_root = tmp_path / "source"
    sector_runtime.enrich_sector_artifact_memo_body(
        _artifact(run_id="concurrent-source"),
        provider=_DeterministicMemoProvider(),
        checkpoint_root=source_root,
    )
    envelope = json.loads(_candidate_checkpoint_files(source_root)[0].read_text(encoding="utf-8"))

    def persist(path: Path, candidate_payload: dict) -> dict:
        return candidate_checkpoint.persist_candidate_memo_checkpoint(
            path,
            checkpoint_key=envelope["checkpoint_key"],
            candidate=candidate_payload,
            usage=envelope["usage"],
            provider_usage=envelope["provider_usage"],
            financial_integrity=envelope["financial_integrity"],
            producer_run_id="concurrent-test",
            created_at="2026-07-22T01:00:00+00:00",
        )

    same_target = tmp_path / "same-key.json"
    with ThreadPoolExecutor(max_workers=2) as executor:
        same_results = list(
            executor.map(
                lambda _index: persist(
                    same_target,
                    envelope["candidate"],
                ),
                range(2),
            )
        )
    assert len(same_results) == 2
    assert same_target.exists()

    different_target = tmp_path / "different-content.json"
    first_candidate = dict(envelope["candidate"])
    second_candidate = {
        **envelope["candidate"],
        "thesis": envelope["candidate"]["thesis"] + " Revised.",
    }

    def attempt(candidate_payload: dict):
        try:
            return persist(different_target, candidate_payload)
        except CandidateMemoCheckpointCorruptError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        different_results = list(executor.map(attempt, [first_candidate, second_candidate]))
    assert sum(isinstance(item, dict) for item in different_results) == 1
    assert (
        sum(isinstance(item, CandidateMemoCheckpointCorruptError) for item in different_results)
        == 1
    )


def test_failure_usage_tail_reconciles_snapshot_without_double_counting():
    snapshot_row = {
        "provider_call_id": "P1",
        "checkpoint_physical_id": "a" * 64,
        "checkpoint_key_sha256": "b" * 64,
        "reused_from_checkpoint": False,
        "status": "OK",
        "provider": "openai",
        "model": "gpt-5-mini",
        "cost_estimate_usd": 0.01,
    }
    duplicate_attached = {
        key: value
        for key, value in snapshot_row.items()
        if key
        not in {
            "provider_call_id",
            "checkpoint_physical_id",
            "checkpoint_key_sha256",
            "reused_from_checkpoint",
        }
    }
    tail_row = {
        **duplicate_attached,
        "status": "ERROR",
        "cost_estimate_usd": 0.02,
        "error": "TimeoutError: tail",
    }

    tail = provider_usage_tail_not_in_snapshot(
        {"provider_usage": [snapshot_row]},
        [duplicate_attached, tail_row],
    )

    assert tail == [tail_row]
    attestation = provider_usage_attestation(
        {
            "artifact_snapshot": {"provider_usage": [snapshot_row]},
            "provider_usage": tail,
        }
    )
    assert attestation["physical_attempt_count"] == 2
    assert attestation["cost_estimate_usd"] == 0.03


def _checkpoint_key() -> dict:
    issuer_identity = {"ticker": "AAA", "issuer_cik": "0000000001"}
    memo_context = {
        "cohort_comparison": {"paragraphs": ["exact"]},
        "triage_surprises": {"items": ["exact"]},
    }
    source_binding = {
        "ordered_tickers": ["AAA"],
        "candidate_packet_sha256": "1" * 64,
        "candidate_scenarios_sha256": "2" * 64,
        "all_scenarios_sha256": "a" * 64,
        "evidence_sha256": "3" * 64,
        "shared_context_checkpoint_key_sha256": "b" * 64,
        "shared_context_source_binding_sha256": "c" * 64,
    }
    price_share_basis = {"quote_snapshot_id": "b" * 64}
    response_config = {
        "max_output_tokens": 2200,
        "strict_paid_execution": True,
    }
    return {
        "campaign_id": "campaign-001",
        "run_identity": "campaign-001:cell-001",
        "sector": "stored-sector",
        "market_cap_focus": "small_cap",
        "pipeline_version": "v1",
        "effective_as_of_date": "2026-07-22",
        "ticker": "AAA",
        "issuer_identity": issuer_identity,
        "issuer_identity_sha256": canonical_json_sha256(issuer_identity),
        "packet_sha256": "1" * 64,
        "scenarios_sha256": "2" * 64,
        "all_scenarios_sha256": "a" * 64,
        "evidence_sha256": "3" * 64,
        "memo_context": memo_context,
        "memo_context_sha256": canonical_json_sha256(memo_context),
        "source_binding": source_binding,
        "source_binding_sha256": canonical_json_sha256(source_binding),
        "shared_context_checkpoint_key_sha256": "b" * 64,
        "shared_context_source_binding_sha256": "c" * 64,
        "financial_integrity_contract_version": (
            CANDIDATE_MEMO_FINANCIAL_INTEGRITY_CONTRACT_VERSION
        ),
        "financial_integrity_scope_fingerprint": "4" * 64,
        "quote_snapshot_ids": {"AAA": "5" * 64},
        "price_share_basis": price_share_basis,
        "price_share_basis_sha256": canonical_json_sha256(price_share_basis),
        "prompt_version": CANDIDATE_MEMO_PROMPT_VERSION,
        "prompt_sha256": "6" * 64,
        "schema_name": "autonomous_sector_candidate_memo_aaa",
        "schema_sha256": "7" * 64,
        "provider": "openai",
        "model": "gpt-5-mini",
        "reasoning_level": "low",
        "response_config": response_config,
        "response_config_sha256": canonical_json_sha256(response_config),
    }


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("packet_sha256", "8" * 64),
        ("scenarios_sha256", "d" * 64),
        ("evidence_sha256", "9" * 64),
        ("all_scenarios_sha256", "e" * 64),
        ("shared_context_checkpoint_key_sha256", "f" * 64),
        ("shared_context_source_binding_sha256", "0" * 64),
        ("quote_snapshot_ids", {"AAA": "a" * 64}),
        ("financial_integrity_scope_fingerprint", "b" * 64),
        ("effective_as_of_date", "2026-07-23"),
        ("model", "gpt-5.6"),
        ("schema_sha256", "c" * 64),
        ("prompt_sha256", "d" * 64),
        ("reasoning_level", "medium"),
    ],
)
def test_complete_key_change_invalidates_reuse(field_name, replacement):
    baseline = _checkpoint_key()
    changed = copy.deepcopy(baseline)
    changed[field_name] = replacement
    source_field = {
        "packet_sha256": "candidate_packet_sha256",
        "scenarios_sha256": "candidate_scenarios_sha256",
        "all_scenarios_sha256": "all_scenarios_sha256",
        "evidence_sha256": "evidence_sha256",
        "shared_context_checkpoint_key_sha256": ("shared_context_checkpoint_key_sha256"),
        "shared_context_source_binding_sha256": ("shared_context_source_binding_sha256"),
    }.get(field_name)
    if source_field is not None:
        changed["source_binding"][source_field] = replacement
        changed["source_binding_sha256"] = canonical_json_sha256(changed["source_binding"])

    assert candidate_checkpoint_key_sha256(changed) != (candidate_checkpoint_key_sha256(baseline))


def test_issuer_and_output_configuration_changes_invalidate_reuse():
    baseline = _checkpoint_key()
    issuer_changed = copy.deepcopy(baseline)
    issuer_changed["issuer_identity"]["issuer_cik"] = "0000000002"
    issuer_changed["issuer_identity_sha256"] = canonical_json_sha256(
        issuer_changed["issuer_identity"]
    )
    output_changed = copy.deepcopy(baseline)
    output_changed["response_config"]["max_output_tokens"] = 2300
    output_changed["response_config_sha256"] = canonical_json_sha256(
        output_changed["response_config"]
    )

    baseline_digest = candidate_checkpoint_key_sha256(baseline)
    assert candidate_checkpoint_key_sha256(issuer_changed) != baseline_digest
    assert candidate_checkpoint_key_sha256(output_changed) != baseline_digest


def test_checkpoint_contract_rejects_memo_context_hash_drift():
    key = _checkpoint_key()
    key["memo_context"]["cohort_comparison"]["paragraphs"] = ["drifted"]

    with pytest.raises(
        ValueError,
        match="memo_context_sha256 does not match bound content",
    ):
        candidate_checkpoint.candidate_checkpoint_key_sha256(key)


def test_strict_checkpoint_rejects_multiple_physical_responses(tmp_path):
    key = _checkpoint_key()
    candidate = {
        "ticker": "AAA",
        "source": "llm",
        "status": "OK",
        "thesis": "AAA has an evidence-bound thesis with an exact quote basis.",
        "key_risks": ["Execution can miss the exact packet case."],
        "falsifiers": ["The dated evidence no longer supports the thesis."],
        "open_questions": ["What filing evidence changes the case?"],
    }
    usage_row = {
        "status": "OK",
        "lane": "parent_research",
        "provider": "openai",
        "model": "gpt-5-mini",
        "schema_name": "autonomous_sector_candidate_memo_aaa",
        "input_tokens": 100,
        "cached_input_tokens": 0,
        "output_tokens": 20,
        "reserved_output_tokens": 0,
        "cost_estimate_usd": 0.001,
        "max_output_tokens_applied": True,
        "requested_max_output_tokens": 2200,
    }
    integrity = {
        "context": "autonomous_sector_candidate_memo_aaa",
        "run_as_of_date": "2026-07-22",
        "status": "PASS",
        "passed": True,
        "violations": [],
        "scope_fingerprint": "4" * 64,
        "packet_count": 1,
        "scenario_count": 0,
        "ticker_snapshot_ids": {"AAA": "5" * 64},
    }

    with pytest.raises(
        CandidateMemoCheckpointCorruptError,
        match="exactly one successful physical response",
    ):
        candidate_checkpoint.persist_candidate_memo_checkpoint(
            tmp_path / "strict.json",
            checkpoint_key=key,
            candidate=candidate,
            usage={"input_tokens": 100, "output_tokens": 20},
            provider_usage=[usage_row, dict(usage_row)],
            financial_integrity=integrity,
            producer_run_id="strict-test",
            created_at="2026-07-22T00:00:00+00:00",
        )
