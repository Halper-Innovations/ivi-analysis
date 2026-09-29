from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    InvalidFinancialInputError,
    require_financial_integrity_scope,
)
from app.llm.usage_capture import (
    attached_provider_usage_records,
    provider_usage_capture,
)
from app.research.filing_context import FilingContext, FilingDocument
from app.research.hypothesis_generator import EvidenceNeed, Hypothesis
from tests.test_financial_integrity import _valid_packet


class _CountingProvider:
    provider_name = "openai"

    def __init__(self) -> None:
        self.calls = 0

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, **_kwargs):
        self.calls += 1
        return SimpleNamespace(
            json_text="{}",
            model="fixture-model",
            usage_input_tokens=1,
            usage_output_tokens=1,
        )


def _raise_invalid_financial_scope(context: str) -> None:
    require_financial_integrity_scope(
        FinancialIntegrityScope(
            context=context,
            run_as_of_date="",
            packets=(),
        )
    )


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    from app.config import get_config

    get_config.cache_clear()
    try:
        yield get_config()
    finally:
        get_config.cache_clear()


def test_evidence_integrity_failure_is_not_converted_to_inconclusive() -> None:
    from app.research.evidence_searcher import (
        CandidateBlock,
        FilingBlock,
        _adjudicate_evidence_item,
    )

    provider = _CountingProvider()
    need = EvidenceNeed(
        need_id="need_1",
        description="customer concentration",
        importance="REQUIRED",
    )
    hypothesis = Hypothesis(
        claim="Revenue may be concentrated.",
        direction="BEARISH",
        evidence_needed=[need],
        falsification="No customer exceeds 10% of revenue.",
        priority="HIGH",
        source="TEST",
    )
    block = FilingBlock(
        block_id="risk_factors_p0",
        section="risk_factors",
        ordinal=0,
        text="One customer represented 25 percent of revenue.",
    )

    with pytest.raises(InvalidFinancialInputError):
        _adjudicate_evidence_item(
            need,
            hypothesis,
            [CandidateBlock(block=block, score=1.0, match_terms=["customer"])],
            provider,
            block_index={block.block_id: block},
            financial_integrity_scope=None,
        )

    assert provider.calls == 0


def test_analyst_notes_integrity_failure_is_not_converted_to_none(
    monkeypatch,
) -> None:
    from app.research.analyst_notes import (
        generate_analyst_notes_from_filing_context,
    )
    from app.research.evidence_searcher import FilingBlock

    provider = _CountingProvider()
    block = FilingBlock(
        block_id="mda_p0",
        section="mda",
        ordinal=0,
        text="Management discussed cash flow and customer concentration.",
    )
    monkeypatch.setattr(
        "app.research.analyst_notes.get_llm_provider",
        lambda: provider,
    )
    monkeypatch.setattr(
        "app.research.analyst_notes.extract_curated_sections_from_filing_context",
        lambda _context: (["mda"], [block]),
    )
    context = FilingContext(
        documents=[
            FilingDocument(
                ticker="TEST",
                cik="0000000001",
                accession="0000000001-26-000001",
                form_type="10-K",
                filing_date="2026-03-01",
                period_end="2025-12-31",
                role="annual",
                local_path=None,
                primary_doc_url=None,
                html="<html>fixture</html>",
            )
        ]
    )

    with pytest.raises(InvalidFinancialInputError):
        generate_analyst_notes_from_filing_context(
            context,
            scorecard={"pricing_zone_detail": {"current_price": 10.0}},
            ticker="TEST",
            financial_integrity_scope=None,
        )

    assert provider.calls == 0


def test_deep_research_propagates_upstream_integrity_failure_without_artifact(
    monkeypatch,
) -> None:
    from unittest.mock import Mock

    from app.research.deep_research import run_deep_research

    provider = _CountingProvider()
    load_scorecard = Mock()
    persist = Mock()
    monkeypatch.setattr(
        "app.research.deep_research._load_scorecard",
        load_scorecard,
    )
    monkeypatch.setattr(
        "app.research.deep_research._persist_to_db",
        persist,
    )
    monkeypatch.setattr(
        "app.llm.providers.get_llm_provider",
        lambda: provider,
    )
    monkeypatch.setattr(
        "app.ingest.facts_writer.ensure_all_facts",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "app.dossier.collector.collect_10k_docket",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "app.valuation.valuation_writer.ensure_valuation",
        lambda *_args, **_kwargs: _raise_invalid_financial_scope("deep_research_upstream"),
    )

    with pytest.raises(InvalidFinancialInputError):
        run_deep_research("TEST", as_of_date="2026-03-25")

    assert provider.calls == 0
    load_scorecard.assert_not_called()
    persist.assert_not_called()


def test_cheapness_invalid_canonical_context_has_no_call_or_cache_write(
    monkeypatch,
) -> None:
    from app.events.cheapness import build_cheapness_report

    provider = _CountingProvider()
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE cheapness_reports(id INTEGER PRIMARY KEY)")
    adapter = SimpleNamespace(client=SimpleNamespace())
    monkeypatch.setattr(
        "app.events.cheapness.cik_for_ticker",
        lambda *_args, **_kwargs: "0000000001",
    )
    monkeypatch.setattr(
        "app.events.cheapness.trailing_filing_index",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "app.events.cheapness.build_canonical_v1_financial_context",
        lambda **_kwargs: _raise_invalid_financial_scope("cheapness_context"),
    )

    with pytest.raises(InvalidFinancialInputError):
        build_cheapness_report(
            "TEST",
            as_of="2026-03-25",
            conn=conn,
            adapter=adapter,
            provider=provider,
        )

    assert provider.calls == 0
    assert conn.execute("SELECT COUNT(*) FROM cheapness_reports").fetchone()[0] == 0
    conn.close()


def test_rlm_integrity_failure_cannot_become_stop_fallback(
    monkeypatch,
    isolated_config,
    tmp_path,
) -> None:
    from app.rlm.planner import generate_plan
    from app.rlm.state import init_loop_state

    provider = _CountingProvider()
    value_gates_path = tmp_path / "value_gates.json"
    value_gates_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "ticker": "TEST",
                        "gate_status": "PASS",
                        "valuation_gap": 0.25,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    state = init_loop_state(
        run_id="rlm_gate_followup",
        sector="Software",
        as_of_date="2026-03-25",
        max_iterations=1,
        llm_budget_usd=1.0,
        sec_budget_count=1,
    )
    state.top_k_current = ["TEST"]
    state.artifacts["value_gates_path"] = str(value_gates_path)
    monkeypatch.setattr(
        "app.rlm.planner.get_llm_provider",
        lambda: provider,
    )
    monkeypatch.setattr(
        "app.rlm.planner.build_canonical_v1_financial_context",
        lambda **_kwargs: _raise_invalid_financial_scope("rlm_context"),
    )

    with pytest.raises(InvalidFinancialInputError):
        generate_plan(state=state, top_k=1)

    assert provider.calls == 0


def test_rlm_initial_iteration_without_value_gates_has_no_call_or_fallback(
    monkeypatch,
    isolated_config,
) -> None:
    from app.rlm.planner import generate_plan
    from app.rlm.state import init_loop_state

    provider = _CountingProvider()
    provider_factory_calls = 0
    fallback_calls = 0
    state = init_loop_state(
        run_id="rlm_initial_scope_missing",
        sector="Software",
        as_of_date="2026-03-25",
        max_iterations=1,
        llm_budget_usd=1.0,
        sec_budget_count=1,
    )
    state.top_k_current = ["TEST"]

    def provider_factory():
        nonlocal provider_factory_calls
        provider_factory_calls += 1
        return provider

    def forbidden_fallback(*_args, **_kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        raise AssertionError("unscoped RLM fallback must not be emitted")

    monkeypatch.setattr(
        "app.rlm.planner.get_llm_provider",
        provider_factory,
    )
    monkeypatch.setattr(
        "app.rlm.planner._fallback_planner_output",
        forbidden_fallback,
    )
    monkeypatch.setattr(
        "app.rlm.planner.build_canonical_v1_financial_context",
        lambda **_kwargs: _raise_invalid_financial_scope("rlm_initial_top_k_context"),
    )

    with pytest.raises(InvalidFinancialInputError):
        generate_plan(state=state, top_k=1)

    assert provider_factory_calls == 0
    assert provider.calls == 0
    assert fallback_calls == 0


def test_rlm_planner_failed_call_cannot_hide_mutation_and_carries_usage(
    monkeypatch,
    isolated_config,
    tmp_path,
) -> None:
    from app.rlm.planner import generate_plan
    from app.rlm.state import init_loop_state

    packet = _valid_packet()

    class _MutatingPlannerProvider:
        provider_name = "openai"

        def __init__(self) -> None:
            self.calls = 0

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, **_kwargs):
            from app.llm.providers.retry_guard import _notify_failed_attempt

            self.calls += 1
            packet["current_price"] = 101.0
            error = RuntimeError("planner provider failed after mutation")
            _notify_failed_attempt(
                provider="openai",
                schema_name="sector_rlm_planner_v0",
                attempt=1,
                exc=error,
                retryable=False,
                will_retry=False,
            )
            raise error

    value_gates_path = tmp_path / "value_gates.json"
    value_gates_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "ticker": "MEGA",
                        "gate_status": "PASS",
                        "valuation_gap": 0.25,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    state = init_loop_state(
        run_id="rlm_mutated_bound_scenario",
        sector="Software",
        as_of_date="2026-07-21",
        max_iterations=1,
        llm_budget_usd=1.0,
        sec_budget_count=1,
    )
    state.top_k_current = ["MEGA"]
    state.artifacts["value_gates_path"] = str(value_gates_path)
    provider = _MutatingPlannerProvider()
    monkeypatch.setattr("app.rlm.planner.get_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.rlm.planner.build_canonical_v1_financial_context",
        lambda **_kwargs: SimpleNamespace(
            as_of_date="2026-07-21",
            packets={"MEGA": packet},
        ),
    )

    with (
        provider_usage_capture("rlm_planner") as captured,
        pytest.raises(InvalidFinancialInputError) as exc_info,
    ):
        generate_plan(state=state, top_k=1)

    assert provider.calls == 1
    assert len(captured) == 1
    assert captured[0]["status"] == "ERROR"
    attached = attached_provider_usage_records(exc_info.value)
    assert len(attached) == 1
    assert attached[0]["cost_estimate_usd"] == captured[0]["cost_estimate_usd"]
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_filing_diff_integrity_failure_writes_no_report(
    monkeypatch,
    isolated_config,
) -> None:
    from app.diff.engine import build_filing_diff_for_ticker

    provider = _CountingProvider()
    run_id = "diff_gate_followup"
    dossier_dir = isolated_config.dossiers_dir / run_id / "TEST"
    dossier_dir.mkdir(parents=True, exist_ok=True)
    (dossier_dir / "dossier.json").write_text(
        json.dumps(
            {
                "ticker": "TEST",
                "run_id": run_id,
                "as_of_date": "2026-03-25",
                "docket": [
                    {
                        "accession": "0000000001-26-000001",
                        "filing_date": "2026-03-01",
                        "period_end": "2025-12-31",
                        "local_path": str(dossier_dir / "2025.txt"),
                    },
                    {
                        "accession": "0000000001-25-000001",
                        "filing_date": "2025-03-01",
                        "period_end": "2024-12-31",
                        "local_path": str(dossier_dir / "2024.txt"),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.diff.engine.get_llm_provider",
        lambda: provider,
    )
    monkeypatch.setattr(
        "app.diff.engine.build_canonical_v1_financial_context",
        lambda **_kwargs: _raise_invalid_financial_scope("filing_diff_context"),
    )
    expected = isolated_config.outputs_dir / "diffs" / f"TEST_{run_id}_diff.json"

    with pytest.raises(InvalidFinancialInputError):
        build_filing_diff_for_ticker(
            ticker="TEST",
            run_id=run_id,
            years_back=2,
        )

    assert provider.calls == 0
    assert not expected.exists()


def test_filing_diff_revalidates_scope_after_successful_provider_call(
    monkeypatch,
) -> None:
    from app.diff import engine

    packet = {"ticker": "TEST", "current_price": 10.0}
    state = {"postprocess_calls": 0}

    class _PacketTrackingScope:
        def __init__(self) -> None:
            self.require_calls = 0

        def require(self, **_kwargs):
            self.require_calls += 1
            if packet["current_price"] != 10.0:
                _raise_invalid_financial_scope("filing_diff_post_provider")

    class _MutatingProvider:
        def __init__(self) -> None:
            self.calls = 0

        def synthesize_json(self, **_kwargs):
            self.calls += 1
            packet["current_price"] = 11.0
            return SimpleNamespace(
                json_text=json.dumps(
                    {
                        "changes": [
                            {
                                "section": "risk_factors",
                                "fiscal_year_from": 2024,
                                "fiscal_year_to": 2025,
                                "change_type": "RISK_SIGNAL",
                                "materiality": "HIGH",
                                "summary": "A new material operating dependency was disclosed.",
                                "from_excerpt": "Prior disclosure.",
                                "to_excerpt": "Updated disclosure.",
                            }
                        ]
                    }
                )
            )

    scope = _PacketTrackingScope()
    provider = _MutatingProvider()

    def _forbidden_postprocess(*_args, **_kwargs):
        state["postprocess_calls"] += 1
        raise AssertionError("drifted provider output must not become FilingChange output")

    monkeypatch.setattr(engine, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(engine, "bind_v1_financial_scope", lambda **_kwargs: scope)
    monkeypatch.setattr(engine, "_postprocess_changes", _forbidden_postprocess)

    with pytest.raises(InvalidFinancialInputError):
        engine._classify_material_changes(
            ticker="TEST",
            section="risk_factors",
            fiscal_year_from=2024,
            fiscal_year_to=2025,
            older_text="Older material disclosure.",
            newer_text="Newer material disclosure.",
            canonical_packet=packet,
            run_as_of_date="2026-03-25",
        )

    assert provider.calls == 1
    assert scope.require_calls == 2
    assert state["postprocess_calls"] == 0


def test_filing_diff_revalidates_scope_immediately_before_return(
    monkeypatch,
) -> None:
    from app.diff import engine

    packet = {"ticker": "TEST", "current_price": 10.0}

    class _PacketTrackingScope:
        def __init__(self) -> None:
            self.require_calls = 0

        def require(self, **_kwargs):
            self.require_calls += 1
            if packet["current_price"] != 10.0:
                _raise_invalid_financial_scope("filing_diff_pre_return")

    class _Provider:
        def synthesize_json(self, **_kwargs):
            return SimpleNamespace(json_text=json.dumps({"changes": []}))

    scope = _PacketTrackingScope()

    def _mutating_postprocess(*_args, **_kwargs):
        packet["current_price"] = 11.0
        return []

    monkeypatch.setattr(engine, "get_llm_provider", lambda: _Provider())
    monkeypatch.setattr(engine, "bind_v1_financial_scope", lambda **_kwargs: scope)
    monkeypatch.setattr(engine, "_postprocess_changes", _mutating_postprocess)

    with pytest.raises(InvalidFinancialInputError):
        engine._classify_material_changes(
            ticker="TEST",
            section="risk_factors",
            fiscal_year_from=2024,
            fiscal_year_to=2025,
            older_text="Older material disclosure.",
            newer_text="Newer material disclosure.",
            canonical_packet=packet,
            run_as_of_date="2026-03-25",
        )

    assert scope.require_calls == 3


def _install_minimal_dossier_layer_fakes(monkeypatch):
    from app.dossier import runner

    filing = SimpleNamespace(
        accession="0000000001-26-000001",
        form_type="10-K",
        filing_date=date(2026, 3, 1),
        period_end="2025-12-31",
        primary_doc_url="https://www.sec.gov/example",
        filing_id=1,
    )
    stage1 = [
        SimpleNamespace(
            ticker="TEST",
            cik="0000000001",
            filing=filing,
            local_path="/tmp/fixture-filing.txt",
        )
    ]
    monkeypatch.setattr(
        runner,
        "materialize_and_parse_docket_stage1",
        lambda **_kwargs: [filing],
    )
    monkeypatch.setattr(runner, "ensure_all_facts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "read_filing_text", lambda _filing: "fixture filing text")
    monkeypatch.setattr(runner, "segment_10k_sections", lambda _text: [])
    monkeypatch.setattr(
        runner,
        "extract_annual_items",
        lambda **_kwargs: [
            {
                "ticker": "TEST",
                "year": 2025,
                "metric": "revenue",
                "value": 100.0,
                "derived_from": ["financials.revenue"],
            }
        ],
    )
    monkeypatch.setattr(runner, "build_time_series", lambda _items: {})
    monkeypatch.setattr(
        runner,
        "write_ticker_dossier",
        lambda **_kwargs: {
            "ticker": "TEST",
            "artifacts": {"dossier_md_path": "/tmp/dossier.md"},
        },
    )
    monkeypatch.setattr(runner, "append_valuation_section", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "append_synthesis_section", lambda **_kwargs: None)
    return runner, stage1


@pytest.mark.parametrize("boundary", ["valuation", "filing_diff", "synthesis"])
def test_dossier_financial_integrity_failures_bypass_layer_recovery(
    monkeypatch,
    boundary,
) -> None:
    runner, stage1 = _install_minimal_dossier_layer_fakes(monkeypatch)

    def _maybe_raise(name: str):
        if boundary == name:
            _raise_invalid_financial_scope(f"dossier_{name}")

    monkeypatch.setattr(
        runner,
        "ensure_valuation",
        lambda *_args, **_kwargs: _maybe_raise("valuation"),
    )
    monkeypatch.setattr(
        "app.diff.engine.build_filing_diff_for_ticker",
        lambda **_kwargs: _maybe_raise("filing_diff"),
    )
    monkeypatch.setattr(runner, "build_packet_for_ticker", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "run_synthesis_for_ticker",
        lambda *_args, **_kwargs: _maybe_raise("synthesis"),
    )

    with pytest.raises(InvalidFinancialInputError):
        runner._build_ticker_payload_from_stage1(
            ticker="TEST",
            as_of_date="2026-03-25",
            run_id="dossier_integrity",
            stage1=stage1,
            years_back=2,
        )


def test_dossier_peer_set_propagates_integrity_from_ticker_build(
    monkeypatch,
    isolated_config,
) -> None:
    from app.dossier.runner import run_dossier_for_peer_set

    monkeypatch.setattr(
        "app.dossier.runner._collect_stage1_for_ticker",
        lambda **_kwargs: [object()],
    )
    monkeypatch.setattr(
        "app.dossier.runner._build_ticker_payload_from_stage1",
        lambda **_kwargs: _raise_invalid_financial_scope("dossier_peer_set"),
    )

    with pytest.raises(InvalidFinancialInputError) as caught:
        run_dossier_for_peer_set(
            tickers=["TEST"],
            as_of_date="2026-03-25",
            run_id="dossier_integrity_boundary",
            workers=1,
        )
    summary_path = (
        isolated_config.dossiers_dir / "dossier_integrity_boundary" / "dossier_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED"
    assert summary["stop_reason_code"] == caught.value.status
    assert summary["ticker_results"]["TEST"]["status"] == "FAILED"
    assert summary["financial_integrity"]["status"] == caught.value.status


def test_escalation_filing_diff_recovery_propagates_integrity(
    monkeypatch,
    isolated_config,
) -> None:
    from app.universe.escalation import ACTION_DEEPEN_FILING_DIFF
    from app.universe.escalation_runner import _execute_queue_item, _runner_paths

    monkeypatch.setattr(
        "app.diff.engine.build_filing_diff_report",
        lambda **_kwargs: _raise_invalid_financial_scope("escalation_filing_diff"),
    )
    queue_item = {
        "queue_rank": 1,
        "ticker": "TEST",
        "priority_lane": "LANE_2_RESEARCH_QUEUE",
        "action_type": ACTION_DEEPEN_FILING_DIFF,
        "blocking_reason_code": "NONE",
        "source_universe_run_ids": ["source_run"],
    }

    with pytest.raises(InvalidFinancialInputError):
        _execute_queue_item(
            campaign_run_id="campaign_integrity",
            queue_item=queue_item,
            as_of_date="2026-03-25",
            results_rows=[],
            paths=_runner_paths("campaign_integrity"),
        )


def test_escalation_runner_does_not_reclassify_integrity_failure(
    monkeypatch,
    isolated_config,
) -> None:
    from app.universe import escalation_runner

    queue_item = {
        "queue_rank": 1,
        "ticker": "TEST",
        "priority_lane": "LANE_2_RESEARCH_QUEUE",
        "action_type": "DEEPEN_FILING_DIFF",
        "blocking_reason_code": "NONE",
    }
    monkeypatch.setattr(
        escalation_runner,
        "load_escalation_queue",
        lambda _campaign_run_id: [queue_item],
    )
    monkeypatch.setattr(
        escalation_runner,
        "_execute_queue_item",
        lambda **_kwargs: _raise_invalid_financial_scope("escalation_runner"),
    )

    with pytest.raises(InvalidFinancialInputError) as caught:
        escalation_runner.run_escalation_queue(
            "campaign_integrity_outer",
            resume=False,
        )
    paths = escalation_runner._runner_paths("campaign_integrity_outer")
    state = json.loads(paths["state_path"].read_text(encoding="utf-8"))
    results = json.loads(paths["results_path"].read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    assert state["stop_reason_code"] == caught.value.status
    assert state["current_item"] == {}
    assert state["financial_integrity"]["status"] == caught.value.status
    assert results["rows"][0]["status"] == "FAILED"


def test_autopilot_filing_diff_recovery_propagates_integrity(
    monkeypatch,
    isolated_config,
) -> None:
    from app.universe import autopilot

    monkeypatch.setattr(
        autopilot,
        "_batch_state_payload",
        lambda **_kwargs: {
            "status": "DONE",
            "completed_runs": [
                {
                    "status": "DONE",
                    "run_id": "source_run",
                    "tickers": ["TEST"],
                }
            ],
        },
    )
    monkeypatch.setattr(
        "app.diff.engine.build_filing_diff_report",
        lambda **_kwargs: _raise_invalid_financial_scope("autopilot_filing_diff"),
    )
    expected = autopilot._stage_expected_paths(
        "universe_integrity",
        "universe_integrity_depth_batch",
    )[autopilot.STAGE_FILING_DIFF]["filing_diff_summary_path"]

    with pytest.raises(InvalidFinancialInputError):
        autopilot._run_filing_diff(
            universe_run_id="universe_integrity",
            batch_run_id="universe_integrity_depth_batch",
            as_of_date="2026-03-25",
            filing_diff_params={},
        )

    assert not Path(expected).exists()


def test_autopilot_runner_does_not_reclassify_integrity_failure(
    monkeypatch,
    isolated_config,
) -> None:
    from app.universe import autopilot

    monkeypatch.setattr(
        autopilot,
        "_STAGE_ORDER",
        [autopilot.STAGE_FILING_DIFF],
    )
    monkeypatch.setattr(
        autopilot,
        "_run_filing_diff",
        lambda **_kwargs: _raise_invalid_financial_scope("autopilot_runner"),
    )

    with pytest.raises(InvalidFinancialInputError) as caught:
        autopilot.run_universe_autopilot(
            universe_run_id="universe_integrity_outer",
            as_of_date="2026-03-25",
            depth="full",
            resume=False,
        )
    paths = autopilot._autopilot_paths("universe_integrity_outer")
    state = json.loads(paths["state_path"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary_path"].read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    assert state["stop_reason_code"] == caught.value.status
    assert state["stages"][autopilot.STAGE_FILING_DIFF]["status"] == "FAILED"
    assert state["financial_integrity"]["status"] == caught.value.status
    assert summary["status"] == "FAILED"
    assert summary["financial_integrity"]["status"] == caught.value.status


def test_sector_tier3_filing_diff_recovery_propagates_integrity(
    monkeypatch,
    isolated_config,
) -> None:
    from app.universe import sector_universe

    monkeypatch.setattr(
        sector_universe,
        "run_dossier_for_peer_set",
        lambda **_kwargs: {"status": "DONE"},
    )
    monkeypatch.setattr(
        sector_universe,
        "build_filing_diff_for_ticker",
        lambda **_kwargs: _raise_invalid_financial_scope("sector_tier3_filing_diff"),
    )

    with pytest.raises(InvalidFinancialInputError):
        sector_universe._run_tier3(
            run_id="sector_integrity",
            tier3_tickers=["TEST"],
            as_of_date="2026-03-25",
            cfg=isolated_config,
        )
