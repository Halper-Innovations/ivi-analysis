"""Tests for app.autonomous.evidence_resolution — the Phase D evidence loop."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.autonomous.evidence_resolution as er


# ---------------------------------------------------------------------------
# Fetchable-gap classification
# ---------------------------------------------------------------------------


def test_fetchable_only_true_for_keq_class_holds() -> None:
    assert er.is_held_on_fetchable_gaps_only(
        [
            "MISSING_VALUATION",
            "MISSING_BASE_RETURN_CASE",
            "MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE",
            "MISSING_COMPANY_SPECIFIC_EVIDENCE",
        ]
    )
    assert er.is_held_on_fetchable_gaps_only(
        ["NO_READABLE_ANNUAL_FILING", "FILING_RISK_NO_FILING", "MISSING_VALUATION"]
    )


def test_fetchable_only_false_for_genuine_blockers() -> None:
    assert not er.is_held_on_fetchable_gaps_only(["MISSING_VALUATION", "GATE_BLOCK"])
    assert not er.is_held_on_fetchable_gaps_only(["SOLVENCY_CRITICAL"])
    assert not er.is_held_on_fetchable_gaps_only(["PROBABLE_PERMANENT_CAPITAL_LOSS"])
    assert not er.is_held_on_fetchable_gaps_only([])


# ---------------------------------------------------------------------------
# Pre-assembly repair hook
# ---------------------------------------------------------------------------


@pytest.fixture()
def repair_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.config import get_config

    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    cfg = get_config()
    from app.db import init_db

    init_db(cfg)
    conn = sqlite3.connect(str(cfg.db_path))
    # READY: readable filing + fresh scorecard -> no repair needed.
    filing_doc = tmp_path / "ready_10k.txt"
    filing_doc.write_text("annual report text")
    conn.execute(
        "INSERT INTO filings(cik, ticker, accession, form_type, filing_date, period_end, "
        "primary_doc_url, local_path, status, created_at, updated_at) "
        "VALUES ('1', 'REDY', 'a1', '10-K', '2026-02-01', '2025-12-31', 'u', ?, 'OK', 'x', 'x')",
        (str(filing_doc),),
    )
    conn.execute(
        "INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at) "
        "VALUES ('REDY', '2026-06-01', 'scorecard', '{}', '{}', '[]', 'x')"
    )
    # GAPS: no filing, stale scorecard -> needs repair.
    conn.execute(
        "INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at) "
        "VALUES ('GAPS', '2026-01-15', 'scorecard', '{}', '{}', '[]', 'x')"
    )
    conn.commit()
    conn.close()
    yield cfg
    get_config.cache_clear()


def test_pre_assembly_repair_targets_only_gapped_names(repair_env, monkeypatch) -> None:
    cfg = repair_env
    repaired: list[str] = []

    def fake_repair(ticker: str, **kwargs):
        repaired.append(ticker)
        return {"ticker": ticker, "actions": ["VALUATION_REFRESHED"], "errors": []}

    monkeypatch.setattr(er, "resolve_data_gaps_for_ticker", fake_repair)
    result = er.pre_assembly_data_gap_repair(
        tickers=["REDY", "GAPS"], as_of_date="2026-06-11", db_path=cfg.db_path, cfg=cfg
    )
    assert result["status"] == "APPLIED"
    assert result["examined"] == 2
    assert result["needed_repair"] == 1
    assert repaired == ["GAPS"]
    assert result["skipped_over_cap"] == []


def test_pre_assembly_repair_respects_cap(repair_env, monkeypatch) -> None:
    cfg = repair_env
    monkeypatch.setattr(
        er,
        "resolve_data_gaps_for_ticker",
        lambda t, **k: {"ticker": t, "actions": [], "errors": []},
    )
    result = er.pre_assembly_data_gap_repair(
        tickers=["GAPS", "GAP2", "GAP3"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        max_repairs=1,
    )
    assert result["needed_repair"] == 3
    assert result["repaired_count"] == 1
    assert result["skipped_over_cap"] == ["GAP2", "GAP3"]


def test_v1_pre_assembly_observation_never_invokes_repairs_before_authorization(
    repair_env, monkeypatch
) -> None:
    cfg = repair_env
    monkeypatch.setattr(
        er,
        "resolve_data_gaps_for_ticker",
        lambda *_args, **_kwargs: pytest.fail(
            "pre-authorization V1 observation must not invoke SEC or market-data repair"
        ),
    )

    result = er.pre_assembly_data_gap_repair(
        tickers=["REDY", "GAPS"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        apply_repairs=False,
    )

    assert result == {
        "status": "SKIPPED_PRE_AUTHORIZATION",
        "examined": 2,
        "needed_repair": 1,
        "repaired": [],
        "repaired_count": 0,
        "skipped_over_cap": ["GAPS"],
        "max_repairs": 25,
        "apply_repairs": False,
    }


def test_v2_repair_ceiling_is_checkpoint_batch_size_not_exclusion(
    repair_env, monkeypatch, tmp_path: Path
) -> None:
    cfg = repair_env
    calls: list[tuple[str, str]] = []

    def fake_stage(stage: str, *, ticker: str, **kwargs):
        calls.append((ticker, stage))
        return {"outcome": "OK", "actions": [f"{stage}:{ticker}"]}

    monkeypatch.setattr(er, "_execute_v2_repair_stage", fake_stage)
    result = er.pre_assembly_data_gap_repair(
        tickers=["GAPS", "GAP2", "GAP3"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        max_repairs=1,
        pipeline_version="v2",
        checkpoint_path=tmp_path / "repair_checkpoint.json",
        checkpoint_scope="energy:large_and_mega",
    )

    assert result["status"] == "COMPLETED"
    assert result["execution_status"] == "COMPLETED"
    assert result["checkpoint_batch_size"] == 1
    assert result["batch_count"] == 3
    assert result["examined"] == 3
    assert result["needed_repair"] == 3
    assert result["repaired_count"] == 3
    assert result["completed_tickers"] == ["GAPS", "GAP2", "GAP3"]
    assert result["pending_tickers"] == []
    assert result["skipped_over_cap"] == []
    assert calls == [
        ("GAPS", "MEMBERSHIP"),
        ("GAPS", "IDENTITY"),
        ("GAPS", "CAP"),
        ("GAPS", "FACTS_AVAILABILITY"),
        ("GAPS", "FILINGS"),
        ("GAPS", "PARSING"),
        ("GAPS", "PRICE"),
        ("GAPS", "VALUATION"),
        ("GAPS", "PACKET"),
        ("GAP2", "MEMBERSHIP"),
        ("GAP2", "IDENTITY"),
        ("GAP2", "CAP"),
        ("GAP2", "FACTS_AVAILABILITY"),
        ("GAP2", "FILINGS"),
        ("GAP2", "PARSING"),
        ("GAP2", "PRICE"),
        ("GAP2", "VALUATION"),
        ("GAP2", "PACKET"),
        ("GAP3", "MEMBERSHIP"),
        ("GAP3", "IDENTITY"),
        ("GAP3", "CAP"),
        ("GAP3", "FACTS_AVAILABILITY"),
        ("GAP3", "FILINGS"),
        ("GAP3", "PARSING"),
        ("GAP3", "PRICE"),
        ("GAP3", "VALUATION"),
        ("GAP3", "PACKET"),
    ]


def test_v2_repair_threads_injected_free_source_dependencies_to_every_stage(
    repair_env,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = repair_env
    provider = object()
    annual_windows = {
        "10-K": 800,
        "10-K/A": 800,
        "20-F": 800,
        "20-F/A": 800,
        "40-F": 800,
        "40-F/A": 800,
    }
    calls: list[tuple[str, object, dict[str, int] | None]] = []

    def fake_stage(
        stage: str,
        *,
        price_provider: object,
        filing_windows_days: dict[str, int] | None,
        **_kwargs,
    ) -> dict[str, str]:
        calls.append((stage, price_provider, filing_windows_days))
        return {"outcome": "OK"}

    monkeypatch.setattr(er, "_execute_v2_repair_stage", fake_stage)
    result = er.pre_assembly_data_gap_repair(
        tickers=["GAPS"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        max_repairs=1,
        pipeline_version="v2",
        checkpoint_path=tmp_path / "free_source_dependencies_checkpoint.json",
        checkpoint_scope="energy:large_and_mega",
        price_provider=provider,
        filing_windows_days=annual_windows,
    )

    assert result["execution_status"] == "COMPLETED"
    assert [stage for stage, _provider, _windows in calls] == [
        "MEMBERSHIP",
        "IDENTITY",
        "CAP",
        "FACTS_AVAILABILITY",
        "FILINGS",
        "PARSING",
        "PRICE",
        "VALUATION",
        "PACKET",
    ]
    for _stage, injected_provider, injected_windows in calls:
        assert injected_provider is provider
        assert injected_windows is annual_windows


def test_v2_packet_stage_is_explicit_readiness_not_materialization(
    repair_env, tmp_path: Path
) -> None:
    cfg = repair_env
    checkpoint_path = tmp_path / "readiness_checkpoint.json"
    result = er.pre_assembly_data_gap_repair(
        tickers=["GAPS"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        max_repairs=25,
        pipeline_version="v2",
        checkpoint_path=checkpoint_path,
        checkpoint_scope="energy:large_and_mega",
        candidate_context={
            "cap_classifications": {
                "GAPS": {
                    "market_cap_mm": 12_500.0,
                    "price_used": 100.0,
                    "cap_source": "offline_fixture",
                    "as_of_date": "2026-06-11",
                }
            }
        },
        apply_repairs=False,
    )

    checkpoint = json.loads(checkpoint_path.read_text())
    stages = checkpoint["candidates"]["GAPS"]["stages"]
    packet = stages["PACKET"]["result"]
    # Source-exhausted missing inputs are a completed repair attempt carrying
    # a terminal NEEDS_DATA decision, not an endlessly retrying execution.
    assert result["execution_status"] == "COMPLETED"
    assert result["completed_tickers"] == ["GAPS"]
    assert result["pending_tickers"] == []
    assert checkpoint["checkpoint_version"] == 4
    assert stages["IDENTITY"]["result"]["repair_policy"] == "LOCAL_EVIDENCE_ONLY"
    assert stages["CAP"]["result"]["repair_policy"] == "LOCAL_EVIDENCE_ONLY"
    assert stages["FACTS_AVAILABILITY"]["result"]["repair_policy"] == "OBSERVE_ONLY"
    assert stages["FILINGS"]["result"]["repair_policy"] == "LOCAL_EVIDENCE_ONLY"
    assert stages["PARSING"]["result"]["repair_policy"] == "LOCAL_EVIDENCE_ONLY"
    assert stages["PRICE"]["result"]["repair_policy"] == "LOCAL_EVIDENCE_ONLY"
    assert stages["VALUATION"]["result"]["repair_policy"] == "LOCAL_EVIDENCE_ONLY"
    assert packet["stage_kind"] == "PACKET_READINESS_CHECK"
    assert packet["packet_materialized"] is False
    assert packet["outcome"] == "NEEDS_DATA"
    assert packet["missing_inputs"] == [
        "IDENTITY",
        "FACTS",
        "FILING",
        "PRICE",
        "VALUATION",
    ]
    assert packet["valuation_status"] == "PROVENANCE_MISMATCH"


def test_v2_repair_recognizes_same_cik_foreign_annual_alias(
    repair_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = repair_env
    monkeypatch.setattr(
        er,
        "repair_issuer_annual_companyfacts",
        lambda *args, **kwargs: {
            "outcome": "NEEDS_DATA",
            "reason_code": "COMPANYFACTS_NOT_IN_TEST_SCOPE",
            "terminal": True,
            "network_attempted": False,
            "attempts_made": 0,
            "actions": [],
        },
    )
    foreign_path = tmp_path / "issuer-20f.htm"
    foreign_path.write_text("readable foreign annual", encoding="utf-8")
    conn = sqlite3.connect(str(cfg.db_path))
    conn.execute(
        "INSERT INTO companies(ticker, cik, name, created_at) "
        "VALUES ('ALIAS', '77', 'Alias Security', 'x')"
    )
    conn.execute(
        "INSERT INTO filings(cik, ticker, accession, form_type, filing_date, period_end, "
        "primary_doc_url, local_path, status, created_at, updated_at) "
        "VALUES ('77', 'PRIMARY', 'f1', '20-F', '2026-03-01', '2025-12-31', "
        "'https://www.sec.gov/Archives/f1.htm', ?, 'OK', 'x', 'x')",
        (str(foreign_path),),
    )
    conn.execute(
        "INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, "
        "warnings_json, created_at) VALUES "
        "('ALIAS', '2026-06-01', 'scorecard', '{}', '{}', '[]', 'x')"
    )
    conn.commit()
    conn.close()

    checkpoint_path = tmp_path / "alias_checkpoint.json"
    result = er.pre_assembly_data_gap_repair(
        tickers=["ALIAS"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        pipeline_version="v2",
        checkpoint_path=checkpoint_path,
        checkpoint_scope="healthcare:large_and_mega",
        candidate_context={
            "cap_classifications": {
                "ALIAS": {
                    "market_cap_mm": 12_500.0,
                    "price_used": 100.0,
                    "price_currency": "USD",
                    "price_as_of_date": "2026-06-11",
                    "cap_source": "terminal_exchange",
                    "as_of_date": "2026-06-11",
                }
            }
        },
        apply_repairs=True,
    )

    checkpoint = json.loads(checkpoint_path.read_text())
    stages = checkpoint["candidates"]["ALIAS"]["stages"]
    assert result["needed_repair"] == 1
    assert stages["FILINGS"]["result"]["outcome"] == "AVAILABLE"
    assert stages["PARSING"]["result"]["outcome"] == "READABLE"
    assert "FILING" not in stages["PACKET"]["result"]["missing_inputs"]


def test_v2_membership_uses_cap_scope_and_does_not_repair_out_of_scope_name(
    repair_env, tmp_path: Path
) -> None:
    cfg = repair_env
    checkpoint_path = tmp_path / "out_of_scope_checkpoint.json"
    result = er.pre_assembly_data_gap_repair(
        tickers=["LOWC"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        pipeline_version="v2",
        checkpoint_path=checkpoint_path,
        checkpoint_scope="energy:large_and_mega",
        candidate_context={
            "market_cap_focus": "large_and_mega",
            "cap_classifications": {
                "LOWC": {
                    "market_cap_mm": 5_000.0,
                    "cap_band": "mid",
                    "cap_source": "terminal_exchange",
                }
            },
        },
        apply_repairs=False,
    )

    checkpoint = json.loads(checkpoint_path.read_text())
    stages = checkpoint["candidates"]["LOWC"]["stages"]
    assert result["execution_status"] == "COMPLETED"
    assert stages["MEMBERSHIP"]["result"] == {
        "cap_band": "mid",
        "market_cap_focus": "large_and_mega",
        "market_cap_mm": 5_000.0,
        "outcome": "OUT_OF_SCOPE",
        "scope_status": "OUT_OF_SCOPE",
        "source": "cap_classification_bounds",
    }
    assert stages["IDENTITY"]["result"]["outcome"] == "NOT_APPLICABLE"
    assert stages["PACKET"]["result"]["reason"] == "CANDIDATE_OUT_OF_SCOPE"


def test_v2_cap_stage_invokes_only_authorized_terminal_search_and_persists_provenance(
    repair_env, tmp_path: Path
) -> None:
    from app.autonomous.terminal_cap_search import (
        WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE,
        build_authorized_terminal_cap_search,
        estimate_terminal_cap_search_worst_case_cost_usd,
        terminal_cap_search_ledger_fingerprint,
    )
    from app.llm.providers.disabled_provider import LLMResult

    class Provider:
        provider_name = "openai"

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def synthesize_json(self, **kwargs):
            self.calls.append(kwargs)
            source_url = "https://issuer.example/market-cap"
            return LLMResult(
                json_text=json.dumps(
                    {
                        "status": "RESOLVED",
                        "ticker": "UNKN",
                        "issuer_name": "Unknown Incorporated",
                        "issuer_cik": "0000000042",
                        "market_cap_basis": "DIRECT_ISSUER_MARKET_CAP",
                        "market_cap_mm": 15_250.0,
                        "market_cap_currency": "USD",
                        "market_cap_units": "USD_MILLIONS",
                        "as_of_date": "2026-06-10",
                        "source_name": "Issuer market data",
                        "source_url": source_url,
                        "confidence": "HIGH",
                        "detail": "Direct issuer market capitalization.",
                    }
                ),
                model="gpt-5.5",
                usage_input_tokens=100,
                usage_output_tokens=50,
                usage_cached_input_tokens=25,
                raw={
                    "id": "resp-cap-stage",
                    "output": [
                        {
                            "type": "web_search_call",
                            "id": "ws-cap-stage",
                            "action": {"sources": [{"url": source_url}]},
                        }
                    ],
                },
            )

    reserve = estimate_terminal_cap_search_worst_case_cost_usd(
        max_attempts=1,
        max_tool_calls_per_attempt=2,
    )
    provider = Provider()
    ledger_path = tmp_path / "terminal-cap-ledger.json"
    callback = build_authorized_terminal_cap_search(
        preflight={
            "artifact_type": WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE,
            "status": "AUTHORIZED",
            "run_id": "all-sector-preflight-cap-stage",
            "authorized_at": "2026-06-11T12:00:00+00:00",
            "model": "gpt-5.5",
            "request_fingerprint": "a" * 64,
            "max_cost_usd": 100.0,
            "worst_case_cost_usd": 50.0,
            "terminal_cap_search_reserved_cost_usd": reserve,
            "lane_worst_case_costs_usd": {
                "provider_preflight": 0.05,
                "parent_research": 5.0,
                "company_underwriting": 50.0 - reserve - 7.05,
                "selected_company_validation": 2.0,
                "repair_fallback": reserve,
            },
            "terminal_cap_search_max_attempts": 1,
            "terminal_cap_search_max_tool_calls_per_attempt": 2,
            "terminal_cap_search_ledger_fingerprint": (
                terminal_cap_search_ledger_fingerprint(ledger_path)
            ),
        },
        provider=provider,
        ledger_path=ledger_path,
    )
    context = {
        "cap_classifications": {
            "UNKN": {
                "market_cap_mm": None,
                "cap_source": "unknown",
                "issuer_cik": "42",
                "issuer_primary_ticker": "UNKN",
                "issuer_listed_tickers": ["UNKN"],
                "security_role": "PRIMARY",
                "is_adr": False,
                "is_secondary_class": False,
                "identity_source": "sec_registry",
            }
        }
    }

    cap = er._execute_v2_repair_stage(
        "CAP",
        ticker="UNKN",
        as_of_date="2026-06-11",
        db_path=Path(repair_env.db_path),
        cfg=repair_env,
        candidate_context=context,
        apply_repairs=True,
        terminal_cap_search=callback,
    )

    assert cap["outcome"] == "RESOLVED"
    assert cap["market_cap_mm"] == 15_250.0
    assert cap["cap_band"] == "large_cap"
    assert cap["cap_source_url"] == "https://issuer.example/market-cap"
    assert cap["cap_confidence"] == "HIGH"
    assert cap["terminal_cap_search"]["ledger_validation_status"] == "VALIDATED"
    assert cap["terminal_cap_search"]["cost_estimate_usd"] == 0.011888
    assert [row["call_type"] for row in cap["usage_records"]] == [
        "responses_model",
        "web_search_call",
    ]
    assert context["cap_classifications"]["UNKN"]["market_cap_mm"] == 15_250.0
    assert len(provider.calls) == 1


def test_v2_cap_stage_never_calls_unpreflighted_terminal_callback(
    repair_env,
) -> None:
    class Bomb:
        def __init__(self) -> None:
            self.called = False

        def __call__(self, *args, **kwargs):
            self.called = True
            raise AssertionError("unpreflighted callback must not run")

    callback = Bomb()
    cap = er._execute_v2_repair_stage(
        "CAP",
        ticker="UNKN",
        as_of_date="2026-06-11",
        db_path=Path(repair_env.db_path),
        cfg=repair_env,
        candidate_context={
            "cap_classifications": {
                "UNKN": {
                    "market_cap_mm": None,
                    "cap_source": "unknown",
                    "issuer_cik": "42",
                }
            }
        },
        apply_repairs=True,
        terminal_cap_search=callback,
    )

    assert callback.called is False
    assert cap["outcome"] == "INCOMPLETE"
    assert cap["reason_code"] == "MARKET_CAP_UNRESOLVED_TERMINAL_SEARCH_NOT_AUTHORIZED"
    assert cap["terminal"] is False
    assert cap["retryable"] is True
    assert cap["source_exhausted"] is False


def test_v2_terminal_cap_provider_failure_and_authorized_exhaustion_stay_retryable(
    repair_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.autonomous.terminal_cap_search import (
        WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE,
        build_authorized_terminal_cap_search,
        estimate_terminal_cap_search_worst_case_cost_usd,
        terminal_cap_search_ledger_fingerprint,
    )

    class FailingProvider:
        provider_name = "openai"

        def __init__(self) -> None:
            self.calls = 0

        def synthesize_json(self, **kwargs):
            self.calls += 1
            raise RuntimeError("temporary provider outage")

    reserve = estimate_terminal_cap_search_worst_case_cost_usd(
        max_attempts=1,
        max_tool_calls_per_attempt=2,
    )
    provider = FailingProvider()
    ledger_path = tmp_path / "provider-failure-ledger.json"
    callback = build_authorized_terminal_cap_search(
        preflight={
            "artifact_type": WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE,
            "status": "AUTHORIZED",
            "run_id": "provider-failure-preflight",
            "authorized_at": "2026-06-11T12:00:00+00:00",
            "model": "gpt-5.5",
            "request_fingerprint": "b" * 64,
            "max_cost_usd": 100.0,
            "worst_case_cost_usd": 50.0,
            "terminal_cap_search_reserved_cost_usd": reserve,
            "lane_worst_case_costs_usd": {
                "provider_preflight": 0.05,
                "parent_research": 5.0,
                "company_underwriting": 50.0 - reserve - 7.05,
                "selected_company_validation": 2.0,
                "repair_fallback": reserve,
            },
            "terminal_cap_search_max_attempts": 1,
            "terminal_cap_search_max_tool_calls_per_attempt": 2,
            "terminal_cap_search_ledger_fingerprint": (
                terminal_cap_search_ledger_fingerprint(ledger_path)
            ),
        },
        provider=provider,
        ledger_path=ledger_path,
    )
    candidate_context = {
        "market_cap_focus": "large_and_mega",
        "cap_classifications": {
            "UNKN": {
                "market_cap_mm": None,
                "cap_source": "unknown",
                "issuer_cik": "42",
                "issuer_primary_ticker": "UNKN",
                "issuer_listed_tickers": ["UNKN"],
                "security_role": "PRIMARY",
                "is_adr": False,
                "is_secondary_class": False,
            }
        },
    }
    monkeypatch.setattr(er, "_ticker_needs_v2_repair", lambda *args, **kwargs: True)
    real_stage = er._execute_v2_repair_stage

    def bounded_stage(stage: str, **kwargs):
        if stage in {"MEMBERSHIP", "IDENTITY", "CAP"}:
            return real_stage(stage, **kwargs)
        return {"outcome": "OK", "terminal": True, "retryable": False}

    monkeypatch.setattr(er, "_execute_v2_repair_stage", bounded_stage)
    kwargs = {
        "tickers": ["UNKN"],
        "as_of_date": "2026-06-11",
        "db_path": repair_env.db_path,
        "cfg": repair_env,
        "pipeline_version": "v2",
        "checkpoint_path": tmp_path / "provider-failure-checkpoint.json",
        "checkpoint_scope": "energy:large_and_mega",
        "candidate_context": candidate_context,
        "apply_repairs": True,
        "terminal_cap_search": callback,
    }

    first = er.pre_assembly_data_gap_repair(**kwargs)
    first_checkpoint = json.loads(Path(first["checkpoint_path"]).read_text())
    first_cap = first_checkpoint["candidates"]["UNKN"]["stages"]["CAP"]
    assert first["execution_status"] == "INCOMPLETE"
    assert first["pending_tickers"] == ["UNKN"]
    assert first_cap["status"] == "NEEDS_RETRY"
    assert first_cap["result"]["outcome"] == "INCOMPLETE"
    assert first_cap["result"]["reason_code"] == "PROVIDER_ERROR:RuntimeError"
    assert first_cap["result"]["terminal"] is False
    assert first_cap["result"]["retryable"] is True

    second = er.pre_assembly_data_gap_repair(**kwargs)
    second_checkpoint = json.loads(Path(second["checkpoint_path"]).read_text())
    second_cap = second_checkpoint["candidates"]["UNKN"]["stages"]["CAP"]
    assert provider.calls == 1
    assert second["execution_status"] == "INCOMPLETE"
    assert second["pending_tickers"] == ["UNKN"]
    assert second_cap["status"] == "NEEDS_RETRY"
    assert second_cap["result"]["outcome"] == "INCOMPLETE"
    assert second_cap["result"]["reason_code"] == "AUTHORIZED_ATTEMPT_LIMIT_EXHAUSTED"
    assert second_cap["result"]["terminal"] is False
    assert second_cap["result"]["retryable"] is True
    assert "final_verdict" not in second


def test_v2_identity_persists_issuer_security_and_ratio_evidence(
    repair_env, tmp_path: Path
) -> None:
    cfg = repair_env
    checkpoint_path = tmp_path / "identity_checkpoint.json"
    er.pre_assembly_data_gap_repair(
        tickers=["ADRX"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        pipeline_version="v2",
        checkpoint_path=checkpoint_path,
        checkpoint_scope="technology:large_and_mega",
        candidate_context={
            "market_cap_focus": "large_and_mega",
            "cap_classifications": {
                "ADRX": {
                    "market_cap_mm": 12_500.0,
                    "issuer_cik": "77",
                    "issuer_primary_ticker": "ORDX",
                    "issuer_listed_tickers": ["ORDX", "ADRX"],
                    "security_role": "ADR",
                    "is_adr": True,
                    "adr_ratio": 2.0,
                    "ratio_source_url": "https://www.sec.gov/Archives/ratio.htm",
                    "identity_source": "issuer_filing",
                    "identity_source_url": "https://www.sec.gov/Archives/identity.htm",
                    "identity_as_of_date": "2026-06-01",
                    "identity_confidence": "HIGH",
                }
            },
        },
        apply_repairs=False,
    )

    checkpoint = json.loads(checkpoint_path.read_text())
    identity = checkpoint["candidates"]["ADRX"]["stages"]["IDENTITY"]["result"]
    assert identity["outcome"] == "RESOLVED"
    assert identity["issuer_cik"] == "0000000077"
    assert identity["issuer_primary_ticker"] == "ORDX"
    assert identity["issuer_listed_tickers"] == ["ORDX", "ADRX"]
    assert identity["issuer_aliases"] == ["ADRX", "ORDX"]
    assert identity["security_role"] == "ADR"
    assert identity["is_adr"] is True
    assert identity["adr_ratio"] == 2.0
    assert identity["ratio_source_url"] == "https://www.sec.gov/Archives/ratio.htm"
    assert identity["identity_source"] == "issuer_filing"
    assert identity["identity_as_of_date"] == "2026-06-01"
    assert identity["identity_confidence"] == "HIGH"


def test_v2_facts_and_packet_use_complete_issuer_alias_coverage(
    repair_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = repair_env
    import app.valuation.valuation_writer as valuation_writer

    def forbid_ticker_reacquisition(*args, **kwargs):
        raise AssertionError("v2 valuation must not reacquire ticker-based CompanyFacts")

    monkeypatch.setattr(
        valuation_writer, "resolve_financial_facts_asof", forbid_ticker_reacquisition
    )
    for module_name in (
        "app.valuation.fcf",
        "app.valuation.graham_dodd",
        "app.valuation.intangible_economics",
        "app.valuation.net_debt",
        "app.valuation.owner_earnings",
        "app.valuation.owner_earnings_quality",
        "app.valuation.rnd_capitalization",
        "app.valuation.shares",
        "app.valuation.tech_category",
    ):
        monkeypatch.setattr(
            f"{module_name}.resolve_financial_facts_asof",
            forbid_ticker_reacquisition,
        )
    monkeypatch.setattr(
        er,
        "repair_issuer_annual_companyfacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("complete issuer-alias facts must not trigger SEC repair")
        ),
    )
    filing_path = tmp_path / "primary-10k.htm"
    filing_path.write_text("issuer annual filing", encoding="utf-8")
    conn = sqlite3.connect(str(cfg.db_path))
    conn.execute(
        "INSERT INTO companies(ticker, cik, name, created_at) "
        "VALUES ('AL2', '88', 'Alias Two', 'x')"
    )
    conn.execute(
        "INSERT INTO filings(cik, ticker, accession, form_type, filing_date, period_end, "
        "primary_doc_url, local_path, status, created_at, updated_at) "
        "VALUES ('88', 'PRI2', 'a88', '10-K', '2026-02-01', '2025-12-31', "
        "'https://www.sec.gov/Archives/a88.htm', ?, 'OK', 'x', 'x')",
        (str(filing_path),),
    )
    for year in (2023, 2024, 2025):
        for line_item, value in (
            ("revenue", 100.0),
            ("operating_income", 20.0),
            ("net_income", 15.0),
            ("cfo", 18.0),
            ("capex", 4.0),
            ("cash", 30.0),
            ("total_assets", 150.0),
            ("total_debt", 40.0),
            ("equity", 75.0),
            ("shares_outstanding", 10.0),
        ):
            conn.execute(
                "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, "
                "period_end, line_item, value, units, source_url, fetched_at, "
                "filed_date, form, accession) "
                "VALUES ('PRI2', ?, 'FY', ?, ?, ?, ?, ?, 'x', '2026-02-15', "
                "'10-K', '0000000088-26-000001')",
                (
                    year,
                    f"{year}-12-31",
                    line_item,
                    value,
                    "shares_millions" if line_item == "shares_outstanding" else "USD_millions",
                    "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000088.json",
                ),
            )
    conn.execute(
        "INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, "
        "warnings_json, created_at) VALUES "
        "('AL2', '2026-06-01', 'scorecard', '{}', '{}', '[]', 'x')"
    )
    conn.commit()
    conn.close()

    checkpoint_path = tmp_path / "alias_facts_checkpoint.json"
    er.pre_assembly_data_gap_repair(
        tickers=["AL2"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        pipeline_version="v2",
        checkpoint_path=checkpoint_path,
        checkpoint_scope="technology:large_and_mega",
        candidate_context={
            "market_cap_focus": "large_and_mega",
            "cap_classifications": {
                "AL2": {
                    "market_cap_mm": 12_500.0,
                    "price_used": 100.0,
                    "price_currency": "USD",
                    "price_as_of_date": "2026-06-11",
                    "cap_source": "terminal_exchange",
                }
            },
        },
        apply_repairs=True,
    )

    stages = json.loads(checkpoint_path.read_text())["candidates"]["AL2"]["stages"]
    facts = stages["FACTS_AVAILABILITY"]["result"]
    assert facts["outcome"] == "AVAILABLE"
    assert facts["issuer_cik"] == "88"
    assert facts["source_tickers"] == ["PRI2"]
    assert facts["missing_line_items"] == []
    assert facts["insufficient_history_line_items"] == []
    assert facts["terminal"] is True
    assert facts["retryable"] is False
    assert facts["source_attempts"] == [
        {
            "source": "LOCAL_NORMALIZED_COMPANYFACTS",
            "as_of_date": "2026-06-11",
            "outcome": "AVAILABLE",
            "issuer_cik": "88",
            "annual_rows": 30,
            "annual_years": 3,
            "latest_period_end": "2025-12-31",
            "missing_line_items": [],
            "insufficient_history_line_items": [],
        }
    ]
    assert stages["PACKET"]["result"]["outcome"] == "READY_FOR_ASSEMBLY"


@pytest.mark.parametrize(
    ("raw_facts", "expected_reason"),
    [
        (
            {"ifrs-full": {"Revenue": {"units": {"USD": []}}}},
            "IFRS_FACTS_UNSUPPORTED",
        ),
        (
            {"us-gaap": {"Revenue": {"units": {"EUR": []}}}},
            "NON_USD_FACTS_UNNORMALIZED",
        ),
    ],
)
def test_v2_partial_foreign_facts_get_precise_normalization_reason(
    repair_env,
    tmp_path: Path,
    raw_facts: dict,
    expected_reason: str,
) -> None:
    cfg = repair_env
    filing_path = tmp_path / f"foreign-{expected_reason}.htm"
    filing_path.write_text("foreign annual filing", encoding="utf-8")
    raw_path = cfg.cache_dir / "companyfacts" / "0000000099.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({"facts": raw_facts}), encoding="utf-8")
    conn = sqlite3.connect(str(cfg.db_path))
    conn.execute(
        "INSERT INTO companies(ticker, cik, name, created_at) "
        "VALUES ('FRGN', '99', 'Foreign Issuer', 'x')"
    )
    conn.execute(
        "INSERT INTO filings(cik, ticker, accession, form_type, filing_date, period_end, "
        "primary_doc_url, local_path, status, created_at, updated_at) "
        "VALUES ('99', 'FRGN', ?, '20-F', '2026-03-01', '2025-12-31', "
        "'https://www.sec.gov/Archives/foreign.htm', ?, 'downloaded', 'x', 'x')",
        (f"a-{expected_reason}", str(filing_path)),
    )
    conn.execute(
        "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, period_end, "
        "line_item, value, units, source_url, fetched_at) VALUES "
        "('FRGN', 2025, 'FY', '2025-12-31', 'revenue', 100, 'USD_millions', "
        "'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000099.json', 'x')"
    )
    conn.execute(
        "INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, "
        "warnings_json, created_at) VALUES "
        "('FRGN', '2026-06-01', 'scorecard', '{}', '{}', '[]', 'x')"
    )
    conn.commit()
    conn.close()

    checkpoint_path = tmp_path / f"foreign-{expected_reason}.json"
    result = er.pre_assembly_data_gap_repair(
        tickers=["FRGN"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        pipeline_version="v2",
        checkpoint_path=checkpoint_path,
        checkpoint_scope="technology:large_and_mega",
        candidate_context={
            "market_cap_focus": "large_and_mega",
            "cap_classifications": {
                "FRGN": {
                    "market_cap_mm": 12_500.0,
                    "price_used": 100.0,
                    "cap_source": "terminal_exchange",
                }
            },
        },
        apply_repairs=False,
    )

    stages = json.loads(checkpoint_path.read_text())["candidates"]["FRGN"]["stages"]
    facts = stages["FACTS_AVAILABILITY"]["result"]
    assert facts["reason_code"] == expected_reason
    assert facts["outcome"] == "NEEDS_DATA"
    assert facts["terminal"] is True
    assert facts["retryable"] is False
    assert stages["FACTS_AVAILABILITY"]["status"] == "COMPLETED"
    assert stages["PACKET"]["result"]["fact_coverage"]["reason_code"] == expected_reason
    assert stages["PACKET"]["result"]["outcome"] == "NEEDS_DATA"
    assert stages["PACKET"]["result"]["terminal"] is True
    assert stages["PACKET"]["status"] == "COMPLETED"
    assert result["execution_status"] == "COMPLETED"
    assert result["pending_tickers"] == []
    summary = result["candidate_states"][0]
    assert summary["facts_reason_code"] == expected_reason
    assert summary["packet_inputs"]["facts"] == facts


def test_v2_fact_provider_materialization_recomputes_issuer_coverage(
    repair_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = repair_env
    conn = sqlite3.connect(str(cfg.db_path))
    conn.execute(
        "INSERT INTO companies(ticker, cik, name, created_at) "
        "VALUES ('PROV', '42', 'Provider Repair', 'x')"
    )
    conn.commit()
    conn.close()

    def materialize(ticker: str, **kwargs):
        assert ticker == "PROV"
        assert kwargs["issuer_cik"] == "42"
        assert kwargs["as_of_date"] == "2026-06-11"
        assert kwargs["db_path"] == Path(cfg.db_path)
        assert kwargs["storage_ticker"] == "PROV"
        target = sqlite3.connect(str(kwargs["db_path"]))
        for year in (2023, 2024, 2025):
            for line_item, value in (
                ("revenue", 100.0),
                ("operating_income", 20.0),
                ("shares_outstanding", 10.0),
            ):
                target.execute(
                    "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, "
                    "period_end, line_item, value, units, source_url, fetched_at, "
                    "filed_date, form, accession) VALUES "
                    "('PROV', ?, 'FY', ?, ?, ?, ?, ?, 'x', '2026-02-15', "
                    "'10-K', '0000000042-26-000001')",
                    (
                        year,
                        f"{year}-12-31",
                        line_item,
                        value,
                        "shares_millions" if line_item == "shares_outstanding" else "USD_millions",
                        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json",
                    ),
                )
        for line_item, value in (
            ("net_income", 15.0),
            ("cfo", 18.0),
            ("capex", 4.0),
            ("cash", 30.0),
            ("total_assets", 150.0),
            ("total_debt", 40.0),
            ("equity", 75.0),
        ):
            target.execute(
                "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, "
                "period_end, line_item, value, units, source_url, fetched_at, "
                "filed_date, form, accession) VALUES "
                "('PROV', 2025, 'FY', '2025-12-31', ?, ?, 'USD_millions', ?, "
                "'x', '2026-02-15', '10-K', '0000000042-26-000001')",
                (
                    line_item,
                    value,
                    "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json",
                ),
            )
        target.commit()
        target.close()
        return {
            "outcome": "FETCHED",
            "reason_code": None,
            "terminal": False,
            "issuer_cik": "42",
            "storage_ticker": "PROV",
            "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json",
            "source_resolution": "companyfacts_fetch",
            "network_attempted": True,
            "attempts_made": 1,
            "normalized_rows": 16,
            "visible_rows": 16,
            "future_rows_rejected": 0,
            "undated_rows_rejected": 0,
            "rows_written": 16,
            "vintages_written": 16,
            "actions": ["COMPANYFACTS_FETCHED_AND_NORMALIZED"],
        }

    monkeypatch.setattr(er, "repair_issuer_annual_companyfacts", materialize)
    facts = er._execute_v2_repair_stage(
        "FACTS_AVAILABILITY",
        ticker="PROV",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context=None,
        apply_repairs=True,
    )

    assert facts["outcome"] == "AVAILABLE"
    assert facts["terminal"] is True
    assert facts["retryable"] is False
    assert facts["source_exhausted"] is False
    assert facts["coverage_before"]["reason_code"] == (
        "NORMALIZED_FACTS_REQUIRED_LINE_ITEMS_MISSING"
    )
    assert facts["coverage_after"]["missing_line_items"] == []
    assert facts["coverage_after"]["insufficient_history_line_items"] == []
    assert [attempt["source"] for attempt in facts["source_attempts"]] == [
        "LOCAL_NORMALIZED_COMPANYFACTS",
        "SEC_COMPANYFACTS",
    ]
    assert facts["source_attempts"][1]["rows_written"] == 16
    assert facts["source_attempts"][1]["network_attempted"] is True


def test_v2_transient_companyfacts_failure_remains_retryable(
    repair_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = repair_env
    conn = sqlite3.connect(str(cfg.db_path))
    conn.execute(
        "INSERT INTO companies(ticker, cik, name, created_at) "
        "VALUES ('TRNS', '42', 'Transient Facts', 'x')"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        er,
        "repair_issuer_annual_companyfacts",
        lambda *args, **kwargs: {
            "outcome": "NEEDS_DATA",
            "reason_code": "COMPANYFACTS_FETCH_5XX",
            "reason_detail": "SEC unavailable",
            "terminal": False,
            "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json",
            "source_resolution": "companyfacts_fetch",
            "network_attempted": True,
            "attempts_made": 3,
            "actions": [],
        },
    )

    facts = er._execute_v2_repair_stage(
        "FACTS_AVAILABILITY",
        ticker="TRNS",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context=None,
        apply_repairs=True,
    )

    assert facts["outcome"] == "INCOMPLETE"
    assert facts["reason_code"] == "COMPANYFACTS_FETCH_5XX"
    assert facts["terminal"] is False
    assert facts["retryable"] is True
    assert facts["source_exhausted"] is False
    assert facts["source_attempts"][1]["attempts_made"] == 3
    assert er._v2_stage_requires_retry("FACTS_AVAILABILITY", facts) is True


def test_v2_fact_coverage_rejects_rows_filed_after_fixed_asof(
    repair_env,
) -> None:
    cfg = repair_env
    conn = sqlite3.connect(str(cfg.db_path))
    conn.execute(
        "INSERT INTO companies(ticker, cik, name, created_at) "
        "VALUES ('FUTR', '42', 'Future Facts', 'x')"
    )
    conn.execute(
        "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, "
        "period_end, line_item, value, units, source_url, fetched_at, filed_date) "
        "VALUES ('FUTR', 2025, 'FY', '2025-12-31', 'revenue', 999, "
        "'USD_millions', "
        "'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json', "
        "'x', '2026-07-01')"
    )
    conn.commit()
    conn.close()

    facts = er._execute_v2_repair_stage(
        "FACTS_AVAILABILITY",
        ticker="FUTR",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context=None,
        apply_repairs=False,
    )

    assert facts["outcome"] == "NEEDS_DATA"
    assert facts["annual_rows"] == 0
    assert facts["available_line_items"] == []
    assert facts["reason_code"] == "NORMALIZED_FACTS_REQUIRED_LINE_ITEMS_MISSING"
    assert facts["source_attempts"][1] == {
        "source": "SEC_COMPANYFACTS",
        "as_of_date": "2026-06-11",
        "outcome": "NOT_ATTEMPTED",
        "reason_code": "FACTS_REPAIR_DISABLED",
    }


def test_v2_candidate_packet_inputs_preserve_full_facts_and_price_evidence(
    repair_env,
) -> None:
    cfg = repair_env
    price = er._execute_v2_repair_stage(
        "PRICE",
        ticker="NOPX",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context=None,
        apply_repairs=False,
    )
    facts = {
        "outcome": "NEEDS_DATA",
        "reason_code": "IFRS_FACTS_UNSUPPORTED",
        "terminal": True,
        "retryable": False,
        "source_attempts": [
            {
                "source": "SEC_COMPANYFACTS",
                "outcome": "NEEDS_DATA",
                "reason_code": "IFRS_FACTS_UNSUPPORTED",
            }
        ],
    }
    summary = er._candidate_stage_summary(
        {
            "ticker": "NOPX",
            "queue_status": "COMPLETED",
            "last_completed_stage": "PACKET",
            "next_stage": None,
            "stages": {
                "FACTS_AVAILABILITY": {
                    "status": "COMPLETED",
                    "result": facts,
                },
                "PRICE": {"status": "COMPLETED", "result": price},
                "PACKET": {
                    "status": "COMPLETED",
                    "result": {
                        "outcome": "NEEDS_DATA",
                        "missing_inputs": ["FACTS", "PRICE"],
                    },
                },
            },
        }
    )

    assert price["outcome"] == "NEEDS_DATA"
    assert price["reason_code"] == "OFFLINE_NO_CACHE"
    assert price["terminal"] is True
    assert price["retryable"] is False
    assert price["source_exhausted"] is True
    assert price["repair_policy"] == "LOCAL_EVIDENCE_ONLY"
    assert price["snapshot"] is None
    assert price["source_attempts"][-1]["source"] == "provider"
    assert price["source_attempts"][-1]["status"] == "SKIPPED"
    assert price["source_attempts"][-1]["reason_code"] == "OFFLINE_NO_CACHE"
    assert summary["facts_reason_code"] == "IFRS_FACTS_UNSUPPORTED"
    assert summary["price_reason_code"] == "OFFLINE_NO_CACHE"
    assert summary["packet_inputs"] == {"facts": facts, "price": price}
    assert summary["stage_states"]["FACTS_AVAILABILITY"] == {
        "status": "COMPLETED",
        "outcome": "NEEDS_DATA",
        "reason_code": "IFRS_FACTS_UNSUPPORTED",
        "terminal": True,
        "retryable": False,
        "error_type": None,
    }


def test_v2_price_stage_uses_injected_provider(
    repair_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = repair_env
    provider = object()
    captured: dict[str, object] = {}

    def fake_resolve_v2_price(ticker: str, **kwargs):
        captured["ticker"] = ticker
        captured.update(kwargs)
        return SimpleNamespace(
            to_dict=lambda: {
                "status": "RESOLVED",
                "reason_code": "PRICE_RESOLVED",
                "source_resolution": "provider",
                "snapshot": {
                    "ticker": "GAPS",
                    "as_of_date": "2026-06-10",
                    "price": 81.5,
                    "currency": "USD",
                    "source": "stooq",
                    "url": "https://stooq.com/q/d/l/",
                    "confidence": "MEDIUM",
                },
                "attempts": [
                    {
                        "source": "provider",
                        "status": "HIT",
                        "reason_code": "PRICE_RESOLVED",
                    }
                ],
                "persisted": True,
                "persistence_error": None,
            }
        )

    monkeypatch.setattr(er, "resolve_v2_price", fake_resolve_v2_price)
    result = er._execute_v2_repair_stage(
        "PRICE",
        ticker="GAPS",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context=None,
        apply_repairs=True,
        run_id="free-repair-price-test",
        price_provider=provider,
    )

    assert captured["ticker"] == "GAPS"
    assert captured["provider"] is provider
    assert captured["allow_provider"] is True
    assert captured["persist"] is True
    assert captured["run_id"] == "free-repair-price-test"
    assert result["outcome"] == "AVAILABLE"
    assert result["provider"] == "stooq"
    assert result["persisted"] is True
    assert result["actions"] == ["PRICE_SNAPSHOT_PERSISTED"]


def test_v2_price_stage_preserves_cap_quote_as_immutable_snapshot(repair_env) -> None:
    cfg = repair_env
    context = {
        "cap_classifications": {
            "GAPS": {
                "price_used": 81.5,
                "price_as_of_date": "2026-06-10",
                "price_currency": "USD",
                "price_source": "terminal_exchange",
                "price_source_url": "https://exchange.example/gaps",
                "price_confidence": "HIGH",
            }
        }
    }

    price = er._execute_v2_repair_stage(
        "PRICE",
        ticker="GAPS",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context=context,
        apply_repairs=False,
        run_id="price-stage-test",
    )

    assert price["outcome"] == "AVAILABLE"
    assert price["reason_code"] == "PRICE_RESOLVED"
    assert price["terminal"] is True
    assert price["retryable"] is False
    assert price["price"] == 81.5
    assert price["currency"] == "USD"
    assert price["provider"] == "terminal_exchange"
    assert price["source"] == "cap_stage"
    assert price["as_of_date"] == "2026-06-10"
    assert price["source_url"] == "https://exchange.example/gaps"
    assert price["confidence"] == "HIGH"
    assert price["snapshot"] == {
        "ticker": "GAPS",
        "as_of_date": "2026-06-10",
        "price": 81.5,
        "currency": "USD",
        "source": "terminal_exchange",
        "retrieved_at": "2026-06-10T00:00:00+00:00",
        "url": "https://exchange.example/gaps",
        "confidence": "HIGH",
        "raw_price": None,
        "volume": None,
    }
    assert price["source_attempts"] == [
        {
            "source": "cap_stage",
            "status": "HIT",
            "reason_code": "PRICE_RESOLVED",
            "retryable": False,
            "terminal_for_attempt": False,
            "detail": (
                "Positive USD price is on or before the requested date and within age bounds."
            ),
            "observed_price": 81.5,
            "observed_as_of_date": "2026-06-10",
            "observed_currency": "USD",
            "source_url": "https://exchange.example/gaps",
            "provider_diagnostic": None,
        }
    ]


def test_v2_insurance_fact_coverage_uses_literal_specialized_contract(
    repair_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = repair_env
    import app.insurance.routing as insurance_routing

    routing_calls: list[dict] = []

    def fake_route_security(*args, **kwargs):
        routing_calls.append({"args": args, **kwargs})
        return SimpleNamespace(
            issuer_type=insurance_routing.ISSUER_INSURANCE_UNDERWRITER,
            security_type=insurance_routing.SECURITY_COMMON,
        )

    monkeypatch.setattr(
        insurance_routing,
        "route_security",
        fake_route_security,
    )
    conn = sqlite3.connect(str(cfg.db_path))
    for line_item, value in (
        ("net_income", 10.0),
        ("equity", 100.0),
        ("shares_outstanding", 20.0),
    ):
        conn.execute(
            "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, "
            "period_end, line_item, value, units, source_url, fetched_at, "
            "filed_date, form, accession) VALUES "
            "('INSR', 2025, 'FY', '2025-12-31', ?, ?, ?, "
            "'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000099.json', "
            "'x', '2026-02-15', '10-K', '0000000099-26-000001')",
            (
                line_item,
                value,
                "shares_millions" if line_item == "shares_outstanding" else "USD_millions",
            ),
        )
    conn.commit()
    conn.close()

    insurance = er._execute_v2_repair_stage(
        "FACTS_AVAILABILITY",
        ticker="INSR",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context={"sector": "financial_services"},
        apply_repairs=False,
    )
    operating = er._execute_v2_repair_stage(
        "FACTS_AVAILABILITY",
        ticker="INSR",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context={"sector": "enterprise_software"},
        apply_repairs=False,
    )

    assert insurance["issuer_classification"] == "INSURANCE_COMMON"
    assert insurance["required_history_years_by_line_item"] == {
        "equity": 1,
        "net_income": 1,
        "shares_outstanding": 1,
    }
    assert insurance["outcome"] == "AVAILABLE"
    assert routing_calls == [
        {
            "args": ("INSR",),
            "as_of_date": "2026-06-11",
            "pipeline_version": "v2",
            "issuer_cik": None,
            "aliases": ("INSR",),
            "db_path": Path(cfg.db_path),
            "cfg": cfg,
        }
    ]
    assert operating["issuer_classification"] == "OPERATING"
    assert operating["outcome"] == "NEEDS_DATA"
    assert "cfo" in operating["missing_line_items"]


def test_v2_bank_contract_is_selected_independently_when_loans_are_missing(
    repair_env,
) -> None:
    cfg = repair_env
    conn = sqlite3.connect(str(cfg.db_path))
    conn.execute(
        "INSERT INTO companies(ticker, cik, name, created_at) "
        "VALUES ('BANKX', '55', 'Example National Bank', 'x')"
    )
    conn.execute(
        """
        INSERT INTO sec_registrants(
            cik, primary_ticker, name, sic, sic_description, sector,
            exchange_scope, operating_status, first_seen_at, last_seen_at
        ) VALUES(
            '55', 'BANKX', 'Example National Bank', '6021',
            'National Commercial Banks', 'Financials', 'US_LISTED',
            'OPERATING', 'x', 'x'
        )
        """
    )
    facts = {
        "revenue": 100.0,
        "net_income": 12.0,
        "cash": 30.0,
        "equity": 80.0,
        "total_assets": 500.0,
        "deposits": 350.0,
    }
    for line_item, value in facts.items():
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value,
                units, source_url, fetched_at, filed_date, form, accession
            ) VALUES(
                'BANKX', 2025, 'FY', '2025-12-31', ?, ?, 'USD_millions',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000055.json',
                'x', '2026-02-15', '10-K', '0000000055-26-000001'
            )
            """,
            (line_item, value),
        )
    for year in (2024, 2025):
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value,
                units, source_url, fetched_at, filed_date, form, accession
            ) VALUES(
                'BANKX', ?, 'FY', ?, 'shares_outstanding', 20.0,
                'shares_millions',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000055.json',
                'x', '2026-02-15', '10-K', '0000000055-26-000001'
            )
            """,
            (year, f"{year}-12-31"),
        )
    conn.commit()
    conn.close()

    result = er._execute_v2_repair_stage(
        "FACTS_AVAILABILITY",
        ticker="BANKX",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context={"sector": "capital_markets"},
        apply_repairs=False,
    )

    assert result["issuer_classification"] == "FINANCIAL"
    assert result["missing_line_items"] == ["loans"]
    assert "cfo" not in result["missing_line_items"]
    assert "capex" not in result["missing_line_items"]


def test_v2_stale_scorecard_remains_unready_when_refresh_does_not_advance(
    repair_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = repair_env
    import app.valuation.valuation_writer as valuation_writer

    monkeypatch.setattr(valuation_writer, "ensure_valuation", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        er,
        "_issuer_normalized_fact_coverage",
        lambda *args, **kwargs: {"outcome": "AVAILABLE"},
    )
    context = {
        "cap_classifications": {
            "GAPS": {
                "market_cap_mm": 12_500.0,
                "price_used": 100.0,
                "price_currency": "USD",
                "cap_source": "terminal_exchange",
                "as_of_date": "2026-06-11",
            }
        }
    }
    valuation = er._execute_v2_repair_stage(
        "VALUATION",
        ticker="GAPS",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context=context,
        apply_repairs=True,
    )
    packet = er._execute_v2_repair_stage(
        "PACKET",
        ticker="GAPS",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context=context,
        apply_repairs=False,
    )

    assert valuation["outcome"] == "NEEDS_DATA"
    assert valuation["valuation_status"] == "PROVENANCE_MISMATCH"
    assert valuation["reason_code"] == "VALUATION_PROVENANCE_MISMATCH_AFTER_REFRESH"
    assert valuation["terminal"] is True
    assert valuation["retryable"] is False
    assert valuation["scorecard_asof_after"] is None
    assert valuation["raw_scorecard_asof_after"] == "2026-01-15"
    assert valuation["is_stale_after"] is True
    assert valuation["refresh_attempted"] is True
    assert valuation["actions"] == ["VALUATION_REFRESH_ATTEMPTED"]
    assert packet["valuation_status"] == "PROVENANCE_MISMATCH"
    assert "VALUATION" in packet["missing_inputs"]


def test_v2_valuation_refresh_uses_injected_config_db_and_price(
    repair_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = repair_env
    import app.valuation.valuation_writer as valuation_writer

    monkeypatch.setattr(
        er,
        "_issuer_normalized_fact_coverage",
        lambda *args, **kwargs: {"outcome": "AVAILABLE"},
    )
    captured: dict[str, object] = {}

    def routed_refresh(ticker: str, as_of_date: str, **kwargs) -> list[dict]:
        from app.valuation.lineage import valuation_source_record

        captured.update({"ticker": ticker, "as_of_date": as_of_date, **kwargs})
        conn = sqlite3.connect(str(kwargs["db_path"]))
        conn.execute(
            "INSERT INTO valuations(ticker, as_of_date, method, inputs_json, "
            "outputs_json, warnings_json, created_at) VALUES "
            "('GAPS', '2026-06-11', 'scorecard', '{}', '{}', '[]', 'x')"
        )
        conn.commit()
        conn.close()
        record = valuation_source_record(
            {
                "ticker": ticker,
                "as_of_date": as_of_date,
                "method": "scorecard",
                "inputs_json": "{}",
                "outputs_json": "{}",
                "warnings_json": "[]",
                "created_at": "x",
                "valuation_writer_version": "test",
                "quality_gate_verdict": "PROCEED",
                "confidence_class": "HIGH",
                "gate_reason_codes": "[]",
                "valuation_headwinds": "[]",
                "valuation_supports": "[]",
                "source_run_id": kwargs["run_id"],
            }
        )
        assert record is not None
        return [record]

    monkeypatch.setattr(valuation_writer, "ensure_valuation", routed_refresh)
    provenance_reads = 0

    def routed_provenance(*args, **kwargs):
        nonlocal provenance_reads
        provenance_reads += 1
        if provenance_reads == 1:
            return {
                "raw_asof": "2026-01-15",
                "validated_asof": None,
                "mismatch_reasons": ["VALUATION_PIPELINE_VERSION_MISMATCH"],
                "inputs": {},
            }
        return {
            "raw_asof": "2026-06-11",
            "validated_asof": "2026-06-11",
            "mismatch_reasons": [],
            "inputs": {"pipeline_version": "v2"},
        }

    monkeypatch.setattr(er, "_v2_scorecard_provenance_state", routed_provenance)
    valuation = er._execute_v2_repair_stage(
        "VALUATION",
        ticker="GAPS",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context={
            "cap_classifications": {
                "GAPS": {
                    "market_cap_mm": 12_500.0,
                    "price_used": 101.25,
                    "price_as_of_date": "2026-06-10",
                    "price_currency": "USD",
                    "cap_source": "terminal_exchange",
                    "issuer_cik": "42",
                    "issuer_primary_ticker": "PRIMARY",
                    "issuer_listed_tickers": ["PRIMARY", "GAPS"],
                    "identity_source": "issuer_filing",
                    "identity_source_url": "https://www.sec.gov/Archives/identity.htm",
                    "identity_as_of_date": "2026-06-01",
                    "identity_confidence": "HIGH",
                }
            }
        },
        apply_repairs=True,
        run_id="valuation-route-test",
    )

    assert captured["ticker"] == "GAPS"
    assert captured["as_of_date"] == "2026-06-11"
    assert captured["cfg"] is cfg
    assert captured["db_path"] == Path(cfg.db_path)
    assert captured["run_id"] == "valuation-route-test"
    assert captured["price_override"] == 101.25
    assert captured["force_refresh"] is True
    assert captured["raise_on_error"] is True
    assert captured["issuer_cik"] == "0000000042"
    assert captured["issuer_aliases"] == ("GAPS", "PRIMARY")
    assert captured["require_filed_asof"] is True
    assert captured["price_provenance"]["ticker"] == "GAPS"
    assert captured["price_provenance"]["price"] == 101.25
    assert captured["price_provenance"]["currency"] == "USD"
    assert captured["price_provenance"]["as_of_date"] == "2026-06-10"
    assert captured["price_provenance"]["source"] == "terminal_exchange"
    assert captured["price_provenance"]["source_resolution"] == "cap_stage"
    assert valuation["outcome"] == "FRESH"
    assert valuation["valuation_status"] == "FRESH"
    assert valuation["actions"] == [
        "VALUATION_REFRESH_ATTEMPTED",
        "VALUATION_REFRESHED",
    ]
    assert valuation["valuation_source_records"][0]["row"]["source_run_id"] == (
        "valuation-route-test"
    )


def test_v2_valuation_refresh_exception_remains_retryable(
    repair_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = repair_env
    import app.valuation.valuation_writer as valuation_writer

    monkeypatch.setattr(
        er,
        "_issuer_normalized_fact_coverage",
        lambda *args, **kwargs: {"outcome": "AVAILABLE"},
    )
    monkeypatch.setattr(
        valuation_writer,
        "ensure_valuation",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("temporary lock")),
    )
    valuation = er._execute_v2_repair_stage(
        "VALUATION",
        ticker="GAPS",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context={
            "cap_classifications": {
                "GAPS": {
                    "market_cap_mm": 12_500.0,
                    "price_used": 101.25,
                    "price_as_of_date": "2026-06-10",
                    "price_currency": "USD",
                    "cap_source": "terminal_exchange",
                }
            }
        },
        apply_repairs=True,
    )

    assert valuation["outcome"] == "INCOMPLETE"
    assert valuation["reason_code"] == "VALUATION_REFRESH_EXCEPTION"
    assert valuation["terminal"] is False
    assert valuation["retryable"] is True
    assert valuation["error_type"] == "RuntimeError"
    assert valuation["error_detail"] == "temporary lock"
    assert valuation["scorecard_asof_after"] is None
    assert valuation["raw_scorecard_asof_after"] == "2026-01-15"


def test_v2_future_scorecard_is_not_visible_at_run_cutoff(repair_env) -> None:
    cfg = repair_env
    conn = sqlite3.connect(str(cfg.db_path))
    conn.execute(
        "INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, "
        "warnings_json, created_at) VALUES "
        "('FUTR', '2026-07-01', 'scorecard', '{}', '{}', '[]', 'x')"
    )
    conn.commit()
    conn.close()
    context = {
        "cap_classifications": {
            "FUTR": {
                "market_cap_mm": 12_500.0,
                "price_used": 100.0,
                "cap_source": "terminal_exchange",
            }
        }
    }

    valuation = er._execute_v2_repair_stage(
        "VALUATION",
        ticker="FUTR",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context=context,
        apply_repairs=False,
    )
    packet = er._execute_v2_repair_stage(
        "PACKET",
        ticker="FUTR",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context=context,
        apply_repairs=False,
    )

    assert valuation["outcome"] == "NEEDS_DATA"
    assert valuation["valuation_status"] == "MISSING"
    assert valuation["terminal"] is True
    assert valuation["retryable"] is False
    assert valuation["reason_code"] == "VALUATION_INPUTS_MISSING"
    assert valuation["missing_inputs"] == ["FACTS", "PRICE"]
    assert valuation["refresh_attempted"] is False
    assert valuation["scorecard_asof_before"] is None
    assert valuation["scorecard_asof_after"] is None
    assert packet["valuation_status"] == "MISSING"
    assert packet["scorecard_asof"] is None
    assert "VALUATION" in packet["missing_inputs"]


def test_v2_attempt_fingerprint_separates_diagnostics_and_prevents_stale_reuse(
    repair_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = repair_env
    checkpoint_path = tmp_path / "attempt_checkpoint.json"
    calls: list[tuple[str, str]] = []

    def fake_stage(stage: str, *, ticker: str, **kwargs):
        calls.append((ticker, stage))
        return {"outcome": "OK"}

    monkeypatch.setattr(er, "_execute_v2_repair_stage", fake_stage)
    common = {
        "tickers": ["GAPS"],
        "as_of_date": "2026-06-11",
        "db_path": cfg.db_path,
        "cfg": cfg,
        "pipeline_version": "v2",
        "checkpoint_path": checkpoint_path,
        "checkpoint_scope": "energy:large_and_mega",
        "candidate_context": {"cap_classifications": {"GAPS": {"market_cap_mm": 12_500.0}}},
    }
    first = er.pre_assembly_data_gap_repair(
        **common,
        run_id="attempt-one",
        evidence_revision="filings-r1",
    )
    first_diagnostic = Path(first["diagnostic_path"])
    calls.clear()
    second = er.pre_assembly_data_gap_repair(
        **common,
        run_id="attempt-two",
        evidence_revision="filings-r2",
    )

    assert first["input_fingerprint"] != second["input_fingerprint"]
    assert first["attempt_id"] != second["attempt_id"]
    assert first_diagnostic.is_file()
    assert Path(second["diagnostic_path"]).is_file()
    assert first["diagnostic_path"] != second["diagnostic_path"]
    assert second["resumed_from_checkpoint"] is False
    assert calls == [("GAPS", stage) for stage in er.V2_REPAIR_STAGE_SEQUENCE]
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["prior_attempt"]["attempt_id"] == first["attempt_id"]

    calls.clear()
    changed_cap_context = {"cap_classifications": {"GAPS": {"market_cap_mm": 13_250.0}}}
    third = er.pre_assembly_data_gap_repair(
        **{**common, "candidate_context": changed_cap_context},
        run_id="attempt-two",
        evidence_revision="filings-r2",
    )
    assert third["input_fingerprint"] != second["input_fingerprint"]
    assert third["resumed_from_checkpoint"] is False
    assert calls == [("GAPS", stage) for stage in er.V2_REPAIR_STAGE_SEQUENCE]


def test_v2_default_checkpoint_resumes_across_ephemeral_runtime_ids(
    repair_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = repair_env
    context = {
        "sector": "energy",
        "market_cap_focus": "large_and_mega",
        "cap_classifications": {"GAPS": {"market_cap_mm": 12_500.0}},
    }
    calls: list[tuple[str, str]] = []

    def interrupt_once(stage: str, *, ticker: str, **kwargs):
        calls.append((ticker, stage))
        if stage == "FILINGS":
            raise KeyboardInterrupt("test interruption")
        return {"outcome": "OK"}

    monkeypatch.setattr(er, "_execute_v2_repair_stage", interrupt_once)
    with pytest.raises(KeyboardInterrupt, match="test interruption"):
        er.pre_assembly_data_gap_repair(
            tickers=["GAPS"],
            as_of_date="2026-06-11",
            db_path=cfg.db_path,
            cfg=cfg,
            pipeline_version="v2",
            checkpoint_scope="energy:large_and_mega",
            candidate_context=context,
            run_id="runtime-first",
        )

    stable_path = er.v2_repair_checkpoint_path(
        tickers=["GAPS"],
        as_of_date="2026-06-11",
        checkpoint_scope="energy:large_and_mega",
        candidate_context=context,
        cfg=cfg,
    )
    interrupted = json.loads(stable_path.read_text())
    first_attempt_id = interrupted["attempt_id"]
    first_diagnostic_path = interrupted["diagnostic_path"]

    resumed_calls: list[tuple[str, str]] = []

    def finish(stage: str, *, ticker: str, **kwargs):
        resumed_calls.append((ticker, stage))
        return {"outcome": "OK"}

    monkeypatch.setattr(er, "_execute_v2_repair_stage", finish)
    resumed = er.pre_assembly_data_gap_repair(
        tickers=["GAPS"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        pipeline_version="v2",
        checkpoint_scope="energy:large_and_mega",
        candidate_context=context,
        run_id="runtime-second",
    )

    assert resumed["checkpoint_path"] == str(stable_path)
    assert resumed["resumed_from_checkpoint"] is True
    assert resumed["execution_status"] == "COMPLETED"
    assert resumed["attempt_id"] != first_attempt_id
    assert resumed["diagnostic_path"] != first_diagnostic_path
    assert resumed["attempt_history"][-1]["attempt_id"] == first_attempt_id
    assert resumed_calls == [
        ("GAPS", "FILINGS"),
        ("GAPS", "PARSING"),
        ("GAPS", "PRICE"),
        ("GAPS", "VALUATION"),
        ("GAPS", "PACKET"),
    ]


def test_v2_unresolved_stage_remains_retryable_and_refreshes_downstream(
    repair_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = repair_env
    checkpoint_path = tmp_path / "unresolved_checkpoint.json"
    first_calls: list[tuple[str, str]] = []

    def unresolved_cap(stage: str, *, ticker: str, **kwargs):
        first_calls.append((ticker, stage))
        if stage == "CAP":
            return {"outcome": "UNRESOLVED"}
        return {"outcome": "OK"}

    monkeypatch.setattr(er, "_execute_v2_repair_stage", unresolved_cap)
    first = er.pre_assembly_data_gap_repair(
        tickers=["GAPS"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        pipeline_version="v2",
        checkpoint_path=checkpoint_path,
        checkpoint_scope="energy:large_and_mega",
    )

    assert first["execution_status"] == "INCOMPLETE"
    assert first["pending_tickers"] == ["GAPS"]
    first_state = first["candidate_states"][0]
    assert first_state["last_completed_stage"] == "IDENTITY"
    assert first_state["next_stage"] == "CAP"
    assert first_state["stage_states"]["CAP"]["status"] == "NEEDS_RETRY"

    retry_calls: list[tuple[str, str]] = []

    def resolved(stage: str, *, ticker: str, **kwargs):
        retry_calls.append((ticker, stage))
        return {"outcome": "OK"}

    monkeypatch.setattr(er, "_execute_v2_repair_stage", resolved)
    second = er.pre_assembly_data_gap_repair(
        tickers=["GAPS"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        pipeline_version="v2",
        checkpoint_path=checkpoint_path,
        checkpoint_scope="energy:large_and_mega",
    )

    assert second["resumed_from_checkpoint"] is True
    assert second["execution_status"] == "COMPLETED"
    assert retry_calls == [("GAPS", stage) for stage in er.V2_REPAIR_STAGE_SEQUENCE[2:]]


def test_v2_resume_invalidates_completed_stage_after_filing_evidence_changes(
    repair_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = repair_env
    checkpoint_path = tmp_path / "evidence_change_checkpoint.json"
    calls: list[tuple[str, str]] = []

    def fake_stage(stage: str, *, ticker: str, **kwargs):
        calls.append((ticker, stage))
        return {"outcome": "OK"}

    monkeypatch.setattr(er, "_execute_v2_repair_stage", fake_stage)
    kwargs = {
        "tickers": ["NEWF"],
        "as_of_date": "2026-06-11",
        "db_path": cfg.db_path,
        "cfg": cfg,
        "pipeline_version": "v2",
        "checkpoint_path": checkpoint_path,
        "checkpoint_scope": "energy:large_and_mega",
        "candidate_context": {"cap_classifications": {"NEWF": {"market_cap_mm": 12_500.0}}},
        "run_id": "same-attempt",
        "evidence_revision": "same-revision",
    }
    er.pre_assembly_data_gap_repair(**kwargs)
    calls.clear()

    filing_path = tmp_path / "newf-10k.htm"
    filing_path.write_text("new filing evidence", encoding="utf-8")
    conn = sqlite3.connect(str(cfg.db_path))
    conn.execute(
        "INSERT INTO filings(cik, ticker, accession, form_type, filing_date, period_end, "
        "primary_doc_url, local_path, status, created_at, updated_at) "
        "VALUES ('123', 'NEWF', 'newf-a1', '10-K', '2026-03-01', '2025-12-31', "
        "'https://www.sec.gov/Archives/newf.htm', ?, 'OK', 'x', 'changed')",
        (str(filing_path),),
    )
    conn.commit()
    conn.close()

    resumed = er.pre_assembly_data_gap_repair(**kwargs)
    checkpoint = json.loads(checkpoint_path.read_text())

    assert resumed["resumed_from_checkpoint"] is True
    assert calls[0] == ("NEWF", "IDENTITY")
    assert calls[-1] == ("NEWF", "PACKET")
    assert ("NEWF", "MEMBERSHIP") not in calls
    assert checkpoint["candidates"]["NEWF"]["invalidation_events"][-1] == {
        "at": checkpoint["candidates"]["NEWF"]["invalidation_events"][-1]["at"],
        "reason": "STAGE_EVIDENCE_FINGERPRINT_CHANGED",
        "stage": "IDENTITY",
    }


def test_v2_resume_reopens_first_source_limited_stage_when_repairs_are_enabled(
    repair_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = repair_env
    checkpoint_path = tmp_path / "observe_then_repair_checkpoint.json"
    calls: list[tuple[str, bool]] = []

    def stage_with_observe_only_filing(
        stage: str,
        *,
        apply_repairs: bool,
        **kwargs,
    ) -> dict[str, object]:
        calls.append((stage, apply_repairs))
        if stage == "FILINGS" and not apply_repairs:
            return {
                "outcome": "NEEDS_DATA",
                "reason_code": "ANNUAL_FILING_NOT_CACHED",
                "terminal": True,
                "retryable": False,
                "source_exhausted": True,
                "repair_policy": "LOCAL_EVIDENCE_ONLY",
            }
        return {
            "outcome": "OK",
            "source_exhausted": False,
            "repair_policy": ("LOCAL_THEN_REPAIR" if apply_repairs else "LOCAL_EVIDENCE_ONLY"),
        }

    monkeypatch.setattr(er, "_execute_v2_repair_stage", stage_with_observe_only_filing)
    common = {
        "tickers": ["GAPS"],
        "as_of_date": "2026-06-11",
        "db_path": cfg.db_path,
        "cfg": cfg,
        "pipeline_version": "v2",
        "checkpoint_path": checkpoint_path,
        "checkpoint_scope": "energy:large_and_mega",
        "candidate_context": {"cap_classifications": {"GAPS": {"market_cap_mm": 12_500.0}}},
    }

    observed = er.pre_assembly_data_gap_repair(**common, apply_repairs=False)
    assert observed["execution_status"] == "COMPLETED"
    assert calls == [(stage, False) for stage in er.V2_REPAIR_STAGE_SEQUENCE]

    calls.clear()
    repaired = er.pre_assembly_data_gap_repair(**common, apply_repairs=True)
    checkpoint = json.loads(checkpoint_path.read_text())
    candidate = checkpoint["candidates"]["GAPS"]

    assert repaired["resumed_from_checkpoint"] is True
    assert repaired["execution_status"] == "COMPLETED"
    assert calls == [(stage, True) for stage in er.V2_REPAIR_STAGE_SEQUENCE[4:]]
    assert candidate["invalidation_events"][-1] == {
        "stage": "FILINGS",
        "reason": "REPAIR_POLICY_EXPANDED",
        "at": candidate["invalidation_events"][-1]["at"],
    }
    assert candidate["stages"]["IDENTITY"]["result"]["repair_policy"] == ("LOCAL_EVIDENCE_ONLY")
    assert candidate["stages"]["FILINGS"]["result"]["repair_policy"] == ("LOCAL_THEN_REPAIR")
    assert candidate["stages"]["FILINGS"]["invalidation_reason"] == ("REPAIR_POLICY_EXPANDED")


def test_v2_checkpoint_batches_process_more_than_twenty_five_without_exclusion(
    repair_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = repair_env
    tickers = [f"Z{i:03d}" for i in range(31)]
    monkeypatch.setattr(
        er,
        "_execute_v2_repair_stage",
        lambda stage, *, ticker, **kwargs: {"outcome": "OK"},
    )
    result = er.pre_assembly_data_gap_repair(
        tickers=tickers,
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        max_repairs=25,
        pipeline_version="v2",
        checkpoint_path=tmp_path / "thirty_one_checkpoint.json",
        checkpoint_scope="energy:large_and_mega",
    )

    assert result["examined"] == 31
    assert result["batch_count"] == 2
    assert result["completed_tickers"] == tickers
    assert result["pending_tickers"] == []
    assert result["skipped_over_cap"] == []


def test_v2_repair_checkpoint_cannot_widen_frozen_execution_set(
    repair_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = repair_env
    checkpoint_path = tmp_path / "frozen_execution_checkpoint.json"
    stage_calls: list[tuple[str, str]] = []

    def complete_stage(stage: str, *, ticker: str, **kwargs):
        stage_calls.append((ticker, stage))
        return {"outcome": "OK"}

    monkeypatch.setattr(er, "_execute_v2_repair_stage", complete_stage)
    er.pre_assembly_data_gap_repair(
        tickers=["AAA"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        pipeline_version="v2",
        checkpoint_path=checkpoint_path,
        checkpoint_scope="energy:large_and_mega",
    )
    assert len(stage_calls) == 9

    stage_calls.clear()
    with pytest.raises(
        er.V2ExecutionBoundDriftError,
        match="frozen execution set does not match",
    ):
        er.pre_assembly_data_gap_repair(
            tickers=["AAA", "BBB"],
            as_of_date="2026-06-11",
            db_path=cfg.db_path,
            cfg=cfg,
            pipeline_version="v2",
            checkpoint_path=checkpoint_path,
            checkpoint_scope="energy:large_and_mega",
        )

    assert stage_calls == []
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["candidate_order"] == ["AAA"]
    assert sorted(checkpoint["candidates"]) == ["AAA"]


def test_v2_repair_resume_continues_after_last_completed_stage_without_loss(
    repair_env, monkeypatch, tmp_path: Path
) -> None:
    cfg = repair_env
    checkpoint_path = tmp_path / "interrupted_checkpoint.json"
    interrupted_calls: list[tuple[str, str]] = []

    def interrupt_at_second_filing(stage: str, *, ticker: str, **kwargs):
        interrupted_calls.append((ticker, stage))
        if ticker == "GAP2" and stage == "FILINGS":
            raise KeyboardInterrupt("operator interrupt")
        return {"outcome": "OK"}

    monkeypatch.setattr(er, "_execute_v2_repair_stage", interrupt_at_second_filing)
    with pytest.raises(KeyboardInterrupt, match="operator interrupt"):
        er.pre_assembly_data_gap_repair(
            tickers=["GAPS", "GAP2", "GAP3"],
            as_of_date="2026-06-11",
            db_path=cfg.db_path,
            cfg=cfg,
            max_repairs=2,
            pipeline_version="v2",
            checkpoint_path=checkpoint_path,
            checkpoint_scope="energy:large_and_mega",
        )

    interrupted = json.loads(checkpoint_path.read_text())
    assert interrupted["execution_status"] == "INTERRUPTED"
    assert interrupted["candidate_order"] == ["GAPS", "GAP2", "GAP3"]
    assert interrupted["candidates"]["GAPS"]["queue_status"] == "COMPLETED"
    assert interrupted["candidates"]["GAP2"]["last_completed_stage"] == "FACTS_AVAILABILITY"
    assert interrupted["candidates"]["GAP2"]["next_stage"] == "FILINGS"
    assert interrupted["candidates"]["GAP2"]["stages"]["FILINGS"]["status"] == "INTERRUPTED"
    assert interrupted["candidates"]["GAP3"]["queue_status"] == "PENDING"

    resumed_calls: list[tuple[str, str]] = []

    def finish_remaining(stage: str, *, ticker: str, **kwargs):
        resumed_calls.append((ticker, stage))
        return {"outcome": "OK"}

    monkeypatch.setattr(er, "_execute_v2_repair_stage", finish_remaining)
    resumed = er.pre_assembly_data_gap_repair(
        tickers=["GAPS", "GAP2", "GAP3"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        max_repairs=2,
        pipeline_version="v2",
        checkpoint_path=checkpoint_path,
        checkpoint_scope="energy:large_and_mega",
    )

    assert resumed["resumed_from_checkpoint"] is True
    assert resumed["execution_status"] == "COMPLETED"
    assert resumed["completed_tickers"] == ["GAPS", "GAP2", "GAP3"]
    assert resumed["pending_tickers"] == []
    assert resumed["skipped_over_cap"] == []
    assert resumed_calls == [
        ("GAP2", "FILINGS"),
        ("GAP2", "PARSING"),
        ("GAP2", "PRICE"),
        ("GAP2", "VALUATION"),
        ("GAP2", "PACKET"),
        ("GAP3", "MEMBERSHIP"),
        ("GAP3", "IDENTITY"),
        ("GAP3", "CAP"),
        ("GAP3", "FACTS_AVAILABILITY"),
        ("GAP3", "FILINGS"),
        ("GAP3", "PARSING"),
        ("GAP3", "PRICE"),
        ("GAP3", "VALUATION"),
        ("GAP3", "PACKET"),
    ]


def test_v2_repair_failed_stage_stays_queued_while_other_candidates_finish(
    repair_env, monkeypatch, tmp_path: Path
) -> None:
    cfg = repair_env
    checkpoint_path = tmp_path / "failed_checkpoint.json"

    def fail_one_valuation(stage: str, *, ticker: str, **kwargs):
        if ticker == "GAPS" and stage == "VALUATION":
            raise RuntimeError("valuation temporarily unavailable")
        return {"outcome": "OK"}

    monkeypatch.setattr(er, "_execute_v2_repair_stage", fail_one_valuation)
    first = er.pre_assembly_data_gap_repair(
        tickers=["GAPS", "GAP2"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        max_repairs=1,
        pipeline_version="v2",
        checkpoint_path=checkpoint_path,
        checkpoint_scope="energy:large_and_mega",
    )

    assert first["execution_status"] == "INCOMPLETE"
    assert first["completed_tickers"] == ["GAP2"]
    assert first["pending_tickers"] == ["GAPS"]
    assert first["skipped_over_cap"] == []
    states = {row["ticker"]: row for row in first["candidate_states"]}
    assert states["GAPS"]["last_completed_stage"] == "PRICE"
    assert states["GAPS"]["next_stage"] == "VALUATION"
    assert states["GAPS"]["stage_states"]["VALUATION"]["status"] == "FAILED"

    retry_calls: list[tuple[str, str]] = []

    def finish(stage: str, *, ticker: str, **kwargs):
        retry_calls.append((ticker, stage))
        return {"outcome": "OK"}

    monkeypatch.setattr(er, "_execute_v2_repair_stage", finish)
    second = er.pre_assembly_data_gap_repair(
        tickers=["GAPS", "GAP2"],
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        max_repairs=1,
        pipeline_version="v2",
        checkpoint_path=checkpoint_path,
        checkpoint_scope="energy:large_and_mega",
    )

    assert second["execution_status"] == "COMPLETED"
    assert second["completed_tickers"] == ["GAPS", "GAP2"]
    assert retry_calls == [("GAPS", "VALUATION"), ("GAPS", "PACKET")]


def test_sector_runtime_activates_staged_repair_only_for_explicit_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.autonomous import sector_runtime

    captured: dict[str, object] = {}
    assembly: dict[str, object] = {}

    def fake_repair(**kwargs):
        captured.update(kwargs)
        return {
            "status": "COMPLETED",
            "pipeline_version": "v2",
            "checkpoint_path": "/tmp/resumed.json",
            "skipped_over_cap": [],
            "candidate_states": [
                {
                    "ticker": "AAA",
                    "issuer_identity": {
                        "issuer_cik": "42",
                        "issuer_primary_ticker": "AAA.PRIMARY",
                        "issuer_aliases": ["AAA", "AAA.PRIMARY"],
                    },
                    "packet_inputs": {
                        "price": {
                            "outcome": "AVAILABLE",
                            "snapshot": {
                                "ticker": "AAA",
                                "price": 73.25,
                                "currency": "USD",
                                "as_of_date": "2026-06-10",
                                "source": "stooq",
                                "url": "https://stooq.example/aaa",
                                "confidence": "MEDIUM",
                            },
                        }
                    },
                }
            ],
        }

    monkeypatch.setattr(er, "pre_assembly_data_gap_repair", fake_repair)

    def fake_assemble(tickers, **kwargs):
        assembly.update({"tickers": tickers, **kwargs})
        raise RuntimeError("stop after repair")

    monkeypatch.setattr(sector_runtime, "assemble_sector_packets", fake_assemble)
    selection = {
        "loaded_tickers": ["AAA"],
        "selected_tickers": ["AAA"],
        "membership_tickers": ["AAA"],
        "execution_tickers": ["AAA"],
        "deferred_by_bound_tickers": [],
        "execution_bound": None,
        "membership_fingerprint": (
            "48e6f6a04a3a679b4e2bdb382af448541fff6a9ae6b13a5ea16146559adaa3f5"
        ),
        "execution_fingerprint": (
            "48e6f6a04a3a679b4e2bdb382af448541fff6a9ae6b13a5ea16146559adaa3f5"
        ),
        "execution_bound_frozen": True,
        "execution_as_of_date": "2026-06-11",
        "data_gap_repair_checkpoint_path": "/tmp/prior.json",
        "cap_classifications": {"AAA": {"market_cap_mm": 12_500.0}},
    }
    artifact = sector_runtime._run_sector_autonomous_financial_analysis_impl(
        sector="energy",
        tickers=["AAA"],
        as_of_date="2026-06-11",
        market_cap_focus="large_and_mega",
        candidate_selection=selection,
        pipeline_version="v2",
    )

    assert captured["tickers"] == ["AAA"]
    assert captured["as_of_date"] == "2026-06-11"
    assert captured["pipeline_version"] == "v2"
    assert assembly["filing_risk_use_llm"] is False
    assert captured["checkpoint_path"] == "/tmp/prior.json"
    assert captured["checkpoint_scope"] == "energy:large_and_mega"
    assert captured["candidate_context"] == selection
    assert assembly["issuer_contexts"] == {
        "AAA": {
            "market_cap_mm": 12_500.0,
            "issuer_cik": "42",
            "issuer_primary_ticker": "AAA.PRIMARY",
            "issuer_aliases": ["AAA", "AAA.PRIMARY"],
            "current_price": 73.25,
            "current_price_as_of_date": "2026-06-10",
            "current_price_currency": "USD",
            "current_price_source": "stooq",
            "current_price_source_url": "https://stooq.example/aaa",
            "current_price_confidence": "MEDIUM",
        }
    }
    assert assembly["current_prices"] == {"AAA": 73.25}
    assert artifact.candidate_selection["data_gap_repair"]["checkpoint_path"] == "/tmp/resumed.json"
    assert artifact.degraded_states == ["PACKET_ASSEMBLY_FAILED"]


def test_v2_filing_availability_recovers_raw_cache_without_local_path(
    repair_env,
) -> None:
    cfg = repair_env
    from app.parse.document_store import filing_local_path

    accession = "0000000042-26-000001"
    raw_path = filing_local_path("42", accession, "annual.htm")
    raw_path.write_text("cached annual filing bytes", encoding="utf-8")
    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        INSERT INTO filings(
            cik, ticker, accession, form_type, filing_date, period_end,
            primary_doc_url, local_path, status, created_at, updated_at
        ) VALUES('42', 'PRIMARY', ?, '20-F', '2026-03-01', '2025-12-31',
                 'https://www.sec.gov/Archives/edgar/data/42/annual.htm',
                 NULL, 'download_error', 'x', 'x')
        """,
        (accession,),
    )
    conn.commit()

    available = er._has_readable_issuer_annual_filing(
        conn,
        "ADR",
        as_of_date="2026-04-01",
        issuer_cik="42",
        aliases=("ADR", "PRIMARY"),
    )
    recovered = conn.execute(
        "SELECT status, local_path FROM filings WHERE accession = ?", (accession,)
    ).fetchone()
    conn.close()

    assert available is True
    assert recovered["status"] == "downloaded"
    assert recovered["local_path"] == str(raw_path)

    parsing = er._execute_v2_repair_stage(
        "PARSING",
        ticker="ADR",
        as_of_date="2026-04-01",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context={
            "cap_classifications": {
                "ADR": {
                    "issuer_cik": "42",
                    "issuer_primary_ticker": "PRIMARY",
                    "issuer_listed_tickers": ["PRIMARY", "ADR"],
                }
            }
        },
        apply_repairs=True,
    )
    assert parsing["outcome"] == "READABLE"
    assert parsing["parsed_count"] == 1


def test_v2_observation_only_filing_availability_does_not_persist_raw_recovery(
    repair_env,
) -> None:
    cfg = repair_env
    from app.parse.document_store import filing_local_path

    accession = "0000000042-26-000002"
    raw_path = filing_local_path("42", accession, "annual-observe-only.htm")
    raw_path.write_text("cached annual filing bytes", encoding="utf-8")
    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        INSERT INTO filings(
            cik, ticker, accession, form_type, filing_date, period_end,
            primary_doc_url, local_path, status, created_at, updated_at
        ) VALUES('42', 'PRIMARY', ?, '20-F', '2026-03-01', '2025-12-31',
                 'https://www.sec.gov/Archives/edgar/data/42/annual-observe-only.htm',
                 NULL, 'download_error', 'x', 'x')
        """,
        (accession,),
    )
    conn.commit()

    available = er._has_readable_issuer_annual_filing(
        conn,
        "ADR",
        as_of_date="2026-04-01",
        issuer_cik="42",
        aliases=("ADR", "PRIMARY"),
        persist_recovered=False,
    )
    observed = conn.execute(
        "SELECT status, local_path, hash FROM filings WHERE accession = ?", (accession,)
    ).fetchone()
    conn.close()

    assert available is True
    assert observed["status"] == "download_error"
    assert observed["local_path"] is None
    assert observed["hash"] is None

    parsing = er._execute_v2_repair_stage(
        "PARSING",
        ticker="ADR",
        as_of_date="2026-04-01",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context={
            "cap_classifications": {
                "ADR": {
                    "issuer_cik": "42",
                    "issuer_primary_ticker": "PRIMARY",
                    "issuer_listed_tickers": ["PRIMARY", "ADR"],
                }
            }
        },
        apply_repairs=False,
    )
    assert parsing["outcome"] == "NEEDS_DATA"
    assert parsing["reason_code"] == "ANNUAL_FILING_EXTRACTION_GAP"
    assert parsing["parsed_count"] == 0

    conn = sqlite3.connect(str(cfg.db_path))
    unchanged = conn.execute(
        "SELECT status, local_path, hash FROM filings WHERE accession = ?", (accession,)
    ).fetchone()
    conn.close()
    assert unchanged == ("download_error", None, None)


def test_v2_filing_stage_uses_injected_annual_windows(
    repair_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = repair_env
    annual_windows = {
        "10-K": 800,
        "10-K/A": 800,
        "20-F": 800,
        "20-F/A": 800,
        "40-F": 800,
        "40-F/A": 800,
    }
    monkeypatch.setattr(
        er,
        "_resolved_security_identity_payload",
        lambda *args, **kwargs: {
            "outcome": "RESOLVED",
            "issuer_cik": "42",
            "issuer_primary_ticker": "PRIMARY",
            "issuer_aliases": ["ADR", "PRIMARY"],
        },
    )
    readable_results = iter([False, True])
    monkeypatch.setattr(
        er,
        "_has_readable_issuer_annual_filing",
        lambda *args, **kwargs: next(readable_results),
    )
    monkeypatch.setattr(er, "ensure_company_row", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        er,
        "_filing_evidence_snapshot",
        lambda *args, **kwargs: {"filings": []},
    )
    captured: dict[str, object] = {}

    import app.ingest.filings as filings_mod

    def fake_ingest(**kwargs):
        captured.update(kwargs)
        return {
            "filings_considered": 1,
            "filings_upserted": 1,
            "filings_downloaded": 1,
            "filing_download_errors": 0,
        }

    monkeypatch.setattr(filings_mod, "ingest_with_policy", fake_ingest)
    result = er._execute_v2_repair_stage(
        "FILINGS",
        ticker="ADR",
        as_of_date="2026-07-17",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context={},
        apply_repairs=True,
        filing_windows_days=annual_windows,
    )

    assert captured == {
        "as_of_date": "2026-07-17",
        "run_id": "evidence_resolution_20260717",
        "tickers": ["PRIMARY"],
        "db_path": Path(cfg.db_path),
        "windows_days": annual_windows,
    }
    assert result["outcome"] == "AVAILABLE"
    assert result["reason_code"] is None
    assert result["had_readable_annual_filing"] is True


def test_v2_filing_download_failure_retries_under_primary_issuer_ticker(
    repair_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = repair_env
    identity = {
        "outcome": "RESOLVED",
        "issuer_cik": "42",
        "issuer_primary_ticker": "PRIMARY",
        "issuer_aliases": ["ADR", "PRIMARY"],
    }
    monkeypatch.setattr(
        er,
        "_resolved_security_identity_payload",
        lambda *args, **kwargs: dict(identity),
    )
    ensured: list[str] = []
    monkeypatch.setattr(
        er,
        "ensure_company_row",
        lambda ticker, **kwargs: ensured.append(ticker) or True,
    )
    filing_path = tmp_path / "retried-annual.htm"
    filing_path.write_text("retried annual filing", encoding="utf-8")
    calls = 0

    import app.ingest.filings as filings_mod

    def ingest(**kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["tickers"] == ["PRIMARY"]
        conn = sqlite3.connect(str(cfg.db_path))
        if calls == 1:
            conn.execute(
                """
                INSERT INTO filings(
                    cik, ticker, accession, form_type, filing_date, period_end,
                    primary_doc_url, local_path, status, created_at, updated_at
                ) VALUES(
                    '42', 'PRIMARY', 'retry-annual', '20-F', '2026-03-01',
                    '2025-12-31', 'https://www.sec.gov/retry.htm', NULL,
                    'download_error', 'x', 'x'
                )
                """
            )
            result = {
                "filings_considered": 1,
                "filings_upserted": 1,
                "filings_downloaded": 0,
                "filing_download_errors": 1,
            }
        else:
            conn.execute(
                "UPDATE filings SET local_path = ?, status = 'downloaded' "
                "WHERE accession = 'retry-annual'",
                (str(filing_path),),
            )
            result = {
                "filings_considered": 1,
                "filings_upserted": 1,
                "filings_downloaded": 1,
                "filing_download_errors": 0,
            }
        conn.commit()
        conn.close()
        return result

    monkeypatch.setattr(filings_mod, "ingest_with_policy", ingest)
    first = er._execute_v2_repair_stage(
        "FILINGS",
        ticker="ADR",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context={},
        apply_repairs=True,
    )
    second = er._execute_v2_repair_stage(
        "FILINGS",
        ticker="ADR",
        as_of_date="2026-06-11",
        db_path=Path(cfg.db_path),
        cfg=cfg,
        candidate_context={},
        apply_repairs=True,
    )

    assert first["outcome"] == "INCOMPLETE"
    assert first["reason_code"] == "ANNUAL_FILING_DOWNLOAD_FAILED"
    assert first["retryable"] is True
    assert first["source_exhausted"] is False
    assert second["outcome"] == "AVAILABLE"
    assert second["reason_code"] is None
    assert ensured == ["PRIMARY", "PRIMARY"]
    conn = sqlite3.connect(str(cfg.db_path))
    owner = conn.execute("SELECT ticker FROM filings WHERE accession = 'retry-annual'").fetchone()[
        0
    ]
    conn.close()
    assert owner == "PRIMARY"


def test_parser_uses_injected_db_and_authoritative_cik(
    repair_env,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = repair_env
    correct_path = tmp_path / "correct.htm"
    correct_path.write_text("<html><body>correct issuer filing</body></html>")
    wrong_path = tmp_path / "wrong.htm"
    wrong_path.write_text("<html><body>wrong issuer filing</body></html>")
    conn = sqlite3.connect(str(cfg.db_path))
    for cik, accession, filing_date, path in (
        ("42", "correct", "2026-02-01", correct_path),
        ("999", "wrong-newer", "2026-03-01", wrong_path),
    ):
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at
            ) VALUES(?, 'ALIAS', ?, '20-F', ?, '2025-12-31',
                     'https://www.sec.gov/filing.htm', ?, 'downloaded', 'x', 'x')
            """,
            (cik, accession, filing_date, str(path)),
        )
    conn.commit()
    conn.close()

    # Point the ambient config elsewhere. The explicit repair DB must still
    # be the only database parsed.
    from app.config import get_config
    from app.db import init_db

    wrong_db = tmp_path / "ambient" / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(wrong_db.parent))
    monkeypatch.setenv("VOE_DB_PATH", str(wrong_db))
    get_config.cache_clear()
    init_db(get_config())

    from app.parse.filing_parser import parse_pending_filings

    parsed = parse_pending_filings(
        limit=10,
        tickers=["ALIAS"],
        issuer_cik="42",
        db_path=cfg.db_path,
    )

    conn = sqlite3.connect(str(cfg.db_path))
    statuses = dict(conn.execute("SELECT accession, status FROM filings"))
    conn.close()
    assert parsed == 1
    assert statuses["correct"] == "parsed"
    assert statuses["wrong-newer"] == "downloaded"


def test_resolve_data_gaps_invokes_filing_and_valuation(repair_env, monkeypatch) -> None:
    cfg = repair_env
    calls: list[str] = []
    monkeypatch.setattr(er, "ensure_company_row", lambda t, **k: True)

    import app.ingest.filings as filings_mod
    import app.valuation.valuation_writer as vw_mod

    monkeypatch.setattr(
        filings_mod, "ingest_with_policy", lambda **kw: calls.append(f"filing:{kw['tickers']}")
    )
    monkeypatch.setattr(
        vw_mod, "ensure_valuation", lambda t, asof, **k: calls.append(f"valuation:{t}:{asof}")
    )

    result = er.resolve_data_gaps_for_ticker(
        "GAPS", as_of_date="2026-06-11", db_path=cfg.db_path, cfg=cfg
    )
    assert calls == ["filing:['GAPS']", "valuation:GAPS:2026-06-11"]
    assert result["actions"] == ["FILING_INGEST_ATTEMPTED", "VALUATION_REFRESHED"]
    assert result["errors"] == []


# ---------------------------------------------------------------------------
# Pool resolution over persisted artifacts
# ---------------------------------------------------------------------------


def _write_artifact(runs_dir: Path, *, sector: str, created_at: str, rows: list[dict]) -> None:
    run_dir = runs_dir / f"autonomous_sector_{sector}_{created_at.replace('-', '')}_abc123"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "autonomous_sector_run.json").write_text(
        json.dumps({"sector": sector, "created_at": created_at, "relative_ranking": rows})
    )


def test_resolve_and_reaudit_pool_promotes_fetchable_holds(tmp_path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    _write_artifact(
        runs_dir,
        sector="industrial_tech",
        created_at="2026-06-11",
        rows=[
            {
                "ticker": "KEQX",
                "audit_status": "BLOCKED",
                "hard_blockers": ["MISSING_VALUATION", "MISSING_BASE_RETURN_CASE"],
                "buy_candidate": True,
                "rank": 3,
            },
            {
                "ticker": "SOLV",
                "audit_status": "BLOCKED",
                "hard_blockers": ["SOLVENCY_CRITICAL"],
                "buy_candidate": False,
                "rank": 9,
            },
            {"ticker": "FINE", "audit_status": "WATCHLIST_ONLY", "hard_blockers": []},
        ],
    )
    # An older artifact for the same sector must be ignored.
    _write_artifact(
        runs_dir,
        sector="industrial_tech",
        created_at="2026-05-16",
        rows=[
            {"ticker": "OLDN", "audit_status": "BLOCKED", "hard_blockers": ["MISSING_VALUATION"]}
        ],
    )

    monkeypatch.setattr(
        er,
        "resolve_data_gaps_for_ticker",
        lambda t, **k: {"ticker": t, "actions": ["VALUATION_REFRESHED"], "errors": []},
    )
    monkeypatch.setattr(
        er,
        "reaudit_held_candidate",
        lambda t, **k: {
            "ticker": t,
            "status": "DATA_INCOMPLETE",
            "hard_blockers": ["MISSING_COMPANY_SPECIFIC_EVIDENCE"],
            "confidence_caps": [],
        },
    )

    report = er.resolve_and_reaudit_pool(as_of_date="2026-06-11", runs_dir=runs_dir)

    assert report["held_total"] == 2
    assert report["held_fetchable_only"] == 1
    assert report["reaudited"] == 1
    assert report["transitions"] == {"BLOCKED->DATA_INCOMPLETE": 1}
    assert report["promoted_count"] == 1
    assert [g["ticker"] for g in report["held_genuine"]] == ["SOLV"]
    outcome = report["outcomes"][0]
    assert outcome["ticker"] == "KEQX"
    assert outcome["transition"] == "BLOCKED->DATA_INCOMPLETE"


def _accepted_identity_cap() -> dict:
    return {
        "ticker": "ACPT",
        "as_of_date": "2026-07-17",
        "market_cap_mm": 25_000.0,
        "cap_source": "accepted_census",
        "cap_source_kind": "ACCEPTED_CENSUS",
        "cap_effective_as_of_date": "2026-07-17",
        "issuer_cik": "0000000123",
        "issuer_primary_ticker": "ACPT",
        "issuer_listed_tickers": ["ACPT"],
        "security_role": "PRIMARY",
        "is_secondary_class": False,
        "is_adr": False,
        "issuer_key": "issuer-123",
        "security_key": "security-123",
        "census_run_id": "accepted-run-1",
        "census_input_fingerprint": "input-fingerprint",
        "census_semantic_output_fingerprint": "semantic-fingerprint",
        "census_cohort_fingerprint": "cohort-fingerprint",
        "identity_source": "sec_registry",
        "identity_source_url": "https://www.sec.gov/files/company_tickers_exchange.json",
        "identity_as_of_date": "2026-07-17",
        "identity_confidence": "HIGH",
    }


def test_accepted_census_identity_bypasses_legacy_security_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "engine.db"
    sqlite3.connect(db_path).close()
    monkeypatch.setattr(
        "app.autonomous.cap_resolver.resolve_security_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("accepted identity must not consult legacy resolution")
        ),
    )

    result = er._resolved_security_identity_payload(
        "ACPT",
        as_of_date="2026-07-17",
        db_path=db_path,
        candidate_context={"cap_classifications": {"ACPT": _accepted_identity_cap()}},
    )

    assert result["outcome"] == "RESOLVED"
    assert result["issuer_cik"] == "123"
    assert result["issuer_primary_ticker"] == "ACPT"
    assert result["issuer_listed_tickers"] == ["ACPT"]
    assert result["issuer_aliases"] == ["ACPT"]
    assert result["identity_authority"] == "accepted_census"
    assert result["issuer_key"] == "issuer-123"
    assert result["security_key"] == "security-123"


def test_accepted_census_identity_drift_fails_closed() -> None:
    cap = _accepted_identity_cap()
    cap["issuer_listed_tickers"] = ["ACPT", "OTHER"]

    with pytest.raises(
        er.V2ExecutionBoundDriftError,
        match="Accepted-census identity drifted for ACPT: issuer_listed_tickers",
    ):
        er._accepted_census_identity_payload(
            "ACPT",
            as_of_date="2026-07-17",
            cap=cap,
        )
