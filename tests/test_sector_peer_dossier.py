from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analyst.thesis_contract import (
    AnalysisFinding,
    AnalysisReport,
    OpenQuestion,
    Falsifier,
    ValuationConclusion,
)
from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.db import get_db, init_db, utc_now_iso
from app.dossier.peer_report import build_peer_report
from app.dossier.time_series import build_time_series
from app.llm.schemas import validate_sector_synthesis_packet
from app.llm.usage_capture import (
    attached_provider_usage_records,
    provider_usage_capture,
)
from app.sector.cycle import run_sector_cycle, sector_run_status
from app.sector.peer_set import select_sector_peers
from app.sector.synthesis import run_sector_synthesis
from tests.financial_integrity_helpers import materialized_no_split_proof


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    taxonomy_path = data_dir / "universe" / "sector_taxonomy.csv"
    taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    monkeypatch.setenv("VOE_SECTOR_TAXONOMY_PATH", str(taxonomy_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg, taxonomy_path


def _seed_sector_synthesis_inputs(cfg, run_id: str, tickers: list[str]) -> None:
    dossier_run_dir = cfg.dossiers_dir / run_id
    dossier_run_dir.mkdir(parents=True, exist_ok=True)
    (dossier_run_dir / "peer_rankings.json").write_text(
        json.dumps(
            {
                "tickers": tickers,
                "future_whale_rank": tickers,
                "quality_rank": tickers,
                "valuation_rank": tickers,
                "risk_rank": tickers,
                "sections": {},
            }
        ),
        encoding="utf-8",
    )
    for index, ticker in enumerate(tickers, start=1):
        cik = f"{index:010d}"
        ddir = dossier_run_dir / ticker
        ddir.mkdir(parents=True, exist_ok=True)
        (ddir / "dossier.json").write_text(
            json.dumps(
                {
                    "ticker": ticker,
                    "as_of_date": "2026-02-13",
                    "time_series": {"standardized_rows": [], "derived_signals": []},
                }
            ),
            encoding="utf-8",
        )
        scorecard = {
            "pricing_zone": "MARGIN_OF_SAFETY",
            "pricing_zone_detail": {
                "current_price": 50.0,
                "current_price_as_of_date": "2026-02-13",
                "current_price_currency": "USD",
                "current_price_source": "fixture_quote",
                "current_price_source_url": "https://example.test/quote",
                "current_price_basis": "UNADJUSTED",
                "current_raw_price": 50.0,
                "split_adjustment_factor": 1.0,
                "no_intervening_split_proof": materialized_no_split_proof(
                    ticker=ticker,
                    period_start="2025-12-31",
                    period_end="2026-02-13",
                    issuer_cik=cik,
                ),
                "dcf_base": 75.0,
                "epv_adjusted": 65.0,
                "gate_action": "PROCEED",
            },
            "quality_context": {
                "gate_action": "PROCEED",
                "confidence_class": "MODERATE",
            },
        }
        with get_db() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO companies(ticker, cik, name, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (ticker, cik, f"{ticker} Fixture", utc_now_iso()),
            )
            conn.execute(
                """
                INSERT INTO valuations(
                    ticker, as_of_date, method, inputs_json, outputs_json,
                    warnings_json, created_at
                )
                VALUES(?, '2026-02-13', 'scorecard', '{}', ?, '[]', ?)
                """,
                (ticker, json.dumps(scorecard), utc_now_iso()),
            )
            conn.execute(
                """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date, form,
                    accession
                )
                VALUES(
                    ?, 2025, 'FY', '2025-12-31', 'shares_outstanding',
                    10.0, 'shares_millions', ?,
                    ?, '2026-02-01', '10-K', ?
                )
                """,
                (
                    ticker,
                    (f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"),
                    utc_now_iso(),
                    f"{cik}-26-{index:06d}",
                ),
            )
        submissions_dir = cfg.cache_dir / "submissions"
        submissions_dir.mkdir(parents=True, exist_ok=True)
        (submissions_dir / f"{cik}.json").write_text(
            json.dumps(
                {
                    "tickers": [ticker],
                    "exchanges": ["NYSE"],
                    "filings": {
                        "recent": {
                            "form": ["10-K"],
                            "filingDate": ["2026-02-01"],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )


def _seed_analysis_report(
    cfg,
    *,
    ticker: str,
    as_of_date: str,
    thesis_summary: str,
    positives: list[str],
    risks: list[str],
    recent_events: list[str],
    open_questions: list[str],
    falsifiers: list[str],
    verdict: str = "BUY",
    confidence_label: str = "HIGH",
    confidence_score: int = 80,
    warnings: list[str] | None = None,
) -> None:
    out_dir = cfg.analyst_outputs_dir / f"{ticker}_{as_of_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "analysis_report.json"
    report = AnalysisReport(
        analysis_id=f"{ticker}_{as_of_date}_test",
        ticker=ticker,
        as_of_date=as_of_date,
        generated_at="2026-02-13T00:00:00+00:00",
        verdict=verdict,
        confidence_label=confidence_label,
        confidence_score=confidence_score,
        thesis_summary=thesis_summary,
        valuation=ValuationConclusion(
            price=100.0,
            base_case_value=140.0,
            bear_case_value=110.0,
            bull_case_value=170.0,
            margin_of_safety=0.28,
        ),
        positives=[
            AnalysisFinding(
                finding_id=f"P{i}",
                category="POSITIVE",
                claim=claim,
                direction="BULLISH",
                severity="HIGH",
                source_basis="filings",
                citation_ids=[],
            )
            for i, claim in enumerate(positives, start=1)
        ],
        risks=[
            AnalysisFinding(
                finding_id=f"R{i}",
                category="RISK",
                claim=claim,
                direction="BEARISH",
                severity="MODERATE",
                source_basis="filings",
                citation_ids=[],
            )
            for i, claim in enumerate(risks, start=1)
        ],
        recent_event_impacts=[
            AnalysisFinding(
                finding_id=f"E{i}",
                category="RECENT_EVENT",
                claim=claim,
                direction=None,
                severity="MODERATE",
                source_basis="recent_events",
                citation_ids=[],
            )
            for i, claim in enumerate(recent_events, start=1)
        ],
        open_questions=[
            OpenQuestion(question=question, importance="MEDIUM") for question in open_questions
        ],
        falsifiers=[
            Falsifier(description=description, trigger_type="THESIS_BREAK")
            for description in falsifiers
        ],
        warnings=warnings or [],
    )
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO analyst_outputs(ticker, as_of_date, output_type, output_path, output_hash, created_at)
            VALUES(?, ?, 'analysis_report', ?, 'hash', ?)
            """,
            (ticker, as_of_date, str(path), utc_now_iso()),
        )


def _seed_legacy_research_packet(
    cfg,
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
    findings: list[str],
    risks: list[str],
    catalysts: list[str],
    disconfirming: list[str],
    evidence_gaps: list[str] | None = None,
) -> None:
    out_dir = cfg.outputs_dir / "research" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}_{as_of_date}.json"
    payload = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "run_id": run_id,
        "findings": [
            {"entry_id": f"F{i}", "summary": claim} for i, claim in enumerate(findings, start=1)
        ],
        "risks": [
            {"entry_id": f"R{i}", "summary": claim} for i, claim in enumerate(risks, start=1)
        ],
        "catalysts": [
            {"entry_id": f"C{i}", "summary": claim} for i, claim in enumerate(catalysts, start=1)
        ],
        "disconfirming_evidence": [
            {"entry_id": f"D{i}", "summary": claim}
            for i, claim in enumerate(disconfirming, start=1)
        ],
        "evidence_gaps": list(evidence_gaps or []),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO research_packets(ticker, as_of_date, run_id, packet_path, packet_hash, created_at)
            VALUES(?, ?, ?, ?, 'hash', ?)
            """,
            (ticker, as_of_date, run_id, str(path), utc_now_iso()),
        )


class _CaptureSectorProvider:
    provider_name = "openai"

    def __init__(self) -> None:
        self.inputs: dict[str, object] | None = None

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
        _ = (schema, schema_name)
        self.inputs = json.loads(prompt.split("INPUT_PAYLOAD:\n", 1)[1])
        peer_tickers = list(self.inputs.get("peer_tickers") or [])
        payload = {
            "sector": self.inputs.get("sector"),
            "as_of_date": self.inputs.get("as_of_date"),
            "run_id": self.inputs.get("run_id"),
            "sector_summary": "Summary",
            "valuation_interpretation": "Interpretation",
            "risk_frame": "Risk frame",
            "catalyst_frame": "Catalyst frame",
            "evidence_gaps": [],
            "recommended_next_actions": ["Read the next filing."],
            "confidence_notes": "Moderate confidence.",
            "peer_tickers": peer_tickers,
            "top_pick": peer_tickers[0],
            "runner_ups": peer_tickers[1:2],
            "avoid_list": [],
            "whale_checklist": ["Durable moat"],
            "per_ticker_narrative": [
                {
                    "ticker": ticker,
                    "moat_indicators": ["Strong moat"],
                    "operating_leverage": "Operating leverage improving",
                    "capital_allocation": "Disciplined capital allocation",
                    "reinvestment_runway": "Long reinvestment runway",
                    "citations": [],
                    "derived_from": [f"peer_rankings.{ticker}"],
                }
                for ticker in peer_tickers
            ],
            "falsifiers": ["Execution weakens"],
            "what_to_read_next": ["Latest annual and quarterly filings"],
            "claims": [
                {
                    "id": "sector_claim",
                    "text": "Sector synthesis used normalized analysis inputs.",
                    "type": "non_numeric",
                    "citations": [],
                    "derived_from": ["ticker_analysis"],
                }
            ],
            "llm_meta": {
                "model": "gpt-5-mini",
                "prompt_hash": "placeholder",
                "input_hash": "placeholder",
                "cost_estimate_usd": 0.0,
                "created_at": "2026-02-13T00:00:00+00:00",
            },
        }

        class _Result:
            pass

        result = _Result()
        result.json_text = json.dumps(payload)
        result.model = "gpt-5-mini"
        result.usage_input_tokens = 1000
        result.usage_output_tokens = 300
        return result


class _ForbiddenSectorProvider:
    provider_name = "openai"

    def __init__(self) -> None:
        self.calls = 0

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, **kwargs):
        self.calls += 1
        raise AssertionError(f"invalid financial context reached provider: {kwargs}")


def test_sector_synthesis_invalid_financial_context_has_zero_provider_calls_and_no_output(
    monkeypatch,
    tmp_path,
):
    cfg, _ = _init_temp_db(monkeypatch, tmp_path)
    run_id = "invalid-financial-context"
    dossier_run_dir = cfg.dossiers_dir / run_id
    dossier_run_dir.mkdir(parents=True, exist_ok=True)
    (dossier_run_dir / "peer_rankings.json").write_text(
        json.dumps(
            {
                "tickers": ["AAA"],
                "future_whale_rank": ["AAA"],
                "quality_rank": ["AAA"],
                "valuation_rank": ["AAA"],
                "risk_rank": ["AAA"],
                "sections": {},
            }
        ),
        encoding="utf-8",
    )
    ticker_dir = dossier_run_dir / "AAA"
    ticker_dir.mkdir(parents=True, exist_ok=True)
    (ticker_dir / "dossier.json").write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "as_of_date": "2026-02-13",
                "time_series": {"standardized_rows": [], "derived_signals": []},
            }
        ),
        encoding="utf-8",
    )
    provider = _ForbiddenSectorProvider()
    monkeypatch.setattr("app.sector.synthesis.get_llm_provider", lambda: provider)

    with pytest.raises(InvalidFinancialInputError):
        run_sector_synthesis(
            sector="Software",
            as_of_date="2026-02-13",
            run_id=run_id,
        )

    assert provider.calls == 0
    assert not (cfg.sectors_dir / run_id / "sector_synthesis.json").exists()


def test_sector_synthesis_excludes_unbound_financial_rows_and_publishes_scope_binding(
    monkeypatch,
    tmp_path,
):
    cfg, _ = _init_temp_db(monkeypatch, tmp_path)
    run_id = "sector-synth-canonical-prompt"
    _seed_sector_synthesis_inputs(cfg, run_id, ["AAA"])
    contradictory_value = 987_654_321.0

    dossier_path = cfg.dossiers_dir / run_id / "AAA" / "dossier.json"
    dossier_path.write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "as_of_date": "2026-02-13",
                "time_series": {
                    "standardized_rows": [
                        {"metric": "current_price", "value": contradictory_value}
                    ],
                    "derived_signals": [{"metric": "market_cap_mm", "value": contradictory_value}],
                },
            }
        ),
        encoding="utf-8",
    )
    stale_synthesis_path = cfg.outputs_dir / "synthesis" / "AAA_2026-02-13_unbound.json"
    stale_synthesis_path.parent.mkdir(parents=True, exist_ok=True)
    stale_synthesis_path.write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "valuation_interpretation": (f"Unbound mutable valuation {contradictory_value}"),
            }
        ),
        encoding="utf-8",
    )
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at
            )
            VALUES(
                'AAA', '2026-02-13', 'unbound_mutable_valuation', '{}', ?,
                '[]', ?
            )
            """,
            (
                json.dumps({"market_cap_mm": contradictory_value}),
                utc_now_iso(),
            ),
        )
        conn.execute(
            """
            INSERT INTO synthesis_packets(
                ticker, as_of_date, run_id, packet_path, packet_hash,
                packet_json, prompt_hash, input_hash, provider, model,
                usage_json, cost_estimate_usd, from_cache, created_at
            )
            VALUES(
                'AAA', '2026-02-13', ?, ?, 'hash', '{}', 'prompt', 'input',
                'openai', 'test', '{}', 0, 0, ?
            )
            """,
            (run_id, str(stale_synthesis_path), utc_now_iso()),
        )

    provider = _CaptureSectorProvider()
    monkeypatch.setattr("app.sector.synthesis.get_llm_provider", lambda: provider)

    summary = run_sector_synthesis(
        sector="Software",
        as_of_date="2026-02-13",
        run_id=run_id,
    )

    assert provider.inputs is not None
    assert "top_dossiers" not in provider.inputs
    assert "valuation_snapshots" not in provider.inputs
    assert "ticker_layer3_synthesis" not in provider.inputs
    assert str(int(contradictory_value)) not in json.dumps(
        provider.inputs,
        sort_keys=True,
    )
    canonical = provider.inputs["canonical_financial_packets"]["AAA"]
    assert canonical["current_price"] == 50.0
    assert canonical["shares_outstanding_mm"] == 10.0
    assert canonical["market_cap_mm"] == 500.0
    binding = provider.inputs["financial_integrity_binding"]
    assert binding["status"] == "PASS"
    assert binding["scope_fingerprint"] == summary["financial_integrity_scope_fingerprint"]

    payload = json.loads(Path(summary["path"]).read_text(encoding="utf-8"))
    assert payload["financial_integrity"]["scope_fingerprint"] == binding["scope_fingerprint"]
    assert payload["financial_integrity"]["input_hash"] == summary["input_hash"]
    assert payload["financial_integrity"]["prompt_hash"] == summary["prompt_hash"]
    assert summary["cost_estimate_usd"] == 0.0019
    assert summary["provider_usage"][0]["cost_estimate_usd"] == 0.0019
    assert payload["llm_meta"]["cost_estimate_usd"] == 0.0019


def test_sector_synthesis_rebinds_canonical_prompt_payload_after_provider(
    monkeypatch,
    tmp_path,
) -> None:
    from app.sector import synthesis as synthesis_module

    cfg, _ = _init_temp_db(monkeypatch, tmp_path)
    run_id = "sector-synth-mutated-prompt"
    _seed_sector_synthesis_inputs(cfg, run_id, ["AAA"])
    original_build_inputs = synthesis_module._build_inputs
    bound_inputs: dict[str, dict] = {}

    def _capture_inputs(**kwargs):
        payload = original_build_inputs(**kwargs)
        bound_inputs["payload"] = payload
        return payload

    class _MutatingSectorProvider(_CaptureSectorProvider):
        def synthesize_json(self, **kwargs):
            bound_inputs["payload"]["canonical_financial_packets"]["AAA"]["current_price"] = 51.0
            return super().synthesize_json(**kwargs)

    provider = _MutatingSectorProvider()
    monkeypatch.setattr(synthesis_module, "_build_inputs", _capture_inputs)
    monkeypatch.setattr(synthesis_module, "get_llm_provider", lambda: provider)

    with (
        provider_usage_capture("rlm_executor") as captured,
        pytest.raises(InvalidFinancialInputError) as exc_info,
    ):
        run_sector_synthesis(
            sector="Software",
            as_of_date="2026-02-13",
            run_id=run_id,
        )

    assert len(captured) == 1
    assert captured[0]["cost_estimate_usd"] == 0.0019
    attached = attached_provider_usage_records(exc_info.value)
    assert len(attached) == 1
    assert attached[0]["cost_estimate_usd"] == 0.0019
    assert not (cfg.sectors_dir / run_id / "sector_synthesis.json").exists()


def test_sector_synthesis_failed_call_cannot_hide_scope_mutation(
    monkeypatch,
    tmp_path,
) -> None:
    from app.llm.providers.retry_guard import _notify_failed_attempt
    from app.sector import synthesis as synthesis_module

    cfg, _ = _init_temp_db(monkeypatch, tmp_path)
    run_id = "sector-synth-failed-mutated-prompt"
    _seed_sector_synthesis_inputs(cfg, run_id, ["AAA"])
    original_build_inputs = synthesis_module._build_inputs
    bound_inputs: dict[str, dict] = {}

    def _capture_inputs(**kwargs):
        payload = original_build_inputs(**kwargs)
        bound_inputs["payload"] = payload
        return payload

    class _FailingMutatingSectorProvider(_CaptureSectorProvider):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def synthesize_json(self, **_kwargs):
            self.calls += 1
            bound_inputs["payload"]["canonical_financial_packets"]["AAA"]["current_price"] = 51.0
            error = RuntimeError("sector provider failed after mutation")
            _notify_failed_attempt(
                provider="openai",
                schema_name="sector_synthesis_packet_v1",
                attempt=1,
                exc=error,
                retryable=False,
                will_retry=False,
            )
            raise error

    provider = _FailingMutatingSectorProvider()
    monkeypatch.setattr(synthesis_module, "_build_inputs", _capture_inputs)
    monkeypatch.setattr(synthesis_module, "get_llm_provider", lambda: provider)

    with (
        provider_usage_capture("rlm_executor") as captured,
        pytest.raises(InvalidFinancialInputError) as exc_info,
    ):
        run_sector_synthesis(
            sector="Software",
            as_of_date="2026-02-13",
            run_id=run_id,
        )

    assert provider.calls == 1
    assert len(captured) == 1
    assert captured[0]["status"] == "ERROR"
    attached = attached_provider_usage_records(exc_info.value)
    assert len(attached) == 1
    assert attached[0]["status"] == "ERROR"
    assert attached[0]["cost_estimate_usd"] == captured[0]["cost_estimate_usd"]
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert not (cfg.sectors_dir / run_id / "sector_synthesis.json").exists()


def test_sector_peer_selection_is_deterministic(monkeypatch, tmp_path):
    cfg, taxonomy_path = _init_temp_db(monkeypatch, tmp_path)
    taxonomy_path.write_text(
        "ticker,sector\nAAA,Software\nBBB,Software\nCCC,Software\n",
        encoding="utf-8",
    )
    now = utc_now_iso()
    with get_db() as conn:
        for ticker in ["AAA", "BBB", "CCC"]:
            conn.execute(
                """
                INSERT INTO companies(ticker, cik, name, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (ticker, ticker, f"{ticker} Inc", now),
            )
            conn.execute(
                """
                INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
                VALUES(?, '2026-02-01', ?, '{}', ?)
                """,
                (ticker, json.dumps({"revenue": 100.0}), now),
            )
    first = select_sector_peers(sector="Software", as_of_date="2026-02-13", limit=3)
    second = select_sector_peers(sector="Software", as_of_date="2026-02-13", limit=3)
    assert first["selected_tickers"] == second["selected_tickers"]
    assert first["selected_tickers"] == sorted(first["selected_tickers"])


def test_sector_peer_selection_hybrid_order_is_deterministic(monkeypatch, tmp_path):
    _, taxonomy_path = _init_temp_db(monkeypatch, tmp_path)
    taxonomy_path.write_text("ticker,sector\nAAA,Software\n", encoding="utf-8")
    now = utc_now_iso()
    with get_db() as conn:
        for ticker in ["AAA", "BBB", "CCC", "DDD"]:
            conn.execute(
                """
                INSERT INTO companies(ticker, cik, name, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (ticker, ticker, f"{ticker} Inc", now),
            )
            conn.execute(
                """
                INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
                VALUES(?, '2026-02-01', ?, '{}', ?)
                """,
                (ticker, json.dumps({"revenue": 100.0}), now),
            )

    inferred = {
        "AAA": ("software", "keywords"),
        "BBB": ("software", "keywords"),
        "CCC": ("software", "keywords"),
        "DDD": ("consumer", "keywords"),
    }

    monkeypatch.setattr(
        "app.sector.peer_set._infer_filings_category",
        lambda conn, *, ticker, as_of_date, sec_client, sec_cache: inferred.get(
            ticker, (None, "unknown")
        ),
    )
    first = select_sector_peers(sector="Software", as_of_date="2026-02-13", limit=3, mode="hybrid")
    second = select_sector_peers(sector="Software", as_of_date="2026-02-13", limit=3, mode="hybrid")
    assert first["selected_tickers"] == ["AAA", "BBB", "CCC"]
    assert first["selected_tickers"] == second["selected_tickers"]
    assert first["peer_selection_summary"]["taxonomy_hits"] == 1
    assert first["peer_selection_summary"]["filings_inferred_hits"] == 2


def test_sector_peer_selection_fallback_when_taxonomy_empty(monkeypatch, tmp_path):
    _, taxonomy_path = _init_temp_db(monkeypatch, tmp_path)
    taxonomy_path.write_text("ticker,sector\nZZZ,Energy\n", encoding="utf-8")
    now = utc_now_iso()
    with get_db() as conn:
        for ticker in ["AAA", "BBB"]:
            conn.execute(
                """
                INSERT INTO companies(ticker, cik, name, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (ticker, ticker, f"{ticker} Inc", now),
            )
            conn.execute(
                """
                INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
                VALUES(?, '2026-02-01', ?, '{}', ?)
                """,
                (ticker, json.dumps({"revenue": 120.0}), now),
            )

    monkeypatch.setattr(
        "app.sector.peer_set._infer_filings_category",
        lambda conn, *, ticker, as_of_date, sec_client, sec_cache: ("software", "keywords"),
    )
    payload = select_sector_peers(
        sector="Software", as_of_date="2026-02-13", limit=2, mode="hybrid"
    )
    assert payload["selected_tickers"] == ["AAA", "BBB"]
    assert payload["peer_selection_summary"]["taxonomy_hits"] == 0
    assert payload["peer_selection_summary"]["filings_inferred_hits"] == 2


def test_sector_peer_selection_sparse_taxonomy_triggers_filings_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_SECTOR_TAXONOMY_SPARSE_THRESHOLD", "3")
    _, taxonomy_path = _init_temp_db(monkeypatch, tmp_path)
    taxonomy_path.write_text("ticker,sector\nAAA,Software\nBBB,Software\n", encoding="utf-8")
    now = utc_now_iso()
    with get_db() as conn:
        for ticker in ["AAA", "BBB", "CCC", "DDD"]:
            conn.execute(
                """
                INSERT INTO companies(ticker, cik, name, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (ticker, ticker, f"{ticker} Inc", now),
            )
            conn.execute(
                """
                INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
                VALUES(?, '2026-02-01', ?, '{}', ?)
                """,
                (ticker, json.dumps({"revenue": 200.0}), now),
            )

    monkeypatch.setattr(
        "app.sector.peer_set._infer_filings_category",
        lambda conn, *, ticker, as_of_date, sec_client, sec_cache: ("software", "keywords"),
    )
    payload = select_sector_peers(
        sector="Software", as_of_date="2026-02-13", limit=4, mode="hybrid"
    )
    summary = payload["peer_selection_summary"]
    assert payload["selected_tickers"] == ["AAA", "BBB", "CCC", "DDD"]
    assert summary["taxonomy_mapped_count"] == 2
    assert summary["taxonomy_sparse_threshold"] == 3
    assert summary["taxonomy_sparse_fallback_triggered"] is True
    assert summary["filings_inferred_hits"] == 2


def test_sector_peer_selection_suppression_defaults_off(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_SECTOR_SUPPRESS_FINANCIALS", "false")
    monkeypatch.setenv("VOE_SECTOR_SUPPRESS_PREREVENUE", "false")
    _, taxonomy_path = _init_temp_db(monkeypatch, tmp_path)
    taxonomy_path.write_text("ticker,sector\nAAA,Software\n", encoding="utf-8")
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('AAA', '1', 'AAA Bank Holdings', ?)
            """,
            (now,),
        )

    monkeypatch.setattr(
        "app.sector.peer_set.preflight_annual_eligibility",
        lambda **kwargs: {"ticker": kwargs["ticker"], "eligible": True, "skip_reason": None},
    )
    payload = select_sector_peers(
        sector="Software", as_of_date="2026-02-13", limit=1, mode="taxonomy"
    )
    assert payload["selected_tickers"] == ["AAA"]
    assert payload["peer_selection_summary"]["dropped_by_suppression"] == 0


def test_sector_peer_selection_suppression_enabled_via_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_SECTOR_SUPPRESS_FINANCIALS", "true")
    _, taxonomy_path = _init_temp_db(monkeypatch, tmp_path)
    taxonomy_path.write_text("ticker,sector\nAAA,Software\n", encoding="utf-8")
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('AAA', '1', 'AAA Bank Holdings', ?)
            """,
            (now,),
        )

    monkeypatch.setattr(
        "app.sector.peer_set.preflight_annual_eligibility",
        lambda **kwargs: {"ticker": kwargs["ticker"], "eligible": True, "skip_reason": None},
    )
    payload = select_sector_peers(
        sector="Software", as_of_date="2026-02-13", limit=1, mode="taxonomy"
    )
    assert payload["selected_tickers"] == []
    assert payload["peer_selection_summary"]["dropped_by_suppression"] == 1


def test_sector_peer_selection_filters_preflight_ineligible(monkeypatch, tmp_path):
    _, taxonomy_path = _init_temp_db(monkeypatch, tmp_path)
    taxonomy_path.write_text("ticker,sector\nAAA,Software\nBBB,Software\n", encoding="utf-8")
    now = utc_now_iso()
    with get_db() as conn:
        for ticker, cik in [("AAA", "1"), ("BBB", "2")]:
            conn.execute(
                """
                INSERT INTO companies(ticker, cik, name, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (ticker, cik, f"{ticker} Inc", now),
            )

    monkeypatch.setattr(
        "app.sector.peer_set.load_ticker_cik_map",
        lambda refresh_if_missing=False: {"AAA": "1", "BBB": "2"},
    )
    monkeypatch.setattr(
        "app.sector.peer_set.preflight_annual_eligibility",
        lambda **kwargs: {
            "ticker": kwargs["ticker"],
            "eligible": kwargs["ticker"] == "AAA",
            "skip_reason": None if kwargs["ticker"] == "AAA" else "NO_ANNUAL_FILING_IN_WINDOW",
        },
    )
    monkeypatch.setattr(
        "app.sector.peer_set._candidate_row",
        lambda conn, *, ticker, source, sector_value, inferred_category, inferred_source, as_of_date, cap_min, cap_max, provider, run_id, cfg: {
            "ticker": ticker,
            "peer_source": source,
            "sector": sector_value,
            "inferred_category": inferred_category,
            "inferred_category_source": inferred_source,
            "company_name": f"{ticker} Inc",
            "market_cap": 100.0,
            "market_cap_status": "OK",
            "market_cap_in_band": True,
            "price": 1.0,
            "price_provider": "mock",
            "shares_outstanding": 100.0,
            "shares_citation": None,
            "suppression_reasons": [],
            "excluded_reason": None,
            "selected": True,
        },
    )

    payload = select_sector_peers(
        sector="Software",
        as_of_date="2026-02-13",
        years_back=10,
        limit=5,
        mode="taxonomy",
    )
    assert payload["selected_tickers"] == ["AAA"]
    summary = payload["peer_selection_summary"]
    assert summary["preflight_checked"] >= 2
    assert summary["preflight_ineligible"] >= 1


def test_sector_peer_selection_sic_expansion_hits_min_peers(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_SECTOR_TAXONOMY_SPARSE_THRESHOLD", "20")
    _, taxonomy_path = _init_temp_db(monkeypatch, tmp_path)
    taxonomy_path.write_text("ticker,sector\nAAA,Software\n", encoding="utf-8")
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('AAA', '1', 'AAA Inc', ?)
            """,
            (now,),
        )
        conn.execute(
            """
            INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
            VALUES('AAA', '2026-02-01', ?, '{}', ?)
            """,
            (json.dumps({"revenue": 100.0}), now),
        )

    monkeypatch.setattr(
        "app.sector.peer_set.load_ticker_cik_map",
        lambda refresh_if_missing=False: {
            "AAA": "1",
            "BBB": "2",
            "CCC": "3",
            "DDD": "4",
            "EEE": "5",
        },
    )
    monkeypatch.setattr(
        "app.sector.peer_set._resolve_ticker_sic",
        lambda conn, *, ticker, cik_by_ticker, as_of_date, sec_client, sec_cache, allow_sec_network=True: (
            3571 if ticker in {"AAA", "BBB", "CCC", "DDD"} else 2834,
            "sec_submissions",
        ),
    )
    monkeypatch.setattr(
        "app.sector.peer_set._infer_filings_category",
        lambda conn, *, ticker, as_of_date, sec_client, sec_cache: (None, "unknown"),
    )
    monkeypatch.setattr(
        "app.sector.peer_set.preflight_annual_eligibility",
        lambda **kwargs: {"ticker": kwargs["ticker"], "eligible": True, "skip_reason": None},
    )

    payload = select_sector_peers(
        sector="Software",
        as_of_date="2026-02-13",
        limit=8,
        min_peers=4,
        max_peers=8,
        mode="hybrid",
        sic_expand=True,
        sic_family=False,
    )
    assert len(payload["selected_tickers"]) >= 4
    assert payload["selected_tickers"][:4] == ["AAA", "BBB", "CCC", "DDD"]
    summary = payload["peer_selection_summary"]
    assert summary["sic_hits"] >= 3
    assert summary["final_selected_count"] >= 4
    assert "sic_expand" in summary["fallback_steps_taken"]


def test_sector_rlm_peer_expansion_to_min_dossierable(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_SECTOR_TAXONOMY_SPARSE_THRESHOLD", "20")
    _, taxonomy_path = _init_temp_db(monkeypatch, tmp_path)
    taxonomy_path.write_text("ticker,sector\nAAA,Software\n", encoding="utf-8")
    now = utc_now_iso()
    with get_db() as conn:
        for ticker, cik in [("AAA", "1"), ("BBB", "2"), ("CCC", "3"), ("DDD", "4"), ("EEE", "5")]:
            conn.execute(
                """
                INSERT INTO companies(ticker, cik, name, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (ticker, cik, f"{ticker} Inc", now),
            )
            conn.execute(
                """
                INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
                VALUES(?, '2026-02-01', ?, '{}', ?)
                """,
                (ticker, json.dumps({"revenue": 100.0}), now),
            )

    monkeypatch.setattr(
        "app.sector.peer_set.load_ticker_cik_map",
        lambda refresh_if_missing=False: {
            "AAA": "1",
            "BBB": "2",
            "CCC": "3",
            "DDD": "4",
            "EEE": "5",
        },
    )
    monkeypatch.setattr(
        "app.sector.peer_set._resolve_ticker_sic",
        lambda conn, *, ticker, cik_by_ticker, as_of_date, sec_client, sec_cache, allow_sec_network=True: (
            3571 if ticker in {"AAA", "BBB", "CCC"} else 2834,
            "sec_submissions",
        ),
    )
    monkeypatch.setattr(
        "app.sector.peer_set._infer_filings_category",
        lambda conn, *, ticker, as_of_date, sec_client, sec_cache: ("software", "keywords"),
    )
    monkeypatch.setattr(
        "app.sector.peer_set.preflight_annual_eligibility",
        lambda **kwargs: {
            "ticker": kwargs["ticker"],
            "eligible": kwargs["ticker"] in {"AAA", "BBB", "CCC", "DDD"},
            "skip_reason": None
            if kwargs["ticker"] in {"AAA", "BBB", "CCC", "DDD"}
            else "NO_ANNUAL_FILING_IN_WINDOW",
            "annual_forms_found": {"10-K": 3},
        },
    )

    payload = select_sector_peers(
        sector="Software",
        as_of_date="2026-02-13",
        years_back=10,
        limit=8,
        min_peers=3,
        max_peers=8,
        max_peer_scan=50,
        stop_when_min_reached=True,
        mode="hybrid",
        min_annual_filings=2,
        sic_expand=True,
        sic_family=True,
    )
    assert payload["selected_tickers"][:3] == ["AAA", "BBB", "CCC"]
    summary = payload["peer_selection_summary"]
    assert summary["dossierable_target_reached"] is True
    assert summary["expansion_stopped_at_target"] is True


def test_sector_peer_selection_sic_expansion_increases_count_deterministically(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VOE_SECTOR_TAXONOMY_SPARSE_THRESHOLD", "20")
    _, taxonomy_path = _init_temp_db(monkeypatch, tmp_path)
    taxonomy_path.write_text("ticker,sector\nAAA,Software\n", encoding="utf-8")
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('AAA', '1', 'AAA Inc', ?)
            """,
            (now,),
        )
        conn.execute(
            """
            INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
            VALUES('AAA', '2026-02-01', ?, '{}', ?)
            """,
            (json.dumps({"revenue": 100.0}), now),
        )

    monkeypatch.setattr(
        "app.sector.peer_set.load_ticker_cik_map",
        lambda refresh_if_missing=False: {"AAA": "1", "BBB": "2", "CCC": "3", "DDD": "4"},
    )
    monkeypatch.setattr(
        "app.sector.peer_set._resolve_ticker_sic",
        lambda conn, *, ticker, cik_by_ticker, as_of_date, sec_client, sec_cache, allow_sec_network=True: (
            3571 if ticker in {"AAA", "BBB", "CCC", "DDD"} else None,
            "sec_submissions",
        ),
    )
    monkeypatch.setattr(
        "app.sector.peer_set._infer_filings_category",
        lambda conn, *, ticker, as_of_date, sec_client, sec_cache: (None, "unknown"),
    )
    monkeypatch.setattr(
        "app.sector.peer_set.preflight_annual_eligibility",
        lambda **kwargs: {"ticker": kwargs["ticker"], "eligible": True, "skip_reason": None},
    )
    without_sic = select_sector_peers(
        sector="Software",
        as_of_date="2026-02-13",
        limit=6,
        min_peers=4,
        max_peers=6,
        mode="hybrid",
        sic_expand=False,
        sic_family=False,
    )
    with_sic = select_sector_peers(
        sector="Software",
        as_of_date="2026-02-13",
        limit=6,
        min_peers=4,
        max_peers=6,
        mode="hybrid",
        sic_expand=True,
        sic_family=False,
    )
    assert with_sic["selected_tickers"] == ["AAA", "BBB", "CCC", "DDD"]
    assert len(with_sic["selected_tickers"]) > len(without_sic["selected_tickers"])
    assert with_sic["peer_selection_summary"]["sic_hits"] >= 3


def test_sector_peer_selection_stops_when_min_target_reached(monkeypatch, tmp_path):
    _, taxonomy_path = _init_temp_db(monkeypatch, tmp_path)
    taxonomy_path.write_text("ticker,sector\nAAA,Software\n", encoding="utf-8")
    now = utc_now_iso()
    with get_db() as conn:
        for ticker, cik in [("AAA", "1"), ("BBB", "2"), ("CCC", "3"), ("DDD", "4"), ("EEE", "5")]:
            conn.execute(
                """
                INSERT INTO companies(ticker, cik, name, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (ticker, cik, f"{ticker} Inc", now),
            )

    monkeypatch.setattr(
        "app.sector.peer_set.load_ticker_cik_map",
        lambda refresh_if_missing=False: {
            "AAA": "1",
            "BBB": "2",
            "CCC": "3",
            "DDD": "4",
            "EEE": "5",
        },
    )
    monkeypatch.setattr(
        "app.sector.peer_set._infer_filings_category",
        lambda conn, *, ticker, as_of_date, sec_client, sec_cache: ("software", "keywords"),
    )
    preflight_calls: list[str] = []

    def _preflight(**kwargs):
        preflight_calls.append(kwargs["ticker"])
        return {
            "ticker": kwargs["ticker"],
            "eligible": True,
            "skip_reason": None,
            "annual_forms_found": {"10-K": 5},
        }

    monkeypatch.setattr("app.sector.peer_set.preflight_annual_eligibility", _preflight)
    monkeypatch.setattr(
        "app.sector.peer_set._candidate_row",
        lambda conn, *, ticker, source, sector_value, inferred_category, inferred_source, as_of_date, cap_min, cap_max, provider, run_id, cfg: {
            "ticker": ticker,
            "peer_source": source,
            "stage_source": source,
            "sector": sector_value,
            "inferred_category": inferred_category,
            "inferred_category_source": inferred_source,
            "company_name": f"{ticker} Inc",
            "market_cap": 100.0,
            "market_cap_status": "OK",
            "market_cap_in_band": True,
            "price": 1.0,
            "price_provider": "mock",
            "shares_outstanding": 100.0,
            "shares_citation": None,
            "suppression_reasons": [],
            "excluded_reason": None,
            "selected": True,
        },
    )

    payload = select_sector_peers(
        sector="Software",
        as_of_date="2026-02-13",
        years_back=10,
        limit=10,
        min_peers=2,
        max_peers=10,
        max_peer_scan=50,
        mode="filings",
        stop_when_min_reached=True,
    )
    assert payload["selected_tickers"] == ["AAA", "BBB"]
    assert len(preflight_calls) == 2
    summary = payload["peer_selection_summary"]
    assert summary["peer_scan_count"] == 2
    assert summary["dossierable_target_reached"] is True
    assert summary["expansion_stopped_at_target"] is True


def test_time_series_shape_and_derived_trace():
    items = []
    for i, year in enumerate(range(2016, 2026), start=1):
        rev = float(100 + 10 * i)
        gp = rev * 0.5
        op = rev * 0.2
        fcf = rev * 0.12
        items.extend(
            [
                {
                    "year": year,
                    "metric": "revenue",
                    "value": rev,
                    "citations": [{"source_url": "https://www.sec.gov/r", "snippet": "rev"}],
                    "derived_from": [],
                },
                {
                    "year": year,
                    "metric": "gross_profit",
                    "value": gp,
                    "citations": [{"source_url": "https://www.sec.gov/gp", "snippet": "gp"}],
                    "derived_from": [],
                },
                {
                    "year": year,
                    "metric": "operating_income",
                    "value": op,
                    "citations": [{"source_url": "https://www.sec.gov/op", "snippet": "op"}],
                    "derived_from": [],
                },
                {
                    "year": year,
                    "metric": "net_income",
                    "value": op * 0.8,
                    "citations": [{"source_url": "https://www.sec.gov/ni", "snippet": "ni"}],
                    "derived_from": [],
                },
                {
                    "year": year,
                    "metric": "cfo",
                    "value": rev * 0.18,
                    "citations": [{"source_url": "https://www.sec.gov/cfo", "snippet": "cfo"}],
                    "derived_from": [],
                },
                {
                    "year": year,
                    "metric": "capex",
                    "value": rev * 0.06,
                    "citations": [{"source_url": "https://www.sec.gov/capex", "snippet": "capex"}],
                    "derived_from": [],
                },
                {
                    "year": year,
                    "metric": "fcf",
                    "value": fcf,
                    "citations": [{"source_url": "https://www.sec.gov/fcf", "snippet": "fcf"}],
                    "derived_from": [],
                },
                {
                    "year": year,
                    "metric": "shares_outstanding",
                    "value": float(1000 + i * 10),
                    "citations": [{"source_url": "https://www.sec.gov/sh", "snippet": "sh"}],
                    "derived_from": [],
                },
                {
                    "year": year,
                    "metric": "net_debt",
                    "value": float(200 - i * 5),
                    "citations": [{"source_url": "https://www.sec.gov/nd", "snippet": "nd"}],
                    "derived_from": [],
                },
            ]
        )
    ts = build_time_series(items)
    assert len(ts["standardized_rows"]) == 10
    assert ts["required_fields"] == [
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "cfo",
        "capex",
        "fcf",
        "shares_outstanding",
        "shares_yoy_change",
        "net_debt",
    ]
    for row in ts["standardized_rows"]:
        assert all(field in row for field in ts["required_fields"])
    signal_map = {row["signal"]: row for row in ts["derived_signals"]}
    assert signal_map["revenue_cagr_3y"]["derived_from"]
    assert signal_map["revenue_cagr_5y"]["derived_from"]
    assert signal_map["revenue_cagr_10y"]["derived_from"]
    assert signal_map["roic_proxy"]["signal"] == "roic_proxy"


def test_peer_report_outputs_rankings(monkeypatch, tmp_path):
    cfg, _ = _init_temp_db(monkeypatch, tmp_path)
    dossiers = [
        {
            "ticker": "AAA",
            "time_series": {
                "derived_signals": [
                    {"signal": "revenue_cagr_5y", "value": 0.15},
                    {"signal": "revenue_cagr_10y", "value": 0.12},
                    {"signal": "operating_margin_trend_slope", "value": 0.01},
                    {"signal": "fcf_margin_trend_slope", "value": 0.01},
                    {"signal": "gross_margin_trend_slope", "value": 0.005},
                    {"signal": "dilution_rate_shares_cagr", "value": 0.01},
                    {"signal": "risk_factor_keyword_delta", "value": 1.0},
                    {"signal": "roic_proxy", "value": 0.12},
                ],
                "standardized_rows": [
                    {"year": 2025, "net_debt": 100.0, "cfo": 50.0, "net_income": 40.0}
                ],
                "standardized_row_traces": {
                    "2025": {
                        "net_debt": {"citations": [], "derived_from": ["a.net_debt"]},
                        "cfo": {"citations": [], "derived_from": ["a.cfo"]},
                        "net_income": {"citations": [], "derived_from": ["a.net_income"]},
                    }
                },
            },
            "claims": [
                {
                    "citations": [{"source_url": "https://www.sec.gov/aaa", "snippet": "aaa"}],
                }
            ],
            "items": [],
        },
        {
            "ticker": "BBB",
            "time_series": {
                "derived_signals": [
                    {"signal": "revenue_cagr_5y", "value": 0.08},
                    {"signal": "revenue_cagr_10y", "value": 0.05},
                    {"signal": "operating_margin_trend_slope", "value": 0.003},
                    {"signal": "fcf_margin_trend_slope", "value": 0.001},
                    {"signal": "gross_margin_trend_slope", "value": 0.002},
                    {"signal": "dilution_rate_shares_cagr", "value": 0.03},
                    {"signal": "risk_factor_keyword_delta", "value": 2.0},
                    {"signal": "roic_proxy", "value": 0.04},
                ],
                "standardized_rows": [
                    {"year": 2025, "net_debt": 300.0, "cfo": 25.0, "net_income": 30.0}
                ],
                "standardized_row_traces": {
                    "2025": {
                        "net_debt": {"citations": [], "derived_from": ["b.net_debt"]},
                        "cfo": {"citations": [], "derived_from": ["b.cfo"]},
                        "net_income": {"citations": [], "derived_from": ["b.net_income"]},
                    }
                },
            },
            "claims": [
                {
                    "citations": [{"source_url": "https://www.sec.gov/bbb", "snippet": "bbb"}],
                }
            ],
            "items": [],
        },
    ]
    summary = build_peer_report(
        run_id="sector_peer_report_test", as_of_date="2026-02-13", dossiers=dossiers
    )
    md_path = Path(summary["peer_report_path"])
    json_path = Path(summary["peer_rankings_path"])
    scoreboard_path = Path(summary["peer_scoreboard_path"])
    assert md_path.exists()
    assert json_path.exists()
    assert scoreboard_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert "AAA" in text
    assert "BBB" in text
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "future_whale_rank" in payload
    assert "whale_signature_rank" in payload
    assert "quality_rank" in payload
    assert "valuation_rank" in payload
    assert "risk_rank" in payload
    assert payload["tickers"] == ["AAA", "BBB"]
    assert "Whale Signature Leaders" in text
    assert "Winner vs Peer Deltas" in text
    assert (cfg.dossiers_dir / "sector_peer_report_test" / "peer_report.md").exists()
    scoreboard = json.loads(scoreboard_path.read_text(encoding="utf-8"))
    assert scoreboard["rows"]
    required_metric_keys = {
        "revenue_cagr_10y",
        "operating_margin_trend_slope",
        "fcf_margin_trend_slope",
        "dilution_rate_shares_cagr",
        "roic_proxy",
        "cash_conversion_proxy",
        "net_debt_proxy",
        "whale_signature_score",
    }
    row_metric_keys = set((scoreboard["rows"][0].get("metric_values") or {}).keys())
    assert required_metric_keys.issubset(row_metric_keys)


def test_sector_synthesis_disabled_provider(monkeypatch, tmp_path):
    cfg, taxonomy_path = _init_temp_db(monkeypatch, tmp_path)
    taxonomy_path.write_text("ticker,sector\nAAA,Software\nBBB,Software\n", encoding="utf-8")
    run_id = "sector_synth_test"
    _seed_sector_synthesis_inputs(cfg, run_id, ["AAA", "BBB"])
    _seed_analysis_report(
        cfg,
        ticker="AAA",
        as_of_date="2026-02-13",
        thesis_summary="AAA has a durable moat.",
        positives=["AAA positive"],
        risks=["AAA risk"],
        recent_events=["AAA event"],
        open_questions=["AAA question"],
        falsifiers=["AAA falsifier"],
    )
    _seed_analysis_report(
        cfg,
        ticker="BBB",
        as_of_date="2026-02-13",
        thesis_summary="BBB has improving economics.",
        positives=["BBB positive"],
        risks=["BBB risk"],
        recent_events=["BBB event"],
        open_questions=["BBB question"],
        falsifiers=["BBB falsifier"],
    )

    summary = run_sector_synthesis(sector="Software", as_of_date="2026-02-13", run_id=run_id)
    path = Path(summary["path"])
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    packet = validate_sector_synthesis_packet(payload)
    assert packet.top_pick in packet.peer_tickers
    assert packet.llm_meta.model == "disabled"


def test_sector_synthesis_prefers_analysis_report_inputs(monkeypatch, tmp_path):
    cfg, taxonomy_path = _init_temp_db(monkeypatch, tmp_path)
    taxonomy_path.write_text("ticker,sector\nAAA,Software\nBBB,Software\n", encoding="utf-8")
    run_id = "sector_synth_analysis"
    _seed_sector_synthesis_inputs(cfg, run_id, ["AAA", "BBB"])
    _seed_analysis_report(
        cfg,
        ticker="AAA",
        as_of_date="2026-02-13",
        thesis_summary="AAA thesis from analysis report",
        positives=["AAA positive from analysis"],
        risks=["AAA risk from analysis"],
        recent_events=["AAA event from analysis"],
        open_questions=["AAA question from analysis"],
        falsifiers=["AAA falsifier from analysis"],
    )

    provider = _CaptureSectorProvider()
    monkeypatch.setattr("app.sector.synthesis.get_llm_provider", lambda: provider)

    summary = run_sector_synthesis(sector="Software", as_of_date="2026-02-13", run_id=run_id)
    assert Path(summary["path"]).exists()
    assert provider.inputs is not None

    ticker_analysis = provider.inputs["ticker_analysis"]
    assert ticker_analysis["AAA"]["verdict"] == "BUY"
    assert ticker_analysis["AAA"]["confidence_label"] == "HIGH"
    assert ticker_analysis["AAA"]["thesis_summary"] == "AAA thesis from analysis report"
    assert ticker_analysis["AAA"]["positives"] == ["AAA positive from analysis"]
    assert ticker_analysis["AAA"]["risks"] == ["AAA risk from analysis"]
    assert ticker_analysis["AAA"]["recent_event_impacts"] == ["AAA event from analysis"]
    assert ticker_analysis["AAA"]["open_questions"] == ["AAA question from analysis"]
    assert ticker_analysis["AAA"]["falsifiers"] == ["AAA falsifier from analysis"]
    assert ticker_analysis["AAA"]["warnings"] == []


def test_sector_synthesis_falls_back_to_legacy_research_packet(monkeypatch, tmp_path):
    cfg, taxonomy_path = _init_temp_db(monkeypatch, tmp_path)
    taxonomy_path.write_text("ticker,sector\nAAA,Software\nBBB,Software\n", encoding="utf-8")
    run_id = "sector_synth_fallback"
    _seed_sector_synthesis_inputs(cfg, run_id, ["AAA", "BBB"])
    _seed_legacy_research_packet(
        cfg,
        ticker="AAA",
        as_of_date="2026-02-13",
        run_id=run_id,
        findings=["AAA finding from legacy"],
        risks=["AAA risk from legacy"],
        catalysts=["AAA catalyst from legacy"],
        disconfirming=["AAA disconfirming from legacy"],
        evidence_gaps=["AAA gap from legacy"],
    )

    provider = _CaptureSectorProvider()
    monkeypatch.setattr("app.sector.synthesis.get_llm_provider", lambda: provider)

    summary = run_sector_synthesis(sector="Software", as_of_date="2026-02-13", run_id=run_id)
    assert Path(summary["path"]).exists()
    assert provider.inputs is not None

    ticker_analysis = provider.inputs["ticker_analysis"]
    assert ticker_analysis["AAA"]["verdict"] == ""
    assert ticker_analysis["AAA"]["thesis_summary"] == ""
    assert ticker_analysis["AAA"]["positives"] == ["AAA finding from legacy"]
    assert ticker_analysis["AAA"]["risks"] == ["AAA risk from legacy"]
    assert ticker_analysis["AAA"]["recent_event_impacts"] == ["AAA catalyst from legacy"]
    assert ticker_analysis["AAA"]["open_questions"] == ["AAA gap from legacy"]
    assert ticker_analysis["AAA"]["falsifiers"] == ["AAA disconfirming from legacy"]
    assert ticker_analysis["AAA"]["warnings"] == ["legacy_research_packet_fallback"]


def test_sector_synthesis_prefers_analysis_report_over_legacy_fallback(monkeypatch, tmp_path):
    cfg, taxonomy_path = _init_temp_db(monkeypatch, tmp_path)
    taxonomy_path.write_text("ticker,sector\nAAA,Software\nBBB,Software\n", encoding="utf-8")
    run_id = "sector_synth_precedence"
    _seed_sector_synthesis_inputs(cfg, run_id, ["AAA", "BBB"])
    _seed_analysis_report(
        cfg,
        ticker="AAA",
        as_of_date="2026-02-13",
        thesis_summary="AAA thesis from analysis report",
        positives=["AAA positive from analysis"],
        risks=["AAA risk from analysis"],
        recent_events=["AAA event from analysis"],
        open_questions=["AAA question from analysis"],
        falsifiers=["AAA falsifier from analysis"],
        warnings=["analysis_warning"],
    )
    _seed_legacy_research_packet(
        cfg,
        ticker="AAA",
        as_of_date="2026-02-13",
        run_id=run_id,
        findings=["AAA finding from legacy"],
        risks=["AAA risk from legacy"],
        catalysts=["AAA catalyst from legacy"],
        disconfirming=["AAA disconfirming from legacy"],
        evidence_gaps=["AAA gap from legacy"],
    )

    provider = _CaptureSectorProvider()
    monkeypatch.setattr("app.sector.synthesis.get_llm_provider", lambda: provider)

    summary = run_sector_synthesis(sector="Software", as_of_date="2026-02-13", run_id=run_id)
    assert Path(summary["path"]).exists()
    assert provider.inputs is not None

    ticker_analysis = provider.inputs["ticker_analysis"]
    assert ticker_analysis["AAA"]["thesis_summary"] == "AAA thesis from analysis report"
    assert ticker_analysis["AAA"]["positives"] == ["AAA positive from analysis"]
    assert ticker_analysis["AAA"]["warnings"] == ["analysis_warning"]


def test_sector_cycle_emits_peer_selection_summary(monkeypatch, tmp_path):
    cfg, _ = _init_temp_db(monkeypatch, tmp_path)
    run_id = "sector_cycle_summary_test"
    dossier_dir = cfg.dossiers_dir / run_id
    dossier_dir.mkdir(parents=True, exist_ok=True)
    peer_report_path = dossier_dir / "peer_report.md"
    peer_rankings_path = dossier_dir / "peer_rankings.json"
    peer_report_path.write_text("# peer", encoding="utf-8")
    peer_rankings_path.write_text(json.dumps({"rankings": []}), encoding="utf-8")

    monkeypatch.setattr(
        "app.sector.cycle.select_sector_peers",
        lambda **kwargs: {
            "sector": kwargs["sector"],
            "as_of_date": kwargs["as_of_date"],
            "selected_tickers": ["AAA"],
            "rows": [],
            "all_rows": [
                {
                    "ticker": "ZZZ",
                    "suppression_reasons": ["SUPPRESSION_FINANCIALS"],
                }
            ],
            "counts": {"selected": 1},
            "peer_selection_summary": {
                "mode": kwargs.get("mode", "hybrid"),
                "anchor_category": "software",
                "taxonomy_hits": 1,
                "filings_inferred_hits": 0,
                "taxonomy_mapped_count": 1,
                "taxonomy_sparse_threshold": 20,
                "taxonomy_sparse_fallback_triggered": False,
                "missing_category": 0,
                "dropped_by_capband": 0,
                "dropped_by_suppression": 1,
            },
        },
    )
    monkeypatch.setattr(
        "app.sector.cycle.run_dossier_for_peer_set", lambda **kwargs: {"status": "DONE"}
    )
    monkeypatch.setattr(
        "app.sector.cycle.run_whale_signals_for_run",
        lambda **kwargs: {
            "summary_path": str(dossier_dir / "whale_signals_summary.json"),
            "rows": [{"ticker": "AAA"}],
        },
    )
    (dossier_dir / "whale_signals_summary.json").write_text(
        json.dumps({"rows": [{"ticker": "AAA"}]}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "app.sector.cycle.build_peer_report_from_run",
        lambda **kwargs: {
            "peer_report_path": str(peer_report_path),
            "peer_rankings_path": str(peer_rankings_path),
            "rankings": [],
        },
    )
    monkeypatch.setattr(
        "app.sector.cycle.build_sector_decision_pack",
        lambda **kwargs: {
            "decision_pack_path": str(cfg.sectors_dir / run_id / "decision_pack.json"),
            "decision_pack_md_path": str(cfg.sectors_dir / run_id / "decision_pack.md"),
            "top_candidates": ["AAA"],
        },
    )
    monkeypatch.setattr("app.sector.cycle.open_dossier_run", lambda run_id: {"status": "DONE"})

    summary = run_sector_cycle(
        sector="Software",
        as_of_date="2026-02-13",
        run_id=run_id,
        peer_limit=1,
        with_research=False,
        with_synthesis=False,
        peer_mode="hybrid",
    )
    summary_path = cfg.sectors_dir / run_id / "peer_selection_summary.json"
    assert summary["status"] == "DONE"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["taxonomy_hits"] == 1
    sector_report = (cfg.sectors_dir / run_id / "peer_report.md").read_text(encoding="utf-8")
    assert "Suppressed Tickers" in sector_report
    assert "SUPPRESSION_FINANCIALS" in sector_report


def test_sector_cycle_sec_budget_reflected_in_summary(monkeypatch, tmp_path):
    cfg, _ = _init_temp_db(monkeypatch, tmp_path)
    run_id = "sector_cycle_budget_summary_test"
    dossier_dir = cfg.dossiers_dir / run_id
    dossier_dir.mkdir(parents=True, exist_ok=True)
    peer_report_path = dossier_dir / "peer_report.md"
    peer_rankings_path = dossier_dir / "peer_rankings.json"
    peer_report_path.write_text("# peer", encoding="utf-8")
    peer_rankings_path.write_text(json.dumps({"rankings": []}), encoding="utf-8")

    monkeypatch.setattr(
        "app.sector.cycle.select_sector_peers",
        lambda **kwargs: {
            "sector": kwargs["sector"],
            "as_of_date": kwargs["as_of_date"],
            "selected_tickers": ["AAA"],
            "rows": [],
            "all_rows": [],
            "counts": {"selected": 1},
            "peer_selection_summary": {"taxonomy_hits": 1},
        },
    )
    monkeypatch.setattr(
        "app.sector.cycle.run_dossier_for_peer_set",
        lambda **kwargs: {
            "status": "DONE",
            "sec_budget": {
                "requested": kwargs.get("sec_budget"),
                "effective": {"sec.gov": 1500, "data.sec.gov": 1500, "www.sec.gov": 1500},
            },
        },
    )
    monkeypatch.setattr(
        "app.sector.cycle.run_whale_signals_for_run",
        lambda **kwargs: {
            "summary_path": str(dossier_dir / "whale_signals_summary.json"),
            "rows": [],
        },
    )
    (dossier_dir / "whale_signals_summary.json").write_text(
        json.dumps({"rows": []}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "app.sector.cycle.build_peer_report_from_run",
        lambda **kwargs: {
            "peer_report_path": str(peer_report_path),
            "peer_rankings_path": str(peer_rankings_path),
            "peer_scoreboard_path": str(dossier_dir / "peer_scoreboard.json"),
            "rankings": [],
        },
    )
    (dossier_dir / "peer_scoreboard.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    monkeypatch.setattr("app.sector.cycle.build_sector_decision_pack", lambda **kwargs: {})
    monkeypatch.setattr("app.sector.cycle.open_dossier_run", lambda run_id: {"status": "DONE"})

    summary = run_sector_cycle(
        sector="Software",
        as_of_date="2026-02-13",
        run_id=run_id,
        peer_limit=1,
        limit_dossiers=1,
        with_research=False,
        with_synthesis=False,
        sec_budget=1500,
    )
    assert summary["sec_budget_requested"] == 1500
    assert summary["sec_budget_effective"]["data.sec.gov"] == 1500


def test_sector_run_status_reports_dossier_status_buckets(monkeypatch, tmp_path):
    cfg, _ = _init_temp_db(monkeypatch, tmp_path)
    run_id = "sector_status_buckets_test"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sector_summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "PARTIAL",
                "sector": "Software",
                "as_of_date": "2026-02-13",
                "peer_tickers": ["AAA", "BBB", "CCC", "DDD"],
                "dossier_tickers": ["AAA", "BBB", "CCC", "DDD"],
                "dossier": {
                    "ticker_results": {
                        "AAA": {"status": "OK"},
                        "BBB": {"status": "FAILED"},
                        "CCC": {"status": "SKIPPED"},
                        "DDD": {"status": "SKIPPED_BUDGET"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    payload = sector_run_status(run_id=run_id)
    assert payload["dossier_status_counts"]["OK"] == 1
    assert payload["dossier_status_counts"]["FAILED"] == 1
    assert payload["dossier_status_counts"]["SKIPPED"] == 1
    assert payload["dossier_status_counts"]["SKIPPED_BUDGET"] == 1
    assert payload["dossierable_count"] == 4
    assert payload["peer_quality_report_path"] is None


def test_sector_cycle_writes_peer_quality_report_with_shortfall(monkeypatch, tmp_path):
    cfg, _ = _init_temp_db(monkeypatch, tmp_path)
    run_id = "sector_peer_quality_shortfall_test"
    dossier_dir = cfg.dossiers_dir / run_id
    dossier_dir.mkdir(parents=True, exist_ok=True)
    peer_report_path = dossier_dir / "peer_report.md"
    peer_rankings_path = dossier_dir / "peer_rankings.json"
    peer_scoreboard_path = dossier_dir / "peer_scoreboard.json"
    peer_report_path.write_text("# peer", encoding="utf-8")
    peer_rankings_path.write_text(json.dumps({"rankings": []}), encoding="utf-8")
    peer_scoreboard_path.write_text(json.dumps({"rows": []}), encoding="utf-8")
    whale_summary_path = dossier_dir / "whale_signals_summary.json"
    whale_summary_path.write_text(json.dumps({"rows": []}), encoding="utf-8")

    monkeypatch.setattr(
        "app.sector.cycle.select_sector_peers",
        lambda **kwargs: {
            "sector": kwargs["sector"],
            "as_of_date": kwargs["as_of_date"],
            "selected_tickers": ["AAA"],
            "rows": [
                {
                    "ticker": "AAA",
                    "peer_source": "taxonomy",
                    "stage_source": "taxonomy",
                    "cik_present": True,
                    "annual_forms_found": {"10-K": 5},
                    "eligible": True,
                    "selected": True,
                }
            ],
            "all_rows": [
                {"ticker": "AAA", "selected": True},
                {
                    "ticker": "NOCIK",
                    "stage_source": "seed_fallback",
                    "excluded_reason": "MISSING_CIK",
                    "selected": False,
                },
                {
                    "ticker": "OTCXF",
                    "stage_source": "sic_expand",
                    "excluded_reason": "OTC_EXCLUDED",
                    "selected": False,
                },
                {
                    "ticker": "FRGN",
                    "stage_source": "filings_inferred",
                    "excluded_reason": "PRECHECK_FOREIGN_EXCLUDED",
                    "selected": False,
                },
                {
                    "ticker": "NOANN",
                    "stage_source": "filings_inferred",
                    "excluded_reason": "PRECHECK_NO_ANNUAL_FILING_IN_WINDOW",
                    "selected": False,
                },
                {
                    "ticker": "LOWANN",
                    "stage_source": "filings_inferred",
                    "excluded_reason": "PRECHECK_INSUFFICIENT_ANNUAL_FILINGS_IN_WINDOW",
                    "selected": False,
                },
                {
                    "ticker": "BUDG",
                    "stage_source": "seed_fallback",
                    "excluded_reason": "PRECHECK_BUDGET_EXCEEDED",
                    "selected": False,
                },
            ],
            "counts": {"selected": 1},
            "peer_selection_summary": {
                "mode": "hybrid",
                "stage_contribution_counts": {
                    "taxonomy": 1,
                    "sic_expand": 0,
                    "sic_family": 0,
                    "filings_inferred": 0,
                    "seed_fallback": 0,
                },
                "peer_scan_count": 7,
                "max_peer_scan": 400,
                "scan_exhausted": False,
            },
        },
    )
    monkeypatch.setattr(
        "app.sector.cycle.run_dossier_for_peer_set",
        lambda **kwargs: {
            "status": "DONE",
            "sec_budget": {
                "requested": kwargs.get("sec_budget"),
                "effective": {"data.sec.gov": 1500, "www.sec.gov": 1500},
            },
        },
    )
    monkeypatch.setattr(
        "app.sector.cycle.run_whale_signals_for_run",
        lambda **kwargs: {"summary_path": str(whale_summary_path), "rows": []},
    )
    monkeypatch.setattr(
        "app.sector.cycle.build_peer_report_from_run",
        lambda **kwargs: {
            "peer_report_path": str(peer_report_path),
            "peer_rankings_path": str(peer_rankings_path),
            "peer_scoreboard_path": str(peer_scoreboard_path),
            "rankings": [],
        },
    )
    monkeypatch.setattr("app.sector.cycle.build_sector_decision_pack", lambda **kwargs: {})
    monkeypatch.setattr("app.sector.cycle.open_dossier_run", lambda run_id: {"status": "DONE"})

    summary = run_sector_cycle(
        sector="Software",
        as_of_date="2026-02-13",
        run_id=run_id,
        peer_limit=10,
        min_peers_dossierable=3,
        limit_dossiers=1,
        with_research=False,
        with_synthesis=False,
    )
    assert summary["status"] == "DONE"
    assert summary["dossierable_count"] == 1
    report_json = cfg.sectors_dir / run_id / "peer_quality_report.json"
    report_md = cfg.sectors_dir / run_id / "peer_quality_report.md"
    assert report_json.exists()
    assert report_md.exists()
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["requested_dossierable_count"] == 3
    assert payload["achieved_dossierable_count"] == 1
    assert payload["dossierable_shortfall"] == 2
    assert payload["final_peer_list"][0]["ticker"] == "AAA"
    assert payload["filtered_reason_buckets"]["NO_CIK"][0]["ticker"] == "NOCIK"
    assert payload["filtered_reason_buckets"]["OTC_EXCLUDED"][0]["ticker"] == "OTCXF"
    assert payload["filtered_reason_buckets"]["FOREIGN_EXCLUDED"][0]["ticker"] == "FRGN"
    assert payload["filtered_reason_buckets"]["NO_ANNUAL_FORMS"][0]["ticker"] == "NOANN"
    assert payload["filtered_reason_buckets"]["BELOW_MIN_ANNUAL"][0]["ticker"] == "LOWANN"
    assert payload["filtered_reason_buckets"]["BUDGET_SKIPPED"][0]["ticker"] == "BUDG"
    report_text = report_md.read_text(encoding="utf-8")
    assert "Filtered Reason Buckets" in report_text
    assert "NO_ANNUAL_FORMS" in report_text
