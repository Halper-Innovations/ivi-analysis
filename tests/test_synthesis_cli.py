from __future__ import annotations

import json
from dataclasses import dataclass

from app.analyst.thesis_contract import (
    AnalysisCitation,
    AnalysisFinding,
    AnalysisReport,
    OpenQuestion,
    ResearchGapSummary,
    ResearchQualitySummary,
    ValuationConclusion,
)
from typer.testing import CliRunner

from app.cli import app
from app.db import get_db, init_db, utc_now_iso
from app.llm.providers.disabled_provider import LLMResult
from tests.financial_integrity_helpers import materialized_no_split_proof


runner = CliRunner()


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _seed_canonical_financial_context(
    cfg,
    *,
    ticker: str,
    as_of_date: str,
) -> None:
    cik = str(sum(ord(char) for char in ticker.upper())).zfill(10)
    companyfacts_source_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    fiscal_year = int(as_of_date[:4]) - 1
    shares_period_end = f"{fiscal_year}-12-31"
    accession = f"{cik}-{as_of_date[2:4]}-000001"
    scorecard = {
        "pricing_zone": "MARGIN_OF_SAFETY",
        "pricing_zone_detail": {
            "current_price": 10.0,
            "current_price_as_of_date": as_of_date,
            "current_price_currency": "USD",
            "current_price_source": "fixture_quote",
            "current_price_source_url": f"https://example.test/quotes/{ticker.upper()}",
            "current_price_basis": "UNADJUSTED",
            "current_raw_price": 10.0,
            "split_adjustment_factor": 1.0,
            "no_intervening_split_proof": materialized_no_split_proof(
                ticker=ticker,
                period_start=shares_period_end,
                period_end=as_of_date,
                verified_as_of=as_of_date,
                issuer_cik=cik,
            ),
            "dcf_base": 12.0,
            "epv_adjusted": 11.0,
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
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET cik=excluded.cik
            """,
            (ticker.upper(), cik, f"{ticker.upper()} Fixture", utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at
            )
            VALUES(?, ?, 'scorecard', '{}', ?, '[]', ?)
            ON CONFLICT(ticker, as_of_date, method) DO UPDATE SET
                outputs_json=excluded.outputs_json
            """,
            (ticker.upper(), as_of_date, json.dumps(scorecard), utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form,
                accession
            )
            VALUES(
                ?, ?, 'FY', ?, 'shares_outstanding', 10.0,
                'shares_millions', ?, ?, ?, '10-K', ?
            )
            ON CONFLICT(ticker, fiscal_year, period_type, line_item) DO UPDATE SET
                value=excluded.value,
                period_end=excluded.period_end,
                filed_date=excluded.filed_date,
                source_url=excluded.source_url,
                form=excluded.form,
                accession=excluded.accession
            """,
            (
                ticker.upper(),
                fiscal_year,
                shares_period_end,
                companyfacts_source_url,
                utc_now_iso(),
                as_of_date,
                accession,
            ),
        )
    companyfacts = {
        "cik": cik,
        "entityName": f"{ticker.upper()} Fixture",
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {
                                "val": 10_000_000.0,
                                "end": shares_period_end,
                                "filed": as_of_date,
                                "form": "10-K",
                                "fy": fiscal_year,
                                "accn": accession,
                            }
                        ]
                    }
                }
            }
        },
    }
    companyfacts_dir = cfg.cache_dir / "companyfacts"
    companyfacts_dir.mkdir(parents=True, exist_ok=True)
    (companyfacts_dir / f"{cik}.json").write_text(
        json.dumps(
            {
                "cik": cik,
                "retrieved_at": utc_now_iso(),
                "source_url": companyfacts_source_url,
                "http_status": 200,
                "size_bytes": len(json.dumps(companyfacts, sort_keys=True).encode("utf-8")),
                "companyfacts": companyfacts,
            }
        ),
        encoding="utf-8",
    )
    submissions_dir = cfg.cache_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    (submissions_dir / f"{cik}.json").write_text(
        json.dumps(
            {
                "tickers": [ticker.upper()],
                "exchanges": ["NYSE"],
                "filings": {
                    "recent": {
                        "form": ["10-K"],
                        "filingDate": [as_of_date],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _seed_evidence(cfg, ticker: str, as_of_date: str):
    packet_path = cfg.evidence_dir / f"{ticker}_{as_of_date}.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(
        json.dumps(
            {
                "ticker": ticker,
                "as_of_date": as_of_date,
                "fundamentals": {"revenue": 1000.0},
                "fundamentals_provenance": {
                    "revenue": {
                        "value": 1000.0,
                        "unit": "USD_millions",
                        "source": "fixture_filing",
                        "period_end": as_of_date,
                        "filed_date": as_of_date,
                        "source_reference": (
                            "https://www.sec.gov/Archives/edgar/data/"
                            f"fixture/{ticker.lower()}-{as_of_date}.htm"
                        ),
                    }
                },
                "valuations": {
                    "dcf": {"outputs": {"status": "OK", "base": 10.0, "low": 9.0, "high": 11.0}}
                },
                "filings_used": [],
                "extracted_facts": [],
                "financials": [],
            }
        ),
        encoding="utf-8",
    )
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO evidence_packets(ticker, as_of_date, packet_path, packet_hash, created_at)
            VALUES(?, ?, ?, 'h', ?)
            """,
            (ticker, as_of_date, str(packet_path), utc_now_iso()),
        )
    _seed_canonical_financial_context(
        cfg,
        ticker=ticker,
        as_of_date=as_of_date,
    )
    _seed_analysis_report(cfg, ticker=ticker, as_of_date=as_of_date)


def _seed_analysis_report(cfg, *, ticker: str, as_of_date: str):
    out_dir = cfg.analyst_outputs_dir / f"{ticker}_{as_of_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "analysis_report.json"
    report = AnalysisReport(
        analysis_id=f"analysis_{ticker}_{as_of_date}",
        ticker=ticker,
        as_of_date=as_of_date,
        generated_at=utc_now_iso(),
        verdict="WATCH",
        confidence_label="MODERATE",
        confidence_score=60,
        thesis_summary="CLI synthesis has an analyst-contract input available.",
        valuation=ValuationConclusion(
            price=10.0,
            base_case_value=10.5,
            bear_case_value=9.0,
            bull_case_value=12.0,
            margin_of_safety=0.05,
        ),
        research_quality=ResearchQualitySummary(
            coverage_score=0.8,
            freshness_score=0.8,
            gap_score=0.2,
            overall_score=0.8,
            incomplete=False,
            evidence_count=2,
            top_gaps=[
                ResearchGapSummary(
                    severity="LOW",
                    summary="Read the next 10-Q.",
                    recommended_action="Read the next 10-Q.",
                )
            ],
        ),
        positives=[
            AnalysisFinding(
                finding_id="P1",
                category="POSITIVE",
                claim="Evidence is ready for synthesis.",
                direction="BULLISH",
                severity="LOW",
                source_basis="filings",
                citation_ids=["C1"],
            )
        ],
        risks=[],
        recent_event_impacts=[],
        open_questions=[
            OpenQuestion(
                question="What changed next?", importance="MEDIUM", next_step="Read the next 10-Q."
            )
        ],
        falsifiers=[],
        citations=[
            AnalysisCitation(
                citation_id="C1",
                source_type="EDGAR",
                source_label="10-Q",
                source_date=as_of_date,
                section="MD&A",
                excerpt="Prompt-ready analyst context exists.",
                source_url="https://www.sec.gov/example",
            )
        ],
        warnings=[],
    )
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO analyst_outputs(ticker, as_of_date, output_type, output_path, output_hash, created_at)
            VALUES(?, ?, 'analysis_report', ?, 'hash', ?)
            ON CONFLICT(ticker, as_of_date, output_type) DO UPDATE SET
                output_path=excluded.output_path,
                output_hash=excluded.output_hash
            """,
            (ticker, as_of_date, str(report_path), utc_now_iso()),
        )


@dataclass
class _FakeProvider:
    calls: int = 0
    provider_name: str = "openai"

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
        _ = (prompt, schema, schema_name)
        self.calls += 1
        payload = {
            "ticker": "AAPL",
            "as_of_date": "2026-02-13",
            "run_id": "run_cli",
            "hypotheses": [
                {
                    "id": "h1",
                    "statement": "Possible upside if fundamentals hold.",
                    "why_it_might_be_true": "Valuation range remains constructive.",
                    "falsifiers": ["cash flow weakens"],
                    "required_evidence": ["next filing"],
                }
            ],
            "claims": [
                {
                    "id": "c1",
                    "text": "DCF value uses valuation outputs.",
                    "type": "numeric",
                    "citations": [],
                    "derived_from": ["evidence_packet.valuations.dcf.base"],
                }
            ],
            "priced_in_assessment": {
                "what_market_assumes": "soft demand",
                "what_is_not_priced": "margin stability",
                "uncertainty_notes": "limited non-EDGAR evidence",
            },
            "next_actions": [
                {
                    "action_type": "edgar_extract",
                    "target_source": "EDGAR",
                    "query_or_url_hint": "next 10-Q",
                    "why": "refresh assumptions",
                }
            ],
            "decision_frame": {
                "stance": "watchlist",
                "key_risks": ["dilution"],
                "catalysts": ["next filing"],
                "time_horizon_days": 90,
            },
            "llm_meta": {
                "model": "gpt-5-mini",
                "prompt_hash": "p",
                "input_hash": "i",
                "cost_estimate_usd": 0.0,
                "created_at": "2026-02-13T00:00:00+00:00",
            },
        }
        return LLMResult(
            json_text=json.dumps(payload),
            model="gpt-5-mini",
            usage_input_tokens=1200,
            usage_output_tokens=300,
            raw={},
        )


def test_synth_cli_run_and_export(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence(cfg, "AAPL", "2026-02-13")
    fake = _FakeProvider()
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: fake)

    run = runner.invoke(
        app,
        [
            "synth-run",
            "--ticker",
            "AAPL",
            "--as-of",
            "2026-02-13",
            "--run-id",
            "run_cli",
        ],
    )
    assert run.exit_code == 0, run.output
    assert "AAPL_2026-02-13_run_run_cli.json" in run.output
    assert (
        "requested_as_of_date=2026-02-13 effective_as_of_date=2026-02-13 as_of_resolution=exact"
        in run.output
    )
    assert fake.calls == 1

    export_path = cfg.outputs_dir / "synth_export_aapl.json"
    export = runner.invoke(
        app,
        [
            "synth-export",
            "--ticker",
            "AAPL",
            "--run-id",
            "run_cli",
            "--out",
            str(export_path),
        ],
    )
    assert export.exit_code == 0, export.output
    assert export_path.exists()


def test_synth_cli_run_no_cache_forces_fresh_call(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence(cfg, "AAPL", "2026-02-13")
    fake = _FakeProvider()
    refresh_calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: fake)
    monkeypatch.setattr(
        "app.cli._build_synthesis_evidence_packet",
        lambda ticker, as_of_date=None, **_kwargs: refresh_calls.append((ticker, as_of_date)),
    )

    first = runner.invoke(
        app,
        [
            "synth-run",
            "--ticker",
            "AAPL",
            "--as-of",
            "2026-02-13",
            "--run-id",
            "run_cli_nocache_a",
        ],
    )
    assert first.exit_code == 0, first.output
    assert fake.calls == 1

    second = runner.invoke(
        app,
        [
            "synth-run",
            "--ticker",
            "AAPL",
            "--as-of",
            "2026-02-13",
            "--run-id",
            "run_cli_nocache_b",
            "--no-cache",
        ],
    )
    assert second.exit_code == 0, second.output
    assert fake.calls == 2
    assert refresh_calls == [("AAPL", "2026-02-13")]


def test_synth_cli_run_with_disabled_provider_writes_fallback(monkeypatch, tmp_path):
    data_dir = tmp_path / "data_disabled"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    _seed_evidence(cfg, "MSFT", "2026-02-13")

    run = runner.invoke(
        app,
        [
            "synth-run",
            "--ticker",
            "MSFT",
            "--as-of",
            "2026-02-13",
            "--run-id",
            "run_disabled_cli",
        ],
    )
    assert run.exit_code == 0, run.output
    assert "MSFT_2026-02-13_run_run_disabled_cli.json" in run.output


def test_synth_cli_reports_fallback_effective_as_of(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence(cfg, "META", "2026-01-29")
    fake = _FakeProvider()
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: fake)

    run = runner.invoke(
        app,
        [
            "synth-run",
            "--ticker",
            "META",
            "--as-of",
            "2026-03-19",
            "--run-id",
            "run_cli_fallback",
        ],
    )
    assert run.exit_code == 0, run.output
    assert "META_2026-01-29_run_run_cli_fallback.json" in run.output
    assert (
        "requested_as_of_date=2026-03-19 effective_as_of_date=2026-01-29 as_of_resolution=fallback"
        in run.output
    )


def test_synth_cli_strict_as_of_fails_on_fallback(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence(cfg, "META", "2026-01-29")
    fake = _FakeProvider()
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: fake)

    run = runner.invoke(
        app,
        [
            "synth-run",
            "--ticker",
            "META",
            "--as-of",
            "2026-03-19",
            "--run-id",
            "run_cli_strict",
            "--strict-as-of",
        ],
    )
    assert run.exit_code == 1, run.output
    assert (
        "No exact evidence packet for META on 2026-03-19; latest available is 2026-01-29"
        in run.output
    )
    assert fake.calls == 0
