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
from app.db import get_db, init_db, utc_now_iso
from app.llm.providers.disabled_provider import LLMResult
from app.llm.synthesis_agent import run_synthesis_for_scope
from app.score.ranker import score_and_rank
from tests.financial_integrity_helpers import materialized_no_split_proof


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    monkeypatch.setenv("VOE_PRICE_PROVIDER", "disabled")
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
    cik = {
        "AAPL": "320193",
        "MSFT": "789019",
    }.get(
        ticker.upper(),
        str(sum(ord(char) for char in ticker.upper())),
    ).zfill(10)
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
                period_start=as_of_date,
                period_end=as_of_date,
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
                value, units, source_url, fetched_at, filed_date, accession
            )
            VALUES(
                ?, ?, 'FY', ?, 'shares_outstanding', 10.0,
                'shares_millions', ?, ?, ?, ?
            )
            ON CONFLICT(ticker, fiscal_year, period_type, line_item) DO UPDATE SET
                value=excluded.value,
                period_end=excluded.period_end,
                filed_date=excluded.filed_date,
                source_url=excluded.source_url
            """,
            (
                ticker.upper(),
                int(as_of_date[:4]),
                as_of_date,
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                utc_now_iso(),
                as_of_date,
                f"{ticker.upper()}-{as_of_date}",
            ),
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


def _seed_effective_date_rows(cfg, *, run_id: str, old_candidate_run_id: str):
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO scores(
                ticker, as_of_date, run_id, subscores_json, total_score, decision,
                is_candidate, is_publishable, candidate_run_id, reasons_json, created_at
            ) VALUES
                ('AAPL', '2026-01-30', ?, '{}', 88, 'Watchlist', 1, 0, ?, '[]', ?),
                ('MSFT', '2026-01-28', ?, '{}', 77, 'Watchlist', 1, 0, ?, '[]', ?)
            """,
            (run_id, old_candidate_run_id, now, run_id, old_candidate_run_id, now),
        )
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES
                ('320193', 'AAPL', '0000320193-26-010001', '10-Q', '2026-01-30', '2025-12-31',
                 'https://www.sec.gov/Archives/edgar/data/320193/doc.htm', NULL, NULL, '2026-02-13', 'parsed', ?, ?),
                ('789019', 'MSFT', '0000789019-26-010001', '10-Q', '2026-01-28', '2025-12-31',
                 'https://www.sec.gov/Archives/edgar/data/789019/doc.htm', NULL, NULL, '2026-02-13', 'parsed', ?, ?)
            """,
            (now, now, now, now),
        )
        conn.execute(
            """
            INSERT INTO research_signals(
                ticker, as_of_date, run_id, recency_days_min, item_count_30d, has_earnings_release, has_investor_presentation,
                sentiment_flags_json, key_topics_json, evidence_item_ids_json, summary_json, created_at
            ) VALUES
                ('AAPL', '2026-01-30', ?, 5, 3, 0, 0, '[]', '[]', '[]', '{}', ?),
                ('MSFT', '2026-01-28', ?, 7, 2, 0, 0, '[]', '[]', '[]', '{}', ?)
            """,
            (run_id, now, run_id, now),
        )

    for ticker, as_of in [("AAPL", "2026-01-30"), ("MSFT", "2026-01-28")]:
        packet_path = cfg.evidence_dir / f"{ticker}_{as_of}.json"
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ticker": ticker,
            "as_of_date": as_of,
            "fundamentals": {
                "revenue": 1000.0,
                "operating_margin": 0.2,
                "fcf": 100.0,
                "net_debt": 0.0,
            },
            "fundamentals_provenance": {
                metric: {
                    "value": value,
                    "unit": ("ratio" if metric == "operating_margin" else "USD_millions"),
                    "source": "fixture_filing",
                    "period_end": as_of,
                    "filed_date": as_of,
                    "source_reference": (
                        "https://www.sec.gov/Archives/edgar/data/"
                        f"fixture/{ticker.lower()}-{as_of}.htm"
                    ),
                }
                for metric, value in {
                    "revenue": 1000.0,
                    "operating_margin": 0.2,
                    "fcf": 100.0,
                    "net_debt": 0.0,
                }.items()
            },
            "valuations": {
                "dcf": {"outputs": {"status": "OK", "base": 10.0, "low": 8.0, "high": 12.0}}
            },
            "filings_used": [{"form_type": "10-Q", "filing_date": as_of}],
            "extracted_facts": [],
            "financials": [],
        }
        packet_path.write_text(json.dumps(payload), encoding="utf-8")
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO evidence_packets(ticker, as_of_date, packet_path, packet_hash, created_at)
                VALUES(?, ?, ?, 'hash', ?)
                """,
                (ticker, as_of, str(packet_path), utc_now_iso()),
            )
        _seed_canonical_financial_context(
            cfg,
            ticker=ticker,
            as_of_date=as_of,
        )
        _seed_analysis_report(cfg, ticker=ticker, as_of_date=as_of)


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
        confidence_score=55,
        thesis_summary=f"{ticker} has a synthesized analyst context at the effective as-of date.",
        valuation=ValuationConclusion(
            price=10.0,
            base_case_value=10.0,
            bear_case_value=8.0,
            bull_case_value=12.0,
            margin_of_safety=0.0,
        ),
        research_quality=ResearchQualitySummary(
            coverage_score=0.7,
            freshness_score=0.7,
            gap_score=0.3,
            overall_score=0.7,
            incomplete=False,
            evidence_count=2,
            top_gaps=[
                ResearchGapSummary(
                    severity="LOW",
                    summary="Read the next filing.",
                    recommended_action="Read the next filing.",
                )
            ],
        ),
        positives=[
            AnalysisFinding(
                finding_id="P1",
                category="POSITIVE",
                claim="Effective as-of evidence is available.",
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
                question="What changed after the filing?",
                importance="MEDIUM",
                next_step="Read the next filing.",
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
                excerpt="Effective as-of report exists.",
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
            "ticker": "PLACEHOLDER",
            "as_of_date": "PLACEHOLDER",
            "run_id": "PLACEHOLDER",
            "hypotheses": [
                {
                    "id": "h1",
                    "statement": "Regression payload",
                    "why_it_might_be_true": "Test fixture",
                    "falsifiers": ["fixture"],
                    "required_evidence": ["fixture"],
                }
            ],
            "claims": [
                {
                    "id": "c1",
                    "text": "Derived claim",
                    "type": "numeric",
                    "citations": [],
                    "derived_from": ["evidence_packet.valuations.dcf.base"],
                }
            ],
            "priced_in_assessment": {
                "what_market_assumes": "fixture",
                "what_is_not_priced": "fixture",
                "uncertainty_notes": "fixture",
            },
            "next_actions": [
                {
                    "action_type": "edgar_extract",
                    "target_source": "EDGAR",
                    "query_or_url_hint": "fixture",
                    "why": "fixture",
                }
            ],
            "decision_frame": {
                "stance": "watchlist",
                "key_risks": ["fixture"],
                "catalysts": ["fixture"],
                "time_horizon_days": 30,
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
            usage_input_tokens=100,
            usage_output_tokens=80,
            raw={},
        )


def test_run_as_of_vs_effective_as_of_scoping_and_synthesis_dates(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "run_20260221T074210077408Z"
    _seed_effective_date_rows(cfg, run_id=run_id, old_candidate_run_id="run_20260218T215544937771Z")

    summary = score_and_rank(
        top_n=2,
        memo_mode="triage",
        run_id=run_id,
        as_of_date="2026-02-13",
        candidate_scope=["AAPL", "MSFT"],
        rescore=False,
    )
    assert summary["score_rows_considered"] == 2
    assert summary["tickers_ranked"] == 2

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT ticker, as_of_date, is_candidate, candidate_run_id
            FROM scores
            WHERE run_id = ?
            ORDER BY ticker
            """,
            (run_id,),
        ).fetchall()
    assert len(rows) == 2
    assert all(int(row["is_candidate"]) == 1 for row in rows)
    assert all(row["candidate_run_id"] == run_id for row in rows)

    fake = _FakeProvider()
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: fake)
    synth_summary = run_synthesis_for_scope(
        as_of_date="2026-02-13",
        run_id=run_id,
        tickers=["AAPL", "MSFT"],
    )
    assert synth_summary["built"] == 2
    assert fake.calls == 2

    with get_db() as conn:
        synth_rows = conn.execute(
            """
            SELECT ticker, as_of_date, packet_path
            FROM synthesis_packets
            WHERE run_id = ?
            ORDER BY ticker
            """,
            (run_id,),
        ).fetchall()
    assert [(row["ticker"], row["as_of_date"]) for row in synth_rows] == [
        ("AAPL", "2026-01-30"),
        ("MSFT", "2026-01-28"),
    ]
    assert f"AAPL_2026-01-30_run_{run_id}.json" in synth_rows[0]["packet_path"]
    assert f"MSFT_2026-01-28_run_{run_id}.json" in synth_rows[1]["packet_path"]
    payload = json.loads(
        (cfg.synthesis_dir / f"AAPL_2026-01-30_run_{run_id}.json").read_text(encoding="utf-8")
    )
    assert payload["requested_as_of_date"] == "2026-02-13"
    assert payload["effective_as_of_date"] == "2026-01-30"
    assert payload["as_of_resolution"] == "fallback"


def test_run_synthesis_scope_no_cache_forces_second_provider_call(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_effective_date_rows(cfg, run_id="scope_seed_run", old_candidate_run_id="scope_old_run")

    fake = _FakeProvider()
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: fake)

    first = run_synthesis_for_scope(
        as_of_date="2026-02-13",
        run_id="scope_cache_a",
        tickers=["AAPL"],
    )
    assert first["built"] == 1
    assert fake.calls == 1

    second = run_synthesis_for_scope(
        as_of_date="2026-02-13",
        run_id="scope_cache_b",
        tickers=["AAPL"],
        allow_cache=False,
    )
    assert second["built"] == 1
    assert fake.calls == 2


def test_run_synthesis_scope_strict_as_of_skips_fallback_tickers(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_effective_date_rows(
        cfg, run_id="scope_strict_seed", old_candidate_run_id="scope_strict_old"
    )

    fake = _FakeProvider()
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: fake)

    summary = run_synthesis_for_scope(
        as_of_date="2026-02-13",
        run_id="scope_strict_run",
        tickers=["AAPL"],
        strict_as_of=True,
    )
    assert summary["built"] == 0
    assert fake.calls == 0


def test_candidate_run_id_backfill_migration(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO scores(
                ticker, as_of_date, run_id, subscores_json, total_score, decision,
                is_candidate, is_publishable, candidate_run_id, reasons_json, created_at
            ) VALUES('AAPL', '2026-01-30', 'run_backfill', '{}', 50, 'Watchlist', 1, 0, 'run_old', '[]', ?)
            """,
            (now,),
        )

    # Re-running init_db should apply schema evolution backfill on existing rows.
    init_db(cfg)

    with get_db() as conn:
        row = conn.execute(
            "SELECT run_id, candidate_run_id FROM scores WHERE ticker='AAPL' AND as_of_date='2026-01-30'"
        ).fetchone()
    assert row["run_id"] == "run_backfill"
    assert row["candidate_run_id"] == "run_backfill"
