from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.analyst.thesis_contract import (
    AnalysisCitation,
    AnalysisFinding,
    AnalysisReport,
    Falsifier,
    OpenQuestion,
    ResearchGapSummary,
    ResearchQualitySummary,
    ValuationConclusion,
)
from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.db import get_db, init_db, utc_now_iso
from app.llm.providers.disabled_provider import LLMResult
from app.llm.synthesis_agent import (
    _estimate_cost_usd,
    _normalize_claim_text_units,
    _normalize_money_mentions,
    _polish_narrative_text,
    _run_spend_usd,
    append_synthesis_section,
    run_synthesis_for_ticker,
)
from tests.financial_integrity_helpers import materialized_no_split_proof


class TestEstimateCostFamilyPricing:
    def test_deepseek_v4_pro_exact_and_cached_rates(self):
        uncached = _estimate_cost_usd(
            "deepseek-v4-pro",
            1000,
            1000,
            provider_name="deepseek",
        )
        cached = _estimate_cost_usd(
            "deepseek-v4-pro",
            1000,
            1000,
            provider_name="deepseek",
            cached_input_tokens=1000,
        )
        assert uncached == 0.001305
        assert cached == 0.000874

    def test_gpt_5_4_mini_uses_gpt5_family_pricing(self):
        # gpt-5.4-mini is the deployed model; must resolve to gpt-5 family
        # standard pricing of (0.00075, 0.0045) per 1k, not the unknown default.
        cost = _estimate_cost_usd("gpt-5.4-mini", 1000, 1000, provider_name="openai")
        assert cost == 0.00525

    def test_gpt_5_4_mini_uses_standard_cached_input_rate(self):
        cost = _estimate_cost_usd(
            "gpt-5.4-mini",
            2000,
            1000,
            provider_name="openai",
            cached_input_tokens=1000,
        )

        assert cost == 0.005325

    def test_haiku_uses_real_haiku_pricing(self):
        # claude-haiku-4-5 real pricing ~ $1/$5 per 1M = (0.001, 0.005) per 1k.
        # 1000 in + 1000 out -> 0.001 + 0.005 = 0.006
        cost = _estimate_cost_usd("claude-haiku-4-5", 1000, 1000, provider_name="anthropic")
        assert cost == 0.006

    def test_opus_uses_opus_pricing(self):
        # opus pricing (0.015, 0.075) per 1k -> 0.015 + 0.075 = 0.09
        cost = _estimate_cost_usd("claude-opus-4-6", 1000, 1000, provider_name="anthropic")
        assert cost == 0.09

    def test_exact_gpt_5_mini_still_supported(self):
        cost = _estimate_cost_usd("gpt-5-mini", 1000, 1000, provider_name="openai")
        assert cost == 0.004

    def test_gpt_5_5_uses_standard_and_cached_input_rates(self):
        cost = _estimate_cost_usd(
            "gpt-5.5",
            2000,
            1000,
            provider_name="openai",
            cached_input_tokens=1000,
        )
        # 1k uncached input ($0.005) + 1k cached input ($0.0005)
        # + 1k output ($0.03).
        assert cost == 0.0355

    def test_gpt_5_5_long_context_multiplier_starts_above_272k(self):
        at_threshold = _estimate_cost_usd(
            "gpt-5.5",
            272_000,
            1_000,
            provider_name="openai",
        )
        above_threshold = _estimate_cost_usd(
            "gpt-5.5",
            272_001,
            1_000,
            provider_name="openai",
        )

        assert at_threshold == 1.39
        assert above_threshold == 2.76501

    def test_gpt_5_5_snapshot_uses_base_and_long_context_rates(self):
        cost = _estimate_cost_usd(
            "gpt-5.5-2026-04-23",
            272_001,
            1_000,
            provider_name="openai",
            cached_input_tokens=1_000,
        )

        assert cost == 2.75601


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_BUDGET_USD_PER_RUN", "5.0")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


# A failed provider call reserves spend priced from ``len(prompt) // 4`` input
# tokens, and the prompt carries the split-proof envelope's ``materialized_path``
# verbatim.  That path is rooted at ``cfg.cache_dir``, i.e. under pytest's temp
# directory, so an exact cost literal would otherwise track the width of the
# basetemp counter (``pytest-999`` -> ``pytest-1000`` adds a character) and the
# length of the temp root on whatever machine runs the suite.  Tests that pin an
# exact failed-call cost seed the proof cache under a root padded to a constant
# width, which holds the prompt length -- and therefore the price -- fixed
# everywhere.  Padding can only lengthen a path, so a temp root wider than the
# budget fails loudly rather than silently drifting the literal.
_PROOF_CACHE_ROOT_WIDTH = 200
_PROOF_CACHE_ROOT_STEM = "cache_"


def _fixed_width_cache_dir(tmp_path: Path) -> Path:
    base = tmp_path.resolve()
    padding = _PROOF_CACHE_ROOT_WIDTH - len(str(base)) - len(_PROOF_CACHE_ROOT_STEM) - 1
    if padding < 0:
        raise RuntimeError(
            f"temp root {base} exceeds the {_PROOF_CACHE_ROOT_WIDTH}-character "
            "budget for a fixed-width split-proof cache root"
        )
    root = base / (_PROOF_CACHE_ROOT_STEM + "x" * padding)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _seed_canonical_financial_context(
    cfg,
    *,
    ticker: str,
    as_of_date: str,
) -> None:
    cik = str(sum(ord(char) for char in ticker.upper())).zfill(10)
    source_url = f"https://example.test/quotes/{ticker.upper()}"
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
            "current_price_source_url": source_url,
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


def _seed_evidence_packet(cfg, ticker: str, as_of_date: str):
    packet_path = cfg.evidence_dir / f"{ticker}_{as_of_date}.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "fundamentals": {
            "revenue": 1000.0,
            "fcf": 120.0,
            "operating_margin": 0.2,
            "r_and_d_total": 180.0,
            "sales_marketing_total": 140.0,
            "g_and_a_total": 60.0,
            "deferred_revenue_amount": 400.0,
            "deferred_revenue_to_revenue_latest": 0.4,
            "rpo_amount": 900.0,
            "rpo_to_revenue_latest": 0.9,
            "customer_concentration_present": 0,
            "customer_concentration_pct": 0.12,
            "segment_count": 3.0,
            "share_repurchases_amount": 80.0,
            "dividends_paid_amount": 50.0,
            "revenue_cagr_3y": 0.12,
            "revenue_cagr_5y": 0.14,
            "gross_margin_trend_slope": 0.01,
            "operating_margin_trend_slope": 0.015,
            "fcf_margin_trend_slope": -0.002,
            "r_and_d_intensity_latest": 0.18,
            "r_and_d_intensity_delta": -0.01,
            "segment_count_delta": 1.0,
            "customer_concentration_delta": -0.02,
            "dilution_rate_shares_cagr": -0.005,
        },
        "valuations": {
            "owner_earnings": {
                "outputs": {
                    "status": "OK",
                    "owner_earnings_latest": 120.0,
                    "confidence": "LOWER",
                    "flags": ["SBC_NOT_ADJUSTED"],
                }
            },
            "dcf": {
                "outputs": {"status": "OK", "low": 10.0, "base": 12.0, "high": 14.0, "flags": []}
            },
            "epv": {"outputs": {"status": "OK", "value_per_share": 11.0}},
            "graham": {"outputs": {"status": "OK", "value_per_share": 9.0}},
            "ncav": {
                "outputs": {
                    "status": "OK",
                    "value_per_share": 3.0,
                    "signal": "NCAV_PARTIAL_PROTECTION",
                }
            },
            "scorecard": {"outputs": {"signal": "UNDERVALUED", "type": "EARNINGS_DRIVEN"}},
            "roic": {"outputs": {"status": "OK", "signal": "ROIC_STRONG"}},
            "capital_structure": {
                "outputs": {"de_ratio": 0.2, "cash_coverage": 0.8, "interest_coverage": 12.0}
            },
            "reverse_dcf": {"outputs": {"status": "OK", "outputs": {"implied_growth": 0.08}}},
        },
        "filings_used": [{"form_type": "10-Q", "filing_date": as_of_date}],
        "extracted_facts": [
            {
                "fact_type": "deferred_revenue_amount",
                "value": {"metric": "deferred_revenue_amount", "value": 400.0},
                "citation": {
                    "source_url": "https://www.sec.gov/example",
                    "snippet": "Deferred revenue was 400",
                    "section_label": "notes",
                },
            },
            {
                "fact_type": "r_and_d_total",
                "value": {"metric": "r_and_d_total", "value": 180.0},
                "citation": {
                    "source_url": "https://www.sec.gov/example",
                    "snippet": "Research and Development 180",
                    "section_label": "financial_statements",
                },
            },
        ],
        "financials": [],
    }
    if ticker == "JPM":
        payload["fundamentals"]["issuer_classification"] = "financial"
        payload["fundamentals"]["fcf"] = "UNKNOWN"
        payload["fundamentals"]["allowance_for_credit_losses"] = 25765.0
        payload["fundamentals"]["provision_for_credit_losses"] = 10462.0
        payload["fundamentals"]["net_charge_offs"] = 3142.0
        payload["fundamentals"]["nonaccrual_loans"] = 8650.0
        payload["fundamentals"]["allowance_to_loans_latest"] = 25765.0 / 1411992.0
        payload["financials"] = [
            {
                "statement_type": "balance_sheet",
                "line_item": "deposits",
                "value": 2562380.0,
                "units": "USD_millions",
                "period": "2025-12-31",
                "citation": {
                    "source_url": "https://www.sec.gov/example",
                    "snippet": "Deposits 2562380",
                    "section_label": "financials",
                },
            },
            {
                "statement_type": "balance_sheet",
                "line_item": "loans",
                "value": 1411992.0,
                "units": "USD_millions",
                "period": "2025-12-31",
                "citation": {
                    "source_url": "https://www.sec.gov/example",
                    "snippet": "Loans 1411992",
                    "section_label": "financials",
                },
            },
            {
                "statement_type": "balance_sheet",
                "line_item": "allowance_for_credit_losses",
                "value": 25765.0,
                "units": "USD_millions",
                "period": "2025-12-31",
                "citation": {
                    "source_url": "https://www.sec.gov/example",
                    "snippet": "Allowance for credit losses 25765",
                    "section_label": "financials",
                },
            },
            {
                "statement_type": "income_statement",
                "line_item": "provision_for_credit_losses",
                "value": 10462.0,
                "units": "USD_millions",
                "period": "2025-12-31",
                "citation": {
                    "source_url": "https://www.sec.gov/example",
                    "snippet": "Provision for credit losses 10462",
                    "section_label": "financials",
                },
            },
            {
                "statement_type": "credit_quality",
                "line_item": "net_charge_offs",
                "value": 3142.0,
                "units": "USD_millions",
                "period": "2025-12-31",
                "citation": {
                    "source_url": "https://www.sec.gov/example",
                    "snippet": "Net charge-offs 3142",
                    "section_label": "financials",
                },
            },
        ]
    source_reference = (
        f"https://www.sec.gov/Archives/edgar/data/fixture/{ticker.lower()}-{as_of_date}.htm"
    )

    def provenance(value: float, units: str) -> dict:
        return {
            "value": float(value),
            "unit": units,
            "source": "fixture_filing",
            "period_end": as_of_date,
            "filed_date": as_of_date,
            "source_reference": source_reference,
        }

    ratio_metrics = {
        "operating_margin",
        "customer_concentration_pct",
        "revenue_cagr_3y",
        "revenue_cagr_5y",
        "gross_margin_trend_slope",
        "operating_margin_trend_slope",
        "fcf_margin_trend_slope",
        "r_and_d_intensity_latest",
        "r_and_d_intensity_delta",
        "customer_concentration_delta",
        "dilution_rate_shares_cagr",
        "allowance_to_loans_latest",
    }
    count_metrics = {
        "customer_concentration_present",
        "segment_count",
        "segment_count_delta",
    }
    payload["fundamentals_provenance"] = {
        metric: provenance(
            float(value),
            (
                "ratio"
                if metric in ratio_metrics
                else "count"
                if metric in count_metrics
                else "USD_millions"
            ),
        )
        for metric, value in payload["fundamentals"].items()
        if not isinstance(value, bool) and isinstance(value, (int, float))
    }
    for row in payload["extracted_facts"]:
        value_payload = row.get("value")
        if not isinstance(value_payload, dict):
            continue
        numeric_value = value_payload.get("value")
        if isinstance(numeric_value, bool) or not isinstance(
            numeric_value,
            (int, float),
        ):
            continue
        row["provenance"] = provenance(
            float(numeric_value),
            "USD_millions",
        )
    for row in payload["financials"]:
        numeric_value = row.get("value")
        if isinstance(numeric_value, bool) or not isinstance(
            numeric_value,
            (int, float),
        ):
            continue
        row["provenance"] = provenance(
            float(numeric_value),
            str(row.get("units") or "USD_millions"),
        )
    packet_path.write_text(json.dumps(payload), encoding="utf-8")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO evidence_packets(ticker, as_of_date, packet_path, packet_hash, created_at)
            VALUES(?, ?, ?, 'hash', ?)
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
    report_md_path = out_dir / "analysis_report.md"
    report = AnalysisReport(
        analysis_id=f"analysis_{ticker}_{as_of_date}",
        ticker=ticker,
        as_of_date=as_of_date,
        generated_at=utc_now_iso(),
        verdict="WATCH",
        confidence_label="MODERATE",
        confidence_score=62,
        thesis_summary=f"{ticker} has enough current evidence to support synthesis framing.",
        valuation=ValuationConclusion(
            price=10.0,
            base_case_value=12.0,
            bear_case_value=9.0,
            bull_case_value=14.0,
            margin_of_safety=0.2,
        ),
        research_quality=ResearchQualitySummary(
            coverage_score=0.8,
            freshness_score=0.7,
            gap_score=0.2,
            overall_score=0.76,
            incomplete=False,
            evidence_count=3,
            top_gaps=[
                ResearchGapSummary(
                    severity="MEDIUM",
                    summary="Need updated filing detail.",
                    recommended_action="Read the next 10-Q.",
                )
            ],
        ),
        positives=[
            AnalysisFinding(
                finding_id="P1",
                category="POSITIVE",
                claim="Revenue visibility remains constructive.",
                direction="BULLISH",
                severity="MEDIUM",
                source_basis="filings",
                citation_ids=["C1"],
            )
        ],
        risks=[
            AnalysisFinding(
                finding_id="R1",
                category="RISK",
                claim="Execution risk remains material.",
                direction="BEARISH",
                severity="MEDIUM",
                source_basis="filings",
                citation_ids=["C1"],
            )
        ],
        recent_event_impacts=[
            AnalysisFinding(
                finding_id="E1",
                category="RECENT_EVENT",
                claim="The next filing remains the key near-term catalyst.",
                direction=None,
                severity="LOW",
                source_basis="filings",
                citation_ids=["C1"],
            )
        ],
        open_questions=[
            OpenQuestion(
                question="What changed most recently?",
                importance="HIGH",
                next_step="Read the next 10-Q.",
            )
        ],
        falsifiers=[
            Falsifier(
                description="Subsequent filings contradict the thesis.",
                trigger_type="FILING_CHANGE",
                monitoring_hint="Watch the next 10-Q.",
            )
        ],
        citations=[
            AnalysisCitation(
                citation_id="C1",
                source_type="EDGAR",
                source_label="10-Q",
                source_date=as_of_date,
                section="MD&A",
                excerpt="Current evidence remains grounded in the filing.",
                source_url="https://www.sec.gov/example",
            )
        ],
        sources_used=["EDGAR"],
        warnings=[],
    )
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    report_md_path.write_text(f"# {ticker} analysis\n", encoding="utf-8")
    created_at = utc_now_iso()
    with get_db() as conn:
        conn.executemany(
            """
            INSERT INTO analyst_outputs(ticker, as_of_date, output_type, output_path, output_hash, created_at)
            VALUES(?, ?, ?, ?, 'hash', ?)
            ON CONFLICT(ticker, as_of_date, output_type) DO UPDATE SET
                output_path=excluded.output_path,
                output_hash=excluded.output_hash
            """,
            [
                (ticker, as_of_date, "analysis_report", str(report_path), created_at),
                (ticker, as_of_date, "analysis_report_markdown", str(report_md_path), created_at),
            ],
        )


def _seed_legacy_research_packet(
    cfg,
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
    suffix: str,
    summary: str,
    source_url: str,
):
    packet_path = cfg.outputs_dir / "research" / suffix / f"{ticker}_{as_of_date}.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "ticker": ticker,
        "as_of_date": as_of_date,
        "generated_at": utc_now_iso(),
        "quality": {
            "coverage_score": 0.7,
            "freshness_score": 0.6,
            "gap_score": 0.2,
            "overall_research_score": 0.68,
            "incomplete": False,
            "top_gaps": [
                {
                    "severity": "LOW",
                    "summary": "Read the next filing.",
                    "recommended_action": "Read the next filing.",
                }
            ],
        },
        "key_questions": [
            {
                "question_id": "Q1",
                "question": "What changed?",
                "importance": "HIGH",
                "next_step": "Read the next filing.",
            }
        ],
        "evidence_items": [
            {
                "id": "ev_1",
                "ticker": ticker,
                "as_of_date": as_of_date,
                "source_type": "EDGAR",
                "source_url": source_url,
                "retrieved_at": utc_now_iso(),
                "excerpt_text": summary,
                "citations": [
                    {"source_url": source_url, "snippet": summary, "section_label": "financials"}
                ],
                "hash": "hash",
            }
        ],
        "findings": [
            {
                "entry_id": "F1",
                "summary": summary,
                "evidence_item_ids": ["ev_1"],
                "citations": [
                    {"source_url": source_url, "snippet": summary, "section_label": "financials"}
                ],
                "derived_from": ["x"],
            }
        ],
        "risks": [
            {
                "entry_id": "R1",
                "summary": f"risk {summary}",
                "evidence_item_ids": ["ev_1"],
                "citations": [
                    {"source_url": source_url, "snippet": summary, "section_label": "financials"}
                ],
                "derived_from": [],
            }
        ],
        "catalysts": [
            {
                "entry_id": "C1",
                "summary": f"catalyst {summary}",
                "evidence_item_ids": ["ev_1"],
                "citations": [
                    {"source_url": source_url, "snippet": summary, "section_label": "financials"}
                ],
                "derived_from": [],
            }
        ],
        "disconfirming_evidence": [
            {
                "entry_id": "D1",
                "summary": f"falsifier {summary}",
                "evidence_item_ids": ["ev_1"],
                "citations": [
                    {"source_url": source_url, "snippet": summary, "section_label": "financials"}
                ],
                "derived_from": [],
            }
        ],
        "next_actions": [
            {
                "step_id": "A1",
                "question_id": "Q1",
                "action": "Read the next filing.",
                "section_targets": [],
                "keywords": [],
                "disconfirmation_check": "",
                "evidence_gap": "",
                "tied_metrics": [],
                "allowed_source": "EDGAR",
            }
        ],
        "evidence_gaps": [],
        "claims": [],
    }
    packet_path.write_text(json.dumps(payload), encoding="utf-8")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO research_packets(ticker, as_of_date, run_id, packet_path, packet_hash, created_at)
            VALUES(?, ?, ?, ?, 'hash', ?)
            """,
            (ticker, as_of_date, run_id, str(packet_path), utc_now_iso()),
        )


@dataclass
class _FakeProvider:
    calls: int = 0
    provider_name: str = "openai"
    last_prompt: str = ""

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
        _ = (schema, schema_name)
        self.calls += 1
        self.last_prompt = prompt
        payload = {
            "ticker": "AAPL",
            "as_of_date": "2026-02-13",
            "run_id": "placeholder",
            "hypotheses": [
                {
                    "id": "h1",
                    "statement": "Upside may be underappreciated.",
                    "why_it_might_be_true": "Valuation ranges are above stressed assumptions.",
                    "falsifiers": ["Cash flow deterioration in next filing."],
                    "required_evidence": ["next 10-Q cash flow"],
                }
            ],
            "claims": [
                {
                    "id": "c1",
                    "text": "DCF base uses filing-derived valuation outputs.",
                    "type": "numeric",
                    "citations": [],
                    "derived_from": ["evidence_packet.valuations.dcf.base"],
                }
            ],
            "priced_in_assessment": {
                "what_market_assumes": "Margins remain pressured.",
                "what_is_not_priced": "Cash flow resilience.",
                "uncertainty_notes": "Research coverage moderate.",
            },
            "next_actions": [
                {
                    "action_type": "edgar_extract",
                    "target_source": "EDGAR",
                    "query_or_url_hint": "next 10-Q margin bridge",
                    "why": "Validate margin trajectory.",
                }
            ],
            "decision_frame": {
                "stance": "watchlist",
                "key_risks": ["dilution"],
                "catalysts": ["next filing"],
                "time_horizon_days": 120,
            },
            "llm_meta": {
                "model": "gpt-5-mini",
                "prompt_hash": "placeholder",
                "input_hash": "placeholder",
                "cost_estimate_usd": 0.0,
                "created_at": "2026-02-13T00:00:00+00:00",
            },
        }
        return LLMResult(
            json_text=json.dumps(payload),
            model="gpt-5-mini",
            usage_input_tokens=1000,
            usage_output_tokens=300,
            raw={},
        )


def test_synthesis_invalid_financial_context_suppresses_provider_and_fallback(
    monkeypatch,
    tmp_path,
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    packet_path = cfg.evidence_dir / "MISS_2026-02-13.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(
        json.dumps(
            {
                "ticker": "MISS",
                "as_of_date": "2026-02-13",
                "fundamentals": {"revenue": 1000.0},
                "valuations": {},
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
            INSERT INTO evidence_packets(
                ticker, as_of_date, packet_path, packet_hash, created_at
            )
            VALUES('MISS', '2026-02-13', ?, 'hash', ?)
            """,
            (str(packet_path), utc_now_iso()),
        )
    provider = _FakeProvider()
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: provider)

    with pytest.raises(InvalidFinancialInputError):
        run_synthesis_for_ticker(
            "MISS",
            as_of_date="2026-02-13",
            run_id="run_invalid_financial_context",
        )

    assert provider.calls == 0
    assert not any(cfg.synthesis_dir.glob("MISS_*"))


def test_synthesis_post_filed_as_of_evidence_makes_zero_provider_calls(
    monkeypatch,
    tmp_path,
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")
    packet_path = cfg.evidence_dir / "AAPL_2026-02-13.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["fundamentals_provenance"]["revenue"]["filed_date"] = "2026-02-14"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    provider = _FakeProvider()
    monkeypatch.setattr(
        "app.llm.synthesis_agent.get_llm_provider",
        lambda: provider,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_synthesis_for_ticker(
            "AAPL",
            as_of_date="2026-02-13",
            run_id="run_post_filed_as_of",
        )

    assert "PROMPT_FINANCIAL_PROVENANCE_ASOF_INVALID" in {
        violation.code for violation in exc_info.value.violations
    }
    assert provider.calls == 0
    assert not any(cfg.synthesis_dir.glob("AAPL_2026-02-13_run_run_post_filed_as_of.json"))
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM synthesis_packets
            WHERE ticker = 'AAPL'
              AND run_id = 'run_post_filed_as_of'
            """
        ).fetchone()
    assert row is None


def test_synthesis_publication_rejects_mutated_full_evidence_scope(
    monkeypatch,
    tmp_path,
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")
    from app.llm import synthesis_agent as synthesis_module

    captured: dict[str, dict] = {}
    original_compact_evidence = synthesis_module._compact_evidence

    def capture_evidence(packet: dict) -> dict:
        captured["packet"] = packet
        return original_compact_evidence(packet)

    class _MutatingProvider(_FakeProvider):
        def synthesize_json(
            self,
            *,
            prompt: str,
            schema: dict,
            schema_name: str | None = None,
        ):
            result = super().synthesize_json(
                prompt=prompt,
                schema=schema,
                schema_name=schema_name,
            )
            captured["packet"]["valuations"]["dcf"]["outputs"]["base"] = 99.0
            return result

    provider = _MutatingProvider()
    monkeypatch.setattr(
        "app.llm.synthesis_agent._compact_evidence",
        capture_evidence,
    )
    monkeypatch.setattr(
        "app.llm.synthesis_agent.get_llm_provider",
        lambda: provider,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_synthesis_for_ticker(
            "AAPL",
            as_of_date="2026-02-13",
            run_id="run_mutated_publication_scope",
        )

    assert {violation.code for violation in exc_info.value.violations} == {
        "BOUND_FINANCIAL_INPUT_MUTATED"
    }
    assert provider.calls == 1
    assert not any(cfg.synthesis_dir.glob("AAPL_2026-02-13_run_run_mutated_publication_scope.json"))
    with get_db() as conn:
        packet_row = conn.execute(
            """
            SELECT 1
            FROM synthesis_packets
            WHERE ticker = 'AAPL'
              AND run_id = 'run_mutated_publication_scope'
            """
        ).fetchone()
        attempt_rows = conn.execute(
            """
            SELECT physical_sequence, status, cost_estimate_usd
            FROM synthesis_paid_attempts
            WHERE ticker = 'AAPL'
              AND run_id = 'run_mutated_publication_scope'
            ORDER BY physical_sequence
            """
        ).fetchall()
        spent = _run_spend_usd(conn, "run_mutated_publication_scope")
    assert packet_row is None
    assert [tuple(row) for row in attempt_rows] == [(1, "OK", 0.0019)]
    assert spent == 0.0019


def test_synthesis_failed_call_persists_spend_and_cannot_hide_scope_mutation(
    monkeypatch,
    tmp_path,
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(cfg, "cache_dir", _fixed_width_cache_dir(tmp_path))
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")
    from app.llm import synthesis_agent as synthesis_module
    from app.llm.providers.retry_guard import _notify_failed_attempt

    captured: dict[str, dict] = {}
    original_compact_evidence = synthesis_module._compact_evidence

    def capture_evidence(packet: dict) -> dict:
        captured["packet"] = packet
        return original_compact_evidence(packet)

    class _FailingMutatingProvider(_FakeProvider):
        def synthesize_json(
            self,
            *,
            prompt: str,
            schema: dict,
            schema_name: str | None = None,
        ):
            _ = (prompt, schema)
            self.calls += 1
            captured["packet"]["valuations"]["dcf"]["outputs"]["base"] = 99.0
            error = RuntimeError("synthesis provider failed after mutation")
            _notify_failed_attempt(
                provider="openai",
                schema_name=str(schema_name),
                attempt=1,
                exc=error,
                retryable=False,
                will_retry=False,
            )
            raise error

    provider = _FailingMutatingProvider()
    monkeypatch.setattr(synthesis_module, "_compact_evidence", capture_evidence)
    monkeypatch.setattr(synthesis_module, "get_llm_provider", lambda: provider)

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_synthesis_for_ticker(
            "AAPL",
            as_of_date="2026-02-13",
            run_id="run_failed_mutated_scope",
        )

    assert provider.calls == 1
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert not any(cfg.synthesis_dir.glob("AAPL_2026-02-13_run_run_failed_mutated_scope.json"))
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT physical_sequence, status, cost_estimate_usd
            FROM synthesis_paid_attempts
            WHERE ticker = 'AAPL' AND run_id = 'run_failed_mutated_scope'
            ORDER BY physical_sequence
            """
        ).fetchall()
        spent = _run_spend_usd(conn, "run_failed_mutated_scope")
    assert [tuple(row) for row in rows] == [(1, "ERROR", 0.016503)]
    assert spent == 0.016503


def test_synthesis_failed_publication_spend_blocks_repeat_before_provider(
    monkeypatch,
    tmp_path,
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")
    from app.config import get_config as _get_config
    from app.llm import synthesis_agent as synthesis_module

    monkeypatch.setenv("VOE_OPENAI_BUDGET_USD_PER_RUN", "0.5005")
    _get_config.cache_clear()
    captured: dict[str, dict] = {}
    original_compact_evidence = synthesis_module._compact_evidence

    def capture_evidence(packet: dict) -> dict:
        captured["packet"] = packet
        return original_compact_evidence(packet)

    class _MutatingProvider(_FakeProvider):
        def synthesize_json(
            self,
            *,
            prompt: str,
            schema: dict,
            schema_name: str | None = None,
        ):
            result = super().synthesize_json(
                prompt=prompt,
                schema=schema,
                schema_name=schema_name,
            )
            captured["packet"]["valuations"]["dcf"]["outputs"]["base"] = 99.0
            return result

    provider = _MutatingProvider()
    monkeypatch.setattr(
        synthesis_module,
        "_compact_evidence",
        capture_evidence,
    )
    monkeypatch.setattr(
        synthesis_module,
        "_estimate_cost_usd",
        lambda *args, **kwargs: 0.5,
    )
    monkeypatch.setattr(
        synthesis_module,
        "get_llm_provider",
        lambda: provider,
    )

    with pytest.raises(InvalidFinancialInputError):
        run_synthesis_for_ticker(
            "AAPL",
            as_of_date="2026-02-13",
            run_id="run_persisted_failed_spend",
        )

    repeated = run_synthesis_for_ticker(
        "AAPL",
        as_of_date="2026-02-13",
        run_id="run_persisted_failed_spend",
    )

    assert repeated is None
    assert provider.calls == 1
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT status, cost_estimate_usd
            FROM synthesis_paid_attempts
            WHERE run_id = 'run_persisted_failed_spend'
            ORDER BY physical_sequence
            """
        ).fetchall()
        spent = _run_spend_usd(conn, "run_persisted_failed_spend")
    assert [tuple(row) for row in rows] == [("OK", 0.0019)]
    assert spent == 0.0019
    _get_config.cache_clear()


def test_synthesis_caching_reuses_existing_and_hash_cache(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")
    fake = _FakeProvider()
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: fake)

    first = run_synthesis_for_ticker("AAPL", as_of_date="2026-02-13", run_id="run_a")
    assert first is not None and first.exists()
    assert fake.calls == 1

    second_same_run = run_synthesis_for_ticker("AAPL", as_of_date="2026-02-13", run_id="run_a")
    assert second_same_run is not None
    assert fake.calls == 1

    third_new_run = run_synthesis_for_ticker("AAPL", as_of_date="2026-02-13", run_id="run_b")
    assert third_new_run is not None and third_new_run.exists()
    assert fake.calls == 1
    cached_payload = json.loads(third_new_run.read_text(encoding="utf-8"))
    assert "R&D $180m" in cached_payload["business_quality_summary"]
    assert "deferred revenue of $400m" in cached_payload["business_quality_summary"]
    assert "DCF base value is about $12.00 per share" in cached_payload["valuation_interpretation"]
    with get_db() as conn:
        attempt_count = conn.execute(
            "SELECT COUNT(*) AS count FROM synthesis_paid_attempts"
        ).fetchone()["count"]
        run_a_packet = conn.execute(
            """
            SELECT paid_invocation_id
            FROM synthesis_packets
            WHERE ticker = 'AAPL' AND run_id = 'run_a'
            """
        ).fetchone()
        run_b_packet = conn.execute(
            """
            SELECT paid_invocation_id
            FROM synthesis_packets
            WHERE ticker = 'AAPL' AND run_id = 'run_b'
            """
        ).fetchone()
        run_a_spend = _run_spend_usd(conn, "run_a")
        run_b_spend = _run_spend_usd(conn, "run_b")
    assert attempt_count == 1
    assert run_a_packet["paid_invocation_id"]
    assert run_b_packet["paid_invocation_id"] is None
    assert run_a_spend == 0.0019
    assert run_b_spend == 0.0
    assert '"owner_earnings"' in fake.last_prompt
    assert '"dcf"' in fake.last_prompt
    assert '"epv"' in fake.last_prompt
    assert '"graham"' in fake.last_prompt
    assert '"ncav"' in fake.last_prompt
    assert '"reverse_dcf"' in fake.last_prompt
    assert '"scorecard"' in fake.last_prompt
    assert '"deferred_revenue_amount"' in fake.last_prompt
    assert '"rpo_amount"' in fake.last_prompt
    assert '"r_and_d_total"' in fake.last_prompt
    assert '"sales_marketing_total"' in fake.last_prompt
    assert '"g_and_a_total"' in fake.last_prompt
    assert '"dossier_focus"' in fake.last_prompt
    assert '"reinvestment"' in fake.last_prompt
    assert '"revenue_visibility"' in fake.last_prompt
    assert '"capital_allocation"' in fake.last_prompt
    assert '"concentration_and_segments"' in fake.last_prompt
    assert '"trend_signals"' in fake.last_prompt
    assert '"revenue_cagr_3y"' in fake.last_prompt
    assert '"operating_margin_trend_slope"' in fake.last_prompt
    assert '"dilution_rate_shares_cagr"' in fake.last_prompt
    assert '"canonical_financial_context"' in fake.last_prompt
    assert (
        "Use canonical_financial_context as the sole authority for price, shares, market cap"
        in fake.last_prompt
    )
    assert '"analysis_context"' in fake.last_prompt
    assert "research_packet.*" not in fake.last_prompt
    assert "Round user-facing numbers so they read naturally" in fake.last_prompt
    assert (
        "prefer deterministic dossier facts for reinvestment and revenue quality"
        in fake.last_prompt.lower()
    )
    assert "explicitly interpret deferred revenue and RPO as visibility signals" in fake.last_prompt
    assert "explicitly mention repurchases and dividends when present" in fake.last_prompt
    assert (
        "If customer_concentration_present = 0 and customer_concentration_pct is UNKNOWN"
        in fake.last_prompt
    )
    assert "Use evidence_packet.dossier_focus.trend_signals when available" in fake.last_prompt
    assert (
        "Writing style should read like a concise investor memo, not a checklist"
        in fake.last_prompt
    )

    with get_db() as conn:
        row = conn.execute(
            "SELECT from_cache FROM synthesis_packets WHERE ticker='AAPL' AND run_id='run_b'"
        ).fetchone()
        assert int(row["from_cache"]) == 1


def test_synthesis_prefers_analysis_report_context(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")
    fake = _FakeProvider()
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: fake)

    path = run_synthesis_for_ticker("AAPL", as_of_date="2026-02-13", run_id="run_analysis_context")
    assert path is not None
    assert '"analysis_context"' in fake.last_prompt
    assert "AAPL has enough current evidence to support synthesis framing." in fake.last_prompt
    assert "legacy_research_packet_fallback" not in fake.last_prompt


def test_synthesis_falls_back_to_legacy_packet_when_analysis_report_missing(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")
    with get_db() as conn:
        conn.execute(
            "DELETE FROM analyst_outputs WHERE ticker = 'AAPL' AND as_of_date = '2026-02-13'"
        )
    _seed_legacy_research_packet(
        cfg,
        ticker="AAPL",
        as_of_date="2026-02-13",
        run_id="run_legacy_fallback",
        suffix="legacy_fallback",
        summary="legacy only summary",
        source_url="https://www.sec.gov/legacy",
    )
    fake = _FakeProvider()
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: fake)

    path = run_synthesis_for_ticker("AAPL", as_of_date="2026-02-13", run_id="run_legacy_fallback")
    assert path is not None
    assert "legacy only summary" in fake.last_prompt
    assert "legacy_research_packet_fallback" in fake.last_prompt


def test_synthesis_prefers_analysis_report_over_legacy_fallback(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")
    _seed_legacy_research_packet(
        cfg,
        ticker="AAPL",
        as_of_date="2026-02-13",
        run_id="run_precedence",
        suffix="legacy_precedence",
        summary="legacy summary should not win",
        source_url="https://www.sec.gov/legacy-precedence",
    )
    fake = _FakeProvider()
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: fake)

    path = run_synthesis_for_ticker("AAPL", as_of_date="2026-02-13", run_id="run_precedence")
    assert path is not None
    assert "AAPL has enough current evidence to support synthesis framing." in fake.last_prompt
    assert "legacy summary should not win" not in fake.last_prompt


def test_synthesis_polishes_numeric_text_and_canonicalizes_paths(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")

    class _PolishProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "AAPL",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Operating margin is 45.62195624085985% and reverse DCF is 35.24113818911284%.",
                "valuation_interpretation": "EPV looks like 100.235 and cover page extraction was used for shares. valuations.dcf.base remains the anchor.",
                "risk_frame": "Risk remains 12.909443 due to leverage assumptions.",
                "catalyst_frame": "Next filing may clarify 35.24113818911284% growth assumptions.",
                "evidence_gaps": [
                    "Need 45.62195624085985% margin bridge detail from research_packet.quality.top_gaps."
                ],
                "recommended_next_actions": [
                    "Recheck 100.235 per-share assumptions from research_packet.next_actions.A1."
                ],
                "confidence_notes": "Model saw 35.24113818911284% implied growth.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "Margin may stay at 45.62195624085985%.",
                        "why_it_might_be_true": "Recent trend shows 12.909443 support from evidence_packet.extracted_facts.r_and_d_pct_revenue.",
                        "falsifiers": ["Falls to 10.12345%."],
                        "required_evidence": ["Confirm 100.235 DCF anchor."],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Operating margin is 45.62195624085985% and shares came from cover page extraction.",
                        "type": "numeric",
                        "citations": [],
                        "derived_from": [
                            "financials.operating_income",
                            "valuations.dcf.base",
                            "extracted_facts.shares_outstanding",
                            "dossier.extractors.r_and_d_total",
                            "dossier.metrics.rpo_to_revenue_latest",
                        ],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "Growth is 35.24113818911284%.",
                    "what_is_not_priced": "EPV at 100.235.",
                    "uncertainty_notes": "Value gap of 12.909443 remains.",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "Check 35.24113818911284% growth.",
                        "why": "Need 100.235 precision check.",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["Growth at 35.24113818911284% may be too high."],
                    "catalysts": ["DCF may fall to 100.235."],
                    "time_horizon_days": 120,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _PolishProvider())
    path = run_synthesis_for_ticker("AAPL", as_of_date="2026-02-13", run_id="run_polish")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["business_quality_summary"].startswith(
        "Operating margin is 45.6% and reverse DCF is 35.2%"
    )
    assert "EPV looks like 100.23" in payload["valuation_interpretation"]
    assert "fallback filing extraction was used for shares" in payload["valuation_interpretation"]
    assert "evidence_packet.valuations" not in payload["valuation_interpretation"]
    assert "evidence_packet.evidence_packet." not in payload["valuation_interpretation"]
    assert "research_packet." not in payload["evidence_gaps"][0]
    assert "research_packet." not in payload["recommended_next_actions"][0]
    assert "evidence_packet." not in payload["hypotheses"][0]["why_it_might_be_true"]
    assert payload["claims"][0]["derived_from"] == [
        "evidence_packet.financials.operating_income",
        "evidence_packet.valuations.dcf.base",
        "evidence_packet.extracted_facts.shares_outstanding",
        "evidence_packet.extracted_facts.r_and_d_total",
        "evidence_packet.fundamentals.rpo_to_revenue_latest",
    ]
    assert "45.6%" in payload["claims"][0]["text"]


def test_synthesis_claim_cleanup_normalizes_aliases_and_drops_index_only_traces(
    monkeypatch, tmp_path
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "MSFT", "2026-02-13")

    class _ClaimCleanupProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "MSFT",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Strong business.",
                "valuation_interpretation": "Mixed valuation.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "R&D supports durability.",
                        "why_it_might_be_true": "Current spending remains healthy.",
                        "falsifiers": ["Margins compress."],
                        "required_evidence": ["Next filing."],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Reportable segments: 3",
                        "type": "non_numeric",
                        "citations": [],
                        "derived_from": [
                            "evidence_packet.fundamentals.r_and_d_pct_revenue",
                            "evidence_packet.financials[0]",
                            "evidence_packet.fundamentals.segment_count",
                        ],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "No change.",
                    "what_is_not_priced": "Optionality.",
                    "uncertainty_notes": "Still limited.",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "next 10-Q",
                        "why": "Confirm segment count.",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _ClaimCleanupProvider())
    path = run_synthesis_for_ticker("MSFT", as_of_date="2026-02-13", run_id="run_claim_cleanup")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["claims"][0]["type"] == "numeric"
    assert payload["claims"][0]["derived_from"] == [
        "evidence_packet.fundamentals.r_and_d_intensity_latest",
        "evidence_packet.fundamentals.segment_count",
    ]


def test_synthesis_promotes_obviously_numeric_claims(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "MSFT", "2026-02-13")

    class _NumericClaimProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "MSFT",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Segment breadth supports resilience.",
                "valuation_interpretation": "Valuation remains mixed.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "Segment count remains stable.",
                        "why_it_might_be_true": "The filing still shows 3 segments.",
                        "falsifiers": ["Segment structure changes."],
                        "required_evidence": ["Next filing."],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Reportable segment count = 3",
                        "type": "non_numeric",
                        "citations": [
                            {
                                "source_url": "https://www.sec.gov/example",
                                "snippet": "segment count 3",
                                "section_label": "segment_info",
                            }
                        ],
                        "derived_from": [],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "No change.",
                    "what_is_not_priced": "Optionality.",
                    "uncertainty_notes": "Still limited.",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "next 10-Q",
                        "why": "Confirm segment count.",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _NumericClaimProvider())
    path = run_synthesis_for_ticker(
        "MSFT", as_of_date="2026-02-13", run_id="run_numeric_claim_type"
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["claims"][0]["type"] == "numeric"


def test_synthesis_disabled_provider_writes_fallback_packet(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    _seed_evidence_packet(cfg, "MSFT", "2026-02-13")

    path = run_synthesis_for_ticker("MSFT", as_of_date="2026-02-13", run_id="run_disabled")
    assert path is not None
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ticker"] == "MSFT"
    assert payload["run_id"] == "run_disabled"
    assert payload["llm_meta"]["model"] == "gpt-5-mini"


def test_synthesis_anthropic_provider_uses_anthropic_model_and_budget(monkeypatch, tmp_path):
    data_dir = tmp_path / "data_anthropic"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("VOE_ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("VOE_ANTHROPIC_MODEL", "claude-test-model")
    monkeypatch.setenv("VOE_ANTHROPIC_BUDGET_USD", "100.0")
    monkeypatch.setenv("VOE_ANTHROPIC_MAX_OUTPUT_TOKENS", "100")
    monkeypatch.setenv("VOE_OPENAI_BUDGET_USD_PER_RUN", "0.000001")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")

    fake = _FakeProvider(provider_name="anthropic")
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: fake)

    path = run_synthesis_for_ticker("AAPL", as_of_date="2026-02-13", run_id="run_anthropic")

    assert path is not None
    assert fake.calls == 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["llm_meta"]["model"] == "claude-test-model"
    _get_config.cache_clear()


def test_synthesis_backfills_required_sections_when_model_omits_them(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")

    class _MissingSectionsProvider(_FakeProvider):
        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            result = super().synthesize_json(prompt=prompt, schema=schema, schema_name=schema_name)
            payload = json.loads(result.json_text)
            payload.pop("claims")
            payload.pop("priced_in_assessment")
            payload.pop("decision_frame")
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr(
        "app.llm.synthesis_agent.get_llm_provider", lambda: _MissingSectionsProvider()
    )

    path = run_synthesis_for_ticker("AAPL", as_of_date="2026-02-13", run_id="run_missing_claims")

    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["claims"][0]["id"] == "c_deterministic_revenue"
    assert payload["claims"][0]["text"] == "Revenue latest $1,000m"
    assert payload["claims"][0]["derived_from"] == ["evidence_packet.fundamentals.revenue"]
    assert (
        payload["priced_in_assessment"]["what_market_assumes"]
        == "Market expectations require manual interpretation because the model omitted its priced-in assessment."
    )
    assert payload["decision_frame"]["stance"] == "watchlist"
    assert payload["decision_frame"]["key_risks"] == ["model_output_missing_decision_frame"]
    assert "model_output_missing_claims" in payload["evidence_gaps"]
    assert "model_output_missing_priced_in_assessment" in payload["evidence_gaps"]
    assert "model_output_missing_decision_frame" in payload["evidence_gaps"]


def test_synthesis_prompt_includes_enrichment_payload(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")
    fake = _FakeProvider()
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: fake)

    path = run_synthesis_for_ticker("AAPL", as_of_date="2026-02-13", run_id="run_enrichment_prompt")
    assert path is not None
    assert '"enrichment"' in fake.last_prompt
    assert '"valuation_spread_analysis"' in fake.last_prompt
    assert '"trend_narratives"' in fake.last_prompt
    assert '"owner_earnings_quality_flags"' in fake.last_prompt
    assert '"moat_signals_summary"' in fake.last_prompt
    assert "For banks, insurers, brokers, and other financial institutions" in fake.last_prompt


def test_synthesis_backfills_only_missing_headline_sections(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")

    class _HeadlineProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "AAPL",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Custom analyst view anchored in product breadth.",
                "valuation_interpretation": "Custom valuation view anchored in growth expectations.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "DCF is $12",
                        "type": "numeric",
                        "citations": [],
                        "derived_from": ["evidence_packet.valuations.dcf.base"],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "z",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "q",
                        "why": "w",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _HeadlineProvider())
    path = run_synthesis_for_ticker("AAPL", as_of_date="2026-02-13", run_id="run_headline_preserve")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["business_quality_summary"].startswith("Custom analyst view")
    assert payload["valuation_interpretation"].startswith("Custom valuation view")


def test_synthesis_backfills_bank_credit_claims_when_model_omits_them(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _MissingBankClaimsProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Large diversified bank.",
                "valuation_interpretation": "Valuation depends on balance-sheet durability.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [],
                "priced_in_assessment": {
                    "what_market_assumes": "x",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "z",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "q",
                        "why": "w",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr(
        "app.llm.synthesis_agent.get_llm_provider", lambda: _MissingBankClaimsProvider()
    )
    path = run_synthesis_for_ticker(
        "JPM", as_of_date="2026-02-13", run_id="run_bank_credit_backfill"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    claim_texts = [str(claim.get("text") or "") for claim in payload["claims"]]

    assert any("Allowance for credit losses" in text for text in claim_texts)
    assert any("Provision for credit losses" in text for text in claim_texts)
    assert any("Net charge-offs" in text for text in claim_texts)


def test_synthesis_polish_removes_stray_valuation_path_fragments(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _PathLeakProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Bank balance sheet is large.",
                "valuation_interpretation": "Absence of deterministic repurchases/dividends prevents connecting capital returns to valuation before per-share evidence_packet.valuations.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Deposits are $100",
                        "type": "numeric",
                        "citations": [],
                        "derived_from": ["evidence_packet.financials.deposits"],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "z",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "q",
                        "why": "w",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _PathLeakProvider())
    path = run_synthesis_for_ticker("JPM", as_of_date="2026-02-13", run_id="run_path_leak_cleanup")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "before per-share valuation work." in payload["valuation_interpretation"]
    assert "evidence_packet.valuations." not in payload["valuation_interpretation"]


def test_synthesis_financial_issuer_rewrites_generic_fcf_gap_language(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _FinancialGapProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Large, diversified bank.",
                "valuation_interpretation": "Given missing FCF/operating-margin inputs, reverse-DCF and DCF outputs are currently UNKNOWN and multiples/DCF ranges are not actionable without further extraction.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": [
                    "fcf and fcf_margin are UNKNOWN — cash-flow bridge and working-capital adjustments need extraction."
                ],
                "recommended_next_actions": [
                    "Extract cash flow bridge and working-capital adjustments from recent filings to compute CFO→FCF conversion"
                ],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Deposits are $100",
                        "type": "numeric",
                        "citations": [],
                        "derived_from": ["evidence_packet.financials.deposits"],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "Market price used in packet.",
                    "what_is_not_priced": "Sustainable free-cash-flow conversion, operating-margin durability, and explicit capital-return amounts (repurchases/dividends) are not priced because FCF, operating_margin, and capital allocation amounts are UNKNOWN or absent from dossier_focus.",
                    "uncertainty_notes": "Valuation is constrained by missing FCF and operating-margin data; reverse-DCF and DCF are UNKNOWN in the packet and multiples guidance warns of insufficient inputs.",
                },
                "next_actions": [
                    {
                        "action_type": "extract_cashflow_bridge",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "Cash Flows section",
                        "why": "To reconcile CFO to FCF and identify one-off working-capital movements flagged in cash_flow_quality signals.",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _FinancialGapProvider())
    path = run_synthesis_for_ticker(
        "JPM", as_of_date="2026-02-13", run_id="run_financial_gap_cleanup"
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert (
        "Traditional FCF-style metrics are sector-limited for bank-like issuers"
        in payload["evidence_gaps"][0]
    )
    assert "compute CFO→FCF conversion" not in payload["recommended_next_actions"][0]
    assert (
        "Traditional FCF-style DCF outputs are less informative for bank-like issuers"
        in payload["valuation_interpretation"]
    )
    assert "generic FCF bridge" in payload["priced_in_assessment"]["what_is_not_priced"]
    assert (
        "traditional FCF-style reverse-DCF checks are less informative for bank-like issuers"
        in payload["priced_in_assessment"]["uncertainty_notes"]
    )
    assert (
        "one-off balance-sheet noise and assess capital-return capacity"
        in payload["next_actions"][0]["why"]
    )


def test_synthesis_polish_strips_trace_refs_from_narrative_and_normalizes_financial_queries(
    monkeypatch, tmp_path
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "MSFT", "2026-02-13")

    class _NarrativeCleanupProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "MSFT",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Revenue visibility is strong (evidence_packet.dossier_focus.revenue_visibility; evidence_packet.extracted_facts.deferred_revenue_amount).",
                "valuation_interpretation": "DCF remains below price (evidence_packet.valuations.dcf.base; evidence_packet.valuations.reverse_dcf.inputs.price).",
                "risk_frame": "Execution risk remains (research_packet.evidence_items covenant signal).",
                "catalyst_frame": "Next filing matters (research_packet.catalysts).",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence (research_packet.quality).",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "Backlog supports durability (evidence_packet.extracted_facts.rpo_amount).",
                        "why_it_might_be_true": "Deferred revenue remains high (evidence_packet.extracted_facts.deferred_revenue_amount).",
                        "falsifiers": ["Backlog falls."],
                        "required_evidence": ["Next filing."],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Revenue is $100.",
                        "type": "numeric",
                        "citations": [],
                        "derived_from": [
                            "evidence_packet.financials[?(@.line_item=='revenue' && @.period=='2025')].value"
                        ],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "No change.",
                    "what_is_not_priced": "Optionality.",
                    "uncertainty_notes": "Still limited.",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "next 10-Q",
                        "why": "Confirm revenue.",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr(
        "app.llm.synthesis_agent.get_llm_provider", lambda: _NarrativeCleanupProvider()
    )
    path = run_synthesis_for_ticker("MSFT", as_of_date="2026-02-13", run_id="run_narrative_cleanup")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["business_quality_summary"] == "Revenue visibility is strong."
    assert payload["valuation_interpretation"] == "DCF remains below price."
    assert payload["risk_frame"] == "Execution risk remains."
    assert payload["confidence_notes"] == "Moderate confidence."
    assert payload["hypotheses"][0]["statement"] == "Backlog supports durability."
    assert payload["claims"][0]["derived_from"] == ["evidence_packet.fundamentals.revenue"]


def test_synthesis_backfills_numeric_claim_derived_from_from_companyfacts_citations(
    monkeypatch, tmp_path
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _CitationOnlyProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Large bank.",
                "valuation_interpretation": "Price is present.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Deposits are $100",
                        "type": "numeric",
                        "citations": [
                            {
                                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json",
                                "snippet": "deposits: 2559320.0",
                                "section_label": "financial_statements",
                            }
                        ],
                        "derived_from": [],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "z",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "q",
                        "why": "w",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _CitationOnlyProvider())
    path = run_synthesis_for_ticker(
        "JPM", as_of_date="2026-02-13", run_id="run_claim_trace_backfill"
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["claims"][0]["derived_from"] == ["evidence_packet.financials.deposits"]


def test_synthesis_polish_strips_inline_reference_leaks(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _InlineLeakProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Large bank.",
                "valuation_interpretation": "Value depends on filing evidence_items ev_1ede5c5735bb57fdc125116e and research_packet.next_actions[0..3].",
                "risk_frame": "Execution risk remains because research_packet.evidence_items covenant signal.",
                "catalyst_frame": "Next filing matters via research_packet.catalysts and evidence_packet.extracted_facts.deposits.",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence via research_packet.quality.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Deposits are $100",
                        "type": "numeric",
                        "citations": [],
                        "derived_from": ["evidence_packet.financials.deposits"],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "z",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "q",
                        "why": "w",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _InlineLeakProvider())
    path = run_synthesis_for_ticker(
        "JPM", as_of_date="2026-02-13", run_id="run_inline_leak_cleanup"
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "evidence_items" not in payload["valuation_interpretation"]
    assert "research_packet.next_actions" not in payload["valuation_interpretation"]
    assert "research_packet.evidence_items" not in payload["risk_frame"]
    assert "evidence_packet.extracted_facts" not in payload["catalyst_frame"]


def test_synthesis_financial_issuer_rewrites_additional_generic_bank_gap_language(
    monkeypatch, tmp_path
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _FinancialGapProvider2:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Large, diversified bank.",
                "valuation_interpretation": "large negative CFO in 2025 increases uncertainty around sustainable free cash generation and the feasibility of repeatable repurchases at that scale.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": [
                    "operating_margin (UNKNOWN) required to judge margin durability and to run canonical DCF/EPV inputs",
                    "fcf (UNKNOWN) — missing FCF prevents canonical DCF/multiples inputs",
                ],
                "recommended_next_actions": [
                    "Extract cash-flow bridge and working-capital adjustments from recent filings to reconcile CFO vs net income"
                ],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Deposits are $100",
                        "type": "numeric",
                        "citations": [],
                        "derived_from": ["evidence_packet.financials.deposits"],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "z",
                },
                "next_actions": [
                    {
                        "action_type": "extract_cashflow_bridge",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "Cash Flows section",
                        "why": "w",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr(
        "app.llm.synthesis_agent.get_llm_provider", lambda: _FinancialGapProvider2()
    )
    path = run_synthesis_for_ticker(
        "JPM", as_of_date="2026-02-13", run_id="run_financial_gap_cleanup2"
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert (
        "Traditional FCF-style metrics are sector-limited for bank-like issuers"
        in payload["evidence_gaps"][1]
    )
    assert "Operating-profitability detail is still incomplete" in payload["evidence_gaps"][0]
    assert (
        "treasury timing items, recurring funding flows, and capital-return capacity"
        in payload["recommended_next_actions"][0]
    )
    assert (
        "recurring liquidity generation, funding mix, and the durability of repurchases"
        in payload["valuation_interpretation"]
    )


def test_synthesis_replaces_generic_numeric_claim_refs_with_specific_paths(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _GenericTraceProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Large bank.",
                "valuation_interpretation": "Price is present.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Revenue is $100.",
                        "type": "numeric",
                        "citations": [
                            {
                                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json",
                                "snippet": "revenue: 182447.0",
                                "section_label": "financial_statements",
                            }
                        ],
                        "derived_from": ["evidence_packet.fundamentals", "evidence_packet.rows"],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "z",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "q",
                        "why": "w",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _GenericTraceProvider())
    path = run_synthesis_for_ticker(
        "JPM", as_of_date="2026-02-13", run_id="run_generic_trace_cleanup"
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert not any(
        claim.get("derived_from") == ["evidence_packet.fundamentals.revenue"]
        for claim in payload["claims"]
    )
    allowance_claim = next(
        claim for claim in payload["claims"] if "Allowance for credit losses" in claim["text"]
    )
    assert allowance_claim["derived_from"] == [
        "evidence_packet.fundamentals.allowance_for_credit_losses"
    ]


def test_synthesis_strips_trace_leaks_from_gap_and_action_lists(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _ListLeakProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Large bank.",
                "valuation_interpretation": "Bank valuation is mixed.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": [
                    "Need research_packet.next_actions A4 and evidence_packet.fundamentals.gaps before underwriting.",
                    "Review extracted_facts.covenant_signal and research_packet.evidence_gaps GAP_SEC_EXHIBITS_NOT_FOUND.",
                ],
                "recommended_next_actions": [
                    "Pull the next filing using research_packet.next_actions A1.",
                    "Recheck evidence_packet.enrichment.liquidity_stress_score and extracted_facts.debt_maturity_signal.",
                ],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["research_packet.next_actions A3"],
                        "required_evidence": ["evidence_packet.fundamentals.gaps"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Deposits are $100",
                        "type": "numeric",
                        "citations": [],
                        "derived_from": ["evidence_packet.financials.deposits"],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "z",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "research_packet.next_actions A1",
                        "why": "evidence_packet.extracted_facts.debt_maturity_signal",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["research_packet.evidence_gaps GAP_UNKNOWN_FCF"],
                    "catalysts": ["evidence_packet.fundamentals.gaps"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _ListLeakProvider())
    path = run_synthesis_for_ticker("JPM", as_of_date="2026-02-13", run_id="run_list_trace_cleanup")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    joined = " ".join(
        payload["evidence_gaps"]
        + payload["recommended_next_actions"]
        + payload["hypotheses"][0]["falsifiers"]
        + payload["hypotheses"][0]["required_evidence"]
    )
    assert "research_packet." not in joined
    assert "evidence_packet." not in joined
    assert "extracted_facts." not in joined


def test_synthesis_normalizes_large_bank_money_mentions(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _MoneyProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Deposits were $2,559,320k and CFO was -$147,782,000,000 USD.",
                "valuation_interpretation": "Net debt is -$8,864,000,000 USD and repurchases were $31,591,000,000.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "CFO was -$147,782,000,000 USD.",
                        "type": "numeric",
                        "citations": [],
                        "derived_from": ["evidence_packet.fundamentals.cfo"],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "z",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "q",
                        "why": "w",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _MoneyProvider())
    path = run_synthesis_for_ticker(
        "JPM", as_of_date="2026-02-13", run_id="run_money_normalization"
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "$2,559,320k" in payload["business_quality_summary"]
    assert "-$147.8b" in payload["business_quality_summary"]
    assert "-$8.9b" in payload["valuation_interpretation"]
    assert "$31.6b" in payload["valuation_interpretation"]
    assert "-$147.8b" in payload["claims"][0]["text"]


def test_synthesis_financial_issuer_prioritizes_bank_first_leads(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _BankLeadProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Large bank with active capital returns.",
                "valuation_interpretation": "Canonical DCF-style per-share valuations are limited by missing FCF and operating-margin inputs.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Deposits are $100",
                        "type": "numeric",
                        "citations": [],
                        "derived_from": ["evidence_packet.financials.deposits"],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "z",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "q",
                        "why": "w",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _BankLeadProvider())
    path = run_synthesis_for_ticker("JPM", as_of_date="2026-02-13", run_id="run_bank_first_leads")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["business_quality_summary"].startswith(
        "For a bank-like issuer, business quality is best judged"
    )
    assert payload["valuation_interpretation"].startswith(
        "For a large bank, valuation should start with deposits"
    )
    assert payload["risk_frame"].startswith("Bank-specific risk is driven first by funding mix")


def test_synthesis_rewrites_remaining_systemish_phrases(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _SystemishProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Large bank with segments_signal and material_weakness flagged in filings.",
                "valuation_interpretation": "research packet identifies the next catalyst.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Deposits are $100",
                        "type": "numeric",
                        "citations": [],
                        "derived_from": ["evidence_packet.financials.deposits"],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "z",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "q",
                        "why": "w",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _SystemishProvider())
    path = run_synthesis_for_ticker("JPM", as_of_date="2026-02-13", run_id="run_systemish_cleanup")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "segments_signal" not in payload["business_quality_summary"]
    assert "segment reporting" in payload["business_quality_summary"]
    assert "material_weakness flagged" not in payload["business_quality_summary"]
    assert "material-weakness disclosure" in payload["business_quality_summary"]
    assert "research packet identifies" not in payload["valuation_interpretation"]
    assert "current research identifies" in payload["valuation_interpretation"]


def test_synthesis_prefers_tighter_support_for_cagr_claims(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _CagrProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Large bank.",
                "valuation_interpretation": "Mixed valuation.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Revenue 3-year CAGR: 12.34%",
                        "type": "numeric",
                        "citations": [
                            {
                                "source_url": "https://www.sec.gov/Archives/edgar/data/19617/000162828026008131/jpm-20251231.htm",
                                "snippet": "very long unrelated principal transactions table",
                                "section_label": None,
                            }
                        ],
                        "derived_from": [
                            "evidence_packet.fundamentals.revenue",
                            "evidence_packet.financials.revenue",
                        ],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "z",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "q",
                        "why": "w",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _CagrProvider())
    path = run_synthesis_for_ticker(
        "JPM", as_of_date="2026-02-13", run_id="run_cagr_support_cleanup"
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert any("Allowance for credit losses" in claim["text"] for claim in payload["claims"])
    assert any("Provision for credit losses" in claim["text"] for claim in payload["claims"])
    assert any("Net charge-offs" in claim["text"] for claim in payload["claims"])


def test_synthesis_strips_bracket_semicolon_corruption_from_narrative(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _CorruptionProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Large bank [~12.34%, 5y CAGR ~8.82%) supporting continued top-line expansion [;;",
                "valuation_interpretation": "Net debt is negative [; and capital returns matter [;;",
                "risk_frame": "Execution risk remains [;;",
                "catalyst_frame": "Next filing matters [;",
                "evidence_gaps": ["Need updated filing [;"],
                "recommended_next_actions": ["Read the next 10-Q [;;"],
                "confidence_notes": "Moderate confidence [;",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test [;",
                        "why_it_might_be_true": "test [;;",
                        "falsifiers": ["x [;"],
                        "required_evidence": ["y [;;"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Deposits are $100",
                        "type": "numeric",
                        "citations": [],
                        "derived_from": ["evidence_packet.financials.deposits"],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x [;",
                    "what_is_not_priced": "y [;;",
                    "uncertainty_notes": "z [;",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "q [;",
                        "why": "w [;;",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution [;"],
                    "catalysts": ["next 10-Q [;;"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _CorruptionProvider())
    path = run_synthesis_for_ticker("JPM", as_of_date="2026-02-13", run_id="run_corruption_cleanup")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    blob = " ".join(
        [
            payload["business_quality_summary"],
            payload["valuation_interpretation"],
            payload["risk_frame"],
            payload["catalyst_frame"],
            payload["evidence_gaps"][0],
            payload["recommended_next_actions"][0],
            payload["hypotheses"][0]["statement"],
            payload["hypotheses"][0]["why_it_might_be_true"],
            payload["priced_in_assessment"]["what_market_assumes"],
            payload["next_actions"][0]["why"],
            payload["decision_frame"]["key_risks"][0],
        ]
    )
    assert "[" not in blob
    assert ";;" not in blob


def test_claim_unit_normalization_preserves_million_scale_values():
    evidence_packet = {
        "fundamentals": {"revenue": 182447.0},
        "financials": [
            {
                "line_item": "revenue",
                "units": "USD_millions",
            }
        ],
    }
    claim = {
        "id": "c1",
        "text": "2025 revenue: $182,447m. Share repurchases in 2025: $31,591m. Dividends paid in 2025: $16,625m.",
        "type": "numeric",
        "citations": [
            {
                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json",
                "snippet": "revenue: 182447.0 (in millions)",
                "section_label": "financial_statements",
            }
        ],
        "derived_from": ["evidence_packet.fundamentals.revenue"],
    }

    accepted = _normalize_claim_text_units(
        claim,
        evidence_packet=evidence_packet,
    )

    assert accepted is True
    assert "$182,447m" in claim["text"]
    assert "$31,591m" in claim["text"]
    assert "$16,625m" in claim["text"]
    assert "$0mm" not in claim["text"]


def test_narrative_cleanup_removes_systemish_fragments_without_distorting_amounts():
    text = (
        "Filings show debt maturities and covenant discussion and material-weakness language "
        "in internal-control disclosures (extracted_facts debt_maturity and material_weakness signals), "
        "while capital_allocation remains active. Deposits = $2,559,320m and share repurchases $31,591m."
    )

    cleaned = _polish_narrative_text(text)

    assert "extracted_facts" not in cleaned
    assert "capital_allocation" not in cleaned
    assert "debt_maturity" not in cleaned
    assert "material_weakness" not in cleaned
    assert "debt maturity" in cleaned
    assert "material weakness" in cleaned
    assert "$2,559,320m" in cleaned
    assert "$31,591m" in cleaned


def test_rendered_text_sanitizer_removes_duplicate_units_and_internal_phrasing():
    cleaned = _polish_narrative_text(
        "Share repurchases were $31,591m million. Dividends were $16,625m million. "
        "Net debt was -$8,864million and deposits = 2,559,320 mm. "
        "Research packet flags cash_flow_quality signals and debt maturity_signal; covenant_signal. "
        "Large bank with segment disclosures present and companyfacts feed references."
    )

    assert "m million" not in cleaned
    assert "-$8,864m" in cleaned
    assert "2,559,320 mm" not in cleaned
    assert "2,559,320m" in cleaned
    assert "research packet" not in cleaned.lower()
    assert "companyfacts feed" not in cleaned.lower()
    assert "segment disclosures" not in cleaned.lower()
    assert "segment reporting" in cleaned.lower()
    assert "cash-flow quality disclosures" in cleaned.lower()
    assert "debt maturity and covenant disclosures" in cleaned.lower()


def test_claim_unit_normalization_formats_packet_million_values_from_raw_integers():
    evidence_packet = {
        "fundamentals": {
            "revenue": 182447.0,
            "cfo": -147782.0,
            "share_repurchases_amount": 31591.0,
            "dividends_paid_amount": 16625.0,
        },
        "financials": [
            {"line_item": metric, "units": "USD_millions"}
            for metric in (
                "revenue",
                "cfo",
                "share_repurchases_amount",
                "dividends_paid_amount",
            )
        ],
    }
    claims = [
        {
            "id": "c1",
            "text": "Revenue (2025, companyfacts): 182,447,000,000",
            "type": "numeric",
            "citations": [
                {
                    "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json",
                    "snippet": "revenue: 182447.0",
                    "section_label": "financial_statements",
                }
            ],
            "derived_from": ["evidence_packet.fundamentals.revenue"],
        },
        {
            "id": "c2",
            "text": "CFO (2025, companyfacts): -147,782,000,000",
            "type": "numeric",
            "citations": [
                {
                    "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json",
                    "snippet": "cfo: -147782.0",
                    "section_label": "financial_statements",
                }
            ],
            "derived_from": ["evidence_packet.fundamentals.cfo"],
        },
        {
            "id": "c3",
            "text": "Share repurchases in 2025: 31,591,000,000",
            "type": "numeric",
            "citations": [
                {
                    "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json",
                    "snippet": "share_repurchases_amount: 31591.0",
                    "section_label": "financial_statements",
                }
            ],
            "derived_from": ["evidence_packet.fundamentals.share_repurchases_amount"],
        },
        {
            "id": "c4",
            "text": "Dividends paid in 2025: 16,625,000,000",
            "type": "numeric",
            "citations": [
                {
                    "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json",
                    "snippet": "dividends_paid_amount: 16625.0",
                    "section_label": "financial_statements",
                }
            ],
            "derived_from": ["evidence_packet.fundamentals.dividends_paid_amount"],
        },
    ]

    for claim in claims:
        assert (
            _normalize_claim_text_units(
                claim,
                evidence_packet=evidence_packet,
            )
            is True
        )

    assert claims[0]["text"] == "Revenue (2025, companyfacts): $182,447m"
    assert claims[1]["text"] == "CFO (2025, companyfacts): -$147,782m"
    assert claims[2]["text"] == "Share repurchases in 2025: $31,591m"
    assert claims[3]["text"] == "Dividends paid in 2025: $16,625m"


def test_claim_unit_normalization_formats_market_price_and_net_debt():
    evidence_packet = {
        "fundamentals": {
            "market_price": 302.55,
            "net_debt": -8_864_000_000.0,
        },
        "financials": [
            {
                "line_item": "market_price",
                "units": "USD_per_share",
            },
            {
                "line_item": "net_debt",
                "units": "USD",
            },
        ],
    }
    price_claim = {
        "id": "c1",
        "text": "Market price used in packet: 302.55",
        "type": "numeric",
        "citations": [
            {
                "source_url": "https://stooq.com/q/l/",
                "snippet": "price: 302.55",
                "section_label": "valuations",
            }
        ],
        "derived_from": ["evidence_packet.valuations.reverse_dcf.inputs.price"],
    }
    net_debt_claim = {
        "id": "c2",
        "text": "Net debt (companyfacts-derived): -8,864,000,000",
        "type": "numeric",
        "citations": [
            {
                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json",
                "snippet": "net_debt: -8864000000.0",
                "section_label": "financial_statements",
            }
        ],
        "derived_from": ["evidence_packet.fundamentals.net_debt"],
    }

    assert (
        _normalize_claim_text_units(
            price_claim,
            evidence_packet=evidence_packet,
        )
        is True
    )
    assert (
        _normalize_claim_text_units(
            net_debt_claim,
            evidence_packet=evidence_packet,
        )
        is True
    )

    assert "$302.55" in price_claim["text"]
    assert "-$8.9b" in net_debt_claim["text"]


def test_claim_unit_normalization_rejects_missing_units():
    evidence_packet = {
        "fundamentals": {"revenue": 750000.0},
        "financials": [{"line_item": "revenue", "units": None}],
    }
    claim = {
        "id": "c1",
        "text": "Revenue was 750,000.",
        "type": "numeric",
        "citations": [
            {
                "source_url": "https://example.test/companyfacts",
                "snippet": "revenue: 750000.0",
                "section_label": "financial_statements",
            }
        ],
        "derived_from": ["evidence_packet.fundamentals.revenue"],
    }

    accepted = _normalize_claim_text_units(
        claim,
        evidence_packet=evidence_packet,
    )

    assert accepted is False
    assert claim["unit_integrity_status"] == "INVALID_FINANCIAL_INPUT"
    assert claim["unit_integrity_reason"] == "MISSING_OR_AMBIGUOUS_UNIT"
    assert claim["text"] == "Revenue was 750,000."


def test_claim_unit_normalization_rejects_ambiguous_units():
    evidence_packet = {
        "fundamentals": {"revenue": 750000.0},
        "financials": [
            {"line_item": "revenue", "units": "USD"},
            {"line_item": "revenue", "units": "USD_millions"},
        ],
    }
    claim = {
        "id": "c1",
        "text": "Revenue was 750,000.",
        "type": "numeric",
        "citations": [
            {
                "source_url": "https://example.test/companyfacts",
                "snippet": "revenue: 750000.0",
                "section_label": "financial_statements",
            }
        ],
        "derived_from": ["evidence_packet.fundamentals.revenue"],
    }

    accepted = _normalize_claim_text_units(
        claim,
        evidence_packet=evidence_packet,
    )

    assert accepted is False
    assert claim["unit_integrity_status"] == "INVALID_FINANCIAL_INPUT"
    assert claim["unit_integrity_reason"] == "MISSING_OR_AMBIGUOUS_UNIT"


def test_claim_unit_normalization_formats_exact_raw_usd_below_one_million():
    evidence_packet = {
        "fundamentals": {"cfo": 750000.0},
        "financials": [{"line_item": "cfo", "units": "USD"}],
    }
    claim = {
        "id": "c1",
        "text": "CFO was 750,000 USD.",
        "type": "numeric",
        "citations": [
            {
                "source_url": "https://example.test/companyfacts",
                "snippet": "cfo: 750000.0",
                "section_label": "financial_statements",
            }
        ],
        "derived_from": ["evidence_packet.fundamentals.cfo"],
    }

    accepted = _normalize_claim_text_units(
        claim,
        evidence_packet=evidence_packet,
    )

    assert accepted is True
    assert claim["text"] == "CFO was $750,000."


def test_money_normalization_preserves_explicit_thousand_unit():
    normalized = _normalize_money_mentions("Liquidity was $500k.")

    assert normalized == "Liquidity was $500k."
    assert "$500m" not in normalized


def test_claim_unit_normalization_rejects_conflicting_packet_and_citation_units():
    evidence_packet = {
        "fundamentals": {
            "cfo": -147782.0,
        },
        "financials": [
            {
                "line_item": "cfo",
                "units": "USD_millions",
            }
        ],
    }
    claim = {
        "id": "c1",
        "text": "Net cash provided by/(used in) operating activities (CFO) = -147,782,000,000 USD",
        "type": "numeric",
        "citations": [
            {
                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json",
                "snippet": "cfo: -147782000000.0",
                "section_label": "financial_statements",
            }
        ],
        "derived_from": ["evidence_packet.fundamentals.cfo"],
    }

    accepted = _normalize_claim_text_units(claim, evidence_packet=evidence_packet)

    assert accepted is False
    assert claim["unit_integrity_status"] == "INVALID_FINANCIAL_INPUT"
    assert claim["text"] == (
        "Net cash provided by/(used in) operating activities (CFO) = -147,782,000,000 USD"
    )


def test_polish_narrative_text_strips_trailing_sources_residue():
    cleaned = _polish_narrative_text(
        "Active capital returns support valuation. Sources; share_repurchases_amount, dividends_paid_amount),"
    )
    assert "Sources;" not in cleaned
    assert "share_repurchases_amount" not in cleaned


def test_polish_narrative_text_rewrites_last_internal_financial_field_names():
    cleaned = _polish_narrative_text(
        "The enrichment layer classifies the posture as shareholder_return_active and operating_margin_trend_slope is positive; "
        "allowance_for_credit_losses (25,765) supports reserve coverage."
    )
    assert "shareholder_return_active" not in cleaned
    assert "operating_margin_trend_slope" not in cleaned
    assert "allowance_for_credit_losses" not in cleaned
    assert "shareholder-return active" in cleaned
    assert "operating-margin trend" in cleaned
    assert "allowance for credit losses" in cleaned


def test_polish_narrative_text_collapses_decimal_million_artifacts():
    cleaned = _polish_narrative_text(
        "Share repurchases were $31,591m.0 million and deposits were $2,559,320m.0; loans equal 1,408,905.0 (millions USD)."
    )
    assert "$31,591m" in cleaned
    assert "$2,559,320m" in cleaned
    assert "$1,408,905m" in cleaned
    assert "m.0" not in cleaned
    assert "millions USD" not in cleaned


def test_synthesis_snapshot_freeze_point_for_jpm_like_output(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _SnapshotProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": (
                    "Large bank with segment signals present and cash_flow_quality and debt_liquidity signals. "
                    "Capital allocation remains active; share repurchases 31,591,000,000 and dividends 16,625,000,000."
                ),
                "valuation_interpretation": (
                    "Market price 302.55 and companyfacts-derived net debt -8,864,000,000 are present, "
                    "but fundamentals.gaps and sbc_dilution_signal remain."
                ),
                "risk_frame": "material_weakness flagged in filings and debt_maturity signals remain important.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": ["Need companyfacts loans and no 8-K exhibits yet."],
                "recommended_next_actions": ["Read the next 10-Q and 8-K exhibits."],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "cash_flow_quality matters",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Market price used in packet: 302.55",
                        "type": "numeric",
                        "citations": [
                            {
                                "source_url": "https://stooq.com/q/l/",
                                "snippet": "price: 302.55",
                                "section_label": "valuations",
                            }
                        ],
                        "derived_from": ["evidence_packet.valuations.reverse_dcf.inputs.price"],
                    },
                    {
                        "id": "c2",
                        "text": "Revenue (2025, companyfacts): 182,447,000,000",
                        "type": "numeric",
                        "citations": [
                            {
                                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json",
                                "snippet": "revenue: 182447.0",
                                "section_label": "financial_statements",
                            }
                        ],
                        "derived_from": ["evidence_packet.fundamentals.revenue"],
                    },
                    {
                        "id": "c3",
                        "text": "Share repurchases in 2025 (companyfacts): 31,591,000,000",
                        "type": "numeric",
                        "citations": [
                            {
                                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json",
                                "snippet": "share_repurchases_amount: 31591.0",
                                "section_label": "financial_statements",
                            }
                        ],
                        "derived_from": ["evidence_packet.fundamentals.share_repurchases_amount"],
                    },
                    {
                        "id": "c4",
                        "text": "Revenue 3-year CAGR: 12.34%",
                        "type": "numeric",
                        "citations": [
                            {
                                "source_url": "https://www.sec.gov/Archives/edgar/data/19617/000162828026008131/jpm-20251231.htm",
                                "snippet": "very long unrelated principal transactions table",
                                "section_label": None,
                            }
                        ],
                        "derived_from": ["evidence_packet.fundamentals.revenue"],
                    },
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "fundamentals.gaps remain",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "companyfacts loans",
                        "why": "cash_flow_quality and sbc_dilution_signal",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["debt_liquidity and material_weakness"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _SnapshotProvider())
    path = run_synthesis_for_ticker("JPM", as_of_date="2026-02-13", run_id="run_jpm_freeze_point")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    blob = json.dumps(payload)
    assert "cash_flow_quality" not in blob
    assert "debt_liquidity" not in blob
    assert "fundamentals.gaps" not in blob
    assert "sbc_dilution_signal" not in blob
    assert "segment signals" not in blob
    assert "research packet signal" not in blob
    assert "&D are unavailable" not in blob
    assert "segment reporting across filings" in payload["business_quality_summary"]
    retained_paths = {tuple(claim.get("derived_from") or []) for claim in payload["claims"]}
    assert ("evidence_packet.valuations.reverse_dcf.inputs.price",) not in retained_paths
    assert ("evidence_packet.fundamentals.revenue",) not in retained_paths
    assert ("evidence_packet.fundamentals.share_repurchases_amount",) not in retained_paths
    assert any("Allowance for credit losses" in claim["text"] for claim in payload["claims"])
    assert any("Provision for credit losses" in claim["text"] for claim in payload["claims"])
    assert any("Net charge-offs" in claim["text"] for claim in payload["claims"])
    assert len(payload["claims"]) == 3


def test_synthesis_financial_issuer_backfills_allowance_claim(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _AllowanceProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Large bank with deposits and loans.",
                "valuation_interpretation": "Traditional DCF inputs are incomplete.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Loans = $1,411,992m USD",
                        "type": "numeric",
                        "citations": [],
                        "derived_from": ["evidence_packet.fundamentals.loans"],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "z",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "q",
                        "why": "w",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _AllowanceProvider())
    path = run_synthesis_for_ticker("JPM", as_of_date="2026-02-13", run_id="run_allowance_backfill")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert any("Allowance for credit losses" in claim["text"] for claim in payload["claims"])


def test_synthesis_replaces_internal_claim_citations_with_real_packet_sources(
    monkeypatch, tmp_path
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "JPM", "2026-02-13")

    class _InternalCitationProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "JPM",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Large bank.",
                "valuation_interpretation": "Price is present.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Share repurchases amount: $31,591.00 million.",
                        "type": "numeric",
                        "citations": [
                            {
                                "source_url": "evidence_packet.dossier_focus.capital_allocation.share_repurchases_amount",
                                "snippet": "share_repurchases_amount: 31591.0",
                                "section_label": None,
                            }
                        ],
                        "derived_from": ["evidence_packet.dossier_focus.capital_allocation"],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "z",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "q",
                        "why": "w",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr(
        "app.llm.synthesis_agent.get_llm_provider", lambda: _InternalCitationProvider()
    )
    path = run_synthesis_for_ticker(
        "JPM", as_of_date="2026-02-13", run_id="run_internal_citation_cleanup"
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    citation = payload["claims"][0]["citations"][0]
    assert citation["source_url"].startswith("https://")
    assert "evidence_packet." not in citation["source_url"]


def test_synthesis_refreshes_canonical_research_when_only_stale_fallback_exists(
    monkeypatch, tmp_path
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")
    with get_db() as conn:
        conn.execute(
            "DELETE FROM analyst_outputs WHERE ticker = 'AAPL' AND as_of_date = '2026-02-13'"
        )
    _seed_legacy_research_packet(
        cfg,
        ticker="AAPL",
        as_of_date="2026-02-13",
        run_id="stale_run",
        suffix="stale_run",
        summary="old legacy summary",
        source_url="https://www.sec.gov/old",
    )

    class _RefreshAwareProvider(_FakeProvider):
        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            result = super().synthesize_json(prompt=prompt, schema=schema, schema_name=schema_name)
            self.last_prompt = prompt
            return result

    def _fake_refresh(**kwargs):
        assert kwargs["ticker"] == "AAPL"
        assert kwargs["run_id"] == "fresh_run"
        _seed_analysis_report(cfg, ticker="AAPL", as_of_date="2026-02-13")
        out_dir = cfg.analyst_outputs_dir / "AAPL_2026-02-13"
        report_path = out_dir / "analysis_report.json"
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["thesis_summary"] = "Fresh canonical analyst summary."
        payload["citations"][0]["source_url"] = "https://www.sec.gov/new"
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return report_path

    fake = _RefreshAwareProvider()
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: fake)
    monkeypatch.setattr("app.research.run_research_agent_for_ticker", _fake_refresh)
    path = run_synthesis_for_ticker("AAPL", as_of_date="2026-02-13", run_id="fresh_run")
    assert path is not None
    assert "Fresh canonical analyst summary." in fake.last_prompt
    assert "https://www.sec.gov/new" in fake.last_prompt


def test_synthesis_strips_derived_from_annotations_from_narrative_fields(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "MSFT", "2026-02-13")

    class _DerivedFromNarrativeProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "MSFT",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Revenue visibility is strong, derived_from: dossier_focus.revenue_visibility.",
                "valuation_interpretation": "DCF remains below price, derived_from: evidence_packet.valuations.dcf.base.",
                "risk_frame": "Execution risk remains, derived_from: analysis_context.citations covenant signal.",
                "catalyst_frame": "Next filing matters, derived_from: analysis_context.recent_event_impacts.",
                "evidence_gaps": ["Need updated filing from analysis_context.research_quality."],
                "recommended_next_actions": [
                    "Read the next 10-Q, derived_from: analysis_context.next_actions A1."
                ],
                "confidence_notes": "Moderate confidence, derived_from: analysis_context.research_quality.",
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "test",
                        "why_it_might_be_true": "test",
                        "falsifiers": ["x"],
                        "required_evidence": ["y"],
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Revenue is $100",
                        "type": "numeric",
                        "citations": [],
                        "derived_from": ["evidence_packet.financials.revenue"],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "x, derived_from: evidence_packet.valuations.reverse_dcf.inputs.price.",
                    "what_is_not_priced": "y",
                    "uncertainty_notes": "z, derived_from: analysis_context.next_actions A3.",
                },
                "next_actions": [
                    {
                        "action_type": "edgar_extract",
                        "target_source": "EDGAR",
                        "query_or_url_hint": "next 10-Q, derived_from: analysis_context.next_actions A1.",
                        "why": "Confirm revenue, derived_from: evidence_packet.extracted_facts.revenue.",
                    }
                ],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution, derived_from: analysis_context.citations covenant"],
                    "catalysts": ["next 10-Q, derived_from: analysis_context.recent_event_impacts"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr(
        "app.llm.synthesis_agent.get_llm_provider", lambda: _DerivedFromNarrativeProvider()
    )
    path = run_synthesis_for_ticker(
        "MSFT", as_of_date="2026-02-13", run_id="run_derived_from_narrative_cleanup"
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "derived_from:" not in payload["business_quality_summary"]
    assert "derived_from:" not in payload["valuation_interpretation"]
    assert "derived_from:" not in payload["confidence_notes"]
    assert "derived_from:" not in payload["priced_in_assessment"]["what_market_assumes"]
    assert "derived_from:" not in payload["next_actions"][0]["why"]
    assert "derived_from:" not in payload["decision_frame"]["key_risks"][0]
    assert "analysis_context." not in " ".join(
        payload["evidence_gaps"]
        + payload["recommended_next_actions"]
        + payload["decision_frame"]["key_risks"]
    )


def test_append_synthesis_section_renders_memo_format(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")
    fake = _FakeProvider()
    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: fake)
    path = run_synthesis_for_ticker("AAPL", as_of_date="2026-02-13", run_id="run_render")
    assert path is not None

    dossier_md = tmp_path / "dossier.md"
    dossier_md.write_text("# Dossier\n", encoding="utf-8")
    append_synthesis_section(ticker="AAPL", run_id="run_render", dossier_md_path=str(dossier_md))

    output = dossier_md.read_text(encoding="utf-8")
    assert "## Layer 3: Qualitative Synthesis" in output
    assert "### Verdict" in output
    assert "WATCHLIST -" in output
    assert "| Method | Intrinsic Value | Current Price | Gap |" in output
    assert "| DCF | $12.00 | UNKNOWN | UNKNOWN |" in output
    assert "### Open Hypotheses" in output
    assert "### Confidence & Evidence Gaps" in output


def test_synthesis_backfills_missing_hypotheses_and_next_actions(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "MSFT", "2026-02-13")

    class _SparseProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
            _ = (prompt, schema, schema_name)
            payload = {
                "ticker": "MSFT",
                "as_of_date": "2026-02-13",
                "run_id": "placeholder",
                "business_quality_summary": "Sparse model output.",
                "valuation_interpretation": "Sparse valuation output.",
                "risk_frame": "Execution risk remains.",
                "catalyst_frame": "Next filing matters.",
                "evidence_gaps": ["Need updated filing."],
                "recommended_next_actions": ["Read the next 10-Q."],
                "confidence_notes": "Moderate confidence.",
                "hypotheses": [],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Deferred revenue remains material.",
                        "type": "non_numeric",
                        "citations": [],
                        "derived_from": [
                            "evidence_packet.dossier_focus.revenue_visibility.deferred_revenue_amount"
                        ],
                    }
                ],
                "priced_in_assessment": {
                    "what_market_assumes": "No change.",
                    "what_is_not_priced": "Optionality.",
                    "uncertainty_notes": "Still limited.",
                },
                "next_actions": [],
                "decision_frame": {
                    "stance": "watchlist",
                    "key_risks": ["execution"],
                    "catalysts": ["next 10-Q"],
                    "time_horizon_days": 90,
                },
                "llm_meta": {
                    "model": "gpt-5-mini",
                    "prompt_hash": "placeholder",
                    "input_hash": "placeholder",
                    "cost_estimate_usd": 0.0,
                    "created_at": "2026-02-13T00:00:00+00:00",
                },
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    monkeypatch.setattr("app.llm.synthesis_agent.get_llm_provider", lambda: _SparseProvider())
    path = run_synthesis_for_ticker("MSFT", as_of_date="2026-02-13", run_id="run_sparse_model")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["hypotheses"]
    assert payload["hypotheses"][0]["id"] == "H_fallback_visibility"
    assert "deferred revenue is $400m" in payload["hypotheses"][0]["why_it_might_be_true"]
    assert payload["next_actions"]
    assert payload["next_actions"][0]["action_type"] == "pull_filings"


def test_synthesis_openai_without_key_raises_clear_error(monkeypatch, tmp_path):
    data_dir = tmp_path / "data_missing_key"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.delenv("VOE_OPENAI_API_KEY", raising=False)
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")

    try:
        run_synthesis_for_ticker("AAPL", as_of_date="2026-02-13", run_id="run_missing_key")
        raise AssertionError("expected RuntimeError for missing API key")
    except RuntimeError as exc:
        text = str(exc)
        assert "VOE_OPENAI_API_KEY" in text


def test_synthesis_anthropic_without_key_raises_clear_error(monkeypatch, tmp_path):
    data_dir = tmp_path / "data_missing_anthropic_key"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("VOE_ANTHROPIC_API_KEY", raising=False)
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")

    try:
        run_synthesis_for_ticker(
            "AAPL", as_of_date="2026-02-13", run_id="run_missing_anthropic_key"
        )
        raise AssertionError("expected RuntimeError for missing API key")
    except RuntimeError as exc:
        text = str(exc)
        assert "VOE_ANTHROPIC_API_KEY" in text
    _get_config.cache_clear()
