from __future__ import annotations

import json

import pytest

from app.alpha.llm_runtime import (
    AlphaCandidateLoopResult,
    _apply_deterministic_evidence_gates,
    decide_alpha_winner,
    get_alpha_llm_provider,
    plan_alpha_investigations,
    run_candidate_investigation,
)
from app.alpha.llm_tools import AlphaToolContext, dispatch_alpha_tool
from app.alpha.schemas import TickerSignalPacket
from app.autonomous.financial_integrity import FinancialIntegrityScope, stable_quote_hash
from app.db import get_db, init_db
from app.llm.providers.disabled_provider import LLMResult
from app.research.adapters.base import AdapterResult
from app.research.current_event_context import CurrentEventContext, CurrentEventDocument
from app.research.filing_context import FilingContext, FilingDocument
from app.research.schemas import CitationRef, EvidenceItem
from tests.financial_integrity_helpers import canonicalize_financial_packet


def _packet(ticker: str = "AAA") -> TickerSignalPacket:
    return TickerSignalPacket(
        ticker=ticker,
        dcf_value=150.0,
        epv_value=120.0,
        current_price=100.0,
        gate_verdict="PROCEED",
        confidence_class="HIGH",
        moat_score=4,
        moat_classification="MODERATE_MOAT",
        valuation_supports=["PEER_LEADER_SUPPORT"],
        valuation_headwinds=["DILUTION_HEADWIND"],
        research_report={
            "anomalies": [],
            "solvency": {
                "risk": "LOW",
                "signals": [],
                "details": "",
                "going_concern_language": False,
                "no_assurance_financing": False,
            },
        },
    )


def _init_companyfacts_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _integrity_scope(
    packet: TickerSignalPacket,
    *,
    as_of_date: str = "2026-04-19",
) -> FinancialIntegrityScope:
    packet.current_price_unit = "USD_per_share"
    packet.current_price_as_of_date = as_of_date
    packet.current_price_currency = "USD"
    packet.current_price_source = "test_quote"
    packet.price_basis = "UNADJUSTED"
    packet.raw_price = packet.current_price
    packet.split_adjustment_factor = 1.0
    packet.shares_outstanding_mm = 10.0
    packet.shares_unit = "shares_millions"
    packet.shares_basis = "UNADJUSTED"
    packet.shares_as_of_date = as_of_date
    packet.shares_source = "test_filing"
    packet.split_lineage_proof = {
        "status": "PASS",
        "period_start": as_of_date,
        "period_end": as_of_date,
        "verified_as_of": as_of_date,
        "source": "test_corporate_actions_ledger",
        "source_reference": f"test://corporate-actions/{packet.ticker}",
    }
    packet.issuer_quote_ratio = 1.0
    packet.market_cap_mm = float(packet.current_price or 0.0) * 10.0
    packet.market_cap_unit = "USD_millions"
    packet.market_cap_source = "derived_from_quote_and_shares"
    packet.market_cap_method = "price_times_shares"
    packet.market_cap_effective_as_of_date = as_of_date
    packet.quote_snapshot_id = stable_quote_hash(
        ticker=packet.ticker,
        price=packet.current_price,
        as_of_date=as_of_date,
        currency="USD",
        source="test_quote",
        price_basis="UNADJUSTED",
        raw_price=packet.current_price,
        split_adjustment_factor=1.0,
    )
    packet.cap_stage_price = packet.current_price
    packet.cap_stage_price_as_of_date = as_of_date
    packet.cap_stage_price_currency = "USD"
    packet.cap_stage_price_source = "test_quote"
    packet.cap_stage_quote_snapshot_id = packet.quote_snapshot_id
    canonicalize_financial_packet(packet, as_of_date=as_of_date, shares_mm=10.0)
    return FinancialIntegrityScope(
        context="alpha_test",
        run_as_of_date=as_of_date,
        packets=(packet,),
    )


def test_alpha_planner_revalidates_exact_scope_before_provider_retry(monkeypatch):
    from app.autonomous.financial_integrity import InvalidFinancialInputError
    from app.llm.providers.retry_guard import call_with_llm_retry_guard

    class _RetryableProviderError(RuntimeError):
        status_code = 503

    packet = _packet()
    scope = _integrity_scope(packet)
    physical_calls = 0

    class _RetryingProvider:
        provider_name = "openai"

        def enabled(self):
            return True

        def synthesize_json(self, **_kwargs):
            def physical_call():
                nonlocal physical_calls
                physical_calls += 1
                if physical_calls == 1:
                    packet.moat_score = 5
                    raise _RetryableProviderError("temporary provider failure")
                return LLMResult(
                    json_text=json.dumps(
                        {
                            "summary": "retry should never publish",
                            "selected_targets": [],
                            "skipped_candidates": [],
                        }
                    ),
                    model="fake",
                    usage_input_tokens=0,
                    usage_output_tokens=0,
                    raw={},
                )

            return call_with_llm_retry_guard(
                provider_name="openai",
                schema_name="alpha_shortlist_plan_v1",
                call=physical_call,
                max_retries=1,
                backoff_seconds=(0.0,),
                sleep_fn=lambda _seconds: None,
            )

    monkeypatch.setattr(
        "app.alpha.llm_runtime.get_alpha_llm_provider",
        lambda: _RetryingProvider(),
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        plan_alpha_investigations(
            sector="software",
            prior_candidates=[],
            hard_blocked_candidates=[],
            max_targets=1,
            integrity_scope=scope,
        )

    assert physical_calls == 1
    assert {item.code for item in exc_info.value.result.violations} == {
        "BOUND_FINANCIAL_INPUT_MUTATED"
    }


def test_alpha_provider_prefers_configured_openai_over_anthropic_key(monkeypatch):
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("VOE_ANTHROPIC_API_KEY", "anthropic-test-key")
    from app.config import get_config

    get_config.cache_clear()

    provider = get_alpha_llm_provider()

    assert provider.provider_name == "openai"
    assert provider.enabled() is True
    get_config.cache_clear()


def test_alpha_provider_does_not_use_anthropic_when_global_provider_disabled(monkeypatch):
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    monkeypatch.setenv("VOE_ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.delenv("VOE_OPENAI_API_KEY", raising=False)
    from app.config import get_config

    get_config.cache_clear()

    provider = get_alpha_llm_provider()

    assert provider.provider_name == "disabled"
    assert provider.enabled() is False
    get_config.cache_clear()


def test_alpha_provider_allows_explicit_anthropic_opt_in(monkeypatch):
    monkeypatch.setenv("VOE_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("VOE_ANTHROPIC_API_KEY", "anthropic-test-key")
    from app.config import get_config

    get_config.cache_clear()

    provider = get_alpha_llm_provider()

    assert provider.provider_name == "anthropic"
    assert provider.enabled() is True
    get_config.cache_clear()


def test_fetch_filing_section_prefers_readable_mdna_over_xbrl_and_glossary(monkeypatch):
    filing_text = """
    Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations 56
    Item 7A. Quantitative and Qualitative Disclosures About Market Risk 79

    us-gaap:StatementTable dei:EntityRegistrantName PMIERs duration_2025 instant_2025
    PMIERs Private mortgage insurer eligibility requirements
    GSE Government-sponsored enterprise
    IIF Insurance-in-force
    RIF Risk-in-force
    NIW New insurance written
    RBC Risk-based capital
    DAC Deferred acquisition costs
    ALM Asset liability management

    Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
    The following analysis reviews our liquidity and operating performance.
    Liquidity and Capital Resources
    Our PMIERs available assets exceeded minimum required assets by $512 million at year-end.
    The excess capital supported holding-company liquidity and gave management flexibility to write new insurance.
    We did not rely on short-term borrowings to satisfy regulatory capital requirements.
    Results of Operations
    Net premiums earned increased because persistency improved and claim severity remained stable.
    Item 7A. Quantitative and Qualitative Disclosures About Market Risk
    """
    monkeypatch.setattr("app.alpha.llm_tools._load_full_text", lambda ticker: (filing_text, "10-K"))
    ctx = AlphaToolContext(
        sector="insurance",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_filing_section",
        {
            "keywords": ["PMIERs", "available assets", "minimum required assets"],
            "section_focus": "liquidity",
            "max_chars": 1200,
        },
        ctx,
    )

    assert result["status"] == "ok"
    assert result["source_strategy"] == "section:mdna"
    assert (
        "Our PMIERs available assets exceeded minimum required assets by $512 million at year-end."
        in result["excerpt"]
    )
    assert "us-gaap:StatementTable" not in result["excerpt"]
    assert "PMIERs Private mortgage insurer eligibility requirements" not in result["excerpt"]
    assert result["warnings"] == []


def test_fetch_filing_section_handles_dash_separated_mdna_heading(monkeypatch):
    filing_text = """
    Item 7 — Management’s Discussion and Analysis of Financial Condition and Results of Operations 62
    Item 7A — Quantitative and Qualitative Disclosures About Market Risk 88

    us-gaap:StatementTable dei:EntityRegistrantName duration_2025 instant_2025
    Revenue Gross Profit Operating Income Net Income Cash and Cash Equivalents Assets Liabilities

    Item 7 — Management’s Discussion and Analysis of Financial Condition and Results of Operations
    Management reviewed liquidity and operating performance in readable narrative.
    Liquidity remained strong because recurring cash flow funded product investment and debt service.
    The company preserved balance-sheet flexibility and did not rely on short-term borrowings.
    Results of Operations
    Revenue growth improved as retention increased and operating leverage expanded.
    Item 7A — Quantitative and Qualitative Disclosures About Market Risk
    """
    monkeypatch.setattr("app.alpha.llm_tools._load_full_text", lambda ticker: (filing_text, "10-K"))
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_filing_section",
        {
            "keywords": ["liquidity", "cash flow", "balance-sheet flexibility"],
            "section_focus": "liquidity",
            "max_chars": 1200,
        },
        ctx,
    )

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["source_strategy"] == "section:mdna"
    assert (
        "Liquidity remained strong because recurring cash flow funded product investment"
        in result["excerpt"]
    )
    assert "us-gaap:StatementTable" not in result["excerpt"]
    assert result["warnings"] == []


def test_fetch_filing_section_rejects_stripped_financial_statement_table(monkeypatch):
    filing_text = (
        "Revenue Cost of Revenue Gross Profit Operating Income Net Income "
        "Basic Earnings Per Share Diluted Earnings Per Share Weighted Average Shares "
        "Cash and Cash Equivalents Accounts Receivable Assets Liabilities "
        "Stockholders Equity Accumulated Deficit Cash Flows "
    ) * 6
    monkeypatch.setattr("app.alpha.llm_tools._load_full_text", lambda ticker: (filing_text, "10-K"))
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_filing_section",
        {
            "keywords": ["revenue", "gross profit", "operating income"],
            "section_focus": "margins",
            "max_chars": 1200,
        },
        ctx,
    )

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is False
    assert result["evidence_status"] == "NO_READABLE_NARRATIVE"
    assert result["source_strategy"] == "full_text_noisy_fallback"
    assert result["warnings"] == ["readable_narrative_section_not_found"]
    assert result["noise_filtered_passages"] > 0


def test_fetch_transcript_excerpt_returns_explicit_unavailable(monkeypatch):
    monkeypatch.setenv("VOE_RESEARCH_ENABLE_TRANSCRIPTS", "false")
    monkeypatch.setenv("VOE_RESEARCH_TRANSCRIPT_PROVIDER", "disabled")
    from app.config import get_config

    get_config.cache_clear()
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool("fetch_transcript_excerpt", {"focus": "guidance"}, ctx)

    assert result == {
        "status": "unavailable",
        "ticker": "AAA",
        "reason": "transcripts_disabled",
        "focus": "guidance",
    }


def test_fetch_transcript_excerpt_returns_focused_alpha_vantage_excerpt(monkeypatch):
    monkeypatch.setenv("VOE_SAFE_MODE", "false")
    monkeypatch.setenv("VOE_RESEARCH_ENABLE_TRANSCRIPTS", "true")
    monkeypatch.setenv("VOE_RESEARCH_TRANSCRIPT_PROVIDER", "alpha_vantage")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
    from app.config import get_config

    get_config.cache_clear()

    def fake_collect(self, adapter_ctx):
        return AdapterResult(
            evidence_items=[
                EvidenceItem(
                    id="ev_transcript",
                    ticker="AAA",
                    as_of_date="2026-04-19",
                    source_type="TRANSCRIPT",
                    source_url="https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol=AAA&quarter=2026Q1",
                    source_title="AAA earnings call transcript 2026Q1",
                    source_published_at="2026-03-31",
                    retrieved_at="2026-04-19T00:00:00+00:00",
                    excerpt_text=(
                        "Opening remarks covered demand. Management raised guidance after retention improved. "
                        "Q&A focused on margin durability."
                    ),
                    citations=[
                        CitationRef(
                            source_url="https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol=AAA&quarter=2026Q1",
                            snippet="Management raised guidance after retention improved.",
                            section_label="TRANSCRIPT",
                        )
                    ],
                    hash="hash",
                )
            ]
        )

    monkeypatch.setattr("app.alpha.llm_tools.TranscriptAdapter.collect", fake_collect)
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool("fetch_transcript_excerpt", {"focus": "guidance"}, ctx)

    assert result["status"] == "ok"
    assert result["ticker"] == "AAA"
    assert result["source_type"] == "TRANSCRIPT"
    assert result["source_title"] == "AAA earnings call transcript 2026Q1"
    assert result["source_published_at"] == "2026-03-31"
    assert "raised guidance" in result["excerpt"]
    assert result["focus"] == "guidance"
    assert result["warnings"] == []


def test_fetch_kpi_trends_includes_insurance_operating_metrics():
    packet = _packet()
    packet.insurance_packet = {
        "operating_metrics": {
            "status": "OK",
            "confidence": "HIGH",
            "combined_ratio": 0.924,
            "loss_ratio": 0.613,
            "expense_ratio": 0.311,
            "combined_ratio_assessment": "STRONG_UNDERWRITING_PROFIT",
            "reserve_development": {"status": "FAVORABLE"},
            "reinsurance_program": {"structures": ["quota_share"]},
            "catastrophe_exposure": {"terms": ["hurricane"]},
            "missing_components": [],
        }
    }
    ctx = AlphaToolContext(
        sector="insurance",
        ticker="AAA",
        packet=packet,
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool("fetch_kpi_trends", {}, ctx)

    assert result["usable_for_decision"] is True
    assert result["evidence_status"] == "KPI_TRENDS_AVAILABLE"
    assert result["summary"] == (
        "AAA KPI context: 5Y revenue CAGR UNKNOWN, earnings quality UNKNOWN, "
        "quarterly revenue trend UNKNOWN, method tension UNKNOWN."
    )
    assert result["insurance_context"]["operating_metrics"] == {
        "status": "OK",
        "confidence": "HIGH",
        "combined_ratio": 0.924,
        "loss_ratio": 0.613,
        "expense_ratio": 0.311,
        "combined_ratio_assessment": "STRONG_UNDERWRITING_PROFIT",
        "reserve_development": {"status": "FAVORABLE"},
        "reinsurance_program": {"structures": ["quota_share"]},
        "catastrophe_exposure": {"terms": ["hurricane"]},
        "missing_components": [],
    }


def test_fetch_current_events_empty_results_are_non_usable(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_current_event_context",
        lambda ticker, as_of_date: CurrentEventContext(
            documents=[],
            warnings=["current_event_gap:company_news:GAP_HOMEPAGE_URL_MISSING"],
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool("fetch_current_events", {"max_items": 5}, ctx)

    assert result["status"] == "unavailable"
    assert result["usable_for_decision"] is False
    assert result["evidence_status"] == "NO_CURRENT_EVENTS"
    assert result["document_count"] == 0
    assert (
        result["summary"]
        == "No current-event documents were available from configured company-controlled sources."
    )
    assert result["warnings"] == ["current_event_gap:company_news:GAP_HOMEPAGE_URL_MISSING"]


def test_fetch_current_events_missing_metadata_exposes_diagnostics(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_current_event_context",
        lambda ticker, as_of_date: CurrentEventContext(
            documents=[],
            warnings=["current_event_gap:company_news:GAP_HOMEPAGE_URL_MISSING"],
            metadata_source="missing",
            homepage_url_present=False,
            ir_rss_url_present=False,
            allowlist_domains=[],
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool("fetch_current_events", {"max_items": 5}, ctx)

    assert result["status"] == "unavailable"
    assert result["usable_for_decision"] is False
    assert result["evidence_status"] == "CURRENT_EVENT_METADATA_MISSING"
    assert result["metadata_source"] == "missing"
    assert result["homepage_url_present"] is False
    assert result["ir_rss_url_present"] is False
    assert result["source_warnings"] == ["current_event_gap:company_news:GAP_HOMEPAGE_URL_MISSING"]
    assert result["allowlist_source"] == "none"
    assert result["summary"] == (
        "No current-event documents were available from configured company-controlled sources "
        "(metadata=missing, homepage=missing, ir_rss=missing, allowlist=none)."
    )


def test_fetch_current_events_populated_results_are_usable(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_current_event_context",
        lambda ticker, as_of_date: CurrentEventContext(
            documents=[
                CurrentEventDocument(
                    ticker="AAA",
                    source_type="company_news",
                    published_at="2026-04-18T00:00:00Z",
                    title="AAA announces durable free cash flow growth",
                    source_url="https://example.com/news",
                    summary="Management reported higher free cash flow from recurring revenue.",
                    source_quality={
                        "source_family": "company_controlled",
                        "freshness_bucket": "same_day",
                        "source_quality_score": 0.86,
                        "calibration_status": "deterministic_heuristic",
                    },
                )
            ],
            warnings=[],
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool("fetch_current_events", {"max_items": 5}, ctx)

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["evidence_status"] == "CURRENT_EVENTS_AVAILABLE"
    assert result["document_count"] == 1
    assert result["documents"][0]["title"] == "AAA announces durable free cash flow growth"
    assert result["documents"][0]["source_quality"]["source_family"] == "company_controlled"


def test_fetch_recent_filing_context_populated_results_are_usable(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="0000000001",
                        accession="0000000001-26-000010",
                        form_type="10-Q",
                        filing_date="2026-04-10",
                        period_end="2026-03-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url="https://sec.example/aaa-10q.htm",
                        html="<html><body><p>Recent quarterly revenue grew with strong free cash flow.</p></body></html>",
                    )
                ],
                warnings=[],
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_recent_filing_context",
        {"quarters": 1, "material_event_window_days": 365, "max_documents": 3, "max_chars": 500},
        ctx,
    )

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["evidence_status"] == "RECENT_FILING_CONTEXT_AVAILABLE"
    assert result["fresh_document_count"] == 1
    assert result["documents"][0]["role"] == "quarterly"
    assert "Recent quarterly revenue grew" in result["documents"][0]["excerpt"]
    assert result["documents"][0]["source_strategy"] == "full_text_keyword"
    assert result["documents"][0]["narrative_chars"] > 0
    assert result["summary"] == (
        "1 recent quarterly/material-event filing document(s) available with readable narrative "
        "(source_strategy=full_text_keyword)."
    )


def test_fetch_recent_filing_context_uses_readable_10q_body_over_inline_xbrl_front_matter(
    monkeypatch,
):
    filing_html = """
    <html><body>
    payc-20260331 us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax iso4217:USD
    xbrli:shares dei:EntityCommonStockSharesOutstanding Member Axis Domain contextRef duration_
    <div>Table of Contents Item 2. Management's Discussion and Analysis Item 3. Quantitative</div>
    <h1>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h1>
    <p>Revenue grew because recurring subscription demand remained durable, cash flow improved,
    and liquidity remained strong despite ongoing product investment. Management also discussed
    margin expansion, customer retention, and disciplined capital allocation in readable narrative.</p>
    <h1>Item 3. Quantitative and Qualitative Disclosures About Market Risk</h1>
    </body></html>
    """
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="0000000001",
                        accession="0000000001-26-000011",
                        form_type="10-Q",
                        filing_date="2026-04-10",
                        period_end="2026-03-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url="https://sec.example/aaa-10q.htm",
                        html=filing_html,
                    )
                ],
                warnings=[],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_recent_filing_context",
        {"quarters": 1, "material_event_window_days": 365, "max_documents": 3, "max_chars": 900},
        ctx,
    )

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["documents"][0]["source_strategy"] == "section:quarterly_mdna"
    assert result["documents"][0]["readable_for_decision"] is True
    assert result["documents"][0]["noise_filtered_passages"] == 0
    assert "recurring subscription demand remained durable" in result["documents"][0]["excerpt"]


def test_fetch_recent_filing_context_handles_typographic_10q_mdna_heading(monkeypatch):
    filing_html = """
    <html><body>
    us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax iso4217:USD xbrli:shares
    dei:EntityCommonStockSharesOutstanding srt:ProductOrServiceAxis stpr:ScenarioForecastMember
    http://fasb.org/us-gaap/2026 contextRef duration_ unitRef instant_20260331 000000000123
    <div>Table of Contents Item 2. Management’s Discussion and Analysis Item 3. Quantitative</div>
    <h1>Item 2. Management’s Discussion and Analysis of Financial Condition and Results of Operations</h1>
    <p>Revenue grew because customer expansion improved, liquidity remained strong,
    and cash flow conversion exceeded management's prior outlook. Management also discussed
    margin durability, renewal trends, operating leverage, and product investment trade-offs
    in readable narrative that should be usable for a decision.</p>
    <h1>Item 3. Quantitative and Qualitative Disclosures About Market Risk</h1>
    </body></html>
    """
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="0000000001",
                        accession="0000000001-26-000012",
                        form_type="10-Q",
                        filing_date="2026-04-10",
                        period_end="2026-03-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url="https://sec.example/aaa-10q.htm",
                        html=filing_html,
                    )
                ],
                warnings=[],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_recent_filing_context",
        {"quarters": 1, "material_event_window_days": 365, "max_documents": 3, "max_chars": 900},
        ctx,
    )

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["documents"][0]["source_strategy"] == "section:quarterly_mdna"
    assert result["documents"][0]["readable_for_decision"] is True
    assert (
        "cash flow conversion exceeded management's prior outlook"
        in result["documents"][0]["excerpt"]
    )
    assert (
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
        not in result["documents"][0]["excerpt"]
    )


def test_fetch_recent_filing_context_prefers_8k_results_over_later_reg_fd(monkeypatch):
    reg_fd_text = (
        "The investor presentation includes guidance summary slides and supplemental material for reference. "
        * 12
    )
    filing_html = f"""
    <html><body>
    <div>Table of Contents Item 2.02 — Results of Operations and Financial Condition
    Item 7.01 — Regulation FD Disclosure Item 9.01 — Financial Statements and Exhibits</div>
    <h1>Item 2.02 — Results of Operations and Financial Condition.</h1>
    <p>Results exceeded guidance because renewal demand improved and liquidity remained strong.
    Management raised full-year guidance after revenue growth accelerated and margins expanded.
    Cash flow was positive and management reaffirmed disciplined capital allocation priorities.</p>
    <h1>Item 7.01 — Regulation FD Disclosure.</h1>
    <p>{reg_fd_text}</p>
    <h1>Item 9.01 — Financial Statements and Exhibits.</h1>
    <p>Exhibit 99.1 Press release. Exhibit 104 Cover Page Interactive Data File.</p>
    <p>SIGNATURES</p>
    </body></html>
    """
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="0000000001",
                        accession="0000000001-26-000013",
                        form_type="8-K",
                        filing_date="2026-04-12",
                        period_end=None,
                        role="material_event",
                        local_path=None,
                        primary_doc_url="https://sec.example/aaa-8k.htm",
                        html=filing_html,
                    )
                ],
                warnings=[],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_recent_filing_context",
        {"quarters": 1, "material_event_window_days": 365, "max_documents": 3, "max_chars": 900},
        ctx,
    )

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["fresh_document_count"] == 1
    assert result["documents"][0]["role"] == "material_event"
    assert result["documents"][0]["source_strategy"] == "section:8k_results"
    assert result["documents"][0]["readable_for_decision"] is True
    assert (
        "Results exceeded guidance because renewal demand improved"
        in result["documents"][0]["excerpt"]
    )
    assert (
        "investor presentation includes guidance summary slides"
        not in result["documents"][0]["excerpt"]
    )


def test_fetch_recent_filing_context_prefers_8k_leadership_over_later_reg_fd(monkeypatch):
    reg_fd_text = (
        "The investor presentation includes supplemental material and governance overview for reference. "
        * 12
    )
    filing_html = f"""
    <html><body>
    <div>Table of Contents Item 5.02 Departure of Directors or Certain Officers
    Item 7.01 Regulation FD Disclosure Item 9.01 Financial Statements and Exhibits</div>
    <h1>Item 5.02 Departure of Directors or Certain Officers; Election of Directors;
    Appointment of Certain Officers; Compensatory Arrangements of Certain Officers.</h1>
    <p>The board appointed a new chief financial officer after the prior executive resigned.
    Management said the appointment supports operating discipline and capital allocation priorities.
    The officer will oversee forecasting, controls, and financial planning.</p>
    <h1>Item 7.01 Regulation FD Disclosure.</h1>
    <p>{reg_fd_text}</p>
    <h1>Item 9.01 Financial Statements and Exhibits.</h1>
    <p>Exhibit 99.1 Press release.</p>
    <p>SIGNATURES</p>
    </body></html>
    """
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="0000000001",
                        accession="0000000001-26-000014",
                        form_type="8-K",
                        filing_date="2026-04-13",
                        period_end=None,
                        role="material_event",
                        local_path=None,
                        primary_doc_url="https://sec.example/aaa-8k-leadership.htm",
                        html=filing_html,
                    )
                ],
                warnings=[],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_recent_filing_context",
        {"quarters": 1, "material_event_window_days": 365, "max_documents": 3, "max_chars": 900},
        ctx,
    )

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["fresh_document_count"] == 1
    assert result["documents"][0]["role"] == "material_event"
    assert result["documents"][0]["source_strategy"] == "section:8k_leadership"
    assert result["documents"][0]["readable_for_decision"] is True
    assert "appointed a new chief financial officer" in result["documents"][0]["excerpt"]
    assert (
        "investor presentation includes supplemental material"
        not in result["documents"][0]["excerpt"]
    )


def test_fetch_recent_filing_context_prefers_8k_restructuring_over_later_reg_fd(monkeypatch):
    reg_fd_text = (
        "The investor presentation includes supplemental material and overview slides for reference. "
        * 12
    )
    filing_html = f"""
    <html><body>
    <div>Table of Contents Item 2.05 Costs Associated with Exit or Disposal Activities
    Item 7.01 Regulation FD Disclosure Item 9.01 Financial Statements and Exhibits</div>
    <h1>Item 2.05 Costs Associated with Exit or Disposal Activities.</h1>
    <p>The company approved a restructuring plan and expects severance charges,
    facility closure costs, and asset write-downs. Management said the exit activities
    will reduce capacity and create execution risk during the transition.
    Cash costs are expected over the next two quarters.</p>
    <h1>Item 7.01 Regulation FD Disclosure.</h1>
    <p>{reg_fd_text}</p>
    <h1>Item 9.01 Financial Statements and Exhibits.</h1>
    <p>Exhibit 99.1 Press release.</p>
    <p>SIGNATURES</p>
    </body></html>
    """
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="0000000001",
                        accession="0000000001-26-000015",
                        form_type="8-K",
                        filing_date="2026-04-14",
                        period_end=None,
                        role="material_event",
                        local_path=None,
                        primary_doc_url="https://sec.example/aaa-8k-restructuring.htm",
                        html=filing_html,
                    )
                ],
                warnings=[],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_recent_filing_context",
        {"quarters": 1, "material_event_window_days": 365, "max_documents": 3, "max_chars": 900},
        ctx,
    )

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["documents"][0]["role"] == "material_event"
    assert result["documents"][0]["source_strategy"] == "section:8k_restructuring"
    assert result["documents"][0]["readable_for_decision"] is True
    assert (
        "approved a restructuring plan and expects severance charges"
        in result["documents"][0]["excerpt"]
    )
    assert (
        "investor presentation includes supplemental material"
        not in result["documents"][0]["excerpt"]
    )


def test_fetch_recent_filing_context_prefers_8k_impairment_over_later_reg_fd(monkeypatch):
    reg_fd_text = (
        "The investor presentation includes supplemental material and overview slides for reference. "
        * 12
    )
    filing_html = f"""
    <html><body>
    <div>Table of Contents Item 2.06 Material Impairments
    Item 7.01 Regulation FD Disclosure Item 9.01 Financial Statements and Exhibits</div>
    <h1>Item 2.06 Material Impairments.</h1>
    <p>The company recorded a material impairment charge after demand deteriorated
    and management reduced the expected cash flows from the asset group. The impairment
    reflects lower utilization, weaker margin expectations, and elevated execution risk.</p>
    <h1>Item 7.01 Regulation FD Disclosure.</h1>
    <p>{reg_fd_text}</p>
    <h1>Item 9.01 Financial Statements and Exhibits.</h1>
    <p>Exhibit 99.1 Press release.</p>
    <p>SIGNATURES</p>
    </body></html>
    """
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="0000000001",
                        accession="0000000001-26-000016",
                        form_type="8-K",
                        filing_date="2026-04-15",
                        period_end=None,
                        role="material_event",
                        local_path=None,
                        primary_doc_url="https://sec.example/aaa-8k-impairment.htm",
                        html=filing_html,
                    )
                ],
                warnings=[],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_recent_filing_context",
        {"quarters": 1, "material_event_window_days": 365, "max_documents": 3, "max_chars": 900},
        ctx,
    )

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["documents"][0]["role"] == "material_event"
    assert result["documents"][0]["source_strategy"] == "section:8k_impairment"
    assert result["documents"][0]["readable_for_decision"] is True
    assert (
        "recorded a material impairment charge after demand deteriorated"
        in result["documents"][0]["excerpt"]
    )
    assert (
        "investor presentation includes supplemental material"
        not in result["documents"][0]["excerpt"]
    )


def test_fetch_recent_filing_context_prefers_8k_default_over_later_reg_fd(monkeypatch):
    reg_fd_text = (
        "The investor presentation includes supplemental material and overview slides for reference. "
        * 12
    )
    filing_html = f"""
    <html><body>
    <div>Table of Contents Item 2.04 Triggering Events That Accelerate or Increase
    a Direct Financial Obligation Item 7.01 Regulation FD Disclosure</div>
    <h1>Item 2.04 Triggering Events That Accelerate or Increase a Direct Financial Obligation.</h1>
    <p>The company received a notice of default under its credit agreement after
    failing to maintain minimum liquidity. The default could accelerate debt
    obligations and creates material refinancing risk.</p>
    <h1>Item 7.01 Regulation FD Disclosure.</h1>
    <p>{reg_fd_text}</p>
    <p>SIGNATURES</p>
    </body></html>
    """
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="0000000001",
                        accession="0000000001-26-000017",
                        form_type="8-K",
                        filing_date="2026-04-16",
                        period_end=None,
                        role="material_event",
                        local_path=None,
                        primary_doc_url="https://sec.example/aaa-8k-default.htm",
                        html=filing_html,
                    )
                ],
                warnings=[],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_recent_filing_context",
        {"quarters": 1, "material_event_window_days": 365, "max_documents": 3, "max_chars": 900},
        ctx,
    )

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["documents"][0]["role"] == "material_event"
    assert result["documents"][0]["source_strategy"] == "section:8k_default"
    assert result["documents"][0]["readable_for_decision"] is True
    assert (
        "received a notice of default under its credit agreement"
        in result["documents"][0]["excerpt"]
    )
    assert (
        "investor presentation includes supplemental material"
        not in result["documents"][0]["excerpt"]
    )


def test_fetch_recent_filing_context_prefers_8k_non_reliance_over_later_reg_fd(monkeypatch):
    reg_fd_text = (
        "The investor presentation includes supplemental material and overview slides for reference. "
        * 12
    )
    filing_html = f"""
    <html><body>
    <div>Table of Contents Item 4.02 Non-Reliance on Previously Issued Financial Statements
    Item 7.01 Regulation FD Disclosure</div>
    <h1>Item 4.02 Non-Reliance on Previously Issued Financial Statements or a Related Audit Report.</h1>
    <p>The audit committee determined that prior financial statements should no longer
    be relied upon because revenue recognition errors require a restatement. Management
    identified a material weakness in internal control over financial reporting.</p>
    <h1>Item 7.01 Regulation FD Disclosure.</h1>
    <p>{reg_fd_text}</p>
    <p>SIGNATURES</p>
    </body></html>
    """
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="0000000001",
                        accession="0000000001-26-000018",
                        form_type="8-K",
                        filing_date="2026-04-17",
                        period_end=None,
                        role="material_event",
                        local_path=None,
                        primary_doc_url="https://sec.example/aaa-8k-nonreliance.htm",
                        html=filing_html,
                    )
                ],
                warnings=[],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_recent_filing_context",
        {"quarters": 1, "material_event_window_days": 365, "max_documents": 3, "max_chars": 900},
        ctx,
    )

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["documents"][0]["role"] == "material_event"
    assert result["documents"][0]["source_strategy"] == "section:8k_non_reliance"
    assert result["documents"][0]["readable_for_decision"] is True
    assert "revenue recognition errors require a restatement" in result["documents"][0]["excerpt"]
    assert (
        "investor presentation includes supplemental material"
        not in result["documents"][0]["excerpt"]
    )


def test_fetch_recent_filing_context_prefers_8k_termination_over_later_reg_fd(monkeypatch):
    reg_fd_text = (
        "The investor presentation includes supplemental material and overview slides for reference. "
        * 12
    )
    filing_html = f"""
    <html><body>
    <div>Table of Contents Item 1.02 Termination of a Material Definitive Agreement
    Item 7.01 Regulation FD Disclosure</div>
    <h1>Item 1.02 Termination of a Material Definitive Agreement.</h1>
    <p>The company terminated a material supply agreement after counterparty performance
    deteriorated. Management said the termination could disrupt customer deliveries and
    create transition costs while replacement vendors are qualified.</p>
    <h1>Item 7.01 Regulation FD Disclosure.</h1>
    <p>{reg_fd_text}</p>
    <p>SIGNATURES</p>
    </body></html>
    """
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="0000000001",
                        accession="0000000001-26-000019",
                        form_type="8-K",
                        filing_date="2026-04-18",
                        period_end=None,
                        role="material_event",
                        local_path=None,
                        primary_doc_url="https://sec.example/aaa-8k-termination.htm",
                        html=filing_html,
                    )
                ],
                warnings=[],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_recent_filing_context",
        {"quarters": 1, "material_event_window_days": 365, "max_documents": 3, "max_chars": 900},
        ctx,
    )

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["documents"][0]["role"] == "material_event"
    assert result["documents"][0]["source_strategy"] == "section:8k_agreement_termination"
    assert result["documents"][0]["readable_for_decision"] is True
    assert "terminated a material supply agreement" in result["documents"][0]["excerpt"]
    assert (
        "investor presentation includes supplemental material"
        not in result["documents"][0]["excerpt"]
    )


def test_fetch_recent_filing_context_prefers_8k_delisting_over_later_reg_fd(monkeypatch):
    reg_fd_text = (
        "The investor presentation includes supplemental material and overview slides for reference. "
        * 12
    )
    filing_html = f"""
    <html><body>
    <div>Table of Contents Item 3.01 Notice of Delisting or Failure to Satisfy a
    Continued Listing Rule or Standard Item 7.01 Regulation FD Disclosure</div>
    <h1>Item 3.01 Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard.</h1>
    <p>The company received a listing deficiency notice after its market value fell
    below the exchange requirement. Management said potential delisting risk could
    affect liquidity, capital access, and investor demand.</p>
    <h1>Item 7.01 Regulation FD Disclosure.</h1>
    <p>{reg_fd_text}</p>
    <p>SIGNATURES</p>
    </body></html>
    """
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="0000000001",
                        accession="0000000001-26-000020",
                        form_type="8-K",
                        filing_date="2026-04-18",
                        period_end=None,
                        role="material_event",
                        local_path=None,
                        primary_doc_url="https://sec.example/aaa-8k-delisting.htm",
                        html=filing_html,
                    )
                ],
                warnings=[],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_recent_filing_context",
        {"quarters": 1, "material_event_window_days": 365, "max_documents": 3, "max_chars": 900},
        ctx,
    )

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["documents"][0]["role"] == "material_event"
    assert result["documents"][0]["source_strategy"] == "section:8k_delisting"
    assert result["documents"][0]["readable_for_decision"] is True
    assert "received a listing deficiency notice" in result["documents"][0]["excerpt"]
    assert (
        "investor presentation includes supplemental material"
        not in result["documents"][0]["excerpt"]
    )


def test_fetch_recent_filing_context_without_fresh_documents_is_non_usable(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="0000000001",
                        accession="0000000001-25-000010",
                        form_type="10-K",
                        filing_date="2024-02-15",
                        period_end="2023-12-31",
                        role="annual",
                        local_path=None,
                        primary_doc_url="https://sec.example/aaa-10k.htm",
                        html="<html><body><p>Annual filing text is readable but not fresh follow-up evidence.</p></body></html>",
                    )
                ],
                warnings=["no_recent_quarterly_or_material_event_filings"],
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_recent_filing_context",
        {"quarters": 1, "material_event_window_days": 365, "max_documents": 3, "max_chars": 500},
        ctx,
    )

    assert result["status"] == "unavailable"
    assert result["usable_for_decision"] is False
    assert result["evidence_status"] == "NO_RECENT_FILING_CONTEXT"
    assert result["fresh_document_count"] == 0
    assert result["fresh_annual_document_count"] == 0
    assert result["warnings"] == ["no_recent_quarterly_or_material_event_filings"]


def test_fetch_recent_filing_context_latest_annual_can_clear_freshness(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="0000000001",
                        accession="0000000001-26-000010",
                        form_type="10-K",
                        filing_date="2026-02-15",
                        period_end="2025-12-31",
                        role="annual",
                        local_path=None,
                        primary_doc_url="https://sec.example/aaa-10k.htm",
                        html="<html><body><p>Fresh annual risk and MD&A context is readable.</p></body></html>",
                        materialized_from="sec_primary_document_fetch",
                    )
                ],
                warnings=[
                    "annual_filing_recovered:0000000001-26-000010:sec_primary_document_fetch"
                ],
                recent_filing_status="NO_RECENT_FILINGS_CACHED",
                recovered_document_count=1,
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_recent_filing_context",
        {"quarters": 1, "material_event_window_days": 365, "max_documents": 3, "max_chars": 500},
        ctx,
    )

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["evidence_status"] == "RECENT_FILING_CONTEXT_AVAILABLE"
    assert result["fresh_document_count"] == 0
    assert result["fresh_annual_document_count"] == 1
    assert result["recovered_document_count"] == 1
    assert result["documents"][0]["materialized_from"] == "sec_primary_document_fetch"
    assert result["documents"][0]["readable_for_decision"] is True
    assert result["documents"][0]["source_strategy"] == "full_text_start"
    assert result["summary"] == (
        "1 readable latest annual filing document(s) available as fresh filing evidence "
        "(source_strategy=full_text_start)."
    )


def test_fetch_recent_filing_context_noisy_recent_annual_is_non_usable(monkeypatch):
    noisy_html = (
        "<html><body>payc-20251231 us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax "
        "iso4217:USD xbrli:shares dei:EntityCommonStockSharesOutstanding Member Axis Domain "
        "us-gaap:Assets us-gaap:Liabilities xbrli:pure contextRef duration_ instant_ unitRef"
        "</body></html>"
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="0000000001",
                        accession="0000000001-26-000010",
                        form_type="10-K",
                        filing_date="2026-02-15",
                        period_end="2025-12-31",
                        role="annual",
                        local_path=None,
                        primary_doc_url="https://sec.example/aaa-10k.htm",
                        html=noisy_html,
                        materialized_from="sec_primary_document_fetch",
                    )
                ],
                warnings=[],
                recent_filing_status="NO_RECENT_FILINGS_CACHED",
                recovered_document_count=1,
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_recent_filing_context",
        {"quarters": 1, "material_event_window_days": 365, "max_documents": 3, "max_chars": 500},
        ctx,
    )

    assert result["status"] == "unavailable"
    assert result["usable_for_decision"] is False
    assert result["evidence_status"] == "RECENT_FILINGS_UNREADABLE"
    assert result["fresh_annual_document_count"] == 0
    assert result["documents"][0]["readable_for_decision"] is False


def test_fetch_recent_filing_context_rejects_stripped_statement_table(monkeypatch):
    statement_table_html = (
        "<html><body>"
        + (
            "Revenue Cost of Revenue Gross Profit Operating Income Net Income "
            "Basic Earnings Per Share Diluted Earnings Per Share Weighted Average Shares "
            "Cash and Cash Equivalents Accounts Receivable Assets Liabilities "
            "Stockholders Equity Accumulated Deficit Cash Flows "
        )
        * 6
        + "</body></html>"
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="0000000001",
                        accession="0000000001-26-000010",
                        form_type="10-K",
                        filing_date="2026-02-15",
                        period_end="2025-12-31",
                        role="annual",
                        local_path=None,
                        primary_doc_url="https://sec.example/aaa-10k.htm",
                        html=statement_table_html,
                        materialized_from="sec_primary_document_fetch",
                    )
                ],
                warnings=[],
                recent_filing_status="NO_RECENT_FILINGS_CACHED",
                recovered_document_count=1,
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_recent_filing_context",
        {"quarters": 1, "material_event_window_days": 365, "max_documents": 3, "max_chars": 500},
        ctx,
    )

    assert result["status"] == "unavailable"
    assert result["usable_for_decision"] is False
    assert result["evidence_status"] == "RECENT_FILINGS_UNREADABLE"
    assert result["fresh_annual_document_count"] == 0
    assert result["documents"][0]["readable_for_decision"] is False
    assert result["documents"][0]["source_strategy"] == "full_text_noisy_fallback"


def test_fetch_kpi_trends_includes_insurance_subtype_peer_context(monkeypatch):
    packet = _packet()
    packet.issuer_type = "insurance_underwriter"
    packet.insurance_subtype = "title_mortgage_specialty"
    monkeypatch.setattr(
        "app.alpha.llm_tools.compute_insurance_subtype_peer_relative_metrics",
        lambda ticker, as_of_date: {
            "status": "OK",
            "peer_scope": "insurance_subtype",
            "peer_group": "insurance:title_mortgage_specialty",
            "insurance_subtype": "title_mortgage_specialty",
            "peer_count": 6,
            "relative_position": "LEADER",
            "relative_ratios": {
                "roic_vs_median": 1.7,
                "operating_margin_vs_median": 1.3,
                "revenue_growth_vs_median": 1.1,
            },
            "sector_medians": {
                "roic": 0.08,
                "operating_margin": 0.22,
                "revenue_growth_5y": 0.04,
            },
        },
    )
    ctx = AlphaToolContext(
        sector="insurance",
        ticker="AAA",
        packet=packet,
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool("fetch_kpi_trends", {}, ctx)

    assert result["insurance_context"]["peer_context"] == {
        "status": "OK",
        "peer_scope": "insurance_subtype",
        "peer_group": "insurance:title_mortgage_specialty",
        "insurance_subtype": "title_mortgage_specialty",
        "peer_count": 6,
        "relative_position": "LEADER",
        "relative_ratios": {
            "roic_vs_median": 1.7,
            "operating_margin_vs_median": 1.3,
            "revenue_growth_vs_median": 1.1,
        },
        "sector_medians": {
            "roic": 0.08,
            "operating_margin": 0.22,
            "revenue_growth_5y": 0.04,
        },
        "fallback_reason": None,
    }


def test_compare_peer_metric_uses_insurance_subtype_peer_ratio(monkeypatch):
    packet = _packet()
    packet.issuer_type = "insurance_underwriter"
    packet.insurance_subtype = "title_mortgage_specialty"
    monkeypatch.setattr(
        "app.alpha.llm_tools.compute_insurance_subtype_peer_relative_metrics",
        lambda ticker, as_of_date: {
            "status": "OK",
            "sector": "insurance",
            "peer_scope": "insurance_subtype",
            "peer_group": "insurance:title_mortgage_specialty",
            "insurance_subtype": "title_mortgage_specialty",
            "peer_count": 6,
            "relative_position": "LEADER",
            "ticker_metrics": {"roic": 0.13, "operating_margin": 0.71, "revenue_growth_5y": 0.21},
            "sector_medians": {"roic": 0.08, "operating_margin": 0.22, "revenue_growth_5y": 0.04},
            "relative_ratios": {
                "roic_vs_median": 1.7,
                "operating_margin_vs_median": 3.2,
                "revenue_growth_vs_median": 5.3,
            },
        },
    )
    ctx = AlphaToolContext(
        sector="insurance",
        ticker="AAA",
        packet=packet,
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool("compare_peer_metric", {"metric": "roic"}, ctx)

    assert result["status"] == "ok"
    assert result["peer_scope"] == "insurance_subtype"
    assert result["peer_group"] == "insurance:title_mortgage_specialty"
    assert result["insurance_subtype"] == "title_mortgage_specialty"
    assert result["peer_count"] == 6
    assert result["relative_position"] == "LEADER"
    assert result["ratio"] == 1.7


def test_compare_peer_metric_supports_sector_valuation_metrics(monkeypatch):
    observed: list[tuple[str, str | None]] = []

    def fake_peer_context(ticker, as_of_date, sector=None):
        observed.append((ticker, sector))
        return {
            "status": "OK",
            "sector": "semiconductors",
            "peer_count": 6,
            "relative_position": "AVERAGE",
            "ticker_metrics": {
                "ev_ebitda": 4.0,
                "ev_ebit": 4.0,
                "ev_sales": 4.0,
                "p_e": 4.0,
                "p_b": 4.0,
                "fcf_yield": 4.0,
                "dividend_yield": 4.0,
            },
            "sector_medians": {
                "ev_ebitda": 3.5,
                "ev_ebit": 3.5,
                "ev_sales": 3.5,
                "p_e": 3.5,
                "p_b": 3.5,
                "fcf_yield": 3.5,
                "dividend_yield": 3.5,
            },
            "sector_q1": {
                "ev_ebitda": 2.0,
                "ev_ebit": 2.0,
                "ev_sales": 2.0,
                "p_e": 2.0,
                "p_b": 2.0,
                "fcf_yield": 2.0,
                "dividend_yield": 2.0,
            },
            "sector_q3": {
                "ev_ebitda": 5.0,
                "ev_ebit": 5.0,
                "ev_sales": 5.0,
                "p_e": 5.0,
                "p_b": 5.0,
                "fcf_yield": 5.0,
                "dividend_yield": 5.0,
            },
            "relative_ratios": {
                "ev_ebitda_vs_median": 1.14,
                "ev_ebit_vs_median": 1.14,
                "ev_sales_vs_median": 1.14,
                "p_e_vs_median": 1.14,
                "p_b_vs_median": 1.14,
                "fcf_yield_vs_median": 1.14,
                "dividend_yield_vs_median": 1.14,
            },
            "percentile_ranks": {
                "ev_ebitda": 66.7,
                "ev_ebit": 66.7,
                "ev_sales": 66.7,
                "p_e": 66.7,
                "p_b": 66.7,
                "fcf_yield": 66.7,
                "dividend_yield": 66.7,
            },
            "peer_set_used": ["PEER_A", "PEER_B", "PEER_C", "PEER_D", "PEER_E", "PEER_F"],
        }

    monkeypatch.setattr("app.alpha.llm_tools.compute_peer_relative_metrics", fake_peer_context)
    ctx = AlphaToolContext(
        sector="semiconductors",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    expected = {
        "EV/EBITDA": "ev_ebitda",
        "EV/EBIT": "ev_ebit",
        "EV/Sales": "ev_sales",
        "P/E": "p_e",
        "P/B": "p_b",
        "FCF yield": "fcf_yield",
        "Dividend yield": "dividend_yield",
    }
    for requested_metric, metric_key in expected.items():
        result = dispatch_alpha_tool("compare_peer_metric", {"metric": requested_metric}, ctx)
        assert result["status"] == "ok"
        assert result["metric_key"] == metric_key
        assert result["stock_value"] == 4.0
        assert result["sector_median"] == 3.5
        assert result["sector_q1"] == 2.0
        assert result["sector_q3"] == 5.0
        assert result["percentile_rank"] == 66.7
        assert result["peer_set_used"] == [
            "PEER_A",
            "PEER_B",
            "PEER_C",
            "PEER_D",
            "PEER_E",
            "PEER_F",
        ]

    assert observed == [
        ("AAA", "semiconductors"),
        ("AAA", "semiconductors"),
        ("AAA", "semiconductors"),
        ("AAA", "semiconductors"),
        ("AAA", "semiconductors"),
        ("AAA", "semiconductors"),
        ("AAA", "semiconductors"),
    ]


def test_compare_peer_metric_rejects_unsupported_metric_without_substitution():
    ctx = AlphaToolContext(
        sector="insurance",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool("compare_peer_metric", {"metric": "combined_ratio"}, ctx)

    assert result["status"] == "unavailable"
    assert result["metric"] == "combined_ratio"
    assert result["reason"] == "unsupported_peer_metric:combined_ratio"
    assert result["usable_for_decision"] is False
    assert result["evidence_status"] == "UNSUPPORTED_PEER_METRIC"
    assert result["supported_metrics"] == [
        "Dividend yield",
        "EV/EBIT",
        "EV/EBITDA",
        "EV/Sales",
        "FCF yield",
        "Operating margin",
        "P/B",
        "P/E",
        "ROIC",
        "Revenue growth 5Y",
        "dividend_yield",
        "ev_ebit",
        "ev_ebitda",
        "ev_sales",
        "fcf_yield",
        "operating_margin",
        "p_b",
        "p_e",
        "revenue_growth_5y",
        "roic",
    ]


def test_analyze_dilution_exposes_decision_usable_summary(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {
            "shares_outstanding": [
                {
                    "fiscal_year": 2021,
                    "value": 100.0,
                    "raw_value": 100.0,
                    "normalized_value": 100.0,
                    "shares_basis": "UNADJUSTED",
                    "split_adjustment_factor": 1.0,
                    "split_effective_date": None,
                },
                {
                    "fiscal_year": 2025,
                    "value": 92.0,
                    "raw_value": 92.0,
                    "normalized_value": 92.0,
                    "shares_basis": "UNADJUSTED",
                    "split_adjustment_factor": 1.0,
                    "split_effective_date": None,
                },
            ]
        },
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool("analyze_dilution", {"years": 5}, ctx)

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["evidence_status"] == "DILUTION_SERIES_AVAILABLE"
    assert result["dilution_direction"] == "BUYBACKS_OR_SHARE_REDUCTION"
    assert result["summary"] == (
        "AAA share-count CAGR -2.1% over the available annual series; "
        "dilution direction BUYBACKS_OR_SHARE_REDUCTION."
    )


def test_analyze_dilution_rejects_probable_split_without_exact_lineage(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {
            "shares_outstanding": [
                {"fiscal_year": 2021, "value": 100.0},
                {"fiscal_year": 2025, "value": 1000.0},
            ]
        },
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool("analyze_dilution", {"years": 5}, ctx)

    assert result["status"] == "unavailable"
    assert result["usable_for_decision"] is False
    assert result["evidence_status"] == "SHARE_SPLIT_LINEAGE_MISSING"
    assert result["warnings"] == ["share_count_split_lineage_missing"]
    assert result["dilution_direction"] == "UNKNOWN"
    assert result["share_count_cagr"] is None
    assert result["share_count_cagr_raw"] is None
    assert result["share_count_latest"] == 1000.0
    assert result["share_count_oldest"] == 100.0
    assert result["share_count_latest_adjusted"] is None
    assert result["share_count_oldest_adjusted"] is None
    assert result["split_adjusted_share_series"] == []
    assert result["share_count_split_adjustments"] == []
    assert result["summary"] == (
        "AAA share-count history lacks exact split-basis lineage; "
        "dilution direction is unavailable."
    )


def test_analyze_dilution_empty_series_is_non_usable(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {"shares_outstanding": []},
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool("analyze_dilution", {"years": 5}, ctx)

    assert result["status"] == "unavailable"
    assert result["usable_for_decision"] is False
    assert result["evidence_status"] == "NO_SHARE_COUNT_SERIES"
    assert (
        result["summary"]
        == "No annual share-count series was available for AAA; dilution direction is UNKNOWN."
    )


def test_analyze_liquidity_stress_exposes_decision_usable_summary():
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool("analyze_liquidity_stress", {}, ctx)

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["evidence_status"] == "LIQUIDITY_STRESS_AVAILABLE"
    assert result["summary"] == "AAA liquidity/solvency risk is LOW; signals: none."


def test_analyze_capital_structure_resolution_clear_low_risk(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {
            "cash": [{"fiscal_year": 2025, "value": 150.0}],
            "total_debt": [{"fiscal_year": 2025, "value": 25.0}],
            "current_assets": [{"fiscal_year": 2025, "value": 300.0}],
            "current_liabilities": [{"fiscal_year": 2025, "value": 100.0}],
            "cfo": [{"fiscal_year": 2025, "value": 40.0}],
        },
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0001",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html="Liquidity remained strong with ample cash and no covenant defaults under the credit facility.",
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["evidence_status"] == "CAPITAL_STRUCTURE_RESOLVED_CLEAR"
    assert result["liquidity"]["net_debt"] == -125.0
    assert result["stress_flags"]["waiver_refinancing_or_compliance_language_found"] is True
    assert result["covenant_status"] == "COMPLIANCE_EVIDENCED"
    assert "capital structure appears clear" in result["summary"]


def test_analyze_capital_structure_resolution_missing_debt_or_cash_is_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {
            "cash": [{"fiscal_year": 2025, "value": 10.0}],
            "total_debt": [],
            "current_assets": [],
            "current_liabilities": [],
            "cfo": [],
        },
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0001",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html="Liquidity remained adequate during the quarter.",
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool(
        "analyze_capital_structure_resolution",
        {},
        ctx,
    )

    assert result["status"] == "unavailable"
    assert result["usable_for_decision"] is False
    assert result["liquidity"]["cash"] == 10.0
    assert result["liquidity"]["total_debt"] is None
    assert result["liquidity"]["net_debt"] is None
    assert result["liquidity"]["net_debt_trace"]["status"] == "NEEDS_DATA"
    assert "capital_structure_total_debt_unavailable" in result["warnings"]


def test_analyze_capital_structure_resolution_explicit_zero_inputs_are_valid(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {
            "cash": [{"fiscal_year": 2025, "value": 0.0}],
            "total_debt": [{"fiscal_year": 2025, "value": 0.0}],
            "current_assets": [],
            "current_liabilities": [],
            "cfo": [],
        },
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0001",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html="Liquidity remained adequate during the quarter.",
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool(
        "analyze_capital_structure_resolution",
        {},
        ctx,
    )

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["liquidity"]["cash"] == 0.0
    assert result["liquidity"]["total_debt"] == 0.0
    assert result["liquidity"]["net_debt"] == 0.0
    assert result["liquidity"]["net_debt_trace"]["status"] == "OK"


def test_analyze_capital_structure_resolution_flags_active_distress(monkeypatch):
    packet = _packet()
    packet.solvency_risk = "CRITICAL"
    packet.research_report["solvency"] = {
        "risk": "CRITICAL",
        "signals": ["NO_ASSURANCE_FINANCING"],
        "details": "No assurance about financing.",
        "going_concern_language": False,
        "no_assurance_financing": True,
        "debt_due_within_12mo": True,
    }
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0001",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html="Liquidity risk is acute and no assurance can be given that we will obtain sufficient financing.",
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(sector="software", ticker="AAA", packet=packet, as_of_date="2026-04-19")

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["evidence_status"] == "CAPITAL_STRUCTURE_ACTIVE_DISTRESS"
    assert result["stress_flags"]["no_assurance_language"] is True
    assert "active distress" in result["summary"]


def test_analyze_capital_structure_resolution_clears_refinanced_no_assurance_flag(monkeypatch):
    packet = _packet()
    packet.solvency_risk = "ELEVATED"
    packet.research_report["solvency"] = {
        "risk": "ELEVATED",
        "signals": ["NO_ASSURANCE_FINANCING"],
        "details": "No assurance language in prior filing.",
        "going_concern_language": False,
        "no_assurance_financing": True,
        "debt_due_within_12mo": False,
    }
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {
            "cash": [{"fiscal_year": 2025, "value": 75.0}],
            "total_debt": [{"fiscal_year": 2025, "value": 40.0}],
            "current_assets": [],
            "current_liabilities": [],
            "cfo": [],
        },
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0002",
                        form_type="8-K",
                        filing_date="2026-03-01",
                        period_end=None,
                        role="material_event",
                        local_path=None,
                        primary_doc_url=None,
                        html="The company completed refinancing and entered into an amended credit facility. It is in compliance with all covenants.",
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(sector="software", ticker="AAA", packet=packet, as_of_date="2026-04-19")

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["evidence_status"] == "CAPITAL_STRUCTURE_RESOLVED_CLEAR"
    assert result["stress_flags"]["waiver_refinancing_or_compliance_language_found"] is True
    assert result["capital_structure_terms"]["covenant_status"] == "COMPLIANCE_EVIDENCED"
    assert "resolves the prior financing flag" in result["summary"]


def test_analyze_capital_structure_resolution_extracts_maturity_schedule_and_covenants(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0004",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all financial covenants under its credit agreement. "
                            "The facility requires a maximum consolidated net leverage ratio of 3.50 to 1.00 "
                            "and minimum liquidity of $20 million. "
                            "$25 million of term debt is due in 2027. "
                            "$40 million of senior notes mature in 2028."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["evidence_status"] == "CAPITAL_STRUCTURE_RESOLVED_CLEAR"
    assert result["capital_structure_terms"]["extraction_status"] == "STRUCTURED_TERMS_EXTRACTED"
    assert (
        result["capital_structure_terms"]["maturity_schedule_status"]
        == "MATURITY_SCHEDULE_EXTRACTED"
    )
    assert result["capital_structure_terms"]["maturity_schedule"] == [
        {
            "year": 2027,
            "amount": 25.0,
            "unit": "million",
            "context": "$25 million of term debt is due in 2027.",
        },
        {
            "year": 2028,
            "amount": 40.0,
            "unit": "million",
            "context": "$40 million of senior notes mature in 2028.",
        },
    ]
    assert result["maturity_schedule"] == result["capital_structure_terms"]["maturity_schedule"]
    assert result["covenant_status"] == "COMPLIANCE_EVIDENCED"
    assert result["capital_structure_terms"]["covenant_evidence"] == (
        "The company was in compliance with all financial covenants under its credit agreement."
    )
    assert result["capital_structure_terms"]["covenant_terms_status"] == "COVENANT_TERMS_EXTRACTED"
    assert result["covenant_terms"] == [
        {
            "metric": "consolidated net leverage ratio",
            "condition": "MAXIMUM",
            "threshold": 3.5,
            "unit": "ratio",
            "context": (
                "The facility requires a maximum consolidated net leverage ratio of 3.50 to 1.00 "
                "and minimum liquidity of $20 million."
            ),
        },
        {
            "metric": "liquidity",
            "condition": "MINIMUM",
            "threshold": 20.0,
            "unit": "million",
            "context": (
                "The facility requires a maximum consolidated net leverage ratio of 3.50 to 1.00 "
                "and minimum liquidity of $20 million."
            ),
        },
    ]


def test_analyze_capital_structure_resolution_extracts_year_first_maturities_and_post_metric_covenants(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0005",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all financial covenants under its credit facility. "
                            "Debt maturities are as follows: 2026 $10 million; 2027 $15 million; 2028 $30 million; thereafter $45 million. "
                            "The credit agreement requires total net leverage ratio not to exceed 4.00 to 1.00, "
                            "secured leverage ratio not to exceed 2.50 to 1.00, and debt service coverage ratio of at least 1.25 to 1.00."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["maturity_schedule"] == [
        {
            "year": 2026,
            "amount": 10.0,
            "unit": "million",
            "context": "Debt maturities are as follows: 2026 $10 million; 2027 $15 million; 2028 $30 million; thereafter $45 million.",
        },
        {
            "year": 2027,
            "amount": 15.0,
            "unit": "million",
            "context": "Debt maturities are as follows: 2026 $10 million; 2027 $15 million; 2028 $30 million; thereafter $45 million.",
        },
        {
            "year": 2028,
            "amount": 30.0,
            "unit": "million",
            "context": "Debt maturities are as follows: 2026 $10 million; 2027 $15 million; 2028 $30 million; thereafter $45 million.",
        },
        {
            "year": None,
            "period": "thereafter",
            "amount": 45.0,
            "unit": "million",
            "context": "Debt maturities are as follows: 2026 $10 million; 2027 $15 million; 2028 $30 million; thereafter $45 million.",
        },
    ]
    assert result["covenant_terms"] == [
        {
            "metric": "total net leverage ratio",
            "condition": "MAXIMUM",
            "threshold": 4.0,
            "unit": "ratio",
            "context": (
                "The credit agreement requires total net leverage ratio not to exceed 4.00 to 1.00, "
                "secured leverage ratio not to exceed 2.50 to 1.00, and debt service coverage ratio of at least 1.25 to 1.00."
            ),
        },
        {
            "metric": "secured leverage ratio",
            "condition": "MAXIMUM",
            "threshold": 2.5,
            "unit": "ratio",
            "context": (
                "The credit agreement requires total net leverage ratio not to exceed 4.00 to 1.00, "
                "secured leverage ratio not to exceed 2.50 to 1.00, and debt service coverage ratio of at least 1.25 to 1.00."
            ),
        },
        {
            "metric": "debt service coverage ratio",
            "condition": "MINIMUM",
            "threshold": 1.25,
            "unit": "ratio",
            "context": (
                "The credit agreement requires total net leverage ratio not to exceed 4.00 to 1.00, "
                "secured leverage ratio not to exceed 2.50 to 1.00, and debt service coverage ratio of at least 1.25 to 1.00."
            ),
        },
    ]


def test_analyze_capital_structure_resolution_extracts_common_covenant_condition_variants(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0019",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all financial covenants. "
                            "The credit agreement requires senior secured leverage ratio shall not exceed 3.25 to 1.00, "
                            "fixed charge coverage ratio no less than 1.10 to 1.00, "
                            "and debt to capitalization ratio no greater than 0.50."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["covenant_terms_status"] == "COVENANT_TERMS_EXTRACTED"
    assert result["covenant_terms"] == [
        {
            "metric": "senior secured leverage ratio",
            "condition": "MAXIMUM",
            "threshold": 3.25,
            "unit": "ratio",
            "context": (
                "The credit agreement requires senior secured leverage ratio shall not exceed 3.25 to 1.00, "
                "fixed charge coverage ratio no less than 1.10 to 1.00, "
                "and debt to capitalization ratio no greater than 0.50."
            ),
        },
        {
            "metric": "fixed charge coverage ratio",
            "condition": "MINIMUM",
            "threshold": 1.1,
            "unit": "ratio",
            "context": (
                "The credit agreement requires senior secured leverage ratio shall not exceed 3.25 to 1.00, "
                "fixed charge coverage ratio no less than 1.10 to 1.00, "
                "and debt to capitalization ratio no greater than 0.50."
            ),
        },
        {
            "metric": "debt to capitalization ratio",
            "condition": "MAXIMUM",
            "threshold": 0.5,
            "unit": "ratio",
            "context": (
                "The credit agreement requires senior secured leverage ratio shall not exceed 3.25 to 1.00, "
                "fixed charge coverage ratio no less than 1.10 to 1.00, "
                "and debt to capitalization ratio no greater than 0.50."
            ),
        },
    ]


def test_analyze_capital_structure_resolution_extracts_equality_covenant_conditions(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0020",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all financial covenants. "
                            "The credit agreement requires interest coverage ratio greater than or equal to 2.00 to 1.00, "
                            "current ratio equal to or greater than 1.20 to 1.00, "
                            "and total net leverage ratio equal to or less than 4.00 to 1.00."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["covenant_terms_status"] == "COVENANT_TERMS_EXTRACTED"
    assert result["covenant_terms"] == [
        {
            "metric": "interest coverage ratio",
            "condition": "MINIMUM",
            "threshold": 2.0,
            "unit": "ratio",
            "context": (
                "The credit agreement requires interest coverage ratio greater than or equal to 2.00 to 1.00, "
                "current ratio equal to or greater than 1.20 to 1.00, "
                "and total net leverage ratio equal to or less than 4.00 to 1.00."
            ),
        },
        {
            "metric": "current ratio",
            "condition": "MINIMUM",
            "threshold": 1.2,
            "unit": "ratio",
            "context": (
                "The credit agreement requires interest coverage ratio greater than or equal to 2.00 to 1.00, "
                "current ratio equal to or greater than 1.20 to 1.00, "
                "and total net leverage ratio equal to or less than 4.00 to 1.00."
            ),
        },
        {
            "metric": "total net leverage ratio",
            "condition": "MAXIMUM",
            "threshold": 4.0,
            "unit": "ratio",
            "context": (
                "The credit agreement requires interest coverage ratio greater than or equal to 2.00 to 1.00, "
                "current ratio equal to or greater than 1.20 to 1.00, "
                "and total net leverage ratio equal to or less than 4.00 to 1.00."
            ),
        },
    ]


def test_analyze_capital_structure_resolution_extracts_symbolic_covenant_conditions(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0021",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all financial covenants. "
                            "The credit agreement requires total net leverage ratio &lt;= 3.75 to 1.00, "
                            "fixed charge coverage ratio &gt;= 1.15 to 1.00, "
                            "asset coverage ratio ≥ 1.50 to 1.00, and loan-to-value ratio ≤ 60%."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="real estate",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["covenant_terms_status"] == "COVENANT_TERMS_EXTRACTED"
    assert result["covenant_terms"] == [
        {
            "metric": "total net leverage ratio",
            "condition": "MAXIMUM",
            "threshold": 3.75,
            "unit": "ratio",
            "context": (
                "The credit agreement requires total net leverage ratio <= 3.75 to 1.00, "
                "fixed charge coverage ratio >= 1.15 to 1.00, "
                "asset coverage ratio ≥ 1.50 to 1.00, and loan-to-value ratio ≤ 60%."
            ),
        },
        {
            "metric": "fixed charge coverage ratio",
            "condition": "MINIMUM",
            "threshold": 1.15,
            "unit": "ratio",
            "context": (
                "The credit agreement requires total net leverage ratio <= 3.75 to 1.00, "
                "fixed charge coverage ratio >= 1.15 to 1.00, "
                "asset coverage ratio ≥ 1.50 to 1.00, and loan-to-value ratio ≤ 60%."
            ),
        },
        {
            "metric": "asset coverage ratio",
            "condition": "MINIMUM",
            "threshold": 1.5,
            "unit": "ratio",
            "context": (
                "The credit agreement requires total net leverage ratio <= 3.75 to 1.00, "
                "fixed charge coverage ratio >= 1.15 to 1.00, "
                "asset coverage ratio ≥ 1.50 to 1.00, and loan-to-value ratio ≤ 60%."
            ),
        },
        {
            "metric": "loan-to-value ratio",
            "condition": "MAXIMUM",
            "threshold": 60.0,
            "unit": "percent",
            "context": (
                "The credit agreement requires total net leverage ratio <= 3.75 to 1.00, "
                "fixed charge coverage ratio >= 1.15 to 1.00, "
                "asset coverage ratio ≥ 1.50 to 1.00, and loan-to-value ratio ≤ 60%."
            ),
        },
    ]


def test_analyze_capital_structure_resolution_extracts_net_worth_and_capitalization_covenants(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0006",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all covenants. "
                            "The credit agreement requires minimum tangible net worth of $150 million, "
                            "asset coverage ratio of at least 1.50 to 1.00, and debt to capitalization ratio not to exceed 0.45."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["covenant_terms_status"] == "COVENANT_TERMS_EXTRACTED"
    assert result["covenant_terms"] == [
        {
            "metric": "asset coverage ratio",
            "condition": "MINIMUM",
            "threshold": 1.5,
            "unit": "ratio",
            "context": (
                "The credit agreement requires minimum tangible net worth of $150 million, "
                "asset coverage ratio of at least 1.50 to 1.00, and debt to capitalization ratio not to exceed 0.45."
            ),
        },
        {
            "metric": "debt to capitalization ratio",
            "condition": "MAXIMUM",
            "threshold": 0.45,
            "unit": "ratio",
            "context": (
                "The credit agreement requires minimum tangible net worth of $150 million, "
                "asset coverage ratio of at least 1.50 to 1.00, and debt to capitalization ratio not to exceed 0.45."
            ),
        },
        {
            "metric": "tangible net worth",
            "condition": "MINIMUM",
            "threshold": 150.0,
            "unit": "million",
            "context": (
                "The credit agreement requires minimum tangible net worth of $150 million, "
                "asset coverage ratio of at least 1.50 to 1.00, and debt to capitalization ratio not to exceed 0.45."
            ),
        },
    ]


def test_analyze_capital_structure_resolution_extracts_ebitda_and_capex_covenants(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0007",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all covenants. "
                            "The credit agreement requires minimum adjusted EBITDA of $60 million "
                            "and annual capital expenditures not to exceed $25 million."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["covenant_terms_status"] == "COVENANT_TERMS_EXTRACTED"
    assert result["covenant_terms"] == [
        {
            "metric": "adjusted ebitda",
            "condition": "MINIMUM",
            "threshold": 60.0,
            "unit": "million",
            "context": (
                "The credit agreement requires minimum adjusted EBITDA of $60 million "
                "and annual capital expenditures not to exceed $25 million."
            ),
        },
        {
            "metric": "annual capital expenditures",
            "condition": "MAXIMUM",
            "threshold": 25.0,
            "unit": "million",
            "context": (
                "The credit agreement requires minimum adjusted EBITDA of $60 million "
                "and annual capital expenditures not to exceed $25 million."
            ),
        },
    ]


def test_analyze_capital_structure_resolution_extracts_post_metric_liquidity_covenants(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0014",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all covenants. "
                            "The credit agreement requires liquidity to be at least $35 million "
                            "and unused availability not less than $40 million."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["covenant_terms_status"] == "COVENANT_TERMS_EXTRACTED"
    assert result["covenant_terms"] == [
        {
            "metric": "liquidity",
            "condition": "MINIMUM",
            "threshold": 35.0,
            "unit": "million",
            "context": (
                "The credit agreement requires liquidity to be at least $35 million "
                "and unused availability not less than $40 million."
            ),
        },
        {
            "metric": "unused availability",
            "condition": "MINIMUM",
            "threshold": 40.0,
            "unit": "million",
            "context": (
                "The credit agreement requires liquidity to be at least $35 million "
                "and unused availability not less than $40 million."
            ),
        },
    ]


def test_analyze_capital_structure_resolution_extracts_ltv_and_debt_yield_covenants(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0015",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all covenants. "
                            "The mortgage facility requires loan-to-value ratio not to exceed 65% "
                            "and debt yield of at least 9.0%."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="real estate",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["covenant_terms_status"] == "COVENANT_TERMS_EXTRACTED"
    assert result["covenant_terms"] == [
        {
            "metric": "loan-to-value ratio",
            "condition": "MAXIMUM",
            "threshold": 65.0,
            "unit": "percent",
            "context": (
                "The mortgage facility requires loan-to-value ratio not to exceed 65% "
                "and debt yield of at least 9.0%."
            ),
        },
        {
            "metric": "debt yield",
            "condition": "MINIMUM",
            "threshold": 9.0,
            "unit": "percent",
            "context": (
                "The mortgage facility requires loan-to-value ratio not to exceed 65% "
                "and debt yield of at least 9.0%."
            ),
        },
    ]


def test_analyze_capital_structure_resolution_extracts_reit_debt_covenant_families(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0022",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all covenants. "
                            "The unsecured credit facility requires unsecured leverage ratio not to exceed 60%, "
                            "secured debt to total assets ratio &lt;= 40%, "
                            "unencumbered interest coverage ratio no less than 1.75 to 1.00, "
                            "and unencumbered debt yield &gt;= 10.0%."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="real estate",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["covenant_terms_status"] == "COVENANT_TERMS_EXTRACTED"
    assert result["covenant_terms"] == [
        {
            "metric": "unsecured leverage ratio",
            "condition": "MAXIMUM",
            "threshold": 60.0,
            "unit": "percent",
            "context": (
                "The unsecured credit facility requires unsecured leverage ratio not to exceed 60%, "
                "secured debt to total assets ratio <= 40%, "
                "unencumbered interest coverage ratio no less than 1.75 to 1.00, "
                "and unencumbered debt yield >= 10.0%."
            ),
        },
        {
            "metric": "secured debt to total assets ratio",
            "condition": "MAXIMUM",
            "threshold": 40.0,
            "unit": "percent",
            "context": (
                "The unsecured credit facility requires unsecured leverage ratio not to exceed 60%, "
                "secured debt to total assets ratio <= 40%, "
                "unencumbered interest coverage ratio no less than 1.75 to 1.00, "
                "and unencumbered debt yield >= 10.0%."
            ),
        },
        {
            "metric": "unencumbered interest coverage ratio",
            "condition": "MINIMUM",
            "threshold": 1.75,
            "unit": "ratio",
            "context": (
                "The unsecured credit facility requires unsecured leverage ratio not to exceed 60%, "
                "secured debt to total assets ratio <= 40%, "
                "unencumbered interest coverage ratio no less than 1.75 to 1.00, "
                "and unencumbered debt yield >= 10.0%."
            ),
        },
        {
            "metric": "unencumbered debt yield",
            "condition": "MINIMUM",
            "threshold": 10.0,
            "unit": "percent",
            "context": (
                "The unsecured credit facility requires unsecured leverage ratio not to exceed 60%, "
                "secured debt to total assets ratio <= 40%, "
                "unencumbered interest coverage ratio no less than 1.75 to 1.00, "
                "and unencumbered debt yield >= 10.0%."
            ),
        },
    ]


def test_analyze_capital_structure_resolution_infers_maturity_table_units(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0016",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all covenants. "
                            "Debt maturities ($ in millions): 2026 12; 2027 18; 2028 and thereafter 44."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert (
        result["capital_structure_terms"]["maturity_schedule_status"]
        == "MATURITY_SCHEDULE_EXTRACTED"
    )
    assert result["maturity_schedule"] == [
        {
            "year": 2026,
            "amount": 12.0,
            "unit": "million",
            "context": "Debt maturities ($ in millions): 2026 12; 2027 18; 2028 and thereafter 44.",
        },
        {
            "year": 2027,
            "amount": 18.0,
            "unit": "million",
            "context": "Debt maturities ($ in millions): 2026 12; 2027 18; 2028 and thereafter 44.",
        },
        {
            "year": None,
            "period": "2028 and thereafter",
            "amount": 44.0,
            "unit": "million",
            "context": "Debt maturities ($ in millions): 2026 12; 2027 18; 2028 and thereafter 44.",
        },
    ]


def test_analyze_capital_structure_resolution_extracts_scheduled_debt_payment_table(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0017",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all covenants. "
                            "Scheduled debt payments ($ in millions): 2026 14; 2027 16; thereafter 25."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert (
        result["capital_structure_terms"]["maturity_schedule_status"]
        == "MATURITY_SCHEDULE_EXTRACTED"
    )
    assert result["maturity_schedule"] == [
        {
            "year": 2026,
            "amount": 14.0,
            "unit": "million",
            "context": "Scheduled debt payments ($ in millions): 2026 14; 2027 16; thereafter 25.",
        },
        {
            "year": 2027,
            "amount": 16.0,
            "unit": "million",
            "context": "Scheduled debt payments ($ in millions): 2026 14; 2027 16; thereafter 25.",
        },
        {
            "year": None,
            "period": "thereafter",
            "amount": 25.0,
            "unit": "million",
            "context": "Scheduled debt payments ($ in millions): 2026 14; 2027 16; thereafter 25.",
        },
    ]


def test_analyze_capital_structure_resolution_preserves_year_and_thereafter_maturity_bucket(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0007",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all covenants. "
                            "Future debt maturities are as follows: 2026 $10 million; "
                            "2027 $20 million; 2028 and thereafter $70 million."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["maturity_schedule"] == [
        {
            "year": 2026,
            "amount": 10.0,
            "unit": "million",
            "context": (
                "Future debt maturities are as follows: 2026 $10 million; "
                "2027 $20 million; 2028 and thereafter $70 million."
            ),
        },
        {
            "year": 2027,
            "amount": 20.0,
            "unit": "million",
            "context": (
                "Future debt maturities are as follows: 2026 $10 million; "
                "2027 $20 million; 2028 and thereafter $70 million."
            ),
        },
        {
            "year": None,
            "period": "2028 and thereafter",
            "amount": 70.0,
            "unit": "million",
            "context": (
                "Future debt maturities are as follows: 2026 $10 million; "
                "2027 $20 million; 2028 and thereafter $70 million."
            ),
        },
    ]


def test_analyze_capital_structure_resolution_preserves_year_range_maturity_bucket(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0008",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all covenants. "
                            "Debt maturities are as follows: 2026 $10 million; "
                            "2027 $20 million; 2029-2031 $90 million."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["maturity_schedule"] == [
        {
            "year": 2026,
            "amount": 10.0,
            "unit": "million",
            "context": (
                "Debt maturities are as follows: 2026 $10 million; "
                "2027 $20 million; 2029-2031 $90 million."
            ),
        },
        {
            "year": 2027,
            "amount": 20.0,
            "unit": "million",
            "context": (
                "Debt maturities are as follows: 2026 $10 million; "
                "2027 $20 million; 2029-2031 $90 million."
            ),
        },
        {
            "year": None,
            "period": "2029-2031",
            "start_year": 2029,
            "end_year": 2031,
            "amount": 90.0,
            "unit": "million",
            "context": (
                "Debt maturities are as follows: 2026 $10 million; "
                "2027 $20 million; 2029-2031 $90 million."
            ),
        },
    ]


def test_analyze_capital_structure_resolution_extracts_duration_bucket_maturity_schedule(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0009",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all covenants. "
                            "Debt maturities are due as follows: less than one year $10 million; "
                            "one to three years $25 million; three to five years $40 million; "
                            "more than five years $55 million."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["maturity_schedule"] == [
        {
            "year": None,
            "period": "less than one year",
            "amount": 10.0,
            "unit": "million",
            "context": (
                "Debt maturities are due as follows: less than one year $10 million; "
                "one to three years $25 million; three to five years $40 million; "
                "more than five years $55 million."
            ),
        },
        {
            "year": None,
            "period": "1-3 years",
            "amount": 25.0,
            "unit": "million",
            "context": (
                "Debt maturities are due as follows: less than one year $10 million; "
                "one to three years $25 million; three to five years $40 million; "
                "more than five years $55 million."
            ),
        },
        {
            "year": None,
            "period": "3-5 years",
            "amount": 40.0,
            "unit": "million",
            "context": (
                "Debt maturities are due as follows: less than one year $10 million; "
                "one to three years $25 million; three to five years $40 million; "
                "more than five years $55 million."
            ),
        },
        {
            "year": None,
            "period": "more than five years",
            "amount": 55.0,
            "unit": "million",
            "context": (
                "Debt maturities are due as follows: less than one year $10 million; "
                "one to three years $25 million; three to five years $40 million; "
                "more than five years $55 million."
            ),
        },
    ]


def test_analyze_capital_structure_resolution_extracts_amount_first_duration_buckets(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0010",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all covenants. "
                            "Principal payments are $12 million within one year; "
                            "$24 million in one to three years; $36 million in three to five years; "
                            "$48 million in more than five years."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["maturity_schedule"] == [
        {
            "year": None,
            "period": "less than one year",
            "amount": 12.0,
            "unit": "million",
            "context": (
                "Principal payments are $12 million within one year; "
                "$24 million in one to three years; $36 million in three to five years; "
                "$48 million in more than five years."
            ),
        },
        {
            "year": None,
            "period": "1-3 years",
            "amount": 24.0,
            "unit": "million",
            "context": (
                "Principal payments are $12 million within one year; "
                "$24 million in one to three years; $36 million in three to five years; "
                "$48 million in more than five years."
            ),
        },
        {
            "year": None,
            "period": "3-5 years",
            "amount": 36.0,
            "unit": "million",
            "context": (
                "Principal payments are $12 million within one year; "
                "$24 million in one to three years; $36 million in three to five years; "
                "$48 million in more than five years."
            ),
        },
        {
            "year": None,
            "period": "more than five years",
            "amount": 48.0,
            "unit": "million",
            "context": (
                "Principal payments are $12 million within one year; "
                "$24 million in one to three years; $36 million in three to five years; "
                "$48 million in more than five years."
            ),
        },
    ]


def test_analyze_capital_structure_resolution_extracts_specific_maturity_dates(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0011",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all covenants. "
                            "Debt maturities are as follows: March 15, 2028 $125 million; "
                            "September 30, 2029 $175 million. "
                            "The $200 million senior notes mature on 2030-06-01."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["maturity_schedule"] == [
        {
            "year": 2028,
            "maturity_date": "2028-03-15",
            "amount": 125.0,
            "unit": "million",
            "context": (
                "Debt maturities are as follows: March 15, 2028 $125 million; "
                "September 30, 2029 $175 million."
            ),
        },
        {
            "year": 2029,
            "maturity_date": "2029-09-30",
            "amount": 175.0,
            "unit": "million",
            "context": (
                "Debt maturities are as follows: March 15, 2028 $125 million; "
                "September 30, 2029 $175 million."
            ),
        },
        {
            "year": 2030,
            "maturity_date": "2030-06-01",
            "amount": 200.0,
            "unit": "million",
            "context": "The $200 million senior notes mature on 2030-06-01.",
        },
    ]


def test_analyze_capital_structure_resolution_extracts_month_year_maturity_dates(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0018",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all covenants. "
                            "The $75 million term loan matures in May 2027. "
                            "Debt maturities are as follows: November 2028 $125 million."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["maturity_schedule"] == [
        {
            "year": 2027,
            "maturity_date": "2027-05-01",
            "amount": 75.0,
            "unit": "million",
            "context": "The $75 million term loan matures in May 2027.",
        },
        {
            "year": 2028,
            "maturity_date": "2028-11-01",
            "amount": 125.0,
            "unit": "million",
            "context": "Debt maturities are as follows: November 2028 $125 million.",
        },
    ]


def test_analyze_capital_structure_resolution_extracts_amount_first_year_buckets(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0012",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all covenants. "
                            "Debt maturities are as follows: $10 million in 2026; "
                            "$20 million in 2027; $30 million during 2028; $40 million thereafter."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["maturity_schedule"] == [
        {
            "year": 2026,
            "amount": 10.0,
            "unit": "million",
            "context": (
                "Debt maturities are as follows: $10 million in 2026; "
                "$20 million in 2027; $30 million during 2028; $40 million thereafter."
            ),
        },
        {
            "year": 2027,
            "amount": 20.0,
            "unit": "million",
            "context": (
                "Debt maturities are as follows: $10 million in 2026; "
                "$20 million in 2027; $30 million during 2028; $40 million thereafter."
            ),
        },
        {
            "year": 2028,
            "amount": 30.0,
            "unit": "million",
            "context": (
                "Debt maturities are as follows: $10 million in 2026; "
                "$20 million in 2027; $30 million during 2028; $40 million thereafter."
            ),
        },
        {
            "year": None,
            "period": "thereafter",
            "amount": 40.0,
            "unit": "million",
            "context": (
                "Debt maturities are as follows: $10 million in 2026; "
                "$20 million in 2027; $30 million during 2028; $40 million thereafter."
            ),
        },
    ]


def test_analyze_capital_structure_resolution_extracts_conjunction_year_buckets(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0013",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html=(
                            "The company was in compliance with all covenants. "
                            "Debt maturities are as follows: 2026 and 2027 $30 million; 2028 $40 million. "
                            "Principal payments are $50 million in 2029 and 2030; $60 million in 2031."
                        ),
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
    )

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["capital_structure_terms"]["maturity_schedule"] == [
        {
            "year": None,
            "period": "2026 and 2027",
            "start_year": 2026,
            "end_year": 2027,
            "amount": 30.0,
            "unit": "million",
            "context": "Debt maturities are as follows: 2026 and 2027 $30 million; 2028 $40 million.",
        },
        {
            "year": 2028,
            "amount": 40.0,
            "unit": "million",
            "context": "Debt maturities are as follows: 2026 and 2027 $30 million; 2028 $40 million.",
        },
        {
            "year": None,
            "period": "2029 and 2030",
            "start_year": 2029,
            "end_year": 2030,
            "amount": 50.0,
            "unit": "million",
            "context": "Principal payments are $50 million in 2029 and 2030; $60 million in 2031.",
        },
        {
            "year": 2031,
            "amount": 60.0,
            "unit": "million",
            "context": "Principal payments are $50 million in 2029 and 2030; $60 million in 2031.",
        },
    ]


def test_analyze_capital_structure_resolution_keeps_near_term_maturity_watchlist(monkeypatch):
    packet = _packet()
    packet.solvency_risk = "ELEVATED"
    packet.research_report["solvency"] = {
        "risk": "ELEVATED",
        "signals": ["DEBT_DUE_WITHIN_12MO"],
        "details": "Debt maturities within 12 months",
        "going_concern_language": False,
        "no_assurance_financing": False,
        "debt_due_within_12mo": True,
    }
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {
            "cash": [{"fiscal_year": 2025, "value": 10.0}],
            "total_debt": [{"fiscal_year": 2025, "value": 80.0}],
            "current_assets": [],
            "current_liabilities": [],
            "cfo": [],
        },
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[
                    FilingDocument(
                        ticker=ticker,
                        cik="1",
                        accession="0003",
                        form_type="10-Q",
                        filing_date="2026-02-01",
                        period_end="2025-12-31",
                        role="quarterly",
                        local_path=None,
                        primary_doc_url=None,
                        html="Liquidity discussion notes debt maturities within the next twelve months and ongoing refinancing evaluation.",
                    )
                ],
                recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            )
        ),
    )
    ctx = AlphaToolContext(sector="software", ticker="AAA", packet=packet, as_of_date="2026-04-19")

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "ok"
    assert result["evidence_status"] == "CAPITAL_STRUCTURE_RESOLVED_WATCHLIST"
    assert result["stress_flags"]["debt_due_within_12mo"] is True
    assert "watchlist cap" in result["summary"]


def test_analyze_capital_structure_resolution_unreadable_context_is_non_usable(monkeypatch):
    packet = _packet()
    packet.solvency_risk = None
    packet.research_report["solvency"] = {}
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    monkeypatch.setattr(
        "app.alpha.llm_tools.load_research_filing_context",
        lambda ticker, as_of_date, quarters, include_material_events, material_event_window_days, material_event_limit: (
            FilingContext(
                documents=[],
                warnings=["no_recent_quarterly_filings_cached"],
                recent_filing_status="NO_RECENT_FILINGS_CACHED",
            )
        ),
    )
    ctx = AlphaToolContext(sector="software", ticker="AAA", packet=packet, as_of_date="2026-04-19")

    result = dispatch_alpha_tool("analyze_capital_structure_resolution", {}, ctx)

    assert result["status"] == "unavailable"
    assert result["usable_for_decision"] is False
    assert result["evidence_status"] == "CAPITAL_STRUCTURE_UNRESOLVED"
    assert result["warnings"] == [
        "capital_structure_filing_context_unreadable",
        "capital_structure_companyfacts_unavailable",
    ]


def test_analyze_capital_allocation_exposes_decision_usable_summary(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {
            "shares_outstanding": [
                {
                    "fiscal_year": 2021,
                    "value": 100.0,
                    "raw_value": 100.0,
                    "normalized_value": 100.0,
                    "shares_basis": "UNADJUSTED",
                    "split_adjustment_factor": 1.0,
                    "split_effective_date": None,
                },
                {
                    "fiscal_year": 2025,
                    "value": 92.0,
                    "raw_value": 92.0,
                    "normalized_value": 92.0,
                    "shares_basis": "UNADJUSTED",
                    "split_adjustment_factor": 1.0,
                    "split_effective_date": None,
                },
            ]
        },
    )
    packet = _packet()
    packet.raw_quality_ctx = {
        "earnings_quality": "HIGH",
        "revenue_cagr_5y": 0.12,
        "revenue_cagr_3y": 0.15,
    }
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=packet,
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool("analyze_capital_allocation", {}, ctx)

    assert result["status"] == "ok"
    assert result["usable_for_decision"] is True
    assert result["evidence_status"] == "CAPITAL_ALLOCATION_CONTEXT_AVAILABLE"
    assert result["summary"] == (
        "AAA capital allocation context: earnings quality HIGH, 5Y revenue CAGR 12.0%, "
        "dilution BUYBACKS_OR_SHARE_REDUCTION, supports PEER_LEADER_SUPPORT, "
        "headwinds DILUTION_HEADWIND."
    )


def test_fetch_companyfacts_timeseries_empty_series_is_non_usable(monkeypatch):
    monkeypatch.setattr(
        "app.alpha.llm_tools._companyfacts_series",
        lambda ticker, line_items, years, as_of_date: {item: [] for item in line_items},
    )
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_companyfacts_timeseries",
        {"line_items": ["equity"], "years": 5},
        ctx,
    )

    assert result["status"] == "unavailable"
    assert result["usable_for_decision"] is False
    assert result["evidence_status"] == "NO_COMPANYFACTS_SERIES"
    assert result["requested_line_items"] == ["equity"]
    assert result["populated_line_items"] == []
    assert result["missing_line_items"] == ["equity"]


def test_fetch_companyfacts_timeseries_normalizes_common_gpt_aliases(monkeypatch):
    observed: list[list[str]] = []

    def fake_series(ticker, line_items, years, as_of_date):
        observed.append(line_items)
        return {
            "cfo": [{"fiscal_year": 2025, "value": 120.0}],
            "capex": [{"fiscal_year": 2025, "value": -20.0}],
            "shares_outstanding": [{"fiscal_year": 2025, "value": 10.0}],
            "sbc": [{"fiscal_year": 2025, "value": 5.0}],
        }

    monkeypatch.setattr("app.alpha.llm_tools._companyfacts_series", fake_series)
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-04-19",
        consensus_rank=1,
        consensus_score=20.0,
    )

    result = dispatch_alpha_tool(
        "fetch_companyfacts_timeseries",
        {
            "line_items": [
                "free_cash_flow",
                "operating_cash_flow",
                "diluted_shares",
                "stock_based_compensation",
            ],
            "years": 5,
        },
        ctx,
    )

    assert observed == [["cfo", "capex", "shares_outstanding", "sbc"]]
    assert result["status"] == "ok"
    assert result["requested_line_items"] == ["cfo", "capex", "shares_outstanding", "sbc"]
    assert result["translated_line_items"] == [
        "free_cash_flow",
        "operating_cash_flow",
        "diluted_shares",
        "stock_based_compensation",
    ]
    assert result["populated_line_items"] == ["cfo", "capex", "shares_outstanding", "sbc"]


def test_fetch_companyfacts_timeseries_rejects_filed_after_asof_and_carries_provenance(
    monkeypatch,
    tmp_path,
):
    cfg = _init_companyfacts_db(monkeypatch, tmp_path)
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"
    with get_db(cfg) as conn:
        conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form,
                accession
            )
            VALUES(
                'AAA', ?, 'FY', ?, 'revenue', ?, 'USD_millions', ?,
                '2026-03-01T00:00:00+00:00', ?, '10-K', ?
            )
            """,
            [
                (
                    2024,
                    "2024-12-31",
                    100.0,
                    source_url,
                    "2025-02-15",
                    "0001-valid",
                ),
                (
                    2025,
                    "2025-12-31",
                    200.0,
                    source_url,
                    "2026-02-15",
                    "0001-future",
                ),
            ],
        )
        conn.commit()
    ctx = AlphaToolContext(
        sector="software",
        ticker="AAA",
        packet=_packet(),
        as_of_date="2026-01-31",
    )

    result = dispatch_alpha_tool(
        "fetch_companyfacts_timeseries",
        {"line_items": ["revenue"], "years": 5},
        ctx,
    )

    assert result["status"] == "ok"
    assert result["series"]["revenue"] == [
        {
            "fiscal_year": 2024,
            "value": 100.0,
            "unit": "USD_millions",
            "units": "USD_millions",
            "period_end": "2024-12-31",
            "filed_date": "2025-02-15",
            "source": "SEC_COMPANYFACTS",
            "source_url": source_url,
            "source_reference": "0001-valid",
            "accession": "0001-valid",
            "as_of_date": "2026-01-31",
        }
    ]
    from app.config import get_config

    get_config.cache_clear()


def test_insurance_evidence_gate_blocks_missing_pc_core_metrics():
    packet = _packet()
    packet.issuer_type = "insurance_underwriter"
    packet.insurance_subtype = "pc_insurer"
    packet.filing_risk_signals = {
        "competitive_disruption": "LOW",
        "secular_decline": "LOW",
        "regulatory_legal": "LOW",
        "customer_concentration": "LOW",
    }
    packet.insurance_packet = {
        "operating_metrics": {
            "metric_family": "pc_insurance",
            "status": "LIMITED",
            "missing_components": ["COMBINED_RATIO", "LOSS_RATIO"],
        }
    }
    ctx = AlphaToolContext(
        sector="insurance",
        ticker="AAA",
        packet=packet,
        as_of_date="2026-04-19",
    )
    result = AlphaCandidateLoopResult(
        ticker="AAA",
        verdict="PROCEED",
        confidence="HIGH",
        model_validity="VALID",
        eligible_for_selection=True,
    )

    _apply_deterministic_evidence_gates(ctx, result)

    assert result.selection_blockers == ["PC_CORE_OPERATING_METRICS_MISSING"]
    assert result.confidence_cap_reasons == ["PC_CORE_OPERATING_METRICS_MISSING"]
    assert result.confidence == "LOW"
    assert result.eligible_for_selection is False


def test_insurance_evidence_gate_caps_confidence_for_missing_reserve_and_statutory_evidence():
    packet = _packet()
    packet.issuer_type = "insurance_underwriter"
    packet.insurance_subtype = "pc_insurer"
    packet.model_fit_warnings = ["RBC_OR_BCAR"]
    packet.filing_risk_signals = {
        "competitive_disruption": "LOW",
        "secular_decline": "LOW",
        "regulatory_legal": "LOW",
        "customer_concentration": "LOW",
    }
    packet.insurance_packet = {
        "operating_metrics": {
            "metric_family": "pc_insurance",
            "status": "OK",
            "combined_ratio": 0.92,
            "loss_ratio": 0.61,
            "expense_ratio": 0.31,
            "missing_components": ["RESERVE_DEVELOPMENT"],
        }
    }
    ctx = AlphaToolContext(
        sector="insurance",
        ticker="AAA",
        packet=packet,
        as_of_date="2026-04-19",
    )
    result = AlphaCandidateLoopResult(
        ticker="AAA",
        verdict="PROCEED",
        confidence="HIGH",
        model_validity="VALID",
        eligible_for_selection=True,
    )

    _apply_deterministic_evidence_gates(ctx, result)

    assert result.selection_blockers == []
    assert result.confidence_cap_reasons == [
        "PC_RESERVE_DEVELOPMENT_UNKNOWN",
        "INSURANCE_STATUTORY_RATING_OR_ALM_EVIDENCE_MISSING",
    ]
    assert result.confidence == "MODERATE"
    assert result.eligible_for_selection is True


def test_insurance_evidence_gate_blocks_missing_mortgage_core_metrics():
    packet = _packet()
    packet.issuer_type = "insurance_underwriter"
    packet.insurance_subtype = "title_mortgage_specialty"
    packet.filing_risk_signals = {
        "competitive_disruption": "LOW",
        "secular_decline": "LOW",
        "regulatory_legal": "LOW",
        "customer_concentration": "LOW",
    }
    packet.insurance_packet = {
        "operating_metrics": {
            "metric_family": "mortgage_insurance",
            "status": "LIMITED",
            "missing_components": ["PMIER_EXCESS_RATIO", "DEFAULT_RATE"],
        }
    }
    ctx = AlphaToolContext(
        sector="insurance",
        ticker="AAA",
        packet=packet,
        as_of_date="2026-04-19",
    )
    result = AlphaCandidateLoopResult(
        ticker="AAA",
        verdict="PROCEED",
        confidence="MODERATE",
        model_validity="VALID",
        eligible_for_selection=True,
    )

    _apply_deterministic_evidence_gates(ctx, result)

    assert result.selection_blockers == ["MORTGAGE_CORE_OPERATING_METRICS_MISSING"]
    assert result.confidence_cap_reasons == ["MORTGAGE_CORE_OPERATING_METRICS_MISSING"]
    assert result.confidence == "LOW"
    assert result.eligible_for_selection is False


def test_run_candidate_investigation_fallback_records_tool_transcript(monkeypatch):
    class _FakeProvider:
        # The fallback bundle only pays for a summary when the provider
        # affirmatively reports enabled; a mock without enabled() is treated
        # as unavailable and never called.
        def enabled(self):
            return True

        def synthesize_json(self, *, prompt, schema, schema_name=None, max_output_tokens=None):
            _ = (prompt, schema, schema_name, max_output_tokens)
            payload = {
                "verdict": "PROCEED",
                "confidence": "HIGH",
                "key_findings": ["Primary evidence stayed supportive."],
                "open_questions": ["Need one more peer check."],
                "key_risk": "Execution miss.",
                "falsification_trigger": "If execution slips materially.",
                "reasoning_trace": "Fallback summary converted deterministic tool outputs into a candidate verdict.",
                "model_validity": "VALID",
                "selection_blockers": [],
                "eligible_for_selection": True,
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="fake",
                usage_input_tokens=None,
                usage_output_tokens=None,
                raw={},
            )

    monkeypatch.setattr("app.alpha.llm_runtime.get_anthropic_provider", lambda: None)
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: _FakeProvider())
    monkeypatch.setattr(
        "app.alpha.llm_runtime.suggested_tool_calls_for_gaps",
        lambda evidence_gaps: [
            ("fetch_kpi_trends", {}),
            ("analyze_dilution", {"years": 5}),
        ],
    )
    monkeypatch.setattr(
        "app.alpha.llm_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "tool": name,
            "input_echo": dict(tool_input),
            "ticker": ctx.ticker,
        },
    )

    packet = _packet()
    result = run_candidate_investigation(
        ctx=AlphaToolContext(
            sector="software",
            ticker="AAA",
            packet=packet,
            as_of_date="2026-04-19",
            consensus_rank=1,
            consensus_score=20.0,
        ),
        investigation_request={
            "ticker": "AAA",
            "reason": "Need KPI and dilution evidence.",
            "evidence_gaps": ["Need KPI trend.", "Need dilution check."],
        },
        hard_block_reasons=[],
        integrity_scope=_integrity_scope(packet),
    )

    assert result["investigation_mode"] == "fallback_tool_bundle"
    assert result["verdict"] == "PROCEED"
    assert result["confidence"] == "HIGH"
    assert result["tool_call_counts"] == {"fetch_kpi_trends": 1, "analyze_dilution": 1}
    assert len(result["tool_transcript"]) == 2
    assert result["tool_transcript"][0]["tool"] == "fetch_kpi_trends"
    assert result["requested_focus"] == {
        "reason": "Need KPI and dilution evidence.",
        "evidence_gaps": ["Need KPI trend.", "Need dilution check."],
    }
    assert result["model_validity"] == "VALID"
    assert result["selection_blockers"] == []
    assert result["eligible_for_selection"] is True


def test_run_candidate_investigation_labels_configured_openai_tool_bundle(monkeypatch):
    class _FakeOpenAIProvider:
        provider_name = "openai"

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, *, prompt, schema, schema_name=None, max_output_tokens=None):
            _ = (prompt, schema, schema_name, max_output_tokens)
            payload = {
                "verdict": "PROCEED",
                "confidence": "MODERATE",
                "key_findings": ["OpenAI summary evaluated deterministic tool output."],
                "open_questions": [],
                "key_risk": "Evidence still needs monitoring.",
                "falsification_trigger": "If KPI trend deteriorates.",
                "reasoning_trace": "Configured GPT provider summarized the alpha tool bundle.",
                "model_validity": "VALID",
                "selection_blockers": [],
                "eligible_for_selection": True,
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="fake-gpt",
                usage_input_tokens=None,
                usage_output_tokens=None,
                raw={},
            )

    monkeypatch.setattr(
        "app.alpha.llm_runtime.get_alpha_llm_provider", lambda: _FakeOpenAIProvider()
    )
    monkeypatch.setattr(
        "app.alpha.llm_runtime.suggested_tool_calls_for_gaps",
        lambda evidence_gaps: [("fetch_kpi_trends", {})],
    )
    monkeypatch.setattr(
        "app.alpha.llm_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {"status": "ok", "ticker": ctx.ticker, "tool": name},
    )

    packet = _packet()
    result = run_candidate_investigation(
        ctx=AlphaToolContext(
            sector="software",
            ticker="AAA",
            packet=packet,
            as_of_date="2026-04-19",
            consensus_rank=1,
            consensus_score=20.0,
        ),
        investigation_request={
            "ticker": "AAA",
            "reason": "Validate with GPT.",
            "evidence_gaps": ["Need KPI trend."],
        },
        hard_block_reasons=[],
        integrity_scope=_integrity_scope(packet),
    )

    assert result["investigation_mode"] == "openai_tool_bundle"
    assert result["reasoning_trace"] == "Configured GPT provider summarized the alpha tool bundle."


def test_run_candidate_investigation_model_mismatch_blocks_selection(monkeypatch):
    class _FakeProvider:
        def enabled(self):
            return True

        def synthesize_json(self, *, prompt, schema, schema_name=None, max_output_tokens=None):
            _ = (prompt, schema, schema_name, max_output_tokens)
            payload = {
                "verdict": "AVOID",
                "confidence": "HIGH",
                "key_findings": [
                    "The requested ticker is a preferred/depositary security, not common equity."
                ],
                "open_questions": [],
                "key_risk": "Common-equity valuation is inapplicable.",
                "falsification_trigger": "Only a security-specific preferred model could revive the idea.",
                "reasoning_trace": "Model mismatch discovered during investigation.",
                "model_validity": "INVALID_SECURITY_TYPE",
                "selection_blockers": [],
                "eligible_for_selection": False,
            }
            return LLMResult(
                json_text=json.dumps(payload),
                model="fake",
                usage_input_tokens=None,
                usage_output_tokens=None,
                raw={},
            )

    monkeypatch.setattr("app.alpha.llm_runtime.get_anthropic_provider", lambda: None)
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: _FakeProvider())
    monkeypatch.setattr(
        "app.alpha.llm_runtime.suggested_tool_calls_for_gaps", lambda evidence_gaps: []
    )

    packet = _packet("BADP")
    result = run_candidate_investigation(
        ctx=AlphaToolContext(
            sector="insurance",
            ticker="BADP",
            packet=packet,
            as_of_date="2026-04-19",
            consensus_rank=1,
            consensus_score=90.0,
        ),
        investigation_request={
            "ticker": "BADP",
            "reason": "Validate security identity.",
            "evidence_gaps": [],
        },
        hard_block_reasons=[],
        integrity_scope=_integrity_scope(packet),
    )

    assert result["verdict"] == "AVOID"
    assert result["model_validity"] == "INVALID_SECURITY_TYPE"
    assert result["selection_blockers"] == ["INVALID_SECURITY_TYPE"]
    assert result["eligible_for_selection"] is False


def test_final_decision_failure_returns_no_winner(monkeypatch):
    class _FailingProvider:
        def synthesize_json(self, **kwargs):
            raise ConnectionError("provider offline")

    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: _FailingProvider())

    result = decide_alpha_winner(
        sector="insurance",
        prior_ranking=[{"ticker": "AAA", "consensus_rank": 1}],
        investigation_plan={"selected_targets": [{"ticker": "AAA"}]},
        candidate_investigations=[
            {
                "ticker": "AAA",
                "eligible_for_selection": True,
                "hard_block_reasons": [],
                "selection_blockers": [],
            }
        ],
        hard_blocked_candidates=[],
        integrity_scope=_integrity_scope(_packet()),
    )

    assert result["winner"] is None
    assert result["runner_up"] is None
    assert result["decision_mode"] == "fallback"
    assert "non-actionable" in result["key_risk"]
