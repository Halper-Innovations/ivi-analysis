"""Tests for app.alpha.filing_risk_scan."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.db import get_db, init_db, utc_now_iso
from tests.financial_integrity_helpers import canonicalize_financial_packet


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config as _gc

    _gc.cache_clear()
    cfg = _gc()
    init_db(cfg)
    return cfg


def _seed_filing(tmp_path, ticker: str = "NICE", form_type: str = "20-F"):
    """Create a fake filing in the DB with a cached HTML file on disk."""
    filing_dir = tmp_path / "filings"
    filing_dir.mkdir(exist_ok=True)
    filing_path = filing_dir / f"{ticker.lower()}-test.htm"
    filing_path.write_text(
        """<html><body>
        <div>Item 1A. Risk Factors</div>
        <div>
        Our business faces significant risks from artificial intelligence and
        machine learning technologies. Generative AI products from competitors
        could disrupt our core contact center and customer engagement platform.
        Large language models may automate tasks that our software currently handles,
        reducing demand for our solutions. We face secular decline in our legacy
        voice solutions business. Revenue concentration risk exists as our top
        10 customers account for 35% of revenue.
        Additionally, regulatory requirements in the EU and US around AI governance
        could increase our compliance costs substantially.
        </div>
        <div>Item 1B. Unresolved Staff Comments</div>
        </body></html>""",
        encoding="utf-8",
    )
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO filings
               (cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "0001003935",
                ticker,
                "0001003935-26-000010",
                form_type,
                "2026-02-26",
                "2025-12-31",
                "https://sec.gov/test",
                str(filing_path),
                "OK",
                now,
                now,
            ),
        )
    return str(filing_path)


def _long_risk_body(marker: str = "material risk") -> str:
    return (
        f"Our business faces {marker} from competition, customer concentration, cybersecurity, "
        "regulatory change, capital allocation mistakes, acquisition execution, macroeconomic pressure, "
        "and platform reliability. These risks could materially harm revenue, margins, liquidity, "
        "cash flows, customer retention, and the trading price of our common stock. "
    ) * 4


def _integrity_scope(ticker: str, as_of_date: str):
    from app.alpha.schemas import TickerSignalPacket
    from app.autonomous.financial_integrity import (
        FinancialIntegrityScope,
        stable_quote_hash,
    )

    snapshot_id = stable_quote_hash(
        ticker=ticker,
        price=50.0,
        as_of_date=as_of_date,
        currency="USD",
        source="fixture_quote",
        price_basis="UNADJUSTED",
        raw_price=50.0,
        split_adjustment_factor=1.0,
    )
    packet = TickerSignalPacket(
        ticker=ticker,
        current_price=50.0,
        current_price_unit="USD_per_share",
        current_price_as_of_date=as_of_date,
        current_price_currency="USD",
        current_price_source="fixture_quote",
        quote_snapshot_id=snapshot_id,
        price_basis="UNADJUSTED",
        raw_price=50.0,
        split_adjustment_factor=1.0,
        market_cap_mm=500.0,
        market_cap_unit="USD_millions",
        market_cap_source="fixture_market_cap",
        market_cap_effective_as_of_date=as_of_date,
        market_cap_method="price_times_shares_divided_by_issuer_quote_ratio",
        shares_outstanding_mm=10.0,
        raw_shares_outstanding_mm=10.0,
        shares_unit="shares_millions",
        shares_basis="UNADJUSTED",
        shares_as_of_date=as_of_date,
        shares_source="fixture_filing",
        split_lineage_proof={
            "status": "PASS",
            "period_start": as_of_date,
            "period_end": as_of_date,
            "verified_as_of": as_of_date,
            "source": "fixture_corporate_actions_ledger",
            "source_reference": f"fixture://corporate-actions/{ticker}",
        },
        issuer_quote_ratio=1.0,
        issuer_primary_ticker=ticker,
        issuer_listed_tickers=[ticker],
        security_role="PRIMARY",
        is_secondary_class=False,
        is_adr=False,
        identity_source="sec_submissions_exchange_binding",
        identity_source_url=(f"https://data.sec.gov/submissions/CIK0000000001.json#{ticker}"),
        identity_as_of_date=as_of_date,
        identity_confidence="HIGH",
        cap_stage_price=50.0,
        cap_stage_price_as_of_date=as_of_date,
        cap_stage_price_currency="USD",
        cap_stage_price_source="fixture_quote",
        cap_stage_quote_snapshot_id=snapshot_id,
    )
    canonicalize_financial_packet(packet, as_of_date=as_of_date, shares_mm=10.0)
    return FinancialIntegrityScope(
        context=f"filing_risk_test:{ticker}",
        run_as_of_date=as_of_date,
        packets=(packet,),
    )


def test_scan_filing_risks_basic(monkeypatch, tmp_path):
    """Should return risk classifications for a ticker with a cached filing."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "NICE")

    mock_llm_response = json.dumps(
        {
            "competitive_disruption": "HIGH",
            "secular_decline": "MODERATE",
            "regulatory_legal": "MODERATE",
            "customer_concentration": "LOW",
            "summary": "Core contact center business faces direct AI disruption threat.",
        }
    )
    mock_result = MagicMock()
    mock_result.json_text = mock_llm_response
    mock_provider = MagicMock()
    mock_provider.synthesize_json.return_value = mock_result
    mock_provider.provider_name = "openai"

    with patch("app.alpha.filing_risk_scan.get_llm_provider", return_value=mock_provider):
        from app.alpha.filing_risk_scan import scan_filing_risks, _RISK_CACHE

        _RISK_CACHE.clear()
        result = scan_filing_risks(
            "NICE",
            integrity_scope=_integrity_scope("NICE", date.today().isoformat()),
        )

    assert result["competitive_disruption"] == "HIGH"
    assert result["secular_decline"] == "MODERATE"
    assert result["status"] == "OK"
    assert result["evidence_status"] == "READABLE_RISK_SECTION"
    assert result["source_accession"] == "0001003935-26-000010"
    assert result["risk_text_chars"] > 200
    assert len(result["provider_usage"]) == 1
    assert result["cost_usd"] >= 0.0
    call_args = mock_provider.synthesize_json.call_args
    assert "risk" in call_args.kwargs["prompt"].lower()


def test_filing_risk_zero_budget_suppresses_provider_call(monkeypatch, tmp_path):
    from app.alpha.filing_risk_scan import _RISK_CACHE, scan_filing_risks
    from app.llm.providers.retry_guard import LLMCostBudgetExceeded
    from app.llm.usage_capture import provider_usage_budget

    class _Provider:
        provider_name = "openai"
        model = "gpt-5-mini"

        def __init__(self):
            self.calls = 0

        def synthesize_json(self, **kwargs):
            self.calls += 1
            raise AssertionError(f"zero budget must suppress filing risk: {kwargs}")

    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "NICE")
    provider = _Provider()
    _RISK_CACHE.clear()

    with (
        patch("app.alpha.filing_risk_scan.get_llm_provider", return_value=provider),
        provider_usage_budget(0.0),
        pytest.raises(LLMCostBudgetExceeded),
    ):
        scan_filing_risks(
            "NICE",
            integrity_scope=_integrity_scope("NICE", date.today().isoformat()),
        )

    assert provider.calls == 0
    assert _RISK_CACHE == {}


def test_filing_risk_failed_billed_response_is_reported_in_fallback(
    monkeypatch,
    tmp_path,
):
    from app.alpha.filing_risk_scan import _RISK_CACHE, scan_filing_risks

    class _BilledFailureProvider:
        provider_name = "openai"
        model = "gpt-5-mini"

        def __init__(self):
            self.calls = 0

        def synthesize_json(self, **_kwargs):
            self.calls += 1
            error = RuntimeError("parse failed after billed filing-risk response")
            error._provider_response_attempts = [
                {
                    "status": "completed",
                    "model": self.model,
                    "usage": {"input_tokens": 120, "output_tokens": 25},
                }
            ]
            raise error

    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "NICE")
    provider = _BilledFailureProvider()
    _RISK_CACHE.clear()

    with patch("app.alpha.filing_risk_scan.get_llm_provider", return_value=provider):
        result = scan_filing_risks(
            "NICE",
            integrity_scope=_integrity_scope("NICE", date.today().isoformat()),
        )

    assert provider.calls == 1
    assert result["status"] == "KEYWORD_FALLBACK"
    assert result["provider_usage"][0]["status"] == "OK"
    assert result["provider_usage"][0]["input_tokens"] == 120
    assert result["provider_usage"][0]["output_tokens"] == 25
    assert result["cost_usd"] > 0.0


def test_filing_risk_invalid_billed_json_is_reported_in_fallback(
    monkeypatch,
    tmp_path,
):
    from app.alpha.filing_risk_scan import _RISK_CACHE, scan_filing_risks

    class _InvalidJsonProvider:
        provider_name = "openai"
        model = "gpt-5-mini"

        def __init__(self):
            self.calls = 0

        def synthesize_json(self, **_kwargs):
            self.calls += 1
            result = MagicMock()
            result.json_text = "{not valid json"
            result.usage_input_tokens = 130
            result.usage_output_tokens = 20
            result.model = self.model
            return result

    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "NICE")
    provider = _InvalidJsonProvider()
    _RISK_CACHE.clear()

    with patch("app.alpha.filing_risk_scan.get_llm_provider", return_value=provider):
        result = scan_filing_risks(
            "NICE",
            integrity_scope=_integrity_scope("NICE", date.today().isoformat()),
        )

    assert provider.calls == 1
    assert result["status"] == "KEYWORD_FALLBACK"
    assert result["provider_usage"][0]["status"] == "OK"
    assert result["provider_usage"][0]["input_tokens"] == 130
    assert result["provider_usage"][0]["output_tokens"] == 20
    assert result["cost_usd"] > 0.0


def test_invalid_financial_scope_suppresses_filing_risk_provider_and_fallback(
    monkeypatch,
    tmp_path,
):
    from app.alpha.filing_risk_scan import _RISK_CACHE, scan_filing_risks
    from app.alpha.schemas import TickerSignalPacket
    from app.autonomous.financial_integrity import (
        FinancialIntegrityScope,
        InvalidFinancialInputError,
    )

    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "NICE")
    provider = MagicMock()
    provider.provider_name = "openai"
    provider.synthesize_json.side_effect = AssertionError(
        "invalid financial context must suppress the provider"
    )
    invalid_scope = FinancialIntegrityScope(
        context="filing_risk_invalid_scope",
        run_as_of_date=date.today().isoformat(),
        packets=(TickerSignalPacket(ticker="NICE", current_price=50.0),),
    )
    _RISK_CACHE.clear()

    with patch("app.alpha.filing_risk_scan.get_llm_provider", return_value=provider):
        with pytest.raises(InvalidFinancialInputError):
            scan_filing_risks("NICE", integrity_scope=invalid_scope)

    assert provider.synthesize_json.call_count == 0
    assert _RISK_CACHE == {}


def test_filing_risk_retry_revalidates_exact_scope_before_second_attempt(
    monkeypatch,
    tmp_path,
):
    from app.alpha.filing_risk_scan import _RISK_CACHE, scan_filing_risks
    from app.autonomous.financial_integrity import InvalidFinancialInputError
    from app.llm.providers.disabled_provider import LLMResult
    from app.llm.providers.retry_guard import call_with_llm_retry_guard

    class _RetryableProviderError(RuntimeError):
        status_code = 503

    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "NICE")
    scope = _integrity_scope("NICE", date.today().isoformat())
    packet = scope.packets[0]
    physical_calls = 0

    class _RetryingProvider:
        provider_name = "openai"

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
                            "competitive_disruption": "LOW",
                            "secular_decline": "LOW",
                            "regulatory_legal": "LOW",
                            "customer_concentration": "LOW",
                            "summary": "retry should never publish",
                        }
                    ),
                    model="fake",
                    usage_input_tokens=0,
                    usage_output_tokens=0,
                    raw={},
                )

            return call_with_llm_retry_guard(
                provider_name="openai",
                schema_name="filing_risk_scan_v1",
                call=physical_call,
                max_retries=1,
                backoff_seconds=(0.0,),
                sleep_fn=lambda _seconds: None,
            )

    _RISK_CACHE.clear()
    with patch(
        "app.alpha.filing_risk_scan.get_llm_provider",
        return_value=_RetryingProvider(),
    ):
        with pytest.raises(InvalidFinancialInputError) as exc_info:
            scan_filing_risks("NICE", integrity_scope=scope)

    assert physical_calls == 1
    assert {item.code for item in exc_info.value.result.violations} == {
        "BOUND_FINANCIAL_INPUT_MUTATED"
    }
    assert _RISK_CACHE == {}


def test_scan_no_filing_returns_unknown(monkeypatch, tmp_path):
    """Ticker with no cached filing should return UNKNOWN for all dimensions."""
    _init_temp_db(monkeypatch, tmp_path)
    from app.alpha.filing_risk_scan import scan_filing_risks, _RISK_CACHE

    _RISK_CACHE.clear()
    result = scan_filing_risks("ZZZZ")
    assert result["status"] == "NO_FILING"
    assert result["evidence_status"] == "NO_READABLE_ANNUAL_FILING"
    assert result["competitive_disruption"] == "UNKNOWN"
    assert result["secular_decline"] == "UNKNOWN"


def test_scan_llm_disabled_returns_keyword_fallback(monkeypatch, tmp_path):
    """When LLM is disabled, fall back to keyword-based heuristic scan."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "NICE")

    mock_provider = MagicMock()
    mock_provider.provider_name = "disabled"

    with patch("app.alpha.filing_risk_scan.get_llm_provider", return_value=mock_provider):
        from app.alpha.filing_risk_scan import scan_filing_risks, _RISK_CACHE

        _RISK_CACHE.clear()
        result = scan_filing_risks("NICE")

    assert result["status"] == "KEYWORD_FALLBACK"
    assert result["competitive_disruption"] in ("HIGH", "MODERATE")


def test_scan_caches_result(monkeypatch, tmp_path):
    """Second call for same ticker should not call LLM again."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "TEST")

    mock_llm_response = json.dumps(
        {
            "competitive_disruption": "LOW",
            "secular_decline": "LOW",
            "regulatory_legal": "LOW",
            "customer_concentration": "LOW",
            "summary": "No major risks identified.",
        }
    )
    mock_result = MagicMock()
    mock_result.json_text = mock_llm_response
    mock_provider = MagicMock()
    mock_provider.synthesize_json.return_value = mock_result
    mock_provider.provider_name = "openai"

    with patch("app.alpha.filing_risk_scan.get_llm_provider", return_value=mock_provider):
        from app.alpha.filing_risk_scan import scan_filing_risks, _RISK_CACHE

        _RISK_CACHE.clear()
        scope = _integrity_scope("TEST", date.today().isoformat())
        r1 = scan_filing_risks("TEST", integrity_scope=scope)
        r2 = scan_filing_risks("TEST", integrity_scope=scope)

    assert r1 == r2
    assert mock_provider.synthesize_json.call_count == 1


def test_v1_cache_keys_database_issuer_accession_asof_and_content_revision(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    first_path = _seed_filing(tmp_path, "V1KEY", form_type="10-K")

    from app.alpha import filing_risk_scan as module

    module._RISK_CACHE.clear()
    with patch.object(
        module,
        "_keyword_fallback",
        wraps=module._keyword_fallback,
    ) as fallback:
        first = module.scan_filing_risks(
            "V1KEY",
            as_of_date="2026-04-01",
            use_llm=False,
        )
        cached = module.scan_filing_risks(
            "V1KEY",
            as_of_date="2026-04-01",
            use_llm=False,
        )
        assert fallback.call_count == 1

        Path(first_path).write_text(
            (
                "<h1>Item 1A. Risk Factors</h1>"
                f"<p>{_long_risk_body('changed V1 revision')}</p>"
                "<h1>Item 1B. Unresolved Staff Comments</h1>"
            ),
            encoding="utf-8",
        )
        revision_miss = module.scan_filing_risks(
            "V1KEY",
            as_of_date="2026-04-01",
            use_llm=False,
        )
        assert fallback.call_count == 2

        asof_miss = module.scan_filing_risks(
            "V1KEY",
            as_of_date="2026-04-02",
            use_llm=False,
        )
        assert fallback.call_count == 3

        second_path = tmp_path / "v1key-newer.htm"
        second_path.write_text(
            (
                "<h1>Item 1A. Risk Factors</h1>"
                f"<p>{_long_risk_body('new V1 accession')}</p>"
                "<h1>Item 1B. Unresolved Staff Comments</h1>"
            ),
            encoding="utf-8",
        )
        now = utc_now_iso()
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO filings(
                    cik, ticker, accession, form_type, filing_date, period_end,
                    primary_doc_url, local_path, status, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "0001003935",
                    "V1KEY",
                    "0001003935-26-000011",
                    "10-K/A",
                    "2026-03-15",
                    "2025-12-31",
                    "https://www.sec.gov/Archives/v1key-newer.htm",
                    str(second_path),
                    "OK",
                    now,
                    now,
                ),
            )
        accession_miss = module.scan_filing_risks(
            "V1KEY",
            as_of_date="2026-04-02",
            use_llm=False,
        )

    assert cached == first
    assert revision_miss["source_content_revision"] != first["source_content_revision"]
    assert asof_miss["analysis_as_of_date"] == "2026-04-02"
    assert accession_miss["source_accession"] == "0001003935-26-000011"
    assert fallback.call_count == 4


def test_scan_cache_separates_keyword_and_llm_modes(monkeypatch, tmp_path):
    """Provider-free cache targeting should not poison later LLM-enabled scans."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "TEST")

    mock_llm_response = json.dumps(
        {
            "competitive_disruption": "LOW",
            "secular_decline": "NOT_MENTIONED",
            "regulatory_legal": "LOW",
            "customer_concentration": "LOW",
            "summary": "LLM-reviewed risk section has limited decision-critical flags.",
        }
    )
    mock_result = MagicMock()
    mock_result.json_text = mock_llm_response
    mock_provider = MagicMock()
    mock_provider.synthesize_json.return_value = mock_result
    mock_provider.provider_name = "openai"

    with patch("app.alpha.filing_risk_scan.get_llm_provider", return_value=mock_provider):
        from app.alpha.filing_risk_scan import scan_filing_risks, _RISK_CACHE

        _RISK_CACHE.clear()
        keyword_result = scan_filing_risks("TEST", use_llm=False)
        llm_result = scan_filing_risks(
            "TEST",
            integrity_scope=_integrity_scope("TEST", date.today().isoformat()),
        )

    assert keyword_result["status"] == "KEYWORD_FALLBACK"
    assert llm_result["status"] == "OK"
    assert llm_result["secular_decline"] == "NOT_MENTIONED"
    assert mock_provider.synthesize_json.call_count == 1


def test_extract_risk_section_10k():
    """Should extract Item 1A risk factors from 10-K HTML."""
    from app.alpha.filing_risk_scan import _extract_risk_text

    html = """<html><body>
    <div>Item 1. Business</div>
    <div>We are a technology company.</div>
    <div>Item 1A. Risk Factors</div>
    <div>We face significant competition from AI-powered alternatives.
    Our products may become obsolete. Generative AI could disrupt our market.</div>
    <div>Item 1B. Unresolved Staff Comments</div>
    </body></html>"""
    text = _extract_risk_text(html, form_type="10-K")
    assert "competition from AI" in text
    assert len(text) > 50


def test_extract_risk_section_ignores_table_of_contents_10k():
    """Should skip TOC Item 1A hits and return the narrative body section."""
    from app.alpha.filing_risk_scan import _extract_risk_text

    risk_body = _long_risk_body("body-section risk")
    html = f"""<html><body>
    <div>Table of Contents</div>
    <div>Item 1. Business 3 Item 1A. Risk Factors 9 Item 1B. Unresolved Staff Comments 29 Item 2. Properties 31</div>
    <h1>Item 1. Business</h1><p>Business overview.</p>
    <h1>Item 1A. Risk Factors</h1><p>{risk_body}</p>
    <h1>Item 1B. Unresolved Staff Comments</h1>
    </body></html>"""

    text = _extract_risk_text(html, form_type="10-K")

    assert "body-section risk" in text
    assert "Table of Contents" not in text
    assert "Item 1B" not in text
    assert len(text) > 200


def test_extract_risk_section_20f():
    """Should extract risk factors from 20-F HTML (Item 3.D pattern)."""
    from app.alpha.filing_risk_scan import _extract_risk_text

    html = """<html><body>
    <div>Item 3. Key Information</div>
    <div>D. Risk Factors</div>
    <div>Artificial intelligence threatens our core business. Generative AI
    could automate customer service workflows. Machine learning competitors
    are emerging rapidly.</div>
    <div>Item 4. Information on the Company</div>
    </body></html>"""
    text = _extract_risk_text(html, form_type="20-F")
    assert "artificial intelligence" in text.lower()


def test_scan_materializes_stale_local_path_from_cached_sec_filing(monkeypatch, tmp_path):
    """A stale DB local_path should not make filing risk look absent when cache has the filing."""
    cfg = _init_temp_db(monkeypatch, tmp_path)
    cik = "0001003935"
    accession = "0001003935-26-000010"
    primary_doc = "nice-20251231.htm"
    stale_path = tmp_path / "missing" / primary_doc
    cached_path = cfg.cache_dir / "filings" / cik / accession / "primary_document.html"
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_text(
        f"<html><body><h1>Item 1A. Risk Factors</h1><p>{_long_risk_body('cached-filing risk')}</p>"
        "<h1>Item 1B. Unresolved Staff Comments</h1></body></html>",
        encoding="utf-8",
    )
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO filings
               (cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cik,
                "NICE",
                accession,
                "10-K",
                "2026-02-26",
                "2025-12-31",
                f"https://www.sec.gov/Archives/edgar/data/1003935/{accession.replace('-', '')}/{primary_doc}",
                str(stale_path),
                "parsed",
                now,
                now,
            ),
        )

    mock_provider = MagicMock()
    mock_provider.provider_name = "disabled"

    with patch("app.alpha.filing_risk_scan.get_llm_provider", return_value=mock_provider):
        from app.alpha.filing_risk_scan import scan_filing_risks, _RISK_CACHE

        _RISK_CACHE.clear()
        result = scan_filing_risks("NICE")

    assert result["status"] == "KEYWORD_FALLBACK"
    assert result["evidence_status"] == "READABLE_RISK_SECTION"
    assert result["source_accession"] == accession
    assert result["source_form_type"] == "10-K"
    assert result["source_filing_date"] == "2026-02-26"
    assert result["risk_text_chars"] > 200
    assert result["warnings"] == []


def test_scan_falls_back_to_older_readable_annual(monkeypatch, tmp_path):
    """Newest unreadable annual should warn but not hide an older readable filing."""
    _init_temp_db(monkeypatch, tmp_path)
    old_path = tmp_path / "older.html"
    old_path.write_text(
        f"<html><body><h1>Item 1A. Risk Factors</h1><p>{_long_risk_body('older-filing risk')}</p>"
        "<h1>Item 1B. Unresolved Staff Comments</h1></body></html>",
        encoding="utf-8",
    )
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO filings
               (cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "0001003935",
                "NICE",
                "newest",
                "10-K",
                "2026-02-26",
                "2025-12-31",
                "",
                "",
                "OK",
                now,
                now,
            ),
        )
        conn.execute(
            """INSERT INTO filings
               (cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "0001003935",
                "NICE",
                "older",
                "10-K",
                "2025-02-26",
                "2024-12-31",
                "https://www.sec.gov/Archives/older.htm",
                str(old_path),
                "OK",
                now,
                now,
            ),
        )

    mock_provider = MagicMock()
    mock_provider.provider_name = "disabled"

    with patch("app.alpha.filing_risk_scan.get_llm_provider", return_value=mock_provider):
        from app.alpha.filing_risk_scan import scan_filing_risks, _RISK_CACHE

        _RISK_CACHE.clear()
        # This test exercises readable-filing fallback, not evidence aging.
        result = scan_filing_risks("NICE", as_of_date="2026-04-01")

    assert result["status"] == "KEYWORD_FALLBACK"
    assert result["evidence_status"] == "READABLE_RISK_SECTION"
    assert result["source_accession"] == "older"
    assert result["analysis_as_of_date"] == "2026-04-01"
    assert result["source_filing_age_days"] == 399
    assert "annual_filing_unreadable:newest" in result["warnings"]


def test_scan_existing_filing_with_missing_risk_section_reports_evidence_status(
    monkeypatch, tmp_path
):
    """Readable filing without Item 1A should be visible as section failure, not polished evidence."""
    _init_temp_db(monkeypatch, tmp_path)
    filing_path = tmp_path / "no-risk.html"
    filing_path.write_text(
        "<html><body><h1>Item 1. Business</h1><p>Business only.</p></body></html>", encoding="utf-8"
    )
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO filings
               (cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "0001003935",
                "NICE",
                "missing-risk",
                "10-K",
                "2026-02-26",
                "2025-12-31",
                "https://www.sec.gov/Archives/no-risk.htm",
                str(filing_path),
                "OK",
                now,
                now,
            ),
        )

    from app.alpha.filing_risk_scan import scan_filing_risks, _RISK_CACHE

    _RISK_CACHE.clear()
    result = scan_filing_risks("NICE")

    assert result["status"] == "NO_FILING"
    assert result["evidence_status"] == "RISK_SECTION_NOT_FOUND"
    assert result["source_accession"] == "missing-risk"
    assert result["risk_text_chars"] == 0


def test_scan_amendment_without_risk_section_falls_back_to_full_40f(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    amendment_path = tmp_path / "foreign-40f-a.html"
    amendment_path.write_text(
        "<html><body><h1>Exhibits</h1><p>Amendment-only exhibits.</p></body></html>",
        encoding="utf-8",
    )
    full_path = tmp_path / "foreign-40f.html"
    full_path.write_text(
        f"<html><body><h1>Risk Factors</h1><p>{_long_risk_body('full-40f risk')}</p>"
        "<h1>Item 4. Information on the Company</h1></body></html>",
        encoding="utf-8",
    )
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO sec_registrants(
                cik, primary_ticker, all_tickers, exchange_scope,
                operating_status, first_seen_at, last_seen_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "0000000042",
                "PRIMARY",
                '["PRIMARY", "ADR"]',
                "US_EXCHANGE",
                "OPERATING",
                now,
                now,
            ),
        )
        for accession, form_type, filing_date, local_path in (
            ("0000000042-26-000002", "40-F/A", "2026-03-15", amendment_path),
            ("0000000042-26-000001", "40-F", "2026-03-01", full_path),
        ):
            conn.execute(
                """
                INSERT INTO filings(
                    cik, ticker, accession, form_type, filing_date, period_end,
                    primary_doc_url, local_path, status, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "42",
                    "PRIMARY",
                    accession,
                    form_type,
                    filing_date,
                    "2025-12-31",
                    "https://www.sec.gov/Archives/foreign.htm",
                    str(local_path),
                    "parsed",
                    now,
                    now,
                ),
            )

    mock_provider = MagicMock()
    mock_provider.provider_name = "disabled"
    with patch("app.alpha.filing_risk_scan.get_llm_provider", return_value=mock_provider):
        from app.alpha.filing_risk_scan import _RISK_CACHE, scan_filing_risks

        _RISK_CACHE.clear()
        result = scan_filing_risks("ADR", as_of_date="2026-04-01", issuer_aware=True)

    assert result["status"] == "KEYWORD_FALLBACK"
    assert result["source_issuer_cik"] == "42"
    assert result["source_accession"] == "0000000042-26-000001"
    assert result["source_form_type"] == "40-F"
    assert result["analysis_as_of_date"] == "2026-04-01"
    assert result["source_content_revision"] is not None
    assert "risk_section_unavailable:0000000042-26-000002" in result["warnings"]
    assert (
        "annual_risk_section_fallback:0000000042-26-000002->0000000042-26-000001"
        in result["warnings"]
    )


def test_risk_cache_keys_issuer_accession_asof_and_content_revision(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    first_path = tmp_path / "first-20f.html"
    first_path.write_text(
        f"<h1>Risk Factors</h1><p>{_long_risk_body('first revision')}</p>"
        "<h1>Item 4. Information on the Company</h1>",
        encoding="utf-8",
    )
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO sec_registrants(
                cik, primary_ticker, all_tickers, exchange_scope,
                operating_status, first_seen_at, last_seen_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "0000000042",
                "PRIMARY",
                '["PRIMARY", "ADR"]',
                "US_EXCHANGE",
                "OPERATING",
                now,
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "42",
                "PRIMARY",
                "0000000042-26-000001",
                "20-F",
                "2026-03-01",
                "2025-12-31",
                "https://www.sec.gov/Archives/first.htm",
                str(first_path),
                "parsed",
                now,
                now,
            ),
        )

    response = MagicMock()
    response.json_text = json.dumps(
        {
            "competitive_disruption": "LOW",
            "secular_decline": "LOW",
            "regulatory_legal": "LOW",
            "customer_concentration": "LOW",
            "summary": "Reviewed annual filing risks.",
        }
    )
    provider = MagicMock()
    provider.provider_name = "openai"
    provider.synthesize_json.return_value = response
    with patch("app.alpha.filing_risk_scan.get_llm_provider", return_value=provider):
        from app.alpha.filing_risk_scan import _RISK_CACHE, scan_filing_risks

        _RISK_CACHE.clear()
        first = scan_filing_risks(
            "ADR",
            as_of_date="2026-04-01",
            issuer_aware=True,
            integrity_scope=_integrity_scope("ADR", "2026-04-01"),
        )
        alias_hit = scan_filing_risks(
            "PRIMARY",
            as_of_date="2026-04-01",
            issuer_aware=True,
            integrity_scope=_integrity_scope("ADR", "2026-04-01"),
        )
        assert provider.synthesize_json.call_count == 1

        asof_miss = scan_filing_risks(
            "ADR",
            as_of_date="2026-04-02",
            issuer_aware=True,
            integrity_scope=_integrity_scope("ADR", "2026-04-02"),
        )
        assert provider.synthesize_json.call_count == 2

        first_path.write_text(
            f"<h1>Risk Factors</h1><p>{_long_risk_body('second revision')}</p>"
            "<h1>Item 4. Information on the Company</h1>",
            encoding="utf-8",
        )
        revision_miss = scan_filing_risks(
            "ADR",
            as_of_date="2026-04-02",
            issuer_aware=True,
            integrity_scope=_integrity_scope("ADR", "2026-04-02"),
        )
        assert provider.synthesize_json.call_count == 3

        second_path = tmp_path / "second-20f.html"
        second_path.write_text(
            f"<h1>Risk Factors</h1><p>{_long_risk_body('new accession')}</p>"
            "<h1>Item 4. Information on the Company</h1>",
            encoding="utf-8",
        )
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO filings(
                    cik, ticker, accession, form_type, filing_date, period_end,
                    primary_doc_url, local_path, status, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "42",
                    "PRIMARY",
                    "0000000042-26-000002",
                    "20-F/A",
                    "2026-03-15",
                    "2025-12-31",
                    "https://www.sec.gov/Archives/second.htm",
                    str(second_path),
                    "parsed",
                    now,
                    now,
                ),
            )
        accession_miss = scan_filing_risks(
            "ADR",
            as_of_date="2026-04-02",
            issuer_aware=True,
            integrity_scope=_integrity_scope("ADR", "2026-04-02"),
        )

    assert first["source_accession"] == "0000000042-26-000001"
    assert alias_hit == first
    assert asof_miss["analysis_as_of_date"] == "2026-04-02"
    assert revision_miss["source_content_revision"] != asof_miss["source_content_revision"]
    assert accession_miss["source_accession"] == "0000000042-26-000002"
    assert provider.synthesize_json.call_count == 4


def test_missing_filing_result_is_not_cached(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    response = MagicMock()
    response.json_text = json.dumps(
        {
            "competitive_disruption": "LOW",
            "secular_decline": "LOW",
            "regulatory_legal": "LOW",
            "customer_concentration": "LOW",
            "summary": "New filing is now available.",
        }
    )
    provider = MagicMock()
    provider.provider_name = "openai"
    provider.synthesize_json.return_value = response
    with patch("app.alpha.filing_risk_scan.get_llm_provider", return_value=provider):
        from app.alpha.filing_risk_scan import _RISK_CACHE, scan_filing_risks

        _RISK_CACHE.clear()
        missing = scan_filing_risks("NEW", as_of_date="2026-04-01", issuer_aware=True)
        filing_path = tmp_path / "new-10k.html"
        filing_path.write_text(
            f"<h1>Item 1A. Risk Factors</h1><p>{_long_risk_body('newly cached filing')}</p>"
            "<h1>Item 1B. Unresolved Staff Comments</h1>",
            encoding="utf-8",
        )
        now = utc_now_iso()
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO filings(
                    cik, ticker, accession, form_type, filing_date, period_end,
                    primary_doc_url, local_path, status, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "44",
                    "NEW",
                    "0000000044-26-000001",
                    "10-K",
                    "2026-03-01",
                    "2025-12-31",
                    "https://www.sec.gov/Archives/new.htm",
                    str(filing_path),
                    "parsed",
                    now,
                    now,
                ),
            )
        recovered = scan_filing_risks(
            "NEW",
            as_of_date="2026-04-01",
            issuer_aware=True,
            integrity_scope=_integrity_scope("NEW", "2026-04-01"),
        )

    assert missing["status"] == "NO_FILING"
    assert recovered["status"] == "OK"
    assert recovered["source_accession"] == "0000000044-26-000001"
    assert provider.synthesize_json.call_count == 1
