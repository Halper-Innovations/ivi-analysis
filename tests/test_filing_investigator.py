"""Tests for app.alpha.filing_investigator."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import patch, MagicMock

import pytest

from app.alpha.schemas import Anomaly
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


def _seed_filing(tmp_path, ticker, text_content, form_type="10-K"):
    filing_dir = tmp_path / "filings"
    filing_dir.mkdir(exist_ok=True)
    path = filing_dir / f"{ticker.lower()}-test.htm"
    path.write_text(f"<html><body>{text_content}</body></html>", encoding="utf-8")
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO filings
               (cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "0001234567",
                ticker,
                "0001234567-26-000001",
                form_type,
                "2026-02-26",
                "2025-12-31",
                "https://sec.gov/test",
                str(path),
                "OK",
                now,
                now,
            ),
        )


def _bound_scope(ticker: str = "TEST"):
    from app.alpha.schemas import TickerSignalPacket
    from app.autonomous.financial_integrity import stable_quote_hash
    from app.autonomous.v1_financial_context import bind_v1_financial_scope

    as_of_date = date.today().isoformat()
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
        market_cap_method="price_times_shares",
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
        cap_stage_price=50.0,
        cap_stage_price_as_of_date=as_of_date,
        cap_stage_price_currency="USD",
        cap_stage_price_source="fixture_quote",
        cap_stage_quote_snapshot_id=snapshot_id,
    )
    canonicalize_financial_packet(packet, as_of_date=as_of_date, shares_mm=10.0)
    return bind_v1_financial_scope(
        context=f"filing_investigator_test:{ticker}",
        run_as_of_date=as_of_date,
        packets=(packet,),
        scenarios=(),
    )


def test_investigate_anomalies_basic(monkeypatch, tmp_path):
    """Should produce investigations matching the anomaly questions."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(
        tmp_path,
        "TEST",
        "We recorded a valuation allowance of $10.8 million against our deferred tax assets. "
        "Impairment charges of $1.5 million were recognized for underperforming game titles.",
    )

    anomalies = [
        Anomaly(
            anomaly_type="Q4_EARNINGS_BOMB",
            severity="HIGH",
            description="Q4 loss was -$28M",
            question="What caused the $28M Q4 loss?",
        ),
    ]

    mock_response = json.dumps(
        {
            "answer": "The Q4 loss was driven by a $10.8M valuation allowance on deferred tax assets and $1.5M impairment of game titles.",
            "evidence_excerpt": "We recorded a valuation allowance of $10.8 million against our deferred tax assets.",
            "follow_up": None,
        }
    )
    mock_result = MagicMock()
    mock_result.json_text = mock_response
    mock_provider = MagicMock()
    mock_provider.synthesize_json.return_value = mock_result
    mock_provider.provider_name = "openai"

    with patch("app.alpha.filing_investigator.get_llm_provider", return_value=mock_provider):
        from app.alpha.filing_investigator import investigate_anomalies

        investigations = investigate_anomalies(
            "TEST",
            anomalies,
            financial_integrity_scope=_bound_scope(),
        )

    assert len(investigations) == 1
    assert "valuation allowance" in investigations[0].answer.lower()
    assert investigations[0].anomaly_type == "Q4_EARNINGS_BOMB"


def test_no_filing_returns_empty(monkeypatch, tmp_path):
    """No cached filing should return empty investigation list."""
    _init_temp_db(monkeypatch, tmp_path)
    anomalies = [Anomaly("TEST", "HIGH", "desc", "question")]
    from app.alpha.filing_investigator import investigate_anomalies

    investigations = investigate_anomalies("ZZZZ", anomalies)
    assert investigations == []


def test_empty_anomalies_returns_empty(monkeypatch, tmp_path):
    """No anomalies should return empty investigation list."""
    _init_temp_db(monkeypatch, tmp_path)
    from app.alpha.filing_investigator import investigate_anomalies

    investigations = investigate_anomalies("TEST", [])
    assert investigations == []


def test_llm_disabled_returns_empty(monkeypatch, tmp_path):
    """When LLM is disabled, should return empty."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "TEST", "Some filing text")
    anomalies = [Anomaly("Q4_BOMB", "HIGH", "desc", "question")]

    mock_provider = MagicMock()
    mock_provider.provider_name = "disabled"

    with patch("app.alpha.filing_investigator.get_llm_provider", return_value=mock_provider):
        from app.alpha.filing_investigator import investigate_anomalies

        investigations = investigate_anomalies("TEST", anomalies)

    assert investigations == []


def test_configured_but_unavailable_llm_returns_empty_without_scope(
    monkeypatch,
    tmp_path,
):
    """A provider with no usable credential remains a zero-call no-LLM lane."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "TEST", "Some filing text")
    anomalies = [Anomaly("Q4_BOMB", "HIGH", "desc", "question")]

    mock_provider = MagicMock()
    mock_provider.provider_name = "openai"
    mock_provider.enabled.return_value = False

    with patch(
        "app.alpha.filing_investigator.get_llm_provider",
        return_value=mock_provider,
    ):
        from app.alpha.filing_investigator import investigate_anomalies

        investigations = investigate_anomalies("TEST", anomalies)

    assert investigations == []
    mock_provider.synthesize_json.assert_not_called()


def test_follow_up_question_investigated(monkeypatch, tmp_path):
    """If LLM returns a follow_up question, it should be investigated in the next iteration."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "TEST", "Filing text with details about convertible notes due 2026.")

    anomalies = [Anomaly("DEBT_SPIKE", "HIGH", "desc", "Why did debt increase?")]

    responses = [
        json.dumps(
            {
                "answer": "Debt increased due to convertible notes issued in Oct and Nov 2025.",
                "evidence_excerpt": "convertible notes due 2026",
                "follow_up": "What are the terms of the convertible notes?",
            }
        ),
        json.dumps(
            {
                "answer": "The notes have repayment dates during 2026 with no assurance of refinancing.",
                "evidence_excerpt": "repayment dates during the year ending December 31, 2026",
                "follow_up": None,
            }
        ),
    ]
    call_count = [0]

    def mock_synthesize(**kwargs):
        result = MagicMock()
        result.json_text = responses[min(call_count[0], len(responses) - 1)]
        call_count[0] += 1
        return result

    mock_provider = MagicMock()
    mock_provider.provider_name = "openai"
    mock_provider.synthesize_json.side_effect = mock_synthesize

    with patch("app.alpha.filing_investigator.get_llm_provider", return_value=mock_provider):
        from app.alpha.filing_investigator import investigate_anomalies

        investigations = investigate_anomalies(
            "TEST",
            anomalies,
            financial_integrity_scope=_bound_scope(),
        )

    assert len(investigations) == 2
    assert "convertible notes" in investigations[0].answer.lower()
    assert investigations[1].question == "What are the terms of the convertible notes?"


def test_missing_scope_suppresses_provider_and_investigation_prose(
    monkeypatch,
    tmp_path,
):
    from app.alpha.filing_investigator import investigate_anomalies
    from app.autonomous.financial_integrity import InvalidFinancialInputError

    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "TEST", "Impairment charges materially reduced earnings.")
    anomalies = [
        Anomaly(
            "Q4_EARNINGS_BOMB",
            "HIGH",
            "material loss",
            "What caused the loss?",
        )
    ]
    provider = MagicMock()
    provider.provider_name = "openai"
    provider.enabled.return_value = True
    provider.synthesize_json.side_effect = AssertionError(
        "missing financial scope must suppress provider"
    )

    with patch(
        "app.alpha.filing_investigator.get_llm_provider",
        return_value=provider,
    ):
        with pytest.raises(InvalidFinancialInputError):
            investigate_anomalies("TEST", anomalies)

    provider.synthesize_json.assert_not_called()


def test_mutated_bound_scope_suppresses_provider_and_investigation_prose(
    monkeypatch,
    tmp_path,
):
    from app.alpha.filing_investigator import investigate_anomalies
    from app.autonomous.financial_integrity import InvalidFinancialInputError

    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "TEST", "Impairment charges materially reduced earnings.")
    anomalies = [
        Anomaly(
            "Q4_EARNINGS_BOMB",
            "HIGH",
            "material loss",
            "What caused the loss?",
        )
    ]
    scope = _bound_scope()
    scope.packets[0]["current_price"] = 51.0
    provider = MagicMock()
    provider.provider_name = "openai"
    provider.enabled.return_value = True

    with patch(
        "app.alpha.filing_investigator.get_llm_provider",
        return_value=provider,
    ):
        with pytest.raises(InvalidFinancialInputError):
            investigate_anomalies(
                "TEST",
                anomalies,
                financial_integrity_scope=scope,
            )

    provider.synthesize_json.assert_not_called()


def test_retry_revalidates_exact_prompt_scenario_before_second_attempt(
    monkeypatch,
    tmp_path,
):
    from app.alpha.filing_investigator import investigate_anomalies
    from app.autonomous.financial_integrity import InvalidFinancialInputError
    from app.llm.providers.retry_guard import call_with_llm_retry_guard

    class _RetryableProviderError(RuntimeError):
        status_code = 503

    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "TEST", "Impairment charges materially reduced earnings.")
    anomalies = [
        Anomaly(
            "Q4_EARNINGS_BOMB",
            "HIGH",
            "material loss",
            "What caused the loss?",
        )
    ]
    physical_calls = 0

    class _RetryingProvider:
        provider_name = "openai"

        def synthesize_json(self, **kwargs):
            def physical_call():
                nonlocal physical_calls
                physical_calls += 1
                if physical_calls == 1:
                    kwargs["schema"]["properties"]["answer"]["type"] = "number"
                    raise _RetryableProviderError("temporary provider failure")
                raise AssertionError("mutated exact scenario must suppress retry")

            return call_with_llm_retry_guard(
                provider_name="openai",
                schema_name="filing_investigator_v1",
                call=physical_call,
                max_retries=1,
                backoff_seconds=(0.0,),
                sleep_fn=lambda _seconds: None,
            )

    with patch(
        "app.alpha.filing_investigator.get_llm_provider",
        return_value=_RetryingProvider(),
    ):
        with pytest.raises(InvalidFinancialInputError):
            investigate_anomalies(
                "TEST",
                anomalies,
                financial_integrity_scope=_bound_scope(),
            )

    assert physical_calls == 1


def test_failed_provider_call_revalidates_exact_prompt_scenario(
    monkeypatch,
    tmp_path,
):
    from app.alpha.filing_investigator import investigate_anomalies
    from app.autonomous.financial_integrity import InvalidFinancialInputError

    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "TEST", "Impairment charges materially reduced earnings.")
    anomalies = [
        Anomaly(
            "Q4_EARNINGS_BOMB",
            "HIGH",
            "material loss",
            "What caused the loss?",
        )
    ]
    physical_calls = 0

    class _FailingMutatingProvider:
        provider_name = "openai"

        @staticmethod
        def enabled():
            return True

        @staticmethod
        def synthesize_json(**kwargs):
            nonlocal physical_calls
            physical_calls += 1
            kwargs["schema"]["properties"]["answer"]["type"] = "number"
            raise RuntimeError("synthetic provider failure after input mutation")

    with patch(
        "app.alpha.filing_investigator.get_llm_provider",
        return_value=_FailingMutatingProvider(),
    ):
        with pytest.raises(InvalidFinancialInputError) as caught:
            investigate_anomalies(
                "TEST",
                anomalies,
                financial_integrity_scope=_bound_scope(),
            )

    assert {violation.code for violation in caught.value.violations} == {
        "BOUND_FINANCIAL_INPUT_MUTATED"
    }
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert physical_calls == 1


def test_failed_provider_call_without_mutation_remains_empty(
    monkeypatch,
    tmp_path,
):
    from app.alpha.filing_investigator import investigate_anomalies

    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing(tmp_path, "TEST", "Impairment charges materially reduced earnings.")
    anomalies = [
        Anomaly(
            "Q4_EARNINGS_BOMB",
            "HIGH",
            "material loss",
            "What caused the loss?",
        )
    ]
    physical_calls = 0

    class _FailingProvider:
        provider_name = "openai"

        @staticmethod
        def enabled():
            return True

        @staticmethod
        def synthesize_json(**_kwargs):
            nonlocal physical_calls
            physical_calls += 1
            raise RuntimeError("synthetic provider failure")

    with patch(
        "app.alpha.filing_investigator.get_llm_provider",
        return_value=_FailingProvider(),
    ):
        investigations = investigate_anomalies(
            "TEST",
            anomalies,
            financial_integrity_scope=_bound_scope(),
        )

    assert investigations == []
    assert physical_calls == 1
