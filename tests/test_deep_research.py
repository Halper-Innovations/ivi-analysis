"""Tests for app.research.deep_research."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from app.research.analyst_notes import AnalystCitation, AnalystNote, AnalystNotes
from app.research.current_event_context import CurrentEventContext, CurrentEventDocument
from app.research.evidence_searcher import (
    _require_paid_financial_scope as _real_require_paid_financial_scope,
)
from app.research.filing_context import FilingContext, FilingDocument
from app.research.findings_reconciliation import PipelineFinding, MatchedFinding, MergedFindings


@pytest.fixture(autouse=True)
def _stub_financial_authorization(monkeypatch):
    """Legacy pipeline tests isolate research behavior from canonical setup."""

    class _Scope:
        def __init__(self, *, packets=(), scenarios=()):
            self.packets = tuple(packets)
            self.scenarios = tuple(scenarios)

        def require(self, **_kwargs):
            return None

    monkeypatch.setattr(
        "app.research.deep_research.bind_v1_financial_scope",
        lambda **kwargs: _Scope(
            packets=kwargs.get("packets") or (),
            scenarios=kwargs.get("scenarios") or (),
        ),
    )

    def _canonical_context(*, tickers, scorecard_evidence, **_kwargs):
        from app.alpha.schemas import TickerSignalPacket

        packets = {}
        for ticker in tickers:
            scorecard = deepcopy(scorecard_evidence[ticker][1])
            detail = scorecard.get("pricing_zone_detail") or {}
            packets[ticker] = TickerSignalPacket(
                ticker=ticker,
                current_price=detail.get("current_price"),
                dcf_value=detail.get("dcf_base"),
                epv_value=detail.get("epv_adjusted"),
                issuer_cik="0000000001",
                issuer_primary_ticker=ticker,
                issuer_listed_tickers=[ticker, f"{ticker}.A"],
                raw_valuation=scorecard,
            )
        return SimpleNamespace(packets=packets)

    monkeypatch.setattr(
        "app.research.deep_research.build_canonical_v1_financial_context",
        _canonical_context,
    )
    monkeypatch.setattr(
        "app.research.analyst_notes.require_financial_integrity_scope",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "app.research.evidence_searcher._require_paid_financial_scope",
        lambda *_args, **_kwargs: None,
    )


# ---------------------------------------------------------------------------
# Minimal stubs for upstream types (avoid importing heavy modules in unit tests)
# ---------------------------------------------------------------------------


def _make_scorecard(
    dcf=100.0,
    epv=60.0,
    price=80.0,
    wacc=0.10,
    tg=0.02,
    graham_disc=0.10,
    gate_action="PROCEED",
) -> dict[str, Any]:
    sc = {
        "pricing_zone_detail": {"current_price": price, "terminal_growth_used": tg},
        "wacc_detail": {"adjusted_wacc": wacc},
        "discounts": {},
        "quality_context": {"gate_action": gate_action},
    }
    if dcf is not None:
        sc["pricing_zone_detail"]["dcf_base"] = dcf
    if epv is not None:
        sc["pricing_zone_detail"]["epv_adjusted"] = epv
    if graham_disc is not None:
        sc["discounts"]["graham"] = graham_disc
    return sc


def _make_tensions():
    return {
        "tension_type": "NONE",
        "methods_agree": True,
        "method_count": 2,
        "consensus_strength": 2,
        "method_values": {"dcf": 100.0, "epv": 60.0},
        "sensitivity": {},
        "intrinsic_range": {"low": 60.0, "mid": 80.0, "high": 100.0},
        "assumption_sensitivity": {},
    }


def _make_filing_context(
    *,
    html: str = "<html>filing</html>",
    form_type: str = "10-K",
    filing_date: str = "2025-11-15",
    accession: str = "0000000000-25-000001",
    role: str = "annual",
) -> FilingContext:
    if not html:
        return FilingContext()
    return FilingContext(
        documents=[
            FilingDocument(
                ticker="TEST",
                cik="0000000001",
                accession=accession,
                form_type=form_type,
                filing_date=filing_date,
                period_end=None,
                role=role,
                local_path=None,
                primary_doc_url=None,
                html=html,
            )
        ]
    )


def _make_current_event_context(
    *, published_at: str = "2026-04-17T10:00:00+00:00"
) -> CurrentEventContext:
    return CurrentEventContext(
        documents=[
            CurrentEventDocument(
                ticker="TEST",
                source_type="ir_press",
                published_at=published_at,
                title="Guidance update",
                source_url="https://example.com/press/guidance",
                summary="Management raised revenue guidance.",
                citations=[],
            )
        ]
    )


def _mutation_rejecting_scope(expected_price: float):
    from app.autonomous.financial_integrity import (
        FinancialIntegrityScope,
        InvalidFinancialInputError,
        require_financial_integrity_scope,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        require_financial_integrity_scope(
            FinancialIntegrityScope(
                context="deep_research_live_rebinding",
                run_as_of_date="2026-03-25",
                packets=(),
            )
        )
    integrity_error = exc_info.value

    class _LiveScope:
        packets = ({"ticker": "TEST", "current_price": 80.0},)
        scenarios: tuple[Any, ...] = ()

        def __init__(self) -> None:
            self.check_count = 0

        def require(self, *, scenarios=None):
            self.check_count += 1
            live_price = scenarios[0]["financial_inputs"]["scorecard"]["pricing_zone_detail"][
                "current_price"
            ]
            if live_price != expected_price:
                raise integrity_error
            return None

    return _LiveScope(), integrity_error


def test_deep_research_stops_after_first_paid_response_mutates_live_price(
    monkeypatch,
):
    from app.llm.usage_capture import (
        attached_provider_usage_records,
        provider_usage_capture,
    )
    from app.research.deep_research import assemble_research_from_filing_context
    from app.research.hypothesis_generator import EvidenceNeed, Hypothesis

    scorecard = _make_scorecard()
    scope, integrity_error = _mutation_rejecting_scope(80.0)
    hypothesis = Hypothesis(
        claim="Customer concentration may impair debt service.",
        direction="BEARISH",
        evidence_needed=[
            EvidenceNeed(
                need_id="need_customer",
                description="customer concentration",
                importance="REQUIRED",
            ),
            EvidenceNeed(
                need_id="need_debt",
                description="debt maturity",
                importance="REQUIRED",
            ),
        ],
        falsification="Customers are diversified and maturities are funded.",
        priority="HIGH",
        source="TEST",
    )

    class _MutatingProvider:
        provider_name = "openai"
        model = "gpt-5-mini"

        def __init__(self) -> None:
            self.calls = 0

        def synthesize_json(self, **_kwargs):
            self.calls += 1
            scorecard["pricing_zone_detail"]["current_price"] = 81.0
            return SimpleNamespace(
                json_text=json.dumps(
                    {
                        "status": "INCONCLUSIVE",
                        "cited_blocks": [],
                        "structured_fact": None,
                        "reasoning_short": "Insufficient evidence.",
                    }
                ),
                model="gpt-5-mini",
                usage_input_tokens=1000,
                usage_output_tokens=300,
                raw={},
            )

    provider = _MutatingProvider()
    monkeypatch.setattr(
        "app.research.hypothesis_generator.generate_hypotheses",
        lambda *args, **kwargs: [hypothesis],
    )
    monkeypatch.setattr(
        "app.research.evidence_searcher._require_paid_financial_scope",
        _real_require_paid_financial_scope,
    )
    monkeypatch.setattr(
        "app.research.evidence_searcher.get_llm_provider",
        lambda: provider,
    )

    filing_context = _make_filing_context(
        html=(
            "<html><body><p>Customer concentration and debt maturity risks "
            "remain material.</p></body></html>"
        )
    )
    with (
        provider_usage_capture("deep_research_evidence") as usage,
        pytest.raises(type(integrity_error)) as exc_info,
    ):
        assemble_research_from_filing_context(
            ticker="TEST",
            as_of_date="2026-03-25",
            scorecard=scorecard,
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={"gate_action": "PROCEED"},
            solvency=None,
            filing_risk=None,
            filing_context=filing_context,
            financial_integrity_scope=scope,
        )

    assert exc_info.value is integrity_error
    assert provider.calls == 1
    assert scope.check_count >= 3
    assert len(usage) == 1
    assert usage[0]["cost_estimate_usd"] == 0.0019
    attached_usage = attached_provider_usage_records(exc_info.value)
    assert len(attached_usage) == 1
    assert attached_usage[0]["cost_estimate_usd"] == 0.0019


def test_direct_deep_research_assembly_rechecks_live_scope_before_return(
    monkeypatch,
):
    from app.research.deep_research import assemble_research_from_filing_context

    scorecard = _make_scorecard(dcf=50.0, epv=40.0, price=200.0)
    scope, integrity_error = _mutation_rejecting_scope(200.0)
    monkeypatch.setattr(
        "app.research.hypothesis_generator.generate_hypotheses",
        lambda *args, **kwargs: [],
    )

    def mutate_live_inputs(*_args, **_kwargs):
        scorecard["pricing_zone_detail"]["current_price"] = 201.0

    monkeypatch.setattr(
        "app.research.deep_research._run_analyst_stage_from_context",
        mutate_live_inputs,
    )

    with pytest.raises(type(integrity_error)) as exc_info:
        assemble_research_from_filing_context(
            ticker="TEST",
            as_of_date="2026-03-25",
            scorecard=scorecard,
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={"gate_action": "PROCEED"},
            solvency=None,
            filing_risk=None,
            filing_context=_make_filing_context(),
            financial_integrity_scope=scope,
        )

    assert exc_info.value is integrity_error
    assert scope.check_count == 2


class TestAssembleResearchEdgeCases:
    """Tests for assemble_research() status resolution and edge cases."""

    def test_no_hypotheses_status(self):
        """Zero hypotheses → NO_HYPOTHESES, investigation_ran=False, thesis=None."""
        from app.research.deep_research import assemble_research

        # Price above all methods so PLAUSIBLE_UNDERVALUATION doesn't fire
        report = assemble_research(
            ticker="TEST",
            as_of_date="2026-03-25",
            scorecard=_make_scorecard(dcf=50.0, epv=40.0, price=200.0),
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={"gate_action": "PROCEED"},
            solvency=None,
            filing_risk=None,
            filing_html="<html>filing</html>",
            form_type="10-K",
        )
        assert report.status == "NO_HYPOTHESES"
        assert report.investigation_ran is False
        assert report.thesis is None
        assert report.hypotheses_generated == 0
        assert report.total_adjustments == 0
        assert report.researchable_items == []
        assert report.not_researchable_items == []

    def test_no_filing_status(self):
        """Hypotheses generated but no filing → NO_FILING."""
        from app.research.deep_research import assemble_research

        # Use a scorecard that will produce at least one hypothesis via BLOCK gate
        report = assemble_research(
            ticker="TEST",
            as_of_date="2026-03-25",
            scorecard=_make_scorecard(gate_action="BLOCK"),
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={
                "gate_action": "BLOCK",
                "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
                "negative_oe_years": 3,
            },
            solvency=None,
            filing_risk=None,
            filing_html=None,
            form_type=None,
        )
        assert report.status == "NO_FILING"
        assert report.investigation_ran is False
        assert report.thesis is None
        assert report.hypotheses_generated > 0

    def test_no_hypotheses_checked_before_no_filing(self):
        """NO_HYPOTHESES takes precedence over NO_FILING when both apply."""
        from app.research.deep_research import assemble_research

        report = assemble_research(
            ticker="TEST",
            as_of_date="2026-03-25",
            scorecard=_make_scorecard(dcf=50.0, epv=40.0, price=200.0),
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={"gate_action": "PROCEED"},
            solvency=None,
            filing_risk=None,
            filing_html=None,  # no filing
            form_type=None,
        )
        # Zero hypotheses AND no filing — but NO_HYPOTHESES should win
        assert report.status == "NO_HYPOTHESES"

    def test_blocked_gate_still_investigates(self):
        """BLOCK gate_action does NOT prevent research."""
        from app.research.deep_research import assemble_research

        report = assemble_research(
            ticker="TEST",
            as_of_date="2026-03-25",
            scorecard=_make_scorecard(gate_action="BLOCK"),
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={
                "gate_action": "BLOCK",
                "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
                "negative_oe_years": 3,
            },
            solvency=None,
            filing_risk=None,
            filing_html="<html><body><p>The company has negative owner earnings for 3 years.</p></body></html>",
            form_type="10-K",
        )
        assert report.gate_action == "BLOCK"
        assert report.hypotheses_generated > 0
        # Investigation should run if hypotheses exist and filing is present
        assert report.investigation_ran is True
        assert report.thesis is not None

    def test_ok_status_with_full_pipeline(self):
        """Full pipeline with hypotheses and filing → OK."""
        from app.research.deep_research import assemble_research

        report = assemble_research(
            ticker="TEST",
            as_of_date="2026-03-25",
            scorecard=_make_scorecard(gate_action="BLOCK"),
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={
                "gate_action": "BLOCK",
                "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
                "negative_oe_years": 3,
            },
            solvency=None,
            filing_risk=None,
            filing_html="<html><body><p>Capital expenditure breakdown shows growth capex at 65%.</p></body></html>",
            form_type="10-K",
        )
        assert report.status == "OK"
        assert report.investigation_ran is True
        assert report.thesis is not None
        assert report.hypotheses_generated > 0

    def test_gate_action_passed_through(self):
        """gate_action from quality_ctx appears on report."""
        from app.research.deep_research import assemble_research

        report = assemble_research(
            ticker="TEST",
            as_of_date="2026-03-25",
            scorecard=_make_scorecard(gate_action="ADJUST"),
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={"gate_action": "ADJUST"},
            solvency=None,
            filing_risk=None,
            filing_html=None,
            form_type=None,
        )
        assert report.gate_action == "ADJUST"

    def test_summary_counts_zero_when_no_investigation(self):
        """When investigation_ran is False, all summary counts are 0."""
        from app.research.deep_research import assemble_research

        report = assemble_research(
            ticker="TEST",
            as_of_date="2026-03-25",
            scorecard=_make_scorecard(),
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={"gate_action": "PROCEED"},
            solvency=None,
            filing_risk=None,
            filing_html=None,
            form_type=None,
        )
        assert report.total_adjustments == 0
        assert report.fact_calibrated_count == 0
        assert report.heuristic_count == 0

    def test_unresolved_classification(self):
        """Unresolved items are split into researchable vs not_researchable."""
        from app.research.deep_research import assemble_research

        report = assemble_research(
            ticker="TEST",
            as_of_date="2026-03-25",
            scorecard=_make_scorecard(gate_action="BLOCK"),
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={
                "gate_action": "BLOCK",
                "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
                "negative_oe_years": 3,
            },
            solvency=None,
            filing_risk=None,
            filing_html="<html><body><p>Some filing text.</p></body></html>",
            form_type="10-K",
        )
        if report.investigation_ran and report.thesis:
            # All unresolved items should be in one of the two lists
            total_unresolved = len(report.researchable_items) + len(report.not_researchable_items)
            assert total_unresolved == len(report.thesis.unresolved)
            # Check classification correctness
            for item in report.researchable_items:
                assert item.unresolved_reason in ("INCONCLUSIVE", "UNCLASSIFIED")
            for item in report.not_researchable_items:
                assert item.unresolved_reason == "NOT_FOUND"


from unittest.mock import patch, MagicMock
import json


def test_financial_scan_wrappers_preserve_legacy_and_forward_strict_pit():
    import app.research.deep_research as deep_research
    from app.alpha.schemas import SolvencyAssessment

    with (
        patch("app.alpha.anomaly_detector.detect_anomalies", return_value=[]) as anomaly_scan,
        patch(
            "app.alpha.solvency_scanner.assess_solvency",
            return_value=SolvencyAssessment(solvency_risk="UNKNOWN"),
        ) as solvency_scan,
    ):
        deep_research.detect_anomalies("LEGACY")
        deep_research.assess_solvency("LEGACY")
        anomaly_scan.assert_called_once_with("LEGACY")
        solvency_scan.assert_called_once_with("LEGACY")

        anomaly_scan.reset_mock()
        solvency_scan.reset_mock()
        strict_kwargs = {
            "as_of_date": "2024-06-30",
            "require_filed_asof": True,
            "issuer_cik": "0000000001",
            "aliases": ("HIST", "HIST.A"),
        }
        deep_research.detect_anomalies("HIST", **strict_kwargs)
        deep_research.assess_solvency("HIST", **strict_kwargs)

        anomaly_scan.assert_called_once_with("HIST", **strict_kwargs)
        solvency_scan.assert_called_once_with("HIST", **strict_kwargs)


def test_historical_anomaly_wrapper_excludes_future_filed_fact(isolated_data_root):
    from app.db import get_db, init_db, utc_now_iso
    from app.research.deep_research import detect_anomalies

    init_db()
    now = utc_now_iso()
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"
    with get_db() as conn:
        conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value,
                units, source_url, fetched_at, filed_date, form, accession
            ) VALUES (?, ?, 'FY', ?, 'equity', ?, 'USD_millions', ?, ?, ?, '10-K', ?)
            """,
            [
                (
                    "HIST",
                    2022,
                    "2022-12-31",
                    100.0,
                    source_url,
                    now,
                    "2023-02-15",
                    "0000000001-23-000001",
                ),
                (
                    "HIST",
                    2023,
                    "2023-12-31",
                    -500.0,
                    source_url,
                    now,
                    "2024-08-15",
                    "0000000001-24-000002",
                ),
            ],
        )

    assert [item.anomaly_type for item in detect_anomalies("HIST")] == ["NEGATIVE_EQUITY"]
    historical = detect_anomalies(
        "HIST",
        as_of_date="2024-06-30",
        require_filed_asof=True,
        issuer_cik="0000000001",
        aliases=("HIST",),
    )
    assert [item.anomaly_type for item in historical] == []


def test_strict_solvency_wrapper_disables_network_materialization():
    import app.research.deep_research as deep_research

    with (
        patch("app.alpha.solvency_scanner._load_latest_annual", return_value={}),
        patch("app.alpha.solvency_scanner._load_market_cap", return_value=None),
        patch(
            "app.alpha.solvency_scanner._load_filing_evidence",
            return_value=None,
        ) as filing_evidence,
    ):
        result = deep_research.assess_solvency(
            "HIST",
            as_of_date="2024-06-30",
            require_filed_asof=True,
            issuer_cik="0000000001",
            aliases=("HIST", "HIST.A"),
        )

    assert result.solvency_risk == "UNKNOWN"
    assert filing_evidence.call_args.kwargs["allow_network_materialization"] is False
    assert filing_evidence.call_args.kwargs["as_of_date"] == "2024-06-30"
    assert filing_evidence.call_args.kwargs["issuer_cik"] == "0000000001"
    assert filing_evidence.call_args.kwargs["aliases"] == ("HIST", "HIST.A")


class TestRunDeepResearch:
    """Tests for run_deep_research() orchestration entry point."""

    def _mock_scorecard_row(self, as_of_date="2026-03-25", gate_action="BLOCK"):
        sc = _make_scorecard(gate_action=gate_action)
        sc["quality_context"] = {
            "gate_action": gate_action,
            "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
            "negative_oe_years": 3,
        }
        return {"outputs_json": json.dumps(sc), "as_of_date": as_of_date}

    @patch("app.research.deep_research._persist_to_db")
    @patch("app.research.deep_research._load_scorecard", return_value=(None, None))
    @patch("app.research.deep_research.detect_anomalies", return_value=[])
    @patch("app.research.deep_research.assess_solvency", return_value=None)
    @patch("app.research.deep_research.scan_filing_risks", return_value={"status": "NO_FILING"})
    def test_no_scorecard_returns_no_scorecard(
        self, mock_fr, mock_solv, mock_anom, mock_load_sc, mock_persist
    ):
        from app.research.deep_research import run_deep_research

        report = run_deep_research("TEST", as_of_date="2026-03-25")
        assert report.status == "NO_SCORECARD"
        assert report.scorecard_present is False

    @patch("app.research.deep_research._write_markdown_artifact", return_value=None)
    @patch("app.research.deep_research._write_artifact")
    @patch("app.research.deep_research._persist_to_db")
    @patch("app.research.deep_research._load_filing_context")
    @patch("app.research.deep_research.scan_filing_risks", return_value={"status": "NO_FILING"})
    @patch("app.research.deep_research.assess_solvency", return_value=None)
    @patch("app.research.deep_research.detect_anomalies", return_value=[])
    @patch("app.research.deep_research._load_scorecard")
    def test_scorecard_present_calls_assemble(
        self,
        mock_load_sc,
        mock_anom,
        mock_solv,
        mock_fr,
        mock_filing,
        mock_persist,
        mock_artifact,
        mock_md,
    ):
        from app.research.deep_research import run_deep_research

        sc = _make_scorecard(gate_action="BLOCK")
        sc["quality_context"] = {
            "gate_action": "BLOCK",
            "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
            "negative_oe_years": 3,
        }
        mock_load_sc.return_value = (sc, "2026-03-25")
        mock_filing.return_value = FilingContext()  # no filing

        report = run_deep_research("TEST", as_of_date="2026-03-25")
        assert report.scorecard_present is True
        assert report.hypotheses_generated > 0
        filing_risk_args, filing_risk_kwargs = mock_fr.call_args
        assert filing_risk_args == ("TEST",)
        assert filing_risk_kwargs["as_of_date"] == "2026-03-25"
        assert filing_risk_kwargs["allow_network_materialization"] is False
        assert filing_risk_kwargs["integrity_scope"].packets[0].ticker == "TEST"
        assert filing_risk_kwargs["cfg"].db_path is not None
        assert filing_risk_kwargs["allowed_filing_roots"] == (
            filing_risk_kwargs["cfg"].raw_filings_dir,
            filing_risk_kwargs["cfg"].cache_dir,
        )
        filing_args, filing_kwargs = mock_filing.call_args
        assert filing_args == ("TEST", "2026-03-25")
        assert filing_kwargs["allow_network_materialization"] is False
        assert filing_kwargs["cfg"] is filing_risk_kwargs["cfg"]
        assert filing_kwargs["allowed_filing_roots"] == filing_risk_kwargs["allowed_filing_roots"]

    def test_strict_research_inputs_are_bound_before_provider_scope(
        self,
        monkeypatch,
    ):
        import app.research.deep_research as deep_research
        from app.alpha.schemas import Anomaly, SolvencyAssessment

        scorecard = _make_scorecard(gate_action="BLOCK")
        scorecard["quality_context"] = {
            "gate_action": "BLOCK",
            "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
            "negative_oe_years": 3,
        }
        anomaly = Anomaly(
            anomaly_type="NEGATIVE_EQUITY",
            severity="HIGH",
            description="Equity is negative.",
            question="What caused negative equity?",
            data={"equity": -25.0},
        )
        solvency = SolvencyAssessment(
            solvency_risk="ELEVATED",
            negative_equity=True,
            signals=["NEGATIVE_EQUITY"],
            details="Negative equity requires review.",
        )
        events: list[str] = []
        anomaly_scan = MagicMock(
            side_effect=lambda *_args, **_kwargs: (
                events.append("anomalies"),
                [anomaly],
            )[1]
        )
        solvency_scan = MagicMock(
            side_effect=lambda *_args, **_kwargs: (
                events.append("solvency"),
                solvency,
            )[1]
        )
        captured: dict[str, Any] = {}

        class _Scope:
            def __init__(self, *, packets, scenarios):
                self.packets = tuple(packets)
                self.scenarios = tuple(scenarios)

            def require(self, *, scenarios=None):
                captured["required_scenarios"] = tuple(scenarios or self.scenarios)
                return None

        def _bind_scope(**kwargs):
            events.append("bind")
            captured["bind"] = kwargs
            return _Scope(
                packets=kwargs["packets"],
                scenarios=kwargs["scenarios"],
            )

        monkeypatch.setattr(deep_research, "detect_anomalies", anomaly_scan)
        monkeypatch.setattr(deep_research, "assess_solvency", solvency_scan)
        monkeypatch.setattr(deep_research, "bind_v1_financial_scope", _bind_scope)
        monkeypatch.setattr(
            deep_research,
            "_load_scorecard",
            lambda *_args, **_kwargs: (scorecard, "2024-06-30"),
        )
        monkeypatch.setattr(
            deep_research,
            "_load_reverse_dcf",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            deep_research,
            "scan_filing_risks",
            lambda *_args, **_kwargs: {"status": "NO_FILING"},
        )
        monkeypatch.setattr(
            deep_research,
            "_load_filing_context",
            lambda *_args, **_kwargs: FilingContext(),
        )
        monkeypatch.setattr(
            deep_research,
            "_load_current_event_context",
            lambda *_args, **_kwargs: CurrentEventContext(),
        )
        monkeypatch.setattr(
            deep_research,
            "_write_markdown_artifact",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            deep_research,
            "_write_artifact",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            deep_research,
            "_persist_to_db",
            lambda *_args, **_kwargs: None,
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
            lambda *_args, **_kwargs: None,
        )

        deep_research.run_deep_research("TEST", as_of_date="2024-06-30")

        expected_scan_kwargs = {
            "as_of_date": "2024-06-30",
            "require_filed_asof": True,
            "issuer_cik": "0000000001",
            "aliases": ("TEST", "TEST.A"),
        }
        anomaly_scan.assert_called_once_with("TEST", **expected_scan_kwargs)
        solvency_scan.assert_called_once_with("TEST", **expected_scan_kwargs)
        assert events[:3] == ["anomalies", "solvency", "bind"]

        bound_inputs = captured["bind"]["scenarios"][0]["financial_inputs"]
        assert bound_inputs["scorecard"]["quality_context"] == scorecard["quality_context"]
        assert bound_inputs["scorecard"]["pricing_zone_detail"]["current_price"] == 80.0
        assert bound_inputs["anomalies"] == [asdict(anomaly)]
        assert bound_inputs["solvency"] == asdict(solvency)
        assert captured["required_scenarios"][0]["financial_inputs"] == bound_inputs

    def test_financial_scope_is_revalidated_before_any_publication(
        self,
        monkeypatch,
    ):
        import app.research.deep_research as deep_research
        from app.autonomous.financial_integrity import (
            FinancialIntegrityScope,
            InvalidFinancialInputError,
            require_financial_integrity_scope,
        )
        from app.alpha.schemas import Anomaly

        with pytest.raises(InvalidFinancialInputError) as exc_info:
            require_financial_integrity_scope(
                FinancialIntegrityScope(
                    context="deep_research_publication_test",
                    run_as_of_date="2026-03-25",
                    packets=(),
                )
            )
        integrity_error = exc_info.value
        anomaly = Anomaly(
            anomaly_type="NEGATIVE_EQUITY",
            severity="HIGH",
            description="Equity is negative.",
            question="What caused negative equity?",
            data={"equity": -25.0},
        )

        class _PublicationScope:
            packets: tuple[Any, ...] = ()
            scenarios: tuple[Any, ...] = ()

            def require(self, *, scenarios=None):
                if scenarios is not None and tuple(scenarios) != self.scenarios:
                    raise integrity_error
                return None

        publication_scope = _PublicationScope()
        original_assemble = deep_research.assemble_research_from_filing_context

        def _assemble_then_drift(*args, **kwargs):
            report = original_assemble(*args, **kwargs)
            anomaly.data["equity"] = -999.0
            return report

        sc = _make_scorecard(gate_action="BLOCK")
        sc["quality_context"] = {
            "gate_action": "BLOCK",
            "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
            "negative_oe_years": 3,
        }
        writes: list[str] = []

        def _bind_scope(**kwargs):
            publication_scope.packets = tuple(kwargs.get("packets") or ())
            publication_scope.scenarios = tuple(kwargs.get("scenarios") or ())
            return publication_scope

        monkeypatch.setattr(
            deep_research,
            "bind_v1_financial_scope",
            _bind_scope,
        )
        monkeypatch.setattr(
            deep_research,
            "assemble_research_from_filing_context",
            _assemble_then_drift,
        )
        monkeypatch.setattr(
            deep_research,
            "_load_scorecard",
            lambda *_args, **_kwargs: (sc, "2026-03-25"),
        )
        monkeypatch.setattr(
            deep_research,
            "_load_reverse_dcf",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            deep_research,
            "detect_anomalies",
            lambda *_args, **_kwargs: [anomaly],
        )
        monkeypatch.setattr(
            deep_research,
            "assess_solvency",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            deep_research,
            "scan_filing_risks",
            lambda *_args, **_kwargs: {"status": "NO_FILING"},
        )
        monkeypatch.setattr(
            deep_research,
            "_load_filing_context",
            lambda *_args, **_kwargs: FilingContext(),
        )
        monkeypatch.setattr(
            deep_research,
            "_load_current_event_context",
            lambda *_args, **_kwargs: CurrentEventContext(),
        )
        monkeypatch.setattr(
            deep_research,
            "_write_markdown_artifact",
            lambda *_args, **_kwargs: writes.append("markdown"),
        )
        monkeypatch.setattr(
            deep_research,
            "_write_artifact",
            lambda *_args, **_kwargs: writes.append("json"),
        )
        monkeypatch.setattr(
            deep_research,
            "_persist_to_db",
            lambda *_args, **_kwargs: writes.append("db"),
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
            lambda *_args, **_kwargs: None,
        )

        with pytest.raises(InvalidFinancialInputError) as raised:
            deep_research.run_deep_research(
                "TEST",
                as_of_date="2026-03-25",
            )

        assert raised.value is integrity_error
        assert writes == []

    @patch("app.research.deep_research._write_markdown_artifact", return_value=None)
    @patch("app.research.deep_research._write_artifact")
    @patch("app.research.deep_research._persist_to_db")
    @patch("app.research.deep_research._load_filing_context")
    @patch("app.research.deep_research.scan_filing_risks", return_value={"status": "NO_FILING"})
    @patch("app.research.deep_research.assess_solvency", return_value=None)
    @patch("app.research.deep_research.detect_anomalies", return_value=[])
    @patch("app.research.deep_research._load_scorecard")
    def test_persist_called_on_success(
        self,
        mock_load_sc,
        mock_anom,
        mock_solv,
        mock_fr,
        mock_filing,
        mock_persist,
        mock_artifact,
        mock_md,
    ):
        from app.research.deep_research import run_deep_research

        sc = _make_scorecard(gate_action="BLOCK")
        sc["quality_context"] = {
            "gate_action": "BLOCK",
            "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
            "negative_oe_years": 3,
        }
        mock_load_sc.return_value = (sc, "2026-03-25")
        mock_filing.return_value = _make_filing_context(html="<html>filing</html>")

        report = run_deep_research("TEST", as_of_date="2026-03-25")
        assert mock_persist.called
        assert mock_artifact.called

    @patch("app.research.deep_research._write_markdown_artifact", return_value=None)
    @patch("app.research.deep_research._write_artifact")
    @patch("app.research.deep_research._persist_to_db")
    @patch("app.research.deep_research._load_filing_context")
    @patch("app.research.deep_research.scan_filing_risks", return_value={"status": "NO_FILING"})
    @patch("app.research.deep_research.assess_solvency", return_value=None)
    @patch("app.research.deep_research.detect_anomalies", return_value=[])
    @patch("app.research.deep_research._load_scorecard")
    def test_filing_date_set_from_query(
        self,
        mock_load_sc,
        mock_anom,
        mock_solv,
        mock_fr,
        mock_filing,
        mock_persist,
        mock_artifact,
        mock_md,
    ):
        from app.research.deep_research import run_deep_research

        sc = _make_scorecard(gate_action="BLOCK")
        sc["quality_context"] = {
            "gate_action": "BLOCK",
            "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
            "negative_oe_years": 3,
        }
        mock_load_sc.return_value = (sc, "2026-03-25")
        mock_filing.return_value = _make_filing_context(html="<html>filing</html>")

        report = run_deep_research("TEST", as_of_date="2026-03-25")
        assert report.filing_date == "2025-11-15"

    @patch("app.research.deep_research._write_markdown_artifact", return_value=None)
    @patch("app.research.deep_research._write_artifact")
    @patch("app.research.deep_research._persist_to_db")
    @patch("app.research.deep_research._load_filing_context")
    @patch("app.research.deep_research._ensure_recent_filing_context_cache", return_value=[])
    @patch("app.research.deep_research.scan_filing_risks", return_value={"status": "NO_FILING"})
    @patch("app.research.deep_research.assess_solvency", return_value=None)
    @patch("app.research.deep_research.detect_anomalies", return_value=[])
    @patch("app.research.deep_research._load_scorecard")
    def test_quarterly_context_sets_latest_filing_metadata(
        self,
        mock_load_sc,
        mock_anom,
        mock_solv,
        mock_fr,
        mock_recent_refresh,
        mock_filing,
        mock_persist,
        mock_artifact,
        mock_md,
    ):
        from app.research.deep_research import run_deep_research

        sc = _make_scorecard(gate_action="BLOCK")
        sc["quality_context"] = {
            "gate_action": "BLOCK",
            "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
            "negative_oe_years": 3,
        }
        mock_load_sc.return_value = (sc, "2026-03-25")
        mock_filing.return_value = FilingContext(
            documents=[
                FilingDocument(
                    ticker="TEST",
                    cik="0000000001",
                    accession="annual-01",
                    form_type="10-K",
                    filing_date="2025-02-20",
                    period_end="2024-12-31",
                    role="annual",
                    local_path=None,
                    primary_doc_url=None,
                    html="<html><body><p>Item 7. Management's Discussion and Analysis</p><p>"
                    + ("Annual filing discussion content. " * 20)
                    + "</p></body></html>",
                ),
                FilingDocument(
                    ticker="TEST",
                    cik="0000000001",
                    accession="quarter-01",
                    form_type="10-Q",
                    filing_date="2025-05-01",
                    period_end="2025-03-31",
                    role="quarterly",
                    local_path=None,
                    primary_doc_url=None,
                    html="<html><body><p>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</p><p>"
                    + ("Quarterly filing discussion content. " * 20)
                    + "</p></body></html>",
                ),
            ]
        )

        report = run_deep_research("TEST", as_of_date="2026-03-25", quarters=1)

        assert report.form_type == "10-Q"
        assert report.filing_date == "2025-05-01"
        assert report.analysis_quarters == 1
        mock_recent_refresh.assert_called_once_with("TEST", "2026-03-25", 1)

    @patch("app.research.deep_research._write_markdown_artifact", return_value=None)
    @patch("app.research.deep_research._write_artifact", return_value=None)  # simulate failure
    @patch("app.research.deep_research._persist_to_db")
    @patch("app.research.deep_research._load_filing_context")
    @patch("app.research.deep_research.scan_filing_risks", return_value={"status": "NO_FILING"})
    @patch("app.research.deep_research.assess_solvency", return_value=None)
    @patch("app.research.deep_research.detect_anomalies", return_value=[])
    @patch("app.research.deep_research._load_scorecard")
    def test_artifact_failure_warning_persisted_to_db(
        self,
        mock_load_sc,
        mock_anom,
        mock_solv,
        mock_fr,
        mock_filing,
        mock_persist,
        mock_artifact,
        mock_md,
    ):
        """When artifact write fails, 'artifact_write_failed' must appear in the warnings passed to _persist_to_db."""
        from app.research.deep_research import run_deep_research

        sc = _make_scorecard(gate_action="BLOCK")
        sc["quality_context"] = {
            "gate_action": "BLOCK",
            "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
            "negative_oe_years": 3,
        }
        mock_load_sc.return_value = (sc, "2026-03-25")
        mock_filing.return_value = FilingContext()

        report = run_deep_research("TEST", as_of_date="2026-03-25")

        # _persist_to_db should have been called with warnings containing artifact failure
        assert mock_persist.called
        persist_args = mock_persist.call_args
        warnings_arg = persist_args[0][3]  # 4th positional arg is warnings list
        assert "artifact_write_failed" in warnings_arg

    @patch("app.research.deep_research._persist_to_db")
    @patch("app.research.deep_research._load_scorecard", return_value=(None, None))
    @patch("app.research.deep_research.detect_anomalies", return_value=[])
    @patch("app.research.deep_research.assess_solvency", return_value=None)
    @patch("app.research.deep_research.scan_filing_risks", return_value={"status": "NO_FILING"})
    def test_no_scorecard_conviction_populated(
        self, mock_fr, mock_solv, mock_anom, mock_load_sc, mock_persist
    ):
        """NO_SCORECARD path: conviction is 0/INSUFFICIENT before persistence."""
        from app.research.deep_research import run_deep_research

        report = run_deep_research("TEST", as_of_date="2026-03-25")
        assert report.conviction_score == 0
        assert report.conviction_class == "INSUFFICIENT"

    @patch("app.research.deep_research._write_markdown_artifact", return_value=None)
    @patch("app.research.deep_research._write_artifact")
    @patch("app.research.deep_research._persist_to_db")
    @patch("app.research.deep_research._load_filing_context")
    @patch("app.research.deep_research.scan_filing_risks", return_value={"status": "NO_FILING"})
    @patch("app.research.deep_research.assess_solvency", return_value=None)
    @patch("app.research.deep_research.detect_anomalies", return_value=[])
    @patch("app.research.deep_research._load_scorecard")
    def test_normal_path_conviction_populated(
        self,
        mock_load_sc,
        mock_anom,
        mock_solv,
        mock_fr,
        mock_filing,
        mock_persist,
        mock_artifact,
        mock_md,
    ):
        """Normal path: conviction is populated (non-None) before persistence."""
        from app.research.deep_research import run_deep_research

        sc = _make_scorecard(gate_action="BLOCK")
        sc["quality_context"] = {
            "gate_action": "BLOCK",
            "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
            "negative_oe_years": 3,
        }
        mock_load_sc.return_value = (sc, "2026-03-25")
        mock_filing.return_value = _make_filing_context(html="<html><body>filing</body></html>")

        report = run_deep_research("TEST", as_of_date="2026-03-25")
        assert report.conviction_score is not None
        assert report.conviction_class is not None
        assert report.conviction_class in ("HIGH", "MODERATE", "LOW", "INSUFFICIENT")


class TestResearchReportTensionSummary:
    """Tests for tension summary and conviction fields on ResearchReport."""

    def test_no_hypotheses_has_tension_summary(self):
        """NO_HYPOTHESES reports retain populated tension summary fields."""
        from app.research.deep_research import assemble_research

        report = assemble_research(
            ticker="TEST",
            as_of_date="2026-03-25",
            scorecard=_make_scorecard(dcf=50.0, epv=40.0, price=200.0),
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={"gate_action": "PROCEED"},
            solvency=None,
            filing_risk=None,
            filing_html="<html>filing</html>",
            form_type="10-K",
        )
        assert report.status == "NO_HYPOTHESES"
        assert report.methods_agree is True
        assert report.consensus_strength == 2
        assert report.method_count == 2
        assert report.tension_type == "NONE"

    def test_no_filing_has_tension_summary(self):
        """NO_FILING reports retain populated tension summary fields."""
        from app.research.deep_research import assemble_research

        report = assemble_research(
            ticker="TEST",
            as_of_date="2026-03-25",
            scorecard=_make_scorecard(gate_action="BLOCK"),
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={
                "gate_action": "BLOCK",
                "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
                "negative_oe_years": 3,
            },
            solvency=None,
            filing_risk=None,
            filing_html=None,
            form_type=None,
        )
        assert report.status == "NO_FILING"
        assert report.methods_agree is True
        assert report.method_count == 2
        assert report.tension_type == "NONE"

    def test_tension_summary_populated_regardless_of_status(self):
        """Tension summary fields populated on all assemble_research paths."""
        from app.research.deep_research import assemble_research

        report = assemble_research(
            ticker="TEST",
            as_of_date="2026-03-25",
            scorecard=_make_scorecard(gate_action="BLOCK"),
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={
                "gate_action": "BLOCK",
                "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
                "negative_oe_years": 3,
            },
            solvency=None,
            filing_risk=None,
            filing_html="<html><body><p>filing text</p></body></html>",
            form_type="10-K",
        )
        # Tension summary is always populated via common dict, regardless of status
        assert report.methods_agree is True
        assert report.method_count == 2
        assert report.tension_type == "NONE"

    def test_conviction_defaults_to_none(self):
        """Conviction fields default to None before compute_conviction runs."""
        from app.research.deep_research import assemble_research

        report = assemble_research(
            ticker="TEST",
            as_of_date="2026-03-25",
            scorecard=_make_scorecard(dcf=50.0, epv=40.0, price=200.0),
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={"gate_action": "PROCEED"},
            solvency=None,
            filing_risk=None,
            filing_html="<html>filing</html>",
            form_type="10-K",
        )
        assert report.conviction_score is None
        assert report.conviction_class is None

    @patch("app.research.deep_research._persist_to_db")
    @patch("app.research.deep_research._load_scorecard", return_value=(None, None))
    @patch("app.research.deep_research.detect_anomalies", return_value=[])
    @patch("app.research.deep_research.assess_solvency", return_value=None)
    @patch("app.research.deep_research.scan_filing_risks", return_value={"status": "NO_FILING"})
    def test_no_scorecard_tension_summary_none(
        self, mock_fr, mock_solv, mock_anom, mock_load_sc, mock_persist
    ):
        """NO_SCORECARD reports have None tension summary fields."""
        from app.research.deep_research import run_deep_research

        report = run_deep_research("TEST", as_of_date="2026-03-25")
        assert report.status == "NO_SCORECARD"
        assert report.methods_agree is None
        assert report.consensus_strength is None
        assert report.method_count is None
        assert report.tension_type is None


class TestReportCitationSchema:
    """Tests for ReportCitation and new ResearchReport fields."""

    def test_report_citation_dataclass(self):
        from app.research.deep_research import ReportCitation

        c = ReportCitation(
            citation_id="C1",
            need_id="need_001",
            section="Item 1A Risk Factors",
            excerpt="Top 10 customers account for 35%",
            relevance="customer concentration data",
            hypothesis_source="GROWTH_VS_EARNINGS_POWER",
        )
        assert c.citation_id == "C1"
        assert c.section == "Item 1A Risk Factors"

    def test_research_report_citations_default_empty(self):
        from app.research.deep_research import assemble_research

        report = assemble_research(
            ticker="TEST",
            as_of_date="2026-03-25",
            scorecard=_make_scorecard(),
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={"gate_action": "PROCEED"},
            solvency=None,
            filing_risk=None,
            filing_html="<html>filing</html>",
            form_type="10-K",
        )
        assert hasattr(report, "citations")
        assert isinstance(report.citations, list)
        assert report.report_path is None

    def test_from_dict_roundtrip_with_citations(self):
        from app.research.deep_research import ResearchReport, ReportCitation
        from dataclasses import asdict

        report = ResearchReport(
            ticker="TEST",
            as_of_date="2026-03-25",
            status="OK",
            started_at="t0",
            completed_at="t1",
            scorecard_present=True,
            filing_present=True,
            filing_date="2025-11-15",
            form_type="10-K",
            anomaly_count=0,
            solvency_status=None,
            filing_risk_status=None,
            gate_action="PROCEED",
            investigation_ran=False,
            hypotheses_generated=0,
            thesis=None,
            researchable_items=[],
            not_researchable_items=[],
            total_adjustments=0,
            fact_calibrated_count=0,
            heuristic_count=0,
            methods_agree=True,
            consensus_strength=2,
            method_count=2,
            tension_type="NONE",
            citations=[
                ReportCitation(
                    citation_id="C1",
                    need_id="n1",
                    section="Item 1A",
                    excerpt="text",
                    relevance="rel",
                    hypothesis_source="SRC",
                ),
            ],
            report_path="/tmp/test_report.md",
        )
        d = asdict(report)
        rebuilt = ResearchReport.from_dict(d)
        assert len(rebuilt.citations) == 1
        assert rebuilt.citations[0].citation_id == "C1"
        assert rebuilt.citations[0].relevance == "rel"
        assert rebuilt.report_path == "/tmp/test_report.md"


class TestCitationExtraction:
    """Tests for _extract_citations helper."""

    def test_extract_citations_basic(self):
        from app.research.deep_research import _extract_citations
        from app.research.evidence_searcher import Citation, EvidenceItemResult, EvidenceResult
        from app.research.hypothesis_generator import EvidenceNeed, Hypothesis

        hyp = Hypothesis(
            source="GROWTH_VS_EARNINGS_POWER",
            claim="Growth may be overstated",
            direction="BEARISH",
            evidence_needed=[EvidenceNeed("n1", "customer concentration", "REQUIRED")],
            falsification="",
            priority="HIGH",
        )
        item1 = EvidenceItemResult(
            need_id="n1",
            needed="customer concentration",
            importance="REQUIRED",
            status="CONFIRMS",
            classification_method="LLM",
            citations=[
                Citation(block_id="blk1", section="Item 1A", ordinal=1, excerpt="35% revenue")
            ],
            excerpt="35% revenue",
            structured_fact="35%",
            reasoning_short="Confirmed.",
            candidates_considered=3,
            top_candidate_score=0.9,
            candidate_rankings=[],
        )
        item2 = EvidenceItemResult(
            need_id="n2",
            needed="retention rate",
            importance="IMPORTANT",
            status="NOT_FOUND",
            classification_method="NONE",
            citations=[],
            excerpt="",
            structured_fact=None,
            reasoning_short="Not found.",
            candidates_considered=0,
            top_candidate_score=0.0,
            candidate_rankings=[],
        )
        results = [
            EvidenceResult(
                hypothesis=hyp,
                evidence_item_results=[item1, item2],
                hypothesis_status="PARTIALLY_CONFIRMED",
                coverage_score=0.5,
                classification_method="LLM",
            )
        ]

        citations = _extract_citations(results)
        assert len(citations) == 1
        assert citations[0].citation_id == "C1"
        assert citations[0].need_id == "n1"
        assert citations[0].section == "Item 1A"
        assert citations[0].excerpt == "35% revenue"
        assert citations[0].relevance == "customer concentration"
        assert citations[0].hypothesis_source == "GROWTH_VS_EARNINGS_POWER"

    def test_extract_citations_preserves_source_metadata(self):
        from app.research.deep_research import _extract_citations
        from app.research.evidence_searcher import Citation, EvidenceItemResult, EvidenceResult
        from app.research.hypothesis_generator import Hypothesis, EvidenceNeed

        hyp = Hypothesis(
            source="SRC_A",
            claim="claim",
            direction="BEARISH",
            evidence_needed=[EvidenceNeed("n1", "need A", "REQUIRED")],
            falsification="",
            priority="HIGH",
        )
        cite = Citation(
            block_id="blk1",
            section="mda",
            ordinal=1,
            excerpt="Quarterly margin improved.",
            source_form_type="10-Q",
            source_filing_date="2026-05-01",
            source_accession="0000000000-26-000001",
            source_role="quarterly",
        )
        item = EvidenceItemResult(
            need_id="n1",
            needed="need A",
            importance="REQUIRED",
            status="CONFIRMS",
            classification_method="LLM",
            citations=[cite],
            excerpt="Quarterly margin improved.",
            structured_fact=None,
            reasoning_short="ok",
            candidates_considered=1,
            top_candidate_score=0.9,
            candidate_rankings=[],
        )
        results = [
            EvidenceResult(
                hypothesis=hyp,
                evidence_item_results=[item],
                hypothesis_status="CONFIRMED",
                coverage_score=1.0,
                classification_method="LLM",
            )
        ]

        citations = _extract_citations(results)
        assert citations[0].source_form_type == "10-Q"
        assert citations[0].source_filing_date == "2026-05-01"
        assert citations[0].source_accession == "0000000000-26-000001"
        assert citations[0].source_role == "quarterly"
        assert citations[0].item_code is None
        assert citations[0].event_category is None

    def test_extract_citations_preserves_material_event_provenance(self):
        from app.research.deep_research import _extract_citations
        from app.research.evidence_searcher import Citation, EvidenceItemResult, EvidenceResult
        from app.research.hypothesis_generator import Hypothesis, EvidenceNeed

        hyp = Hypothesis(
            source="SRC_EVENT",
            claim="claim",
            direction="BEARISH",
            evidence_needed=[EvidenceNeed("n1", "leadership transition", "REQUIRED")],
            falsification="",
            priority="HIGH",
        )
        cite = Citation(
            block_id="blk_event",
            section="leadership_governance",
            ordinal=1,
            excerpt="The board appointed a new chief executive officer.",
            source_form_type="8-K",
            source_filing_date="2026-04-10",
            source_accession="0000000000-26-000101",
            source_role="material_event",
            item_code="5.02",
            event_category="leadership_governance",
        )
        item = EvidenceItemResult(
            need_id="n1",
            needed="leadership transition",
            importance="REQUIRED",
            status="CONFIRMS",
            classification_method="LLM",
            citations=[cite],
            excerpt="The board appointed a new chief executive officer.",
            structured_fact=None,
            reasoning_short="ok",
            candidates_considered=1,
            top_candidate_score=0.9,
            candidate_rankings=[],
        )
        results = [
            EvidenceResult(
                hypothesis=hyp,
                evidence_item_results=[item],
                hypothesis_status="CONFIRMED",
                coverage_score=1.0,
                classification_method="LLM",
            )
        ]

        citations = _extract_citations(results)
        assert citations[0].source_form_type == "8-K"
        assert citations[0].source_accession == "0000000000-26-000101"
        assert citations[0].source_role == "material_event"
        assert citations[0].item_code == "5.02"
        assert citations[0].event_category == "leadership_governance"

    def test_extract_citations_preserves_current_event_provenance(self):
        from app.research.deep_research import _extract_citations
        from app.research.evidence_searcher import Citation, EvidenceItemResult, EvidenceResult
        from app.research.hypothesis_generator import Hypothesis, EvidenceNeed

        hyp = Hypothesis(
            source="SRC_EVENT",
            claim="claim",
            direction="BULLISH",
            evidence_needed=[EvidenceNeed("n1", "guidance update", "REQUIRED")],
            falsification="",
            priority="HIGH",
        )
        cite = Citation(
            block_id="event_1",
            section="ir_press",
            ordinal=1,
            excerpt="Management raised revenue guidance for the year.",
            source_role="current_event",
            source_type="ir_press",
            source_title="Guidance update",
            source_url="https://example.com/press/guidance",
            source_published_at="2026-04-17T10:00:00+00:00",
            source_quality={
                "source_family": "company_controlled",
                "source_origin": "primary_company_controlled",
                "source_independence": "company_controlled",
                "source_domain": "example.com",
                "freshness_days": 1,
                "freshness_bucket": "recent_7d",
                "source_quality_score": 0.85,
                "calibration_status": "deterministic_heuristic",
                "reason_codes": [
                    "SOURCE_COMPANY_CONTROLLED",
                    "SOURCE_ISSUER_BIAS_POSSIBLE",
                    "FRESHNESS_RECENT_7D",
                ],
            },
        )
        item = EvidenceItemResult(
            need_id="n1",
            needed="guidance update",
            importance="REQUIRED",
            status="CONFIRMS",
            classification_method="LLM",
            citations=[cite],
            excerpt="Management raised revenue guidance for the year.",
            structured_fact=None,
            reasoning_short="ok",
            candidates_considered=1,
            top_candidate_score=0.9,
            candidate_rankings=[],
        )
        results = [
            EvidenceResult(
                hypothesis=hyp,
                evidence_item_results=[item],
                hypothesis_status="CONFIRMED",
                coverage_score=1.0,
                classification_method="LLM",
            )
        ]

        citations = _extract_citations(results)
        assert citations[0].source_role == "current_event"
        assert citations[0].source_type == "ir_press"
        assert citations[0].source_title == "Guidance update"
        assert citations[0].source_url == "https://example.com/press/guidance"
        assert citations[0].source_published_at == "2026-04-17T10:00:00+00:00"
        assert citations[0].source_quality["source_family"] == "company_controlled"

    def test_assemble_research_from_context_uses_latest_material_event_metadata(self, monkeypatch):
        from app.research.deep_research import assemble_research_from_filing_context

        annual = FilingDocument(
            ticker="TEST",
            cik="0000000001",
            accession="0000000000-25-000001",
            form_type="10-K",
            filing_date="2025-11-15",
            period_end=None,
            role="annual",
            local_path=None,
            primary_doc_url=None,
            html="<html><body><p>Annual filing</p></body></html>",
        )
        event = FilingDocument(
            ticker="TEST",
            cik="0000000001",
            accession="0000000000-26-000101",
            form_type="8-K",
            filing_date="2026-04-10",
            period_end=None,
            role="material_event",
            local_path=None,
            primary_doc_url=None,
            html="<html><body><p>Item 5.02 leadership update</p></body></html>",
        )
        context = FilingContext(documents=[annual, event])

        monkeypatch.setattr(
            "app.research.hypothesis_generator.generate_hypotheses",
            lambda *args, **kwargs: [SimpleNamespace(source="SRC_EVENT")],
        )
        monkeypatch.setattr(
            "app.research.evidence_searcher.search_evidence_in_filing_context",
            lambda *args, **kwargs: [],
        )
        monkeypatch.setattr(
            "app.research.thesis_updater.update_thesis",
            lambda *args, **kwargs: SimpleNamespace(unresolved=[], adjustments=[]),
        )
        monkeypatch.setattr(
            "app.research.deep_research._run_analyst_stage_from_context",
            lambda *args, **kwargs: None,
        )

        report = assemble_research_from_filing_context(
            ticker="TEST",
            as_of_date="2026-04-18",
            scorecard=_make_scorecard(),
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={"gate_action": "PROCEED"},
            solvency=None,
            filing_risk=None,
            filing_context=context,
        )

        assert report.status == "OK"
        assert report.form_type == "8-K"
        assert report.filing_date == "2026-04-10"

    def test_assemble_research_from_context_sets_latest_evidence_from_current_events(
        self, monkeypatch
    ):
        from app.research.deep_research import assemble_research_from_filing_context

        monkeypatch.setattr(
            "app.research.hypothesis_generator.generate_hypotheses",
            lambda *args, **kwargs: [],
        )

        report = assemble_research_from_filing_context(
            ticker="TEST",
            as_of_date="2026-04-18",
            scorecard=_make_scorecard(dcf=50.0, epv=40.0, price=200.0),
            tensions=_make_tensions(),
            anomalies=[],
            quality_ctx={"gate_action": "PROCEED"},
            solvency=None,
            filing_risk=None,
            filing_context=_make_filing_context(
                form_type="10-Q",
                filing_date="2026-04-01",
            ),
            current_event_context=_make_current_event_context(
                published_at="2026-04-17T10:00:00+00:00"
            ),
        )

        assert report.status == "NO_HYPOTHESES"
        assert report.filing_date == "2026-04-01"
        assert report.form_type == "10-Q"
        assert report.latest_evidence_date == "2026-04-17T10:00:00+00:00"
        assert report.latest_evidence_source_type == "ir_press"

    def test_extract_citations_dedup_by_block_excerpt(self):
        from app.research.deep_research import _extract_citations
        from app.research.evidence_searcher import Citation, EvidenceItemResult, EvidenceResult
        from app.research.hypothesis_generator import Hypothesis

        hyp = Hypothesis(
            source="SRC_A",
            claim="claim",
            direction="BEARISH",
            evidence_needed=[],
            falsification="",
            priority="HIGH",
        )
        shared_cite = Citation(
            block_id="blk1", section="Item 1A", ordinal=1, excerpt="same excerpt"
        )
        item1 = EvidenceItemResult(
            need_id="n1",
            needed="need A",
            importance="REQUIRED",
            status="CONFIRMS",
            classification_method="LLM",
            citations=[shared_cite],
            excerpt="same excerpt",
            structured_fact=None,
            reasoning_short="ok",
            candidates_considered=1,
            top_candidate_score=0.9,
            candidate_rankings=[],
        )
        item2 = EvidenceItemResult(
            need_id="n2",
            needed="need B",
            importance="IMPORTANT",
            status="CONFIRMS",
            classification_method="LLM",
            citations=[shared_cite],
            excerpt="same excerpt",
            structured_fact=None,
            reasoning_short="ok",
            candidates_considered=1,
            top_candidate_score=0.8,
            candidate_rankings=[],
        )
        results = [
            EvidenceResult(
                hypothesis=hyp,
                evidence_item_results=[item1, item2],
                hypothesis_status="CONFIRMED",
                coverage_score=1.0,
                classification_method="LLM",
            )
        ]

        citations = _extract_citations(results)
        assert len(citations) == 1
        assert citations[0].need_id == "n1"  # first encountered wins
        assert citations[0].additional_need_ids == ["n2"]  # second need_id tracked

    def test_extract_citations_empty_when_no_evidence(self):
        from app.research.deep_research import _extract_citations

        assert _extract_citations([]) == []


class TestArtifactPersistence:
    """Tests for artifact path generation and markdown write."""

    def test_artifact_base_path_format(self, monkeypatch):
        from app.config import get_config
        from app.research.deep_research import _artifact_base_path

        monkeypatch.setenv("VOE_DATA_DIR", "data")
        get_config.cache_clear()

        # Config anchors relative data dirs to the repo root, so the artifact
        # path is absolute even when VOE_DATA_DIR is the relative default.
        prefix = str(get_config().project_root / "data/outputs/research/FSLR_2026-04-04_")
        base = _artifact_base_path("FSLR", "2026-04-04")
        assert base.startswith(prefix)
        assert len(base) > len(prefix)
        get_config.cache_clear()

    def test_artifact_base_path_deterministic_prefix(self, monkeypatch):
        from app.config import get_config
        from app.research.deep_research import _artifact_base_path

        monkeypatch.setenv("VOE_DATA_DIR", "data")
        get_config.cache_clear()

        prefix = str(get_config().project_root / "data/outputs/research/AAPL_2026-01-01_")
        b1 = _artifact_base_path("AAPL", "2026-01-01")
        b2 = _artifact_base_path("AAPL", "2026-01-01")
        assert b1.startswith(prefix)
        assert b2.startswith(prefix)
        get_config.cache_clear()

    def test_artifact_base_path_uses_configured_data_dir(self, monkeypatch, tmp_path):
        from pathlib import Path

        from app.config import get_config
        from app.research.deep_research import _artifact_base_path

        data_dir = tmp_path / "isolated-data"
        monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
        get_config.cache_clear()

        base = _artifact_base_path("FSLR", "2026-04-04")

        assert Path(base).parent == data_dir / "outputs" / "research"
        assert Path(base).name.startswith("FSLR_2026-04-04_")
        get_config.cache_clear()

    def test_write_markdown_artifact_success(self, tmp_path):
        from app.research.deep_research import _write_markdown_artifact

        base = str(tmp_path / "TEST_2026-04-04_20260404T000000Z")
        md_path = _write_markdown_artifact(base, "# Report\nContent here")
        assert md_path is not None
        assert md_path.endswith("_report.md")
        from pathlib import Path as _P

        assert _P(md_path).read_text() == "# Report\nContent here"

    def test_write_markdown_artifact_diagnostic_suffix(self, tmp_path):
        from app.research.deep_research import _write_markdown_artifact

        base = str(tmp_path / "TEST_2026-04-04_20260404T000000Z")
        md_path = _write_markdown_artifact(base, "# Diagnostic\nContent here", diagnostic=True)
        assert md_path is not None
        assert md_path.endswith("_diagnostic_report.md")
        from pathlib import Path as _P

        assert _P(md_path).read_text() == "# Diagnostic\nContent here"

    def test_write_markdown_artifact_failure(self):
        from app.research.deep_research import _write_markdown_artifact
        from pathlib import Path as _P

        with patch.object(_P, "write_text", side_effect=OSError("disk full")):
            md_path = _write_markdown_artifact("/tmp/test_base", "# Report")
        assert md_path is None


class TestMarkdownIntegration:
    """Tests for markdown rendering wired into run_deep_research."""

    @patch("app.research.deep_research._write_markdown_artifact", return_value="/tmp/report.md")
    @patch("app.research.deep_research._write_artifact", return_value="/tmp/artifact.json")
    @patch("app.research.deep_research._persist_to_db")
    @patch("app.research.deep_research._load_filing_context")
    @patch("app.research.deep_research.scan_filing_risks", return_value={"status": "NO_FILING"})
    @patch("app.research.deep_research.assess_solvency", return_value=None)
    @patch("app.research.deep_research.detect_anomalies", return_value=[])
    @patch("app.research.deep_research._load_scorecard")
    def test_report_path_set_on_success(
        self,
        mock_load_sc,
        mock_anom,
        mock_solv,
        mock_fr,
        mock_filing,
        mock_persist,
        mock_artifact,
        mock_md,
    ):
        from app.research.deep_research import run_deep_research

        sc = _make_scorecard(gate_action="BLOCK")
        sc["quality_context"] = {
            "gate_action": "BLOCK",
            "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
            "negative_oe_years": 3,
        }
        mock_load_sc.return_value = (sc, "2026-03-25")
        mock_filing.return_value = _make_filing_context(html="<html><body>filing</body></html>")

        report = run_deep_research("TEST", as_of_date="2026-03-25")
        assert report.report_path == "/tmp/report.md"
        assert mock_md.called

    @patch("app.research.deep_research._write_markdown_artifact", return_value=None)
    @patch("app.research.deep_research._write_artifact", return_value="/tmp/artifact.json")
    @patch("app.research.deep_research._persist_to_db")
    @patch("app.research.deep_research._load_filing_context")
    @patch("app.research.deep_research.scan_filing_risks", return_value={"status": "NO_FILING"})
    @patch("app.research.deep_research.assess_solvency", return_value=None)
    @patch("app.research.deep_research.detect_anomalies", return_value=[])
    @patch("app.research.deep_research._load_scorecard")
    def test_markdown_failure_adds_warning(
        self,
        mock_load_sc,
        mock_anom,
        mock_solv,
        mock_fr,
        mock_filing,
        mock_persist,
        mock_artifact,
        mock_md,
    ):
        from app.research.deep_research import run_deep_research

        sc = _make_scorecard(gate_action="BLOCK")
        sc["quality_context"] = {
            "gate_action": "BLOCK",
            "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
            "negative_oe_years": 3,
        }
        mock_load_sc.return_value = (sc, "2026-03-25")
        mock_filing.return_value = _make_filing_context(html="<html><body>filing</body></html>")

        report = run_deep_research("TEST", as_of_date="2026-03-25")
        assert report.report_path is None
        warnings_arg = mock_persist.call_args[0][3]
        assert "markdown_write_failed" in warnings_arg


class TestUpstreamIngestion:
    """Tests for auto-ingestion wired into run_deep_research."""

    @patch("app.research.deep_research._write_markdown_artifact", return_value=None)
    @patch("app.research.deep_research._write_artifact")
    @patch("app.research.deep_research._persist_to_db")
    @patch("app.research.deep_research._load_filing_context", return_value=FilingContext())
    @patch("app.research.deep_research.scan_filing_risks", return_value={"status": "NO_FILING"})
    @patch("app.research.deep_research.assess_solvency", return_value=None)
    @patch("app.research.deep_research.detect_anomalies", return_value=[])
    @patch("app.research.deep_research._load_scorecard")
    @patch("app.valuation.valuation_writer.ensure_valuation")
    @patch("app.dossier.collector.collect_10k_docket", return_value=[])
    @patch("app.ingest.facts_writer.ensure_all_facts")
    def test_upstream_calls_made(
        self,
        mock_facts,
        mock_docket,
        mock_valuation,
        mock_load_sc,
        mock_anom,
        mock_solv,
        mock_fr,
        mock_filing,
        mock_persist,
        mock_artifact,
        mock_md,
    ):
        from app.research.deep_research import run_deep_research

        sc = _make_scorecard(gate_action="BLOCK")
        sc["quality_context"] = {
            "gate_action": "BLOCK",
            "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
            "negative_oe_years": 3,
        }
        mock_load_sc.return_value = (sc, "2026-03-25")

        run_deep_research("TEST", as_of_date="2026-03-25")

        mock_facts.assert_called_once_with("TEST", years_back=5)
        mock_docket.assert_called_once()
        mock_valuation.assert_called_once()

    @patch("app.parse.filing_parser.parse_pending_filings", return_value=2)
    @patch("app.ingest.filings.ingest_filings_between", return_value=3)
    def test_recent_filing_context_refresh_ingests_and_parses_ticker(
        self, mock_ingest, mock_parse, monkeypatch
    ):
        from app.research.deep_research import _ensure_recent_filing_context_cache
        from app.config import get_config

        monkeypatch.setenv("VOE_LOOKBACK_10Q_DAYS", "120")
        get_config.cache_clear()

        warnings = _ensure_recent_filing_context_cache("test", "2026-03-25", 2)

        assert warnings == []
        mock_ingest.assert_called_once_with(
            date(2025, 11, 25),
            date(2026, 3, 25),
            ["10-Q", "10-Q/A", "8-K", "8-K/A"],
            as_of_date="2026-03-25",
            tickers=["TEST"],
            limit=1,
        )
        mock_parse.assert_called_once_with(limit=10, tickers=["TEST"])
        get_config.cache_clear()

    @patch("app.ingest.filings.ingest_filings_between", side_effect=RuntimeError("sec unavailable"))
    def test_recent_filing_context_refresh_returns_warning_on_failure(self, mock_ingest):
        from app.research.deep_research import _ensure_recent_filing_context_cache

        warnings = _ensure_recent_filing_context_cache("TEST", "2026-03-25", 2)

        assert warnings == ["recent_filing_context_refresh_failed: sec unavailable"]
        mock_ingest.assert_called_once()

    @patch("app.research.deep_research._persist_to_db")
    @patch("app.research.deep_research._load_scorecard", return_value=(None, None))
    @patch("app.valuation.valuation_writer.ensure_valuation", side_effect=Exception("price fail"))
    @patch("app.dossier.collector.collect_10k_docket", side_effect=Exception("sec fail"))
    @patch("app.ingest.facts_writer.ensure_all_facts", side_effect=Exception("net fail"))
    def test_upstream_failures_non_fatal(
        self, mock_facts, mock_docket, mock_valuation, mock_load_sc, mock_persist
    ):
        from app.research.deep_research import run_deep_research

        report = run_deep_research("TEST", as_of_date="2026-03-25")
        # Pipeline should still complete (with NO_SCORECARD) despite upstream failures
        assert report.status == "NO_SCORECARD"


class TestResearchReportAnalystNotes:
    def test_new_fields_default_none(self):
        from app.research.deep_research import ResearchReport

        report = ResearchReport(
            ticker="TEST",
            as_of_date="2026-01-01",
            status="OK",
            started_at="t0",
            completed_at="t1",
            scorecard_present=True,
            filing_present=True,
            filing_date=None,
            form_type="10-K",
            anomaly_count=0,
            solvency_status=None,
            filing_risk_status=None,
            gate_action="PROCEED",
            investigation_ran=True,
            hypotheses_generated=1,
            thesis=None,
            researchable_items=[],
            not_researchable_items=[],
            total_adjustments=0,
            fact_calibrated_count=0,
            heuristic_count=0,
            methods_agree=True,
            consensus_strength=3,
            method_count=3,
            tension_type="NONE",
        )
        assert report.analyst_notes is None
        assert report.merged_findings is None

    def test_from_dict_round_trip_with_analyst_notes(self):
        from app.research.deep_research import ResearchReport

        citation = AnalystCitation(section="mda", excerpt="test excerpt", block_id="mda_p0")
        note = AnalystNote(
            category="RISK",
            claim="Customer concentration",
            direction="BEARISH",
            severity="HIGH",
            citations=[citation],
            suggested_adjustment=None,
            validation_status="VERIFIED",
        )
        analyst_notes = AnalystNotes(
            ticker="TEST",
            positives=[],
            risks=[note],
            surprises=[],
            adjustment_triggers=[],
            overall_assessment="Test assessment.",
            filing_sections_read=["mda"],
        )
        pf = PipelineFinding(
            claim="Revenue decline",
            direction="BEARISH",
            section="mda",
            content_words={"revenue", "decline"},
            source_hypothesis="REV_DECLINE",
            evidence_status="CONFIRMED",
        )
        matched = MatchedFinding(
            llm_note=note,
            pipeline_finding=pf,
            shared_words=["revenue"],
        )
        merged = MergedFindings(
            both_paths=[matched],
            llm_only=[],
            pipeline_only=[],
            agreement_score=1.0,
        )

        report = ResearchReport(
            ticker="TEST",
            as_of_date="2026-01-01",
            status="OK",
            started_at="t0",
            completed_at="t1",
            scorecard_present=True,
            filing_present=True,
            filing_date=None,
            form_type="10-K",
            anomaly_count=0,
            solvency_status=None,
            filing_risk_status=None,
            gate_action="PROCEED",
            investigation_ran=True,
            hypotheses_generated=1,
            thesis=None,
            researchable_items=[],
            not_researchable_items=[],
            total_adjustments=0,
            fact_calibrated_count=0,
            heuristic_count=0,
            methods_agree=True,
            consensus_strength=3,
            method_count=3,
            tension_type="NONE",
            analyst_notes=analyst_notes,
            merged_findings=merged,
        )

        d = asdict(report)
        restored = ResearchReport.from_dict(d)
        assert restored.analyst_notes is not None
        assert restored.analyst_notes.ticker == "TEST"
        assert len(restored.analyst_notes.risks) == 1
        assert restored.analyst_notes.risks[0].claim == "Customer concentration"
        assert restored.analyst_notes.risks[0].citations[0].section == "mda"
        assert restored.merged_findings is not None
        assert len(restored.merged_findings.both_paths) == 1
        assert restored.merged_findings.agreement_score == 1.0

    def test_from_dict_round_trip_without_analyst_notes(self):
        from app.research.deep_research import ResearchReport

        report = ResearchReport(
            ticker="TEST",
            as_of_date="2026-01-01",
            status="OK",
            started_at="t0",
            completed_at="t1",
            scorecard_present=True,
            filing_present=True,
            filing_date=None,
            form_type="10-K",
            anomaly_count=0,
            solvency_status=None,
            filing_risk_status=None,
            gate_action="PROCEED",
            investigation_ran=True,
            hypotheses_generated=1,
            thesis=None,
            researchable_items=[],
            not_researchable_items=[],
            total_adjustments=0,
            fact_calibrated_count=0,
            heuristic_count=0,
            methods_agree=True,
            consensus_strength=3,
            method_count=3,
            tension_type="NONE",
        )

        d = asdict(report)
        restored = ResearchReport.from_dict(d)
        assert restored.analyst_notes is None
        assert restored.merged_findings is None


class TestAssembleResearchWithAnalystNotes:
    """Integration test: assemble_research with analyst notes enabled."""

    def _mock_llm_response(self):
        return {
            "overall_assessment": "Solid company.",
            "positives": [
                {
                    "category": "POSITIVE",
                    "claim": "Strong revenue growth",
                    "direction": "BULLISH",
                    "severity": "HIGH",
                    "citations": [
                        {"section": "mda", "excerpt": "Revenue grew 15%", "block_id": "mda_p0"}
                    ],
                    "suggested_adjustment": None,
                }
            ],
            "risks": [],
            "surprises": [],
            "adjustment_triggers": [],
        }

    def test_analyst_notes_attached_when_enabled(self, monkeypatch):
        from app.config import get_config

        monkeypatch.setenv("VOE_ANALYST_NOTES", "enabled")
        monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
        monkeypatch.setenv("VOE_OPENAI_API_KEY", "test-key")
        get_config.cache_clear()

        mock_result = MagicMock()
        mock_result.json_text = json.dumps(self._mock_llm_response())
        mock_provider = MagicMock()
        mock_provider.provider_name = "openai"
        mock_provider.synthesize_json.return_value = mock_result

        filing_html = """
        <h1>Item 7. Management's Discussion and Analysis</h1>
        <p>Revenue grew 15% driven by enterprise expansion and new product launches during the fiscal year.</p>
        <h1>Item 1A. Risk Factors</h1>
        <p>We face significant competition in all markets where we operate and sell our products.</p>
        """

        from app.research.deep_research import assemble_research

        with (
            patch("app.research.analyst_notes.get_llm_provider", return_value=mock_provider),
            patch("app.research.deep_research._write_markdown_artifact", return_value=None),
        ):
            report = assemble_research(
                ticker="TEST",
                as_of_date="2026-01-01",
                scorecard={
                    "pricing_zone_detail": {
                        "dcf_base": 50.0,
                        "epv_adjusted": 45.0,
                        "current_price": 35.0,
                    },
                    "pre_valuation_gate": {"action": "PROCEED"},
                    "discounts": {},
                },
                tensions={
                    "tension_type": "NONE",
                    "methods_agree": True,
                    "consensus_strength": 3,
                    "method_count": 3,
                },
                anomalies=[],
                quality_ctx={"gate_action": "PROCEED"},
                solvency=None,
                filing_risk=None,
                filing_html=filing_html,
                form_type="10-K",
            )
        assert report.analyst_notes is not None
        assert report.analyst_notes.ticker == "TEST"
        get_config.cache_clear()

    def test_analyst_notes_none_when_disabled(self, monkeypatch):
        from app.config import get_config

        monkeypatch.setenv("VOE_ANALYST_NOTES", "disabled")
        get_config.cache_clear()

        filing_html = (
            "<h1>Item 7. Management's Discussion and Analysis</h1><p>Test content here.</p>"
        )

        from app.research.deep_research import assemble_research

        with patch("app.research.deep_research._write_markdown_artifact", return_value=None):
            report = assemble_research(
                ticker="TEST",
                as_of_date="2026-01-01",
                scorecard={
                    "pricing_zone_detail": {
                        "dcf_base": 50.0,
                        "epv_adjusted": 45.0,
                        "current_price": 35.0,
                    },
                    "pre_valuation_gate": {"action": "PROCEED"},
                    "discounts": {},
                },
                tensions={
                    "tension_type": "NONE",
                    "methods_agree": True,
                    "consensus_strength": 3,
                    "method_count": 3,
                },
                anomalies=[],
                quality_ctx={"gate_action": "PROCEED"},
                solvency=None,
                filing_risk=None,
                filing_html=filing_html,
                form_type="10-K",
            )
        assert report.analyst_notes is None
        get_config.cache_clear()

    def test_no_hypotheses_gets_analyst_notes_but_no_reconciliation(self, monkeypatch):
        """NO_HYPOTHESES path: Stage A runs but Stage C is skipped (spec: either path missing → no reconciliation)."""
        from app.config import get_config

        monkeypatch.setenv("VOE_ANALYST_NOTES", "enabled")
        monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
        monkeypatch.setenv("VOE_OPENAI_API_KEY", "test-key")
        get_config.cache_clear()

        mock_result = MagicMock()
        mock_result.json_text = json.dumps(self._mock_llm_response())
        mock_provider = MagicMock()
        mock_provider.provider_name = "openai"
        mock_provider.synthesize_json.return_value = mock_result

        # Filing that won't produce hypotheses (no tensions, no anomalies, no gate signals)
        filing_html = """
        <h1>Item 7. Management's Discussion and Analysis</h1>
        <p>Revenue grew 15% driven by enterprise expansion and new product launches during the fiscal year.</p>
        <h1>Item 1A. Risk Factors</h1>
        <p>We face significant competition in all markets where we operate and sell our products.</p>
        """

        from app.research.deep_research import assemble_research

        with (
            patch("app.research.analyst_notes.get_llm_provider", return_value=mock_provider),
            patch("app.research.deep_research._write_markdown_artifact", return_value=None),
        ):
            report = assemble_research(
                ticker="TEST",
                as_of_date="2026-01-01",
                scorecard={
                    "pricing_zone_detail": {
                        "dcf_base": 50.0,
                        "epv_adjusted": 45.0,
                        "current_price": 35.0,
                    },
                    "pre_valuation_gate": {"action": "PROCEED"},
                    "discounts": {},
                },
                tensions={
                    "tension_type": "NONE",
                    "methods_agree": True,
                    "consensus_strength": 3,
                    "method_count": 3,
                },
                anomalies=[],
                quality_ctx={"gate_action": "PROCEED"},
                solvency=None,
                filing_risk=None,
                filing_html=filing_html,
                form_type="10-K",
            )

        # Stage A ran — analyst notes present
        assert report.analyst_notes is not None
        assert report.analyst_notes.ticker == "TEST"
        # Stage C skipped — no reconciliation (spec: either path missing → skip)
        assert report.merged_findings is None
        # Confirm this was a NO_HYPOTHESES report
        assert report.status == "NO_HYPOTHESES"
        get_config.cache_clear()


class TestExpectationsGapPassthrough:
    """The implied/supportable legs are sourced from the
    method='reverse_dcf' valuation row (merged under scorecard['reverse_dcf']),
    NOT from non-existent top-level scorecard keys. These tests exercise the
    live data shape rather than injecting resolved scorecard_* fields directly,
    so a regression that dead-wires the source is caught."""

    def test_legacy_reverse_dcf_row_recomputed_via_canonical_authority(self):
        from app.research.deep_research import _expectations_gap_passthrough

        scorecard = {
            "pricing_zone_detail": {},
            "discounts": {},
            "reverse_dcf": {
                "status": "OK",
                "revenue_cagr_5y_used": 0.10,
                "feasibility": "REASONABLE",
                "outputs": {
                    "implied_growth": 0.04,
                    "feasibility_gap_score": 0.5,
                    "target_ev": 1.0e9,
                },
            },
        }
        implied, supportable, gap, bucket, line = _expectations_gap_passthrough(scorecard)
        assert implied == 0.04
        assert supportable == 0.10
        assert gap == -0.06
        assert bucket == "CHEAP_VS_EXPECTATIONS"
        assert line == "Market implies 4% growth; supportable 10%; gap -6% (CHEAP_VS_EXPECTATIONS)"

    def test_persisted_expectations_gap_dict_used_verbatim(self):
        from app.research.deep_research import _expectations_gap_passthrough

        # Production shape: expectations_gap lives at the TOP LEVEL of the
        # reverse_dcf outputs_json (sibling to "outputs"), NOT under "outputs".
        # The persisted dict is deliberately DIVERGENT from a quality_flags=[]
        # recompute: revenue_cagr_5y_used=0.10 would recompute supportable=0.10
        # (gap -0.06 -> CHEAP), but the persisted T4-haircut authority says
        # supportable=0.06 (gap +0.06 -> EXPENSIVE). Consuming the persisted
        # dict verbatim must win, proving the persisted branch is live.
        scorecard = {
            "reverse_dcf": {
                "status": "OK",
                "revenue_cagr_5y_used": 0.10,
                "outputs": {
                    "implied_growth": 0.12,
                },
                "expectations_gap": {
                    "gap": 0.06,
                    "bucket": "EXPENSIVE_VS_EXPECTATIONS",
                    "supportable_growth": 0.06,
                    "implied_growth_saturated": False,
                    "line": "Market implies 12% growth; supportable 6%; gap 6% (EXPENSIVE_VS_EXPECTATIONS)",
                },
            },
        }
        implied, supportable, gap, bucket, line = _expectations_gap_passthrough(scorecard)
        assert implied == 0.12
        assert supportable == 0.06
        assert gap == 0.06
        assert bucket == "EXPENSIVE_VS_EXPECTATIONS"
        assert (
            line == "Market implies 12% growth; supportable 6%; gap 6% (EXPENSIVE_VS_EXPECTATIONS)"
        )

    def test_missing_reverse_dcf_row_is_silent(self):
        from app.research.deep_research import _expectations_gap_passthrough

        scorecard = {"pricing_zone_detail": {}, "discounts": {}}
        implied, supportable, gap, bucket, line = _expectations_gap_passthrough(scorecard)
        assert implied is None
        assert gap is None
        assert bucket is None
        assert line is None

    def test_saturated_implied_growth_is_silent(self):
        from app.research.deep_research import _expectations_gap_passthrough

        scorecard = {
            "reverse_dcf": {
                "status": "OK",
                "revenue_cagr_5y_used": 0.06,
                "outputs": {
                    "implied_growth": 0.60,
                    "implied_growth_saturated": True,
                },
            },
        }
        implied, supportable, gap, bucket, line = _expectations_gap_passthrough(scorecard)
        assert gap is None
        assert bucket is None
        assert line is None

    def test_persisted_unreliable_bucket_is_silent(self):
        from app.research.deep_research import _expectations_gap_passthrough

        # Production shape: expectations_gap at TOP LEVEL, sibling to "outputs".
        scorecard = {
            "reverse_dcf": {
                "outputs": {
                    "implied_growth": 0.60,
                },
                "expectations_gap": {
                    "gap": None,
                    "bucket": "EXPECTATIONS_GAP_UNRELIABLE",
                    "supportable_growth": 0.06,
                    "implied_growth_saturated": True,
                    "line": None,
                },
            },
        }
        implied, supportable, gap, bucket, line = _expectations_gap_passthrough(scorecard)
        assert gap is None
        assert bucket is None
        assert line is None


def test_deep_research_persistence_binds_exact_canonical_artifact(monkeypatch, tmp_path):
    import hashlib
    import json

    from app.config import get_config
    from app.db import get_db, init_db
    from app.research.deep_research import ResearchReport, _persist_to_db
    from app.valuation.lineage import valuation_integrity_fingerprint

    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()
    cfg = get_config()

    report = ResearchReport(
        ticker="TEST",
        as_of_date="2026-03-25",
        status="OK",
        started_at="2026-03-25T00:00:00Z",
        completed_at="2026-03-25T00:01:00Z",
        scorecard_present=True,
        filing_present=True,
        filing_date="2026-02-01",
        form_type="10-K",
        anomaly_count=0,
        solvency_status="PASS",
        filing_risk_status="PASS",
        gate_action="PROCEED",
        investigation_ran=True,
        hypotheses_generated=1,
        thesis=None,
        researchable_items=[],
        not_researchable_items=[],
        total_adjustments=0,
        fact_calibrated_count=0,
        heuristic_count=0,
        methods_agree=True,
        consensus_strength=2,
        method_count=2,
        tension_type="NONE",
        run_id="TEST_2026-03-25_exact",
        report_path=str((cfg.research_dir / "TEST_2026-03-25_exact.md").resolve()),
        artifact_path=str((cfg.research_dir / "TEST_2026-03-25_exact.json").resolve()),
    )
    artifact_path = cfg.research_dir
    artifact_path.mkdir(parents=True, exist_ok=True)
    canonical_path = artifact_path / "TEST_2026-03-25_exact.json"
    assert str(canonical_path.resolve()) == report.artifact_path
    canonical_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    _persist_to_db("TEST", "2026-03-25", report, report.warnings)

    with get_db() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM valuations
            WHERE ticker = 'TEST' AND method = 'deep_research'
            """
        ).fetchone()
    assert row["source_run_id"] == "TEST_2026-03-25_exact"
    assert row["source_artifact_path"] == str(canonical_path.resolve())
    assert row["source_artifact_sha256"] == hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    assert row["financial_integrity_fingerprint"] == (valuation_integrity_fingerprint(row))
