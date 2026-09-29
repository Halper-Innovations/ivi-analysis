"""Tests for app.analyst.bundle_builder (Redirect Task 2)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.analyst import AnalysisEvidenceBundle
from app.research.current_event_context import CurrentEventContext, CurrentEventDocument
from app.research.filing_context import FilingContext, FilingDocument
from app.analyst.bundle_builder import (
    _canonical_section,
    _is_live_as_of_date,
    _resolve_as_of_date,
    _validate_as_of_date,
    build_analysis_evidence_bundle,
)
from tests.financial_integrity_helpers import canonicalize_financial_packet


def test_bundle_builder_module_imports():
    """The builder module should import cleanly."""
    assert build_analysis_evidence_bundle is not None


# ---------------------------------------------------------------------------
# _validate_as_of_date
# ---------------------------------------------------------------------------


def test_validate_as_of_date_none_returns_silently():
    _validate_as_of_date(None)  # must not raise


def test_validate_as_of_date_valid_iso_returns_silently():
    _validate_as_of_date("2025-01-15")
    _validate_as_of_date("2026-04-10")
    _validate_as_of_date("2024-12-31")


def test_validate_as_of_date_malformed_string_raises_value_error():
    with pytest.raises(ValueError, match="banana"):
        _validate_as_of_date("banana")


def test_validate_as_of_date_invalid_month_raises_value_error():
    with pytest.raises(ValueError, match="2026-13-99"):
        _validate_as_of_date("2026-13-99")


def test_validate_as_of_date_empty_string_raises_value_error():
    with pytest.raises(ValueError):
        _validate_as_of_date("")


# ---------------------------------------------------------------------------
# _resolve_as_of_date
# ---------------------------------------------------------------------------


def test_resolve_as_of_date_none_returns_today():
    result = _resolve_as_of_date(None)
    assert result == date.today().isoformat()


def test_resolve_as_of_date_echoes_explicit_string():
    assert _resolve_as_of_date("2025-01-15") == "2025-01-15"
    assert _resolve_as_of_date("2024-12-31") == "2024-12-31"


# ---------------------------------------------------------------------------
# _is_live_as_of_date
# ---------------------------------------------------------------------------


def test_is_live_as_of_date_none_is_live():
    assert _is_live_as_of_date(None) is True


def test_is_live_as_of_date_today_is_live():
    today = date.today().isoformat()
    assert _is_live_as_of_date(today) is True


def test_is_live_as_of_date_yesterday_is_historical():
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    assert _is_live_as_of_date(yesterday) is False


def test_is_live_as_of_date_historical_year_is_historical():
    assert _is_live_as_of_date("2022-06-15") is False
    assert _is_live_as_of_date("2020-01-01") is False


def test_is_live_as_of_date_malformed_string_is_not_live():
    assert _is_live_as_of_date("banana") is False
    assert _is_live_as_of_date("2026-13-99") is False


# ---------------------------------------------------------------------------
# _canonical_section
# ---------------------------------------------------------------------------


def test_canonical_section_maps_known_labels():
    assert _canonical_section("md_and_a") == "mda"
    assert _canonical_section("risk_factors") == "risk_factors"
    assert _canonical_section("notes") == "fin_notes"
    assert _canonical_section("business") == "business"
    assert _canonical_section("item_502") == "leadership_governance"
    assert _canonical_section("item_402") == "restatement_controls"


def test_canonical_section_drops_unsupported_labels():
    assert _canonical_section("financial_statements") is None
    assert _canonical_section("segment_info") is None


def test_canonical_section_unknown_label_returns_none():
    assert _canonical_section("banana") is None
    assert _canonical_section("") is None


# ---------------------------------------------------------------------------
# _build_valuation_snapshot
# ---------------------------------------------------------------------------


from dataclasses import dataclass as _dc

from app.analyst.bundle_builder import _build_valuation_snapshot
from app.analyst.evidence_bundle import ValuationSnapshot


def _sample_scorecard() -> dict:
    return {
        "pricing_zone_detail": {
            "current_price": 45.50,
            "market_cap": 12000.0,
            "dcf_base": 62.10,
            "epv_adjusted": 48.75,
            "gate_action": "PROCEED",
        },
        "discounts": {"graham": 0.20},
        "quality_context": {"gate_action": "PROCEED"},
    }


def _canonical_financial_packet(ticker: str = "EXAMPLE"):
    from app.alpha.schemas import TickerSignalPacket
    from app.autonomous.financial_integrity import stable_quote_hash

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
    return canonicalize_financial_packet(
        packet,
        as_of_date=as_of_date,
        shares_mm=10.0,
    )


@_dc
class _FakeSolvency:
    solvency_risk: str = "LOW"


def test_build_valuation_snapshot_populated_scorecard():
    scorecard = _sample_scorecard()
    tensions = {"methods_agree": True, "tension_type": "NONE"}
    solvency = _FakeSolvency(solvency_risk="LOW")
    filing_risk = {"status": "OK"}
    snapshot = _build_valuation_snapshot(scorecard, tensions, solvency, filing_risk)
    assert isinstance(snapshot, ValuationSnapshot)
    assert snapshot.current_price == 45.50
    assert snapshot.market_cap == 12000.0
    assert snapshot.dcf_base == 62.10
    assert snapshot.epv_adjusted == 48.75
    # TEXTBOOK-discount inversion: iv = price/(1 - d) (audit: graham-discount-inversion)
    assert snapshot.graham_value == pytest.approx(45.50 / 0.80)
    assert snapshot.methods_agree is True
    assert snapshot.tension_type == "NONE"
    assert snapshot.gate_action == "PROCEED"
    assert snapshot.solvency_status == "LOW"
    assert snapshot.filing_risk_status == "OK"


def test_build_valuation_snapshot_none_scorecard():
    snapshot = _build_valuation_snapshot(None, {}, None, None)
    assert snapshot.current_price is None
    assert snapshot.market_cap is None
    assert snapshot.dcf_base is None
    assert snapshot.epv_adjusted is None
    assert snapshot.graham_value is None
    assert snapshot.methods_agree is None
    assert snapshot.tension_type is None
    assert snapshot.gate_action is None
    assert snapshot.solvency_status is None
    assert snapshot.filing_risk_status is None


def test_build_valuation_snapshot_missing_pricing_zone_detail():
    scorecard = {"quality_context": {"gate_action": "BLOCK"}}
    snapshot = _build_valuation_snapshot(scorecard, {}, None, None)
    assert snapshot.current_price is None
    assert snapshot.dcf_base is None
    assert snapshot.gate_action == "BLOCK"


def test_build_valuation_snapshot_solvency_and_filing_risk_none():
    scorecard = _sample_scorecard()
    snapshot = _build_valuation_snapshot(scorecard, {}, None, None)
    assert snapshot.solvency_status is None
    assert snapshot.filing_risk_status is None


def test_build_valuation_snapshot_graham_skipped_when_discount_missing():
    scorecard = _sample_scorecard()
    scorecard["discounts"] = {}
    snapshot = _build_valuation_snapshot(scorecard, {}, None, None)
    assert snapshot.graham_value is None


def test_build_valuation_snapshot_graham_skipped_when_discounts_key_missing():
    scorecard = _sample_scorecard()
    scorecard.pop("discounts")
    snapshot = _build_valuation_snapshot(scorecard, {}, None, None)
    assert snapshot.graham_value is None


def test_build_valuation_snapshot_graham_skipped_when_price_missing():
    scorecard = _sample_scorecard()
    scorecard["pricing_zone_detail"]["current_price"] = None
    snapshot = _build_valuation_snapshot(scorecard, {}, None, None)
    assert snapshot.graham_value is None


def test_build_valuation_snapshot_graham_skipped_when_discount_is_minus_one_sentinel():
    """graham_disc == -1 is the legacy "not computable" sentinel and must be skipped."""
    scorecard = _sample_scorecard()
    scorecard["discounts"] = {"graham": -1}
    snapshot = _build_valuation_snapshot(scorecard, {}, None, None)
    assert snapshot.graham_value is None


# ---------------------------------------------------------------------------
# _build_bundle_filings
# ---------------------------------------------------------------------------


from app.analyst.bundle_builder import _build_bundle_filings
from app.analyst.evidence_bundle import BundleFiling
from app.dossier.collector import DossierFiling


def _dossier_filing(form_type: str = "10-K", accession: str = "0001-25-01") -> DossierFiling:
    return DossierFiling(
        ticker="EXAMPLE",
        cik="0000000001",
        accession=accession,
        form_type=form_type,
        filing_date="2025-02-14",
        period_end="2024-12-31",
        primary_doc_url="https://sec.gov/example.htm",
        local_path="/tmp/example.html",
        filing_id=1,
    )


@_dc
class _Span:
    section_label: str
    text: str


def test_build_bundle_filings_annual_form_happy_path(monkeypatch):
    filing = _dossier_filing(form_type="10-K")

    def fake_reader(f: DossierFiling) -> str | None:
        return "full filing text"

    def fake_segment(text: str):
        return [
            _Span(section_label="md_and_a", text="MDA text"),
            _Span(section_label="risk_factors", text="Risk text"),
            _Span(section_label="financial_statements", text="FS text"),  # dropped
        ]

    monkeypatch.setattr("app.analyst.bundle_builder.segment_10k_sections", fake_segment)

    filings, warnings = _build_bundle_filings([filing], reader=fake_reader)
    assert len(filings) == 1
    bundle_filing = filings[0]
    assert isinstance(bundle_filing, BundleFiling)
    assert bundle_filing.form_type == "10-K"
    assert bundle_filing.role == "annual"
    assert bundle_filing.accession == "0001-25-01"
    assert bundle_filing.sections_included == ["mda", "risk_factors"]
    assert bundle_filing.section_text == {"mda": "MDA text", "risk_factors": "Risk text"}
    assert warnings == []


def test_build_bundle_filings_skips_non_annual_forms(monkeypatch):
    annual = _dossier_filing(form_type="10-K", accession="ANNUAL-01")
    quarterly = _dossier_filing(form_type="10-Q", accession="Q-01")

    def fake_reader(f: DossierFiling) -> str | None:
        return "text"

    def fake_segment(text: str):
        return [_Span(section_label="md_and_a", text="MDA")]

    monkeypatch.setattr("app.analyst.bundle_builder.segment_10k_sections", fake_segment)

    filings, warnings = _build_bundle_filings([annual, quarterly], reader=fake_reader)
    assert len(filings) == 1
    assert filings[0].accession == "ANNUAL-01"
    assert warnings == []


def test_build_bundle_filings_reader_returns_none(monkeypatch):
    filing = _dossier_filing(accession="MISSING-01")

    def fake_reader(f: DossierFiling) -> str | None:
        return None

    filings, warnings = _build_bundle_filings([filing], reader=fake_reader)
    assert filings == []
    assert "filing_text_unavailable:MISSING-01" in warnings


def test_build_bundle_filings_empty_sections(monkeypatch):
    filing = _dossier_filing(accession="EMPTY-01")

    def fake_reader(f: DossierFiling) -> str | None:
        return "raw text"

    monkeypatch.setattr("app.analyst.bundle_builder.segment_10k_sections", lambda t: [])

    filings, warnings = _build_bundle_filings([filing], reader=fake_reader)
    assert filings == []
    assert "filing_sections_empty:EMPTY-01" in warnings


def test_build_bundle_filings_all_sections_dropped(monkeypatch):
    filing = _dossier_filing(accession="DROPPED-01")

    def fake_reader(f: DossierFiling) -> str | None:
        return "raw text"

    def fake_segment(text: str):
        return [
            _Span(section_label="financial_statements", text="FS"),
            _Span(section_label="segment_info", text="SI"),
        ]

    monkeypatch.setattr("app.analyst.bundle_builder.segment_10k_sections", fake_segment)

    filings, warnings = _build_bundle_filings([filing], reader=fake_reader)
    assert filings == []
    assert "filing_sections_empty:DROPPED-01" in warnings


# ---------------------------------------------------------------------------
# _build_recent_events
# ---------------------------------------------------------------------------


from app.analyst.bundle_builder import _build_bundle_filing_record, _build_recent_events
from app.analyst.evidence_bundle import BundleEvent


def _filing_document(
    *,
    form_type: str,
    role: str,
    accession: str,
    filing_date: str = "2026-04-09",
    html: str = "<h1>Section</h1><p>Body</p>",
    url: str | None = "https://www.sec.gov/Archives/example.htm",
) -> FilingDocument:
    return FilingDocument(
        ticker="EXAMPLE",
        cik="0000000001",
        accession=accession,
        form_type=form_type,
        filing_date=filing_date,
        period_end=None,
        role=role,
        local_path=None,
        primary_doc_url=url,
        html=html,
    )


def test_build_recent_events_happy_path(monkeypatch):
    monkeypatch.setattr(
        "app.analyst.bundle_builder.segment_8k_sections",
        lambda text: [
            _Span(
                section_label="item_502", text="Chief executive officer transition announced. " * 20
            )
        ],
    )
    events = _build_recent_events(
        [_filing_document(form_type="8-K", role="material_event", accession="8K-01")]
    )
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, BundleEvent)
    assert ev.source_type == "8-K"
    assert ev.published_at == "2026-04-09"
    assert ev.title == "8-K Item 5.02 — leadership governance"
    assert ev.source_url == "https://www.sec.gov/Archives/example.htm"
    assert ev.materiality is None
    assert ev.accession == "8K-01"
    assert ev.item_code == "5.02"
    assert ev.event_category == "leadership_governance"
    assert ev.source_quality == {
        "source_family": "regulatory_filing",
        "source_origin": "primary",
        "source_independence": "regulatory",
        "source_domain": "www.sec.gov",
        "freshness_days": None,
        "freshness_bucket": "undated",
        "source_quality_score": 0.95,
        "calibration_status": "deterministic_heuristic",
        "reason_codes": [
            "SOURCE_PRIMARY_REGULATORY",
            "FRESHNESS_UNDATED",
        ],
    }


def test_build_recent_events_recovers_item_102_termination_without_stubbed_segmenter():
    html = (
        "<html><body>"
        "<div>Table of Contents Item 1.02 Termination of a Material Definitive Agreement "
        "Item 7.01 Regulation FD Disclosure</div>"
        "<h1>Item 1.02 Termination of a Material Definitive Agreement.</h1>"
        + (
            "<p>The company terminated a material supply agreement after counterparty performance deteriorated. "
            "Management said the termination could disrupt deliveries and create transition costs.</p>"
            * 10
        )
        + "<h1>Item 7.01 Regulation FD Disclosure.</h1>"
        + (
            "<p>The investor presentation includes supplemental overview material for reference.</p>"
            * 10
        )
        + "<p>SIGNATURES</p>"
        "</body></html>"
    )

    events = _build_recent_events(
        [_filing_document(form_type="8-K", role="material_event", accession="8K-102", html=html)]
    )

    assert len(events) == 2
    assert events[0].title == "8-K Item 1.02 — agreement termination"
    assert events[0].item_code == "1.02"
    assert events[0].event_category == "agreement_termination"
    assert "terminated a material supply agreement" in events[0].summary
    assert "investor presentation includes supplemental" not in events[0].summary


def test_build_recent_events_truncates_long_excerpt(monkeypatch):
    long_text = "x" * 1000
    monkeypatch.setattr(
        "app.analyst.bundle_builder.segment_8k_sections",
        lambda text: [_Span(section_label="item_202", text=long_text)],
    )
    events = _build_recent_events(
        [_filing_document(form_type="8-K", role="material_event", accession="8K-02")]
    )
    assert len(events[0].summary) == 500


def test_build_recent_events_skips_unrecognized_sections(monkeypatch):
    monkeypatch.setattr(
        "app.analyst.bundle_builder.segment_8k_sections",
        lambda text: [_Span(section_label="other", text="Generic cover page text")],
    )
    events = _build_recent_events(
        [_filing_document(form_type="8-K", role="material_event", accession="8K-03")]
    )
    assert events == []


def test_build_recent_events_empty_result():
    events = _build_recent_events([])
    assert events == []


def test_build_bundle_filing_record_for_material_event(monkeypatch):
    monkeypatch.setattr(
        "app.analyst.bundle_builder.segment_8k_sections",
        lambda text: [
            _Span(section_label="item_502", text="Leadership change."),
            _Span(section_label="item_701", text="Results discussion."),
        ],
    )

    filing, warning = _build_bundle_filing_record(
        text="<html></html>",
        form_type="8-K",
        filing_date="2026-04-09",
        accession="8K-04",
        role="material_event",
    )

    assert warning is None
    assert filing is not None
    assert filing.role == "material_event"
    assert filing.sections_included == ["leadership_governance", "results_guidance"]


# ---------------------------------------------------------------------------
# Integration tests with monkeypatched IO
# ---------------------------------------------------------------------------

from typing import Any


def _monkeypatch_builder_io(
    monkeypatch,
    *,
    scorecard: tuple | None = None,
    docket: list | None = None,
    solvency: Any | None = None,
    filing_risk: dict | None = None,
    tensions: dict | None = None,
    filing_context: FilingContext | None = None,
    reader_result: str | None = "filing text",
    segment_spans: list | None = None,
    quarterly_segment_spans: list | None = None,
    material_event_segment_spans: list | None = None,
    ensure_all_facts_fails: bool = False,
    collect_docket_fails: bool = False,
    ensure_valuation_fails: bool = False,
    load_scorecard_fails: bool = False,
    load_filing_context_fails: bool = False,
    assess_solvency_fails: bool = False,
    scan_filing_risks_fails: bool = False,
    tensions_fails: bool = False,
):
    """Helper that monkeypatches every IO seam the builder uses."""

    def fake_ensure_all_facts(ticker, years_back=5):
        if ensure_all_facts_fails:
            raise RuntimeError("facts failed")

    def fake_collect_docket(*, ticker, as_of_date, years_back):
        if collect_docket_fails:
            raise RuntimeError("docket failed")
        return docket if docket is not None else []

    def fake_ensure_valuation(ticker, as_of_date, **_kwargs):
        if ensure_valuation_fails:
            raise RuntimeError("valuation failed")

    def fake_load_scorecard(ticker, as_of_date):
        if load_scorecard_fails:
            raise RuntimeError("load_scorecard raised")
        return scorecard if scorecard is not None else (None, None)

    def fake_assess_solvency(ticker):
        if assess_solvency_fails:
            raise RuntimeError("solvency failed")
        return solvency

    def fake_scan_filing_risks(ticker, **_kwargs):
        if scan_filing_risks_fails:
            raise RuntimeError("filing risk failed")
        return filing_risk

    def fake_compute_tensions(scorecard_arg, quality_ctx):
        if tensions_fails:
            raise RuntimeError("tensions failed")
        return tensions if tensions is not None else {}

    def fake_load_filing_context(ticker, as_of_date, quarters=0):
        if load_filing_context_fails:
            raise RuntimeError("filing context failed")
        return filing_context if filing_context is not None else FilingContext()

    def fake_reader(filing):
        return reader_result

    def fake_segment(text):
        if segment_spans is None:
            return [_Span(section_label="md_and_a", text="MDA body")]
        return segment_spans

    def fake_segment_10q(text):
        if quarterly_segment_spans is None:
            return [_Span(section_label="md_and_a", text="Quarterly MD&A body")]
        return quarterly_segment_spans

    def fake_segment_8k(text):
        if material_event_segment_spans is None:
            return [
                _Span(
                    section_label="item_502",
                    text="Chief executive officer transition announced with governance continuity plans.",
                )
            ]
        return material_event_segment_spans

    monkeypatch.setattr("app.analyst.bundle_builder.ensure_all_facts", fake_ensure_all_facts)
    monkeypatch.setattr("app.analyst.bundle_builder.collect_10k_docket", fake_collect_docket)
    monkeypatch.setattr("app.analyst.bundle_builder.ensure_valuation", fake_ensure_valuation)
    monkeypatch.setattr("app.analyst.bundle_builder._load_scorecard", fake_load_scorecard)
    monkeypatch.setattr("app.analyst.bundle_builder.assess_solvency", fake_assess_solvency)
    monkeypatch.setattr("app.analyst.bundle_builder.scan_filing_risks", fake_scan_filing_risks)
    monkeypatch.setattr(
        "app.analyst.bundle_builder._compute_tensions_from_scorecard", fake_compute_tensions
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder.load_research_filing_context", fake_load_filing_context
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder.load_current_event_context",
        lambda ticker, as_of_date: CurrentEventContext(),
    )
    monkeypatch.setattr("app.analyst.bundle_builder.read_filing_text", fake_reader)
    monkeypatch.setattr("app.analyst.bundle_builder.segment_10k_sections", fake_segment)
    monkeypatch.setattr("app.analyst.bundle_builder.segment_10q_sections", fake_segment_10q)
    monkeypatch.setattr("app.analyst.bundle_builder.segment_8k_sections", fake_segment_8k)


def test_build_bundle_happy_path_none_as_of_date(monkeypatch):
    today = date.today().isoformat()
    scorecard_dict = _sample_scorecard()
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(scorecard_dict, today),
        docket=[_dossier_filing(form_type="10-K", accession="HAPPY-01")],
        solvency=_FakeSolvency(solvency_risk="LOW"),
        filing_risk={"status": "OK"},
        tensions={"methods_agree": True, "tension_type": "NONE"},
        filing_context=FilingContext(
            documents=[
                _filing_document(
                    form_type="8-K",
                    role="material_event",
                    accession="8K-HAPPY-01",
                )
            ]
        ),
    )

    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None, years=5, quarters=0)

    assert isinstance(bundle, AnalysisEvidenceBundle)
    assert bundle.ticker == "EXAMPLE"
    assert bundle.as_of_date == today
    assert bundle.analysis_years == 5
    assert bundle.analysis_quarters == 0
    assert bundle.freshness_window_days == 90
    assert bundle.valuation.current_price == 45.50
    assert bundle.valuation.solvency_status == "LOW"
    assert bundle.valuation.filing_risk_status == "OK"
    assert bundle.valuation.methods_agree is True
    assert [filing.role for filing in bundle.filings] == ["annual", "material_event"]
    assert bundle.filings[0].accession == "HAPPY-01"
    assert bundle.filings[1].accession == "8K-HAPPY-01"
    assert "mda" in bundle.filings[0].sections_included
    assert len(bundle.recent_events) == 1
    assert bundle.recent_events[0].item_code == "5.02"
    assert bundle.prior_thesis is None
    assert "quarterly_filings_not_loaded" not in bundle.warnings
    assert "prior_thesis_not_loaded" not in bundle.warnings
    assert "solvency_gated_on_live_as_of_date" not in bundle.warnings
    assert "filing_risk_gated_on_live_as_of_date" not in bundle.warnings


def test_build_bundle_includes_quarterly_filings_when_requested(monkeypatch):
    today = date.today().isoformat()
    scorecard_dict = _sample_scorecard()
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(scorecard_dict, today),
        docket=[_dossier_filing(form_type="10-K", accession="ANNUAL-01")],
        solvency=_FakeSolvency(solvency_risk="LOW"),
        filing_risk={"status": "OK"},
        tensions={"methods_agree": True, "tension_type": "NONE"},
    )

    monkeypatch.setattr(
        "app.analyst.bundle_builder.load_research_filing_context",
        lambda ticker, as_of_date, quarters=0: FilingContext(
            documents=[
                FilingDocument(
                    ticker="EXAMPLE",
                    cik="0000000001",
                    accession="Q-01",
                    form_type="10-Q",
                    filing_date=today,
                    period_end=today,
                    role="quarterly",
                    local_path=None,
                    primary_doc_url=None,
                    html="<h1>Item 2. Management's Discussion and Analysis</h1>",
                )
            ]
        ),
    )

    monkeypatch.setattr(
        "app.analyst.bundle_builder.segment_10q_sections",
        lambda text: [_Span(section_label="md_and_a", text="Quarterly MD&A body")],
    )

    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=today, years=5, quarters=2)

    assert bundle.analysis_quarters == 2
    assert [filing.role for filing in bundle.filings] == ["annual", "quarterly"]
    assert bundle.filings[1].form_type == "10-Q"
    assert bundle.filings[1].sections_included == ["mda"]


def test_build_bundle_happy_path_explicit_today(monkeypatch):
    today = date.today().isoformat()
    scorecard_dict = _sample_scorecard()
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(scorecard_dict, today),
        docket=[_dossier_filing(form_type="10-K", accession="TODAY-01")],
        solvency=_FakeSolvency(solvency_risk="LOW"),
        filing_risk={"status": "OK"},
        tensions={"methods_agree": True, "tension_type": "NONE"},
    )
    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=today, years=5, quarters=0)
    assert bundle.as_of_date == today
    assert bundle.valuation.solvency_status == "LOW"
    assert "solvency_gated_on_live_as_of_date" not in bundle.warnings


# ---------------------------------------------------------------------------
# Live gate + temporal coherence tests
# ---------------------------------------------------------------------------


def test_build_bundle_historical_as_of_date_skips_scanners(monkeypatch):
    historical = "2025-06-15"
    scanners_called = {"solvency": False, "filing_risk": False}

    scorecard_dict = _sample_scorecard()
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(scorecard_dict, historical),
        docket=[_dossier_filing(form_type="10-K", accession="HIST-01")],
        tensions={"methods_agree": True, "tension_type": "NONE"},
    )

    # Override the scanner patches set by _monkeypatch_builder_io with traps
    def trap_solvency(ticker):
        scanners_called["solvency"] = True
        raise AssertionError("solvency scanner must NOT run for historical as_of_date")

    def trap_filing_risk(ticker):
        scanners_called["filing_risk"] = True
        raise AssertionError("filing_risk scanner must NOT run for historical as_of_date")

    monkeypatch.setattr("app.analyst.bundle_builder.assess_solvency", trap_solvency)
    monkeypatch.setattr("app.analyst.bundle_builder.scan_filing_risks", trap_filing_risk)

    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=historical)
    assert scanners_called == {"solvency": False, "filing_risk": False}
    assert bundle.as_of_date == historical
    assert bundle.valuation.solvency_status is None
    assert bundle.valuation.filing_risk_status is None
    assert "solvency_gated_on_live_as_of_date" in bundle.warnings
    assert "filing_risk_gated_on_live_as_of_date" in bundle.warnings


def test_build_bundle_temporal_coherence_historical_with_divergent_scorecard(monkeypatch):
    """Caller passes explicit historical date; scorecard resolves to a different date.

    bundle.as_of_date MUST equal the caller's original, not the scorecard's.
    """
    historical = "2025-01-15"
    scorecard_dict = _sample_scorecard()
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(scorecard_dict, "2024-12-31"),  # divergent
        docket=[_dossier_filing(form_type="10-K")],
        tensions={"methods_agree": True, "tension_type": "NONE"},
    )
    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=historical)
    assert bundle.as_of_date == historical  # frozen at the start
    assert "solvency_gated_on_live_as_of_date" in bundle.warnings


def test_build_bundle_temporal_coherence_none_with_stale_scorecard(monkeypatch):
    """Caller passes None; scorecard resolves to a date days old.

    bundle.as_of_date MUST equal today, not the scorecard's stale date
    (temporal coherence rule). Scanners MUST still run (caller intent = live).
    A stale scorecard warning MUST surface so Task 3 can reason about
    mixed-freshness inputs instead of consuming them silently.
    """
    today = date.today().isoformat()
    days_old = (date.today() - timedelta(days=2)).isoformat()
    scorecard_dict = _sample_scorecard()
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(scorecard_dict, days_old),
        docket=[_dossier_filing(form_type="10-K")],
        solvency=_FakeSolvency(solvency_risk="LOW"),
        filing_risk={"status": "OK"},
        tensions={"methods_agree": True, "tension_type": "NONE"},
    )
    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None)
    assert bundle.as_of_date == today
    assert bundle.valuation.solvency_status == "LOW"
    assert "solvency_gated_on_live_as_of_date" not in bundle.warnings
    assert "filing_risk_gated_on_live_as_of_date" not in bundle.warnings
    assert f"scorecard_stale:{days_old}" in bundle.warnings


def test_build_bundle_scorecard_resolved_date_matching_effective_emits_no_stale_warning(
    monkeypatch,
):
    """When resolved_date == effective_as_of_date, no scorecard_stale warning fires."""
    today = date.today().isoformat()
    scorecard_dict = _sample_scorecard()
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(scorecard_dict, today),
        docket=[_dossier_filing(form_type="10-K")],
        solvency=_FakeSolvency(solvency_risk="LOW"),
        filing_risk={"status": "OK"},
        tensions={"methods_agree": True, "tension_type": "NONE"},
    )
    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None)
    stale_warnings = [w for w in bundle.warnings if w.startswith("scorecard_stale:")]
    assert stale_warnings == []


# ---------------------------------------------------------------------------
# Runtime failure tests
# ---------------------------------------------------------------------------


def test_build_bundle_scorecard_missing_warning(monkeypatch):
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(None, None),
        docket=[_dossier_filing(form_type="10-K")],
    )
    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None)
    assert "scorecard_missing" in bundle.warnings
    assert bundle.valuation.current_price is None


def test_build_bundle_docket_empty_warning(monkeypatch):
    scorecard_dict = _sample_scorecard()
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(scorecard_dict, date.today().isoformat()),
        docket=[],
    )
    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None)
    assert "no_annual_filings_available" in bundle.warnings
    assert bundle.filings == []


def test_build_bundle_docket_raises_warning(monkeypatch):
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(_sample_scorecard(), date.today().isoformat()),
        collect_docket_fails=True,
    )
    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None)
    assert "filing_docket_failed" in bundle.warnings
    assert bundle.filings == []


def test_build_bundle_facts_ingestion_failed_warning(monkeypatch):
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(_sample_scorecard(), date.today().isoformat()),
        docket=[_dossier_filing()],
        ensure_all_facts_fails=True,
    )
    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None)
    assert "facts_ingestion_failed" in bundle.warnings


def test_build_bundle_solvency_raises_warning(monkeypatch):
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(_sample_scorecard(), date.today().isoformat()),
        docket=[_dossier_filing()],
        assess_solvency_fails=True,
    )
    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None)
    assert "solvency_scan_failed" in bundle.warnings
    assert bundle.valuation.solvency_status is None
    assert "solvency_gated_on_live_as_of_date" not in bundle.warnings


def test_build_bundle_filing_risk_raises_warning(monkeypatch):
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(_sample_scorecard(), date.today().isoformat()),
        docket=[_dossier_filing()],
        scan_filing_risks_fails=True,
    )
    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None)
    assert "filing_risk_scan_failed" in bundle.warnings
    assert bundle.valuation.filing_risk_status is None
    assert "filing_risk_gated_on_live_as_of_date" not in bundle.warnings


def test_build_bundle_load_scorecard_raises_does_not_propagate(monkeypatch):
    """_load_scorecard does DB + json.loads with no internal guard.

    A DB exception or JSON parse error must not escape the builder —
    the contract says the only intended raise path is malformed as_of_date.
    """
    _monkeypatch_builder_io(
        monkeypatch,
        docket=[_dossier_filing()],
        solvency=_FakeSolvency(solvency_risk="LOW"),
        filing_risk={"status": "OK"},
        tensions={"methods_agree": True, "tension_type": "NONE"},
        load_scorecard_fails=True,
    )
    # Must not raise
    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None)
    assert "scorecard_load_failed" in bundle.warnings
    # When the load raises, scorecard_missing must NOT also fire — they are
    # disjoint failure modes. The builder distinguishes "no row found" from
    # "load raised".
    assert "scorecard_missing" not in bundle.warnings
    # Downstream valuation fields degrade to None because there's no scorecard
    assert bundle.valuation.current_price is None
    assert bundle.valuation.dcf_base is None
    # Other data still populates from its own sources
    assert len(bundle.filings) == 1
    assert bundle.valuation.solvency_status == "LOW"


def test_build_bundle_load_scorecard_raises_directly_via_monkeypatch(monkeypatch):
    """Belt-and-suspenders: patch _load_scorecard directly to raise the
    specific exception type (RuntimeError with known message) and assert
    the builder returns instead of propagating."""
    scorecard_dict = _sample_scorecard()
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(scorecard_dict, date.today().isoformat()),
        docket=[_dossier_filing()],
        solvency=_FakeSolvency(solvency_risk="LOW"),
        filing_risk={"status": "OK"},
        tensions={"methods_agree": True, "tension_type": "NONE"},
    )

    def boom(ticker, as_of_date):
        raise RuntimeError("db boom")

    monkeypatch.setattr("app.analyst.bundle_builder._load_scorecard", boom)

    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None)
    assert "scorecard_load_failed" in bundle.warnings
    assert "scorecard_missing" not in bundle.warnings
    assert bundle.valuation.current_price is None


def test_build_bundle_ensure_valuation_raises_warning(monkeypatch):
    """The helper has had an ensure_valuation_fails switch from day one,
    but the previous test matrix never exercised it. Cover it now."""
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(_sample_scorecard(), date.today().isoformat()),
        docket=[_dossier_filing()],
        solvency=_FakeSolvency(solvency_risk="LOW"),
        filing_risk={"status": "OK"},
        tensions={"methods_agree": True, "tension_type": "NONE"},
        ensure_valuation_fails=True,
    )
    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None)
    assert "valuation_failed" in bundle.warnings
    # Scorecard is still returned by the monkeypatched loader, so downstream
    # mapping still populates. The warning is the only runtime signal.
    assert bundle.valuation.current_price == 45.50


def test_build_bundle_tension_analysis_raises_warning(monkeypatch):
    """Same deal — tensions_fails switch existed but was untested."""
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(_sample_scorecard(), date.today().isoformat()),
        docket=[_dossier_filing()],
        solvency=_FakeSolvency(solvency_risk="LOW"),
        filing_risk={"status": "OK"},
        tensions_fails=True,
    )
    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None)
    assert "tension_analysis_failed" in bundle.warnings
    # tensions dict fell back to empty, so methods_agree / tension_type are None
    assert bundle.valuation.methods_agree is None
    assert bundle.valuation.tension_type is None
    # Other valuation fields still populate
    assert bundle.valuation.current_price == 45.50


# ---------------------------------------------------------------------------
# Filing-context / material-event coverage
# ---------------------------------------------------------------------------


def test_build_bundle_filing_context_failure_warns(monkeypatch):
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(_sample_scorecard(), date.today().isoformat()),
        docket=[_dossier_filing()],
        load_filing_context_fails=True,
    )
    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None)
    assert "filing_context_failed" in bundle.warnings
    assert bundle.recent_events == []


def test_build_bundle_material_event_context_populates_recent_events(monkeypatch):
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(_sample_scorecard(), date.today().isoformat()),
        docket=[_dossier_filing()],
        filing_context=FilingContext(
            documents=[
                _filing_document(
                    form_type="8-K",
                    role="material_event",
                    accession="8K-BUNDLE-01",
                    filing_date="2026-04-12",
                )
            ]
        ),
    )
    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None)
    assert any(filing.role == "material_event" for filing in bundle.filings)
    assert len(bundle.recent_events) == 1
    assert bundle.recent_events[0].source_type == "8-K"
    assert bundle.recent_events[0].accession == "8K-BUNDLE-01"
    assert bundle.recent_events[0].item_code == "5.02"
    assert bundle.recent_events[0].event_category == "leadership_governance"


def test_build_bundle_current_event_context_appends_recent_events(monkeypatch):
    _monkeypatch_builder_io(
        monkeypatch,
        scorecard=(_sample_scorecard(), date.today().isoformat()),
        docket=[_dossier_filing()],
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder.load_current_event_context",
        lambda ticker, as_of_date: CurrentEventContext(
            documents=[
                CurrentEventDocument(
                    ticker="EXAMPLE",
                    source_type="ir_press",
                    published_at="2026-04-17T10:00:00+00:00",
                    title="Guidance update",
                    source_url="https://example.com/press/guidance",
                    summary="Management raised revenue guidance.",
                    citations=[],
                )
            ]
        ),
    )

    bundle = build_analysis_evidence_bundle("EXAMPLE", as_of_date=None)

    assert len(bundle.recent_events) == 1
    assert bundle.recent_events[0].source_type == "ir_press"
    assert bundle.recent_events[0].title == "Guidance update"
    assert bundle.recent_events[0].source_url == "https://example.com/press/guidance"


# ---------------------------------------------------------------------------
# Malformed date test
# ---------------------------------------------------------------------------


def test_build_bundle_malformed_as_of_date_raises_before_io(monkeypatch):
    """Malformed as_of_date must raise ValueError before any IO."""
    io_called = {"facts": False, "docket": False}

    def trap_facts(ticker, years_back=5):
        io_called["facts"] = True
        raise AssertionError("ensure_all_facts must NOT run on malformed as_of_date")

    def trap_docket(*, ticker, as_of_date, years_back):
        io_called["docket"] = True
        raise AssertionError("collect_10k_docket must NOT run on malformed as_of_date")

    monkeypatch.setattr("app.analyst.bundle_builder.ensure_all_facts", trap_facts)
    monkeypatch.setattr("app.analyst.bundle_builder.collect_10k_docket", trap_docket)

    with pytest.raises(ValueError, match="banana"):
        build_analysis_evidence_bundle("EXAMPLE", as_of_date="banana")

    assert io_called == {"facts": False, "docket": False}


# ---------------------------------------------------------------------------
# build_analysis_evidence_bundle_from_cached_scorecard (fast path for sweeps)
# ---------------------------------------------------------------------------


def _install_trap_for_slow_ensure_chain(monkeypatch):
    """Monkeypatch ensure_all_facts, ensure_valuation, and _load_scorecard
    to raise AssertionError if called. Used to verify the fast-path function
    genuinely skips the slow ensure_* chain."""

    def trap_ensure_all_facts(*args, **kwargs):
        raise AssertionError("ensure_all_facts must NOT be called from the cached-scorecard path")

    def trap_ensure_valuation(*args, **kwargs):
        raise AssertionError("ensure_valuation must NOT be called from the cached-scorecard path")

    def trap_load_scorecard(*args, **kwargs):
        raise AssertionError("_load_scorecard must NOT be called from the cached-scorecard path")

    monkeypatch.setattr("app.analyst.bundle_builder.ensure_all_facts", trap_ensure_all_facts)
    monkeypatch.setattr("app.analyst.bundle_builder.ensure_valuation", trap_ensure_valuation)
    monkeypatch.setattr("app.analyst.bundle_builder._load_scorecard", trap_load_scorecard)


def _monkeypatch_fast_path_io(
    monkeypatch,
    *,
    docket: list | None = None,
    solvency: Any | None = None,
    filing_risk: dict | None = None,
    tensions: dict | None = None,
    filing_context: FilingContext | None = None,
    reader_result: str | None = "filing text",
    segment_spans: list | None = None,
    quarterly_segment_spans: list | None = None,
    material_event_segment_spans: list | None = None,
):
    """Subset of _monkeypatch_builder_io that only stubs the IO the fast path
    actually runs (docket, scanners, tensions, filings, filing context)."""

    def fake_collect_docket(*, ticker, as_of_date, years_back):
        return docket if docket is not None else []

    def fake_assess_solvency(ticker):
        return solvency

    def fake_scan_filing_risks(ticker, **_kwargs):
        return filing_risk

    def fake_compute_tensions(scorecard_arg, quality_ctx):
        return tensions if tensions is not None else {}

    def fake_load_filing_context(ticker, as_of_date, quarters=0):
        return filing_context if filing_context is not None else FilingContext()

    def fake_reader(filing):
        return reader_result

    def fake_segment(text):
        if segment_spans is None:
            return [_Span(section_label="md_and_a", text="MDA body")]
        return segment_spans

    def fake_segment_10q(text):
        if quarterly_segment_spans is None:
            return [_Span(section_label="md_and_a", text="Quarterly MD&A body")]
        return quarterly_segment_spans

    def fake_segment_8k(text):
        if material_event_segment_spans is None:
            return [
                _Span(
                    section_label="item_502",
                    text="Leadership change announced with transition support.",
                )
            ]
        return material_event_segment_spans

    monkeypatch.setattr("app.analyst.bundle_builder.collect_10k_docket", fake_collect_docket)
    monkeypatch.setattr("app.analyst.bundle_builder.assess_solvency", fake_assess_solvency)
    monkeypatch.setattr("app.analyst.bundle_builder.scan_filing_risks", fake_scan_filing_risks)
    monkeypatch.setattr(
        "app.analyst.bundle_builder._compute_tensions_from_scorecard", fake_compute_tensions
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder.load_research_filing_context", fake_load_filing_context
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder.load_current_event_context",
        lambda ticker, as_of_date: CurrentEventContext(),
    )
    monkeypatch.setattr("app.analyst.bundle_builder.read_filing_text", fake_reader)
    monkeypatch.setattr("app.analyst.bundle_builder.segment_10k_sections", fake_segment)
    monkeypatch.setattr("app.analyst.bundle_builder.segment_10q_sections", fake_segment_10q)
    monkeypatch.setattr("app.analyst.bundle_builder.segment_8k_sections", fake_segment_8k)


def test_live_paid_filing_risk_without_packet_fails_before_scanner(
    monkeypatch,
):
    from app.analyst.bundle_builder import (
        build_analysis_evidence_bundle_from_cached_scorecard,
    )
    from app.autonomous.financial_integrity import InvalidFinancialInputError

    class _EnabledProvider:
        provider_name = "openai"

        def enabled(self):
            return True

    _install_trap_for_slow_ensure_chain(monkeypatch)
    _monkeypatch_fast_path_io(monkeypatch)
    scanner_calls = 0

    def forbidden_scan(*_args, **_kwargs):
        nonlocal scanner_calls
        scanner_calls += 1
        raise AssertionError("missing packet must suppress filing-risk scanner")

    monkeypatch.setattr(
        "app.analyst.bundle_builder.get_llm_provider",
        lambda: _EnabledProvider(),
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder.scan_filing_risks",
        forbidden_scan,
    )

    with pytest.raises(InvalidFinancialInputError):
        build_analysis_evidence_bundle_from_cached_scorecard(
            ticker="EXAMPLE",
            scorecard=_sample_scorecard(),
            scorecard_as_of_date=date.today().isoformat(),
        )

    assert scanner_calls == 0


def test_live_paid_filing_risk_rejects_different_ticker_packet(
    monkeypatch,
):
    from app.analyst.bundle_builder import (
        build_analysis_evidence_bundle_from_cached_scorecard,
    )
    from app.autonomous.financial_integrity import InvalidFinancialInputError

    class _EnabledProvider:
        provider_name = "openai"

        def enabled(self):
            return True

    _install_trap_for_slow_ensure_chain(monkeypatch)
    _monkeypatch_fast_path_io(monkeypatch)
    scanner_calls = 0

    def forbidden_scan(*_args, **_kwargs):
        nonlocal scanner_calls
        scanner_calls += 1
        raise AssertionError("wrong-ticker packet must suppress filing-risk scanner")

    monkeypatch.setattr(
        "app.analyst.bundle_builder.get_llm_provider",
        lambda: _EnabledProvider(),
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder.scan_filing_risks",
        forbidden_scan,
    )

    with pytest.raises(InvalidFinancialInputError):
        build_analysis_evidence_bundle_from_cached_scorecard(
            ticker="EXAMPLE",
            scorecard=_sample_scorecard(),
            scorecard_as_of_date=date.today().isoformat(),
            financial_packet=_canonical_financial_packet("OTHER"),
        )

    assert scanner_calls == 0


def test_live_paid_filing_risk_receives_canonical_parent_packet(
    monkeypatch,
):
    from app.analyst.bundle_builder import (
        build_analysis_evidence_bundle_from_cached_scorecard,
    )
    from app.autonomous.financial_integrity import (
        require_financial_integrity_scope,
    )

    class _EnabledProvider:
        provider_name = "openai"

        def enabled(self):
            return True

    _install_trap_for_slow_ensure_chain(monkeypatch)
    _monkeypatch_fast_path_io(monkeypatch)
    captured: dict[str, Any] = {}

    def validating_scan(ticker, **kwargs):
        captured["ticker"] = ticker
        captured.update(kwargs)
        result = require_financial_integrity_scope(kwargs["integrity_scope"])
        captured["scope_fingerprint"] = result.scope_fingerprint
        return {"status": "OK"}

    monkeypatch.setattr(
        "app.analyst.bundle_builder.get_llm_provider",
        lambda: _EnabledProvider(),
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder.scan_filing_risks",
        validating_scan,
    )
    packet = _canonical_financial_packet()

    bundle = build_analysis_evidence_bundle_from_cached_scorecard(
        ticker="EXAMPLE",
        scorecard=_sample_scorecard(),
        scorecard_as_of_date=date.today().isoformat(),
        financial_packet=packet,
    )

    assert bundle.valuation.filing_risk_status == "OK"
    assert captured["ticker"] == "EXAMPLE"
    assert captured["use_llm"] is True
    assert captured["as_of_date"] == date.today().isoformat()
    assert captured["allow_network_materialization"] is False
    assert captured["integrity_scope"].packets[0]["quote_snapshot_id"] == (packet.quote_snapshot_id)
    assert captured["scope_fingerprint"]


def test_filing_risk_integrity_failure_propagates_before_downstream_analysis(
    monkeypatch,
):
    from app.analyst.bundle_builder import (
        build_analysis_evidence_bundle_from_cached_scorecard,
    )
    from app.autonomous.financial_integrity import (
        FinancialIntegrityScope,
        InvalidFinancialInputError,
        require_financial_integrity_scope,
    )

    class _EnabledProvider:
        provider_name = "openai"

        def enabled(self):
            return True

    _install_trap_for_slow_ensure_chain(monkeypatch)
    _monkeypatch_fast_path_io(monkeypatch)
    downstream_calls = 0

    def invalid_scan(*_args, **_kwargs):
        require_financial_integrity_scope(
            FinancialIntegrityScope(
                context="bundle_filing_risk_invalid",
                run_as_of_date=date.today().isoformat(),
                packets=(),
            )
        )
        raise AssertionError("empty scope unexpectedly passed")

    def forbidden_tensions(*_args, **_kwargs):
        nonlocal downstream_calls
        downstream_calls += 1
        raise AssertionError("integrity failure must stop downstream analysis")

    monkeypatch.setattr(
        "app.analyst.bundle_builder.get_llm_provider",
        lambda: _EnabledProvider(),
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder.scan_filing_risks",
        invalid_scan,
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder._compute_tensions_from_scorecard",
        forbidden_tensions,
    )

    with pytest.raises(InvalidFinancialInputError):
        build_analysis_evidence_bundle_from_cached_scorecard(
            ticker="EXAMPLE",
            scorecard=_sample_scorecard(),
            scorecard_as_of_date=date.today().isoformat(),
            financial_packet=_canonical_financial_packet(),
        )

    assert downstream_calls == 0


def test_filing_risk_budget_exhaustion_propagates_before_downstream_analysis(
    monkeypatch,
):
    from app.analyst.bundle_builder import (
        build_analysis_evidence_bundle_from_cached_scorecard,
    )
    from app.llm.providers.retry_guard import LLMCostBudgetExceeded

    class _EnabledProvider:
        provider_name = "openai"

        def enabled(self):
            return True

    _install_trap_for_slow_ensure_chain(monkeypatch)
    _monkeypatch_fast_path_io(monkeypatch)
    downstream_calls = 0

    def exhausted_scan(*_args, **_kwargs):
        raise LLMCostBudgetExceeded("nested filing-risk budget exhausted")

    def forbidden_tensions(*_args, **_kwargs):
        nonlocal downstream_calls
        downstream_calls += 1
        raise AssertionError("budget exhaustion must stop downstream analysis")

    monkeypatch.setattr(
        "app.analyst.bundle_builder.get_llm_provider",
        lambda: _EnabledProvider(),
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder.scan_filing_risks",
        exhausted_scan,
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder._compute_tensions_from_scorecard",
        forbidden_tensions,
    )

    with pytest.raises(LLMCostBudgetExceeded):
        build_analysis_evidence_bundle_from_cached_scorecard(
            ticker="EXAMPLE",
            scorecard=_sample_scorecard(),
            scorecard_as_of_date=date.today().isoformat(),
            financial_packet=_canonical_financial_packet(),
        )

    assert downstream_calls == 0


def test_provider_disabled_bundle_keeps_deterministic_no_packet_lane(
    monkeypatch,
):
    from app.analyst.bundle_builder import (
        build_analysis_evidence_bundle_from_cached_scorecard,
    )

    class _DisabledProvider:
        provider_name = "disabled"

        def enabled(self):
            return False

    _install_trap_for_slow_ensure_chain(monkeypatch)
    _monkeypatch_fast_path_io(monkeypatch)
    scanner_calls = 0

    def deterministic_scan(ticker, **kwargs):
        nonlocal scanner_calls
        scanner_calls += 1
        assert ticker == "EXAMPLE"
        assert kwargs["use_llm"] is False
        assert kwargs["allow_network_materialization"] is False
        assert "integrity_scope" not in kwargs
        return {"status": "KEYWORD_FALLBACK"}

    monkeypatch.setattr(
        "app.analyst.bundle_builder.get_llm_provider",
        lambda: _DisabledProvider(),
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder.scan_filing_risks",
        deterministic_scan,
    )

    bundle = build_analysis_evidence_bundle_from_cached_scorecard(
        ticker="EXAMPLE",
        scorecard=_sample_scorecard(),
        scorecard_as_of_date=date.today().isoformat(),
    )

    assert scanner_calls == 1
    assert bundle.valuation.filing_risk_status == "KEYWORD_FALLBACK"


def test_from_cached_skips_ensure_all_facts_and_ensure_valuation(monkeypatch):
    """Trap ensure_all_facts, ensure_valuation, and _load_scorecard. Fast path
    must not call any of them."""
    from app.analyst.bundle_builder import (
        build_analysis_evidence_bundle_from_cached_scorecard,
    )

    _install_trap_for_slow_ensure_chain(monkeypatch)
    _monkeypatch_fast_path_io(
        monkeypatch,
        docket=[_dossier_filing()],
        solvency=_FakeSolvency(solvency_risk="LOW"),
        filing_risk={"status": "OK"},
        tensions={"methods_agree": True, "tension_type": "NONE"},
    )

    scorecard = _sample_scorecard()
    bundle = build_analysis_evidence_bundle_from_cached_scorecard(
        ticker="EXAMPLE",
        scorecard=scorecard,
        scorecard_as_of_date=date.today().isoformat(),
    )

    assert isinstance(bundle, AnalysisEvidenceBundle)
    assert bundle.ticker == "EXAMPLE"
    assert bundle.as_of_date == date.today().isoformat()


def test_from_cached_populates_valuation_from_provided_scorecard(monkeypatch):
    """The returned ValuationSnapshot should come from the injected scorecard."""
    from app.analyst.bundle_builder import (
        build_analysis_evidence_bundle_from_cached_scorecard,
    )

    _install_trap_for_slow_ensure_chain(monkeypatch)
    _monkeypatch_fast_path_io(
        monkeypatch,
        docket=[_dossier_filing()],
        solvency=_FakeSolvency(solvency_risk="LOW"),
        filing_risk={"status": "OK"},
        tensions={"methods_agree": True, "tension_type": "NONE"},
    )

    scorecard = _sample_scorecard()
    bundle = build_analysis_evidence_bundle_from_cached_scorecard(
        ticker="EXAMPLE",
        scorecard=scorecard,
        scorecard_as_of_date=date.today().isoformat(),
    )

    # Valuation fields should match the scorecard we injected (_sample_scorecard
    # has current_price=45.50, dcf_base=62.10, epv_adjusted=48.75)
    assert bundle.valuation.current_price == 45.50
    assert bundle.valuation.dcf_base == 62.10
    assert bundle.valuation.epv_adjusted == 48.75
    assert bundle.valuation.solvency_status == "LOW"
    assert bundle.valuation.filing_risk_status == "OK"
    assert bundle.valuation.methods_agree is True


def test_from_cached_still_populates_filings(monkeypatch):
    """The fast path must still read filings from the dossier (collect_10k_docket
    is NOT on the slow-chain trap list)."""
    from app.analyst.bundle_builder import (
        build_analysis_evidence_bundle_from_cached_scorecard,
    )

    _install_trap_for_slow_ensure_chain(monkeypatch)
    _monkeypatch_fast_path_io(
        monkeypatch,
        docket=[_dossier_filing(form_type="10-K", accession="FAST-01")],
        solvency=_FakeSolvency(solvency_risk="LOW"),
        filing_risk={"status": "OK"},
        tensions={"methods_agree": True, "tension_type": "NONE"},
    )

    bundle = build_analysis_evidence_bundle_from_cached_scorecard(
        ticker="EXAMPLE",
        scorecard=_sample_scorecard(),
        scorecard_as_of_date=date.today().isoformat(),
    )

    assert len(bundle.filings) == 1
    assert bundle.filings[0].accession == "FAST-01"
    assert bundle.filings[0].role == "annual"


def test_from_cached_stale_warning_when_scorecard_date_differs(monkeypatch):
    """When scorecard_as_of_date != effective date, emit scorecard_stale warning."""
    from app.analyst.bundle_builder import (
        build_analysis_evidence_bundle_from_cached_scorecard,
    )

    _install_trap_for_slow_ensure_chain(monkeypatch)
    _monkeypatch_fast_path_io(
        monkeypatch,
        docket=[_dossier_filing()],
        solvency=_FakeSolvency(solvency_risk="LOW"),
        filing_risk={"status": "OK"},
    )

    stale_date = (date.today() - timedelta(days=5)).isoformat()
    bundle = build_analysis_evidence_bundle_from_cached_scorecard(
        ticker="EXAMPLE",
        scorecard=_sample_scorecard(),
        scorecard_as_of_date=stale_date,
    )

    assert f"scorecard_stale:{stale_date}" in bundle.warnings


def test_from_cached_no_stale_warning_when_dates_match(monkeypatch):
    from app.analyst.bundle_builder import (
        build_analysis_evidence_bundle_from_cached_scorecard,
    )

    _install_trap_for_slow_ensure_chain(monkeypatch)
    _monkeypatch_fast_path_io(
        monkeypatch,
        docket=[_dossier_filing()],
        solvency=_FakeSolvency(solvency_risk="LOW"),
        filing_risk={"status": "OK"},
    )

    today = date.today().isoformat()
    bundle = build_analysis_evidence_bundle_from_cached_scorecard(
        ticker="EXAMPLE",
        scorecard=_sample_scorecard(),
        scorecard_as_of_date=today,
    )

    stale = [w for w in bundle.warnings if w.startswith("scorecard_stale:")]
    assert stale == []


def test_from_cached_no_stale_warning_when_scorecard_date_is_none(monkeypatch):
    """When scorecard_as_of_date is None, staleness check is skipped entirely."""
    from app.analyst.bundle_builder import (
        build_analysis_evidence_bundle_from_cached_scorecard,
    )

    _install_trap_for_slow_ensure_chain(monkeypatch)
    _monkeypatch_fast_path_io(
        monkeypatch,
        docket=[_dossier_filing()],
        solvency=_FakeSolvency(solvency_risk="LOW"),
        filing_risk={"status": "OK"},
    )

    bundle = build_analysis_evidence_bundle_from_cached_scorecard(
        ticker="EXAMPLE",
        scorecard=_sample_scorecard(),
        scorecard_as_of_date=None,
    )

    stale = [w for w in bundle.warnings if w.startswith("scorecard_stale:")]
    assert stale == []


def test_from_cached_historical_as_of_still_gates_scanners(monkeypatch):
    """Same live-vs-historical gate as the main function."""
    from app.analyst.bundle_builder import (
        build_analysis_evidence_bundle_from_cached_scorecard,
    )

    _install_trap_for_slow_ensure_chain(monkeypatch)
    _monkeypatch_fast_path_io(
        monkeypatch,
        docket=[_dossier_filing()],
    )

    # Trap scanners — they must NOT be called for a historical as_of_date
    def trap_solvency(ticker):
        raise AssertionError("assess_solvency must NOT run for historical as_of_date")

    def trap_filing_risk(ticker):
        raise AssertionError("scan_filing_risks must NOT run for historical as_of_date")

    monkeypatch.setattr("app.analyst.bundle_builder.assess_solvency", trap_solvency)
    monkeypatch.setattr("app.analyst.bundle_builder.scan_filing_risks", trap_filing_risk)

    bundle = build_analysis_evidence_bundle_from_cached_scorecard(
        ticker="EXAMPLE",
        scorecard=_sample_scorecard(),
        scorecard_as_of_date="2025-06-15",
        as_of_date="2025-06-15",
    )

    assert bundle.as_of_date == "2025-06-15"
    assert "solvency_gated_on_live_as_of_date" in bundle.warnings
    assert "filing_risk_gated_on_live_as_of_date" in bundle.warnings
    assert bundle.valuation.solvency_status is None
    assert bundle.valuation.filing_risk_status is None


def test_from_cached_malformed_as_of_date_raises(monkeypatch):
    from app.analyst.bundle_builder import (
        build_analysis_evidence_bundle_from_cached_scorecard,
    )

    _install_trap_for_slow_ensure_chain(monkeypatch)
    _monkeypatch_fast_path_io(monkeypatch)

    with pytest.raises(ValueError, match="banana"):
        build_analysis_evidence_bundle_from_cached_scorecard(
            ticker="EXAMPLE",
            scorecard=_sample_scorecard(),
            as_of_date="banana",
        )


# ---------------------------------------------------------------------------
# Import-safety regression
# ---------------------------------------------------------------------------


@pytest.mark.subprocess
def test_app_analyst_package_root_is_pure_import():
    """Importing app.analyst for the contract layer must not pull in
    app.config, app.db, or app.research adapters.

    The bundle_builder module transitively imports
    app.research.adapters.base, which calls get_config() at module scope.
    That side effect is acceptable for callers that explicitly import the
    builder, but app.analyst's package root must stay pure so that
    'from app.analyst import AnalysisEvidenceBundle' remains a no-I/O
    contract import.

    Regression test for: code review round 4 finding that Task 2's
    __init__ export regressed Task 1's import-safety boundary.
    """
    import subprocess
    import sys

    # Run in a subprocess so we get a clean module cache
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import app.analyst; "
                "mods = set(sys.modules); "
                "concerning = [m for m in mods if m.startswith(('app.config', 'app.db', 'app.llm', 'app.research', 'app.valuation', 'app.dossier', 'app.ingest', 'app.alpha'))]; "
                "assert not concerning, f'pure import pulled in {len(concerning)} concerning modules: {sorted(concerning)[:5]}'; "
                "print('pure')"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"app.analyst package root is not import-safe:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "pure" in result.stdout


def test_builder_importable_from_its_own_module():
    """The builder is still importable via its own module path."""
    from app.analyst.bundle_builder import build_analysis_evidence_bundle as explicit

    assert explicit is build_analysis_evidence_bundle


def test_builder_not_exposed_at_package_root():
    """Package root must not re-export the builder (it would regress import safety)."""
    import app.analyst as analyst_pkg

    assert not hasattr(analyst_pkg, "build_analysis_evidence_bundle")
    assert "build_analysis_evidence_bundle" not in analyst_pkg.__all__
