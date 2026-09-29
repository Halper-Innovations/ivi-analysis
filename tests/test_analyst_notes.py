"""Tests for LLM-first analyst notes (Stage A)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.research.current_event_context import CurrentEventContext, CurrentEventDocument
from app.research.filing_context import FilingContext, FilingDocument
from app.research.analyst_notes import (
    AnalystCitation,
    AnalystNote,
    AnalystNotes,
    validate_citations,
    extract_curated_sections,
    extract_curated_sections_from_filing_context,
    generate_analyst_notes,
    ANALYST_NOTES_SCHEMA,
)
from app.research.evidence_searcher import FilingBlock


@pytest.fixture(autouse=True)
def _stub_financial_authorization(monkeypatch):
    """Analyst-note tests use synthetic scorecards without quote scope."""

    monkeypatch.setattr(
        "app.research.analyst_notes.require_financial_integrity_scope",
        lambda *_args, **_kwargs: None,
    )


class TestDataclasses:
    def test_analyst_citation_construction(self):
        c = AnalystCitation(section="mda", excerpt="Revenue grew 12%", block_id="mda_p3")
        assert c.section == "mda"
        assert c.excerpt == "Revenue grew 12%"
        assert c.block_id == "mda_p3"

    def test_analyst_citation_no_block_id(self):
        c = AnalystCitation(section="risk_factors", excerpt="Customer concentration", block_id=None)
        assert c.block_id is None

    def test_analyst_note_construction(self):
        citation = AnalystCitation(section="mda", excerpt="test excerpt", block_id="mda_p1")
        note = AnalystNote(
            category="RISK",
            claim="Customer concentration above 30%",
            direction="BEARISH",
            severity="HIGH",
            citations=[citation],
            suggested_adjustment="Growth haircut 2-3pp",
            validation_status="VERIFIED",
        )
        assert note.category == "RISK"
        assert note.direction == "BEARISH"
        assert note.validation_status == "VERIFIED"
        assert len(note.citations) == 1

    def test_analyst_notes_container(self):
        notes = AnalystNotes(
            ticker="ACME",
            positives=[],
            risks=[],
            surprises=[],
            adjustment_triggers=[],
            overall_assessment="Solid business with moderate risks.",
            filing_sections_read=["mda", "risk_factors", "fin_notes"],
        )
        assert notes.ticker == "ACME"
        assert notes.filing_sections_read == ["mda", "risk_factors", "fin_notes"]


class TestConfig:
    def test_analyst_notes_default_disabled(self):
        from app.config import get_config

        get_config.cache_clear()
        cfg = get_config()
        assert cfg.analyst_notes_enabled is False

    def test_analyst_notes_enabled_via_env(self, monkeypatch):
        from app.config import get_config

        monkeypatch.setenv("VOE_ANALYST_NOTES", "enabled")
        get_config.cache_clear()
        cfg = get_config()
        assert cfg.analyst_notes_enabled is True
        get_config.cache_clear()

    def test_analyst_notes_disabled_via_env(self, monkeypatch):
        from app.config import get_config

        monkeypatch.setenv("VOE_ANALYST_NOTES", "disabled")
        get_config.cache_clear()
        cfg = get_config()
        assert cfg.analyst_notes_enabled is False
        get_config.cache_clear()


def _make_blocks():
    """Build a small set of filing blocks for validation tests."""
    return [
        FilingBlock(
            block_id="mda_p0",
            section="mda",
            ordinal=0,
            text="Revenue increased 12% year over year driven by strong demand in enterprise segment.",
        ),
        FilingBlock(
            block_id="mda_p1",
            section="mda",
            ordinal=1,
            text="Operating margins expanded from 18% to 22% due to cost optimization initiatives.",
        ),
        FilingBlock(
            block_id="risk_factors_p0",
            section="risk_factors",
            ordinal=0,
            text="We depend on a limited number of customers for a significant portion of our revenue.",
        ),
        FilingBlock(
            block_id="fin_notes_p0",
            section="fin_notes",
            ordinal=0,
            text="Long-term debt matures in 2028 with an outstanding balance of $450 million.",
        ),
    ]


class TestCitationValidation:
    def test_block_level_match_verified(self):
        blocks = _make_blocks()
        citation = AnalystCitation(
            section="mda",
            excerpt="Revenue increased 12% year over year driven by strong demand",
            block_id="mda_p0",
        )
        note = AnalystNote(
            category="POSITIVE",
            claim="Strong revenue growth",
            direction="BULLISH",
            severity="HIGH",
            citations=[citation],
            suggested_adjustment=None,
            validation_status="UNVERIFIED",
        )
        result = validate_citations([note], blocks)
        assert result[0].validation_status == "VERIFIED"
        assert result[0].citations[0].block_id == "mda_p0"

    def test_block_id_missing_falls_back_to_section(self):
        blocks = _make_blocks()
        citation = AnalystCitation(
            section="risk_factors",
            excerpt="limited number of customers for a significant portion of our revenue",
            block_id=None,
        )
        note = AnalystNote(
            category="RISK",
            claim="Customer concentration",
            direction="BEARISH",
            severity="HIGH",
            citations=[citation],
            suggested_adjustment=None,
            validation_status="UNVERIFIED",
        )
        result = validate_citations([note], blocks)
        assert result[0].validation_status == "VERIFIED"

    def test_block_id_invalid_falls_back_to_section(self):
        blocks = _make_blocks()
        citation = AnalystCitation(
            section="fin_notes",
            excerpt="Long-term debt matures in 2028 with an outstanding balance",
            block_id="fin_notes_p999",
        )
        note = AnalystNote(
            category="RISK",
            claim="Debt maturity risk",
            direction="BEARISH",
            severity="MODERATE",
            citations=[citation],
            suggested_adjustment=None,
            validation_status="UNVERIFIED",
        )
        result = validate_citations([note], blocks)
        assert result[0].validation_status == "VERIFIED"

    def test_no_match_unverified(self):
        blocks = _make_blocks()
        citation = AnalystCitation(
            section="mda",
            excerpt="The company plans to acquire three competitors in Asia-Pacific",
            block_id="mda_p0",
        )
        note = AnalystNote(
            category="SURPRISE",
            claim="Acquisition plans",
            direction="BULLISH",
            severity="HIGH",
            citations=[citation],
            suggested_adjustment=None,
            validation_status="UNVERIFIED",
        )
        result = validate_citations([note], blocks)
        assert result[0].validation_status == "UNVERIFIED"

    def test_multiple_citations_all_must_pass(self):
        blocks = _make_blocks()
        good_cite = AnalystCitation(
            section="mda",
            excerpt="Revenue increased 12% year over year",
            block_id="mda_p0",
        )
        bad_cite = AnalystCitation(
            section="mda",
            excerpt="completely fabricated hallucinated content",
            block_id="mda_p1",
        )
        note = AnalystNote(
            category="POSITIVE",
            claim="Growth story",
            direction="BULLISH",
            severity="MODERATE",
            citations=[good_cite, bad_cite],
            suggested_adjustment=None,
            validation_status="UNVERIFIED",
        )
        result = validate_citations([note], blocks)
        assert result[0].validation_status == "UNVERIFIED"

    def test_empty_notes_returns_empty(self):
        result = validate_citations([], _make_blocks())
        assert result == []

    def test_cross_block_excerpt_verified_via_section_fallback(self):
        """Excerpt spanning two adjacent blocks in same section — VERIFIED via section concatenation."""
        blocks = _make_blocks()
        # Excerpt spans content from mda_p0 and mda_p1
        citation = AnalystCitation(
            section="mda",
            excerpt="strong demand in enterprise segment Operating margins expanded from 18%",
            block_id=None,
        )
        note = AnalystNote(
            category="POSITIVE",
            claim="Growth with margin expansion",
            direction="BULLISH",
            severity="HIGH",
            citations=[citation],
            suggested_adjustment=None,
            validation_status="UNVERIFIED",
        )
        result = validate_citations([note], blocks)
        assert result[0].validation_status == "VERIFIED"


class TestCuratedSections:
    def test_extracts_mda_risk_factors_fin_notes(self):
        html = """
        <h1>Item 1. Business</h1>
        <p>We sell widgets to enterprise customers worldwide with a focus on durability.</p>
        <h1>Item 1A. Risk Factors</h1>
        <p>Customer concentration is a significant risk to our revenue stability and growth prospects.</p>
        <h1>Item 7. Management's Discussion and Analysis</h1>
        <p>Revenue grew 15% driven by enterprise expansion and new product launches in Q3 and Q4.</p>
        <h1>Item 8. Financial Statements</h1>
        <p>Notes to Consolidated Financial Statements follow the audited financial data below.</p>
        <p>Long-term debt consists of senior notes due 2028 with a principal balance of $500 million.</p>
        """
        sections, blocks = extract_curated_sections(html, "10-K")
        section_labels = set(sections)
        assert "mda" in section_labels
        assert "risk_factors" in section_labels
        assert "fin_notes" in section_labels
        assert "business" not in section_labels
        assert "other" not in section_labels
        assert len(blocks) > 0
        assert all(b.section in ("mda", "risk_factors", "fin_notes") for b in blocks)

    def test_missing_sections_returns_available(self):
        html = """
        <h1>Item 7. Management's Discussion and Analysis</h1>
        <p>Revenue declined 8% due to macroeconomic headwinds affecting our core enterprise segment.</p>
        """
        sections, blocks = extract_curated_sections(html, "10-K")
        assert "mda" in sections
        assert "risk_factors" not in sections
        assert len(blocks) >= 1

    def test_empty_html_returns_empty(self):
        sections, blocks = extract_curated_sections("", "10-K")
        assert sections == []
        assert blocks == []

    def test_none_html_returns_empty(self):
        sections, blocks = extract_curated_sections(None, "10-K")
        assert sections == []
        assert blocks == []

    def test_extracts_material_event_sections_from_filing_context(self):
        context = FilingContext(
            documents=[
                FilingDocument(
                    ticker="TEST",
                    cik="0000000001",
                    accession="0000000001-26-000101",
                    form_type="8-K",
                    filing_date="2026-04-10",
                    period_end=None,
                    role="material_event",
                    local_path=None,
                    primary_doc_url=None,
                    html=(
                        "<h1>Item 5.02 Departure of Directors or Certain Officers; Election of Directors; "
                        "Appointment of Certain Officers; Compensatory Arrangements of Certain Officers.</h1>"
                        "<p>"
                        + (
                            "The board appointed a new chief executive officer and transition plan. "
                            * 6
                        )
                        + "</p>"
                    ),
                )
            ]
        )

        sections, blocks = extract_curated_sections_from_filing_context(context)

        assert "leadership_governance" in sections
        assert len(blocks) >= 1
        assert blocks[0].section == "leadership_governance"
        assert blocks[0].item_code == "5.02"

    def test_extracts_agreement_termination_material_event_sections(self):
        context = FilingContext(
            documents=[
                FilingDocument(
                    ticker="TEST",
                    cik="0000000001",
                    accession="0000000001-26-000102",
                    form_type="8-K",
                    filing_date="2026-04-11",
                    period_end=None,
                    role="material_event",
                    local_path=None,
                    primary_doc_url=None,
                    html=(
                        "<h1>Item 1.02 Termination of a Material Definitive Agreement.</h1>"
                        "<p>"
                        + (
                            "The company terminated a material supply agreement and expects transition costs. "
                            * 6
                        )
                        + "</p>"
                    ),
                )
            ]
        )

        sections, blocks = extract_curated_sections_from_filing_context(context)

        assert "agreement_termination" in sections
        assert len(blocks) >= 1
        assert blocks[0].section == "agreement_termination"
        assert blocks[0].item_code == "1.02"

    def test_extracts_quarterly_sections(self):
        html = """
        <h1>Item 1. Financial Statements</h1>
        <p>Condensed financial statements content that is long enough to parse meaningfully.</p>
        <h1>Notes to Condensed Consolidated Financial Statements</h1>
        <p>Quarterly note disclosures and debt footnotes appear here in the quarter.</p>
        <h1>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h1>
        <p>Quarterly management discussion covers margins, revenue, and operating cash flow.</p>
        <h1>Item 1A. Risk Factors</h1>
        <p>Quarterly risk updates describe customer concentration and macro exposure.</p>
        """
        sections, blocks = extract_curated_sections(html, "10-Q")
        assert "mda" in sections
        assert "risk_factors" in sections
        assert "fin_notes" in sections
        assert all(b.section in ("mda", "risk_factors", "fin_notes") for b in blocks)


class TestGenerateAnalystNotes:
    def _mock_llm_response(self):
        return {
            "overall_assessment": "Solid company with moderate customer concentration risk.",
            "positives": [
                {
                    "category": "POSITIVE",
                    "claim": "Revenue grew 15% year over year",
                    "direction": "BULLISH",
                    "severity": "HIGH",
                    "citations": [
                        {
                            "section": "mda",
                            "excerpt": "Revenue grew 15% driven by enterprise",
                            "block_id": "mda_p0",
                        }
                    ],
                    "suggested_adjustment": None,
                }
            ],
            "risks": [
                {
                    "category": "RISK",
                    "claim": "Top 3 customers represent 45% of revenue",
                    "direction": "BEARISH",
                    "severity": "HIGH",
                    "citations": [
                        {
                            "section": "risk_factors",
                            "excerpt": "limited number of customers for a significant portion",
                            "block_id": "risk_factors_p0",
                        }
                    ],
                    "suggested_adjustment": "Growth haircut 2pp for concentration",
                }
            ],
            "surprises": [],
            "adjustment_triggers": [],
        }

    def test_returns_analyst_notes_on_success(self):
        filing_html = """
        <h1>Item 1A. Risk Factors</h1>
        <p>We depend on a limited number of customers for a significant portion of our revenue.</p>
        <h1>Item 7. Management's Discussion and Analysis</h1>
        <p>Revenue grew 15% driven by enterprise expansion and new product launches during the fiscal year.</p>
        """
        scorecard = {
            "pricing_zone_detail": {"dcf_base": 50.0, "epv_adjusted": 45.0, "current_price": 35.0}
        }

        mock_result = MagicMock()
        mock_result.json_text = json.dumps(self._mock_llm_response())
        mock_provider = MagicMock()
        mock_provider.provider_name = "openai"
        mock_provider.synthesize_json.return_value = mock_result

        with patch("app.research.analyst_notes.get_llm_provider", return_value=mock_provider):
            result = generate_analyst_notes(filing_html, "10-K", scorecard, "ACME")

        assert result is not None
        assert result.ticker == "ACME"
        assert len(result.positives) == 1
        assert len(result.risks) == 1
        assert result.positives[0].claim == "Revenue grew 15% year over year"
        assert result.risks[0].validation_status in ("VERIFIED", "UNVERIFIED")
        assert "mda" in result.filing_sections_read

    def test_rechecks_callable_financial_scope_after_paid_response(self):
        from app.autonomous.financial_integrity import (
            FinancialIntegrityScope,
            InvalidFinancialInputError,
            require_financial_integrity_scope,
        )

        with pytest.raises(InvalidFinancialInputError) as invalid:
            require_financial_integrity_scope(
                FinancialIntegrityScope(
                    context="analyst_notes_live_rebinding",
                    run_as_of_date="2026-03-25",
                    packets=(),
                )
            )
        integrity_error = invalid.value
        filing_html = """
        <h1>Item 7. Management's Discussion and Analysis</h1>
        <p>Revenue grew 15% driven by enterprise expansion and new product launches.</p>
        """
        scorecard = {
            "pricing_zone_detail": {
                "dcf_base": 50.0,
                "epv_adjusted": 45.0,
                "current_price": 35.0,
            }
        }

        class _LiveScope:
            def __init__(self) -> None:
                self.checks = 0

            def require(self, *, scenarios=None):
                self.checks += 1
                price = scenarios[0]["financial_inputs"]["scorecard"]["pricing_zone_detail"][
                    "current_price"
                ]
                if price != 35.0:
                    raise integrity_error

        class _MutatingProvider:
            provider_name = "openai"
            model = "gpt-5-mini"

            def __init__(self) -> None:
                self.calls = 0

            def synthesize_json(self, **_kwargs):
                self.calls += 1
                scorecard["pricing_zone_detail"]["current_price"] = 36.0
                return SimpleNamespace(
                    json_text=json.dumps(self_response),
                    model="gpt-5-mini",
                    usage_input_tokens=1000,
                    usage_output_tokens=300,
                    raw={},
                )

        self_response = self._mock_llm_response()
        scope = _LiveScope()
        provider = _MutatingProvider()

        with (
            patch(
                "app.research.analyst_notes.get_llm_provider",
                return_value=provider,
            ),
            pytest.raises(InvalidFinancialInputError) as raised,
        ):
            generate_analyst_notes(
                filing_html,
                "10-K",
                scorecard,
                "ACME",
                financial_integrity_scope=scope,
                financial_scenarios=lambda: ({"financial_inputs": {"scorecard": scorecard}},),
            )

        assert raised.value is integrity_error
        assert provider.calls == 1
        assert scope.checks == 2

    def test_paid_failure_rechecks_callable_financial_scope_before_fallback(self):
        from app.autonomous.financial_integrity import (
            FinancialIntegrityScope,
            InvalidFinancialInputError,
            require_financial_integrity_scope,
        )

        with pytest.raises(InvalidFinancialInputError) as invalid:
            require_financial_integrity_scope(
                FinancialIntegrityScope(
                    context="analyst_notes_failed_call_rebinding",
                    run_as_of_date="2026-03-25",
                    packets=(),
                )
            )
        integrity_error = invalid.value
        filing_html = """
        <h1>Item 7. Management's Discussion and Analysis</h1>
        <p>Revenue grew 15% driven by enterprise expansion and new product launches.</p>
        """
        scorecard = {
            "pricing_zone_detail": {
                "dcf_base": 50.0,
                "epv_adjusted": 45.0,
                "current_price": 35.0,
            }
        }

        class _LiveScope:
            def __init__(self) -> None:
                self.checks = 0

            def require(self, *, scenarios=None):
                self.checks += 1
                price = scenarios[0]["financial_inputs"]["scorecard"]["pricing_zone_detail"][
                    "current_price"
                ]
                if price != 35.0:
                    raise integrity_error

        class _MutatingFailedProvider:
            provider_name = "openai"
            model = "gpt-5-mini"

            def __init__(self) -> None:
                self.calls = 0

            def synthesize_json(self, **_kwargs):
                self.calls += 1
                scorecard["pricing_zone_detail"]["current_price"] = 36.0
                raise RuntimeError("provider failed after mutating live inputs")

        scope = _LiveScope()
        provider = _MutatingFailedProvider()

        with (
            patch(
                "app.research.analyst_notes.get_llm_provider",
                return_value=provider,
            ),
            pytest.raises(InvalidFinancialInputError) as raised,
        ):
            generate_analyst_notes(
                filing_html,
                "10-K",
                scorecard,
                "ACME",
                financial_integrity_scope=scope,
                financial_scenarios=lambda: ({"financial_inputs": {"scorecard": scorecard}},),
            )

        assert raised.value is integrity_error
        assert provider.calls == 1
        assert scope.checks == 2

    def test_ordinary_paid_failure_still_returns_none_after_scope_recheck(self):
        filing_html = """
        <h1>Item 7. Management's Discussion and Analysis</h1>
        <p>Revenue grew 15% driven by enterprise expansion and new product launches.</p>
        """
        scorecard = {
            "pricing_zone_detail": {
                "dcf_base": 50.0,
                "epv_adjusted": 45.0,
                "current_price": 35.0,
            }
        }

        class _LiveScope:
            def __init__(self) -> None:
                self.checks = 0

            def require(self, *, scenarios=None):
                self.checks += 1
                assert (
                    scenarios[0]["financial_inputs"]["scorecard"]["pricing_zone_detail"][
                        "current_price"
                    ]
                    == 35.0
                )

        class _FailedProvider:
            provider_name = "openai"
            model = "gpt-5-mini"

            def __init__(self) -> None:
                self.calls = 0

            def synthesize_json(self, **_kwargs):
                self.calls += 1
                raise RuntimeError("ordinary provider failure")

        scope = _LiveScope()
        provider = _FailedProvider()

        with patch(
            "app.research.analyst_notes.get_llm_provider",
            return_value=provider,
        ):
            result = generate_analyst_notes(
                filing_html,
                "10-K",
                scorecard,
                "ACME",
                financial_integrity_scope=scope,
                financial_scenarios=lambda: ({"financial_inputs": {"scorecard": scorecard}},),
            )

        assert result is None
        assert provider.calls == 1
        assert scope.checks == 2

    def test_returns_none_when_no_filing(self):
        result = generate_analyst_notes(None, "10-K", {}, "ACME")
        assert result is None

    def test_returns_none_when_provider_disabled(self):
        filing_html = "<h1>Item 7. Management's Discussion and Analysis</h1><p>Content here for testing purposes.</p>"
        mock_provider = MagicMock()
        mock_provider.provider_name = "disabled"

        with patch("app.research.analyst_notes.get_llm_provider", return_value=mock_provider):
            result = generate_analyst_notes(filing_html, "10-K", {}, "ACME")
        assert result is None

    def test_returns_none_when_no_curated_sections(self):
        filing_html = (
            "<p>Some random preamble text that does not contain any section headers at all.</p>"
        )
        mock_provider = MagicMock()
        mock_provider.provider_name = "openai"

        with patch("app.research.analyst_notes.get_llm_provider", return_value=mock_provider):
            result = generate_analyst_notes(filing_html, "10-K", {}, "ACME")
        assert result is None

    def test_schema_is_valid_json_schema(self):
        assert ANALYST_NOTES_SCHEMA["type"] == "object"
        assert "positives" in ANALYST_NOTES_SCHEMA["properties"]
        assert "risks" in ANALYST_NOTES_SCHEMA["properties"]
        assert "overall_assessment" in ANALYST_NOTES_SCHEMA["properties"]

    def test_includes_current_events_appendix_in_prompt(self):
        filing_html = """
        <h1>Item 7. Management's Discussion and Analysis</h1>
        <p>Revenue grew 15% driven by enterprise expansion and new product launches during the fiscal year.</p>
        """
        scorecard = {
            "pricing_zone_detail": {"dcf_base": 50.0, "epv_adjusted": 45.0, "current_price": 35.0}
        }
        mock_result = MagicMock()
        mock_result.json_text = json.dumps(self._mock_llm_response())
        captured_prompt: dict[str, str] = {}

        class FakeProvider:
            provider_name = "openai"

            def synthesize_json(self, *, prompt, schema, schema_name=None, **kwargs):
                captured_prompt["prompt"] = prompt
                return mock_result

        current_event_context = CurrentEventContext(
            documents=[
                CurrentEventDocument(
                    ticker="ACME",
                    source_type="ir_press",
                    published_at="2026-04-17T10:00:00+00:00",
                    title="Guidance update",
                    source_url="https://example.com/press/guidance",
                    summary="Management raised revenue guidance for the full year.",
                    citations=[],
                )
            ]
        )

        with patch("app.research.analyst_notes.get_llm_provider", return_value=FakeProvider()):
            result = generate_analyst_notes(
                filing_html,
                "10-K",
                scorecard,
                "ACME",
                current_event_context=current_event_context,
            )

        assert result is not None
        assert "Recent company-controlled current events:" in captured_prompt["prompt"]
        assert "IR_PRESS (2026-04-17T10:00:00+00:00)" in captured_prompt["prompt"]
        assert "Guidance update" in captured_prompt["prompt"]
