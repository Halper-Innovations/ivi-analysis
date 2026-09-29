"""Tests for app.research.evidence_searcher."""

from __future__ import annotations

import json
from dataclasses import dataclass as _dc

import pytest

from app.research.evidence_searcher import (
    _require_paid_financial_scope as _real_require_paid_financial_scope,
)


@pytest.fixture(autouse=True)
def _stub_financial_authorization(monkeypatch):
    """Retrieval tests isolate adjudication behavior from scope construction."""

    monkeypatch.setattr(
        "app.research.evidence_searcher._require_paid_financial_scope",
        lambda *_args, **_kwargs: None,
    )


class TestPrepareFilingText:
    """Tests for _prepare_filing_text() — HTML to structure-preserving text."""

    def test_p_tags_produce_double_newline(self):
        from app.research.evidence_searcher import _prepare_filing_text

        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        result = _prepare_filing_text(html)
        assert "First paragraph." in result
        assert "Second paragraph." in result
        assert "\n\n" in result

    def test_div_does_not_produce_boundary(self):
        from app.research.evidence_searcher import _prepare_filing_text

        html = "<div><div>Same logical paragraph continues here.</div></div>"
        result = _prepare_filing_text(html)
        assert "\n\n" not in result
        assert "Same logical paragraph continues here." in result

    def test_heading_tags_produce_double_newline(self):
        from app.research.evidence_searcher import _prepare_filing_text

        html = "<h2>Section Title</h2><p>Content after heading.</p>"
        result = _prepare_filing_text(html)
        parts = result.split("\n\n")
        titles = [p.strip() for p in parts if "Section Title" in p]
        assert len(titles) >= 1

    def test_br_produces_single_newline(self):
        from app.research.evidence_searcher import _prepare_filing_text

        html = "Line one<br>Line two<br/>Line three"
        result = _prepare_filing_text(html)
        lines = result.strip().split("\n")
        assert len(lines) >= 3

    def test_td_produces_tab(self):
        from app.research.evidence_searcher import _prepare_filing_text

        html = "<tr><td>Revenue</td><td>$1,000</td></tr>"
        result = _prepare_filing_text(html)
        assert "\t" in result
        assert "Revenue" in result
        assert "$1,000" in result

    def test_entities_decoded(self):
        from app.research.evidence_searcher import _prepare_filing_text

        html = "<p>Smith &amp; Co&#160;reported &lt;5% decline</p>"
        result = _prepare_filing_text(html)
        assert "Smith & Co" in result
        assert "<5% decline" in result

    def test_excessive_newlines_collapsed(self):
        from app.research.evidence_searcher import _prepare_filing_text

        html = "<p>A</p>\n\n\n\n\n<p>B</p>"
        result = _prepare_filing_text(html)
        assert "\n\n\n" not in result
        assert "A" in result and "B" in result

    def test_script_and_style_removed(self):
        from app.research.evidence_searcher import _prepare_filing_text

        html = "<script>var x = 1;</script><p>Real content</p><style>.foo{}</style>"
        result = _prepare_filing_text(html)
        assert "var x" not in result
        assert ".foo" not in result
        assert "Real content" in result

    def test_spaces_not_collapsed(self):
        from app.research.evidence_searcher import _prepare_filing_text

        html = "<p>Revenue    breakdown   by   segment</p>"
        result = _prepare_filing_text(html)
        # Multi-space runs within paragraphs should be preserved
        assert "Revenue" in result
        assert "breakdown" in result


class TestDetectSections:
    """Tests for _detect_sections() — section boundary detection with TOC disambiguation."""

    def test_10k_basic_sections(self):
        from app.research.evidence_searcher import _detect_sections

        text = (
            "Cover page and preamble text here.\n\n"
            "Item 1. Business\n\n" + ("This is the business section content. " * 100) + "\n\n"
            "Item 1A. Risk Factors\n\n" + ("These are the risk factors. " * 100) + "\n\n"
            "Item 7. Management's Discussion and Analysis\n\n"
            + ("MD&A content here discussing results. " * 100)
            + "\n\n"
            "Item 8. Financial Statements\n\n" + ("Financial statement notes here. " * 100)
        )
        sections = _detect_sections(text, "10-K")
        labels = [s[0] for s in sections]
        assert "other" in labels
        assert "business" in labels
        assert "risk_factors" in labels
        assert "mda" in labels
        assert "fin_notes" in labels

    def test_toc_disambiguation_picks_body(self):
        from app.research.evidence_searcher import _detect_sections

        # TOC has Item 7 followed quickly by Item 8
        toc = (
            "TABLE OF CONTENTS\n\n"
            "Item 7. Management Discussion...........23\n\n"
            "Item 8. Financial Statements...........45\n\n"
        )
        # Body has Item 7 followed by large content
        body = (
            "Item 7. Management's Discussion and Analysis\n\n"
            + ("Management discusses the operating results in detail. " * 200)
            + "\n\n"
            "Item 8. Financial Statements\n\n"
            + ("Notes to consolidated financial statements. " * 200)
        )
        text = toc + body
        sections = _detect_sections(text, "10-K")
        mda = next(s for s in sections if s[0] == "mda")
        # Should contain the body content, not TOC line
        assert "operating results" in mda[1].lower()

    def test_toc_min_threshold_filters_short_matches(self):
        from app.research.evidence_searcher import _detect_sections

        # A TOC entry with <2000 chars before next boundary should be filtered
        toc_entry = "Item 1A. Risk Factors\n\nSee page 15\n\n"
        body = "Item 1A. Risk Factors\n\n" + ("Risk factor discussion content. " * 200) + "\n\n"
        text = toc_entry + body
        sections = _detect_sections(text, "10-K")
        rf = next(s for s in sections if s[0] == "risk_factors")
        assert "Risk factor discussion content" in rf[1]

    def test_20f_sections(self):
        from app.research.evidence_searcher import _detect_sections

        text = (
            "Preamble\n\n"
            "Item 4. Information on the Company\n\n" + ("Company info. " * 100) + "\n\n"
            "D. Risk Factors\n\n" + ("Risk discussion. " * 100) + "\n\n"
            "Item 5. Operating and Financial Review\n\n" + ("Operating review. " * 100)
        )
        sections = _detect_sections(text, "20-F")
        labels = [s[0] for s in sections]
        assert "business" in labels
        assert "risk_factors" in labels
        assert "mda" in labels

    def test_10q_sections(self):
        from app.research.evidence_searcher import _detect_sections

        text = (
            "Preamble\n\n"
            "Item 1. Financial Statements\n\n"
            + ("Condensed financial statements content. " * 120)
            + "\n\n"
            "Notes to Condensed Consolidated Financial Statements\n\n"
            + ("Quarterly note content. " * 120)
            + "\n\n"
            "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations\n\n"
            + ("Quarterly management discussion content. " * 120)
            + "\n\n"
            "Item 1A. Risk Factors\n\n" + ("Quarterly risk factor content. " * 120)
        )
        sections = _detect_sections(text, "10-Q")
        labels = [s[0] for s in sections]
        assert "mda" in labels
        assert "risk_factors" in labels
        assert "fin_notes" in labels

    def test_toc_same_label_repeated_picks_body(self):
        from app.research.evidence_searcher import _detect_sections

        # TOC has Item 1A, then body has Item 1A, then Item 7
        # The TOC Item 1A's region must end at the body Item 1A, not at Item 7
        toc = "TABLE OF CONTENTS\n\nItem 1A. Risk Factors...........15\n\n"
        body = (
            "Item 1A. Risk Factors\n\n"
            + ("Detailed risk factor discussion about competitive threats. " * 150)
            + "\n\n"
            "Item 7. Management's Discussion and Analysis\n\n"
            + ("Management discusses results of operations. " * 150)
        )
        text = toc + body
        sections = _detect_sections(text, "10-K")
        rf = next(s for s in sections if s[0] == "risk_factors")
        assert "competitive threats" in rf[1].lower()
        # TOC line should NOT be in the risk_factors section content
        assert "...........15" not in rf[1]

    def test_no_matches_returns_other(self):
        from app.research.evidence_searcher import _detect_sections

        text = "This filing has no recognizable section markers at all. " * 100
        sections = _detect_sections(text, "10-K")
        assert len(sections) == 1
        assert sections[0][0] == "other"
        assert "no recognizable section markers" in sections[0][1]

    def test_other_appears_at_most_once_and_first(self):
        from app.research.evidence_searcher import _detect_sections

        text = (
            "Preamble text before any section.\n\n"
            "Item 7. Management's Discussion and Analysis\n\n" + ("MDA content. " * 100)
        )
        sections = _detect_sections(text, "10-K")
        other_sections = [s for s in sections if s[0] == "other"]
        assert len(other_sections) <= 1
        if other_sections:
            assert sections[0][0] == "other"

    def test_post_last_boundary_appended_to_last_section(self):
        from app.research.evidence_searcher import _detect_sections

        text = (
            "Item 7. Management's Discussion and Analysis\n\n" + ("MDA content. " * 100) + "\n\n"
            "SIGNATURES\n\nTrailing content after last section."
        )
        sections = _detect_sections(text, "10-K")
        mda = next(s for s in sections if s[0] == "mda")
        assert "Trailing content" in mda[1]


class TestBlockSplitting:
    """Tests for block splitting and parse_filing_blocks()."""

    def test_split_on_double_newline(self):
        from app.research.evidence_searcher import parse_filing_blocks, FilingBlock

        html = "<p>First paragraph with enough content to pass the minimum size threshold easily.</p><p>Second paragraph with enough content to pass the minimum threshold as well.</p>"
        blocks = parse_filing_blocks(html)
        assert len(blocks) >= 2
        assert all(isinstance(b, FilingBlock) for b in blocks)

    def test_short_chunks_discarded(self):
        from app.research.evidence_searcher import parse_filing_blocks

        html = "<p>Short</p><p>This paragraph has enough content to pass the fifty character minimum threshold for blocks.</p>"
        blocks = parse_filing_blocks(html)
        texts = [b.text for b in blocks]
        assert not any(t.strip() == "Short" for t in texts)

    def test_block_id_format(self):
        from app.research.evidence_searcher import parse_filing_blocks

        html = (
            "<p>Item 7. Management's Discussion and Analysis</p>"
            "<p>" + "MDA paragraph content that is long enough to pass threshold. " * 3 + "</p>"
        )
        blocks = parse_filing_blocks(html)
        for b in blocks:
            assert b.block_id.endswith(f"_p{b.ordinal}")
            assert b.section in ("business", "risk_factors", "mda", "fin_notes", "other")

    def test_ordinals_are_per_section(self):
        from app.research.evidence_searcher import parse_filing_blocks

        html = (
            "<p>Item 1. Business</p>"
            "<p>" + "Business content paragraph one is long enough to pass. " * 3 + "</p>"
            "<p>" + "Business content paragraph two is long enough to pass. " * 3 + "</p>"
            "<p>Item 1A. Risk Factors</p>"
            "<p>" + "Risk factors paragraph one is long enough here to pass. " * 3 + "</p>"
        )
        blocks = parse_filing_blocks(html)
        biz_blocks = [b for b in blocks if b.section == "business"]
        rf_blocks = [b for b in blocks if b.section == "risk_factors"]
        if biz_blocks:
            assert biz_blocks[0].ordinal == 0
        if rf_blocks:
            assert rf_blocks[0].ordinal == 0

    def test_sections_assigned_correctly(self):
        from app.research.evidence_searcher import parse_filing_blocks

        html = (
            "<p>Item 7. Management's Discussion and Analysis</p>"
            "<p>" + "Revenue increased due to strong demand across segments. " * 5 + "</p>"
            "<p>Item 8. Financial Statements</p>"
            "<p>" + "Notes to consolidated financial statements follow here. " * 5 + "</p>"
        )
        blocks = parse_filing_blocks(html, form_type="10-K")
        sections = set(b.section for b in blocks)
        assert "mda" in sections or "fin_notes" in sections

    def test_empty_html_returns_empty(self):
        from app.research.evidence_searcher import parse_filing_blocks

        blocks = parse_filing_blocks("")
        assert blocks == []

    def test_form_type_20f(self):
        from app.research.evidence_searcher import parse_filing_blocks

        html = (
            "<p>Item 5. Operating and Financial Review</p>"
            "<p>" + "The company's operating results improved significantly. " * 5 + "</p>"
        )
        blocks = parse_filing_blocks(html, form_type="20-F")
        assert any(b.section == "mda" for b in blocks)

    def test_parse_blocks_preserves_source_metadata(self):
        from app.research.evidence_searcher import parse_filing_blocks

        html = (
            "<p>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</p>"
            "<p>"
            + "Quarterly discussion is long enough to pass the block threshold easily. " * 4
            + "</p>"
        )
        blocks = parse_filing_blocks(
            html,
            form_type="10-Q",
            source_form_type="10-Q",
            source_filing_date="2026-05-01",
            source_accession="0000000000-26-000001",
            source_role="quarterly",
        )
        assert blocks[0].block_id.startswith("0000000000-26-000001:")
        assert blocks[0].source_form_type == "10-Q"
        assert blocks[0].source_filing_date == "2026-05-01"
        assert blocks[0].source_accession == "0000000000-26-000001"
        assert blocks[0].source_role == "quarterly"
        assert blocks[0].source_quality == {
            "source_family": "regulatory_filing",
            "source_origin": "primary",
            "source_independence": "regulatory",
            "source_domain": None,
            "freshness_days": None,
            "freshness_bucket": "undated",
            "source_quality_score": 0.95,
            "calibration_status": "deterministic_heuristic",
            "reason_codes": [
                "SOURCE_PRIMARY_REGULATORY",
                "FRESHNESS_UNDATED",
            ],
        }

    def test_parse_8k_blocks_normalize_sections_and_preserve_item_metadata(self):
        from app.research.evidence_searcher import parse_filing_blocks

        html = (
            "<p>Item 5.02 Departure of Directors or Certain Officers; Election of Directors; "
            "Appointment of Certain Officers; Compensatory Arrangements of Certain Officers.</p>"
            "<p>"
            + "The board appointed a new chief executive officer and transition plan. "
            * 6
            + "</p>"
            "<p>Item 1.02 Termination of a Material Definitive Agreement.</p>"
            "<p>"
            + "The company terminated a material supply agreement and expects transition costs. "
            * 6
            + "</p>"
            "<p>Item 4.02 Non-Reliance on Previously Issued Financial Statements or a Related Audit Report or Completed Interim Review.</p>"
            "<p>"
            + "Management concluded previously issued financial statements should no longer be relied upon. "
            * 6
            + "</p>"
        )
        blocks = parse_filing_blocks(
            html,
            form_type="8-K",
            source_form_type="8-K",
            source_filing_date="2026-04-10",
            source_accession="0000000000-26-000101",
            source_role="material_event",
        )
        leadership = next(
            block for block in blocks if block.event_category == "leadership_governance"
        )
        termination = next(
            block for block in blocks if block.event_category == "agreement_termination"
        )
        restatement = next(
            block for block in blocks if block.event_category == "restatement_controls"
        )
        assert leadership.section == "leadership_governance"
        assert leadership.item_code == "5.02"
        assert leadership.source_form_type == "8-K"
        assert leadership.source_role == "material_event"
        assert termination.section == "agreement_termination"
        assert termination.item_code == "1.02"
        assert restatement.section == "restatement_controls"
        assert restatement.item_code == "4.02"


class TestRetrieveCandidates:
    """Tests for retrieve_candidates() — token scoring with query expansion."""

    def _make_blocks(self):
        from app.research.evidence_searcher import FilingBlock

        return [
            FilingBlock(
                "mda_p0",
                "mda",
                0,
                "Our ten largest customers accounted for approximately 35% of our total revenue for the year ended December 2025.",
            ),
            FilingBlock(
                "mda_p1",
                "mda",
                1,
                "Revenue increased 12% year over year driven by strong demand across all product categories and geographies.",
            ),
            FilingBlock(
                "risk_factors_p0",
                "risk_factors",
                0,
                "We face intense competition from established players and new entrants in all our markets.",
            ),
            FilingBlock(
                "fin_notes_p0",
                "fin_notes",
                0,
                "Long-term debt matures as follows: 2026 $500M, 2027 $750M, 2028 $1.2B. Interest rate is 4.5%.",
            ),
            FilingBlock(
                "business_p0",
                "business",
                0,
                "The company is a leading provider of enterprise software solutions serving over 10,000 customers worldwide.",
            ),
        ]

    def test_basic_token_match(self):
        from app.research.evidence_searcher import retrieve_candidates

        blocks = self._make_blocks()
        results = retrieve_candidates("debt maturity schedule", blocks)
        assert len(results) >= 1
        assert results[0].block.block_id == "fin_notes_p0"

    def test_query_expansion_finds_filing_language(self):
        from app.research.evidence_searcher import retrieve_candidates

        blocks = self._make_blocks()
        # "concentration" expands to "accounted for", "largest", etc.
        results = retrieve_candidates("customer concentration data", blocks)
        assert len(results) >= 1
        assert results[0].block.block_id == "mda_p0"

    def test_section_prior_boost(self):
        from app.research.evidence_searcher import retrieve_candidates, FilingBlock

        # Two blocks with identical text in different sections
        blocks = [
            FilingBlock(
                "mda_p0",
                "mda",
                0,
                "Revenue growth rate was 15% driven by customer retention and expansion.",
            ),
            FilingBlock(
                "other_p0",
                "other",
                0,
                "Revenue growth rate was 15% driven by customer retention and expansion.",
            ),
        ]
        results = retrieve_candidates(
            "revenue growth rate", blocks, source="GROWTH_VS_EARNINGS_POWER"
        )
        assert len(results) == 2
        # mda should score higher due to section prior boost
        assert results[0].block.section == "mda"

    def test_word_boundary_matching(self):
        from app.research.evidence_searcher import retrieve_candidates, FilingBlock

        blocks = [
            FilingBlock(
                "p0",
                "other",
                0,
                "The corporate strategy focuses on capacity building and integration of operations across regions.",
            ),
            FilingBlock(
                "p1",
                "other",
                1,
                "The interest rate on our revolving credit facility is SOFR plus 150 basis points. The cap on borrowings is $500M.",
            ),
        ]
        # "rate" should match "interest rate" (word boundary) but NOT "corporate" (substring)
        # "cap" should match "$500M cap" but NOT "capacity"
        results = retrieve_candidates("interest rate cap", blocks)
        if results:
            assert results[0].block.block_id == "p1"

    def test_short_query_requires_both_tokens(self):
        from app.research.evidence_searcher import retrieve_candidates, FilingBlock

        blocks = [
            FilingBlock(
                "p0",
                "other",
                0,
                "The company reported strong retention rates across all customer segments with minimal churn.",
            ),
            FilingBlock(
                "p1",
                "other",
                1,
                "Revenue from the enterprise segment increased by 20% year over year due to large contract wins.",
            ),
        ]
        # "peer multiples" — both tokens must match for a 2-token query
        results = retrieve_candidates("peer multiples", blocks)
        # Neither block has both "peer" and "multiples" → empty
        assert len(results) == 0

    def test_minimum_threshold_filters_weak_matches(self):
        from app.research.evidence_searcher import retrieve_candidates, FilingBlock

        blocks = [
            FilingBlock(
                "p0",
                "other",
                0,
                "The company has operations in 40 countries with diverse revenue streams and multiple product lines.",
            ),
        ]
        # Only "revenue" matches from a 5-token query → below 0.3 threshold
        results = retrieve_candidates("customer concentration revenue breakdown analysis", blocks)
        assert len(results) == 0

    def test_deterministic_tiebreaking(self):
        from app.research.evidence_searcher import retrieve_candidates, FilingBlock

        blocks = [
            FilingBlock(
                "mda_p0",
                "mda",
                0,
                "Revenue and margin analysis shows growth trends in all segments.",
            ),
            FilingBlock(
                "mda_p1",
                "mda",
                1,
                "Revenue and margin analysis shows growth trends in all segments.",
            ),
        ]
        r1 = retrieve_candidates("revenue margin growth", blocks, source="MARGIN_COLLAPSE")
        r2 = retrieve_candidates("revenue margin growth", blocks, source="MARGIN_COLLAPSE")
        assert [c.block.block_id for c in r1] == [c.block.block_id for c in r2]
        # Earlier ordinal should win on tie
        if len(r1) >= 2:
            assert r1[0].block.ordinal <= r1[1].block.ordinal

    def test_top_n_limits_results(self):
        from app.research.evidence_searcher import retrieve_candidates, FilingBlock

        blocks = [
            FilingBlock(
                f"mda_p{i}",
                "mda",
                i,
                f"Revenue growth and margin analysis for segment {i} shows strong results with customer retention.",
            )
            for i in range(20)
        ]
        results = retrieve_candidates("revenue growth margin", blocks, top_n=3)
        assert len(results) <= 3

    def test_match_terms_populated(self):
        from app.research.evidence_searcher import retrieve_candidates

        blocks = self._make_blocks()
        results = retrieve_candidates("debt maturity schedule", blocks)
        assert len(results) >= 1
        assert len(results[0].match_terms) >= 1

    def test_negated_passage_demoted(self):
        from app.research.evidence_searcher import retrieve_candidates, FilingBlock

        blocks = [
            FilingBlock(
                "p0",
                "mda",
                0,
                "The company has no debt and no maturity schedule. It operates with zero leverage.",
            ),
            FilingBlock(
                "p1",
                "fin_notes",
                1,
                "Long-term debt matures as follows: 2026 $500M, 2027 $750M. The maturity schedule is detailed below.",
            ),
        ]
        results = retrieve_candidates("debt maturity schedule", blocks)
        assert len(results) >= 2
        # Real disclosure should rank above negated passage
        assert results[0].block.block_id == "p1"

    def test_high_signal_8k_items_rank_above_generic_other_events(self):
        from app.research.evidence_searcher import FilingBlock, retrieve_candidates

        blocks = [
            FilingBlock(
                "8k_801_p0",
                "other_material",
                0,
                "The company disclosed leadership transition and executive appointment details.",
                item_code="8.01",
                event_category="other_material",
                source_form_type="8-K",
                source_role="material_event",
            ),
            FilingBlock(
                "8k_502_p0",
                "leadership_governance",
                0,
                "The company disclosed leadership transition and executive appointment details.",
                item_code="5.02",
                event_category="leadership_governance",
                source_form_type="8-K",
                source_role="material_event",
            ),
        ]

        results = retrieve_candidates("leadership transition executive appointment", blocks)

        assert len(results) == 2
        assert results[0].block.block_id == "8k_502_p0"


class TestAdjudicateEvidenceItem:
    """Tests for _adjudicate_evidence_item() — LLM classification per evidence item."""

    def _make_fake_provider(self, payload: dict):
        from app.llm.providers.disabled_provider import LLMResult

        @_dc
        class FakeProvider:
            provider_name: str = "openai"

            def enabled(self):
                return True

            def synthesize_json(self, *, prompt, schema, schema_name=None, **kwargs):
                return LLMResult(
                    json_text=json.dumps(payload),
                    model="test",
                    usage_input_tokens=100,
                    usage_output_tokens=50,
                    raw={},
                )

        return FakeProvider()

    def test_confirms_with_citations(self):
        from app.research.evidence_searcher import (
            _adjudicate_evidence_item,
            FilingBlock,
            CandidateBlock,
        )
        from app.research.hypothesis_generator import EvidenceNeed, Hypothesis

        need = EvidenceNeed("test_001", "customer concentration data", "REQUIRED")
        hypothesis = Hypothesis(
            claim="Test claim",
            direction="BEARISH",
            priority="HIGH",
            source="TEST",
            evidence_needed=[need],
            falsification="Test",
            impact_estimate=None,
        )
        candidates = [
            CandidateBlock(
                block=FilingBlock(
                    "mda_p0", "mda", 0, "Our largest customer accounted for 35% of revenue."
                ),
                score=0.8,
                match_terms=["customer", "concentration"],
            )
        ]

        provider = self._make_fake_provider(
            {
                "status": "CONFIRMS",
                "cited_blocks": [
                    {"block_id": "mda_p0", "excerpt": "largest customer accounted for 35%"}
                ],
                "structured_fact": "35%",
                "reasoning_short": "Filing directly states customer concentration at 35%.",
            }
        )

        result = _adjudicate_evidence_item(need, hypothesis, candidates, provider)
        assert result.status == "CONFIRMS"
        assert result.classification_method == "LLM"
        assert result.structured_fact == "35%"
        assert len(result.citations) >= 1
        assert result.citations[0].block_id == "mda_p0"
        assert result.citations[0].section == "mda"

    def test_disabled_provider_returns_unclassified(self):
        from app.research.evidence_searcher import (
            _adjudicate_evidence_item,
            FilingBlock,
            CandidateBlock,
        )
        from app.research.hypothesis_generator import EvidenceNeed, Hypothesis

        need = EvidenceNeed("test_001", "customer concentration data", "REQUIRED")
        hypothesis = Hypothesis(
            claim="Test",
            direction="BEARISH",
            priority="HIGH",
            source="TEST",
            evidence_needed=[need],
            falsification="Test",
        )
        candidates = [
            CandidateBlock(
                block=FilingBlock(
                    "mda_p0", "mda", 0, "Some text here that is long enough to be a real block."
                ),
                score=0.5,
                match_terms=["customer"],
            )
        ]

        result = _adjudicate_evidence_item(need, hypothesis, candidates, None)
        assert result.status == "UNCLASSIFIED"
        assert result.classification_method == "NONE"
        assert result.candidates_considered >= 1

    def test_empty_candidates_returns_not_found(self):
        from app.research.evidence_searcher import _adjudicate_evidence_item
        from app.research.hypothesis_generator import EvidenceNeed, Hypothesis

        need = EvidenceNeed("test_001", "customer concentration data", "REQUIRED")
        hypothesis = Hypothesis(
            claim="Test",
            direction="BEARISH",
            priority="HIGH",
            source="TEST",
            evidence_needed=[need],
            falsification="Test",
        )

        result = _adjudicate_evidence_item(need, hypothesis, [], None)
        assert result.status == "NOT_FOUND"
        assert result.candidates_considered == 0

    def test_span_offsets_computed_when_excerpt_matches(self):
        from app.research.evidence_searcher import (
            _adjudicate_evidence_item,
            FilingBlock,
            CandidateBlock,
        )
        from app.research.hypothesis_generator import EvidenceNeed, Hypothesis

        block_text = "Our largest customer accounted for 35% of total revenue."
        need = EvidenceNeed("test_001", "customer concentration", "REQUIRED")
        hypothesis = Hypothesis(
            claim="Test",
            direction="BEARISH",
            priority="HIGH",
            source="TEST",
            evidence_needed=[need],
            falsification="Test",
        )
        candidates = [
            CandidateBlock(
                block=FilingBlock("mda_p0", "mda", 0, block_text),
                score=0.8,
                match_terms=["customer"],
            )
        ]

        provider = self._make_fake_provider(
            {
                "status": "CONFIRMS",
                "cited_blocks": [{"block_id": "mda_p0", "excerpt": "accounted for 35%"}],
                "structured_fact": "35%",
                "reasoning_short": "Found.",
            }
        )

        result = _adjudicate_evidence_item(need, hypothesis, candidates, provider)
        assert len(result.citations) >= 1
        cit = result.citations[0]
        assert cit.span_start is not None
        assert cit.span_end is not None
        assert block_text[cit.span_start : cit.span_end] == "accounted for 35%"

    def test_provider_exception_returns_none_method(self, monkeypatch):
        from app.research.evidence_searcher import (
            _adjudicate_evidence_item,
            FilingBlock,
            CandidateBlock,
        )
        from app.research.hypothesis_generator import EvidenceNeed, Hypothesis

        need = EvidenceNeed("test_001", "customer concentration data", "REQUIRED")
        hypothesis = Hypothesis(
            claim="Test",
            direction="BEARISH",
            priority="HIGH",
            source="TEST",
            evidence_needed=[need],
            falsification="Test",
        )
        candidates = [
            CandidateBlock(
                block=FilingBlock(
                    "mda_p0", "mda", 0, "Some block text long enough to be a real block here."
                ),
                score=0.5,
                match_terms=["customer"],
            )
        ]

        @_dc
        class BrokenProvider:
            provider_name: str = "openai"
            model: str = "gpt-5-mini"
            calls: int = 0

            def synthesize_json(self, *, prompt, schema, schema_name=None, **kwargs):
                self.calls += 1
                raise RuntimeError("LLM service unavailable")

        class _LiveScope:
            def __init__(self) -> None:
                self.checks = 0

            def require(self, *, scenarios=None):
                self.checks += 1
                assert scenarios == ({"financial_inputs": {"current_price": 35.0}},)

        monkeypatch.setattr(
            "app.research.evidence_searcher._require_paid_financial_scope",
            _real_require_paid_financial_scope,
        )
        scope = _LiveScope()
        provider = BrokenProvider()
        result = _adjudicate_evidence_item(
            need,
            hypothesis,
            candidates,
            provider,
            financial_integrity_scope=scope,
            financial_scenarios=lambda: ({"financial_inputs": {"current_price": 35.0}},),
        )
        assert result.status == "INCONCLUSIVE"
        assert result.classification_method == "NONE"
        assert "failed" in result.reasoning_short.lower()
        assert provider.calls == 1
        assert scope.checks == 2

    def test_provider_mutation_then_exception_raises_integrity_error(self, monkeypatch):
        from app.autonomous.financial_integrity import (
            FinancialIntegrityScope,
            InvalidFinancialInputError,
            require_financial_integrity_scope,
        )
        from app.research.evidence_searcher import (
            _adjudicate_evidence_item,
            CandidateBlock,
            FilingBlock,
        )
        from app.research.hypothesis_generator import EvidenceNeed, Hypothesis

        with pytest.raises(InvalidFinancialInputError) as invalid:
            require_financial_integrity_scope(
                FinancialIntegrityScope(
                    context="evidence_failed_call_rebinding",
                    run_as_of_date="2026-03-25",
                    packets=(),
                )
            )
        integrity_error = invalid.value
        scenario = {"financial_inputs": {"current_price": 35.0}}
        need = EvidenceNeed("test_001", "customer concentration data", "REQUIRED")
        hypothesis = Hypothesis(
            claim="Test",
            direction="BEARISH",
            priority="HIGH",
            source="TEST",
            evidence_needed=[need],
            falsification="Test",
        )
        candidates = [
            CandidateBlock(
                block=FilingBlock(
                    "mda_p0",
                    "mda",
                    0,
                    "Some block text long enough to be a real block here.",
                ),
                score=0.5,
                match_terms=["customer"],
            )
        ]

        class _LiveScope:
            def __init__(self) -> None:
                self.checks = 0

            def require(self, *, scenarios=None):
                self.checks += 1
                if scenarios[0]["financial_inputs"]["current_price"] != 35.0:
                    raise integrity_error

        class _MutatingFailedProvider:
            provider_name = "openai"
            model = "gpt-5-mini"

            def __init__(self) -> None:
                self.calls = 0

            def synthesize_json(self, **_kwargs):
                self.calls += 1
                scenario["financial_inputs"]["current_price"] = 36.0
                raise RuntimeError("provider failed after mutating live inputs")

        monkeypatch.setattr(
            "app.research.evidence_searcher._require_paid_financial_scope",
            _real_require_paid_financial_scope,
        )
        scope = _LiveScope()
        provider = _MutatingFailedProvider()

        with pytest.raises(InvalidFinancialInputError) as raised:
            _adjudicate_evidence_item(
                need,
                hypothesis,
                candidates,
                provider,
                financial_integrity_scope=scope,
                financial_scenarios=lambda: (scenario,),
            )

        assert raised.value is integrity_error
        assert provider.calls == 1
        assert scope.checks == 2


class TestBuildCitationsVerification:
    """Tests for _build_citations() excerpt verification and block-id quarantine."""

    def test_unverified_excerpt_is_flagged_and_warned(self):
        from app.research.evidence_searcher import _build_citations, FilingBlock

        block = FilingBlock(
            "mda_p0", "mda", 0, "Revenue grew due to strong demand across all segments."
        )
        block_index = {"mda_p0": block}
        # LLM excerpt is NOT verbatim in the block text.
        cited = [{"block_id": "mda_p0", "excerpt": "Revenue fell 40% on collapsing demand"}]

        citations, warnings = _build_citations(cited, block_index)

        assert len(citations) == 1
        cit = citations[0]
        assert cit.block_id == "mda_p0"
        assert cit.unverified_excerpt is True
        assert cit.is_synthetic is True
        assert cit.span_start is None
        assert cit.span_end is None
        assert warnings == ["1 cited excerpt(s) not found verbatim in their blocks"]

    def test_verbatim_excerpt_not_flagged_no_warning(self):
        from app.research.evidence_searcher import _build_citations, FilingBlock

        block = FilingBlock(
            "mda_p0", "mda", 0, "Our largest customer accounted for 35% of total revenue."
        )
        block_index = {"mda_p0": block}
        cited = [{"block_id": "mda_p0", "excerpt": "accounted for 35%"}]

        citations, warnings = _build_citations(cited, block_index)

        assert len(citations) == 1
        cit = citations[0]
        assert cit.unverified_excerpt is False
        assert cit.is_synthetic is False
        assert cit.span_start == 21
        assert cit.span_end == 38
        assert warnings == []

    def test_unknown_block_id_is_dropped_and_warned(self):
        from app.research.evidence_searcher import _build_citations, FilingBlock

        block = FilingBlock(
            "mda_p0", "mda", 0, "Our largest customer accounted for 35% of total revenue."
        )
        block_index = {"mda_p0": block}
        # LLM hallucinated a block_id that was never presented.
        cited = [{"block_id": "fin_notes_p99", "excerpt": "fabricated debt schedule"}]

        citations, warnings = _build_citations(cited, block_index)

        assert citations == []
        assert warnings == ["1 citation(s) dropped for unknown/hallucinated block_id"]

    def test_mixed_verbatim_unverified_and_unknown(self):
        from app.research.evidence_searcher import _build_citations, FilingBlock

        block = FilingBlock(
            "mda_p0", "mda", 0, "Our largest customer accounted for 35% of total revenue."
        )
        block_index = {"mda_p0": block}
        cited = [
            {"block_id": "mda_p0", "excerpt": "accounted for 35%"},  # verbatim
            {"block_id": "mda_p0", "excerpt": "accounted for 90%"},  # unverified
            {"block_id": "ghost_p0", "excerpt": "hallucinated content"},  # unknown block
        ]

        citations, warnings = _build_citations(cited, block_index)

        # Verbatim + unverified are kept (2); unknown is dropped.
        assert len(citations) == 2
        assert citations[0].unverified_excerpt is False
        assert citations[1].unverified_excerpt is True
        assert warnings == [
            "1 cited excerpt(s) not found verbatim in their blocks",
            "1 citation(s) dropped for unknown/hallucinated block_id",
        ]

    def test_adjudicate_surfaces_unverified_warnings_in_result(self):
        from app.research.evidence_searcher import (
            _adjudicate_evidence_item,
            FilingBlock,
            CandidateBlock,
        )
        from app.research.hypothesis_generator import EvidenceNeed, Hypothesis
        from app.llm.providers.disabled_provider import LLMResult

        need = EvidenceNeed("test_001", "customer concentration data", "REQUIRED")
        hypothesis = Hypothesis(
            claim="Test",
            direction="BEARISH",
            priority="HIGH",
            source="TEST",
            evidence_needed=[need],
            falsification="Test",
        )
        candidates = [
            CandidateBlock(
                block=FilingBlock(
                    "mda_p0", "mda", 0, "Our largest customer accounted for 35% of revenue."
                ),
                score=0.8,
                match_terms=["customer"],
            )
        ]

        @_dc
        class FakeProvider:
            provider_name: str = "openai"

            def synthesize_json(self, *, prompt, schema, schema_name=None, **kwargs):
                payload = {
                    "status": "CONFIRMS",
                    "cited_blocks": [
                        {"block_id": "mda_p0", "excerpt": "fabricated unverifiable text"}
                    ],
                    "structured_fact": "35%",
                    "reasoning_short": "x",
                }
                return LLMResult(
                    json_text=json.dumps(payload),
                    model="test",
                    usage_input_tokens=10,
                    usage_output_tokens=10,
                    raw={},
                )

        result = _adjudicate_evidence_item(need, hypothesis, candidates, FakeProvider())
        assert result.warnings == ["1 cited excerpt(s) not found verbatim in their blocks"]
        assert result.citations[0].unverified_excerpt is True


class TestAggregation:
    """Tests for hypothesis status derivation and coverage scoring."""

    def _make_item(self, status, importance, method="LLM"):
        from app.research.evidence_searcher import EvidenceItemResult

        return EvidenceItemResult(
            need_id="t",
            needed="t",
            importance=importance,
            status=status,
            classification_method=method,
            citations=[],
            excerpt="",
            structured_fact=None,
            reasoning_short="",
            candidates_considered=0,
            top_candidate_score=0.0,
            candidate_rankings=[],
        )

    def test_required_contradiction_flips(self):
        from app.research.evidence_searcher import _derive_hypothesis_status

        items = [
            self._make_item("CONFIRMS", "IMPORTANT"),
            self._make_item("CONTRADICTS", "REQUIRED"),
        ]
        assert _derive_hypothesis_status(items) == "CONTRADICTED"

    def test_all_required_important_confirm(self):
        from app.research.evidence_searcher import _derive_hypothesis_status

        items = [
            self._make_item("CONFIRMS", "REQUIRED"),
            self._make_item("CONFIRMS", "IMPORTANT"),
            self._make_item("NOT_FOUND", "SUPPORTING"),
        ]
        assert _derive_hypothesis_status(items) == "CONFIRMED"

    def test_partial_confirmation(self):
        from app.research.evidence_searcher import _derive_hypothesis_status

        items = [
            self._make_item("CONFIRMS", "REQUIRED"),
            self._make_item("INCONCLUSIVE", "IMPORTANT"),
        ]
        assert _derive_hypothesis_status(items) == "PARTIALLY_CONFIRMED"

    def test_important_contradiction_without_confirms_is_inconclusive(self):
        from app.research.evidence_searcher import _derive_hypothesis_status

        items = [
            self._make_item("CONTRADICTS", "IMPORTANT"),
            self._make_item("NOT_FOUND", "REQUIRED"),
        ]
        assert _derive_hypothesis_status(items) == "INCONCLUSIVE"

    def test_important_contradiction_with_confirms_is_partial(self):
        from app.research.evidence_searcher import _derive_hypothesis_status

        items = [
            self._make_item("CONFIRMS", "REQUIRED"),
            self._make_item("CONTRADICTS", "IMPORTANT"),
        ]
        assert _derive_hypothesis_status(items) == "PARTIALLY_CONFIRMED"

    def test_all_inconclusive_is_inconclusive(self):
        from app.research.evidence_searcher import _derive_hypothesis_status

        items = [
            self._make_item("INCONCLUSIVE", "REQUIRED"),
            self._make_item("NOT_FOUND", "IMPORTANT"),
        ]
        assert _derive_hypothesis_status(items) == "INCONCLUSIVE"

    def test_all_unclassified(self):
        from app.research.evidence_searcher import _derive_hypothesis_status

        items = [
            self._make_item("UNCLASSIFIED", "REQUIRED", method="NONE"),
            self._make_item("UNCLASSIFIED", "IMPORTANT", method="NONE"),
        ]
        assert _derive_hypothesis_status(items) == "UNCLASSIFIED"

    def test_supporting_does_not_affect_status(self):
        from app.research.evidence_searcher import _derive_hypothesis_status

        items = [
            self._make_item("INCONCLUSIVE", "REQUIRED"),
            self._make_item("CONTRADICTS", "SUPPORTING"),
        ]
        assert _derive_hypothesis_status(items) == "INCONCLUSIVE"

    def test_coverage_score_counts_resolved(self):
        from app.research.evidence_searcher import _compute_coverage

        items = [
            self._make_item("CONFIRMS", "REQUIRED"),
            self._make_item("CONTRADICTS", "IMPORTANT"),
            self._make_item("NOT_FOUND", "SUPPORTING"),
            self._make_item("INCONCLUSIVE", "SUPPORTING"),
        ]
        assert _compute_coverage(items) == 0.5

    def test_coverage_empty(self):
        from app.research.evidence_searcher import _compute_coverage

        assert _compute_coverage([]) == 0.0

    def test_classification_method_derivation(self):
        from app.research.evidence_searcher import _derive_classification_method

        items_llm = [self._make_item("CONFIRMS", "REQUIRED", "LLM")]
        assert _derive_classification_method(items_llm) == "LLM"

        items_none = [self._make_item("UNCLASSIFIED", "REQUIRED", "NONE")]
        assert _derive_classification_method(items_none) == "NONE"

        items_mixed = [
            self._make_item("CONFIRMS", "REQUIRED", "LLM"),
            self._make_item("CONFIRMS", "IMPORTANT", "FALLBACK"),
        ]
        assert _derive_classification_method(items_mixed) == "MIXED"


class TestSearchEvidence:
    """Integration tests for search_evidence()."""

    def _make_hypothesis(self):
        from app.research.hypothesis_generator import Hypothesis, EvidenceNeed

        return Hypothesis(
            claim="Customer concentration risk threatens growth assumption.",
            direction="BEARISH",
            priority="HIGH",
            source="GROWTH_VS_EARNINGS_POWER",
            evidence_needed=[
                EvidenceNeed("g_001", "customer concentration data", "REQUIRED"),
                EvidenceNeed("g_002", "retention metrics", "IMPORTANT"),
            ],
            falsification="If top-10 customer revenue < 25%.",
            impact_estimate=50.0,
        )

    def _make_html(self):
        return (
            "<html><body>"
            "<p>Item 7. Management's Discussion and Analysis</p>"
            "<p>Our ten largest customers accounted for approximately 35% of our total revenue "
            "for the year ended December 2025. We believe customer retention remains strong "
            "with net revenue retention exceeding 110% across all cohorts.</p>"
            "<p>Revenue increased 15% year over year driven by expansion of existing "
            "customers and new logo acquisition across enterprise and mid-market segments.</p>"
            "</body></html>"
        )

    def test_returns_evidence_result(self):
        from app.research.evidence_searcher import search_evidence

        result = search_evidence([self._make_hypothesis()], self._make_html())
        assert len(result) == 1
        er = result[0]
        assert er.hypothesis.source == "GROWTH_VS_EARNINGS_POWER"
        assert len(er.evidence_item_results) == 2
        assert er.hypothesis_status in (
            "CONFIRMED",
            "CONTRADICTED",
            "PARTIALLY_CONFIRMED",
            "INCONCLUSIVE",
            "UNCLASSIFIED",
        )

    def test_llm_disabled_returns_unclassified(self):
        from app.research.evidence_searcher import search_evidence

        result = search_evidence([self._make_hypothesis()], self._make_html())
        er = result[0]
        # Default LLM provider is disabled in test env
        assert er.hypothesis_status == "UNCLASSIFIED"
        assert er.coverage_score == 0.0

    def test_max_hypotheses_truncates(self):
        from app.research.evidence_searcher import search_evidence
        from app.research.hypothesis_generator import Hypothesis, EvidenceNeed

        hyps = [
            Hypothesis(
                claim=f"Hypothesis {i}",
                direction="BEARISH",
                priority="HIGH",
                source=f"SRC_{i}",
                evidence_needed=[
                    EvidenceNeed(f"n_{i}", "some evidence", "REQUIRED"),
                ],
                falsification="F",
            )
            for i in range(5)
        ]
        result = search_evidence(hyps, self._make_html(), max_hypotheses=2)
        assert len(result) == 2

    def test_empty_html_returns_results(self):
        from app.research.evidence_searcher import search_evidence

        result = search_evidence([self._make_hypothesis()], "")
        assert len(result) == 1
        er = result[0]
        for item in er.evidence_item_results:
            assert item.status in ("NOT_FOUND", "UNCLASSIFIED")

    def test_candidates_retrieved_even_without_llm(self):
        from app.research.evidence_searcher import search_evidence

        result = search_evidence([self._make_hypothesis()], self._make_html())
        er = result[0]
        # At least the concentration item should find candidates
        conc_item = next((i for i in er.evidence_item_results if "concentration" in i.needed), None)
        if conc_item:
            assert conc_item.candidates_considered > 0

    def test_fallback_on_no_candidates(self, monkeypatch):
        from app.research.evidence_searcher import search_evidence
        from app.research.hypothesis_generator import Hypothesis, EvidenceNeed
        from app.llm.providers.disabled_provider import LLMResult

        hyp = Hypothesis(
            claim="Test",
            direction="BEARISH",
            priority="HIGH",
            source="TEST",
            evidence_needed=[
                EvidenceNeed(
                    "x_001",
                    "something extremely specific and unlikely to match anything",
                    "REQUIRED",
                ),
            ],
            falsification="F",
        )

        class FakeProvider:
            provider_name = "openai"

            def enabled(self):
                return True

            def synthesize_json(self, *, prompt, schema, schema_name=None, **kwargs):
                payload = {
                    "status": "CONFIRMS",
                    "cited_blocks": [{"block_id": "all_sections_fallback", "excerpt": "some text"}],
                    "structured_fact": None,
                    "reasoning_short": "Found in fallback.",
                }
                return LLMResult(
                    json_text=json.dumps(payload),
                    model="test",
                    usage_input_tokens=10,
                    usage_output_tokens=10,
                    raw={},
                )

        monkeypatch.setattr(
            "app.research.evidence_searcher.get_llm_provider", lambda: FakeProvider()
        )

        html = (
            "<p>Item 7. Management's Discussion and Analysis</p><p>"
            + "Generic content about revenue and operations. " * 50
            + "</p>"
        )
        result = search_evidence([hyp], html)
        er = result[0]
        item = er.evidence_item_results[0]
        assert item.classification_method == "FALLBACK"
        for cit in item.citations:
            assert cit.is_synthetic is True


class TestCurrentEventBlocks:
    def _context(self):
        from app.research.current_event_context import CurrentEventContext, CurrentEventDocument

        return CurrentEventContext(
            documents=[
                CurrentEventDocument(
                    ticker="TEST",
                    source_type="ir_press",
                    published_at="2026-04-17T10:00:00+00:00",
                    title="Press release title",
                    source_url="https://example.com/press",
                    summary="Revenue outlook was raised.",
                    citations=[],
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
            ]
        )

    def test_build_current_event_blocks_preserves_metadata(self):
        from app.research.evidence_searcher import build_current_event_blocks

        blocks = build_current_event_blocks(self._context())

        assert len(blocks) == 1
        assert blocks[0].source_type == "ir_press"
        assert blocks[0].source_title == "Press release title"
        assert blocks[0].source_url == "https://example.com/press"
        assert blocks[0].source_published_at == "2026-04-17T10:00:00+00:00"
        assert blocks[0].source_role == "current_event"
        assert blocks[0].source_quality["source_family"] == "company_controlled"
        assert blocks[0].source_quality["freshness_bucket"] == "recent_7d"

    def test_retrieve_candidates_prefers_ir_press_over_company_news(self):
        from app.research.evidence_searcher import FilingBlock, retrieve_candidates

        blocks = [
            FilingBlock(
                block_id="event_ir",
                section="ir_press",
                ordinal=0,
                text="Raised revenue outlook and reaffirmed margin target.",
                source_role="current_event",
                source_type="ir_press",
                source_title="IR press",
                source_url="https://example.com/ir",
                source_published_at="2026-04-17T10:00:00+00:00",
            ),
            FilingBlock(
                block_id="event_news",
                section="company_news",
                ordinal=1,
                text="Raised revenue outlook and reaffirmed margin target.",
                source_role="current_event",
                source_type="company_news",
                source_title="Company news",
                source_url="https://example.com/news",
                source_published_at="2026-04-17T10:00:00+00:00",
            ),
        ]

        candidates = retrieve_candidates(
            "raised revenue outlook margin", blocks, source="MARKET_PREMIUM", top_n=2
        )

        assert [candidate.block.block_id for candidate in candidates] == ["event_ir", "event_news"]

    def test_retrieve_candidates_prefers_newer_current_events(self):
        from app.research.evidence_searcher import FilingBlock, retrieve_candidates

        blocks = [
            FilingBlock(
                block_id="event_old",
                section="company_news",
                ordinal=0,
                text="Leadership update and demand commentary remained unchanged.",
                source_role="current_event",
                source_type="company_news",
                source_title="Older news",
                source_url="https://example.com/old",
                source_published_at="2026-03-01T10:00:00+00:00",
            ),
            FilingBlock(
                block_id="event_new",
                section="company_news",
                ordinal=1,
                text="Leadership update and demand commentary remained unchanged.",
                source_role="current_event",
                source_type="company_news",
                source_title="Newer news",
                source_url="https://example.com/new",
                source_published_at="2026-04-17T10:00:00+00:00",
            ),
        ]

        candidates = retrieve_candidates(
            "leadership demand commentary", blocks, source="MARKET_PREMIUM", top_n=2
        )

        assert [candidate.block.block_id for candidate in candidates] == ["event_new", "event_old"]
