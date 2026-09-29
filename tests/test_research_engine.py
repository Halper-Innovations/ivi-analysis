from __future__ import annotations

import pytest

from app.research.engine import _build_key_questions, _build_next_actions, _detect_base_gaps


def _ev(id_, *, source_type="EDGAR", title="", excerpt="", section=None, snippet=""):
    from app.research.schemas import CitationRef, EvidenceItem

    return EvidenceItem(
        id=id_,
        ticker="ABC",
        as_of_date="2026-02-13",
        source_type=source_type,
        source_url=f"https://example.com/{id_}",
        source_title=title,
        source_published_at=None,
        retrieved_at="2026-02-13T00:00:00+00:00",
        excerpt_text=excerpt,
        citations=[CitationRef(source_url=f"https://example.com/{id_}", snippet=snippet, section_label=section)],
        hash=f"h_{id_}",
        dedupe_key=None,
    )


def test_detect_base_gaps_points_to_companyfacts_and_evidence_packet_rebuild():
    packet = {
        "fundamentals": {
            "revenue": "UNKNOWN",
            "operating_margin": "UNKNOWN",
            "fcf": "UNKNOWN",
            "net_debt": "UNKNOWN",
        },
        "valuations": {},
    }
    gaps = _detect_base_gaps(packet)
    assert gaps
    assert "companyfacts" in gaps[0].recommended_action.lower()
    assert "evidence packet" in gaps[0].recommended_action.lower()


def test_detect_base_gaps_uses_financial_issuer_metrics():
    packet = {
        "fundamentals": {
            "issuer_classification": "financial",
            "revenue": 182447.0,
            "operating_margin": "UNKNOWN",
            "fcf": "UNKNOWN",
            "net_debt": "UNKNOWN",
            "deposits": 2559320.0,
            "loans": 1400000.0,
            "total_assets": 3900000.0,
            "allowance_for_credit_losses": "UNKNOWN",
        },
        "financials": [
            {"line_item": "deposits", "citation": {"snippet": "Deposits 2559320"}},
            {"line_item": "loans", "citation": {"snippet": "Loans 1400000"}},
            {"line_item": "total_assets", "citation": {"snippet": "Assets 3900000"}},
        ],
        "valuations": {},
    }

    gaps = _detect_base_gaps(packet)
    gap_ids = {gap.gap_id for gap in gaps}

    assert "GAP_UNKNOWN_FCF" not in gap_ids
    assert "GAP_UNKNOWN_OPERATING_MARGIN" not in gap_ids
    assert "GAP_UNKNOWN_NET_DEBT" not in gap_ids
    assert "GAP_UNKNOWN_ALLOWANCE_FOR_CREDIT_LOSSES" in gap_ids
    assert "GAP_UNKNOWN_PROVISION_FOR_CREDIT_LOSSES" in gap_ids
    assert "GAP_UNKNOWN_NET_CHARGE_OFFS" in gap_ids
    assert "GAP_UNKNOWN_NONACCRUAL_LOANS" in gap_ids


def test_build_next_actions_uses_bank_native_priority_for_financial_issuers():
    packet = {
        "fundamentals": {
            "issuer_classification": "financial",
            "revenue": 182447.0,
            "deposits": 2559320.0,
            "loans": "UNKNOWN",
            "total_assets": 4424900.0,
            "allowance_for_credit_losses": "UNKNOWN",
        },
        "financials": [
            {"line_item": "deposits", "citation": {"snippet": "Deposits 2559320"}},
            {"line_item": "total_assets", "citation": {"snippet": "Assets 4424900"}},
        ],
    }
    gaps = _detect_base_gaps(packet)
    actions = _build_next_actions(_build_key_questions(), gaps, packet=packet)

    assert actions
    assert "deposit, loan, and funding-mix bridges" in actions[0].action
    assert "Hydrate loans, allowance, provision, charge-off, and nonaccrual-credit history" in actions[1].action
    assert "covenant definitions, maturities, refinancing paths, and liquidity buffers" in actions[2].action
    assert "8-K exhibits and investor presentations" in actions[6].action


def test_build_findings_uses_bank_credit_history_when_present():
    from app.research.engine import _build_findings
    from app.research.schemas import CitationRef, EvidenceItem

    packet = {
        "fundamentals": {
            "issuer_classification": "financial",
            "deposits": 2559320.0,
            "total_assets": 4424900.0,
            "loans": 1408905.0,
            "allowance_for_credit_losses": 25765.0,
            "provision_for_credit_losses": 14212.0,
            "net_charge_offs": 9849.0,
            "derived_signals": {
                "allowance_to_loans_history": {"value": [{"year": 2023, "value": 0.017}, {"year": 2024, "value": 0.018}, {"year": 2025, "value": 0.0183}]},
                "provision_to_loans_history": {"value": [{"year": 2023, "value": 0.008}, {"year": 2024, "value": 0.009}, {"year": 2025, "value": 0.0101}]},
                "net_charge_offs_to_loans_history": {"value": [{"year": 2023, "value": 0.004}, {"year": 2024, "value": 0.005}, {"year": 2025, "value": 0.007}]},
            },
        }
    }
    findings = _build_findings(
        packet,
        evidence_items=[
            EvidenceItem(
                id="E1",
                ticker="JPM",
                as_of_date="2026-02-13",
                source_type="EDGAR",
                source_url="https://www.sec.gov/example",
                source_title="10-K",
                source_published_at=None,
                retrieved_at="2026-02-13T00:00:00+00:00",
                excerpt_text="Reserve coverage and charge-off trends are discussed.",
                citations=[CitationRef(source_url="https://www.sec.gov/example", snippet="reserve coverage", section_label="notes")],
                hash="h",
                dedupe_key=None,
            )
        ],
    )
    assert findings
    summary = findings[0].summary
    assert "Historical reserve coverage shows" in summary
    assert "allowance-to-loans history spans" in summary
    assert "net charge-offs-to-loans history spans" in summary


def test_select_relevant_ranks_by_keyword_relevance_not_position():
    from app.research.engine import _select_relevant

    # E1 is first positionally but mentions nothing about debt/covenant.
    # E3 is last positionally but is the most relevant to the query keywords.
    items = [
        _ev("E1", title="General overview", excerpt="The company sells widgets to consumers."),
        _ev("E2", title="Earnings", excerpt="Revenue grew this year."),
        _ev("E3", title="Debt notes", excerpt="Covenant headroom and debt maturity refinancing risk.", snippet="covenant"),
    ]
    picked = _select_relevant(items, ["covenant", "debt", "maturity", "refinancing"], n=1)
    assert [it.id for it in picked] == ["E3"]


def test_select_relevant_excludes_ids():
    from app.research.engine import _select_relevant

    items = [
        _ev("E1", excerpt="Debt covenant maturity refinancing discussion."),
        _ev("E2", excerpt="Debt covenant maturity refinancing discussion."),
    ]
    picked = _select_relevant(items, ["debt", "covenant"], n=2, exclude_ids={"E1"})
    assert [it.id for it in picked] == ["E2"]


def test_select_relevant_falls_back_when_no_keyword_match():
    from app.research.engine import _select_relevant

    # No keyword matches anywhere -> deterministic fallback to id-sorted order.
    items = [
        _ev("E2", excerpt="alpha beta gamma"),
        _ev("E1", excerpt="alpha beta gamma"),
    ]
    picked = _select_relevant(items, ["nonexistentkeyword"], n=1)
    assert [it.id for it in picked] == ["E1"]


def test_disconfirming_uses_disjoint_evidence_from_findings():
    from app.research.engine import _build_findings, _build_disconfirming

    packet = {
        "fundamentals": {"net_debt": 1000.0, "liquidity_stress_score": 0.4},
        "valuations": {
            "dcf": {"outputs": {"base": 42.0}},
            "epv": {"outputs": {"value_per_share": 30.0}},
        },
    }
    items = [
        _ev("E1", title="10-K valuation", excerpt="Valuation ranges and DCF base from filing inputs.", snippet="valuation"),
        _ev("E2", title="Balance sheet", excerpt="Net debt and liquidity disclosures from filing.", snippet="liquidity"),
        _ev("E3", title="Cash flow", excerpt="Operating cash flow weakened and dilution from SBC rose.", snippet="cash flow"),
        _ev("E4", title="Equity footnote", excerpt="Shares outstanding rose due to stock-based compensation dilution.", snippet="dilution"),
    ]
    findings = _build_findings(packet, items)
    finding_ids = set()
    for f in findings:
        finding_ids.update(f.evidence_item_ids)

    disconfirming = _build_disconfirming(items, exclude_ids=finding_ids)
    disconfirming_ids = set()
    for d in disconfirming:
        disconfirming_ids.update(d.evidence_item_ids)

    assert disconfirming_ids
    assert finding_ids.isdisjoint(disconfirming_ids)


def test_assert_opposing_buckets_disjoint_raises_on_identical_evidence():
    from app.research.engine import _assert_opposing_buckets_disjoint
    from app.research.schemas import EvidenceLinkedEntry

    findings = [EvidenceLinkedEntry(entry_id="F1", summary="x", evidence_item_ids=["E1", "E2"])]
    disconfirming = [EvidenceLinkedEntry(entry_id="D1", summary="y", evidence_item_ids=["E1", "E2"])]

    with pytest.raises(ValueError, match="share evidence"):
        _assert_opposing_buckets_disjoint(findings, disconfirming, evidence_pool_size=4)


def test_assert_opposing_buckets_disjoint_passes_when_disjoint():
    from app.research.engine import _assert_opposing_buckets_disjoint
    from app.research.schemas import EvidenceLinkedEntry

    findings = [EvidenceLinkedEntry(entry_id="F1", summary="x", evidence_item_ids=["E1", "E2"])]
    disconfirming = [EvidenceLinkedEntry(entry_id="D1", summary="y", evidence_item_ids=["E3"])]

    # Should not raise.
    _assert_opposing_buckets_disjoint(findings, disconfirming, evidence_pool_size=4)


def test_assert_opposing_buckets_disjoint_allows_overlap_when_pool_too_small():
    from app.research.engine import _assert_opposing_buckets_disjoint
    from app.research.schemas import EvidenceLinkedEntry

    # Only one evidence item exists; disjointness is impossible, so overlap is tolerated.
    findings = [EvidenceLinkedEntry(entry_id="F1", summary="x", evidence_item_ids=["E1"])]
    disconfirming = [EvidenceLinkedEntry(entry_id="D1", summary="y", evidence_item_ids=["E1"])]

    _assert_opposing_buckets_disjoint(findings, disconfirming, evidence_pool_size=1)
