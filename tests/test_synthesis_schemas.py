from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.llm.schemas import synthesis_schema_for_prompt, validate_synthesis_packet


def _base_payload() -> dict:
    return {
        "ticker": "AAPL",
        "as_of_date": "2026-02-13",
        "run_id": "run_synth_1",
        "hypotheses": [
            {
                "id": "h1",
                "statement": "Valuation assumptions may be too pessimistic.",
                "why_it_might_be_true": "DCF and multiples ranges imply upside under conservative assumptions.",
                "falsifiers": ["Next filing shows sustained margin compression."],
                "required_evidence": ["10-Q margin trend", "cash flow stability"],
            }
        ],
        "claims": [
            {
                "id": "c1",
                "text": "DCF base case is above current implied expectations.",
                "type": "non_numeric",
                "citations": [
                    {
                        "source_url": "https://www.sec.gov/Archives/example",
                        "snippet": "valuation snippet",
                        "section_label": "valuation",
                    }
                ],
                "derived_from": [],
            }
        ],
        "priced_in_assessment": {
            "what_market_assumes": "Margin compression persists.",
            "what_is_not_priced": "Stabilization in operating cash flow.",
            "uncertainty_notes": "Research coverage remains moderate.",
        },
        "next_actions": [
            {
                "action_type": "edgar_extract",
                "target_source": "EDGAR",
                "query_or_url_hint": "latest 10-Q cash flow statement",
                "why": "Confirm trend durability.",
            }
        ],
        "decision_frame": {
            "stance": "watchlist",
            "key_risks": ["dilution", "refinancing"],
            "catalysts": ["next 10-Q"],
            "time_horizon_days": 90,
        },
        "llm_meta": {
            "model": "gpt-5-mini",
            "prompt_hash": "p",
            "input_hash": "i",
            "cost_estimate_usd": 0.01,
            "created_at": "2026-02-13T00:00:00+00:00",
        },
    }


def test_synthesis_schema_accepts_valid_payload():
    packet = validate_synthesis_packet(_base_payload())
    assert packet.ticker == "AAPL"
    assert packet.business_quality_summary == ""
    assert packet.as_of_resolution == "exact"


def test_synthesis_schema_rejects_numeric_claim_without_trace():
    payload = _base_payload()
    payload["claims"] = [
        {
            "id": "c_numeric",
            "text": "Revenue growth is 20%",
            "type": "numeric",
            "citations": [],
            "derived_from": [],
        }
    ]
    with pytest.raises(ValidationError):
        validate_synthesis_packet(payload)


def test_synthesis_prompt_schema_sets_additional_properties_false_for_objects():
    schema = synthesis_schema_for_prompt()

    def _walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                props = node.get("properties")
                if isinstance(props, dict):
                    assert node.get("required") == list(props.keys())
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(schema)


def test_synthesis_schema_accepts_layer3_fields():
    payload = _base_payload()
    payload["business_quality_summary"] = "Business quality appears durable based on cited filing trends."
    payload["valuation_interpretation"] = "Valuation support appears modestly favorable versus current expectations."
    payload["risk_frame"] = "Execution and dilution remain the primary risks."
    payload["catalyst_frame"] = "The next filing and margin progression are the key catalysts."
    payload["evidence_gaps"] = ["Need cleaner segment margin evidence."]
    payload["recommended_next_actions"] = ["Read latest 10-Q segment note."]
    payload["confidence_notes"] = "Moderate confidence because several claims remain qualitative."
    packet = validate_synthesis_packet(payload)
    assert packet.evidence_gaps == ["Need cleaner segment margin evidence."]


def test_synthesis_schema_accepts_requested_vs_effective_as_of_metadata():
    payload = _base_payload()
    payload["requested_as_of_date"] = "2026-03-19"
    payload["effective_as_of_date"] = "2026-01-29"
    payload["as_of_resolution"] = "fallback"
    payload["as_of_resolution_reason"] = "Used the latest available evidence packet at or before the requested as-of date."
    packet = validate_synthesis_packet(payload)
    assert packet.requested_as_of_date == "2026-03-19"
    assert packet.effective_as_of_date == "2026-01-29"
    assert packet.as_of_resolution == "fallback"
