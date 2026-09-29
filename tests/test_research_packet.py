from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.research.schemas import validate_research_packet


def _base_payload() -> dict:
    return {
        "run_id": "r1",
        "ticker": "AAPL",
        "as_of_date": "2026-02-13",
        "generated_at": "2026-02-13T00:00:00+00:00",
        "key_questions": [
            {"question_id": "Q1", "question": "q1"},
            {"question_id": "Q2", "question": "q2"},
            {"question_id": "Q3", "question": "q3"},
            {"question_id": "Q4", "question": "q4"},
            {"question_id": "Q5", "question": "q5"},
        ],
        "evidence_items": [
            {
                "id": "ev_1",
                "ticker": "AAPL",
                "as_of_date": "2026-02-13",
                "source_type": "EDGAR",
                "source_url": "https://www.sec.gov/Archives/example",
                "retrieved_at": "2026-02-13T00:00:00+00:00",
                "excerpt_text": "example",
                "citations": [
                    {
                        "source_url": "https://www.sec.gov/Archives/example",
                        "snippet": "snippet",
                        "section_label": "MD&A",
                    }
                ],
                "hash": "h1",
            }
        ],
        "findings": [
            {
                "entry_id": "F1",
                "summary": "finding",
                "evidence_item_ids": ["ev_1"],
                "citations": [{"source_url": "https://www.sec.gov/Archives/example", "snippet": "snippet"}],
                "derived_from": [],
            }
        ],
        "risks": [
            {
                "entry_id": "R1",
                "summary": "risk",
                "evidence_item_ids": ["ev_1"],
                "citations": [],
                "derived_from": [],
            }
        ],
        "catalysts": [
            {
                "entry_id": "C1",
                "summary": "cat",
                "evidence_item_ids": ["ev_1"],
                "citations": [],
                "derived_from": [],
            }
        ],
        "disconfirming_evidence": [
            {
                "entry_id": "D1",
                "summary": "disc",
                "evidence_item_ids": ["ev_1"],
                "citations": [{"source_url": "https://www.sec.gov/Archives/example", "snippet": "snippet"}],
                "derived_from": [],
            }
        ],
        "next_actions": [
            {
                "step_id": "A1",
                "question_id": "Q1",
                "action": "pull filing",
                "section_targets": ["MD&A"],
                "keywords": ["revenue"],
                "disconfirmation_check": "fail if decline",
                "evidence_gap": "gap",
                "tied_metrics": ["revenue"],
                "allowed_source": "EDGAR",
            }
        ],
        "claims": [
            {
                "claim_id": "cl1",
                "label": "metric",
                "value": 1.0,
                "unit": "ratio",
                "citations": [{"source_url": "https://www.sec.gov/Archives/example", "snippet": "snippet"}],
                "derived_from": [],
            }
        ],
    }


def test_research_packet_validator_accepts_valid_payload():
    payload = _base_payload()
    packet = validate_research_packet(payload)
    assert packet.ticker == "AAPL"


def test_research_packet_validator_rejects_missing_evidence_links():
    payload = _base_payload()
    payload["findings"][0]["evidence_item_ids"] = []
    with pytest.raises(ValidationError):
        validate_research_packet(payload)
