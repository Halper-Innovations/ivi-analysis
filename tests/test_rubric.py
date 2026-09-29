from app.score.rubric import score_packet


def _sample_packet():
    return {
        "fundamentals": {
            "revenue": 1000,
            "operating_margin": 0.12,
            "fcf": 100,
            "fcf_margin": 0.10,
            "net_debt": 200,
            "liquidity_stress_score": 3,
        },
        "valuations": {
            "dcf_lite": {"outputs": {"confidence": "MEDIUM"}},
            "reverse_dcf": {"inputs": {"market_price": "UNKNOWN"}},
        },
        "deltas_vs_prior_period": {"revenue": 50, "fcf": 10},
        "extracted_facts": [],
    }


def test_rubric_is_deterministic():
    packet = _sample_packet()
    s1 = score_packet(packet, {"classification": "WATCHLIST"})
    s2 = score_packet(packet, {"classification": "WATCHLIST"})
    assert s1 == s2


def test_price_unknown_caps_gap_score():
    packet = _sample_packet()
    subscores, total, decision, reasons = score_packet(packet, {"classification": "WATCHLIST"})
    assert subscores["valuation_gap"] <= 12.0
    assert any("Market price UNKNOWN" in r for r in reasons)


def test_research_incomplete_applies_penalty():
    packet = _sample_packet()
    subscores, total, decision, reasons = score_packet(
        packet,
        {"classification": "WATCHLIST"},
        research_quality={
            "overall_research_score": 40.0,
            "incomplete": True,
            "top_gaps": [{"summary": "missing IR RSS metadata"}],
        },
    )
    assert subscores["research_penalty"] < 0
    assert total < 100
    assert any("Research incomplete" in r for r in reasons)
