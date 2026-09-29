from __future__ import annotations

from app.evidence.enrichment import enrich_evidence_packet


def test_enrich_evidence_packet_adds_trends_and_valuation_spreads():
    packet = {
        "ticker": "AAPL",
        "fundamentals": {
            "revenue_cagr_3y": 0.12,
            "revenue_cagr_5y": 0.14,
            "operating_margin_trend_slope": 0.015,
            "fcf_margin_trend_slope": -0.002,
            "dilution_rate_shares_cagr": -0.005,
            "share_repurchases_amount": 80.0,
            "dividends_paid_amount": 50.0,
            "r_and_d_intensity_latest": 0.18,
        },
        "valuations": {
            "owner_earnings": {"outputs": {"confidence": "LOWER", "flags": ["SBC_NOT_ADJUSTED"]}},
            "dcf": {"outputs": {"base": 12.0}},
            "epv": {"outputs": {"value_per_share": 11.0}},
            "graham": {"outputs": {"value_per_share": 9.0}},
            "reverse_dcf": {"inputs": {"price": 24.0}, "outputs": {"outputs": {"implied_growth": 0.08}}},
        },
        "extracted_facts": [
            {
                "fact_type": "deferred_revenue_amount",
                "value": {"metric": "deferred_revenue_amount", "value": 400.0},
                "citation": {"snippet": "Deferred revenue was 400"},
            }
        ],
    }

    enriched = enrich_evidence_packet(packet)
    enrichment = enriched["enrichment"]
    assert enrichment["trend_narratives"]["revenue_cagr_3y_pct"] == 12.0
    assert enrichment["trend_narratives"]["operating_margin_trend_direction"] == "improving"
    assert enrichment["trend_narratives"]["fcf_conversion_trend_direction"] == "declining"
    assert enrichment["trend_narratives"]["dilution_direction"] == "share_count_shrinking"
    assert enrichment["valuation_spread_analysis"]["dcf"]["gap_pct"] == 100.0
    assert enrichment["valuation_spread_analysis"]["implied_growth_feasibility"] == "PLAUSIBLE"
    assert enrichment["capital_allocation_quality"]["capital_allocation_posture"] == "shareholder_return_active"
    assert enrichment["owner_earnings_quality_flags"]["flags"] == ["SBC_NOT_ADJUSTED"]
    assert enrichment["moat_signals_summary"][0]["signal"] == "deferred_revenue_amount"
