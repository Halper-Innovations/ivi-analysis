"""Tests for app.alpha.consensus_ranker."""
from __future__ import annotations

from app.alpha.schemas import TickerSignalPacket


def _make_packet(ticker, dcf=None, epv=None, graham=None, ncav=None,
                 price=None, gate="PROCEED", moat=None, downside=None,
                 solvency=None, tension="NONE", growth_dep=None) -> TickerSignalPacket:
    return TickerSignalPacket(
        ticker=ticker, dcf_value=dcf, epv_value=epv, graham_value=graham,
        ncav_value=ncav, current_price=price, gate_verdict=gate,
        moat_score=moat, downside_risk_class=downside, solvency_risk=solvency,
        method_tension_type=tension, growth_dependency_ratio=growth_dep,
        methods_agree=tension == "NONE",
        consensus_direction="UNDERVALUED" if dcf and price and dcf > price else "OVERVALUED",
    )


def test_ranks_by_consensus_discount():
    """Ticker with discount across multiple methods ranks higher than single-method discount."""
    packets = {
        "MULTI": _make_packet("MULTI", dcf=150, epv=120, graham=100, price=80),  # all 3 agree: undervalued
        "SINGLE": _make_packet("SINGLE", dcf=200, epv=30, graham=40, price=100),  # only DCF says undervalued
    }
    from app.alpha.consensus_ranker import rank_by_consensus
    result = rank_by_consensus(packets)
    ranked = result.ranked
    assert ranked[0].ticker == "MULTI"


def test_blocked_tickers_excluded():
    """BLOCK gate tickers should not appear in the ranked list."""
    packets = {
        "GOOD": _make_packet("GOOD", dcf=150, epv=120, price=100, gate="PROCEED"),
        "BAD": _make_packet("BAD", dcf=200, epv=180, price=50, gate="BLOCK"),
    }
    from app.alpha.consensus_ranker import rank_by_consensus
    result = rank_by_consensus(packets)
    ranked = result.ranked
    tickers = [r.ticker for r in ranked]
    assert "BAD" not in tickers
    assert "GOOD" in tickers


def test_blocked_tickers_can_be_included_for_downstream_sector_gate():
    packets = {
        "GOOD": _make_packet("GOOD", dcf=150, epv=120, price=100, gate="PROCEED"),
        "BANK": _make_packet("BANK", dcf=None, epv=None, price=40, gate="BLOCK"),
    }

    from app.alpha.consensus_ranker import rank_by_consensus

    result = rank_by_consensus(packets, include_blocked=True)

    assert [entry.ticker for entry in result.ranked] == ["GOOD"]
    assert [entry.ticker for entry in result.ranked_insufficient] == ["BANK"]
    assert result.ranked_insufficient[0].adjustments == ["NO_DISCOUNT_DATA", "GATE_BLOCKED"]


def test_critical_solvency_penalized():
    """CRITICAL solvency risk should heavily penalize ranking."""
    packets = {
        "HEALTHY": _make_packet("HEALTHY", dcf=150, epv=120, price=100, solvency="LOW"),
        "DISTRESSED": _make_packet("DISTRESSED", dcf=200, epv=180, price=80, solvency="CRITICAL"),
    }
    from app.alpha.consensus_ranker import rank_by_consensus
    result = rank_by_consensus(packets)
    ranked = result.ranked
    assert ranked[0].ticker == "HEALTHY"


def test_growth_tension_penalized():
    """High growth dependency should reduce ranking vs multi-method agreement."""
    packets = {
        "SOLID": _make_packet("SOLID", dcf=150, epv=130, graham=110, price=100, tension="NONE"),
        "GROWTH": _make_packet("GROWTH", dcf=300, epv=50, price=100,
                               tension="GROWTH_VS_EARNINGS_POWER", growth_dep=0.83),
    }
    from app.alpha.consensus_ranker import rank_by_consensus
    result = rank_by_consensus(packets)
    ranked = result.ranked
    assert ranked[0].ticker == "SOLID"


def test_no_price_ranked_last():
    """Tickers without price cannot compute discount — ranked at bottom."""
    packets = {
        "PRICED": _make_packet("PRICED", dcf=150, epv=120, price=100),
        "UNPRICED": _make_packet("UNPRICED", dcf=200, epv=180, price=None),
    }
    from app.alpha.consensus_ranker import rank_by_consensus
    result = rank_by_consensus(packets)
    ranked = result.ranked
    assert ranked[0].ticker == "PRICED"


def test_insufficient_data_separated():
    """Tickers without gate_verdict should be in ranked_insufficient."""
    packets = {
        "FULL": _make_packet("FULL", dcf=150, epv=120, price=100, gate="PROCEED"),
        "SPARSE": _make_packet("SPARSE", dcf=200, epv=180, price=80, gate=None),
    }
    from app.alpha.consensus_ranker import rank_by_consensus
    result = rank_by_consensus(packets)
    assert len(result.ranked) == 1
    assert result.ranked[0].ticker == "FULL"
    assert len(result.ranked_insufficient) == 1
    assert result.ranked_insufficient[0].ticker == "SPARSE"


def test_returns_score_and_metadata():
    """Each ranked entry should include the consensus score and component breakdown."""
    packets = {
        "TEST": _make_packet("TEST", dcf=150, epv=120, graham=100, price=80, moat=5),
    }
    from app.alpha.consensus_ranker import rank_by_consensus
    result = rank_by_consensus(packets)
    ranked = result.ranked
    assert len(ranked) == 1
    entry = ranked[0]
    assert hasattr(entry, 'consensus_score')
    assert hasattr(entry, 'method_discounts')
    assert entry.consensus_score > 0


def test_empty_input():
    """Empty packet dict should return empty RankingResult."""
    from app.alpha.consensus_ranker import rank_by_consensus
    result = rank_by_consensus({})
    assert result.ranked == []
    assert result.ranked_insufficient == []


def test_dcf_weight_reduced_for_high_growth_dependency():
    """DCF weight should decrease when growth_dependency_ratio is high."""
    from app.alpha.consensus_ranker import rank_by_consensus
    # Two tickers: same DCF discount, but GROWTH has high growth dependency
    packets = {
        "STABLE": _make_packet("STABLE", dcf=200, epv=180, price=100, growth_dep=0.1),
        "GROWTH": _make_packet("GROWTH", dcf=200, epv=50, price=100, growth_dep=0.8),
    }
    result = rank_by_consensus(packets)
    ranked = result.ranked
    # STABLE should rank higher — its DCF carries full weight
    assert ranked[0].ticker == "STABLE"
    # GROWTH should have a DCF_WEIGHT_REDUCED adjustment
    growth_entry = next(e for e in ranked if e.ticker == "GROWTH")
    assert any("DCF_WEIGHT_REDUCED" in a for a in growth_entry.adjustments)


def test_asset_vs_earnings_tension_penalized():
    """ASSET_VS_EARNINGS tension should reduce score vs no-tension ticker."""
    from app.alpha.consensus_ranker import rank_by_consensus
    packets = {
        "CLEAN": _make_packet("CLEAN", dcf=150, epv=130, graham=110, price=100, tension="NONE"),
        "ASSET": _make_packet("ASSET", dcf=150, epv=130, graham=110, price=100,
                              tension="ASSET_VS_EARNINGS"),
    }
    result = rank_by_consensus(packets)
    ranked = result.ranked
    clean = next(e for e in ranked if e.ticker == "CLEAN")
    asset = next(e for e in ranked if e.ticker == "ASSET")
    assert clean.consensus_score > asset.consensus_score
    assert any("ASSET_TENSION" in a for a in asset.adjustments)


def test_insufficient_data_in_separate_list():
    """Tickers without gate_verdict should appear in ranked_insufficient, not ranked."""
    from app.alpha.consensus_ranker import rank_by_consensus
    packets = {
        "FULL": _make_packet("FULL", dcf=150, epv=120, price=100, gate="PROCEED"),
        "SPARSE": _make_packet("SPARSE", dcf=200, epv=180, price=80, gate=None),
    }
    result = rank_by_consensus(packets)
    full_tickers = [e.ticker for e in result.ranked]
    sparse_tickers = [e.ticker for e in result.ranked_insufficient]
    assert "FULL" in full_tickers
    assert "SPARSE" in sparse_tickers
    assert "SPARSE" not in full_tickers


def test_all_insufficient_fallback():
    """When all tickers lack gate_verdict, ranked_insufficient should contain them all."""
    from app.alpha.consensus_ranker import rank_by_consensus
    packets = {
        "A": _make_packet("A", dcf=200, epv=180, price=100, gate=None),
        "B": _make_packet("B", dcf=150, epv=120, price=100, gate=None),
    }
    result = rank_by_consensus(packets)
    assert len(result.ranked) == 0
    assert len(result.ranked_insufficient) == 2
