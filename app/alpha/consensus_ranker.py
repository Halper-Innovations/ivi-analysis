"""Rank tickers by consensus margin of safety across all valuation methods.

Replaces the LLM tournament bracket with deterministic scoring.
The LLM's role shifts from "pick a winner" to "investigate the top N."

Scoring logic:
  1. Compute discount-to-intrinsic for each method that produced a value
  2. Average the discounts (weighted: methods with higher confidence get more weight)
     - Dynamic DCF weight: reduced when growth_dependency_ratio > 50%
  3. Apply bonuses/penalties:
     - Consensus bonus: all methods agree on direction → +20%
     - Tension penalty: GROWTH_VS_EARNINGS_POWER → scale by growth_dependency_ratio
     - Tension penalty: ASSET_VS_EARNINGS → -15% of score
     - Gate penalty: ADJUST → -10%, BLOCK → excluded, None → separate tier
     - Solvency penalty: CRITICAL → -50%, ELEVATED → -20%
     - Moat bonus: score 5+ → +10%
  4. Sort descending by final score
"""
from __future__ import annotations

from dataclasses import dataclass

from app.alpha.schemas import TickerSignalPacket


@dataclass
class RankedEntry:
    """A ticker's position in the consensus ranking."""
    ticker: str
    consensus_score: float
    method_discounts: dict[str, float]  # {dcf: 0.30, epv: -0.15, ...}
    methods_with_value: int
    methods_agreeing: int
    adjustments: list[str]  # human-readable adjustment descriptions
    packet: TickerSignalPacket


@dataclass
class RankingResult:
    """Consensus ranking with separate tiers for full vs insufficient data."""
    ranked: list[RankedEntry]              # tickers with quality gate
    ranked_insufficient: list[RankedEntry]  # tickers without quality gate


def _compute_discount(value: float | None, price: float | None) -> float | None:
    """Compute discount: positive = undervalued, negative = overvalued."""
    if value is None or price is None or price <= 0 or value <= 0:
        return None
    return (value - price) / value


def _score_ticker(packet: TickerSignalPacket) -> RankedEntry:
    """Compute consensus score for a single ticker."""
    adjustments: list[str] = []

    # Step 1: Compute per-method discounts
    method_discounts: dict[str, float] = {}
    for name, value in [
        ("insurance", packet.insurance_value),
        ("dcf", packet.dcf_value),
        ("epv", packet.epv_value),
        ("graham", packet.graham_value),
        ("ncav", packet.ncav_value),
    ]:
        d = _compute_discount(value, packet.current_price)
        if d is not None:
            method_discounts[name] = round(d, 4)

    methods_with_value = len(method_discounts)

    if not method_discounts:
        return RankedEntry(
            ticker=packet.ticker,
            consensus_score=-999.0,
            method_discounts={},
            methods_with_value=0,
            methods_agreeing=0,
            adjustments=["NO_DISCOUNT_DATA"],
            packet=packet,
        )

    # Step 2: Weighted average discount
    # Base weights: EPV premium (no growth assumptions), NCAV least reliable
    weights = {"insurance": 1.3, "dcf": 1.0, "epv": 1.2, "graham": 0.8, "ncav": 0.5}
    if "insurance" in method_discounts:
        adjustments.append(f"INSURANCE_MODEL_USED ({packet.insurance_method or 'insurance'})")

    # Dynamic: reduce DCF weight when most of its value is growth premium
    growth_dep = packet.growth_dependency_ratio or 0
    if growth_dep > 0.5:
        weights["dcf"] = max(0.3, 1.0 - growth_dep)
        adjustments.append(f"DCF_WEIGHT_REDUCED ({growth_dep:.0%} growth dependent)")

    weighted_sum = sum(method_discounts[m] * weights.get(m, 1.0) for m in method_discounts)
    weight_total = sum(weights.get(m, 1.0) for m in method_discounts)
    avg_discount = weighted_sum / weight_total

    # Step 3: Count agreement
    undervalued = [m for m, d in method_discounts.items() if d > 0.05]
    overvalued = [m for m, d in method_discounts.items() if d < -0.05]
    methods_agreeing = max(len(undervalued), len(overvalued))

    score = avg_discount * 100  # convert to percentage scale

    # Step 4: Bonuses and penalties
    # Consensus bonus
    if len(undervalued) >= 2 and not overvalued:
        bonus = 20.0 * (len(undervalued) / methods_with_value)
        score += bonus
        adjustments.append(f"CONSENSUS_BONUS +{bonus:.0f} ({len(undervalued)} methods agree undervalued)")
    elif len(overvalued) >= 2 and not undervalued:
        adjustments.append("ALL_OVERVALUED")

    # Growth tension penalty
    if packet.method_tension_type == "GROWTH_VS_EARNINGS_POWER" and growth_dep > 0.4:
        penalty = score * growth_dep * 0.5  # reduce score proportional to growth dependency
        score -= abs(penalty)
        adjustments.append(f"GROWTH_TENSION -{abs(penalty):.0f} ({growth_dep:.0%} growth dependent)")

    # Asset vs earnings tension penalty
    if packet.method_tension_type == "ASSET_VS_EARNINGS":
        penalty = abs(score) * 0.15
        score -= penalty
        adjustments.append(f"ASSET_TENSION -{penalty:.0f} (asset/earnings disagreement)")

    # Gate penalty
    gate = (packet.gate_verdict or "").upper()
    if gate == "ADJUST":
        score *= 0.9
        adjustments.append("ADJUST_GATE -10%")

    # Solvency penalty
    sol = (packet.solvency_risk or "").upper()
    if sol == "CRITICAL":
        score *= 0.5
        adjustments.append("CRITICAL_SOLVENCY -50%")
    elif sol == "ELEVATED":
        score *= 0.8
        adjustments.append("ELEVATED_SOLVENCY -20%")

    # Moat bonus (mild)
    if isinstance(packet.moat_score, int) and packet.moat_score >= 5:
        score *= 1.1
        adjustments.append(f"MOAT_BONUS +10% (score={packet.moat_score})")

    return RankedEntry(
        ticker=packet.ticker,
        consensus_score=round(score, 2),
        method_discounts=method_discounts,
        methods_with_value=methods_with_value,
        methods_agreeing=methods_agreeing,
        adjustments=adjustments,
        packet=packet,
    )


def rank_by_consensus(
    packets: dict[str, TickerSignalPacket],
    *,
    include_blocked: bool = False,
) -> RankingResult:
    """Rank all tickers by consensus margin of safety.

    Excludes BLOCKED tickers. Partitions remaining into:
      - ranked: tickers with a quality gate verdict
      - ranked_insufficient: tickers without a quality gate verdict
    Both lists sorted by consensus_score descending.

    ``include_blocked`` keeps BLOCK-gated packets in the insufficient tier for
    sector-specific downstream gates that can evaluate a non-generic data shape
    without relaxing the normal consensus ranking contract.
    """
    full: list[RankedEntry] = []
    insufficient: list[RankedEntry] = []

    for ticker, packet in packets.items():
        if (packet.gate_verdict or "").upper() == "BLOCK":
            if include_blocked:
                entry = _score_ticker(packet)
                entry.adjustments.append("GATE_BLOCKED")
                insufficient.append(entry)
            continue
        entry = _score_ticker(packet)
        if packet.gate_verdict is None:
            insufficient.append(entry)
        else:
            full.append(entry)

    full.sort(key=lambda e: e.consensus_score, reverse=True)
    insufficient.sort(key=lambda e: e.consensus_score, reverse=True)
    return RankingResult(ranked=full, ranked_insufficient=insufficient)
