"""Pure within-sector cross-sectional ranker.

Replaces the flat 12% absolute-return cliff as the selection driver with a
within-sector cross-sectional rank. Each candidate is scored on a blended
within-sector z-score of three orthogonal, already-computed signals:

  * value   — cheapness (e.g. ``discount_to_anchor``), higher-is-better
  * quality — economics (e.g. ROIC, ROIC-WACC spread, FCF margin), higher-is-better
  * gap     — the reverse-DCF expectations gap (``implied_growth``), LOWER-is-better
              (a low bar to beat is good), so it is NEGATED before z-scoring

The top quantile (default top 20%) is flagged as a BUY-candidate regardless of
the absolute DCF discount. This stops the cliff from rewarding distressed names
(which clear 12% only because they are distressed) while rejecting quality
leaders trading at fair-to-slightly-cheap prices.

This module is intentionally pure and deterministic: it takes pre-extracted
factor vectors and performs no DB or network I/O, so it is fully unit-testable
on synthetic peer sets. Callers (sector_runtime) extract the factor vectors
from packets and feed them in; the ``gap`` field is passed as the RAW
``implied_growth`` value and negated internally here, so the ranker treats all
three factors as higher-is-better.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

# Plan defaults: top-quantile BUY-candidate cutoff and factor-blend weights.
CROSS_SECTIONAL_BUY_QUANTILE = 0.20
DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {"value": 0.40, "quality": 0.40, "gap": 0.20}

# Factor names in canonical order. ``gap`` is lower-is-better (negated internally).
_FACTOR_NAMES = ("value", "quality", "gap")
_HIGHER_IS_BETTER = {"value": True, "quality": True, "gap": False}


@dataclass
class FactorVector:
    """Pre-extracted per-member factor inputs for the cross-sectional ranker.

    ``gap`` is the RAW reverse-DCF ``implied_growth`` (lower-is-better); the
    ranker negates it internally so all three factors are scored
    higher-is-better.
    """

    ticker: str
    value: float | None
    quality: float | None
    gap: float | None


@dataclass
class CrossSectionalRank:
    """Per-member result of the within-sector cross-sectional rank."""

    ticker: str
    composite: float | None
    rank: int
    percentile: float
    factor_zscores: dict[str, float | None]
    buy_candidate: bool
    buy_candidate_reason: str | None


def _zscore(values: list[float]) -> list[float]:
    """Population (ddof=0) z-scores rounded to 4 dp; all-equal -> all 0.0."""
    if not values:
        return []
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0:
        return [0.0 for _ in values]
    return [round((v - mean) / stdev, 4) for v in values]


def rank_cross_sectional(
    factors: list[FactorVector],
    *,
    weights: dict[str, float] | None = None,
    buy_quantile: float = CROSS_SECTIONAL_BUY_QUANTILE,
) -> list[CrossSectionalRank]:
    """Rank ``factors`` within their sector on a blended within-sector z-score.

    Steps:
      1. For each factor, collect the sub-list of non-None member values
         (negating ``gap`` so lower implied growth scores higher), z-score them,
         and scatter back by ticker keeping None where the member lacked it.
      2. Composite = weighted mean over the member's PRESENT factor z-scores,
         with the weights of present factors renormalized to sum to 1 (so a
         member missing a factor is not penalized as if z=0). Composite is None
         when the member has no rankable factors.
      3. Sort by composite descending (None last), assign ranks 1..N.
      4. percentile = round((N - rank + 1) / N * 100, 1).
      5. buy_candidate = rank <= ceil(N * buy_quantile) AND composite is not None.
    """
    if weights is None:
        weights = DEFAULT_FACTOR_WEIGHTS

    n = len(factors)
    if n == 0:
        return []

    # 1) Per-factor z-scores scattered back by member index.
    zscores_by_factor: dict[str, list[float | None]] = {}
    for name in _FACTOR_NAMES:
        higher_is_better = _HIGHER_IS_BETTER[name]
        present_idx: list[int] = []
        present_vals: list[float] = []
        for i, fv in enumerate(factors):
            raw = getattr(fv, name)
            if raw is None:
                continue
            present_idx.append(i)
            present_vals.append(raw if higher_is_better else -raw)
        zs = _zscore(present_vals)
        scattered: list[float | None] = [None] * n
        for slot, idx in enumerate(present_idx):
            scattered[idx] = zs[slot]
        zscores_by_factor[name] = scattered

    # 2) Composite per member over present factors, renormalizing their weights.
    rows: list[tuple[int, FactorVector, dict[str, float | None], float | None]] = []
    for i, fv in enumerate(factors):
        member_z: dict[str, float | None] = {
            name: zscores_by_factor[name][i] for name in _FACTOR_NAMES
        }
        present = [name for name in _FACTOR_NAMES if member_z[name] is not None]
        if not present:
            composite: float | None = None
        else:
            weight_sum = sum(weights[name] for name in present)
            if weight_sum == 0:
                composite = None
            else:
                blended = sum(weights[name] * member_z[name] for name in present)
                composite = round(blended / weight_sum, 4)
        rows.append((i, fv, member_z, composite))

    # 3) Sort by composite desc (None last), original order as final tiebreak.
    def _sort_key(item: tuple[int, FactorVector, dict[str, float | None], float | None]):
        original_order, _fv, _z, composite = item
        has_composite = composite is not None
        return (
            0 if has_composite else 1,
            -(composite if composite is not None else 0.0),
            original_order,
        )

    ordered = sorted(rows, key=_sort_key)

    # 4/5) Assign rank, percentile, buy_candidate.
    buy_cutoff = math.ceil(n * buy_quantile)
    results: list[CrossSectionalRank] = []
    for position, (_original_order, fv, member_z, composite) in enumerate(ordered):
        rank = position + 1
        percentile = round((n - rank + 1) / n * 100, 1)
        if composite is None:
            buy_candidate = False
            reason = "NO_RANKABLE_FACTORS"
        else:
            buy_candidate = rank <= buy_cutoff
            reason = None
        results.append(
            CrossSectionalRank(
                ticker=fv.ticker,
                composite=composite,
                rank=rank,
                percentile=percentile,
                factor_zscores=member_z,
                buy_candidate=buy_candidate,
                buy_candidate_reason=reason,
            )
        )
    return results
