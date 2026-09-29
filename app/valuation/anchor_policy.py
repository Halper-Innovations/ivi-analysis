"""Canonical valuation-anchor selection — ONE rule for live packets and the
backtest.

Audit findings backtest-live-anchor-divergence + anchor-rule-backtest-vs-live-
divergence (both HIGH): the backtest measured deploy_ready against
anchor = max(positive dcf_base, epv_adjusted) while the live watchlist used
FIRST-AVAILABLE(dcf, epv, graham, ncav) with no positivity check — so the
measured signal and the production buy trigger were different rules (~17-26%
of names diverged; negative DCF silently blocked positive EPV on ~8%).

The canonical rule (documented core signal: price <= 0.75 x max(DCF, EPV)):

  1. A sector-specific anchor (insurance residual-income, technology-adjusted)
     wins when positive — these model classes the generic stack mis-values.
  2. Otherwise anchor = MAX of the POSITIVE earnings anchors {dcf, epv}.
  3. Otherwise fall through graham -> ncav (first positive). Non-positive
     values are SKIPPED, never selected, and never block a lower-priority
     positive method.

Both app/autonomous/sector_financial_packets.py and app/backtest/reconstruct.py
must select anchors through this module. Do not re-derive the rule locally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AnchorSelection:
    method: str | None
    value: float | None
    # Full candidate set for provenance/output surfaces ({method: value}).
    candidates: dict[str, float | None] = field(default_factory=dict)
    # Why this method won: SECTOR_SPECIFIC | MAX_POSITIVE_DCF_EPV |
    # FALLBACK_GRAHAM | FALLBACK_NCAV | NO_POSITIVE_ANCHOR |
    # DECLINE_CAPPED_NO_GROWTH (decline-class name capped at the no-growth basis)
    reason: str = "NO_POSITIVE_ANCHOR"


# Revenue-trend classes whose names must not anchor on a growth-bearing
# basis: projecting growth scenarios for a business the classifier says is
# shrinking lets the DCF leg set buy targets the no-growth economics cannot
# support. VOLATILE (cyclical) is intentionally NOT a decline class — the
# cyclical trough is handled by the EPV OI-median normalization, not by
# capping (a trough-year cap would punish exactly the names the
# normalization exists to value through the cycle).
DECLINE_TREND_CLASSES = ("DECLINING", "SECULAR_DECLINE")


def is_decline_class(revenue_trend_class: object) -> bool:
    """ONE predicate for 'a decline-class gate-adjust fired' across all
    select_anchor call sites (live packets, backtest, render)."""
    return str(revenue_trend_class or "") in DECLINE_TREND_CLASSES


def _pos(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0:
        return float(value)
    return None


def select_anchor(
    *,
    dcf: float | None = None,
    epv: float | None = None,
    graham: float | None = None,
    ncav: float | None = None,
    sector_specific: tuple[str, float] | None = None,
    extra_candidates: dict[str, float | None] | None = None,
    decline_class: bool = False,
) -> AnchorSelection:
    """Select the valuation anchor under the canonical rule (module docstring).

    ``extra_candidates`` (e.g. the ev_ebit / fcf_yield / tangible_floor
    lenses) are PROVENANCE ONLY: they appear in the candidates dict for
    output surfaces but can never be selected — no single lens may act as a
    gate or an anchor.

    ``decline_class`` (decline-cap policy): when the caller's quality context
    classifies the revenue trend as a decline (``is_decline_class``), the
    selected anchor is CAPPED at the no-growth (EPV-class) basis — a
    growth-bearing DCF must not set buy targets for a shrinking business.
    When EPV is uncomputable or anomalous (non-positive), the cap falls to
    the most conservative POSITIVE anchor-eligible method (dcf/epv/graham/
    ncav/sector-specific; provenance lenses never anchor, so they never cap).
    The conviction/confidence-widened margin of safety stacks ON TOP of the
    capped anchor (buy targets are computed downstream from the returned
    value) — documented policy, not an accident.
    """
    candidates: dict[str, float | None] = {
        "dcf": float(dcf) if isinstance(dcf, (int, float)) else None,
        "epv": float(epv) if isinstance(epv, (int, float)) else None,
        "graham": float(graham) if isinstance(graham, (int, float)) else None,
        "ncav": float(ncav) if isinstance(ncav, (int, float)) else None,
    }
    for name, value in (extra_candidates or {}).items():
        # A provenance lens never replaces a core method's value: a lens arriving
        # under an anchor-eligible name ("dcf", ...) would otherwise become the anchor.
        if str(name) in candidates:
            continue
        candidates[str(name)] = float(value) if isinstance(value, (int, float)) else None
    anchor_eligible: list[str] = ["dcf", "epv", "graham", "ncav"]
    selection: AnchorSelection | None = None
    if sector_specific is not None:
        sector_method, sector_value = sector_specific
        candidates[str(sector_method)] = (
            float(sector_value) if isinstance(sector_value, (int, float)) else None
        )
        anchor_eligible.append(str(sector_method))
        sector_pos = _pos(sector_value)
        if sector_pos is not None:
            selection = AnchorSelection(
                method=str(sector_method),
                value=sector_pos,
                candidates=candidates,
                reason="SECTOR_SPECIFIC",
            )

    if selection is None:
        earnings = {name: _pos(candidates[name]) for name in ("dcf", "epv")}
        positive_earnings = {k: v for k, v in earnings.items() if v is not None}
        if positive_earnings:
            method = max(positive_earnings, key=lambda k: positive_earnings[k])
            selection = AnchorSelection(
                method=method,
                value=positive_earnings[method],
                candidates=candidates,
                reason="MAX_POSITIVE_DCF_EPV",
            )

    if selection is None:
        for name, reason in (("graham", "FALLBACK_GRAHAM"), ("ncav", "FALLBACK_NCAV")):
            value = _pos(candidates[name])
            if value is not None:
                selection = AnchorSelection(
                    method=name, value=value, candidates=candidates, reason=reason
                )
                break

    if selection is None:
        return AnchorSelection(method=None, value=None, candidates=candidates)

    if decline_class and selection.value is not None:
        # Decline-cap policy (docstring): no-growth basis = EPV when positive,
        # else the most conservative positive anchor-eligible method.
        epv_pos = _pos(candidates.get("epv"))
        if epv_pos is not None:
            cap_method, cap_value = "epv", epv_pos
        else:
            eligible_pos = {
                name: _pos(candidates.get(name))
                for name in anchor_eligible
                if _pos(candidates.get(name)) is not None
            }
            if eligible_pos:
                cap_method = min(eligible_pos, key=lambda k: eligible_pos[k])
                cap_value = eligible_pos[cap_method]
            else:
                cap_method, cap_value = None, None
        if cap_value is not None and selection.value > cap_value:
            return AnchorSelection(
                method=cap_method,
                value=cap_value,
                candidates=candidates,
                reason="DECLINE_CAPPED_NO_GROWTH",
            )

    return selection


def published_dcf_base(pricing_zone_detail: Any, raw_base: Any) -> float | None:
    """The DCF value the platform stands behind, given the scorecard's zone detail.

    When a non-recurring revenue spike is detected the writer computes a
    "durable" DCF with the spike removed, anchors the valuation and the buy
    target on it, and flags the raw figure DCF_INFLATED_BY_NONRECURRING_REVENUE
    — but the ``dcf`` method row it persists is the RAW result. The durable base
    lives only in the scorecard's ``pricing_zone_detail``. Every consumer that
    shows a DCF resolves it through here. Without it the dossier printed $30 in
    its decision block and $50 in its own table three paragraphs below, the
    company page and the gauge drew $50, and the memo argued a 79% discount
    from the number the writer had already rejected.
    """
    pzd = pricing_zone_detail if isinstance(pricing_zone_detail, dict) else {}
    durable = pzd.get("dcf_base")
    raw = pzd.get("dcf_raw_base")
    if isinstance(durable, (int, float)) and isinstance(raw, (int, float)):
        return float(durable)
    return float(raw_base) if isinstance(raw_base, (int, float)) else None
