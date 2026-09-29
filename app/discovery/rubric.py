from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.fundamentals.normalize import UNKNOWN
from app.valuation.reverse_dcf import implied_growth_from_price


@dataclass
class DiscoveryScoreResult:
    total_score: float
    subscores: dict[str, float]
    reasons: list[str]
    reason_objects: list[dict[str, Any]]
    flags: list[str]


def _num(value: Any) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


CRITICAL_METRICS = ["ttm_revenue", "gross_margin", "operating_margin", "fcf", "shares_outstanding"]


def count_critical_unknown(metrics: dict[str, Any]) -> int:
    return sum(1 for name in CRITICAL_METRICS if metrics.get(name) == UNKNOWN)


def explainability_subscore(
    *,
    cited_reason_count: int,
    reason_count: int,
    critical_unknown_count: int,
    missing_citation_penalty: float = 0.5,
) -> float:
    cited = max(0, int(cited_reason_count))
    reasons = max(0, int(reason_count))
    unknowns = max(0, int(critical_unknown_count))
    uncited = max(0, reasons - cited)
    score = (
        (cited * 2.5)
        + min(2.0, uncited * 0.5)
        - (unknowns * 1.0)
        - (uncited * missing_citation_penalty)
    )
    return round(max(0.0, min(10.0, score)), 2)


def _add_reason(
    reason_objects: list[dict[str, Any]],
    reasons: list[str],
    *,
    reason_code: str,
    summary: str,
    severity: str,
    derived_from: list[str],
    notes: str | None = None,
) -> None:
    reason_objects.append(
        {
            "reason_code": reason_code,
            "summary": summary,
            "severity": severity,
            "derived_from": derived_from,
            "notes": notes,
        }
    )
    reasons.append(summary)


def score_evidence_strength(
    *,
    critical_unknown_count: int,
    explainability_score: float,
    metrics: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    reasons: list[dict[str, Any]] = []
    completeness_component = max(0.0, 5.0 - (float(critical_unknown_count) * 1.5))
    explainability_component = max(0.0, min(5.0, float(explainability_score) / 2.0))

    score = completeness_component + explainability_component
    if metrics.get("ttm_revenue") != UNKNOWN:
        score += 0.5
        reasons.append(
            {
                "reason_code": "EVIDENCE_REVENUE_PRESENT",
                "summary": "Revenue base is present in filing-derived metrics",
                "severity": "boost",
                "derived_from": ["discovery.metrics.ttm_revenue"],
                "notes": None,
            }
        )
    else:
        reasons.append(
            {
                "reason_code": "EVIDENCE_REVENUE_UNKNOWN",
                "summary": "Revenue base is missing from current filing extraction",
                "severity": "penalty",
                "derived_from": ["discovery.metrics.ttm_revenue"],
                "notes": None,
            }
        )
    reasons.extend(
        [
            {
                "reason_code": "EVIDENCE_COMPLETENESS",
                "summary": "Evidence completeness reflects critical metric availability",
                "severity": "info",
                "derived_from": ["discovery.metrics", "discovery.rubric.critical_unknown_count"],
                "notes": f"critical_unknown_count={int(critical_unknown_count)}",
            },
            {
                "reason_code": "EVIDENCE_EXPLAINABILITY",
                "summary": "Explainability contribution reflects cited vs uncited reasons",
                "severity": "info",
                "derived_from": ["discovery.rubric.explainability"],
                "notes": f"explainability_score={float(explainability_score):.2f}",
            },
        ]
    )
    return round(max(0.0, min(10.0, score)), 2), reasons


def determine_discovery_stage(
    *,
    total_score: float,
    whale_fit_score: float,
    evidence_strength_score: float,
    market_cap: float | str,
    market_cap_in_band: bool,
    critical_unknown_count: int,
    suppressed: bool,
) -> tuple[str, list[dict[str, Any]]]:
    reasons: list[dict[str, Any]] = []
    if suppressed:
        reasons.append(
            {
                "reason_code": "STAGE_SUPPRESSED",
                "summary": "Suppression rules forced rejection",
                "severity": "penalty",
                "derived_from": ["discovery.suppression"],
                "notes": None,
            }
        )
        return "REJECT", reasons

    if evidence_strength_score < 5.0:
        reasons.append(
            {
                "reason_code": "STAGE_LOW_EVIDENCE_STRENGTH",
                "summary": "Evidence strength below advance floor",
                "severity": "penalty",
                "derived_from": ["discovery.rubric.evidence_strength"],
                "notes": f"evidence_strength={evidence_strength_score:.2f}",
            }
        )
        return ("WATCHLIST_ONLY" if total_score >= 45.0 else "REJECT"), reasons

    if critical_unknown_count >= 2:
        reasons.append(
            {
                "reason_code": "STAGE_CRITICAL_UNKNOWNS",
                "summary": "Too many critical unknown metrics for deep advancement",
                "severity": "penalty",
                "derived_from": ["discovery.rubric.critical_unknown_count"],
                "notes": f"critical_unknown_count={critical_unknown_count}",
            }
        )
        return ("WATCHLIST_ONLY" if total_score >= 45.0 else "REJECT"), reasons

    if isinstance(market_cap, (int, float)) and not market_cap_in_band:
        reasons.append(
            {
                "reason_code": "STAGE_MCAP_OUT_OF_BAND",
                "summary": "Market cap outside 5B-50B Future Whale target band",
                "severity": "penalty",
                "derived_from": ["discovery.metrics.market_cap", "discovery.rubric.cap_band"],
                "notes": None,
            }
        )
        return "WATCHLIST_ONLY", reasons

    if not isinstance(market_cap, (int, float)) and whale_fit_score < 18.0:
        reasons.append(
            {
                "reason_code": "STAGE_MCAP_UNKNOWN_LOW_WHALEFIT",
                "summary": "Market cap unknown and whale fit below high-confidence threshold",
                "severity": "penalty",
                "derived_from": ["discovery.rubric.whale_fit", "discovery.metrics.market_cap"],
                "notes": None,
            }
        )
        return ("WATCHLIST_ONLY" if total_score >= 45.0 else "REJECT"), reasons

    if total_score >= 60.0:
        reasons.append(
            {
                "reason_code": "STAGE_ADVANCE",
                "summary": "Passed quality floor and is eligible for deep pipeline",
                "severity": "boost",
                "derived_from": ["discovery.rubric.total_score", "discovery.rubric.whale_fit"],
                "notes": None,
            }
        )
        return "ADVANCE_TO_DEEP", reasons

    reasons.append(
        {
            "reason_code": "STAGE_WATCHLIST_SCORE",
            "summary": "Quality floor passed but total score suggests watchlist",
            "severity": "info",
            "derived_from": ["discovery.rubric.total_score"],
            "notes": None,
        }
    )
    return "WATCHLIST_ONLY", reasons


def score_discovery_candidate(
    *,
    metrics: dict[str, Any],
    market_cap: float | str,
    market_cap_in_band: bool,
    price: float | None,
    shares_outstanding: float | None,
) -> DiscoveryScoreResult:
    reasons: list[str] = []
    reason_objects: list[dict[str, Any]] = []
    flags: list[str] = []

    quality = 0.0
    inflection = 0.0
    valuation = 0.0
    coverage = 10.0
    whale_fit = 0.0
    cap_band = 0.0

    gross_margin = _num(metrics.get("gross_margin"))
    op_margin = _num(metrics.get("operating_margin"))
    fcf = _num(metrics.get("fcf"))
    liq = _num(metrics.get("liquidity_stress_score"))
    revenue = _num(metrics.get("ttm_revenue"))
    net_debt = _num(metrics.get("net_debt"))
    cfo_margin = _num(metrics.get("cfo_margin"))
    fcf_margin = _num(metrics.get("fcf_margin"))
    cash_conversion = _num(metrics.get("cfo_to_net_income"))
    shares_change_4q = _num(metrics.get("shares_change_4q"))

    if gross_margin is not None and gross_margin > 0:
        quality += 8
        _add_reason(
            reason_objects,
            reasons,
            reason_code="QUALITY_GROSS_MARGIN_POSITIVE",
            summary="Positive gross margin",
            severity="boost",
            derived_from=["discovery.metrics.gross_margin"],
        )
    if op_margin is not None and op_margin > -0.02:
        quality += 8
        _add_reason(
            reason_objects,
            reasons,
            reason_code="QUALITY_OPERATING_MARGIN_NOT_DEEP_NEGATIVE",
            summary="Operating margin not deeply negative",
            severity="boost",
            derived_from=["discovery.metrics.operating_margin"],
        )
    if fcf is not None and fcf > 0:
        quality += 8
        _add_reason(
            reason_objects,
            reasons,
            reason_code="QUALITY_FCF_POSITIVE",
            summary="Free cash flow positive",
            severity="boost",
            derived_from=["discovery.metrics.fcf"],
        )
    elif fcf is not None:
        quality += 3
    if liq is not None:
        if liq <= 3:
            quality += 6
            _add_reason(
                reason_objects,
                reasons,
                reason_code="QUALITY_LOW_LIQUIDITY_STRESS",
                summary="Low liquidity stress signals",
                severity="boost",
                derived_from=["discovery.metrics.liquidity_stress_score"],
            )
        elif liq <= 6:
            quality += 3
        else:
            flags.append("LIQUIDITY_STRESS_ELEVATED")
            _add_reason(
                reason_objects,
                reasons,
                reason_code="QUALITY_LIQUIDITY_STRESS_ELEVATED",
                summary="Liquidity stress is elevated",
                severity="penalty",
                derived_from=["discovery.metrics.liquidity_stress_score"],
            )
    quality = min(30.0, quality)

    rev_accel = _num(metrics.get("revenue_acceleration"))
    gm_change = _num(metrics.get("gross_margin_change_qoq"))
    om_change = _num(metrics.get("operating_margin_change_qoq"))
    fcf_change = _num(metrics.get("fcf_change_qoq"))

    if rev_accel is not None:
        if rev_accel > 0.03:
            inflection += 12
            _add_reason(
                reason_objects,
                reasons,
                reason_code="INFLECTION_REVENUE_ACCELERATION",
                summary="Revenue growth acceleration",
                severity="boost",
                derived_from=["discovery.metrics.revenue_acceleration"],
            )
        elif rev_accel > 0:
            inflection += 6
    if gm_change is not None:
        if gm_change > 0.01:
            inflection += 8
            _add_reason(
                reason_objects,
                reasons,
                reason_code="INFLECTION_GROSS_MARGIN_EXPANSION",
                summary="Gross margin expansion",
                severity="boost",
                derived_from=["discovery.metrics.gross_margin_change_qoq"],
            )
        elif gm_change > 0:
            inflection += 4
    if om_change is not None:
        if om_change > 0.01:
            inflection += 8
            _add_reason(
                reason_objects,
                reasons,
                reason_code="INFLECTION_OPERATING_LEVERAGE_IMPROVING",
                summary="Operating leverage improving",
                severity="boost",
                derived_from=["discovery.metrics.operating_margin_change_qoq"],
            )
        elif om_change > 0:
            inflection += 4
    if fcf_change is not None:
        if fcf_change > 0:
            inflection += 7
            _add_reason(
                reason_objects,
                reasons,
                reason_code="INFLECTION_FCF_IMPROVING",
                summary="FCF trend improving",
                severity="boost",
                derived_from=["discovery.metrics.fcf_change_qoq"],
            )
    inflection = min(35.0, inflection)

    implied_growth = None
    reverse_out = None
    reverse_warn = []
    if (
        price is not None
        and shares_outstanding not in (None, 0)
        and revenue is not None
        and op_margin is not None
    ):
        reverse_out, reverse_warn = implied_growth_from_price(
            market_price=price,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt,
            base_revenue=revenue,
            margin=op_margin,
        )
        implied_growth = _num(reverse_out.get("implied_growth"))
        if reverse_out.get("implied_growth_saturated"):
            # A clipped bound is not a feasible-growth solve: a saturated-LOW
            # net-cash shell would otherwise collect the +14 feasibility
            # boost (audit: saturated-bound-leaks-to-flag-ignoring-consumers).
            # Nulling routes scoring to the EV/FCF branch below.
            implied_growth = None
        if reverse_warn:
            flags.extend([f"REVERSE_DCF_WARN:{item}" for item in reverse_warn])
    if implied_growth is not None:
        if implied_growth <= 0.10:
            valuation += 14
            _add_reason(
                reason_objects,
                reasons,
                reason_code="VALUATION_IMPLIED_GROWTH_FEASIBLE",
                summary="Implied growth appears feasible",
                severity="boost",
                derived_from=[
                    "valuation.reverse_dcf.implied_growth",
                    "discovery.metrics.ttm_revenue",
                ],
            )
        elif implied_growth <= 0.16:
            valuation += 8
        else:
            valuation += 2
            flags.append("IMPLIED_EXPECTATIONS_STRETCHED")
    elif isinstance(market_cap, (int, float)) and fcf not in (None, 0) and net_debt is not None:
        # Reached when reverse-DCF inputs were missing OR the solve saturated
        # (implied nulled above) — saturated deep-cheap names score on EV/FCF
        # instead of collecting zero valuation-plausibility points.
        ev_approx = float(market_cap) + net_debt
        ev_fcf = ev_approx / fcf if fcf else None
        if ev_fcf is not None:
            if 8 <= ev_fcf <= 25:
                valuation += 10
            elif 5 <= ev_fcf <= 35:
                valuation += 6
            else:
                valuation += 2
    else:
        flags.append("VALUATION_PLAUSIBILITY_LOW_CONFIDENCE")
    valuation = min(25.0, valuation)

    # Future Whale shaping: durability and unit economics quality proxies (0-25).
    if revenue is not None:
        if revenue >= 10_000_000_000:
            whale_fit += 6
        elif revenue >= 5_000_000_000:
            whale_fit += 5
        elif revenue >= 1_000_000_000:
            whale_fit += 3
        _add_reason(
            reason_objects,
            reasons,
            reason_code="WHALE_REVENUE_SCALE",
            summary="Revenue scale supports durable operating platform",
            severity="boost",
            derived_from=["discovery.metrics.ttm_revenue"],
        )
    else:
        _add_reason(
            reason_objects,
            reasons,
            reason_code="WHALE_REVENUE_SCALE_UNKNOWN",
            summary="Revenue scale unavailable for whale-fit shaping",
            severity="penalty",
            derived_from=["discovery.metrics.ttm_revenue"],
        )

    if gross_margin is not None and gross_margin >= 0.35:
        whale_fit += 4
    if op_margin is not None and op_margin >= 0.08:
        whale_fit += 4
    if gm_change is not None and gm_change > 0:
        whale_fit += 3
    if om_change is not None and om_change > 0:
        whale_fit += 3
    if cfo_margin is not None and cfo_margin >= 0.10:
        whale_fit += 2
    if fcf_margin is not None and fcf_margin >= 0.05:
        whale_fit += 2
    if cash_conversion is not None and cash_conversion >= 1.0:
        whale_fit += 1
    if net_debt is not None and net_debt <= 0:
        whale_fit += 2
    elif (
        net_debt is not None
        and revenue is not None
        and revenue > 0
        and (net_debt / revenue) <= 0.30
    ):
        whale_fit += 1
    if shares_change_4q is not None:
        if shares_change_4q <= 0.03:
            whale_fit += 2
        elif shares_change_4q > 0.08:
            whale_fit -= 2
            flags.append("DILUTION_PRESSURE")
            _add_reason(
                reason_objects,
                reasons,
                reason_code="WHALE_DILUTION_PRESSURE",
                summary="Shares outstanding trend indicates dilution pressure",
                severity="penalty",
                derived_from=["discovery.metrics.shares_change_4q"],
            )
    whale_fit = max(0.0, min(25.0, whale_fit))

    if whale_fit >= 18:
        _add_reason(
            reason_objects,
            reasons,
            reason_code="WHALE_FIT_STRONG",
            summary="Future Whale fit is strong on durability and unit economics",
            severity="boost",
            derived_from=["discovery.rubric.whale_fit", "discovery.metrics"],
        )
    elif whale_fit < 10:
        _add_reason(
            reason_objects,
            reasons,
            reason_code="WHALE_FIT_WEAK",
            summary="Future Whale fit is currently weak",
            severity="penalty",
            derived_from=["discovery.rubric.whale_fit", "discovery.metrics"],
        )

    missing_critical = [name for name in CRITICAL_METRICS if metrics.get(name) == UNKNOWN]
    coverage -= min(10.0, float(len(missing_critical) * 2))
    if missing_critical:
        flags.append("FINANCIALS_INCOMPLETE")
    if not isinstance(market_cap, (int, float)):
        coverage -= 4
        flags.append("MARKET_CAP_UNKNOWN")
    if isinstance(market_cap, (int, float)) and market_cap_in_band:
        cap_band += 2.0
        _add_reason(
            reason_objects,
            reasons,
            reason_code="CAP_BAND_IN_TARGET",
            summary="Market cap is inside 5B-50B target band",
            severity="boost",
            derived_from=["discovery.metrics.market_cap", "discovery.rubric.cap_band"],
        )
    elif isinstance(market_cap, (int, float)) and not market_cap_in_band:
        cap_band -= 10.0
        flags.append("MARKET_CAP_OUT_OF_BAND")
        _add_reason(
            reason_objects,
            reasons,
            reason_code="CAP_BAND_OUT_OF_TARGET",
            summary="Market cap is outside 5B-50B target band",
            severity="penalty",
            derived_from=["discovery.metrics.market_cap", "discovery.rubric.cap_band"],
        )
    else:
        cap_band -= 4.0
        _add_reason(
            reason_objects,
            reasons,
            reason_code="CAP_BAND_UNKNOWN",
            summary="Market cap unavailable; advancement confidence reduced",
            severity="penalty",
            derived_from=["discovery.metrics.market_cap", "discovery.rubric.cap_band"],
        )
    coverage = max(0.0, coverage)

    total = round(
        max(0.0, min(100.0, quality + inflection + valuation + coverage + whale_fit + cap_band)), 2
    )
    subscores = {
        "quality_durability": round(quality, 2),
        "inflection_signals": round(inflection, 2),
        "valuation_plausibility": round(valuation, 2),
        "coverage_penalty_component": round(coverage, 2),
        "whale_fit": round(whale_fit, 2),
        "cap_band": round(cap_band, 2),
    }
    ordered_reasons: list[str] = []
    for item in reasons:
        if item not in ordered_reasons:
            ordered_reasons.append(item)
    return DiscoveryScoreResult(
        total_score=total,
        subscores=subscores,
        reasons=ordered_reasons,
        reason_objects=reason_objects,
        flags=sorted(set(flags)),
    )
