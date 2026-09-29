"""Thesis updater — maps evidence findings to calibrated valuation adjustments.

Takes Task 5 evidence results + scorecard valuations and produces method-specific,
evidence-traced ThesisAdjustments using calibrated bands.

Public API: update_thesis(ticker, scorecard, tensions, evidence_results, iteration)

Pure function. No DB calls. No file I/O. No LLM calls.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Any

from app.research.evidence_searcher import EvidenceResult
from app.valuation.mos_conventions import graham_value_from_textbook_discount

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ValuationInputs:
    """Extracted from scorecard — the raw numbers needed for sensitivity math.

    Note: graham is back-computed from discounts.graham + price.
    It is supplementary, not a primary method output.
    """
    dcf: float | None = None
    epv: float | None = None
    graham: float | None = None
    price: float | None = None
    wacc: float | None = None
    terminal_growth: float | None = None


@dataclass
class ThesisAdjustment:
    """One directional adjustment to a specific valuation method, traced to evidence.

    Every record represents an actual valuation adjustment. Sources without
    clear valuation mechanics do NOT produce ThesisAdjustment records.
    """
    hypothesis_source: str
    hypothesis_claim: str
    hypothesis_direction: str
    hypothesis_status: str
    affected_method: str         # "dcf" / "epv" — always single method
    adjustment_magnitude: float  # signed $/share (negative = bearish)
    adjustment_confidence: str   # FACT_CALIBRATED / HEURISTIC
    calibration_detail: str
    evidence_item_ids: list[str]
    structured_facts_used: list[str]


@dataclass
class UnresolvedEvidence:
    """An evidence item that was not resolved — preserves Task 5 status semantics."""
    need_id: str
    description: str
    importance: str
    unresolved_reason: str       # NOT_FOUND / INCONCLUSIVE / UNCLASSIFIED
    hypothesis_source: str
    hypothesis_priority: str
    hypothesis_direction: str


@dataclass
class ThesisResult:
    """Complete thesis output after evidence-based adjustment."""
    ticker: str
    iteration: int
    status: str                    # OK / NO_EVIDENCE / NO_VALUATION

    original_dcf: float | None
    original_epv: float | None
    original_graham: float | None
    current_price: float | None

    adjustments: list[ThesisAdjustment]

    adjusted_dcf: float | None
    adjusted_epv: float | None
    adjusted_intrinsic_mid: float | None
    adjusted_margin_of_safety: float | None

    hypotheses_confirmed: int
    hypotheses_contradicted: int
    hypotheses_partially_confirmed: int
    hypotheses_inconclusive: int
    hypotheses_unclassified: int
    average_coverage: float

    unresolved: list[UnresolvedEvidence]
    high_priority_unresolved: int

    # FIX 5: True when adjusted_dcf/adjusted_epv was truncated from a negative
    # raw value to the 0.0 floor (defaults False to preserve existing callers).
    adjusted_value_floored: bool = False


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------

_PCT_RE = re.compile(r"(-?[\d.]+)\s*%")
_PP_RE = re.compile(r"(-?[\d.]+)\s*pp", re.IGNORECASE)
_RATIO_RE = re.compile(r"(-?[\d.]+)\s*x", re.IGNORECASE)
_NUM_RE = re.compile(r"-?[\d.]+")


def _parse_fact(fact_str: str | None, expected: str) -> float | None:
    """Extract numeric value from structured_fact strings.

    expected: "pct" (-2% -> -0.02, 35% -> 0.35), "pp" (-3pp -> -3.0),
              "ratio" (3.5x -> 3.5), "years" (4 -> 4.0)
    Returns None if unparseable.
    """
    if not fact_str or not fact_str.strip():
        return None

    s = fact_str.strip()
    try:
        if expected == "pct":
            m = _PCT_RE.search(s)
            if m:
                return float(m.group(1)) / 100.0
            # Try bare decimal (already a fraction)
            val = float(s)
            return val if val <= 1.0 else val / 100.0
        elif expected == "pp":
            m = _PP_RE.search(s)
            if m:
                return float(m.group(1))
            return float(s)
        elif expected == "ratio":
            m = _RATIO_RE.search(s)
            if m:
                return float(m.group(1))
            return float(s)
        elif expected == "years":
            m = _NUM_RE.search(s)
            if m:
                return float(m.group())
            return None
        return float(s)
    except (ValueError, TypeError):
        return None


def _dcf_growth_impact(growth_haircut_pp: float, inputs: ValuationInputs) -> float:
    """$/share DCF impact from a growth rate reduction.

    Formula: dcf * haircut_pp * 0.01 / (wacc - terminal_growth)
    Returns 0.0 if inputs are missing or spread <= 0.005.
    """
    if inputs.dcf is None or inputs.wacc is None or inputs.terminal_growth is None:
        return 0.0
    spread = inputs.wacc - inputs.terminal_growth
    if spread <= 0.005:
        return 0.0
    return inputs.dcf * growth_haircut_pp * 0.01 / spread


def _epv_margin_impact(margin_haircut_pp: float, inputs: ValuationInputs) -> float:
    """$/share EPV impact from an operating margin reduction.

    Formula: epv * haircut_pp * 0.01 / wacc
    Returns 0.0 if inputs are missing or wacc <= 0.
    """
    if inputs.epv is None or inputs.wacc is None or inputs.wacc <= 0:
        return 0.0
    return inputs.epv * margin_haircut_pp * 0.01 / inputs.wacc


def _extract_valuation_inputs(scorecard: dict[str, Any]) -> ValuationInputs:
    """Extract the raw numbers Task 6 needs from a scorecard dict."""
    pzd = scorecard.get("pricing_zone_detail") or {}
    discounts = scorecard.get("discounts") or {}
    wacc_detail = scorecard.get("wacc_detail") or {}
    price = pzd.get("current_price")
    if isinstance(price, str):
        try:
            price = float(price)
        except (ValueError, TypeError):
            price = None

    # Graham: invert the TEXTBOOK discount d=(iv-price)/iv => iv = price/(1-d)
    # (audit: graham-discount-inversion)
    graham = graham_value_from_textbook_discount(
        price if isinstance(price, (int, float)) else None,
        discounts.get("graham") if isinstance(discounts.get("graham"), (int, float)) else None,
    )

    return ValuationInputs(
        dcf=pzd.get("dcf_base"),
        epv=pzd.get("epv_adjusted"),
        graham=graham,
        price=price,
        wacc=wacc_detail.get("adjusted_wacc"),
        terminal_growth=pzd.get("terminal_growth_used"),
    )


def _find_matching_item(
    er: EvidenceResult,
    need_id_fragment: str,
) -> Any:
    """Find a CONFIRMS evidence item by need_id fragment.

    Only returns items with status == CONFIRMS. CONTRADICTS items should
    not drive calibration or modulation — contradictory item-level evidence
    does not support adjustment sizing even when the hypothesis overall is
    CONFIRMED (other items may have confirmed it).

    Returns the first matching CONFIRMS EvidenceItemResult or None.
    """
    for item in er.evidence_item_results:
        if need_id_fragment not in item.need_id:
            continue
        if item.status == "CONFIRMS":
            return item
    return None


def _extract_fact_from_items(
    er: EvidenceResult,
    need_id_fragment: str,
    expected_type: str,
) -> float | None:
    """Find an evidence item by need_id fragment, parse its structured_fact.

    Precedence: CONFIRMS > CONTRADICTS, first in list order.
    """
    best = _find_matching_item(er, need_id_fragment)
    if best is None or not best.structured_fact:
        return None
    return _parse_fact(best.structured_fact, expected_type)


# ---------------------------------------------------------------------------
# Calibrator Helpers
# ---------------------------------------------------------------------------

_TRIGGER_STATUSES = {"CONFIRMED", "PARTIALLY_CONFIRMED"}


def _build_adjustment(
    er: EvidenceResult,
    method: str,
    magnitude: float,
    confidence: str,
    detail: str,
    *,
    used_item_ids: list[str] | None = None,
    used_facts: list[str] | None = None,
) -> ThesisAdjustment:
    """Build a ThesisAdjustment from common fields.

    used_item_ids and used_facts should contain ONLY the specific evidence
    items and facts that the calibrator actually consumed for this adjustment.
    Do not include unrelated items from the same hypothesis.
    """
    return ThesisAdjustment(
        hypothesis_source=er.hypothesis.source,
        hypothesis_claim=er.hypothesis.claim,
        hypothesis_direction=er.hypothesis.direction,
        hypothesis_status=er.hypothesis_status,
        affected_method=method,
        adjustment_magnitude=magnitude,
        adjustment_confidence=confidence,
        calibration_detail=detail,
        evidence_item_ids=used_item_ids or [],
        structured_facts_used=used_facts or [],
    )


def _sign(direction: str, magnitude: float) -> float:
    """Apply direction sign: BEARISH -> negative, BULLISH -> positive."""
    if direction == "BEARISH":
        return -abs(magnitude)
    elif direction == "BULLISH":
        return abs(magnitude)
    return 0.0


# ---------------------------------------------------------------------------
# Source Calibrators
# ---------------------------------------------------------------------------

def _calibrate_growth_tension(
    er: EvidenceResult, inputs: ValuationInputs, tensions: dict[str, Any],
) -> list[ThesisAdjustment]:
    """GROWTH_VS_EARNINGS_POWER: customer concentration -> DCF growth haircut."""
    if er.hypothesis_status not in _TRIGGER_STATUSES:
        return []

    is_partial = er.hypothesis_status == "PARTIALLY_CONFIRMED"

    # Priority 1: structured_fact from Task 5
    concentration = _extract_fact_from_items(er, "customer_concentration", "pct")
    if concentration is not None and not is_partial:
        if concentration < 0.20:
            haircut_pp = 1.0
        elif concentration < 0.30:
            haircut_pp = 2.0
        else:
            haircut_pp = 3.0
        mag = _dcf_growth_impact(haircut_pp, inputs)
        fact_item = _find_matching_item(er, "customer_concentration")
        return [_build_adjustment(er, "dcf", _sign(er.hypothesis.direction, mag), "FACT_CALIBRATED",
                f"concentration {concentration:.0%} -> {haircut_pp}pp growth haircut -> ${mag:.2f} DCF impact",
                used_item_ids=[fact_item.need_id] if fact_item else [],
                used_facts=[fact_item.structured_fact] if fact_item and fact_item.structured_fact else [])]

    # Priority 2: calibration_context — use growth_dependency_ratio to select band
    cal = er.hypothesis.calibration_context
    if cal and cal.get("growth_dependency_ratio") is not None:
        gdr = cal["growth_dependency_ratio"]
        # Band: 50-65% dependency -> 1.5pp, 65-80% -> 2.5pp, >80% -> 3.5pp
        if gdr < 0.65:
            haircut_pp = 1.5
        elif gdr < 0.80:
            haircut_pp = 2.5
        else:
            haircut_pp = 3.5
        mag = _dcf_growth_impact(haircut_pp, inputs)
        return [_build_adjustment(er, "dcf", _sign(er.hypothesis.direction, mag), "HEURISTIC",
                f"calibration_context growth_dependency={gdr:.0%} -> {haircut_pp}pp growth haircut -> ${mag:.2f} DCF impact",
                used_item_ids=[], used_facts=[])]

    # Priority 3: heuristic (2pp)
    mag = _dcf_growth_impact(2.0, inputs)
    return [_build_adjustment(er, "dcf", _sign(er.hypothesis.direction, mag), "HEURISTIC",
            f"heuristic 2pp growth haircut -> ${mag:.2f} DCF impact",
            used_item_ids=[], used_facts=[])]


def _calibrate_revenue_decline(
    er: EvidenceResult, inputs: ValuationInputs, tensions: dict[str, Any],
) -> list[ThesisAdjustment]:
    """REVENUE_DECLINE_FROM_PEAK: peak decline % -> DCF growth haircut.

    Primary band input is calibration_context.peak_decline_pct (from anomaly
    detector data, populated by Task 4). Filing evidence from revenue_by_segment
    is NOT used as decline magnitude — segment percentages (e.g., "Cloud was 40%
    of revenue") are composition, not decline. Filing evidence confirms/contradicts
    the hypothesis but does not supply the band input.
    """
    if er.hypothesis_status not in _TRIGGER_STATUSES:
        return []

    # Primary: calibration_context (the anomaly detector already computed decline %)
    cal = er.hypothesis.calibration_context
    if cal and cal.get("peak_decline_pct") is not None:
        decline_pct = cal["peak_decline_pct"]
        if decline_pct < 0.25:
            haircut_pp = 1.0
        elif decline_pct < 0.40:
            haircut_pp = 2.5
        else:
            haircut_pp = 4.0
        mag = _dcf_growth_impact(haircut_pp, inputs)
        return [_build_adjustment(er, "dcf", _sign(er.hypothesis.direction, mag), "HEURISTIC",
                f"calibration_context peak_decline={decline_pct:.0%} -> {haircut_pp}pp -> ${mag:.2f} DCF impact",
                used_item_ids=[], used_facts=[])]

    # Fallback: heuristic (2pp)
    mag = _dcf_growth_impact(2.0, inputs)
    return [_build_adjustment(er, "dcf", _sign(er.hypothesis.direction, mag), "HEURISTIC",
            f"heuristic 2pp growth haircut -> ${mag:.2f} DCF impact",
            used_item_ids=[], used_facts=[])]


def _calibrate_margin_collapse(
    er: EvidenceResult, inputs: ValuationInputs, tensions: dict[str, Any],
) -> list[ThesisAdjustment]:
    """MARGIN_COLLAPSE: margin decline pp -> EPV margin haircut."""
    if er.hypothesis_status not in _TRIGGER_STATUSES:
        return []

    is_partial = er.hypothesis_status == "PARTIALLY_CONFIRMED"

    # Priority 1: structured_fact
    fact_fragment = "segment_margins"
    margin_fact = _extract_fact_from_items(er, fact_fragment, "pp")
    if margin_fact is None:
        fact_fragment = "input_cost_trends"
        margin_fact = _extract_fact_from_items(er, fact_fragment, "pp")

    if margin_fact is not None and not is_partial:
        decline_pp = abs(margin_fact)
        if decline_pp < 5:
            haircut_pp = 1.0
        elif decline_pp < 10:
            haircut_pp = 3.0
        else:
            haircut_pp = 5.0
        mag = _epv_margin_impact(haircut_pp, inputs)
        fact_item = _find_matching_item(er, fact_fragment)
        return [_build_adjustment(er, "epv", _sign(er.hypothesis.direction, mag), "FACT_CALIBRATED",
                f"margin decline {decline_pp:.0f}pp -> {haircut_pp}pp margin haircut -> ${mag:.2f} EPV impact",
                used_item_ids=[fact_item.need_id] if fact_item else [],
                used_facts=[fact_item.structured_fact] if fact_item and fact_item.structured_fact else [])]

    # Priority 2: calibration_context
    cal = er.hypothesis.calibration_context
    if cal and cal.get("margin_decline_pp") is not None:
        decline_pp = abs(cal["margin_decline_pp"])
        if decline_pp < 5:
            haircut_pp = 1.0
        elif decline_pp < 10:
            haircut_pp = 3.0
        else:
            haircut_pp = 5.0
        mag = _epv_margin_impact(haircut_pp, inputs)
        return [_build_adjustment(er, "epv", _sign(er.hypothesis.direction, mag), "HEURISTIC",
                f"calibration_context margin_decline={decline_pp:.0f}pp -> {haircut_pp}pp -> ${mag:.2f} EPV impact",
                used_item_ids=[], used_facts=[])]

    # Priority 3: heuristic (3pp)
    mag = _epv_margin_impact(3.0, inputs)
    return [_build_adjustment(er, "epv", _sign(er.hypothesis.direction, mag), "HEURISTIC",
            f"heuristic 3pp margin haircut -> ${mag:.2f} EPV impact",
            used_item_ids=[], used_facts=[])]


def _calibrate_secular_decline(
    er: EvidenceResult, inputs: ValuationInputs, tensions: dict[str, Any],
) -> list[ThesisAdjustment]:
    """SECULAR_DECLINE: TAM growth -> DCF growth haircut."""
    if er.hypothesis_status not in _TRIGGER_STATUSES:
        return []

    is_partial = er.hypothesis_status == "PARTIALLY_CONFIRMED"

    # Priority 1: structured_fact from projections or tam
    tam_fact = _extract_fact_from_items(er, "projections", "pct")
    if tam_fact is None:
        tam_fact = _extract_fact_from_items(er, "tam", "pct")

    if tam_fact is not None and not is_partial:
        growth_rate = tam_fact
        if growth_rate >= 0:
            haircut_pp = 1.0
        elif growth_rate >= -0.03:
            haircut_pp = 2.5
        else:
            haircut_pp = 4.0
        mag = _dcf_growth_impact(haircut_pp, inputs)
        fact_item = _find_matching_item(er, "projections") or _find_matching_item(er, "tam")
        return [_build_adjustment(er, "dcf", _sign(er.hypothesis.direction, mag), "FACT_CALIBRATED",
                f"TAM growth {growth_rate:.1%} -> {haircut_pp}pp growth haircut -> ${mag:.2f} DCF impact",
                used_item_ids=[fact_item.need_id] if fact_item else [],
                used_facts=[fact_item.structured_fact] if fact_item and fact_item.structured_fact else [])]

    # Priority 2/3: heuristic (3pp — no calibration_context for this source)
    mag = _dcf_growth_impact(3.0, inputs)
    return [_build_adjustment(er, "dcf", _sign(er.hypothesis.direction, mag), "HEURISTIC",
            f"heuristic 3pp growth haircut -> ${mag:.2f} DCF impact",
            used_item_ids=[], used_facts=[])]


def _calibrate_competitive_disruption(
    er: EvidenceResult, inputs: ValuationInputs, tensions: dict[str, Any],
) -> list[ThesisAdjustment]:
    """COMPETITIVE_DISRUPTION: market share change -> DCF growth haircut."""
    if er.hypothesis_status not in _TRIGGER_STATUSES:
        return []

    is_partial = er.hypothesis_status == "PARTIALLY_CONFIRMED"

    share_fact = _extract_fact_from_items(er, "market_share", "pct")

    if share_fact is not None and not is_partial:
        change = share_fact
        if change >= 0:
            haircut_pp = 0.5
        elif change >= -0.15:
            haircut_pp = 2.0
        else:
            haircut_pp = 3.5
        mag = _dcf_growth_impact(haircut_pp, inputs)
        fact_item = _find_matching_item(er, "market_share")
        return [_build_adjustment(er, "dcf", _sign(er.hypothesis.direction, mag), "FACT_CALIBRATED",
                f"market share change {change:.0%} -> {haircut_pp}pp growth haircut -> ${mag:.2f} DCF impact",
                used_item_ids=[fact_item.need_id] if fact_item else [],
                used_facts=[fact_item.structured_fact] if fact_item and fact_item.structured_fact else [])]

    # Heuristic (2pp)
    mag = _dcf_growth_impact(2.0, inputs)
    return [_build_adjustment(er, "dcf", _sign(er.hypothesis.direction, mag), "HEURISTIC",
            f"heuristic 2pp growth haircut -> ${mag:.2f} DCF impact",
            used_item_ids=[], used_facts=[])]


def _dual_method_bands(years: int, dcf_table: list[tuple[int, float]], epv_table: list[tuple[int, float]]) -> tuple[float, float]:
    """Look up DCF and EPV haircuts from year-based band tables."""
    dcf_pp = dcf_table[-1][1]  # default to highest band
    for threshold, pp in dcf_table:
        if years <= threshold:
            dcf_pp = pp
            break
    epv_pp = epv_table[-1][1]
    for threshold, pp in epv_table:
        if years <= threshold:
            epv_pp = pp
            break
    return dcf_pp, epv_pp


_NOE_DCF_BANDS = [(2, 2.0), (3, 3.0), (99, 4.0)]
_NOE_EPV_BANDS = [(2, 3.0), (3, 4.5), (99, 6.0)]


def _calibrate_negative_oe(
    er: EvidenceResult, inputs: ValuationInputs, tensions: dict[str, Any],
) -> list[ThesisAdjustment]:
    """NEGATIVE_OWNER_EARNINGS: years -> DCF growth + EPV margin haircut.

    Base band from calibration_context.negative_oe_years.
    Task 5 facts from capex/sbc modulate the base:
    - growth capex > 60% of total -> reduce DCF haircut 1pp (investment phase)
    - SBC > 10% of revenue -> increase EPV haircut 1pp (earnings quality)
    """
    if er.hypothesis_status not in _TRIGGER_STATUSES:
        return []

    cal = er.hypothesis.calibration_context
    years = None
    if cal and cal.get("negative_oe_years") is not None:
        years = int(cal["negative_oe_years"])

    if years is not None:
        dcf_pp, epv_pp = _dual_method_bands(years, _NOE_DCF_BANDS, _NOE_EPV_BANDS)
        base_detail = f"calibration_context {years}yr negative OE"
    else:
        dcf_pp, epv_pp = 2.5, 3.0
        base_detail = "heuristic (no year count)"

    # Task 5 fact modulation — per-method detail strings
    dcf_detail = base_detail
    epv_detail = base_detail
    dcf_item_ids, dcf_facts = [], []
    epv_item_ids, epv_facts = [], []

    capex_fact = _extract_fact_from_items(er, "capex", "pct")
    if capex_fact is not None and capex_fact > 0.60:
        dcf_pp = max(dcf_pp - 1.0, 1.0)  # reduce, floor at 1pp
        dcf_detail += f"; capex growth {capex_fact:.0%} -> DCF -1pp"
        capex_item = _find_matching_item(er, "capex")
        if capex_item:
            dcf_item_ids.append(capex_item.need_id)
            if capex_item.structured_fact:
                dcf_facts.append(capex_item.structured_fact)

    sbc_fact = _extract_fact_from_items(er, "sbc", "pct")
    if sbc_fact is not None and sbc_fact > 0.10:
        epv_pp = min(epv_pp + 1.0, 6.0)  # increase, cap at 6pp
        epv_detail += f"; SBC {sbc_fact:.0%} -> EPV +1pp"
        sbc_item = _find_matching_item(er, "sbc")
        if sbc_item:
            epv_item_ids.append(sbc_item.need_id)
            if sbc_item.structured_fact:
                epv_facts.append(sbc_item.structured_fact)

    # Always HEURISTIC — base band comes from calibration_context/heuristic,
    # Task 5 facts only modulate severity, not select the band.
    dcf_mag = _dcf_growth_impact(dcf_pp, inputs)
    epv_mag = _epv_margin_impact(epv_pp, inputs)

    return [
        _build_adjustment(er, "dcf", _sign(er.hypothesis.direction, dcf_mag), "HEURISTIC",
                          f"{dcf_detail} -> {dcf_pp}pp growth haircut -> ${dcf_mag:.2f} DCF impact",
                          used_item_ids=dcf_item_ids, used_facts=dcf_facts),
        _build_adjustment(er, "epv", _sign(er.hypothesis.direction, epv_mag), "HEURISTIC",
                          f"{epv_detail} -> {epv_pp}pp margin haircut -> ${epv_mag:.2f} EPV impact",
                          used_item_ids=epv_item_ids, used_facts=epv_facts),
    ]


_PCB_DCF_BANDS = [(2, 2.5), (3, 3.5), (99, 5.0)]
_PCB_EPV_BANDS = [(2, 1.5), (3, 2.5), (99, 3.5)]


def _calibrate_persistent_burn(
    er: EvidenceResult, inputs: ValuationInputs, tensions: dict[str, Any],
) -> list[ThesisAdjustment]:
    """PERSISTENT_CASH_BURN: burn years -> DCF growth + EPV margin haircut.

    Base band from calibration_context.burn_years.
    Task 5 facts from capex_breakdown/r&d modulate the base:
    - R&D increasing AND revenue growing -> reduce both haircuts 1pp (deliberate investment)
    - growth capex < 30% of total -> increase DCF haircut 1pp (no investment in growth)
    """
    if er.hypothesis_status not in _TRIGGER_STATUSES:
        return []

    cal = er.hypothesis.calibration_context
    years = None
    if cal and cal.get("burn_years") is not None:
        years = int(cal["burn_years"])

    if years is not None:
        dcf_pp, epv_pp = _dual_method_bands(years, _PCB_DCF_BANDS, _PCB_EPV_BANDS)
        base_detail = f"calibration_context {years}yr cash burn"
    else:
        dcf_pp, epv_pp = 3.0, 2.0
        base_detail = "heuristic (no year count)"

    # Task 5 fact modulation — per-method detail strings
    dcf_detail = base_detail
    epv_detail = base_detail
    dcf_item_ids, dcf_facts = [], []
    epv_item_ids, epv_facts = [], []

    capex_fact = _extract_fact_from_items(er, "capex_breakdown", "pct")
    if capex_fact is not None and capex_fact < 0.30:
        dcf_pp = min(dcf_pp + 1.0, 5.0)  # increase, cap at 5pp
        dcf_detail += f"; growth capex {capex_fact:.0%} -> DCF +1pp"
        capex_item = _find_matching_item(er, "capex_breakdown")
        if capex_item:
            dcf_item_ids.append(capex_item.need_id)
            if capex_item.structured_fact:
                dcf_facts.append(capex_item.structured_fact)

    # R&D modulation: only reduce haircuts when BOTH capex_breakdown shows
    # growth investment (>= 30%) AND R&D fact confirms.
    rd_fact = _extract_fact_from_items(er, "r&d", "pct")
    capex_shows_growth = capex_fact is not None and capex_fact >= 0.30
    if rd_fact is not None and capex_shows_growth:
        rd_item = _find_matching_item(er, "r&d")
        if rd_item and rd_item.status == "CONFIRMS":
            dcf_pp = max(dcf_pp - 1.0, 1.5)
            epv_pp = max(epv_pp - 1.0, 1.0)
            rd_note = f"; R&D {rd_fact:.0%} + growth capex -> -1pp"
            dcf_detail += rd_note
            epv_detail += rd_note
            dcf_item_ids.append(rd_item.need_id)
            epv_item_ids.append(rd_item.need_id)
            if rd_item.structured_fact:
                dcf_facts.append(rd_item.structured_fact)
                epv_facts.append(rd_item.structured_fact)

    # Always HEURISTIC — base band comes from calibration_context/heuristic,
    # Task 5 facts only modulate severity, not select the band.
    dcf_mag = _dcf_growth_impact(dcf_pp, inputs)
    epv_mag = _epv_margin_impact(epv_pp, inputs)

    return [
        _build_adjustment(er, "dcf", _sign(er.hypothesis.direction, dcf_mag), "HEURISTIC",
                          f"{dcf_detail} -> {dcf_pp}pp growth haircut -> ${dcf_mag:.2f} DCF impact",
                          used_item_ids=dcf_item_ids, used_facts=dcf_facts),
        _build_adjustment(er, "epv", _sign(er.hypothesis.direction, epv_mag), "HEURISTIC",
                          f"{epv_detail} -> {epv_pp}pp margin haircut -> ${epv_mag:.2f} EPV impact",
                          used_item_ids=epv_item_ids, used_facts=epv_facts),
    ]


# ---------------------------------------------------------------------------
# Dispatch Table
# ---------------------------------------------------------------------------

_SOURCE_CALIBRATORS: dict[str, Any] = {
    "GROWTH_VS_EARNINGS_POWER":   _calibrate_growth_tension,
    "REVENUE_DECLINE_FROM_PEAK":  _calibrate_revenue_decline,
    "MARGIN_COLLAPSE":            _calibrate_margin_collapse,
    "SECULAR_DECLINE":            _calibrate_secular_decline,
    "COMPETITIVE_DISRUPTION":     _calibrate_competitive_disruption,
    "NEGATIVE_OWNER_EARNINGS":    _calibrate_negative_oe,
    "PERSISTENT_CASH_BURN":       _calibrate_persistent_burn,
}


# ---------------------------------------------------------------------------
# Adjusted Valuation Computation
# ---------------------------------------------------------------------------

def _compute_adjusted_valuations(
    inputs: ValuationInputs,
    adjustments: list[ThesisAdjustment],
) -> tuple[float | None, float | None, float | None, float | None, bool]:
    """Returns (adjusted_dcf, adjusted_epv, adjusted_intrinsic_mid, adjusted_mos, floored).

    The returned adjusted DCF/EPV are floored at 0.0: a thesis adjustment can never
    drive intrinsic value negative. A deeply-bearish adjustment that *would* have
    produced a negative raw value is truncated to 0.0, and ``floored`` is set True so
    downstream consumers can distinguish "genuinely ~zero intrinsic value" from
    "truncated from a much more negative number". The floored value itself is NOT
    changed — only the flag is added (FIX 5).
    """
    dcf_delta = sum(a.adjustment_magnitude for a in adjustments if a.affected_method == "dcf")
    epv_delta = sum(a.adjustment_magnitude for a in adjustments if a.affected_method == "epv")

    raw_dcf = inputs.dcf + dcf_delta if inputs.dcf is not None else None
    raw_epv = inputs.epv + epv_delta if inputs.epv is not None else None

    adj_dcf = max(0.0, raw_dcf) if raw_dcf is not None else None
    adj_epv = max(0.0, raw_epv) if raw_epv is not None else None

    floored = bool(
        (raw_dcf is not None and raw_dcf < 0.0)
        or (raw_epv is not None and raw_epv < 0.0)
    )

    if adj_dcf is not None and adj_epv is not None:
        adj_mid = (adj_dcf + adj_epv) / 2
    elif adj_dcf is not None:
        adj_mid = adj_dcf
    elif adj_epv is not None:
        adj_mid = adj_epv
    else:
        adj_mid = None

    # MoS here is the TEXTBOOK convention: (intrinsic - price) / intrinsic
    # (the fraction of intrinsic value not paid for at the current price).
    # NOTE: this differs from the upside-ratio convention (intrinsic/price - 1)
    # used in graham_dodd.py and intrinsic_discipline.py — same field name,
    # different math. See FIX 6 / deviations.
    adj_mos = None
    if adj_mid is not None and adj_mid > 0 and inputs.price is not None:
        adj_mos = (adj_mid - inputs.price) / adj_mid

    return adj_dcf, adj_epv, adj_mid, adj_mos, floored


# ---------------------------------------------------------------------------
# Unresolved Evidence Collection
# ---------------------------------------------------------------------------

_UNRESOLVED_STATUSES = {"NOT_FOUND", "INCONCLUSIVE", "UNCLASSIFIED"}


def _collect_unresolved(evidence_results: list[EvidenceResult]) -> tuple[list[UnresolvedEvidence], int]:
    """Collect unresolved evidence items and count high-priority ones."""
    unresolved: list[UnresolvedEvidence] = []
    high_count = 0

    for er in evidence_results:
        for item in er.evidence_item_results:
            if item.status in _UNRESOLVED_STATUSES:
                u = UnresolvedEvidence(
                    need_id=item.need_id,
                    description=item.needed,
                    importance=item.importance,
                    unresolved_reason=item.status,
                    hypothesis_source=er.hypothesis.source,
                    hypothesis_priority=er.hypothesis.priority,
                    hypothesis_direction=er.hypothesis.direction,
                )
                unresolved.append(u)
                if er.hypothesis.priority == "HIGH":
                    high_count += 1

    return unresolved, high_count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_STATUS_COUNTS = {"CONFIRMED", "CONTRADICTED", "PARTIALLY_CONFIRMED", "INCONCLUSIVE", "UNCLASSIFIED"}


def update_thesis(
    ticker: str,
    scorecard: dict[str, Any],
    tensions: dict[str, Any],
    evidence_results: list[EvidenceResult],
    iteration: int = 0,
) -> ThesisResult:
    """Map evidence findings to calibrated valuation adjustments.

    Pure function. No DB calls, no file I/O, no LLM.
    """
    inputs = _extract_valuation_inputs(scorecard)

    # Edge case: no evidence
    if not evidence_results:
        adj_dcf, adj_epv, adj_mid, adj_mos, adj_floored = _compute_adjusted_valuations(inputs, [])
        return ThesisResult(
            ticker=ticker, iteration=iteration, status="NO_EVIDENCE",
            original_dcf=inputs.dcf, original_epv=inputs.epv,
            original_graham=inputs.graham, current_price=inputs.price,
            adjustments=[], adjusted_dcf=adj_dcf, adjusted_epv=adj_epv,
            adjusted_intrinsic_mid=adj_mid, adjusted_margin_of_safety=adj_mos,
            hypotheses_confirmed=0, hypotheses_contradicted=0,
            hypotheses_partially_confirmed=0, hypotheses_inconclusive=0,
            hypotheses_unclassified=0, average_coverage=0.0,
            unresolved=[], high_priority_unresolved=0,
            adjusted_value_floored=adj_floored,
        )

    # Edge case: no valuation methods available
    if inputs.dcf is None and inputs.epv is None:
        unresolved, high_count = _collect_unresolved(evidence_results)
        counts = _count_statuses(evidence_results)
        return ThesisResult(
            ticker=ticker, iteration=iteration, status="NO_VALUATION",
            original_dcf=None, original_epv=None,
            original_graham=inputs.graham, current_price=inputs.price,
            adjustments=[], adjusted_dcf=None, adjusted_epv=None,
            adjusted_intrinsic_mid=None, adjusted_margin_of_safety=None,
            **counts, average_coverage=_avg_coverage(evidence_results),
            unresolved=unresolved, high_priority_unresolved=high_count,
        )

    # Map evidence to adjustments
    adjustments: list[ThesisAdjustment] = []
    for er in evidence_results:
        calibrator = _SOURCE_CALIBRATORS.get(er.hypothesis.source)
        if calibrator is not None:
            adjustments.extend(calibrator(er, inputs, tensions))

    # Compute adjusted valuations
    adj_dcf, adj_epv, adj_mid, adj_mos, adj_floored = _compute_adjusted_valuations(inputs, adjustments)

    # Collect unresolved evidence
    unresolved, high_count = _collect_unresolved(evidence_results)

    # Count statuses
    counts = _count_statuses(evidence_results)

    return ThesisResult(
        ticker=ticker, iteration=iteration, status="OK",
        original_dcf=inputs.dcf, original_epv=inputs.epv,
        original_graham=inputs.graham, current_price=inputs.price,
        adjustments=adjustments, adjusted_dcf=adj_dcf, adjusted_epv=adj_epv,
        adjusted_intrinsic_mid=adj_mid, adjusted_margin_of_safety=adj_mos,
        **counts, average_coverage=_avg_coverage(evidence_results),
        unresolved=unresolved, high_priority_unresolved=high_count,
        adjusted_value_floored=adj_floored,
    )


def _count_statuses(evidence_results: list[EvidenceResult]) -> dict[str, int]:
    """Count hypothesis statuses across all evidence results."""
    counts = {
        "hypotheses_confirmed": 0,
        "hypotheses_contradicted": 0,
        "hypotheses_partially_confirmed": 0,
        "hypotheses_inconclusive": 0,
        "hypotheses_unclassified": 0,
    }
    for er in evidence_results:
        key = f"hypotheses_{er.hypothesis_status.lower()}"
        if key in counts:
            counts[key] += 1
    return counts


def _avg_coverage(evidence_results: list[EvidenceResult]) -> float:
    """Mean of coverage_score across hypotheses."""
    if not evidence_results:
        return 0.0
    return sum(er.coverage_score for er in evidence_results) / len(evidence_results)
