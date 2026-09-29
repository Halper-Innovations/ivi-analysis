"""Deterministic hypothesis generation from valuation tensions, anomalies, and quality signals.

Converts quantitative patterns (method disagreements, anomalies, filing flags) into
testable investment claims with evidence needs and falsification criteria.
No LLM calls — purely template-based.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.alpha.schemas import Anomaly, SolvencyAssessment


@dataclass
class EvidenceNeed:
    """A specific piece of evidence needed to evaluate a hypothesis."""
    need_id: str            # stable identifier e.g. "growth_ep_001_customer_concentration"
    description: str        # "customer concentration data"
    importance: str         # REQUIRED / IMPORTANT / SUPPORTING


@dataclass
class Hypothesis:
    """A testable investment claim derived from quantitative signals."""
    claim: str
    direction: str          # BEARISH / BULLISH / NEUTRAL
    evidence_needed: list[EvidenceNeed]
    falsification: str
    priority: str           # HIGH / MODERATE / LOW
    source: str             # trigger identifier
    impact_estimate: float | None = None  # $/share; None when not computable
    calibration_context: dict[str, Any] | None = None  # source-specific calibration inputs for Task 6


# ---------------------------------------------------------------------------
# Tier 1: Method Tensions
# ---------------------------------------------------------------------------

def _tier1_method_tensions(tensions: dict[str, Any], valuation: dict[str, Any]) -> list[Hypothesis]:
    """Generate hypotheses from cross-method valuation disagreements."""
    if tensions.get("tension_type") == "INSUFFICIENT_METHODS":
        return []

    results: list[Hypothesis] = []
    mv = tensions.get("method_values", {})
    dcf = mv.get("dcf")
    epv = mv.get("epv")
    ncav = mv.get("ncav")
    pzd = valuation.get("pricing_zone_detail") or {}
    price = pzd.get("current_price")
    growth_dep = tensions.get("growth_value_pct") or tensions.get("sensitivity", {}).get("growth_dependency_ratio") or 0

    # GROWTH_VS_EARNINGS_POWER
    if tensions.get("tension_type") == "GROWTH_VS_EARNINGS_POWER" and growth_dep > 0.5:
        gap = (dcf or 0) - (epv or 0)
        results.append(Hypothesis(
            claim=(
                f"DCF (${dcf:.0f}) assumes growth, but EPV (${epv:.0f}) values current earnings only. "
                f"${gap:.0f}/share ({growth_dep:.0%}) depends on sustained growth."
            ),
            direction="BEARISH",
            evidence_needed=[
                EvidenceNeed("growth_ep_001_revenue_segment", "revenue by segment", "REQUIRED"),
                EvidenceNeed("growth_ep_002_customer_concentration", "customer concentration data", "REQUIRED"),
                EvidenceNeed("growth_ep_003_mgmt_guidance", "management guidance on growth", "IMPORTANT"),
                EvidenceNeed("growth_ep_004_retention", "retention metrics", "SUPPORTING"),
            ],
            falsification="If revenue growth is sustained for 3+ years AND customer concentration < 25%, growth assumption is defensible.",
            priority="HIGH",
            source="GROWTH_VS_EARNINGS_POWER",
            impact_estimate=round(gap, 2) if gap > 0 else None,
            calibration_context={"growth_dependency_ratio": round(growth_dep, 3), "dcf_epv_gap": round(gap, 2)},
        ))

    # ASSET_VS_EARNINGS
    if tensions.get("tension_type") == "ASSET_VS_EARNINGS" and ncav is not None:
        earnings_methods = {k: v for k, v in mv.items() if k != "ncav"}
        earnings_avg = sum(earnings_methods.values()) / max(1, len(earnings_methods)) if earnings_methods else 0
        impact = round(ncav - earnings_avg, 2) if earnings_avg else None
        results.append(Hypothesis(
            claim=(
                f"NCAV (${ncav:.0f}) exceeds earnings-derived estimates (avg ${earnings_avg:.0f}). "
                f"Company may be worth more in liquidation, or earnings are temporarily depressed."
            ),
            direction="BEARISH",
            evidence_needed=[
                EvidenceNeed("asset_earn_001_asset_composition", "asset composition (tangible vs intangible)", "REQUIRED"),
                EvidenceNeed("asset_earn_002_inventory_quality", "inventory quality", "IMPORTANT"),
                EvidenceNeed("asset_earn_003_receivables_aging", "receivables aging", "IMPORTANT"),
                EvidenceNeed("asset_earn_004_book_market_value", "book vs market value of assets", "SUPPORTING"),
            ],
            falsification="If operating income is positive and growing for 2+ years, earnings depression is likely temporary.",
            priority="MODERATE",
            source="ASSET_VS_EARNINGS",
            impact_estimate=impact,
        ))

    # UNANIMOUS UNDERVALUATION — requires >=2 methods agreeing
    method_count = tensions.get("method_count", 0)
    undervalued_count = tensions.get("undervalued_count", 0)
    if (tensions.get("consensus_direction") == "UNDERVALUED"
            and method_count >= 2 and undervalued_count >= 2):
        results.append(Hypothesis(
            claim=(
                f"{undervalued_count} methods agree on undervaluation. "
                f"Look for hidden risks the numbers don't show."
            ),
            direction="BULLISH",
            evidence_needed=[
                EvidenceNeed("unanimous_001_competitive", "competitive position", "REQUIRED"),
                EvidenceNeed("unanimous_002_mgmt_quality", "management quality", "IMPORTANT"),
                EvidenceNeed("unanimous_003_regulatory", "regulatory exposure", "IMPORTANT"),
                EvidenceNeed("unanimous_004_concentration", "customer concentration", "SUPPORTING"),
            ],
            falsification="If filing reveals HIGH competitive_disruption or going concern language, the discount may be a value trap.",
            priority="MODERATE",
            source="UNANIMOUS_UNDERVALUATION",
        ))

    # PLAUSIBLE UNDERVALUATION — DCF and/or EPV above price but consensus is not UNDERVALUED.
    # This is the fallback for clean value candidates that don't cross the unanimous threshold.
    if not any(h.source == "UNANIMOUS_UNDERVALUATION" for h in results):
        dcf_undervalued = dcf is not None and price is not None and dcf > price
        epv_undervalued = epv is not None and price is not None and epv > price
        if dcf_undervalued or epv_undervalued:
            methods_above = []
            if dcf_undervalued:
                methods_above.append(f"DCF ${dcf:.0f}")
            if epv_undervalued:
                methods_above.append(f"EPV ${epv:.0f}")
            results.append(Hypothesis(
                claim=(
                    f"{' and '.join(methods_above)} above price ${price:.0f}. "
                    f"Verify the discount is real and not explained by hidden risks."
                ),
                direction="BULLISH",
                evidence_needed=[
                    EvidenceNeed("plausible_001_revenue_sustainability", "revenue sustainability and growth drivers", "REQUIRED"),
                    EvidenceNeed("plausible_002_competitive_position", "competitive position and moat", "REQUIRED"),
                    EvidenceNeed("plausible_003_customer_concentration", "customer concentration data", "IMPORTANT"),
                    EvidenceNeed("plausible_004_mgmt_capital_allocation", "management capital allocation track record", "SUPPORTING"),
                ],
                falsification="If filing reveals customer concentration >30%, declining revenue, or competitive disruption, the discount may be warranted.",
                priority="HIGH",
                source="PLAUSIBLE_UNDERVALUATION",
            ))

    # MARKET PREMIUM — price above DCF but gate says PROCEED
    if dcf is not None and price is not None and price > dcf and dcf > 0:
        gate = str(pzd.get("gate_action") or "")
        if gate.upper() == "PROCEED":
            results.append(Hypothesis(
                claim=(
                    f"Market prices in more growth than the DCF model "
                    f"(price ${price:.0f} vs DCF ${dcf:.0f}). Implied growth exceeds model assumption."
                ),
                direction="NEUTRAL",
                evidence_needed=[
                    EvidenceNeed("mkt_prem_001_reverse_dcf", "reverse DCF implied growth rate", "REQUIRED"),
                    EvidenceNeed("mkt_prem_002_market_expectations", "market expectations", "SUPPORTING"),
                    EvidenceNeed("mkt_prem_003_peer_multiples", "peer multiples", "SUPPORTING"),
                ],
                falsification="If implied growth rate < industry median, premium may be justified by quality.",
                priority="LOW",
                source="MARKET_PREMIUM",
            ))

    return results


# ---------------------------------------------------------------------------
# Tier 2: Anomalies
# ---------------------------------------------------------------------------

_ANOMALY_TEMPLATES: dict[str, dict[str, Any]] = {
    "Q4_EARNINGS_BOMB": {
        "priority": "HIGH",
        "evidence_needed": [
            "Q4 press release",
            "restructuring charges",
            "goodwill writedown disclosures",
            "MD&A discussion of Q4",
        ],
        "falsification": "If Q4 loss is from a one-time writedown or restructuring charge with no recurring impact.",
        "transient": "a one-time writedown",
        "structural": "recurring operational losses",
    },
    "MARGIN_COLLAPSE": {
        "priority": "HIGH",
        "evidence_needed": [
            "segment margins",
            "input cost trends",
            "pricing power indicators",
            "competitive dynamics in MD&A",
        ],
        "falsification": "If margin decline is < 200bps AND attributable to a specific one-time cost.",
        "transient": "a temporary cost spike",
        "structural": "permanent competitive pressure",
    },
    "REVENUE_DECLINE_FROM_PEAK": {
        "priority": "MODERATE",
        "evidence_needed": [
            "revenue by segment",
            "geographic breakdown",
            "backlog/order data",
            "management guidance",
        ],
        "falsification": "If decline is < 10% AND concentrated in one segment with recovery indicators.",
        "transient": "a cyclical trough",
        "structural": "secular decline in core market",
    },
    "DEBT_SPIKE": {
        "priority": "MODERATE",
        "evidence_needed": [
            "debt maturity schedule",
            "interest coverage",
            "purpose of borrowing (acquisition vs operations)",
            "covenant compliance",
        ],
        "falsification": "If debt increase funded an accretive acquisition AND interest coverage > 3x.",
        "transient": "strategic acquisition financing",
        "structural": "distress borrowing to fund operations",
    },
    "PERSISTENT_CASH_BURN": {
        "priority": "HIGH",
        "evidence_needed": [
            "capex breakdown (growth vs maintenance)",
            "R&D as % of revenue trend",
            "management commentary on path to profitability",
        ],
        "falsification": "If cash burn is declining YoY AND driven by R&D in a growing market.",
        "transient": "an investment phase",
        "structural": "fundamental inability to generate cash",
    },
    "NEGATIVE_EQUITY": {
        "priority": "MODERATE",
        "evidence_needed": [
            "equity composition (retained earnings vs accumulated deficit)",
            "share buyback history",
            "intangible asset valuation",
        ],
        "falsification": "If negative equity is due to aggressive buybacks AND operating income is positive and growing.",
        "transient": "aggressive capital returns",
        "structural": "accumulated losses eroding the balance sheet",
    },
    "INTANGIBLE_ASSET_JUMP": {
        "priority": "LOW",
        "evidence_needed": [
            "acquisition disclosures",
            "goodwill allocation",
            "impairment testing methodology",
            "purchase price allocation",
        ],
        "falsification": "If intangible jump is from a disclosed acquisition AND goodwill is < 50% of total intangibles.",
        "transient": "a well-priced acquisition",
        "structural": "serial overpayment for acquisitions",
    },
    "WORKING_CAPITAL_CRISIS": {
        "priority": "HIGH",
        "evidence_needed": [
            "receivables aging",
            "inventory turnover trend",
            "payables stretching",
            "credit facility availability",
        ],
        "falsification": "If working capital deterioration is seasonal AND reverts within 2 quarters historically.",
        "transient": "seasonal working capital needs",
        "structural": "deteriorating collections and inventory management",
    },
}

_IMPACT_COMPUTABLE = {
    "Q4_EARNINGS_BOMB": lambda data: data.get("q4_implied"),
    "MARGIN_COLLAPSE": lambda data: -(data.get("decline_pp") or 0),
}


_IMPORTANCE_ORDER = ["REQUIRED", "IMPORTANT", "IMPORTANT", "SUPPORTING", "SUPPORTING"]


def _make_evidence_needs(slug_prefix: str, descriptions: list[str]) -> list[EvidenceNeed]:
    """Convert a list of evidence description strings to EvidenceNeed objects."""
    needs: list[EvidenceNeed] = []
    for i, desc in enumerate(descriptions):
        slug = desc.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")[:30]
        imp = _IMPORTANCE_ORDER[i] if i < len(_IMPORTANCE_ORDER) else "SUPPORTING"
        needs.append(EvidenceNeed(f"{slug_prefix}_{i + 1:03d}_{slug}", desc, imp))
    return needs


def _tier2_anomalies(anomalies: list[Anomaly]) -> list[Hypothesis]:
    """Generate one hypothesis per anomaly using templates."""
    results: list[Hypothesis] = []
    for anomaly in anomalies:
        template = _ANOMALY_TEMPLATES.get(anomaly.anomaly_type)
        slug_prefix = anomaly.anomaly_type.lower()
        if template is None:
            results.append(Hypothesis(
                claim=f"{anomaly.description} Is this transient or structural?",
                direction="BEARISH",
                evidence_needed=[
                    EvidenceNeed(f"{slug_prefix}_001_mda", "filing MD&A section", "REQUIRED"),
                    EvidenceNeed(f"{slug_prefix}_002_mgmt", "management commentary", "IMPORTANT"),
                ],
                falsification="Requires manual investigation — no template available.",
                priority="MODERATE",
                source=anomaly.anomaly_type,
            ))
            continue

        claim = f"{anomaly.description} Is this {template['transient']} or {template['structural']}?"

        impact_fn = _IMPACT_COMPUTABLE.get(anomaly.anomaly_type)
        impact = impact_fn(anomaly.data) if impact_fn else None
        if impact is not None:
            impact = round(impact, 2)

        # Build calibration_context from anomaly data
        cal_ctx = None
        atype = anomaly.anomaly_type
        if atype == "REVENUE_DECLINE_FROM_PEAK":
            peak = anomaly.data.get("peak_rev")
            current = anomaly.data.get("current_rev")
            if peak and current and peak > 0:
                cal_ctx = {"peak_decline_pct": round(1 - current / peak, 3)}
        elif atype == "MARGIN_COLLAPSE":
            decline_pp = anomaly.data.get("decline_pp")
            if decline_pp is not None:
                cal_ctx = {"margin_decline_pp": round(float(decline_pp), 1)}
        elif atype == "PERSISTENT_CASH_BURN":
            neg_years = anomaly.data.get("neg_years")
            if neg_years is not None:
                cal_ctx = {"burn_years": int(neg_years)}

        results.append(Hypothesis(
            claim=claim,
            direction="BEARISH",
            evidence_needed=_make_evidence_needs(slug_prefix, list(template["evidence_needed"])),
            falsification=template["falsification"],
            priority=template["priority"],
            source=anomaly.anomaly_type,
            impact_estimate=impact,
            calibration_context=cal_ctx,
        ))

    return results


# ---------------------------------------------------------------------------
# Tier 3: Filing Risk + Quality Gate Signals
# ---------------------------------------------------------------------------

def _tier3_filing_and_gate(
    quality_ctx: dict[str, Any],
    valuation: dict[str, Any],
    filing_risk: dict[str, Any] | None,
    solvency: SolvencyAssessment | None,
) -> list[Hypothesis]:
    """Generate hypotheses from filing risk signals and quality gate flags."""
    results: list[Hypothesis] = []

    # Filing risk: competitive disruption
    if filing_risk and filing_risk.get("competitive_disruption") == "HIGH":
        results.append(Hypothesis(
            claim="Filing acknowledges existential competitive threat. Moat may be eroding.",
            direction="BEARISH",
            evidence_needed=[
                EvidenceNeed("comp_disrupt_001_competitor", "competitor analysis in 10-K", "REQUIRED"),
                EvidenceNeed("comp_disrupt_002_market_share", "market share data", "IMPORTANT"),
                EvidenceNeed("comp_disrupt_003_tech_risk", "technology displacement risk", "IMPORTANT"),
                EvidenceNeed("comp_disrupt_004_response", "management response plan", "SUPPORTING"),
            ],
            falsification="If company has > 30% market share AND R&D/revenue > industry median.",
            priority="HIGH",
            source="COMPETITIVE_DISRUPTION",
        ))

    # Filing risk: secular decline
    if filing_risk and filing_risk.get("secular_decline") == "HIGH":
        results.append(Hypothesis(
            claim="Filing signals structural market decline. DCF discount may exist because of this risk, not despite it.",
            direction="BEARISH",
            evidence_needed=[
                EvidenceNeed("sec_decline_001_projections", "industry growth projections", "REQUIRED"),
                EvidenceNeed("sec_decline_002_diversification", "segment diversification", "IMPORTANT"),
                EvidenceNeed("sec_decline_003_pivot", "pivot strategy", "IMPORTANT"),
                EvidenceNeed("sec_decline_004_tam", "TAM trends", "SUPPORTING"),
            ],
            falsification="If company has growing segments that offset declining ones.",
            priority="HIGH",
            source="SECULAR_DECLINE",
        ))

    # PROCEED gate + SEVERE downside
    gate = str(quality_ctx.get("gate_action") or "")
    pzd = valuation.get("pricing_zone_detail") or {}
    downside = str(pzd.get("downside_risk_class") or "")
    if gate.upper() == "PROCEED" and downside.upper() == "SEVERE":
        results.append(Hypothesis(
            claim="Quality gate says PROCEED but bear case is catastrophic. What's the tail risk?",
            direction="BEARISH",
            evidence_needed=[
                EvidenceNeed("severe_ds_001_bear_case", "bear-case scenario details", "REQUIRED"),
                EvidenceNeed("severe_ds_002_tail_risk", "tail risk exposures", "IMPORTANT"),
                EvidenceNeed("severe_ds_003_insurance", "insurance/hedging", "SUPPORTING"),
                EvidenceNeed("severe_ds_004_contingent", "contingent liabilities", "SUPPORTING"),
            ],
            falsification="If bear-case probability is < 10% based on historical frequency.",
            priority="MODERATE",
            source="PROCEED_SEVERE_DOWNSIDE",
        ))

    # Negative owner earnings (from gate reason codes)
    reason_codes = quality_ctx.get("gate_reason_codes") or []
    if "ZERO_OWNER_EARNINGS" in reason_codes:
        neg_oe_years = quality_ctx.get("negative_oe_years")
        noe_cal_ctx = {"negative_oe_years": int(neg_oe_years)} if neg_oe_years is not None else None
        results.append(Hypothesis(
            claim="The company isn't generating real cash (CFO < capex + SBC). Is this temporary (investment phase) or structural?",
            direction="BEARISH",
            evidence_needed=[
                EvidenceNeed("neg_oe_001_capex", "capex breakdown (growth vs maintenance)", "REQUIRED"),
                EvidenceNeed("neg_oe_002_sbc", "SBC trend", "IMPORTANT"),
                EvidenceNeed("neg_oe_003_cap_alloc", "management commentary on capital allocation", "SUPPORTING"),
            ],
            falsification="If owner earnings turned positive in the most recent year.",
            priority="HIGH",
            source="NEGATIVE_OWNER_EARNINGS",
            calibration_context=noe_cal_ctx,
        ))

    # Revenue decline + R&D increase
    headwinds = quality_ctx.get("valuation_headwinds") or []
    has_decline = any("DECLINE" in h for h in headwinds) or any("DECLINE" in str(c) for c in reason_codes)
    has_rd_increase = any("R&D" in h.upper() or "RD" in h.upper() for h in quality_ctx.get("valuation_supports") or [])
    if not has_decline:
        has_decline = any("SECULAR_DECLINE" in str(c) for c in reason_codes)
    if has_decline and has_rd_increase:
        results.append(Hypothesis(
            claim="Revenue is falling while R&D rises. Is this a pivot to new markets or a death spiral?",
            direction="NEUTRAL",
            evidence_needed=[
                EvidenceNeed("rev_rd_001_pipeline", "R&D project pipeline", "REQUIRED"),
                EvidenceNeed("rev_rd_002_patents", "patent filings", "IMPORTANT"),
                EvidenceNeed("rev_rd_003_segment_rev", "segment revenue trends", "IMPORTANT"),
                EvidenceNeed("rev_rd_004_rd_roi", "management guidance on R&D ROI", "SUPPORTING"),
            ],
            falsification="If R&D is yielding new product revenue within 2 years AND decline is in legacy segment only.",
            priority="MODERATE",
            source="REVENUE_DECLINE_RD_INCREASE",
        ))

    return results


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

_IMPORTANCE_RANK = {"REQUIRED": 0, "IMPORTANT": 1, "SUPPORTING": 2}


def _merge_evidence_needs(winner_needs: list[EvidenceNeed], loser_needs: list[EvidenceNeed]) -> list[EvidenceNeed]:
    """Merge evidence needs from two hypotheses. Match by description (case-insensitive), keep higher importance."""
    merged: list[EvidenceNeed] = list(winner_needs)
    seen_descs = {n.description.lower() for n in merged}
    for need in loser_needs:
        desc_lower = need.description.lower()
        if desc_lower not in seen_descs:
            merged.append(need)
            seen_descs.add(desc_lower)
        else:
            # Same description — keep higher importance
            for i, existing_need in enumerate(merged):
                if existing_need.description.lower() == desc_lower:
                    if _IMPORTANCE_RANK.get(need.importance, 9) < _IMPORTANCE_RANK.get(existing_need.importance, 9):
                        merged[i] = need
                    break
    return merged


def _deduplicate(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    """Deduplicate by source field. Keep higher priority, merge evidence_needed."""
    priority_rank = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    seen: dict[str, Hypothesis] = {}

    for h in hypotheses:
        if h.source not in seen:
            seen[h.source] = h
            continue

        existing = seen[h.source]
        existing_rank = priority_rank.get(existing.priority, 9)
        new_rank = priority_rank.get(h.priority, 9)

        if new_rank < existing_rank or (new_rank == existing_rank and (h.impact_estimate or 0) > (existing.impact_estimate or 0)):
            winner, loser = h, existing
        else:
            winner, loser = existing, h

        merged_evidence = _merge_evidence_needs(winner.evidence_needed, loser.evidence_needed)
        merged_cal_ctx = winner.calibration_context if winner.calibration_context is not None else loser.calibration_context
        seen[h.source] = Hypothesis(
            claim=winner.claim, direction=winner.direction, evidence_needed=merged_evidence,
            falsification=winner.falsification, priority=winner.priority, source=winner.source,
            impact_estimate=winner.impact_estimate,
            calibration_context=merged_cal_ctx,
        )

    return list(seen.values())


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def _sort_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    """Sort: HIGH > MODERATE > LOW, BEARISH > BULLISH > NEUTRAL, impact desc."""
    priority_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    direction_order = {"BEARISH": 0, "BULLISH": 1, "NEUTRAL": 2}

    def sort_key(h: Hypothesis) -> tuple:
        return (
            priority_order.get(h.priority, 9),
            direction_order.get(h.direction, 9),
            -(h.impact_estimate or 0),
        )

    return sorted(hypotheses, key=sort_key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_hypotheses(
    ticker: str,
    valuation: dict[str, Any],
    tensions: dict[str, Any],
    anomalies: list[Anomaly],
    quality_ctx: dict[str, Any],
    solvency: SolvencyAssessment | None = None,
    filing_risk: dict[str, Any] | None = None,
) -> list[Hypothesis]:
    """Generate testable investment hypotheses from all available signals.

    Returns hypotheses sorted by priority (HIGH first), direction (BEARISH first),
    then impact_estimate descending.
    """
    results: list[Hypothesis] = []
    results.extend(_tier1_method_tensions(tensions, valuation))
    results.extend(_tier2_anomalies(anomalies))
    results.extend(_tier3_filing_and_gate(quality_ctx, valuation, filing_risk, solvency))
    results = _deduplicate(results)
    return _sort_hypotheses(results)
