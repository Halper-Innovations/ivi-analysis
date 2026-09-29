from __future__ import annotations

import math
from typing import Any

from app.fundamentals.normalize import UNKNOWN
from app.score.change_momentum import compute_change_momentum_adjustments
from app.score.research_adjustments import compute_research_adjustments


# Keys the packet builders add to a fundamentals dict that are not measured values:
# identity, provenance containers, and issuer-class labels. They are never "known
# metrics" and must not be counted (or credited) in data completeness.
NON_METRIC_FUNDAMENTALS_KEYS = frozenset(
    {
        "ticker",
        "run_id",
        "as_of_date",
        "rows",
        "row_traces",
        "derived_signals",
        "gaps",
        "issuer_classification",
        "fcf_applicability",
        "_prompt_financial_provenance",
    }
)

# Fields only a bank / lender reports. An operating company is not "missing" them, so
# they leave its completeness denominator (they stay in a financial issuer's).
BANK_ONLY_FUNDAMENTALS_KEYS = frozenset(
    {
        "deposits",
        "loans",
        "investment_securities",
        "assets_under_management",
        "allowance_for_credit_losses",
        "provision_for_credit_losses",
        "net_charge_offs",
        "nonaccrual_loans",
        "deposits_to_assets",
        "loans_to_deposits",
        "allowance_to_loans",
        "provision_to_loans",
        "net_charge_offs_to_loans",
    }
)

# The penalty a known filing-coverage score of 0 earns. A coverage row that is missing
# altogether is at least as bad: unknown must never score better than known bad.
UNKNOWN_FILING_COVERAGE_PENALTY = 5.0


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_known_metric(value: Any) -> bool:
    # NaN and infinity are not measurements: counting them as known inflated
    # data completeness for a packet whose numbers had failed to compute.
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return value is not None and value != UNKNOWN


def clamp(value: float, low: float, high: float) -> float:
    # NaN compares False to everything, so max/min would hand back the HIGH bound and
    # a NaN input would score as the best possible company. It takes the low bound.
    if isinstance(value, float) and math.isnan(value):
        return low
    return max(low, min(high, value))


def score_packet(
    packet: dict[str, Any],
    analyst_decision: dict[str, Any] | None = None,
    research_quality: dict[str, Any] | None = None,
    research_signals: dict[str, Any] | None = None,
    filing_coverage: dict[str, Any] | None = None,
    change_context: dict[str, Any] | None = None,
) -> tuple[dict[str, float], float, str, list[str]]:
    fundamentals = packet.get("fundamentals", {})
    valuations = packet.get("valuations", {})

    reasons: list[str] = []

    metric_items = {k: v for k, v in fundamentals.items() if k not in NON_METRIC_FUNDAMENTALS_KEYS}
    if str(fundamentals.get("issuer_classification") or "").strip().lower() == "operating":
        metric_items = {k: v for k, v in metric_items.items() if k not in BANK_ONLY_FUNDAMENTALS_KEYS}
    known_metrics = sum(1 for v in metric_items.values() if _is_known_metric(v))
    total_metrics = max(1, len(metric_items))
    data_completeness = clamp((known_metrics / total_metrics) * 20.0, 0, 20)
    if data_completeness < 8:
        reasons.append("Low data completeness from filings")

    operating_margin = fundamentals.get("operating_margin")
    fcf_margin = fundamentals.get("fcf_margin")
    quality = 10.0
    if _finite_number(operating_margin):
        quality += clamp(operating_margin * 20.0, -5, 5)
    if _finite_number(fcf_margin):
        quality += clamp(fcf_margin * 20.0, -5, 5)
    business_quality = clamp(quality, 0, 20)

    reverse_dcf = valuations.get("reverse_dcf", {}) if isinstance(valuations, dict) else {}
    reverse_inputs = reverse_dcf.get("inputs", {}) if isinstance(reverse_dcf, dict) else {}
    price_known = (
        reverse_inputs.get("market_price", UNKNOWN) != UNKNOWN
        and reverse_inputs.get("price", UNKNOWN) != UNKNOWN
    ) or reverse_inputs.get("market_price", UNKNOWN) != UNKNOWN or reverse_inputs.get("price", UNKNOWN) != UNKNOWN
    dcf = valuations.get("dcf", {}) if isinstance(valuations, dict) else {}
    dcf_outputs = dcf.get("outputs", {}) if isinstance(dcf, dict) else {}
    if not isinstance(dcf_outputs, dict):
        dcf_outputs = dcf if isinstance(dcf, dict) else {}
    dcf_conf = dcf_outputs.get("confidence")
    if not dcf_conf:
        dcf_status = str(dcf_outputs.get("status") or "").upper()
        if dcf_status == "OK":
            dcf_conf = "MEDIUM"
        elif dcf_status:
            dcf_conf = "LOW"
        else:
            legacy = valuations.get("dcf_lite", {}).get("outputs", {}) if isinstance(valuations, dict) else {}
            dcf_conf = legacy.get("confidence", "LOW") if isinstance(legacy, dict) else "LOW"
    gap_score = 10.0
    if dcf_conf == "MEDIUM":
        gap_score += 8.0
    if dcf_conf == "LOW":
        gap_score -= 4.0
    valuation_gap = clamp(gap_score, 0, 25)
    if not price_known:
        valuation_gap = min(valuation_gap, 12.0)
        reasons.append("Market price UNKNOWN; valuation gap score capped")

    momentum = 7.0
    deltas = packet.get("deltas_vs_prior_period", {})
    rev_delta = deltas.get("revenue", 0)
    fcf_delta = deltas.get("fcf", 0)
    if isinstance(rev_delta, (int, float)) and rev_delta > 0:
        momentum += 4.0
    if isinstance(fcf_delta, (int, float)) and fcf_delta > 0:
        momentum += 4.0
    deterioration_or_improvement = clamp(momentum, 0, 15)

    liquidity = fundamentals.get("liquidity_stress_score", UNKNOWN)
    if _finite_number(liquidity):
        balance_sheet = clamp(10.0 - liquidity, 0, 10)
    else:
        balance_sheet = 4.0
    if balance_sheet < 4:
        reasons.append("Elevated balance sheet / dilution risk")

    # A restatement, an auditor change or a material weakness is evidence against
    # the filings' reliability, never a reason to score a company higher. They used
    # to add 2.5 points each; they now earn nothing here (the change-momentum
    # penalty already charges a newly raised material weakness).
    special_flags = 0.0
    adverse = [
        fact.get("fact_type")
        for fact in packet.get("extracted_facts", [])
        if fact.get("fact_type") in {"restatement", "auditor_change", "material_weakness"}
    ]
    if adverse:
        reasons.append("Adverse accounting events on file: " + ", ".join(sorted(set(adverse))))
    special_situations = clamp(special_flags, 0, 10)

    subscores = {
        "data_completeness": round(data_completeness, 2),
        "business_quality_durability": round(business_quality, 2),
        "valuation_gap": round(valuation_gap, 2),
        "momentum": round(deterioration_or_improvement, 2),
        "balance_sheet_dilution": round(balance_sheet, 2),
        "special_situations": round(special_situations, 2),
    }

    research_penalty = 0.0
    research_overall = None
    if isinstance(research_quality, dict):
        overall = research_quality.get("overall_research_score")
        incomplete = bool(research_quality.get("incomplete"))
        if isinstance(overall, (int, float)):
            research_overall = float(overall)
            subscores["research_overall"] = round(research_overall, 2)
        if incomplete:
            research_penalty = 12.0
            reasons.append("Research incomplete: quality gate below threshold")
            top_gaps = research_quality.get("top_gaps") or []
            if top_gaps:
                first = top_gaps[0]
                if isinstance(first, dict) and first.get("summary"):
                    reasons.append(f"Top evidence gap: {first['summary']}")

    if research_penalty > 0:
        subscores["research_penalty"] = -round(research_penalty, 2)

    research_adjust_subscores, research_signal_net, research_signal_reasons = compute_research_adjustments(research_signals)
    subscores.update(research_adjust_subscores)
    reasons.extend(research_signal_reasons)

    filing_coverage_penalty = 0.0
    coverage_score = filing_coverage.get("coverage_score") if isinstance(filing_coverage, dict) else None
    if _finite_number(coverage_score):
        subscores["filing_coverage_score"] = round(float(coverage_score), 2)
        filing_coverage_penalty = max(0.0, round((100.0 - float(coverage_score)) / 20.0, 2))
        subscores["filing_coverage_penalty"] = -filing_coverage_penalty
    else:
        # No coverage row (or an unreadable score) is unknown, and unknown must not
        # outscore a company whose coverage was measured at zero.
        filing_coverage_penalty = UNKNOWN_FILING_COVERAGE_PENALTY
        subscores["filing_coverage_penalty"] = -filing_coverage_penalty
        reasons.append("Filing coverage unknown; scored as zero coverage.")
    if isinstance(filing_coverage, dict):
        missing_required = filing_coverage.get("missing_required") or []
        if isinstance(missing_required, list):
            if any(x == "10-K_OR_20-F" for x in missing_required):
                reasons.append("Filing coverage missing recent annual filing (10-K/20-F).")
            if any(x == "10-Q" for x in missing_required):
                reasons.append("Filing coverage missing recent quarterly filing (10-Q).")

    change_adjust_subscores, change_momentum_net, change_reasons = compute_change_momentum_adjustments(change_context)
    subscores.update(change_adjust_subscores)
    reasons.extend(change_reasons)

    base_total = (
        data_completeness
        + business_quality
        + valuation_gap
        + deterioration_or_improvement
        + balance_sheet
        + special_situations
    )
    total = round(base_total - research_penalty + research_signal_net + change_momentum_net - filing_coverage_penalty, 2)

    classification = "Abstain"
    analyst_class = (analyst_decision or {}).get("classification", "")
    if analyst_class == "RESEARCH_ONLY":
        classification = "Research Only"
    elif total >= 70 and analyst_class in {"LONG", "WATCHLIST"}:
        classification = "Long Candidate"
    elif total >= 60 and analyst_class == "SHORT":
        classification = "Short Candidate"
    elif total >= 45:
        classification = "Watchlist"
    else:
        classification = "Abstain"

    if analyst_class == "RESEARCH_ONLY":
        reasons.append("Analyst assumptions include >=2 critical UNKNOWN states")

    return subscores, total, classification, reasons
