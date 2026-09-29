from __future__ import annotations

from typing import Any


def _get_num(mapping: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _pct(value: float | None) -> float | None:
    return None if value is None else round(value * 100.0, 2)


def _direction(value: float | None, *, positive: str, negative: str, flat: str = "stable") -> str | None:
    if value is None:
        return None
    if value > 0:
        return positive
    if value < 0:
        return negative
    return flat


def _top_moat_signals(packet: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in packet.get("extracted_facts") or []:
        if not isinstance(row, dict):
            continue
        fact_type = str(row.get("fact_type") or "").strip()
        if not fact_type:
            continue
        lowered = fact_type.lower()
        if not any(token in lowered for token in ("deferred", "rpo", "segment", "customer", "repurchase", "dividend", "r_and_d")):
            continue
        citation = row.get("citation") if isinstance(row.get("citation"), dict) else {}
        value = row.get("value") if isinstance(row.get("value"), dict) else {}
        out.append(
            {
                "signal": fact_type,
                "metric": value.get("metric"),
                "value": value.get("value"),
                "source_snippet": str(citation.get("snippet") or "")[:240],
                "derived_from": [f"evidence_packet.extracted_facts.{fact_type}"],
            }
        )
        if len(out) >= limit:
            break
    return out


def _valuation_spread_analysis(packet: dict[str, Any]) -> dict[str, Any]:
    valuations = packet.get("valuations") if isinstance(packet.get("valuations"), dict) else {}
    reverse_dcf = valuations.get("reverse_dcf") if isinstance(valuations.get("reverse_dcf"), dict) else {}
    reverse_inputs = reverse_dcf.get("inputs") if isinstance(reverse_dcf.get("inputs"), dict) else {}
    price = _get_num(reverse_inputs, "price")

    def _method_gap(method: str, key: str) -> dict[str, Any] | None:
        method_payload = valuations.get(method) if isinstance(valuations.get(method), dict) else {}
        outputs = method_payload.get("outputs") if isinstance(method_payload.get("outputs"), dict) else {}
        nested = outputs.get("outputs") if isinstance(outputs.get("outputs"), dict) else {}
        intrinsic = outputs.get(key)
        if not isinstance(intrinsic, (int, float)):
            intrinsic = nested.get(key)
        if not isinstance(intrinsic, (int, float)):
            return None
        intrinsic_value = float(intrinsic)
        gap_pct = None
        flag = "VALUE_PRESENT_NO_PRICE"
        if price not in (None, 0):
            gap_pct = round(((price - intrinsic_value) / intrinsic_value) * 100.0, 2) if intrinsic_value else None
            if gap_pct is not None:
                if gap_pct >= 200:
                    flag = "PRICE_EXCEEDS_3X_METHOD"
                elif gap_pct >= 100:
                    flag = "PRICE_EXCEEDS_2X_METHOD"
                elif gap_pct > 0:
                    flag = "PRICE_ABOVE_METHOD"
                elif gap_pct < 0:
                    flag = "PRICE_BELOW_METHOD"
                else:
                    flag = "PRICE_NEAR_METHOD"
        return {
            "price": price,
            "intrinsic_value": intrinsic_value,
            "gap_pct": gap_pct,
            "flag": flag,
            "derived_from": [
                f"evidence_packet.valuations.{method}.{key}",
                "evidence_packet.valuations.reverse_dcf.inputs.price",
            ],
        }

    implied_growth = None
    reverse_outputs = reverse_dcf.get("outputs") if isinstance(reverse_dcf.get("outputs"), dict) else {}
    nested_reverse = reverse_outputs.get("outputs") if isinstance(reverse_outputs.get("outputs"), dict) else {}
    if isinstance(reverse_outputs.get("implied_growth"), (int, float)):
        implied_growth = float(reverse_outputs["implied_growth"])
    elif isinstance(nested_reverse.get("implied_growth"), (int, float)):
        implied_growth = float(nested_reverse["implied_growth"])
    if bool(reverse_outputs.get("implied_growth_saturated")) or bool(
        nested_reverse.get("implied_growth_saturated")
    ):
        # Clipped bound, not a solve — do not grade feasibility off it
        # (audit: saturated-bound-leaks-to-flag-ignoring-consumers).
        implied_growth = None

    feasibility = None
    if implied_growth is not None:
        if implied_growth >= 0.2:
            feasibility = "IMPLAUSIBLE"
        elif implied_growth >= 0.12:
            feasibility = "STRETCHED"
        elif implied_growth >= 0.06:
            feasibility = "PLAUSIBLE"
        else:
            feasibility = "MODEST"

    return {
        "price": price,
        "dcf": _method_gap("dcf", "base"),
        "epv": _method_gap("epv", "value_per_share"),
        "graham": _method_gap("graham", "value_per_share"),
        "ncav": _method_gap("ncav", "value_per_share"),
        "implied_growth_rate": implied_growth,
        "implied_growth_rate_pct": _pct(implied_growth),
        "implied_growth_feasibility": feasibility,
    }


def _quality_assessment(packet: dict[str, Any]) -> dict[str, Any]:
    """Extract quality signals from scorecard outputs."""
    valuations = packet.get("valuations") if isinstance(packet.get("valuations"), dict) else {}
    scorecard = valuations.get("scorecard") if isinstance(valuations.get("scorecard"), dict) else {}
    outputs = scorecard.get("outputs") if isinstance(scorecard.get("outputs"), dict) else {}
    qc = outputs.get("quality_context") if isinstance(outputs.get("quality_context"), dict) else {}
    moat = outputs.get("moat_strength") if isinstance(outputs.get("moat_strength"), dict) else {}
    downside = outputs.get("downside_scenario") if isinstance(outputs.get("downside_scenario"), dict) else {}
    nr = qc.get("nonrecurring_detection") if isinstance(qc.get("nonrecurring_detection"), dict) else {}
    sbc = qc.get("sbc_trajectory") if isinstance(qc.get("sbc_trajectory"), dict) else {}
    dep = qc.get("depreciation_audit") if isinstance(qc.get("depreciation_audit"), dict) else {}

    return {
        "gate_verdict": qc.get("gate_action"),
        "confidence_class": qc.get("confidence_class"),
        "gate_reason_codes": qc.get("gate_reason_codes") or [],
        "valuation_headwinds": qc.get("valuation_headwinds") or [],
        "valuation_supports": qc.get("valuation_supports") or [],
        "moat_class": moat.get("moat_class"),
        "moat_score": moat.get("moat_score"),
        "signal_context": outputs.get("signal_context"),
        "downside_risk_class": downside.get("downside_risk_class"),
        "bear_case_intrinsic": downside.get("bear_case_intrinsic"),
        "nonrecurring_flags": nr.get("nonrecurring_flags") or [],
        "sbc_flags": sbc.get("sbc_flags") or [],
        "depreciation_flags": dep.get("depreciation_flags") or [],
    }


def _filing_intelligence(packet: dict[str, Any]) -> dict[str, Any]:
    """Extract filing diff and pattern scan results from the evidence packet.

    These are populated when a dossier run has executed the filing diff engine
    and/or pattern scanner. Returns empty dicts when not available.
    """
    # Filing diff results (if attached to packet by dossier runner)
    diff_report = packet.get("filing_diff") if isinstance(packet.get("filing_diff"), dict) else {}
    high_materiality_changes: list[dict[str, Any]] = []
    if isinstance(diff_report.get("changes"), list):
        for change in diff_report["changes"]:
            if isinstance(change, dict) and change.get("materiality") == "HIGH":
                high_materiality_changes.append({
                    "section": change.get("section"),
                    "change_type": change.get("change_type"),
                    "summary": change.get("summary"),
                    "fiscal_years": f"{change.get('fiscal_year_from')}→{change.get('fiscal_year_to')}",
                })

    # Pattern scan results (if attached)
    pattern_hits = packet.get("pattern_hits") if isinstance(packet.get("pattern_hits"), list) else []
    confirmed_patterns: list[dict[str, Any]] = []
    for hit in pattern_hits:
        if isinstance(hit, dict) and hit.get("outcome_confirmed"):
            confirmed_patterns.append({
                "pattern_id": hit.get("pattern_id"),
                "name": hit.get("name"),
                "hypothesis": hit.get("hypothesis"),
                "years_detected": hit.get("years_detected"),
                "hit_rate": hit.get("hit_rate"),
                "summary_text": hit.get("summary_text"),
            })

    return {
        "filing_diff_available": bool(diff_report),
        "high_materiality_changes": high_materiality_changes,
        "high_materiality_count": len(high_materiality_changes),
        "pattern_hits_available": bool(pattern_hits),
        "confirmed_patterns": confirmed_patterns,
        "confirmed_pattern_count": len(confirmed_patterns),
    }


def enrich_evidence_packet(packet: dict[str, Any]) -> dict[str, Any]:
    fundamentals = packet.get("fundamentals") if isinstance(packet.get("fundamentals"), dict) else {}
    owner_earnings = (
        packet.get("valuations", {}).get("owner_earnings")
        if isinstance(packet.get("valuations"), dict)
        else {}
    )
    owner_outputs = owner_earnings.get("outputs") if isinstance(owner_earnings, dict) else {}
    owner_warnings = owner_earnings.get("warnings") if isinstance(owner_earnings, dict) else None
    if not isinstance(owner_outputs, dict):
        owner_outputs = {}
    trend_signals = {
        "revenue_cagr_3y_pct": _pct(_get_num(fundamentals, "revenue_cagr_3y")),
        "revenue_cagr_5y_pct": _pct(_get_num(fundamentals, "revenue_cagr_5y")),
        "operating_margin_trend_direction": _direction(
            _get_num(fundamentals, "operating_margin_trend_slope"),
            positive="improving",
            negative="declining",
        ),
        "roic_trend_direction": _direction(
            _get_num(fundamentals, "roic_trend"),
            positive="improving",
            negative="declining",
        ),
        "fcf_conversion_trend_direction": _direction(
            _get_num(fundamentals, "fcf_margin_trend_slope"),
            positive="improving",
            negative="declining",
        ),
        "dilution_direction": _direction(
            _get_num(fundamentals, "dilution_rate_shares_cagr"),
            positive="dilutive",
            negative="share_count_shrinking",
        ),
    }
    capital_allocation = {
        "share_repurchases_amount": _get_num(fundamentals, "share_repurchases_amount"),
        "dividends_paid_amount": _get_num(fundamentals, "dividends_paid_amount"),
        "dilution_rate_shares_cagr_pct": _pct(_get_num(fundamentals, "dilution_rate_shares_cagr")),
        "capex_intensity_pct": _pct(_get_num(fundamentals, "capex_intensity")),
        "r_and_d_intensity_pct": _pct(_get_num(fundamentals, "r_and_d_intensity_latest")),
        "capital_allocation_posture": (
            "shareholder_return_active"
            if _get_num(fundamentals, "share_repurchases_amount") is not None
            or _get_num(fundamentals, "dividends_paid_amount") is not None
            else "reinvestment_led_or_unknown"
        ),
    }
    enriched = dict(packet)
    enriched["enrichment"] = {
        "trend_narratives": trend_signals,
        "moat_signals_summary": _top_moat_signals(packet),
        "valuation_spread_analysis": _valuation_spread_analysis(packet),
        "capital_allocation_quality": capital_allocation,
        "owner_earnings_quality_flags": {
            "confidence": owner_outputs.get("confidence"),
            "flags": owner_outputs.get("flags") if isinstance(owner_outputs.get("flags"), list) else owner_warnings or [],
        },
        "quality_assessment": _quality_assessment(packet),
        "filing_intelligence": _filing_intelligence(packet),
    }
    return enriched
