from __future__ import annotations

from typing import Any


def _method_outputs(valuations: dict[str, Any], method: str) -> dict[str, Any]:
    payload = valuations.get(method, {}) if isinstance(valuations, dict) else {}
    if not isinstance(payload, dict):
        return {}
    outputs = payload.get("outputs")
    if isinstance(outputs, dict):
        return outputs
    return payload


def _citations_for_line_item(packet: dict[str, Any], line_item: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in packet.get("financials", []):
        if row.get("line_item") != line_item:
            continue
        citation = row.get("citation", {})
        if citation.get("source_url"):
            out.append(
                {
                    "source_url": citation.get("source_url", ""),
                    "snippet": citation.get("snippet", ""),
                    "section_label": citation.get("section_label"),
                }
            )
            if len(out) >= 3:
                break
    return out


def build_numeric_claims(packet: dict[str, Any]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    fundamentals = packet.get("fundamentals", {})

    direct_metric_to_line_item = {
        "revenue": "revenue",
        "net_income": "net_income",
        "cfo": "cfo",
        "capex": "capex",
    }

    for metric, line_item in direct_metric_to_line_item.items():
        value = fundamentals.get(metric)
        if not isinstance(value, (int, float)):
            continue
        claims.append(
            {
                "claim_id": f"metric_{metric}",
                "label": metric,
                "value": value,
                "unit": "USD",
                "citations": _citations_for_line_item(packet, line_item),
                "derived_from": [f"financials.{line_item}"],
            }
        )

    derived_metrics = {
        "operating_margin": ["financials.operating_income", "financials.revenue"],
        "fcf": ["financials.cfo", "financials.capex"],
        "fcf_margin": ["financials.cfo", "financials.capex", "financials.revenue"],
        "net_debt": ["financials.total_debt", "financials.cash"],
    }
    for metric, deps in derived_metrics.items():
        value = fundamentals.get(metric)
        if not isinstance(value, (int, float)):
            continue
        citations: list[dict[str, str]] = []
        if metric == "operating_margin":
            citations.extend(_citations_for_line_item(packet, "operating_income"))
            citations.extend(_citations_for_line_item(packet, "revenue"))
        elif metric == "fcf":
            citations.extend(_citations_for_line_item(packet, "cfo"))
            citations.extend(_citations_for_line_item(packet, "capex"))
        elif metric == "fcf_margin":
            citations.extend(_citations_for_line_item(packet, "cfo"))
            citations.extend(_citations_for_line_item(packet, "capex"))
            citations.extend(_citations_for_line_item(packet, "revenue"))
        elif metric == "net_debt":
            citations.extend(_citations_for_line_item(packet, "total_debt"))
            citations.extend(_citations_for_line_item(packet, "cash"))
        claims.append(
            {
                "claim_id": f"metric_{metric}",
                "label": metric,
                "value": value,
                "unit": "ratio" if metric.endswith("margin") else "USD",
                "citations": citations[:6],
                "derived_from": deps,
            }
        )

    valuations = packet.get("valuations", {})
    valuation_methods = [
        ("dcf", _method_outputs(valuations, "dcf"), ["low", "base", "high"]),
        ("graham", _method_outputs(valuations, "graham"), ["value_per_share"]),
        ("epv", _method_outputs(valuations, "epv"), ["value_per_share"]),
        ("ncav", _method_outputs(valuations, "ncav"), ["value_per_share"]),
    ]
    legacy_dcf = _method_outputs(valuations, "dcf_lite")
    if not valuation_methods[0][1] and isinstance(legacy_dcf.get("per_share_range"), dict):
        valuation_methods[0] = ("dcf", legacy_dcf.get("per_share_range", {}), ["low", "base", "high"])

    for method, outputs, keys in valuation_methods:
        for key in keys:
            value = outputs.get(key)
            if not isinstance(value, (int, float)):
                continue
            claims.append(
                {
                    "claim_id": f"valuation_{method}_{key}",
                    "label": f"{method}_per_share_{key}",
                    "value": value,
                    "unit": "USD/share",
                    "citations": [],
                    "derived_from": [f"valuations.{method}.outputs.{key}"],
                }
            )

    reverse_outputs = _method_outputs(valuations, "reverse_dcf")
    nested_outputs = (
        reverse_outputs.get("outputs")
        if isinstance(reverse_outputs.get("outputs"), dict)
        else {}
    )
    implied_growth = reverse_outputs.get("implied_growth")
    if not isinstance(implied_growth, (int, float)):
        implied_growth = nested_outputs.get("implied_growth")
    # A saturated solve returns the clipped bound in implied_growth by design;
    # presenting it as a point-estimate claim fabricates a deep-value-looking
    # number for exactly the unsolvable population (review RDCF-2; mirrors
    # evidence/enrichment).
    implied_growth_saturated = bool(reverse_outputs.get("implied_growth_saturated")) or bool(
        nested_outputs.get("implied_growth_saturated")
    )
    if isinstance(implied_growth, (int, float)) and not implied_growth_saturated:
        claims.append(
            {
                "claim_id": "valuation_reverse_dcf_implied_growth",
                "label": "reverse_dcf_implied_growth",
                "value": implied_growth,
                "unit": "ratio",
                "citations": [],
                "derived_from": ["valuations.reverse_dcf.outputs.implied_growth"],
            }
        )

    return claims


def validate_claims_have_evidence_or_derivation(claims: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for claim in claims:
        value = claim.get("value")
        if not isinstance(value, (int, float)):
            continue
        citations = claim.get("citations") or []
        derived_from = claim.get("derived_from") or []
        if not citations and not derived_from:
            failures.append(f"numeric claim missing citation/derivation: {claim.get('label')}")
    return failures
