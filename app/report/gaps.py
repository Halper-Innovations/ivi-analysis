from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import utc_now_iso
from app.fundamentals.normalize import UNKNOWN
from app.util.issuer_classification import ISSUER_CLASS_FINANCIAL, resolve_issuer_classification


def _missing_key_metrics(packet: dict[str, Any]) -> list[str]:
    fundamentals = packet.get("fundamentals", {}) if isinstance(packet, dict) else {}
    if _issuer_classification_from_packet(packet) == ISSUER_CLASS_FINANCIAL:
        required = ["revenue", "deposits", "loans", "total_assets"]
    else:
        required = ["revenue", "operating_margin", "net_debt", "fcf"]
    missing: list[str] = []
    for key in required:
        value = fundamentals.get(key)
        if value in {None, UNKNOWN}:
            missing.append(key)
    if (
        _issuer_classification_from_packet(packet) == ISSUER_CLASS_FINANCIAL
        and isinstance(fundamentals.get("loans"), (int, float))
        and fundamentals.get("allowance_for_credit_losses") in {None, UNKNOWN}
    ):
        missing.append("allowance_for_credit_losses")
    if _issuer_classification_from_packet(packet) == ISSUER_CLASS_FINANCIAL and isinstance(fundamentals.get("loans"), (int, float)):
        for key in ("provision_for_credit_losses", "net_charge_offs", "nonaccrual_loans"):
            if fundamentals.get(key) in {None, UNKNOWN}:
                missing.append(key)
    return missing


def _parse_warnings(packet: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    filings_used = packet.get("filings_used", []) if isinstance(packet, dict) else []
    if not filings_used:
        warnings.append("No filings linked in evidence packet.")

    required_sections = {"revenue", "cfo", "cash", "total_debt"}
    if _issuer_classification_from_packet(packet) != ISSUER_CLASS_FINANCIAL:
        required_sections.add("operating_income")
        required_sections.add("capex")
    else:
        required_sections.update({"deposits", "loans", "total_assets"})
    line_items = {
        row.get("line_item")
        for row in (packet.get("financials", []) if isinstance(packet, dict) else [])
        if row.get("line_item")
    }
    missing_sections = sorted(required_sections.difference(line_items))
    if missing_sections:
        warnings.append(f"Missing parsed financial sections: {', '.join(missing_sections)}")
    return warnings[:12]


def _claim_warnings(suppressed_claims: list[str]) -> list[str]:
    warnings: list[str] = []
    for label in suppressed_claims:
        warnings.append(f"Suppressed numeric claim without trace: {label}")
    return warnings[:12]


def _missing_research_sources(research: dict[str, Any] | None) -> list[str]:
    if not isinstance(research, dict):
        return ["research_packet_missing"]
    gaps = research.get("evidence_gaps") or []
    out: list[str] = []
    for gap in gaps:
        if isinstance(gap, dict):
            source_type = gap.get("source_type")
        else:
            source_type = str(gap) if gap else None
        if source_type and source_type not in out:
            out.append(str(source_type))
    if not out:
        out.append("none")
    return out


def _research_warnings(research: dict[str, Any] | None) -> list[str]:
    warnings: list[str] = []
    if not isinstance(research, dict):
        return ["Research packet missing for this run/ticker."]
    quality = research.get("quality")
    if isinstance(quality, dict) and quality.get("incomplete"):
        warnings.append("Research quality is incomplete.")
    for gap in (research.get("evidence_gaps") or [])[:8]:
        if isinstance(gap, dict) and gap.get("summary"):
            warnings.append(str(gap["summary"]))
    if not warnings:
        warnings.append("No research warnings.")
    return warnings[:12]


def _recommended_actions(
    packet: dict[str, Any],
    research: dict[str, Any] | None,
    missing_metrics: list[str],
    missing_price: bool,
    missing_filings: bool,
    claim_warnings: list[str],
) -> list[str]:
    actions: list[str] = []
    is_financial = _issuer_classification_from_packet(packet) == ISSUER_CLASS_FINANCIAL

    if isinstance(research, dict):
        for step in research.get("next_actions", []):
            action = (step or {}).get("action")
            if action and action not in actions:
                actions.append(str(action))

    if missing_filings:
        actions.append("Ingest and parse latest 10-Q/10-K filings for this ticker.")
    if missing_metrics:
        if is_financial:
            actions.append(
                "Hydrate bank credit and funding metrics/disclosures: "
                f"{', '.join(missing_metrics)}."
            )
        else:
            actions.append(f"Extract filing values required for metrics: {', '.join(missing_metrics)}.")
    if missing_price:
        actions.append("Enable allowed price provider or continue in price-unknown research-only mode.")
    if claim_warnings:
        actions.append("Add citation or derived_from trace for suppressed numeric claims.")

    dedup: list[str] = []
    for action in actions:
        if action not in dedup:
            dedup.append(action)
    return dedup[:8]


def _issuer_classification_from_packet(packet: dict[str, Any]) -> str:
    fundamentals = packet.get("fundamentals", {}) if isinstance(packet, dict) else {}
    classification = str(fundamentals.get("issuer_classification") or "").strip().lower()
    if classification:
        return classification
    financials = packet.get("financials", []) if isinstance(packet, dict) else []
    line_items = [str(row.get("line_item") or "") for row in financials if isinstance(row, dict)]
    texts = [str((row.get("citation") or {}).get("snippet") or "") for row in financials if isinstance(row, dict)]
    # SIC-first (VOE_ISSUER_CLASSIFICATION_BY_SIC); the substring rule is the fallback.
    return resolve_issuer_classification(
        ticker=packet.get("ticker") if isinstance(packet, dict) else None,
        texts=texts,
        line_items=line_items,
    )[0]


def write_gaps_artifact(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
    packet: dict[str, Any],
    research: dict[str, Any] | None,
    status_flags: list[str],
    suppressed_claims: list[str],
) -> Path:
    cfg = get_config()
    cfg.gaps_dir.mkdir(parents=True, exist_ok=True)

    valuations = packet.get("valuations", {}) if isinstance(packet, dict) else {}
    reverse_inputs = valuations.get("reverse_dcf", {}).get("inputs", {}) if isinstance(valuations, dict) else {}
    market_price = reverse_inputs.get("market_price", UNKNOWN) if isinstance(reverse_inputs, dict) else UNKNOWN

    missing_price = market_price == UNKNOWN
    missing_filings = not bool(packet.get("filings_used"))
    missing_metrics = _missing_key_metrics(packet)
    parse_warnings = _parse_warnings(packet)
    claim_warnings = _claim_warnings(suppressed_claims)
    missing_sources = _missing_research_sources(research)
    research_warnings = _research_warnings(research)

    payload = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "status_flags": status_flags,
        "missing_price": missing_price,
        "missing_filings": missing_filings,
        "missing_key_metrics": missing_metrics,
        "missing_research_sources": missing_sources,
        "parse_warnings": parse_warnings,
        "claim_warnings": claim_warnings,
        "research_warnings": research_warnings,
        "recommended_next_actions": _recommended_actions(
            packet,
            research,
            missing_metrics,
            missing_price,
            missing_filings,
            claim_warnings,
        ),
    }

    out_path = cfg.gaps_dir / f"{ticker}_{run_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
