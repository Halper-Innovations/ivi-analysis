"""Decision-useful per-ticker valuation rendering (goal task OUTPUT).

ONE shared block consumed by the analyze CLI and the dossier valuation
section:

  - per-method anchor/basis/warnings table (incl. the ev_ebit / fcf_yield /
    tangible_floor lenses)
  - anchor-selection provenance: which method won and WHY (via
    app.valuation.anchor_policy)
  - explicit convention labels on every margin-of-safety number
    (see app/valuation/mos_conventions.py — unlabeled cross-convention
    numbers produced the graham-discount-inversion bug)
  - durable / normalized values rendered beside raw ones
  - blocked names show which gate fired and what would unblock them
"""

from __future__ import annotations

import json
from typing import Any

from app.db import get_db
from app.valuation.anchor_policy import is_decline_class, select_anchor
from app.valuation.lineage import (
    latest_decision_eligible_valuation_row,
    latest_decision_eligible_valuation_rows,
)

TEXTBOOK_LABEL = "textbook: (anchor − price) / anchor"

# What would unblock each gate reason code — rendered for blocked names.
GATE_UNBLOCK_HINTS: dict[str, str] = {
    "SEVERE_SECULAR_DECLINE": "latest-FY revenue back within 30% of the non-spike peak, or the trend no longer classifies as secular decline",
    "ZERO_OWNER_EARNINGS": "any of the 3 most recent CFO years covering maintenance capex + SBC (growth-aware ratio)",
    "LIQUIDITY_CRISIS": "aligned-year current ratio recovering to >= 0.5",
    "BOOK_INSOLVENCY": "positive book equity, or revenue no longer declining",
    "LOW_QUALITY_HIGH_LEVERAGE": "earnings quality above LOW, or leverage stress below HIGH",
    "GOING_CONCERN": "going-concern language removed in the next filing",
    "FINANCIAL_ISSUER": "requires the sector-specific model (insurance/bank), not DCF/EPV",
    "SOLVENCY_CONCERN": "valuation allowance released / solvency evidence improves",
}

_METHOD_ORDER = [
    ("dcf", "DCF"),
    ("dcf_adjusted", "DCF (R&D-adj)"),
    ("epv", "EPV"),
    ("epv_adjusted", "EPV (R&D-adj)"),
    ("graham", "Graham"),
    ("ncav", "NCAV"),
    ("ev_ebit", "EV/EBIT lens"),
    ("fcf_yield", "FCF-yield lens"),
    ("tangible_floor", "Tangible floor"),
]


def _load_method_payloads(ticker: str, as_of_date: str | None) -> dict[str, dict[str, Any]]:
    with get_db() as conn:
        if as_of_date:
            rows = latest_decision_eligible_valuation_rows(
                conn,
                ticker=ticker,
                as_of_date=as_of_date,
                exact_as_of_date=True,
            )
        else:
            newest = latest_decision_eligible_valuation_row(
                conn,
                ticker=ticker,
            )
            rows = (
                latest_decision_eligible_valuation_rows(
                    conn,
                    ticker=ticker,
                    as_of_date=str(newest["as_of_date"]),
                    exact_as_of_date=True,
                )
                if newest is not None
                else []
            )
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            out[str(row["method"])] = json.loads(row["outputs_json"] or "{}")
        except Exception:
            out[str(row["method"])] = {}
    return out


def _method_value(payload: dict[str, Any], method: str) -> float | None:
    value = payload.get("base") if method.startswith("dcf") else payload.get("value_per_share")
    return float(value) if isinstance(value, (int, float)) else None


def _method_basis(payload: dict[str, Any], method: str) -> str:
    basis = payload.get("basis")
    if isinstance(basis, dict):
        convention = str(basis.get("convention") or "")
        return convention.split(";")[0] if convention else ""
    if method.startswith("dcf"):
        return "full-capex FCFF OE @ WACC − net debt − senior claims"
    if method.startswith("epv"):
        return "normalized margin × current revenue × (1−tax) / WACC − net debt − senior claims"
    if method == "graham":
        return "sqrt(22.5 × EPS × BVPS)"
    if method == "ncav":
        return "haircut current assets − TOTAL liabilities − preferred"
    return ""


def format_method_value(value: float | None, status: str = "") -> str:
    """A method's per-share value as the table prints it.

    A negative value is not a price: it is written with the sign before the
    dollar (-$6.74, never $-6.74) and its status beside it, so a reader cannot
    take a capitalized loss for a floor.
    """
    if not isinstance(value, (int, float)):
        return f"— ({status or 'missing'})"
    if value < 0:
        return f"-${abs(value):.2f} ({status or 'NEGATIVE'}; not a price)"
    return f"${value:.2f}"


def build_method_rows(methods: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-method rows: name, value, basis, warnings — None values included
    so missing methods are visible, not silently absent."""
    scorecard = methods.get("scorecard") or {}
    pzd = (
        scorecard.get("pricing_zone_detail")
        if isinstance(scorecard.get("pricing_zone_detail"), dict)
        else {}
    )
    rows: list[dict[str, Any]] = []
    for method, label in _METHOD_ORDER:
        payload = methods.get(method)
        if payload is None:
            continue
        value = _method_value(payload, method)
        warnings = [str(f) for f in (payload.get("flags") or [])]
        status = str(payload.get("status") or "")
        extra = ""
        if (
            method == "dcf"
            and isinstance(pzd.get("dcf_raw_base"), (int, float))
            and isinstance(pzd.get("dcf_base"), (int, float))
        ):
            # Durable beside raw: the zone/anchor uses the durable base.
            value = float(pzd["dcf_base"])
            extra = f"durable (spike-corrected); raw ${float(pzd['dcf_raw_base']):.2f}"
        if method.startswith("epv") and "EPV_CYCLICALLY_NORMALIZED" in warnings:
            extra = (extra + "; " if extra else "") + "cyclically normalized (OI-median basis)"
        rows.append(
            {
                "method": method,
                "label": label,
                "value_per_share": value,
                "status": status,
                "basis": _method_basis(payload, method),
                "note": extra,
                "warnings": warnings,
            }
        )
    return rows


def build_anchor_provenance(methods: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Re-derive the anchor selection from the persisted values so the
    rendered provenance always matches the canonical rule."""
    scorecard = methods.get("scorecard") or {}
    pzd = (
        scorecard.get("pricing_zone_detail")
        if isinstance(scorecard.get("pricing_zone_detail"), dict)
        else {}
    )

    def _num(key: str) -> float | None:
        v = pzd.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    diagnostics = scorecard.get("tech_valuation_divergence_diagnostics")
    sector_specific = None
    if isinstance(diagnostics, dict) and diagnostics.get("status") in (None, "OK"):
        adj = diagnostics.get("adjusted_anchor")
        if isinstance(adj, (int, float)) and float(adj) > 0:
            sector_specific = ("technology_adjusted_dcf", float(adj))

    quality_ctx = (
        scorecard.get("quality_context")
        if isinstance(scorecard.get("quality_context"), dict)
        else {}
    )
    selection = select_anchor(
        dcf=_num("dcf_base"),
        epv=_num("epv_adjusted"),
        graham=_num("graham_value_per_share"),
        ncav=_num("ncav_value_per_share"),
        sector_specific=sector_specific,
        extra_candidates={
            "ev_ebit": _num("ev_ebit_value_per_share"),
            "fcf_yield": _num("fcf_yield_value_per_share"),
            "tangible_floor": _num("tangible_floor_per_share"),
        },
        # Decline-cap policy — the displayed anchor/buy-at must match what
        # the live packets and backtest select (ONE predicate).
        decline_class=is_decline_class(quality_ctx.get("revenue_trend_class")),
    )
    zone = scorecard.get("pricing_zone")
    if str(zone or "") == "VALUATION_ANOMALY":
        # Production suppresses anchors for anomaly names (signal_assembler
        # nulls the methods; the backtest excludes the rows) — the surviving
        # positive method in the detail dict must not become buy guidance
        # here either (review: RENDER-ANOMALY-ANCHOR).
        return {
            "anchor_method": None,
            "anchor_value": None,
            "reason": "ZONE_VALUATION_ANOMALY_SUPPRESSED",
            "candidates": selection.candidates,
            "pricing_zone": zone,
            "current_price": _num("current_price"),
        }
    return {
        "anchor_method": selection.method,
        "anchor_value": selection.value,
        "reason": selection.reason,
        "candidates": selection.candidates,
        "pricing_zone": zone,
        "current_price": _num("current_price"),
    }


def render_valuation_decision_block(ticker: str, as_of_date: str | None = None) -> str:
    """Markdown block for the analyze/report surfaces. Empty string when no
    valuation rows exist."""
    methods = _load_method_payloads(ticker, as_of_date)
    if not methods:
        return ""
    scorecard = methods.get("scorecard") or {}
    lines: list[str] = ["## Valuation Methods & Anchor", ""]

    # ── Blocked names: which gate fired, what would unblock ──────────────────
    if str(scorecard.get("pricing_zone") or "") == "VALUATION_BLOCKED":
        qc = (
            scorecard.get("quality_context")
            if isinstance(scorecard.get("quality_context"), dict)
            else {}
        )
        codes = [str(c) for c in (qc.get("gate_reason_codes") or [])]
        lines.append(f"**GATE BLOCKED** — {', '.join(codes) or 'unknown reason'}")
        for code in codes:
            hint = GATE_UNBLOCK_HINTS.get(code)
            if hint:
                lines.append(f"- {code}: would unblock when {hint}")
        return "\n".join(lines)

    provenance = build_anchor_provenance(methods)
    anchor_value = provenance["anchor_value"]
    price = provenance["current_price"]
    if provenance["reason"] == "ZONE_VALUATION_ANOMALY_SUPPRESSED":
        lines.append(
            "**Anchor:** suppressed — production suppresses anchors for "
            "VALUATION_ANOMALY names (negative intrinsic value in one or more "
            "methods); no buy guidance"
        )
    elif anchor_value is not None:
        lines.append(
            f"**Anchor:** ${anchor_value:.2f} via **{provenance['anchor_method']}** "
            f"({provenance['reason']}) — buy at <= ${anchor_value * 0.75:.2f} "
            f"(25% margin of safety, {TEXTBOOK_LABEL})"
        )
        if isinstance(price, (int, float)) and anchor_value > 0:
            mos = (anchor_value - price) / anchor_value
            lines.append(f"**Margin of safety at ${price:.2f}:** {mos:.1%} ({TEXTBOOK_LABEL})")
    else:
        lines.append("**Anchor:** none — no positive anchor method")
    zone = provenance.get("pricing_zone")
    if zone:
        lines.append(f"**Pricing zone:** {zone}")
    lines.append("")

    # ── Per-method table ──────────────────────────────────────────────────────
    lines.append("| Method | Value/sh | Basis | Notes | Warnings |")
    lines.append("|--------|----------|-------|-------|----------|")
    anchor_method = provenance.get("anchor_method")
    for row in build_method_rows(methods):
        value_str = format_method_value(row["value_per_share"], row["status"])
        marker = " **(anchor)**" if row["method"] == anchor_method else ""
        warn = ", ".join(row["warnings"][:4]) or "none"
        lines.append(
            f"| {row['label']}{marker} | {value_str} | {row['basis']} | {row['note'] or ''} | {warn} |"
        )

    lines.append("")
    lines.append(
        "_Conventions: margin-of-safety figures above are TEXTBOOK ((anchor−price)/anchor); "
        "graham_dodd/scout surfaces report UPSIDE ratios (anchor/price − 1). "
        "Lenses (EV/EBIT, FCF-yield, tangible floor) are provenance only — never the anchor, never a gate._"
    )
    return "\n".join(lines)
