"""Generate a readable markdown report from a SectorAlphaReport."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.alpha.schemas import SectorAlphaReport
from app.db import get_db
from app.util.financial_data_access import ANNUAL_COMPANYFACTS_PERIOD_TYPES, companyfacts_rows
from app.util.financial_data_access import issuer_companyfacts_rows
from app.valuation.lineage import latest_decision_eligible_valuation_row

logger = logging.getLogger(__name__)

_FINANCIAL_SUMMARY_LINE_ITEMS = (
    "revenue",
    "operating_income",
    "net_income",
    "cfo",
    "capex",
)
_FINANCIAL_SOURCE_COLUMNS = (
    "id",
    "ticker",
    "fiscal_year",
    "period_type",
    "period_end",
    "line_item",
    "value",
    "units",
    "source_url",
    "fetched_at",
    "filed_date",
    "form",
    "accession",
)


class AlphaFinancialUnitError(ValueError):
    """A supplemental report row is not in the normalized monetary unit."""

    def __init__(self, row: dict[str, Any]):
        self.row = dict(row)
        super().__init__(
            "Alpha report financial row requires units='USD_millions': "
            f"ticker={row.get('ticker')!r}, line_item={row.get('line_item')!r}, "
            f"units={row.get('units')!r}"
        )


def derive_alpha_report_financials(
    rows: list[dict[str, Any]],
    *,
    years: int,
) -> list[dict[str, Any]]:
    """Derive the displayed financial summary from an already frozen PIT snapshot."""

    rows_by_year: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            fiscal_year = int(row["fiscal_year"])
        except (KeyError, TypeError, ValueError):
            continue
        rows_by_year.setdefault(fiscal_year, []).append(row)
    result = []
    for yr in sorted(rows_by_year, reverse=True):
        items: dict[str, float] = {}
        for row in rows_by_year[yr]:
            line_item = str(row.get("line_item") or "")
            if line_item not in _FINANCIAL_SUMMARY_LINE_ITEMS:
                continue
            if row.get("units") != "USD_millions":
                raise AlphaFinancialUnitError(row)
            try:
                items[line_item] = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
        if not any(key in items for key in ("revenue", "operating_income", "net_income", "cfo")):
            continue
        rev = items.get("revenue")
        operating_income = items.get("operating_income")
        cfo = items.get("cfo")
        capex = items.get("capex")
        result.append(
            {
                "year": yr,
                "revenue": rev,
                "operating_income": operating_income,
                "net_income": items.get("net_income"),
                "cfo": cfo,
                "capex": capex,
                "fcf": (cfo - capex) if cfo is not None and capex is not None else None,
                "operating_margin": (operating_income / rev * 100)
                if rev and operating_income is not None
                else None,
            }
        )
        if len(result) >= years:
            break
    return result


def load_alpha_report_financial_sources(
    report: SectorAlphaReport,
    *,
    as_of_date: str,
    conn: sqlite3.Connection | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load the exact PIT CompanyFacts rows used by Alpha report prose.

    Publication callers pass a connection held under ``BEGIN IMMEDIATE`` for
    the final comparison.  Direct report rendering can omit it and gets the
    same query semantics from a short read connection.
    """

    years_by_ticker: dict[str, int] = {}
    if report.winner:
        years_by_ticker[str(report.winner).strip().upper()] = 5
    if report.runner_up:
        ticker = str(report.runner_up).strip().upper()
        years_by_ticker[ticker] = max(years_by_ticker.get(ticker, 0), 3)
    if not years_by_ticker:
        return {}

    context = nullcontext(conn) if conn is not None else get_db()
    snapshots: dict[str, list[dict[str, Any]]] = {}
    with context as selected_conn:
        assert selected_conn is not None
        for ticker, years in sorted(years_by_ticker.items()):
            query = {
                "columns": _FINANCIAL_SOURCE_COLUMNS,
                "period_types": ANNUAL_COMPANYFACTS_PERIOD_TYPES,
                "line_items": _FINANCIAL_SUMMARY_LINE_ITEMS,
                "value_not_null": True,
                "as_of_date": as_of_date,
                "require_filed_asof": True,
                "order_by": "fiscal_year DESC, line_item ASC, id ASC",
            }
            packet = report.signal_packets.get(ticker, {})
            issuer_cik = str(packet.get("issuer_cik") or "").strip() or None
            aliases = tuple(
                str(item).strip().upper()
                for item in packet.get("issuer_listed_tickers") or ()
                if str(item).strip()
            )
            if issuer_cik is not None or aliases:
                _issuer_scope, rows = issuer_companyfacts_rows(
                    selected_conn,
                    ticker,
                    issuer_cik=issuer_cik,
                    aliases=aliases,
                    **query,
                )
            else:
                rows = companyfacts_rows(
                    selected_conn,
                    ticker,
                    **query,
                )
            row_payloads = [dict(row) for row in rows]
            selected_years = {
                int(item["year"])
                for item in derive_alpha_report_financials(row_payloads, years=years)
            }
            snapshots[ticker] = [
                item for item in row_payloads if int(item["fiscal_year"]) in selected_years
            ]
    return snapshots


def _load_financials(
    ticker: str,
    *,
    as_of_date: str | None,
    years: int = 5,
) -> list[dict[str, Any]]:
    """Load annual facts that were actually filed by the report date."""
    if not as_of_date:
        return []
    try:
        probe = SectorAlphaReport(
            sector="",
            total_candidates=0,
            rounds=[],
            winner=ticker,
            winner_thesis="",
            winner_conviction="",
            runner_up=None,
            runner_up_thesis="",
            key_risk="",
            falsification_trigger="",
            time_horizon="",
            signal_packets={},
        )
        rows = load_alpha_report_financial_sources(
            probe,
            as_of_date=as_of_date,
        ).get(str(ticker).strip().upper(), [])
        return derive_alpha_report_financials(rows, years=years)
    except Exception:
        return []


def _scorecard_summary_from_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Render only the canonical packet values used by the Alpha decision."""

    if not packet:
        return {}
    current_price = packet.get("current_price")
    dcf_base = packet.get("dcf_value")
    epv_adjusted = packet.get("epv_value")
    dcf_discount = None
    epv_discount = None
    if (
        isinstance(current_price, (int, float))
        and not isinstance(current_price, bool)
        and isinstance(dcf_base, (int, float))
        and not isinstance(dcf_base, bool)
        and dcf_base != 0
    ):
        dcf_discount = (dcf_base - current_price) / dcf_base
    if (
        isinstance(current_price, (int, float))
        and not isinstance(current_price, bool)
        and isinstance(epv_adjusted, (int, float))
        and not isinstance(epv_adjusted, bool)
        and epv_adjusted != 0
    ):
        epv_discount = (epv_adjusted - current_price) / epv_adjusted
    return {
        "current_price": current_price,
        "dcf_base": dcf_base,
        "epv_adjusted": epv_adjusted,
        "dcf_discount": dcf_discount,
        "epv_discount": epv_discount,
        "gate_action": packet.get("gate_verdict"),
        "signal": packet.get("consensus_direction") or packet.get("margin_of_safety_verdict"),
        "moat_class": packet.get("moat_classification"),
        "moat_score": packet.get("moat_score"),
        "downside_risk": packet.get("downside_risk_class"),
        "earnings_quality": packet.get("confidence_class"),
        "revenue_cagr_5y": None,
        "revenue_cagr_3y": None,
        "headwinds": packet.get("valuation_headwinds") or [],
        "supports": packet.get("valuation_supports") or [],
    }


def _load_insurance_packet_summary(ticker: str) -> dict[str, Any]:
    """Load latest persisted insurance packet for report rendering."""
    try:
        with get_db() as conn:
            row = latest_decision_eligible_valuation_row(
                conn,
                ticker=ticker,
                method="insurance_packet",
            )
        if row and row["outputs_json"]:
            return json.loads(row["outputs_json"])
    except Exception:
        pass
    return {}


def _fmt(val: Any, fmt: str = ".0f", prefix: str = "$", suffix: str = "M") -> str:
    """Format a numeric value, returning '—' for None/0."""
    if val is None or val == 0:
        return "—"
    try:
        return f"{prefix}{float(val):{fmt}}{suffix}"
    except (ValueError, TypeError):
        return str(val)


def _pct(val: Any) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.1%}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_plain_number(val: Any) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):,.0f}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_plain_pct(val: Any) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.1f}%"
    except (ValueError, TypeError):
        return str(val)


def _insurance_packet_for(report: SectorAlphaReport, ticker: str | None) -> dict[str, Any]:
    if not ticker:
        return {}
    packet_summary = report.signal_packets.get(ticker, {})
    if isinstance(packet_summary.get("insurance_packet"), dict):
        return packet_summary["insurance_packet"]
    return {}


def _insurance_active(insurance_packet: dict[str, Any]) -> bool:
    return bool(
        isinstance(insurance_packet, dict)
        and insurance_packet
        and insurance_packet.get("model_status") != "NOT_APPLICABLE"
    )


def _generic_allowed(insurance_packet: dict[str, Any]) -> bool:
    return (
        not _insurance_active(insurance_packet)
        or insurance_packet.get("generic_valuation_valid") is not False
    )


def _append_pc_operating_summary(lines: list[str], operating_metrics: dict[str, Any]) -> None:
    if (
        not isinstance(operating_metrics, dict)
        or operating_metrics.get("status") == "NOT_APPLICABLE"
    ):
        return
    lines.append("**P&C Operating Metrics**")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Status | {operating_metrics.get('status', '—')} |")
    lines.append(f"| Confidence | {operating_metrics.get('confidence', '—')} |")
    lines.append(f"| Combined Ratio | {_pct(operating_metrics.get('combined_ratio'))} |")
    lines.append(f"| Loss Ratio | {_pct(operating_metrics.get('loss_ratio'))} |")
    lines.append(f"| Expense Ratio | {_pct(operating_metrics.get('expense_ratio'))} |")
    lines.append(
        f"| Underwriting Assessment | {operating_metrics.get('combined_ratio_assessment', '—')} |"
    )
    reserve = (
        operating_metrics.get("reserve_development")
        if isinstance(operating_metrics.get("reserve_development"), dict)
        else {}
    )
    lines.append(f"| Reserve Development | {reserve.get('status', '—')} |")
    reinsurance = (
        operating_metrics.get("reinsurance_program")
        if isinstance(operating_metrics.get("reinsurance_program"), dict)
        else {}
    )
    reinsurance_structures = ", ".join(reinsurance.get("structures") or []) or "—"
    lines.append(f"| Reinsurance Structures | {reinsurance_structures} |")
    cat = (
        operating_metrics.get("catastrophe_exposure")
        if isinstance(operating_metrics.get("catastrophe_exposure"), dict)
        else {}
    )
    cat_terms = ", ".join(cat.get("terms") or []) or "—"
    lines.append(f"| Catastrophe Terms | {cat_terms} |")
    missing = ", ".join(operating_metrics.get("missing_components") or []) or "—"
    lines.append(f"| Missing Components | {missing} |")
    lines.append("")


def _append_mortgage_operating_summary(lines: list[str], operating_metrics: dict[str, Any]) -> None:
    if (
        not isinstance(operating_metrics, dict)
        or operating_metrics.get("status") == "NOT_APPLICABLE"
    ):
        return
    lines.append("**Mortgage Insurance Metrics**")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Status | {operating_metrics.get('status', '—')} |")
    lines.append(f"| Confidence | {operating_metrics.get('confidence', '—')} |")
    lines.append(f"| PMIERs Excess Ratio | {_pct(operating_metrics.get('pmier_excess_ratio'))} |")
    lines.append(
        f"| PMIERs Available / Required | {_fmt(operating_metrics.get('pmier_available_to_required_ratio'), '.2f', '', 'x')} |"
    )
    lines.append(
        f"| Primary IIF | {_fmt(operating_metrics.get('primary_iif_billion'), '.1f', '$', 'B')} |"
    )
    lines.append(
        f"| Primary RIF | {_fmt(operating_metrics.get('primary_rif_billion'), '.1f', '$', 'B')} |"
    )
    lines.append(
        f"| New Insurance Written | {_fmt(operating_metrics.get('new_insurance_written_billion'), '.1f', '$', 'B')} |"
    )
    lines.append(f"| Default Rate | {_pct(operating_metrics.get('default_rate'))} |")
    lines.append(
        f"| Loans In Default | {_fmt_plain_number(operating_metrics.get('loans_in_default'))} |"
    )
    lines.append(
        f"| Policies In Force | {_fmt_plain_number(operating_metrics.get('policies_in_force'))} |"
    )
    lines.append(f"| Annual Persistency | {_pct(operating_metrics.get('annual_persistency'))} |")
    lines.append(
        f"| Claims Paid | {_fmt_plain_number(operating_metrics.get('claims_paid_count'))} |"
    )
    lines.append(
        f"| Claims Paid Amount | {_fmt(operating_metrics.get('claims_paid_million'), '.1f', '$', 'M')} |"
    )
    lines.append(
        f"| Credit / Capital Assessment | {operating_metrics.get('credit_capital_assessment', '—')} |"
    )
    reserve = (
        operating_metrics.get("reserve_development")
        if isinstance(operating_metrics.get("reserve_development"), dict)
        else {}
    )
    lines.append(f"| Reserve Development | {reserve.get('status', '—')} |")
    reinsurance = (
        operating_metrics.get("reinsurance_program")
        if isinstance(operating_metrics.get("reinsurance_program"), dict)
        else {}
    )
    reinsurance_structures = ", ".join(reinsurance.get("structures") or []) or "—"
    lines.append(f"| Reinsurance Structures | {reinsurance_structures} |")
    missing = ", ".join(operating_metrics.get("missing_components") or []) or "—"
    lines.append(f"| Missing Components | {missing} |")
    lines.append("")


def _append_operating_summary(lines: list[str], operating_metrics: dict[str, Any]) -> None:
    metric_family = str(operating_metrics.get("metric_family") or "")
    if metric_family == "mortgage_insurance":
        _append_mortgage_operating_summary(lines, operating_metrics)
    else:
        _append_pc_operating_summary(lines, operating_metrics)


def _append_insurance_summary(
    lines: list[str], *, sc: dict[str, Any], insurance_packet: dict[str, Any]
) -> bool:
    if not _insurance_active(insurance_packet):
        return False
    routing = (
        insurance_packet.get("routing") if isinstance(insurance_packet.get("routing"), dict) else {}
    )
    valuation = (
        insurance_packet.get("valuation")
        if isinstance(insurance_packet.get("valuation"), dict)
        else {}
    )
    lines.append("| Insurance / Security Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(
        f"| Current Price | {_fmt(sc.get('current_price') or valuation.get('current_price'), '.2f', '$', '')} |"
    )
    lines.append(f"| Security Type | {routing.get('security_type', '—')} |")
    lines.append(f"| Identity Status | {routing.get('security_identity_status', '—')} |")
    lines.append(f"| Issuer Type | {routing.get('issuer_type', '—')} |")
    lines.append(f"| Insurance Subtype | {routing.get('insurance_subtype', '—') or '—'} |")
    lines.append(f"| Accounting Regime | {routing.get('accounting_regime', '—')} |")
    lines.append(f"| Model Status | {insurance_packet.get('model_status', '—')} |")
    if valuation.get("method") == "insurance_common":
        lines.append(
            f"| Insurance Anchor | {_fmt(valuation.get('valuation_anchor'), '.2f', '$', '')} |"
        )
        lines.append(
            f"| Adjusted Book / Share | {_fmt(valuation.get('adjusted_book_value_per_share'), '.2f', '$', '')} |"
        )
        lines.append(f"| Normalized ROE | {_pct(valuation.get('normalized_roe'))} |")
        lines.append(
            f"| Justified P/B | {_fmt(valuation.get('justified_price_to_book'), '.2f', '', 'x')} |"
        )
    elif valuation.get("method") == "insurance_preferred":
        terms = (
            valuation.get("preferred_terms")
            if isinstance(valuation.get("preferred_terms"), dict)
            else {}
        )
        lines.append(
            f"| Liquidation Preference | {_fmt(valuation.get('valuation_anchor'), '.2f', '$', '')} |"
        )
        lines.append(f"| Coupon Rate | {_pct(terms.get('coupon_rate'))} |")
        lines.append(f"| Current Yield | {_pct(valuation.get('current_yield'))} |")
        lines.append(f"| Yield To Worst | {_pct(valuation.get('yield_to_worst'))} |")
        lines.append(f"| Cumulative | {terms.get('cumulative', '—')} |")
    blockers = ", ".join(insurance_packet.get("model_blockers") or []) or "—"
    warnings = ", ".join(insurance_packet.get("model_fit_warnings") or []) or "—"
    lines.append(f"| Model Blockers | {blockers} |")
    lines.append(f"| Model Warnings | {warnings} |")
    lines.append("")
    operating_metrics = (
        insurance_packet.get("operating_metrics")
        if isinstance(insurance_packet.get("operating_metrics"), dict)
        else {}
    )
    _append_operating_summary(lines, operating_metrics)
    if not _generic_allowed(insurance_packet):
        lines.append(
            "Generic DCF/EPV anchors are suppressed for this security because the insurance/security routing says the generic model is not valid."
        )
        lines.append("")
    return True


def render_alpha_report(
    report: SectorAlphaReport,
    *,
    as_of_date: str | None = None,
    financial_source_rows: dict[str, list[dict[str, Any]]] | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Render Alpha Markdown from the supplied immutable financial snapshot."""
    lines: list[str] = []
    rendered_at = generated_at or datetime.now(timezone.utc)
    if rendered_at.tzinfo is None:
        rendered_at = rendered_at.replace(tzinfo=timezone.utc)
    now = rendered_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Header
    lines.append(f"# Alpha Scan Report: {report.sector}")
    lines.append(f"")
    lines.append(f"**Generated:** {now}")
    if as_of_date:
        lines.append(f"**Financial evidence as of:** {as_of_date}")
    if report.financial_integrity_binding:
        fingerprint = str(
            report.financial_integrity_binding.get("publication_scope_fingerprint") or ""
        )
        if fingerprint:
            lines.append(f"**Financial scope fingerprint:** `{fingerprint}`")
    lines.append(f"**Total candidates:** {report.total_candidates}")
    lines.append(f"**Elimination rounds:** {len(report.rounds)}")
    lines.append(f"**Selection basis:** {report.selection_basis.replace('_', ' ')}")
    lines.append(f"**Winner conviction:** {report.winner_conviction}")
    if report.pre_investigation_winner:
        if report.winner and report.pre_investigation_winner != report.winner:
            lines.append(
                f"**Consensus leader before investigation:** {report.pre_investigation_winner}"
            )
        elif report.winner is None:
            lines.append(
                f"**Consensus leader rejected by post-investigation screen:** "
                f"{report.pre_investigation_winner}"
            )
    lines.append("")

    # Winner
    lines.append("---")
    lines.append("")
    lines.append(f"## Winner: {report.winner or 'None'}")
    lines.append("")
    if report.winner is None and report.pre_investigation_winner:
        lines.append(
            f"Post-investigation screening found no long candidate. "
            f"Consensus leader was **{report.pre_investigation_winner}**."
        )
        lines.append("")
    if report.selection_basis == "llm_no_winner":
        decision_mode = str(report.llm_decision_trace.get("decision_mode") or "")
        if decision_mode == "fallback":
            lines.append(
                "**DEGRADED RESULT:** The final LLM decision failed, so this report is non-actionable until the run is repeated."
            )
        else:
            lines.append(
                "**NO ACTIONABLE WINNER:** The investigated candidates did not clear the eligibility and evidence threshold."
            )
        lines.append("")
    if report.winner:
        winner_pkt = report.signal_packets.get(report.winner, {})
        insurance_packet = _insurance_packet_for(report, report.winner)
        generic_allowed = _generic_allowed(insurance_packet)
        sc = _scorecard_summary_from_packet(winner_pkt)
        _append_insurance_summary(lines, sc=sc, insurance_packet=insurance_packet)

        if sc and generic_allowed:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Current Price | {_fmt(sc.get('current_price'), '.2f', '$', '')} |")
            lines.append(f"| DCF Intrinsic | {_fmt(sc.get('dcf_base'), '.2f', '$', '')} |")
            lines.append(f"| EPV Intrinsic | {_fmt(sc.get('epv_adjusted'), '.2f', '$', '')} |")
            lines.append(f"| DCF Discount | {_pct(sc.get('dcf_discount'))} |")
            lines.append(f"| Gate Verdict | {sc.get('gate_action', '—')} |")
            lines.append(f"| Signal | {sc.get('signal', '—')} |")
            lines.append(f"| Moat | {sc.get('moat_class', '—')} ({sc.get('moat_score', '—')}) |")
            lines.append(f"| Downside Risk | {sc.get('downside_risk', '—')} |")
            lines.append(f"| Earnings Quality | {sc.get('earnings_quality', '—')} |")
            lines.append(f"| Revenue CAGR (5y) | {_pct(sc.get('revenue_cagr_5y'))} |")
            lines.append(f"| Revenue CAGR (3y) | {_pct(sc.get('revenue_cagr_3y'))} |")
            lines.append("")
            if sc.get("headwinds"):
                lines.append(f"**Headwinds:** {', '.join(sc['headwinds'])}")
                lines.append("")
            if sc.get("supports"):
                lines.append(f"**Supports:** {', '.join(sc['supports'])}")
                lines.append("")

        # Method tension
        tension_type = winner_pkt.get("method_tension_type")
        if tension_type and tension_type not in ("NONE", "INSUFFICIENT_METHODS"):
            lines.append(f"**Method Tension: {tension_type}**")
            lines.append("")
            growth_dep = winner_pkt.get("growth_dependency_ratio")
            if growth_dep and growth_dep > 0:
                lines.append(
                    f"- Growth dependency: {growth_dep:.0%} of DCF value is growth premium"
                )
            ir_low = winner_pkt.get("intrinsic_range_low")
            ir_high = winner_pkt.get("intrinsic_range_high")
            if ir_low is not None and ir_high is not None:
                lines.append(f"- Intrinsic value range: ${ir_low:.0f} - ${ir_high:.0f}")
            consensus = winner_pkt.get("consensus_direction")
            if consensus:
                lines.append(f"- Method consensus: {consensus}")
            lines.append("")

    lines.append("### Thesis")
    lines.append("")
    lines.append(report.winner_thesis)
    lines.append("")
    if report.llm_decision_trace:
        decision_trace = str(report.llm_decision_trace.get("decision_trace") or "").strip()
        if decision_trace:
            lines.append("### Final LLM Decision")
            lines.append("")
            lines.append(decision_trace)
            lines.append("")
        rejected = report.llm_decision_trace.get("rejected_candidates") or []
        if rejected:
            lines.append("**Rejected Candidates:**")
            lines.append("")
            for item in rejected:
                if isinstance(item, dict):
                    lines.append(
                        f"- {item.get('ticker', '—')}: {str(item.get('reason') or '').strip() or 'Rejected after investigation.'}"
                    )
            lines.append("")

    if report.prior_ranking:
        lines.append("### Deterministic Prior")
        lines.append("")
        lines.append("| Rank | Ticker | Consensus Score | Hard Blocks | Method Tension |")
        lines.append("|------|--------|-----------------|-------------|----------------|")
        for item in report.prior_ranking:
            hard_blocks = ", ".join(item.get("hard_block_reasons") or []) or "—"
            lines.append(
                f"| {item.get('consensus_rank', '—')} "
                f"| {item.get('ticker', '—')} "
                f"| {float(item.get('consensus_score', 0.0)):+.2f} "
                f"| {hard_blocks} "
                f"| {item.get('method_tension_type', '—') or '—'} |"
            )
        lines.append("")

    # Financials
    if report.winner:
        if financial_source_rows is None:
            financials = _load_financials(
                report.winner,
                as_of_date=as_of_date,
            )
        else:
            financials = derive_alpha_report_financials(
                financial_source_rows.get(str(report.winner).strip().upper(), []),
                years=5,
            )
        if financials:
            lines.append("### Financial Summary")
            lines.append("")
            lines.append(
                "| Year | Revenue ($M) | Op Income ($M) | Margin | Net Income ($M) | CFO ($M) | FCF ($M) |"
            )
            lines.append(
                "|------|-------------|---------------|--------|----------------|---------|---------|"
            )
            for f in financials:
                lines.append(
                    f"| {f['year']} "
                    f"| {_fmt_plain_number(f['revenue'])} "
                    f"| {_fmt_plain_number(f['operating_income'])} "
                    f"| {_fmt_plain_pct(f['operating_margin'])} "
                    f"| {_fmt_plain_number(f['net_income'])} "
                    f"| {_fmt_plain_number(f['cfo'])} "
                    f"| {_fmt_plain_number(f['fcf'])} |"
                )
            lines.append("")
            # Margin trend warning
            if len(financials) >= 3:
                margins = [
                    f["operating_margin"]
                    for f in financials[:3]
                    if f["operating_margin"] is not None
                ]
                if len(margins) == 3 and margins[0] < margins[1] < margins[2]:
                    lines.append(
                        f"**WARNING: Operating margins declining** — {margins[2]:.1f}% -> {margins[1]:.1f}% -> {margins[0]:.1f}% over 3 years. DCF may overstate intrinsic value if margin compression continues."
                    )
                    lines.append("")
                elif len(margins) == 3 and margins[0] > margins[1] > margins[2]:
                    lines.append(
                        f"**Positive: Operating margins expanding** — {margins[2]:.1f}% -> {margins[1]:.1f}% -> {margins[0]:.1f}% over 3 years."
                    )
                    lines.append("")

    # Research findings for winner
    if report.winner and report.winner in report.signal_packets:
        winner_pkt = report.signal_packets[report.winner]
        research = winner_pkt.get("research_report", {})
        anomalies = research.get("anomalies", [])
        investigations = research.get("investigations", [])
        solvency = research.get("solvency", {})

        if anomalies or solvency.get("signals"):
            lines.append("### Research Findings")
            lines.append("")

            # Solvency
            sol_risk = solvency.get("risk", "UNKNOWN")
            if sol_risk in ("CRITICAL", "ELEVATED"):
                lines.append(f"**Solvency Risk: {sol_risk}**")
                lines.append("")
                lines.append(f"{solvency.get('details', '')}")
                lines.append("")

            # Anomalies
            if anomalies:
                lines.append("**Anomalies Detected:**")
                lines.append("")
                lines.append("| Type | Severity | Description |")
                lines.append("|------|----------|-------------|")
                for a in anomalies:
                    lines.append(f"| {a['type']} | {a['severity']} | {a['description'][:120]} |")
                lines.append("")

            # Investigations
            if investigations:
                lines.append("**Filing Investigation Results:**")
                lines.append("")
                for inv in investigations:
                    lines.append(f"**Q:** {inv['question']}")
                    lines.append("")
                    lines.append(f"**A:** {inv['answer']}")
                    lines.append("")
                    if inv.get("evidence"):
                        lines.append(f"> {inv['evidence'][:300]}")
                        lines.append("")

    # Data quality
    if report.signal_packets:
        total_p = len(report.signal_packets)
        with_price = sum(
            1
            for p in report.signal_packets.values()
            if isinstance(p.get("current_price"), (int, float)) and p["current_price"] > 0
        )
        with_gate = sum(
            1 for p in report.signal_packets.values() if p.get("gate_verdict") is not None
        )
        with_risk = sum(
            1
            for p in report.signal_packets.values()
            if isinstance(p.get("filing_risk_signals"), dict)
            and p["filing_risk_signals"].get("competitive_disruption") != "UNKNOWN"
        )
        lines.append("---")
        lines.append("")
        lines.append("## Data Quality")
        lines.append("")
        lines.append(f"| Metric | Count | % |")
        lines.append(f"|--------|-------|---|")
        lines.append(f"| Total tickers | {total_p} | 100% |")
        lines.append(f"| With live price | {with_price} | {with_price / total_p * 100:.0f}% |")
        lines.append(f"| With quality gate | {with_gate} | {with_gate / total_p * 100:.0f}% |")
        lines.append(f"| With filing risk scan | {with_risk} | {with_risk / total_p * 100:.0f}% |")
        lines.append("")
        if with_price < total_p * 0.5:
            lines.append(
                "**WARNING:** Less than 50% of tickers have live price data. Valuation rankings may be biased toward tickers with available prices."
            )
            lines.append("")

    # Runner-up
    lines.append("---")
    lines.append("")
    lines.append(f"## Runner-Up: {report.runner_up or 'None'}")
    lines.append("")
    if report.runner_up:
        runner_packet = report.signal_packets.get(report.runner_up, {})
        sc2 = _scorecard_summary_from_packet(runner_packet)
        ru_insurance_packet = _insurance_packet_for(report, report.runner_up)
        ru_generic_allowed = _generic_allowed(ru_insurance_packet)
        _append_insurance_summary(lines, sc=sc2, insurance_packet=ru_insurance_packet)
        if sc2 and ru_generic_allowed:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Current Price | {_fmt(sc2.get('current_price'), '.2f', '$', '')} |")
            lines.append(f"| DCF Intrinsic | {_fmt(sc2.get('dcf_base'), '.2f', '$', '')} |")
            lines.append(f"| Gate Verdict | {sc2.get('gate_action', '—')} |")
            lines.append(f"| Moat | {sc2.get('moat_class', '—')} ({sc2.get('moat_score', '—')}) |")
            lines.append(f"| Downside Risk | {sc2.get('downside_risk', '—')} |")
            lines.append("")
    lines.append(report.runner_up_thesis)
    lines.append("")
    if report.runner_up:
        if financial_source_rows is None:
            ru_financials = _load_financials(
                report.runner_up,
                as_of_date=as_of_date,
                years=3,
            )
        else:
            ru_financials = derive_alpha_report_financials(
                financial_source_rows.get(str(report.runner_up).strip().upper(), []),
                years=3,
            )
        if ru_financials:
            lines.append("| Year | Revenue ($M) | Op Income ($M) | Margin | CFO ($M) | FCF ($M) |")
            lines.append("|------|-------------|---------------|--------|---------|---------|")
            for f in ru_financials:
                lines.append(
                    f"| {f['year']} "
                    f"| {_fmt_plain_number(f['revenue'])} "
                    f"| {_fmt_plain_number(f['operating_income'])} "
                    f"| {_fmt_plain_pct(f['operating_margin'])} "
                    f"| {_fmt_plain_number(f['cfo'])} "
                    f"| {_fmt_plain_number(f['fcf'])} |"
                )
            lines.append("")

    if report.investigation_plan:
        lines.append("---")
        lines.append("")
        lines.append("## LLM Investigation Plan")
        lines.append("")
        summary = str(report.investigation_plan.get("summary") or "").strip()
        if summary:
            lines.append(summary)
            lines.append("")
        targets = report.investigation_plan.get("selected_targets") or []
        if targets:
            lines.append("| Ticker | Why Investigate | Evidence Gaps |")
            lines.append("|--------|------------------|---------------|")
            for target in targets:
                lines.append(
                    f"| {target.get('ticker', '—')} "
                    f"| {str(target.get('reason') or '—').replace('|', '/')} "
                    f"| {', '.join(target.get('evidence_gaps') or []) or '—'} |"
                )
            lines.append("")

    if report.candidate_investigations:
        lines.append("---")
        lines.append("")
        lines.append("## Candidate Investigations")
        lines.append("")
        lines.append(
            "| Ticker | Prior Rank | Verdict | Confidence | Eligible | Selection Blockers | Confidence Caps | Tool Mode |"
        )
        lines.append(
            "|--------|------------|---------|------------|----------|--------------------|-----------------|-----------|"
        )
        for investigation in report.candidate_investigations:
            blockers = (
                ", ".join(
                    investigation.get("selection_blockers")
                    or investigation.get("hard_block_reasons")
                    or []
                )
                or "—"
            )
            confidence_caps = ", ".join(investigation.get("confidence_cap_reasons") or []) or "—"
            lines.append(
                f"| {investigation.get('ticker', '—')} "
                f"| {investigation.get('consensus_rank', '—')} "
                f"| {investigation.get('verdict', '—')} "
                f"| {investigation.get('confidence', '—')} "
                f"| {investigation.get('eligible_for_selection', '—')} "
                f"| {blockers} "
                f"| {confidence_caps} "
                f"| {investigation.get('investigation_mode', '—')} |"
            )
        lines.append("")
        for investigation in report.candidate_investigations:
            lines.append(f"### {investigation.get('ticker', '—')}")
            lines.append("")
            findings = investigation.get("key_findings") or []
            if findings:
                lines.append("**Decision-Relevant Findings:**")
                lines.append("")
                for finding in findings:
                    lines.append(f"- {finding}")
                lines.append("")
            open_questions = investigation.get("open_questions") or []
            if open_questions:
                lines.append("**Open Questions:**")
                lines.append("")
                for question in open_questions:
                    lines.append(f"- {question}")
                lines.append("")
            if investigation.get("key_risk"):
                lines.append(f"**Candidate Risk:** {investigation.get('key_risk')}")
                lines.append("")
            if investigation.get("falsification_trigger"):
                lines.append(
                    f"**Candidate Falsifier:** {investigation.get('falsification_trigger')}"
                )
                lines.append("")

    if report.tool_budget_summary:
        lines.append("---")
        lines.append("")
        lines.append("## Tool Budget")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Provider Mode | {report.tool_budget_summary.get('provider_mode', '—')} |")
        lines.append(
            f"| Investigated Candidates | {report.tool_budget_summary.get('investigated_candidates', '—')} |"
        )
        lines.append(
            f"| Total Tool Calls | {report.tool_budget_summary.get('total_tool_calls', '—')} |"
        )
        lines.append(f"| Total Turns | {report.tool_budget_summary.get('total_turns', '—')} |")
        lines.append(
            f"| Total Cost (USD) | {report.tool_budget_summary.get('total_cost_usd', '—')} |"
        )
        lines.append("")

    # Risk and falsification
    lines.append("---")
    lines.append("")
    lines.append("## Key Risk")
    lines.append("")
    lines.append(report.key_risk)
    lines.append("")
    lines.append("## Falsification Trigger")
    lines.append("")
    lines.append(report.falsification_trigger)
    lines.append("")
    lines.append(f"**Time Horizon:** {report.time_horizon}")
    lines.append("")

    # Elimination rounds
    lines.append("---")
    lines.append("")
    lines.append("## Elimination Rounds")
    lines.append("")
    for r in report.rounds:
        entering = len(r.candidates_entering)
        remaining = len(r.candidates_remaining)
        cut = len(r.candidates_eliminated)
        lines.append(f"### Round {r.round_number}: {entering} -> {remaining} (cut {cut})")
        lines.append("")
        lines.append(f"**Criteria:** {r.elimination_criteria}")
        lines.append("")
        if cut <= 15:
            lines.append(f"**Eliminated:** {', '.join(r.candidates_eliminated)}")
        else:
            lines.append(f"**Eliminated:** {cut} tickers")
        lines.append("")
        if r.reasoning:
            lines.append(f"**Reasoning:** {r.reasoning[:500]}")
            lines.append("")

    # Finalists comparison table
    lines.append("---")
    lines.append("")
    lines.append("## Finalists Comparison")
    lines.append("")

    # Get the last round's remaining candidates
    if report.rounds:
        last_round = max(report.rounds, key=lambda r: r.round_number)
        finalist_tickers = last_round.candidates_remaining
    else:
        finalist_tickers = [report.winner] if report.winner else []

    if finalist_tickers:
        lines.append(
            "| Ticker | Price | Primary Anchor | Discount | Gate | Moat | Downside | Rev CAGR 5y |"
        )
        lines.append(
            "|--------|-------|----------------|----------|------|------|----------|------------|"
        )
        for t in finalist_tickers:
            packet_summary = report.signal_packets.get(t, {})
            sc = _scorecard_summary_from_packet(packet_summary)
            insurance_packet = (
                packet_summary.get("insurance_packet")
                if isinstance(packet_summary.get("insurance_packet"), dict)
                else {}
            )
            valuation = (
                insurance_packet.get("valuation")
                if isinstance(insurance_packet.get("valuation"), dict)
                else {}
            )
            anchor = (
                valuation.get("valuation_anchor")
                if insurance_packet.get("generic_valuation_valid") is False
                else sc.get("dcf_base")
            )
            discount = None
            price = sc.get("current_price") or valuation.get("current_price")
            if isinstance(anchor, (int, float)) and isinstance(price, (int, float)) and anchor > 0:
                discount = (anchor - price) / anchor
            lines.append(
                f"| **{t}** "
                f"| {_fmt(price, '.2f', '$', '')} "
                f"| {_fmt(anchor, '.2f', '$', '')} "
                f"| {_pct(discount)} "
                f"| {sc.get('gate_action', '—')} "
                f"| {sc.get('moat_class', '—')} ({sc.get('moat_score', '—')}) "
                f"| {sc.get('downside_risk', '—')} "
                f"| {_pct(sc.get('revenue_cagr_5y'))} |"
            )
        lines.append("")

    return "\n".join(lines)


def generate_alpha_report(
    report: SectorAlphaReport,
    output_path: Path,
    *,
    as_of_date: str | None = None,
    financial_source_rows: dict[str, list[dict[str, Any]]] | None = None,
    generated_at: datetime | None = None,
) -> Path:
    """Generate a Markdown report and save it to disk.

    The optional source rows keep direct callers backward compatible while
    allowing the CLI publication path to prohibit post-gate database rereads.
    """

    md = render_alpha_report(
        report,
        as_of_date=as_of_date,
        financial_source_rows=financial_source_rows,
        generated_at=generated_at,
    )
    output_path.write_text(md, encoding="utf-8")
    logger.info("Alpha report written to %s", output_path)
    return output_path
