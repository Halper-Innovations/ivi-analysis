from __future__ import annotations

import json
from typing import Any

from app.config import get_config
from app.logging import get_logger
from app.report.claims import validate_claims_have_evidence_or_derivation


UNKNOWN = "UNKNOWN"
logger = get_logger(__name__)


def _claims_from_items(items: list[dict[str, Any]], derived_signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for item in items:
        value = item.get("value")
        if not isinstance(value, (int, float)):
            continue
        claims.append(
            {
                "claim_id": f"{item.get('year')}_{item.get('metric')}",
                "label": f"{item.get('metric')}::{item.get('year')}",
                "value": value,
                "unit": None,
                "citations": item.get("citations") or [],
                "derived_from": item.get("derived_from") or [],
            }
        )
    for row in derived_signals:
        value = row.get("value")
        if not isinstance(value, (int, float)):
            continue
        claims.append(
            {
                "claim_id": f"derived_{row.get('signal')}",
                "label": str(row.get("signal")),
                "value": value,
                "unit": None,
                "citations": [],
                "derived_from": row.get("derived_from") or [],
            }
        )
    return claims


def _latest_value(rows: list[dict[str, Any]], metric: str) -> Any:
    if not rows:
        return UNKNOWN
    return rows[-1].get(metric, UNKNOWN)


def _delta_value(rows: list[dict[str, Any]], metric: str) -> Any:
    if len(rows) < 2:
        return UNKNOWN
    first = rows[0].get(metric, UNKNOWN)
    last = rows[-1].get(metric, UNKNOWN)
    if isinstance(first, (int, float)) and isinstance(last, (int, float)):
        return float(last) - float(first)
    return UNKNOWN


def write_ticker_dossier(
    *,
    run_id: str,
    ticker: str,
    as_of_date: str,
    docket: list[dict[str, Any]],
    section_spans: dict[str, list[dict[str, Any]]],
    items: list[dict[str, Any]],
    time_series: dict[str, Any],
    dossier_quality: str = "FULL",
    filing_body_cached: bool = True,
) -> dict[str, Any]:
    cfg = get_config()
    out_dir = cfg.dossiers_dir / run_id / ticker.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(time_series.get("rows") or [])
    standardized_rows = list(time_series.get("standardized_rows") or [])
    derived_signals = list(time_series.get("derived_signals") or [])
    claims = _claims_from_items(items, derived_signals)
    claim_failures = validate_claims_have_evidence_or_derivation(claims)
    if claim_failures:
        raise ValueError(f"dossier claim trace validation failed for {ticker}: {claim_failures}")

    latest_revenue = _latest_value(rows, "revenue")
    gross_margin_delta = _delta_value(rows, "gross_margin")
    op_margin_delta = _delta_value(rows, "operating_margin")
    fcf_margin_delta = _delta_value(rows, "fcf_margin")
    risk_delta = _delta_value(rows, "risk_factor_keyword_count")
    dilution_proxy = next((x.get("value") for x in derived_signals if x.get("signal") == "dilution_rate_proxy"), UNKNOWN)
    deferred_revenue_ratio = next((x.get("value") for x in derived_signals if x.get("signal") == "deferred_revenue_to_revenue_latest"), UNKNOWN)
    rpo_ratio = next((x.get("value") for x in derived_signals if x.get("signal") == "rpo_to_revenue_latest"), UNKNOWN)
    rnd_intensity_delta = next((x.get("value") for x in derived_signals if x.get("signal") == "r_and_d_intensity_delta"), UNKNOWN)
    segment_count_delta = next((x.get("value") for x in derived_signals if x.get("signal") == "segment_count_delta"), UNKNOWN)
    customer_concentration_delta = next((x.get("value") for x in derived_signals if x.get("signal") == "customer_concentration_delta"), UNKNOWN)

    md = [
        f"# {ticker.upper()} 10-K Dossier",
        "",
        f"- Run ID: `{run_id}`",
        f"- As Of Date: `{as_of_date}`",
        f"- Dossier Quality: `{str(dossier_quality or 'FULL').upper()}`",
        f"- Filing Body Cached: `{bool(filing_body_cached)}`",
        f"- Years Covered: `{len(rows)}`",
        "",
        "## Compounding Engine",
        f"- Latest Revenue: {latest_revenue if isinstance(latest_revenue, (int, float)) else 'UNKNOWN'}",
        f"- Gross Profit Dollars Trend: {next((x.get('value') for x in derived_signals if x.get('signal') == 'gross_profit_dollars_trend'), UNKNOWN)}",
        "",
        "## Unit Economics Proxies",
        f"- Gross Margin Delta: {gross_margin_delta if isinstance(gross_margin_delta, (int, float)) else 'UNKNOWN'}",
        f"- Operating Margin Delta: {op_margin_delta if isinstance(op_margin_delta, (int, float)) else 'UNKNOWN'}",
        f"- FCF Margin Delta: {fcf_margin_delta if isinstance(fcf_margin_delta, (int, float)) else 'UNKNOWN'}",
        "",
        "## Moat Signals",
        f"- Segment Count (latest): {_latest_value(rows, 'segment_count')}",
        f"- Segment Count Delta: {segment_count_delta if isinstance(segment_count_delta, (int, float)) else 'UNKNOWN'}",
        f"- Customer Concentration Flag (latest): {_latest_value(rows, 'customer_concentration_present')}",
        f"- Customer Concentration (latest): {_latest_value(rows, 'customer_concentration_pct')}",
        f"- Customer Concentration Delta: {customer_concentration_delta if isinstance(customer_concentration_delta, (int, float)) else 'UNKNOWN'}",
        "",
        "## Capital Allocation",
        f"- Acquisition Mentions (latest): {_latest_value(rows, 'acquisition_mentions_count')}",
        f"- R&D Intensity (latest): {next((x.get('value') for x in derived_signals if x.get('signal') == 'r_and_d_intensity_latest'), UNKNOWN)}",
        f"- R&D Intensity Delta: {rnd_intensity_delta if isinstance(rnd_intensity_delta, (int, float)) else 'UNKNOWN'}",
        f"- Sales and Marketing (latest): {_latest_value(rows, 'sales_marketing_total')}",
        f"- General and Administrative (latest): {_latest_value(rows, 'g_and_a_total')}",
        f"- Share Repurchases (latest): {_latest_value(rows, 'share_repurchases_amount')}",
        f"- Dividends Paid (latest): {_latest_value(rows, 'dividends_paid_amount')}",
        "",
        "## Revenue Visibility",
        f"- Deferred Revenue (latest): {_latest_value(rows, 'deferred_revenue_amount')}",
        f"- Deferred Revenue / Revenue (latest): {deferred_revenue_ratio if isinstance(deferred_revenue_ratio, (int, float)) else 'UNKNOWN'}",
        f"- RPO (latest): {_latest_value(rows, 'rpo_amount')}",
        f"- RPO / Revenue (latest): {rpo_ratio if isinstance(rpo_ratio, (int, float)) else 'UNKNOWN'}",
        "",
        "## Risk Evolution",
        f"- Risk Factor Keyword Delta: {risk_delta if isinstance(risk_delta, (int, float)) else 'UNKNOWN'}",
        "",
        "## Accounting/Dilution",
        f"- SBC/Dilution Indicator (latest): {_latest_value(rows, 'sbc_dilution_indicator')}",
        f"- Dilution Rate Proxy: {dilution_proxy if isinstance(dilution_proxy, (int, float)) else 'UNKNOWN'}",
        "",
        "## What Changed Over 10 Years",
        "- Trends are computed from filing-derived yearly series only; UNKNOWN values are preserved when evidence is missing.",
        "",
        "## 10-Year Standardized Time Series",
        "| Year | Revenue | Gross Profit | Operating Income | Net Income | CFO | Capex | FCF | Shares | Shares YoY | Net Debt | R&D | S&M | G&A | Deferred Rev | RPO | Customer % | Segments |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in standardized_rows:
        md.append(
            f"| {row.get('year')} | {row.get('revenue', UNKNOWN)} | {row.get('gross_profit', UNKNOWN)} | "
            f"{row.get('operating_income', UNKNOWN)} | {row.get('net_income', UNKNOWN)} | {row.get('cfo', UNKNOWN)} | "
            f"{row.get('capex', UNKNOWN)} | {row.get('fcf', UNKNOWN)} | {row.get('shares_outstanding', UNKNOWN)} | "
            f"{row.get('shares_yoy_change', UNKNOWN)} | {row.get('net_debt', UNKNOWN)} | {row.get('r_and_d_total', UNKNOWN)} | "
            f"{row.get('sales_marketing_total', UNKNOWN)} | {row.get('g_and_a_total', UNKNOWN)} | "
            f"{row.get('deferred_revenue_amount', UNKNOWN)} | {row.get('rpo_amount', UNKNOWN)} | "
            f"{row.get('customer_concentration_pct', UNKNOWN)} | {row.get('segment_count', UNKNOWN)} |"
        )
    md.extend(
        [
            "",
        "## Evidence Appendix",
        ]
    )
    for item in items[:25]:
        src = item.get("source_url") or ""
        snippet = str(item.get("snippet") or "")[:220]
        md.append(f"- {item.get('year')} {item.get('metric')}: {src} :: {snippet}")

    md_path = out_dir / "dossier.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    payload = {
        "ticker": ticker.upper(),
        "run_id": run_id,
        "as_of_date": as_of_date,
        "dossier_quality": str(dossier_quality or "FULL").upper(),
        "filing_body_cached": bool(filing_body_cached),
        "docket": docket,
        "section_spans": section_spans,
        "items": items,
        "time_series": time_series,
        "claims": claims,
        "claim_failures": claim_failures,
        "artifacts": {
            "dossier_md_path": str(md_path),
        },
    }
    json_path = out_dir / "dossier.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["artifacts"]["dossier_json_path"] = str(json_path)
    logger.info("Dossier written for %s: %s", ticker.upper(), json_path)
    return payload
