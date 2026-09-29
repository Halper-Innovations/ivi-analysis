from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from app.fundamentals.normalize import UNKNOWN
from app.util.financial_data_access import ANNUAL_COMPANYFACTS_PERIOD_TYPES
from app.util.issuer_classification import ISSUER_CLASS_FINANCIAL, resolve_issuer_classification
from app.util.text import try_parse_number


@dataclass
class DiscoveryMetricsResult:
    effective_as_of_date: str
    metrics: dict[str, Any]
    claims: list[dict[str, Any]]
    flags: list[str]
    filing_accessions_used: list[str]


def _safe_div(numerator: float | None, denominator: float | None) -> float | str:
    if numerator is None or denominator in (None, 0):
        return UNKNOWN
    try:
        return float(numerator) / float(denominator)
    except Exception:
        return UNKNOWN


def _parse_shares(value_json: str) -> float | None:
    try:
        payload = json.loads(value_json)
    except Exception:
        return None
    value = payload.get("value")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return try_parse_number(value)
    return None


def _shares_series_for_accessions(conn, ticker: str, selected_accessions: list[str]) -> list[float]:
    if not selected_accessions:
        return []
    placeholders = ",".join("?" for _ in selected_accessions)
    rows = conn.execute(
        f"""
        SELECT ef.value_json
        FROM extracted_facts ef
        JOIN filings f ON ef.filing_id = f.id
        WHERE f.ticker = ? AND ef.fact_type = 'shares_outstanding' AND f.accession IN ({placeholders})
        ORDER BY COALESCE(f.filing_date, '1900-01-01') DESC
        LIMIT 8
        """,
        (ticker, *selected_accessions),
    ).fetchall()
    values: list[float] = []
    for row in rows:
        parsed = _parse_shares(str(row["value_json"] or ""))
        if isinstance(parsed, (int, float)) and parsed > 0:
            values.append(float(parsed))
    if values:
        return values
    fallback_rows = conn.execute(
        f"""
        SELECT value
        FROM companyfacts_facts
        WHERE ticker = ?
          AND period_type IN ({",".join("?" for _ in ANNUAL_COMPANYFACTS_PERIOD_TYPES)})
          AND line_item = 'shares_outstanding'
          AND value IS NOT NULL
          AND filed_date IS NOT NULL
          AND TRIM(filed_date) != ''
          AND accession IN ({placeholders})
        ORDER BY filed_date DESC, fiscal_year DESC, id DESC
        LIMIT 8
        """,
        (
            ticker.upper(),
            *ANNUAL_COMPANYFACTS_PERIOD_TYPES,
            *selected_accessions,
        ),
    ).fetchall()
    for row in fallback_rows:
        value = row["value"]
        if isinstance(value, (int, float)) and value > 0:
            values.append(float(value))
    return values


def _line_items_for_filing(conn, filing_id: int) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT line_item, value, source_url, snippet
        FROM financials
        WHERE filing_id = ?
        ORDER BY id DESC
        """,
        (filing_id,),
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        line_item = str(row["line_item"] or "")
        if not line_item or line_item in out:
            continue
        out[line_item] = {
            "value": row["value"],
            "source_url": row["source_url"] or "",
            "snippet": row["snippet"] or "",
        }
    return out


def _year_for_filing_row(row: Any) -> int | None:
    period_end = str(row["period_end"] or "") if "period_end" in row.keys() else ""
    filing_date = str(row["filing_date"] or "") if "filing_date" in row.keys() else ""
    for raw in (period_end, filing_date):
        if len(raw) >= 4 and raw[:4].isdigit():
            return int(raw[:4])
    return None


def _companyfacts_line_items_for_year(
    conn,
    ticker: str,
    fiscal_year: int | None,
    *,
    period_end: str | None,
    filed_date: str | None,
    accession: str | None,
) -> dict[str, dict[str, Any]]:
    if (
        fiscal_year is None
        or not str(period_end or "").strip()
        or not str(filed_date or "").strip()
        or not str(accession or "").strip()
    ):
        return {}
    annual_period_types = ANNUAL_COMPANYFACTS_PERIOD_TYPES
    placeholders = ",".join("?" for _ in annual_period_types)
    rows = conn.execute(
        f"""
        SELECT
            line_item, value, units, source_url, period_end, filed_date,
            form, accession
        FROM companyfacts_facts
        WHERE ticker = ?
          AND fiscal_year = ?
          AND period_type IN ({placeholders})
          AND period_end = ?
          AND filed_date IS NOT NULL
          AND TRIM(filed_date) != ''
          AND filed_date <= ?
          AND accession = ?
        ORDER BY filed_date DESC, id DESC
        """,
        (
            ticker.upper(),
            fiscal_year,
            *annual_period_types,
            str(period_end),
            str(filed_date),
            str(accession),
        ),
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        line_item = str(row["line_item"] or "")
        if not line_item or line_item in out:
            continue
        out[line_item] = {
            "value": row["value"],
            "units": row["units"],
            "source_url": row["source_url"] or "",
            "snippet": "",
            "period_end": row["period_end"],
            "filed_date": row["filed_date"],
            "form": row["form"],
            "accession": row["accession"],
        }
    return out


def _sum_quarter_values(
    quarters: list[dict[str, dict[str, Any]]], line_item: str, required: int = 4
) -> float | None:
    if len(quarters) < required:
        return None
    values: list[float] = []
    for q in quarters[:required]:
        row = q.get(line_item, {})
        value = row.get("value")
        if not isinstance(value, (int, float)):
            return None
        values.append(float(value))
    return float(sum(values))


def _latest_citation(
    quarters: list[dict[str, dict[str, Any]]],
    annual: dict[str, dict[str, Any]] | None,
    line_item: str,
) -> list[dict[str, str]]:
    for bucket in [*(quarters or []), annual or {}]:
        row = bucket.get(line_item, {}) if isinstance(bucket, dict) else {}
        source_url = row.get("source_url")
        if source_url:
            return [
                {
                    "source_url": source_url,
                    "snippet": row.get("snippet", ""),
                    "section_label": None,
                }
            ]
    return []


def _growth(current: float | None, prior: float | None) -> float | str:
    if current is None or prior in (None, 0):
        return UNKNOWN
    return (float(current) - float(prior)) / abs(float(prior))


def _liquidity_score_for_filings(conn, filing_ids: list[int]) -> int:
    if not filing_ids:
        return 10
    placeholders = ",".join("?" for _ in filing_ids)
    rows = conn.execute(
        f"""
        SELECT fact_type
        FROM extracted_facts
        WHERE filing_id IN ({placeholders})
        """,
        tuple(filing_ids),
    ).fetchall()
    penalty = 0
    for row in rows:
        fact_type = str(row["fact_type"] or "")
        if fact_type in {
            "going_concern",
            "covenant_pressure",
            "refinancing_need",
            "material_weakness",
        }:
            penalty += 2
        elif fact_type in {"dilution_risk", "debt_liquidity_signal"}:
            penalty += 1
    return min(10, penalty)


def compute_discovery_metrics(
    conn,
    *,
    ticker: str,
    selected_accessions: list[str],
) -> DiscoveryMetricsResult | None:
    if not selected_accessions:
        return None
    placeholders = ",".join("?" for _ in selected_accessions)
    rows = conn.execute(
        f"""
        SELECT id, accession, form_type, filing_date, period_end
        FROM filings
        WHERE ticker = ? AND accession IN ({placeholders})
        ORDER BY COALESCE(filing_date, '1900-01-01') DESC
        """,
        (ticker, *selected_accessions),
    ).fetchall()
    if not rows:
        return None

    latest_filing_date = str(rows[0]["filing_date"] or "")
    annual_row = next(
        (row for row in rows if str(row["form_type"]).upper() in {"10-K", "20-F"}), None
    )
    quarter_rows = [row for row in rows if str(row["form_type"]).upper() == "10-Q"]

    annual_fin = None
    if annual_row:
        annual_year = _year_for_filing_row(annual_row)
        annual_fin = _companyfacts_line_items_for_year(
            conn,
            ticker,
            annual_year,
            period_end=annual_row["period_end"],
            filed_date=annual_row["filing_date"],
            accession=annual_row["accession"],
        ) or _line_items_for_filing(conn, int(annual_row["id"]))
    quarter_fin = [_line_items_for_filing(conn, int(row["id"])) for row in quarter_rows]

    ttm_revenue = _sum_quarter_values(quarter_fin, "revenue", required=4)
    ttm_gross_profit = _sum_quarter_values(quarter_fin, "gross_profit", required=4)
    ttm_operating_income = _sum_quarter_values(quarter_fin, "operating_income", required=4)
    ttm_cfo = _sum_quarter_values(quarter_fin, "cfo", required=4)
    ttm_capex = _sum_quarter_values(quarter_fin, "capex", required=4)
    ttm_net_income = _sum_quarter_values(quarter_fin, "net_income", required=4)

    annual_revenue = annual_fin.get("revenue", {}).get("value") if annual_fin else None
    annual_gross_profit = annual_fin.get("gross_profit", {}).get("value") if annual_fin else None
    annual_operating_income = (
        annual_fin.get("operating_income", {}).get("value") if annual_fin else None
    )
    annual_cfo = annual_fin.get("cfo", {}).get("value") if annual_fin else None
    annual_capex = annual_fin.get("capex", {}).get("value") if annual_fin else None
    annual_net_income = annual_fin.get("net_income", {}).get("value") if annual_fin else None
    annual_cash = annual_fin.get("cash", {}).get("value") if annual_fin else None
    annual_total_debt = annual_fin.get("total_debt", {}).get("value") if annual_fin else None
    # SIC-first (VOE_ISSUER_CLASSIFICATION_BY_SIC); tag-name substring rule only when
    # no SIC code is on file for the ticker.
    issuer_classification, _classification_source = resolve_issuer_classification(
        ticker=ticker,
        line_items=set((annual_fin or {}).keys())
        | {key for bucket in quarter_fin for key in bucket.keys()},
        conn=conn,
    )
    is_financial = issuer_classification == ISSUER_CLASS_FINANCIAL

    revenue_base = ttm_revenue if isinstance(ttm_revenue, (int, float)) else annual_revenue
    gross_profit_base = (
        ttm_gross_profit if isinstance(ttm_gross_profit, (int, float)) else annual_gross_profit
    )
    operating_income_base = (
        ttm_operating_income
        if isinstance(ttm_operating_income, (int, float))
        else annual_operating_income
    )
    cfo_base = ttm_cfo if isinstance(ttm_cfo, (int, float)) else annual_cfo
    capex_base = ttm_capex if isinstance(ttm_capex, (int, float)) else annual_capex
    net_income_base = (
        ttm_net_income if isinstance(ttm_net_income, (int, float)) else annual_net_income
    )

    fcf_base = None
    if (
        not is_financial
        and isinstance(cfo_base, (int, float))
        and isinstance(capex_base, (int, float))
    ):
        fcf_base = float(cfo_base) - float(capex_base)

    gross_margin = _safe_div(
        float(gross_profit_base) if isinstance(gross_profit_base, (int, float)) else None,
        float(revenue_base) if isinstance(revenue_base, (int, float)) else None,
    )
    operating_margin = _safe_div(
        float(operating_income_base) if isinstance(operating_income_base, (int, float)) else None,
        float(revenue_base) if isinstance(revenue_base, (int, float)) else None,
    )

    latest_q = quarter_fin[0] if len(quarter_fin) >= 1 else {}
    prior_q = quarter_fin[1] if len(quarter_fin) >= 2 else {}
    third_q = quarter_fin[2] if len(quarter_fin) >= 3 else {}

    rev_recent_growth = _growth(
        latest_q.get("revenue", {}).get("value"), prior_q.get("revenue", {}).get("value")
    )
    rev_prior_growth = _growth(
        prior_q.get("revenue", {}).get("value"), third_q.get("revenue", {}).get("value")
    )
    revenue_acceleration = (
        (float(rev_recent_growth) - float(rev_prior_growth))
        if isinstance(rev_recent_growth, (int, float))
        and isinstance(rev_prior_growth, (int, float))
        else UNKNOWN
    )

    gm_recent = _safe_div(
        latest_q.get("gross_profit", {}).get("value"), latest_q.get("revenue", {}).get("value")
    )
    gm_prior = _safe_div(
        prior_q.get("gross_profit", {}).get("value"), prior_q.get("revenue", {}).get("value")
    )
    gross_margin_change = (
        float(gm_recent) - float(gm_prior)
        if isinstance(gm_recent, (int, float)) and isinstance(gm_prior, (int, float))
        else UNKNOWN
    )

    om_recent = _safe_div(
        latest_q.get("operating_income", {}).get("value"),
        latest_q.get("revenue", {}).get("value"),
    )
    om_prior = _safe_div(
        prior_q.get("operating_income", {}).get("value"),
        prior_q.get("revenue", {}).get("value"),
    )
    operating_margin_change = (
        float(om_recent) - float(om_prior)
        if isinstance(om_recent, (int, float)) and isinstance(om_prior, (int, float))
        else UNKNOWN
    )

    latest_fcf = None
    if (
        not is_financial
        and isinstance(latest_q.get("cfo", {}).get("value"), (int, float))
        and isinstance(latest_q.get("capex", {}).get("value"), (int, float))
    ):
        latest_fcf = float(latest_q["cfo"]["value"]) - float(latest_q["capex"]["value"])
    prior_fcf = None
    if (
        not is_financial
        and isinstance(prior_q.get("cfo", {}).get("value"), (int, float))
        and isinstance(prior_q.get("capex", {}).get("value"), (int, float))
    ):
        prior_fcf = float(prior_q["cfo"]["value"]) - float(prior_q["capex"]["value"])
    fcf_change = (
        (latest_fcf - prior_fcf) if latest_fcf is not None and prior_fcf is not None else UNKNOWN
    )

    shares_row = conn.execute(
        """
        SELECT ef.value_json
        FROM extracted_facts ef
        JOIN filings f ON ef.filing_id = f.id
        WHERE f.ticker = ? AND ef.fact_type = 'shares_outstanding' AND f.accession IN ({placeholders})
        ORDER BY COALESCE(f.filing_date, '1900-01-01') DESC
        LIMIT 1
        """.format(placeholders=placeholders),
        (ticker, *selected_accessions),
    ).fetchone()
    shares = _parse_shares(shares_row["value_json"]) if shares_row else None
    if not isinstance(shares, (int, float)) and annual_fin:
        annual_shares = annual_fin.get("shares_outstanding", {}).get("value")
        if isinstance(annual_shares, (int, float)):
            shares = float(annual_shares)

    filing_ids = [int(row["id"]) for row in rows]
    liquidity_score = _liquidity_score_for_filings(conn, filing_ids)

    net_debt = None
    if isinstance(annual_total_debt, (int, float)) and isinstance(annual_cash, (int, float)):
        net_debt = float(annual_total_debt) - float(annual_cash)

    cfo_margin = _safe_div(
        float(cfo_base) if isinstance(cfo_base, (int, float)) else None,
        float(revenue_base) if isinstance(revenue_base, (int, float)) else None,
    )
    fcf_margin = _safe_div(
        float(fcf_base) if isinstance(fcf_base, (int, float)) else None,
        float(revenue_base) if isinstance(revenue_base, (int, float)) else None,
    )
    cfo_to_net_income = _safe_div(
        float(cfo_base) if isinstance(cfo_base, (int, float)) else None,
        float(net_income_base) if isinstance(net_income_base, (int, float)) else None,
    )
    revenue_growth_yoy = _growth(
        float(ttm_revenue) if isinstance(ttm_revenue, (int, float)) else None,
        float(annual_revenue) if isinstance(annual_revenue, (int, float)) else None,
    )
    revenue_scale_log = (
        round(math.log10(float(revenue_base)), 4)
        if isinstance(revenue_base, (int, float)) and revenue_base > 0
        else UNKNOWN
    )

    shares_series = _shares_series_for_accessions(conn, ticker, selected_accessions)
    shares_change_4q: float | str = UNKNOWN
    if len(shares_series) >= 2 and shares_series[-1] != 0:
        shares_change_4q = (shares_series[0] - shares_series[-1]) / abs(shares_series[-1])

    metrics: dict[str, Any] = {
        "ttm_revenue": float(revenue_base) if isinstance(revenue_base, (int, float)) else UNKNOWN,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "cfo": float(cfo_base) if isinstance(cfo_base, (int, float)) else UNKNOWN,
        "capex": float(capex_base) if isinstance(capex_base, (int, float)) else UNKNOWN,
        "fcf": float(fcf_base) if isinstance(fcf_base, (int, float)) else UNKNOWN,
        "shares_outstanding": float(shares) if isinstance(shares, (int, float)) else UNKNOWN,
        "revenue_growth_recent": rev_recent_growth,
        "revenue_acceleration": revenue_acceleration,
        "gross_margin_change_qoq": gross_margin_change,
        "operating_margin_change_qoq": operating_margin_change,
        "fcf_change_qoq": fcf_change,
        "liquidity_stress_score": liquidity_score,
        "net_debt": float(net_debt) if isinstance(net_debt, (int, float)) else UNKNOWN,
        "cfo_margin": cfo_margin,
        "fcf_margin": fcf_margin,
        "cfo_to_net_income": cfo_to_net_income,
        "revenue_growth_yoy": revenue_growth_yoy,
        "revenue_scale_log": revenue_scale_log,
        "shares_change_4q": shares_change_4q,
        "issuer_classification": issuer_classification,
        "fcf_applicability": "sector_limited" if is_financial else "standard",
    }

    claims = [
        {
            "claim_id": "ttm_revenue",
            "label": "ttm_revenue",
            "value": metrics["ttm_revenue"],
            "unit": "USD",
            "citations": _latest_citation(quarter_fin, annual_fin, "revenue"),
            "derived_from": [
                "discovery.metrics.ttm_revenue",
                "companyfacts_facts.revenue",
                "financials.revenue",
            ],
        },
        {
            "claim_id": "gross_margin",
            "label": "gross_margin",
            "value": metrics["gross_margin"],
            "unit": "ratio",
            "citations": _latest_citation(quarter_fin, annual_fin, "gross_profit")
            + _latest_citation(quarter_fin, annual_fin, "revenue"),
            "derived_from": [
                "discovery.metrics.gross_margin",
                "companyfacts_facts.gross_profit",
                "companyfacts_facts.revenue",
                "financials.gross_profit",
                "financials.revenue",
            ],
        },
        {
            "claim_id": "operating_margin",
            "label": "operating_margin",
            "value": metrics["operating_margin"],
            "unit": "ratio",
            "citations": _latest_citation(quarter_fin, annual_fin, "operating_income")
            + _latest_citation(quarter_fin, annual_fin, "revenue"),
            "derived_from": [
                "discovery.metrics.operating_margin",
                "companyfacts_facts.operating_income",
                "companyfacts_facts.revenue",
                "financials.operating_income",
                "financials.revenue",
            ],
        },
        {
            "claim_id": "fcf",
            "label": "fcf",
            "value": metrics["fcf"],
            "unit": "USD",
            "citations": _latest_citation(quarter_fin, annual_fin, "cfo")
            + _latest_citation(quarter_fin, annual_fin, "capex"),
            "derived_from": [
                "discovery.metrics.fcf",
                "companyfacts_facts.cfo",
                "companyfacts_facts.capex",
                "financials.cfo",
                "financials.capex",
            ],
        },
        {
            "claim_id": "shares_outstanding",
            "label": "shares_outstanding",
            "value": metrics["shares_outstanding"],
            "unit": "shares",
            "citations": [],
            "derived_from": [
                "discovery.metrics.shares_outstanding",
                "companyfacts_facts.shares_outstanding",
                "extracted_facts.shares_outstanding",
            ],
        },
        {
            "claim_id": "cfo_margin",
            "label": "cfo_margin",
            "value": metrics["cfo_margin"],
            "unit": "ratio",
            "citations": _latest_citation(quarter_fin, annual_fin, "cfo")
            + _latest_citation(quarter_fin, annual_fin, "revenue"),
            "derived_from": [
                "discovery.metrics.cfo_margin",
                "companyfacts_facts.cfo",
                "companyfacts_facts.revenue",
                "financials.cfo",
                "financials.revenue",
            ],
        },
        {
            "claim_id": "fcf_margin",
            "label": "fcf_margin",
            "value": metrics["fcf_margin"],
            "unit": "ratio",
            "citations": _latest_citation(quarter_fin, annual_fin, "cfo")
            + _latest_citation(quarter_fin, annual_fin, "capex")
            + _latest_citation(quarter_fin, annual_fin, "revenue"),
            "derived_from": [
                "discovery.metrics.fcf_margin",
                "companyfacts_facts.cfo",
                "companyfacts_facts.capex",
                "companyfacts_facts.revenue",
                "financials.cfo",
                "financials.capex",
                "financials.revenue",
            ],
        },
        {
            "claim_id": "cfo_to_net_income",
            "label": "cfo_to_net_income",
            "value": metrics["cfo_to_net_income"],
            "unit": "ratio",
            "citations": _latest_citation(quarter_fin, annual_fin, "cfo")
            + _latest_citation(quarter_fin, annual_fin, "net_income"),
            "derived_from": [
                "discovery.metrics.cfo_to_net_income",
                "companyfacts_facts.cfo",
                "companyfacts_facts.net_income",
                "financials.cfo",
                "financials.net_income",
            ],
        },
        {
            "claim_id": "revenue_growth_yoy",
            "label": "revenue_growth_yoy",
            "value": metrics["revenue_growth_yoy"],
            "unit": "ratio",
            "citations": _latest_citation(quarter_fin, annual_fin, "revenue"),
            "derived_from": [
                "discovery.metrics.revenue_growth_yoy",
                "companyfacts_facts.revenue",
                "financials.revenue",
            ],
        },
        {
            "claim_id": "shares_change_4q",
            "label": "shares_change_4q",
            "value": metrics["shares_change_4q"],
            "unit": "ratio",
            "citations": [],
            "derived_from": [
                "discovery.metrics.shares_change_4q",
                "extracted_facts.shares_outstanding",
            ],
        },
        {
            "claim_id": "net_debt",
            "label": "net_debt",
            "value": metrics["net_debt"],
            "unit": "USD",
            "citations": _latest_citation(quarter_fin, annual_fin, "total_debt")
            + _latest_citation(quarter_fin, annual_fin, "cash"),
            "derived_from": [
                "discovery.metrics.net_debt",
                "companyfacts_facts.total_debt",
                "companyfacts_facts.cash",
                "financials.total_debt",
                "financials.cash",
            ],
        },
    ]

    flags: list[str] = []
    critical = ["ttm_revenue", "gross_margin", "operating_margin", "shares_outstanding"]
    if not is_financial:
        critical.append("fcf")
    if any(metrics.get(k) == UNKNOWN for k in critical):
        flags.append("FINANCIALS_INCOMPLETE")
    if is_financial:
        flags.append("FINANCIAL_ISSUER_CASHFLOW_FRAME")

    return DiscoveryMetricsResult(
        effective_as_of_date=latest_filing_date,
        metrics=metrics,
        claims=claims,
        flags=flags,
        filing_accessions_used=[str(row["accession"]) for row in rows],
    )
