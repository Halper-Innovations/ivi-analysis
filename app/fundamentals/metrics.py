from __future__ import annotations

import json
from typing import Any

from app.db import get_db, utc_now_iso
from app.fundamentals.normalize import UNKNOWN, maybe_unknown, safe_div
from app.ingest.facts_writer import ensure_facts
from app.logging import get_logger
from app.util.financial_data_access import (
    ANNUAL_COMPANYFACTS_PERIOD_TYPES,
    FINANCIAL_CACHED_FILING_FORM_TYPES,
    companyfacts_map_for_latest_period,
    latest_filing_ids_by_ticker as load_latest_filing_ids_by_ticker,
    latest_filing_row,
)
from app.util.issuer_classification import (
    ISSUER_CLASS_FINANCIAL,
    infer_issuer_classification,
    resolve_issuer_classification,
)


logger = get_logger(__name__)


def _latest_filing_ids_by_ticker(
    conn,
    *,
    as_of_date: str | None = None,
) -> dict[str, int]:
    return load_latest_filing_ids_by_ticker(
        conn,
        form_types=FINANCIAL_CACHED_FILING_FORM_TYPES,
        as_of_date=as_of_date,
    )


def _financial_map(conn, filing_id: int) -> dict[str, float | None]:
    rows = conn.execute(
        "SELECT line_item, value FROM financials WHERE filing_id = ?",
        (filing_id,),
    ).fetchall()
    out: dict[str, float | None] = {}
    for row in rows:
        out[row["line_item"]] = row["value"]
    return out


def _liquidity_stress_score(conn, filing_id: int) -> int:
    rows = conn.execute(
        "SELECT fact_type FROM extracted_facts WHERE filing_id = ?",
        (filing_id,),
    ).fetchall()
    penalty = 0
    for row in rows:
        fact_type = row["fact_type"]
        if fact_type in {
            "going_concern",
            "covenant_pressure",
            "refinancing_need",
            "material_weakness",
        }:
            penalty += 2
        if fact_type in {"dilution_risk", "debt_liquidity_signal"}:
            penalty += 1
    return min(10, penalty)


def compute_metrics_from_financials(
    fin: dict[str, float | None],
    liquidity_score: int | str,
    computed_as_of_date: str | None = None,
    *,
    issuer_classification: str | None = None,
    issuer_classification_source: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``issuer_classification`` is the caller's SIC-first answer (see
    ``resolve_issuer_classification``); without one, the tag-name substring rule
    over ``fin``'s keys decides, as before."""
    revenue = fin.get("revenue")
    gross_profit = fin.get("gross_profit")
    operating_income = fin.get("operating_income")
    net_income = fin.get("net_income")
    cfo = fin.get("cfo")
    capex = fin.get("capex")
    cash = fin.get("cash")
    total_debt = fin.get("total_debt")
    r_and_d_total = fin.get("r_and_d_total")
    share_repurchases_amount = fin.get("share_repurchases_amount")
    dividends_paid_amount = fin.get("dividends_paid_amount")
    deposits = fin.get("deposits")
    loans = fin.get("loans")
    investment_securities = fin.get("investment_securities")
    total_assets = fin.get("total_assets")
    assets_under_management = fin.get("assets_under_management")
    allowance_for_credit_losses = fin.get("allowance_for_credit_losses")
    provision_for_credit_losses = fin.get("provision_for_credit_losses")
    net_charge_offs = fin.get("net_charge_offs")
    nonaccrual_loans = fin.get("nonaccrual_loans")

    if not issuer_classification:
        issuer_classification = infer_issuer_classification(line_items=fin.keys())
    fcf = (
        None
        if cfo is None or capex is None or issuer_classification == ISSUER_CLASS_FINANCIAL
        else (cfo - capex)
    )
    net_debt = None if total_debt is None or cash is None else (total_debt - cash)
    r_and_d_intensity = safe_div(r_and_d_total, revenue)
    deposits_to_assets = safe_div(deposits, total_assets)
    loans_to_deposits = safe_div(loans, deposits)
    allowance_to_loans = safe_div(allowance_for_credit_losses, loans)
    provision_to_loans = safe_div(provision_for_credit_losses, loans)
    net_charge_offs_to_loans = safe_div(net_charge_offs, loans)
    debt_to_assets = safe_div(total_debt, total_assets)

    metrics = {
        "revenue": maybe_unknown(revenue),
        "revenue_growth": UNKNOWN,
        "gross_margin": maybe_unknown(safe_div(gross_profit, revenue)),
        "operating_margin": maybe_unknown(safe_div(operating_income, revenue)),
        "net_income": maybe_unknown(net_income),
        "cfo": maybe_unknown(cfo),
        "capex": maybe_unknown(capex),
        "fcf": maybe_unknown(fcf),
        "fcf_margin": maybe_unknown(safe_div(fcf, revenue)),
        "net_debt": maybe_unknown(net_debt),
        "r_and_d_total": maybe_unknown(r_and_d_total),
        "share_repurchases_amount": maybe_unknown(share_repurchases_amount),
        "dividends_paid_amount": maybe_unknown(dividends_paid_amount),
        "deposits": maybe_unknown(deposits),
        "loans": maybe_unknown(loans),
        "investment_securities": maybe_unknown(investment_securities),
        "total_assets": maybe_unknown(total_assets),
        "assets_under_management": maybe_unknown(assets_under_management),
        "allowance_for_credit_losses": maybe_unknown(allowance_for_credit_losses),
        "provision_for_credit_losses": maybe_unknown(provision_for_credit_losses),
        "net_charge_offs": maybe_unknown(net_charge_offs),
        "nonaccrual_loans": maybe_unknown(nonaccrual_loans),
        "deposits_to_assets": maybe_unknown(deposits_to_assets),
        "loans_to_deposits": maybe_unknown(loans_to_deposits),
        "allowance_to_loans": maybe_unknown(allowance_to_loans),
        "provision_to_loans": maybe_unknown(provision_to_loans),
        "net_charge_offs_to_loans": maybe_unknown(net_charge_offs_to_loans),
        "debt_to_assets": maybe_unknown(debt_to_assets),
        "r_and_d_intensity_latest": maybe_unknown(r_and_d_intensity),
        "liquidity_stress_score": liquidity_score,
        "sbc_proxy_flag": "UNKNOWN",
        "issuer_classification": issuer_classification,
        "fcf_applicability": "sector_limited"
        if issuer_classification == ISSUER_CLASS_FINANCIAL
        else "standard",
    }

    quality_flags = {
        "has_revenue": revenue is not None,
        "has_cashflow_pair": cfo is not None and capex is not None,
        "has_balance_sheet_pair": total_debt is not None and cash is not None,
        "issuer_classification": issuer_classification,
        "computed_as_of_date": computed_as_of_date or utc_now_iso()[:10],
    }
    if issuer_classification_source:
        quality_flags["issuer_classification_source"] = issuer_classification_source
    return metrics, quality_flags


def compute_all_fundamentals(as_of_date: str | None = None) -> int:
    count = 0
    effective_as_of = as_of_date or utc_now_iso()[:10]
    with get_db() as conn:
        ticker_to_filing = _latest_filing_ids_by_ticker(
            conn,
            as_of_date=effective_as_of,
        )
        for ticker, filing_id in ticker_to_filing.items():
            if _compute_fundamentals_for_filing(
                conn, ticker, filing_id, computed_as_of_date=as_of_date
            ):
                count += 1

    logger.info(
        "fundamentals_completed", extra={"stage_name": "fundamentals", "stage_count": count}
    )
    return count


def _latest_filing_id_for_ticker(
    conn,
    ticker: str,
    *,
    as_of_date: str,
) -> int | None:
    row = latest_filing_row(
        conn,
        ticker,
        columns=("id",),
        form_types=FINANCIAL_CACHED_FILING_FORM_TYPES,
        as_of_date=as_of_date,
    )
    if not row:
        return None
    return int(row["id"])


def _companyfacts_map_for_ticker(
    conn, ticker: str, as_of_date: str | None = None
) -> tuple[dict[str, float | None], str | None]:
    return companyfacts_map_for_latest_period(
        conn,
        ticker,
        as_of_date=as_of_date,
        period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
        require_filed_asof=True,
    )


def _compute_fundamentals_for_filing(
    conn, ticker: str, filing_id: int, computed_as_of_date: str | None = None
) -> bool:
    fin = _financial_map(conn, filing_id)
    liq_score = _liquidity_stress_score(conn, filing_id)
    issuer_classification, classification_source = resolve_issuer_classification(
        ticker=ticker, line_items=fin.keys(), conn=conn
    )
    metrics, quality_flags = compute_metrics_from_financials(
        fin,
        liq_score,
        computed_as_of_date=computed_as_of_date,
        issuer_classification=issuer_classification,
        issuer_classification_source=classification_source,
    )

    as_of_date = conn.execute(
        "SELECT filing_date FROM filings WHERE id = ?", (filing_id,)
    ).fetchone()["filing_date"]
    as_of_date = as_of_date or utc_now_iso()[:10]

    conn.execute(
        """
        INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(ticker, as_of_date) DO UPDATE SET
            metrics_json=excluded.metrics_json,
            quality_flags_json=excluded.quality_flags_json
        """,
        (ticker, as_of_date, json.dumps(metrics), json.dumps(quality_flags), utc_now_iso()),
    )
    return True


def compute_fundamentals_for_ticker(ticker: str, as_of_date: str | None = None) -> bool:
    effective_as_of = as_of_date or utc_now_iso()[:10]
    with get_db() as conn:
        filing_id = _latest_filing_id_for_ticker(
            conn,
            ticker,
            as_of_date=effective_as_of,
        )
        if filing_id is None:
            fin, period_end = _companyfacts_map_for_ticker(
                conn,
                ticker,
                as_of_date=effective_as_of,
            )
            if not fin:
                ensure_facts(ticker)
                fin, period_end = _companyfacts_map_for_ticker(
                    conn,
                    ticker,
                    as_of_date=effective_as_of,
                )
            if not fin:
                return False
            # No parsed filing means no extracted facts were ever examined for going
            # concern, covenant or refinancing stress. That is UNKNOWN, not "no stress":
            # a stored 0 would let the rubric award the top balance-sheet score.
            liq_score = UNKNOWN
            issuer_classification, classification_source = resolve_issuer_classification(
                ticker=ticker, line_items=fin.keys(), conn=conn
            )
            metrics, quality_flags = compute_metrics_from_financials(
                fin,
                liq_score,
                computed_as_of_date=effective_as_of,
                issuer_classification=issuer_classification,
                issuer_classification_source=classification_source,
            )
            quality_flags["source_resolution"] = "companyfacts_fallback"
            fundamentals_as_of = effective_as_of
            conn.execute(
                """
                INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(ticker, as_of_date) DO UPDATE SET
                    metrics_json=excluded.metrics_json,
                    quality_flags_json=excluded.quality_flags_json
                """,
                (
                    ticker,
                    fundamentals_as_of,
                    json.dumps(metrics),
                    json.dumps(quality_flags),
                    utc_now_iso(),
                ),
            )
            return True
        return _compute_fundamentals_for_filing(
            conn,
            ticker,
            filing_id,
            computed_as_of_date=effective_as_of,
        )
