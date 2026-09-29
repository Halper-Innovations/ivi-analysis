"""Detect quantitative anomalies in financial data.

Scans companyfacts_facts for patterns that warrant investigation:
- Q4 earnings bombs (implied Q4 wildly different from Q1-Q3 trend)
- Negative equity
- Operating margin collapse (>10pp decline over 2 years)
- Intangible asset jumps (>3x YoY — possible acquisition or capitalization change)
- Revenue decline from peak (>20% decline from highest annual revenue)
- Persistent cash burn (negative CFO for 2+ consecutive years)
- Working capital crisis (current ratio < 0.8)
- Debt spike (total debt >2x YoY increase)

All checks are deterministic — no LLM calls. Each anomaly includes a
plain-English question for the filing investigator to answer.
"""

from __future__ import annotations

import logging
from contextlib import closing
from pathlib import Path
from typing import Sequence

from app.alpha.schemas import Anomaly
from app.db import connect, get_db
from app.util.financial_data_access import companyfacts_rows, issuer_companyfacts_rows

logger = logging.getLogger(__name__)


def _db_context(db_path: str | Path | None):
    """Open the requested DB without consulting the configured DB path."""

    return get_db() if db_path is None else closing(connect(db_path))


def _fact_rows(
    conn,
    ticker: str,
    *,
    columns: Sequence[str],
    issuer_cik: str | None,
    aliases: Sequence[str],
    **kwargs,
):
    if issuer_cik is not None or aliases:
        _scope, rows = issuer_companyfacts_rows(
            conn,
            ticker,
            columns=columns,
            issuer_cik=issuer_cik,
            aliases=aliases,
            **kwargs,
        )
        return rows
    return companyfacts_rows(conn, ticker, columns=columns, **kwargs)


def _load_annual(
    ticker: str,
    *,
    as_of_date: str | None = None,
    require_filed_asof: bool = False,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
) -> dict[int, dict[str, float]]:
    """Load all FY data as {fiscal_year: {line_item: value}}."""
    try:
        with _db_context(db_path) as conn:
            rows = _fact_rows(
                conn,
                ticker,
                columns=("fiscal_year", "line_item", "value"),
                issuer_cik=issuer_cik,
                aliases=aliases,
                period_types=("FY",),
                as_of_date=as_of_date,
                value_not_null=True,
                require_filed_asof=require_filed_asof,
                order_by="fiscal_year",
            )
    except Exception:
        return {}
    out: dict[int, dict[str, float]] = {}
    for r in rows:
        out.setdefault(int(r["fiscal_year"]), {})[str(r["line_item"])] = float(r["value"])
    return out


def _load_quarterly(
    ticker: str,
    *,
    as_of_date: str | None = None,
    require_filed_asof: bool = False,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Load quarterly data as {'FY2025Q1': {line_item: value}}."""
    try:
        with _db_context(db_path) as conn:
            rows = _fact_rows(
                conn,
                ticker,
                columns=("fiscal_year", "period_type", "line_item", "value"),
                issuer_cik=issuer_cik,
                aliases=aliases,
                exclude_period_types=("FY",),
                as_of_date=as_of_date,
                value_not_null=True,
                require_filed_asof=require_filed_asof,
                order_by="fiscal_year, period_type",
            )
    except Exception:
        return {}
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        key = f"FY{r['fiscal_year']}{r['period_type']}"
        out.setdefault(key, {})[str(r["line_item"])] = float(r["value"])
    return out


def _latest_years(annual: dict[int, dict[str, float]], n: int = 3) -> list[int]:
    return sorted(annual.keys(), reverse=True)[:n]


def detect_anomalies(
    ticker: str,
    *,
    as_of_date: str | None = None,
    require_filed_asof: bool = False,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
) -> list[Anomaly]:
    """Scan financial data for quantitative anomalies. Returns list of Anomaly objects."""
    upper = ticker.upper()
    annual = _load_annual(
        upper,
        as_of_date=as_of_date,
        require_filed_asof=require_filed_asof,
        issuer_cik=issuer_cik,
        aliases=aliases,
        db_path=db_path,
    )
    quarterly = _load_quarterly(
        upper,
        as_of_date=as_of_date,
        require_filed_asof=require_filed_asof,
        issuer_cik=issuer_cik,
        aliases=aliases,
        db_path=db_path,
    )
    if not annual:
        return []

    anomalies: list[Anomaly] = []
    years = sorted(annual.keys(), reverse=True)
    latest_yr = years[0] if years else None
    latest = annual.get(latest_yr, {}) if latest_yr else {}

    # --- Q4 Earnings Bomb ---
    if latest_yr:
        fy_ni = latest.get("net_income")
        q_keys = [f"FY{latest_yr}{q}" for q in ("Q1", "Q2", "Q3")]
        q_ni_values = [quarterly.get(k, {}).get("net_income") for k in q_keys]
        if fy_ni is not None and all(v is not None for v in q_ni_values):
            q123_total = sum(q_ni_values)
            q4_implied = fy_ni - q123_total
            swing = abs(q4_implied - (q123_total / 3.0)) if q123_total != 0 else abs(q4_implied)
            if q4_implied < -5.0 and abs(q4_implied) > abs(q123_total) * 2:
                anomalies.append(
                    Anomaly(
                        anomaly_type="Q4_EARNINGS_BOMB",
                        severity="HIGH",
                        description=f"Q1-Q3 net income totaled ${q123_total:.1f}M, but FY was ${fy_ni:.1f}M — implying Q4 was ${q4_implied:.1f}M.",
                        question=f"What caused the ${abs(q4_implied):.0f}M Q4 loss? Search for impairment charges, writedowns, valuation allowances, restructuring, or one-time items.",
                        data={"fy_ni": fy_ni, "q123_total": q123_total, "q4_implied": q4_implied},
                    )
                )

    # --- Negative Equity ---
    equity = latest.get("equity")
    if equity is not None and equity < 0:
        prev_equity = annual.get(years[1], {}).get("equity") if len(years) > 1 else None
        anomalies.append(
            Anomaly(
                anomaly_type="NEGATIVE_EQUITY",
                severity="HIGH",
                description=f"Stockholders' equity is ${equity:.1f}M (negative). "
                + (f"Prior year was ${prev_equity:.1f}M." if prev_equity is not None else ""),
                question="What drove equity negative? Search for accumulated deficit, large losses, dividends in excess of earnings, or share repurchases funded by debt.",
                data={"equity": equity, "prev_equity": prev_equity},
            )
        )

    # --- Margin Collapse ---
    if len(years) >= 3:
        margins = []
        for yr in years[:3]:
            rev = annual[yr].get("revenue")
            oi = annual[yr].get("operating_income")
            if rev and rev > 0 and oi is not None:
                margins.append((yr, oi / rev * 100))
        if len(margins) >= 3:
            newest_margin = margins[0][1]
            oldest_margin = margins[2][1]
            decline = oldest_margin - newest_margin
            if decline > 10.0:
                anomalies.append(
                    Anomaly(
                        anomaly_type="MARGIN_COLLAPSE",
                        severity="HIGH" if decline > 20 else "MODERATE",
                        description=f"Operating margin fell from {oldest_margin:.1f}% (FY{margins[2][0]}) to {newest_margin:.1f}% (FY{margins[0][0]}) — a {decline:.1f}pp decline.",
                        question="What is causing margin compression? Search for cost increases, pricing pressure, mix shift, R&D ramp, or restructuring costs.",
                        data={
                            "newest": newest_margin,
                            "oldest": oldest_margin,
                            "decline_pp": decline,
                        },
                    )
                )

    # --- Intangible Asset Jump ---
    if len(years) >= 2:
        ia_now = latest.get("intangible_assets")
        ia_prev = annual.get(years[1], {}).get("intangible_assets")
        if ia_now and ia_prev and ia_prev > 0 and ia_now / ia_prev > 3.0:
            anomalies.append(
                Anomaly(
                    anomaly_type="INTANGIBLE_ASSET_JUMP",
                    severity="MODERATE",
                    description=f"Intangible assets jumped from ${ia_prev:.1f}M to ${ia_now:.1f}M ({ia_now / ia_prev:.1f}x increase).",
                    question="What caused the intangible asset increase? Search for acquisitions, capitalized development costs, or purchased IP/licenses.",
                    data={"current": ia_now, "prior": ia_prev, "ratio": ia_now / ia_prev},
                )
            )

    # --- Revenue Decline from Peak ---
    rev_series = [(yr, annual[yr].get("revenue")) for yr in years if annual[yr].get("revenue")]
    if len(rev_series) >= 3:
        peak_yr, peak_rev = max(rev_series, key=lambda x: x[1])
        latest_rev = rev_series[0][1]
        if peak_rev > 0 and latest_rev < peak_rev * 0.80:
            anomalies.append(
                Anomaly(
                    anomaly_type="REVENUE_DECLINE_FROM_PEAK",
                    severity="HIGH" if latest_rev < peak_rev * 0.60 else "MODERATE",
                    description=f"Revenue peaked at ${peak_rev:.0f}M in FY{peak_yr}, now ${latest_rev:.0f}M ({(1 - latest_rev / peak_rev) * 100:.0f}% decline).",
                    question="Is the revenue decline structural (lost customers, obsolete products) or cyclical (timing, one-time)? Search for customer losses, product lifecycle, or market conditions.",
                    data={"peak_yr": peak_yr, "peak_rev": peak_rev, "current_rev": latest_rev},
                )
            )

    # --- Persistent Cash Burn ---
    cfo_series = [
        (yr, annual[yr].get("cfo")) for yr in years[:3] if annual[yr].get("cfo") is not None
    ]
    neg_cfo_years = [yr for yr, cfo in cfo_series if cfo < 0]
    if len(neg_cfo_years) >= 2:
        cash = latest.get("cash")
        avg_burn = sum(annual[yr]["cfo"] for yr in neg_cfo_years) / len(neg_cfo_years)
        quarters_left = (
            cash / abs(avg_burn) * 4 if avg_burn < 0 and cash is not None and cash > 0 else None
        )
        cash_description = f"${cash:.1f}M" if cash is not None else "unavailable"
        anomalies.append(
            Anomaly(
                anomaly_type="PERSISTENT_CASH_BURN",
                severity="HIGH"
                if (quarters_left is not None and quarters_left < 8)
                else "MODERATE",
                description=f"Negative CFO for {len(neg_cfo_years)} of last {len(cfo_series)} years. Cash: {cash_description}, avg annual burn: ${avg_burn:.1f}M."
                + (f" Estimated runway: {quarters_left:.0f} quarters." if quarters_left else ""),
                question="How is the company funding operations? Search for debt raises, equity issuance, asset sales, or planned financing. Is there a path to positive cash flow?",
                data={
                    "neg_years": len(neg_cfo_years),
                    "cash": cash,
                    "avg_burn": avg_burn,
                    "quarters_left": quarters_left,
                },
            )
        )

    # --- Working Capital Crisis ---
    ca = latest.get("current_assets")
    cl = latest.get("current_liabilities")
    if ca and cl and cl > 0:
        ratio = ca / cl
        if ratio < 0.8:
            anomalies.append(
                Anomaly(
                    anomaly_type="WORKING_CAPITAL_CRISIS",
                    severity="HIGH" if ratio < 0.6 else "MODERATE",
                    description=f"Current ratio is {ratio:.2f} (CA ${ca:.1f}M / CL ${cl:.1f}M) — below 1.0 indicates inability to cover short-term obligations.",
                    question="How will the company meet near-term obligations? Search for credit facilities, revolving credit lines, or planned refinancing.",
                    data={"current_ratio": ratio, "current_assets": ca, "current_liabilities": cl},
                )
            )

    # --- Debt Spike ---
    if len(years) >= 2:
        debt_now = latest.get("total_debt")
        debt_prev = annual.get(years[1], {}).get("total_debt")
        if debt_now and debt_prev and debt_prev > 0 and debt_now / debt_prev > 2.0:
            anomalies.append(
                Anomaly(
                    anomaly_type="DEBT_SPIKE",
                    severity="MODERATE",
                    description=f"Total debt increased from ${debt_prev:.1f}M to ${debt_now:.1f}M ({debt_now / debt_prev:.1f}x).",
                    question="What drove the debt increase? Search for new credit facilities, convertible notes, acquisition financing, or refinancing.",
                    data={"current": debt_now, "prior": debt_prev},
                )
            )

    return anomalies
