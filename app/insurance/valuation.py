"""Insurance-specific valuation primitives.

These functions intentionally prefer explicit ``MODEL_BLOCKED`` / ``UNKNOWN``
outputs over generic DCF/EPV fallbacks. V1 is designed to be auditable and
defensive rather than falsely precise.
"""

from __future__ import annotations

import os
import re
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from app.config import AppConfig
from app.db import connect, get_db
from app.insurance.routing import (
    ISSUER_INSURANCE_UNDERWRITER,
    SECURITY_COMMON,
    SECURITY_DEPOSITARY,
    SECURITY_PREFERRED,
    SecurityRoutingResult,
)
from app.insurance.sources import latest_cached_filing_text, latest_company_profile
from app.util.financial_data_access import (
    ANNUAL_COMPANYFACTS_PERIOD_TYPES,
    companyfacts_rows,
    issuer_companyfacts_rows,
)

UNKNOWN = "UNKNOWN"


def _float_env(name: str, default: float | None = None) -> float | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value / 100.0 if value > 1.0 else value


def insurance_cost_of_equity() -> dict[str, Any]:
    """Return deterministic v1 cost-of-equity assumptions."""
    direct = _float_env("VOE_INSURANCE_COST_OF_EQUITY")
    if direct is not None:
        base = direct
        source = "VOE_INSURANCE_COST_OF_EQUITY"
    elif any(
        os.getenv(name) not in (None, "")
        for name in (
            "VOE_INSURANCE_RISK_FREE_RATE",
            "VOE_INSURANCE_EQUITY_RISK_PREMIUM",
            "VOE_INSURANCE_COMPANY_SPECIFIC_PREMIUM",
        )
    ):
        risk_free = _float_env("VOE_INSURANCE_RISK_FREE_RATE", 0.04) or 0.04
        erp = _float_env("VOE_INSURANCE_EQUITY_RISK_PREMIUM", 0.05) or 0.05
        company_premium = _float_env("VOE_INSURANCE_COMPANY_SPECIFIC_PREMIUM", 0.01) or 0.01
        base = risk_free + erp + company_premium
        source = "rf_plus_erp_plus_company_premium"
    else:
        base = 0.10
        source = "default_base_10pct"
    return {
        "base": round(base, 6),
        "source": source,
        "sensitivity_rates": [0.09, 0.10, 0.11, 0.12],
    }


def _as_float(value: Any) -> float | None:
    if value is None or value == UNKNOWN:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _annual_fact_rows(
    ticker: str,
    as_of_date: str | None,
    *,
    pipeline_version: str = "v1",
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> list[dict[str, Any]]:
    is_v2 = str(pipeline_version or "v1").strip().lower() == "v2"
    if not as_of_date:
        raise ValueError("insurance annual facts require as_of_date")
    db_context = get_db(cfg) if db_path is None else closing(connect(db_path, cfg=cfg))
    with db_context as conn:
        kwargs = {
            "columns": (
                "fiscal_year",
                "period_end",
                "line_item",
                "value",
                "units",
                "source_url",
                "filed_date",
                "accession",
            ),
            "period_types": ANNUAL_COMPANYFACTS_PERIOD_TYPES,
            "as_of_date": as_of_date,
            "value_not_null": True,
            "require_filed_asof": True,
            "order_by": "period_end ASC, fiscal_year ASC, line_item ASC",
        }
        if is_v2:
            _scope, rows = issuer_companyfacts_rows(
                conn,
                ticker,
                issuer_cik=issuer_cik,
                aliases=aliases,
                **kwargs,
            )
        else:
            rows = companyfacts_rows(conn, ticker, **kwargs)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        period_end = str(row["period_end"] or "").strip()
        filed_date = str(row["filed_date"] or "").strip()
        source_url = str(row["source_url"] or "").strip()
        if not period_end or not filed_date or not source_url or period_end > filed_date:
            continue
        normalized.append(
            {
                "fiscal_year": int(row["fiscal_year"]),
                "period_end": period_end,
                "line_item": str(row["line_item"]),
                "value": float(row["value"]),
                "units": row["units"],
                "source_url": source_url,
                "filed_date": filed_date,
                "accession": str(row["accession"] or "").strip() or None,
            }
        )
    return normalized


def _rows_by_year(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_year: dict[int, dict[str, Any]] = {}
    for row in rows:
        year = int(row["fiscal_year"])
        by_year.setdefault(year, {"fiscal_year": year, "period_end": row.get("period_end")})
        line_item = str(row["line_item"])
        by_year[year][line_item] = row["value"]
        by_year[year][f"{line_item}_filed_date"] = row.get("filed_date")
        by_year[year][f"{line_item}_accession"] = row.get("accession")
        by_year[year][f"{line_item}_source_url"] = row.get("source_url")
    return by_year


def _latest_year_with(by_year: dict[int, dict[str, Any]], item: str) -> dict[str, Any] | None:
    for year in sorted(by_year.keys(), reverse=True):
        if _as_float(by_year[year].get(item)) is not None:
            return by_year[year]
    return None


def _component_score(
    label: str, status: str, value: Any, weight: float, reason: str
) -> dict[str, Any]:
    return {
        "component": label,
        "status": status,
        "value": value,
        "weight": weight,
        "reason": reason,
    }


# Sanity bounds for the no-fade perpetual-spread model (audit coverage gap
# 1): the ROE spread is clamped to a plausible underwriting band and the
# anchor is bounded to a defensible price-to-book range so env-overridable
# cost/growth inputs can never drive a 1000x multiplier.
ROE_SPREAD_CLAMP = 0.20
RESIDUAL_BOOK_MULTIPLE_CAP = 4.0
MIN_COE_GROWTH_SPREAD = 0.005


def _residual_value(
    book_value_per_share: float, normalized_roe: float, cost_of_equity: float, growth: float
) -> float:
    value, _guards = _residual_value_guarded(
        book_value_per_share, normalized_roe, cost_of_equity, growth
    )
    return value


def _residual_value_guarded(
    book_value_per_share: float,
    normalized_roe: float,
    cost_of_equity: float,
    growth: float,
) -> tuple[float, list[str]]:
    guards: list[str] = []
    spread = normalized_roe - cost_of_equity
    if abs(spread) > ROE_SPREAD_CLAMP:
        spread = ROE_SPREAD_CLAMP if spread > 0 else -ROE_SPREAD_CLAMP
        guards.append("ROE_SPREAD_CLAMPED")
    denominator = max(MIN_COE_GROWTH_SPREAD, cost_of_equity - growth)
    value = book_value_per_share + (book_value_per_share * spread / denominator)
    cap = RESIDUAL_BOOK_MULTIPLE_CAP * book_value_per_share
    if value > cap:
        value = cap
        guards.append("RESIDUAL_CAPPED_AT_BOOK_MULTIPLE")
    if value < 0.0:
        value = 0.0
        guards.append("RESIDUAL_FLOORED_AT_ZERO")
    return value, guards


def calculate_insurance_common_valuation(
    ticker: str,
    *,
    as_of_date: str | None = None,
    routing: SecurityRoutingResult | dict[str, Any] | None = None,
    current_price: float | None = None,
    pipeline_version: str = "v1",
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Calculate a v1 common-equity insurer valuation using book and ROE."""
    upper = ticker.upper()
    routing_dict = (
        routing.to_dict() if isinstance(routing, SecurityRoutingResult) else (routing or {})
    )
    if routing_dict and (
        routing_dict.get("security_type") != SECURITY_COMMON
        or routing_dict.get("issuer_type") != ISSUER_INSURANCE_UNDERWRITER
    ):
        return {
            "status": "NOT_APPLICABLE",
            "model_status": "NOT_APPLICABLE",
            "method": "insurance_common",
            "ticker": upper,
            "reason_codes": ["NOT_INSURANCE_COMMON_EQUITY"],
        }

    rows = _annual_fact_rows(
        upper,
        as_of_date,
        pipeline_version=pipeline_version,
        issuer_cik=issuer_cik,
        aliases=aliases,
        db_path=db_path,
        cfg=cfg,
    )
    by_year = _rows_by_year(rows)
    latest_equity_row = _latest_year_with(by_year, "equity")
    latest_shares_row = _latest_year_with(by_year, "shares_outstanding")
    latest_equity = _as_float((latest_equity_row or {}).get("equity"))
    latest_shares = _as_float((latest_shares_row or {}).get("shares_outstanding"))

    net_income_values = [
        _as_float(by_year[year].get("net_income"))
        for year in sorted(by_year.keys(), reverse=True)[:5]
        if _as_float(by_year[year].get("net_income")) is not None
    ]

    blockers: list[str] = []
    if str(pipeline_version or "v1").strip().lower() == "v2":
        monetary_units = {
            str(row.get("units") or "").strip().upper()
            for row in rows
            if row.get("line_item") in {"equity", "net_income"}
        }
        if any(unit and not unit.startswith("USD") for unit in monetary_units):
            blockers.append("NON_USD_FACTS_UNNORMALIZED")
        elif "" in monetary_units:
            blockers.append("FACT_CURRENCY_UNRESOLVED")
    if latest_equity is None or latest_equity <= 0:
        blockers.append("MISSING_OR_NONPOSITIVE_BOOK_VALUE")
    if latest_shares is None or latest_shares <= 0:
        blockers.append("MISSING_SHARES_OUTSTANDING")
    if not net_income_values:
        blockers.append("MISSING_NORMALIZED_ROE_INPUT")

    coe = insurance_cost_of_equity()
    cost = float(coe["base"])
    growth = _float_env("VOE_INSURANCE_LONG_RUN_GROWTH", 0.02) or 0.02
    if cost - growth < MIN_COE_GROWTH_SPREAD:
        # Perpetuity undefined / explosive — refuse rather than extrapolate
        # (audit coverage gap 1).
        blockers.append("INVALID_COE_GROWTH_ASSUMPTIONS")

    if blockers:
        return {
            "status": "MODEL_BLOCKED",
            "model_status": "MODEL_BLOCKED",
            "method": "insurance_common",
            "ticker": upper,
            "as_of_date": as_of_date,
            "valuation_anchor": None,
            "reason_codes": blockers,
            "missing_components": blockers,
            "cost_of_equity": coe,
            "source_references": {
                "annual_companyfacts_rows": len(rows),
                "latest_equity_period": (latest_equity_row or {}).get("period_end"),
                "latest_shares_period": (latest_shares_row or {}).get("period_end"),
                "latest_equity_filed_date": (latest_equity_row or {}).get("equity_filed_date"),
                "latest_shares_filed_date": (latest_shares_row or {}).get(
                    "shares_outstanding_filed_date"
                ),
            },
        }

    assert latest_equity is not None
    assert latest_shares is not None
    avg_net_income = sum(value for value in net_income_values if value is not None) / len(
        net_income_values
    )
    normalized_roe = avg_net_income / latest_equity
    book_value_per_share = latest_equity / latest_shares
    residual_value, residual_guards = _residual_value_guarded(
        book_value_per_share, normalized_roe, cost, growth
    )
    justified_pb = (normalized_roe - growth) / max(0.001, cost - growth)
    price = _as_float(current_price)

    component_scores = [
        _component_score(
            "normalized_roe_vs_cost_of_equity",
            "OK" if normalized_roe >= cost else "WEAK",
            round(normalized_roe, 6),
            0.35,
            "ROE above cost of equity creates residual income; below cost of equity destroys value.",
        ),
        _component_score(
            "book_value_base",
            "OK",
            round(book_value_per_share, 6),
            0.25,
            "Latest annual common equity divided by actual shares outstanding.",
        ),
        _component_score(
            "capital_adequacy",
            UNKNOWN,
            UNKNOWN,
            0.20,
            "RBC/BCAR/statutory capital is not standardized in v1 primary-source inputs.",
        ),
        _component_score(
            "ratings_evidence",
            UNKNOWN,
            UNKNOWN,
            0.10,
            "Rating agency capital model evidence is captured only when explicitly disclosed.",
        ),
        _component_score(
            "alm_investment_risk",
            UNKNOWN,
            UNKNOWN,
            0.10,
            "Investment portfolio and ALM detail require filing/supplement extraction beyond normalized facts.",
        ),
    ]

    sensitivities = {
        f"{int(rate * 100)}pct": round(
            _residual_value(book_value_per_share, normalized_roe, rate, growth), 6
        )
        for rate in coe["sensitivity_rates"]
        if float(rate) - growth >= MIN_COE_GROWTH_SPREAD
    }
    return {
        "status": "OK",
        "model_status": "OK",
        "method": "insurance_common",
        "ticker": upper,
        "as_of_date": as_of_date,
        "valuation_anchor": round(residual_value, 6),
        "valuation_anchor_label": "residual_income_value_per_share",
        "adjusted_book_value_per_share": round(book_value_per_share, 6),
        "residual_income_value_per_share": round(residual_value, 6),
        "normalized_roe": round(normalized_roe, 6),
        "average_net_income": round(avg_net_income, 6),
        "justified_price_to_book": round(justified_pb, 6),
        "cost_of_equity": coe,
        "long_run_growth": round(growth, 6),
        "sensitivity": sensitivities,
        "current_price": price,
        "upside_to_anchor": round((residual_value - price) / residual_value, 6)
        if price and residual_value > 0
        else None,
        "reason_codes": list(residual_guards),
        "missing_components": ["RBC_OR_BCAR", "RATINGS_MODEL", "STATUTORY_SURPLUS", "ALM_DETAIL"],
        "component_scores": component_scores,
        "source_references": {
            "annual_companyfacts_rows": len(rows),
            "latest_equity_period": (latest_equity_row or {}).get("period_end"),
            "latest_shares_period": (latest_shares_row or {}).get("period_end"),
            "latest_equity_filed_date": (latest_equity_row or {}).get("equity_filed_date"),
            "latest_shares_filed_date": (latest_shares_row or {}).get(
                "shares_outstanding_filed_date"
            ),
            "net_income_year_count": len(net_income_values),
        },
    }


def _extract_preferred_terms(text: str, name: str) -> dict[str, Any]:
    haystack = f"{name}\n{text[:220_000]}"
    lower = haystack.lower()

    coupon_rate = None
    for match in re.finditer(r"(?<!\d)(\d{1,2}(?:\.\d+)?)\s*%", haystack):
        value = float(match.group(1)) / 100.0
        window = haystack[max(0, match.start() - 120) : match.end() + 120].lower()
        if 0 < value < 0.20 and any(
            token in window for token in ("preferred", "depositary", "series", "dividend")
        ):
            coupon_rate = value
            break

    liquidation_preference = None
    patterns = [
        r"liquidation preference(?: of)?\s*\$?([0-9][0-9,]*(?:\.\d+)?)",
        r"\$?([0-9][0-9,]*(?:\.\d+)?)\s+liquidation preference",
        r"depositary share[^.\n]{0,120}\$([0-9][0-9,]*(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, haystack, flags=re.IGNORECASE)
        if match:
            liquidation_preference = float(match.group(1).replace(",", ""))
            break

    cumulative = None
    if "non-cumulative" in lower or "noncumulative" in lower:
        cumulative = False
    elif "cumulative" in lower:
        cumulative = True

    call_date = None
    call_match = re.search(
        r"(?:on or after|redeemable on or after)\s+([A-Z][a-z]+ \d{1,2}, \d{4})",
        haystack,
    )
    if call_match:
        call_date = call_match.group(1)

    return {
        "coupon_rate": coupon_rate,
        "liquidation_preference": liquidation_preference,
        "cumulative": cumulative,
        "call_date": call_date,
        "source_policy": "cached_primary_filing_and_company_name",
    }


def _yield_to_call(
    current_price: float,
    liquidation_preference: float,
    annual_coupon: float,
    call_date: str | None,
    *,
    valuation_date: str | None = None,
) -> float | None:
    if not call_date:
        return None
    try:
        call_dt = datetime.strptime(call_date, "%B %d, %Y").date()
    except ValueError:
        return None
    if valuation_date:
        try:
            base_date = datetime.strptime(valuation_date[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    else:
        base_date = datetime.now(timezone.utc).date()
    years = (call_dt - base_date).days / 365.25
    if years <= 0:
        return None
    return ((liquidation_preference - current_price) / years + annual_coupon) / (
        (liquidation_preference + current_price) / 2.0
    )


def calculate_insurance_preferred_valuation(
    ticker: str,
    *,
    as_of_date: str | None = None,
    routing: SecurityRoutingResult | dict[str, Any] | None = None,
    current_price: float | None = None,
    pipeline_version: str = "v1",
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Calculate v1 preferred/depositary security valuation terms."""
    upper = ticker.upper()
    routing_dict = (
        routing.to_dict() if isinstance(routing, SecurityRoutingResult) else (routing or {})
    )
    if routing_dict and routing_dict.get("security_type") not in (
        SECURITY_PREFERRED,
        SECURITY_DEPOSITARY,
    ):
        return {
            "status": "NOT_APPLICABLE",
            "model_status": "NOT_APPLICABLE",
            "method": "insurance_preferred",
            "ticker": upper,
            "reason_codes": ["NOT_PREFERRED_OR_DEPOSITARY_SECURITY"],
        }

    is_v2 = str(pipeline_version or "v1").strip().lower() == "v2"
    if is_v2:
        if not as_of_date:
            raise ValueError("v2 insurance preferred valuation requires as_of_date")
        profile = latest_company_profile(
            upper,
            issuer_cik=issuer_cik,
            aliases=aliases,
            db_path=db_path,
            cfg=cfg,
        )
        text, filing_meta = latest_cached_filing_text(
            upper,
            as_of_date=as_of_date,
            issuer_cik=issuer_cik,
            aliases=aliases,
            db_path=db_path,
            cfg=cfg,
        )
    else:
        profile = latest_company_profile(upper)
        text, filing_meta = latest_cached_filing_text(upper, as_of_date=as_of_date)
    terms = _extract_preferred_terms(text, str(profile.get("name") or ""))
    price = _as_float(current_price)

    missing: list[str] = []
    if terms.get("coupon_rate") is None:
        missing.append("PREFERRED_COUPON_MISSING")
    if terms.get("liquidation_preference") is None:
        missing.append("LIQUIDATION_PREFERENCE_MISSING")
    if terms.get("cumulative") is None:
        missing.append("CUMULATIVE_STATUS_MISSING")
    if price is None or price <= 0:
        missing.append("CURRENT_PRICE_MISSING")

    if missing:
        return {
            "status": "MODEL_BLOCKED",
            "model_status": "MODEL_BLOCKED",
            "method": "insurance_preferred",
            "ticker": upper,
            "as_of_date": as_of_date,
            "valuation_anchor": None,
            "preferred_terms": terms,
            "reason_codes": missing,
            "missing_components": missing,
            "source_references": {
                "company_name": profile.get("name"),
                "filing": filing_meta or UNKNOWN,
            },
        }

    coupon_rate = float(terms["coupon_rate"])
    liquidation_preference = float(terms["liquidation_preference"])
    assert price is not None
    annual_coupon = liquidation_preference * coupon_rate
    current_yield = annual_coupon / price
    ytc = _yield_to_call(
        price,
        liquidation_preference,
        annual_coupon,
        terms.get("call_date"),
        valuation_date=as_of_date if is_v2 else None,
    )
    yield_to_worst = min(current_yield, ytc) if ytc is not None else current_yield

    return {
        "status": "OK",
        "model_status": "OK",
        "method": "insurance_preferred",
        "ticker": upper,
        "as_of_date": as_of_date,
        "valuation_anchor": round(liquidation_preference, 6),
        "valuation_anchor_label": "liquidation_preference",
        "preferred_terms": {
            **terms,
            "annual_coupon": round(annual_coupon, 6),
        },
        "current_price": price,
        "current_yield": round(current_yield, 6),
        "yield_to_call": round(ytc, 6) if ytc is not None else UNKNOWN,
        "yield_to_worst": round(yield_to_worst, 6),
        "upside_to_liquidation_preference": round(
            (liquidation_preference - price) / liquidation_preference, 6
        ),
        "dividend_safety": UNKNOWN,
        "seniority": "preferred_equity",
        "change_of_control_treatment": UNKNOWN,
        "liquidity_delisting_risk": UNKNOWN,
        "reason_codes": [],
        "missing_components": [
            "DIVIDEND_SAFETY",
            "CHANGE_OF_CONTROL_TREATMENT",
            "LIQUIDITY_DELISTING_RISK",
        ],
        "component_scores": [
            _component_score(
                "current_yield",
                "OK",
                round(current_yield, 6),
                0.30,
                "Coupon income relative to current market price.",
            ),
            _component_score(
                "liquidation_preference_discount",
                "OK",
                round((liquidation_preference - price) / liquidation_preference, 6),
                0.25,
                "Discount to contractual liquidation preference.",
            ),
            _component_score(
                "cumulative_status",
                "OK" if terms.get("cumulative") else "WEAK",
                terms.get("cumulative"),
                0.20,
                "Cumulative preferred dividends are structurally stronger than non-cumulative dividends.",
            ),
            _component_score(
                "change_of_control_terms",
                UNKNOWN,
                UNKNOWN,
                0.25,
                "Merger/change-of-control treatment requires explicit security terms.",
            ),
        ],
        "source_references": {
            "company_name": profile.get("name"),
            "filing": filing_meta or UNKNOWN,
        },
    }
