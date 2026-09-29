from __future__ import annotations

from typing import Any


UNKNOWN = "UNKNOWN"

REQUIRED_FIELDS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "cfo",
    "capex",
    "fcf",
    "shares_outstanding",
    "shares_yoy_change",
    "net_debt",
]

EXTENDED_STANDARDIZED_FIELDS = [
    "r_and_d_total",
    "sales_marketing_total",
    "g_and_a_total",
    "risk_factor_keyword_count",
    "acquisition_mentions_count",
    "deferred_revenue_amount",
    "rpo_amount",
    "customer_concentration_pct",
    "segment_count",
]


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _safe_div(a: Any, b: Any) -> float | str:
    if not _is_num(a) or not _is_num(b) or float(b) == 0:
        return UNKNOWN
    return float(a) / float(b)


def _metric_map_by_year(items: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, dict[str, Any]]]]:
    rows: dict[int, dict[str, Any]] = {}
    traces: dict[int, dict[str, dict[str, Any]]] = {}
    for item in items:
        year = item.get("year")
        metric = item.get("metric")
        if not isinstance(year, int) or not isinstance(metric, str):
            continue
        if year not in rows:
            rows[year] = {"year": year}
            traces[year] = {}
        rows[year][metric] = item.get("value", UNKNOWN)
        if item.get("source_url"):
            rows[year][f"{metric}__source_url"] = item.get("source_url")
        if item.get("snippet"):
            rows[year][f"{metric}__snippet"] = item.get("snippet")
        traces[year][metric] = {
            "citations": list(item.get("citations") or []),
            "derived_from": list(item.get("derived_from") or []),
        }
    return rows, traces


def _row_metric(row: dict[str, Any], *labels: str) -> Any:
    for label in labels:
        if label in row:
            return row.get(label, UNKNOWN)
    return UNKNOWN


def _cagr(values: list[tuple[int, Any]], years_window: int, label: str) -> dict[str, Any]:
    known = [(y, float(v)) for y, v in values if _is_num(v) and float(v) > 0]
    if len(known) < 2:
        if known:
            first_year = known[0][0]
            last_year = known[-1][0]
            refs = [
                f"dossier.time_series.standardized_rows[{first_year}].{label}",
                f"dossier.time_series.standardized_rows[{last_year}].{label}",
            ]
        else:
            refs = [f"dossier.time_series.standardized_rows[*].{label}"]
        return {
            "signal": f"{label}_cagr_{years_window}y",
            "value": UNKNOWN,
            "derived_from": refs,
        }
    end_year, end_value = known[-1]
    start_target = end_year - years_window
    candidates = [(y, v) for y, v in known if y <= start_target]
    if candidates:
        start_year, start_value = candidates[-1]
    else:
        start_year, start_value = known[0]
    span = max(1, end_year - start_year)
    value = (end_value / start_value) ** (1.0 / span) - 1.0
    return {
        "signal": f"{label}_cagr_{years_window}y",
        "value": value,
        "derived_from": [
            f"dossier.time_series.standardized_rows[{start_year}].{label}",
            f"dossier.time_series.standardized_rows[{end_year}].{label}",
        ],
    }


def _slope(values: list[tuple[int, Any]], signal_name: str, label: str) -> dict[str, Any]:
    known = [(y, float(v)) for y, v in values if _is_num(v)]
    if len(known) < 2:
        if known:
            refs = [f"dossier.time_series.standardized_rows[{known[0][0]}].{label}"]
        else:
            refs = [f"dossier.time_series.standardized_rows[*].{label}"]
        return {
            "signal": signal_name,
            "value": UNKNOWN,
            "derived_from": refs,
        }
    first_year, first_value = known[0]
    last_year, last_value = known[-1]
    span = max(1, last_year - first_year)
    return {
        "signal": signal_name,
        "value": (last_value - first_value) / span,
        "derived_from": [
            f"dossier.time_series.standardized_rows[{first_year}].{label}",
            f"dossier.time_series.standardized_rows[{last_year}].{label}",
        ],
    }


def build_time_series(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_year, traces_by_year = _metric_map_by_year(items)
    years = sorted(by_year.keys())
    rows = [by_year[y] for y in years]
    derived_signals: list[dict[str, Any]] = []
    if not years:
        return {
            "years": [],
            "rows": [],
            "standardized_rows": [],
            "standardized_row_traces": {},
            "required_fields": REQUIRED_FIELDS,
            "derived_signals": [],
        }

    standardized_rows: list[dict[str, Any]] = []
    standardized_row_traces: dict[str, dict[str, dict[str, Any]]] = {}
    prev_shares: float | None = None
    for year in years:
        row = by_year[year]
        traces = traces_by_year.get(year, {})

        revenue = _row_metric(row, "revenue")
        gross_profit = _row_metric(row, "gross_profit", "gross_profit_dollars")
        operating_income = _row_metric(row, "operating_income", "operating_income_dollars")
        net_income = _row_metric(row, "net_income")
        cfo = _row_metric(row, "cfo")
        capex = _row_metric(row, "capex")
        fcf = _row_metric(row, "fcf", "fcf_dollars")
        shares = _row_metric(row, "shares_outstanding")
        net_debt = _row_metric(row, "net_debt")
        r_and_d_total = _row_metric(row, "r_and_d_total")
        sales_marketing_total = _row_metric(row, "sales_marketing_total")
        g_and_a_total = _row_metric(row, "g_and_a_total")
        risk_factor_keyword_count = _row_metric(row, "risk_factor_keyword_count")
        acquisition_mentions_count = _row_metric(row, "acquisition_mentions_count")
        deferred_revenue_amount = _row_metric(row, "deferred_revenue_amount")
        rpo_amount = _row_metric(row, "rpo_amount")
        customer_concentration_pct = _row_metric(row, "customer_concentration_pct")
        segment_count = _row_metric(row, "segment_count")
        if _is_num(shares) and prev_shares is not None and prev_shares != 0:
            shares_yoy = (float(shares) / float(prev_shares)) - 1.0
            shares_yoy_trace = {
                "citations": [],
                "derived_from": [
                    f"dossier.time_series.standardized_rows[{year-1}].shares_outstanding",
                    f"dossier.time_series.standardized_rows[{year}].shares_outstanding",
                ],
            }
        else:
            shares_yoy = UNKNOWN
            shares_yoy_trace = {"citations": [], "derived_from": []}
        if _is_num(shares):
            prev_shares = float(shares)

        std_row = {
            "year": year,
            "revenue": revenue,
            "gross_profit": gross_profit,
            "operating_income": operating_income,
            "net_income": net_income,
            "cfo": cfo,
            "capex": capex,
            "fcf": fcf,
            "shares_outstanding": shares,
            "shares_yoy_change": shares_yoy,
            "net_debt": net_debt,
            "r_and_d_total": r_and_d_total,
            "sales_marketing_total": sales_marketing_total,
            "g_and_a_total": g_and_a_total,
            "risk_factor_keyword_count": risk_factor_keyword_count,
            "acquisition_mentions_count": acquisition_mentions_count,
            "deferred_revenue_amount": deferred_revenue_amount,
            "rpo_amount": rpo_amount,
            "customer_concentration_pct": customer_concentration_pct,
            "segment_count": segment_count,
        }
        standardized_rows.append(std_row)

        standardized_row_traces[str(year)] = {
            "revenue": traces.get("revenue", {"citations": [], "derived_from": []}),
            "gross_profit": traces.get("gross_profit", traces.get("gross_profit_dollars", {"citations": [], "derived_from": []})),
            "operating_income": traces.get(
                "operating_income",
                traces.get("operating_income_dollars", {"citations": [], "derived_from": []}),
            ),
            "net_income": traces.get("net_income", {"citations": [], "derived_from": []}),
            "cfo": traces.get("cfo", {"citations": [], "derived_from": []}),
            "capex": traces.get("capex", {"citations": [], "derived_from": []}),
            "fcf": traces.get("fcf", traces.get("fcf_dollars", {"citations": [], "derived_from": []})),
            "shares_outstanding": traces.get("shares_outstanding", {"citations": [], "derived_from": []}),
            "shares_yoy_change": shares_yoy_trace,
            "net_debt": traces.get("net_debt", {"citations": [], "derived_from": []}),
            "r_and_d_total": traces.get("r_and_d_total", {"citations": [], "derived_from": []}),
            "sales_marketing_total": traces.get("sales_marketing_total", {"citations": [], "derived_from": []}),
            "g_and_a_total": traces.get("g_and_a_total", {"citations": [], "derived_from": []}),
            "risk_factor_keyword_count": traces.get("risk_factor_keyword_count", {"citations": [], "derived_from": []}),
            "acquisition_mentions_count": traces.get("acquisition_mentions_count", {"citations": [], "derived_from": []}),
            "deferred_revenue_amount": traces.get("deferred_revenue_amount", {"citations": [], "derived_from": []}),
            "rpo_amount": traces.get("rpo_amount", {"citations": [], "derived_from": []}),
            "customer_concentration_pct": traces.get("customer_concentration_pct", {"citations": [], "derived_from": []}),
            "segment_count": traces.get("segment_count", {"citations": [], "derived_from": []}),
        }

    first = rows[0]
    last = rows[-1]
    first_year = years[0]
    last_year = years[-1]

    first_shares = _row_metric(first, "shares_outstanding")
    last_shares = _row_metric(last, "shares_outstanding")
    if _is_num(first_shares) and _is_num(last_shares) and float(first_shares) != 0:
        dilution_value = (float(last_shares) / float(first_shares)) - 1.0
    else:
        dilution_value = UNKNOWN
    derived_signals.append(
        {
            "signal": "dilution_rate_proxy",
            "value": dilution_value,
            "derived_from": [
                f"dossier.time_series.rows[{first_year}].shares_outstanding",
                f"dossier.time_series.rows[{last_year}].shares_outstanding",
            ],
        }
    )

    for signal_name, metric in [
        ("gross_margin_delta", "gross_margin"),
        ("operating_margin_delta", "operating_margin"),
        ("fcf_margin_delta", "fcf_margin"),
        ("risk_factor_keyword_delta", "risk_factor_keyword_count"),
    ]:
        first_val = _row_metric(first, metric)
        last_val = _row_metric(last, metric)
        if _is_num(first_val) and _is_num(last_val):
            value = float(last_val) - float(first_val)
        else:
            value = UNKNOWN
        derived_signals.append(
            {
                "signal": signal_name,
                "value": value,
                "derived_from": [
                    f"dossier.time_series.rows[{first_year}].{metric}",
                    f"dossier.time_series.rows[{last_year}].{metric}",
                ],
            }
        )

    first_gp = _row_metric(first, "gross_profit_dollars", "gross_profit")
    last_gp = _row_metric(last, "gross_profit_dollars", "gross_profit")
    if _is_num(first_gp) and _is_num(last_gp):
        gp_trend = float(last_gp) - float(first_gp)
    else:
        gp_trend = UNKNOWN
    derived_signals.append(
        {
            "signal": "gross_profit_dollars_trend",
            "value": gp_trend,
            "derived_from": [
                f"dossier.time_series.rows[{first_year}].gross_profit_dollars",
                f"dossier.time_series.rows[{last_year}].gross_profit_dollars",
            ],
        }
    )

    first_rev = _row_metric(first, "revenue")
    last_rnd = _row_metric(last, "r_and_d_total")
    last_rev = _row_metric(last, "revenue")
    if _is_num(last_rnd) and _is_num(last_rev):
        rnd_intensity = _safe_div(last_rnd, last_rev)
    else:
        rnd_intensity = _row_metric(last, "r_and_d_pct_revenue")
    derived_signals.append(
        {
            "signal": "r_and_d_intensity_latest",
            "value": rnd_intensity if _is_num(rnd_intensity) else UNKNOWN,
            "derived_from": [
                f"dossier.time_series.rows[{last_year}].r_and_d_total",
                f"dossier.time_series.rows[{last_year}].revenue",
                f"dossier.time_series.rows[{last_year}].r_and_d_pct_revenue",
            ],
        }
    )

    first_rnd = _row_metric(first, "r_and_d_total")
    if _is_num(first_rnd) and _is_num(last_rnd) and _is_num(first_rev) and _is_num(last_rev):
        first_rnd_intensity = _safe_div(first_rnd, first_rev)
        last_rnd_intensity = _safe_div(last_rnd, last_rev)
        if _is_num(first_rnd_intensity) and _is_num(last_rnd_intensity):
            rnd_intensity_delta = float(last_rnd_intensity) - float(first_rnd_intensity)
        else:
            rnd_intensity_delta = UNKNOWN
    else:
        rnd_intensity_delta = UNKNOWN
    derived_signals.append(
        {
            "signal": "r_and_d_intensity_delta",
            "value": rnd_intensity_delta,
            "derived_from": [
                f"dossier.time_series.rows[{first_year}].r_and_d_total",
                f"dossier.time_series.rows[{first_year}].revenue",
                f"dossier.time_series.rows[{last_year}].r_and_d_total",
                f"dossier.time_series.rows[{last_year}].revenue",
            ],
        }
    )

    # Keep legacy revenue CAGR proxy over entire horizon.
    last_rev = _row_metric(last, "revenue")
    if _is_num(first_rev) and _is_num(last_rev) and float(first_rev) > 0 and len(years) > 1:
        n = len(years) - 1
        value = (float(last_rev) / float(first_rev)) ** (1 / n) - 1.0
    else:
        value = UNKNOWN
    derived_signals.append(
        {
            "signal": "revenue_cagr_proxy",
            "value": value,
            "derived_from": [
                f"dossier.time_series.rows[{first_year}].revenue",
                f"dossier.time_series.rows[{last_year}].revenue",
            ],
        }
    )

    revenue_values = [(row["year"], row.get("revenue")) for row in standardized_rows]
    derived_signals.extend(
        [
            _cagr(revenue_values, 3, "revenue"),
            _cagr(revenue_values, 5, "revenue"),
            _cagr(revenue_values, 10, "revenue"),
        ]
    )

    gross_margin_values = [(row["year"], _safe_div(row.get("gross_profit"), row.get("revenue"))) for row in standardized_rows]
    operating_margin_values = [(row["year"], _safe_div(row.get("operating_income"), row.get("revenue"))) for row in standardized_rows]
    fcf_margin_values = [(row["year"], _safe_div(row.get("fcf"), row.get("revenue"))) for row in standardized_rows]

    derived_signals.extend(
        [
            _slope(gross_margin_values, "gross_margin_trend_slope", "gross_margin"),
            _slope(operating_margin_values, "operating_margin_trend_slope", "operating_margin"),
            _slope(fcf_margin_values, "fcf_margin_trend_slope", "fcf_margin"),
        ]
    )

    # Shares CAGR as dilution rate.
    shares_values = [(row["year"], row.get("shares_outstanding")) for row in standardized_rows]
    shares_cagr = _cagr(shares_values, 10, "shares_outstanding")
    shares_cagr["signal"] = "dilution_rate_shares_cagr"
    derived_signals.append(shares_cagr)

    # ROIC proxy: operating income / (equity + net_debt) using latest year.
    last_std = standardized_rows[-1]
    op_income = last_std.get("operating_income")
    equity = _row_metric(last, "equity")
    net_debt_latest = last_std.get("net_debt")
    if _is_num(op_income) and _is_num(equity) and _is_num(net_debt_latest) and (float(equity) + float(net_debt_latest)) != 0:
        roic_proxy = float(op_income) / (float(equity) + float(net_debt_latest))
    else:
        roic_proxy = UNKNOWN
    derived_signals.append(
        {
            "signal": "roic_proxy",
            "value": roic_proxy,
            "derived_from": [
                f"dossier.time_series.standardized_rows[{last_std['year']}].operating_income",
                f"dossier.time_series.rows[{last_year}].equity",
                f"dossier.time_series.standardized_rows[{last_std['year']}].net_debt",
            ],
        }
    )

    for signal_name, metric in [
        ("segment_count_delta", "segment_count"),
        ("customer_concentration_delta", "customer_concentration_pct"),
    ]:
        first_val = _row_metric(first, metric)
        last_val = _row_metric(last, metric)
        if _is_num(first_val) and _is_num(last_val):
            value = float(last_val) - float(first_val)
        else:
            value = UNKNOWN
        derived_signals.append(
            {
                "signal": signal_name,
                "value": value,
                "derived_from": [
                    f"dossier.time_series.rows[{first_year}].{metric}",
                    f"dossier.time_series.rows[{last_year}].{metric}",
                ],
            }
        )

    deferred_revenue_amount = _row_metric(last, "deferred_revenue_amount")
    rpo_amount = _row_metric(last, "rpo_amount")
    derived_signals.append(
        {
            "signal": "deferred_revenue_to_revenue_latest",
            "value": _safe_div(deferred_revenue_amount, last_rev),
            "derived_from": [
                f"dossier.time_series.rows[{last_year}].deferred_revenue_amount",
                f"dossier.time_series.rows[{last_year}].revenue",
            ],
        }
    )
    derived_signals.append(
        {
            "signal": "rpo_to_revenue_latest",
            "value": _safe_div(rpo_amount, last_rev),
            "derived_from": [
                f"dossier.time_series.rows[{last_year}].rpo_amount",
                f"dossier.time_series.rows[{last_year}].revenue",
            ],
        }
    )

    return {
        "years": years,
        "rows": rows,
        "required_fields": REQUIRED_FIELDS,
        "standardized_rows": standardized_rows,
        "standardized_row_traces": standardized_row_traces,
        "derived_signals": derived_signals,
    }
