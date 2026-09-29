from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import utc_now_iso


UNKNOWN = "UNKNOWN"
FUNDAMENTALS_VERSION = "v1.1"

_BASE_FIELDS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "cfo",
    "capex",
    "fcf",
    "shares_outstanding",
    "net_debt",
]
_DERIVED_FIELDS = [
    "gross_margin",
    "op_margin",
    "fcf_margin",
    "cfo_margin",
]
_DERIVED_SIGNALS = [
    "revenue_cagr_3y",
    "revenue_cagr_5y",
    "revenue_cagr_10y",
    "gross_margin_trend_slope",
    "operating_margin_trend_slope",
    "fcf_margin_trend_slope",
    "dilution_rate_shares_cagr",
]


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _safe_div(a: Any, b: Any) -> float | str:
    if not _is_num(a) or not _is_num(b):
        return UNKNOWN
    if float(b) == 0.0:
        return UNKNOWN
    return float(a) / float(b)


def _slope(series: list[tuple[int, float]]) -> float | str:
    if len(series) < 2:
        return UNKNOWN
    start_year, start_value = series[0]
    end_year, end_value = series[-1]
    span = max(1, int(end_year) - int(start_year))
    return (float(end_value) - float(start_value)) / float(span)


def _cagr(series: list[tuple[int, float]], years: int) -> float | str:
    if len(series) < 2:
        return UNKNOWN
    end_year, end_value = series[-1]
    if float(end_value) <= 0:
        return UNKNOWN
    target_year = int(end_year) - int(years)
    candidates = [(year, value) for year, value in series if int(year) <= target_year and float(value) > 0]
    if candidates:
        start_year, start_value = candidates[-1]
    else:
        start_year, start_value = series[0]
    if float(start_value) <= 0:
        return UNKNOWN
    span = max(1, int(end_year) - int(start_year))
    return (float(end_value) / float(start_value)) ** (1.0 / float(span)) - 1.0


def _metric_trace(trace_map: dict[str, Any], year: int, metric: str) -> dict[str, Any]:
    year_bucket = trace_map.get(str(year), {}) if isinstance(trace_map, dict) else {}
    metric_bucket = year_bucket.get(metric, {}) if isinstance(year_bucket, dict) else {}
    derived = [str(x) for x in (metric_bucket.get("derived_from") or []) if str(x).strip()]
    cits = [x for x in (metric_bucket.get("citations") or []) if isinstance(x, dict)]
    if not derived:
        derived = [f"dossier.time_series.standardized_rows[{year}].{metric}"]
    return {"derived_from": derived, "citations": cits}


def _signal_lookup(dossier: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in (dossier.get("time_series") or {}).get("derived_signals") or []:
        if not isinstance(row, dict):
            continue
        signal = str(row.get("signal") or "").strip()
        if not signal:
            continue
        value = row.get("value")
        out[signal] = {
            "value": float(value) if _is_num(value) else UNKNOWN,
            "derived_from": [str(x) for x in (row.get("derived_from") or []) if str(x).strip()]
            or [f"dossier.time_series.derived_signals.{signal}"],
        }
    return out


def build_fundamentals_frame(
    dossier: dict[str, Any],
    *,
    years_back: int | None = None,
) -> dict[str, Any]:
    ticker = str(dossier.get("ticker") or "").upper()
    run_id = str(dossier.get("run_id") or "")
    as_of_date = str(dossier.get("as_of_date") or "")
    rows_raw = [row for row in ((dossier.get("time_series") or {}).get("standardized_rows") or []) if isinstance(row, dict)]
    rows_raw = sorted(rows_raw, key=lambda row: int(row.get("year", 0)))
    if years_back is not None and int(years_back) > 0 and rows_raw:
        rows_raw = rows_raw[-int(years_back) :]
    traces_map = (dossier.get("time_series") or {}).get("standardized_row_traces") or {}
    raw_rows_by_year = {
        int(row.get("year", 0)): row
        for row in ((dossier.get("time_series") or {}).get("rows") or [])
        if isinstance(row, dict) and int(row.get("year", 0)) > 0
    }

    frame_rows: list[dict[str, Any]] = []
    row_traces: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows_raw:
        year = int(row.get("year", 0))
        raw_row = raw_rows_by_year.get(year, {})
        revenue = row.get("revenue", UNKNOWN)
        gross_profit = row.get("gross_profit", UNKNOWN)
        operating_income = row.get("operating_income", UNKNOWN)
        cfo = row.get("cfo", UNKNOWN)
        fcf = row.get("fcf", UNKNOWN)
        r_and_d_total = raw_row.get("r_and_d_total", row.get("r_and_d_total", UNKNOWN))
        sales_marketing_total = raw_row.get("sales_marketing_total", row.get("sales_marketing_total", UNKNOWN))
        g_and_a_total = raw_row.get("g_and_a_total", row.get("g_and_a_total", UNKNOWN))
        sga_total = raw_row.get("sga_total", row.get("sga_total", UNKNOWN))
        deferred_revenue_amount = raw_row.get("deferred_revenue_amount", row.get("deferred_revenue_amount", UNKNOWN))
        rpo_amount = raw_row.get("rpo_amount", row.get("rpo_amount", UNKNOWN))
        customer_concentration_pct = raw_row.get("customer_concentration_pct", row.get("customer_concentration_pct", UNKNOWN))
        customer_concentration_present = raw_row.get("customer_concentration_present", row.get("customer_concentration_present", UNKNOWN))
        segment_count = raw_row.get("segment_count", row.get("segment_count", UNKNOWN))
        share_repurchases_amount = raw_row.get("share_repurchases_amount", row.get("share_repurchases_amount", UNKNOWN))
        dividends_paid_amount = raw_row.get("dividends_paid_amount", row.get("dividends_paid_amount", UNKNOWN))
        if not _is_num(sga_total) and _is_num(sales_marketing_total) and _is_num(g_and_a_total):
            sga_total = float(sales_marketing_total) + float(g_and_a_total)
        values = {
            "year": year,
            "revenue": revenue,
            "gross_profit": gross_profit,
            "operating_income": operating_income,
            "net_income": row.get("net_income", UNKNOWN),
            "cfo": cfo,
            "capex": row.get("capex", UNKNOWN),
            "fcf": fcf,
            "shares_outstanding": row.get("shares_outstanding", UNKNOWN),
            "net_debt": row.get("net_debt", UNKNOWN),
            "r_and_d_total": r_and_d_total,
            "sales_marketing_total": sales_marketing_total,
            "g_and_a_total": g_and_a_total,
            "sga_total": sga_total,
            "deferred_revenue_amount": deferred_revenue_amount,
            "rpo_amount": rpo_amount,
            "customer_concentration_pct": customer_concentration_pct,
            "customer_concentration_present": customer_concentration_present,
            "segment_count": segment_count,
            "share_repurchases_amount": share_repurchases_amount,
            "dividends_paid_amount": dividends_paid_amount,
            "gross_margin": _safe_div(gross_profit, revenue),
            "op_margin": _safe_div(operating_income, revenue),
            "fcf_margin": _safe_div(fcf, revenue),
            "cfo_margin": _safe_div(cfo, revenue),
        }
        frame_rows.append(values)
        row_traces[str(year)] = {}
        for metric in _BASE_FIELDS:
            row_traces[str(year)][metric] = _metric_trace(traces_map, year, metric)
        row_traces[str(year)]["gross_margin"] = {
            "derived_from": [
                f"dossier.time_series.standardized_rows[{year}].gross_profit",
                f"dossier.time_series.standardized_rows[{year}].revenue",
            ],
            "citations": [],
        }
        row_traces[str(year)]["op_margin"] = {
            "derived_from": [
                f"dossier.time_series.standardized_rows[{year}].operating_income",
                f"dossier.time_series.standardized_rows[{year}].revenue",
            ],
            "citations": [],
        }
        row_traces[str(year)]["fcf_margin"] = {
            "derived_from": [
                f"dossier.time_series.standardized_rows[{year}].fcf",
                f"dossier.time_series.standardized_rows[{year}].revenue",
            ],
            "citations": [],
        }
        row_traces[str(year)]["cfo_margin"] = {
            "derived_from": [
                f"dossier.time_series.standardized_rows[{year}].cfo",
                f"dossier.time_series.standardized_rows[{year}].revenue",
            ],
            "citations": [],
        }
        row_traces[str(year)]["r_and_d_total"] = {
            "derived_from": [f"dossier.time_series.rows[{year}].r_and_d_total"],
            "citations": [],
        }
        row_traces[str(year)]["sales_marketing_total"] = {
            "derived_from": [f"dossier.time_series.rows[{year}].sales_marketing_total"],
            "citations": [],
        }
        row_traces[str(year)]["g_and_a_total"] = {
            "derived_from": [f"dossier.time_series.rows[{year}].g_and_a_total"],
            "citations": [],
        }
        row_traces[str(year)]["sga_total"] = {
            "derived_from": (
                [
                    f"dossier.time_series.rows[{year}].sales_marketing_total",
                    f"dossier.time_series.rows[{year}].g_and_a_total",
                ]
                if not _is_num(raw_row.get("sga_total", UNKNOWN))
                and _is_num(sales_marketing_total)
                and _is_num(g_and_a_total)
                else [f"dossier.time_series.rows[{year}].sga_total"]
            ),
            "citations": [],
        }
        for metric in (
            "deferred_revenue_amount",
            "rpo_amount",
            "customer_concentration_pct",
            "customer_concentration_present",
            "segment_count",
            "share_repurchases_amount",
            "dividends_paid_amount",
        ):
            row_traces[str(year)][metric] = _metric_trace(traces_map, year, metric)

    revenue_series = [(int(row["year"]), float(row["revenue"])) for row in frame_rows if _is_num(row.get("revenue")) and float(row["revenue"]) > 0]
    shares_series = [(int(row["year"]), float(row["shares_outstanding"])) for row in frame_rows if _is_num(row.get("shares_outstanding")) and float(row["shares_outstanding"]) > 0]
    gm_series = [(int(row["year"]), float(row["gross_margin"])) for row in frame_rows if _is_num(row.get("gross_margin"))]
    om_series = [(int(row["year"]), float(row["op_margin"])) for row in frame_rows if _is_num(row.get("op_margin"))]
    fcfm_series = [(int(row["year"]), float(row["fcf_margin"])) for row in frame_rows if _is_num(row.get("fcf_margin"))]

    existing_signals = _signal_lookup(dossier)
    derived_signals: dict[str, dict[str, Any]] = {}

    def _set_signal(name: str, value: float | str, refs: list[str]) -> None:
        if name in existing_signals:
            derived_signals[name] = existing_signals[name]
            return
        derived_signals[name] = {
            "value": value,
            "derived_from": refs,
        }

    _set_signal(
        "revenue_cagr_3y",
        _cagr(revenue_series, 3),
        ["dossier.time_series.standardized_rows[*].revenue"],
    )
    _set_signal(
        "revenue_cagr_5y",
        _cagr(revenue_series, 5),
        ["dossier.time_series.standardized_rows[*].revenue"],
    )
    _set_signal(
        "revenue_cagr_10y",
        _cagr(revenue_series, 10),
        ["dossier.time_series.standardized_rows[*].revenue"],
    )
    _set_signal(
        "gross_margin_trend_slope",
        _slope(gm_series),
        ["dossier.time_series.standardized_rows[*].gross_profit", "dossier.time_series.standardized_rows[*].revenue"],
    )
    _set_signal(
        "operating_margin_trend_slope",
        _slope(om_series),
        ["dossier.time_series.standardized_rows[*].operating_income", "dossier.time_series.standardized_rows[*].revenue"],
    )
    _set_signal(
        "fcf_margin_trend_slope",
        _slope(fcfm_series),
        ["dossier.time_series.standardized_rows[*].fcf", "dossier.time_series.standardized_rows[*].revenue"],
    )
    _set_signal(
        "dilution_rate_shares_cagr",
        _cagr(shares_series, 10),
        ["dossier.time_series.standardized_rows[*].shares_outstanding"],
    )
    for signal_name in (
        "r_and_d_intensity_latest",
        "r_and_d_intensity_delta",
        "segment_count_delta",
        "customer_concentration_delta",
        "deferred_revenue_to_revenue_latest",
        "rpo_to_revenue_latest",
        "roic_proxy",
    ):
        if signal_name in derived_signals:
            continue
        signal_payload = existing_signals.get(signal_name)
        if signal_payload:
            derived_signals[signal_name] = signal_payload

    gaps: list[dict[str, Any]] = []
    for field in _BASE_FIELDS:
        missing = len([row for row in frame_rows if not _is_num(row.get(field))])
        if missing <= 0:
            continue
        gaps.append(
            {
                "field": field,
                "missing_count": int(missing),
                "derived_from": [f"dossier.time_series.standardized_rows[*].{field}"],
            }
        )
    latest = frame_rows[-1] if frame_rows else {}
    for field in _BASE_FIELDS:
        if _is_num(latest.get(field)):
            continue
        gaps.append(
            {
                "field": field,
                "missing_latest": True,
                "derived_from": [f"dossier.time_series.standardized_rows[{latest.get('year', '*')}].{field}"],
            }
        )

    return {
        "fundamentals_version": FUNDAMENTALS_VERSION,
        "ticker": ticker,
        "run_id": run_id,
        "as_of_date": as_of_date,
        "rows": frame_rows,
        "row_traces": row_traces,
        "derived_signals": {
            name: {
                "value": data.get("value", UNKNOWN),
                "derived_from": data.get("derived_from") or [f"dossier.time_series.derived_signals.{name}"],
            }
            for name, data in sorted(derived_signals.items())
        },
        "gaps": gaps,
        "generated_at": utc_now_iso(),
    }


def _load_dossier(run_dir: Path, ticker: str) -> dict[str, Any] | None:
    dossier_path = run_dir / ticker / "dossier.json"
    if not dossier_path.exists():
        return None
    try:
        payload = json.loads(dossier_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _iter_dossier_tickers(run_dir: Path) -> list[str]:
    if not run_dir.exists():
        return []
    return sorted([p.name for p in run_dir.iterdir() if p.is_dir() and p.name.isupper()])


def write_fundamentals_for_run(
    *,
    run_id: str,
    tickers: list[str] | None = None,
    years_back: int = 10,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    dossier_dir = cfg.dossiers_dir / run_id
    out_dir = output_dir or (cfg.sectors_dir / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_tickers = sorted({str(t).strip().upper() for t in (tickers or _iter_dossier_tickers(dossier_dir)) if str(t).strip()})
    rows: list[dict[str, Any]] = []
    missingness: dict[str, int] = {field: 0 for field in (_BASE_FIELDS + _DERIVED_FIELDS)}

    for ticker in target_tickers:
        dossier = _load_dossier(dossier_dir, ticker)
        if not dossier:
            rows.append(
                {
                    "ticker": ticker,
                    "status": "MISSING_DOSSIER",
                    "path": None,
                    "gaps": [{"field": "dossier", "missing": True, "derived_from": [f"dossiers/{ticker}/dossier.json"]}],
                }
            )
            continue
        fundamentals = build_fundamentals_frame(dossier, years_back=years_back)
        out_path = out_dir / f"fundamentals_{ticker}.json"
        out_path.write_text(json.dumps(fundamentals, indent=2), encoding="utf-8")
        latest_row = fundamentals.get("rows", [])[-1] if fundamentals.get("rows") else {}
        for field in (_BASE_FIELDS + _DERIVED_FIELDS):
            if not _is_num(latest_row.get(field)):
                missingness[field] = int(missingness.get(field, 0)) + 1
        rows.append(
            {
                "ticker": ticker,
                "status": "OK",
                "path": str(out_path),
                "gaps": fundamentals.get("gaps") or [],
            }
        )

    rows_sorted = sorted(rows, key=lambda row: str(row.get("ticker") or ""))
    summary = {
        "run_id": run_id,
        "fundamentals_version": FUNDAMENTALS_VERSION,
        "years_back": int(years_back),
        "output_dir": str(out_dir),
        "ticker_count": len(rows_sorted),
        "ok_count": len([row for row in rows_sorted if row.get("status") == "OK"]),
        "rows": rows_sorted,
        "missingness_by_field": {key: int(value) for key, value in sorted(missingness.items())},
        "generated_at": utc_now_iso(),
    }
    summary_path = out_dir / "fundamentals_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
