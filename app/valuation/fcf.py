from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.valuation.facts import resolve_financial_facts_asof
from app.valuation.fundamentals import UNKNOWN


# What this resolver returns, stated so no consumer has to infer it.
#
# FCF = CFO - capex. US GAAP classifies cash interest paid as an operating
# outflow, so CFO is already net of interest and this is a LEVERED stream -
# free cash flow to equity before net borrowing, not free cash flow to the
# firm. Two consequences a caller must respect:
#   1. It belongs with a cost-of-equity discount rate (or an equity multiple),
#      not a WACC, and it yields EQUITY value directly. Capitalising it and
#      then subtracting net debt counts the debt twice - once through the
#      interest already deducted inside CFO, and again explicitly.
#      app/valuation/engine.py::_owner_earnings_intrinsic does exactly that.
#   2. Stock-based compensation is ADDED BACK as a non-cash charge, because
#      CFO adds it back. app/valuation/valuation_writer.py subtracts SBC and
#      adds back after-tax interest, so its owner-earnings figure is an
#      unlevered, SBC-expensed stream and is NOT comparable with this one.


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d")
    except Exception:
        return None


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _trace_refs(trace_bucket: dict[str, Any] | None, fallback: str) -> list[str]:
    refs = [str(x) for x in ((trace_bucket or {}).get("derived_from") or []) if str(x).strip()]
    return refs or [fallback]


def _extract_fcf_from_dossier(
    payload: dict[str, Any],
    *,
    run_id: str,
    ticker: str,
) -> dict[str, Any]:
    ts = payload.get("time_series") if isinstance(payload.get("time_series"), dict) else {}
    rows = [row for row in (ts.get("standardized_rows") or []) if isinstance(row, dict)]
    trace_map = ts.get("standardized_row_traces") if isinstance(ts.get("standardized_row_traces"), dict) else {}
    rows = sorted(rows, key=lambda row: int(row.get("year", 0)))
    out = {
        "fcf_value": None,
        "fcf_reason_code": "DOSSIER_NO_FCF",
        "fcf_asof_used": str(payload.get("as_of_date") or ""),
        "cfo_status": "UNKNOWN",
        "cfo_reason_code": "MISSING_CFO",
        "cfo_value": UNKNOWN,
        "cfo_asof_used": str(payload.get("as_of_date") or ""),
        "capex_status": "UNKNOWN",
        "capex_reason_code": "MISSING_CAPEX",
        "capex_value": UNKNOWN,
        "capex_asof_used": str(payload.get("as_of_date") or ""),
        "derived_from": [],
    }
    if not rows:
        return out

    latest = rows[-1]
    year = int(latest.get("year", 0))
    year_trace = trace_map.get(str(year)) if isinstance(trace_map.get(str(year)), dict) else {}

    cfo = latest.get("cfo")
    if _is_num(cfo):
        out["cfo_status"] = "OK"
        out["cfo_reason_code"] = "OK"
        out["cfo_value"] = float(cfo)
    cfo_refs = _trace_refs(
        (year_trace or {}).get("cfo") if isinstance(year_trace, dict) else None,
        f"dossiers.{run_id}.{ticker}.time_series.standardized_rows[{year}].cfo",
    )

    capex = latest.get("capex")
    if _is_num(capex):
        out["capex_status"] = "OK"
        out["capex_reason_code"] = "OK"
        out["capex_value"] = float(capex)
    capex_refs = _trace_refs(
        (year_trace or {}).get("capex") if isinstance(year_trace, dict) else None,
        f"dossiers.{run_id}.{ticker}.time_series.standardized_rows[{year}].capex",
    )

    fcf_value = latest.get("fcf")
    if _is_num(fcf_value):
        out["fcf_value"] = float(fcf_value)
        out["fcf_reason_code"] = "OK"
        fcf_refs = _trace_refs(
            (year_trace or {}).get("fcf") if isinstance(year_trace, dict) else None,
            f"dossiers.{run_id}.{ticker}.time_series.standardized_rows[{year}].fcf",
        )
        out["derived_from"] = fcf_refs + cfo_refs + capex_refs
        return out

    if out["cfo_status"] != "OK":
        out["fcf_reason_code"] = "MISSING_CFO"
        out["derived_from"] = cfo_refs
        return out
    if out["capex_status"] != "OK":
        out["fcf_reason_code"] = "MISSING_CAPEX"
        out["derived_from"] = capex_refs
        return out

    # Capex as a MAGNITUDE: a filer that reports the element negated would
    # otherwise turn this subtraction into an addition and publish FCF above
    # CFO. "capex_value" keeps the filed sign; only the deduction is normalised.
    out["fcf_value"] = float(out["cfo_value"]) - abs(float(out["capex_value"]))
    out["fcf_reason_code"] = "OK"
    out["derived_from"] = cfo_refs + capex_refs + ["derived:fcf=cfo-capex"]
    return out


def _iter_historical_dossiers(
    *,
    cfg: AppConfig,
    ticker: str,
    as_of_date: str,
    skip_run_id: str | None,
) -> list[tuple[datetime, str, Path, dict[str, Any]]]:
    requested_dt = _parse_date(as_of_date)
    if requested_dt is None:
        return []
    candidates: list[tuple[datetime, str, Path, dict[str, Any]]] = []
    pattern = f"*/{ticker.upper()}/dossier.json"
    for path in sorted(cfg.dossiers_dir.glob(pattern)):
        run_id = path.parent.parent.name
        if skip_run_id and run_id == skip_run_id:
            continue
        payload = _safe_json(path)
        asof = str(payload.get("as_of_date") or "").strip()
        asof_dt = _parse_date(asof)
        if asof_dt is None or asof_dt > requested_dt:
            continue
        candidates.append((asof_dt, run_id, path, payload))
    return candidates


def _ordered_historical_candidates(candidates: list[tuple[datetime, str, Path, dict[str, Any]]]) -> list[tuple[datetime, str, Path, dict[str, Any]]]:
    return sorted(candidates, key=lambda row: (-row[0].toordinal(), row[1]))


def resolve_fcf_asof(ticker: str, as_of_date: str, run_id: str | None) -> tuple[float | None, dict[str, Any]]:
    cfg = get_config()
    ticker_norm = str(ticker or "").upper().strip()
    asof_norm = str(as_of_date or "").strip()
    coverage = {
        "ticker": ticker_norm,
        "requested_as_of": asof_norm,
        "run_id": run_id,
        "fcf_status": "UNKNOWN",
        "fcf_reason_code": "DOSSIER_NO_FCF",
        "fcf_reason_detail": "No current-run dossier FCF was found.",
        "fcf_value": UNKNOWN,
        "fcf_asof_used": None,
        "fcf_source": None,
        "fcf_source_resolution": "unknown",
        "cfo_status": "UNKNOWN",
        "cfo_reason_code": "MISSING_CFO",
        "cfo_value": UNKNOWN,
        "cfo_asof_used": None,
        "capex_status": "UNKNOWN",
        "capex_reason_code": "MISSING_CAPEX",
        "capex_value": UNKNOWN,
        "capex_asof_used": None,
        "fetch_reason_code": None,
        "fetch_reason_detail": None,
        "cache_path": None,
        "source_url": None,
        "http_status": None,
        "network_attempted": False,
        "derived_from": [],
    }
    try:
        if not ticker_norm or not asof_norm:
            coverage["fcf_reason_code"] = "EXCEPTION"
            coverage["fcf_reason_detail"] = "Ticker or as_of_date is empty."
            return None, coverage

        if run_id:
            dossier_path = cfg.dossiers_dir / run_id / ticker_norm / "dossier.json"
            dossier_payload = _safe_json(dossier_path)
            dossier_dt = _parse_date(str(dossier_payload.get("as_of_date") or ""))
            requested_dt = _parse_date(asof_norm)
            # The current-run route needs the same date proof as historical dossiers.
            if dossier_dt is None or requested_dt is None or dossier_dt > requested_dt:
                dossier_payload = {}
            extracted = _extract_fcf_from_dossier(
                dossier_payload,
                run_id=run_id,
                ticker=ticker_norm,
            )
            coverage.update(
                {
                    "fcf_reason_code": extracted["fcf_reason_code"],
                    "fcf_asof_used": extracted["fcf_asof_used"] or None,
                    "cfo_status": extracted["cfo_status"],
                    "cfo_reason_code": extracted["cfo_reason_code"],
                    "cfo_value": extracted["cfo_value"],
                    "cfo_asof_used": extracted["cfo_asof_used"] or None,
                    "capex_status": extracted["capex_status"],
                    "capex_reason_code": extracted["capex_reason_code"],
                    "capex_value": extracted["capex_value"],
                    "capex_asof_used": extracted["capex_asof_used"] or None,
                    "derived_from": list(extracted["derived_from"]) + ([str(dossier_path)] if dossier_payload else []),
                }
            )
            if _is_num(extracted["fcf_value"]):
                value = float(extracted["fcf_value"])
                coverage.update(
                    {
                        "fcf_status": "OK",
                        "fcf_reason_code": "OK",
                        "fcf_reason_detail": "Resolved from current run dossier.",
                        "fcf_value": value,
                        "fcf_source": "current_run_dossier",
                        "fcf_source_resolution": "current_run_dossier",
                    }
                )
                return value, coverage
            if extracted["fcf_reason_code"] in {"MISSING_CFO", "MISSING_CAPEX", "DOSSIER_NO_FCF"}:
                coverage["fcf_reason_detail"] = "Current run dossier missing required FCF inputs."

        historical_candidates = _iter_historical_dossiers(
            cfg=cfg,
            ticker=ticker_norm,
            as_of_date=asof_norm,
            skip_run_id=run_id,
        )
        for _, hist_run_id, path, payload in _ordered_historical_candidates(historical_candidates):
            extracted = _extract_fcf_from_dossier(
                payload,
                run_id=hist_run_id,
                ticker=ticker_norm,
            )
            if not _is_num(extracted["fcf_value"]):
                continue
            value = float(extracted["fcf_value"])
            coverage.update(
                {
                    "fcf_status": "OK",
                    "fcf_reason_code": "HISTORICAL_DOSSIER_HIT",
                    "fcf_reason_detail": "Resolved FCF from historical dossier fallback.",
                    "fcf_value": value,
                    "fcf_asof_used": extracted["fcf_asof_used"] or asof_norm,
                    "fcf_source": f"historical_dossier:{hist_run_id}",
                    "fcf_source_resolution": "historical_dossier",
                    "cfo_status": extracted["cfo_status"],
                    "cfo_reason_code": extracted["cfo_reason_code"],
                    "cfo_value": extracted["cfo_value"],
                    "cfo_asof_used": extracted["cfo_asof_used"] or None,
                    "capex_status": extracted["capex_status"],
                    "capex_reason_code": extracted["capex_reason_code"],
                    "capex_value": extracted["capex_value"],
                    "capex_asof_used": extracted["capex_asof_used"] or None,
                    "derived_from": list(extracted["derived_from"]) + [str(path)],
                }
            )
            return value, coverage

        base_reason_code = "NO_HISTORICAL_DOSSIER" if not historical_candidates else str(coverage.get("fcf_reason_code") or "DOSSIER_NO_FCF")
        base_reason_detail = (
            "No historical dossier found at or before requested as_of_date."
            if not historical_candidates
            else str(coverage.get("fcf_reason_detail") or "No FCF in current or historical dossiers.")
        )
        coverage["fcf_reason_code"] = base_reason_code
        coverage["fcf_reason_detail"] = base_reason_detail

        facts_row = resolve_financial_facts_asof(
            ticker=ticker_norm,
            as_of_date=asof_norm,
            run_id=run_id,
            refresh=False,
            cfg=cfg,
        )
        coverage.update(
            {
                "fetch_reason_code": facts_row.get("fetch_reason_code"),
                "fetch_reason_detail": facts_row.get("fetch_reason_detail"),
                "cache_path": facts_row.get("cache_path"),
                "source_url": facts_row.get("source_url"),
                "http_status": facts_row.get("http_status"),
                "network_attempted": bool(facts_row.get("network_attempted")),
                "cfo_status": str(facts_row.get("cfo_status") or coverage.get("cfo_status")),
                "cfo_reason_code": str(facts_row.get("cfo_reason") or coverage.get("cfo_reason_code")),
                "cfo_value": facts_row.get("cfo_value", coverage.get("cfo_value")),
                "cfo_asof_used": facts_row.get("cfo_asof_used") or coverage.get("cfo_asof_used"),
                "capex_status": str(facts_row.get("capex_status") or coverage.get("capex_status")),
                "capex_reason_code": str(facts_row.get("capex_reason") or coverage.get("capex_reason_code")),
                "capex_value": facts_row.get("capex_value", coverage.get("capex_value")),
                "capex_asof_used": facts_row.get("capex_asof_used") or coverage.get("capex_asof_used"),
            }
        )
        facts_refs = [str(ref) for ref in (facts_row.get("derived_from") or []) if str(ref).strip()]
        facts_value = facts_row.get("fcf_value")
        facts_source = str(facts_row.get("source_resolution") or "unknown")
        facts_reason = str(facts_row.get("fcf_reason") or "COMPANYFACTS_MISS").upper()
        if _is_num(facts_value):
            reason_code = "COMPANYFACTS_FCF_HIT"
            if facts_reason in {"OK"} and str(facts_row.get("cfo_status") or "").upper() == "OK" and str(facts_row.get("capex_status") or "").upper() == "OK":
                reason_code = "COMPANYFACTS_CFO_CAPEX_HIT"
            value = float(facts_value)
            coverage.update(
                {
                    "fcf_status": "OK",
                    "fcf_reason_code": reason_code,
                    "fcf_reason_detail": "Resolved FCF from SEC companyfacts fallback.",
                    "fcf_value": value,
                    "fcf_asof_used": facts_row.get("fcf_asof_used") or asof_norm,
                    "fcf_source": facts_source,
                    "fcf_source_resolution": facts_source,
                    "derived_from": facts_refs,
                }
            )
            return value, coverage

        if str(facts_row.get("status") or "").upper() in {"OK", "PARTIAL"}:
            coverage.update(
                {
                    "fcf_reason_code": "COMPANYFACTS_MISS",
                    "fcf_reason_detail": "Companyfacts payload available but required FCF facts are missing.",
                    "fcf_source": facts_source,
                    "fcf_source_resolution": facts_source,
                    "derived_from": facts_refs,
                }
            )
        return None, coverage
    except Exception as exc:  # noqa: BLE001
        coverage.update(
            {
                "fcf_status": "UNKNOWN",
                "fcf_reason_code": "EXCEPTION",
                "fcf_reason_detail": str(exc),
                "fcf_value": UNKNOWN,
            }
        )
        return None, coverage


def write_fcf_coverage_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str] | None = None,
    output_dir: Path | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    run_dir = output_dir or (cfg.sectors_dir / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        dossier_root = cfg.dossiers_dir / run_id
        candidates = sorted([path.name.upper() for path in dossier_root.iterdir() if path.is_dir()]) if dossier_root.exists() else []
        if not candidates:
            candidates = sorted(
                {
                    str(path.stem).replace("fundamentals_", "").upper()
                    for path in run_dir.glob("fundamentals_*.json")
                    if str(path.stem).replace("fundamentals_", "").strip()
                }
            )
    else:
        candidates = sorted({str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()})

    entries: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    ok_count = 0
    for ticker in candidates:
        value, row = resolve_fcf_asof(ticker=ticker, as_of_date=as_of_date, run_id=run_id)
        if _is_num(value):
            ok_count += 1
        code = str(row.get("fcf_reason_code") or "UNKNOWN")
        reason_counts[code] = reason_counts.get(code, 0) + 1
        entries.append(row)

    entries = sorted(entries, key=lambda row: str(row.get("ticker") or ""))
    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(entries),
        "ok_count": int(ok_count),
        "unknown_count": int(len(entries) - ok_count),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: kv[0])),
        "entries": entries,
        "generated_at": utc_now_iso(),
    }
    out_path = run_dir / "fcf_coverage.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["fcf_coverage_path"] = str(out_path)
    return payload
