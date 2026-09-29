from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso


UNKNOWN = "UNKNOWN"
OK = "OK"

CONSISTENT = "CONSISTENT"
INCONSISTENT = "INCONSISTENT"

ALIGNED = "ALIGNED"
DIFFERENT_DATES = "DIFFERENT_DATES"

SHARES_SOURCE_DEI = "DEI_COVER_PAGE"
SHARES_SOURCE_US_GAAP = "US_GAAP_OUTSTANDING"
SHARES_SOURCE_DOSSIER = "DOSSIER"
SHARES_SOURCE_UNKNOWN = UNKNOWN

FCF_SOURCE_DIRECT = "COMPANYFACTS_DIRECT_FCF"
FCF_SOURCE_BRIDGE = "COMPANYFACTS_CFO_MINUS_CAPEX"
FCF_SOURCE_DOSSIER = "DOSSIER"
FCF_SOURCE_UNKNOWN = UNKNOWN

TOTAL_DEBT_SOURCE_DIRECT = "DIRECT_TAG"
TOTAL_DEBT_SOURCE_COMPONENT_SUM = "COMPONENT_SUM"
TOTAL_DEBT_SOURCE_SINGLE_TAG = "SINGLE_TAG_FALLBACK"
TOTAL_DEBT_SOURCE_ESTIMATED_ZERO = "ESTIMATED_ZERO"
TOTAL_DEBT_SOURCE_UNKNOWN = UNKNOWN

CASH_SOURCE_DIRECT = "DIRECT_TAG"
CASH_SOURCE_UNKNOWN = UNKNOWN

FLAG_SHARES_UNKNOWN = "SHARES_UNKNOWN"
FLAG_FCF_UNKNOWN = "FCF_UNKNOWN"
FLAG_NET_DEBT_UNKNOWN = "NET_DEBT_UNKNOWN"
FLAG_TOTAL_DEBT_UNKNOWN = "TOTAL_DEBT_UNKNOWN"
FLAG_CASH_UNKNOWN = "CASH_UNKNOWN"
FLAG_MARKET_CAP_FORMULA_MISMATCH = "MARKET_CAP_FORMULA_MISMATCH"
FLAG_ENTERPRISE_VALUE_FORMULA_MISMATCH = "ENTERPRISE_VALUE_FORMULA_MISMATCH"
FLAG_FCF_BRIDGE_MISMATCH = "FCF_BRIDGE_MISMATCH"
FLAG_NET_DEBT_BRIDGE_MISMATCH = "NET_DEBT_BRIDGE_MISMATCH"
FLAG_DEBT_CASH_DATE_MISMATCH = "DEBT_CASH_DATE_MISMATCH"

DIRECT_TOTAL_DEBT_TAGS = {
    "LongTermDebtAndCapitalLeaseObligations",
    "DebtAndCapitalLeaseObligations",
    "DebtLongtermAndShorttermCombinedAmount",
    "DebtInstrumentCarryingAmount",
    "Debt",
}

DEFAULT_THRESHOLDS = {
    "shares_unknown_rate_max": 0.15,
    "fcf_unknown_rate_max": 0.20,
    "net_debt_unknown_rate_max": 0.20,
    "debt_cash_date_mismatch_rate_max": 0.10,
    "market_cap_formula_mismatch_max": 0,
    "enterprise_value_formula_mismatch_max": 0,
    "fcf_bridge_mismatch_max": 0,
    "net_debt_bridge_mismatch_max": 0,
}


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _markdown_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_num(value: Any) -> float | str:
    return float(value) if _is_num(value) else UNKNOWN


def _dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _input_payload(score_row: dict[str, Any], key: str) -> dict[str, Any]:
    inputs_used = score_row.get("inputs_used") if isinstance(score_row.get("inputs_used"), dict) else {}
    payload = inputs_used.get(key) if isinstance(inputs_used, dict) else {}
    return payload if isinstance(payload, dict) else {}


def _input_value(score_row: dict[str, Any], key: str) -> float | str:
    return _to_num(_input_payload(score_row, key).get("value"))


def _input_refs(score_row: dict[str, Any], key: str) -> list[str]:
    payload = _input_payload(score_row, key)
    return [str(ref) for ref in (payload.get("derived_from") or []) if str(ref).strip()]


def _source_class_for_shares(score_row: dict[str, Any]) -> str:
    refs = _input_refs(score_row, "shares_outstanding")
    if any("companyfacts.dei.EntityCommonStockSharesOutstanding[" in ref for ref in refs):
        return SHARES_SOURCE_DEI
    if any(
        token in ref
        for ref in refs
        for token in (
            "companyfacts.us-gaap.CommonStockSharesOutstanding[",
            "companyfacts.us-gaap.CommonStockOtherSharesOutstanding[",
        )
    ):
        return SHARES_SOURCE_US_GAAP
    if any("dossiers." in ref or "dossier." in ref for ref in refs):
        return SHARES_SOURCE_DOSSIER
    return SHARES_SOURCE_UNKNOWN


def _source_class_for_fcf(score_row: dict[str, Any]) -> str:
    refs = _input_refs(score_row, "fcf_value")
    if any("companyfacts.us-gaap.FreeCashFlow[" in ref for ref in refs):
        return FCF_SOURCE_DIRECT
    if any("companyfacts.us-gaap.NetCashProvidedByUsedInOperatingActivities[" in ref for ref in refs) and any(
        token in ref
        for ref in refs
        for token in (
            "companyfacts.us-gaap.PaymentsToAcquirePropertyPlantAndEquipment[",
            "companyfacts.us-gaap.PaymentsToAcquireProductiveAssets[",
            "companyfacts.us-gaap.CapitalExpendituresIncurringObligation[",
            "companyfacts.us-gaap.PaymentsForCapitalImprovements[",
            "companyfacts.us-gaap.PaymentsToAcquirePremisesAndEquipment[",
            "companyfacts.us-gaap.CapitalExpendituresIncurredButNotYetPaid[",
        )
    ):
        return FCF_SOURCE_BRIDGE
    if any("dossiers." in ref or "dossier." in ref for ref in refs):
        return FCF_SOURCE_DOSSIER
    return FCF_SOURCE_UNKNOWN


def _source_class_for_total_debt(net_debt_payload: dict[str, Any]) -> str:
    total_debt = net_debt_payload.get("total_debt") if isinstance(net_debt_payload.get("total_debt"), dict) else {}
    if not isinstance(total_debt, dict) or not _is_num(total_debt.get("value")):
        return TOTAL_DEBT_SOURCE_UNKNOWN
    if bool(total_debt.get("estimated_zero")) or str(total_debt.get("tag") or "") == "EstimatedZeroDebtNoDebtTags":
        return TOTAL_DEBT_SOURCE_ESTIMATED_ZERO
    tag = str(total_debt.get("tag") or "")
    if tag == "DebtCurrent_plus_LongTermDebtNoncurrent":
        return TOTAL_DEBT_SOURCE_COMPONENT_SUM
    if tag in DIRECT_TOTAL_DEBT_TAGS:
        return TOTAL_DEBT_SOURCE_DIRECT
    if tag:
        return TOTAL_DEBT_SOURCE_SINGLE_TAG
    return TOTAL_DEBT_SOURCE_UNKNOWN


def _source_class_for_cash(net_debt_payload: dict[str, Any]) -> str:
    cash = net_debt_payload.get("cash_equivalents") if isinstance(net_debt_payload.get("cash_equivalents"), dict) else {}
    if isinstance(cash, dict) and _is_num(cash.get("value")) and str(cash.get("tag") or "").strip():
        return CASH_SOURCE_DIRECT
    return CASH_SOURCE_UNKNOWN


def _consistency_status(
    actual: Any,
    expected: Any,
    *,
    rel_tol: float = 0.01,
    abs_tol: float = 0.5,
) -> str:
    if not (_is_num(actual) and _is_num(expected)):
        return UNKNOWN
    actual_num = float(actual)
    expected_num = float(expected)
    allowed = max(float(abs_tol), float(rel_tol) * max(abs(actual_num), abs(expected_num), 1.0))
    return CONSISTENT if abs(actual_num - expected_num) <= allowed else INCONSISTENT


def _date_alignment_status(left: Any, right: Any) -> str:
    left_token = str(left or "").strip()
    right_token = str(right or "").strip()
    if not left_token or not right_token:
        return UNKNOWN
    return ALIGNED if left_token == right_token else DIFFERENT_DATES


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if int(denominator) > 0 else 0.0


def _build_row(*, ticker: str, score_row: dict[str, Any], net_debt_payload: dict[str, Any]) -> dict[str, Any]:
    price_value = _input_value(score_row, "current_price")
    shares_value = _input_value(score_row, "shares_outstanding")
    market_cap_value = _input_value(score_row, "market_cap")
    cfo_value = _input_value(score_row, "cfo_value")
    capex_value = _input_value(score_row, "capex_value")
    fcf_value = _input_value(score_row, "fcf_value")
    net_debt_value = _input_value(score_row, "net_debt_proxy")
    total_debt_value = _to_num(
        (net_debt_payload.get("total_debt") or {}).get("value")
        if isinstance(net_debt_payload.get("total_debt"), dict)
        else UNKNOWN
    )
    cash_value = _to_num(
        (net_debt_payload.get("cash_equivalents") or {}).get("value")
        if isinstance(net_debt_payload.get("cash_equivalents"), dict)
        else UNKNOWN
    )
    enterprise_value = _input_value(score_row, "enterprise_value")

    market_cap_consistency = _consistency_status(market_cap_value, float(price_value) * float(shares_value) if _is_num(price_value) and _is_num(shares_value) else UNKNOWN)
    enterprise_value_consistency = _consistency_status(
        enterprise_value,
        float(market_cap_value) + float(net_debt_value) if _is_num(market_cap_value) and _is_num(net_debt_value) else UNKNOWN,
    )
    fcf_bridge_consistency = _consistency_status(
        fcf_value,
        float(cfo_value) - float(capex_value) if _is_num(cfo_value) and _is_num(capex_value) else UNKNOWN,
        abs_tol=0.1,
    )
    net_debt_bridge_consistency = _consistency_status(
        net_debt_value,
        float(total_debt_value) - float(cash_value) if _is_num(total_debt_value) and _is_num(cash_value) else UNKNOWN,
        abs_tol=0.1,
    )
    debt_cash_date_alignment = _date_alignment_status(
        (net_debt_payload.get("total_debt") or {}).get("date")
        if isinstance(net_debt_payload.get("total_debt"), dict)
        else None,
        (net_debt_payload.get("cash_equivalents") or {}).get("date")
        if isinstance(net_debt_payload.get("cash_equivalents"), dict)
        else None,
    )

    flags: list[str] = []
    if str(score_row.get("shares_status") or UNKNOWN).upper() != OK or not _is_num(shares_value):
        flags.append(FLAG_SHARES_UNKNOWN)
    if str(score_row.get("fcf_status") or UNKNOWN).upper() != OK or not _is_num(fcf_value):
        flags.append(FLAG_FCF_UNKNOWN)
    if not _is_num(net_debt_value):
        flags.append(FLAG_NET_DEBT_UNKNOWN)
    if not _is_num(total_debt_value):
        flags.append(FLAG_TOTAL_DEBT_UNKNOWN)
    if not _is_num(cash_value):
        flags.append(FLAG_CASH_UNKNOWN)
    if market_cap_consistency == INCONSISTENT:
        flags.append(FLAG_MARKET_CAP_FORMULA_MISMATCH)
    if enterprise_value_consistency == INCONSISTENT:
        flags.append(FLAG_ENTERPRISE_VALUE_FORMULA_MISMATCH)
    if fcf_bridge_consistency == INCONSISTENT:
        flags.append(FLAG_FCF_BRIDGE_MISMATCH)
    if net_debt_bridge_consistency == INCONSISTENT:
        flags.append(FLAG_NET_DEBT_BRIDGE_MISMATCH)
    if debt_cash_date_alignment == DIFFERENT_DATES:
        flags.append(FLAG_DEBT_CASH_DATE_MISMATCH)

    derived = _dedupe(
        [str(ref) for ref in (score_row.get("derived_from") or []) if str(ref).strip()]
        + [str(ref) for ref in (net_debt_payload.get("derived_from") or []) if str(ref).strip()]
    )

    return {
        "ticker": ticker,
        "shares_status": str(score_row.get("shares_status") or UNKNOWN).upper(),
        "fcf_status": str(score_row.get("fcf_status") or UNKNOWN).upper(),
        "facts_status": str(score_row.get("facts_status") or UNKNOWN).upper(),
        "shares_source_class": _source_class_for_shares(score_row),
        "fcf_source_class": _source_class_for_fcf(score_row),
        "total_debt_source_class": _source_class_for_total_debt(net_debt_payload),
        "cash_source_class": _source_class_for_cash(net_debt_payload),
        "market_cap_formula_consistency": market_cap_consistency,
        "enterprise_value_formula_consistency": enterprise_value_consistency,
        "fcf_bridge_consistency": fcf_bridge_consistency,
        "net_debt_bridge_consistency": net_debt_bridge_consistency,
        "debt_cash_date_alignment": debt_cash_date_alignment,
        "price_value": price_value,
        "shares_value": shares_value,
        "market_cap_value": market_cap_value,
        "cfo_value": cfo_value,
        "capex_value": capex_value,
        "fcf_value": fcf_value,
        "total_debt_value": total_debt_value,
        "cash_value": cash_value,
        "net_debt_value": net_debt_value,
        "enterprise_value": enterprise_value,
        "total_debt_tag": str((net_debt_payload.get("total_debt") or {}).get("tag") or "")
        if isinstance(net_debt_payload.get("total_debt"), dict)
        else "",
        "cash_tag": str((net_debt_payload.get("cash_equivalents") or {}).get("tag") or "")
        if isinstance(net_debt_payload.get("cash_equivalents"), dict)
        else "",
        "total_debt_date": (net_debt_payload.get("total_debt") or {}).get("date")
        if isinstance(net_debt_payload.get("total_debt"), dict)
        else None,
        "cash_date": (net_debt_payload.get("cash_equivalents") or {}).get("date")
        if isinstance(net_debt_payload.get("cash_equivalents"), dict)
        else None,
        "flag_count": len(flags),
        "flags": flags,
        "derived_from": derived,
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Fundamental Regression Analytics",
        "",
        f"- run_id: `{payload.get('run_id')}`",
        f"- as_of_date: `{payload.get('as_of_date')}`",
        f"- ticker_count: `{payload.get('ticker_count')}`",
        "",
        "## Source Mix",
        "",
        f"- shares_source_counts: `{payload.get('counts_by_shares_source_class')}`",
        f"- fcf_source_counts: `{payload.get('counts_by_fcf_source_class')}`",
        f"- total_debt_source_counts: `{payload.get('counts_by_total_debt_source_class')}`",
        f"- cash_source_counts: `{payload.get('counts_by_cash_source_class')}`",
        "",
        "## Unknowns",
        "",
        f"- unknown_input_counts: `{payload.get('unknown_input_counts')}`",
        "",
        "## Mismatches",
        "",
        f"- formula_mismatch_counts: `{payload.get('formula_mismatch_counts')}`",
        f"- debt_cash_date_alignment_counts: `{payload.get('debt_cash_date_alignment_counts')}`",
        "",
        "## Threshold Breaches",
        "",
    ]
    breaches = [row for row in (payload.get("threshold_breaches") or []) if isinstance(row, dict)]
    if not breaches:
        lines.append("- none")
    else:
        for breach in breaches:
            lines.append(
                f"- `{breach.get('metric')}` actual=`{breach.get('actual')}` threshold=`{breach.get('threshold')}`"
            )
    lines.extend(["", "## Top Flagged Tickers", ""])
    top_rows = [row for row in (payload.get("top_10_flagged_tickers") or []) if isinstance(row, dict)]
    if not top_rows:
        lines.append("- none")
    else:
        for row in top_rows:
            lines.append(f"- **{row.get('ticker')}**: flags=`{row.get('flags')}`")
    lines.append("")
    return "\n".join(lines)


def write_fundamental_regression_analytics_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    output_path: Path,
    markdown_path: Path | None = None,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    net_debt_resolved_by_ticker: dict[str, dict[str, Any]] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    del cfg
    scoreboard_rows = [row for row in (scoreboard_rows or []) if isinstance(row, dict)]
    score_by_ticker = {
        str(row.get("ticker") or "").strip().upper(): row
        for row in scoreboard_rows
        if str(row.get("ticker") or "").strip()
    }
    net_debt_resolved_by_ticker = {
        str(ticker or "").strip().upper(): payload
        for ticker, payload in (net_debt_resolved_by_ticker or {}).items()
        if str(ticker or "").strip() and isinstance(payload, dict)
    }

    rows: list[dict[str, Any]] = []
    for ticker in sorted({str(token or "").strip().upper() for token in tickers if str(token or "").strip()}):
        rows.append(
            _build_row(
                ticker=ticker,
                score_row=score_by_ticker.get(ticker, {}),
                net_debt_payload=net_debt_resolved_by_ticker.get(ticker, {}),
            )
        )

    def _count_by(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            token = str(row.get(key) or UNKNOWN)
            counts[token] = counts.get(token, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (str(item[0]), int(item[1]))))

    shares_unknown_count = len([row for row in rows if FLAG_SHARES_UNKNOWN in (row.get("flags") or [])])
    fcf_unknown_count = len([row for row in rows if FLAG_FCF_UNKNOWN in (row.get("flags") or [])])
    net_debt_unknown_count = len([row for row in rows if FLAG_NET_DEBT_UNKNOWN in (row.get("flags") or [])])
    total_debt_unknown_count = len([row for row in rows if FLAG_TOTAL_DEBT_UNKNOWN in (row.get("flags") or [])])
    cash_unknown_count = len([row for row in rows if FLAG_CASH_UNKNOWN in (row.get("flags") or [])])

    formula_mismatch_counts = {
        FLAG_MARKET_CAP_FORMULA_MISMATCH: len([row for row in rows if FLAG_MARKET_CAP_FORMULA_MISMATCH in (row.get("flags") or [])]),
        FLAG_ENTERPRISE_VALUE_FORMULA_MISMATCH: len([row for row in rows if FLAG_ENTERPRISE_VALUE_FORMULA_MISMATCH in (row.get("flags") or [])]),
        FLAG_FCF_BRIDGE_MISMATCH: len([row for row in rows if FLAG_FCF_BRIDGE_MISMATCH in (row.get("flags") or [])]),
        FLAG_NET_DEBT_BRIDGE_MISMATCH: len([row for row in rows if FLAG_NET_DEBT_BRIDGE_MISMATCH in (row.get("flags") or [])]),
    }
    debt_cash_date_alignment_counts = _count_by("debt_cash_date_alignment")

    threshold_breaches: list[dict[str, Any]] = []
    ticker_count = len(rows)
    rate_checks = [
        ("shares_unknown_rate", _rate(shares_unknown_count, ticker_count), DEFAULT_THRESHOLDS["shares_unknown_rate_max"]),
        ("fcf_unknown_rate", _rate(fcf_unknown_count, ticker_count), DEFAULT_THRESHOLDS["fcf_unknown_rate_max"]),
        ("net_debt_unknown_rate", _rate(net_debt_unknown_count, ticker_count), DEFAULT_THRESHOLDS["net_debt_unknown_rate_max"]),
        (
            "debt_cash_date_mismatch_rate",
            _rate(int(debt_cash_date_alignment_counts.get(DIFFERENT_DATES, 0)), ticker_count),
            DEFAULT_THRESHOLDS["debt_cash_date_mismatch_rate_max"],
        ),
    ]
    for metric, actual, threshold in rate_checks:
        if float(actual) > float(threshold):
            threshold_breaches.append({"metric": metric, "actual": round(float(actual), 4), "threshold": float(threshold)})
    for metric, count in (
        ("market_cap_formula_mismatch_count", int(formula_mismatch_counts[FLAG_MARKET_CAP_FORMULA_MISMATCH])),
        ("enterprise_value_formula_mismatch_count", int(formula_mismatch_counts[FLAG_ENTERPRISE_VALUE_FORMULA_MISMATCH])),
        ("fcf_bridge_mismatch_count", int(formula_mismatch_counts[FLAG_FCF_BRIDGE_MISMATCH])),
        ("net_debt_bridge_mismatch_count", int(formula_mismatch_counts[FLAG_NET_DEBT_BRIDGE_MISMATCH])),
    ):
        if count > 0:
            threshold_breaches.append({"metric": metric, "actual": int(count), "threshold": 0})

    flag_reason_counts: dict[str, int] = {}
    for row in rows:
        for flag in row.get("flags") or []:
            token = str(flag or "").strip()
            if not token:
                continue
            flag_reason_counts[token] = flag_reason_counts.get(token, 0) + 1

    top_flagged = sorted(
        rows,
        key=lambda row: (-int(row.get("flag_count") or 0), str(row.get("ticker") or "")),
    )

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": ticker_count,
        "counts_by_shares_source_class": _count_by("shares_source_class"),
        "counts_by_fcf_source_class": _count_by("fcf_source_class"),
        "counts_by_total_debt_source_class": _count_by("total_debt_source_class"),
        "counts_by_cash_source_class": _count_by("cash_source_class"),
        "unknown_input_counts": {
            "shares_unknown_count": shares_unknown_count,
            "fcf_unknown_count": fcf_unknown_count,
            "net_debt_unknown_count": net_debt_unknown_count,
            "total_debt_unknown_count": total_debt_unknown_count,
            "cash_unknown_count": cash_unknown_count,
        },
        "formula_mismatch_counts": formula_mismatch_counts,
        "debt_cash_date_alignment_counts": debt_cash_date_alignment_counts,
        "flag_reason_counts": dict(sorted(flag_reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))),
        "thresholds_effective": dict(DEFAULT_THRESHOLDS),
        "threshold_breaches": threshold_breaches,
        "top_10_flagged_tickers": [
            {
                "ticker": str(row.get("ticker") or ""),
                "flag_count": int(row.get("flag_count") or 0),
                "flags": [str(flag) for flag in (row.get("flags") or []) if str(flag).strip()],
                "shares_source_class": str(row.get("shares_source_class") or UNKNOWN),
                "fcf_source_class": str(row.get("fcf_source_class") or UNKNOWN),
                "total_debt_source_class": str(row.get("total_debt_source_class") or UNKNOWN),
            }
            for row in top_flagged[:10]
            if int(row.get("flag_count") or 0) > 0
        ],
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    payload["fundamental_regression_analytics_path"] = str(output_path)
    if markdown_path is not None:
        _markdown_write(markdown_path, _markdown_report(payload))
        payload["fundamental_regression_analytics_markdown_path"] = str(markdown_path)
    return payload


def _fundamental_regression_analytics_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "fundamental_regression_analytics.json",
        cfg.sectors_dir / run_id / "fundamental_regression_analytics.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_fundamental_regression_analytics(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    path = _fundamental_regression_analytics_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "fundamental_regression_analytics_path": str(path) if path is not None else "",
        }
    payload = _safe_json(path)
    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or 0),
        "counts_by_shares_source_class": payload.get("counts_by_shares_source_class")
        if isinstance(payload.get("counts_by_shares_source_class"), dict)
        else {},
        "counts_by_fcf_source_class": payload.get("counts_by_fcf_source_class")
        if isinstance(payload.get("counts_by_fcf_source_class"), dict)
        else {},
        "counts_by_total_debt_source_class": payload.get("counts_by_total_debt_source_class")
        if isinstance(payload.get("counts_by_total_debt_source_class"), dict)
        else {},
        "counts_by_cash_source_class": payload.get("counts_by_cash_source_class")
        if isinstance(payload.get("counts_by_cash_source_class"), dict)
        else {},
        "unknown_input_counts": payload.get("unknown_input_counts")
        if isinstance(payload.get("unknown_input_counts"), dict)
        else {},
        "formula_mismatch_counts": payload.get("formula_mismatch_counts")
        if isinstance(payload.get("formula_mismatch_counts"), dict)
        else {},
        "debt_cash_date_alignment_counts": payload.get("debt_cash_date_alignment_counts")
        if isinstance(payload.get("debt_cash_date_alignment_counts"), dict)
        else {},
        "threshold_breaches": [
            row for row in (payload.get("threshold_breaches") or []) if isinstance(row, dict)
        ],
        "top_10_flagged_tickers": [
            row for row in (payload.get("top_10_flagged_tickers") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "fundamental_regression_analytics_path": str(path),
    }
