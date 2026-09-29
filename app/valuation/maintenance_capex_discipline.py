from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso


UNKNOWN = "UNKNOWN"

LOW_ASSET_INTENSITY = "LOW_ASSET_INTENSITY"
MODERATE_ASSET_INTENSITY = "MODERATE_ASSET_INTENSITY"
HIGH_ASSET_INTENSITY = "HIGH_ASSET_INTENSITY"
ASSET_INTENSITY_UNKNOWN = "ASSET_INTENSITY_UNKNOWN"

HIGH_MAINTENANCE_CAPEX_CREDIBILITY = "HIGH_MAINTENANCE_CAPEX_CREDIBILITY"
MODERATE_MAINTENANCE_CAPEX_CREDIBILITY = "MODERATE_MAINTENANCE_CAPEX_CREDIBILITY"
LOW_MAINTENANCE_CAPEX_CREDIBILITY = "LOW_MAINTENANCE_CAPEX_CREDIBILITY"
MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN = "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"

OWNER_EARNINGS_SUPPORTIVE = "OWNER_EARNINGS_SUPPORTIVE"
OWNER_EARNINGS_MIXED = "OWNER_EARNINGS_MIXED"
OWNER_EARNINGS_HEADWIND = "OWNER_EARNINGS_HEADWIND"
OWNER_EARNINGS_UNCLEAR = "OWNER_EARNINGS_UNCLEAR"

SIG_CAPEX_LIGHT_SCALING_PRESENT = "CAPEX_LIGHT_SCALING_PRESENT"
SIG_LOW_CAPEX_TO_REVENUE = "LOW_CAPEX_TO_REVENUE"
SIG_LOW_CAPEX_TO_CFO = "LOW_CAPEX_TO_CFO"
SIG_OWNER_EARNINGS_PROXY_APPEARS_REASONABLE = "OWNER_EARNINGS_PROXY_APPEARS_REASONABLE"
SIG_DEPRECIATION_AND_CAPEX_BROADLY_ALIGNED = "DEPRECIATION_AND_CAPEX_BROADLY_ALIGNED"
SIG_MAINTENANCE_BURDEN_NOT_DOMINANT = "MAINTENANCE_BURDEN_NOT_DOMINANT"

SIG_HIGH_CAPEX_TO_REVENUE = "HIGH_CAPEX_TO_REVENUE"
SIG_HIGH_CAPEX_TO_CFO = "HIGH_CAPEX_TO_CFO"
SIG_HIGH_CAPEX_TO_OWNER_EARNINGS = "HIGH_CAPEX_TO_OWNER_EARNINGS"
SIG_MAINTENANCE_BURDEN_MAY_BE_UNDERSTATED = "MAINTENANCE_BURDEN_MAY_BE_UNDERSTATED"
SIG_CAPEX_VOLATILITY_HEADWIND = "CAPEX_VOLATILITY_HEADWIND"
SIG_HEAVY_ASSET_REPLACEMENT_BURDEN = "HEAVY_ASSET_REPLACEMENT_BURDEN"
SIG_OWNER_EARNINGS_DENOMINATOR_TOO_FLATTERING = "OWNER_EARNINGS_DENOMINATOR_TOO_FLATTERING"
SIG_MAINTENANCE_CAPEX_EVIDENCE_THIN = "MAINTENANCE_CAPEX_EVIDENCE_THIN"

REASON_LOW_ASSET_INTENSITY_SUPPORT = "LOW_ASSET_INTENSITY_SUPPORT"
REASON_HIGH_ASSET_INTENSITY_HEADWIND = "HIGH_ASSET_INTENSITY_HEADWIND"
REASON_LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND = "LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND"
REASON_OWNER_EARNINGS_DENOMINATOR_SENSITIVE = "OWNER_EARNINGS_DENOMINATOR_SENSITIVE"
REASON_MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN = "MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN"
REASON_MISSING_MAINTENANCE_CAPEX_INPUTS = "MISSING_MAINTENANCE_CAPEX_INPUTS"
REASON_MAINTENANCE_CAPEX_MIXED = "MAINTENANCE_CAPEX_MIXED"
REASON_CAPEX_BURDEN_NOT_DOMINANT = "CAPEX_BURDEN_NOT_DOMINANT"
REASON_HEAVY_CAPEX_BURDEN = "HEAVY_CAPEX_BURDEN"
REASON_CAPEX_PROXY_APPEARS_REASONABLE = "CAPEX_PROXY_APPEARS_REASONABLE"
REASON_CAPEX_PROXY_POSSIBLY_TOO_FLATTERING = "CAPEX_PROXY_POSSIBLY_TOO_FLATTERING"
REASON_CAPEX_VOLATILITY = "CAPEX_VOLATILITY"

_ASSET_INTENSITY_ORDER = {
    LOW_ASSET_INTENSITY: 0,
    MODERATE_ASSET_INTENSITY: 1,
    ASSET_INTENSITY_UNKNOWN: 2,
    HIGH_ASSET_INTENSITY: 3,
}
_CREDIBILITY_ORDER = {
    HIGH_MAINTENANCE_CAPEX_CREDIBILITY: 0,
    MODERATE_MAINTENANCE_CAPEX_CREDIBILITY: 1,
    MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN: 2,
    LOW_MAINTENANCE_CAPEX_CREDIBILITY: 3,
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


def _coalesce_num(*values: Any) -> float | str:
    for value in values:
        if _is_num(value):
            return float(value)
    return UNKNOWN


def _series_rows(fundamentals: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in (fundamentals.get("rows") or []) if isinstance(row, dict)]
    rows.sort(key=lambda row: int(row.get("year") or 0))
    return rows


def _rows_from_owner_payload(owner_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in [row for row in (owner_payload.get("series") or []) if isinstance(row, dict)]:
        year = int(raw.get("year") or 0)
        if year <= 0:
            continue
        rows.append(
            {
                "year": year,
                "revenue": raw.get("revenue", UNKNOWN),
                "cfo": raw.get("cfo", UNKNOWN),
                "capex": raw.get("capex", UNKNOWN),
                "maintenance_capex_proxy": raw.get("maintenance_capex_proxy", UNKNOWN),
                "owner_earnings": raw.get("owner_earnings", UNKNOWN),
            }
        )
    rows.sort(key=lambda row: int(row.get("year") or 0))
    return rows


def _metric_refs(row: dict[str, Any], metric: str) -> list[str]:
    year = int(row.get("year") or 0)
    if year <= 0:
        return []
    return [f"fundamentals.rows[{year}].{metric}"]


def _median_ratio(
    rows: list[dict[str, Any]],
    numerator_key: str,
    denominator_key: str,
    *,
    allow_negative_denominator: bool = False,
    window: int = 5,
) -> tuple[float | str, list[str], int]:
    numeric: list[tuple[float, list[str]]] = []
    for row in rows:
        numerator = row.get(numerator_key, UNKNOWN)
        denominator = row.get(denominator_key, UNKNOWN)
        if not _is_num(numerator) or not _is_num(denominator):
            continue
        denominator_value = float(denominator)
        if allow_negative_denominator:
            if denominator_value == 0.0:
                continue
        elif denominator_value <= 0.0:
            continue
        numeric.append(
            (
                abs(float(numerator)) / abs(denominator_value),
                _dedupe(_metric_refs(row, numerator_key) + _metric_refs(row, denominator_key)),
            )
        )
    numeric = numeric[-max(1, int(window)) :]
    if not numeric:
        return UNKNOWN, [], 0
    return (
        float(median([value for value, _refs in numeric])),
        _dedupe([ref for _value, refs in numeric for ref in refs]),
        len(numeric),
    )


def _capex_volatility(rows: list[dict[str, Any]], *, window: int = 5) -> tuple[float | str, list[str], int]:
    numeric_rows = [
        row
        for row in rows
        if _is_num(row.get("capex")) and abs(float(row.get("capex"))) > 0.0
    ][-max(1, int(window)) :]
    if len(numeric_rows) < 2:
        return UNKNOWN, [], len(numeric_rows)
    values = [abs(float(row["capex"])) for row in numeric_rows]
    median_capex = median(values)
    if median_capex <= 0.0:
        return UNKNOWN, _dedupe([ref for row in numeric_rows for ref in _metric_refs(row, "capex")]), len(values)
    volatility = (max(values) - min(values)) / float(median_capex)
    refs = _dedupe([ref for row in numeric_rows for ref in _metric_refs(row, "capex")])
    return float(volatility), refs, len(values)


def _latest_metric(rows: list[dict[str, Any]], *keys: str) -> tuple[float | str, list[str]]:
    for row in sorted(rows, key=lambda item: int(item.get("year") or 0), reverse=True):
        for key in keys:
            value = row.get(key, UNKNOWN)
            if _is_num(value):
                return float(value), _metric_refs(row, key)
    return UNKNOWN, []


def _collect_derived_from(*payloads: dict[str, Any], row_refs: list[Any] | None = None) -> list[str]:
    refs: list[Any] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        refs.extend(payload.get("derived_from") or [])
        claims = payload.get("claims") if isinstance(payload.get("claims"), dict) else {}
        for claim in claims.values():
            if isinstance(claim, dict):
                refs.extend(claim.get("derived_from") or [])
    refs.extend(row_refs or [])
    return _dedupe(refs)


def _claim(*, value: Any, refs: list[Any], reason_code: str) -> dict[str, Any]:
    derived = _dedupe(refs)
    token = str(value or "").strip()
    if token.upper() in {ASSET_INTENSITY_UNKNOWN, MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN}:
        return {
            "value": value, "status": UNKNOWN, "reason_code": reason_code, "derived_from": derived
        }
    if token and token.upper() != UNKNOWN:
        return {"value": value, "status": "OK", "reason_code": reason_code, "derived_from": derived}
    return {
        "value": UNKNOWN,
        "status": UNKNOWN,
        "reason_code": str(reason_code or UNKNOWN),
        "derived_from": derived,
    }


def _summary_for_profile(
    *,
    asset_intensity_class: str,
    credibility_class: str,
    headwinds: list[str],
    supports: list[str],
) -> str:
    if (
        asset_intensity_class == LOW_ASSET_INTENSITY
        and credibility_class == HIGH_MAINTENANCE_CAPEX_CREDIBILITY
    ):
        return "owner earnings appear reasonably supported by low to manageable sustaining capital demands"
    if (
        asset_intensity_class == HIGH_ASSET_INTENSITY
        or credibility_class == LOW_MAINTENANCE_CAPEX_CREDIBILITY
    ):
        if SIG_OWNER_EARNINGS_DENOMINATOR_TOO_FLATTERING in headwinds:
            return "sustaining capital needs may be heavy enough that the current owner-earnings denominator looks too flattering"
        return "capital burden appears meaningful enough to make owner earnings less trustworthy than headline cash generation suggests"
    if (
        asset_intensity_class == MODERATE_ASSET_INTENSITY
        or credibility_class == MODERATE_MAINTENANCE_CAPEX_CREDIBILITY
    ):
        return "capital burden is meaningful but manageable; owner earnings remain usable with caution"
    if headwinds:
        return "maintenance-capex discipline is unclear, with some burden signals but not enough support for a clean judgment"
    if supports:
        return "some capex-light support exists, but evidence remains too thin to trust the maintenance-capex proxy fully"
    return "evidence too thin to judge whether owner earnings are being flattered by sustaining capital needs"


def compute_maintenance_capex_discipline(
    ticker: str,
    as_of_date: str,
    *,
    fundamentals: dict[str, Any] | None = None,
    owner_payload: dict[str, Any] | None = None,
    owner_quality_payload: dict[str, Any] | None = None,
    accounting_quality_payload: dict[str, Any] | None = None,
    reinvestment_efficiency_payload: dict[str, Any] | None = None,
    returns_persistence_payload: dict[str, Any] | None = None,
    intangible_payload: dict[str, Any] | None = None,
    price_status: Any = UNKNOWN,
    facts_status: Any = UNKNOWN,
    shares_status: Any = UNKNOWN,
    fcf_status: Any = UNKNOWN,
    row_derived_from: list[Any] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    del cfg

    ticker_norm = str(ticker or "").strip().upper()
    fundamentals = fundamentals if isinstance(fundamentals, dict) else {}
    owner_payload = owner_payload if isinstance(owner_payload, dict) else {}
    owner_quality_payload = owner_quality_payload if isinstance(owner_quality_payload, dict) else {}
    accounting_quality_payload = (
        accounting_quality_payload if isinstance(accounting_quality_payload, dict) else {}
    )
    reinvestment_efficiency_payload = (
        reinvestment_efficiency_payload
        if isinstance(reinvestment_efficiency_payload, dict)
        else {}
    )
    returns_persistence_payload = (
        returns_persistence_payload if isinstance(returns_persistence_payload, dict) else {}
    )
    intangible_payload = intangible_payload if isinstance(intangible_payload, dict) else {}

    rows = _series_rows(fundamentals)
    if not rows:
        rows = _rows_from_owner_payload(owner_payload)

    capex_to_revenue, capex_to_revenue_refs, revenue_points = _median_ratio(rows, "capex", "revenue")
    capex_to_cfo, capex_to_cfo_refs, cfo_points = _median_ratio(rows, "capex", "cfo")
    capex_to_owner, capex_to_owner_refs, owner_points = _median_ratio(rows, "capex", "owner_earnings")
    maintenance_proxy_to_cfo, maintenance_proxy_to_cfo_refs, maintenance_points = _median_ratio(
        rows,
        "maintenance_capex_proxy",
        "cfo",
    )
    depreciation_to_capex, depreciation_refs, depreciation_points = _median_ratio(
        rows,
        "depreciation",
        "capex",
        allow_negative_denominator=True,
    )
    capex_volatility, capex_volatility_refs, capex_volatility_points = _capex_volatility(rows)
    latest_capex, latest_capex_refs = _latest_metric(rows, "capex")
    latest_cfo, latest_cfo_refs = _latest_metric(rows, "cfo")
    latest_owner, latest_owner_refs = _latest_metric(rows, "owner_earnings")

    maintenance_ratio = _coalesce_num(
        owner_payload.get("maintenance_capex_ratio"),
        fundamentals.get("maintenance_capex_ratio"),
    )
    accounting_class = str(
        accounting_quality_payload.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN"
    ).upper()
    accounting_headwinds = {
        str(code)
        for code in (accounting_quality_payload.get("cash_earnings_headwind_signals") or [])
        if str(code).strip()
    }
    reinvestment_class = str(
        reinvestment_efficiency_payload.get("reinvestment_efficiency_class")
        or "REINVESTMENT_EFFICIENCY_UNKNOWN"
    ).upper()
    reinvestment_headwinds = {
        str(code)
        for code in (reinvestment_efficiency_payload.get("reinvestment_headwind_signals") or [])
        if str(code).strip()
    }
    returns_class = str(
        returns_persistence_payload.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN"
    ).upper()
    optionality_score = intangible_payload.get("balance_sheet_optionality_score", UNKNOWN)

    evidence_count = sum(
        1
        for value in [
            capex_to_revenue,
            capex_to_cfo,
            capex_to_owner,
            maintenance_proxy_to_cfo,
            capex_volatility,
        ]
        if _is_num(value)
    )

    support_signals: list[str] = []
    headwind_signals: list[str] = []
    asset_reason_codes: list[str] = []
    credibility_reason_codes: list[str] = []

    if _is_num(capex_to_revenue):
        if float(capex_to_revenue) <= 0.05:
            support_signals.append(SIG_LOW_CAPEX_TO_REVENUE)
        elif float(capex_to_revenue) >= 0.12:
            headwind_signals.append(SIG_HIGH_CAPEX_TO_REVENUE)
    if _is_num(capex_to_cfo):
        if float(capex_to_cfo) <= 0.30:
            support_signals.append(SIG_LOW_CAPEX_TO_CFO)
        elif float(capex_to_cfo) >= 0.60:
            headwind_signals.extend([SIG_HIGH_CAPEX_TO_CFO, SIG_HEAVY_ASSET_REPLACEMENT_BURDEN])
    if _is_num(capex_to_owner):
        if float(capex_to_owner) <= 0.35:
            support_signals.append(SIG_OWNER_EARNINGS_PROXY_APPEARS_REASONABLE)
        elif float(capex_to_owner) >= 0.75:
            headwind_signals.extend(
                [SIG_HIGH_CAPEX_TO_OWNER_EARNINGS, SIG_OWNER_EARNINGS_DENOMINATOR_TOO_FLATTERING]
            )
    if _is_num(maintenance_proxy_to_cfo):
        if float(maintenance_proxy_to_cfo) <= 0.20:
            support_signals.append(SIG_MAINTENANCE_BURDEN_NOT_DOMINANT)
        elif float(maintenance_proxy_to_cfo) >= 0.45:
            headwind_signals.append(SIG_MAINTENANCE_BURDEN_MAY_BE_UNDERSTATED)
    if _is_num(depreciation_to_capex) and 0.70 <= float(depreciation_to_capex) <= 1.30:
        support_signals.append(SIG_DEPRECIATION_AND_CAPEX_BROADLY_ALIGNED)
    if _is_num(capex_volatility) and float(capex_volatility) >= 0.75:
        headwind_signals.append(SIG_CAPEX_VOLATILITY_HEADWIND)

    if {
        SIG_LOW_CAPEX_TO_REVENUE,
        SIG_LOW_CAPEX_TO_CFO,
        SIG_MAINTENANCE_BURDEN_NOT_DOMINANT,
    } <= set(support_signals):
        support_signals.append(SIG_CAPEX_LIGHT_SCALING_PRESENT)

    if accounting_class == "LOW_ACCOUNTING_QUALITY" and (
        "REPORTED_EARNINGS_NOT_OWNER_RELEVANT" in accounting_headwinds
        or "CASH_EARNINGS_DIVERGENCE" in accounting_headwinds
    ):
        headwind_signals.extend(
            [SIG_MAINTENANCE_BURDEN_MAY_BE_UNDERSTATED, SIG_OWNER_EARNINGS_DENOMINATOR_TOO_FLATTERING]
        )
    if reinvestment_class == "LOW_REINVESTMENT_EFFICIENCY" and {
        "HIGH_CAPEX_BURDEN",
        "CAPITAL_HUNGRY_GROWTH",
    } & reinvestment_headwinds:
        headwind_signals.extend([SIG_HIGH_CAPEX_TO_CFO, SIG_HEAVY_ASSET_REPLACEMENT_BURDEN])
    if returns_class == "LOW_RETURNS_PERSISTENCE":
        headwind_signals.append(SIG_MAINTENANCE_BURDEN_MAY_BE_UNDERSTATED)
    if _is_num(optionality_score) and float(optionality_score) >= 4.0 and _is_num(capex_to_cfo) and float(capex_to_cfo) <= 0.35:
        support_signals.append(SIG_MAINTENANCE_BURDEN_NOT_DOMINANT)

    support_signals = _dedupe(support_signals)
    headwind_signals = _dedupe(headwind_signals)

    if evidence_count == 0:
        headwind_signals.append(SIG_MAINTENANCE_CAPEX_EVIDENCE_THIN)
        asset_intensity_class = ASSET_INTENSITY_UNKNOWN
        credibility_class = MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN
        caution = OWNER_EARNINGS_UNCLEAR
        asset_reason_codes.extend([REASON_MISSING_MAINTENANCE_CAPEX_INPUTS, REASON_MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN])
        credibility_reason_codes.extend([REASON_MISSING_MAINTENANCE_CAPEX_INPUTS, REASON_MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN])
    else:
        asset_headwind_strength = 0
        asset_support_strength = 0
        credibility_headwind_strength = 0
        credibility_support_strength = 0

        if SIG_HIGH_CAPEX_TO_REVENUE in headwind_signals:
            asset_headwind_strength += 2
        if SIG_HIGH_CAPEX_TO_CFO in headwind_signals:
            asset_headwind_strength += 2
            credibility_headwind_strength += 1
        if SIG_HIGH_CAPEX_TO_OWNER_EARNINGS in headwind_signals:
            asset_headwind_strength += 1
            credibility_headwind_strength += 2
        if SIG_HEAVY_ASSET_REPLACEMENT_BURDEN in headwind_signals:
            asset_headwind_strength += 1
            credibility_headwind_strength += 1
        if SIG_CAPEX_VOLATILITY_HEADWIND in headwind_signals:
            credibility_headwind_strength += 1
        if SIG_MAINTENANCE_BURDEN_MAY_BE_UNDERSTATED in headwind_signals:
            credibility_headwind_strength += 2
        if SIG_OWNER_EARNINGS_DENOMINATOR_TOO_FLATTERING in headwind_signals:
            credibility_headwind_strength += 2

        if SIG_LOW_CAPEX_TO_REVENUE in support_signals:
            asset_support_strength += 2
        if SIG_LOW_CAPEX_TO_CFO in support_signals:
            asset_support_strength += 2
            credibility_support_strength += 1
        if SIG_CAPEX_LIGHT_SCALING_PRESENT in support_signals:
            asset_support_strength += 1
        if SIG_MAINTENANCE_BURDEN_NOT_DOMINANT in support_signals:
            asset_support_strength += 1
            credibility_support_strength += 1
        if SIG_OWNER_EARNINGS_PROXY_APPEARS_REASONABLE in support_signals:
            credibility_support_strength += 2
        if SIG_DEPRECIATION_AND_CAPEX_BROADLY_ALIGNED in support_signals:
            credibility_support_strength += 1

        if asset_headwind_strength >= 3:
            asset_intensity_class = HIGH_ASSET_INTENSITY
            asset_reason_codes.append(REASON_HIGH_ASSET_INTENSITY_HEADWIND)
        elif asset_support_strength >= 3 and asset_headwind_strength == 0:
            asset_intensity_class = LOW_ASSET_INTENSITY
            asset_reason_codes.extend([REASON_LOW_ASSET_INTENSITY_SUPPORT, REASON_CAPEX_BURDEN_NOT_DOMINANT])
        else:
            asset_intensity_class = MODERATE_ASSET_INTENSITY
            asset_reason_codes.append(REASON_MAINTENANCE_CAPEX_MIXED)

        if credibility_headwind_strength >= 3:
            credibility_class = LOW_MAINTENANCE_CAPEX_CREDIBILITY
            credibility_reason_codes.extend(
                [
                    REASON_LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND,
                    REASON_OWNER_EARNINGS_DENOMINATOR_SENSITIVE,
                    REASON_CAPEX_PROXY_POSSIBLY_TOO_FLATTERING,
                ]
            )
        elif (
            credibility_support_strength >= 3
            and credibility_headwind_strength == 0
            and asset_intensity_class == LOW_ASSET_INTENSITY
        ):
            credibility_class = HIGH_MAINTENANCE_CAPEX_CREDIBILITY
            credibility_reason_codes.extend(
                [REASON_CAPEX_PROXY_APPEARS_REASONABLE, REASON_CAPEX_BURDEN_NOT_DOMINANT]
            )
        else:
            credibility_class = MODERATE_MAINTENANCE_CAPEX_CREDIBILITY
            credibility_reason_codes.append(REASON_MAINTENANCE_CAPEX_MIXED)

        if SIG_CAPEX_VOLATILITY_HEADWIND in headwind_signals:
            credibility_reason_codes.append(REASON_CAPEX_VOLATILITY)
        if asset_intensity_class == HIGH_ASSET_INTENSITY and REASON_HIGH_ASSET_INTENSITY_HEADWIND not in credibility_reason_codes:
            credibility_reason_codes.append(REASON_HIGH_ASSET_INTENSITY_HEADWIND)
        if asset_intensity_class == LOW_ASSET_INTENSITY and REASON_LOW_ASSET_INTENSITY_SUPPORT not in credibility_reason_codes:
            credibility_reason_codes.append(REASON_LOW_ASSET_INTENSITY_SUPPORT)

        if credibility_class == LOW_MAINTENANCE_CAPEX_CREDIBILITY:
            caution = OWNER_EARNINGS_HEADWIND
        elif credibility_class == HIGH_MAINTENANCE_CAPEX_CREDIBILITY and asset_intensity_class == LOW_ASSET_INTENSITY:
            caution = OWNER_EARNINGS_SUPPORTIVE
        else:
            caution = OWNER_EARNINGS_MIXED

    reason_codes = _dedupe(asset_reason_codes + credibility_reason_codes)
    derived_from = _collect_derived_from(
        fundamentals,
        owner_payload,
        owner_quality_payload,
        accounting_quality_payload,
        reinvestment_efficiency_payload,
        returns_persistence_payload,
        intangible_payload,
        row_refs=(
            list(fundamentals.get("derived_from") or [])
            + list(owner_payload.get("derived_from") or [])
            + capex_to_revenue_refs
            + capex_to_cfo_refs
            + capex_to_owner_refs
            + maintenance_proxy_to_cfo_refs
            + depreciation_refs
            + capex_volatility_refs
            + latest_capex_refs
            + latest_cfo_refs
            + latest_owner_refs
            + list(row_derived_from or [])
        ),
    )

    payload = {
        "ticker": ticker_norm,
        "as_of_date": str(as_of_date or ""),
        "asset_intensity_class": asset_intensity_class,
        "asset_intensity_reason_codes": _dedupe(asset_reason_codes),
        "maintenance_capex_credibility_class": credibility_class,
        "maintenance_capex_credibility_reason_codes": _dedupe(credibility_reason_codes),
        "maintenance_capex_support_signals": support_signals,
        "maintenance_capex_headwind_signals": headwind_signals,
        "primary_maintenance_capex_caution": caution,
        "maintenance_capex_discipline_summary": _summary_for_profile(
            asset_intensity_class=asset_intensity_class,
            credibility_class=credibility_class,
            headwinds=headwind_signals,
            supports=support_signals,
        ),
        "capex_to_revenue_median": _to_num(capex_to_revenue),
        "capex_to_cfo_median": _to_num(capex_to_cfo),
        "capex_to_owner_earnings_median": _to_num(capex_to_owner),
        "maintenance_capex_proxy_to_cfo_median": _to_num(maintenance_proxy_to_cfo),
        "depreciation_to_capex_median": _to_num(depreciation_to_capex),
        "capex_volatility_proxy": _to_num(capex_volatility),
        "maintenance_capex_ratio": _to_num(maintenance_ratio),
        "derived_from": derived_from,
        "claims": {
            "asset_intensity_class": _claim(
                value=asset_intensity_class,
                refs=derived_from,
                reason_code=(_dedupe(asset_reason_codes) or [REASON_MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN])[0],
            ),
            "maintenance_capex_credibility_class": _claim(
                value=credibility_class,
                refs=derived_from,
                reason_code=(_dedupe(credibility_reason_codes) or [REASON_MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN])[0],
            ),
        },
        "generated_at": utc_now_iso(),
    }
    return payload


def write_maintenance_capex_discipline_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    output_path: Path,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    del cfg

    rows_in = [row for row in (scoreboard_rows or []) if isinstance(row, dict)]
    detail_by_ticker = {
        str(row.get("ticker") or "").upper(): row.get("maintenance_capex_discipline_detail")
        for row in rows_in
        if isinstance(row.get("maintenance_capex_discipline_detail"), dict)
    }

    rows: list[dict[str, Any]] = []
    counts_by_credibility_class: dict[str, int] = {}
    counts_by_asset_intensity_class: dict[str, int] = {}
    counts_by_caution: dict[str, int] = {}
    reason_counts: dict[str, int] = {}

    for ticker in sorted({str(token).strip().upper() for token in tickers if str(token).strip()}):
        detail = detail_by_ticker.get(ticker)
        if not isinstance(detail, dict):
            score_row = next((row for row in rows_in if str(row.get("ticker") or "").upper() == ticker), {})
            detail = compute_maintenance_capex_discipline(
                ticker=ticker,
                as_of_date=as_of_date,
                fundamentals=score_row.get("fundamentals_detail")
                if isinstance(score_row.get("fundamentals_detail"), dict)
                else {},
                owner_payload=score_row.get("owner_earnings_detail")
                if isinstance(score_row.get("owner_earnings_detail"), dict)
                else {},
                owner_quality_payload=score_row.get("owner_earnings_quality_detail")
                if isinstance(score_row.get("owner_earnings_quality_detail"), dict)
                else {},
                accounting_quality_payload=score_row.get("accounting_quality_detail")
                if isinstance(score_row.get("accounting_quality_detail"), dict)
                else {},
                reinvestment_efficiency_payload=score_row.get("reinvestment_efficiency_detail")
                if isinstance(score_row.get("reinvestment_efficiency_detail"), dict)
                else {},
                returns_persistence_payload=score_row.get("returns_persistence_detail")
                if isinstance(score_row.get("returns_persistence_detail"), dict)
                else {},
                intangible_payload=score_row.get("intangible_economics_detail")
                if isinstance(score_row.get("intangible_economics_detail"), dict)
                else {},
                price_status=score_row.get("price_status", UNKNOWN),
                facts_status=score_row.get("facts_status", UNKNOWN),
                shares_status=score_row.get("shares_status", UNKNOWN),
                fcf_status=score_row.get("fcf_status", UNKNOWN),
                row_derived_from=list(score_row.get("derived_from") or []),
            )

        credibility_class = str(
            detail.get("maintenance_capex_credibility_class") or MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN
        )
        asset_intensity_class = str(detail.get("asset_intensity_class") or ASSET_INTENSITY_UNKNOWN)
        caution = str(detail.get("primary_maintenance_capex_caution") or OWNER_EARNINGS_UNCLEAR)
        counts_by_credibility_class[credibility_class] = int(
            counts_by_credibility_class.get(credibility_class) or 0
        ) + 1
        counts_by_asset_intensity_class[asset_intensity_class] = int(
            counts_by_asset_intensity_class.get(asset_intensity_class) or 0
        ) + 1
        counts_by_caution[caution] = int(counts_by_caution.get(caution) or 0) + 1
        for code in _dedupe(
            list(detail.get("asset_intensity_reason_codes") or [])
            + list(detail.get("maintenance_capex_credibility_reason_codes") or [])
        ):
            reason_counts[code] = int(reason_counts.get(code) or 0) + 1
        rows.append(
            {
                "ticker": ticker,
                "asset_intensity_class": asset_intensity_class,
                "asset_intensity_reason_codes": [
                    str(code)
                    for code in (detail.get("asset_intensity_reason_codes") or [])
                    if str(code).strip()
                ],
                "maintenance_capex_credibility_class": credibility_class,
                "maintenance_capex_credibility_reason_codes": [
                    str(code)
                    for code in (detail.get("maintenance_capex_credibility_reason_codes") or [])
                    if str(code).strip()
                ],
                "maintenance_capex_support_signals": [
                    str(code)
                    for code in (detail.get("maintenance_capex_support_signals") or [])
                    if str(code).strip()
                ],
                "maintenance_capex_headwind_signals": [
                    str(code)
                    for code in (detail.get("maintenance_capex_headwind_signals") or [])
                    if str(code).strip()
                ],
                "primary_maintenance_capex_caution": caution,
                "maintenance_capex_discipline_summary": str(
                    detail.get("maintenance_capex_discipline_summary") or ""
                ),
                "derived_from": [str(ref) for ref in (detail.get("derived_from") or []) if str(ref).strip()],
            }
        )

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            _CREDIBILITY_ORDER.get(
                str(
                    row.get("maintenance_capex_credibility_class")
                    or MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN
                ),
                len(_CREDIBILITY_ORDER),
            ),
            _ASSET_INTENSITY_ORDER.get(
                str(row.get("asset_intensity_class") or ASSET_INTENSITY_UNKNOWN),
                len(_ASSET_INTENSITY_ORDER),
            ),
            str(row.get("ticker") or ""),
        ),
    )

    def _credibility_rows(value: str) -> list[dict[str, Any]]:
        return [
            row
            for row in rows_sorted
            if str(row.get("maintenance_capex_credibility_class") or MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN)
            == value
        ]

    def _asset_rows(value: str) -> list[dict[str, Any]]:
        return [
            row
            for row in rows_sorted
            if str(row.get("asset_intensity_class") or ASSET_INTENSITY_UNKNOWN) == value
        ]

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "generated_at": utc_now_iso(),
        "ticker_count": len(rows_sorted),
        "rows": rows_sorted,
        "counts_by_maintenance_capex_credibility_class": dict(sorted(counts_by_credibility_class.items())),
        "counts_by_asset_intensity_class": dict(sorted(counts_by_asset_intensity_class.items())),
        "counts_by_primary_maintenance_capex_caution": dict(sorted(counts_by_caution.items())),
        "top_10_high_maintenance_capex_credibility": [
            {
                "ticker": row.get("ticker", ""),
                "maintenance_capex_credibility_reason_codes": [
                    str(code)
                    for code in (row.get("maintenance_capex_credibility_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in _credibility_rows(HIGH_MAINTENANCE_CAPEX_CREDIBILITY)[:10]
        ],
        "top_10_low_maintenance_capex_credibility": [
            {
                "ticker": row.get("ticker", ""),
                "maintenance_capex_credibility_reason_codes": [
                    str(code)
                    for code in (row.get("maintenance_capex_credibility_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in _credibility_rows(LOW_MAINTENANCE_CAPEX_CREDIBILITY)[:10]
        ],
        "top_10_high_asset_intensity": [
            {
                "ticker": row.get("ticker", ""),
                "asset_intensity_reason_codes": [
                    str(code)
                    for code in (row.get("asset_intensity_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in _asset_rows(HIGH_ASSET_INTENSITY)[:10]
        ],
        "most_common_maintenance_capex_reason_codes": [
            {"reason_code": code, "count": count}
            for code, count in sorted(reason_counts.items(), key=lambda item: (-int(item[1]), item[0]))[:10]
        ],
        "maintenance_capex_discipline_path": str(output_path),
    }
    _json_write(output_path, payload)
    return payload


def _maintenance_capex_discipline_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "maintenance_capex_discipline.json",
        cfg.sectors_dir / run_id / "maintenance_capex_discipline.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else None


def open_maintenance_capex_discipline(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    path = _maintenance_capex_discipline_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "maintenance_capex_discipline_path": str(path) if path is not None else "",
        }

    payload = _safe_json(path)
    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or 0),
        "counts_by_maintenance_capex_credibility_class": (
            payload.get("counts_by_maintenance_capex_credibility_class")
            if isinstance(payload.get("counts_by_maintenance_capex_credibility_class"), dict)
            else {}
        ),
        "counts_by_asset_intensity_class": (
            payload.get("counts_by_asset_intensity_class")
            if isinstance(payload.get("counts_by_asset_intensity_class"), dict)
            else {}
        ),
        "counts_by_primary_maintenance_capex_caution": (
            payload.get("counts_by_primary_maintenance_capex_caution")
            if isinstance(payload.get("counts_by_primary_maintenance_capex_caution"), dict)
            else {}
        ),
        "top_10_high_maintenance_capex_credibility": [
            row
            for row in (payload.get("top_10_high_maintenance_capex_credibility") or [])
            if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_10_low_maintenance_capex_credibility": [
            row
            for row in (payload.get("top_10_low_maintenance_capex_credibility") or [])
            if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_10_high_asset_intensity": [
            row
            for row in (payload.get("top_10_high_asset_intensity") or [])
            if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "most_common_maintenance_capex_reason_codes": [
            row
            for row in (payload.get("most_common_maintenance_capex_reason_codes") or [])
            if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "maintenance_capex_discipline_path": str(path),
    }
