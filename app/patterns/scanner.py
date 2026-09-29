from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.diff.engine import list_dossier_run_tickers
from app.patterns.catalog import ALL_CATEGORIES, get_pattern_definition, get_pattern_definitions
from app.patterns.schemas import PatternHit, PatternResult, PatternScanReport
from app.valuation.facts import resolve_financial_facts_asof
from app.valuation.intangible_economics import (
    _CAPEX_TAG_PRIORITY,
    _GROSS_PROFIT_TAG_PRIORITY,
    _OPERATING_INCOME_TAG_PRIORITY,
    _RND_TAG_PRIORITY,
    _REVENUE_TAG_PRIORITY,
    _dedupe_refs,
    _is_num,
    _series_from_companyfacts,
)
from app.valuation.owner_earnings import _CFO_TAG_PRIORITY, _load_companyfacts_payload
from app.valuation.tech_category import classify_company_category

_DEFERRED_REVENUE_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "ContractWithCustomerLiability"),
    ("us-gaap", "ContractWithCustomerLiabilityCurrent"),
    ("us-gaap", "DeferredRevenue"),
    ("us-gaap", "DeferredRevenueCurrent"),
]
_DEPRECIATION_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "Depreciation"),
    ("us-gaap", "DepreciationAndAmortization"),
    ("us-gaap", "DepreciationDepletionAndAmortization"),
]
_NET_INCOME_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "NetIncomeLoss"),
    ("us-gaap", "ProfitLoss"),
]
_SHARES_TAG_PRIORITY: list[tuple[str, str]] = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockOtherSharesOutstanding"),
]


def pattern_scan_report_path(run_id: str, *, cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_config()
    return cfg.outputs_dir / "patterns" / f"{str(run_id).strip()}_pattern_scan.json"


def load_pattern_scan_report(run_id: str, *, cfg: AppConfig | None = None) -> PatternScanReport | None:
    path = pattern_scan_report_path(run_id, cfg=cfg)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return PatternScanReport.model_validate(payload)
    except Exception:
        return None


def count_patterns_with_hits(report: PatternScanReport | dict[str, Any] | None) -> int:
    if isinstance(report, PatternScanReport):
        pattern_results = report.pattern_results
    elif isinstance(report, dict):
        pattern_results = report.get("pattern_results") or []
    else:
        pattern_results = []
    distinct: set[str] = set()
    for result in pattern_results:
        if isinstance(result, PatternResult):
            pattern_id = result.pattern_id
            hit_count = int(result.hit_count or 0)
        elif isinstance(result, dict):
            pattern_id = str(result.get("pattern_id") or "").strip()
            hit_count = int(result.get("hit_count") or 0)
        else:
            continue
        if pattern_id and hit_count > 0:
            distinct.add(pattern_id)
    return len(distinct)


def _metric_series(
    *,
    companyfacts: dict[str, Any],
    as_of_date: str,
    priority: list[tuple[str, str]],
    expected_unit_exact: tuple[str, ...] = ("usd",),
) -> list[dict[str, Any]]:
    return _series_from_companyfacts(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=priority,
        expected_unit_exact=expected_unit_exact,
    )


def _series_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        year = int(row.get("year") or 0)
        if year <= 0 or not _is_num(row.get("value")):
            continue
        out[year] = {
            "year": year,
            "value": float(row["value"]),
            "derived_from": [str(ref) for ref in (row.get("derived_from") or []) if str(ref).strip()],
        }
    return out


def _ratio_series(numerator: list[dict[str, Any]], denominator: list[dict[str, Any]]) -> list[dict[str, Any]]:
    denominator_by_year = _series_map(denominator)
    out: list[dict[str, Any]] = []
    for row in numerator:
        year = int(row.get("year") or 0)
        denom = denominator_by_year.get(year)
        if not isinstance(denom, dict) or not _is_num(row.get("value")) or not _is_num(denom.get("value")):
            continue
        denom_value = float(denom["value"])
        if denom_value == 0.0:
            continue
        out.append(
            {
                "year": year,
                "value": float(row["value"]) / denom_value,
                "derived_from": _dedupe_refs(list(row.get("derived_from") or []) + list(denom.get("derived_from") or [])),
            }
        )
    return sorted(out, key=lambda row: int(row["year"]))


def _derived_difference_series(first: list[dict[str, Any]], second: list[dict[str, Any]], *, metric_name: str) -> list[dict[str, Any]]:
    second_by_year = _series_map(second)
    out: list[dict[str, Any]] = []
    for row in first:
        year = int(row.get("year") or 0)
        other = second_by_year.get(year)
        if not isinstance(other, dict) or not _is_num(row.get("value")):
            continue
        out.append(
            {
                "year": year,
                "value": float(row["value"]) - float(other["value"]),
                "derived_from": _dedupe_refs(list(row.get("derived_from") or []) + list(other.get("derived_from") or [])),
                "metric": metric_name,
            }
        )
    return out


def _load_dossier_as_of(*, run_id: str, ticker: str, cfg: AppConfig) -> str | None:
    path = cfg.dossiers_dir / run_id / ticker.upper() / "dossier.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    as_of_date = str(payload.get("as_of_date") or "").strip()
    return as_of_date or None


def _extract_ticker_timeseries(
    *,
    ticker: str,
    as_of_date: str,
    companyfacts: dict[str, Any],
    cfg: AppConfig,
) -> dict[str, Any]:
    revenue = _metric_series(companyfacts=companyfacts, as_of_date=as_of_date, priority=_REVENUE_TAG_PRIORITY)
    gross_profit = _metric_series(companyfacts=companyfacts, as_of_date=as_of_date, priority=_GROSS_PROFIT_TAG_PRIORITY)
    operating_income = _metric_series(companyfacts=companyfacts, as_of_date=as_of_date, priority=_OPERATING_INCOME_TAG_PRIORITY)
    net_income = _metric_series(companyfacts=companyfacts, as_of_date=as_of_date, priority=_NET_INCOME_TAG_PRIORITY)
    cfo = _metric_series(companyfacts=companyfacts, as_of_date=as_of_date, priority=_CFO_TAG_PRIORITY)
    capex = _metric_series(companyfacts=companyfacts, as_of_date=as_of_date, priority=_CAPEX_TAG_PRIORITY)
    r_and_d_total = _metric_series(companyfacts=companyfacts, as_of_date=as_of_date, priority=_RND_TAG_PRIORITY)
    shares_outstanding = _metric_series(companyfacts=companyfacts, as_of_date=as_of_date, priority=_SHARES_TAG_PRIORITY, expected_unit_exact=("shares",))
    deferred_revenue = _metric_series(companyfacts=companyfacts, as_of_date=as_of_date, priority=_DEFERRED_REVENUE_TAG_PRIORITY)
    depreciation = _metric_series(companyfacts=companyfacts, as_of_date=as_of_date, priority=_DEPRECIATION_TAG_PRIORITY)
    fcf = _derived_difference_series(cfo, capex, metric_name="fcf")
    gross_margin = _ratio_series(gross_profit, revenue)
    operating_margin = _ratio_series(operating_income, revenue)
    category_payload = classify_company_category(
        ticker=ticker,
        as_of_date=as_of_date,
        companyfacts=companyfacts,
        cfg=cfg,
    )
    return {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "category": category_payload.get("category"),
        "category_confidence": category_payload.get("confidence"),
        "category_derived_from": category_payload.get("derived_from") or [],
        "revenue": revenue,
        "gross_profit": gross_profit,
        "operating_income": operating_income,
        "net_income": net_income,
        "cfo": cfo,
        "capex": capex,
        "r_and_d_total": r_and_d_total,
        "shares_outstanding": shares_outstanding,
        "deferred_revenue": deferred_revenue,
        "depreciation": depreciation,
        "fcf": fcf,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
    }


def _applicable_category(category: Any, allowed: tuple[str, ...]) -> bool:
    if allowed == ALL_CATEGORIES:
        return True
    token = str(category or "").strip().upper()
    return token in {str(item).strip().upper() for item in allowed}


def _recency_weight(years_detected: list[int]) -> float:
    valid_years = [int(year) for year in years_detected if isinstance(year, int) or str(year).isdigit()]
    if not valid_years:
        return 0.25
    latest_year = max(valid_years)
    delta = max(0, latest_year - max(valid_years))
    # latest detected year anchors the hit; weight is based on how old that most-recent
    # supporting year is within the detected set.
    delta = 0
    if delta <= 0:
        return 1.0
    if delta == 1:
        return 0.75
    if delta == 2:
        return 0.5
    return 0.25


def _recency_weight_from_years(years_detected: list[int], *, reference_year: int | None = None) -> float:
    valid_years = [int(year) for year in years_detected if isinstance(year, int) or str(year).isdigit()]
    if not valid_years:
        return 0.25
    latest_year = max(valid_years)
    reference = int(reference_year) if isinstance(reference_year, int) else latest_year
    age = max(0, reference - latest_year)
    if age <= 0:
        return 1.0
    if age == 1:
        return 0.75
    if age == 2:
        return 0.5
    return 0.25


def scan_peer_set(
    *,
    run_id: str,
    tickers: list[str] | None = None,
    patterns: list[str] | None = None,
    cfg: AppConfig | None = None,
) -> PatternScanReport:
    cfg = cfg or get_config()
    selected_tickers = [str(ticker).strip().upper() for ticker in (tickers or []) if str(ticker).strip()]
    if not selected_tickers:
        selected_tickers = list_dossier_run_tickers(run_id=run_id)
    selected_patterns = get_pattern_definitions(patterns)
    if patterns and len(selected_patterns) != len({str(pattern_id).strip() for pattern_id in patterns if str(pattern_id).strip()}):
        missing = sorted({str(pattern_id).strip() for pattern_id in patterns if str(pattern_id).strip()} - {definition.pattern_id for definition in selected_patterns})
        raise ValueError(f"unknown pattern ids: {', '.join(missing)}")

    timeseries_by_ticker: dict[str, dict[str, Any]] = {}
    for ticker in selected_tickers:
        as_of_date = _load_dossier_as_of(run_id=run_id, ticker=ticker, cfg=cfg)
        if not as_of_date:
            continue
        facts_row = resolve_financial_facts_asof(ticker=ticker, as_of_date=as_of_date, refresh=False, cfg=cfg)
        companyfacts = _load_companyfacts_payload(facts_row)
        if not companyfacts:
            continue
        timeseries_by_ticker[ticker] = _extract_ticker_timeseries(
            ticker=ticker,
            as_of_date=as_of_date,
            companyfacts=companyfacts,
            cfg=cfg,
        )

    pattern_results: list[PatternResult] = []
    all_detected_years: list[int] = []
    for data in timeseries_by_ticker.values():
        revenue_rows = data.get("revenue") if isinstance(data.get("revenue"), list) else []
        all_detected_years.extend(int(row.get("year")) for row in revenue_rows if isinstance(row, dict) and row.get("year"))
    reference_year = max(all_detected_years) if all_detected_years else None
    for definition in selected_patterns:
        hits: list[PatternHit] = []
        for ticker, data in timeseries_by_ticker.items():
            if not _applicable_category(data.get("category"), definition.category):
                continue
            detection = definition.detection_fn(data)
            if not detection.get("present"):
                continue
            outcome = definition.outcome_fn(data, detection)
            years_detected = [int(year) for year in (detection.get("years_detected") or [])]
            hits.append(
                PatternHit(
                    ticker=ticker,
                    pattern_id=definition.pattern_id,
                    years_detected=years_detected,
                    detection_strength=float(detection.get("detection_strength") or 0.0),
                    recency_weight=_recency_weight_from_years(years_detected, reference_year=reference_year),
                    outcome_confirmed=outcome.get("outcome_confirmed"),
                    outcome_value=float(outcome["outcome_value"]) if isinstance(outcome.get("outcome_value"), (int, float)) else None,
                    outcome_details=str(outcome.get("outcome_details") or "") or None,
                    derived_from=_dedupe_refs(
                        [str(ref) for ref in (detection.get("derived_from") or []) if str(ref).strip()]
                        + [str(ref) for ref in (outcome.get("derived_from") or []) if str(ref).strip()]
                        + [str(ref) for ref in (data.get("category_derived_from") or []) if str(ref).strip()]
                    ),
                )
            )
        confirmed_count = len([hit for hit in hits if hit.outcome_confirmed is True])
        sample_size = len([hit for hit in hits if hit.outcome_confirmed is not None])
        unconfirmed_count = len([hit for hit in hits if hit.outcome_confirmed is None])
        hit_rate = (confirmed_count / sample_size) if sample_size > 0 else None
        pattern_results.append(
            PatternResult(
                pattern_id=definition.pattern_id,
                hypothesis=definition.hypothesis,
                hit_count=len(hits),
                confirmed_count=confirmed_count,
                unconfirmed_count=unconfirmed_count,
                hit_rate=hit_rate,
                sample_size=sample_size,
                hits=hits,
            )
        )

    report = PatternScanReport(
        run_id=run_id,
        scan_date=utc_now_iso(),
        peer_set_size=len(timeseries_by_ticker),
        peer_set_tickers=sorted(timeseries_by_ticker.keys()),
        pattern_results=pattern_results,
        patterns_with_signal=[
            result.pattern_id
            for result in pattern_results
            if result.hit_rate is not None and result.hit_rate > 0.5 and result.sample_size >= 3
        ],
        recency_weighted_hit_count=round(
            sum(float(hit.recency_weight) for result in pattern_results for hit in result.hits),
            4,
        ),
    )

    if tickers is None and patterns is None:
        path = pattern_scan_report_path(run_id, cfg=cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    return report


def summarize_pattern_scan_for_ticker(report: PatternScanReport, ticker: str) -> dict[str, Any]:
    ticker_norm = str(ticker or "").strip().upper()
    pattern_hits: list[dict[str, Any]] = []
    for result in report.pattern_results:
        definition = get_pattern_definition(result.pattern_id)
        for hit in result.hits:
            if hit.ticker.upper() != ticker_norm:
                continue
            years_text = "-".join(str(year) for year in hit.years_detected) if hit.years_detected else "unknown years"
            summary_text = (
                f"Cross-sectional analysis detected {definition.name if definition else result.pattern_id} in {ticker_norm} "
                f"({years_text}), with a historical hit rate of "
                f"{(result.hit_rate * 100):.0f}% across {result.sample_size} checkable peers."
                if result.hit_rate is not None and result.sample_size > 0
                else f"Cross-sectional analysis detected {definition.name if definition else result.pattern_id} in {ticker_norm} ({years_text})."
            )
            pattern_hits.append(
                {
                    "pattern_id": result.pattern_id,
                    "name": definition.name if definition else result.pattern_id,
                    "hypothesis": result.hypothesis,
                    "years_detected": list(hit.years_detected),
                    "detection_strength": hit.detection_strength,
                    "recency_weight": hit.recency_weight,
                    "outcome_confirmed": hit.outcome_confirmed,
                    "outcome_value": hit.outcome_value,
                    "outcome_details": hit.outcome_details,
                    "hit_rate": result.hit_rate,
                    "sample_size": result.sample_size,
                    "summary_text": summary_text,
                    "derived_from": list(hit.derived_from),
                }
            )
    return {
        "ticker": ticker_norm,
        "run_id": report.run_id,
        "pattern_hit_count": len(pattern_hits),
        "pattern_hits": pattern_hits,
        "summary_text": " ".join(item["summary_text"] for item in pattern_hits[:3]) if pattern_hits else None,
    }
